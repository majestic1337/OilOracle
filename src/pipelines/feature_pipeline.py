"""Feature engineering pipeline for ML and DL forecasting workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from loguru import logger
from statsmodels.tsa.ar_model import AutoReg

ROLLING_MEAN_WINDOWS: tuple[int, ...] = (5, 10, 20)
ROLLING_STD_WINDOWS: tuple[int, ...] = (5, 20)
REQUIRED_RETURN_COLUMNS: tuple[str, ...] = (
    "brent_return",
    "wti_return",
    "dxy_return",
    "gold_return",
)
REQUIRED_DL_PRICE_COLUMNS: tuple[str, ...] = ("brent",)
LAG_AIC_CACHE: dict[int, float] = {}
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 7)
SEMANTIC_LAGS: tuple[int, ...] = (5, 10)
VOLATILITY_WINDOWS: tuple[int, ...] = (22, 63)

def _validate_returns_columns(df: pd.DataFrame) -> None:
    return_columns = {column for column in df.columns if column.endswith("_return")}
    missing = [column for column in REQUIRED_RETURN_COLUMNS if column not in return_columns]
    if missing:
        raise ValueError(f"Missing required return columns: {missing}")


def _validate_dl_price_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_DL_PRICE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required price columns for DL: {missing}")


def _compute_cumulative_target(returns_df: pd.DataFrame, horizon: int) -> pd.Series:
    """Build cumulative log-return target over horizon h.

    The target at row t equals sum(r_{t+1}, ..., r_{t+h}), i.e. the
    cumulative log-return over the next h trading days. This is achieved by
    .rolling(h).sum() (which at row t gives sum of [t-h+1..t]) followed by
    .shift(-h) (which moves that value to row t-h, so row t receives the
    forward sum [t+1..t+h]).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    _validate_returns_columns(returns_df)
    target = (
        returns_df["brent_return"]
        .rolling(window=horizon, min_periods=horizon)
        .sum()
        .shift(-horizon)
        .rename(f"target_h{horizon}")
    )
    return target


def _compute_aic_by_lag(train_df: pd.DataFrame, max_search: int) -> dict[int, float]:
    """Compute AIC values for AR(p) models on brent_return."""
    _validate_returns_columns(train_df)

    series = train_df["brent_return"].dropna()
    if series.empty:
        raise ValueError("train_df['brent_return'] is empty after dropping NaN")

    aic_values: dict[int, float] = {}
    for lag in range(1, max_search + 1):
        try:
            model = AutoReg(series, lags=lag, old_names=False).fit()
        except Exception as exc:  # noqa: BLE001 - keep lag search robust
            logger.warning("Failed to fit AR({lag}): {error}", lag=lag, error=str(exc))
            continue

        aic_values[lag] = float(model.aic)
        logger.info("AIC for lag {lag}: {aic}", lag=lag, aic=aic_values[lag])

    if not aic_values:
        raise ValueError("No AR models were successfully fit for AIC evaluation")

    return aic_values


def determine_max_lag_order(
    train_df: pd.DataFrame, max_search: int = 5
) -> tuple[int, dict[int, float]]:
    """Determine optimal lag order using AIC on the training segment.

    Returns:
        Tuple of (best_lag, aic_values) where aic_values maps lag -> AIC score.
    """
    aic_values = _compute_aic_by_lag(train_df, max_search=max_search)

    best_lag = min(aic_values, key=aic_values.get)
    logger.info(
        "Selected max lag order {lag} with AIC {aic}",
        lag=best_lag,
        aic=aic_values[best_lag],
    )
    return best_lag, aic_values


def _generate_ml_features(
    returns_df: pd.DataFrame,
    max_lag: int,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Generate lag/rolling/technical predictors for *_return columns.

    Note:
        Rolling features use only the rows available in returns_df. When called
        on val or test splits without a warmup prefix, the first
        max(ROLLING_MEAN_WINDOWS) - 1 rows will be NaN.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    features = returns_df.copy()
    return_columns = [column for column in returns_df.columns if column.endswith("_return")]

    # --- Лагові ознаки ---
    for col in return_columns:
        base_lags = list(range(1, max_lag + 1))
        extra_lags = [lag for lag in SEMANTIC_LAGS if lag > max_lag]

        for lag in base_lags + extra_lags:
            features[f"{col}_lag{lag}"] = returns_df[col].shift(lag)

        if include_rolling:
            for window in ROLLING_MEAN_WINDOWS:
                features[f"{col}_rollmean{window}"] = (
                    returns_df[col].rolling(window=window).mean().shift(1)
                )
            for window in ROLLING_STD_WINDOWS:
                features[f"{col}_rollstd{window}"] = (
                    returns_df[col].rolling(window=window).std().shift(1)
                )

    # --- Похідні RSI ознаки ---
    if "brent_rsi" in returns_df.columns:
        features["brent_delta_rsi"] = returns_df["brent_rsi"].diff()

    # --- Похідні MACD ознаки ---
    if "brent_macd" in returns_df.columns and "brent_signal" in returns_df.columns:
        features["brent_macd_hist"] = (
            returns_df["brent_macd"] - returns_df["brent_signal"]
        )

    # --- Ознаки волатильності Brent ---
    if "brent_return" in returns_df.columns:
        for window in VOLATILITY_WINDOWS:
            features[f"brent_vol{window}"] = (
                returns_df["brent_return"]
                .rolling(window=window)
                .std()
                .shift(1)
            )

        vol_22 = features.get("brent_vol22")
        vol_63 = features.get("brent_vol63")
        if vol_22 is not None and vol_63 is not None:
            features["brent_vol_ratio"] = vol_22 / (vol_63 + 1e-8)

    return features

def prepare_ml_data(
    returns_df: pd.DataFrame,
    max_lag: int,
    horizon: int,
    output_dir: str | Path,
    include_rolling: bool = True,
    split_name: str = "",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare ML-ready feature matrix with cumulative log-return target.

    Steps:
    1) Build lag/rolling predictors.
    2) Compute cumulative log-return target over horizon h.
    3) Drop all rows containing NaN.
    4) Save `{split}_feature_matrix_ml_lag{max_lag}.parquet` and
       `{split}_target_h{horizon}.parquet`.

    Note:
        The first max(ROLLING_MEAN_WINDOWS) - 1 = 19 rows of each split are
        dropped via dropna() because rolling features require a warm-up period.
        For val/test splits this means up to 19 leading observations are lost.
        To avoid this, pass a warmup prefix from the previous split's tail
        (analogous to EWM_WARMUP_PERIODS in data_pipeline.py) — not implemented.
    """
    _validate_returns_columns(returns_df)

    features = _generate_ml_features(
        returns_df=returns_df,
        max_lag=max_lag,
        include_rolling=include_rolling,
    )
    target = _compute_cumulative_target(returns_df, horizon=horizon)

    joined = pd.concat([features, target], axis=1).dropna(how="any")
    feature_cols = [column for column in joined.columns if column != target.name]

    X_ml = joined[feature_cols]
    y_ml = joined[target.name].rename("brent_return_target")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{split_name}_" if split_name else ""
    feature_path = output_path / f"{prefix}feature_matrix_ml_lag{max_lag}.parquet"
    target_path = output_path / f"{prefix}target_h{horizon}.parquet"

    X_ml.to_parquet(feature_path)
    y_ml.to_frame(name=target.name).to_parquet(target_path)

    logger.info("Saved ML features to {path}", path=feature_path)
    logger.info("Saved ML target to {path}", path=target_path)

    return X_ml, y_ml


def save_shifted_targets(
    returns_df: pd.DataFrame,
    horizons: Iterable[int],
    output_dir: str | Path,
    valid_index: pd.Index,
    split_name: str = "",
) -> dict[int, pd.Series]:
    """Save cumulative Brent log-return targets aligned to valid_index."""
    _validate_returns_columns(returns_df)

    unique_horizons = sorted({int(h) for h in horizons})
    if not unique_horizons:
        raise ValueError("At least one horizon is required")
    if any(h < 1 for h in unique_horizons):
        raise ValueError("All horizons must be >= 1")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = f"{split_name}_" if split_name else ""
    targets: dict[int, pd.Series] = {}
    for horizon in unique_horizons:
        if len(returns_df) <= horizon:
            raise ValueError(
                f"returns_df is too short for horizon h={horizon}: "
                f"{len(returns_df)} rows <= {horizon}. "
                f"Check split boundaries or reduce horizon."
            )
        target = _compute_cumulative_target(returns_df, horizon=horizon).dropna()
        target = target.reindex(valid_index)

        nan_count = target.isna().sum()
        if nan_count > 0:
            logger.warning(
                "save_shifted_targets: horizon h={h} has {n} NaN values after reindex — "
                "valid_index contains dates not present in target",
                h=horizon, n=nan_count,
            )

        target_path = output_path / f"{prefix}target_h{horizon}.parquet"
        target.to_frame(name=f"target_h{horizon}").to_parquet(target_path)
        targets[horizon] = target
        logger.info(
            "Saved cumulative target for horizon h={h} to {path}",
            h=horizon,
            path=target_path,
        )

    return targets


def prepare_dl_data(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    output_dir: str | Path,
    brent_index: pd.DatetimeIndex | None = None,
    split_name: str = "",
) -> pd.DataFrame:
    """Prepare DL-ready data for NeuralForecast.

    Target y is sourced from raw Brent prices, while exogenous regressors come
    from return-space series.
    Output schema: [unique_id, ds, y, wti_return, dxy_return, gold_return].

    Note:
        Calendar gap filling is performed only here (not in the ML branch)
        because NeuralForecast requires a contiguous daily calendar for its
        internal temporal encoding. If ``brent_index`` is provided, the DL
        DataFrame is reindexed to that Brent-anchored trading calendar and
        forward-filled (instead of using ``asfreq("B")`` which would insert
        ISO business days absent from the original Brent calendar).
    """
    _validate_dl_price_columns(prices_df)
    _validate_returns_columns(returns_df)

    common_index = pd.DatetimeIndex(prices_df.index).intersection(
        pd.DatetimeIndex(returns_df.index)
    )
    if common_index.empty:
        raise ValueError("No overlapping index between prices_df and returns_df")

    prices_aligned = prices_df.loc[common_index].sort_index()
    returns_aligned = returns_df.loc[common_index].sort_index()

    if brent_index is not None:
        brent_subset = brent_index[
            (brent_index >= common_index.min()) & (brent_index <= common_index.max())
        ]
        prices_aligned = prices_aligned.reindex(brent_subset).ffill()
        returns_aligned = returns_aligned.reindex(brent_subset).ffill()

    dl_df = pd.DataFrame(
        {
            "unique_id": "brent",
            "ds": pd.DatetimeIndex(prices_aligned.index),
            "y": prices_aligned["brent"],
            "wti_return": returns_aligned["wti_return"],
            "dxy_return": returns_aligned["dxy_return"],
            "gold_return": returns_aligned["gold_return"],
        },
        index=prices_aligned.index,
    )

    dl_df = dl_df.dropna(how="any")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"{split_name}_" if split_name else ""
    dl_path = output_path / f"{prefix}feature_matrix_dl.parquet"
    dl_df.to_parquet(dl_path)

    logger.info("Saved DL feature matrix to {path}", path=dl_path)
    return dl_df


def save_lag_config(
    max_lag_order: int,
    max_search: int,
    train_end_date: str,
    aic_values: dict[int, float],
    config_path: str | Path = "configs/lag_order_config.json",
) -> dict[str, Any]:
    """Persist lag-selection metadata used by the ML branch."""
    payload: dict[str, Any] = {
        "max_lag_order": max_lag_order,
        "max_search": max_search,
        "determined_on": "train",
        "train_end_date": train_end_date,
        "aic_values": {str(lag): value for lag, value in aic_values.items()},
    }

    cfg_path = Path(config_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved lag config to {path}", path=cfg_path)
    return payload

def _add_spread_feature(
    returns_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add Brent-WTI spread as an additional feature.

    Args:
        returns_df: DataFrame of log returns with indicator columns.
        prices_df: DataFrame of aligned prices.

    Returns:
        returns_df with appended brent_wti_spread column.
    """
    if "brent" not in prices_df.columns or "wti" not in prices_df.columns:
        logger.warning("Cannot compute spread: brent or wti missing from prices_df")
        return returns_df

    spread = (prices_df["brent"] - prices_df["wti"]).rename("brent_wti_spread")
    spread_shifted = spread.shift(1)

    result = returns_df.copy()
    result["brent_wti_spread"] = spread_shifted.reindex(returns_df.index)
    return result

def run_feature_pipeline(
    processed_dir: str | Path = "data/processed",
    horizon: int = 1,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_search: int = 5,
) -> dict[str, Any]:
    """Run both ML and DL feature preparation branches.

    Returns:
        Dictionary with two keys:
        - "ml": dict mapping split name ("train", "val", "test") to
          {"X_ml": pd.DataFrame, "y_ml": pd.Series, "targets_ml": dict[int, pd.Series]}
        - "dl": dict mapping split name ("train", "val", "test") to pd.DataFrame
          with columns [unique_id, ds, y, wti_return, dxy_return, gold_return]
    """
    output_dir = Path(processed_dir)
    train_returns = pd.read_parquet(output_dir / "train_returns.parquet")
    val_returns = pd.read_parquet(output_dir / "val_returns.parquet")
    test_returns = pd.read_parquet(output_dir / "test_returns.parquet")

    train_prices = pd.read_parquet(output_dir / "train_prices.parquet")
    val_prices = pd.read_parquet(output_dir / "val_prices.parquet")
    test_prices = pd.read_parquet(output_dir / "test_prices.parquet")

    train_returns = _add_spread_feature(train_returns, train_prices)
    val_returns = _add_spread_feature(val_returns, val_prices)
    test_returns = _add_spread_feature(test_returns, test_prices)

    # --- Lag order selection (train only) ---
    max_lag_order, aic_values = determine_max_lag_order(train_returns, max_search=max_search)

    save_lag_config(
        max_lag_order=max_lag_order,
        max_search=max_search,
        train_end_date=str(train_returns.index.max().date()),
        aic_values=aic_values,
    )

    all_horizons = tuple(sorted(set(horizons + (horizon,))))

    # --- ML branch: per-split feature generation ---
    splits_returns = {"train": train_returns, "val": val_returns, "test": test_returns}
    splits_prices = {"train": train_prices, "val": val_prices, "test": test_prices}

    ml_results: dict[str, dict[str, Any]] = {}
    for split_name, split_returns in splits_returns.items():
        X_ml, y_ml = prepare_ml_data(
            returns_df=split_returns,
            max_lag=max_lag_order,
            horizon=horizon,
            output_dir=output_dir,
            include_rolling=True,
            split_name=split_name,
        )

        extra_horizons = tuple(h for h in all_horizons if h != horizon)
        extra_targets = (
            save_shifted_targets(
                returns_df=split_returns,
                horizons=extra_horizons,
                output_dir=output_dir,
                valid_index=X_ml.index,
                split_name=split_name,
            )
            if extra_horizons
            else {}
        )

        ml_results[split_name] = {
            "X_ml": X_ml,
            "y_ml": y_ml,
            "targets_ml": extra_targets,
        }

    # --- DL branch: per-split feature generation ---
    dl_results: dict[str, pd.DataFrame] = {}
    for split_name in splits_returns:
        dl_df = prepare_dl_data(
            prices_df=splits_prices[split_name],
            returns_df=splits_returns[split_name],
            output_dir=output_dir,
            split_name=split_name,
        )
        dl_results[split_name] = dl_df

    logger.info("ML matrix shape (train): {shape}", shape=ml_results["train"]["X_ml"].shape)
    logger.info("DL matrix shape (train): {shape}", shape=dl_results["train"].shape)

    return {
        "ml": ml_results,
        "dl": dl_results,
    }


if __name__ == "__main__":
    try:
        run_feature_pipeline(
            processed_dir="data/processed",
            horizon=1,
            horizons=DEFAULT_HORIZONS,
            max_search=5,
        )
    except Exception as exc:  # noqa: BLE001 - keep CLI feedback explicit
        logger.exception("Feature pipeline failed: {error}", error=str(exc))
        raise
