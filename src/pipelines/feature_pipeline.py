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
    """Build cumulative log-return target over horizon h."""
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


def determine_max_lag_order(train_df: pd.DataFrame, max_search: int = 5) -> int:
    """Determine optimal lag order using AIC on the training segment."""
    global LAG_AIC_CACHE
    LAG_AIC_CACHE = _compute_aic_by_lag(train_df, max_search=max_search)

    best_lag = min(LAG_AIC_CACHE, key=LAG_AIC_CACHE.get)
    logger.info(
        "Selected max lag order {lag} with AIC {aic}",
        lag=best_lag,
        aic=LAG_AIC_CACHE[best_lag],
    )
    return best_lag


def _generate_ml_features(
    returns_df: pd.DataFrame,
    max_lag: int,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Generate lag/rolling predictors only for *_return columns."""
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    features = returns_df.copy()
    return_columns = [column for column in returns_df.columns if column.endswith("_return")]

    for col in return_columns:
        for lag in range(1, max_lag + 1):
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

    return features


def prepare_ml_data(
    returns_df: pd.DataFrame,
    max_lag: int,
    horizon: int,
    output_dir: str | Path,
    include_rolling: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare ML-ready feature matrix with cumulative log-return target.

    Steps:
    1) Build lag/rolling predictors.
    2) Compute cumulative log-return target over horizon h.
    3) Drop all rows containing NaN.
    4) Save `feature_matrix_ml.parquet` and `target_h{horizon}.parquet`.
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

    feature_path = output_path / "feature_matrix_ml.parquet"
    target_path = output_path / f"target_h{horizon}.parquet"

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

    targets: dict[int, pd.Series] = {}
    for horizon in unique_horizons:
        target = _compute_cumulative_target(returns_df, horizon=horizon).dropna()
        target = target.reindex(valid_index)
        target_path = output_path / f"target_h{horizon}.parquet"
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
) -> pd.DataFrame:
    """Prepare DL-ready data for NeuralForecast.

    Target y is sourced from raw Brent prices, while exogenous regressors come
    from return-space series.
    Output schema: [unique_id, ds, y, wti_return, dxy_return, gold_return].
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

    dl_df = pd.DataFrame(
        {
            "unique_id": "brent",
            "ds": pd.DatetimeIndex(common_index),
            "y": prices_aligned["brent"],
            "wti_return": returns_aligned["wti_return"],
            "dxy_return": returns_aligned["dxy_return"],
            "gold_return": returns_aligned["gold_return"],
        },
        index=common_index,
    )

    dl_df = dl_df.dropna(how="any")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dl_path = output_path / "feature_matrix_dl.parquet"
    dl_df.to_parquet(dl_path)

    logger.info("Saved DL feature matrix to {path}", path=dl_path)
    return dl_df


def save_lag_config(
    max_lag_order: int,
    max_search: int,
    train_end_date: str,
    config_path: str | Path = "configs/lag_order_config.json",
) -> dict[str, Any]:
    """Persist lag-selection metadata used by the ML branch."""
    payload: dict[str, Any] = {
        "max_lag_order": max_lag_order,
        "max_search": max_search,
        "determined_on": "train",
        "train_end_date": train_end_date,
        "aic_values": {str(lag): value for lag, value in LAG_AIC_CACHE.items()},
    }

    cfg_path = Path(config_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved lag config to {path}", path=cfg_path)
    return payload

def _fill_calendar_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure no missing business days in the index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Ресемплінг до бізнес-днів ('B') та заповнення пропусків останнім значенням
    return df.asfreq("B").ffill()

def run_feature_pipeline(
    processed_dir: str | Path = "data/processed",
    horizon: int = 1,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_search: int = 5,
) -> dict[str, Any]:
    """Run both ML and DL feature preparation branches."""
    output_dir = Path(processed_dir)
    train_returns = pd.read_parquet(output_dir / "train_returns.parquet")
    val_returns = pd.read_parquet(output_dir / "val_returns.parquet")
    test_returns = pd.read_parquet(output_dir / "test_returns.parquet")

    train_prices = pd.read_parquet(output_dir / "train_prices.parquet")
    val_prices = pd.read_parquet(output_dir / "val_prices.parquet")
    test_prices = pd.read_parquet(output_dir / "test_prices.parquet")

    full_returns = pd.concat([train_returns, val_returns, test_returns]).sort_index()
    full_prices = pd.concat([train_prices, val_prices, test_prices]).sort_index()

    # Виправлення пропусків у календарі (важливо для DL моделей)
    full_returns = _fill_calendar_gaps(full_returns)
    full_prices = _fill_calendar_gaps(full_prices)

    common_index = full_returns.index.intersection(full_prices.index)
    if common_index.empty:
        raise ValueError("No overlapping index between full_returns and full_prices")

    full_returns = full_returns.loc[common_index].sort_index()
    full_prices = full_prices.loc[common_index].sort_index()

    max_lag_order = determine_max_lag_order(train_returns, max_search=max_search)
    save_lag_config(
        max_lag_order=max_lag_order,
        max_search=max_search,
        train_end_date=str(train_returns.index.max().date()),
    )

    all_horizons = tuple(sorted(set(horizons + (horizon,))))

    X_ml, y_ml = prepare_ml_data(
        returns_df=full_returns,
        max_lag=max_lag_order,
        horizon=horizon,
        output_dir=output_dir,
        include_rolling=True,
    )
    extra_horizons = tuple(h for h in all_horizons if h != horizon)
    extra_targets = (
        save_shifted_targets(
            returns_df=full_returns,
            horizons=extra_horizons,
            output_dir=output_dir,
            valid_index=X_ml.index,
        )
        if extra_horizons
        else {}
    )
    dl_df = prepare_dl_data(
        prices_df=full_prices,
        returns_df=full_returns,
        output_dir=output_dir,
    )

    logger.info("ML matrix shape: {shape}", shape=X_ml.shape)
    logger.info("DL matrix shape: {shape}", shape=dl_df.shape)

    return {
        "X_ml": X_ml,
        "y_ml": y_ml,
        "targets_ml": extra_targets,
        "dl_df": dl_df,
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
