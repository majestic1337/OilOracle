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
LAG_AIC_CACHE: dict[int, float] = {}
DEFAULT_HORIZONS: tuple[int, ...] = (1, 3, 5, 7)


def _validate_returns_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_RETURN_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required return columns: {missing}")



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
    """Generate lag/rolling predictors for tabular ML models."""
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    features = returns_df.copy()

    for col in returns_df.columns:
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
    """Prepare ML-ready feature matrix with shifted target.

    Steps:
    1) Build lag/rolling predictors.
    2) Shift target by -horizon for direct forecasting.
    3) Drop all rows containing NaN.
    4) Save `feature_matrix_ml.parquet` and `target_h{horizon}.parquet`.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    _validate_returns_columns(returns_df)

    features = _generate_ml_features(
        returns_df=returns_df,
        max_lag=max_lag,
        include_rolling=include_rolling,
    )
    target = returns_df["brent_return"].shift(-horizon).rename(f"target_h{horizon}")

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
) -> dict[int, pd.Series]:
    """Save shifted Brent targets for multiple direct-forecast horizons."""
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
        target = returns_df["brent_return"].shift(-horizon).rename(f"target_h{horizon}").dropna()
        target_path = output_path / f"target_h{horizon}.parquet"
        target.to_frame(name=f"target_h{horizon}").to_parquet(target_path)
        targets[horizon] = target
        logger.info("Saved shifted target for horizon h={h} to {path}", h=horizon, path=target_path)

    return targets



def prepare_dl_data(
    returns_df: pd.DataFrame,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Prepare DL-ready data for NeuralForecast.

    No lag generation, no rolling windows, and no target shift are applied.
    Output schema: [unique_id, ds, y, wti_return, dxy_return, gold_return].
    """
    _validate_returns_columns(returns_df)

    dl_df = pd.DataFrame(
        {
            "unique_id": "brent",
            "ds": pd.DatetimeIndex(returns_df.index),
            "y": returns_df["brent_return"],
            "wti_return": returns_df["wti_return"],
            "dxy_return": returns_df["dxy_return"],
            "gold_return": returns_df["gold_return"],
        },
        index=returns_df.index,
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



def run_feature_pipeline(
    processed_dir: str | Path = "data/processed",
    horizon: int = 1,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_search: int = 5,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run both ML and DL feature preparation branches."""
    output_dir = Path(processed_dir)
    train_df = pd.read_parquet(output_dir / "train_returns.parquet")
    val_df = pd.read_parquet(output_dir / "val_returns.parquet")
    test_df = pd.read_parquet(output_dir / "test_returns.parquet")

    full_returns = pd.concat([train_df, val_df, test_df]).sort_index()

    max_lag_order = determine_max_lag_order(train_df, max_search=max_search)
    save_lag_config(
        max_lag_order=max_lag_order,
        max_search=max_search,
        train_end_date=str(train_df.index.max().date()),
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
        )
        if extra_horizons
        else {}
    )
    dl_df = prepare_dl_data(returns_df=full_returns, output_dir=output_dir)

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
