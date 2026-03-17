"""Feature engineering pipeline for return series."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from loguru import logger
from statsmodels.tsa.ar_model import AutoReg

ROLLING_MEAN_WINDOWS: tuple[int, ...] = (5, 10, 20)
ROLLING_STD_WINDOWS: tuple[int, ...] = (5, 20)
LAG_AIC_CACHE: dict[int, float] = {}


def generate_lag_features(
    df: pd.DataFrame,
    max_lag: int,
    include_rolling: bool = True,
) -> pd.DataFrame:
    """Generate lag and rolling window features.

    Args:
        df: Returns DataFrame with columns of asset returns.
        max_lag: Maximum lag order to include.
        include_rolling: Whether to include rolling mean/std features.

    Returns:
        DataFrame containing original columns plus lag and rolling features.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    features = df.copy()

    for col in df.columns:
        for lag in range(1, max_lag + 1):
            features[f"{col}_lag{lag}"] = df[col].shift(lag)

        if include_rolling:
            for window in ROLLING_MEAN_WINDOWS:
                features[f"{col}_rollmean{window}"] = (
                    df[col].rolling(window=window).mean().shift(1)
                )
            for window in ROLLING_STD_WINDOWS:
                features[f"{col}_rollstd{window}"] = (
                    df[col].rolling(window=window).std().shift(1)
                )

    features = features.dropna()
    return features


def _compute_aic_by_lag(train_df: pd.DataFrame, max_search: int) -> dict[int, float]:
    """Compute AIC values for AR(p) models on brent_return.

    Args:
        train_df: Training DataFrame containing brent_return.
        max_search: Maximum AR order to evaluate.

    Returns:
        Dictionary mapping lag order to AIC value.
    """
    if "brent_return" not in train_df.columns:
        raise ValueError("train_df must contain 'brent_return' column")

    series = train_df["brent_return"].dropna()
    if series.empty:
        raise ValueError("train_df['brent_return'] is empty after dropping NaN")

    aic_values: dict[int, float] = {}
    for lag in range(1, max_search + 1):
        try:
            model = AutoReg(series, lags=lag, old_names=False).fit()
        except Exception as exc:  # noqa: BLE001 - continue on failed lag fits
            logger.warning("Failed to fit AR({lag}): {error}", lag=lag, error=str(exc))
            continue

        aic_values[lag] = float(model.aic)
        logger.info("AIC for lag {lag}: {aic}", lag=lag, aic=aic_values[lag])

    if not aic_values:
        raise ValueError("No AR models were successfully fit for AIC evaluation")

    return aic_values


def determine_max_lag_order(train_df: pd.DataFrame, max_search: int = 20) -> int:
    """Determine optimal lag order using AIC on the training segment.

    Args:
        train_df: Training DataFrame containing brent_return.
        max_search: Maximum AR order to evaluate.

    Returns:
        Lag order that minimizes AIC.
    """
    global LAG_AIC_CACHE
    LAG_AIC_CACHE = _compute_aic_by_lag(train_df, max_search=max_search)

    best_lag = min(LAG_AIC_CACHE, key=LAG_AIC_CACHE.get)
    logger.info(
        "Selected max lag order {lag} with AIC {aic}",
        lag=best_lag,
        aic=LAG_AIC_CACHE[best_lag],
    )
    return best_lag


def build_target_vectors(
    returns_df: pd.DataFrame,
    horizons: Sequence[int] | None = None,
    strategy: str = "direct",
) -> dict[str, pd.Series | pd.DataFrame]:
    """Build target vectors for forecasting horizons.

    Args:
        returns_df: DataFrame of returns with brent_return column.
        horizons: Sequence of forecast horizons.
        strategy: 'direct' for separate targets, 'mimo' for a joint target.

    Returns:
        Dictionary of target series/dataframes keyed by target name.
    """
    if "brent_return" not in returns_df.columns:
        raise ValueError("returns_df must contain 'brent_return' column")

    horizons = list(horizons or [1, 7])
    if any(h < 1 for h in horizons):
        raise ValueError("All horizons must be >= 1")

    if strategy == "direct":
        targets: dict[str, pd.Series] = {}
        for horizon in horizons:
            shifted = returns_df["brent_return"].shift(-horizon)
            targets[f"target_h{horizon}"] = shifted.dropna()
        return targets

    if strategy == "mimo":
        target_frame = pd.DataFrame(
            {
                f"h{horizon}": returns_df["brent_return"].shift(-horizon)
                for horizon in horizons
            }
        )
        target_frame = target_frame.dropna()
        return {"target_mimo": target_frame}

    raise ValueError("strategy must be either 'direct' or 'mimo'")


def save_feature_matrix(
    feature_df: pd.DataFrame,
    targets: dict[str, pd.Series | pd.DataFrame],
    output_dir: str | Path,
    lag_config: dict[str, Any],
) -> None:
    """Save feature matrix, targets, and lag configuration.

    Args:
        feature_df: Feature matrix to save.
        targets: Dictionary of target series/dataframes.
        output_dir: Directory for processed data artifacts.
        lag_config: Configuration describing lag order selection.

    Returns:
        None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    feature_path = output_path / "feature_matrix.parquet"
    feature_df.to_parquet(feature_path)
    logger.info("Saved feature matrix to {path}", path=feature_path)

    config_path = Path("configs") / "lag_order_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(lag_config, handle, indent=2)
    logger.info("Saved lag order config to {path}", path=config_path)

    for name, target in targets.items():
        target_path = output_path / f"{name}.parquet"
        if isinstance(target, pd.Series):
            target.to_frame(name=name).to_parquet(target_path)
        else:
            target.to_parquet(target_path)
        logger.info("Saved target {name} to {path}", name=name, path=target_path)


if __name__ == "__main__":
    processed_dir = Path("data/processed")

    train_df = pd.read_parquet(processed_dir / "train_returns.parquet")
    val_df = pd.read_parquet(processed_dir / "val_returns.parquet")
    test_df = pd.read_parquet(processed_dir / "test_returns.parquet")

    full_returns = pd.concat([train_df, val_df, test_df]).sort_index()

    max_lag_order = determine_max_lag_order(train_df, max_search=20)
    feature_matrix = generate_lag_features(
        full_returns,
        max_lag=max_lag_order,
        include_rolling=True,
    )

    logger.info("Feature matrix shape: {shape}", shape=feature_matrix.shape)
    logger.info(
        "First 5 feature names: {features}",
        features=list(feature_matrix.columns)[:5],
    )

    target_vectors = build_target_vectors(full_returns, horizons=[1, 7], strategy="direct")

    lag_config_payload = {
        "max_lag_order": max_lag_order,
        "determined_on": "train",
        "train_end_date": "2021-12-31",
        "aic_values": {str(lag): value for lag, value in LAG_AIC_CACHE.items()},
    }

    save_feature_matrix(
        feature_df=feature_matrix,
        targets=target_vectors,
        output_dir=processed_dir,
        lag_config=lag_config_payload,
    )
