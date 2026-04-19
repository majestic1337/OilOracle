"""Train XGBoost model with Walk-Forward Validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_base import (  # noqa: E402
    build_wfv_config,
    get_model_config,
    load_ml_data,
    load_runtime_config,
    resolve_config_path,
    resolve_data_dir,
)
from src.models.ml_models import XGBoostForecaster  # noqa: E402
from src.pipelines.wfv_orchestrator import run_wfv, save_wfv_results  # noqa: E402


def run_training(
    horizon: int,
    data_dir: str | Path,
    config: str | Path | None,
    max_lag: int = 5,
    random_state: int | None = None,
) -> Path:
    """Run XGBoost training script end-to-end."""
    resolved_data_dir = resolve_data_dir(data_dir)
    config_payload = load_runtime_config(config)
    model_cfg = get_model_config(config_payload, "xgboost")

    resolved_random_state = (
        random_state if random_state is not None else int(model_cfg.get("random_state", 42))
    )

    wfv_config = build_wfv_config(config_payload, horizon=horizon, model_family="ml")
    
    base_name = "xgboost"
    model_name_resolved = f"{base_name}_cls" if wfv_config.task_type == "classification" else base_name
    
    model = XGBoostForecaster(random_state=resolved_random_state, task_type=wfv_config.task_type)

    try:
        X_train, y_train = load_ml_data(resolved_data_dir, horizon=horizon, split_name="train", max_lag=max_lag)
        X_val, y_val = load_ml_data(resolved_data_dir, horizon=horizon, split_name="val", max_lag=max_lag)
        X_test, y_test = load_ml_data(resolved_data_dir, horizon=horizon, split_name="test", max_lag=max_lag)
        X = pd.concat([X_train, X_val, X_test]).sort_index()
        y = pd.concat([y_train, y_val, y_test]).sort_index()

        X = X[~X.index.duplicated(keep="first")].sort_index()
        y = y[~y.index.duplicated(keep="first")].sort_index()

        # Drop boundary NaN from shift(-h) at split edges
        y = y.dropna()
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index].dropna()
        y = y.loc[X.index]
        logger.info(
            "Loaded full dataset: {n} rows, {start} -> {end}",
            n=len(X),
            start=X.index.min().date(),
            end=X.index.max().date(),
        )
        iterations, predictions_df = run_wfv(X=X, y=y, model=model, config=wfv_config)
        save_wfv_results(
            iterations=iterations,
            predictions=predictions_df,
            model_name=model_name_resolved,
            output_dir=resolved_data_dir,
        )
    except Exception as exc:  # noqa: BLE001 - script-level safety
        logger.exception("XGBoost training failed: {error}", error=str(exc))
        raise

    output_path = resolved_data_dir / f"predictions_{model_name_resolved}_{horizon}.parquet"
    logger.info("XGBoost training completed. Predictions: {path}", path=output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XGBoost with WFV")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (h)")
    parser.add_argument("--max-lag", type=int, default=5, help="Max lag order used during feature engineering")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Processed data directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/wfv_config.json",
        help="Path to JSON/YAML config file",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="XGBoost random state override",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config_arg = resolve_config_path(args.config)

    run_training(
        horizon=args.horizon,
        data_dir=args.data_dir,
        config=config_arg,
        max_lag=args.max_lag,
        random_state=args.random_state,
    )
