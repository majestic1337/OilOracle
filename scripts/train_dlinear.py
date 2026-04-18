"""Train DLinear (AR-Linear) model with Walk-Forward Validation."""

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
    load_dl_unshifted_data,
    load_ml_data,
    load_runtime_config,
    resolve_config_path,
    resolve_data_dir,
)
from src.models.statistical import DLinearModel  # noqa: E402
from src.pipelines.wfv_orchestrator import run_wfv, save_wfv_results  # noqa: E402


def run_training(
    horizon: int,
    data_dir: str | Path,
    config: str | Path | None,
    max_lag: int = 5,
) -> Path:
    """Run DLinear training script end-to-end."""
    resolved_data_dir = resolve_data_dir(data_dir)
    config_payload = load_runtime_config(config)
    
    model = DLinearModel(horizon=horizon)
    wfv_config = build_wfv_config(config_payload, horizon=horizon, model_family="stat")
    
    base_name = "dlinear"
    model_name_resolved = f"{base_name}_cls" if wfv_config.task_type == "classification" else base_name

    try:
        # Raw unshifted brent_return for DLinear training
        X_train, y_train = load_dl_unshifted_data(resolved_data_dir, split_name="train", max_lag=max_lag)
        X_val, y_val = load_dl_unshifted_data(resolved_data_dir, split_name="val", max_lag=max_lag)
        X_test, y_test = load_dl_unshifted_data(resolved_data_dir, split_name="test", max_lag=max_lag)

        X = pd.concat([X_train, X_val, X_test]).sort_index()
        y = pd.concat([y_train, y_val, y_test]).sort_index()
        X = X[~X.index.duplicated(keep="first")].sort_index()
        y = y[~y.index.duplicated(keep="first")].sort_index()
        y = y.dropna()
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index].dropna()
        y = y.loc[X.index]

        # Shifted cumulative target for evaluation metrics only
        _, y_eval_train = load_ml_data(resolved_data_dir, horizon=horizon, split_name="train", max_lag=max_lag)
        _, y_eval_val = load_ml_data(resolved_data_dir, horizon=horizon, split_name="val", max_lag=max_lag)
        _, y_eval_test = load_ml_data(resolved_data_dir, horizon=horizon, split_name="test", max_lag=max_lag)
        y_eval = pd.concat([y_eval_train, y_eval_val, y_eval_test]).sort_index()
        y_eval = y_eval[~y_eval.index.duplicated(keep="first")].sort_index()
        y_eval = y_eval.dropna()

        logger.info(
            "Loaded full dataset: {n} rows, {start} -> {end}",
            n=len(X),
            start=X.index.min().date(),
            end=X.index.max().date(),
        )
        iterations, predictions_df = run_wfv(
            X=X, y=y, model=model, config=wfv_config, y_eval=y_eval,
        )
        
        save_wfv_results(
            iterations=iterations,
            predictions=predictions_df,
            model_name=model_name_resolved,
            output_dir=resolved_data_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("DLinear training failed: {error}", error=str(exc))
        raise

    output_path = resolved_data_dir / f"predictions_{model_name_resolved}_{horizon}.parquet"
    logger.info("DLinear training completed. Predictions: {path}", path=output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DLinear with WFV")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="configs/wfv_config.json")
    parser.add_argument("--max-lag", type=int, default=5, help="Max lag order used during feature engineering")
    args = parser.parse_args()

    config_arg = resolve_config_path(args.config)

    run_training(
        horizon=args.horizon,
        data_dir=args.data_dir,
        config=config_arg,
        max_lag=args.max_lag,
    )