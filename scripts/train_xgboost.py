"""Train XGBoost model with Walk-Forward Validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_base import (  # noqa: E402
    build_wfv_config,
    get_model_config,
    load_ml_data,
    load_runtime_config,
    resolve_data_dir,
)
from src.models.ml_models import XGBoostForecaster  # noqa: E402
from src.pipelines.wfv_orchestrator import run_wfv, save_wfv_results  # noqa: E402


def run_training(
    horizon: int,
    data_dir: str | Path,
    config: str | Path | None,
    random_state: int | None = None,
) -> Path:
    """Run XGBoost training script end-to-end."""
    resolved_data_dir = resolve_data_dir(data_dir)
    config_payload = load_runtime_config(config)
    model_cfg = get_model_config(config_payload, "xgboost")

    resolved_random_state = (
        random_state if random_state is not None else int(model_cfg.get("random_state", 42))
    )

    model = XGBoostForecaster(random_state=resolved_random_state)
    wfv_config = build_wfv_config(config_payload, horizon=horizon, model_family="ml")

    try:
        X, y = load_ml_data(resolved_data_dir, horizon=horizon)
        iterations, predictions_df = run_wfv(X=X, y=y, model=model, config=wfv_config)
        save_wfv_results(
            iterations=iterations,
            predictions=predictions_df,
            model_name="xgboost",
            output_dir=resolved_data_dir,
        )
    except Exception as exc:  # noqa: BLE001 - script-level safety
        logger.exception("XGBoost training failed: {error}", error=str(exc))
        raise

    output_path = resolved_data_dir / f"predictions_xgboost_{horizon}.parquet"
    logger.info("XGBoost training completed. Predictions: {path}", path=output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XGBoost with WFV")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (h)")
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
    config_arg: str | None = args.config
    if config_arg and not (PROJECT_ROOT / config_arg).exists() and not Path(config_arg).exists():
        config_arg = None
        logger.warning("Config file not found; proceeding with defaults and CLI overrides")

    run_training(
        horizon=args.horizon,
        data_dir=args.data_dir,
        config=config_arg,
        random_state=args.random_state,
    )
