"""Train PatchTST model with Walk-Forward Validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_base import (
    build_wfv_config,
    get_model_config,
    load_dl_unshifted_data,
    load_ml_data,
    load_runtime_config,
    resolve_data_dir,
)
from src.models.patchtst_model import PatchTSTForecaster
from src.pipelines.wfv_orchestrator import run_wfv, save_wfv_results


def run_training(
    horizon: int,
    data_dir: str | Path,
    config: str | Path | None,
    input_size: int | None = None,
    max_steps: int | None = None,
    learning_rate: float | None = None,
) -> Path:
    """Run PatchTST training script end-to-end."""
    resolved_data_dir = resolve_data_dir(data_dir)
    config_payload = load_runtime_config(config)
    model_cfg = get_model_config(config_payload, "patchtst")

    # Resolve parameters: CLI > Config > Default
    resolved_input_size = input_size or int(model_cfg.get("input_size", 64))
    resolved_max_steps = int(max_steps if max_steps is not None else model_cfg.get("max_steps", 100))
    resolved_learning_rate = float(
        learning_rate if learning_rate is not None else model_cfg.get("learning_rate", 1e-3)
    )

    # Initialize model
    model = PatchTSTForecaster(
        horizon=horizon,
        input_size=resolved_input_size,
        max_steps=resolved_max_steps,
        learning_rate=resolved_learning_rate,
    )

    wfv_config = build_wfv_config(config_payload, horizon=horizon, model_family="dl")

    try:
        # Load unshifted target: NeuralForecast MIMO projects horizon via h param
        X, y = load_dl_unshifted_data(resolved_data_dir)
        # Load shifted target for evaluation metrics only
        _, y_eval = load_ml_data(resolved_data_dir, horizon=horizon)
        
        logger.info(
            "Starting PatchTST WFV: horizon={h}, steps={s}, input={i}",
            h=horizon, s=resolved_max_steps, i=resolved_input_size
        )
        
        iterations, predictions_df = run_wfv(
            X=X, 
            y=y, 
            model=model, 
            config=wfv_config,
            model_name="patchtst",
            output_dir=resolved_data_dir,
            y_eval=y_eval,
        )
        
        save_wfv_results(
            iterations=iterations,
            predictions=predictions_df,
            model_name="patchtst",
            output_dir=resolved_data_dir,
        )
    except Exception as exc:
        logger.exception("PatchTST training failed: {error}", error=str(exc))
        raise

    output_path = resolved_data_dir / f"predictions_patchtst_{horizon}.parquet"
    logger.info("PatchTST completed. Predictions: {path}", path=output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PatchTST with WFV")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon (h)")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed", help="Processed data directory"
    )
    parser.add_argument(
        "--config", type=str, default="configs/wfv_config.json", help="WFV/Model config file"
    )
    parser.add_argument("--input-size", type=int, default=None, help="Input window size")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    
    config_arg: str | None = args.config
    if config_arg and not (PROJECT_ROOT / config_arg).exists() and not Path(config_arg).exists():
        logger.warning("Config file {} not found; using defaults", config_arg)
        config_arg = None

    run_training(
        horizon=args.horizon,
        data_dir=args.data_dir,
        config=config_arg,
        input_size=args.input_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )