"""Train TFT model with Walk-Forward Validation."""

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
    load_dl_unshifted_data,
    load_ml_data,
    load_runtime_config,
    resolve_data_dir,
)
from src.models.tft_model import TFTForecaster  # noqa: E402
from src.pipelines.wfv_orchestrator import run_wfv, save_wfv_results  # noqa: E402


def run_training(
    horizon: int,
    data_dir: str | Path,
    config: str | Path | None,
    input_size: int | None = 30,
    max_steps: int | None = None,
    learning_rate: float | None = None,
    scaler_type: str | None = None,
    local_scaler_type: str | None = None,
) -> Path:
    """Run TFT training script end-to-end."""
    resolved_data_dir = resolve_data_dir(data_dir)
    config_payload = load_runtime_config(config)
    model_cfg = get_model_config(config_payload, "tft")

    resolved_input_size = input_size
    if resolved_input_size is None and "input_size" in model_cfg:
        resolved_input_size = int(model_cfg["input_size"])
    if resolved_input_size is None:
        resolved_input_size = 30

    resolved_max_steps = int(max_steps if max_steps is not None else model_cfg.get("max_steps", 50))
    resolved_learning_rate = float(
        learning_rate if learning_rate is not None else model_cfg.get("learning_rate", 1e-3)
    )
    cfg_scaler_type = model_cfg.get("scaler_type")
    if scaler_type is not None:
        resolved_scaler_type = str(scaler_type)
    elif cfg_scaler_type is not None:
        resolved_scaler_type = str(cfg_scaler_type)
    else:
        resolved_scaler_type = "standard"

    resolved_local_scaler_type = local_scaler_type
    if resolved_local_scaler_type is None:
        # For stationary log-returns, keep NeuralForecast local scaler disabled by default.
        resolved_local_scaler_type = None
    if isinstance(resolved_local_scaler_type, str) and resolved_local_scaler_type.lower() == "none":
        resolved_local_scaler_type = None

    model = TFTForecaster(
        horizon=horizon,
        input_size=resolved_input_size,
        max_steps=resolved_max_steps,
        learning_rate=resolved_learning_rate,
        scaler_type=resolved_scaler_type,
        local_scaler_type=resolved_local_scaler_type,
    )
    wfv_config = build_wfv_config(config_payload, horizon=horizon, model_family="dl")
    # TFT is trained in direct multi-step mode; recursive DL path expects an already-fitted model.
    wfv_config.dl_recursive = False

    try:
        # Load unshifted target: NeuralForecast MIMO projects horizon via h param
        X, y = load_dl_unshifted_data(resolved_data_dir)
        # Load shifted target for evaluation metrics only
        _, y_eval = load_ml_data(resolved_data_dir, horizon=horizon)
        iterations, predictions_df = run_wfv(
            X=X, 
            y=y, 
            model=model, 
            config=wfv_config,
            model_name="tft",
            output_dir=resolved_data_dir,
            y_eval=y_eval,
        )
        save_wfv_results(
            iterations=iterations,
            predictions=predictions_df,
            model_name="tft",
            output_dir=resolved_data_dir,
        )
    except Exception as exc:  # noqa: BLE001 - script-level safety
        logger.exception("TFT training failed: {error}", error=str(exc))
        raise

    output_path = resolved_data_dir / f"predictions_tft_{horizon}.parquet"
    logger.info("TFT training completed. Predictions: {path}", path=output_path)
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TFT with WFV")
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
        "--input-size",
        type=int,
        default=None,
        help="TFT input_size override",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max training steps (quick-test default is 50)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="TFT learning rate override",
    )
    parser.add_argument(
        "--scaler-type",
        type=str,
        default=None,
        help="Model-level scaler type (e.g., standard)",
    )
    parser.add_argument(
        "--local-scaler-type",
        type=str,
        default=None,
        help="NeuralForecast local scaler type (use 'none' to disable)",
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
        input_size=args.input_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        scaler_type=args.scaler_type,
        local_scaler_type=args.local_scaler_type,
    )
