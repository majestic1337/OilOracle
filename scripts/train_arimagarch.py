"""Train ARIMA-GARCH model with Walk-Forward Validation."""

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
    load_runtime_config,
    load_stat_data,
    resolve_data_dir,
)
from src.models.statistical import ARIMAGARCHModel  # noqa: E402
from src.pipelines.wfv_orchestrator import run_wfv, save_wfv_results  # noqa: E402


def run_training(
    horizon: int,
    data_dir: str | Path,
    config: str | Path | None,
    max_p: int | None = None,
    max_q: int | None = None,
) -> Path:
    """Run ARIMA-GARCH training script end-to-end."""
    resolved_data_dir = resolve_data_dir(data_dir)
    config_payload = load_runtime_config(config)
    model_cfg = get_model_config(config_payload, "arimagarch")

    p = max_p if max_p is not None else int(model_cfg.get("max_p", 5))
    q = max_q if max_q is not None else int(model_cfg.get("max_q", 2))

    model = ARIMAGARCHModel(max_p=p, max_q=q)
    wfv_config = build_wfv_config(config_payload, horizon=horizon, model_family="stat")

    try:
        X, y = load_stat_data(resolved_data_dir, horizon=horizon)
        iterations, predictions_df = run_wfv(X=X, y=y, model=model, config=wfv_config)
        
        save_wfv_results(
            iterations=iterations,
            predictions=predictions_df,
            model_name="arimagarch",
            output_dir=resolved_data_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ARIMA-GARCH training failed: {error}", error=str(exc))
        raise

    output_path = resolved_data_dir / f"predictions_arimagarch_{horizon}.parquet"
    logger.info("ARIMA-GARCH completed. Predictions: {path}", path=output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ARIMA-GARCH with WFV")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default="configs/wfv_config.json")
    parser.add_argument("--max-p", type=int, default=None)
    parser.add_argument("--max-q", type=int, default=None)
    args = parser.parse_args()

    # Логіка перевірки існування конфігу
    config_arg: str | None = args.config
    if config_arg and not Path(config_arg).exists() and not (PROJECT_ROOT / config_arg).exists():
        logger.warning(f"Config file {config_arg} not found; using defaults")
        config_arg = None

    run_training(
        horizon=args.horizon, 
        data_dir=args.data_dir, 
        config=config_arg,
        max_p=args.max_p,
        max_q=args.max_q
    )