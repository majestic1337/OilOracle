"""Temporal Fusion Transformer forecaster."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from neuralforecast.losses.pytorch import MQLoss

from src.models.dl_wrapper import DeepLearningForecasterWrapper


class TFTForecaster(DeepLearningForecasterWrapper):
    """Temporal Fusion Transformer (Lim et al., 2021) forecaster.

    TFT integrates a Variable Selection Network to weight exogenous inputs and
    provides native quantile forecasts (q10/q50/q90), eliminating the need for
    post-hoc calibration. In this experiment it is the only multivariate model
    that ingests all four series (brent, wti, dxy, gold) as exogenous features.
    """

    def __init__(self, horizon: int, input_size: int | None = None) -> None:
        if input_size is None:
            input_size = 4 * horizon

        try:
            from neuralforecast.models import TFT
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("neuralforecast is required for TFTForecaster") from exc

        loss = None
        try:
            from neuralforecast.losses.pytorch import QuantileLoss

            loss = MQLoss(level=[80, 90])
        except Exception as exc:  # noqa: BLE001 - optional dependency
            logger.warning(
                "QuantileLoss unavailable; TFT will output point forecasts: {error}",
                error=str(exc),
            )

        logger.info("Initializing TFT forecaster with horizon {h}", h=horizon)

        model_kwargs: dict[str, Any] = {
            "hidden_size": 64,
            "n_head": 4,
            "dropout": 0.1,
            "attn_dropout": 0.1,
            "max_steps": 300,
            "learning_rate": 1e-3,
            "val_check_steps": 50,
            "early_stop_patience_steps": 10,
            "random_seed": 42,
            "accelerator": "cpu",
        }
        signature = inspect.signature(TFT)
        if "num_encoder_layers" in signature.parameters:
            model_kwargs["num_encoder_layers"] = 2
        if loss is not None:
            model_kwargs["loss"] = loss

        super().__init__(
            model_class=TFT,
            horizon=horizon,
            input_size=input_size,
            **model_kwargs,
        )

    @staticmethod
    def _find_quantile_column(columns: list[str], quantile: float) -> str | None:
        q_str = f"{quantile:.1f}".rstrip("0").rstrip(".")
        q_int = int(round(quantile * 100))
        patterns = [
            f"q{q_str}",
            f"q{quantile}",
            f"q{q_int}",
            f"p{q_int}",
            f"quantile{q_str}",
        ]
        for col in columns:
            lower = col.lower()
            if any(pattern in lower for pattern in patterns):
                return col
        return None

    def predict(self, X: Any) -> np.ndarray:
        """Return the median (q50) forecast from the TFT quantile outputs."""
        forecast_df = self._predict_df(X)
        pred_columns = [col for col in forecast_df.columns if col not in {"unique_id", "ds"}]
        if not pred_columns:
            raise ValueError("No prediction columns returned by NeuralForecast")

        q50_col = self._find_quantile_column(pred_columns, 0.5)
        if q50_col is None:
            q50_col = pred_columns[0]
            logger.warning("q50 column not found; using {col}", col=q50_col)

        preds = forecast_df[q50_col].to_numpy(dtype=float)
        if len(preds) != len(X):
            logger.warning(
                "Prediction length {pred_len} differs from X length {x_len}; aligning",
                pred_len=len(preds),
                x_len=len(X),
            )
            if len(preds) > len(X):
                preds = preds[: len(X)]
            else:
                preds = np.pad(preds, (0, len(X) - len(preds)), mode="edge")

        return preds

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return native quantile intervals (q10, q90) from the TFT output."""
        if alpha != 0.1:
            logger.warning("TFT uses fixed quantiles; ignoring alpha override")

        forecast_df = self._predict_df(X)

        exclude = {"unique_id", "ds"}
        model_cols = [col for col in forecast_df.columns if col not in exclude]
        if not model_cols:
            raise ValueError("No forecast columns in neuralforecast output")

        q10_col = next((col for col in model_cols if "10" in col), None)
        q90_col = next((col for col in model_cols if "90" in col), None)
        if q10_col is None or q90_col is None:
            raise ValueError("Quantile outputs not found for TFT; ensure QuantileLoss is enabled")

        lower = forecast_df[q10_col].values[: len(X)]
        upper = forecast_df[q90_col].values[: len(X)]

        return lower, upper
