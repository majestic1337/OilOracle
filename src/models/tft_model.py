"""Temporal Fusion Transformer forecaster."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
try:
    from neuralforecast.losses.pytorch import MQLoss
except Exception:  # noqa: BLE001 - optional dependency
    MQLoss = None  # type: ignore[assignment]

from src.models.dl_wrapper import DeepLearningForecasterWrapper


class TFTForecaster(DeepLearningForecasterWrapper):
    """Temporal Fusion Transformer (Lim et al., 2021) forecaster.

    TFT integrates a Variable Selection Network to weight exogenous inputs and
    provides native quantile forecasts (q10/q50/q90), eliminating the need for
    post-hoc calibration. In this experiment it is the only multivariate model
    that ingests all four series (brent, wti, dxy, gold) as exogenous features.
    """

    def __init__(
        self,
        horizon: int,
        input_size: int = 30,
        max_steps: int = 50,
        learning_rate: float = 1e-3,
        scaler_type: str = "standard",
        local_scaler_type: str | None = None,
    ) -> None:
        self.max_steps = int(max_steps)
        self.learning_rate = float(learning_rate)
        self.scaler_type = scaler_type
        self.local_scaler_type = local_scaler_type
        self.horizon = horizon
        self.input_size = int(input_size)

        try:
            from neuralforecast.models import TFT
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("neuralforecast is required for TFTForecaster") from exc

        if MQLoss is None:
            raise ImportError(
                "TFTForecaster requires quantile-capable loss (MQLoss) "
                "to produce prediction intervals."
            )
        self.loss = MQLoss(level=[80])

        logger.info("Initializing TFT forecaster with horizon {h}", h=horizon)

        model_kwargs: dict[str, Any] = {
            "hidden_size": 8,
            "n_head": 4,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "scaler_type": self.scaler_type,
            "val_check_steps": 100,
            "early_stop_patience_steps": -1,
            "random_seed": 42,
            "loss": self.loss,
            "accelerator": "cpu",
        }
        signature = inspect.signature(TFT)
        if "num_encoder_layers" in signature.parameters:
            model_kwargs["num_encoder_layers"] = 2

        super().__init__(
            model_class=TFT,
            horizon=horizon,
            input_size=self.input_size,
            local_scaler_type=self.local_scaler_type,
            **model_kwargs,
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "input_size": self.input_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "scaler_type": self.scaler_type,
            "local_scaler_type": self.local_scaler_type,
        }

    def set_params(self, **params: Any) -> "TFTForecaster":
        for key, value in params.items():
            setattr(self, key, value)

        if hasattr(self, "model_kwargs"):
            for key in ("max_steps", "learning_rate", "scaler_type"):
                if key in params:
                    if key == "max_steps":
                        self.model_kwargs[key] = int(params[key])
                    elif key == "learning_rate":
                        self.model_kwargs[key] = float(params[key])
                    else:
                        self.model_kwargs[key] = params[key]
        return self

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

    def predict(
        self,
        X: Any,
        history_X: Any | None = None,
        history_y: Any | None = None,
    ) -> np.ndarray:
        """Return the median (q50) forecast from the TFT quantile outputs."""
        forecast_df = self._predict_df(X, history_X=history_X, history_y=history_y)
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

    def predict_interval(
        self,
        X: Any,
        alpha: float = 0.1,
        history_X: Any | None = None,
        history_y: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return native quantile intervals (q10, q90) from the TFT output."""
        if alpha != 0.1:
            logger.warning("TFT uses fixed quantiles; ignoring alpha override")

        forecast_df = self._predict_df(X, history_X=history_X, history_y=history_y)

        exclude = {"unique_id", "ds"}
        model_cols = [col for col in forecast_df.columns if col not in exclude]
        if not model_cols:
            raise ValueError("No forecast columns in neuralforecast output")

        q10_col = next(
            (
                col
                for col in model_cols
                if "lo-80" in col.lower() or "q.1" in col.lower() or "q10" in col.lower()
            ),
            None,
        )
        q90_col = next(
            (
                col
                for col in model_cols
                if "hi-80" in col.lower() or "q.9" in col.lower() or "q90" in col.lower()
            ),
            None,
        )
        if q10_col is None or q90_col is None:
            raise ValueError(
                f"Quantile columns not found in TFT output. "
                f"Expected patterns: 'lo-80'/'hi-80' or 'q10'/'q90'. "
                f"Available columns: {model_cols}"
            )

        lower = forecast_df[q10_col].values[: len(X)]
        upper = forecast_df[q90_col].values[: len(X)]

        return lower, upper
