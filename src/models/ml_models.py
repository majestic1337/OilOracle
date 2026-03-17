"""Machine-learning forecasters and adaptive conformal wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from src.models.base import BaseForecaster, calculate_metrics

__all__ = [
    "XGBoostForecaster",
    "LightGBMForecaster",
    "AdaptiveConformalWrapper",
    "calculate_metrics",
]


def _as_2d_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _as_1d_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        array = array.squeeze()
    return array


class XGBoostForecaster(BaseForecaster):
    """Quantile-regression XGBoost forecaster.

    We fit three separate quantile models (0.1/0.5/0.9) rather than an ensemble
    so each quantile can carry its own hyperparameters in future experiments.
    Quantile regression is used because it provides asymmetric uncertainty
    bounds that better capture heavy-tailed commodity return dynamics.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._logger = logger.bind(model=self.__class__.__name__)
        self.model_q10: Any | None = None
        self.model_q50: Any | None = None
        self.model_q90: Any | None = None

    def _build_model(self, quantile: float) -> Any:
        try:
            from xgboost import XGBRegressor
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("xgboost is required for XGBoostForecaster") from exc

        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile,
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "XGBoostForecaster":
        """Fit three independent quantile models.

        Early stopping is applied only to the median model because it is the
        primary point forecast in the experimental pipeline.
        """
        self._logger.info("Fitting XGBoost quantile models (q10/q50/q90)")
        X_arr = _as_2d_array(X_train)
        y_arr = _as_1d_array(y_train)

        self.model_q10 = self._build_model(0.1)
        self.model_q50 = self._build_model(0.5)
        self.model_q90 = self._build_model(0.9)

        self.model_q10.fit(X_arr, y_arr)
        self.model_q90.fit(X_arr, y_arr)

        if X_val is not None and y_val is not None:
            X_val_arr = _as_2d_array(X_val)
            y_val_arr = _as_1d_array(y_val)
            self.model_q50.fit(
                X_arr,
                y_arr,
                eval_set=[(X_val_arr, y_val_arr)],
                early_stopping_rounds=50,
                verbose=False,
            )
        else:
            self.model_q50.fit(X_arr, y_arr)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the median (q50) forecast."""
        if self.model_q50 is None:
            raise ValueError("XGBoostForecaster must be fitted before prediction")
        return self.model_q50.predict(_as_2d_array(X)).astype(float)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return the 10th and 90th percentile forecasts as an interval."""
        if self.model_q10 is None or self.model_q90 is None:
            raise ValueError("XGBoostForecaster must be fitted before interval prediction")
        lower = self.model_q10.predict(_as_2d_array(X)).astype(float)
        upper = self.model_q90.predict(_as_2d_array(X)).astype(float)
        return lower, upper


class LightGBMForecaster(BaseForecaster):
    """Quantile-regression LightGBM forecaster.

    We train independent quantile estimators for 0.1/0.5/0.9 to avoid mixing
    quantile objectives in a single model; this keeps calibration explicit and
    allows later hyperparameter specialization per quantile.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._logger = logger.bind(model=self.__class__.__name__)
        self.model_q10: Any | None = None
        self.model_q50: Any | None = None
        self.model_q90: Any | None = None

    def _build_model(self, quantile: float) -> Any:
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("lightgbm is required for LightGBMForecaster") from exc

        return LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1,
        )

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "LightGBMForecaster":
        """Fit three independent quantile models with optional early stopping."""
        self._logger.info("Fitting LightGBM quantile models (q10/q50/q90)")
        X_arr = _as_2d_array(X_train)
        y_arr = _as_1d_array(y_train)

        self.model_q10 = self._build_model(0.1)
        self.model_q50 = self._build_model(0.5)
        self.model_q90 = self._build_model(0.9)

        self.model_q10.fit(X_arr, y_arr)
        self.model_q90.fit(X_arr, y_arr)

        if X_val is not None and y_val is not None:
            X_val_arr = _as_2d_array(X_val)
            y_val_arr = _as_1d_array(y_val)
            try:
                from lightgbm import early_stopping, log_evaluation
            except Exception as exc:  # noqa: BLE001 - optional dependency
                self._logger.warning("LightGBM callbacks unavailable: {error}", error=str(exc))
                self.model_q50.fit(X_arr, y_arr)
            else:
                self.model_q50.fit(
                    X_arr,
                    y_arr,
                    eval_set=[(X_val_arr, y_val_arr)],
                    callbacks=[early_stopping(50), log_evaluation(0)],
                )
        else:
            self.model_q50.fit(X_arr, y_arr)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return the median (q50) forecast."""
        if self.model_q50 is None:
            raise ValueError("LightGBMForecaster must be fitted before prediction")
        return self.model_q50.predict(_as_2d_array(X)).astype(float)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return the 10th and 90th percentile forecasts as an interval."""
        if self.model_q10 is None or self.model_q90 is None:
            raise ValueError("LightGBMForecaster must be fitted before interval prediction")
        lower = self.model_q10.predict(_as_2d_array(X)).astype(float)
        upper = self.model_q90.predict(_as_2d_array(X)).astype(float)
        return lower, upper


class AdaptiveConformalWrapper(BaseForecaster):
    """Adaptive conformal inference wrapper for point-forecast models.

    Adaptive Conformal Inference (Gibbs & Candes, 2021) updates interval widths
    online to stabilize coverage under distribution shifts, which is common in
    financial time series. It is especially useful for deep or black-box models
    that lack native uncertainty estimates. The wrapper stores internal state so
    interval width evolves across sequential calls.
    """

    def __init__(self, base_model: BaseForecaster, gamma: float = 0.05, alpha: float = 0.1) -> None:
        self.base_model = base_model
        self.gamma = gamma
        self.alpha = alpha
        self.q_hat: float | None = None
        self._calibrated = False
        self._last_predictions: np.ndarray | None = None
        self._logger = logger.bind(model=self.__class__.__name__)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "AdaptiveConformalWrapper":
        """Fit the wrapped base model."""
        self.base_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        return self

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> None:
        """Initialize conformal scores on a calibration set.

        Args:
            y_cal: Calibration targets.
            y_pred_cal: Point predictions for calibration targets.
        """
        y_cal_arr = _as_1d_array(y_cal)
        y_pred_arr = _as_1d_array(y_pred_cal)
        if y_cal_arr.size == 0:
            raise ValueError("Calibration targets are empty")
        if y_cal_arr.shape != y_pred_arr.shape:
            raise ValueError("Calibration targets and predictions must have the same shape")

        scores = np.abs(y_cal_arr - y_pred_arr)
        self.q_hat = float(np.quantile(scores, 1.0 - self.alpha))
        self._calibrated = True
        self._logger.info("ACI calibrated with Q_hat={q}", q=self.q_hat)

    def predict(self, X: Any) -> np.ndarray:
        """Return point forecasts from the base model."""
        return self.base_model.predict(X)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return adaptive conformal intervals.

        Note: The interval width is updated online via the `update` method after
        observing true outcomes. The state (Q_hat) persists between calls, so
        callers must manage sequential evaluation order.
        """
        if not self._calibrated or self.q_hat is None:
            raise ValueError("AdaptiveConformalWrapper must be calibrated before prediction")

        if alpha != self.alpha:
            self._logger.warning("Ignoring alpha override; using alpha={alpha}", alpha=self.alpha)

        preds = _as_1d_array(self.base_model.predict(X))
        self._last_predictions = preds
        lower = preds - self.q_hat
        upper = preds + self.q_hat
        return lower.astype(float), upper.astype(float)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray | None = None) -> None:
        """Update Q_hat using newly observed outcomes.

        Args:
            y_true: Observed targets aligned with previous predictions.
            y_pred: Optional point predictions; defaults to last predictions.
        """
        if not self._calibrated or self.q_hat is None:
            raise ValueError("AdaptiveConformalWrapper must be calibrated before update")

        if y_pred is None:
            if self._last_predictions is None:
                raise ValueError("No stored predictions; provide y_pred explicitly")
            y_pred_arr = self._last_predictions
        else:
            y_pred_arr = _as_1d_array(y_pred)

        y_true_arr = _as_1d_array(y_true)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        for err in np.abs(y_true_arr - y_pred_arr):
            indicator = 1.0 if err > self.q_hat else 0.0
            alpha_t = self.alpha + self.gamma * (self.alpha - indicator)
            self.q_hat = self.q_hat + self.gamma * (alpha_t - self.alpha)
