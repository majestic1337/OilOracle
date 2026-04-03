"""Machine-learning forecasters and adaptive conformal wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseForecaster, as_1d_array, as_2d_array, calculate_metrics

__all__ = [
    "XGBoostForecaster",
    "LightGBMForecaster",
    "AdaptiveConformalWrapper",
    "calculate_metrics",
]




class XGBoostForecaster(BaseForecaster):
    def __init__(self, random_state: int = 42, alpha: float = 0.1) -> None:
        self.random_state = random_state
        self.alpha = alpha
        self._logger = logger.bind(model=self.__class__.__name__)
        self.model_lower: Any | None = None
        self.model_q50: Any | None = None
        self.model_upper: Any | None = None
        self.feature_names_: list[str] | None = None

    def _build_model(self, quantile: float) -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required") from exc

        # Використання MSE для медіани (q50) для стабілізації в умовах шуму
        objective = "reg:squarederror" if quantile == 0.5 else "reg:quantileerror"
        
        return XGBRegressor(
            objective=objective,
            quantile_alpha=quantile if objective == "reg:quantileerror" else None,
            n_estimators=200,         
            learning_rate=0.05,       
            max_depth=2,              
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,            
            reg_lambda=1.0,           
            early_stopping_rounds=20,
            random_state=self.random_state,
        )

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "XGBoostForecaster":
        self.feature_names_ = X_train.columns.tolist() if hasattr(X_train, "columns") else None
        X_arr = as_2d_array(X_train)
        y_arr = as_1d_array(y_train)

        if X_val is None or y_val is None:
            # H8: Use last 10% as temporal validation (not random split)
            split_idx = int(len(X_arr) * 0.9)
            if split_idx < 1 or split_idx >= len(X_arr):
                self._logger.warning(
                    "Dataset too small for internal temporal split; using full set"
                )
                X_t, y_t = X_arr, y_arr
                X_v, y_v = X_arr[-1:], y_arr[-1:]
            else:
                self._logger.info(
                    "No validation data provided; using last 10%% as temporal hold-out"
                )
                X_t, y_t = X_arr[:split_idx], y_arr[:split_idx]
                X_v, y_v = X_arr[split_idx:], y_arr[split_idx:]
        else:
            X_t, y_t = X_arr, y_arr
            X_v, y_v = as_2d_array(X_val), as_1d_array(y_val)

        lower_q = self.alpha / 2.0
        upper_q = 1.0 - self.alpha / 2.0

        self.model_lower = self._build_model(lower_q)
        self.model_q50 = self._build_model(0.5)
        self.model_upper = self._build_model(upper_q)

        eval_set = [(X_v, y_v)]
        
        self.model_lower.fit(X_t, y_t, eval_set=eval_set, verbose=False)
        self.model_q50.fit(X_t, y_t, eval_set=eval_set, verbose=False)
        self.model_upper.fit(X_t, y_t, eval_set=eval_set, verbose=False)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model_q50 is None:
            raise ValueError("XGBoostForecaster must be fitted before prediction")
            
        X_input = X
        if not isinstance(X_input, pd.DataFrame) and self.feature_names_:
            X_input = pd.DataFrame(X_input, columns=self.feature_names_)
            
        return self.model_q50.predict(as_2d_array(X_input)).astype(float)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        if self.model_lower is None or self.model_upper is None:
            raise ValueError("XGBoostForecaster must be fitted before interval prediction")
            
        X_arr = as_2d_array(X)
        lower = self.model_lower.predict(X_arr).astype(float)
        upper = self.model_upper.predict(X_arr).astype(float)
        return lower, upper

class LightGBMForecaster(BaseForecaster):
    def __init__(self, random_state: int = 42, alpha: float = 0.1) -> None:
        self.random_state = random_state
        self.alpha = alpha
        self._logger = logger.bind(model=self.__class__.__name__)
        self.model_lower: Any | None = None
        self.model_q50: Any | None = None
        self.model_upper: Any | None = None
        self.feature_names_: list[str] | None = None

    def _build_model(self, quantile: float) -> Any:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required") from exc

        # Використання MSE для медіани (q50) для стабілізації в умовах шуму
        objective = "regression" if quantile == 0.5 else "quantile"
        
        return LGBMRegressor(
            objective=objective,
            alpha=quantile if objective == "quantile" else None,
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
        self.feature_names_ = X_train.columns.tolist() if hasattr(X_train, "columns") else None
        X_arr = as_2d_array(X_train)
        y_arr = as_1d_array(y_train)

        lower_q = self.alpha / 2.0
        upper_q = 1.0 - self.alpha / 2.0

        self.model_lower = self._build_model(lower_q)
        self.model_q50 = self._build_model(0.5)
        self.model_upper = self._build_model(upper_q)

        # C3: Apply early stopping to ALL quantile models consistently
        if X_val is not None and y_val is not None:
            X_v, y_v = as_2d_array(X_val), as_1d_array(y_val)
            try:
                from lightgbm import early_stopping, log_evaluation
                es_callbacks = [early_stopping(50), log_evaluation(0)]
                self.model_lower.fit(
                    X_arr, y_arr, eval_set=[(X_v, y_v)], callbacks=es_callbacks
                )
                self.model_q50.fit(
                    X_arr, y_arr, eval_set=[(X_v, y_v)], callbacks=es_callbacks
                )
                self.model_upper.fit(
                    X_arr, y_arr, eval_set=[(X_v, y_v)], callbacks=es_callbacks
                )
            except (ImportError, Exception):
                self.model_lower.fit(X_arr, y_arr)
                self.model_q50.fit(X_arr, y_arr)
                self.model_upper.fit(X_arr, y_arr)
        else:
            self.model_lower.fit(X_arr, y_arr)
            self.model_q50.fit(X_arr, y_arr)
            self.model_upper.fit(X_arr, y_arr)

        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.model_q50 is None:
            raise ValueError("Model not fitted")
        
        X_input = X
        if not isinstance(X_input, pd.DataFrame) and self.feature_names_:
            X_input = pd.DataFrame(X_input, columns=self.feature_names_)
        
        return self.model_q50.predict(X_input).astype(float)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        if self.model_lower is None or self.model_upper is None:
            raise ValueError("Interval models not fitted")
            
        X_arr = as_2d_array(X)
        lower = self.model_lower.predict(X_arr).astype(float)
        upper = self.model_upper.predict(X_arr).astype(float)
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
        y_cal_arr = as_1d_array(y_cal)
        y_pred_arr = as_1d_array(y_pred_cal)
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

        preds = as_1d_array(self.base_model.predict(X))
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
            y_pred_arr = as_1d_array(y_pred)

        y_true_arr = as_1d_array(y_true)
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        # C4: Correct ACI update (Gibbs & Candès, 2021).
        # q̂_{t+1} = q̂_t + γ(α − 𝟙{|e_t| > q̂_t})
        for err in np.abs(y_true_arr - y_pred_arr):
            indicator = 1.0 if err > self.q_hat else 0.0
            self.q_hat = self.q_hat + self.gamma * (self.alpha - indicator)
