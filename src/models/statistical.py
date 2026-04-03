"""Statistical baselines for oil price forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .base import BaseForecaster, as_1d_array


def _first_column_series(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        series = X.iloc[:, 0]
        return series.to_numpy(dtype=float)
    array = np.asarray(X, dtype=float)
    if array.ndim > 1:
        return array[:, 0]
    return array


class RandomWalkModel(BaseForecaster):
    """Random-walk baseline for stationary financial returns.

    For log-returns, the naive forecast assumes the expected return is zero
    (price remains unchanged). Prediction intervals are based on recent historical volatility.
    """

    def __init__(self, window: int = 20, task_type: str = "regression") -> None:
        self.window = window
        self.task_type = task_type
        self._expected_return: float = 0.0
        self._rolling_std: float | None = None
        self._logger = logger.bind(model=self.__class__.__name__)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "RandomWalkModel":
        series = np.asarray(y_train, dtype=float)
        if series.ndim > 1:
            series = series.squeeze()
            
        if series.size == 0:
            raise ValueError("y_train is empty; cannot fit RandomWalkModel")

        # True random walk on stationary log-returns: naive forecast is zero.
        self._expected_return = 0.0
        
        window = min(self.window, len(series))
        if window < 2:
            self._logger.warning("Insufficient samples to estimate rolling std")
            self._rolling_std = float("nan")
        else:
            self._rolling_std = float(np.std(series[-window:], ddof=1))

        return self

    def predict(self, X: Any) -> np.ndarray:
        steps = len(X) if hasattr(X, "__len__") else 1
        return np.full(steps, self._expected_return, dtype=float)

    def predict_interval(self, X: Any, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        steps = len(X) if hasattr(X, "__len__") else 1
        std = float(self._rolling_std) if self._rolling_std is not None else float("nan")
        
        z = float(norm.ppf(1.0 - alpha / 2.0))
        
        lower = np.full(steps, self._expected_return - z * std, dtype=float)
        upper = np.full(steps, self._expected_return + z * std, dtype=float)
        return lower, upper
    

class DLinearModel(BaseForecaster):
    """Linear autoregressive baseline for log-returns.

    On stationary log-returns the decomposition component of DLinear collapses,
    so the model effectively acts as a linear AR(p) layer. We keep it as a
    transparent baseline that is easy to interpret and compare.
    """

    def __init__(self, task_type: str = "regression") -> None:
        self.task_type = task_type
        self.model = LinearRegression()
        self._residual_std: float | None = None
        self._logger = logger.bind(model=self.__class__.__name__)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "DLinearModel":
        """Fit the linear regression and cache residual volatility."""
        X_arr = np.asarray(X_train)
        y_arr = as_1d_array(y_train)

        if X_val is not None and y_val is not None:
            X_val_arr = np.asarray(X_val)
            y_val_arr = as_1d_array(y_val)
            X_arr = np.concatenate([X_arr, X_val_arr], axis=0)
            y_arr = np.concatenate([y_arr, y_val_arr], axis=0)

        self.model.fit(X_arr, y_arr)
        residuals = y_arr - self.model.predict(X_arr)
        if residuals.size < 2:
            self._logger.warning("Insufficient samples to estimate residual std")
            self._residual_std = float("nan")
        else:
            self._residual_std = float(np.std(residuals, ddof=1))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return linear point forecasts."""
        X_arr = np.asarray(X)
        return self.model.predict(X_arr).astype(float)

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Return symmetric intervals based on training residual variance."""
        preds = self.predict(X)
        std = float(self._residual_std) if self._residual_std is not None else float("nan")
        z = float(norm.ppf(1.0 - alpha / 2.0))
        lower = preds - z * std
        upper = preds + z * std
        return lower, upper


class ARIMAGARCHModel(BaseForecaster):
    """ARIMA mean model with GARCH(1,1) volatility on residuals.

    This univariate baseline ignores multivariate features and relies on the
    first column of X (brent_return lags) to keep the forecasting exercise
    strictly econometric. The model combines conditional mean (ARIMA) with
    conditional variance (GARCH) to emulate classic commodity return models.
    """

    def __init__(
        self,
        max_p: int = 5,
        max_q: int = 2,
        random_state: int = 42,
        horizon: int = 1,
        task_type: str = "regression",
    ) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.random_state = random_state
        self.horizon = horizon
        self.task_type = task_type
        self.arima_model: Any | None = None
        self.garch_model: Any | None = None
        self.garch_scale_: float = 1.0
        self._fit_success = False
        self._fallback = RandomWalkModel(task_type=task_type)
        self._logger = logger.bind(model=self.__class__.__name__)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "ARIMAGARCHModel":
        """Fit ARIMA on the first column and GARCH on its residuals.

        We fix d=0 because the pipeline operates on stationary log-returns.
        If the ARIMA or GARCH fails to converge, we degrade gracefully to a
        random-walk forecast while preserving the model interface.
        """
        X_fit = X_train
        y_fit = y_train
        if X_val is not None and y_val is not None:
            if isinstance(X_train, pd.DataFrame) and isinstance(X_val, pd.DataFrame):
                X_fit = pd.concat([X_train, X_val], axis=0).sort_index()
            else:
                X_fit = np.concatenate([np.asarray(X_train), np.asarray(X_val)], axis=0)

            if isinstance(y_train, pd.Series) and isinstance(y_val, pd.Series):
                y_fit = pd.concat([y_train, y_val], axis=0).sort_index()
            elif isinstance(y_train, pd.DataFrame) and isinstance(y_val, pd.DataFrame):
                y_fit = pd.concat([y_train, y_val], axis=0).sort_index()
            else:
                y_fit = np.concatenate([as_1d_array(y_train), as_1d_array(y_val)], axis=0)

        series = np.asarray(y_fit, dtype=float).squeeze()
        if series.size == 0:
            raise ValueError("y_train is empty; cannot fit ARIMAGARCHModel")

        try:
            self._fallback.fit(X_fit, y_fit)
        except Exception as exc:  # noqa: BLE001 - fallback must never block ARIMA path
            self._logger.warning(
                "RandomWalk fallback fit failed on raw inputs; retrying with sanitized series: {error}",
                error=str(exc),
            )
            try:
                fallback_X = np.zeros((series.size, 1), dtype=float)
                self._fallback.fit(fallback_X, series)
            except Exception as retry_exc:  # noqa: BLE001
                self._logger.warning(
                    "RandomWalk fallback fit retry failed; predictions may use default state: {error}",
                    error=str(retry_exc),
                )

        try:
            from pmdarima import auto_arima
        except Exception as exc:  # noqa: BLE001 - optional dependency
            self._logger.warning(
                "auto_arima unavailable; fallback to RandomWalk: {error}",
                error=str(exc),
            )
            self._fit_success = False
            return self

        try:
            arima_model = auto_arima(
                series,
                d=0,
                max_p=self.max_p,
                max_q=self.max_q,
                seasonal=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                random_state=self.random_state,
            )

            if hasattr(arima_model, "resid"):
                residuals = np.asarray(arima_model.resid())
            else:
                residuals = np.asarray(arima_model.arima_res_.resid)

            try:
                from arch import arch_model
            except Exception as exc:  # noqa: BLE001 - optional dependency
                self._logger.warning(
                    "arch unavailable; fallback to RandomWalk: {error}",
                    error=str(exc),
                )
                self._fit_success = False
                return self

            GARCH_SCALE = 10.0  # log-returns ~0.01-0.03, після x10 = 0.1-0.3
            # Це в межах допустимого для GARCH

            resids_scaled = residuals * GARCH_SCALE

            # Навчаємо GARCH на масштабованих residuals
            garch = arch_model(resids_scaled, vol="Garch", p=1, q=1, dist="normal")
            garch_fit = garch.fit(disp="off")
        except Exception as exc:  # noqa: BLE001 - convergence or fit errors
            self._logger.warning(
                "ARIMA/GARCH fit failed; fallback to RandomWalk: {error}",
                error=str(exc),
            )
            self._fit_success = False
            return self

        self.arima_model = arima_model
        self.garch_model = garch_fit
        self.garch_scale_ = GARCH_SCALE
        self._fit_success = True
        return self

    def _forecast_mean(self, steps: int) -> np.ndarray:
        if not self.arima_model:
            raise ValueError("ARIMA model is not fitted")
        return np.asarray(
            self.arima_model.predict(n_periods=steps),
            dtype=float,
        )

    def _forecast_variance(self, steps: int) -> np.ndarray:
        if not self.garch_model:
            raise ValueError("GARCH model is not fitted")

        forecast = self.garch_model.forecast(horizon=steps, reindex=False)
        variance_series = forecast.variance.iloc[-1]
        variance_array = np.asarray(variance_series, dtype=float).squeeze()
        if variance_array.ndim == 0:
            return np.asarray([float(variance_array)], dtype=float)
        if variance_array.ndim > 1:
            return np.asarray(variance_array, dtype=float).reshape(-1)
        return variance_array

    def predict(self, X: Any) -> np.ndarray:
        """Forecast conditional mean; fallback to RandomWalk on failure.

        In direct-forecasting mode with horizon h and test length n:
        1) forecast total_steps = (h - 1) + n
        2) drop the first (h - 1) embargo steps
        3) keep the last n values aligned with X_test indices
        """
        steps = len(X) if hasattr(X, "__len__") else 1
        if not self._fit_success:
            return self._fallback.predict(X)

        try:
            total_steps = max((self.horizon - 1) + steps, 1)
            mean_forecast = self._forecast_mean(total_steps)
            mean_forecast = np.asarray(mean_forecast, dtype=float).squeeze()
            if mean_forecast.ndim == 0:
                mean_forecast = np.asarray([float(mean_forecast)], dtype=float)
            elif mean_forecast.ndim > 1:
                mean_forecast = np.asarray(mean_forecast, dtype=float).reshape(-1)

            if mean_forecast.size < steps:
                if mean_forecast.size == 0:
                    return np.full(steps, np.nan, dtype=float)
                return np.pad(mean_forecast, (0, steps - mean_forecast.size), mode="edge")
            return mean_forecast[-steps:].astype(float)
        except Exception as exc:  # noqa: BLE001 - robust fallback
            self._logger.warning(
                "ARIMA predict failed; fallback to RandomWalk: {error}",
                error=str(exc),
            )
            return self._fallback.predict(X)

    def predict_interval(
        self,
        X: Any,
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return analytic intervals using GARCH conditional variance."""
        steps = len(X) if hasattr(X, "__len__") else 1
        if not self._fit_success:
            return self._fallback.predict_interval(X, alpha=alpha)

        try:
            total_steps = max((self.horizon - 1) + steps, 1)
            mean = self._forecast_mean(total_steps)
            variance = self._forecast_variance(total_steps)
            mean = np.asarray(mean, dtype=float).squeeze()
            variance = np.asarray(variance, dtype=float).squeeze()
            if mean.ndim == 0:
                mean = np.asarray([float(mean)], dtype=float)
            elif mean.ndim > 1:
                mean = np.asarray(mean, dtype=float).reshape(-1)
            if variance.ndim == 0:
                variance = np.asarray([float(variance)], dtype=float)
            elif variance.ndim > 1:
                variance = np.asarray(variance, dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - robust fallback
            self._logger.warning(
                "GARCH interval failed; fallback to RandomWalk: {error}",
                error=str(exc),
            )
            return self._fallback.predict_interval(X, alpha=alpha)

        z = float(norm.ppf(1.0 - alpha / 2.0))
        garch_scale = self.garch_scale_ if self.garch_scale_ else 1.0
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance) / garch_scale
        if mean.size < steps:
            if mean.size == 0:
                mean = np.full(steps, np.nan, dtype=float)
            else:
                mean = np.pad(mean, (0, steps - mean.size), mode="edge")
        else:
            mean = mean[-steps:]

        if std.size < steps:
            if std.size == 0:
                std = np.full(steps, np.nan, dtype=float)
            else:
                std = np.pad(std, (0, steps - std.size), mode="edge")
        else:
            std = std[-steps:]

        lower = mean - z * std
        upper = mean + z * std
        return lower, upper
