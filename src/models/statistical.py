"""Statistical baselines for oil price forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from .base import BaseForecaster


def _as_1d_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        array = array.squeeze()
    return array


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

    def __init__(self, window: int = 20) -> None:
        self.window = window
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
        
        z = 1.96
        
        lower = np.full(steps, self._expected_return - z * std, dtype=float)
        upper = np.full(steps, self._expected_return + z * std, dtype=float)
        return lower, upper
    

class DLinearModel(BaseForecaster):
    """Linear autoregressive baseline for log-returns.

    On stationary log-returns the decomposition component of DLinear collapses,
    so the model effectively acts as a linear AR(p) layer. We keep it as a
    transparent baseline that is easy to interpret and compare.
    """

    def __init__(self) -> None:
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
        y_arr = _as_1d_array(y_train)

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
        z = 1.96
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

    def __init__(self, max_p: int = 5, max_q: int = 2, random_state: int = 42) -> None:
        self.max_p = max_p
        self.max_q = max_q
        self.random_state = random_state
        self.arima_model: Any | None = None
        self.garch_model: Any | None = None
        self.garch_scale_: float = 1.0
        self._fit_success = False
        self._fallback = RandomWalkModel()
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
        series = _first_column_series(X_train)
        if series.size == 0:
            raise ValueError("X_train is empty; cannot fit ARIMAGARCHModel")

        self._fallback.fit(X_train, y_train)

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
        variance = forecast.variance
        variance_array = np.asarray(variance)
        if variance_array.ndim == 1:
            return variance_array
        return variance_array[-1]

    def predict(self, X: Any) -> np.ndarray:
        """Forecast conditional mean; fallback to RandomWalk on failure."""
        steps = len(X) if hasattr(X, "__len__") else 1
        if not self._fit_success:
            return self._fallback.predict(X)

        try:
            return self._forecast_mean(steps)
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
            mean = self._forecast_mean(steps)
            variance = self._forecast_variance(steps)
        except Exception as exc:  # noqa: BLE001 - robust fallback
            self._logger.warning(
                "GARCH interval failed; fallback to RandomWalk: {error}",
                error=str(exc),
            )
            return self._fallback.predict_interval(X, alpha=alpha)

        z = 1.96
        garch_scale = self.garch_scale_ if self.garch_scale_ else 1.0
        std = np.sqrt(variance) / garch_scale
        lower = mean - z * std
        upper = mean + z * std
        return lower.astype(float), upper.astype(float)
