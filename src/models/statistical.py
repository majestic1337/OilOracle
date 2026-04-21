"""Statistical baselines for oil price forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from .base import BaseForecaster, as_1d_array


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

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
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

    def __init__(self, task_type: str = "regression", horizon: int = 1) -> None:
        self.task_type = task_type
        self.horizon = horizon
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
        """Fit direct multi-step linear regression.

        Note:
            y_train is raw brent_return. We build the h-step-ahead target
            internally via shift(-horizon) so the model learns to predict
            return at t+h from features at t. Rows where shifted target is
            NaN are dropped before fitting.

            X_val and y_val are accepted for interface compatibility but
            intentionally ignored. LinearRegression has no early stopping;
            merging val into train would constitute data leakage in the WFV loop.
        """
        X_arr = np.asarray(X_train)
        y_arr = as_1d_array(y_train)

        # Build direct h-step target: y_{t+h} from features at t
        y_shifted = pd.Series(y_arr).shift(-self.horizon).values
        valid_mask = ~np.isnan(y_shifted)
        X_fit = X_arr[valid_mask]
        y_fit = y_shifted[valid_mask]

        self.model.fit(X_fit, y_fit)
        residuals = y_fit - self.model.predict(X_fit)
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
    GARCH_SCALE: float = 100.0  # scale residuals before GARCH fit; divide std after

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

        Note:
            X_val and y_val are accepted for interface compatibility but intentionally
            ignored. LinearRegression has no early stopping; merging val into train
            would constitute data leakage in the WFV loop.
        """
        series = np.asarray(y_train, dtype=float).squeeze()
        if series.size == 0:
            raise ValueError("y_train is empty; cannot fit ARIMAGARCHModel")

        try:
            self._fallback.fit(X_train, y_train)
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

            resids_scaled = residuals * self.GARCH_SCALE

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
        self.garch_scale_ = self.GARCH_SCALE
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

    def predict(
        self,
        X: Any,
        history_X: Any | None = None,
        history_y: Any | None = None,
    ) -> np.ndarray:
        """Forecast h-step-ahead conditional mean with stateful ARIMA updates.

        Note:
            Trained on raw brent_return. For each step, forecasts horizon h
            periods ahead and returns the h-th element as the point forecast.
            After each forecast, ARIMA state is updated with the most recent
            observed value from history_y so subsequent forecasts reflect
            new information.
        """
        steps = len(X) if hasattr(X, "__len__") else 1
        if not self._fit_success:
            return self._fallback.predict(X)

        try:
            # Build observation history for ARIMA state update
            # history_y contains raw brent_return values up to current step
            obs_history: list[float] = []
            if history_y is not None:
                obs_array = np.asarray(history_y, dtype=float).squeeze()
                if obs_array.ndim == 0:
                    obs_history = [float(obs_array)]
                else:
                    obs_history = obs_array.tolist()

            forecasts: list[float] = []
            for step_idx in range(steps):
                # Forecast h steps ahead from current ARIMA state
                step_forecast = self._forecast_mean(self.horizon)
                step_forecast = np.asarray(step_forecast, dtype=float).squeeze()
                if step_forecast.ndim == 0:
                    step_forecast = np.asarray([float(step_forecast)], dtype=float)

                h_step = (
                    float(step_forecast[self.horizon - 1])
                    if len(step_forecast) >= self.horizon
                    else float(step_forecast[-1])
                )
                forecasts.append(h_step)

                # Update ARIMA state with the most recent observed value
                if obs_history:
                    new_obs = obs_history[-1]
                else:
                    new_obs = 0.0

                try:
                    self.arima_model.update([new_obs])
                except Exception as update_exc:  # noqa: BLE001
                    self._logger.debug(
                        "ARIMA state update failed at step {s}: {error}",
                        s=step_idx,
                        error=str(update_exc),
                    )

            return np.asarray(forecasts, dtype=float)

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
        history_X: Any | None = None,
        history_y: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return analytic intervals using GARCH conditional variance.

        Forecasts horizon h steps and extracts the h-th element for both
        mean and variance, then broadcasts to all test observations.
        """
        steps = len(X) if hasattr(X, "__len__") else 1
        if not self._fit_success:
            return self._fallback.predict_interval(X, alpha=alpha)

        try:
            total_steps = self.horizon  # forecast h steps, take h-th element
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
        garch_scale = self.garch_scale_ if self.garch_scale_ > 0 else 1.0

        # Extract h-th step
        mean_val = float(mean[self.horizon - 1]) if len(mean) >= self.horizon else float(mean[-1])
        var_val = float(variance[self.horizon - 1]) if len(variance) >= self.horizon else float(variance[-1])
        mean = np.full(steps, mean_val, dtype=float)
        std_val = np.sqrt(max(var_val, 0.0)) / garch_scale
        std = np.full(steps, std_val, dtype=float)

        lower = mean - z * std
        upper = mean + z * std
        return lower, upper
