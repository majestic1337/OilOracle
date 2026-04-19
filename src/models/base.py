"""Base interfaces and evaluation utilities for forecasting models."""

from __future__ import annotations

from math import erf
from typing import Any

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin


class BaseForecaster(BaseEstimator, RegressorMixin):
    """Base class for forecasting models.

    This interface is intentionally minimal to allow classical statistics,
    ML regressors, and hybrid models to be swapped inside the WFV loop
    without adapter glue.
    """

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None = None,
        y_val: Any | None = None,
    ) -> "BaseForecaster":
        """Fit the model.

        Args:
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Optional validation feature matrix.
            y_val: Optional validation target vector.

        Returns:
            self
        """
        raise NotImplementedError("fit must be implemented in subclasses")

    def predict(self, X: Any, history_X: Any | None = None, history_y: Any | None = None) -> np.ndarray:
        """Generate point forecasts.

        Args:
            X: Feature matrix for prediction.
            history_X: Optional historical features for models that require state update without fitting.
            history_y: Optional historical targets for models that require state update without fitting.

        Returns:
            Array of point forecasts.
        """
        raise NotImplementedError("predict must be implemented in subclasses")

    def predict_interval(self, X: Any, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals for uncertainty-aware models.

        Args:
            X: Feature matrix for prediction.
            alpha: Interval tail probability (1 - alpha coverage).

        Returns:
            Tuple of (lower, upper) bounds for the prediction interval.
        """
        raise NotImplementedError("predict_interval not supported for this model")


def as_1d_array(values: Any) -> np.ndarray:
    """Convert array-like values to a 1D numpy float array.

    Shared utility — import from here instead of duplicating.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        array = array.squeeze()
    return array


def as_2d_array(values: Any) -> np.ndarray:
    """Convert array-like values to a 2D numpy float array.

    Shared utility — import from here instead of duplicating.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


# Backward-compat alias
_to_numpy = as_1d_array


def calculate_mase(
    y_train: Any,
    y_test: Any,
    y_pred: Any,
    m: int = 1,
    zero_naive: bool = True,
) -> float:
    """Compute mean absolute scaled error (MASE).

    For stationary returns (zero_naive=True), the naive baseline is 0.
    For non-stationary prices (zero_naive=False), the naive baseline is y_{t-m}.

    Args:
        y_train: In-sample target values.
        y_test: Out-of-sample target values.
        y_pred: Forecast values for y_test.
        m: Seasonal period for the naive benchmark.
        zero_naive: Whether to use 0 as the naive forecast (for returns).

    Returns:
        MASE value or NaN if the naive MAE is zero.
    """
    if m < 1:
        raise ValueError("m must be >= 1")

    train = as_1d_array(y_train)
    test = as_1d_array(y_test)
    pred = as_1d_array(y_pred)

    if len(test) != len(pred):
        raise ValueError("y_test and y_pred must have the same length")

    if zero_naive:
        if len(train) == 0:
            logger.warning("Empty training set for MASE")
            return float("nan")
        naive_mae = float(np.mean(np.abs(train)))
    else:
        if len(train) <= m:
            logger.warning("Not enough training samples to compute naive MAE for MASE")
            return float("nan")
        naive_mae = float(np.mean(np.abs(train[m:] - train[:-m])))

    if naive_mae == 0:
        logger.warning("Naive MAE is zero; MASE undefined")
        return float("nan")

    if len(test) == 0:
        logger.warning("y_test is empty; MASE undefined")
        return float("nan")

    forecast_mae = float(np.mean(np.abs(test - pred)))
    return forecast_mae / naive_mae


calculate_window_mase = calculate_mase  # backward-compat alias

def calculate_metrics(y_train: Any, y_test: Any, y_pred: Any) -> dict[str, float]:
    """Compute Experiment A metrics (MASE, RMSE, MAE).

    The metric set mirrors the thesis requirements: MASE for scaled accuracy,
    RMSE for volatility-sensitive penalties, and MAE for robustness to outliers.

    Args:
        y_train: In-sample target values.
        y_test: Out-of-sample target values.
        y_pred: Forecast values for y_test.

    Returns:
        Dictionary of metrics.
    """
    train = as_1d_array(y_train)
    test = as_1d_array(y_test)
    pred = as_1d_array(y_pred)

    if len(test) != len(pred):
        raise ValueError("y_test and y_pred must have the same length")

    rmse = float(np.sqrt(np.mean((test - pred) ** 2)))
    mae = float(np.mean(np.abs(test - pred)))
    mase = float(calculate_mase(train, test, pred, m=1))

    return {"mase": mase, "rmse": rmse, "mae": mae}


def diebold_mariano_test(
    errors_model: Any,
    errors_benchmark: Any,
    h: int = 1,
) -> dict[str, float | str]:
    """Diebold-Mariano test for predictive accuracy differences.

    We compare loss differentials using a Newey-West style HAC variance that
    accounts for multi-step forecast overlap (lag = h-1). The loss function is
    squared error to stay consistent with the WFV regression setting.

    Note:
        HAC variance uses Bartlett kernel weights (w_j = 1 - j/h) following
        Harvey, Leybourne & Newbold (1997). This differs from a flat-weight
        Newey-West estimator.

    Args:
        errors_model: Forecast errors of the model under test (y - y_hat).
        errors_benchmark: Forecast errors of the benchmark model.
        h: Forecast horizon used in the experiment.

    Returns:
        Dictionary containing DM statistic, p-value, and better model label.
    """
    if h < 1:
        raise ValueError("h must be >= 1")

    err_model = as_1d_array(errors_model)
    err_bench = as_1d_array(errors_benchmark)

    if len(err_model) != len(err_bench):
        raise ValueError("errors_model and errors_benchmark must have the same length")

    if len(err_model) < 2:
        logger.warning("Not enough error observations for DM test")
        return {"dm_statistic": float("nan"), "p_value": float("nan"), "better_model": "tie"}

    d = (err_model**2) - (err_bench**2)
    mean_d = float(np.mean(d))

    d_centered = d - mean_d
    gamma0 = float(np.mean(d_centered**2))
    hac_var = gamma0

    for lag in range(1, h):
        weight = 1.0 - lag / h  # Bartlett kernel
        cov = float(np.mean(d_centered[lag:] * d_centered[:-lag]))
        hac_var += 2.0 * weight * cov

    hac_var /= len(d)

    if hac_var <= 0:
        logger.warning("Non-positive HAC variance in DM test")
        dm_stat = float("nan")
        p_value = float("nan")
    else:
        dm_stat = mean_d / np.sqrt(hac_var)
        try:
            from scipy.stats import t

            p_value = float(2.0 * (1.0 - t.cdf(abs(dm_stat), df=len(d) - 1)))
        except Exception:  # noqa: BLE001 - fallback to normal approximation
            p_value = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / np.sqrt(2)))))

    if np.isnan(mean_d) or abs(mean_d) < 1e-12:
        better = "tie"
    elif mean_d < 0:
        better = "model"
    else:
        better = "benchmark"

    return {"dm_statistic": float(dm_stat), "p_value": float(p_value), "better_model": better}
