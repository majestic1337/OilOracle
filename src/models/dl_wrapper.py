"""Deep-learning wrapper for NeuralForecast models."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.models.base import BaseForecaster


def _detect_accelerator() -> str:
    """Detect whether CUDA is available for NeuralForecast models."""
    try:
        import torch

        if torch.cuda.is_available():
            return "gpu"
    except Exception as exc:  # noqa: BLE001 - torch may be unavailable
        logger.debug("CUDA detection skipped: {error}", error=str(exc))
    return "cpu"


def _as_1d_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        array = array.squeeze()
    return array


class DeepLearningForecasterWrapper(BaseForecaster):
    """Adapter that exposes NeuralForecast models through a scikit-like API.

    The wrapper standardizes preprocessing so DL architectures can be compared
    against classical baselines inside the walk-forward loop without extra glue.
    """

    def __init__(
        self,
        model_class: type[Any],
        horizon: int,
        input_size: int,
        **kwargs: Any,
    ) -> None:
        self.model_class = model_class
        self.horizon = horizon
        self.input_size = input_size
        self.model_kwargs = kwargs
        self._logger = logger.bind(model=self.__class__.__name__)
        self._accelerator = _detect_accelerator()
        self._exog_columns: list[str] = []
        self._nf: Any | None = None
        self._model: Any | None = None

        self._logger.info("Using accelerator: {acc}", acc=self._accelerator)

    def _prepare_df(self, X: pd.DataFrame, y: Any | None = None) -> pd.DataFrame:
        """Convert feature matrices into NeuralForecast dataframe format.

        Args:
            X: Feature matrix with DatetimeIndex.
            y: Optional target array/series aligned with X.

        Returns:
            DataFrame with columns [unique_id, ds, y] plus exogenous features.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with a DatetimeIndex")

        if not isinstance(X.index, pd.DatetimeIndex):
            X = X.copy()
            X.index = pd.to_datetime(X.index)

        df = pd.DataFrame({"unique_id": "brent", "ds": X.index})

        for col in X.columns:
            df[col] = X[col].values

        if y is not None:
            y_array = _as_1d_array(y)
            if len(y_array) != len(X):
                raise ValueError("y length must match X length")
            df["y"] = y_array

        return df

    def _build_model(self, exog_columns: list[str]) -> Any:
        model_kwargs = dict(self.model_kwargs)

        if "h" not in model_kwargs:
            model_kwargs["h"] = self.horizon
        if "input_size" not in model_kwargs:
            model_kwargs["input_size"] = self.input_size

        if "accelerator" not in model_kwargs:
            model_kwargs["accelerator"] = self._accelerator

        signature = inspect.signature(self.model_class)
        for param in ("futr_exog_list", "hist_exog_list"):
            if param in signature.parameters and exog_columns:
                model_kwargs[param] = exog_columns

        return self.model_class(**model_kwargs)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Any,
        X_val: pd.DataFrame | None = None,
        y_val: Any | None = None,
    ) -> "DeepLearningForecasterWrapper":
        """Fit the NeuralForecast model on the provided data."""
        if X_train.empty:
            raise ValueError("X_train is empty")

        self._exog_columns = list(X_train.columns)
        df_train = self._prepare_df(X_train, y_train)

        val_size = 0
        if X_val is not None and y_val is not None:
            df_val = self._prepare_df(X_val, y_val)
            df_fit = pd.concat([df_train, df_val]).sort_values("ds")
            val_size = len(df_val)
        else:
            df_fit = df_train

        try:
            from neuralforecast import NeuralForecast
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("neuralforecast is required for deep learning models") from exc

        self._model = self._build_model(self._exog_columns)
        self._nf = NeuralForecast(models=[self._model], freq="B")

        if val_size > 0:
            self._nf.fit(df_fit, val_size=val_size)
        else:
            self._nf.fit(df_fit)

        return self

    def _predict_df(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._nf is None:
            raise ValueError("Model must be fitted before prediction")

        df_test = self._prepare_df(X_test)

        try:
            forecast_df = self._nf.predict(futr_df=df_test)
        except TypeError:
            forecast_df = self._nf.predict(df=df_test)

        if not isinstance(forecast_df, pd.DataFrame):
            raise ValueError("NeuralForecast predict must return a DataFrame")

        return forecast_df

    @staticmethod
    def _select_point_column(columns: list[str]) -> str:
        for candidate in columns:
            lower = candidate.lower()
            if any(token in lower for token in ("q0.5", "median", "quantile0.5", "p50")):
                return candidate
        return columns[0]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts aligned with the test window."""
        forecast_df = self._predict_df(X)
        pred_columns = [col for col in forecast_df.columns if col not in {"unique_id", "ds"}]
        if not pred_columns:
            raise ValueError("No prediction columns returned by NeuralForecast")

        point_col = self._select_point_column(pred_columns)
        preds = forecast_df[point_col].to_numpy(dtype=float)

        if len(preds) != len(X):
            self._logger.warning(
                "Prediction length {pred_len} differs from X length {x_len}; aligning",
                pred_len=len(preds),
                x_len=len(X),
            )
            if len(preds) > len(X):
                preds = preds[: len(X)]
            else:
                pad = len(X) - len(preds)
                preds = np.pad(preds, (0, pad), mode="edge")

        return preds
