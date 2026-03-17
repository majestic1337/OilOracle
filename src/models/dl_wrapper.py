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
        self._accelerator = "cpu"
        self._exog_columns: list[str] = []
        self._nf: Any | None = None
        self._model: Any | None = None

        self._logger.info("Using CPU for DL training (GPU disabled)")

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

        if y is not None:
            y_array = _as_1d_array(y)
            if len(y_array) != len(X):
                raise ValueError("y length must match X length")
            df["y"] = y_array

        try:
            from src.models.patchtst_model import PatchTSTForecaster
        except Exception:  # pragma: no cover - avoid hard dependency
            PatchTSTForecaster = None  # type: ignore[assignment]
        if PatchTSTForecaster is not None and isinstance(self, PatchTSTForecaster):
            # PatchTST channel-independent — ігноруємо exogenous
            keep_cols = ["unique_id", "ds"]
            if "y" in df.columns:
                keep_cols.append("y")
            return df[keep_cols]

        for col in X.columns:
            df[col] = X[col].values

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
            if param in signature.parameters and exog_columns and param not in model_kwargs:
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

        try:
            from neuralforecast import NeuralForecast
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise ImportError("neuralforecast is required for deep learning models") from exc

        self._model = self._build_model(self._exog_columns)
        self.model_instance = self._model
        self._nf = NeuralForecast(
            models=[self.model_instance],
            freq="B",
            local_scaler_type=None,
        )

        if X_val is not None and y_val is not None:
            val_size = len(y_val)
            self._nf.fit(df_train, val_size=val_size)
        else:
            # Без val — вимикаємо early stopping щоб уникнути помилки
            for model in self._nf.models:
                model.early_stop_patience_steps = -1
            self._nf.fit(df_train)

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
        if self._nf is None:
            raise ValueError("Model must be fitted before prediction")

        df_test = self._prepare_df(X)

        try:
            forecast_df = self._nf.predict(futr_df=df_test)
        except TypeError:
            forecast_df = self._nf.predict(df=df_test)

        if not isinstance(forecast_df, pd.DataFrame):
            raise ValueError("NeuralForecast predict must return a DataFrame")

        # neuralforecast може повернути більше рядків ніж треба
        # Фільтруємо тільки дати з X_test
        test_dates = X.index
        model_col = forecast_df.columns[-1]

        forecast_df = forecast_df[forecast_df["ds"].isin(test_dates)]
        forecast_df = forecast_df.set_index("ds").reindex(test_dates)

        result = forecast_df[model_col].to_numpy(dtype=float)

        if len(result) != len(X):
            self._logger.error(
                "predict output length {pred_len} != X length {x_len} after filtering",
                pred_len=len(result),
                x_len=len(X),
            )

        return result
