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
        self.model_instance: Any | None = None
        self.nf: Any | None = None
        self._last_train_df: pd.DataFrame | None = None

        self._logger.info("Using CPU for DL training (GPU disabled)")

    def _prepare_df(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "unique_id": "brent",
                "ds": pd.DatetimeIndex(X.index),
            }
        )
        if y is not None:
            df["y"] = y.values

        # PatchTST не підтримує exogenous variables
        if self._supports_exogenous():
            for col in X.columns:
                df[col] = X[col].values

        return df

    def _supports_exogenous(self) -> bool:
        model_name = self.model_class.__name__ if self.model_class is not None else ""
        return model_name not in {"PatchTST"}

    def _build_model(self, exog_columns: list[str]) -> Any:
        model_kwargs = dict(self.model_kwargs)

        if "h" not in model_kwargs:
            model_kwargs["h"] = self.horizon
        if "input_size" not in model_kwargs:
            model_kwargs["input_size"] = self.input_size

        if "accelerator" not in model_kwargs:
            model_kwargs["accelerator"] = self._accelerator
            
        if model_kwargs.get("accelerator") == "cpu" and "devices" not in model_kwargs:
            model_kwargs["devices"] = 1

        signature = inspect.signature(self.model_class)
        if self._supports_exogenous():
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

        self._exog_columns = list(X_train.columns) if self._supports_exogenous() else []
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
        self.nf = self._nf

        if X_val is not None and y_val is not None:
            val_size = len(y_val)
            self._nf.fit(df_train, val_size=val_size)
        else:
            # Без val — вимикаємо early stopping щоб уникнути помилки
            for model in self._nf.models:
                model.early_stop_patience_steps = -1
            self._nf.fit(df_train)

        self._last_train_df = df_train

        return self

    def _predict_df(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._nf is None:
            raise ValueError("Model must be fitted before prediction")

        # Generate strictly continuous future dates to avoid NeuralForecast missing combinations errors.
        try:
            futr_df = self._nf.make_future_dataframe(df=self._last_train_df)
        except Exception:
            last_date = self._last_train_df["ds"].max() if self._last_train_df is not None else pd.Timestamp.now()
            dates = pd.date_range(start=last_date, periods=len(X_test) + 1, freq="B")[1:]
            futr_df = pd.DataFrame({"unique_id": "brent", "ds": dates})
            
        # Manually align exogenous variables
        if self._supports_exogenous():
            for col in self._exog_columns:
                if col in X_test.columns:
                    vals = X_test[col].values
                    if len(vals) < len(futr_df):
                        vals = np.pad(vals, (0, len(futr_df) - len(vals)), mode="edge")
                    elif len(vals) > len(futr_df):
                        vals = vals[: len(futr_df)]
                    futr_df[col] = vals

        try:
            forecast_df = self._nf.predict(futr_df=futr_df)
        except TypeError:
            try:
                forecast_df = self._nf.predict(df=futr_df)
            except Exception:
                forecast_df = self._nf.predict()

        logger.info(
            "DL Predict Diagnostics - shape: {shape}, cols: {cols}",
            shape=forecast_df.shape,
            cols=list(forecast_df.columns),
        )
        if "ds" in forecast_df.columns:
            logger.info("DL Predict Diagnostics - unique ds count: {count}", count=forecast_df["ds"].nunique())
        elif forecast_df.index.name == "ds":
            logger.info("DL Predict Diagnostics - unique ds count (index): {count}", count=forecast_df.index.nunique())

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
        forecast_df = self._predict_df(X)

        exclude = {"unique_id", "ds"}
        model_cols = [c for c in forecast_df.columns if c not in exclude]

        median_col = self._select_point_column(model_cols)

        # Беремо тільки перші len(X) рядків
        result = forecast_df[median_col].values[: len(X)]

        if len(result) != len(X):
            logger.error(
                "Length mismatch: got {res_len}, expected {x_len}",
                res_len=len(result),
                x_len=len(X),
            )

        return result
