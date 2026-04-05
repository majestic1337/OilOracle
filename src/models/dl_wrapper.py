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
        local_scaler_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_class = model_class
        self.horizon = horizon
        self.input_size = input_size
        self.local_scaler_type = local_scaler_type
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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return clone-safe params.

        For the base wrapper class, preserve dynamic `**kwargs` so direct usage
        of `DeepLearningForecasterWrapper` remains clone-compatible.
        For subclasses (e.g., TFT/NBEATS wrappers), return only constructor
        parameters accepted by the subclass `__init__`.
        """
        if self.__class__ is DeepLearningForecasterWrapper:
            params: dict[str, Any] = {
                "model_class": self.model_class,
                "horizon": self.horizon,
                "input_size": self.input_size,
                "local_scaler_type": self.local_scaler_type,
            }
            params.update(self.model_kwargs)
            return params

        params: dict[str, Any] = {}
        signature = inspect.signature(self.__class__.__init__)
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if hasattr(self, name):
                params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any) -> "DeepLearningForecasterWrapper":
        """Set clone/grid-search params, including dynamic model kwargs."""
        core_keys = {"model_class", "horizon", "input_size", "local_scaler_type"}
        for key, value in params.items():
            if key in core_keys:
                setattr(self, key, value)
            else:
                self.model_kwargs[key] = value
                if self.__class__ is not DeepLearningForecasterWrapper:
                    setattr(self, key, value)
        return self

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
                # Prevent leakage: never pass financial features as future exogenous inputs.
                if "futr_exog_list" in model_kwargs:
                    model_kwargs.pop("futr_exog_list", None)
                    self._logger.warning(
                        "Ignoring futr_exog_list in model kwargs to prevent leakage; "
                        "using hist_exog_list only."
                    )

                if (
                    "hist_exog_list" in signature.parameters
                    and exog_columns
                    and "hist_exog_list" not in model_kwargs
                ):
                    model_kwargs["hist_exog_list"] = exog_columns

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
            local_scaler_type=self.local_scaler_type,
        )
        self.nf = self._nf

        if X_val is not None and y_val is not None:
            df_val = self._prepare_df(X_val, y_val)
            df_full = pd.concat([df_train, df_val], ignore_index=True)
            df_full = df_full.sort_values(["unique_id", "ds"]).reset_index(drop=True)
            val_size = len(y_val)
            self._nf.fit(df_full, val_size=val_size)
            self._last_train_df = df_full
        else:
            # Без val — вимикаємо early stopping щоб уникнути помилки
            for model in self._nf.models:
                model.early_stop_patience_steps = -1
            self._nf.fit(df_train)
            self._last_train_df = df_train

        return self

    def _has_futr_exog(self) -> bool:
        """Check if the underlying NeuralForecast model uses future exogenous features."""
        if self._model is None:
            return False
        futr_list = getattr(self._model, "futr_exog_list", None)
        return bool(futr_list)

    def _predict_df(self, X_test: pd.DataFrame, history_X: pd.DataFrame | None = None, history_y: pd.Series | None = None) -> pd.DataFrame:
        if self._nf is None:
            raise ValueError("Model must be fitted before prediction")

        futr_df = self._build_future_df(X_test, history_X)

        hist_df = None
        if history_X is not None and history_y is not None:
            hist_df = self._prepare_df(history_X, history_y)

        forecast_df = self._predict_with_futr_df(futr_df, hist_df)
        forecast_df = self._normalize_forecast_df(forecast_df, futr_df)

        # MIMO h-th step extraction for step-by-step WFV:
        # NeuralForecast returns h rows [t+1, ..., t+h]. When futr_df has
        # only 1 row (step-by-step recursive), we must extract the h-th
        # element BEFORE _align_forecast_to_future truncates the vector.
        # We then overwrite its date with futr_df's date so alignment works.
        if len(futr_df) == 1 and len(forecast_df) >= self.horizon:
            forecast_df = forecast_df.iloc[[self.horizon - 1]].copy()
            forecast_df["ds"] = futr_df["ds"].values[0]
            self._logger.debug(
                "Step-by-step: extracted h={h} element from forecast",
                h=self.horizon,
            )

        n_expected = min(self.horizon, len(futr_df))
        if len(forecast_df) != len(futr_df) and len(futr_df) > 1:
            self._logger.info(
                "Forecast length {pred_len} vs future length {future_len}; "
                "trimming futr_df to model horizon h={h}",
                pred_len=len(forecast_df),
                future_len=len(futr_df),
                h=self.horizon,
            )

        forecast_df = self._align_forecast_to_future(forecast_df, futr_df, n_expected)

        logger.info(
            "DL Predict Diagnostics - shape: {shape}, cols: {cols}",
            shape=forecast_df.shape,
            cols=list(forecast_df.columns),
        )

        return forecast_df

    def _build_future_df(
        self,
        X_test: pd.DataFrame,
        history_X: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build a future DataFrame for NeuralForecast.predict().

        Generates business-day dates strictly continuous from the last known
        historical observation. During WFV, ``history_X`` (train+val window)
        is used as the authoritative source for both the last date and exog
        fallback values, overriding ``_last_train_df`` which may lag behind
        the current fold's window.
        """
        if X_test.empty:
            raise ValueError("X_test is empty")

        # --- Resolve last date: prefer history_X index over _last_train_df ---
        if history_X is not None and not history_X.empty:
            try:
                last_date = pd.DatetimeIndex(history_X.index).max()
                if last_date.tz is not None:
                    last_date = last_date.tz_convert(None)
                last_date = last_date.normalize()
            except Exception:  # noqa: BLE001
                last_date = (
                    self._last_train_df["ds"].max()
                    if self._last_train_df is not None
                    else pd.Timestamp.now()
                )
        else:
            last_date = (
                self._last_train_df["ds"].max()
                if self._last_train_df is not None
                else pd.Timestamp.now()
            )

        # --- Generate continuous business-day dates from training end ---
        n_periods = len(X_test)
        expected_dates = pd.bdate_range(
            start=last_date + pd.tseries.offsets.BDay(1),
            periods=n_periods,
        ).normalize()

        # --- Try to use X_test's own index if it matches exactly ---
        use_x_index = False
        ds_index: pd.DatetimeIndex | None = None
        try:
            ds_index = pd.DatetimeIndex(X_test.index)
            if ds_index.isna().any():
                raise ValueError("X_test index contains NaT values")
            if ds_index.tz is not None:
                ds_index = ds_index.tz_convert(None)
            ds_index = ds_index.normalize()
            if len(ds_index) == len(expected_dates) and ds_index.equals(expected_dates):
                use_x_index = True
            else:
                self._logger.warning(
                    "X_test index ({x_len} rows) not continuous from last known "
                    "date {last_date}; using generated business-day dates",
                    x_len=len(ds_index),
                    last_date=last_date.date(),
                )
        except Exception as exc:  # noqa: BLE001 - defensive fallback
            self._logger.warning(
                "Failed to parse X_test index as DatetimeIndex: {error}",
                error=str(exc),
            )

        dates = ds_index if (use_x_index and ds_index is not None) else expected_dates
        futr_df = pd.DataFrame({"unique_id": "brent", "ds": dates})

        # --- Attach exogenous columns (only for models that support them) ---
        if self._supports_exogenous() and self._exog_columns:
            # Retrieve last known historical values: prefer history_X, fall back to _last_train_df
            hist_last_values: dict[str, float] = {}
            hist_source = history_X if (history_X is not None and not history_X.empty) else None
            if hist_source is not None:
                for col in self._exog_columns:
                    if col in hist_source.columns:
                        last_val = hist_source[col].dropna()
                        hist_last_values[col] = float(last_val.iloc[-1]) if len(last_val) > 0 else 0.0
            elif self._last_train_df is not None:
                for col in self._exog_columns:
                    if col in self._last_train_df.columns:
                        last_val = self._last_train_df[col].dropna()
                        hist_last_values[col] = float(last_val.iloc[-1]) if len(last_val) > 0 else 0.0

            for col in self._exog_columns:
                if col in X_test.columns:
                    vals = X_test[col].values.copy().astype(float)
                    # Fill NaN from last known historical value
                    nan_mask = np.isnan(vals)
                    if nan_mask.any():
                        fill_val = hist_last_values.get(col, 0.0)
                        vals[nan_mask] = fill_val
                        self._logger.info(
                            "Filled {n} NaN(s) in exog '{col}' with last historical value {v}",
                            n=int(nan_mask.sum()), col=col, v=fill_val,
                        )
                    # Align length
                    if len(vals) < len(futr_df):
                        vals = np.pad(vals, (0, len(futr_df) - len(vals)), mode="edge")
                    elif len(vals) > len(futr_df):
                        vals = vals[: len(futr_df)]
                    futr_df[col] = vals
                else:
                    # Column missing from X_test entirely — use last historical value
                    fill_val = hist_last_values.get(col, 0.0)
                    futr_df[col] = fill_val
                    self._logger.warning(
                        "Exog column '{col}' missing from X_test; filled with historical last={v}",
                        col=col, v=fill_val,
                    )

        return futr_df

    def _predict_with_futr_df(self, futr_df: pd.DataFrame, hist_df: pd.DataFrame | None = None) -> pd.DataFrame:
        kwargs = {}
        # Only pass futr_df if the model actually expects future exogenous features.
        # When only hist_exog_list is configured, passing exog columns in futr_df
        # causes NeuralForecast to raise or silently misinterpret them.
        if futr_df is not None and self._has_futr_exog():
            kwargs["futr_df"] = futr_df
        if hist_df is not None:
            kwargs["df"] = hist_df

        try:
            forecast_df = self._nf.predict(**kwargs)
        except TypeError:
            # Fallback if futr_df is not accepted
            if "futr_df" in kwargs:
                del kwargs["futr_df"]
            forecast_df = self._nf.predict(**kwargs)
        return forecast_df

    @staticmethod
    def _normalize_forecast_df(
        forecast_df: pd.DataFrame,
        futr_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(forecast_df, pd.DataFrame):
            raise ValueError("NeuralForecast predict must return a DataFrame")

        df = forecast_df.copy()
        if "ds" not in df.columns:
            if df.index.name == "ds" or (
                isinstance(df.index, pd.MultiIndex) and "ds" in df.index.names
            ):
                df = df.reset_index()

        if "ds" not in df.columns:
            df.insert(0, "ds", futr_df["ds"].values[: len(df)])

        if "unique_id" not in df.columns:
            df.insert(0, "unique_id", futr_df["unique_id"].values[: len(df)])

        return df

    def _align_forecast_to_future(
        self,
        forecast_df: pd.DataFrame,
        futr_df: pd.DataFrame,
        n_expected: int | None = None,
    ) -> pd.DataFrame:
        """Align forecast output to future dates.

        Uses date-based merge first. If that produces NaN (date mismatch),
        falls back to positional alignment to avoid injecting zeros into
        the prediction output.
        """
        if "ds" not in forecast_df.columns:
            # No dates to align — return as-is, trimmed to n_expected
            if n_expected is not None and len(forecast_df) > n_expected:
                return forecast_df.iloc[:n_expected].copy()
            return forecast_df

        df = forecast_df.copy()
        df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
        base = futr_df.copy()
        base["ds"] = pd.to_datetime(base["ds"]).dt.normalize()

        # Trim base to n_expected if model outputs fewer rows than futr_df
        if n_expected is not None and n_expected < len(base):
            base = base.iloc[:n_expected].copy()

        merge_cols = ["ds"]
        if "unique_id" in df.columns and "unique_id" in base.columns:
            merge_cols = ["unique_id", "ds"]

        df = df.drop_duplicates(subset=merge_cols, keep="last")
        aligned = base[merge_cols].merge(df, on=merge_cols, how="left")

        pred_cols = [col for col in aligned.columns if col not in {"unique_id", "ds"}]

        # Check if date-based merge produced NaN (date mismatch)
        has_nans = pred_cols and aligned[pred_cols].isna().any().any()
        if has_nans:
            self._logger.warning(
                "Date-based alignment produced NaN; falling back to positional alignment"
            )
            # Positional fallback: take the first n_expected rows from forecast
            n_use = min(len(df), len(base))
            aligned = base.iloc[:n_use].copy()
            for col in pred_cols:
                if col in df.columns:
                    aligned[col] = df[col].values[:n_use]

        return aligned

    def _predict_in_chunks(self, futr_df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
        if chunk_size <= 0:
            chunk_size = 1

        chunks: list[pd.DataFrame] = []
        for start in range(0, len(futr_df), chunk_size):
            chunk = futr_df.iloc[start : start + chunk_size].copy()
            forecast_chunk = self._predict_with_futr_df(chunk)
            forecast_chunk = self._normalize_forecast_df(forecast_chunk, chunk)
            forecast_chunk = self._align_forecast_to_future(forecast_chunk, chunk)
            chunks.append(forecast_chunk)

        if not chunks:
            return pd.DataFrame()

        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def _select_point_column(columns: list[str]) -> str:
        for candidate in columns:
            lower = candidate.lower()
            if any(token in lower for token in ("q0.5", "median", "quantile0.5", "p50")):
                return candidate
        return columns[0]

    def predict(self, X: pd.DataFrame, history_X: pd.DataFrame | None = None, history_y: pd.Series | None = None) -> np.ndarray:
        forecast_df = self._predict_df(X, history_X, history_y)

        exclude = {"unique_id", "ds"}
        model_cols = [c for c in forecast_df.columns if c not in exclude]

        if not model_cols:
            raise ValueError("No prediction columns returned by NeuralForecast")

        median_col = self._select_point_column(model_cols)

        if "ds" in forecast_df.columns:
            df = forecast_df.copy()
            df["ds"] = pd.to_datetime(df["ds"])
            df = df.drop_duplicates(subset=["ds"], keep="last").set_index("ds")
            target_index = pd.DatetimeIndex(X.index)
            series = df[median_col].reindex(target_index)
            if series.isna().any():
                series = series.ffill().bfill()
                if series.isna().any():
                    series = series.fillna(0.0)
            result = series.to_numpy(dtype=float)
        else:
            result = forecast_df[median_col].to_numpy(dtype=float)

        if len(result) != len(X):
            self._logger.warning(
                "Length mismatch after alignment: got {res_len}, expected {x_len}",
                res_len=len(result),
                x_len=len(X),
            )
            if len(result) == 0:
                result = np.zeros(len(X), dtype=float)
            elif len(result) > len(X):
                result = result[: len(X)]
            else:
                result = np.pad(result, (0, len(X) - len(result)), mode="edge")

        return result
