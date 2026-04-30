"""Walk-forward validation orchestration for oil price forecasting."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler

ModelFamily = Literal["ml", "dl", "stat"]
TaskType = Literal["regression", "classification"]


@dataclass(slots=True)
class WFVConfig:
    """Configuration for walk-forward validation."""

    w_train: int = 1000
    w_val: int = 200
    w_step: int = 20
    horizon: int = 1
    strategy: str = "direct"
    model_family: ModelFamily = "ml"
    task_type: TaskType = "regression"
    window_type: str = "rolling"
    max_lag_order: int = 10
    granger_p_threshold: float = 0.1  # reserved for future feature selection pipeline
    vif_threshold: float = 10.0  # reserved for future feature selection pipeline
    corr_threshold: float = 0.85  # reserved for future feature selection pipeline
    top_k_shap: int = 15
    random_state: int = 42
    dl_mode: str = "per_fold"
    dl_recursive: bool = True
    interval_alpha: float = 0.1
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        if self.model_family not in {"ml", "dl", "stat"}:
            raise ValueError("model_family must be one of {'ml', 'dl', 'stat'}")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError("task_type must be 'regression' or 'classification'")
        if self.w_step <= self.horizon - 1:
            raise ValueError(
                f"w_step ({self.w_step}) must be > horizon - 1 ({self.horizon - 1}) "
                "to ensure test samples exist after embargo."
            )
        if self.w_train < self.w_val:
            raise ValueError(
                f"w_train ({self.w_train}) must be >= w_val ({self.w_val}): "
                "validation window cannot exceed training window."
            )

    @property
    def model_kwargs(self) -> dict[str, int]:
        """Generic kwargs passed to DL model constructors."""
        return {"horizon": self.horizon}


@dataclass(slots=True)
class WFVIteration:
    """Audit record for a single WFV iteration."""

    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    selected_features: list[str]
    n_features_input: int
    n_features_selected: int
    scaler_center: np.ndarray
    scaler_scale: np.ndarray
    predictions: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    actuals: np.ndarray
    fit_time_seconds: float


def _ensure_series(y: pd.Series | pd.DataFrame, horizon: int) -> pd.Series:
    if isinstance(y, pd.Series):
        return y

    if y.shape[1] == 1:
        return y.iloc[:, 0]

    for candidate in (f"target_h{horizon}", f"h{horizon}"):
        if candidate in y.columns:
            return y[candidate]

    logger.warning("Unable to identify target column for horizon {h}; using first column", h=horizon)
    return y.iloc[:, 0]


def _is_incompatible_model_error(error: Exception, horizon: int) -> bool:
    message = str(error).lower()
    if horizon == 1 and "h=1" in message and "incompatible" in message:
        if "seasonality" in message or "trend" in message:
            return True
    if "does not support future exogenous" in message:
        return True
    return False


def _to_1d_float(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        if array.shape[1] == 0:
            return np.array([], dtype=float)
        array = array[:, 0]
    if array.ndim > 1:
        array = array.squeeze()
    return array


def _align_prediction_length(values: Any, expected: int, context: str) -> np.ndarray:
    arr = _to_1d_float(values)

    if expected < 1:
        raise ValueError("expected must be >= 1")

    if arr.size == expected:
        return arr

    if arr.size == 0:
        logger.warning("{ctx}: empty prediction array; filling NaN", ctx=context)
        return np.full(expected, np.nan, dtype=float)

    if arr.size > expected:
        logger.warning("{ctx}: prediction length {got} > expected {exp}; truncating", ctx=context, got=arr.size, exp=expected)
        return arr[:expected]

    logger.warning("{ctx}: prediction length {got} < expected {exp}; padding with last value", ctx=context, got=arr.size, exp=expected)
    if arr.size == 1:
        return np.repeat(arr[0], expected)

    return np.pad(arr, (0, expected - arr.size), mode="edge")


def _safe_fit_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | pd.DataFrame | None,
) -> None:
    try:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    except TypeError:
        model.fit(X_train, y_train)


def _safe_predict_interval(
    model: Any,
    X_test: pd.DataFrame,
    expected: int,
    alpha: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    predict_interval = getattr(model, "predict_interval", None)
    if not callable(predict_interval):
        return None, None

    try:
        lower, upper = predict_interval(X_test, alpha=alpha)
    except NotImplementedError:
        return None, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Interval prediction failed: {error}", error=str(exc))
        return None, None

    if lower is None or upper is None:
        return None, None

    lower_arr = _align_prediction_length(lower, expected, context="predict_interval(lower)")
    upper_arr = _align_prediction_length(upper, expected, context="predict_interval(upper)")
    return lower_arr, upper_arr


def _select_features_shap(X_train: pd.DataFrame, y_train: pd.Series, config: WFVConfig) -> list[str]:
    try:
        import lightgbm as lgb
        import shap
    except ImportError:
        logger.warning("SHAP dependencies are unavailable; using all features")
        return list(X_train.columns)

    if config.task_type == "classification":
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=config.random_state,
            verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=config.random_state,
            verbose=-1,
        )

    try:
        X_values = X_train.to_numpy()
        y_values = np.asarray(y_train)
        model.fit(X_values, y_values)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_values)
        shap_array = np.asarray(shap_values)

        if shap_array.ndim == 1:
            shap_importance = np.abs(shap_array)
        elif shap_array.ndim == 2:
            # Typical tree-based SHAP output for regression/binary classification:
            # shape (n_samples, n_features). Aggregate over samples.
            shap_importance = np.abs(shap_array).mean(axis=0)
        elif shap_array.ndim == 3:
            # Multiclass SHAP output can include class dimension.
            # Aggregate over all axes except the feature axis.
            shap_importance = np.abs(shap_array).mean(axis=(0, 1))
        else:
            raise ValueError(f"Unexpected SHAP output dimensions: {shap_array.ndim}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP selection failed; using all features: {error}", error=str(exc))
        return list(X_train.columns)

    if shap_importance.ndim != 1 or shap_importance.shape[0] != X_train.shape[1]:
        logger.warning(
            "SHAP importance shape mismatch ({shape}); using all features",
            shape=shap_importance.shape,
        )
        return list(X_train.columns)

    importance_series = pd.Series(shap_importance, index=X_train.columns)
    top_k = min(config.top_k_shap, len(importance_series))
    return importance_series.sort_values(ascending=False).head(top_k).index.tolist()


def select_features_in_fold(X_train: pd.DataFrame, y_train: pd.Series, config: WFVConfig) -> list[str]:
    """Select ML features using SHAP-only marginal contribution ranking."""
    shap_features = _select_features_shap(X_train, y_train, config)
    if not shap_features:
        return list(X_train.columns)
    return shap_features


def _iter_windows(total: int, config: WFVConfig) -> Iterable[tuple[int, int, int, int]]:
    if config.window_type not in {"rolling", "expanding"}:
        raise ValueError("window_type must be 'rolling' or 'expanding'")

    fold = 0
    while True:
        if config.window_type == "rolling":
            train_start = fold * config.w_step
            train_end = train_start + config.w_train
        else:  # expanding
            train_start = 0
            train_end = config.w_train + fold * config.w_step
        val_end = train_end + config.w_val
        test_end = val_end + config.w_step

        if test_end > total:
            break

        yield train_start, train_end, val_end, test_end
        fold += 1


def _extract_target_series(y_data: pd.Series | pd.DataFrame, horizon: int, mimo_target_col: str | None) -> pd.Series:
    if isinstance(y_data, pd.DataFrame) and mimo_target_col and mimo_target_col in y_data.columns:
        return y_data[mimo_target_col]
    return _ensure_series(y_data, horizon)


def _run_dl_recursive_forecast(
    model: Any,
    X_history: pd.DataFrame,
    y_history: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    interval_alpha: float,
    y_raw: pd.Series | None = None,
    max_history: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Run step-by-step recursive DL forecast.

    Parameters
    ----------
    y_test : pd.Series
        Shifted target (target_h{horizon}) — used ONLY for final metric calculation
        by the caller, NOT for updating autoregressive history.
    y_raw : pd.Series | None
        Unshifted raw target (brent_return at time t) — used to update
        ``history_y`` at each step so the model sees actual observed values
        without future leakage.
    """
    predictions: list[float] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    fit_time_total = 0.0

    has_interval = callable(getattr(model, "predict_interval", None))
    history_X = X_history.copy()
    history_y = y_history.copy()

    # Use y_raw for history updates; fall back to y_test if not provided
    y_for_history = y_raw if y_raw is not None else y_test

    for step_idx in range(len(X_test)):
        X_step = X_test.iloc[[step_idx]]
        start_time = time.perf_counter()
        
        # Avoid extremely slow refitting inside test steps
        step_pred_raw = model.predict(X_step, history_X=history_X, history_y=history_y)
        fit_time_total += time.perf_counter() - start_time

        step_pred = _align_prediction_length(step_pred_raw, 1, f"dl_recursive_step_{step_idx}")[0]
        predictions.append(float(step_pred))

        if has_interval:
            lower_step, upper_step = _safe_predict_interval(model, X_step, 1, interval_alpha)
            lower_values.append(float(lower_step[0]) if lower_step is not None else np.nan)
            upper_values.append(float(upper_step[0]) if upper_step is not None else np.nan)

        # Update history with UNSHIFTED raw observation (not shifted target)
        raw_value = float(y_for_history.iloc[step_idx])
        history_X = pd.concat([history_X, X_step], axis=0)
        history_y = pd.concat(
            [history_y, pd.Series([raw_value], index=X_step.index, name=history_y.name)], axis=0
        )
        if max_history is not None and len(history_X) > max_history:
            history_X = history_X.iloc[-max_history:]
            history_y = history_y.iloc[-max_history:]

    preds_array = np.asarray(predictions, dtype=float)
    if has_interval:
        lower_arr = np.asarray(lower_values, dtype=float)
        upper_arr = np.asarray(upper_values, dtype=float)
        if not (np.isnan(lower_arr).all() or np.isnan(upper_arr).all()):
            return preds_array, lower_arr, upper_arr, fit_time_total

    return preds_array, None, None, fit_time_total


def run_wfv(
    X: pd.DataFrame, 
    y: pd.Series | pd.DataFrame, 
    model: Any, 
    config: WFVConfig,
    model_name: str | None = None,
    output_dir: str | Path | None = None,
    y_eval: pd.Series | None = None,
) -> tuple[list[WFVIteration], pd.DataFrame]:
    """Walk-Forward Validation loop.

    Parameters
    ----------
    y : pd.Series | pd.DataFrame
        Target used for model **training**. For ML models this is the
        pre-shifted ``target_h{horizon}``. For DL MIMO models this is the
        raw, unshifted ``brent_return`` series.
    y_eval : pd.Series | None
        Optional shifted target used **only** for evaluation / actuals
        comparison. When provided (typically for DL MIMO models), the
        ``actuals_array`` recorded in each fold comes from ``y_eval``
        rather than ``y``. This prevents conflating the training target
        with the evaluation target. When ``None``, ``y`` is used for both.

    Notes
    -----
    Classification mode (task_type="classification", model_family="ml"):
        Targets are binarized to {0, 1} via ``(y > 0).astype(int)`` before
        fitting. Models are expected to return probability scores via
        ``predict_proba``-style output (floats in [0, 1]). The RMSE/MAE
        metrics computed by ``calculate_metrics`` downstream are not
        meaningful for classification — use AUC or Brier score instead.
        This is a known limitation; a dedicated classification evaluation
        path is not yet implemented.
    """
    import time
    from sklearn.base import clone
    from sklearn.preprocessing import RobustScaler
    from tqdm import tqdm

    y_model: pd.Series | pd.DataFrame
    mimo_target_col: str | None = None

    if config.strategy == "mimo" and isinstance(y, pd.DataFrame):
        y_model = y
        for candidate in (f"h{config.horizon}", f"target_h{config.horizon}"):
            if candidate in y.columns:
                mimo_target_col = candidate
                break
        if mimo_target_col is None:
            mimo_target_col = y.columns[0]
    else:
        y_model = _ensure_series(y, config.horizon)

    common_index = X.index.intersection(y_model.index)
    if y_eval is not None:
        common_index = common_index.intersection(y_eval.index)
    X = X.loc[common_index].sort_index()
    y_model = y_model.loc[common_index].sort_index()
    if y_eval is not None:
        y_eval = y_eval.loc[common_index].sort_index()

    iterations: list[WFVIteration] = []
    predictions_list: list[pd.DataFrame] = []
    pred_col_name = f"pred_h{config.horizon}"
    
    last_completed_fold = -1
    if model_name is not None and output_dir is not None:
        cached = load_cached_results(model_name, config.horizon, output_dir,  model=model)
        if cached is not None:
            cached_iters, cached_preds = cached
            if cached_iters:
                last_completed_fold = max(it.fold_idx for it in cached_iters)
                iterations = cached_iters
                predictions_list = [cached_preds]
                logger.info("Resuming WFV from fold {f}", f=last_completed_fold + 1)
    fresh_start = last_completed_fold == -1
    
    forbidden_cols = {"unique_id", "ds", "y"}
    static_features = [col for col in X.columns if col not in forbidden_cols]
    
    skip_model = False
    consecutive_failures = 0

    windows = list(_iter_windows(len(X), config))
    progress_bar = tqdm(windows, desc=f"WFV [{config.model_family.upper()} | h={config.horizon}]", unit="fold", dynamic_ncols=True)

    # Enable target scaling ONLY for continuous ML models
    scale_target = config.task_type == "regression" and config.model_family == "ml"

    for fold_idx, (train_start, train_end, val_end, test_end) in enumerate(progress_bar):
        if fold_idx <= last_completed_fold:
            continue
            
        try:
            embargo_steps = config.horizon - 1
            test_start_embargo = val_end + embargo_steps
            
            if test_start_embargo >= test_end:
                continue

            fold_model = clone(model)

            # Prevent Target Leakage
            train_end_embargo = train_end - (config.horizon - 1)
            if train_start >= train_end_embargo:
                logger.warning("Empty training set after embargo; skipping fold")
                continue

            X_train = X.iloc[train_start:train_end_embargo]
            X_val = X.iloc[train_end_embargo:val_end]
            X_test = X.iloc[test_start_embargo:test_end]

            y_train_raw = y_model.iloc[train_start:train_end_embargo]
            y_val_raw = y_model.iloc[train_end_embargo:val_end]
            y_test_raw = y_model.iloc[test_start_embargo:test_end]

            # For DL MIMO: y_eval provides the shifted target for metric evaluation
            if y_eval is not None:
                y_eval_test = y_eval.iloc[test_start_embargo:test_end]
            else:
                y_eval_test = None

            y_train_target = _extract_target_series(y_train_raw, config.horizon, mimo_target_col)
            y_val_target = _extract_target_series(y_val_raw, config.horizon, mimo_target_col)
            y_test_target = _extract_target_series(y_test_raw, config.horizon, mimo_target_col)

            # Target Scaling Strategy
            y_scaler: RobustScaler | None = None
            if scale_target:
                y_scaler = RobustScaler()
                y_train_fit = pd.Series(y_scaler.fit_transform(y_train_target.values.reshape(-1, 1)).flatten(), index=y_train_target.index)
                y_val_fit = pd.Series(y_scaler.transform(y_val_target.values.reshape(-1, 1)).flatten(), index=y_val_target.index)
                y_test_fit = pd.Series(y_scaler.transform(y_test_target.values.reshape(-1, 1)).flatten(), index=y_test_target.index)
            else:
                y_train_fit = y_train_target.copy()
                y_val_fit = y_val_target.copy()
                y_test_fit = y_test_target.copy()

            # Feature Scaling Strategy
            features_to_scale = [col for col in X_train.columns if col not in forbidden_cols]
            scaler = RobustScaler()
            X_train_scaled, X_val_scaled, X_test_scaled = X_train.copy(), X_val.copy(), X_test.copy()

            if features_to_scale:
                X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
                X_val_scaled[features_to_scale] = scaler.transform(X_val[features_to_scale])
                X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])
            else:
                scaler.center_, scaler.scale_ = np.array([]), np.array([])

            if config.model_family == "ml":
                selected_features = select_features_in_fold(X_train_scaled, y_train_fit, config)
                if not selected_features:
                    selected_features = list(X_train_scaled.columns)
            else:
                selected_features = static_features

            X_train_sel = X_train_scaled[selected_features]
            X_val_sel = X_val_scaled[selected_features]
            X_test_sel = X_test_scaled[selected_features]

            lower_array, upper_array = None, None

            # 1. ГАРАНТОВАНЕ ТРЕНУВАННЯ ДЛЯ ВСІХ МОДЕЛЕЙ
            start_time = time.perf_counter()
            _safe_fit_model(fold_model, X_train_sel, y_train_fit, X_val=X_val_sel, y_val=y_val_fit)
            fit_time = time.perf_counter() - start_time

            # 2. ГЕНЕРАЦІЯ ПРОГНОЗІВ
            if config.model_family == "dl" and config.dl_recursive:
                X_history = pd.concat([X_train_sel, X_val_sel], axis=0).sort_index()
                y_history = pd.concat([y_train_fit, y_val_fit], axis=0).sort_index()
                # Extract raw brent_return from X for history updates (no leakage)
                y_raw_test = None
                if "brent_return" in X_test.columns:
                    y_raw_test = X_test["brent_return"].iloc[
                        :len(X_test_sel)
                    ].copy()
                    y_raw_test.index = X_test_sel.index
                preds_array, lower_array, upper_array, add_fit_time = _run_dl_recursive_forecast(
                    model=fold_model,
                    X_history=X_history,
                    y_history=y_history,
                    X_test=X_test_sel,
                    y_test=y_test_fit,
                    interval_alpha=config.interval_alpha,
                    y_raw=y_raw_test,
                    max_history=config.w_train + config.w_val,
                )
                fit_time += add_fit_time
            else:
                has_update = hasattr(fold_model, "update") and callable(getattr(fold_model, "update", None))
                if config.model_family == "stat":
                    # Step-by-step prediction with history for ARIMA state updates
                    X_history = pd.concat([X_train_sel, X_val_sel], axis=0).sort_index()
                    y_history = pd.concat([y_train_fit, y_val_fit], axis=0).sort_index()

                    preds_list: list[float] = []
                    lower_list: list[float] = []
                    upper_list: list[float] = []
                    has_interval = callable(getattr(fold_model, "predict_interval", None))

                    for i in range(len(X_test_sel)):
                        X_i = X_test_sel.iloc[[i]]
                        try:
                            p_i = fold_model.predict(
                                X_i,
                                history_X=X_history,
                                history_y=y_history,
                            )
                        except TypeError:
                            p_i = fold_model.predict(X_i)

                        preds_list.append(float(p_i[0]) if len(p_i) > 0 else 0.0)

                        if has_interval:
                            l_i, u_i = _safe_predict_interval(
                                fold_model, X_i, 1, config.interval_alpha
                            )
                            lower_list.append(float(l_i[0]) if l_i is not None and len(l_i) > 0 else np.nan)
                            upper_list.append(float(u_i[0]) if u_i is not None and len(u_i) > 0 else np.nan)

                        # Update history with the raw observed value from X
                        if "brent_return" in X_test_sel.columns:
                            raw_obs = float(X_test_sel["brent_return"].iloc[i])
                        else:
                            raw_obs = float(y_test_fit.iloc[i])

                        X_history = pd.concat([X_history, X_i], axis=0)
                        y_history = pd.concat(
                            [y_history, pd.Series([raw_obs], index=X_i.index, name=y_history.name)],
                            axis=0,
                        )

                    preds_array = np.asarray(preds_list, dtype=float)
                    lower_array = np.asarray(lower_list, dtype=float) if lower_list else None
                    upper_array = np.asarray(upper_list, dtype=float) if upper_list else None

                    if lower_array is not None and np.isnan(lower_array).all():
                        lower_array = None
                    if upper_array is not None and np.isnan(upper_array).all():
                        upper_array = None
                elif has_update:
                    preds_list = []
                    lower_list = []
                    upper_list = []
                    for i in range(len(X_test_sel)):
                        X_i = X_test_sel.iloc[[i]]
                        y_i = y_test_fit.iloc[[i]]
                        p_i = fold_model.predict(X_i)
                        l_i, u_i = _safe_predict_interval(fold_model, X_i, 1, config.interval_alpha)
                        
                        preds_list.append(p_i[0] if p_i.size > 0 else 0.0)
                        if l_i is not None and u_i is not None:
                            lower_list.append(l_i[0] if l_i.size > 0 else np.nan)
                            upper_list.append(u_i[0] if u_i.size > 0 else np.nan)
                            
                        try:
                            fold_model.update(y_true=y_i.values, y_pred=p_i)
                        except Exception:
                            pass
                    
                    preds_array = np.array(preds_list)
                    lower_array = np.array(lower_list) if lower_list else None
                    upper_array = np.array(upper_list) if upper_list else None
                else:
                    preds_raw = fold_model.predict(X_test_sel)
                    preds_array = _align_prediction_length(preds_raw, len(X_test_sel), f"predict_fold_{fold_idx}")
                    lower_array, upper_array = _safe_predict_interval(fold_model, X_test_sel, len(X_test_sel), config.interval_alpha)

            # Inverse Target Scaling
            # NOTE: ACI (AdaptiveConformalWrapper) computes q_hat in scaled space.
            # RobustScaler is linear so inverse_transform preserves interval symmetry.
            # If a non-linear scaler is introduced in future, ACI q_hat must be
            # re-calibrated in original space after inverse_transform.
            if scale_target and y_scaler is not None:
                preds_array = y_scaler.inverse_transform(preds_array.reshape(-1, 1)).flatten()
                if lower_array is not None:
                    lower_array = y_scaler.inverse_transform(lower_array.reshape(-1, 1)).flatten()
                if upper_array is not None:
                    upper_array = y_scaler.inverse_transform(upper_array.reshape(-1, 1)).flatten()

            # Actuals: use y_eval (shifted target) for DL MIMO, else use y_test_target
            if y_eval_test is not None:
                actuals_array = _to_1d_float(y_eval_test)
            else:
                actuals_array = _to_1d_float(y_test_target)

            fold_predictions = pd.DataFrame({pred_col_name: preds_array}, index=X_test_sel.index)
            if lower_array is not None and upper_array is not None:
                fold_predictions["lower"], fold_predictions["upper"] = lower_array, upper_array

            predictions_list.append(fold_predictions)
            consecutive_failures = 0

            iterations.append(
                WFVIteration(
                    fold_idx=fold_idx,
                    train_start=X_train.index[0],
                    train_end=X_train.index[-1],
                    test_start=X_test.index[0],
                    test_end=X_test.index[-1],
                    selected_features=selected_features,
                    n_features_input=X_train.shape[1],
                    n_features_selected=len(selected_features),
                    scaler_center=scaler.center_.copy(),
                    scaler_scale=scaler.scale_.copy(),
                    predictions=preds_array,
                    lower_bounds=lower_array if lower_array is not None else np.array([]),
                    upper_bounds=upper_array if upper_array is not None else np.array([]),
                    actuals=actuals_array,
                    fit_time_seconds=fit_time,
                )
            )

            if model_name is not None and output_dir is not None:
                is_first_save = len(iterations) == 1
                fold_preds = predictions_list[-1]
                fold_preds = fold_preds[~fold_preds.index.duplicated(keep="last")].sort_index()
                fold_preds.attrs["horizon"] = config.horizon
                save_wfv_results(
                    [iterations[-1]],
                    fold_preds,
                    model_name,
                    output_dir,
                    fresh_start=fresh_start and is_first_save,
                    model=fold_model if config.model_family == "dl" else None
                )

            # Memory cleanup for DL models (PyTorch/Lightning graph accumulation)
            if config.model_family == "dl":
                del fold_model
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

        except Exception as exc:
            consecutive_failures += 1
            logger.warning(
                "WFV fold {fold} failed for model_family={family}, h={h}: {error}",
                fold=fold_idx,
                family=config.model_family,
                h=config.horizon,
                error=str(exc),
            )
            if _is_incompatible_model_error(exc, config.horizon) or consecutive_failures >= config.max_consecutive_failures:
                skip_model = True
                break
            continue

    if skip_model:
        return [], pd.DataFrame(columns=[pred_col_name])

    if predictions_list:
        predictions_df = pd.concat(predictions_list, axis=0)
        predictions_df = predictions_df[~predictions_df.index.duplicated(keep='last')].sort_index()
    else:
        predictions_df = pd.DataFrame(columns=[pred_col_name])
    predictions_df.attrs["horizon"] = config.horizon
    return iterations, predictions_df


def load_cached_results(model_name: str, horizon: int, output_dir: str | Path, model: Any | None = None) -> tuple[list[WFVIteration], pd.DataFrame] | None:
    output_path = Path(output_dir)
    predictions_path = output_path / f"predictions_{model_name}_{horizon}.parquet"
    audit_path = output_path / f"audit_{model_name}_{horizon}.json"

    if not (predictions_path.exists() and audit_path.exists()):
        return None

    try:
        predictions_df = pd.read_parquet(predictions_path)
    except Exception:  # noqa: BLE001
        return None

    if predictions_df.empty:
        return None

    if not isinstance(predictions_df.index, pd.DatetimeIndex):
        predictions_df = predictions_df.copy()
        predictions_df.index = pd.to_datetime(predictions_df.index)

    predictions_df = predictions_df.sort_index()
    # Завантажуємо збережену модель якщо передана і підтримує load_model
    if model is not None and hasattr(model, "load_model"):
        model_save_path = output_path / f"model_{model_name}_{horizon}"
        if model_save_path.exists():
            try:
                model.load_model(model_save_path)
                logger.info(
                    "Restored fitted model from {path}",
                    path=model_save_path,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load model from {path}: {error}",
                    path=model_save_path,
                    error=str(exc),
                )

    predictions_df.attrs["horizon"] = horizon

    try:
        with audit_path.open("r", encoding="utf-8") as handle:
            audit_data = json.load(handle)
        
        iterations = []
        for row in audit_data:
            iterations.append(
                WFVIteration(
                    fold_idx=row["fold_idx"],
                    train_start=pd.Timestamp(row["train_start"]),
                    train_end=pd.Timestamp(row["train_end"]),
                    test_start=pd.Timestamp(row["test_start"]),
                    test_end=pd.Timestamp(row["test_end"]),
                    selected_features=row["selected_features"],
                    n_features_input=row["n_features_input"],
                    n_features_selected=row["n_features_selected"],
                    scaler_center=np.array(row["scaler_center"]),
                    scaler_scale=np.array(row["scaler_scale"]),
                    predictions=np.array(row["predictions"]),
                    lower_bounds=np.array(row["lower_bounds"]),
                    upper_bounds=np.array(row["upper_bounds"]),
                    actuals=np.array(row["actuals"]),
                    fit_time_seconds=row["fit_time_seconds"],
                )
            )
        iterations.sort(key=lambda it: it.fold_idx)
    except Exception:  # noqa: BLE001
        iterations = []

    return iterations, predictions_df


def save_wfv_results(
    iterations: list[WFVIteration],
    predictions: pd.Series | pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
    fresh_start: bool = False,
    model: Any | None = None
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(predictions, pd.Series):
        predictions_df = predictions.to_frame(name=predictions.name or "prediction")
    else:
        predictions_df = predictions.copy()

    horizon_attr = predictions_df.attrs.get("horizon")
    if horizon_attr is None:
        horizon_attr = predictions.attrs.get("horizon") if hasattr(predictions, "attrs") else None

    if horizon_attr is None:
        pred_column = next((c for c in predictions_df.columns if c.startswith("pred_h")), "")
        if pred_column.startswith("pred_h"):
            try:
                horizon_attr = int(pred_column.replace("pred_h", ""))
            except ValueError:
                horizon_attr = "unknown"
        else:
            horizon_attr = "unknown"

    predictions_path = output_path / f"predictions_{model_name}_{horizon_attr}.parquet"
    if not fresh_start and predictions_path.exists():
        existing = pd.read_parquet(predictions_path)
        predictions_df = pd.concat([existing, predictions_df], axis=0)
        predictions_df = predictions_df[~predictions_df.index.duplicated(keep="last")].sort_index()
    predictions_df.to_parquet(predictions_path)

    audit_payload: list[dict[str, Any]] = []
    for iteration in iterations:
        audit_payload.append(
            {
                "fold_idx": iteration.fold_idx,
                "train_start": iteration.train_start.isoformat(),
                "train_end": iteration.train_end.isoformat(),
                "test_start": iteration.test_start.isoformat(),
                "test_end": iteration.test_end.isoformat(),
                "selected_features": iteration.selected_features,
                "n_features_input": iteration.n_features_input,
                "n_features_selected": iteration.n_features_selected,
                "scaler_center": iteration.scaler_center.tolist(),
                "scaler_scale": iteration.scaler_scale.tolist(),
                "predictions": iteration.predictions.tolist(),
                "lower_bounds": iteration.lower_bounds.tolist(),
                "upper_bounds": iteration.upper_bounds.tolist(),
                "actuals": iteration.actuals.tolist(),
                "fit_time_seconds": iteration.fit_time_seconds,
            }
        )

    audit_path = output_path / f"audit_{model_name}_{horizon_attr}.json"
    existing_audit: list[dict[str, Any]] = []
    if not fresh_start and audit_path.exists():
        with audit_path.open("r", encoding="utf-8") as handle:
            existing_audit = json.load(handle)
    audit_payload = existing_audit + audit_payload
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2)

    if model is not None and hasattr(model, "save_model"):
        model_save_path = output_path / f"model_{model_name}_{horizon_attr}"
        try:
            model.save_model(model_save_path)
        except Exception as exc:
            logger.warning("Failed to save model {name}: {error}", name=model_name, error=str(exc))


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    # Demo block — update split_name and max_lag to match your feature pipeline output
    DEMO_SPLIT = "train"
    DEMO_MAX_LAG = 5

    processed_dir = Path("data/processed")

    X_path = processed_dir / f"{DEMO_SPLIT}_feature_matrix_ml_lag{DEMO_MAX_LAG}.parquet"
    if not X_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {X_path}. "
            f"Run feature_engineering.py first or update DEMO_SPLIT/DEMO_MAX_LAG."
        )
    X = pd.read_parquet(X_path)

    config_path = Path("configs") / "lag_order_config.json"
    config_data: dict[str, Any] = {}
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))

    config = WFVConfig(
        model_family="ml",
        task_type="regression",
        max_lag_order=int(config_data.get("max_lag_order", 10)),
    )

    target_path = processed_dir / f"{DEMO_SPLIT}_target_h{config.horizon}.parquet"
    y = pd.read_parquet(target_path)

    class _RFWrapper:
        def __init__(self, base_model: RandomForestRegressor) -> None:
            self.base_model = base_model

        def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame | None = None,
            y_val: pd.Series | None = None,
        ) -> None:
            self.base_model.fit(X_train, y_train)

        def predict(self, X_test: pd.DataFrame) -> np.ndarray:
            return self.base_model.predict(X_test)

    model = _RFWrapper(RandomForestRegressor(n_estimators=200, random_state=config.random_state))

    iterations, predictions_df = run_wfv(X, y, model, config)
    save_wfv_results(
        iterations,
        predictions_df,
        model_name="random_forest",
        output_dir=processed_dir,
    )
