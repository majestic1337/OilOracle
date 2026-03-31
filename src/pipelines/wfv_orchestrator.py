"""Walk-forward validation orchestration for oil price forecasting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests

ModelFamily = Literal["ml", "dl", "stat"]


@dataclass(slots=True)
class WFVConfig:
    """Configuration for walk-forward validation."""

    w_train: int = 1000
    w_val: int = 200
    w_step: int = 20
    horizon: int = 1
    strategy: str = "direct"
    model_family: ModelFamily = "ml"
    window_type: str = "rolling"
    max_lag_order: int = 10
    granger_p_threshold: float = 0.1
    vif_threshold: float = 10.0
    corr_threshold: float = 0.85
    top_k_shap: int = 15
    random_state: int = 42
    dl_mode: str = "per_fold"
    dl_recursive: bool = True
    interval_alpha: float = 0.1
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        if self.model_family not in {"ml", "dl", "stat"}:
            raise ValueError("model_family must be one of {'ml', 'dl', 'stat'}")

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
    """Ensure a 1D target series for the configured horizon."""
    if isinstance(y, pd.Series):
        return y

    if y.shape[1] == 1:
        return y.iloc[:, 0]

    for candidate in (f"target_h{horizon}", f"h{horizon}"):
        if candidate in y.columns:
            return y[candidate]

    logger.warning(
        "Unable to identify target column for horizon {h}; using first column",
        h=horizon,
    )
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
        logger.warning(
            "{ctx}: prediction length {got} > expected {exp}; truncating",
            ctx=context,
            got=arr.size,
            exp=expected,
        )
        return arr[:expected]

    logger.warning(
        "{ctx}: prediction length {got} < expected {exp}; padding with last value",
        ctx=context,
        got=arr.size,
        exp=expected,
    )
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
    except Exception as exc:  # noqa: BLE001 - optional capability
        logger.warning("Interval prediction failed: {error}", error=str(exc))
        return None, None

    lower_arr = _align_prediction_length(lower, expected, context="predict_interval(lower)")
    upper_arr = _align_prediction_length(upper, expected, context="predict_interval(upper)")
    return lower_arr, upper_arr



def _select_features_granger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WFVConfig,
) -> tuple[list[str], dict[str, float]]:
    """Apply Granger causality filtering."""
    selected: list[str] = []
    f_stats: dict[str, float] = {}

    if len(y_train) <= config.max_lag_order + 1:
        logger.warning("Not enough observations for Granger tests; skipping filter")
        return list(X_train.columns), {name: 0.0 for name in X_train.columns}

    for feature in X_train.columns:
        combined = pd.concat([y_train, X_train[feature]], axis=1).dropna()
        if combined.shape[0] <= config.max_lag_order + 1:
            logger.warning("Skipping Granger for {feature}: insufficient samples", feature=feature)
            f_stats[feature] = 0.0
            continue

        try:
            results = grangercausalitytests(
                combined,
                maxlag=config.max_lag_order,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - keep pipeline running
            logger.warning(
                "Granger test failed for {feature}: {error}",
                feature=feature,
                error=str(exc),
            )
            f_stats[feature] = 0.0
            continue

        min_p = 1.0
        max_f = 0.0
        for lag_result in results.values():
            f_stat, p_value, *_ = lag_result[0]["ssr_ftest"]
            min_p = min(min_p, float(p_value))
            max_f = max(max_f, float(f_stat))

        f_stats[feature] = max_f
        if min_p < config.granger_p_threshold:
            selected.append(feature)

    if not selected:
        logger.warning("Granger filter removed all features; keeping original set")
        return list(X_train.columns), f_stats

    return selected, f_stats



def _select_features_vif(
    X_train: pd.DataFrame,
    config: WFVConfig,
) -> list[str]:
    """Iteratively remove features with high VIF."""
    features = list(X_train.columns)
    if len(features) <= 1:
        return features

    while True:
        X_values = X_train[features].values
        vif_values: list[float] = []
        try:
            for idx in range(len(features)):
                vif_values.append(float(variance_inflation_factor(X_values, idx)))
        except Exception as exc:  # noqa: BLE001 - keep pipeline running
            logger.warning("VIF computation failed: {error}", error=str(exc))
            break

        max_vif = max(vif_values)
        if max_vif <= config.vif_threshold:
            break

        max_idx = int(np.argmax(vif_values))
        removed = features.pop(max_idx)
        logger.info("Removed {feature} due to high VIF ({vif})", feature=removed, vif=max_vif)
        if len(features) <= 1:
            break

    return features



def _select_features_correlation(
    X_train: pd.DataFrame,
    f_stats: dict[str, float],
    config: WFVConfig,
) -> list[str]:
    """Remove weakly informative features highly correlated with stronger ones."""
    features = list(X_train.columns)
    if len(features) <= 1:
        return features

    corr_matrix = X_train[features].corr().abs()

    def _sort_key(feat: str) -> tuple[float, str]:
        return (f_stats.get(feat, 0.0), feat)

    ordered = sorted(features, key=_sort_key)
    to_remove: set[str] = set()

    for feat in ordered:
        if feat in to_remove:
            continue

        for other in features:
            if other == feat or other in to_remove:
                continue
            stronger = _sort_key(other) > _sort_key(feat)
            if stronger and corr_matrix.loc[feat, other] > config.corr_threshold:
                to_remove.add(feat)
                break

    return [feat for feat in features if feat not in to_remove]



def _select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WFVConfig,
) -> list[str]:
    """Rank features using LightGBM + SHAP."""
    try:
        import lightgbm as lgb  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning("LightGBM unavailable: {error}", error=str(exc))
        return list(X_train.columns)

    try:
        import shap  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency
        logger.warning("SHAP unavailable: {error}", error=str(exc))
        return list(X_train.columns)

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
        else:
            shap_importance = np.abs(shap_array).mean(axis=0)
    except Exception as exc:  # noqa: BLE001 - keep pipeline running
        logger.warning("SHAP ranking failed: {error}", error=str(exc))
        return list(X_train.columns)

    importance_series = pd.Series(shap_importance, index=X_train.columns)
    top_k = min(config.top_k_shap, len(importance_series))
    return importance_series.sort_values(ascending=False).head(top_k).index.tolist()



def select_features_in_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WFVConfig,
) -> list[str]:
    """Select features within a fold using Granger, VIF, correlation, and SHAP."""
    selected, f_stats = _select_features_granger(X_train, y_train, config)

    if not selected:
        logger.warning("Feature selection stopped after Granger: empty feature set")
        return list(X_train.columns)

    vif_features = _select_features_vif(X_train[selected], config)
    if not vif_features:
        logger.warning("Feature selection stopped after VIF: empty feature set")
        return selected

    corr_features = _select_features_correlation(X_train[vif_features], f_stats, config)
    if not corr_features:
        logger.warning("Feature selection stopped after correlation pruning: empty feature set")
        return vif_features

    shap_features = _select_features_shap(X_train[corr_features], y_train, config)
    if not shap_features:
        logger.warning("Feature selection stopped after SHAP: empty feature set")
        return corr_features

    return shap_features



def _iter_windows(
    total: int,
    config: WFVConfig,
) -> Iterable[tuple[int, int, int, int]]:
    """Generate rolling/expanding window indices."""
    if config.window_type not in {"rolling", "expanding"}:
        raise ValueError("window_type must be 'rolling' or 'expanding'")

    fold = 0
    while True:
        if config.window_type == "rolling":
            train_start = fold * config.w_step
        else:
            train_start = 0

        train_end = train_start + config.w_train + (
            fold * config.w_step if config.window_type == "expanding" else 0
        )
        val_end = train_end + config.w_val
        test_end = val_end + config.w_step

        if test_end > total:
            break

        yield train_start, train_end, val_end, test_end
        fold += 1



def _extract_target_series(
    y_data: pd.Series | pd.DataFrame,
    horizon: int,
    mimo_target_col: str | None,
) -> pd.Series:
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
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Recursive DL inference (step-by-step with observed test updates)."""
    predictions: list[float] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    fit_time_total = 0.0

    has_interval = callable(getattr(model, "predict_interval", None))

    history_X = X_history.copy()
    history_y = y_history.copy()

    for step_idx in range(len(X_test)):
        X_step = X_test.iloc[[step_idx]]

        start_time = time.perf_counter()
        _safe_fit_model(model, history_X, history_y, X_val=None, y_val=None)
        fit_time_total += time.perf_counter() - start_time

        step_pred_raw = model.predict(X_step)
        step_pred = _align_prediction_length(
            step_pred_raw,
            expected=1,
            context=f"dl_recursive_predict_fold_step_{step_idx}",
        )[0]
        predictions.append(float(step_pred))

        if has_interval:
            lower_step, upper_step = _safe_predict_interval(
                model,
                X_step,
                expected=1,
                alpha=interval_alpha,
            )
            lower_values.append(float(lower_step[0]) if lower_step is not None else np.nan)
            upper_values.append(float(upper_step[0]) if upper_step is not None else np.nan)

        actual_value = float(y_test.iloc[step_idx])
        history_X = pd.concat([history_X, X_step], axis=0)
        history_y = pd.concat(
            [
                history_y,
                pd.Series([actual_value], index=X_step.index, name=history_y.name),
            ],
            axis=0,
        )

    preds_array = np.asarray(predictions, dtype=float)

    if has_interval:
        lower_arr = np.asarray(lower_values, dtype=float)
        upper_arr = np.asarray(upper_values, dtype=float)
        if np.isnan(lower_arr).all() or np.isnan(upper_arr).all():
            return preds_array, None, None, fit_time_total
        return preds_array, lower_arr, upper_arr, fit_time_total

    return preds_array, None, None, fit_time_total

def run_wfv(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    model: Any,
    config: WFVConfig,
) -> tuple[list[WFVIteration], pd.DataFrame]:
    """Run walk-forward validation loop with feature and target scaling."""
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
    X = X.loc[common_index].sort_index()
    y_model = y_model.loc[common_index].sort_index()

    iterations: list[WFVIteration] = []
    predictions_list: list[pd.DataFrame] = []
    pred_col_name = f"pred_h{config.horizon}"
    
    forbidden_cols = {"unique_id", "ds", "y"}
    static_features = [col for col in X.columns if col not in forbidden_cols]
    
    skip_model = False
    consecutive_failures = 0

    windows = list(_iter_windows(len(X), config))
    progress_bar = tqdm(
        windows, 
        desc=f"WFV [{config.model_family.upper()} | h={config.horizon}]", 
        unit="fold",
        dynamic_ncols=True
    )

    for fold_idx, (train_start, train_end, val_end, test_end) in enumerate(progress_bar):
        try:
            embargo_steps = config.horizon - 1
            test_start_embargo = val_end + embargo_steps
            
            if test_start_embargo >= test_end:
                continue

            fold_model = clone(model)

            X_train = X.iloc[train_start:train_end]
            X_val = X.iloc[train_end:val_end]
            X_test = X.iloc[test_start_embargo:test_end]

            y_train_raw = y_model.iloc[train_start:train_end]
            y_val_raw = y_model.iloc[train_end:val_end]
            y_test_raw = y_model.iloc[test_start_embargo:test_end]

            y_train_target = _extract_target_series(y_train_raw, config.horizon, mimo_target_col)
            y_val_target = _extract_target_series(y_val_raw, config.horizon, mimo_target_col)
            y_test_target = _extract_target_series(y_test_raw, config.horizon, mimo_target_col)

            # Target Scaling
            y_scaler = RobustScaler()
            y_train_fit = pd.Series(
                y_scaler.fit_transform(y_train_target.values.reshape(-1, 1)).flatten(),
                index=y_train_target.index,
                name=y_train_target.name
            )
            y_val_fit = pd.Series(
                y_scaler.transform(y_val_target.values.reshape(-1, 1)).flatten(),
                index=y_val_target.index,
                name=y_val_target.name
            )
            y_test_fit = pd.Series(
                y_scaler.transform(y_test_target.values.reshape(-1, 1)).flatten(),
                index=y_test_target.index,
                name=y_test_target.name
            )

            # Feature Scaling
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

            fit_time = 0.0
            lower_array, upper_array = None, None

            if config.model_family == "dl" and config.dl_recursive:
                X_history = pd.concat([X_train_sel, X_val_sel], axis=0).sort_index()
                y_history = pd.concat([y_train_fit, y_val_fit], axis=0).sort_index()
                preds_array, lower_array, upper_array, fit_time = _run_dl_recursive_forecast(
                    model=fold_model,
                    X_history=X_history,
                    y_history=y_history,
                    X_test=X_test_sel,
                    y_test=y_test_fit,
                    interval_alpha=config.interval_alpha,
                )
            else:
                start_time = time.perf_counter()
                _safe_fit_model(fold_model, X_train_sel, y_train_fit, X_val=X_val_sel, y_val=y_val_fit)
                fit_time = time.perf_counter() - start_time

                preds_raw = fold_model.predict(X_test_sel)
                preds_array = _align_prediction_length(preds_raw, len(X_test_sel), f"predict_fold_{fold_idx}")
                lower_array, upper_array = _safe_predict_interval(fold_model, X_test_sel, len(X_test_sel), config.interval_alpha)

            # Inverse Target Scaling
            preds_array = y_scaler.inverse_transform(preds_array.reshape(-1, 1)).flatten()
            if lower_array is not None:
                lower_array = y_scaler.inverse_transform(lower_array.reshape(-1, 1)).flatten()
            if upper_array is not None:
                upper_array = y_scaler.inverse_transform(upper_array.reshape(-1, 1)).flatten()

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

        except Exception as exc:
            consecutive_failures += 1
            if _is_incompatible_model_error(exc, config.horizon) or consecutive_failures >= config.max_consecutive_failures:
                skip_model = True
                break
            continue

    if skip_model:
        return [], pd.DataFrame(columns=[pred_col_name])

    predictions_df = pd.concat(predictions_list, axis=0).sort_index() if predictions_list else pd.DataFrame(columns=[pred_col_name])
    predictions_df.attrs["horizon"] = config.horizon
    return iterations, predictions_df

def load_cached_results(
    model_name: str,
    horizon: int,
    output_dir: str | Path,
) -> tuple[list[WFVIteration], pd.DataFrame] | None:
    """Load cached predictions/audit if present."""
    output_path = Path(output_dir)
    predictions_path = output_path / f"predictions_{model_name}_{horizon}.parquet"
    audit_path = output_path / f"audit_{model_name}_{horizon}.json"

    if not (predictions_path.exists() and audit_path.exists()):
        return None

    try:
        predictions_df = pd.read_parquet(predictions_path)
    except Exception as exc:  # noqa: BLE001 - fallback to recompute
        logger.warning(
            "Failed to load cached predictions for {model} horizon {h}: {error}",
            model=model_name,
            h=horizon,
            error=str(exc),
        )
        return None

    if predictions_df.empty:
        logger.warning(
            "Cached predictions empty for {model} horizon {h}; rerunning WFV",
            model=model_name,
            h=horizon,
        )
        return None

    if not isinstance(predictions_df.index, pd.DatetimeIndex):
        predictions_df = predictions_df.copy()
        predictions_df.index = pd.to_datetime(predictions_df.index)

    predictions_df = predictions_df.sort_index()
    predictions_df.attrs["horizon"] = horizon

    logger.info(
        "Using cached WFV results for {model} horizon {h}; skipping training",
        model=model_name,
        h=horizon,
    )
    return [], predictions_df



def save_wfv_results(
    iterations: list[WFVIteration],
    predictions: pd.Series | pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Save predictions and audit logs."""
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
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2)

    avg_features = float(np.mean([it.n_features_selected for it in iterations])) if iterations else 0.0
    total_time = float(np.sum([it.fit_time_seconds for it in iterations])) if iterations else 0.0

    logger.info("Saved predictions to {path}", path=predictions_path)
    logger.info("Saved audit log to {path}", path=audit_path)
    logger.info(
        "WFV summary: folds={folds}, avg_features={avg}, total_fit_time={time}",
        folds=len(iterations),
        avg=avg_features,
        time=total_time,
    )


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    processed_dir = Path("data/processed")

    X_path = processed_dir / "feature_matrix_ml.parquet"
    if not X_path.exists():
        X_path = processed_dir / "feature_matrix.parquet"
    X = pd.read_parquet(X_path)

    config_path = Path("configs") / "lag_order_config.json"
    config_data: dict[str, Any] = {}
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))

    config = WFVConfig(
        model_family="ml",
        max_lag_order=int(config_data.get("max_lag_order", 10)),
    )

    target_path = processed_dir / f"target_h{config.horizon}.parquet"
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
