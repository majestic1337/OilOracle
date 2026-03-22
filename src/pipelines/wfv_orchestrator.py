"""Walk-forward validation orchestration for oil price forecasting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests


@dataclass(slots=True)
class WFVConfig:
    """Configuration for walk-forward validation."""

    w_train: int = 1000
    w_val: int = 200
    w_step: int = 20
    horizon: int = 1
    strategy: str = "direct"
    window_type: str = "rolling"
    max_lag_order: int = 10
    granger_p_threshold: float = 0.05
    vif_threshold: float = 5.0
    corr_threshold: float = 0.85
    top_k_shap: int = 15
    random_state: int = 42
    dl_mode: str = "per_fold"
    max_consecutive_failures: int = 3

    @property
    def model_kwargs(self) -> dict:
        """Словник що передається в конструктор будь-якої DL моделі."""
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
    actuals: np.ndarray
    fit_time_seconds: float


def _ensure_series(y: pd.Series | pd.DataFrame, horizon: int) -> pd.Series | pd.DataFrame:
    """Ensure y aligns with expected horizon.

    Args:
        y: Target series or dataframe.
        horizon: Forecast horizon used for fallback selection.

    Returns:
        Series for direct strategy or dataframe for MIMO.
    """
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


def _select_features_granger(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WFVConfig,
) -> tuple[list[str], dict[str, float]]:
    """Apply Granger causality filtering.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        config: WFV configuration.

    Returns:
        Tuple of selected feature names and f-statistics map.
    """
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
                verbose=False
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
    """Iteratively remove features with high VIF.

    Args:
        X_train: Training feature matrix.
        config: WFV configuration.

    Returns:
        List of features that pass the VIF threshold.
    """
    features = list(X_train.columns)
    if len(features) <= 1:
        return features

    while True:
        X_values = X_train[features].values
        vif_values = []
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
    """Remove weakly informative features that are highly correlated with stronger ones.

    Args:
        X_train: Training feature matrix.
        f_stats: F-statistics from Granger filtering.
        config: WFV configuration.

    Returns:
        List of retained features after correlation pruning.
    """
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

    remaining = [feat for feat in features if feat not in to_remove]
    return remaining


def _select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WFVConfig,
) -> list[str]:
    """Rank features using LightGBM + SHAP.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        config: WFV configuration.

    Returns:
        List of top-k features by SHAP importance.
    """
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
        # Convert to ndarray to keep LightGBM from storing feature names.
        # This avoids sklearn's feature-name warning when SHAP calls predict.
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
    top_features = (
        importance_series.sort_values(ascending=False).head(top_k).index.tolist()
    )
    return top_features


def select_features_in_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: WFVConfig,
) -> list[str]:
    """Select features within a single fold using Granger, VIF, correlation, and SHAP.

    Args:
        X_train: Training feature matrix.
        y_train: Training target series.
        config: WFV configuration.

    Returns:
        List of selected feature names.
    """
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
    """Generate rolling/expanding window indices.

    Args:
        total: Total number of samples.
        config: WFV configuration.

    Returns:
        Iterator over (train_start, train_end, val_end, test_end) indices.
    """
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


def run_wfv(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    model: Any,
    config: WFVConfig,
) -> tuple[list[WFVIteration], pd.Series]:
    """Run walk-forward validation loop.

    Args:
        X: Feature matrix.
        y: Target series or dataframe.
        model: Model implementing fit(X, y, X_val, y_val) and predict(X).
        config: WFV configuration.

    Returns:
        Tuple of WFVIteration list and prediction series.
    """
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
            logger.warning(
                "MIMO target column not found for horizon {h}; using {col}",
                h=config.horizon,
                col=mimo_target_col,
            )
    else:
        y_model = _ensure_series(y, config.horizon)

    common_index = X.index.intersection(y_model.index)
    X = X.loc[common_index]
    y_model = y_model.loc[common_index]

    iterations: list[WFVIteration] = []
    predictions_list: list[pd.Series] = []
    skip_model = False
    skip_reason = ""
    consecutive_failures = 0

    global_trained = False
    global_selected_features: list[str] | None = None
    global_scaler: RobustScaler | None = None

    for fold_idx, (train_start, train_end, val_end, test_end) in enumerate(
        _iter_windows(len(X), config)
    ):
        train_slice = slice(train_start, train_end)
        val_slice = slice(train_end, val_end)
        test_slice = slice(val_end, test_end)

        X_train = X.iloc[train_slice]
        X_val = X.iloc[val_slice]
        X_test = X.iloc[test_slice]

        if config.dl_mode == "global_train" and global_trained and global_scaler is not None and global_selected_features is not None:
            # Reuse globally trained artifacts for testing
            scaler = global_scaler
            selected_features = global_selected_features
            
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )
            X_test_sel = X_test_scaled[selected_features]
            
            # Create dummy arrays since we skip training
            X_train_sel = pd.DataFrame()
            X_val_sel = pd.DataFrame()
            
            # Still need actuals for audit
            if isinstance(y_test, pd.DataFrame) and mimo_target_col is not None:
                actuals_series = y_test[mimo_target_col]
            else:
                actuals_series = _ensure_series(y_test, config.horizon)
            actuals_array = np.asarray(actuals_series).astype(float)
        else:
            y_train = y_model.iloc[train_slice]
            y_val = y_model.iloc[val_slice]
            y_test = y_model.iloc[test_slice]

            scaler = RobustScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns,
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                index=X_val.index,
                columns=X_val.columns,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns,
            )

            if isinstance(y_train, pd.DataFrame) and mimo_target_col is not None:
                selection_target = y_train[mimo_target_col]
            else:
                selection_target = _ensure_series(y_train, config.horizon)
            selected_features = select_features_in_fold(X_train_scaled, selection_target, config)
            if not selected_features:
                logger.warning("No features selected in fold {fold}; using all features", fold=fold_idx)
                selected_features = list(X_train_scaled.columns)

            X_train_sel = X_train_scaled[selected_features]
            X_val_sel = X_val_scaled[selected_features]
            X_test_sel = X_test_scaled[selected_features]

            if isinstance(y_test, pd.DataFrame) and mimo_target_col is not None:
                actuals_series = y_test[mimo_target_col]
            else:
                actuals_series = _ensure_series(y_test, config.horizon)
            actuals_array = np.asarray(actuals_series).astype(float)

        fit_time = 0.0
        preds_array: np.ndarray = np.array([])

        try:
            start_time = time.perf_counter()
            if config.dl_mode == "global_train" and global_trained:
                pass  # Skip retraining
            else:
                try:
                    model.fit(X_train_sel, y_train, X_val=X_val_sel, y_val=y_val)
                except TypeError:
                    model.fit(X_train_sel, y_train)
                fit_time = time.perf_counter() - start_time
                
                if config.dl_mode == "global_train":
                    global_trained = True
                    global_scaler = scaler
                    global_selected_features = selected_features

            preds = model.predict(X_test_sel)
            preds_array = np.asarray(preds)
            if preds_array.ndim > 1:
                if isinstance(y_model, pd.DataFrame) and mimo_target_col is not None:
                    col_idx = list(y_model.columns).index(mimo_target_col)
                    preds_array = preds_array[:, col_idx]
                else:
                    preds_array = preds_array[:, 0]
            preds_array = preds_array.astype(float)

            predictions_list.append(pd.Series(preds_array, index=X_test_sel.index))
            consecutive_failures = 0
            
        except Exception as exc:  # noqa: BLE001 - keep pipeline running
            consecutive_failures += 1
            if _is_incompatible_model_error(exc, config.horizon):
                skip_model = True
                skip_reason = str(exc)
                logger.warning(
                    "Skipping model after incompatibility on fold {fold}: {error}",
                    fold=fold_idx,
                    error=skip_reason,
                )
                break
                
            logger.error("Model failed on fold {fold}: {error}", fold=fold_idx, error=str(exc))
            
            if consecutive_failures >= config.max_consecutive_failures:
                skip_model = True
                skip_reason = f"Model failed {config.max_consecutive_failures} consecutive times. Aborting WFV."
                logger.error(skip_reason)
                break

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
                actuals=actuals_array,
                fit_time_seconds=fit_time,
            )
        )

    if skip_model:
        iterations = []
        predictions_list = []

    if predictions_list:
        predictions_series = pd.concat(predictions_list).sort_index()
    else:
        predictions_series = pd.Series(dtype=float)

    predictions_series.name = f"pred_h{config.horizon}"
    predictions_series.attrs["horizon"] = config.horizon

    return iterations, predictions_series


def load_cached_results(
    model_name: str,
    horizon: int,
    output_dir: str | Path,
) -> tuple[list[WFVIteration], pd.Series] | None:
    """Load cached predictions/audit if present.

    Args:
        model_name: Name of the model used.
        horizon: Forecast horizon.
        output_dir: Directory for output artifacts.

    Returns:
        Tuple of empty iterations list and predictions series if cache exists,
        otherwise None.
    """
    output_path = Path(output_dir)
    predictions_path = output_path / f"predictions_{model_name}_{horizon}.parquet"
    audit_path = output_path / f"audit_{model_name}_{horizon}.json"

    if not (predictions_path.exists() and audit_path.exists()):
        return None

    try:
        predictions_df = pd.read_parquet(predictions_path)
        if predictions_df.empty:
            logger.warning(
                "Cached predictions empty for {model} horizon {h}; rerunning WFV",
                model=model_name,
                h=horizon,
            )
            return None
        predictions = predictions_df.iloc[:, 0].copy()
        predictions.name = predictions_df.columns[0]
        predictions.attrs["horizon"] = horizon
    except Exception as exc:  # noqa: BLE001 - fallback to recompute
        logger.warning(
            "Failed to load cached predictions for {model} horizon {h}: {error}",
            model=model_name,
            h=horizon,
            error=str(exc),
        )
        return None

    logger.info(
        "Using cached WFV results for {model} horizon {h}; skipping training",
        model=model_name,
        h=horizon,
    )
    return [], predictions


def save_wfv_results(
    iterations: list[WFVIteration],
    predictions: pd.Series,
    model_name: str,
    output_dir: str | Path,
) -> None:
    """Save predictions and audit logs.

    Args:
        iterations: List of WFV iterations.
        predictions: Predictions series with datetime index.
        model_name: Name of the model used.
        output_dir: Directory for output artifacts.

    Returns:
        None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    horizon = predictions.attrs.get("horizon", "unknown")
    predictions_path = output_path / f"predictions_{model_name}_{horizon}.parquet"
    predictions.to_frame(name=predictions.name).to_parquet(predictions_path)

    audit_payload = []
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
                "actuals": iteration.actuals.tolist(),
                "fit_time_seconds": iteration.fit_time_seconds,
            }
        )

    audit_path = output_path / f"audit_{model_name}_{horizon}.json"
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2)

    if iterations:
        avg_features = float(np.mean([it.n_features_selected for it in iterations]))
    else:
        avg_features = 0.0
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
    X = pd.read_parquet(processed_dir / "feature_matrix.parquet")

    config_path = Path("configs") / "lag_order_config.json"
    config_data: dict[str, Any] = {}
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))

    config = WFVConfig(max_lag_order=int(config_data.get("max_lag_order", 10)))

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

    iterations, predictions = run_wfv(X, y, model, config)
    save_wfv_results(iterations, predictions, model_name="random_forest", output_dir=processed_dir)
