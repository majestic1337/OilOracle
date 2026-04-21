"""Evaluation pipeline for forecasting experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.models.base import calculate_window_mase, diebold_mariano_test


def pesaran_timmermann_test(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Calculate Pesaran-Timmermann test for directional accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    
    n = len(y_t)
    if n == 0:
        return {"pt_statistic": float("nan"), "p_value": float("nan")}
        
    p_hat = np.mean(np.sign(y_p) == np.sign(y_t))
    p_y = np.mean(y_t > 0)
    p_pred = np.mean(y_p > 0)
    
    p_star = p_y * p_pred + (1 - p_y) * (1 - p_pred)
    
    var_diff = (
        p_star * (1 - p_star) / n
        + ((2 * p_y - 1) ** 2) * p_pred * (1 - p_pred) / n
        + ((2 * p_pred - 1) ** 2) * p_y * (1 - p_y) / n
    )
    
    if var_diff <= 0:
        return {"pt_statistic": float("nan"), "p_value": float("nan")}
        
    s_pt = (p_hat - p_star) / np.sqrt(var_diff)
    p_value = 2 * (1 - norm.cdf(np.abs(s_pt)))
    
    return {"pt_statistic": float(s_pt), "p_value": float(p_value)}


def _to_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(data, pd.Series):
        series = data
    else:
        series = data.iloc[:, 0]

    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.to_datetime(series.index)

    return series.sort_index()


def _select_column(df: pd.DataFrame, tokens: list[str]) -> str | None:
    for column in df.columns:
        lowered = column.lower()
        if any(token in lowered for token in tokens):
            return column
    return None


def _extract_prediction_series(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series | None]:
    pred_col = _select_column(df, ["pred", "prediction", "q0.5", "q50", "median", "p50"])
    if pred_col is None:
        pred_col = df.columns[0]

    series = df[pred_col]

    lower_col = _select_column(df, ["lower", "q0.1", "q10", "p10"])
    upper_col = _select_column(df, ["upper", "q0.9", "q90", "p90"])
    interval_width: pd.Series | None = None
    if lower_col and upper_col:
        interval_width = df[upper_col] - df[lower_col]

    return _to_series(series), interval_width


def _parse_prediction_name(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected prediction filename: {path.name}")

    horizon = parts[-1].lstrip("h")
    model_name = "_".join(parts[1:-1])
    return f"{model_name}_h{horizon}"


def validate_predictions_file(df: pd.DataFrame, path_name: str) -> bool:
    """Validate prediction dataframe schema and integrity."""
    if df.empty:
        logger.error(f"Validation failed for {path_name}: DataFrame is completely empty.")
        return False
        
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error(f"Validation failed for {path_name}: Index is {type(df.index).__name__}, expected DatetimeIndex.")
        return False

    if df.index.hasnans:
        logger.error(f"Validation failed for {path_name}: Index contains NaT values.")
        return False

    if df.columns.empty:
        logger.error(f"Validation failed for {path_name}: No columns present.")
        return False
        
    if df.isna().all().any():
        logger.error(f"Validation failed for {path_name}: Contains entirely NaN columns.")
        return False

    return True


def _extract_horizon_from_model_key(model_key: str) -> int:
    match = re.search(r"_h(\d+)$", model_key)
    if not match:
        return 1
    return int(match.group(1))


def load_all_predictions(
    processed_dir: str | Path,
    test_start: str = "2024-01-01",
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load all prediction files from the processed directory.

    Per-file series are filtered to ``test_start`` onward. Cross-model /
    cross-horizon index alignment is not applied here — align per horizon
    in the caller when evaluating.

    Args:
        processed_dir: Directory containing prediction parquet files.
        test_start: Start date for test period filtering (inclusive).

    Returns:
        Tuple of (predictions, interval_widths) where interval_widths maps
        model keys to interval width series. Keys absent from interval_widths
        indicate models without prediction intervals.
    """
    processed_path = Path(processed_dir)
    files = sorted(processed_path.glob("predictions_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No predictions_*.parquet files found in {processed_path}")

    predictions: dict[str, pd.Series] = {}
    interval_widths: dict[str, pd.Series] = {}
    excluded_files = []

    for file_path in files:
        try:
            df = pd.read_parquet(file_path)
            if not validate_predictions_file(df, file_path.name):
                excluded_files.append(file_path.name)
                continue
                
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            series, interval_width = _extract_prediction_series(df)
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
                
            try:
                filtered_series = series.loc[test_start:]
            except KeyError:
                filtered_series = series[series.index >= pd.Timestamp(test_start)]
                
            if filtered_series.empty:
                logger.error(f"File {file_path.name} is empty after filtering by test_start={test_start}.")
                excluded_files.append(file_path.name)
                continue

            key = _parse_prediction_name(file_path)
            predictions[key] = filtered_series
            if interval_width is not None:
                interval_widths[key] = interval_width.loc[filtered_series.index]
        except Exception as exc:
            logger.error(f"Failed to process {file_path.name}: {exc}")
            excluded_files.append(file_path.name)

    if not predictions:
        raise RuntimeError(f"All prediction files were invalid or empty. Excluded: {excluded_files}")

    logger.info(
        "Loaded {n_models} predictions. Excluded: {excl}",
        n_models=len(predictions),
        excl=excluded_files,
    )

    return predictions, interval_widths


def _find_baseline_key(predictions_dict: dict[str, pd.Series]) -> str:
    # Prefer regression (non-cls) random walk over classification variant
    candidates = [
        key for key in predictions_dict
        if "random_walk" in key or "randomwalk" in key
    ]
    if not candidates:
        raise ValueError("Random Walk baseline not found in predictions")
    # Prefer non-classification baseline for Experiment A regression metrics
    regression_candidates = [k for k in candidates if "_cls" not in k]
    return regression_candidates[0] if regression_candidates else candidates[0]


def _align_series(y_test: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    common_index = y_test.index.intersection(y_pred.index)
    return y_test.loc[common_index], y_pred.loc[common_index]


def evaluate_experiment_a(
    predictions_dict: dict[str, pd.Series],
    y_test: pd.Series,
    y_train_last_window: pd.Series,
) -> pd.DataFrame:
    """Evaluate regression metrics and DM test for Experiment A.

    Args:
        predictions_dict: Mapping of model names to prediction series.
        y_test: Test target series.
        y_train_last_window: Last training window for MASE denominator.

    Returns:
        DataFrame with metrics per model.
    """
    y_test_series = _to_series(y_test)
    train_window = _to_series(y_train_last_window)
    # Zero-naive denominator: mean(|y_t|) for stationary log-returns.
    # Consistent with calculate_mase(zero_naive=True) in base.py.
    # Do NOT use np.diff here — that computes random-walk naive which
    # is inappropriate for mean-zero return series.
    mae_naive_train = float(np.mean(np.abs(train_window.values)))

    if not np.isfinite(mae_naive_train) or mae_naive_train <= 0.0:
        logger.warning(
            "Invalid MASE denominator ({value}); fallback to 1.0",
            value=mae_naive_train,
        )
        mae_naive_train = 1.0

    baseline_key: str | None = None
    baseline_errors: pd.Series | None = None
    try:
        baseline_key = _find_baseline_key(predictions_dict)
        baseline_pred = predictions_dict[baseline_key]
        y_test_aligned, baseline_pred = _align_series(y_test_series, baseline_pred)
        baseline_errors = y_test_aligned - baseline_pred
    except ValueError:
        logger.warning("Random Walk baseline not found; DM test will be skipped for Experiment A.")
        baseline_errors = None

    rows: list[dict[str, Any]] = []
    for model_name, preds in predictions_dict.items():
        y_test_model, preds_model = _align_series(y_test_series, preds)

        if len(y_test_model) == 0:
            logger.warning(f"Skipping evaluate_experiment_a for {model_name}: aligned series is empty.")
            continue

        mae = float(np.mean(np.abs(y_test_model - preds_model)))
        mase = mae / mae_naive_train
        rmse = float(np.sqrt(np.mean((y_test_model - preds_model) ** 2)))

        if baseline_errors is not None and model_name == baseline_key:
            dm_stat = 0.0
            dm_p = 1.0
            beats_rw = False
        elif baseline_errors is not None and baseline_key is not None:
            baseline_pred_model = predictions_dict[baseline_key]
            common_index_dm = y_test_model.index.intersection(baseline_pred_model.index)
            if len(common_index_dm) == 0:
                logger.warning(
                    "Skipping DM test for {model}: no overlap with baseline predictions",
                    model=model_name,
                )
                dm_stat = float("nan")
                dm_p = float("nan")
                beats_rw = False
            else:
                model_errors = y_test_model.loc[common_index_dm] - preds_model.loc[common_index_dm]
                baseline_errors_model = (
                    y_test_model.loc[common_index_dm] - baseline_pred_model.loc[common_index_dm]
                )
                horizon = _extract_horizon_from_model_key(model_name)
                dm_result = diebold_mariano_test(model_errors, baseline_errors_model, h=horizon)
                dm_stat = float(dm_result["dm_statistic"])
                dm_p = float(dm_result["p_value"])
                beats_rw = bool(dm_p < 0.05 and dm_stat > 0)
        else:
            dm_stat = float("nan")
            dm_p = float("nan")
            beats_rw = False

        rows.append(
            {
                "model": model_name,
                "MASE": float(mase),
                "RMSE": rmse,
                "MAE": mae,
                "DM_stat": dm_stat,
                "DM_pvalue": dm_p,
                "beats_rw": beats_rw,
            }
        )

    if not rows:
        logger.warning("No valid model results for experiment_a at this horizon")
        return pd.DataFrame(
            columns=["MASE", "RMSE", "MAE", "DM_stat", "DM_pvalue", "beats_rw"]
        ).rename_axis("model")

    results = pd.DataFrame(rows).set_index("model").sort_index()
    return results


def _roc_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        logger.warning("ROC-AUC undefined for constant scores or single-class targets")
        return float("nan")


def evaluate_experiment_b(
    predictions_dict: dict[str, pd.Series],
    y_test: pd.Series,
    interval_widths: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    y_test_series = _to_series(y_test)

    rows: list[dict[str, Any]] = []
    for model_name, preds in predictions_dict.items():
        y_test_model, preds_model = _align_series(y_test_series, preds)
        
        if len(y_test_model) == 0:
            logger.warning("Skipping evaluate_experiment_b for {model}: aligned series is empty.", model=model_name)
            continue
            
        # Detect classification models by name suffix rather than value range.
        # Value-range detection misclassifies log-return regression forecasts
        # (typically in [-0.15, 0.15]) as probability scores.
        is_proba = "_cls" in model_name

        threshold = 0.5 if is_proba else 0.0
        
        y_true = (y_test_model > 0).astype(int).to_numpy()
        y_pred = (preds_model > threshold).astype(int).to_numpy()

        mcc = float(matthews_corrcoef(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        acc = float(accuracy_score(y_true, y_pred))

        scores = preds_model.to_numpy(dtype=float)
        
        if not is_proba:
            interval_width = None
            if interval_widths is not None:
                interval_width = interval_widths.get(model_name)
            if interval_width is not None:
                aligned_interval = interval_width.loc[preds_model.index].to_numpy(dtype=float)
                denom = np.maximum(aligned_interval / 2.0, 1e-12)
                scores = scores / denom

        roc_auc = _roc_auc_from_scores(y_true, scores)

        y_test_raw = y_test_model.to_numpy()
        if is_proba:
            y_pred_raw = preds_model.to_numpy() - threshold
        else:
            y_pred_raw = preds_model.to_numpy()

        da = float(np.mean(np.sign(y_pred_raw) == np.sign(y_test_raw)))
        abs_y_true = np.abs(y_test_raw)
        sum_abs = np.sum(abs_y_true)
        wda = float(np.sum(abs_y_true * (np.sign(y_pred_raw) == np.sign(y_test_raw))) / sum_abs) if sum_abs > 0 else float("nan")

        pt_res = pesaran_timmermann_test(y_test_raw, y_pred_raw)

        rows.append(
            {
                "model": model_name,
                "MCC": mcc,
                "F1_macro": f1,
                "Accuracy": acc,
                "ROC_AUC": roc_auc,
                "DA": da,
                "WDA": wda,
                "PT_stat": pt_res["pt_statistic"],
                "PT_pvalue": pt_res["p_value"],
            }
        )

    if not rows:
        logger.warning("No valid model results for experiment_b at this horizon")
        return pd.DataFrame(
            columns=["MCC", "F1_macro", "Accuracy", "ROC_AUC", "DA", "WDA", "PT_stat", "PT_pvalue"]
        ).rename_axis("model")

    results = pd.DataFrame(rows).set_index("model").sort_index()
    return results


def compare_window_types(
    rolling_results: pd.DataFrame,
    expanding_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare rolling vs expanding window strategies.

    Args:
        rolling_results: Metrics from rolling windows.
        expanding_results: Metrics from expanding windows.

    Returns:
        DataFrame describing window effects per model.
    """
    common_models = rolling_results.index.intersection(expanding_results.index)
    rolling = rolling_results.loc[common_models]
    expanding = expanding_results.loc[common_models]

    summary = pd.DataFrame(index=common_models)
    summary["MASE_rolling"] = rolling.get("MASE")
    summary["MASE_expanding"] = expanding.get("MASE")
    summary["delta_mase"] = summary["MASE_expanding"] - summary["MASE_rolling"]
    summary["better_window"] = np.where(summary["delta_mase"] < 0, "expanding", "rolling")

    winners = summary.index[summary["better_window"] == "expanding"].tolist()
    losers = summary.index[summary["better_window"] == "rolling"].tolist()
    logger.info("Models benefiting from expanding windows: {models}", models=winners)
    logger.info("Models hurt by expanding windows (signal dilution): {models}", models=losers)

    return summary


def regime_analysis(
    predictions_dict: dict[str, pd.Series],
    y_test: pd.Series,
    regimes: dict[str, tuple[str, str]],
    y_train_last_window: pd.Series,
) -> pd.DataFrame:
    """Compute regime-specific metrics within the test period.

    Args:
        predictions_dict: Mapping of model names to prediction series.
        y_test: Test target series.
        regimes: Mapping of regime names to date ranges.
        y_train_last_window: Last training window used as MASE denominator.
            Must be the same window used in evaluate_experiment_a for
            comparability of MASE scores across experiments.

    Returns:
        DataFrame indexed by model and regime with MASE/RMSE/MAE.
    """
    y_test_series = _to_series(y_test)
    rows: list[dict[str, Any]] = []

    for regime_name, (start, end) in regimes.items():
        y_regime = y_test_series.loc[start:end]
        if y_regime.empty:
            logger.warning("No data for regime {regime}", regime=regime_name)
            continue

        for model_name, preds in predictions_dict.items():
            y_regime_model, preds_model = _align_series(y_regime, preds)
            if y_regime_model.empty:
                continue

            mase = calculate_window_mase(y_train_last_window, y_regime_model, preds_model, m=1)
            rmse = float(np.sqrt(np.mean((y_regime_model - preds_model) ** 2)))
            mae = float(np.mean(np.abs(y_regime_model - preds_model)))

            rows.append(
                {
                    "model": model_name,
                    "regime": regime_name,
                    "MASE": float(mase),
                    "RMSE": rmse,
                    "MAE": mae,
                }
            )

    if not rows:
        logger.warning("No valid regime results for this horizon; returning empty DataFrame")
        return pd.DataFrame(
            columns=["MASE", "RMSE", "MAE"]
        ).rename_axis(["model", "regime"])
    return pd.DataFrame(rows).set_index(["model", "regime"]).sort_index()


def generate_results_tables(all_results: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    """Persist evaluation tables and log top performers.

    Args:
        all_results: Mapping of result table names to DataFrames.
        output_dir: Directory for saving CSV tables.

    Returns:
        None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if "experiment_a" in all_results:
        all_results["experiment_a"].to_csv(output_path / "results_experiment_a.csv")
    else:
        experiment_a_keys = sorted(
            key for key in all_results if key.startswith("experiment_a_h")
        )
        for key in experiment_a_keys:
            all_results[key].to_csv(output_path / f"results_{key}.csv")

    if "experiment_b" in all_results:
        all_results["experiment_b"].to_csv(output_path / "results_experiment_b.csv")
    else:
        experiment_b_keys = sorted(
            key for key in all_results if key.startswith("experiment_b_h")
        )
        for key in experiment_b_keys:
            all_results[key].to_csv(output_path / f"results_{key}.csv")

    if "regimes" in all_results:
        all_results["regimes"].to_csv(output_path / "results_regimes.csv")

    horizon_keys = sorted(
        key for key in all_results if key.startswith("experiment_a_h")
    )
    for horizon_key in horizon_keys:
        if horizon_key in all_results:
            result = all_results[horizon_key]
            if "MASE" in result.columns:
                top_models = result.sort_values("MASE").head(3)
                logger.info(
                    "Top-3 models by MASE for {key}: {models}",
                    key=horizon_key,
                    models=top_models.index.tolist(),
                )


if __name__ == "__main__":
    # Update DEMO_SPLIT and DEMO_MAX_LAG to match your feature pipeline output
    DEMO_SPLIT = "train"
    DEMO_MAX_LAG = 5

    processed_dir = Path("data/processed")
    predictions, interval_widths = load_all_predictions(processed_dir)

    preds_by_horizon: dict[int, dict[str, pd.Series]] = {}
    for model_key, series in predictions.items():
        horizon = _extract_horizon_from_model_key(model_key)
        preds_by_horizon.setdefault(horizon, {})[model_key] = series

    if not preds_by_horizon:
        raise ValueError("No predictions with horizon suffix were found (expected *_h<k>).")
    
    TEST_START = "2024-01-01"

    def load_test_target(horizon: int) -> pd.Series:
        for split in ("test", "val", "train"):
            target_path = processed_dir / f"{split}_target_h{horizon}.parquet"
            if target_path.exists():
                series = pd.read_parquet(target_path).iloc[:, 0]
                filtered = series.loc[TEST_START:]
                if not filtered.empty:
                    return filtered
        raise FileNotFoundError(
            f"No target file for h={horizon} with data after {TEST_START}"
        )

    config_path = Path("configs") / "wfv_config.json"
    w_train = 1000
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            w_train = int(config_data.get("w_train", w_train))
        except Exception as exc:  # noqa: BLE001 - fallback to default
            logger.warning("Failed to read w_train from config: {error}", error=str(exc))

    train_df = pd.read_parquet(processed_dir / f"{DEMO_SPLIT}_returns.parquet")
    if "brent_return" in train_df.columns:
        train_series = train_df["brent_return"]
    else:
        train_series = train_df.iloc[:, 0]

    y_train_last_window = train_series.tail(w_train)

    regimes = {
        "opec_cuts": ("2024-01-01", "2024-06-30"),
        "normalization": ("2024-07-01", "2026-03-10"),
    }
    all_results = {}

    regimes_parts: list[pd.DataFrame] = []
    for horizon in sorted(preds_by_horizon):
        preds_h = preds_by_horizon[horizon]

        # Align all models for this horizon to their common index (not across horizons).
        if preds_h:
            h_common: pd.DatetimeIndex | None = None
            for series in preds_h.values():
                h_common = series.index if h_common is None else h_common.intersection(series.index)
            h_common = h_common.sort_values()
            preds_h = {k: v.loc[h_common] for k, v in preds_h.items()}
            h_interval_widths = {
                k: v.loc[h_common]
                for k, v in interval_widths.items()
                if k in preds_h and k in interval_widths
            }
        else:
            h_interval_widths = {}

        y_test_h = load_test_target(horizon=horizon)

        all_results[f"experiment_a_h{horizon}"] = evaluate_experiment_a(
            preds_h,
            y_test_h,
            y_train_last_window,
        )

        all_results[f"experiment_b_h{horizon}"] = evaluate_experiment_b(
            preds_h,
            y_test_h,
            interval_widths=h_interval_widths,
        )
        regime_results_h = regime_analysis(preds_h, y_test_h, regimes, y_train_last_window)
        regime_results_h["horizon"] = horizon
        all_results[f"regimes_h{horizon}"] = regime_results_h
        regimes_parts.append(regime_results_h)

    if regimes_parts:
        all_results["regimes"] = pd.concat(regimes_parts)

    generate_results_tables(all_results, output_dir=processed_dir)

    experiment_a_keys = sorted(
        [key for key in all_results if key.startswith("experiment_a_h")],
        key=lambda k: int(re.search(r"_h(\d+)$", k).group(1)) if re.search(r"_h(\d+)$", k) else 0
    )
    
    for key in experiment_a_keys:
        print(f"\nExperiment A Summary ({key})\n")
        print(all_results[key].to_string())