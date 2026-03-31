"""Evaluation pipeline for forecasting experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from src.models.base import calculate_window_mase, diebold_mariano_test


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


def _extract_prediction_series(df: pd.DataFrame) -> pd.Series:
    pred_col = _select_column(df, ["pred", "prediction", "q0.5", "q50", "median", "p50"])
    if pred_col is None:
        pred_col = df.columns[0]

    series = df[pred_col]

    lower_col = _select_column(df, ["lower", "q0.1", "q10", "p10"])
    upper_col = _select_column(df, ["upper", "q0.9", "q90", "p90"])
    if lower_col and upper_col:
        interval_width = df[upper_col] - df[lower_col]
        series.attrs["interval_width"] = interval_width

    return _to_series(series)


def _parse_prediction_name(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected prediction filename: {path.name}")

    horizon = parts[-1].lstrip("h")
    model_name = "_".join(parts[1:-1])
    return f"{model_name}_h{horizon}"


def load_all_predictions(processed_dir: str | Path) -> dict[str, pd.Series]:
    """Load all prediction files from the processed directory and align indices.

    Args:
        processed_dir: Directory containing prediction parquet files.

    Returns:
        Dictionary mapping model_horizon names to aligned prediction series.
    """
    processed_path = Path(processed_dir)
    files = sorted(processed_path.glob("predictions_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No predictions_*.parquet files found in {processed_path}")

    predictions: dict[str, pd.Series] = {}

    for file_path in files:
        df = pd.read_parquet(file_path)
        series = _extract_prediction_series(df)
        key = _parse_prediction_name(file_path)
        predictions[key] = series

    if not predictions:
        return predictions

    # Знаходження спільного перетину індексів для всіх моделей
    common_index = predictions[list(predictions.keys())[0]].index
    for series in predictions.values():
        common_index = common_index.intersection(series.index)

    if common_index.empty:
        logger.warning("Common index across all predictions is empty.")

    # Вирівнювання всіх прогнозів за спільним індексом
    for key in predictions:
        predictions[key] = predictions[key].loc[common_index]

    logger.info("Aligned all predictions to a common index of length {n}", n=len(common_index))

    return predictions


def _find_baseline_key(predictions_dict: dict[str, pd.Series]) -> str:
    for key in predictions_dict:
        if "random_walk" in key or "randomwalk" in key:
            return key
    raise ValueError("Random Walk baseline not found in predictions")


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

    baseline_key = _find_baseline_key(predictions_dict)
    baseline_pred = predictions_dict[baseline_key]
    y_test_aligned, baseline_pred = _align_series(y_test_series, baseline_pred)
    baseline_errors = y_test_aligned - baseline_pred

    rows: list[dict[str, Any]] = []
    for model_name, preds in predictions_dict.items():
        y_test_model, preds_model = _align_series(y_test_series, preds)

        mase = calculate_window_mase(train_window, y_test_model, preds_model, m=1)
        rmse = float(np.sqrt(np.mean((y_test_model - preds_model) ** 2)))
        mae = float(np.mean(np.abs(y_test_model - preds_model)))

        if model_name == baseline_key:
            dm_stat = 0.0
            dm_p = 1.0
        else:
            model_errors = y_test_model - preds_model
            horizon = 1
            if model_name.endswith("_h7"):
                horizon = 7
            dm_result = diebold_mariano_test(model_errors, baseline_errors, h=horizon)
            dm_stat = float(dm_result["dm_statistic"])
            dm_p = float(dm_result["p_value"])

        beats_rw = bool(dm_p < 0.05 and dm_stat < 0)

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
) -> pd.DataFrame:
    y_test_series = _to_series(y_test)

    rows: list[dict[str, Any]] = []
    for model_name, preds in predictions_dict.items():
        y_test_model, preds_model = _align_series(y_test_series, preds)
        y_true = (y_test_model > 0).astype(int).to_numpy()
        y_pred = (preds_model > 0).astype(int).to_numpy()

        mcc = float(matthews_corrcoef(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        acc = float(accuracy_score(y_true, y_pred))

        scores = np.abs(preds_model.to_numpy(dtype=float))
        interval_width = preds_model.attrs.get("interval_width")
        
        if interval_width is not None:
            aligned_interval = interval_width.loc[preds_model.index].to_numpy(dtype=float)
            denom = np.maximum(aligned_interval / 2.0, 1e-12)
            scores = scores / denom
            
        max_score = float(np.max(scores)) if scores.size else 0.0
        if max_score > 0:
            scores = scores / max_score

        roc_auc = _roc_auc_from_scores(y_true, scores)

        rows.append(
            {
                "model": model_name,
                "MCC": mcc,
                "F1_macro": f1,
                "Accuracy": acc,
                "ROC_AUC": roc_auc,
            }
        )

    results = pd.DataFrame(rows).set_index("model").sort_index()
    return results


def compare_horizons(results_h1: pd.DataFrame, results_h7: pd.DataFrame) -> pd.DataFrame:
    """Compare metric shifts between horizons h=1 and h=7.

    Args:
        results_h1: Metrics for horizon 1.
        results_h7: Metrics for horizon 7.

    Returns:
        DataFrame with side-by-side metrics and MASE delta.
    """
    common_models = results_h1.index.intersection(results_h7.index)
    h1 = results_h1.loc[common_models].copy()
    h7 = results_h7.loc[common_models].copy()

    combined = pd.DataFrame(index=common_models)
    for col in h1.columns:
        combined[f"{col}_h1"] = h1[col]
    for col in h7.columns:
        combined[f"{col}_h7"] = h7[col]

    if "MASE_h1" in combined.columns and "MASE_h7" in combined.columns:
        combined["delta_mase"] = (combined["MASE_h7"] - combined["MASE_h1"]) / combined["MASE_h1"]

    return combined


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
) -> pd.DataFrame:
    """Compute regime-specific metrics within the test period.

    Args:
        predictions_dict: Mapping of model names to prediction series.
        y_test: Test target series.
        regimes: Mapping of regime names to date ranges.

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

            mase = calculate_window_mase(y_regime_model, y_regime_model, preds_model, m=1)
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
        if "experiment_a_h1" in all_results:
            all_results["experiment_a_h1"].to_csv(output_path / "results_experiment_a_h1.csv")
        if "experiment_a_h7" in all_results:
            all_results["experiment_a_h7"].to_csv(output_path / "results_experiment_a_h7.csv")

    if "experiment_b" in all_results:
        all_results["experiment_b"].to_csv(output_path / "results_experiment_b.csv")
    else:
        if "experiment_b_h1" in all_results:
            all_results["experiment_b_h1"].to_csv(output_path / "results_experiment_b_h1.csv")
        if "experiment_b_h7" in all_results:
            all_results["experiment_b_h7"].to_csv(output_path / "results_experiment_b_h7.csv")

    if "regimes" in all_results:
        all_results["regimes"].to_csv(output_path / "results_regimes.csv")

    for horizon_key in ("experiment_a_h1", "experiment_a_h7"):
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
    processed_dir = Path("data/processed")
    predictions = load_all_predictions(processed_dir)

    # Визначаємо наявні горизонти прогнозів
    preds_h1 = {k: v for k, v in predictions.items() if k.endswith("_h1")}
    preds_h7 = {k: v for k, v in predictions.items() if k.endswith("_h7")}

    def load_test_target(horizon: int) -> pd.Series:
        """Load the correctly shifted target for evaluation to prevent data leakage."""
        target_path = processed_dir / f"target_h{horizon}.parquet"
        if target_path.exists():
            df = pd.read_parquet(target_path)
            return df.iloc[:, 0]
        logger.warning("Target file {path} not found. Falling back to unshifted test_returns.", path=target_path)
        test_path = processed_dir / "test_returns.parquet"
        df = pd.read_parquet(test_path)
        return df["brent_return"] if "brent_return" in df.columns else df.iloc[:, 0]

    config_path = Path("configs") / "wfv_config.json"
    w_train = 1000
    if config_path.exists():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            w_train = int(config_data.get("w_train", w_train))
        except Exception as exc:  # noqa: BLE001 - fallback to default
            logger.warning("Failed to read w_train from config: {error}", error=str(exc))

    train_df = pd.read_parquet(processed_dir / "train_returns.parquet")
    val_df = pd.read_parquet(processed_dir / "val_returns.parquet")
    train_series = pd.concat([train_df, val_df]).sort_index()
    if "brent_return" in train_series.columns:
        train_series = train_series["brent_return"]
    else:
        train_series = train_series.iloc[:, 0]

    y_train_last_window = train_series.tail(w_train)

    regimes = {
        "post_covid": ("2021-01-01", "2022-02-23"),
        "ukraine_war": ("2022-02-24", "2023-06-30"),
        "normalization": ("2023-07-01", "2026-03-10"),
    }
    all_results = {}

    def has_baseline(preds_dict: dict) -> bool:
        return any("random_walk" in k or "randomwalk" in k for k in preds_dict)

    if preds_h1:
        y_test_h1 = load_test_target(horizon=1)
        if has_baseline(preds_h1):
            all_results["experiment_a_h1"] = evaluate_experiment_a(preds_h1, y_test_h1, y_train_last_window)
        all_results["experiment_b_h1"] = evaluate_experiment_b(preds_h1, y_test_h1)
        regime_results_h1 = regime_analysis(preds_h1, y_test_h1, regimes)
        regime_results_h1["horizon"] = 1
        all_results["regimes_h1"] = regime_results_h1

    if preds_h7:
        y_test_h7 = load_test_target(horizon=7)
        if has_baseline(preds_h7):
            all_results["experiment_a_h7"] = evaluate_experiment_a(preds_h7, y_test_h7, y_train_last_window)
        all_results["experiment_b_h7"] = evaluate_experiment_b(preds_h7, y_test_h7)
        regime_results_h7 = regime_analysis(preds_h7, y_test_h7, regimes)
        regime_results_h7["horizon"] = 7
        all_results["regimes_h7"] = regime_results_h7

    if "regimes_h1" in all_results and "regimes_h7" in all_results:
        all_results["regimes"] = pd.concat([all_results["regimes_h1"], all_results["regimes_h7"]])
    elif "regimes_h1" in all_results:
        all_results["regimes"] = all_results["regimes_h1"]
    elif "regimes_h7" in all_results:
        all_results["regimes"] = all_results["regimes_h7"]

    generate_results_tables(all_results, output_dir=processed_dir)

    if "experiment_a_h1" in all_results and "experiment_a_h7" in all_results:
        summary_table = compare_horizons(all_results["experiment_a_h1"], all_results["experiment_a_h7"])
        print("\nExperiment A Summary (h1 vs h7)\n")
        print(summary_table.to_string())
    elif "experiment_a_h1" in all_results:
        print("\nExperiment A Summary (h1 only)\n")
        print(all_results["experiment_a_h1"].to_string())
    elif "experiment_a_h7" in all_results:
        print("\nExperiment A Summary (h7 only)\n")
        print(all_results["experiment_a_h7"].to_string())