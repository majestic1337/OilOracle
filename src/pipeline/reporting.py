"""Reporting — per-model reports, master DataFrame, Pandas Styler highlighting.

Changes vs original:
  Change 2: aggregate_dl_runs() — mean/std/median over multiple DL seeds.
  Change 6: per_step_errors storage, build_dm_matrix(), export_csv() with dm_path,
            dm_pvalue column in master table.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.pipeline.evaluation import diebold_mariano_test

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Change 2 · DL multi-run aggregation
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_dl_runs(run_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics over multiple DL seed runs into mean/std/median.

    Parameters
    ----------
    run_results : list of per-run metric dicts (one dict per seed run).

    Returns
    -------
    Flat dict with keys ``<metric>_mean``, ``<metric>_std``, ``<metric>_median``.
    Also stores the raw ``<metric>_mean`` value under the original key so that
    single-run and multi-run rows share a common set of metric column names.
    """
    if not run_results:
        return {}
    keys = run_results[0].keys()
    aggregated: Dict[str, float] = {}
    for k in keys:
        vals = np.array([r[k] for r in run_results if not np.isnan(r.get(k, np.nan))], dtype=float)
        if len(vals) == 0:
            aggregated[k] = float("nan")
            aggregated[f"{k}_mean"]   = float("nan")
            aggregated[f"{k}_std"]    = float("nan")
            aggregated[f"{k}_median"] = float("nan")
        else:
            aggregated[k]             = float(np.mean(vals))   # primary column
            aggregated[f"{k}_mean"]   = float(np.mean(vals))
            aggregated[f"{k}_std"]    = float(np.std(vals, ddof=0))
            aggregated[f"{k}_median"] = float(np.median(vals))
    return aggregated


# ═════════════════════════════════════════════════════════════════════════════
# Report builder
# ═════════════════════════════════════════════════════════════════════════════

class ReportBuilder:
    """Collects per-model results and produces styled master comparison table."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        # Change 6: stores {(model_name, horizon): np.ndarray of per-step errors}
        self._step_errors: Dict[Tuple[str, int], np.ndarray] = {}

    # ── Per-model report ─────────────────────────────────────────────────
    def add_result(
        self,
        model_name: str,
        horizon: int,
        lookback: int,
        validation_type: str,
        metrics: Dict[str, float],
        meta: Dict[str, Any],
        per_step_errors: Optional[np.ndarray] = None,   # Change 6
    ) -> None:
        """Register a single model run.

        Parameters
        ----------
        per_step_errors : per-step squared-error array (y_true − y_pred)².
            Used for the Diebold-Mariano DM matrix.  Pass ``None`` to skip.
        """
        row = {
            "Model": model_name,
            "Horizon": horizon,
            "Lookback": lookback,
            "Validation": validation_type,
            **metrics,
        }
        self._records.append(row)
        # Change 6: store errors keyed by (model, horizon)
        if per_step_errors is not None:
            self._step_errors[(model_name, horizon)] = per_step_errors
        self._print_model_report(model_name, horizon, lookback, validation_type, metrics, meta)

    def _print_model_report(
        self,
        name: str,
        h: int,
        lb: int,
        val_type: str,
        metrics: Dict[str, float],
        meta: Dict[str, Any],
    ) -> None:
        print(f"\n{'='*70}")
        print(f"  {name}  |  h={h}  |  lookback={lb}  |  {val_type}")
        print(f"{'='*70}")
        if "params" in meta:
            print(f"  Params: {meta['params']}")
        if meta.get("early_stop_epoch"):
            print(f"  Early Stop @ epoch {meta['early_stop_epoch']}")
        print(f"  Train time: {meta.get('train_time', 0):.2f}s")
        for k, v in metrics.items():
            if not np.isnan(v) if isinstance(v, float) else True:
                print(f"  {k:>20s}: {v:.6f}")

    # ── Loss curve plotting ──────────────────────────────────────────────
    @staticmethod
    def plot_loss_curves(
        model_name: str,
        loss_history: Dict[str, List[float]],
        horizon: int,
        lookback: int,
    ) -> None:
        """Plot train / val loss curves for DL models."""
        if not loss_history or not loss_history.get("train"):
            return
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(loss_history["train"], label="Train Loss")
        if loss_history.get("val"):
            ax.plot(loss_history["val"], label="Val Loss")
        ax.set_title(f"{model_name} Loss (h={horizon}, L={lookback})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ── Master DataFrame ─────────────────────────────────────────────────
    def build_master_table(self) -> pd.DataFrame:
        """Aggregate all results into one DataFrame."""
        return pd.DataFrame(self._records)

    # ── Change 6 · DM matrix ─────────────────────────────────────────────
    def build_dm_matrix(self, horizon: int) -> pd.DataFrame:
        """Build an N×N p-value matrix for all model pairs at a given horizon.

        Cell [A, B] = p-value of DM test (model A vs model B) on squared errors.
        Diagonal = NaN.  p-value < 0.05 → statistically significant difference.

        Parameters
        ----------
        horizon : forecast horizon (h) to filter stored errors.

        Returns
        -------
        pd.DataFrame with model names as both index and columns.
        """
        # Collect all models that have errors stored for this horizon
        models_h = [
            (name, errs)
            for (name, h), errs in self._step_errors.items()
            if h == horizon
        ]
        if len(models_h) < 2:
            logger.warning("DM matrix requires ≥ 2 models with errors; got %d for h=%d", len(models_h), horizon)
            return pd.DataFrame()

        names = [m[0] for m in models_h]
        errors = [m[1] for m in models_h]
        n = len(names)
        matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                try:
                    _, pv = diebold_mariano_test(errors[i], errors[j], h=horizon)
                    matrix[i, j] = pv
                except Exception as exc:
                    logger.debug("DM(%s, %s) failed: %s", names[i], names[j], exc)

        df_dm = pd.DataFrame(matrix, index=names, columns=names)
        return df_dm

    # ── Change 6 · CSV export with DM ────────────────────────────────────
    def export_csv(
        self,
        path: str,
        dm_path: Optional[str] = None,
    ) -> None:
        """Export the master results table and (optionally) DM matrices.

        Parameters
        ----------
        path    : file path for the main results CSV (all models × metrics).
        dm_path : base path for DM matrices.  If provided, one file per horizon
                  is written as ``{dm_path}_h{horizon}.csv``.
                  Rows/columns marked with p < 0.05 are noted in a separate
                  *_significant.csv companion.
        """
        df = self.build_master_table()

        # Add dm_pvalue column: min p-value across all pairs where this model is A
        dm_pvalue_col: List[float] = []
        for _, row in df.iterrows():
            model = str(row["Model"])
            h = int(row["Horizon"])
            key_errors = self._step_errors.get((model, h))
            if key_errors is None:
                dm_pvalue_col.append(float("nan"))
                continue
            pvals = []
            for (other_name, other_h), other_errs in self._step_errors.items():
                if other_name == model or other_h != h:
                    continue
                try:
                    _, pv = diebold_mariano_test(key_errors, other_errs, h=h)
                    pvals.append(pv)
                except Exception:
                    pass
            dm_pvalue_col.append(float(np.min(pvals)) if pvals else float("nan"))

        df["dm_pvalue"] = dm_pvalue_col
        df.to_csv(path, index=False)
        logger.info("Results saved → %s", path)

        # DM matrices per horizon
        if dm_path is not None:
            horizons = df["Horizon"].dropna().unique().tolist()
            for h in horizons:
                dm_df = self.build_dm_matrix(int(h))
                if dm_df.empty:
                    continue
                out = f"{dm_path}_h{int(h)}.csv"
                dm_df.to_csv(out)
                logger.info("DM matrix h=%d saved → %s", int(h), out)
                # Flag significant pairs
                sig = dm_df < 0.05
                sig.to_csv(f"{dm_path}_h{int(h)}_significant.csv")

    # ── Styled output ────────────────────────────────────────────────────
    @staticmethod
    def styled_master(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
        """Apply highlight_max / highlight_min per metric column."""
        metric_cols = [
            c for c in df.columns
            if c not in ("Model", "Horizon", "Lookback", "Validation")
        ]
        # Higher-is-better
        higher_better = {"Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "Net_PnL"}
        # Lower-is-better — now includes brier_score, ece, dm_pvalue (Change 4 & 6)
        lower_better = {
            "RMSE", "MAE", "MAPE", "Max_Drawdown",
            "Train_Time_s", "Inf_Latency_s",
            "brier_score", "ece", "dm_pvalue",
        }

        styler = df.style
        for col in metric_cols:
            if col in higher_better:
                styler = styler.highlight_max(subset=[col], color="#2ecc71", axis=0)
            elif col in lower_better:
                styler = styler.highlight_min(subset=[col], color="#3498db", axis=0)
        styler = styler.format(precision=4, na_rep="—")
        return styler
