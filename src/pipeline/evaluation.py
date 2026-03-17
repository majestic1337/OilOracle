"""Evaluation metrics — classification, regression, financial, computational.

Changes vs original:
  Change 3: flat_mask parameter — filters NaN (flat-zone) rows before any metric.
  Change 4: y_proba parameter — enables brier_score and ECE calibration metrics.
  Change 6: diebold_mariano_test() standalone function for DM significance testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ═════════════════════════════════════════════════════════════════════════════
# Change 6 · Diebold-Mariano test
# ═════════════════════════════════════════════════════════════════════════════

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    Compares model A (``e1``) against model B (``e2``) using squared-error loss.
    Variance is estimated with Newey-West HAC (lag = h − 1) for robustness to
    serial correlation induced by multi-step forecasting.

    Parameters
    ----------
    e1, e2 : np.ndarray
        1-D arrays of per-step forecast errors for models A and B.
        Must be the same length.
    h : int
        Forecast horizon; controls the Newey-West lag truncation (lag = h − 1).

    Returns
    -------
    dm_stat : float
        DM test statistic (positive → e1 has higher loss than e2).
    p_value : float
        Two-sided p-value under the standard normal approximation.
    """
    from scipy import stats as sp_stats

    d = e1 ** 2 - e2 ** 2          # loss differential
    n = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance estimate
    lag = max(h - 1, 0)
    gamma_0 = np.var(d, ddof=0)
    hac_var = gamma_0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)    # Bartlett kernel weight
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        hac_var += 2.0 * w * gamma_k

    hac_var = max(hac_var, 1e-12)  # numerical floor
    dm_stat = d_bar / np.sqrt(hac_var / n)
    p_value = float(2.0 * (1.0 - sp_stats.norm.cdf(abs(dm_stat))))
    return float(dm_stat), p_value


# ═════════════════════════════════════════════════════════════════════════════
# ECE helper
# ═════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — 10 equal-width probability bins.

    ECE = Σ_b (n_b / n) × |accuracy_b − confidence_b|

    Parameters
    ----------
    y_true  : binary {0, 1} ground-truth array.
    y_proba : predicted probabilities in [0, 1].
    n_bins  : number of equal-width bins (default 10).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi)
        if i == n_bins - 1:          # include right edge in last bin
            mask = (y_proba >= lo) & (y_proba <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b  = float(y_true[mask].mean())
        conf_b = float(y_proba[mask].mean())
        ece += abs(acc_b - conf_b) * n_b / n_total
    return ece


# ═════════════════════════════════════════════════════════════════════════════
# Financial back-tester (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class FinancialBacktester:
    """Simulates a simple long/flat directional strategy with costs."""

    transaction_cost: float = 0.001
    slippage: float = 0.0005

    def backtest(
        self,
        prices_now: np.ndarray,
        prices_future: np.ndarray,
        predicted_direction: np.ndarray,
    ) -> Dict[str, float]:
        """Return Net PnL and Max Drawdown for a directional-signal strategy."""
        realised_ret = (prices_future - prices_now) / (prices_now + 1e-9)
        strategy_ret = predicted_direction * realised_ret

        # Trade costs applied on every signal change
        trades = np.abs(np.diff(predicted_direction, prepend=0))
        costs = trades * (self.transaction_cost + self.slippage)
        net_ret = strategy_ret - costs

        cum = np.cumprod(1.0 + net_ret)
        net_pnl = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / (running_max + 1e-9)
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        return {"Net_PnL": net_pnl, "Max_Drawdown": max_dd}


# ═════════════════════════════════════════════════════════════════════════════
# Metrics calculator
# ═════════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """Computes full metric suite for a single model run."""

    def __init__(self, backtester: FinancialBacktester | None = None) -> None:
        self.backtester = backtester or FinancialBacktester()

    def compute(
        self,
        y_true_price: np.ndarray,
        y_pred_price: np.ndarray,
        current_price: np.ndarray,
        train_time: float,
        inference_time: float,
        *,
        flat_mask: Optional[np.ndarray] = None,   # Change 3: flat-zone mask
        y_proba: Optional[np.ndarray] = None,     # Change 4: calibration metrics
    ) -> Dict[str, float]:
        """Full evaluation: classification + regression + financial + compute.

        Parameters
        ----------
        y_true_price  : actual future prices.
        y_pred_price  : predicted future prices (regression) or 0/1 labels (cls).
        current_price : price at the prediction point (used for direction derivation).
        train_time    : seconds spent training.
        inference_time: average seconds per prediction step.
        flat_mask     : boolean array where True = flat-zone sample.
                        When provided, those rows are excluded before any metric.
        y_proba       : P(direction = UP) per sample.  Required for brier/ECE.
                        Pass None for statistical/regression models.
        """
        # ── Change 3: apply flat-zone exclusion ──────────────────────────
        if flat_mask is not None and flat_mask.any():
            keep = ~flat_mask
            y_true_price  = y_true_price[keep]
            y_pred_price  = y_pred_price[keep]
            current_price = current_price[keep]
            if y_proba is not None:
                y_proba = y_proba[keep]

        # Directional classification derived from price predictions
        true_dir = (y_true_price > current_price).astype(int)
        pred_dir = (y_pred_price > current_price).astype(int)

        # ── Classification ───────────────────────────────────────────────
        try:
            auc = float(roc_auc_score(true_dir, pred_dir))
        except ValueError:
            auc = 0.5

        cls_metrics = {
            "Accuracy": float(accuracy_score(true_dir, pred_dir)),
            "Precision": float(precision_score(true_dir, pred_dir, zero_division=0)),
            "Recall": float(recall_score(true_dir, pred_dir, zero_division=0)),
            "F1": float(f1_score(true_dir, pred_dir, zero_division=0)),
            "ROC_AUC": auc,
        }

        # ── Regression ───────────────────────────────────────────────────
        reg_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
            "MAE": float(mean_absolute_error(y_true_price, y_pred_price)),
            "MAPE": float(mean_absolute_percentage_error(y_true_price, y_pred_price)),
        }

        # ── Financial ────────────────────────────────────────────────────
        fin_metrics = self.backtester.backtest(current_price, y_true_price, pred_dir)

        # ── Compute ──────────────────────────────────────────────────────
        compute_metrics = {
            "Train_Time_s": round(train_time, 3),
            "Inf_Latency_s": round(inference_time, 6),
        }

        # ── Change 4: calibration metrics ────────────────────────────────
        if y_proba is not None:
            cal_metrics: Dict[str, float] = {
                "brier_score": float(brier_score_loss(true_dir, y_proba)),
                "ece": expected_calibration_error(true_dir, y_proba),
            }
        else:
            cal_metrics = {"brier_score": float("nan"), "ece": float("nan")}

        return {
            **cls_metrics,
            **reg_metrics,
            **fin_metrics,
            **compute_metrics,
            **cal_metrics,
        }
