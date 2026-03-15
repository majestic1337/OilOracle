"""Validation engines — hold-out, expanding-window CV with Optuna, walk-forward OOS."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna

from src.pipeline.config import PipelineConfig
from src.pipeline.evaluation import MetricsCalculator
from src.pipeline.models import BaseModel, ModelFactory

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Global Hold-Out Validator
# ═════════════════════════════════════════════════════════════════════════════
class HoldoutValidator:
    """80/20 chronological split on the Development Set (no shuffle)."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.calc = MetricsCalculator()

    def evaluate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y_reg: np.ndarray,
        current_prices: np.ndarray,
    ) -> Dict[str, Any]:
        n = len(X)
        split = int(n * self.cfg.holdout_train_ratio)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y_reg[:split], y_reg[split:]
        p_val = current_prices[split:]

        meta = model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
        t0 = time.time()
        preds = model.predict(X_val)
        inf_time = (time.time() - t0) / max(len(X_val), 1)

        metrics = self.calc.compute(y_val, preds, p_val, meta["train_time"], inf_time)
        return {"metrics": metrics, "meta": meta, "preds": preds, "y_true": y_val}


# ═════════════════════════════════════════════════════════════════════════════
# 2. Expanding Window Cross-Validation with Optuna
# ═════════════════════════════════════════════════════════════════════════════
class ExpandingWindowCV:
    """Time-series expanding-window CV strictly within the Dev set.

    Uses Optuna to search over model hyperparameters.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def _get_search_space(self, model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        if model_name in ("XGBoost", "LightGBM"):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            }
        if model_name in ("LSTM", "TCN", "NBEATS", "TFT"):
            return {
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            }
        return {}

    def optimize(
        self,
        model_name: str,
        X: np.ndarray,
        y_reg: np.ndarray,
        lookback: int,
        horizon: int,
        n_features: int,
    ) -> Dict[str, Any]:
        """Run Optuna study and return best params."""
        # Statistical models have no tunable hyperparams
        if model_name in ("ARIMA", "HoltWinters"):
            return {}

        n = len(X)
        fold_boundaries = [
            int(n * 0.50),
            int(n * 0.65),
            int(n * 0.80),
        ]

        def objective(trial: optuna.Trial) -> float:
            params = self._get_search_space(model_name, trial)
            fold_errors: List[float] = []

            for boundary in fold_boundaries:
                X_tr, X_v = X[:boundary], X[boundary: boundary + max(horizon * 10, 50)]
                y_tr, y_v = y_reg[:boundary], y_reg[boundary: boundary + max(horizon * 10, 50)]
                if len(X_v) < 5:
                    continue

                model = ModelFactory.create(
                    model_name, lookback, horizon, self.cfg, n_features, **params
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_v)
                rmse = float(np.sqrt(np.mean((y_v - preds) ** 2)))
                fold_errors.append(rmse)

            return float(np.mean(fold_errors)) if fold_errors else 1e9

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.cfg.optuna_n_trials, show_progress_bar=False)
        logger.info(
            "%s Optuna best RMSE=%.4f  params=%s",
            model_name,
            study.best_value,
            study.best_params,
        )
        return study.best_params


# ═════════════════════════════════════════════════════════════════════════════
# 3. Walk-Forward (Rolling Window) OOS Tester
# ═════════════════════════════════════════════════════════════════════════════
class WalkForwardTester:
    """Walk-forward test on OOS data (2022–2026).

    Steps forward one observation at a time, refits every N steps,
    and records predictions aligned to actual dates.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.calc = MetricsCalculator()

    def run(
        self,
        model_name: str,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        current_prices_test: np.ndarray,
        lookback: int,
        horizon: int,
        n_features: int,
        best_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute walk-forward and return aggregated metrics + per-step preds."""
        all_preds: List[float] = []
        all_true: List[float] = []
        all_prices: List[float] = []
        total_train_time = 0.0
        total_inf_time = 0.0
        refit_count = 0

        # Growing history starts with full dev set
        X_history = X_dev.copy()
        y_history = y_dev.copy()

        model = ModelFactory.create(
            model_name, lookback, horizon, self.cfg, n_features, **best_params
        )
        meta = model.fit(X_history, y_history)
        total_train_time += meta["train_time"]

        n_test = len(X_test)
        for i in range(n_test):
            # Predict
            t0 = time.time()
            pred = model.predict(X_test[i : i + 1])
            total_inf_time += time.time() - t0

            all_preds.append(float(pred[0]))
            all_true.append(float(y_test[i]))
            all_prices.append(float(current_prices_test[i]))

            # Expand history with actual
            X_history = np.vstack([X_history, X_test[i : i + 1]])
            y_history = np.append(y_history, y_test[i])

            # Periodic refit
            if (i + 1) % self.cfg.walk_forward_refit_every == 0 and i < n_test - 1:
                model = ModelFactory.create(
                    model_name, lookback, horizon, self.cfg, n_features, **best_params
                )
                meta_refit = model.fit(X_history, y_history)
                total_train_time += meta_refit["train_time"]
                refit_count += 1

        preds_arr = np.array(all_preds)
        true_arr = np.array(all_true)
        prices_arr = np.array(all_prices)
        avg_inf = total_inf_time / max(n_test, 1)

        metrics = self.calc.compute(true_arr, preds_arr, prices_arr, total_train_time, avg_inf)
        logger.info(
            "WF %s h=%d L=%d → RMSE=%.4f Acc=%.4f PnL=%.4f  (%d refits)",
            model_name, horizon, lookback, metrics["RMSE"], metrics["Accuracy"],
            metrics["Net_PnL"], refit_count,
        )
        return {
            "metrics": metrics,
            "meta": {**meta, "refit_count": refit_count, "total_train_time": total_train_time},
            "preds": preds_arr,
            "y_true": true_arr,
        }
