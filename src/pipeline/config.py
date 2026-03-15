"""Pipeline configuration — all constants and type-safe dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the entire forecasting pipeline."""

    # ── Temporal Boundaries ──────────────────────────────────────────────
    data_start: str = "2007-07-30"
    data_end: str = "2026-03-10"
    dev_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    ticker: str = "BZ=F"

    # ── Forecast Parameters ──────────────────────────────────────────────
    horizons: List[int] = field(default_factory=lambda: [1, 7])
    lookback_candidates: List[int] = field(
        default_factory=lambda: [5, 10, 20, 30, 60, 90]
    )

    # ── Model Zoo ────────────────────────────────────────────────────────
    statistical_models: List[str] = field(
        default_factory=lambda: ["ARIMA", "HoltWinters"]
    )
    ml_models: List[str] = field(
        default_factory=lambda: ["XGBoost", "LightGBM"]
    )
    ml_cls_models: List[str] = field(
        default_factory=lambda: ["XGBoost_cls", "LightGBM_cls"]
    )
    dl_models: List[str] = field(
        default_factory=lambda: ["LSTM", "TCN", "NBEATS", "TFT"]
    )

    # ── Validation ───────────────────────────────────────────────────────
    holdout_train_ratio: float = 0.8
    cv_n_splits: int = 3
    optuna_n_trials: int = 15
    walk_forward_refit_every: int = 20  # refit every N steps in OOS

    # ── Financial Simulation ─────────────────────────────────────────────
    transaction_cost_bps: float = 10.0  # 10 basis points
    slippage_bps: float = 5.0          # 5 basis points

    # ── DL Training ──────────────────────────────────────────────────────
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 32

    # ── Change 1 · Global Seed ───────────────────────────────────────────
    GLOBAL_SEED: int = 42

    # ── Change 2 · DL Variance (multiple runs) ───────────────────────────
    DL_N_RUNS: int = 3                              # runs per DL model
    DL_RUN_SEEDS: List[int] = field(
        default_factory=lambda: [42, 123, 7]        # len must == DL_N_RUNS
    )

    # ── Change 3 · Target-Variable Thresholding (ε) ──────────────────────
    epsilon: float = -1.0
    # Sentinel values:
    #   -1.0  → compute ATR-based ε dynamically per fold
    #    0.0  → no exclusion (backward-compatible; restores original)
    #   >0.0  → use this fixed ε value for all folds
    atr_window: int = 14
    atr_epsilon_multiplier: float = 0.5

    # ── Change 4 · Probability Calibration ───────────────────────────────
    use_calibration: bool = True
    # True  → wrap XGBoost_cls / LightGBM_cls in CalibratedClassifierCV
    # False → DL sigmoid output used as-is (already a probability)

    # ── Change 5 · Feature Engineering Ablation ──────────────────────────
    USE_TECHNICAL_FEATURES: bool = True
    # True  → include RSI-14, MACD, Bollinger-width, OBV
    # False → baseline: raw OHLCV lags + SMA/STD/ATR/vol only

    @property
    def all_models(self) -> List[str]:
        return (
            self.statistical_models
            + self.ml_models
            + self.ml_cls_models
            + self.dl_models
        )

    @property
    def transaction_cost(self) -> float:
        return self.transaction_cost_bps / 10_000

    @property
    def slippage(self) -> float:
        return self.slippage_bps / 10_000
