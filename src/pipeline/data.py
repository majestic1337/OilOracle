"""Data acquisition, cleaning, feature engineering, and target formulation.

All feature engineering is strictly causal — no future data ever leaks
into the feature or target vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from src.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def compute_epsilon(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 14,
    multiplier: float = 0.5,
) -> float:
    """Compute a data-driven flat-zone threshold.

    Returns ATR(window).mean() × multiplier computed on the provided series.
    Call on the training window of each Walk-Forward fold to avoid look-ahead.

    Parameters
    ----------
    close, high, low : pd.Series
        Aligned price series (training window only).
    window : int
        ATR look-back period (default 14).
    multiplier : float
        Scaling factor applied to the mean ATR (default 0.5).

    Returns
    -------
    float
        ε threshold in price units.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return float(atr.mean()) * multiplier


# ═════════════════════════════════════════════════════════════════════════════
# Main pipeline class
# ═════════════════════════════════════════════════════════════════════════════

class BrentDataPipeline:
    """Downloads Brent crude prices, engineers features, formulates targets."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._raw: pd.DataFrame = pd.DataFrame()
        self._featured: pd.DataFrame = pd.DataFrame()

    # ── 1. Acquisition ───────────────────────────────────────────────────
    def fetch(self) -> pd.DataFrame:
        """Download Brent daily OHLCV from Yahoo Finance."""
        logger.info(
            "Downloading %s [%s → %s]",
            self.cfg.ticker, self.cfg.data_start, self.cfg.data_end,
        )
        df = yf.download(
            self.cfg.ticker,
            start=self.cfg.data_start,
            end=self.cfg.data_end,
            progress=False,
            auto_adjust=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        self._raw = df
        logger.info(
            "Fetched %d rows  [%s → %s]",
            len(df), df.index[0].date(), df.index[-1].date(),
        )
        return df

    # ── 2. Causal Feature Engineering ────────────────────────────────────
    def engineer_features(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build lag / rolling / technical features.  ALL strictly backward-looking."""
        if df is None:
            df = self._raw.copy()
        else:
            df = df.copy()

        close = df["Close"]

        # ── Always-on features ────────────────────────────────────────────

        # Returns
        df["ret_1d"] = close.pct_change(1)
        df["ret_5d"] = close.pct_change(5)
        df["ret_20d"] = close.pct_change(20)

        # Log-returns
        df["log_ret_1d"] = np.log(close / close.shift(1))

        # Rolling statistics (window sizes chosen to stay ≤ min lookback)
        for w in [5, 10, 20, 60]:
            df[f"sma_{w}"] = close.rolling(w, min_periods=w).mean()
            df[f"std_{w}"] = close.rolling(w, min_periods=w).std()
            df[f"vol_{w}"] = df["ret_1d"].rolling(w, min_periods=w).std()

        # Average True Range (used for ε computation — always computed)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - close.shift(1)).abs(),
                (df["Low"] - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr_14"] = tr.rolling(14, min_periods=14).mean()

        # Lag features of Close price (strictly past values)
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f"close_lag_{lag}"] = close.shift(lag)

        # ── Change 5: Technical indicators gated by USE_TECHNICAL_FEATURES ─
        if self.cfg.USE_TECHNICAL_FEATURES:
            # RSI(14)
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
            rs = gain / (loss + 1e-9)
            df["rsi_14"] = 100 - (100 / (1 + rs))

            # MACD (12, 26, 9)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df["macd"] = ema12 - ema26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

            # Bollinger Bands — %B and width
            mid = close.rolling(20, min_periods=20).mean()
            std = close.rolling(20, min_periods=20).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            df["boll_pctb"] = (close - lower) / (4 * std + 1e-9)
            df["boll_width"] = (upper - lower) / (mid + 1e-9)  # Change 5: new

            # OBV (On-Balance Volume)                            # Change 5: new
            direction = np.sign(close.diff()).fillna(0)
            df["obv"] = (df["Volume"] * direction).cumsum()

        df.dropna(inplace=True)
        self._featured = df
        logger.info("Features engineered → %d rows × %d cols", *df.shape)
        return df

    # ── 3. Target Formulation ────────────────────────────────────────────
    def build_targets(
        self,
        df: pd.DataFrame,
        horizon: int,
        epsilon: float = 0.0,
    ) -> pd.DataFrame:
        """Create classification (direction) and regression (price) targets.

        Targets use FUTURE prices — they are **only used as labels** and
        never appear in the feature matrix X.

        Parameters
        ----------
        df : pd.DataFrame
            Feature dataframe (must have ``Close`` column).
        horizon : int
            Forecast horizon in steps.
        epsilon : float
            Flat-zone half-width (Change 3).
            - ``epsilon == 0``: original binary label, no flat zone.
            - ``epsilon > 0``: rows where |P(t+h) − P(t)| ≤ ε are set to NaN
              and must be excluded from training/evaluation downstream.

        Notes
        -----
        Flat-zone NaN rows are **not** dropped here so the index alignment
        between ``X`` and ``y`` is preserved for Walk-Forward slicing.
        Downstream (evaluation.py) applies the flat mask.
        """
        df = df.copy()
        target_price_col = f"target_price_h{horizon}"
        target_dir_col = f"target_dir_h{horizon}"

        df[target_price_col] = df["Close"].shift(-horizon)

        if epsilon == 0.0:
            # Original behavior — binary directional label, no flat zone
            df[target_dir_col] = (df[target_price_col] > df["Close"]).astype(int)
        else:
            # Change 3: three-way label with flat zone
            diff = df[target_price_col] - df["Close"]
            y_cls = pd.Series(np.nan, index=df.index, dtype=float)
            y_cls[diff > epsilon] = 1.0
            y_cls[diff < -epsilon] = 0.0
            # Rows where |diff| <= epsilon stay NaN (flat zone)
            df[target_dir_col] = y_cls

            # Logging (only when flat zone is active)
            n_flat = y_cls.isna().sum()
            n_total = len(y_cls)
            pct = 100.0 * n_flat / max(n_total, 1)
            logger.info(
                "Flat zone excluded: %d rows (%.1f%%)  [h=%d, ε=%.4f]",
                n_flat, pct, horizon, epsilon,
            )

        # Drop rows where the REGRESSION target is missing (end-of-series)
        df.dropna(subset=[target_price_col], inplace=True)
        return df

    # ── 4. Temporal Split ────────────────────────────────────────────────
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Strict chronological dev / out-of-sample test split."""
        dev = df[df.index <= self.cfg.dev_end].copy()
        test = df[df.index >= self.cfg.test_start].copy()
        logger.info(
            "Split → Dev %d rows [→ %s] | Test %d rows [%s →]",
            len(dev), self.cfg.dev_end, len(test), self.cfg.test_start,
        )
        return dev, test

    # ── 5. Feature / Target Separation ───────────────────────────────────
    @staticmethod
    def xy_split(
        df: pd.DataFrame, horizon: int
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return (X_features, y_regression, y_classification).

        Drops all target columns from X to prevent leakage.
        The classification target may contain NaN (flat zone) — callers
        must apply the flat mask before passing y_cls to models/metrics.
        """
        target_cols = [c for c in df.columns if c.startswith("target_")]
        y_reg = df[f"target_price_h{horizon}"]
        y_cls = df[f"target_dir_h{horizon}"]
        X = df.drop(columns=target_cols)
        return X, y_reg, y_cls
