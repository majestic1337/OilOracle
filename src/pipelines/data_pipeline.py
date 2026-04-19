"""Data pipeline for oil price time series."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

DB_URL = "postgresql://admin:adminpassword@localhost:5432/brentprices_data"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TABLE_MAP: dict[str, str] = {
    "brent": "brent_prices",
    "wti": "wti_prices",
    "dxy": "dxy_prices",
    "gold": "gold_prices",
}

EWM_WARMUP_PERIODS = 3 * 26  # ~95% weight convergence for span=26


def _read_price_table(engine: Engine, table_name: str, asset: str) -> pd.DataFrame:
    """Read a price table from PostgreSQL.

    Args:
        engine: SQLAlchemy engine for database connection.
        table_name: Database table name.
        asset: Asset label for column naming.

    Returns:
        DataFrame indexed by date with a single price column.
    """
    query = f"SELECT date, close FROM {table_name}"
    try:
        df = pd.read_sql(query, engine, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001 - re-raise non-table errors
        message = str(exc).lower()
        table_token = table_name.lower()
        missing_table = (
            "does not exist" in message
            or "undefined table" in message
            or ("relation" in message and table_token in message)
        )
        if missing_table:
            raise ValueError(
                f"Table '{table_name}' not found in database. "
                f"Verify DB_URL and that the table exists."
            ) from exc
        raise

    df = df.rename(columns={"close": asset}).set_index("date").sort_index()
    logger.info("Loaded {asset} rows: {rows}", asset=asset, rows=len(df))
    return df


def load_raw_data(engine: Engine) -> pd.DataFrame:
    """Load raw price data for all assets.

    Args:
        engine: SQLAlchemy engine for database connection.

    Returns:
        Wide-format DataFrame with columns [brent, wti, dxy, gold]
        and a DatetimeIndex.
    """
    frames: list[pd.DataFrame] = []
    for asset, table_name in TABLE_MAP.items():
        frames.append(_read_price_table(engine, table_name, asset))

    wide_df = pd.concat(frames, axis=1, join="outer", sort=True).sort_index()
    return wide_df

def compute_indicators(df: pd.DataFrame, warmup_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute technical indicators (RSI, MACD) for Brent.
    
    Args:
        df: Aligned DataFrame with price columns.
        warmup_df: Optional DataFrame with previous observations for warm-up.
        
    Returns:
        Original DataFrame with appended indicator columns.
    """
    if "brent" not in df.columns:
        return df

    if warmup_df is not None and not warmup_df.empty:
        out_df = pd.concat([warmup_df, df]).copy()
    else:
        out_df = df.copy()

    brent = out_df["brent"]

    # RSI (14 days)
    delta = brent.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    out_df["brent_rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = brent.ewm(span=12, adjust=False).mean()
    ema26 = brent.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    out_df["brent_macd"] = macd_line
    out_df["brent_signal"] = signal_line

    if warmup_df is not None and not warmup_df.empty:
        out_df = out_df.iloc[len(warmup_df):]
        assert len(out_df) == len(df), f"Indicator slice mismatch: expected {len(df)}, got {len(out_df)}"
        assert out_df.index.equals(df.index), "Indicator slice index mismatch"

    return out_df


def align_series(df: pd.DataFrame) -> pd.DataFrame:
    """Align series to the target Brent calendar and fill exogenous gaps.

    Args:
        df: Wide-format DataFrame with raw prices.

    Returns:
        DataFrame reindexed to Brent's non-null calendar, with exogenous
        columns forward-filled up to 3 business steps.
    """
    if "brent" not in df.columns:
        raise ValueError("align_series requires a 'brent' column")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    weekend_mask = df.index.dayofweek >= 5
    weekend_count = weekend_mask.sum()
    if weekend_count > 0:
        logger.warning("Dropped {n} weekend rows during alignment", n=weekend_count)

    weekdays_df = df[~weekend_mask].sort_index()
    brent_index = weekdays_df["brent"].dropna().index
    if brent_index.empty:
        raise ValueError("Brent index is empty after dropping NaN values")

    aligned_df = weekdays_df.reindex(brent_index)

    exogenous_columns = [col for col in ("wti", "dxy", "gold") if col in aligned_df.columns]
    for column in exogenous_columns:
        aligned_df[column] = aligned_df[column].ffill(limit=3)

    logger.info("Aligned calendar length (Brent-anchored): {n}", n=len(aligned_df))
    if exogenous_columns:
        remaining_nans = aligned_df[exogenous_columns].isna().sum().to_dict()
        logger.info("Remaining NaN counts in exogenous columns: {counts}", counts=remaining_nans)

    return aligned_df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for each asset series.

    Args:
        df: Aligned DataFrame with price columns and indicators.

    Returns:
        DataFrame of log returns with *_return suffixes and unchanged indicators.

    Note:
        WTI квітень 2020 містить від'ємні futures ціни (артефакт ринкової
        аномалії). Замінюються через ffill(limit=1). Ця дата має бути
        згадана в тезисі як data quality exception.
    """
    df = df.sort_index()
    price_cols = [c for c in ["brent", "wti", "dxy", "gold"] if c in df.columns]
    indicator_cols = [c for c in df.columns if c not in price_cols]

    # Логуємо які саме дати мають проблемні значення
    for col in price_cols:
        bad_dates = df.index[df[col] <= 0].tolist()
        if bad_dates:
            logger.warning(
                f"{col}: non-positive prices on {bad_dates} "
                f"— replacing with NaN before log-returns"
            )

    # Замінюємо non-positive на NaN для цін
    df_prices = df[price_cols].where(df[price_cols] > 0, other=np.nan)

    # Forward fill максимум 1 день (сусідня ціна як proxy)
    # КРИТИЧНО: тільки ffill(limit=1), не більше
    df_prices = df_prices.ffill(limit=1)

    df_combined = pd.concat([df_prices, df[indicator_cols]], axis=1)

    # Якщо після ffill або через розрахунок індикаторів є NaN — дропаємо рядок
    df_combined = df_combined.dropna()

    # Returns across weekends/holidays will naturally accumulate due to the shift.
    returns_prices = np.log(df_combined[price_cols] / df_combined[price_cols].shift(1))
    returns_prices = returns_prices.add_suffix("_return")

    returns_combined = pd.concat([returns_prices, df_combined[indicator_cols]], axis=1)
    returns_combined = returns_combined.iloc[1:]
    
    return returns_combined


def run_stationarity_diagnostics(returns_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Run ADF, KPSS, and Ljung-Box tests for each return series.

    Args:
        returns_df: DataFrame of log returns.

    Returns:
        Dictionary of diagnostics per series.
    """
    diagnostics: dict[str, dict[str, Any]] = {}
    for column in returns_df.columns:
        series = returns_df[column].dropna()

        adf_result = adfuller(series, autolag="AIC")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InterpolationWarning)
            kpss_result = kpss(series, regression="c", nlags="auto")
        kpss_warning = None
        for warning in caught:
            if isinstance(warning.message, InterpolationWarning):
                kpss_warning = str(warning.message)
                break

        lb_result = acorr_ljungbox(series**2, lags=[10, 20], return_df=True)
        lb_pvalues = {
            f"lag_{lag}": float(lb_result.loc[lag, "lb_pvalue"]) for lag in lb_result.index
        }

        adf_pvalue = float(adf_result[1])
        kpss_pvalue = float(kpss_result[1])
        is_stationary = adf_pvalue < 0.05 and kpss_pvalue > 0.05

        diagnostics[column] = {
            "adf_pvalue": adf_pvalue,
            "kpss_pvalue": kpss_pvalue,
            "ljungbox_arch": lb_pvalues,
            "is_stationary": is_stationary,
            "kpss_interpolation_warning": kpss_warning,
        }

    return diagnostics


def _split_by_fixed_dates(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a DataFrame using the project-standard temporal boundaries."""
    train = df.loc["2007-07-30":"2021-12-31"].copy()
    val = df.loc["2022-01-01":"2023-12-31"].copy()
    test = df.loc["2024-01-01":"2026-03-10"].copy()

    if not train.index.intersection(test.index).empty:
        raise ValueError("Train and test indices overlap; test must remain isolated.")
    
    return {"train": train, "val": val, "test": test}


def create_train_val_test_split(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split aligned prices into train/val/test partitions.

    Args:
        df: DataFrame of aligned raw prices.

    Returns:
        Dictionary with "train", "val", "test" splits.
    """
    return _split_by_fixed_dates(df)


def save_processed_data(
    split_sets: dict[str, dict[str, pd.DataFrame]],
    diagnostics: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Save processed splits and diagnostics to disk.

    Args:
        split_sets: Nested dictionary with "prices" and "returns" split maps.
        diagnostics: Diagnostics report data.
        output_dir: Target directory for saved files.

    Returns:
        None
    """
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    return_splits = split_sets.get("returns", {})
    price_splits = split_sets.get("prices", {})

    for split_name, split_df in return_splits.items():
        split_path = output_path / f"{split_name}_returns.parquet"
        split_df.to_parquet(split_path)
        logger.info("Saved returns split {split} to {path}", split=split_name, path=split_path)

    for split_name, split_df in price_splits.items():
        split_path = output_path / f"{split_name}_prices.parquet"
        split_df.to_parquet(split_path)
        logger.info("Saved prices split {split} to {path}", split=split_name, path=split_path)

    diagnostics_path = output_path / "diagnostics_report.json"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    logger.info("Saved diagnostics report to {path}", path=diagnostics_path)


if __name__ == "__main__":
    engine = create_engine(DB_URL)

    raw_df = load_raw_data(engine)
    aligned_df = align_series(raw_df)
    
    price_splits = create_train_val_test_split(aligned_df)
    
    train_prices = compute_indicators(price_splits["train"])
    val_prices = compute_indicators(price_splits["val"], warmup_df=price_splits["train"].tail(EWM_WARMUP_PERIODS))
    test_prices = compute_indicators(price_splits["test"], warmup_df=price_splits["val"].tail(EWM_WARMUP_PERIODS))
    
    train_returns = compute_log_returns(train_prices)
    val_returns = compute_log_returns(val_prices)
    test_returns = compute_log_returns(test_prices)
    
    split_sets = {
        "prices": {"train": train_prices, "val": val_prices, "test": test_prices},
        "returns": {"train": train_returns, "val": val_returns, "test": test_returns},
    }
    
    diagnostics = run_stationarity_diagnostics(train_returns)

    save_processed_data(split_sets, diagnostics, output_dir=DEFAULT_PROCESSED_DIR)

    for split_name, split_df in split_sets["returns"].items():
        summary = split_df.describe().T
        logger.info("Summary for {split}\n{summary}", split=split_name, summary=summary)
