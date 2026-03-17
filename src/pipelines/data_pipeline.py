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

TABLE_MAP: dict[str, str] = {
    "brent": "brent_prices",
    "wti": "wti_prices",
    "dxy": "dxy_prices",
    "gold": "gold_prices",
}


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

    wide_df = pd.concat(frames, axis=1, join="outer").sort_index()
    return wide_df


def align_series(df: pd.DataFrame) -> pd.DataFrame:
    """Align asset series using business days and intersection filtering.

    Args:
        df: Wide-format DataFrame with raw prices.

    Returns:
        Aligned DataFrame restricted to weekdays with no missing values.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    weekdays_df = df[df.index.dayofweek < 5]
    aligned_df = weekdays_df.dropna(how="any")

    dropped_dates = df.index.difference(aligned_df.index)
    dropped_list = [date.strftime("%Y-%m-%d") for date in dropped_dates]
    logger.info("Dropped {count} dates during alignment", count=len(dropped_list))
    logger.info("Dropped dates: {dates}", dates=dropped_list)
    if not dropped_dates.empty:
        missing_by_asset = df.loc[dropped_dates].isna().sum().to_dict()
        logger.info("Missing counts by asset for dropped dates: {counts}", counts=missing_by_asset)

    return aligned_df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for each asset series.

    Args:
        df: Aligned DataFrame with price columns.

    Returns:
        DataFrame of log returns with *_return suffixes.
    """
    df = df.sort_index()
    non_positive = (df <= 0).sum()
    total_non_positive = int(non_positive.sum())
    if total_non_positive > 0:
        logger.warning(
            "Detected non-positive prices before log returns: {counts}",
            counts=non_positive.to_dict(),
        )
        df = df.mask(df <= 0, np.nan).dropna()
    # Returns across weekends/holidays will naturally accumulate due to the shift.
    returns = np.log(df / df.shift(1))
    returns = returns.iloc[1:]
    returns = returns.add_suffix("_return")
    return returns


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


def create_train_val_test_split(returns_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the returns data into train, validation, and test sets.

    Args:
        returns_df: DataFrame of log returns.

    Returns:
        Dictionary with keys train, val, test.
    """
    train = returns_df.loc["2007-07-30":"2021-12-31"].copy()
    val = returns_df.loc["2022-01-01":"2023-12-31"].copy()
    test = returns_df.loc["2024-01-01":"2026-03-10"].copy()

    assert (
        train.index.intersection(test.index).empty
    ), "Train and test indices overlap; test must remain isolated."

    return {"train": train, "val": val, "test": test}


def save_processed_data(
    splits: dict[str, pd.DataFrame],
    diagnostics: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """Save processed splits and diagnostics to disk.

    Args:
        splits: Dictionary of train/val/test DataFrames.
        diagnostics: Diagnostics report data.
        output_dir: Target directory for saved files.

    Returns:
        None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in splits.items():
        split_path = output_path / f"{split_name}_returns.parquet"
        split_df.to_parquet(split_path)
        logger.info("Saved {split} to {path}", split=split_name, path=split_path)

    diagnostics_path = output_path / "diagnostics_report.json"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    logger.info("Saved diagnostics report to {path}", path=diagnostics_path)


if __name__ == "__main__":
    engine = create_engine(DB_URL)

    raw_df = load_raw_data(engine)
    aligned_df = align_series(raw_df)
    returns_df = compute_log_returns(aligned_df)
    diagnostics = run_stationarity_diagnostics(returns_df)
    splits = create_train_val_test_split(returns_df)

    save_processed_data(splits, diagnostics, output_dir="data/processed")

    for split_name, split_df in splits.items():
        summary = split_df.describe().T
        logger.info("Summary for {split}\n{summary}", split=split_name, summary=summary)
