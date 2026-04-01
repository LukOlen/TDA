"""
OHLCV data loader with TTL-based LRU caching.

Fetches daily price data from Yahoo Finance via yfinance and normalises
column names to lowercase ``[open, high, low, close, volume]``.

The cache avoids redundant HTTP round-trips when the same ticker/date
range is requested within a 15-minute window.
"""
from __future__ import annotations

import functools
import time

import pandas as pd
import yfinance as yf


_CACHE_TTL_SECONDS = 900  # 15 minutes


@functools.lru_cache(maxsize=64)
def _fetch_cached(
    ticker: str, start: str, end: str, _ttl_bucket: int
) -> pd.DataFrame:
    """Internal cached fetch.  *_ttl_bucket* rotates every TTL window."""
    df: pd.DataFrame = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' ({start} to {end})"
        )

    # yfinance ≥0.2 may return MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {missing}")

    return df[["open", "high", "low", "close", "volume"]].copy()


def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load OHLCV data from Yahoo Finance with TTL-based LRU caching.

    Parameters
    ----------
    ticker : str
        Symbol, e.g. ``"AAPL"``.
    start : str
        Start date ``"YYYY-MM-DD"``.
    end : str
        End date ``"YYYY-MM-DD"``.

    Returns
    -------
    pd.DataFrame
        Columns ``[open, high, low, close, volume]`` with a DatetimeIndex.

    Raises
    ------
    ValueError
        If no data is returned (bad ticker or date range).
    """
    ttl_bucket = int(time.time()) // _CACHE_TTL_SECONDS
    return _fetch_cached(ticker.upper(), start, end, ttl_bucket)


def clear_cache() -> None:
    """Flush the data cache (useful in tests)."""
    _fetch_cached.cache_clear()
