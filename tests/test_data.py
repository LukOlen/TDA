"""Unit tests for the data loader module."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from backtester.data import load_ohlcv, clear_cache


def _make_yf_dataframe(n: int = 100) -> pd.DataFrame:
    """Synthetic DataFrame mimicking yfinance output."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 150.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close * (1 + abs(rng.normal(0, 0.005, n))),
            "Low": close * (1 - abs(rng.normal(0, 0.005, n))),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=idx,
    )


def _make_yf_multiindex(n: int = 100) -> pd.DataFrame:
    """Synthetic DataFrame with MultiIndex columns (yfinance ≥0.2 quirk)."""
    df = _make_yf_dataframe(n)
    df.columns = pd.MultiIndex.from_tuples(
        [(c, "AAPL") for c in df.columns]
    )
    return df


@pytest.fixture(autouse=True)
def _clear_data_cache():
    """Ensure each test starts with a clean cache."""
    clear_cache()
    yield
    clear_cache()


class TestLoadOHLCV:
    @patch("backtester.data.yf.download")
    def test_returns_dataframe_with_correct_columns(self, mock_dl):
        mock_dl.return_value = _make_yf_dataframe()
        df = load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    @patch("backtester.data.yf.download")
    def test_empty_data_raises_value_error(self, mock_dl):
        mock_dl.return_value = pd.DataFrame()
        with pytest.raises(ValueError, match="No data returned"):
            load_ohlcv("FAKE", "2023-01-01", "2023-06-01")

    @patch("backtester.data.yf.download")
    def test_caching_avoids_duplicate_fetch(self, mock_dl):
        mock_dl.return_value = _make_yf_dataframe()
        load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert mock_dl.call_count == 1

    @patch("backtester.data.yf.download")
    def test_different_tickers_are_separate_cache_entries(self, mock_dl):
        mock_dl.return_value = _make_yf_dataframe()
        load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        load_ohlcv("MSFT", "2023-01-01", "2023-06-01")
        assert mock_dl.call_count == 2

    @patch("backtester.data.yf.download")
    def test_multiindex_columns_normalised(self, mock_dl):
        mock_dl.return_value = _make_yf_multiindex()
        df = load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    @patch("backtester.data.yf.download")
    def test_ticker_uppercased(self, mock_dl):
        mock_dl.return_value = _make_yf_dataframe()
        load_ohlcv("aapl", "2023-01-01", "2023-06-01")
        args = mock_dl.call_args
        assert args[0][0] == "AAPL"

    @patch("backtester.data.yf.download")
    def test_clear_cache_works(self, mock_dl):
        mock_dl.return_value = _make_yf_dataframe()
        load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        clear_cache()
        load_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert mock_dl.call_count == 2
