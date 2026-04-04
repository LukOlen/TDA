"""Unit tests for the stock screener module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from backtester.screener import (
    DEFAULT_UNIVERSE,
    ScreenerResult,
    _momentum,
    _score_single_ticker,
    screen_universe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 300, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(trend, 0.01, n)))
    high = close * (1 + abs(rng.normal(0, 0.003, n)))
    low = close * (1 - abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _mock_load(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Return synthetic data with different trends per ticker."""
    seed_map = {"SPY": 10, "AAPL": 20, "MSFT": 30, "GOOGL": 40}
    trend_map = {"SPY": 0.0005, "AAPL": 0.002, "MSFT": 0.001, "GOOGL": -0.001}
    seed = seed_map.get(ticker, hash(ticker) % 1000)
    trend = trend_map.get(ticker, 0.0005)
    return make_ohlcv(n=300, trend=trend, seed=seed)


def _mock_load_failing(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Raise for certain tickers to test error collection."""
    if ticker == "BAD":
        raise ValueError("no data for BAD")
    return _mock_load(ticker, start=start, end=end)


# ---------------------------------------------------------------------------
# _momentum helper
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_positive_trend(self):
        data = make_ohlcv(n=300, trend=0.002)
        result = _momentum(data["close"], 63)
        assert result is not None
        assert result > 0

    def test_insufficient_data_returns_none(self):
        data = make_ohlcv(n=10)
        assert _momentum(data["close"], 63) is None

    def test_flat_prices_near_zero(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="B")
        close = pd.Series(100.0, index=idx)
        result = _momentum(close, 21)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _score_single_ticker
# ---------------------------------------------------------------------------

class TestScoreSingleTicker:
    def test_returns_all_expected_keys(self):
        data = make_ohlcv(n=300, trend=0.001, seed=42)
        bench = make_ohlcv(n=300, trend=0.0005, seed=10)
        result = _score_single_ticker(data, bench)
        expected_keys = {
            "relative_strength", "momentum_1m", "momentum_3m",
            "momentum_6m", "momentum_12m", "sharpe", "volatility",
            "max_drawdown",
        }
        assert set(result.keys()) == expected_keys

    def test_positive_trend_positive_momentum(self):
        data = make_ohlcv(n=300, trend=0.003, seed=42)
        bench = make_ohlcv(n=300, trend=0.0005, seed=10)
        result = _score_single_ticker(data, bench)
        assert result["momentum_3m"] > 0
        assert result["relative_strength"] > 0

    def test_negative_trend_negative_momentum(self):
        data = make_ohlcv(n=300, trend=-0.003, seed=42)
        bench = make_ohlcv(n=300, trend=0.001, seed=10)
        result = _score_single_ticker(data, bench)
        assert result["momentum_3m"] < 0
        assert result["relative_strength"] < 0

    def test_max_drawdown_is_negative(self):
        data = make_ohlcv(n=300, trend=0.001, seed=42)
        bench = make_ohlcv(n=300, trend=0.0005, seed=10)
        result = _score_single_ticker(data, bench)
        assert result["max_drawdown"] <= 0


# ---------------------------------------------------------------------------
# screen_universe
# ---------------------------------------------------------------------------

class TestScreenUniverse:
    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_sorted_by_composite_score(self, _mock):
        results, errors = screen_universe(
            tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
        )
        assert len(results) == 3
        scores = [r.composite_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_rank_starts_at_one(self, _mock):
        results, _ = screen_universe(
            tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
        )
        assert results[0].rank == 1
        assert results[-1].rank == len(results)

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_top_n_limits_results(self, _mock):
        results, _ = screen_universe(
            tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
            top_n=2,
        )
        assert len(results) == 2

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load_failing)
    def test_failed_tickers_in_errors(self, _mock):
        results, errors = screen_universe(
            tickers=["AAPL", "BAD", "MSFT"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
        )
        assert len(results) == 2
        assert len(errors) == 1
        assert errors[0]["ticker"] == "BAD"

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_single_ticker(self, _mock):
        results, _ = screen_universe(
            tickers=["AAPL"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
        )
        assert len(results) == 1
        assert results[0].rank == 1

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_composite_score_between_zero_and_one(self, _mock):
        results, _ = screen_universe(
            tickers=["AAPL", "MSFT", "GOOGL"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
        )
        for r in results:
            assert 0.0 <= r.composite_score <= 1.0

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_result_is_screener_result_type(self, _mock):
        results, _ = screen_universe(
            tickers=["AAPL", "MSFT"],
            benchmark="SPY",
            start="2020-01-01",
            end="2021-03-01",
        )
        assert all(isinstance(r, ScreenerResult) for r in results)

    def test_all_tickers_fail_returns_empty(self):
        def _all_fail(ticker, *, start, end):
            raise ValueError("fail")

        with patch("backtester.screener.load_ohlcv", side_effect=_all_fail):
            # benchmark also fails — should propagate
            with pytest.raises(ValueError):
                screen_universe(
                    tickers=["BAD1", "BAD2"],
                    benchmark="BADBENCH",
                    start="2020-01-01",
                    end="2021-03-01",
                )


class TestScreenerEdgeCases:
    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_default_dates_used_when_none(self, _mock):
        results, _ = screen_universe(
            tickers=["AAPL", "MSFT"],
            benchmark="SPY",
        )
        assert len(results) == 2

    @patch("backtester.screener.load_ohlcv", side_effect=_mock_load)
    def test_default_universe_used_when_none(self, _mock):
        results, _ = screen_universe(benchmark="SPY", start="2020-01-01", end="2021-03-01")
        # Should attempt to load all DEFAULT_UNIVERSE tickers
        assert len(results) + len([]) <= len(DEFAULT_UNIVERSE)

    def test_short_data_no_12m_momentum(self):
        """With <252 bars, momentum_12m should be None."""
        def _short_load(ticker, *, start, end):
            return make_ohlcv(n=200, trend=0.001, seed=hash(ticker) % 1000)

        with patch("backtester.screener.load_ohlcv", side_effect=_short_load):
            results, _ = screen_universe(
                tickers=["AAPL", "MSFT"],
                benchmark="SPY",
                start="2020-01-01",
                end="2021-03-01",
            )
            for r in results:
                assert r.momentum_12m is None
