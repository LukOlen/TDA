"""Unit tests for the market regime detection module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from backtester.regime import (
    Regime,
    RegimeIndicator,
    RegimeResult,
    _clamp,
    _compute_rsi,
    _score_momentum,
    _score_price_vs_sma200,
    _score_rsi,
    _score_sma_crossover,
    _score_volatility_regime,
    detect_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 400, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
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


def _mock_load_uptrend(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Strong uptrend data."""
    return make_ohlcv(n=400, trend=0.003, seed=42)


def _mock_load_downtrend(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Strong downtrend data."""
    return make_ohlcv(n=400, trend=-0.003, seed=42)


def _mock_load_flat(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Flat / sideways data."""
    return make_ohlcv(n=400, trend=0.0, seed=42)


def _mock_load_no_vix(ticker: str, *, start: str, end: str) -> pd.DataFrame:
    """Raise for ^VIX, return uptrend for everything else."""
    if ticker == "^VIX":
        raise ValueError("VIX unavailable")
    return make_ohlcv(n=400, trend=0.002, seed=42)


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_above_max(self):
        assert _clamp(2.0) == 1.0

    def test_below_min(self):
        assert _clamp(-3.0) == -1.0


# ---------------------------------------------------------------------------
# _compute_rsi
# ---------------------------------------------------------------------------

class TestComputeRSI:
    def test_rsi_range(self):
        data = make_ohlcv(n=300, trend=0.001)
        rsi = _compute_rsi(data["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_uptrend_rsi_above_50(self):
        data = make_ohlcv(n=300, trend=0.005)
        rsi = _compute_rsi(data["close"], 14)
        # Last RSI in a strong uptrend should be >50
        assert rsi.iloc[-1] > 50

    def test_downtrend_rsi_below_50(self):
        data = make_ohlcv(n=300, trend=-0.005)
        rsi = _compute_rsi(data["close"], 14)
        assert rsi.iloc[-1] < 50


# ---------------------------------------------------------------------------
# Individual indicator scorers
# ---------------------------------------------------------------------------

class TestPriceVsSMA200:
    def test_uptrend_positive_score(self):
        data = make_ohlcv(n=400, trend=0.003)
        ind = _score_price_vs_sma200(data["close"])
        assert ind.score > 0
        assert "above" in ind.interpretation

    def test_downtrend_negative_score(self):
        data = make_ohlcv(n=400, trend=-0.003)
        ind = _score_price_vs_sma200(data["close"])
        assert ind.score < 0
        assert "below" in ind.interpretation

    def test_score_clamped(self):
        data = make_ohlcv(n=400, trend=0.003)
        ind = _score_price_vs_sma200(data["close"])
        assert -1 <= ind.score <= 1


class TestSMACrossover:
    def test_golden_cross_positive(self):
        data = make_ohlcv(n=400, trend=0.003)
        ind = _score_sma_crossover(data["close"])
        assert ind.score > 0
        assert "golden" in ind.interpretation

    def test_death_cross_negative(self):
        data = make_ohlcv(n=400, trend=-0.003)
        ind = _score_sma_crossover(data["close"])
        assert ind.score < 0
        assert "death" in ind.interpretation


class TestMomentumIndicator:
    def test_uptrend_positive(self):
        data = make_ohlcv(n=400, trend=0.003)
        ind = _score_momentum(data["close"])
        assert ind.score > 0

    def test_downtrend_negative(self):
        data = make_ohlcv(n=400, trend=-0.003)
        ind = _score_momentum(data["close"])
        assert ind.score < 0

    def test_insufficient_data(self):
        data = make_ohlcv(n=30)
        ind = _score_momentum(data["close"])
        assert ind.score == 0.0
        assert "insufficient" in ind.interpretation


class TestRSIIndicator:
    def test_uptrend_positive(self):
        data = make_ohlcv(n=300, trend=0.005)
        ind = _score_rsi(data["close"])
        assert ind.score > 0

    def test_downtrend_negative(self):
        data = make_ohlcv(n=300, trend=-0.005)
        ind = _score_rsi(data["close"])
        assert ind.score < 0

    def test_score_clamped(self):
        data = make_ohlcv(n=300, trend=0.005)
        ind = _score_rsi(data["close"])
        assert -1 <= ind.score <= 1


class TestVolatilityRegime:
    def test_returns_indicator(self):
        data = make_ohlcv(n=300, trend=0.001)
        ind = _score_volatility_regime(data["close"])
        assert isinstance(ind, RegimeIndicator)
        assert -1 <= ind.score <= 1

    def test_insufficient_data(self):
        data = make_ohlcv(n=30)
        ind = _score_volatility_regime(data["close"])
        assert ind.score == 0.0


# ---------------------------------------------------------------------------
# detect_regime (full integration)
# ---------------------------------------------------------------------------

class TestDetectRegime:
    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_strong_uptrend_is_bullish(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert result.regime == Regime.BULLISH
        assert result.composite_score > 0

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_downtrend)
    def test_strong_downtrend_is_bearish(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert result.regime == Regime.BEARISH
        assert result.composite_score < 0

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_flat)
    def test_flat_market_is_neutral(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert result.regime == Regime.NEUTRAL

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_result_has_indicators(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert len(result.indicators) >= 5
        for ind in result.indicators:
            assert isinstance(ind, RegimeIndicator)
            assert -1 <= ind.score <= 1

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_confidence_between_zero_and_one(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert 0 <= result.confidence <= 1

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_as_of_date_populated(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert result.as_of_date != ""

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_no_vix)
    def test_vix_failure_graceful(self, _mock):
        """Regime detection should work even when VIX data is unavailable."""
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert isinstance(result, RegimeResult)
        # VIX indicator should not be present
        vix_indicators = [i for i in result.indicators if "VIX" in i.name]
        assert len(vix_indicators) == 0

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_custom_thresholds(self, _mock):
        # Very high threshold — should be neutral even in uptrend
        result = detect_regime(
            benchmark="SPY",
            start="2020-01-01",
            end="2021-07-01",
            bullish_threshold=0.99,
            bearish_threshold=-0.99,
        )
        assert result.regime == Regime.NEUTRAL

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_default_dates(self, _mock):
        result = detect_regime(benchmark="SPY")
        assert isinstance(result, RegimeResult)

    @patch("backtester.regime.load_ohlcv", side_effect=_mock_load_uptrend)
    def test_composite_score_clamped(self, _mock):
        result = detect_regime(
            benchmark="SPY", start="2020-01-01", end="2021-07-01"
        )
        assert -1 <= result.composite_score <= 1
