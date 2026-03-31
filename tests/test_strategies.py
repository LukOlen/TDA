"""Unit tests for the built-in trading strategies."""
import numpy as np
import pandas as pd
import pytest

from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import BollingerMeanReversion
from backtester.strategies.momentum import BreakoutMomentum


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


def assert_valid_signals(signals: pd.Series, data: pd.DataFrame):
    """Check that signals are valid integers aligned to the data index."""
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(data)
    assert signals.index.equals(data.index)
    assert set(signals.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# SMA Crossover
# ---------------------------------------------------------------------------

class TestSMACrossover:
    def test_default_params(self):
        data = make_ohlcv()
        strategy = SMACrossover()
        signals = strategy.generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_params(self):
        data = make_ohlcv()
        signals = SMACrossover(fast_period=10, slow_period=30).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            SMACrossover(fast_period=50, slow_period=20)

    def test_long_only_signals(self):
        data = make_ohlcv()
        signals = SMACrossover().generate_signals(data)
        # SMA Crossover is long-only — no short signals
        assert (signals >= 0).all()

    def test_zero_before_slow_warmup(self):
        data = make_ohlcv(60)
        signals = SMACrossover(fast_period=5, slow_period=50).generate_signals(data)
        # First 49 bars should be 0 (not enough data for slow SMA)
        assert (signals.iloc[:49] == 0).all()

    def test_repr(self):
        assert "SMACrossover" in repr(SMACrossover(20, 50))


# ---------------------------------------------------------------------------
# Bollinger Mean Reversion
# ---------------------------------------------------------------------------

class TestBollingerMeanReversion:
    def test_default_params(self):
        data = make_ohlcv()
        signals = BollingerMeanReversion().generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_params(self):
        data = make_ohlcv()
        signals = BollingerMeanReversion(period=10, num_std=1.5).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            BollingerMeanReversion(period=1)

    def test_invalid_num_std(self):
        with pytest.raises(ValueError):
            BollingerMeanReversion(num_std=0)

    def test_long_only_signals(self):
        data = make_ohlcv()
        signals = BollingerMeanReversion().generate_signals(data)
        assert (signals >= 0).all()

    def test_repr(self):
        assert "BollingerMeanReversion" in repr(BollingerMeanReversion())


# ---------------------------------------------------------------------------
# Breakout Momentum
# ---------------------------------------------------------------------------

class TestBreakoutMomentum:
    def test_default_params(self):
        data = make_ohlcv()
        signals = BreakoutMomentum().generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_lookback(self):
        data = make_ohlcv()
        signals = BreakoutMomentum(lookback=10).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_invalid_lookback(self):
        with pytest.raises(ValueError):
            BreakoutMomentum(lookback=1)

    def test_long_only_signals(self):
        data = make_ohlcv()
        signals = BreakoutMomentum().generate_signals(data)
        assert (signals >= 0).all()

    def test_trending_up_generates_longs(self):
        # Strongly trending up data should generate long signals
        data = make_ohlcv(200, trend=0.005)
        signals = BreakoutMomentum(lookback=5).generate_signals(data)
        # Expect at least some long signals in a strong uptrend
        assert (signals == 1).any()

    def test_repr(self):
        assert "BreakoutMomentum" in repr(BreakoutMomentum())
