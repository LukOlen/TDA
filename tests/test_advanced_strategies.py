"""Unit tests for the four advanced trading strategies."""
import numpy as np
import pandas as pd
import pytest

from backtester.strategies.macd_momentum import MACDMomentum
from backtester.strategies.rsi_trend_filter import RSITrendFilter
from backtester.strategies.tsmom import TSMOM
from backtester.strategies.keltner_breakout import KeltnerBreakout


# ---------------------------------------------------------------------------
# Helpers shared with test_strategies.py (duplicated to keep tests isolated)
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 400, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(trend, 0.01, n)))
    high = close * (1 + abs(rng.normal(0, 0.004, n)))
    low = close * (1 - abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def make_downtrend(n: int = 400) -> pd.DataFrame:
    """Strongly trending downward data to elicit short signals."""
    return make_ohlcv(n=n, trend=-0.003, seed=99)


def assert_valid_signals(signals: pd.Series, data: pd.DataFrame) -> None:
    """Confirm signals are integers in {-1, 0, 1} aligned to the data index."""
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(data)
    assert signals.index.equals(data.index)
    assert set(signals.unique()).issubset({-1, 0, 1}), (
        f"Unexpected signal values: {set(signals.unique()) - {-1, 0, 1}}"
    )


# ===========================================================================
# MACD Momentum
# ===========================================================================

class TestMACDMomentum:

    def test_default_params(self):
        data = make_ohlcv()
        signals = MACDMomentum().generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_params(self):
        data = make_ohlcv()
        signals = MACDMomentum(fast_period=5, slow_period=15, signal_period=4).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_invalid_fast_gte_slow_raises(self):
        with pytest.raises(ValueError):
            MACDMomentum(fast_period=26, slow_period=12)

    def test_invalid_fast_period_zero_raises(self):
        with pytest.raises(ValueError):
            MACDMomentum(fast_period=0)

    def test_invalid_signal_period_zero_raises(self):
        with pytest.raises(ValueError):
            MACDMomentum(signal_period=0)

    def test_long_and_short_signals_generated(self):
        """MACD is a long-short strategy — both sides should fire on typical data."""
        data = make_ohlcv(600)
        signals = MACDMomentum().generate_signals(data)
        assert (signals == 1).any(), "Expected at least one long signal"
        assert (signals == -1).any(), "Expected at least one short signal"

    def test_zero_during_warmup(self):
        """No signal before slow_period + signal_period − 1 bars."""
        fast, slow, sig = 5, 15, 4
        data = make_ohlcv(200)
        signals = MACDMomentum(fast_period=fast, slow_period=slow, signal_period=sig).generate_signals(data)
        warmup = slow + sig - 2  # last bar that must still be 0
        assert (signals.iloc[:warmup] == 0).all(), (
            f"Signals fired during warmup: {signals.iloc[:warmup][signals.iloc[:warmup] != 0]}"
        )

    def test_repr(self):
        assert "MACDMomentum" in repr(MACDMomentum())


# ===========================================================================
# RSI Trend Filter
# ===========================================================================

class TestRSITrendFilter:

    def test_default_params(self):
        data = make_ohlcv()
        signals = RSITrendFilter().generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_params(self):
        data = make_ohlcv()
        signals = RSITrendFilter(
            rsi_period=10, oversold=25.0, overbought=75.0, trend_period=50
        ).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_invalid_rsi_period_raises(self):
        with pytest.raises(ValueError):
            RSITrendFilter(rsi_period=1)

    def test_invalid_trend_period_raises(self):
        with pytest.raises(ValueError):
            RSITrendFilter(trend_period=0)

    def test_invalid_oversold_overbought_raises(self):
        with pytest.raises(ValueError):
            RSITrendFilter(oversold=70.0, overbought=30.0)  # inverted

    def test_oversold_equal_overbought_raises(self):
        with pytest.raises(ValueError):
            RSITrendFilter(oversold=50.0, overbought=50.0)

    def test_can_generate_short_signals(self):
        """
        Verify the strategy fires a short signal when RSI > overbought AND price < trend EMA.

        Scenario:
          - 300 bars of -0.5 %/day decline → price 100 → 22, EMA(50) lags at ~25
          - 3 consecutive +2 % days → price bounces to ~23.7, still below EMA(50)
          - RSI(3) after 3 straight up days ≈ 100 → comfortably above overbought=70
          - Both short conditions met simultaneously → short signal must fire
        """
        n = 400
        close = np.zeros(n)
        close[0] = 100.0
        for i in range(1, 300):
            close[i] = close[i - 1] * 0.995     # sustained decline
        for i in range(300, 303):
            close[i] = close[i - 1] * 1.02      # counter-trend rally
        for i in range(303, n):
            close[i] = close[i - 1] * 0.995     # resume decline
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        data = pd.DataFrame(
            {
                "open":   close * 1.001,
                "high":   close * 1.002,
                "low":    close * 0.998,
                "close":  close,
                "volume": np.ones(n) * 1_000_000,
            },
            index=idx,
        )
        signals = RSITrendFilter(
            rsi_period=3, oversold=20.0, overbought=70.0, trend_period=50
        ).generate_signals(data)
        assert (signals == -1).any(), "Expected short signals: RSI(3)≈100 > 70 while price < EMA(50)"

    def test_zero_during_warmup(self):
        """No signals before max(trend_period, rsi_period) − 1 bars have elapsed."""
        rsi_p, trend_p = 5, 50
        data = make_ohlcv(300)
        signals = RSITrendFilter(
            rsi_period=rsi_p, oversold=30, overbought=70, trend_period=trend_p
        ).generate_signals(data)
        warmup = max(trend_p, rsi_p) - 1
        assert (signals.iloc[:warmup] == 0).all()

    def test_repr(self):
        assert "RSITrendFilter" in repr(RSITrendFilter())


# ===========================================================================
# Time-Series Momentum (TSMOM)
# ===========================================================================

class TestTSMOM:

    def test_default_params(self):
        data = make_ohlcv(500)
        signals = TSMOM().generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_params(self):
        data = make_ohlcv(300)
        signals = TSMOM(lookbacks=[10, 20, 40], vol_target=0.10, vol_lookback=20).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_empty_lookbacks_raises(self):
        with pytest.raises(ValueError):
            TSMOM(lookbacks=[])

    def test_negative_lookback_raises(self):
        with pytest.raises(ValueError):
            TSMOM(lookbacks=[20, -1])

    def test_invalid_vol_target_raises(self):
        with pytest.raises(ValueError):
            TSMOM(vol_target=0.0)

    def test_invalid_vol_lookback_raises(self):
        with pytest.raises(ValueError):
            TSMOM(vol_lookback=1)

    def test_no_fractional_signals(self):
        """Engine compatibility: output must be strictly {-1, 0, 1}."""
        data = make_ohlcv(500)
        signals = TSMOM().generate_signals(data)
        assert set(signals.unique()).issubset({-1, 0, 1}), (
            f"Fractional values leaked into signal output: {signals.unique()}"
        )

    def test_long_and_short_signals_generated(self):
        data = make_ohlcv(600)
        signals = TSMOM(lookbacks=[10, 20, 40], vol_lookback=10).generate_signals(data)
        assert (signals == 1).any(), "Expected long signals"
        assert (signals == -1).any(), "Expected short signals"

    def test_zero_during_warmup(self):
        """No signal before max(max(lookbacks), vol_lookback) bars."""
        lookbacks = [10, 20, 40]
        vol_lb = 15
        data = make_ohlcv(300)
        signals = TSMOM(lookbacks=lookbacks, vol_lookback=vol_lb).generate_signals(data)
        warmup = max(max(lookbacks), vol_lb)
        assert (signals.iloc[:warmup] == 0).all()

    def test_uptrend_generates_long(self):
        """A strong uptrend should produce net long signals from TSMOM."""
        data = make_ohlcv(400, trend=0.005, seed=7)
        signals = TSMOM(lookbacks=[20, 40], vol_lookback=20).generate_signals(data)
        # Net long exposure should dominate in a strong uptrend
        assert (signals == 1).sum() > (signals == -1).sum()

    def test_repr(self):
        assert "TSMOM" in repr(TSMOM())


# ===========================================================================
# Keltner Channel Breakout
# ===========================================================================

class TestKeltnerBreakout:

    def test_default_params(self):
        data = make_ohlcv()
        signals = KeltnerBreakout().generate_signals(data)
        assert_valid_signals(signals, data)

    def test_custom_params(self):
        data = make_ohlcv()
        signals = KeltnerBreakout(period=10, atr_period=7, atr_mult=1.0).generate_signals(data)
        assert_valid_signals(signals, data)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            KeltnerBreakout(period=1)

    def test_invalid_atr_period_raises(self):
        with pytest.raises(ValueError):
            KeltnerBreakout(atr_period=0)

    def test_invalid_atr_mult_raises(self):
        with pytest.raises(ValueError):
            KeltnerBreakout(atr_mult=0.0)

    def test_long_and_short_signals_generated(self):
        """Both breakout directions should fire over a long enough history."""
        data = make_ohlcv(600)
        signals = KeltnerBreakout(atr_mult=0.5).generate_signals(data)  # narrow channel
        assert (signals == 1).any(), "Expected long breakout signals"
        assert (signals == -1).any(), "Expected short breakout signals"

    def test_short_signals_on_downtrend(self):
        """A strong downtrend should produce short entries below the lower channel."""
        data = make_downtrend(500)
        signals = KeltnerBreakout(atr_mult=0.5).generate_signals(data)
        assert (signals == -1).any(), "Expected short signals in a strong downtrend"

    def test_zero_during_warmup(self):
        """No signal before max(period, atr_period) bars have elapsed."""
        period, atr_p = 30, 20
        data = make_ohlcv(300)
        signals = KeltnerBreakout(period=period, atr_period=atr_p).generate_signals(data)
        warmup = max(period, atr_p)
        assert (signals.iloc[:warmup] == 0).all()

    def test_repr(self):
        assert "KeltnerBreakout" in repr(KeltnerBreakout())
