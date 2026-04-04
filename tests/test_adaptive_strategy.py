"""Unit tests for the adaptive regime-switching strategy."""
import numpy as np
import pandas as pd
import pytest

from backtester.strategies.adaptive import (
    AdaptiveStrategy,
    _Regime,
    _compute_rolling_regimes,
    _rolling_regime_score,
    DEFAULT_REGIME_MAP,
)
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import BollingerMeanReversion
from backtester.strategies.momentum import BreakoutMomentum
from backtester.strategies.macd_momentum import MACDMomentum
from backtester.strategies.adaptive import _AlwaysLong, _AlwaysFlat
from backtester.engine import BacktestEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 500, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
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


def assert_valid_signals(signals: pd.Series, data: pd.DataFrame) -> None:
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(data)
    assert signals.index.equals(data.index)
    assert set(signals.unique()).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# _rolling_regime_score
# ---------------------------------------------------------------------------

class TestRollingRegimeScore:
    def test_uptrend_positive(self):
        data = make_ohlcv(n=400, trend=0.003)
        score = _rolling_regime_score(data["close"], len(data) - 1)
        assert score > 0

    def test_downtrend_negative(self):
        data = make_ohlcv(n=400, trend=-0.003)
        score = _rolling_regime_score(data["close"], len(data) - 1)
        assert score < 0

    def test_early_bar_neutral(self):
        data = make_ohlcv(n=400, trend=0.003)
        score = _rolling_regime_score(data["close"], 10)
        # Very little data — should be near zero
        assert -1 <= score <= 1

    def test_score_clamped(self):
        data = make_ohlcv(n=400, trend=0.005)
        score = _rolling_regime_score(data["close"], len(data) - 1)
        assert -1 <= score <= 1

    def test_no_look_ahead(self):
        """Score at bar 250 should only use data[:251]."""
        data = make_ohlcv(n=400, trend=0.002)
        s1 = _rolling_regime_score(data["close"], 250)
        # Modify data after bar 250 drastically
        modified = data["close"].copy()
        modified.iloc[251:] = 0.01
        s2 = _rolling_regime_score(modified, 250)
        assert s1 == s2


# ---------------------------------------------------------------------------
# _compute_rolling_regimes
# ---------------------------------------------------------------------------

class TestComputeRollingRegimes:
    def test_length_matches(self):
        data = make_ohlcv(n=400)
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2,
        )
        assert len(regimes) == len(data)

    def test_warmup_uses_warmup_regime(self):
        data = make_ohlcv(n=400, trend=0.005)
        # Default warmup_regime is BULLISH in _compute_rolling_regimes
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2, warmup=200,
            warmup_regime=_Regime.NEUTRAL,
        )
        assert all(r == _Regime.NEUTRAL for r in regimes.iloc[:200])

    def test_warmup_bullish_default(self):
        data = make_ohlcv(n=400, trend=0.005)
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2, warmup=200,
            warmup_regime=_Regime.BULLISH,
        )
        assert all(r == _Regime.BULLISH for r in regimes.iloc[:200])

    def test_uptrend_eventually_bullish(self):
        data = make_ohlcv(n=500, trend=0.003, seed=42)
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2,
        )
        # Should detect bullish at some point after warmup
        assert _Regime.BULLISH in set(regimes)

    def test_downtrend_eventually_bearish(self):
        data = make_ohlcv(n=500, trend=-0.003, seed=42)
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2,
            confirm_bars=1,  # no confirmation delay for this test
        )
        assert _Regime.BEARISH in set(regimes)

    def test_circuit_breaker_forces_bearish(self):
        """A sharp drawdown should force bearish even during warmup."""
        # Create data with a crash: 300 bars up, then sharp -15% drop
        rng = np.random.default_rng(42)
        n = 400
        prices = np.zeros(n)
        prices[0] = 100.0
        for i in range(1, 300):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        # Crash over 10 bars
        for i in range(300, 320):
            prices[i] = prices[i - 1] * 0.99  # ~-18% total
        for i in range(320, n):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series(prices, index=idx)

        regimes = _compute_rolling_regimes(
            close, rebalance_days=5,
            bullish_threshold=0.15, bearish_threshold=-0.15,
            warmup=200, drawdown_exit=-0.07, drawdown_lookback=50,
        )
        # During/after the crash, should detect bearish
        crash_regimes = set(regimes.iloc[305:320])
        assert _Regime.BEARISH in crash_regimes

    def test_circuit_breaker_disabled(self):
        """Setting drawdown_exit=None should disable the circuit breaker."""
        rng = np.random.default_rng(42)
        n = 400
        prices = 100.0 * np.exp(np.cumsum(rng.normal(-0.003, 0.01, n)))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series(prices, index=idx)

        regimes = _compute_rolling_regimes(
            close, rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2,
            warmup=200, drawdown_exit=None,
        )
        # During warmup should stay at warmup_regime even with big drawdowns
        assert all(r == _Regime.BULLISH for r in regimes.iloc[:200])

    def test_confirm_bars_delays_bear(self):
        """With confirm_bars=3, bear regime requires 3 consecutive bearish readings."""
        data = make_ohlcv(n=500, trend=-0.003, seed=42)
        regimes_fast = _compute_rolling_regimes(
            data["close"], rebalance_days=5,
            bullish_threshold=0.2, bearish_threshold=-0.2,
            confirm_bars=1,
        )
        regimes_slow = _compute_rolling_regimes(
            data["close"], rebalance_days=5,
            bullish_threshold=0.2, bearish_threshold=-0.2,
            confirm_bars=3,
        )
        # The slow version should take longer to go bearish
        first_bear_fast = next((i for i, r in enumerate(regimes_fast) if r == _Regime.BEARISH), len(data))
        first_bear_slow = next((i for i, r in enumerate(regimes_slow) if r == _Regime.BEARISH), len(data))
        assert first_bear_slow >= first_bear_fast

    def test_recovery_sma_exits_bearish_early(self):
        """After circuit breaker fires, recovery SMA should get back in faster."""
        rng = np.random.default_rng(42)
        n = 500
        prices = np.zeros(n)
        prices[0] = 100.0
        for i in range(1, 250):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        # Crash over 20 bars (~18%)
        for i in range(250, 270):
            prices[i] = prices[i - 1] * 0.99
        # Slow recovery over 80 bars
        for i in range(270, 350):
            prices[i] = prices[i - 1] * 1.005
        for i in range(350, n):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series(prices, index=idx)

        # Use long rebalance interval so composite score is slow to update
        regimes_no_recovery = _compute_rolling_regimes(
            close, rebalance_days=21,
            bullish_threshold=0.15, bearish_threshold=-0.15,
            warmup=200, drawdown_exit=-0.07, drawdown_lookback=50,
            recovery_sma=None,
        )
        regimes_with_recovery = _compute_rolling_regimes(
            close, rebalance_days=21,
            bullish_threshold=0.15, bearish_threshold=-0.15,
            warmup=200, drawdown_exit=-0.07, drawdown_lookback=50,
            recovery_sma=20,
        )

        # Both should go bearish during the crash
        assert _Regime.BEARISH in set(regimes_no_recovery)
        assert _Regime.BEARISH in set(regimes_with_recovery)
        # With recovery SMA, should spend fewer bars in bearish
        bear_bars_no = (regimes_no_recovery == _Regime.BEARISH).sum()
        bear_bars_with = (regimes_with_recovery == _Regime.BEARISH).sum()
        assert bear_bars_with < bear_bars_no

    def test_recovery_sma_disabled(self):
        """recovery_sma=None should not affect regime transitions."""
        data = make_ohlcv(n=400, trend=-0.002)
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=5,
            bullish_threshold=0.2, bearish_threshold=-0.2,
            recovery_sma=None,
        )
        assert len(regimes) == len(data)

    def test_valid_regime_values(self):
        data = make_ohlcv(n=400)
        regimes = _compute_rolling_regimes(
            data["close"], rebalance_days=21,
            bullish_threshold=0.2, bearish_threshold=-0.2,
        )
        assert set(regimes.unique()).issubset({_Regime.BULLISH, _Regime.BEARISH, _Regime.NEUTRAL})


# ---------------------------------------------------------------------------
# AdaptiveStrategy construction
# ---------------------------------------------------------------------------

class TestAdaptiveConstruction:
    def test_default_params(self):
        s = AdaptiveStrategy()
        assert s.rebalance_days == 5
        assert s.warmup == 200
        assert s.confirm_bars == 2
        assert "bullish" in s.regime_map
        assert "bearish" in s.regime_map
        assert "neutral" in s.regime_map
        # Bullish should be buy-and-hold, bearish should be cash
        assert isinstance(s.regime_map["bullish"], _AlwaysLong)
        assert isinstance(s.regime_map["bearish"], _AlwaysFlat)
        assert isinstance(s.regime_map["neutral"], SMACrossover)

    def test_custom_regime_map(self):
        custom = {
            "bullish": SMACrossover(),
            "bearish": MACDMomentum(),
            "neutral": BollingerMeanReversion(),
        }
        s = AdaptiveStrategy(regime_map=custom)
        assert isinstance(s.regime_map["bullish"], SMACrossover)

    def test_invalid_confirm_bars_raises(self):
        with pytest.raises(ValueError):
            AdaptiveStrategy(confirm_bars=0)

    def test_invalid_rebalance_raises(self):
        with pytest.raises(ValueError):
            AdaptiveStrategy(rebalance_days=0)

    def test_invalid_warmup_raises(self):
        with pytest.raises(ValueError):
            AdaptiveStrategy(warmup=0)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            AdaptiveStrategy(bullish_threshold=-0.5, bearish_threshold=0.5)

    def test_recovery_sma_default(self):
        s = AdaptiveStrategy()
        assert s.recovery_sma == 30

    def test_repr(self):
        s = AdaptiveStrategy()
        r = repr(s)
        assert "AdaptiveStrategy" in r
        assert "rebalance" in r
        assert "recovery_sma" in r


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestAdaptiveSignals:
    def test_valid_signals(self):
        data = make_ohlcv(n=500, trend=0.001)
        s = AdaptiveStrategy()
        signals = s.generate_signals(data)
        assert_valid_signals(signals, data)

    def test_uptrend_signals(self):
        data = make_ohlcv(n=500, trend=0.003)
        s = AdaptiveStrategy()
        signals = s.generate_signals(data)
        assert_valid_signals(signals, data)
        # In uptrend, should have some long signals after warmup
        post_warmup = signals.iloc[220:]
        assert (post_warmup == 1).any()

    def test_downtrend_signals(self):
        data = make_ohlcv(n=500, trend=-0.003)
        s = AdaptiveStrategy()
        signals = s.generate_signals(data)
        assert_valid_signals(signals, data)

    def test_warmup_is_long(self):
        """Default warmup regime is bullish → always long during warmup."""
        data = make_ohlcv(n=500, trend=0.002)
        s = AdaptiveStrategy(warmup=200)
        signals = s.generate_signals(data)
        assert_valid_signals(signals, data)
        # During warmup, regime is bullish → _AlwaysLong → all 1s
        assert (signals.iloc[:200] == 1).all()


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestAdaptiveEngine:
    def test_runs_without_error(self):
        data = make_ohlcv(n=500, trend=0.001)
        engine = BacktestEngine(data, AdaptiveStrategy(), initial_capital=100_000)
        result = engine.run()
        assert result.metrics["total_trades"] >= 0
        assert len(result.equity_curve) == len(data)

    def test_equity_starts_at_capital(self):
        data = make_ohlcv(n=500)
        engine = BacktestEngine(data, AdaptiveStrategy(), initial_capital=100_000)
        result = engine.run()
        assert result.equity_curve.iloc[0] == pytest.approx(100_000)

    def test_metrics_populated(self):
        data = make_ohlcv(n=500)
        engine = BacktestEngine(data, AdaptiveStrategy(), initial_capital=100_000)
        result = engine.run()
        for key in ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]:
            assert key in result.metrics
