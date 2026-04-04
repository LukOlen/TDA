"""Unit tests for the TDA-enhanced adaptive regime-switching strategy."""
import numpy as np
import pandas as pd
import pytest

from backtester.strategies.tda_adaptive import (
    TDAAdaptiveStrategy,
    _TDARegime,
    _compute_tda_regimes,
    _hybrid_regime_score,
    DEFAULT_TDA_REGIME_MAP,
)
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.macd_momentum import MACDMomentum
from backtester.strategies.keltner_breakout import KeltnerBreakout
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
# _hybrid_regime_score
# ---------------------------------------------------------------------------

class TestHybridRegimeScore:
    def test_low_complexity_positive_direction(self):
        """Low complexity + positive 20-bar return → positive score."""
        data = make_ohlcv(n=400, trend=0.003)
        score = _hybrid_regime_score(data["close"], len(data) - 1, tda_complexity=0.1)
        assert score > 0

    def test_low_complexity_negative_direction(self):
        """Low complexity + negative 20-bar return → negative score."""
        data = make_ohlcv(n=400, trend=-0.003)
        score = _hybrid_regime_score(data["close"], len(data) - 1, tda_complexity=0.1)
        assert score < 0

    def test_high_complexity_dampens(self):
        """High complexity should dampen the score toward zero."""
        data = make_ohlcv(n=400, trend=0.003)
        score_low = _hybrid_regime_score(data["close"], len(data) - 1, tda_complexity=0.1)
        score_high = _hybrid_regime_score(data["close"], len(data) - 1, tda_complexity=0.9)
        assert abs(score_high) < abs(score_low)

    def test_score_bounded(self):
        data = make_ohlcv(n=400, trend=0.005)
        score = _hybrid_regime_score(data["close"], len(data) - 1, tda_complexity=0.5)
        assert -1 <= score <= 1

    def test_early_bar_returns_bounded(self):
        data = make_ohlcv(n=400, trend=0.003)
        score = _hybrid_regime_score(data["close"], 5, tda_complexity=0.3)
        assert -1 <= score <= 1


# ---------------------------------------------------------------------------
# _compute_tda_regimes
# ---------------------------------------------------------------------------

class TestComputeTDARegimes:
    # Use minimal TDA parameters for fast tests
    TDA_PARAMS = dict(
        rebalance_days=5,
        bullish_threshold=0.15,
        bearish_threshold=-0.20,
        warmup=200,
        confirm_bars=2,
        drawdown_exit=-0.15,
        drawdown_lookback=80,
        recovery_sma=20,
        complexity_alert=0.6,
        tda_window=60,
        tda_dimension=3,
        tda_delay=5,
        tda_recompute_interval=20,  # recompute less often for speed
        tda_weight=0.6,
    )

    def test_length_matches(self):
        data = make_ohlcv(n=400)
        regimes = _compute_tda_regimes(data["close"], **self.TDA_PARAMS)
        assert len(regimes) == len(data)

    def test_warmup_is_bullish(self):
        data = make_ohlcv(n=400, trend=0.003)
        regimes = _compute_tda_regimes(data["close"], **self.TDA_PARAMS)
        assert all(r == _TDARegime.BULLISH for r in regimes.iloc[:200])

    def test_valid_regime_values(self):
        data = make_ohlcv(n=400)
        regimes = _compute_tda_regimes(data["close"], **self.TDA_PARAMS)
        valid = {_TDARegime.BULLISH, _TDARegime.TRANSITION, _TDARegime.NEUTRAL, _TDARegime.BEARISH}
        assert set(regimes.unique()).issubset(valid)

    def test_uptrend_mostly_bullish(self):
        data = make_ohlcv(n=500, trend=0.003, seed=42)
        regimes = _compute_tda_regimes(data["close"], **self.TDA_PARAMS)
        post_warmup = regimes.iloc[200:]
        bullish_pct = (post_warmup == _TDARegime.BULLISH).mean()
        # Strong uptrend should be mostly bullish (>40%)
        # Some bars may be TRANSITION due to TDA complexity alerts
        assert bullish_pct > 0.4

    def test_circuit_breaker_forces_bearish(self):
        """A sharp drawdown should force bearish."""
        rng = np.random.default_rng(42)
        n = 400
        prices = np.zeros(n)
        prices[0] = 100.0
        for i in range(1, 300):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        # Crash over 15 bars (~-15%)
        for i in range(300, 315):
            prices[i] = prices[i - 1] * 0.99
        for i in range(315, n):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series(prices, index=idx)

        params = dict(self.TDA_PARAMS)
        params["drawdown_exit"] = -0.07
        params["drawdown_lookback"] = 50
        regimes = _compute_tda_regimes(close, **params)
        crash_regimes = set(regimes.iloc[305:320])
        assert _TDARegime.BEARISH in crash_regimes

    def test_recovery_sma_exits_bearish(self):
        """After circuit breaker, recovery SMA should transition out of bearish."""
        rng = np.random.default_rng(42)
        n = 500
        prices = np.zeros(n)
        prices[0] = 100.0
        for i in range(1, 250):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        for i in range(250, 270):
            prices[i] = prices[i - 1] * 0.99
        for i in range(270, 350):
            prices[i] = prices[i - 1] * 1.005
        for i in range(350, n):
            prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        close = pd.Series(prices, index=idx)

        params = dict(self.TDA_PARAMS)
        params["drawdown_exit"] = -0.07
        params["drawdown_lookback"] = 50
        params["rebalance_days"] = 21  # slow rebalance so recovery SMA acts first
        regimes_no_recovery = _compute_tda_regimes(close, **{**params, "recovery_sma": 0})
        regimes_with_recovery = _compute_tda_regimes(close, **{**params, "recovery_sma": 20})

        bear_bars_no = (regimes_no_recovery == _TDARegime.BEARISH).sum()
        bear_bars_with = (regimes_with_recovery == _TDARegime.BEARISH).sum()
        assert bear_bars_with <= bear_bars_no


# ---------------------------------------------------------------------------
# TDAAdaptiveStrategy construction
# ---------------------------------------------------------------------------

class TestTDAAdaptiveConstruction:
    def test_default_params(self):
        s = TDAAdaptiveStrategy()
        assert s.tda_window == 60
        assert s.tda_dimension == 3
        assert s.tda_delay == 5
        assert s.tda_recompute_interval == 5
        assert s.tda_weight == 0.6
        assert s.complexity_alert == 0.6
        assert s.rebalance_days == 5
        assert s.warmup == 200
        assert s.confirm_bars == 2
        assert s.drawdown_exit == -0.15
        assert s.drawdown_lookback == 80
        assert s.recovery_sma == 20

    def test_default_regime_map(self):
        s = TDAAdaptiveStrategy()
        assert "bullish" in s.regime_map
        assert "transition" in s.regime_map
        assert "neutral" in s.regime_map
        assert "bearish" in s.regime_map
        assert isinstance(s.regime_map["bullish"], _AlwaysLong)
        assert isinstance(s.regime_map["transition"], _AlwaysLong)
        assert isinstance(s.regime_map["neutral"], MACDMomentum)
        assert isinstance(s.regime_map["bearish"], KeltnerBreakout)

    def test_custom_regime_map(self):
        custom = {
            "bullish": SMACrossover(),
            "transition": MACDMomentum(),
            "neutral": MACDMomentum(),
            "bearish": _AlwaysFlat(),
        }
        s = TDAAdaptiveStrategy(regime_map=custom)
        assert isinstance(s.regime_map["bullish"], SMACrossover)

    def test_invalid_rebalance_raises(self):
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(rebalance_days=0)

    def test_invalid_warmup_raises(self):
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(warmup=0)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(bullish_threshold=-0.5, bearish_threshold=0.5)

    def test_invalid_confirm_bars_raises(self):
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(confirm_bars=0)

    def test_invalid_tda_window_raises(self):
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(tda_window=5)

    def test_invalid_tda_weight_raises(self):
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(tda_weight=0)
        with pytest.raises(ValueError):
            TDAAdaptiveStrategy(tda_weight=1.5)

    def test_repr(self):
        s = TDAAdaptiveStrategy()
        r = repr(s)
        assert "TDAAdaptiveStrategy" in r
        assert "tda_window" in r


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestTDAAdaptiveSignals:
    def test_valid_signals(self):
        data = make_ohlcv(n=500, trend=0.001)
        s = TDAAdaptiveStrategy(tda_recompute_interval=20)
        signals = s.generate_signals(data)
        assert_valid_signals(signals, data)

    def test_uptrend_has_longs(self):
        data = make_ohlcv(n=500, trend=0.003)
        s = TDAAdaptiveStrategy(tda_recompute_interval=20)
        signals = s.generate_signals(data)
        assert_valid_signals(signals, data)
        post_warmup = signals.iloc[220:]
        assert (post_warmup == 1).any()

    def test_warmup_is_long(self):
        """Warmup regime is bullish → _AlwaysLong → all 1s during warmup."""
        data = make_ohlcv(n=500, trend=0.002)
        s = TDAAdaptiveStrategy(warmup=200, tda_recompute_interval=20)
        signals = s.generate_signals(data)
        assert (signals.iloc[:200] == 1).all()

    def test_signal_name(self):
        data = make_ohlcv(n=300, trend=0.001)
        s = TDAAdaptiveStrategy(warmup=100, tda_recompute_interval=20)
        signals = s.generate_signals(data)
        assert signals.name == "signal"


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestTDAAdaptiveEngine:
    def test_runs_without_error(self):
        data = make_ohlcv(n=500, trend=0.001)
        engine = BacktestEngine(
            data,
            TDAAdaptiveStrategy(tda_recompute_interval=20),
            initial_capital=100_000,
        )
        result = engine.run()
        assert result.metrics["total_trades"] >= 0
        assert len(result.equity_curve) == len(data)

    def test_equity_starts_at_capital(self):
        data = make_ohlcv(n=500)
        engine = BacktestEngine(
            data,
            TDAAdaptiveStrategy(tda_recompute_interval=20),
            initial_capital=100_000,
        )
        result = engine.run()
        assert result.equity_curve.iloc[0] == pytest.approx(100_000)

    def test_metrics_populated(self):
        data = make_ohlcv(n=500)
        engine = BacktestEngine(
            data,
            TDAAdaptiveStrategy(tda_recompute_interval=20),
            initial_capital=100_000,
        )
        result = engine.run()
        for key in ["total_return", "sharpe_ratio", "max_drawdown", "volatility"]:
            assert key in result.metrics
