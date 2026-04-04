"""
TDA-enhanced adaptive regime-switching strategy.

Uses persistent homology (Topological Data Analysis) to detect regime
transitions *earlier* than traditional technical indicators.  The topology
of the return point cloud — measured via persistence entropy and loop
structure — changes before price-level indicators (SMA crossovers,
momentum) react.

Architecture:

    1. Takens embedding converts rolling log-returns into a point cloud.
    2. Persistent homology extracts topological features (H0 clusters,
       H1 loops) from the point cloud.
    3. A ``tda_complexity`` score (0 = clean trend, 1 = fragmented/complex)
       provides an early-warning signal for regime transitions.
    4. A hybrid score blends TDA features (leading) with the existing
       technical composite score (confirming) for robust regime classification.

Four regimes:

    BULLISH    → buy-and-hold (capture the uptrend)
    TRANSITION → stay long (TDA early warning; only circuit breaker escalates)
    NEUTRAL    → MACD Momentum (captures both directions, low lag)
    BEARISH    → Keltner Breakout (can short-sell + catch re-entries)
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from backtester.strategy import BaseStrategy
from backtester.strategies.adaptive import _AlwaysLong, _AlwaysFlat, _rolling_regime_score
from backtester.strategies.macd_momentum import MACDMomentum
from backtester.strategies.keltner_breakout import KeltnerBreakout
from backtester.tda.features import compute_tda_features


# ── Regime enum ──────────────────────────────────────────────────────────

class _TDARegime(str, Enum):
    BULLISH = "bullish"
    TRANSITION = "transition"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


# ── Default regime → strategy mapping ────────────────────────────────────

DEFAULT_TDA_REGIME_MAP: dict[str, BaseStrategy] = {
    "bullish": _AlwaysLong(),
    "transition": _AlwaysLong(),  # stay long during early warning; periodic eval will escalate if needed
    "neutral": MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
    "bearish": KeltnerBreakout(period=10, atr_period=7, atr_mult=1.0),
}


# ── Hybrid regime score ─────────────────────────────────────────────────

def _hybrid_regime_score(
    close: pd.Series,
    idx: int,
    tda_complexity: float,
    tda_weight: float = 0.6,
) -> float:
    """
    Blend TDA topological features with the traditional technical score.

    Returns a score in [-1, +1].  Positive = bullish, negative = bearish.
    """
    tech_score = _rolling_regime_score(close, idx)

    # Direction from 20-bar return
    if idx >= 20:
        ret_20 = close.iloc[idx] / close.iloc[idx - 20] - 1
        direction = float(np.sign(ret_20))
    else:
        direction = 0.0

    # TDA score: low complexity + positive direction = bullish
    tda_score = direction * (1.0 - tda_complexity)
    tda_score = max(-1.0, min(1.0, tda_score))

    tech_weight = 1.0 - tda_weight
    hybrid = tda_weight * tda_score + tech_weight * tech_score

    # Complexity penalty: high complexity dampens score toward neutral
    if tda_complexity > 0.6:
        penalty = (tda_complexity - 0.6) * 2.5  # 0.6→0, 1.0→1
        hybrid *= (1.0 - min(penalty, 1.0))

    return max(-1.0, min(1.0, hybrid))


# ── Rolling regime computation ───────────────────────────────────────────

def _compute_tda_regimes(
    close: pd.Series,
    *,
    rebalance_days: int,
    bullish_threshold: float,
    bearish_threshold: float,
    warmup: int,
    confirm_bars: int,
    drawdown_exit: float,
    drawdown_lookback: int,
    recovery_sma: int,
    complexity_alert: float,
    tda_window: int,
    tda_dimension: int,
    tda_delay: int,
    tda_recompute_interval: int,
    tda_weight: float,
) -> pd.Series:
    """
    Compute rolling regime labels using hybrid TDA + technical scoring.

    Returns a Series of ``_TDARegime`` values aligned to *close*.
    """
    regimes = pd.Series(_TDARegime.BULLISH, index=close.index, dtype=object)
    current_regime = _TDARegime.BULLISH
    bars_since_eval = rebalance_days
    bearish_streak = 0
    close_vals = close.values

    # TDA feature cache
    last_tda_complexity = 0.0
    bars_since_tda = tda_recompute_interval  # force first computation

    for i in range(len(close)):
        # ── Circuit breaker: drawdown from rolling high ──
        if drawdown_exit is not None and i >= drawdown_lookback:
            rolling_high = close_vals[max(0, i - drawdown_lookback) : i + 1].max()
            if rolling_high > 0:
                dd = (close_vals[i] - rolling_high) / rolling_high
                if dd <= drawdown_exit:
                    current_regime = _TDARegime.BEARISH
                    bearish_streak = confirm_bars
                    regimes.iloc[i] = current_regime
                    continue

        if i < warmup:
            regimes.iloc[i] = _TDARegime.BULLISH
            continue

        # ── Recompute TDA features periodically ──
        bars_since_tda += 1
        if bars_since_tda >= tda_recompute_interval and i >= tda_window:
            tda_features = compute_tda_features(
                close_vals[: i + 1],
                window=tda_window,
                dimension=tda_dimension,
                delay=tda_delay,
            )
            last_tda_complexity = tda_features["tda_complexity"]
            bars_since_tda = 0

        # ── TDA early warning: complexity spike during circuit-breaker drawdown ──
        # The circuit breaker at drawdown_exit is the primary BULLISH exit.
        # TDA complexity adds an *earlier* exit: if we're already in a
        # significant drawdown (>7%) AND topology is complex, transition
        # before the circuit breaker fires.  This avoids false alarms from
        # normal pullbacks while still providing early warning for real
        # regime changes.
        if (current_regime == _TDARegime.BULLISH
                and last_tda_complexity > complexity_alert
                and drawdown_exit is not None
                and i >= drawdown_lookback):
            rolling_high = close_vals[max(0, i - drawdown_lookback) : i + 1].max()
            if rolling_high > 0:
                dd = (close_vals[i] - rolling_high) / rolling_high
                if dd < drawdown_exit * 0.5:  # 50% of circuit breaker threshold
                    current_regime = _TDARegime.TRANSITION
                    regimes.iloc[i] = current_regime
                    bars_since_eval = 0
                    continue

        # ── Fast recovery: exit BEARISH when price > short SMA ──
        if (recovery_sma > 0
                and current_regime == _TDARegime.BEARISH
                and i >= recovery_sma):
            recovery_mean = close_vals[i - recovery_sma + 1 : i + 1].mean()
            if close_vals[i] > recovery_mean:
                current_regime = _TDARegime.NEUTRAL
                bearish_streak = 0
                regimes.iloc[i] = current_regime
                bars_since_eval = 0
                continue

        # ── Periodic regime evaluation ──
        bars_since_eval += 1
        if bars_since_eval >= rebalance_days:
            # When BULLISH, only circuit breaker or TDA complexity can exit
            if current_regime == _TDARegime.TRANSITION:
                # TRANSITION can only return to BULLISH or let circuit
                # breaker force BEARISH.  This prevents the same drawdown
                # that triggered the TDA alert from cascading into BEARISH.
                s = _hybrid_regime_score(
                    close, i, last_tda_complexity, tda_weight,
                )
                if s >= bullish_threshold:
                    current_regime = _TDARegime.BULLISH
                    bearish_streak = 0
            elif current_regime not in (_TDARegime.BULLISH, _TDARegime.TRANSITION):
                s = _hybrid_regime_score(
                    close, i, last_tda_complexity, tda_weight,
                )

                if s >= bullish_threshold:
                    current_regime = _TDARegime.BULLISH
                    bearish_streak = 0
                elif s <= bearish_threshold:
                    bearish_streak += 1
                    if bearish_streak >= confirm_bars:
                        current_regime = _TDARegime.BEARISH
                else:
                    bearish_streak = 0
                    current_regime = _TDARegime.NEUTRAL

            bars_since_eval = 0

        regimes.iloc[i] = current_regime

    return regimes


# ── Strategy class ───────────────────────────────────────────────────────

class TDAAdaptiveStrategy(BaseStrategy):
    """
    TDA-enhanced regime-switching strategy.

    Uses persistent homology to detect regime transitions earlier than
    traditional technical indicators, then delegates to specialised
    sub-strategies per regime.

    Parameters
    ----------
    tda_window : int
        Rolling window of bars for TDA computation (default 60).
    tda_dimension : int
        Takens embedding dimension (default 3).
    tda_delay : int
        Takens embedding time delay in bars (default 5).
    tda_recompute_interval : int
        Recompute TDA features every N bars (default 5).
    tda_weight : float
        Weight of TDA component in hybrid score (default 0.6).
    complexity_alert : float
        TDA complexity threshold for TRANSITION regime (default 0.6).
    regime_map : dict or None
        Override regime → strategy mapping.
    rebalance_days : int
        Regime evaluation frequency (default 5).
    bullish_threshold : float
        Hybrid score threshold for bullish (default 0.15).
    bearish_threshold : float
        Hybrid score threshold for bearish (default −0.20).
    confirm_bars : int
        Consecutive bearish evaluations before bear regime (default 2).
    warmup : int
        Bars before regime evaluation begins (default 200).
    drawdown_exit : float
        Circuit breaker drawdown threshold (default −0.15).
    drawdown_lookback : int
        Rolling-high window for circuit breaker (default 80).
    recovery_sma : int
        Fast recovery SMA period (default 20).
    """

    name = "TDA Adaptive Regime"

    def __init__(
        self,
        tda_window: int = 60,
        tda_dimension: int = 3,
        tda_delay: int = 5,
        tda_recompute_interval: int = 5,
        tda_weight: float = 0.6,
        complexity_alert: float = 0.6,
        regime_map: dict[str, BaseStrategy] | None = None,
        rebalance_days: int = 5,
        bullish_threshold: float = 0.15,
        bearish_threshold: float = -0.20,
        confirm_bars: int = 2,
        warmup: int = 200,
        drawdown_exit: float = -0.15,
        drawdown_lookback: int = 80,
        recovery_sma: int = 20,
    ) -> None:
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        if bullish_threshold <= bearish_threshold:
            raise ValueError("bullish_threshold must be > bearish_threshold")
        if confirm_bars < 1:
            raise ValueError("confirm_bars must be >= 1")
        if tda_window < 10:
            raise ValueError("tda_window must be >= 10")
        if not 0 < tda_weight <= 1:
            raise ValueError("tda_weight must be in (0, 1]")

        self.tda_window = tda_window
        self.tda_dimension = tda_dimension
        self.tda_delay = tda_delay
        self.tda_recompute_interval = tda_recompute_interval
        self.tda_weight = tda_weight
        self.complexity_alert = complexity_alert
        self.regime_map = regime_map or dict(DEFAULT_TDA_REGIME_MAP)
        self.rebalance_days = rebalance_days
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.confirm_bars = confirm_bars
        self.warmup = warmup
        self.drawdown_exit = drawdown_exit
        self.drawdown_lookback = drawdown_lookback
        self.recovery_sma = recovery_sma

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        # 1. Classify regime at each bar
        regimes = _compute_tda_regimes(
            close,
            rebalance_days=self.rebalance_days,
            bullish_threshold=self.bullish_threshold,
            bearish_threshold=self.bearish_threshold,
            warmup=self.warmup,
            confirm_bars=self.confirm_bars,
            drawdown_exit=self.drawdown_exit,
            drawdown_lookback=self.drawdown_lookback,
            recovery_sma=self.recovery_sma,
            complexity_alert=self.complexity_alert,
            tda_window=self.tda_window,
            tda_dimension=self.tda_dimension,
            tda_delay=self.tda_delay,
            tda_recompute_interval=self.tda_recompute_interval,
            tda_weight=self.tda_weight,
        )

        # 2. Pre-compute signals for every sub-strategy
        sub_signals: dict[str, pd.Series] = {}
        for regime_key, strategy in self.regime_map.items():
            sub_signals[regime_key] = (
                strategy.generate_signals(data)
                .reindex(data.index)
                .fillna(0)
                .astype(int)
            )

        # 3. Stitch: pick the signal from the active sub-strategy at each bar
        combined = pd.Series(0, index=data.index, dtype=int)
        for i in range(len(data)):
            regime_label = regimes.iloc[i].value
            if regime_label in sub_signals:
                combined.iloc[i] = int(sub_signals[regime_label].iloc[i])

        return combined.rename("signal")

    def __repr__(self) -> str:
        mapping = {k: type(v).__name__ for k, v in self.regime_map.items()}
        return (
            f"TDAAdaptiveStrategy(tda_window={self.tda_window}, "
            f"tda_dim={self.tda_dimension}, "
            f"rebalance={self.rebalance_days}d, "
            f"map={mapping})"
        )
