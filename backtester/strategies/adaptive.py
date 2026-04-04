"""
Adaptive regime-switching strategy.

Detects the market regime (bullish / bearish / neutral) on a rolling basis
using technical indicators computed directly from the price data, then
selects the appropriate posture:

    BULLISH  → fully long (buy-and-hold, capture the uptrend)
    NEUTRAL  → long with an SMA filter (stay invested but with a safety net)
    BEARISH  → flat / cash (preserve capital)

The design philosophy is *stay invested by default, step aside only when
danger is real*.  This captures most of the bull-market upside while
avoiding the worst drawdowns.

The regime is re-evaluated every *rebalance_days* trading days.  To avoid
whipsaw, transitioning **into** a bear regime requires the score to stay
below the threshold for *confirm_bars* consecutive evaluations.

**Asymmetric re-entry**: exiting to cash is slow (requires confirmation),
but recovering from cash is fast — when price crosses above a short-term
SMA (*recovery_sma*), the regime immediately transitions to NEUTRAL,
allowing the SMA-crossover sub-strategy to get back in quickly.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from backtester.strategy import BaseStrategy
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.mean_reversion import BollingerMeanReversion
from backtester.strategies.momentum import BreakoutMomentum


# ── Regime enum ───────────────────────────────────────────────────────────

class _Regime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# ── Lightweight helper strategies ─────────────────────────────────────────

class _AlwaysLong(BaseStrategy):
    """Signal = 1 on every bar.  Pure buy-and-hold."""
    name = "Always Long"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=data.index, dtype=int, name="signal")


class _AlwaysFlat(BaseStrategy):
    """Signal = 0 on every bar.  100 % cash."""
    name = "Always Flat"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=data.index, dtype=int, name="signal")


# ── Default regime → strategy mapping ─────────────────────────────────────
# Key insight from backtests:
#   Bull:  Buy-and-hold beats every active strategy in bull markets
#   Neutral: SMA Crossover — mostly long but steps out on trend break
#   Bear:  Cash — avoiding drawdowns is the single biggest alpha source

DEFAULT_REGIME_MAP: dict[str, BaseStrategy] = {
    "bullish": _AlwaysLong(),
    "neutral": SMACrossover(fast_period=20, slow_period=50),
    "bearish": _AlwaysFlat(),
}


# ── Rolling regime indicators (pure price-based, no external data) ────────

def _rolling_regime_score(close: pd.Series, idx: int) -> float:
    """
    Compute a composite regime score at bar *idx* using data up to that bar.

    Returns a score in [-1, +1].  Positive = bullish, negative = bearish.
    Only uses data ``close[:idx+1]`` — no look-ahead.
    """
    window = close.iloc[: idx + 1]
    n = len(window)
    score = 0.0
    total_weight = 0.0

    def _clamp(v: float) -> float:
        return max(-1.0, min(1.0, v))

    # 1. Price vs SMA-200 (weight 0.30)
    #    The single most important trend indicator.
    if n >= 200:
        sma200 = window.iloc[-200:].mean()
        if sma200 != 0:
            ratio = (window.iloc[-1] - sma200) / sma200
            score += 0.30 * _clamp(ratio * 10)
            total_weight += 0.30

    # 2. SMA-50 vs SMA-200 — golden/death cross (weight 0.25)
    if n >= 200:
        sma50 = window.iloc[-50:].mean()
        sma200 = window.iloc[-200:].mean()
        if sma200 != 0:
            ratio = (sma50 - sma200) / sma200
            score += 0.25 * _clamp(ratio * 10)
            total_weight += 0.25

    # 3. 63-day momentum (weight 0.20)
    if n >= 64:
        roc = window.iloc[-1] / window.iloc[-63] - 1
        score += 0.20 * _clamp(roc * 5)
        total_weight += 0.20

    # 4. RSI-14 (weight 0.10)
    if n >= 15:
        delta = window.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean().iloc[-1]
        avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean().iloc[-1]
        if avg_loss != 0:
            rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        else:
            rsi = 100.0
        score += 0.10 * _clamp((rsi - 50) / 50)
        total_weight += 0.10

    # 5. Volatility regime (weight 0.15)
    #    Elevated short-term vol = stress = bearish.
    if n >= 64:
        rets = window.pct_change().dropna()
        vol21 = rets.iloc[-21:].std() * np.sqrt(252) if len(rets) >= 21 else 0
        vol63 = rets.iloc[-63:].std() * np.sqrt(252) if len(rets) >= 63 else 0
        if vol63 > 0:
            ratio = vol21 / vol63
            score += 0.15 * _clamp(-(ratio - 1) * 3)
            total_weight += 0.15

    if total_weight == 0:
        return 0.0
    return max(-1.0, min(1.0, score / total_weight))


# ── Rolling regime computation with confirmation + circuit breaker ────────

def _compute_rolling_regimes(
    close: pd.Series,
    rebalance_days: int,
    bullish_threshold: float,
    bearish_threshold: float,
    warmup: int = 200,
    warmup_regime: _Regime = _Regime.BULLISH,
    confirm_bars: int = 1,
    drawdown_exit: float = -0.07,
    drawdown_lookback: int = 50,
    recovery_sma: int | None = None,
) -> pd.Series:
    """
    Return a Series of regime labels aligned to *close*.

    Parameters
    ----------
    confirm_bars : int
        Number of consecutive bearish evaluations required before switching
        to bear regime.  Prevents whipsaw on transient dips.
    warmup_regime : _Regime
        Regime to use before *warmup* bars are available.
    drawdown_exit : float
        If the price drops more than this fraction from its rolling high,
        force the regime to BEARISH immediately (circuit breaker).
        Set to ``None`` or a very negative value to disable.
    drawdown_lookback : int
        Window (in bars) for computing the rolling high used by the
        circuit breaker.
    recovery_sma : int or None
        When set, the regime transitions from BEARISH to NEUTRAL as soon
        as the price crosses above its *recovery_sma*-bar simple moving
        average.  This provides fast re-entry after drawdown-driven exits.
        ``None`` disables the recovery check.
    """
    regimes = pd.Series(warmup_regime, index=close.index, dtype=object)
    current_regime = warmup_regime
    bars_since_eval = rebalance_days  # force evaluation on first eligible bar
    bearish_streak = 0  # consecutive bearish evaluations
    close_vals = close.values  # numpy for speed

    for i in range(len(close)):
        # ── Circuit breaker: drawdown from rolling high ──
        if drawdown_exit is not None and i >= drawdown_lookback:
            rolling_high = close_vals[max(0, i - drawdown_lookback) : i + 1].max()
            if rolling_high > 0:
                dd = (close_vals[i] - rolling_high) / rolling_high
                if dd <= drawdown_exit:
                    current_regime = _Regime.BEARISH
                    bearish_streak = confirm_bars  # already confirmed
                    regimes.iloc[i] = current_regime
                    continue

        if i < warmup:
            regimes.iloc[i] = warmup_regime
            continue

        # ── Fast recovery: exit BEARISH early when price > short SMA ──
        if (recovery_sma and recovery_sma > 0
                and current_regime == _Regime.BEARISH and i >= recovery_sma):
            recovery_mean = close_vals[i - recovery_sma + 1 : i + 1].mean()
            if close_vals[i] > recovery_mean:
                current_regime = _Regime.NEUTRAL
                bearish_streak = 0
                regimes.iloc[i] = current_regime
                bars_since_eval = 0
                continue

        bars_since_eval += 1
        if bars_since_eval >= rebalance_days:
            # When BULLISH, only the circuit breaker can force an exit.
            # The composite score is too noisy and causes unnecessary
            # regime switches during normal bull-market pullbacks.
            if current_regime != _Regime.BULLISH:
                s = _rolling_regime_score(close, i)

                if s >= bullish_threshold:
                    current_regime = _Regime.BULLISH
                    bearish_streak = 0
                elif s <= bearish_threshold:
                    bearish_streak += 1
                    if bearish_streak >= confirm_bars:
                        current_regime = _Regime.BEARISH
                    # else: keep previous regime (don't rush into bear)
                else:
                    bearish_streak = 0
                    current_regime = _Regime.NEUTRAL

            bars_since_eval = 0

        regimes.iloc[i] = current_regime

    return regimes


# ── Strategy class ────────────────────────────────────────────────────────

class AdaptiveStrategy(BaseStrategy):
    """
    Regime-switching strategy that stays invested by default.

    The key design choices:

    1. **Bullish → fully long** (buy-and-hold).  No active strategy beats
       the market in a sustained uptrend, so don't try.
    2. **Neutral → SMA Crossover**.  Mostly long, but steps aside when the
       short-term trend breaks below the long-term trend.
    3. **Bearish → cash (flat)**.  Capital preservation is the single
       largest source of alpha.  Avoiding a -30 % drawdown is worth more
       than any clever short-selling.
    4. **Warmup defaults to long**.  Markets go up more often than down;
       being invested during the warmup period captures early gains.
    5. **Bear confirmation** prevents whipsaw.  A single bad reading
       doesn't dump the portfolio to cash — the score must stay bearish
       for *confirm_bars* consecutive evaluations.

    Parameters
    ----------
    regime_map : dict or None
        ``"bullish"``/``"bearish"``/``"neutral"`` → strategy instances.
    rebalance_days : int
        Re-evaluation frequency in trading days (default 5 ≈ weekly).
    bullish_threshold : float
        Score above which the regime is bullish (default 0.05).
    bearish_threshold : float
        Score below which the regime *may* be bearish (default -0.25).
    confirm_bars : int
        Consecutive bearish evaluations before switching to bear (default 2).
    warmup : int
        Bars before regime evaluation begins (default 200).
    drawdown_exit : float
        Drawdown circuit breaker.  If the price drops more than this
        fraction (e.g. -0.07 = -7 %) from its *drawdown_lookback*-bar
        rolling high, the regime is forced to BEARISH immediately,
        bypassing the normal scoring and confirmation logic.
    drawdown_lookback : int
        Rolling-high window for the circuit breaker (default 50 bars).
    recovery_sma : int
        After the regime goes bearish, price crossing above this SMA
        triggers a fast transition back to neutral.  Solves the classic
        "sell the dip, miss the recovery" problem (default 20 bars).
    """

    name = "Adaptive Regime"

    def __init__(
        self,
        regime_map: dict[str, BaseStrategy] | None = None,
        rebalance_days: int = 5,
        bullish_threshold: float = 0.05,
        bearish_threshold: float = -0.25,
        confirm_bars: int = 2,
        warmup: int = 200,
        drawdown_exit: float = -0.20,
        drawdown_lookback: int = 100,
        recovery_sma: int = 30,
    ) -> None:
        if rebalance_days < 1:
            raise ValueError("rebalance_days must be >= 1")
        if warmup < 1:
            raise ValueError("warmup must be >= 1")
        if bullish_threshold <= bearish_threshold:
            raise ValueError("bullish_threshold must be > bearish_threshold")
        if confirm_bars < 1:
            raise ValueError("confirm_bars must be >= 1")

        self.regime_map = regime_map or dict(DEFAULT_REGIME_MAP)
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
        regimes = _compute_rolling_regimes(
            close,
            rebalance_days=self.rebalance_days,
            bullish_threshold=self.bullish_threshold,
            bearish_threshold=self.bearish_threshold,
            warmup=self.warmup,
            warmup_regime=_Regime.BULLISH,
            confirm_bars=self.confirm_bars,
            drawdown_exit=self.drawdown_exit,
            drawdown_lookback=self.drawdown_lookback,
            recovery_sma=self.recovery_sma,
        )

        # 2. Pre-compute signals for every sub-strategy
        sub_signals: dict[str, pd.Series] = {}
        for regime_key, strategy in self.regime_map.items():
            sub_signals[regime_key] = (
                strategy.generate_signals(data).reindex(data.index).fillna(0).astype(int)
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
            f"AdaptiveStrategy(rebalance={self.rebalance_days}d, "
            f"confirm={self.confirm_bars}, recovery_sma={self.recovery_sma}, "
            f"map={mapping})"
        )
