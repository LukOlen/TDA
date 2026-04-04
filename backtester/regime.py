"""
Market regime detection — classify the broad market as bullish, bearish, or neutral.

Uses multiple technical indicators computed from OHLCV data (and optionally VIX)
to produce a composite score that maps to one of three regimes.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class RegimeIndicator:
    """One component signal contributing to the regime classification."""

    name: str
    value: float
    score: float  # normalised to [-1, +1]
    interpretation: str


@dataclass
class RegimeResult:
    """Full regime classification with supporting evidence."""

    regime: Regime
    composite_score: float  # -1 (max bearish) to +1 (max bullish)
    confidence: float  # 0 to 1
    indicators: list[RegimeIndicator] = field(default_factory=list)
    as_of_date: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


# ---------------------------------------------------------------------------
# Individual indicator scorers
# ---------------------------------------------------------------------------

def _score_price_vs_sma200(close: pd.Series) -> RegimeIndicator:
    sma200 = close.rolling(200).mean()
    last_close = close.iloc[-1]
    last_sma = sma200.iloc[-1]
    if np.isnan(last_sma) or last_sma == 0:
        return RegimeIndicator("Price vs SMA200", 0.0, 0.0, "insufficient data")
    ratio = (last_close - last_sma) / last_sma
    score = _clamp(ratio * 10)
    above = "above" if last_close > last_sma else "below"
    interp = f"Price {above} 200-day SMA by {abs(ratio)*100:.1f}%"
    return RegimeIndicator("Price vs SMA200", float(ratio), score, interp)


def _score_sma_crossover(close: pd.Series) -> RegimeIndicator:
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    last_50 = sma50.iloc[-1]
    last_200 = sma200.iloc[-1]
    if np.isnan(last_50) or np.isnan(last_200) or last_200 == 0:
        return RegimeIndicator("SMA50 vs SMA200", 0.0, 0.0, "insufficient data")
    ratio = (last_50 - last_200) / last_200
    score = _clamp(ratio * 10)
    cross = "golden cross" if last_50 > last_200 else "death cross"
    interp = f"50-day SMA {cross} territory ({ratio*100:+.1f}%)"
    return RegimeIndicator("SMA50 vs SMA200", float(ratio), score, interp)


def _score_momentum(close: pd.Series) -> RegimeIndicator:
    if len(close) < 64:
        return RegimeIndicator("3-Month Momentum", 0.0, 0.0, "insufficient data")
    roc = close.iloc[-1] / close.iloc[-63] - 1
    score = _clamp(roc * 5)
    interp = f"63-day rate of change: {roc*100:+.1f}%"
    return RegimeIndicator("3-Month Momentum", float(roc), score, interp)


def _score_rsi(close: pd.Series) -> RegimeIndicator:
    rsi = _compute_rsi(close, 14)
    last_rsi = rsi.iloc[-1]
    if np.isnan(last_rsi):
        return RegimeIndicator("RSI(14)", 50.0, 0.0, "insufficient data")
    score = _clamp((last_rsi - 50) / 50)
    if last_rsi > 60:
        interp = f"RSI at {last_rsi:.1f} — bullish momentum"
    elif last_rsi < 40:
        interp = f"RSI at {last_rsi:.1f} — bearish momentum"
    else:
        interp = f"RSI at {last_rsi:.1f} — neutral"
    return RegimeIndicator("RSI(14)", float(last_rsi), score, interp)


def _score_volatility_regime(close: pd.Series) -> RegimeIndicator:
    returns = close.pct_change().dropna()
    if len(returns) < 63:
        return RegimeIndicator("Volatility Regime", 0.0, 0.0, "insufficient data")
    vol21 = returns.iloc[-21:].std() * np.sqrt(252)
    vol63 = returns.iloc[-63:].std() * np.sqrt(252)
    if vol63 == 0 or np.isnan(vol63):
        return RegimeIndicator("Volatility Regime", 0.0, 0.0, "zero vol")
    ratio = vol21 / vol63
    # Elevated short-term vol relative to medium-term is bearish
    score = _clamp(-(ratio - 1) * 3)
    if ratio > 1.15:
        interp = f"Short-term vol elevated vs medium-term ({ratio:.2f}x) — stressed"
    elif ratio < 0.85:
        interp = f"Short-term vol depressed vs medium-term ({ratio:.2f}x) — calm"
    else:
        interp = f"Volatility regime stable ({ratio:.2f}x)"
    return RegimeIndicator("Volatility Regime", float(ratio), score, interp)


def _score_vix(start: str, end: str) -> RegimeIndicator | None:
    """Try to fetch VIX data. Returns None if unavailable."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vix_data = load_ohlcv("^VIX", start=start, end=end)
        last_vix = float(vix_data["close"].iloc[-1])
        # Below 15 → +1 (bullish), above 30 → -1 (bearish), linear between
        score = _clamp((22.5 - last_vix) / 7.5)
        if last_vix < 15:
            interp = f"VIX at {last_vix:.1f} — very low fear"
        elif last_vix > 30:
            interp = f"VIX at {last_vix:.1f} — high fear"
        else:
            interp = f"VIX at {last_vix:.1f} — moderate"
        return RegimeIndicator("VIX Level", last_vix, score, interp)
    except Exception as exc:
        logger.warning("Could not fetch VIX data: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

_BASE_WEIGHTS: dict[str, float] = {
    "Price vs SMA200": 0.25,
    "SMA50 vs SMA200": 0.20,
    "3-Month Momentum": 0.15,
    "RSI(14)": 0.10,
    "Volatility Regime": 0.15,
    "VIX Level": 0.15,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_regime(
    benchmark: str = "SPY",
    start: str | None = None,
    end: str | None = None,
    bullish_threshold: float = 0.2,
    bearish_threshold: float = -0.2,
) -> RegimeResult:
    """
    Estimate the current market regime.

    Parameters
    ----------
    benchmark : str
        Market index ticker (default ``"SPY"``).
    start, end : str or None
        Date range.  ``None`` defaults to 18 months ending today (enough
        history for the 200-day SMA warmup).
    bullish_threshold, bearish_threshold : float
        Composite-score thresholds for classifying the regime.

    Returns
    -------
    RegimeResult
        Classification with composite score, confidence, and per-indicator
        breakdown.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=548)).strftime("%Y-%m-%d")

    data = load_ohlcv(benchmark, start=start, end=end)
    close = data["close"]

    # Compute all indicators
    indicators: list[RegimeIndicator] = [
        _score_price_vs_sma200(close),
        _score_sma_crossover(close),
        _score_momentum(close),
        _score_rsi(close),
        _score_volatility_regime(close),
    ]

    vix_indicator = _score_vix(start, end)
    if vix_indicator is not None:
        indicators.append(vix_indicator)

    # Build weights — redistribute VIX weight if unavailable
    weights = dict(_BASE_WEIGHTS)
    if vix_indicator is None:
        vix_w = weights.pop("VIX Level")
        total = sum(weights.values())
        weights = {k: v + vix_w * (v / total) for k, v in weights.items()}

    # Weighted composite score
    composite = 0.0
    for ind in indicators:
        w = weights.get(ind.name, 0.0)
        composite += w * ind.score
    composite = _clamp(composite)

    # Classify
    if composite >= bullish_threshold:
        regime = Regime.BULLISH
    elif composite <= bearish_threshold:
        regime = Regime.BEARISH
    else:
        regime = Regime.NEUTRAL

    confidence = min(abs(composite), 1.0)
    as_of = str(close.index[-1].date()) if hasattr(close.index[-1], "date") else end

    return RegimeResult(
        regime=regime,
        composite_score=float(composite),
        confidence=float(confidence),
        indicators=indicators,
        as_of_date=as_of,
    )
