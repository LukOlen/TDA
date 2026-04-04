"""
Stock screener — rank a universe of tickers relative to a market benchmark.

Computes per-ticker momentum, risk-adjusted returns, and relative strength,
then produces a cross-sectional percentile-based composite score.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv
from backtester.stats.metrics import (
    max_drawdown,
    sharpe_ratio,
    volatility,
)

logger = logging.getLogger(__name__)

# Default screening universe — liquid large-caps across sectors
DEFAULT_UNIVERSE: list[str] = [
    # Tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    # Financials
    "JPM", "GS", "BAC", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "LLY", "ABBV",
    # Energy
    "XOM", "CVX",
    # Consumer
    "WMT", "KO", "PG", "PEP", "COST", "MCD", "HD",
    # Industrials / Other
    "AVGO", "CRM", "ACN",
]

# Weights for composite score (must sum to 1.0)
_WEIGHTS_FULL: dict[str, float] = {
    "relative_strength": 0.20,
    "momentum_3m": 0.15,
    "momentum_6m": 0.20,
    "momentum_12m": 0.15,
    "sharpe": 0.20,
    "volatility": 0.05,
    "max_drawdown": 0.05,
}

# Metrics where *lower* is better (ranks are inverted)
_LOWER_IS_BETTER = {"volatility", "max_drawdown"}


@dataclass
class ScreenerResult:
    """Screening scores for a single ticker."""

    ticker: str
    rank: int
    composite_score: float
    relative_strength: float
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float | None
    sharpe: float
    volatility: float
    max_drawdown: float


def _momentum(close: pd.Series, n_days: int) -> float | None:
    """Return the n-day simple return, or None if insufficient data."""
    if len(close) < n_days + 1:
        return None
    return float(close.iloc[-1] / close.iloc[-n_days - 1] - 1)


def _score_single_ticker(
    ticker_data: pd.DataFrame,
    benchmark_data: pd.DataFrame,
) -> dict[str, float | None]:
    """Compute raw screening metrics for one ticker."""
    close = ticker_data["close"]
    bench_close = benchmark_data["close"]

    # Align dates
    common = close.index.intersection(bench_close.index)
    close = close.loc[common]
    bench_close = bench_close.loc[common]

    returns = close.pct_change().dropna()

    # Cumulative return of ticker vs benchmark over the full period
    ticker_cum = close.iloc[-1] / close.iloc[0] - 1
    bench_cum = bench_close.iloc[-1] / bench_close.iloc[0] - 1
    rel_strength = float(ticker_cum - bench_cum)

    # Build equity curve for max_drawdown
    equity = (1 + returns).cumprod()

    return {
        "relative_strength": rel_strength,
        "momentum_1m": _momentum(close, 21),
        "momentum_3m": _momentum(close, 63),
        "momentum_6m": _momentum(close, 126),
        "momentum_12m": _momentum(close, 252),
        "sharpe": sharpe_ratio(returns),
        "volatility": volatility(returns),
        "max_drawdown": max_drawdown(equity),
    }


def screen_universe(
    tickers: list[str] | None = None,
    benchmark: str = "SPY",
    start: str | None = None,
    end: str | None = None,
    top_n: int | None = None,
) -> tuple[list[ScreenerResult], list[dict[str, str]]]:
    """
    Screen and rank stocks relative to a benchmark.

    Parameters
    ----------
    tickers : list[str] or None
        Ticker universe.  ``None`` uses :data:`DEFAULT_UNIVERSE`.
    benchmark : str
        Benchmark ticker for relative-strength calculation (default ``"SPY"``).
    start, end : str or None
        Date range (``"YYYY-MM-DD"``).  ``None`` defaults to the trailing year
        ending today.
    top_n : int or None
        Return only the top *N* results.  ``None`` returns all.

    Returns
    -------
    (results, errors)
        *results* sorted by composite score descending (rank 1 = best).
        *errors* is a list of ``{"ticker": ..., "error": ...}`` dicts for
        tickers that could not be loaded.
    """
    if tickers is None:
        tickers = list(DEFAULT_UNIVERSE)

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    # Load benchmark
    benchmark_data = load_ohlcv(benchmark, start=start, end=end)

    # Score each ticker
    scores: dict[str, dict] = {}
    errors: list[dict[str, str]] = []

    for ticker in tickers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load_ohlcv(ticker, start=start, end=end)
            if len(data) < 30:
                errors.append({"ticker": ticker, "error": "insufficient data"})
                continue
            scores[ticker] = _score_single_ticker(data, benchmark_data)
        except Exception as exc:
            logger.warning("Screener: failed to load %s: %s", ticker, exc)
            errors.append({"ticker": ticker, "error": str(exc)})

    if not scores:
        return [], errors

    # Build DataFrame for cross-sectional ranking
    df = pd.DataFrame(scores).T

    # Determine whether 12-month momentum is available for *all* tickers
    has_12m = df["momentum_12m"].notna().all()

    # Select weights — redistribute momentum_12m weight if unavailable
    weights = dict(_WEIGHTS_FULL)
    if not has_12m:
        extra = weights.pop("momentum_12m")
        total = sum(weights.values())
        weights = {k: v + extra * (v / total) for k, v in weights.items()}

    ranked_cols = {}
    for col, w in weights.items():
        series = df[col].astype(float)
        if col in _LOWER_IS_BETTER:
            # Lower values → higher percentile rank
            ranked_cols[col] = series.rank(pct=True, ascending=False)
        else:
            ranked_cols[col] = series.rank(pct=True, ascending=True)

    rank_df = pd.DataFrame(ranked_cols)
    composite = sum(rank_df[col] * w for col, w in weights.items())

    # Build results
    results: list[ScreenerResult] = []
    for ticker_name in composite.sort_values(ascending=False).index:
        raw = scores[ticker_name]
        results.append(
            ScreenerResult(
                ticker=ticker_name,
                rank=0,  # assigned below
                composite_score=float(composite[ticker_name]),
                relative_strength=raw["relative_strength"],
                momentum_1m=raw["momentum_1m"] if raw["momentum_1m"] is not None else 0.0,
                momentum_3m=raw["momentum_3m"] if raw["momentum_3m"] is not None else 0.0,
                momentum_6m=raw["momentum_6m"] if raw["momentum_6m"] is not None else 0.0,
                momentum_12m=raw["momentum_12m"],
                sharpe=raw["sharpe"],
                volatility=raw["volatility"],
                max_drawdown=raw["max_drawdown"],
            )
        )

    for i, r in enumerate(results, 1):
        r.rank = i

    if top_n is not None:
        results = results[:top_n]

    return results, errors
