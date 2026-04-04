"""
Performance metrics for backtesting results.

All functions accept pandas Series / DataFrames produced by the engine and
return plain Python floats (or dicts) suitable for JSON serialisation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily net returns (decimal, e.g. 0.01 = 1 %).
    risk_free_rate : float
        Annual risk-free rate as a decimal (default 0 %).
    periods : int
        Trading days per year used for annualisation.
    """
    excess = returns - risk_free_rate / periods
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / std)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Annualised Sortino ratio (penalises only downside volatility)."""
    excess = returns - risk_free_rate / periods
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    # Downside deviation is the square root of the mean of squared negative returns.
    downside_variance = (downside ** 2).sum() / len(excess)
    downside_deviation = np.sqrt(downside_variance)
    if downside_deviation == 0 or np.isnan(downside_deviation):
        return float("inf")
    return float(np.sqrt(periods) * excess.mean() / downside_deviation)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative decimal."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Return the full drawdown time series (negative values, 0 at peaks)."""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def cagr(equity_curve: pd.Series, periods: int = 252) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    n_years = (len(equity_curve) - 1) / periods
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / n_years) - 1)


def calmar_ratio(equity_curve: pd.Series, periods: int = 252) -> float:
    """Calmar ratio: CAGR divided by absolute max drawdown."""
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return float("inf")
    return float(cagr(equity_curve, periods) / abs(mdd))


def win_rate(trades: pd.DataFrame) -> float:
    """Fraction of trades with positive P&L."""
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    winners = (trades["pnl"] > 0).sum()
    return float(winners / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    """Gross profit divided by gross loss (> 1 is profitable)."""
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = abs(trades.loc[trades["pnl"] < 0, "pnl"].sum())
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


def avg_trade_return(trades: pd.DataFrame) -> float:
    """Mean P&L percentage per trade."""
    if trades.empty or "pnl_pct" not in trades.columns:
        return 0.0
    return float(trades["pnl_pct"].mean())


def volatility(returns: pd.Series, periods: int = 252) -> float:
    """Annualised standard deviation of daily returns."""
    return float(returns.std(ddof=1) * np.sqrt(periods))


def beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Market beta relative to a benchmark return series."""
    aligned = pd.concat(
        [strategy_returns, benchmark_returns], axis=1, join="inner"
    ).dropna()
    if len(aligned) < 2:
        return float("nan")
    s_ret = aligned.iloc[:, 0]
    b_ret = aligned.iloc[:, 1]
    cov = np.cov(s_ret, b_ret, ddof=1)
    var_b = cov[1, 1]
    if var_b == 0:
        return float("nan")
    return float(cov[0, 1] / var_b)


def alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """Jensen's alpha (annualised)."""
    b = beta(strategy_returns, benchmark_returns)
    if np.isnan(b):
        return float("nan")
    mean_s = strategy_returns.mean() * periods
    mean_b = benchmark_returns.mean() * periods
    return float(mean_s - risk_free_rate - b * (mean_b - risk_free_rate))


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    benchmark_equity: Optional[pd.Series] = None,
) -> dict:
    """
    Compute a full suite of performance statistics.

    Returns a plain dict of metric name → float value, suitable for JSON
    serialisation (``float("inf")`` values are clamped to ``None``).
    """
    bah_returns = (
        benchmark_equity.pct_change().fillna(0)
        if benchmark_equity is not None
        else None
    )

    def _clean(v: float) -> Optional[float]:
        if v is None:
            return None
        if np.isinf(v) or np.isnan(v):
            return None
        return v

    raw: dict[str, Optional[float]] = {
        "total_return": float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1),
        "cagr": cagr(equity_curve),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(equity_curve),
        "volatility": volatility(returns),
        "win_rate": win_rate(trades),
        "profit_factor": profit_factor(trades),
        "avg_trade_return": avg_trade_return(trades),
        "total_trades": int(len(trades)),
        "benchmark_return": float(
            (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
        )
        if benchmark_equity is not None
        else None,
        "beta": _clean(beta(returns, bah_returns)) if bah_returns is not None else None,
        "alpha": _clean(alpha(returns, bah_returns)) if bah_returns is not None else None,
    }

    return {k: _clean(v) if isinstance(v, float) else v for k, v in raw.items()}
