"""
Parameter optimisation and walk-forward analysis for backtesting strategies.

All functions are synchronous (CPU-bound pandas work).  Concurrency is
handled at the API layer via ``ThreadPoolExecutor``.
"""
from __future__ import annotations

import itertools
import random as _random
from typing import Any, Optional

import numpy as np
import pandas as pd

from .engine import BacktestEngine
from .strategy import BaseStrategy
from .stats.metrics import compute_metrics


# Valid metric names that can be used as optimisation targets.
VALID_METRICS = {
    "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "calmar_ratio", "volatility",
    "win_rate", "profit_factor", "avg_trade_return", "total_trades",
    "benchmark_return", "beta", "alpha",
}


def _metric_sort_key(value: Any, maximize: bool) -> float:
    """Return a sort key that pushes ``None`` values to the bottom."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("-inf") if maximize else float("inf")
    return float(value)


def _run_single(
    data: pd.DataFrame,
    strategy_cls: type[BaseStrategy],
    params: dict[str, Any],
    initial_capital: float,
    commission: float,
    ticker: str,
) -> Optional[dict]:
    """Run a single backtest, returning metrics dict or *None* on failure."""
    try:
        strategy = strategy_cls(**params)
    except (TypeError, ValueError):
        return None

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_capital=initial_capital,
        commission=commission,
        ticker=ticker,
    )
    result = engine.run()
    return result.metrics


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    data: pd.DataFrame,
    strategy_cls: type[BaseStrategy],
    param_grid: dict[str, list],
    metric: str,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    ticker: str = "UNKNOWN",
    maximize: bool = True,
) -> list[dict]:
    """
    Exhaustive grid search over the parameter space.

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data.
    strategy_cls : type[BaseStrategy]
        Strategy class (not instance).
    param_grid : dict[str, list]
        ``{"fast_period": [10, 20], "slow_period": [40, 50, 60]}``
    metric : str
        Key in ``BacktestResult.metrics`` to optimise.
    maximize : bool
        ``True`` to maximise, ``False`` to minimise.

    Returns
    -------
    list[dict]
        Sorted (best first) list of ``{"rank": int, "params": dict, "metrics": dict}``.
    """
    names = list(param_grid.keys())
    value_lists = [param_grid[n] for n in names]

    results: list[dict] = []
    for combo in itertools.product(*value_lists):
        params = dict(zip(names, combo))
        metrics = _run_single(data, strategy_cls, params, initial_capital, commission, ticker)
        if metrics is not None:
            results.append({"params": params, "metrics": metrics})

    results.sort(
        key=lambda r: _metric_sort_key(r["metrics"].get(metric), maximize),
        reverse=maximize,
    )

    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

def random_search(
    data: pd.DataFrame,
    strategy_cls: type[BaseStrategy],
    param_grid: dict[str, list],
    metric: str,
    n_samples: int = 50,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    ticker: str = "UNKNOWN",
    maximize: bool = True,
    seed: Optional[int] = None,
) -> list[dict]:
    """
    Random search over the parameter space.

    Falls back to grid search when *n_samples* ≥ total combinations.
    """
    names = list(param_grid.keys())
    value_lists = [param_grid[n] for n in names]
    total_combos = 1
    for vl in value_lists:
        total_combos *= len(vl)

    if n_samples >= total_combos:
        return grid_search(
            data, strategy_cls, param_grid, metric,
            initial_capital, commission, ticker, maximize,
        )

    rng = _random.Random(seed)
    seen: set[frozenset] = set()
    samples: list[dict[str, Any]] = []

    while len(samples) < n_samples:
        combo = tuple(rng.choice(vl) for vl in value_lists)
        key = frozenset(zip(names, combo))
        if key in seen:
            continue
        seen.add(key)
        samples.append(dict(zip(names, combo)))

    results: list[dict] = []
    for params in samples:
        metrics = _run_single(data, strategy_cls, params, initial_capital, commission, ticker)
        if metrics is not None:
            results.append({"params": params, "metrics": metrics})

    results.sort(
        key=lambda r: _metric_sort_key(r["metrics"].get(metric), maximize),
        reverse=maximize,
    )
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

_MIN_IS_BARS = 60
_MIN_OOS_BARS = 20


def walk_forward(
    data: pd.DataFrame,
    strategy_cls: type[BaseStrategy],
    param_grid: dict[str, list],
    metric: str,
    n_splits: int = 5,
    in_sample_pct: float = 0.7,
    anchored: bool = False,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    ticker: str = "UNKNOWN",
    maximize: bool = True,
) -> dict:
    """
    Walk-forward optimisation with in-sample/out-of-sample splits.

    Returns
    -------
    dict
        ``{"splits": [...], "aggregate_oos_metrics": {...}}``
    """
    n_rows = len(data)

    if anchored:
        # Fixed start, expanding IS, fixed-size OOS
        oos_size = max(_MIN_OOS_BARS, int(n_rows * (1 - in_sample_pct) / n_splits))
        step = (n_rows - oos_size) // n_splits
        if step < _MIN_IS_BARS:
            raise ValueError(
                f"Insufficient data for {n_splits} anchored splits "
                f"({n_rows} bars). Need at least "
                f"{_MIN_IS_BARS * n_splits + oos_size} bars."
            )
        windows = []
        for i in range(n_splits):
            is_start = 0
            is_end = step * (i + 1)
            oos_start = is_end
            oos_end = min(oos_start + oos_size, n_rows)
            if is_end - is_start < _MIN_IS_BARS or oos_end - oos_start < _MIN_OOS_BARS:
                continue
            windows.append((is_start, is_end, oos_start, oos_end))
    else:
        # Rolling fixed-size windows
        window_size = n_rows // n_splits
        if window_size < _MIN_IS_BARS + _MIN_OOS_BARS:
            raise ValueError(
                f"Insufficient data for {n_splits} rolling splits "
                f"({n_rows} bars, {window_size} per window). Need at least "
                f"{(_MIN_IS_BARS + _MIN_OOS_BARS) * n_splits} bars."
            )
        windows = []
        for i in range(n_splits):
            w_start = i * window_size
            w_end = n_rows if i == n_splits - 1 else (i + 1) * window_size
            split_point = w_start + int((w_end - w_start) * in_sample_pct)
            if split_point - w_start < _MIN_IS_BARS or w_end - split_point < _MIN_OOS_BARS:
                continue
            windows.append((w_start, split_point, split_point, w_end))

    if not windows:
        raise ValueError("No valid walk-forward windows could be constructed.")

    splits: list[dict] = []
    all_oos_returns: list[pd.Series] = []
    all_oos_equity: list[pd.Series] = []

    for idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
        is_data = data.iloc[is_start:is_end]
        oos_data = data.iloc[oos_start:oos_end]

        # Optimise on in-sample
        opt_results = grid_search(
            is_data, strategy_cls, param_grid, metric,
            initial_capital, commission, ticker, maximize,
        )

        if not opt_results:
            continue

        best_params = opt_results[0]["params"]
        is_metrics = opt_results[0]["metrics"]

        # Validate on out-of-sample
        strategy = strategy_cls(**best_params)
        oos_engine = BacktestEngine(
            data=oos_data,
            strategy=strategy,
            initial_capital=initial_capital,
            commission=commission,
            ticker=ticker,
        )
        oos_result = oos_engine.run()

        # Collect OOS data for aggregate metrics
        all_oos_returns.append(oos_result.returns)
        all_oos_equity.append(oos_result.equity_curve)

        oos_equity_points = [
            {"date": str(d.date()), "value": round(v, 2)}
            for d, v in oos_result.equity_curve.items()
        ]

        splits.append({
            "split_index": idx,
            "in_sample_start": str(is_data.index[0].date()),
            "in_sample_end": str(is_data.index[-1].date()),
            "oos_start": str(oos_data.index[0].date()),
            "oos_end": str(oos_data.index[-1].date()),
            "best_params": best_params,
            "in_sample_metrics": is_metrics,
            "oos_metrics": oos_result.metrics,
            "oos_equity_curve": oos_equity_points,
        })

    # Aggregate OOS metrics across all splits
    if all_oos_returns:
        combined_returns = pd.concat(all_oos_returns)
        combined_equity = initial_capital * (1 + combined_returns).cumprod()
        combined_equity.iloc[0] = initial_capital
        # No benchmark for aggregate (would require stitching)
        aggregate_metrics = compute_metrics(combined_returns, combined_equity, pd.DataFrame())
    else:
        aggregate_metrics = {}

    return {
        "splits": splits,
        "aggregate_oos_metrics": aggregate_metrics,
    }
