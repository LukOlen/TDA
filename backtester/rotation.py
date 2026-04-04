"""
Multi-asset rotation engine.

Bypasses BacktestEngine to work with float position sizes directly.
The primary strategy drives the main asset; spare capacity is allocated
to alternative assets ranked by 21-day momentum.

Usage::

    from backtester.rotation import run_rotation_backtest
    result = run_rotation_backtest(
        spy_data, {"TLT": tlt_data, "GLD": gld_data}, strategy,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .results import BacktestResult
from .stats.metrics import compute_metrics


def run_rotation_backtest(
    primary_data: pd.DataFrame,
    alt_assets: dict[str, pd.DataFrame],
    strategy,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    ticker: str = "ROTATION",
) -> BacktestResult:
    """
    Run a rotation backtest combining primary strategy with alt assets.

    Parameters
    ----------
    primary_data : pd.DataFrame
        OHLCV data for the primary asset (e.g. SPY).
    alt_assets : dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame for alternative assets.
    strategy
        Strategy instance with ``generate_float_signals()`` or
        ``generate_signals()`` method.
    initial_capital : float
        Starting portfolio value.
    commission : float
        Round-trip commission as fraction of trade value.
    ticker : str
        Label for the result.

    Returns
    -------
    BacktestResult
    """
    # 1. Get float signals from strategy
    if hasattr(strategy, "generate_float_signals"):
        signals = strategy.generate_float_signals(primary_data)
    else:
        signals = strategy.generate_signals(primary_data).astype(float)
    signals = signals.reindex(primary_data.index).fillna(0.0)

    # 2. Shift by 1 bar (look-ahead protection)
    primary_positions = signals.shift(1).fillna(0.0)

    # 3. Align all assets to common index
    common_idx = primary_data.index.copy()
    alt_closes = {}
    for name, df in alt_assets.items():
        common_idx = common_idx.intersection(df.index)
        alt_closes[name] = df["close"]

    if len(common_idx) == 0:
        if not alt_assets:
            common_idx = primary_data.index
        else:
            raise ValueError("No overlapping dates between primary and alt assets")

    # Reindex everything to common dates
    primary_close = primary_data["close"].reindex(common_idx)
    primary_positions = primary_positions.reindex(common_idx).fillna(0.0)
    primary_returns = primary_close.pct_change().fillna(0.0)

    alt_returns = {}
    for name, close_s in alt_closes.items():
        alt_returns[name] = close_s.reindex(common_idx).pct_change().fillna(0.0)

    # 4. Compute alt positions: spare capacity allocated by 21-day momentum
    n = len(common_idx)
    alt_position_arrays = {name: np.zeros(n) for name in alt_returns}

    for i in range(n):
        spare = max(0.0, 1.0 - abs(primary_positions.iloc[i]))
        if spare < 0.01 or not alt_returns:
            continue

        # Rank alts by 21-day momentum
        mom_scores = {}
        for name, ret_s in alt_returns.items():
            if i < 21:
                continue
            # 21-day cumulative return
            mom = float((1 + ret_s.iloc[i - 20 : i + 1]).prod() - 1)
            if mom > 0:  # only allocate to positive momentum
                mom_scores[name] = mom

        if not mom_scores:
            continue

        # Winner-takes-all: best momentum gets full spare capacity
        winner = max(mom_scores, key=mom_scores.get)
        alt_position_arrays[winner][i] = spare

    # 5. Composite returns with commissions
    composite_returns = primary_positions.values * primary_returns.values

    total_alt_position_change = np.zeros(n)
    for name in alt_returns:
        alt_pos = alt_position_arrays[name]
        alt_ret = alt_returns[name].values
        composite_returns += alt_pos * alt_ret
        total_alt_position_change += np.abs(np.diff(alt_pos, prepend=0.0))

    # Commission on all position changes
    primary_changes = np.abs(np.diff(primary_positions.values, prepend=0.0))
    commission_costs = (primary_changes + total_alt_position_change) * commission
    net_returns = composite_returns - commission_costs

    net_returns_s = pd.Series(net_returns, index=common_idx)

    # 6. Equity curve
    equity_curve = initial_capital * (1 + net_returns_s).cumprod()
    equity_curve.iloc[0] = initial_capital

    # 7. Buy-and-hold benchmark (primary asset only)
    bah_equity = initial_capital * (1 + primary_returns).cumprod()
    bah_equity.iloc[0] = initial_capital

    # 8. Extract trades from primary signal direction changes
    trade_signals = np.sign(signals.reindex(common_idx).fillna(0.0)).astype(int)
    trades = _extract_trades(trade_signals, primary_close)

    # 9. Compute metrics
    metrics = compute_metrics(net_returns_s, equity_curve, trades, bah_equity)

    return BacktestResult(
        strategy_name=f"{strategy.name} + Rotation",
        ticker=ticker,
        start_date=str(common_idx[0].date()),
        end_date=str(common_idx[-1].date()),
        initial_capital=initial_capital,
        equity_curve=equity_curve,
        returns=net_returns_s,
        trades=trades,
        metrics=metrics,
    )


def _extract_trades(signals: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Extract per-trade records from signal series (mirrors engine logic)."""
    trades: list[dict] = []
    position = 0
    entry_price = 0.0
    entry_date = None

    for date, signal in signals.items():
        sig = int(signal)
        if sig == position:
            continue

        if position != 0 and entry_date is not None:
            exit_price = float(close.loc[date])
            pnl_pct = (exit_price / entry_price - 1.0) * position
            trades.append({
                "entry_date": entry_date,
                "exit_date": date,
                "direction": "long" if position > 0 else "short",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": (exit_price - entry_price) * position,
                "pnl_pct": pnl_pct,
            })

        if sig != 0:
            entry_price = float(close.loc[date])
            entry_date = date

        position = sig

    cols = [
        "entry_date", "exit_date", "direction",
        "entry_price", "exit_price", "pnl", "pnl_pct",
    ]
    if not trades:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(trades)
