from __future__ import annotations

import pandas as pd
import numpy as np

from .strategy import BaseStrategy
from .results import BacktestResult
from .stats.metrics import compute_metrics


class BacktestEngine:
    """
    Vectorized backtesting engine.

    The engine avoids look-ahead bias by shifting signals one bar forward
    before calculating returns — i.e. a signal generated on day *t* is
    executed at the open of day *t+1* (approximated as the close of day *t*
    in this daily model).

    Parameters
    ----------
    data : pd.DataFrame
        OHLCV DataFrame with lowercase column names (``open``, ``high``,
        ``low``, ``close``, ``volume``) and a ``DatetimeIndex``.
    strategy : BaseStrategy
        A concrete strategy instance.
    initial_capital : float
        Starting portfolio value in USD.
    commission : float
        Round-trip commission as a fraction of trade value (default 0.1 %).
    ticker : str
        Symbol label used in the result output.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        ticker: str = "UNKNOWN",
    ) -> None:
        if data.empty:
            raise ValueError("data must not be empty")
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data is missing columns: {missing}")

        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.ticker = ticker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return a :class:`BacktestResult`."""
        signals = self.strategy.generate_signals(self.data)
        signals = signals.reindex(self.data.index).fillna(0)
        # Support both float (fractional sizing) and integer signals
        is_float = not np.allclose(signals.values, np.round(signals.values))
        if not is_float:
            signals = signals.round().astype(int)

        # Shift 1 bar to eliminate look-ahead bias
        positions = signals.shift(1).fillna(0)

        close = self.data["close"]
        daily_returns = close.pct_change().fillna(0)

        # Strategy gross returns
        strategy_returns = positions * daily_returns

        # Commission: charged proportional to the size of position change
        position_changes = positions.diff().abs().fillna(0)
        commission_costs = position_changes * self.commission

        net_returns = strategy_returns - commission_costs

        # Equity curve
        equity_curve = self.initial_capital * (1 + net_returns).cumprod()
        equity_curve.iloc[0] = self.initial_capital

        # Extract individual trades (use directional sign for float signals)
        trade_signals = np.sign(signals).astype(int) if is_float else signals
        trades = self._extract_trades(trade_signals, close)

        # Also build a buy-and-hold benchmark equity curve
        bah_returns = daily_returns.copy()
        bah_equity = self.initial_capital * (1 + bah_returns).cumprod()
        bah_equity.iloc[0] = self.initial_capital

        metrics = compute_metrics(net_returns, equity_curve, trades, bah_equity)

        return BacktestResult(
            strategy_name=self.strategy.name,
            ticker=self.ticker,
            start_date=str(self.data.index[0].date()),
            end_date=str(self.data.index[-1].date()),
            initial_capital=self.initial_capital,
            equity_curve=equity_curve,
            returns=net_returns,
            trades=trades,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_trades(
        self, signals: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        """Walk signals and collect per-trade records."""
        trades: list[dict] = []
        position = 0
        entry_price = 0.0
        entry_date = None

        for date, signal in signals.items():
            if signal == position:
                continue

            # Close the current open position
            if position != 0 and entry_date is not None:
                exit_price = float(close.loc[date])
                pnl_pct = (exit_price / entry_price - 1.0) * position
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": "long" if position > 0 else "short",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": (exit_price - entry_price) * position,
                        "pnl_pct": pnl_pct,
                    }
                )

            # Open a new position (if signal != 0)
            if signal != 0:
                entry_price = float(close.loc[date])
                entry_date = date

            position = int(signal)

        _empty_cols = [
            "entry_date", "exit_date", "direction",
            "entry_price", "exit_price", "pnl", "pnl_pct",
        ]
        if not trades:
            return pd.DataFrame(columns=_empty_cols)

        return pd.DataFrame(trades)
