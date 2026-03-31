from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass
class BacktestResult:
    """Container for all outputs produced by a completed backtest run."""

    strategy_name: str
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float

    #: Daily equity curve indexed by date.
    equity_curve: pd.Series
    #: Daily net returns (after commission).
    returns: pd.Series
    #: Per-trade records with entry/exit dates, prices, and P&L.
    trades: pd.DataFrame
    #: Summary performance statistics.
    metrics: dict[str, Any]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize result to a JSON-safe dictionary for the API layer."""
        equity_points = [
            {"date": str(date.date()), "value": round(value, 2)}
            for date, value in self.equity_curve.items()
        ]

        trades_list = []
        if not self.trades.empty:
            for _, row in self.trades.iterrows():
                trades_list.append(
                    {
                        "entry_date": str(row["entry_date"].date())
                        if hasattr(row["entry_date"], "date")
                        else str(row["entry_date"]),
                        "exit_date": str(row["exit_date"].date())
                        if hasattr(row["exit_date"], "date")
                        else str(row["exit_date"]),
                        "direction": row["direction"],
                        "entry_price": round(float(row["entry_price"]), 4),
                        "exit_price": round(float(row["exit_price"]), 4),
                        "pnl": round(float(row["pnl"]), 2),
                        "pnl_pct": round(float(row["pnl_pct"]) * 100, 4),
                    }
                )

        rounded_metrics = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in self.metrics.items()
        }

        return {
            "strategy_name": self.strategy_name,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "metrics": rounded_metrics,
            "equity_curve": equity_points,
            "trades": trades_list,
        }

    def __repr__(self) -> str:
        n = len(self.trades)
        tr = self.metrics.get("total_return", 0)
        return (
            f"BacktestResult(ticker={self.ticker!r}, strategy={self.strategy_name!r}, "
            f"total_return={tr:.2%}, trades={n})"
        )
