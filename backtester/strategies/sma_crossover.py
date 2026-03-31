"""
Simple Moving Average (SMA) Crossover Strategy
-----------------------------------------------
Generates a long signal (1) when the fast SMA crosses above the slow SMA,
and exits to flat (0) when the fast SMA crosses below the slow SMA.

This is a trend-following strategy — it aims to capture sustained directional
moves while filtering out short-term noise via the dual-MA filter.

Parameters
----------
fast_period : int
    Lookback for the fast moving average (default 20 days).
slow_period : int
    Lookback for the slow moving average (default 50 days).
"""
import pandas as pd

from ..strategy import BaseStrategy


class SMACrossover(BaseStrategy):
    name = "SMA Crossover"

    def __init__(self, fast_period: int = 20, slow_period: int = 50) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        fast_sma = close.rolling(self.fast_period).mean()
        slow_sma = close.rolling(self.slow_period).mean()

        # 1 when fast > slow, 0 otherwise (long-only)
        signal = (fast_sma > slow_sma).astype(int)
        signal = signal.where(slow_sma.notna(), other=0)

        return signal.rename("signal")

    def __repr__(self) -> str:
        return (
            f"SMACrossover(fast={self.fast_period}, slow={self.slow_period})"
        )
