"""
N-Day Breakout Momentum Strategy
----------------------------------
A price-channel breakout strategy.  It goes long when today's close breaks
above the rolling N-day high (excluding today), signalling strong upward
momentum.  The position is exited when the close drops below the rolling
N-day low.

This is a classic trend-following / CTA-style approach inspired by the
Donchian Channel and the Turtle Trading rules.

Parameters
----------
lookback : int
    Number of bars used to compute the rolling high/low channel (default 20).
"""
import pandas as pd

from ..strategy import BaseStrategy


class BreakoutMomentum(BaseStrategy):
    name = "Breakout Momentum"

    def __init__(self, lookback: int = 20) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Use rolling over the prior N bars (shift 1 to exclude current bar)
        rolling_high = high.shift(1).rolling(self.lookback).max()
        rolling_low = low.shift(1).rolling(self.lookback).min()

        signal = pd.Series(0, index=close.index, name="signal")
        position = 0

        for i, (date, price) in enumerate(close.items()):
            if pd.isna(rolling_high.iloc[i]):
                signal.loc[date] = 0
                continue

            if position == 0:
                # Breakout above rolling high → go long
                if price > rolling_high.iloc[i]:
                    position = 1
            elif position == 1:
                # Close below rolling low → exit
                if price < rolling_low.iloc[i]:
                    position = 0

            signal.loc[date] = position

        return signal

    def __repr__(self) -> str:
        return f"BreakoutMomentum(lookback={self.lookback})"
