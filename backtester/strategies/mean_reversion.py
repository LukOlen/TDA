"""
Bollinger Band Mean Reversion Strategy
----------------------------------------
Uses Bollinger Bands to identify statistically extreme price levels and fade
the move back toward the mean.

Entry logic (long-only):
  * Go long (1) when close crosses **below** the lower band
    (price is unusually cheap relative to its recent history).
  * Exit to flat (0) when close crosses **above** the upper band or
    reverts back to the middle band.

The number of standard deviations for the bands controls aggressiveness:
a larger ``num_std`` produces fewer, higher-confidence signals.

Parameters
----------
period : int
    Rolling window for computing the mean and standard deviation (default 20).
num_std : float
    Band width in standard deviations (default 2.0 = ~95 % coverage).
"""
import pandas as pd
import numpy as np

from ..strategy import BaseStrategy


class BollingerMeanReversion(BaseStrategy):
    name = "Bollinger Mean Reversion"

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        if period < 2:
            raise ValueError("period must be at least 2")
        if num_std <= 0:
            raise ValueError("num_std must be positive")
        self.period = period
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std(ddof=1)

        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signal = pd.Series(np.nan, index=close.index, name="signal")

        # Set 0 where moving average is not available yet
        signal[pd.isna(mid)] = 0

        # Create boolean masks for conditions
        entry_mask = close < lower
        exit_mask = (close > upper) | (close > mid)
        valid_mask = ~pd.isna(mid)

        # Apply conditions
        # Entry: price pierces the lower band → oversold → go long
        signal.loc[valid_mask & entry_mask] = 1
        # Exit: price reverts to the upper band or middle band
        signal.loc[valid_mask & exit_mask] = 0

        # Forward fill the gaps to maintain position
        signal = signal.ffill()

        return signal.fillna(0).astype(int)

    def __repr__(self) -> str:
        return (
            f"BollingerMeanReversion(period={self.period}, num_std={self.num_std})"
        )
