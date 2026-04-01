"""
MACD Momentum Strategy
-----------------------
Exploits short-term momentum regime shifts via MACD histogram zero-crossings.

MACD line   = EMA(close, fast_period) − EMA(close, slow_period)
Signal line = EMA(MACD line, signal_period)
Histogram   = MACD line − Signal line

Entry rules:
  Long  (1):  histogram crosses from ≤ 0 to > 0  (bullish momentum shift)
  Short (-1): histogram crosses from ≥ 0 to < 0  (bearish momentum shift)
  Flat  (0):  during EMA/signal warmup only

Once a crossover fires the strategy holds the position until the opposing
crossover fires — it is always long or short once warmed up.

Academic / theoretical basis:
  Appel, G. (1979). The Moving Average Convergence-Divergence Method.
  EMA-based histogram crossovers exploit short-term autocorrelation in
  returns and form the core timing signal of many systematic CTA strategies.
  The histogram (not the MACD-line/zero cross) gives earlier, lower-lag entries.

Parameters
----------
fast_period : int
    EMA period for the fast component (default 12).
slow_period : int
    EMA period for the slow component (default 26).
signal_period : int
    EMA period for smoothing the MACD line into the signal line (default 9).
"""
import numpy as np
import pandas as pd

from ..strategy import BaseStrategy


class MACDMomentum(BaseStrategy):
    name = "MACD Momentum"

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> None:
        if fast_period < 1:
            raise ValueError("fast_period must be at least 1")
        if slow_period < 1:
            raise ValueError("slow_period must be at least 1")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period")
        if signal_period < 1:
            raise ValueError("signal_period must be at least 1")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        # adjust=False matches the standard recursive EMA (alpha = 2/(N+1))
        # used by Bloomberg, TA-Lib, and most trading platforms.
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        # Vectorised crossover detection — only reads data up to bar t, no look-ahead
        cross_above = (histogram > 0) & (histogram.shift(1) <= 0)
        cross_below = (histogram < 0) & (histogram.shift(1) >= 0)

        # Warmup: slow EMA needs slow_period bars; signal line needs
        # an additional signal_period bars on top of that.
        warmup = self.slow_period + self.signal_period - 1

        # Stamp +1 / -1 at each crossover, forward-fill between them
        raw = pd.Series(np.nan, index=close.index)
        raw[cross_above] = 1
        raw[cross_below] = -1
        raw.iloc[:warmup] = np.nan  # suppress signals during warmup

        signal = raw.ffill().fillna(0).astype(int)
        return signal.rename("signal")

    def __repr__(self) -> str:
        return (
            f"MACDMomentum("
            f"fast={self.fast_period}, "
            f"slow={self.slow_period}, "
            f"signal={self.signal_period})"
        )
