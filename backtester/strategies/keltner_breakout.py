"""
Keltner Channel Breakout Strategy
------------------------------------
An ATR-based adaptive channel breakout strategy.  Keltner Channels use
Average True Range (ATR) instead of standard deviation for band width,
making the channel more stable during trending phases and less prone to
volatility-clustering artefacts that inflate Bollinger Band widths.

Channel formulas:
  Midline    = EMA(close, period)
  ATR        = Wilder EMA(True Range, atr_period)   [alpha = 1/atr_period]
  Upper band = Midline + atr_mult × ATR
  Lower band = Midline − atr_mult × ATR

  True Range = max(high−low, |high−prev_close|, |low−prev_close|)

Entry rules:
  Long  (1):  close breaks above upper band  (upside volatility expansion)
  Short (-1): close breaks below lower band  (downside volatility expansion)

Exit rules:
  Exit long:  close falls back to or below midline EMA
  Exit short: close rises back to or above midline EMA

ATR uses Wilder's exponential smoothing (com = atr_period−1, adjust=False),
which gives alpha = 1/atr_period — identical to TA-Lib's ATR function.
Using span=atr_period would over-weight recent bars and diverge from the
standard Wilder (1978) definition.

Academic / theoretical basis:
  Keltner, C.W. (1960). How to Make Money in Commodities.
  Raschke, L.B. (1989). Modified Keltner Channel (ATR-based bands).
  Kaufman, P.J. (2013). Trading Systems and Methods, 5th ed.
  ATR-based channels are documented as superior to SD-based bands for
  breakout detection in futures markets due to their resistance to
  volatility clustering (Kaufman, 2013, ch. 20).

Parameters
----------
period : int
    EMA period for the midline (default 20).
atr_period : int
    Wilder ATR smoothing period (default 14).
atr_mult : float
    Channel half-width in ATR units (default 1.5).
"""
import numpy as np
import pandas as pd

from ..strategy import BaseStrategy


class KeltnerBreakout(BaseStrategy):
    name = "Keltner Channel Breakout"

    def __init__(
        self,
        period: int = 20,
        atr_period: int = 14,
        atr_mult: float = 1.5,
    ) -> None:
        if period < 2:
            raise ValueError("period must be at least 2")
        if atr_period < 1:
            raise ValueError("atr_period must be at least 1")
        if atr_mult <= 0:
            raise ValueError("atr_mult must be positive")
        self.period = period
        self.atr_period = atr_period
        self.atr_mult = atr_mult

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wilder_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        ATR via Wilder's exponential smoothing (alpha = 1/atr_period).

        ``com = atr_period − 1`` gives alpha = 1/(1+com) = 1/atr_period,
        matching TA-Lib's ATR and the original Wilder (1978) formula.
        """
        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        return tr.ewm(com=self.atr_period - 1, adjust=False).mean()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        midline = close.ewm(span=self.period, adjust=False).mean()
        atr = self._wilder_atr(data)

        upper = midline + self.atr_mult * atr
        lower = midline - self.atr_mult * atr

        # Both midline EMA and ATR need to converge before signals are valid
        warmup = max(self.period, self.atr_period)

        price_arr = close.to_numpy()
        mid_arr = midline.to_numpy()
        up_arr = upper.to_numpy()
        lo_arr = lower.to_numpy()

        n = len(close)
        signal_arr = np.zeros(n, dtype=np.int8)
        position = 0

        for i in range(n):
            if i < warmup or np.isnan(up_arr[i]) or np.isnan(lo_arr[i]):
                signal_arr[i] = 0
                continue

            price = price_arr[i]
            mid = mid_arr[i]
            up = up_arr[i]
            lo = lo_arr[i]

            if position == 0:
                if price > up:
                    position = 1
                elif price < lo:
                    position = -1

            elif position == 1:
                # Exit long when price reverts to midline
                if price <= mid:
                    position = 0

            elif position == -1:
                # Exit short when price reverts to midline
                if price >= mid:
                    position = 0

            signal_arr[i] = position

        return pd.Series(signal_arr, index=close.index, name="signal")

    def __repr__(self) -> str:
        return (
            f"KeltnerBreakout("
            f"period={self.period}, "
            f"atr_period={self.atr_period}, "
            f"atr_mult={self.atr_mult})"
        )
