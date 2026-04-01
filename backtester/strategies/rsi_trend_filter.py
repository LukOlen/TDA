"""
RSI Trend Filter Strategy
--------------------------
A mean-reversion strategy gated by a long-term trend EMA to avoid
fading strong momentum moves — one of the most common sources of alpha
destruction in raw RSI strategies.

RSI is computed using Wilder's (1978) exponential smoothing:
  alpha = 1 / rsi_period  →  com = rsi_period − 1

Entry rules:
  Long  (1):  RSI < oversold  AND  close > trend_ema   (oversold in uptrend)
  Short (-1): RSI > overbought AND  close < trend_ema  (overbought in downtrend)

Exit rules:
  Exit long  → flat (0):  RSI > overbought  OR  close < trend_ema
  Exit short → flat (0):  RSI < oversold    OR  close > trend_ema

The strategy can be flat (between extremes, or when RSI and price trend
disagree), unlike MACD Momentum which is always in the market.

Academic / theoretical basis:
  Wilder, J.W. (1978). New Concepts in Technical Trading Systems.
  Connors, L. & Alvarez, C. (2009). Short-Term Trading Strategies That Work.
  Lo, A. & Hasanhodzic, J. (2009). The Heretics of Finance.
  The trend gate converts a low-Sharpe oscillator into a genuinely
  selective entry filter, increasing win rate at the cost of fewer trades.

Parameters
----------
rsi_period : int
    Wilder RSI smoothing period (default 14).
oversold : float
    RSI level below which the asset is considered oversold (default 30).
overbought : float
    RSI level above which the asset is considered overbought (default 70).
trend_period : int
    EMA period for the long-term trend filter (default 200).
"""
import numpy as np
import pandas as pd

from ..strategy import BaseStrategy


class RSITrendFilter(BaseStrategy):
    name = "RSI Trend Filter"

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        trend_period: int = 200,
    ) -> None:
        if rsi_period < 2:
            raise ValueError("rsi_period must be at least 2")
        if not (0 < oversold < overbought < 100):
            raise ValueError(
                "oversold and overbought must satisfy 0 < oversold < overbought < 100"
            )
        if trend_period < 1:
            raise ValueError("trend_period must be at least 1")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.trend_period = trend_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wilder_rsi(self, close: pd.Series) -> pd.Series:
        """
        Wilder's RSI using exponential smoothing with alpha = 1/rsi_period.

        ``com = rsi_period − 1`` gives alpha = 1/(1+com) = 1/rsi_period,
        which is the exact Wilder (1978) formulation.  Using ``span=rsi_period``
        would give alpha = 2/(rsi_period+1), over-weighting recent bars.
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(com=self.rsi_period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)  # neutral 50 during initialization

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        rsi = self._wilder_rsi(close)
        trend_ema = close.ewm(span=self.trend_period, adjust=False).mean()

        # Both indicators need to converge; trend EMA dominates warmup
        warmup = max(self.trend_period, self.rsi_period) - 1

        signal = pd.Series(0, index=close.index, name="signal")
        position = 0

        for i in range(len(close)):
            if i < warmup:
                # signal.iloc[i] already 0
                continue

            price = close.iloc[i]
            r = rsi.iloc[i]
            ema = trend_ema.iloc[i]

            if position == 0:
                if r < self.oversold and price > ema:
                    position = 1
                elif r > self.overbought and price < ema:
                    position = -1

            elif position == 1:
                if r > self.overbought or price < ema:
                    position = 0

            elif position == -1:
                if r < self.oversold or price > ema:
                    position = 0

            signal.iloc[i] = position

        return signal

    def __repr__(self) -> str:
        return (
            f"RSITrendFilter("
            f"rsi_period={self.rsi_period}, "
            f"oversold={self.oversold}, "
            f"overbought={self.overbought}, "
            f"trend_period={self.trend_period})"
        )
