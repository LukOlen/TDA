"""
Time-Series Momentum (TSMOM) Strategy
---------------------------------------
Implements the Moskowitz, Ooi & Pedersen (2012) time-series momentum factor.

For each lookback horizon h in ``lookbacks``:
    past_return(h) = close[t] / close[t−h] − 1
    direction(h)   = sign(past_return(h))        # +1 / −1 / 0
    realized_vol   = std(daily_ret, vol_lookback) × √252   (annualised)
    scaled(h)      = direction(h) × vol_target / max(realized_vol, ε)

final_position = mean(scaled signals across all horizons)   # NaN-safe
final_position = clip(final_position, −1, 1)

The signal returned is sign(final_position) ∈ {−1, 0, 1}.

NOTE — fractional positions and engine compatibility:
  The underlying model produces continuous position sizes proportional to the
  inverse of realised volatility (volatility targeting).  The current engine
  applies ``astype(int)`` at engine.py:63, which would truncate fractional
  values to 0 and discard the vol-scaling benefit entirely.  This
  implementation therefore returns ``np.sign(clipped)`` — a discrete
  {−1, 0, 1} signal that preserves the *direction* of the vol-scaled position.
  If the engine is later updated to support fractional positions, remove the
  ``np.sign`` call and return ``clipped`` directly.

Academic / theoretical basis:
  Moskowitz, T.J., Ooi, Y.H., & Pedersen, L.H. (2012).
  Time series momentum.  Journal of Financial Economics, 104(2), 228–250.
  The strategy has been shown to generate positive risk-adjusted returns
  across 58 liquid futures markets over 25 years, with a Sharpe ratio of ~1.0
  after costs, and low correlation to traditional risk premia.

Parameters
----------
lookbacks : sequence of int
    Past-return horizons in trading days (default [20, 60, 125, 252]).
vol_target : float
    Target annualised volatility for position scaling (default 0.15 = 15 %).
vol_lookback : int
    Rolling window for realised volatility estimation (default 60 days).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..strategy import BaseStrategy


class TSMOM(BaseStrategy):
    name = "Time-Series Momentum"

    def __init__(
        self,
        lookbacks: Sequence[int] = (20, 60, 125, 252),
        vol_target: float = 0.15,
        vol_lookback: int = 60,
    ) -> None:
        lookbacks = list(lookbacks)
        if not lookbacks:
            raise ValueError("lookbacks must not be empty")
        if any(lb < 1 for lb in lookbacks):
            raise ValueError("all lookback periods must be at least 1")
        if vol_target <= 0:
            raise ValueError("vol_target must be positive")
        if vol_lookback < 2:
            raise ValueError("vol_lookback must be at least 2")
        self.lookbacks = lookbacks
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        daily_ret = close.pct_change()

        # Annualised realised volatility — right-aligned window, no look-ahead
        realized_vol = (
            daily_ret
            .rolling(self.vol_lookback)
            .std(ddof=1)
            .mul(np.sqrt(252))
        )

        scaled_signals: list[pd.Series] = []
        for lb in self.lookbacks:
            # pct_change(lb) = close[t]/close[t-lb] - 1, using only data ≤ t
            past_ret = close.pct_change(lb)
            direction = np.sign(past_ret)

            # Volatility scaling: target / realised; floor avoids division by zero
            vol_floored = realized_vol.clip(lower=1e-8)
            scale = self.vol_target / vol_floored

            scaled_signals.append(direction * scale)

        # Average across horizons; pd.concat + mean handles unequal NaN warmups
        combined = pd.concat(scaled_signals, axis=1).mean(axis=1)

        # Clip to [-1, 1] then discretise for engine compatibility (see docstring)
        clipped = combined.clip(-1.0, 1.0)
        signal = np.sign(clipped)

        # Hard warmup floor: no signal until both vol and longest lookback converge
        warmup = max(max(self.lookbacks), self.vol_lookback)
        signal.iloc[:warmup] = 0.0

        return signal.fillna(0.0).astype(int).rename("signal")

    def __repr__(self) -> str:
        return (
            f"TSMOM("
            f"lookbacks={self.lookbacks}, "
            f"vol_target={self.vol_target}, "
            f"vol_lookback={self.vol_lookback})"
        )
