"""
TDA ML Enhanced Strategy
------------------------
Walk-forward ML strategy using TDA topological features plus traditional
technical indicators to predict forward returns.

Architecture:
  1. Feature engineering: multi-horizon returns, volatility, volume ratio,
     RSI, MACD, trend, plus TDA persistence features.
  2. Walk-forward model fitting (GBR or Ridge) with expanding window,
     refitting every 63 bars.
  3. Raw prediction → continuous signal in [-1, 1].
  4. EMA smoothing with fast-exit override for risk protection.
  5. Regime gate (200-day SMA) + circuit breaker on large daily drops.
  6. Discrete output {-1, 0, 1} for BacktestEngine compatibility.
  7. Optional ``generate_float_signals()`` returns pre-discretized floats
     for use by the rotation module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.strategy import BaseStrategy
from backtester.tda.features import compute_tda_features

try:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


class TDAMLEnhancedStrategy(BaseStrategy):
    """ML strategy with GBR/Ridge + TDA features + signal smoothing."""

    name = "TDA ML Enhanced"

    def __init__(
        self,
        lookback: int = 252,
        ridge_alpha: float = 1.0,
        model_type: str = "gbr",
        signal_smoothing: float = 0.15,
        circuit_breaker_pct: float = -0.05,
        # TDA params
        tda_window: int = 60,
        tda_dimension: int = 3,
        tda_delay: int = 5,
        tda_recompute_interval: int = 5,
    ) -> None:
        if lookback < 50:
            raise ValueError("lookback must be >= 50")
        if model_type not in ("gbr", "ridge"):
            raise ValueError("model_type must be 'gbr' or 'ridge'")
        self.lookback = lookback
        self.ridge_alpha = ridge_alpha
        self.model_type = model_type
        self.signal_smoothing = signal_smoothing
        self.circuit_breaker_pct = circuit_breaker_pct
        self.tda_window = tda_window
        self.tda_dimension = tda_dimension
        self.tda_delay = tda_delay
        self.tda_recompute_interval = tda_recompute_interval

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from OHLCV data (no look-ahead)."""
        close = data["close"]
        volume = data["volume"]
        daily_ret = close.pct_change()

        features = pd.DataFrame(index=data.index)

        # Multi-horizon returns
        for h in [5, 10, 21, 63, 126, 252]:
            features[f"ret_{h}d"] = close.pct_change(h)

        # Rolling volatility (annualised)
        for w in [10, 21, 63]:
            features[f"vol_{w}d"] = daily_ret.rolling(w).std() * np.sqrt(252)

        # Volume ratio
        features["vol_ratio_10_63"] = (
            volume.rolling(10).mean() / volume.rolling(63).mean()
        )

        # RSI(14) — Wilder smoothing
        features["rsi"] = self._wilder_rsi(close, period=14)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        features["macd"] = macd_line
        features["macd_hist"] = macd_line - macd_signal

        # Trend: close / 200-SMA - 1
        sma200 = close.rolling(200).mean()
        features["trend_200"] = close / sma200 - 1.0

        # TDA features (computed periodically, forward-filled)
        features = self._add_tda_features(features, close.values)

        return features

    @staticmethod
    def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder RSI: alpha = 1/period via com = period - 1."""
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    def _add_tda_features(
        self, features: pd.DataFrame, close_vals: np.ndarray
    ) -> pd.DataFrame:
        """Compute TDA features periodically and forward-fill."""
        n = len(close_vals)
        tda_cols = [
            "tda_complexity", "h0_entropy", "h0_max_persistence",
            "h0_mean_persistence", "h1_proxy",
        ]
        for col in tda_cols:
            features[col] = np.nan

        bars_since = self.tda_recompute_interval  # force first computation
        for i in range(n):
            bars_since += 1
            if i >= self.tda_window and bars_since >= self.tda_recompute_interval:
                tda = compute_tda_features(
                    close_vals[: i + 1],
                    window=self.tda_window,
                    dimension=self.tda_dimension,
                    delay=self.tda_delay,
                )
                for col in tda_cols:
                    features.iloc[i, features.columns.get_loc(col)] = tda.get(col, 0.0)
                bars_since = 0

        features[tda_cols] = features[tda_cols].ffill().fillna(0.0)
        return features

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model on training data.

        Returns (model, mu, sigma) where model is either a sklearn
        GBR object or a numpy weight vector for ridge regression.
        """
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=1)
        sigma[sigma < 1e-10] = 1.0
        Xn = (X - mu) / sigma

        if self.model_type == "gbr" and _HAS_SKLEARN:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )
            model.fit(Xn, y)
        else:
            # Closed-form ridge: w = (X'X + αI)^{-1} X'y
            n_feat = Xn.shape[1]
            model = np.linalg.solve(
                Xn.T @ Xn + self.ridge_alpha * np.eye(n_feat),
                Xn.T @ y,
            )
        return model, mu, sigma

    def _predict(self, X: np.ndarray, model, mu, sigma) -> np.ndarray:
        """Predict using fitted model. X can be 1D (single row) or 2D."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xn = (X - mu) / sigma
        if self.model_type == "gbr" and _HAS_SKLEARN and hasattr(model, "predict"):
            return model.predict(Xn)
        else:
            return Xn @ model

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return discrete {-1, 0, 1} signals for BacktestEngine."""
        raw = self._generate_raw_signals(data)
        return np.sign(raw).astype(int).rename("signal")

    def generate_float_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return continuous signals in [-1, 1] for the rotation module."""
        return self._generate_raw_signals(data)

    def _generate_raw_signals(self, data: pd.DataFrame) -> pd.Series:
        """Core signal pipeline: features → model → smooth → clip."""
        close = data["close"]
        daily_ret = close.pct_change().fillna(0.0)
        n = len(data)

        # 1. Features
        features_df = self._compute_features(data)
        X_all = features_df.values.copy()

        # 2. Forward 21-day return target (training only)
        fwd_ret = close.pct_change(21).shift(-21).values

        # 3. Walk-forward predictions
        raw_signal = np.zeros(n)
        model = mu = sigma = None
        refit_interval = 63
        start_bar = self.lookback + 21  # need lookback for features + 21 for target

        for i in range(n):
            if i < start_bar:
                continue

            # Refit periodically
            need_refit = (model is None) or ((i - start_bar) % refit_interval == 0)
            if need_refit:
                # Train on data up to current bar minus target gap
                train_end = i - 21
                mask = np.isfinite(X_all[:train_end + 1]).all(axis=1)
                mask &= np.isfinite(fwd_ret[:train_end + 1])
                X_train = X_all[:train_end + 1][mask]
                y_train = fwd_ret[:train_end + 1][mask]
                if len(X_train) >= 60:
                    model, mu, sigma = self._fit_model(X_train, y_train)

            if model is not None:
                x_row = X_all[i]
                if np.isfinite(x_row).all():
                    pred = float(self._predict(x_row, model, mu, sigma)[0])
                    # Scale prediction to [-1, 1]
                    threshold = 0.02  # 2% forward return as full conviction
                    raw_signal[i] = np.clip(
                        np.sign(pred) * min(1.0, abs(pred) / threshold),
                        -1.0, 1.0,
                    )

        # 4. Regime gate: 200-day SMA
        sma200 = close.rolling(200).mean()
        above_trend = (close >= sma200).values

        # 5. Circuit breaker
        circuit_break = daily_ret.values < self.circuit_breaker_pct

        # 6. Apply regime gate and circuit breaker
        for i in range(n):
            if i < 200 or not np.isfinite(sma200.iloc[i]):
                raw_signal[i] = 0.0
                continue
            if circuit_break[i]:
                raw_signal[i] = 0.0
                continue
            # Suppress shorts in uptrend, longs in downtrend
            if above_trend[i] and raw_signal[i] < 0:
                raw_signal[i] = 0.0
            elif not above_trend[i] and raw_signal[i] > 0:
                raw_signal[i] = 0.0

        # 7. Signal smoothing (EMA with fast-exit override)
        if self.signal_smoothing > 0:
            raw_signal = self._smooth_signals(
                raw_signal, daily_ret.values, above_trend, circuit_break,
            )

        return pd.Series(raw_signal, index=data.index, name="signal")

    def _smooth_signals(
        self,
        raw: np.ndarray,
        daily_ret: np.ndarray,
        above_trend: np.ndarray,
        circuit_break: np.ndarray,
    ) -> np.ndarray:
        """EMA smoothing with fast-exit override and regime-change reset."""
        alpha = self.signal_smoothing
        smoothed = np.zeros_like(raw)
        prev = 0.0
        prev_trend = True

        for i in range(len(raw)):
            curr_trend = above_trend[i] if i < len(above_trend) else True
            # Reset on regime change or circuit breaker
            if circuit_break[i] or (curr_trend != prev_trend):
                prev = raw[i]
            else:
                # Fast exit override: if raw drops sharply vs smoothed
                if abs(prev) > 1e-6 and raw[i] < prev * 0.5:
                    prev = raw[i]
                else:
                    prev = alpha * raw[i] + (1.0 - alpha) * prev

            smoothed[i] = prev
            prev_trend = curr_trend

        return smoothed

    def __repr__(self) -> str:
        return (
            f"TDAMLEnhancedStrategy(model={self.model_type}, "
            f"lookback={self.lookback}, smoothing={self.signal_smoothing})"
        )
