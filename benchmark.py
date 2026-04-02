import time
import pandas as pd
import numpy as np

# Original code
class BollingerMeanReversionOrig:
    def __init__(self, period=20, num_std=2.0):
        self.period = period
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std(ddof=1)

        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signal = pd.Series(np.nan, index=close.index, name="signal")
        position = 0

        for i, (date, price) in enumerate(close.items()):
            if pd.isna(mid.iloc[i]):
                signal.loc[date] = 0
                continue

            if position == 0:
                if price < lower.iloc[i]:
                    position = 1
            elif position == 1:
                if price > upper.iloc[i] or price > mid.iloc[i]:
                    position = 0

            signal.loc[date] = position

        return signal.fillna(0).astype(int)

# Vectorized code
class BollingerMeanReversionVectorized:
    def __init__(self, period=20, num_std=2.0):
        self.period = period
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std(ddof=1)

        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        signal = pd.Series(np.nan, index=close.index, name="signal")

        signal[pd.isna(mid)] = 0

        entry_mask = close < lower
        exit_mask = (close > upper) | (close > mid)

        valid_mask = ~pd.isna(mid)

        signal.loc[valid_mask & entry_mask] = 1
        signal.loc[valid_mask & exit_mask] = 0

        # Forward fill the gaps
        signal = signal.ffill()

        return signal.fillna(0).astype(int)

# Create synthetic data
np.random.seed(42)
N = 100000
dates = pd.date_range("2000-01-01", periods=N, freq="min")
# Random walk
close = pd.Series(np.cumsum(np.random.randn(N)), index=dates) + 100
data = pd.DataFrame({"close": close})

# Test correctness
orig = BollingerMeanReversionOrig()
vec = BollingerMeanReversionVectorized()

sig_orig = orig.generate_signals(data)
sig_vec = vec.generate_signals(data)

if not sig_orig.equals(sig_vec):
    print("Mismatch in outputs!")
    diff_mask = sig_orig != sig_vec
    print(pd.DataFrame({'orig': sig_orig[diff_mask], 'vec': sig_vec[diff_mask], 'close': close[diff_mask], 'mid': close.rolling(20).mean()[diff_mask]}))
else:
    print("Outputs match!")

# Benchmark
start = time.perf_counter()
orig.generate_signals(data)
end = time.perf_counter()
print(f"Original time: {end - start:.4f} seconds")

start = time.perf_counter()
vec.generate_signals(data)
end = time.perf_counter()
print(f"Vectorized time: {end - start:.4f} seconds")
