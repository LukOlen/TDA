import time
import pandas as pd
import numpy as np
from backtester.strategies.rsi_trend_filter import RSITrendFilter

def create_dummy_data(n=10000):
    np.random.seed(42)
    # Generate some random walk for price
    price = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    return pd.DataFrame({"close": price})

def benchmark():
    df = create_dummy_data(20000)
    strategy = RSITrendFilter()

    start = time.time()
    for _ in range(10):
        signal = strategy.generate_signals(df)
    end = time.time()

    print(f"Time taken for 10 iterations (20k rows): {end - start:.4f} seconds")

if __name__ == "__main__":
    benchmark()
