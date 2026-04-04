#!/usr/bin/env python3
"""Compare TDA Adaptive vs Adaptive vs Buy-and-Hold across market scenarios."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from backtester.engine import BacktestEngine
from backtester.strategies.adaptive import AdaptiveStrategy
from backtester.strategies.tda_adaptive import TDAAdaptiveStrategy


def make_scenario(name: str, n: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for a named market scenario."""
    rng = np.random.default_rng(seed)

    if name == "steady_bull":
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.008, n)))
    elif name == "strong_bull":
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0015, 0.010, n)))
    elif name == "steady_bear":
        close = 100.0 * np.exp(np.cumsum(rng.normal(-0.0008, 0.010, n)))
    elif name == "crash_recovery":
        rets = np.concatenate([
            rng.normal(0.001, 0.008, n // 3),      # bull
            rng.normal(-0.006, 0.020, n // 6),      # crash
            rng.normal(0.002, 0.012, n // 2),       # recovery
        ])[:n]
        close = 100.0 * np.exp(np.cumsum(rets))
    elif name == "choppy":
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
    elif name == "v_shaped":
        mid = n // 2
        rets = np.concatenate([
            rng.normal(-0.003, 0.012, mid),
            rng.normal(0.003, 0.012, n - mid),
        ])
        close = 100.0 * np.exp(np.cumsum(rets))
    elif name == "double_dip":
        seg = n // 5
        rets = np.concatenate([
            rng.normal(0.001, 0.008, seg),      # bull
            rng.normal(-0.004, 0.015, seg),     # dip 1
            rng.normal(0.002, 0.010, seg),      # recovery
            rng.normal(-0.004, 0.015, seg),     # dip 2
            rng.normal(0.002, 0.010, seg),      # recovery
        ])[:n]
        close = 100.0 * np.exp(np.cumsum(rets))
    elif name == "low_vol_grind":
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.004, n)))
    else:
        raise ValueError(f"Unknown scenario: {name}")

    actual_n = len(close)
    high = close * (1 + np.abs(rng.normal(0, 0.004, actual_n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, actual_n)))
    open_ = close * (1 + rng.normal(0, 0.002, actual_n))
    volume = rng.integers(1_000_000, 5_000_000, actual_n).astype(float)
    idx = pd.date_range("2020-01-01", periods=actual_n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def buy_and_hold_return(data: pd.DataFrame) -> float:
    return (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100


def run_strategy(data: pd.DataFrame, strategy, capital: float = 100_000):
    engine = BacktestEngine(data, strategy, initial_capital=capital)
    result = engine.run()
    return result.metrics


def main():
    scenarios = [
        "steady_bull", "strong_bull", "steady_bear", "crash_recovery",
        "choppy", "v_shaped", "double_dip", "low_vol_grind",
    ]
    n_bars = 1000
    capital = 100_000

    adaptive = AdaptiveStrategy()
    tda_adaptive = TDAAdaptiveStrategy()

    print(f"{'Scenario':<18} {'B&H':>8} {'Adaptive':>10} {'TDA Adpt':>10} │ {'α(Adpt)':>9} {'α(TDA)':>9}")
    print("─" * 80)

    total_bh = 0
    total_adpt = 0
    total_tda = 0

    for scenario in scenarios:
        data = make_scenario(scenario, n_bars)
        bh_ret = buy_and_hold_return(data)

        adpt_metrics = run_strategy(data, adaptive, capital)
        tda_metrics = run_strategy(data, tda_adaptive, capital)

        adpt_ret = adpt_metrics["total_return"] * 100
        tda_ret = tda_metrics["total_return"] * 100

        alpha_adpt = adpt_ret - bh_ret
        alpha_tda = tda_ret - bh_ret

        total_bh += bh_ret
        total_adpt += adpt_ret
        total_tda += tda_ret

        print(
            f"{scenario:<18} {bh_ret:>7.1f}% {adpt_ret:>9.1f}% {tda_ret:>9.1f}% │ "
            f"{alpha_adpt:>+8.1f}% {alpha_tda:>+8.1f}%"
        )

    print("─" * 80)
    n = len(scenarios)
    avg_bh = total_bh / n
    avg_adpt = total_adpt / n
    avg_tda = total_tda / n
    print(
        f"{'AVERAGE':<18} {avg_bh:>7.1f}% {avg_adpt:>9.1f}% {avg_tda:>9.1f}% │ "
        f"{avg_adpt - avg_bh:>+8.1f}% {avg_tda - avg_bh:>+8.1f}%"
    )

    print("\n── Risk Metrics (average across scenarios) ──")
    print(f"{'':18} {'Adaptive':>12} {'TDA Adaptive':>14}")

    # Re-run to collect risk metrics
    sharpe_adpt, sharpe_tda = [], []
    dd_adpt, dd_tda = [], []
    for scenario in scenarios:
        data = make_scenario(scenario, n_bars)
        a = run_strategy(data, adaptive, capital)
        t = run_strategy(data, tda_adaptive, capital)
        sharpe_adpt.append(a["sharpe_ratio"])
        sharpe_tda.append(t["sharpe_ratio"])
        dd_adpt.append(a["max_drawdown"] * 100)
        dd_tda.append(t["max_drawdown"] * 100)

    print(f"{'Avg Sharpe':<18} {np.mean(sharpe_adpt):>11.2f} {np.mean(sharpe_tda):>13.2f}")
    print(f"{'Avg Max DD':<18} {np.mean(dd_adpt):>10.1f}% {np.mean(dd_tda):>12.1f}%")


if __name__ == "__main__":
    main()
