#!/usr/bin/env python
"""
Example: run all three built-in strategies on AAPL and print a comparison table.

Usage:
    python examples/run_backtest.py
    python examples/run_backtest.py --ticker MSFT --start 2019-01-01 --end 2023-12-31
"""
import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies import (
    SMACrossover,
    BollingerMeanReversion,
    BreakoutMomentum,
)


def fmt_pct(v, decimals=2):
    if v is None:
        return "    N/A"
    return f"{v * 100:+.{decimals}f}%"


def fmt_num(v, decimals=3):
    if v is None:
        return "    N/A"
    return f"{v:.{decimals}f}"


def run(ticker: str, start: str, end: str, capital: float):
    print(f"\nFetching {ticker} from {start} to {end}…")
    data = load_ohlcv(ticker, start=start, end=end)
    print(f"  Loaded {len(data)} trading days\n")

    strategies = [
        SMACrossover(fast_period=20, slow_period=50),
        BollingerMeanReversion(period=20, num_std=2.0),
        BreakoutMomentum(lookback=20),
    ]

    results = []
    for strategy in strategies:
        engine = BacktestEngine(data, strategy, initial_capital=capital, ticker=ticker)
        result = engine.run()
        results.append(result)

    # Header
    col_w = 24
    print(f"{'Metric':<22}", end="")
    for r in results:
        print(f"{r.strategy_name:>{col_w}}", end="")
    print()
    print("-" * (22 + col_w * len(results)))

    rows = [
        ("Total Return",   lambda m: fmt_pct(m["total_return"])),
        ("CAGR",           lambda m: fmt_pct(m["cagr"])),
        ("Sharpe Ratio",   lambda m: fmt_num(m["sharpe_ratio"])),
        ("Sortino Ratio",  lambda m: fmt_num(m["sortino_ratio"])),
        ("Max Drawdown",   lambda m: fmt_pct(m["max_drawdown"])),
        ("Calmar Ratio",   lambda m: fmt_num(m["calmar_ratio"])),
        ("Volatility",     lambda m: fmt_pct(m["volatility"])),
        ("Win Rate",       lambda m: fmt_pct(m["win_rate"])),
        ("Profit Factor",  lambda m: fmt_num(m["profit_factor"])),
        ("Total Trades",   lambda m: f"{m['total_trades']:>{col_w - 1}}"),
        ("Benchmark Ret.", lambda m: fmt_pct(m["benchmark_return"])),
    ]

    for label, fn in rows:
        print(f"{label:<22}", end="")
        for r in results:
            val = fn(r.metrics)
            print(f"{val:>{col_w}}", end="")
        print()

    print()


def main():
    parser = argparse.ArgumentParser(description="TDA Backtester example")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()
    run(args.ticker, args.start, args.end, args.capital)


if __name__ == "__main__":
    main()
