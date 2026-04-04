"""
TDA Adaptive + Monte Carlo  vs  Buy & Hold S&P 500
====================================================
Runs the tda_adaptive strategy on SPY, then stress-tests both
the strategy returns and the raw B&H returns via Monte Carlo.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies import REGISTRY
from backtester.monte_carlo import run_monte_carlo


# ── settings ──────────────────────────────────────────────────────────
TICKER      = "SPY"
START       = "2018-01-01"
END         = "2024-12-31"
CAPITAL     = 100_000.0
COMMISSION  = 0.001
N_SIMS      = 5_000
HORIZON     = 252          # 1 trading year forward
SEED        = 42
MC_METHODS  = ["bootstrap", "block_bootstrap", "parametric_normal", "parametric_t"]


def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"

def fmt_f(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def print_section(title: str):
    w = 64
    print(f"\n{'═' * w}")
    print(f"  {title}")
    print(f"{'═' * w}")


def print_mc(label: str, mc):
    print(f"\n  ── {label} ({mc.method}, {mc.n_simulations:,} sims, {mc.horizon}d horizon) ──")
    for vr in mc.var_results:
        cl = int(vr.confidence_level * 100)
        print(f"    VaR  {cl}%: {fmt_pct(vr.var):>10}    CVaR {cl}%: {fmt_pct(vr.cvar):>10}")
    rd = mc.return_distribution
    print(f"    Return  mean={fmt_pct(rd.mean)}  median={fmt_pct(rd.median)}  std={fmt_pct(rd.std)}")
    print(f"            5th={fmt_pct(rd.percentile_5)}  95th={fmt_pct(rd.percentile_95)}")
    dd = mc.drawdown_distribution
    print(f"    Max-DD  median={fmt_pct(dd.median)}  95th={fmt_pct(dd.percentile_95)}")
    sd = mc.sharpe_distribution
    print(f"    Sharpe  mean={fmt_f(sd.mean, 2)}  median={fmt_f(sd.median, 2)}  "
          f"5th={fmt_f(sd.percentile_5, 2)}  95th={fmt_f(sd.percentile_95, 2)}")
    print(f"    P(loss): {mc.probability_of_loss * 100:.1f}%")


# =====================================================================
# 1) Load data
# =====================================================================
print(f"Loading {TICKER} {START} → {END} …")
data = load_ohlcv(TICKER, start=START, end=END)
print(f"  {len(data)} trading days loaded")

# =====================================================================
# 2) Run TDA Adaptive backtest
# =====================================================================
print_section("TDA Adaptive Backtest")

tda_strategy = REGISTRY["tda_adaptive"]()
engine = BacktestEngine(
    data=data, strategy=tda_strategy,
    initial_capital=CAPITAL, commission=COMMISSION, ticker=TICKER,
)
tda_result = engine.run()

m = tda_result.metrics
print(f"  Total return : {fmt_pct(m['total_return'])}")
print(f"  CAGR         : {fmt_pct(m['cagr'])}")
print(f"  Sharpe       : {fmt_f(m['sharpe_ratio'], 2)}")
print(f"  Sortino      : {fmt_f(m['sortino_ratio'], 2)}")
print(f"  Max drawdown : {fmt_pct(m['max_drawdown'])}")
print(f"  Volatility   : {fmt_pct(m['volatility'])}")
print(f"  Trades       : {m['total_trades']}")
print(f"  B&H return   : {fmt_pct(m['benchmark_return'])}")
print(f"  Alpha        : {fmt_pct(m['alpha'])}")
print(f"  Beta         : {fmt_f(m['beta'], 2)}")

# =====================================================================
# 3) Build B&H returns series for Monte Carlo comparison
# =====================================================================
bah_returns = data["close"].pct_change().fillna(0)

# =====================================================================
# 4) Monte Carlo — TDA Adaptive vs Buy & Hold
# =====================================================================
print_section("Monte Carlo Stress Test")

for method in MC_METHODS:
    mc_tda = run_monte_carlo(
        tda_result.returns, n_simulations=N_SIMS, horizon=HORIZON,
        method=method, seed=SEED,
    )
    mc_bh = run_monte_carlo(
        bah_returns, n_simulations=N_SIMS, horizon=HORIZON,
        method=method, seed=SEED,
    )
    print_mc(f"TDA Adaptive", mc_tda)
    print_mc(f"Buy & Hold SPY", mc_bh)

    # quick edge summary
    tda_med = mc_tda.return_distribution.median
    bh_med  = mc_bh.return_distribution.median
    tda_dd  = mc_tda.drawdown_distribution.percentile_95
    bh_dd   = mc_bh.drawdown_distribution.percentile_95
    tda_ploss = mc_tda.probability_of_loss
    bh_ploss  = mc_bh.probability_of_loss

    print(f"\n  ── Edge Summary ({method}) ──")
    print(f"    Median return   TDA {fmt_pct(tda_med)} vs B&H {fmt_pct(bh_med)}  "
          f"(delta {fmt_pct(tda_med - bh_med)})")
    print(f"    95th-pct DD     TDA {fmt_pct(tda_dd)} vs B&H {fmt_pct(bh_dd)}  "
          f"(TDA {'better' if tda_dd > bh_dd else 'worse'})")
    print(f"    P(loss)         TDA {tda_ploss*100:.1f}% vs B&H {bh_ploss*100:.1f}%")

    # VaR comparison at 95%
    tda_var95 = next(v for v in mc_tda.var_results if v.confidence_level == 0.95)
    bh_var95  = next(v for v in mc_bh.var_results  if v.confidence_level == 0.95)
    print(f"    VaR 95%         TDA {fmt_pct(tda_var95.var)} vs B&H {fmt_pct(bh_var95.var)}")
    print(f"    CVaR 95%        TDA {fmt_pct(tda_var95.cvar)} vs B&H {fmt_pct(bh_var95.cvar)}")

print(f"\n{'═' * 64}")
print("  Done.")
print(f"{'═' * 64}")
