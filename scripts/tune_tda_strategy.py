"""
Systematic tuning of TDA Adaptive strategy parameters.
========================================================
Tests the key levers that explain the poor performance:
  1. BEARISH sub-strategy: KeltnerBreakout (can short) vs flat (cash)
  2. TRANSITION sub-strategy: AlwaysLong vs flat
  3. Circuit breaker: drawdown_exit, drawdown_lookback
  4. Re-entry speed: recovery_sma
  5. TDA params: complexity_alert, tda_weight, rebalance_days
"""
from __future__ import annotations

import sys, os, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies.tda_adaptive import TDAAdaptiveStrategy
from backtester.strategies.adaptive import _AlwaysLong, _AlwaysFlat, AdaptiveStrategy
from backtester.strategies.sma_crossover import SMACrossover
from backtester.strategies.macd_momentum import MACDMomentum
from backtester.strategies.keltner_breakout import KeltnerBreakout
from backtester.monte_carlo import run_monte_carlo

# ── Settings ──────────────────────────────────────────────────────────
TICKER   = "SPY"
START    = "2018-01-01"
END      = "2024-12-31"
CAPITAL  = 100_000.0
COMM     = 0.001

def fmt_pct(v): return f"{v*100:+.2f}%"
def fmt_f(v, d=2): return f"{v:.{d}f}"


# ── Load data once ────────────────────────────────────────────────────
print(f"Loading {TICKER} {START} → {END} …")
data = load_ohlcv(TICKER, start=START, end=END)
print(f"  {len(data)} bars\n")


# ── B&H baseline ─────────────────────────────────────────────────────
bah_rets = data["close"].pct_change().fillna(0)
bah_total = float((1 + bah_rets).prod() - 1)
bah_equity = CAPITAL * (1 + bah_rets).cumprod()
bah_dd = float(((bah_equity / bah_equity.cummax()) - 1).min())
bah_sharpe = float(bah_rets.mean() / bah_rets.std() * np.sqrt(252))
print(f"B&H SPY baseline: return={fmt_pct(bah_total)}  sharpe={fmt_f(bah_sharpe)}  max_dd={fmt_pct(bah_dd)}\n")


# ── Run one config ───────────────────────────────────────────────────
def run_config(label, strategy):
    engine = BacktestEngine(data=data, strategy=strategy,
                            initial_capital=CAPITAL, commission=COMM, ticker=TICKER)
    r = engine.run()
    m = r.metrics
    return {
        "label": label,
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe_ratio"],
        "sortino": m["sortino_ratio"],
        "max_dd": m["max_drawdown"],
        "volatility": m["volatility"],
        "trades": m["total_trades"],
        "calmar": m["calmar_ratio"],
        "returns": r.returns,
    }


# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Test BEARISH sub-strategy (biggest lever)
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: BEARISH sub-strategy")
print("=" * 70)

bearish_variants = {
    "Keltner(default)": KeltnerBreakout(period=10, atr_period=7, atr_mult=1.0),
    "Flat (cash)":      _AlwaysFlat(),
    "SMA(10,30)":       SMACrossover(fast_period=10, slow_period=30),
    "MACD(8,21,5)":     MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
}

phase1_results = []
for name, bear_strat in bearish_variants.items():
    regime_map = {
        "bullish":    _AlwaysLong(),
        "transition": _AlwaysLong(),
        "neutral":    MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
        "bearish":    bear_strat,
    }
    strat = TDAAdaptiveStrategy(regime_map=regime_map)
    res = run_config(f"bear={name}", strat)
    phase1_results.append(res)
    print(f"  {name:20s}  ret={fmt_pct(res['total_return'])}  sharpe={fmt_f(res['sharpe'])}  "
          f"max_dd={fmt_pct(res['max_dd'])}  trades={res['trades']}")

best_bear = max(phase1_results, key=lambda x: x["sharpe"])
print(f"\n  → Best BEARISH sub: {best_bear['label']}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Test TRANSITION sub-strategy
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 2: TRANSITION sub-strategy")
print("=" * 70)

trans_variants = {
    "AlwaysLong":   _AlwaysLong(),
    "Flat (cash)":  _AlwaysFlat(),
    "SMA(20,50)":   SMACrossover(fast_period=20, slow_period=50),
}

phase2_results = []
for name, trans_strat in trans_variants.items():
    regime_map = {
        "bullish":    _AlwaysLong(),
        "transition": trans_strat,
        "neutral":    MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
        "bearish":    _AlwaysFlat(),   # use flat from phase 1 insight
    }
    strat = TDAAdaptiveStrategy(regime_map=regime_map)
    res = run_config(f"trans={name}", strat)
    phase2_results.append(res)
    print(f"  {name:20s}  ret={fmt_pct(res['total_return'])}  sharpe={fmt_f(res['sharpe'])}  "
          f"max_dd={fmt_pct(res['max_dd'])}  trades={res['trades']}")

best_trans = max(phase2_results, key=lambda x: x["sharpe"])
print(f"\n  → Best TRANSITION sub: {best_trans['label']}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Grid over circuit breaker + recovery params
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 3: Circuit breaker & recovery tuning")
print("=" * 70)

dd_exits     = [-0.06, -0.08, -0.10, -0.12, -0.15]
dd_lookbacks = [30, 50, 80]
recovery_smas = [15, 20, 30, 50]

phase3_results = []
for dd_exit, dd_lb, rec_sma in itertools.product(dd_exits, dd_lookbacks, recovery_smas):
    regime_map = {
        "bullish":    _AlwaysLong(),
        "transition": _AlwaysLong(),
        "neutral":    MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
        "bearish":    _AlwaysFlat(),
    }
    strat = TDAAdaptiveStrategy(
        regime_map=regime_map,
        drawdown_exit=dd_exit,
        drawdown_lookback=dd_lb,
        recovery_sma=rec_sma,
    )
    res = run_config(f"dd={dd_exit}/lb={dd_lb}/rec={rec_sma}", strat)
    phase3_results.append(res | {"dd_exit": dd_exit, "dd_lb": dd_lb, "rec_sma": rec_sma})

# Sort by Sharpe, show top 10
phase3_results.sort(key=lambda x: x["sharpe"], reverse=True)
print(f"\n  Top 10 by Sharpe (of {len(phase3_results)} combos):")
for i, r in enumerate(phase3_results[:10]):
    print(f"  {i+1:2d}. dd_exit={r['dd_exit']:6.2f} lb={r['dd_lb']:3d} rec={r['rec_sma']:3d}  "
          f"ret={fmt_pct(r['total_return'])}  sharpe={fmt_f(r['sharpe'])}  "
          f"max_dd={fmt_pct(r['max_dd'])}  calmar={fmt_f(r['calmar'])}")

best_p3 = phase3_results[0]


# ══════════════════════════════════════════════════════════════════════
# PHASE 4: TDA-specific params on the best circuit breaker config
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 4: TDA params (complexity_alert, tda_weight, rebalance_days)")
print("=" * 70)

complexities  = [0.3, 0.4, 0.5, 0.6, 0.7]
tda_weights   = [0.4, 0.5, 0.6, 0.7, 0.8]
rebal_days    = [3, 5, 10]

phase4_results = []
for comp, tw, rb in itertools.product(complexities, tda_weights, rebal_days):
    regime_map = {
        "bullish":    _AlwaysLong(),
        "transition": _AlwaysLong(),
        "neutral":    MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
        "bearish":    _AlwaysFlat(),
    }
    strat = TDAAdaptiveStrategy(
        regime_map=regime_map,
        drawdown_exit=best_p3["dd_exit"],
        drawdown_lookback=best_p3["dd_lb"],
        recovery_sma=best_p3["rec_sma"],
        complexity_alert=comp,
        tda_weight=tw,
        rebalance_days=rb,
    )
    res = run_config(f"comp={comp}/tw={tw}/rb={rb}", strat)
    phase4_results.append(res | {"comp": comp, "tw": tw, "rb": rb})

phase4_results.sort(key=lambda x: x["sharpe"], reverse=True)
print(f"\n  Top 10 by Sharpe (of {len(phase4_results)} combos):")
for i, r in enumerate(phase4_results[:10]):
    print(f"  {i+1:2d}. comp={r['comp']:.1f} tw={r['tw']:.1f} rb={r['rb']:2d}  "
          f"ret={fmt_pct(r['total_return'])}  sharpe={fmt_f(r['sharpe'])}  "
          f"max_dd={fmt_pct(r['max_dd'])}  calmar={fmt_f(r['calmar'])}")

best_p4 = phase4_results[0]


# ══════════════════════════════════════════════════════════════════════
# PHASE 5: Also test non-TDA adaptive with tuning
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 5: Non-TDA Adaptive baseline comparison")
print("=" * 70)

adaptive_configs = [
    ("Adaptive(default)", AdaptiveStrategy()),
    ("Adaptive(dd=-0.07,rec=20)", AdaptiveStrategy(drawdown_exit=-0.07, drawdown_lookback=50, recovery_sma=20)),
    ("Adaptive(dd=-0.10,rec=30)", AdaptiveStrategy(drawdown_exit=-0.10, drawdown_lookback=50, recovery_sma=30)),
    ("Adaptive(dd=-0.06,rec=40)", AdaptiveStrategy(drawdown_exit=-0.06, drawdown_lookback=30, recovery_sma=40)),
]

phase5_results = []
for name, strat in adaptive_configs:
    res = run_config(name, strat)
    phase5_results.append(res)
    print(f"  {name:40s}  ret={fmt_pct(res['total_return'])}  sharpe={fmt_f(res['sharpe'])}  "
          f"max_dd={fmt_pct(res['max_dd'])}  calmar={fmt_f(res['calmar'])}")


# ══════════════════════════════════════════════════════════════════════
# PHASE 6: Final shootout + Monte Carlo
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("PHASE 6: FINAL SHOOTOUT — best TDA tuned vs best Adaptive vs B&H")
print("=" * 70)

# Rebuild best TDA config
regime_map_best = {
    "bullish":    _AlwaysLong(),
    "transition": _AlwaysLong(),
    "neutral":    MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
    "bearish":    _AlwaysFlat(),
}
best_tda_strat = TDAAdaptiveStrategy(
    regime_map=regime_map_best,
    drawdown_exit=best_p3["dd_exit"],
    drawdown_lookback=best_p3["dd_lb"],
    recovery_sma=best_p3["rec_sma"],
    complexity_alert=best_p4["comp"],
    tda_weight=best_p4["tw"],
    rebalance_days=best_p4["rb"],
)

# Best non-TDA adaptive
best_adaptive_res = max(phase5_results, key=lambda x: x["sharpe"])

# Run the final three
finalists = {
    "TDA Tuned": run_config("TDA Tuned", best_tda_strat),
    "Adaptive Best": best_adaptive_res,
}

print(f"\n  {'Strategy':25s} {'Return':>10s} {'CAGR':>10s} {'Sharpe':>8s} {'Sortino':>8s} "
      f"{'MaxDD':>10s} {'Calmar':>8s} {'Trades':>7s}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*7}")

print(f"  {'B&H SPY':25s} {fmt_pct(bah_total):>10s} {'':>10s} {fmt_f(bah_sharpe):>8s} "
      f"{'':>8s} {fmt_pct(bah_dd):>10s} {'':>8s} {'0':>7s}")

for name, r in finalists.items():
    print(f"  {name:25s} {fmt_pct(r['total_return']):>10s} {fmt_pct(r['cagr']):>10s} "
          f"{fmt_f(r['sharpe']):>8s} {fmt_f(r['sortino']):>8s} "
          f"{fmt_pct(r['max_dd']):>10s} {fmt_f(r['calmar']):>8s} {r['trades']:>7d}")

# Best tuned params summary
print(f"\n  Best TDA params:")
print(f"    drawdown_exit={best_p3['dd_exit']}, drawdown_lookback={best_p3['dd_lb']}, "
      f"recovery_sma={best_p3['rec_sma']}")
print(f"    complexity_alert={best_p4['comp']}, tda_weight={best_p4['tw']}, "
      f"rebalance_days={best_p4['rb']}")
print(f"    BEARISH=Flat, TRANSITION=AlwaysLong")


# ── Monte Carlo on the winner ─────────────────────────────────────────
# Pick whichever strategy beat B&H on Sharpe, or the best risk-adjusted
all_candidates = list(finalists.values())
winner = max(all_candidates, key=lambda x: x["sharpe"])
print(f"\n  Winner by Sharpe: {winner['label']}")

print(f"\n{'=' * 70}")
print(f"MONTE CARLO: {winner['label']} vs B&H SPY  (5000 sims × 252d)")
print(f"{'=' * 70}")

for method in ["bootstrap", "block_bootstrap"]:
    mc_win = run_monte_carlo(winner["returns"], n_simulations=5000, horizon=252,
                             method=method, seed=42)
    mc_bh  = run_monte_carlo(bah_rets, n_simulations=5000, horizon=252,
                             method=method, seed=42)

    print(f"\n  ── {method} ──")
    print(f"  {'':25s} {'Winner':>12s} {'B&H SPY':>12s} {'Delta':>12s}")
    print(f"  {'Median return':25s} {fmt_pct(mc_win.return_distribution.median):>12s} "
          f"{fmt_pct(mc_bh.return_distribution.median):>12s} "
          f"{fmt_pct(mc_win.return_distribution.median - mc_bh.return_distribution.median):>12s}")
    print(f"  {'P(loss)':25s} {mc_win.probability_of_loss*100:>11.1f}% "
          f"{mc_bh.probability_of_loss*100:>11.1f}%")

    w95 = next(v for v in mc_win.var_results if v.confidence_level == 0.95)
    b95 = next(v for v in mc_bh.var_results  if v.confidence_level == 0.95)
    print(f"  {'VaR 95%':25s} {fmt_pct(w95.var):>12s} {fmt_pct(b95.var):>12s}")
    print(f"  {'CVaR 95%':25s} {fmt_pct(w95.cvar):>12s} {fmt_pct(b95.cvar):>12s}")
    print(f"  {'Max-DD 95th pct':25s} {fmt_pct(mc_win.drawdown_distribution.percentile_95):>12s} "
          f"{fmt_pct(mc_bh.drawdown_distribution.percentile_95):>12s}")
    print(f"  {'Sharpe median':25s} {fmt_f(mc_win.sharpe_distribution.median):>12s} "
          f"{fmt_f(mc_bh.sharpe_distribution.median):>12s}")

print(f"\n{'=' * 70}")
print("  Done.")
print(f"{'=' * 70}")
