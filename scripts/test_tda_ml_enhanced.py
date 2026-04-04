"""
Test TDA ML Enhanced v2: higher returns via asymmetric vol targeting.
Key insight: lever up in calm bulls, protect in chaos. Don't fight bullish regime.
"""
from __future__ import annotations

import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.strategies.tda_adaptive import TDAAdaptiveStrategy
from backtester.strategies.tda_ml_enhanced import TDAMLEnhancedStrategy
from backtester.strategies.adaptive import _AlwaysLong, _AlwaysFlat
from backtester.strategies.macd_momentum import MACDMomentum
from backtester.monte_carlo import run_monte_carlo

TICKER, START, END = "SPY", "2018-01-01", "2024-12-31"
CAPITAL, COMM = 100_000.0, 0.001

def fmt_pct(v): return f"{v*100:+.2f}%"
def fmt_f(v, d=2): return f"{v:.{d}f}"

print(f"Loading {TICKER} {START} → {END} …")
data = load_ohlcv(TICKER, start=START, end=END)
print(f"  {len(data)} bars\n")

bah_rets = data["close"].pct_change().fillna(0)
bah_total = float((1 + bah_rets).prod() - 1)
bah_equity = CAPITAL * (1 + bah_rets).cumprod()
bah_dd = float(((bah_equity / bah_equity.cummax()) - 1).min())
bah_sharpe = float(bah_rets.mean() / bah_rets.std() * np.sqrt(252))
print(f"B&H: ret={fmt_pct(bah_total)}  sharpe={fmt_f(bah_sharpe)}  dd={fmt_pct(bah_dd)}\n")


def run_one(label, strategy):
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
        "calmar": m["calmar_ratio"],
        "volatility": m["volatility"],
        "trades": m["total_trades"],
        "beta": m["beta"],
        "alpha": m["alpha"],
        "returns": r.returns,
    }


# ── Baseline ──────────────────────────────────────────────────────────
print("=" * 72)
print("Baseline: Tuned TDA Adaptive")
print("=" * 72)
tuned_tda = TDAAdaptiveStrategy(
    regime_map={
        "bullish": _AlwaysLong(), "transition": _AlwaysLong(),
        "neutral": MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
        "bearish": _AlwaysFlat(),
    },
    drawdown_exit=-0.10, drawdown_lookback=80, recovery_sma=50,
    complexity_alert=0.6, tda_weight=0.4, rebalance_days=5,
)
baseline = run_one("TDA Tuned", tuned_tda)
print(f"  ret={fmt_pct(baseline['total_return'])}  sharpe={fmt_f(baseline['sharpe'])}  "
      f"dd={fmt_pct(baseline['max_dd'])}  calmar={fmt_f(baseline['calmar'])}\n")


# ══════════════════════════════════════════════════════════════════════
# Sweep: focus on vol targeting + leverage (the return multiplier)
# ══════════════════════════════════════════════════════════════════════
print("=" * 72)
print("ML Enhanced v2: parameter sweep")
print("=" * 72)

target_vols     = [0.15, 0.18, 0.20, 0.22, 0.25]
max_leverages   = [1.2, 1.3, 1.5]
vol_floors      = [0.5, 0.65, 0.8]
bullish_floors  = [0.85, 0.90, 0.95]
ridge_alphas    = [10.0, 20.0]

results = []
total = (len(target_vols) * len(max_leverages) * len(vol_floors)
         * len(bullish_floors) * len(ridge_alphas))
print(f"  Testing {total} configurations …\n")

count = 0
for tv, ml, vf, bf, ra in itertools.product(
    target_vols, max_leverages, vol_floors, bullish_floors, ridge_alphas,
):
    count += 1
    strat = TDAMLEnhancedStrategy(
        target_vol=tv, max_leverage=ml, vol_floor=vf,
        bullish_floor=bf, ridge_alpha=ra,
    )
    try:
        r = run_one(f"tv={tv}/ml={ml}/vf={vf}/bf={bf}/ra={ra}", strat)
        results.append(r | {"tv": tv, "ml": ml, "vf": vf, "bf": bf, "ra": ra})
    except Exception as e:
        print(f"  FAILED: {e}")
    if count % 50 == 0:
        print(f"  … {count}/{total} done")

print(f"  {len(results)} configs completed\n")


# ── Filter: must beat B&H Sharpe (0.76) and have max_dd > -25% ──────
good = [r for r in results if r["sharpe"] > bah_sharpe and r["max_dd"] > -0.25]
good.sort(key=lambda x: x["total_return"], reverse=True)

print(f"  {len(good)} configs beat B&H Sharpe with max_dd > -25%")
print(f"\n  Top 15 (sorted by return, constrained sharpe>{fmt_f(bah_sharpe)} & dd>-25%):")
print(f"  {'#':>3s}  {'tv':>5s} {'lev':>4s} {'vf':>4s} {'bf':>4s} {'ra':>5s}  "
      f"{'Return':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>9s} {'Calmar':>7s} {'Vol':>8s}")
print(f"  {'─'*3}  {'─'*5} {'─'*4} {'─'*4} {'─'*4} {'─'*5}  "
      f"{'─'*10} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8}")
for i, r in enumerate(good[:15]):
    print(f"  {i+1:3d}  {r['tv']:5.2f} {r['ml']:4.1f} {r['vf']:4.1f} {r['bf']:4.2f} {r['ra']:5.0f}  "
          f"{fmt_pct(r['total_return']):>10s} {fmt_f(r['sharpe']):>7s} {fmt_f(r['sortino']):>8s} "
          f"{fmt_pct(r['max_dd']):>9s} {fmt_f(r['calmar']):>7s} {fmt_pct(r['volatility']):>8s}")


# ── Also show best by risk-adjusted return (Calmar) ──────────────────
good_calmar = sorted(good, key=lambda x: x["calmar"] if x["calmar"] else 0, reverse=True)
print(f"\n  Top 5 by Calmar (among filtered):")
for i, r in enumerate(good_calmar[:5]):
    print(f"  {i+1:3d}  tv={r['tv']:.2f} lev={r['ml']:.1f} vf={r['vf']:.1f} bf={r['bf']:.2f} ra={r['ra']:.0f}  "
          f"ret={fmt_pct(r['total_return'])} sharpe={fmt_f(r['sharpe'])} dd={fmt_pct(r['max_dd'])} calmar={fmt_f(r['calmar'])}")


# ══════════════════════════════════════════════════════════════════════
# Pick the best: highest return among configs with Sharpe > B&H and dd > -22%
# ══════════════════════════════════════════════════════════════════════
strict = [r for r in results if r["sharpe"] > 0.85 and r["max_dd"] > -0.22]
if strict:
    strict.sort(key=lambda x: x["total_return"], reverse=True)
    best = strict[0]
else:
    best = good[0] if good else results[0]

print(f"\n{'=' * 72}")
print("FINAL SHOOTOUT")
print("=" * 72)

best_strat = TDAMLEnhancedStrategy(
    target_vol=best["tv"], max_leverage=best["ml"], vol_floor=best["vf"],
    bullish_floor=best["bf"], ridge_alpha=best["ra"],
)
best_result = run_one("ML Enhanced v2", best_strat)

print(f"\n  {'Strategy':30s} {'Return':>10s} {'CAGR':>8s} {'Sharpe':>7s} {'Sortino':>8s} "
      f"{'MaxDD':>9s} {'Calmar':>7s} {'Vol':>8s} {'Beta':>6s} {'Alpha':>8s}")
print(f"  {'─'*30} {'─'*10} {'─'*8} {'─'*7} {'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6} {'─'*8}")

print(f"  {'B&H SPY':30s} {fmt_pct(bah_total):>10s} {'':>8s} {fmt_f(bah_sharpe):>7s} "
      f"{'':>8s} {fmt_pct(bah_dd):>9s} {'':>7s} {'':>8s} {'1.00':>6s} {'':>8s}")

for name, r in [("TDA Tuned (prev best)", baseline), ("TDA ML Enhanced v2", best_result)]:
    print(f"  {name:30s} {fmt_pct(r['total_return']):>10s} {fmt_pct(r['cagr']):>8s} "
          f"{fmt_f(r['sharpe']):>7s} {fmt_f(r['sortino']):>8s} "
          f"{fmt_pct(r['max_dd']):>9s} {fmt_f(r['calmar']):>7s} "
          f"{fmt_pct(r['volatility']):>8s} {fmt_f(r['beta'] or 0):>6s} "
          f"{fmt_pct(r['alpha'] or 0):>8s}")

print(f"\n  Best params: target_vol={best['tv']}, max_leverage={best['ml']}, "
      f"vol_floor={best['vf']},")
print(f"               bullish_floor={best['bf']}, ridge_alpha={best['ra']}")


# ══════════════════════════════════════════════════════════════════════
# Monte Carlo
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print(f"MONTE CARLO: ML Enhanced v2 vs TDA Tuned vs B&H  (5000 sims × 252d)")
print(f"{'=' * 72}")

for method in ["bootstrap", "block_bootstrap"]:
    mc_ml  = run_monte_carlo(best_result["returns"], n_simulations=5000,
                             horizon=252, method=method, seed=42)
    mc_tda = run_monte_carlo(baseline["returns"], n_simulations=5000,
                             horizon=252, method=method, seed=42)
    mc_bh  = run_monte_carlo(bah_rets, n_simulations=5000,
                             horizon=252, method=method, seed=42)

    print(f"\n  ── {method} ──")
    print(f"  {'':30s} {'ML Enhanced':>12s} {'TDA Tuned':>12s} {'B&H SPY':>12s}")
    print(f"  {'Median return':30s} "
          f"{fmt_pct(mc_ml.return_distribution.median):>12s} "
          f"{fmt_pct(mc_tda.return_distribution.median):>12s} "
          f"{fmt_pct(mc_bh.return_distribution.median):>12s}")
    print(f"  {'Mean return':30s} "
          f"{fmt_pct(mc_ml.return_distribution.mean):>12s} "
          f"{fmt_pct(mc_tda.return_distribution.mean):>12s} "
          f"{fmt_pct(mc_bh.return_distribution.mean):>12s}")
    print(f"  {'P(loss)':30s} "
          f"{mc_ml.probability_of_loss*100:>11.1f}% "
          f"{mc_tda.probability_of_loss*100:>11.1f}% "
          f"{mc_bh.probability_of_loss*100:>11.1f}%")

    ml95  = next(v for v in mc_ml.var_results  if v.confidence_level == 0.95)
    tda95 = next(v for v in mc_tda.var_results if v.confidence_level == 0.95)
    bh95  = next(v for v in mc_bh.var_results  if v.confidence_level == 0.95)
    ml99  = next(v for v in mc_ml.var_results  if v.confidence_level == 0.99)
    bh99  = next(v for v in mc_bh.var_results  if v.confidence_level == 0.99)
    print(f"  {'VaR 95%':30s} "
          f"{fmt_pct(ml95.var):>12s} {fmt_pct(tda95.var):>12s} {fmt_pct(bh95.var):>12s}")
    print(f"  {'CVaR 95%':30s} "
          f"{fmt_pct(ml95.cvar):>12s} {fmt_pct(tda95.cvar):>12s} {fmt_pct(bh95.cvar):>12s}")
    print(f"  {'VaR 99%':30s} "
          f"{fmt_pct(ml99.var):>12s} {'':>12s} {fmt_pct(bh99.var):>12s}")
    print(f"  {'Max-DD median':30s} "
          f"{fmt_pct(mc_ml.drawdown_distribution.median):>12s} "
          f"{fmt_pct(mc_tda.drawdown_distribution.median):>12s} "
          f"{fmt_pct(mc_bh.drawdown_distribution.median):>12s}")
    print(f"  {'Max-DD 95th pct':30s} "
          f"{fmt_pct(mc_ml.drawdown_distribution.percentile_95):>12s} "
          f"{fmt_pct(mc_tda.drawdown_distribution.percentile_95):>12s} "
          f"{fmt_pct(mc_bh.drawdown_distribution.percentile_95):>12s}")
    print(f"  {'Sharpe median':30s} "
          f"{fmt_f(mc_ml.sharpe_distribution.median):>12s} "
          f"{fmt_f(mc_tda.sharpe_distribution.median):>12s} "
          f"{fmt_f(mc_bh.sharpe_distribution.median):>12s}")

print(f"\n{'=' * 72}")
print("  Done.")
print(f"{'=' * 72}")
