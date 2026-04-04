"""
Tier 1 Improvements Comparison Script
--------------------------------------
Compares 7 configurations of the TDA ML Enhanced strategy to measure
the individual and combined impact of:
  - GBR (vs Ridge baseline)
  - Signal smoothing
  - Multi-asset rotation (TLT, GLD)

Outputs: results table, improvement attribution, turnover analysis,
and bootstrap confidence intervals for Sharpe ratio.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.rotation import run_rotation_backtest
from backtester.strategies.tda_ml_enhanced import TDAMLEnhancedStrategy

# ── Config ───────────────────────────────────────────────────────────
TICKER = "SPY"
START, END = "2018-01-01", "2024-12-31"
CAPITAL = 100_000.0
COMM = 0.001
N_BOOTSTRAP = 1000
SEED = 42


def fmt_pct(v):
    if v is None:
        return "   N/A"
    return f"{v * 100:+.2f}%"


def fmt_f(v, d=2):
    if v is None:
        return "  N/A"
    return f"{v:.{d}f}"


# ── Load data ────────────────────────────────────────────────────────
print(f"Loading {TICKER} {START} → {END} ...")
spy_data = load_ohlcv(TICKER, start=START, end=END)
print(f"  SPY: {len(spy_data)} bars")

print("Loading TLT, GLD ...")
tlt_data = load_ohlcv("TLT", start=START, end=END)
gld_data = load_ohlcv("GLD", start=START, end=END)
print(f"  TLT: {len(tlt_data)} bars, GLD: {len(gld_data)} bars\n")

alt_assets = {"TLT": tlt_data, "GLD": gld_data}


# ── Define configurations ────────────────────────────────────────────
CONFIGS = [
    {"label": "1. Baseline (Ridge)",        "model": "ridge", "smoothing": 0.0,  "rotation": False},
    {"label": "2. + GBR",                   "model": "gbr",   "smoothing": 0.0,  "rotation": False},
    {"label": "3. + Smoothing",             "model": "ridge", "smoothing": 0.15, "rotation": False},
    {"label": "4. + GBR + Smoothing",       "model": "gbr",   "smoothing": 0.15, "rotation": False},
    {"label": "5. + Rotation",              "model": "ridge", "smoothing": 0.0,  "rotation": True},
    {"label": "6. + GBR + Rotation",        "model": "gbr",   "smoothing": 0.0,  "rotation": True},
    {"label": "7. All Three",               "model": "gbr",   "smoothing": 0.15, "rotation": True},
]


# ── Run all configurations ───────────────────────────────────────────
results = []

for cfg in CONFIGS:
    label = cfg["label"]
    print(f"Running {label} ...", end=" ", flush=True)

    strategy = TDAMLEnhancedStrategy(
        model_type=cfg["model"],
        signal_smoothing=cfg["smoothing"],
    )

    if cfg["rotation"]:
        result = run_rotation_backtest(
            spy_data, alt_assets, strategy,
            initial_capital=CAPITAL, commission=COMM, ticker=TICKER,
        )
    else:
        engine = BacktestEngine(
            data=spy_data, strategy=strategy,
            initial_capital=CAPITAL, commission=COMM, ticker=TICKER,
        )
        result = engine.run()

    m = result.metrics

    # Compute turnover: sum of absolute position changes
    signals = strategy.generate_float_signals(spy_data)
    positions = signals.shift(1).fillna(0.0)
    turnover = float(positions.diff().abs().sum())

    entry = {
        "label": label,
        "total_return": m["total_return"],
        "cagr": m["cagr"],
        "sharpe": m["sharpe_ratio"],
        "sortino": m["sortino_ratio"],
        "max_dd": m["max_drawdown"],
        "calmar": m["calmar_ratio"],
        "volatility": m["volatility"],
        "trades": m["total_trades"],
        "turnover": turnover,
        "returns": result.returns,
    }
    results.append(entry)
    print(f"ret={fmt_pct(m['total_return'])}  sharpe={fmt_f(m['sharpe_ratio'])}")

# B&H benchmark
bah_rets = spy_data["close"].pct_change().fillna(0)
bah_total = float((1 + bah_rets).prod() - 1)
bah_equity = CAPITAL * (1 + bah_rets).cumprod()
bah_dd = float(((bah_equity / bah_equity.cummax()) - 1).min())
bah_sharpe = float(bah_rets.mean() / bah_rets.std() * np.sqrt(252))
n_years = (len(bah_rets) - 1) / 252
bah_cagr = float((bah_equity.iloc[-1] / CAPITAL) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0
bah_vol = float(bah_rets.std() * np.sqrt(252))


# ══════════════════════════════════════════════════════════════════════
# 1. Results Table
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 110}")
print("RESULTS TABLE")
print(f"{'=' * 110}")

header = (
    f"  {'Configuration':30s} {'Return':>9s} {'CAGR':>8s} {'Sharpe':>7s} "
    f"{'Sortino':>8s} {'MaxDD':>9s} {'Calmar':>7s} {'Vol':>7s} {'Trades':>6s}"
)
print(header)
print(f"  {'─' * 30} {'─' * 9} {'─' * 8} {'─' * 7} {'─' * 8} {'─' * 9} {'─' * 7} {'─' * 7} {'─' * 6}")

# B&H row
print(
    f"  {'B&H SPY':30s} {fmt_pct(bah_total):>9s} {fmt_pct(bah_cagr):>8s} "
    f"{fmt_f(bah_sharpe):>7s} {'':>8s} {fmt_pct(bah_dd):>9s} {'':>7s} "
    f"{fmt_pct(bah_vol):>7s} {'':>6s}"
)

for r in results:
    print(
        f"  {r['label']:30s} {fmt_pct(r['total_return']):>9s} {fmt_pct(r['cagr']):>8s} "
        f"{fmt_f(r['sharpe']):>7s} {fmt_f(r['sortino']):>8s} "
        f"{fmt_pct(r['max_dd']):>9s} {fmt_f(r['calmar']):>7s} "
        f"{fmt_pct(r['volatility']):>7s} {r['trades']:>6d}"
    )


# ══════════════════════════════════════════════════════════════════════
# 2. Improvement Attribution
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("IMPROVEMENT ATTRIBUTION (Sharpe delta from Baseline)")
print(f"{'=' * 80}")

baseline_sharpe = results[0]["sharpe"] or 0.0
improvements = {
    "GBR alone":          (results[1]["sharpe"] or 0.0) - baseline_sharpe,
    "Smoothing alone":    (results[2]["sharpe"] or 0.0) - baseline_sharpe,
    "Rotation alone":     (results[4]["sharpe"] or 0.0) - baseline_sharpe,
    "All Three combined": (results[6]["sharpe"] or 0.0) - baseline_sharpe,
}
sum_individual = improvements["GBR alone"] + improvements["Smoothing alone"] + improvements["Rotation alone"]
synergy = improvements["All Three combined"] - sum_individual

for name, delta in improvements.items():
    print(f"  {name:25s}  Sharpe Δ = {delta:+.4f}")
print(f"  {'Sum of individuals':25s}  Sharpe Δ = {sum_individual:+.4f}")
print(f"  {'Synergy/interference':25s}           = {synergy:+.4f}")
if synergy > 0:
    print("  → Positive synergy: improvements reinforce each other")
elif synergy < -0.01:
    print("  → Negative interference: combining hurts vs individual sum")
else:
    print("  → Roughly additive: no significant synergy or interference")


# ══════════════════════════════════════════════════════════════════════
# 3. Turnover Analysis
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("TURNOVER ANALYSIS")
print(f"{'=' * 60}")

for r in results:
    print(f"  {r['label']:30s}  turnover = {r['turnover']:8.1f}")

# Show smoothing reduces turnover
no_smooth = results[0]["turnover"]
with_smooth = results[2]["turnover"]
if no_smooth > 0:
    reduction = (1 - with_smooth / no_smooth) * 100
    print(f"\n  Smoothing reduces turnover by {reduction:.1f}% (Ridge: {no_smooth:.0f} → {with_smooth:.0f})")


# ══════════════════════════════════════════════════════════════════════
# 4. Bootstrap Confidence Intervals for Sharpe
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"BOOTSTRAP CONFIDENCE INTERVALS (Sharpe, {N_BOOTSTRAP} resamples)")
print(f"{'=' * 70}")


def bootstrap_sharpe(returns: np.ndarray, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Resample daily returns and compute Sharpe for each sample."""
    n = len(returns)
    sharpes = np.empty(n_iter)
    for k in range(n_iter):
        sample = returns[rng.integers(0, n, size=n)]
        std = sample.std(ddof=1)
        if std > 0:
            sharpes[k] = sample.mean() / std * np.sqrt(252)
        else:
            sharpes[k] = 0.0
    return sharpes


rng = np.random.default_rng(SEED)

# Bootstrap B&H, Baseline (#1), All Three (#7)
targets = [
    ("B&H SPY", bah_rets.values),
    (results[0]["label"], results[0]["returns"].values),
    (results[6]["label"], results[6]["returns"].values),
]

print(f"  {'Strategy':30s} {'Median':>8s} {'5th pct':>9s} {'95th pct':>9s}")
print(f"  {'─' * 30} {'─' * 8} {'─' * 9} {'─' * 9}")

for name, rets in targets:
    sharpes = bootstrap_sharpe(rets, N_BOOTSTRAP, rng)
    p5, p50, p95 = np.percentile(sharpes, [5, 50, 95])
    print(f"  {name:30s} {p50:>8.3f} {p5:>9.3f} {p95:>9.3f}")


print(f"\n{'=' * 70}")
print("Done.")
print(f"{'=' * 70}")
