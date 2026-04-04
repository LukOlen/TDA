"""
Full Strategy Comparison
========================
B&H SPY  vs  TDA Adaptive (tuned)  vs  TDA ML Enhanced (7 configs)

Sections:
  1. Results table (all strategies + B&H)
  2. Improvement attribution (Sharpe/return deltas)
  3. Turnover analysis
  4. Risk analysis (drawdown, tail risk)
  5. Bootstrap confidence intervals (Sharpe)
  6. Rolling 1-year Sharpe comparison
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from backtester.data import load_ohlcv
from backtester.engine import BacktestEngine
from backtester.rotation import run_rotation_backtest
from backtester.strategies.tda_ml_enhanced import TDAMLEnhancedStrategy
from backtester.strategies.tda_adaptive import TDAAdaptiveStrategy
from backtester.strategies.adaptive import _AlwaysLong, _AlwaysFlat
from backtester.strategies.macd_momentum import MACDMomentum

# ── Config ────────────────────────────────────────────────────────────
TICKER = "SPY"
START, END = "2018-01-01", "2024-12-31"
CAPITAL = 100_000.0
COMM = 0.001
N_BOOTSTRAP = 2000
SEED = 42
W = 110  # print width


def fmt_pct(v):
    if v is None: return "    N/A"
    return f"{v * 100:+.2f}%"

def fmt_f(v, d=2):
    if v is None: return "  N/A"
    return f"{v:.{d}f}"

def hline(ch="═"):
    print(ch * W)

def section(title):
    print(f"\n{title}")
    hline("─")


# ══════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════
print("=" * W)
print("FULL STRATEGY COMPARISON")
hline()
print(f"  Period: {START} → {END}   Capital: ${CAPITAL:,.0f}   Commission: {COMM*100:.1f}%\n")

print("Loading market data ...")
spy_data = load_ohlcv(TICKER, start=START, end=END)
tlt_data = load_ohlcv("TLT", start=START, end=END)
gld_data = load_ohlcv("GLD", start=START, end=END)
alt_assets = {"TLT": tlt_data, "GLD": gld_data}
print(f"  SPY: {len(spy_data)} bars | TLT: {len(tlt_data)} bars | GLD: {len(gld_data)} bars\n")

# ── B&H baseline ─────────────────────────────────────────────────────
bah_rets = spy_data["close"].pct_change().fillna(0)
bah_equity = CAPITAL * (1 + bah_rets).cumprod()
bah_equity.iloc[0] = CAPITAL
bah_total = float(bah_equity.iloc[-1] / CAPITAL - 1)
bah_dd = float(((bah_equity / bah_equity.cummax()) - 1).min())
bah_sharpe = float(bah_rets.mean() / bah_rets.std() * np.sqrt(252))
n_years = (len(bah_rets) - 1) / 252
bah_cagr = float((bah_equity.iloc[-1] / CAPITAL) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0
bah_vol = float(bah_rets.std() * np.sqrt(252))
bah_sortino = float(bah_rets.mean() / bah_rets[bah_rets < 0].std() * np.sqrt(252)) if (bah_rets < 0).any() else 0.0
bah_calmar = bah_cagr / abs(bah_dd) if bah_dd != 0 else 0.0


# ── Helper to run a strategy ─────────────────────────────────────────
def run_strategy(label, strategy, use_rotation=False):
    print(f"  Running {label} ...", end=" ", flush=True)
    if use_rotation:
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

    # Turnover
    if hasattr(strategy, "generate_float_signals"):
        sigs = strategy.generate_float_signals(spy_data)
    else:
        sigs = strategy.generate_signals(spy_data).astype(float)
    positions = sigs.shift(1).fillna(0.0)
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
        "beta": m.get("beta"),
        "alpha": m.get("alpha"),
        "turnover": turnover,
        "returns": result.returns,
        "equity": result.equity_curve,
    }
    print(f"ret={fmt_pct(m['total_return'])}  sharpe={fmt_f(m['sharpe_ratio'])}")
    return entry


# ══════════════════════════════════════════════════════════════════════
# Run all strategies
# ══════════════════════════════════════════════════════════════════════
section("RUNNING STRATEGIES")

all_results = []

# --- TDA Adaptive (tuned baseline from previous work) ---
tuned_tda = TDAAdaptiveStrategy(
    regime_map={
        "bullish": _AlwaysLong(), "transition": _AlwaysLong(),
        "neutral": MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
        "bearish": _AlwaysFlat(),
    },
    drawdown_exit=-0.10, drawdown_lookback=80, recovery_sma=50,
    complexity_alert=0.6, tda_weight=0.4, rebalance_days=5,
)
all_results.append(run_strategy("TDA Adaptive (tuned)", tuned_tda))

# --- 7 ML Enhanced configs ---
ML_CONFIGS = [
    {"label": "ML Ridge (baseline)",    "model": "ridge", "smoothing": 0.0,  "rotation": False},
    {"label": "ML GBR",                 "model": "gbr",   "smoothing": 0.0,  "rotation": False},
    {"label": "ML Ridge + Smooth",      "model": "ridge", "smoothing": 0.15, "rotation": False},
    {"label": "ML GBR + Smooth",        "model": "gbr",   "smoothing": 0.15, "rotation": False},
    {"label": "ML Ridge + Rotation",    "model": "ridge", "smoothing": 0.0,  "rotation": True},
    {"label": "ML GBR + Rotation",      "model": "gbr",   "smoothing": 0.0,  "rotation": True},
    {"label": "ML GBR+Smooth+Rotation", "model": "gbr",   "smoothing": 0.15, "rotation": True},
]

for cfg in ML_CONFIGS:
    strat = TDAMLEnhancedStrategy(
        model_type=cfg["model"],
        signal_smoothing=cfg["smoothing"],
    )
    all_results.append(run_strategy(cfg["label"], strat, use_rotation=cfg["rotation"]))


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Results Table
# ══════════════════════════════════════════════════════════════════════
section("1. PERFORMANCE SUMMARY")

col_w = {
    "name": 28, "ret": 10, "cagr": 8, "sharpe": 7, "sortino": 8,
    "dd": 9, "calmar": 7, "vol": 7, "beta": 6, "alpha": 8, "trades": 6,
}
header = (
    f"  {'Strategy':<{col_w['name']}s} {'Return':>{col_w['ret']}s} {'CAGR':>{col_w['cagr']}s} "
    f"{'Sharpe':>{col_w['sharpe']}s} {'Sortino':>{col_w['sortino']}s} "
    f"{'MaxDD':>{col_w['dd']}s} {'Calmar':>{col_w['calmar']}s} "
    f"{'Vol':>{col_w['vol']}s} {'Beta':>{col_w['beta']}s} {'Alpha':>{col_w['alpha']}s} {'Trades':>{col_w['trades']}s}"
)
print(header)
sep = "  " + " ".join("─" * w for w in col_w.values())
print(sep)

# B&H row
print(
    f"  {'★ B&H SPY':<{col_w['name']}s} {fmt_pct(bah_total):>{col_w['ret']}s} "
    f"{fmt_pct(bah_cagr):>{col_w['cagr']}s} {fmt_f(bah_sharpe):>{col_w['sharpe']}s} "
    f"{fmt_f(bah_sortino):>{col_w['sortino']}s} {fmt_pct(bah_dd):>{col_w['dd']}s} "
    f"{fmt_f(bah_calmar):>{col_w['calmar']}s} {fmt_pct(bah_vol):>{col_w['vol']}s} "
    f"{'1.00':>{col_w['beta']}s} {'+0.00%':>{col_w['alpha']}s} {'—':>{col_w['trades']}s}"
)

for r in all_results:
    b = fmt_f(r["beta"]) if r["beta"] is not None else "  N/A"
    a = fmt_pct(r["alpha"]) if r["alpha"] is not None else "    N/A"
    print(
        f"  {r['label']:<{col_w['name']}s} {fmt_pct(r['total_return']):>{col_w['ret']}s} "
        f"{fmt_pct(r['cagr']):>{col_w['cagr']}s} {fmt_f(r['sharpe']):>{col_w['sharpe']}s} "
        f"{fmt_f(r['sortino']):>{col_w['sortino']}s} {fmt_pct(r['max_dd']):>{col_w['dd']}s} "
        f"{fmt_f(r['calmar']):>{col_w['calmar']}s} {fmt_pct(r['volatility']):>{col_w['vol']}s} "
        f"{b:>{col_w['beta']}s} {a:>{col_w['alpha']}s} {r['trades']:>{col_w['trades']}d}"
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Improvement Attribution
# ══════════════════════════════════════════════════════════════════════
section("2. IMPROVEMENT ATTRIBUTION vs B&H")

print(f"  {'Strategy':<28s} {'Sharpe Δ':>10s} {'Return Δ':>12s} {'MaxDD Δ':>10s}")
print(f"  {'─'*28} {'─'*10} {'─'*12} {'─'*10}")

for r in all_results:
    s_delta = (r["sharpe"] or 0) - bah_sharpe
    r_delta = (r["total_return"] or 0) - bah_total
    d_delta = (r["max_dd"] or 0) - bah_dd  # positive = less drawdown
    print(
        f"  {r['label']:<28s} {s_delta:>+10.4f} {fmt_pct(r_delta):>12s} {fmt_pct(d_delta):>10s}"
    )

# Component attribution (ML configs only)
ml_results = all_results[1:]  # skip TDA Adaptive
baseline_s = ml_results[0]["sharpe"] or 0.0

section("2b. COMPONENT ATTRIBUTION (vs ML Ridge baseline)")
improvements = {
    "GBR alone":          (ml_results[1]["sharpe"] or 0) - baseline_s,
    "Smoothing alone":    (ml_results[2]["sharpe"] or 0) - baseline_s,
    "Rotation alone":     (ml_results[4]["sharpe"] or 0) - baseline_s,
    "All Three combined": (ml_results[6]["sharpe"] or 0) - baseline_s,
}
sum_ind = improvements["GBR alone"] + improvements["Smoothing alone"] + improvements["Rotation alone"]
synergy = improvements["All Three combined"] - sum_ind

for name, delta in improvements.items():
    print(f"  {name:25s}  Sharpe Δ = {delta:+.4f}")
print(f"  {'Sum of individuals':25s}  Sharpe Δ = {sum_ind:+.4f}")
print(f"  {'Synergy/interference':25s}           = {synergy:+.4f}")
if synergy > 0.01:
    print("  → Positive synergy: improvements reinforce each other")
elif synergy < -0.01:
    print("  → Negative interference: combining hurts vs individual sum")
else:
    print("  → Roughly additive")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Turnover Analysis
# ══════════════════════════════════════════════════════════════════════
section("3. TURNOVER ANALYSIS")

print(f"  {'Strategy':<28s} {'Turnover':>10s} {'Trades':>8s} {'Turn/Trade':>10s}")
print(f"  {'─'*28} {'─'*10} {'─'*8} {'─'*10}")
for r in all_results:
    tpt = r["turnover"] / max(r["trades"], 1)
    print(f"  {r['label']:<28s} {r['turnover']:>10.1f} {r['trades']:>8d} {tpt:>10.2f}")

# Smoothing effect
no_smooth_ridge = ml_results[0]["turnover"]
with_smooth_ridge = ml_results[2]["turnover"]
no_smooth_gbr = ml_results[1]["turnover"]
with_smooth_gbr = ml_results[3]["turnover"]
if no_smooth_ridge > 0:
    print(f"\n  Smoothing reduces Ridge turnover: {no_smooth_ridge:.0f} → {with_smooth_ridge:.0f} "
          f"({(1 - with_smooth_ridge/no_smooth_ridge)*100:.1f}% reduction)")
if no_smooth_gbr > 0:
    print(f"  Smoothing reduces GBR turnover:   {no_smooth_gbr:.0f} → {with_smooth_gbr:.0f} "
          f"({(1 - with_smooth_gbr/no_smooth_gbr)*100:.1f}% reduction)")


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Risk Analysis
# ══════════════════════════════════════════════════════════════════════
section("4. RISK ANALYSIS")

# Worst periods: worst 5-day, worst 21-day, max consecutive losses
print(f"  {'Strategy':<28s} {'Worst 5d':>10s} {'Worst 21d':>10s} {'Max Consec':>10s} {'VaR 95%':>10s}")
print(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

def worst_window(rets, w):
    """Worst rolling w-day return."""
    rolling = (1 + rets).rolling(w).apply(lambda x: x.prod() - 1, raw=True)
    return float(rolling.min()) if not rolling.isna().all() else 0.0

def max_consec_losses(rets):
    """Maximum consecutive losing days."""
    is_loss = (rets < 0).astype(int)
    streaks = is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)
    return int(streaks.max()) if len(streaks) > 0 else 0

def var_95(rets):
    """Historical VaR at 95% confidence (daily)."""
    return float(np.percentile(rets, 5))

# B&H risk
print(
    f"  {'★ B&H SPY':<28s} {fmt_pct(worst_window(bah_rets, 5)):>10s} "
    f"{fmt_pct(worst_window(bah_rets, 21)):>10s} "
    f"{max_consec_losses(bah_rets):>10d} "
    f"{fmt_pct(var_95(bah_rets.values)):>10s}"
)

for r in all_results:
    rets = r["returns"]
    print(
        f"  {r['label']:<28s} {fmt_pct(worst_window(rets, 5)):>10s} "
        f"{fmt_pct(worst_window(rets, 21)):>10s} "
        f"{max_consec_losses(rets):>10d} "
        f"{fmt_pct(var_95(rets.values)):>10s}"
    )


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Bootstrap Confidence Intervals
# ══════════════════════════════════════════════════════════════════════
section(f"5. BOOTSTRAP CONFIDENCE INTERVALS (Sharpe, {N_BOOTSTRAP} resamples)")

def bootstrap_sharpe(returns, n_iter, rng):
    n = len(returns)
    sharpes = np.empty(n_iter)
    for k in range(n_iter):
        sample = returns[rng.integers(0, n, size=n)]
        std = sample.std(ddof=1)
        sharpes[k] = (sample.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return sharpes

rng = np.random.default_rng(SEED)

# Key strategies to bootstrap
boot_targets = [
    ("B&H SPY", bah_rets.values),
    (all_results[0]["label"], all_results[0]["returns"].values),  # TDA Adaptive
    (ml_results[0]["label"], ml_results[0]["returns"].values),    # ML Ridge
    (ml_results[1]["label"], ml_results[1]["returns"].values),    # ML GBR
    (ml_results[3]["label"], ml_results[3]["returns"].values),    # ML GBR+Smooth
    (ml_results[6]["label"], ml_results[6]["returns"].values),    # All Three
]

print(f"  {'Strategy':<28s} {'Median':>8s} {'5th pct':>9s} {'95th pct':>9s} {'Width':>8s}")
print(f"  {'─'*28} {'─'*8} {'─'*9} {'─'*9} {'─'*8}")

for name, rets in boot_targets:
    sharpes = bootstrap_sharpe(rets, N_BOOTSTRAP, rng)
    p5, p50, p95 = np.percentile(sharpes, [5, 50, 95])
    width = p95 - p5
    print(f"  {name:<28s} {p50:>8.3f} {p5:>9.3f} {p95:>9.3f} {width:>8.3f}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: Rolling 1-Year Sharpe
# ══════════════════════════════════════════════════════════════════════
section("6. ROLLING 1-YEAR SHARPE (252-day window)")

rolling_targets = [
    ("B&H SPY", bah_rets),
    (all_results[0]["label"], all_results[0]["returns"]),  # TDA Adaptive
    (ml_results[6]["label"], ml_results[6]["returns"]),    # All Three
]

# Compute rolling Sharpe for each year-end
years = sorted(set(spy_data.index.year))
print(f"  {'Year':>6s}", end="")
for name, _ in rolling_targets:
    print(f"  {name[:20]:>20s}", end="")
print()
print(f"  {'─'*6}", end="")
for _ in rolling_targets:
    print(f"  {'─'*20}", end="")
print()

for year in years:
    year_mask_end = spy_data.index.year == year
    if not year_mask_end.any():
        continue
    last_day = spy_data.index[year_mask_end][-1]
    # Get 252 trading days ending at last_day
    loc = spy_data.index.get_loc(last_day)
    start_loc = max(0, loc - 251)

    print(f"  {year:>6d}", end="")
    for name, rets in rolling_targets:
        window_rets = rets.iloc[start_loc:loc+1]
        if len(window_rets) < 50:
            print(f"  {'—':>20s}", end="")
            continue
        std = window_rets.std(ddof=1)
        if std > 0:
            rs = float(window_rets.mean() / std * np.sqrt(252))
        else:
            rs = 0.0
        print(f"  {rs:>20.3f}", end="")
    print()


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: Final Verdict
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * W}")
print("FINAL VERDICT")
hline()

# Find best strategy by Sharpe
best_by_sharpe = max(all_results, key=lambda r: r["sharpe"] or -999)
best_by_return = max(all_results, key=lambda r: r["total_return"] or -999)
best_by_calmar = max(all_results, key=lambda r: r["calmar"] or -999)
least_dd = max(all_results, key=lambda r: r["max_dd"] or -999)  # closest to 0

print(f"  Best Sharpe:        {best_by_sharpe['label']:<28s}  ({fmt_f(best_by_sharpe['sharpe'])})")
print(f"  Best Total Return:  {best_by_return['label']:<28s}  ({fmt_pct(best_by_return['total_return'])})")
print(f"  Best Calmar:        {best_by_calmar['label']:<28s}  ({fmt_f(best_by_calmar['calmar'])})")
print(f"  Shallowest MaxDD:   {least_dd['label']:<28s}  ({fmt_pct(least_dd['max_dd'])})")

# Compare best ML vs TDA Adaptive vs B&H
tda_adap = all_results[0]
print(f"\n  Key comparisons (Sharpe / Return / MaxDD):")
print(f"    B&H SPY:               {fmt_f(bah_sharpe)} / {fmt_pct(bah_total)} / {fmt_pct(bah_dd)}")
print(f"    TDA Adaptive (tuned):  {fmt_f(tda_adap['sharpe'])} / {fmt_pct(tda_adap['total_return'])} / {fmt_pct(tda_adap['max_dd'])}")
print(f"    Best ML config:        {fmt_f(best_by_sharpe['sharpe'])} / {fmt_pct(best_by_sharpe['total_return'])} / {fmt_pct(best_by_sharpe['max_dd'])}")
print(f"                           ({best_by_sharpe['label']})")

print(f"\n{'═' * W}")
print("Done.")
print(f"{'═' * W}")
