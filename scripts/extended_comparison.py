"""
Extended Strategy Comparison — Multi-Period, Multi-Asset
=========================================================
Tests the full strategy suite across:
  - 20-year backtest (2005–2024): SPY, QQQ, IWM
  - Sub-period breakdown: GFC (2007–2009), Recovery (2010–2015),
    Bull (2016–2019), COVID+After (2020–2024)
  - Out-of-sample split: train 2005–2017, test 2018–2024
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
CAPITAL = 100_000.0
COMM = 0.001
N_BOOTSTRAP = 2000
SEED = 42
W = 120

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


# ── Strategy factory ─────────────────────────────────────────────────
def make_tda_adaptive():
    return TDAAdaptiveStrategy(
        regime_map={
            "bullish": _AlwaysLong(), "transition": _AlwaysLong(),
            "neutral": MACDMomentum(fast_period=8, slow_period=21, signal_period=5),
            "bearish": _AlwaysFlat(),
        },
        drawdown_exit=-0.10, drawdown_lookback=80, recovery_sma=50,
        complexity_alert=0.6, tda_weight=0.4, rebalance_days=5,
    )

STRATEGY_DEFS = [
    ("B&H",                     None,                                                           False),
    ("TDA Adaptive",            lambda: make_tda_adaptive(),                                    False),
    ("ML Ridge",                lambda: TDAMLEnhancedStrategy(model_type="ridge", signal_smoothing=0.0),  False),
    ("ML GBR",                  lambda: TDAMLEnhancedStrategy(model_type="gbr",   signal_smoothing=0.0),  False),
    ("ML GBR+Smooth",           lambda: TDAMLEnhancedStrategy(model_type="gbr",   signal_smoothing=0.15), False),
    ("ML Ridge+Rotation",       lambda: TDAMLEnhancedStrategy(model_type="ridge", signal_smoothing=0.0),  True),
    ("ML GBR+Rotation",         lambda: TDAMLEnhancedStrategy(model_type="gbr",   signal_smoothing=0.0),  True),
    ("ML GBR+Smooth+Rot",       lambda: TDAMLEnhancedStrategy(model_type="gbr",   signal_smoothing=0.15), True),
]


def run_backtest(primary_data, alt_assets, label, strat_factory, use_rotation,
                 capital=CAPITAL, comm=COMM, ticker="SPY"):
    """Run one strategy configuration, return metrics dict."""
    if strat_factory is None:
        # B&H
        rets = primary_data["close"].pct_change().fillna(0)
        equity = capital * (1 + rets).cumprod()
        equity.iloc[0] = capital
        total = float(equity.iloc[-1] / capital - 1)
        dd = float(((equity / equity.cummax()) - 1).min())
        std = rets.std(ddof=1)
        sharpe = float(rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        n_y = (len(rets) - 1) / 252
        cagr_v = float((equity.iloc[-1] / capital) ** (1.0 / n_y) - 1) if n_y > 0 else 0.0
        vol = float(std * np.sqrt(252))
        down = rets[rets < 0]
        sortino = float(rets.mean() / down.std(ddof=1) * np.sqrt(252)) if len(down) > 0 and down.std() > 0 else 0.0
        calmar = cagr_v / abs(dd) if dd != 0 else 0.0
        return {
            "label": label, "total_return": total, "cagr": cagr_v,
            "sharpe": sharpe, "sortino": sortino, "max_dd": dd,
            "calmar": calmar, "volatility": vol, "trades": 0,
            "returns": rets, "equity": equity,
        }

    strategy = strat_factory()

    if use_rotation and alt_assets:
        result = run_rotation_backtest(
            primary_data, alt_assets, strategy,
            initial_capital=capital, commission=comm, ticker=ticker,
        )
    else:
        engine = BacktestEngine(
            data=primary_data, strategy=strategy,
            initial_capital=capital, commission=comm, ticker=ticker,
        )
        result = engine.run()

    m = result.metrics
    return {
        "label": label,
        "total_return": m["total_return"], "cagr": m["cagr"],
        "sharpe": m["sharpe_ratio"], "sortino": m["sortino_ratio"],
        "max_dd": m["max_drawdown"], "calmar": m["calmar_ratio"],
        "volatility": m["volatility"], "trades": m["total_trades"],
        "returns": result.returns, "equity": result.equity_curve,
    }


def print_results_table(results_list, title=""):
    if title:
        section(title)
    print(
        f"  {'Strategy':<22s} {'Return':>10s} {'CAGR':>8s} {'Sharpe':>7s} "
        f"{'Sortino':>8s} {'MaxDD':>9s} {'Calmar':>7s} {'Vol':>8s} {'Trades':>6s}"
    )
    print(
        f"  {'─'*22} {'─'*10} {'─'*8} {'─'*7} "
        f"{'─'*8} {'─'*9} {'─'*7} {'─'*8} {'─'*6}"
    )
    for r in results_list:
        t = f"{r['trades']:>6d}" if r['trades'] > 0 else f"{'—':>6s}"
        print(
            f"  {r['label']:<22s} {fmt_pct(r['total_return']):>10s} {fmt_pct(r['cagr']):>8s} "
            f"{fmt_f(r['sharpe']):>7s} {fmt_f(r['sortino']):>8s} "
            f"{fmt_pct(r['max_dd']):>9s} {fmt_f(r['calmar']):>7s} "
            f"{fmt_pct(r['volatility']):>8s} {t}"
        )


def bootstrap_sharpe(returns, n_iter, rng):
    n = len(returns)
    sharpes = np.empty(n_iter)
    for k in range(n_iter):
        sample = returns[rng.integers(0, n, size=n)]
        std = sample.std(ddof=1)
        sharpes[k] = (sample.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return sharpes


# ══════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════
print("=" * W)
print("EXTENDED STRATEGY COMPARISON — Multi-Period, Multi-Asset")
hline()
print(f"  Capital: ${CAPITAL:,.0f}   Commission: {COMM*100:.1f}%\n")

FULL_START, FULL_END = "2005-01-01", "2024-12-31"

print("Loading market data ...")
data_cache = {}
for tkr in ["SPY", "QQQ", "IWM", "TLT", "GLD"]:
    print(f"  {tkr} ...", end=" ", flush=True)
    df = load_ohlcv(tkr, start=FULL_START, end=FULL_END)
    data_cache[tkr] = df
    print(f"{len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")

print()


def slice_data(start, end):
    """Slice all cached data to a date range."""
    sliced = {}
    for tkr, df in data_cache.items():
        s = df.loc[start:end]
        if len(s) > 0:
            sliced[tkr] = s
    return sliced


def run_all_strategies(primary_ticker, start, end, label_prefix=""):
    """Run all strategies on a given ticker and date range."""
    sliced = slice_data(start, end)
    primary = sliced[primary_ticker]
    alts = {k: v for k, v in sliced.items() if k in ("TLT", "GLD")}

    results = []
    for label, factory, rotation in STRATEGY_DEFS:
        full_label = f"{label_prefix}{label}" if label_prefix else label
        try:
            r = run_backtest(primary, alts, full_label, factory, rotation,
                             ticker=primary_ticker)
            results.append(r)
        except Exception as e:
            print(f"    SKIP {full_label}: {e}")
    return results


# ══════════════════════════════════════════════════════════════════════
# PART 1: FULL 20-YEAR BACKTEST (SPY)
# ══════════════════════════════════════════════════════════════════════
hline()
print("PART 1: FULL 20-YEAR BACKTEST — SPY (2005–2024)")
hline()

print("\n  Running strategies on SPY 2005-2024 ...")
full_spy = run_all_strategies("SPY", FULL_START, FULL_END)
print_results_table(full_spy, "SPY 2005–2024 Results")


# ══════════════════════════════════════════════════════════════════════
# PART 2: MULTI-ASSET (QQQ, IWM)
# ══════════════════════════════════════════════════════════════════════
hline()
print("PART 2: MULTI-ASSET TEST (2005–2024)")
hline()

for tkr in ["QQQ", "IWM"]:
    print(f"\n  Running strategies on {tkr} 2005-2024 ...")
    r = run_all_strategies(tkr, FULL_START, FULL_END)
    print_results_table(r, f"{tkr} 2005–2024 Results")


# ══════════════════════════════════════════════════════════════════════
# PART 3: SUB-PERIOD BREAKDOWN (SPY)
# ══════════════════════════════════════════════════════════════════════
hline()
print("PART 3: SUB-PERIOD BREAKDOWN — SPY")
hline()

PERIODS = [
    ("GFC (2007–2009)",     "2007-01-01", "2009-12-31"),
    ("Recovery (2010–2015)","2010-01-01", "2015-12-31"),
    ("Bull (2016–2019)",    "2016-01-01", "2019-12-31"),
    ("COVID+ (2020–2024)",  "2020-01-01", "2024-12-31"),
]

# Compact sub-period table: only key strategies
KEY_STRATS = ["B&H", "TDA Adaptive", "ML GBR", "ML GBR+Smooth+Rot"]

for period_label, p_start, p_end in PERIODS:
    print(f"\n  Running {period_label} ...")
    period_results = run_all_strategies("SPY", p_start, p_end)
    # Filter to key strategies
    filtered = [r for r in period_results if r["label"] in KEY_STRATS]
    print_results_table(filtered, f"SPY — {period_label}")


# ══════════════════════════════════════════════════════════════════════
# PART 4: OUT-OF-SAMPLE SPLIT
# ══════════════════════════════════════════════════════════════════════
hline()
print("PART 4: IN-SAMPLE vs OUT-OF-SAMPLE SPLIT")
hline()

print("\n  In-sample: 2005–2017 | Out-of-sample: 2018–2024\n")

is_results = run_all_strategies("SPY", "2005-01-01", "2017-12-31")
oos_results = run_all_strategies("SPY", "2018-01-01", "2024-12-31")

is_filtered = [r for r in is_results if r["label"] in KEY_STRATS]
oos_filtered = [r for r in oos_results if r["label"] in KEY_STRATS]

print_results_table(is_filtered, "IN-SAMPLE: SPY 2005–2017")
print_results_table(oos_filtered, "OUT-OF-SAMPLE: SPY 2018–2024")

# Sharpe decay analysis
section("Sharpe Decay (IS → OOS)")
print(f"  {'Strategy':<22s} {'IS Sharpe':>10s} {'OOS Sharpe':>11s} {'Decay':>8s}")
print(f"  {'─'*22} {'─'*10} {'─'*11} {'─'*8}")
for is_r, oos_r in zip(is_filtered, oos_filtered):
    is_s = is_r["sharpe"] or 0
    oos_s = oos_r["sharpe"] or 0
    decay = is_s - oos_s
    print(f"  {is_r['label']:<22s} {fmt_f(is_s):>10s} {fmt_f(oos_s):>11s} {decay:>+8.3f}")


# ══════════════════════════════════════════════════════════════════════
# PART 5: ROLLING 1-YEAR SHARPE (20 years, SPY, key strategies)
# ══════════════════════════════════════════════════════════════════════
section("ROLLING 1-YEAR SHARPE — SPY (key strategies)")

# Use full_spy results
rolling_targets = [(r["label"], r["returns"]) for r in full_spy if r["label"] in KEY_STRATS]

spy_full = data_cache["SPY"]
years = sorted(set(spy_full.index.year))

col_names = [name[:18] for name, _ in rolling_targets]
print(f"  {'Year':>6s}", end="")
for cn in col_names:
    print(f"  {cn:>18s}", end="")
print()
print(f"  {'─'*6}", end="")
for _ in col_names:
    print(f"  {'─'*18}", end="")
print()

for year in years:
    mask = spy_full.index.year == year
    if not mask.any():
        continue
    last_day = spy_full.index[mask][-1]
    loc = spy_full.index.get_loc(last_day)
    start_loc = max(0, loc - 251)

    print(f"  {year:>6d}", end="")
    for name, rets in rolling_targets:
        if start_loc >= len(rets) or loc >= len(rets):
            print(f"  {'—':>18s}", end="")
            continue
        window_rets = rets.iloc[start_loc:loc+1]
        if len(window_rets) < 50:
            print(f"  {'—':>18s}", end="")
            continue
        std = window_rets.std(ddof=1)
        rs = float(window_rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
        print(f"  {rs:>18.3f}", end="")
    print()


# ══════════════════════════════════════════════════════════════════════
# PART 6: BOOTSTRAP CONFIDENCE INTERVALS (20-year SPY)
# ══════════════════════════════════════════════════════════════════════
section(f"BOOTSTRAP CONFIDENCE INTERVALS — SPY 20yr ({N_BOOTSTRAP} resamples)")

rng = np.random.default_rng(SEED)

boot_targets = [(r["label"], r["returns"].values) for r in full_spy if r["label"] in KEY_STRATS]

print(f"  {'Strategy':<22s} {'Median':>8s} {'5th pct':>9s} {'95th pct':>9s} {'P(Sharpe>0)':>12s}")
print(f"  {'─'*22} {'─'*8} {'─'*9} {'─'*9} {'─'*12}")

for name, rets in boot_targets:
    sharpes = bootstrap_sharpe(rets, N_BOOTSTRAP, rng)
    p5, p50, p95 = np.percentile(sharpes, [5, 50, 95])
    p_pos = (sharpes > 0).mean() * 100
    print(f"  {name:<22s} {p50:>8.3f} {p5:>9.3f} {p95:>9.3f} {p_pos:>11.1f}%")


# ══════════════════════════════════════════════════════════════════════
# PART 7: CROSS-ASSET SUMMARY
# ══════════════════════════════════════════════════════════════════════
hline()
print("CROSS-ASSET SUMMARY (2005–2024): Best ML config vs B&H")
hline()

print(f"\n  {'Ticker':<8s} {'B&H Sharpe':>11s} {'B&H Return':>11s} {'B&H MaxDD':>10s}  │  "
      f"{'Best ML':>10s} {'ML Return':>10s} {'ML MaxDD':>9s} {'ML Strat':<22s}")
print(f"  {'─'*8} {'─'*11} {'─'*11} {'─'*10}  │  {'─'*10} {'─'*10} {'─'*9} {'─'*22}")

for tkr in ["SPY", "QQQ", "IWM"]:
    results = run_all_strategies(tkr, FULL_START, FULL_END)
    bah = next(r for r in results if r["label"] == "B&H")
    ml_strats = [r for r in results if r["label"] not in ("B&H", "TDA Adaptive")]
    best_ml = max(ml_strats, key=lambda r: r["sharpe"] or -999)
    print(
        f"  {tkr:<8s} {fmt_f(bah['sharpe']):>11s} {fmt_pct(bah['total_return']):>11s} "
        f"{fmt_pct(bah['max_dd']):>10s}  │  "
        f"{fmt_f(best_ml['sharpe']):>10s} {fmt_pct(best_ml['total_return']):>10s} "
        f"{fmt_pct(best_ml['max_dd']):>9s} {best_ml['label']:<22s}"
    )


# ══════════════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * W}")
print("COMPARISON COMPLETE")
hline()
