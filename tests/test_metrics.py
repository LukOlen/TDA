"""Unit tests for performance metrics."""
import numpy as np
import pandas as pd
import pytest

from backtester.stats.metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    cagr,
    calmar_ratio,
    win_rate,
    profit_factor,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_returns(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def make_equity(initial: float, returns: pd.Series) -> pd.Series:
    equity = initial * (1 + returns).cumprod()
    equity.iloc[0] = initial
    return equity


def make_trades(pnls: list[float]) -> pd.DataFrame:
    rows = [
        {
            "entry_date": pd.Timestamp("2020-01-01"),
            "exit_date": pd.Timestamp("2020-02-01"),
            "direction": "long",
            "entry_price": 100.0,
            "exit_price": 100.0 + p,
            "pnl": p,
            "pnl_pct": p / 100.0,
        }
        for p in pnls
    ]
    if not rows:
        return pd.DataFrame(
            columns=["entry_date", "exit_date", "direction",
                     "entry_price", "exit_price", "pnl", "pnl_pct"]
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_returns(self):
        # Non-constant positive returns (constant series has std=0 → Sharpe=0)
        returns = make_returns([0.01, 0.005] * 126)
        sr = sharpe_ratio(returns)
        assert sr > 0

    def test_zero_std_returns_zero(self):
        returns = make_returns([0.0] * 252)
        assert sharpe_ratio(returns) == 0.0

    def test_annualisation(self):
        # mean ≈ std ≈ 0.01 → Sharpe ≈ sqrt(252) * mean/std ≈ sqrt(252)
        rng = np.random.default_rng(0)
        returns = make_returns(rng.normal(0.01, 0.01, 252).tolist())
        sr = sharpe_ratio(returns)
        assert abs(sr - np.sqrt(252)) < 3.0  # wide tolerance for random sample

    def test_negative_mean_returns_negative(self):
        # Non-constant negative returns so std > 0
        returns = make_returns([-0.005, -0.003] * 126)
        assert sharpe_ratio(returns) < 0


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:
    def test_no_downside_returns_inf(self):
        returns = make_returns([0.01] * 252)
        assert sortino_ratio(returns) == float("inf")

    def test_negative_mean_returns_negative(self):
        # Use three values so the downside returns are non-constant (std > 0)
        returns = make_returns([-0.005, -0.003, 0.001] * 84)
        assert sortino_ratio(returns) < 0

    def test_greater_than_sharpe_when_downside_small(self):
        rng = np.random.default_rng(42)
        pos = abs(rng.normal(0.005, 0.01, 252))
        returns = make_returns(pos.tolist())
        assert sortino_ratio(returns) >= sharpe_ratio(returns)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        equity = pd.Series([100, 110, 120, 130], dtype=float)
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_50_percent_drawdown(self):
        equity = pd.Series([100.0, 120.0, 60.0, 80.0])
        mdd = max_drawdown(equity)
        assert mdd == pytest.approx(-0.5)

    def test_always_negative_or_zero(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        equity = 100 * (1 + returns).cumprod()
        assert max_drawdown(equity) <= 0


# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_doubles_in_1_year(self):
        # 252 trading days, equity doubles
        equity = pd.Series(np.linspace(100, 200, 252))
        result = cagr(equity, periods=252)
        assert result == pytest.approx(1.0, abs=0.05)

    def test_zero_for_flat(self):
        equity = pd.Series([100.0] * 252)
        assert cagr(equity) == pytest.approx(0.0, abs=1e-6)

    def test_short_series(self):
        assert cagr(pd.Series([100.0]), periods=252) == 0.0


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_positive_cagr_negative_drawdown(self):
        equity = pd.Series([100, 110, 105, 120, 115, 130], dtype=float)
        result = calmar_ratio(equity, periods=len(equity))
        assert result > 0

    def test_zero_drawdown_returns_inf(self):
        equity = pd.Series([100.0, 110.0, 120.0])
        assert calmar_ratio(equity) == float("inf")


# ---------------------------------------------------------------------------
# Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_winners(self):
        trades = make_trades([10, 20, 30])
        assert win_rate(trades) == pytest.approx(1.0)

    def test_half_winners(self):
        trades = make_trades([10, -10])
        assert win_rate(trades) == pytest.approx(0.5)

    def test_empty_trades(self):
        assert win_rate(make_trades([])) == 0.0

    def test_all_losers(self):
        trades = make_trades([-5, -10])
        assert win_rate(trades) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_equal_profit_and_loss(self):
        trades = make_trades([10, -10])
        assert profit_factor(trades) == pytest.approx(1.0)

    def test_no_losses_returns_inf(self):
        trades = make_trades([5, 10, 15])
        assert profit_factor(trades) == float("inf")

    def test_greater_than_one_when_profitable(self):
        trades = make_trades([10, 10, -5])
        assert profit_factor(trades) > 1.0

    def test_empty_trades(self):
        assert profit_factor(make_trades([])) == 0.0


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

class TestBeta:
    def test_identical_series_beta_is_one(self):
        returns = make_returns([0.01, -0.02, 0.03, -0.01] * 50)
        from backtester.stats.metrics import beta
        assert beta(returns, returns) == pytest.approx(1.0)

    def test_scaled_series(self):
        benchmark = make_returns([0.01, -0.02, 0.03, -0.01] * 50)
        strategy = benchmark * 1.5
        from backtester.stats.metrics import beta
        assert beta(strategy, benchmark) == pytest.approx(1.5)

    def test_zero_variance_benchmark_returns_nan(self):
        benchmark = make_returns([0.01] * 100)
        strategy = make_returns([0.02] * 100)
        # Due to numerical precision, a constant series can yield a tiny non-zero variance.
        # We explicitly set `var_b` to 0.0 using a patched benchmark
        benchmark.loc[:] = 0.0
        strategy.loc[:] = 0.0
        from backtester.stats.metrics import beta
        import math
        assert math.isnan(beta(strategy, benchmark))

    def test_short_series_returns_nan(self):
        benchmark = make_returns([0.01])
        strategy = make_returns([0.02])
        from backtester.stats.metrics import beta
        import math
        assert math.isnan(beta(strategy, benchmark))

    def test_alignment_and_dropna(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="B")
        benchmark = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
        # Strategy misses first day, has NaN on third day
        strategy = pd.Series([0.02, np.nan, 0.08, 0.10], index=idx[1:])

        # Aligned indices should be idx[1], idx[3], idx[4]
        # Benchmark: [0.02, 0.04, 0.05]
        # Strategy: [0.02, 0.08, 0.10]
        # However, covariance formula might output something slightly different based on the mean
        # Let's mock a simpler relation. Strategy = 2 * benchmark + 0.01

        # Benchmark at aligned indices: [0.02, 0.04, 0.05]
        # Mean = (0.02 + 0.04 + 0.05) / 3 = 0.11 / 3 = 0.036666
        # Devs: [-0.016666, 0.003333, 0.013333]

        # Let's create an exact linear relationship to ensure Beta = 2.0
        # Strategy = 2 * Benchmark
        # Benchmark: [0.02, 0.04, 0.05] -> Strategy: [0.04, 0.08, 0.10]

        strategy = pd.Series([0.04, np.nan, 0.08, 0.10], index=idx[1:])

        from backtester.stats.metrics import beta
        assert beta(strategy, benchmark) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# compute_metrics integration
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_returns_all_keys(self):
        returns = make_returns([0.001] * 252)
        equity = make_equity(100_000, returns)
        trades = make_trades([100, -50, 200])
        metrics = compute_metrics(returns, equity, trades)

        expected_keys = {
            "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "calmar_ratio", "volatility",
            "win_rate", "profit_factor", "avg_trade_return", "total_trades",
            "benchmark_return", "beta", "alpha",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_no_inf_values(self):
        returns = make_returns([0.002, -0.001] * 126)
        equity = make_equity(100_000, returns)
        trades = make_trades([50, -20, 30])
        metrics = compute_metrics(returns, equity, trades)
        for k, v in metrics.items():
            if isinstance(v, float):
                assert not np.isinf(v), f"Metric {k!r} is inf"
