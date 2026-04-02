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
    alpha,
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
# Alpha
# ---------------------------------------------------------------------------

class TestAlpha:
    def test_alpha_zero_when_returns_match_benchmark(self):
        # Beta is 1.0, mean_s == mean_b, so alpha = 0.0
        returns = make_returns([0.01, -0.005, 0.002] * 84)
        result = alpha(returns, returns)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_positive_alpha(self):
        # Benchmark returns
        b_returns = make_returns([0.01, -0.005, 0.002] * 84)
        # Strategy returns are benchmark + 0.001 daily
        s_returns = b_returns + 0.001

        # Beta should be exactly 1.0.
        # Mean difference is 0.001 daily, annualised is 0.001 * 252 = 0.252.
        result = alpha(s_returns, b_returns, periods=252)
        assert result == pytest.approx(0.252, abs=1e-6)

    def test_negative_alpha(self):
        # Benchmark returns
        b_returns = make_returns([0.01, -0.005, 0.002] * 84)
        # Strategy returns are benchmark - 0.001 daily
        s_returns = b_returns - 0.001

        # Alpha should be -0.001 * 252 = -0.252
        result = alpha(s_returns, b_returns, periods=252)
        assert result == pytest.approx(-0.252, abs=1e-6)

    def test_risk_free_rate_impact(self):
        # Benchmark returns
        b_returns = make_returns([0.01, -0.005, 0.002] * 84)
        # Strategy has half the volatility (beta = 0.5) + constant
        s_returns = b_returns * 0.5

        # alpha = mean_s - rf - beta * (mean_b - rf)
        # Since mean_s = 0.5 * mean_b and beta = 0.5:
        # alpha(rf=0) = 0.5 * mean_b - 0 - 0.5 * (mean_b - 0) = 0
        # alpha(rf=0.05) = 0.5 * mean_b - 0.05 - 0.5 * (mean_b - 0.05) = -0.05 + 0.025 = -0.025
        res_0 = alpha(s_returns, b_returns, risk_free_rate=0.0, periods=252)
        assert res_0 == pytest.approx(0.0, abs=1e-6)

        res_5 = alpha(s_returns, b_returns, risk_free_rate=0.05, periods=252)
        assert res_5 == pytest.approx(-0.025, abs=1e-6)

    def test_nan_beta_returns_nan(self):
        # Too few data points -> nan beta -> nan alpha
        s_returns = make_returns([0.01])
        b_returns = make_returns([0.01])
        result = alpha(s_returns, b_returns)
        assert np.isnan(result)


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
