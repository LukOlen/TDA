"""Tests for Monte Carlo risk estimation module."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from backtester.monte_carlo import run_monte_carlo, SimulationMethod


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    high = close * (1 + abs(rng.normal(0, 0.005, n)))
    low = close * (1 - abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _mock_load_ohlcv():
    fake = _make_ohlcv(500)
    with patch("api.routes.backtest.load_ohlcv", return_value=fake), \
         patch("backtester.data.yf.download", return_value=fake):
        yield


# ---------------------------------------------------------------------------
# Unit tests — run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    """Direct tests on the run_monte_carlo function."""

    @pytest.mark.parametrize("method", [m.value for m in SimulationMethod])
    def test_all_methods_produce_valid_result(self, method):
        ret = _make_returns()
        mc = run_monte_carlo(ret, n_simulations=200, horizon=63, method=method, seed=1)
        assert mc.n_simulations == 200
        assert mc.horizon == 63
        assert mc.method == method
        assert len(mc.var_results) == 2  # default confidence levels
        assert 0.0 <= mc.probability_of_loss <= 1.0

    def test_seed_reproducibility(self):
        ret = _make_returns()
        a = run_monte_carlo(ret, n_simulations=100, horizon=63, seed=42)
        b = run_monte_carlo(ret, n_simulations=100, horizon=63, seed=42)
        assert a.return_distribution.mean == b.return_distribution.mean
        assert a.var_results[0].var == b.var_results[0].var

    def test_var_cvar_ordering(self):
        ret = _make_returns()
        mc = run_monte_carlo(ret, n_simulations=500, horizon=252, seed=7)
        for vr in mc.var_results:
            # CVaR (expected shortfall) should be <= VaR (further in the tail)
            assert vr.cvar <= vr.var

        # 99% VaR should be <= 95% VaR (more extreme loss)
        var_95 = next(v for v in mc.var_results if v.confidence_level == 0.95)
        var_99 = next(v for v in mc.var_results if v.confidence_level == 0.99)
        assert var_99.var <= var_95.var

    def test_drawdowns_non_positive(self):
        ret = _make_returns()
        mc = run_monte_carlo(ret, n_simulations=200, horizon=63, seed=1)
        assert mc.drawdown_distribution.median <= 0
        assert mc.drawdown_distribution.percentile_95 <= 0

    def test_probability_of_loss_bounds(self):
        ret = _make_returns()
        mc = run_monte_carlo(ret, n_simulations=500, horizon=252, seed=1)
        assert 0.0 <= mc.probability_of_loss <= 1.0

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            run_monte_carlo(pd.Series([], dtype=float))

    def test_single_return_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            run_monte_carlo(pd.Series([0.01]))

    def test_unknown_method_raises(self):
        ret = _make_returns()
        with pytest.raises(ValueError, match="Unknown method"):
            run_monte_carlo(ret, method="nonexistent")

    def test_custom_confidence_levels(self):
        ret = _make_returns()
        mc = run_monte_carlo(ret, n_simulations=200, horizon=63, seed=1,
                             confidence_levels=[0.90, 0.95, 0.99])
        assert len(mc.var_results) == 3
        levels = [v.confidence_level for v in mc.var_results]
        assert levels == [0.90, 0.95, 0.99]

    def test_block_bootstrap_block_size(self):
        ret = _make_returns()
        mc = run_monte_carlo(ret, n_simulations=100, horizon=63,
                             method="block_bootstrap", block_size=10, seed=1)
        assert mc.method == "block_bootstrap"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------

class TestMonteCarloEndpoint:
    """Integration tests for POST /api/backtest/monte-carlo."""

    def test_basic_request(self, client):
        resp = client.post("/api/backtest/monte-carlo", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "n_simulations": 100,
            "horizon": 63,
            "seed": 42,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["ticker"] == "AAPL"
        assert body["strategy"] == "sma_crossover"
        assert body["n_simulations"] == 100
        assert body["horizon"] == 63
        assert "var_results" in body
        assert "return_distribution" in body
        assert "drawdown_distribution" in body
        assert "sharpe_distribution" in body

    def test_invalid_strategy_400(self, client):
        resp = client.post("/api/backtest/monte-carlo", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "nonexistent",
            "n_simulations": 100,
        })
        assert resp.status_code == 400

    @pytest.mark.parametrize("method", [m.value for m in SimulationMethod])
    def test_all_methods_accepted(self, client, method):
        resp = client.post("/api/backtest/monte-carlo", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "n_simulations": 100,
            "horizon": 63,
            "method": method,
            "seed": 1,
        })
        assert resp.status_code == 200
        assert resp.json()["method"] == method

    def test_cvar_leq_var_in_response(self, client):
        resp = client.post("/api/backtest/monte-carlo", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "n_simulations": 500,
            "horizon": 252,
            "seed": 7,
        })
        assert resp.status_code == 200
        for vr in resp.json()["var_results"]:
            assert vr["cvar"] <= vr["var"]

    def test_probability_of_loss_in_response(self, client):
        resp = client.post("/api/backtest/monte-carlo", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "n_simulations": 100,
            "horizon": 63,
            "seed": 1,
        })
        assert resp.status_code == 200
        p = resp.json()["probability_of_loss"]
        assert 0.0 <= p <= 1.0
