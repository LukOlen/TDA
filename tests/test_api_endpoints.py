"""Integration tests for all API endpoints."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# Helpers
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


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def _mock_load_ohlcv():
    """Patch load_ohlcv everywhere it's imported to avoid real network calls."""
    fake = _make_ohlcv(500)
    with patch("api.routes.backtest.load_ohlcv", return_value=fake), \
         patch("backtester.data.yf.download", return_value=fake):
        yield


# ---------------------------------------------------------------------------
# Existing endpoints (smoke)
# ---------------------------------------------------------------------------

class TestHealthAndStrategies:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_list_strategies(self, client):
        resp = client.get("/api/strategies")
        assert resp.status_code == 200
        keys = [s["key"] for s in resp.json()]
        assert "sma_crossover" in keys
        assert len(keys) == 7


class TestSingleBacktest:
    def test_basic_backtest(self, client):
        resp = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "metrics" in body
        assert "equity_curve" in body
        assert "trades" in body

    def test_unknown_strategy_400(self, client):
        resp = client.post("/api/backtest", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "nonexistent",
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Batch backtest
# ---------------------------------------------------------------------------

class TestBatchBacktest:
    def test_single_ticker(self, client):
        resp = client.post("/api/backtest/batch", json={
            "tickers": ["AAPL"],
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 1
        assert len(body["errors"]) == 0

    def test_multiple_tickers(self, client):
        resp = client.post("/api/backtest/batch", json={
            "tickers": ["AAPL", "MSFT", "GOOG"],
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 3

    def test_invalid_strategy_400(self, client):
        resp = client.post("/api/backtest/batch", json={
            "tickers": ["AAPL"],
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "nonexistent",
        })
        assert resp.status_code == 400

    def test_partial_failure(self, client):
        """One ticker fails to load data; the rest succeed."""
        good_data = _make_ohlcv(300)

        def _selective_load(ticker, start, end):
            if ticker.upper() == "BAD":
                raise ValueError("No data returned for ticker 'BAD'")
            return good_data

        with patch("api.routes.backtest.load_ohlcv", side_effect=_selective_load):
            resp = client.post("/api/backtest/batch", json={
                "tickers": ["AAPL", "BAD", "GOOG"],
                "start_date": "2020-01-01",
                "end_date": "2024-01-01",
                "strategy": "sma_crossover",
            })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 2
        assert len(body["errors"]) == 1
        assert body["errors"][0]["ticker"] == "BAD"


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

class TestCompare:
    def test_compare_two_strategies(self, client):
        resp = client.post("/api/backtest/compare", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategies": [
                {"strategy": "sma_crossover", "params": {}},
                {"strategy": "momentum", "params": {}},
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 2

    def test_compare_invalid_strategy_400(self, client):
        resp = client.post("/api/backtest/compare", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategies": [
                {"strategy": "sma_crossover"},
                {"strategy": "fake_strategy"},
            ],
        })
        assert resp.status_code == 400

    def test_compare_returns_same_ticker(self, client):
        resp = client.post("/api/backtest/compare", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategies": [
                {"strategy": "sma_crossover"},
                {"strategy": "momentum"},
            ],
        })
        body = resp.json()
        assert body["ticker"] == "AAPL"
        for r in body["results"]:
            assert r["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Parameter optimisation
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_grid_search(self, client):
        resp = client.post("/api/backtest/optimize", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": [10, 20]},
                {"name": "slow_period", "values": [40, 50]},
            ],
            "target_metric": "sharpe_ratio",
            "method": "grid",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_combinations_evaluated"] > 0
        assert "best_params" in body
        assert body["results"][0]["rank"] == 1

    def test_random_search(self, client):
        resp = client.post("/api/backtest/optimize", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": list(range(5, 25))},
                {"name": "slow_period", "values": list(range(30, 80))},
            ],
            "target_metric": "sharpe_ratio",
            "method": "random",
            "n_samples": 5,
            "seed": 42,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["method"] == "random"
        assert body["total_combinations_evaluated"] <= 5

    def test_unknown_metric_422(self, client):
        resp = client.post("/api/backtest/optimize", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": [10]},
                {"name": "slow_period", "values": [40]},
            ],
            "target_metric": "nonexistent_metric",
        })
        assert resp.status_code == 422

    def test_grid_too_large_422(self, client):
        resp = client.post("/api/backtest/optimize", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": list(range(1, 102))},
                {"name": "slow_period", "values": list(range(1, 102))},
            ],
            "target_metric": "sharpe_ratio",
            "method": "grid",
        })
        assert resp.status_code == 422
        assert "random" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_basic(self, client):
        resp = client.post("/api/backtest/walk-forward", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": [10, 20]},
                {"name": "slow_period", "values": [40, 50]},
            ],
            "target_metric": "sharpe_ratio",
            "n_splits": 3,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_splits"] == 3
        assert len(body["splits"]) > 0
        assert "aggregate_oos_metrics" in body

    def test_split_structure(self, client):
        resp = client.post("/api/backtest/walk-forward", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": [10, 20]},
                {"name": "slow_period", "values": [40, 50]},
            ],
            "n_splits": 3,
        })
        body = resp.json()
        for split in body["splits"]:
            assert "split_index" in split
            assert "in_sample_start" in split
            assert "oos_end" in split
            assert "best_params" in split
            assert "in_sample_metrics" in split
            assert "oos_metrics" in split
            assert "oos_equity_curve" in split

    def test_unknown_metric_422(self, client):
        resp = client.post("/api/backtest/walk-forward", json={
            "ticker": "AAPL",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01",
            "strategy": "sma_crossover",
            "param_grid": [
                {"name": "fast_period", "values": [10]},
                {"name": "slow_period", "values": [40]},
            ],
            "target_metric": "bad_metric",
            "n_splits": 3,
        })
        assert resp.status_code == 422
