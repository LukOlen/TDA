"""Integration tests for the analysis API endpoints (screener + regime)."""
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

def _make_ohlcv(n: int = 400, trend: float = 0.001, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(trend, 0.01, n)))
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
    fake = _make_ohlcv(400, trend=0.002)

    def _fake_load(ticker, *, start, end):
        if ticker == "^VIX":
            # Return synthetic VIX-like data (values around 15-25)
            rng = np.random.default_rng(99)
            n = 400
            close = 20.0 + rng.normal(0, 2, n).cumsum() * 0.1
            close = np.clip(close, 10, 40)
            idx = pd.date_range("2020-01-01", periods=n, freq="B")
            return pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": np.ones(n) * 1_000_000,
                },
                index=idx,
            )
        return fake

    with patch("backtester.screener.load_ohlcv", side_effect=_fake_load), \
         patch("backtester.regime.load_ohlcv", side_effect=_fake_load):
        yield


# ---------------------------------------------------------------------------
# Screen endpoint
# ---------------------------------------------------------------------------

class TestScreenEndpoint:
    def test_basic_screen(self, client):
        resp = client.post("/api/analysis/screen", json={
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "errors" in data
        assert len(data["results"]) == 3

    def test_response_structure(self, client):
        resp = client.post("/api/analysis/screen", json={
            "tickers": ["AAPL", "MSFT"],
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
        })
        data = resp.json()
        assert data["benchmark"] == "SPY"
        assert "start_date" in data
        assert "end_date" in data
        assert "total_screened" in data
        # Check result fields
        r = data["results"][0]
        for key in ["ticker", "rank", "composite_score", "sharpe", "volatility",
                     "max_drawdown", "momentum_1m", "momentum_3m", "momentum_6m"]:
            assert key in r

    def test_top_n(self, client):
        resp = client.post("/api/analysis/screen", json={
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
            "top_n": 2,
        })
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2

    def test_rank_ordering(self, client):
        resp = client.post("/api/analysis/screen", json={
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
        })
        ranks = [r["rank"] for r in resp.json()["results"]]
        assert ranks == [1, 2, 3]


# ---------------------------------------------------------------------------
# Regime endpoint
# ---------------------------------------------------------------------------

class TestRegimeEndpoint:
    def test_basic_regime(self, client):
        resp = client.post("/api/analysis/regime", json={
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["regime"] in ("bullish", "bearish", "neutral")

    def test_response_structure(self, client):
        resp = client.post("/api/analysis/regime", json={
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
        })
        data = resp.json()
        assert "regime" in data
        assert "composite_score" in data
        assert "confidence" in data
        assert "as_of_date" in data
        assert "indicators" in data
        assert len(data["indicators"]) >= 5

    def test_indicator_fields(self, client):
        resp = client.post("/api/analysis/regime", json={
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
        })
        for ind in resp.json()["indicators"]:
            assert "name" in ind
            assert "value" in ind
            assert "score" in ind
            assert "interpretation" in ind
            assert -1 <= ind["score"] <= 1

    def test_custom_thresholds(self, client):
        resp = client.post("/api/analysis/regime", json={
            "benchmark": "SPY",
            "start_date": "2020-01-01",
            "end_date": "2021-07-01",
            "bullish_threshold": 0.99,
            "bearish_threshold": -0.99,
        })
        assert resp.status_code == 200
        # With extreme thresholds, should be neutral
        assert resp.json()["regime"] == "neutral"
