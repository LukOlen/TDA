"""Unit tests for the optimiser module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.optimizer import grid_search, random_search, walk_forward
from backtester.strategies import SMACrossover, BreakoutMomentum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 500, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Synthetic trending OHLCV data (reuses pattern from test_engine.py)."""
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    high = close * (1 + abs(rng.normal(0, 0.005, n)))
    low = close * (1 - abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

class TestGridSearch:
    def test_returns_sorted_results(self):
        data = make_ohlcv(300)
        results = grid_search(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
        )
        assert len(results) > 0
        # Verify sorted descending by sharpe (maximize=True default)
        sharpes = [r["metrics"].get("sharpe_ratio") for r in results]
        for a, b in zip(sharpes, sharpes[1:]):
            if a is not None and b is not None:
                assert a >= b

    def test_ranks_are_sequential(self):
        data = make_ohlcv(300)
        results = grid_search(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
        )
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_all_combinations_evaluated(self):
        data = make_ohlcv(300)
        results = grid_search(
            data, SMACrossover,
            {"fast_period": [5, 10, 15], "slow_period": [30, 50]},
            metric="total_return",
        )
        # 3 x 2 = 6 combos
        assert len(results) == 6

    def test_skips_invalid_param_combos(self):
        data = make_ohlcv(300)
        # fast_period=100 > slow_period=50 should be invalid for SMACrossover
        results = grid_search(
            data, SMACrossover,
            {"fast_period": [10, 100], "slow_period": [50]},
            metric="sharpe_ratio",
        )
        # Only the valid combo should survive
        assert len(results) == 1
        assert results[0]["params"]["fast_period"] == 10

    def test_minimize_mode(self):
        data = make_ohlcv(300)
        results = grid_search(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="max_drawdown",
            maximize=False,
        )
        # max_drawdown is negative; minimising means most negative first
        drawdowns = [r["metrics"].get("max_drawdown") for r in results]
        for a, b in zip(drawdowns, drawdowns[1:]):
            if a is not None and b is not None:
                assert a <= b

    def test_result_contains_metrics_dict(self):
        data = make_ohlcv(300)
        results = grid_search(
            data, SMACrossover,
            {"fast_period": [10], "slow_period": [40]},
            metric="sharpe_ratio",
        )
        assert len(results) == 1
        assert "sharpe_ratio" in results[0]["metrics"]
        assert "total_return" in results[0]["metrics"]


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------

class TestRandomSearch:
    def test_respects_n_samples(self):
        data = make_ohlcv(300)
        results = random_search(
            data, SMACrossover,
            {"fast_period": list(range(5, 25)), "slow_period": list(range(30, 80))},
            metric="sharpe_ratio",
            n_samples=8,
            seed=42,
        )
        assert len(results) <= 8

    def test_seed_reproducibility(self):
        data = make_ohlcv(300)
        grid = {"fast_period": list(range(5, 25)), "slow_period": list(range(30, 80))}
        r1 = random_search(data, SMACrossover, grid, "sharpe_ratio", n_samples=5, seed=99)
        r2 = random_search(data, SMACrossover, grid, "sharpe_ratio", n_samples=5, seed=99)
        assert [r["params"] for r in r1] == [r["params"] for r in r2]

    def test_falls_back_to_grid_when_n_samples_exceeds_total(self):
        data = make_ohlcv(300)
        # 2 x 2 = 4 combos, request 100 samples → should get all 4
        results = random_search(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_samples=100,
        )
        assert len(results) == 4

    def test_results_are_sorted(self):
        data = make_ohlcv(300)
        results = random_search(
            data, SMACrossover,
            {"fast_period": list(range(5, 25)), "slow_period": list(range(30, 80))},
            metric="sharpe_ratio",
            n_samples=10,
            seed=42,
        )
        sharpes = [r["metrics"].get("sharpe_ratio") for r in results]
        for a, b in zip(sharpes, sharpes[1:]):
            if a is not None and b is not None:
                assert a >= b


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_basic_rolling(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
            in_sample_pct=0.7,
        )
        assert "splits" in result
        assert "aggregate_oos_metrics" in result
        assert len(result["splits"]) == 3

    def test_split_dates_are_sequential(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
        )
        for split in result["splits"]:
            assert split["in_sample_start"] < split["in_sample_end"]
            assert split["oos_start"] <= split["oos_end"]

    def test_each_split_has_best_params(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
        )
        for split in result["splits"]:
            assert "best_params" in split
            assert "fast_period" in split["best_params"]
            assert "slow_period" in split["best_params"]

    def test_oos_metrics_populated(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
        )
        for split in result["splits"]:
            assert "oos_metrics" in split
            assert "sharpe_ratio" in split["oos_metrics"]

    def test_aggregate_metrics_present(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
        )
        agg = result["aggregate_oos_metrics"]
        assert "total_return" in agg
        assert "sharpe_ratio" in agg

    def test_anchored_mode(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
            anchored=True,
        )
        # All IS windows should start at the same date in anchored mode
        starts = [s["in_sample_start"] for s in result["splits"]]
        assert all(s == starts[0] for s in starts)

    def test_insufficient_data_raises(self):
        data = make_ohlcv(50)  # very small dataset
        with pytest.raises(ValueError, match="Insufficient data"):
            walk_forward(
                data, SMACrossover,
                {"fast_period": [10, 20], "slow_period": [40, 50]},
                metric="sharpe_ratio",
                n_splits=5,
            )

    def test_oos_equity_curve_present(self):
        data = make_ohlcv(600, seed=7)
        result = walk_forward(
            data, SMACrossover,
            {"fast_period": [10, 20], "slow_period": [40, 50]},
            metric="sharpe_ratio",
            n_splits=3,
        )
        for split in result["splits"]:
            assert "oos_equity_curve" in split
            assert len(split["oos_equity_curve"]) > 0
            assert "date" in split["oos_equity_curve"][0]
            assert "value" in split["oos_equity_curve"][0]
