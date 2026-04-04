"""Unit tests for TDA feature extraction."""
import numpy as np
import pytest

from backtester.tda.features import (
    persistence_entropy,
    max_persistence,
    mean_persistence,
    n_significant_features,
    compute_tda_features,
)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

class TestPersistenceEntropy:
    def test_single_feature_zero_entropy(self):
        """One feature = zero entropy (no uncertainty)."""
        diagram = np.array([[0.0, 1.0]])
        assert persistence_entropy(diagram) == pytest.approx(0.0, abs=0.01)

    def test_two_equal_features(self):
        """Two equal-lifetime features = 1 bit of entropy."""
        diagram = np.array([[0.0, 1.0], [0.0, 1.0]])
        assert persistence_entropy(diagram) == pytest.approx(1.0, abs=0.01)

    def test_many_equal_features_high_entropy(self):
        n = 16
        diagram = np.column_stack([np.zeros(n), np.ones(n)])
        expected = np.log2(n)  # max entropy for n equal features
        assert persistence_entropy(diagram) == pytest.approx(expected, abs=0.01)

    def test_one_dominant_low_entropy(self):
        """One large feature + many tiny ones → low entropy."""
        diagram = np.array([
            [0.0, 10.0],   # dominant
            [0.0, 0.01],
            [0.0, 0.01],
            [0.0, 0.01],
        ])
        ent = persistence_entropy(diagram)
        assert ent < 0.5

    def test_empty_diagram(self):
        diagram = np.empty((0, 2))
        assert persistence_entropy(diagram) == 0.0

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        births = np.zeros(20)
        deaths = rng.uniform(0.1, 5, 20)
        diagram = np.column_stack([births, deaths])
        assert persistence_entropy(diagram) >= 0.0


class TestMaxPersistence:
    def test_known_value(self):
        diagram = np.array([[0.0, 3.0], [1.0, 2.0], [0.0, 5.0]])
        assert max_persistence(diagram) == pytest.approx(5.0)

    def test_empty(self):
        assert max_persistence(np.empty((0, 2))) == 0.0


class TestMeanPersistence:
    def test_known_value(self):
        diagram = np.array([[0.0, 2.0], [0.0, 4.0]])
        assert mean_persistence(diagram) == pytest.approx(3.0)


class TestNSignificantFeatures:
    def test_all_significant(self):
        diagram = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        assert n_significant_features(diagram, threshold=0.1) == 3

    def test_one_dominant(self):
        diagram = np.array([[0.0, 10.0], [0.0, 0.5], [0.0, 0.5]])
        # threshold = 0.1 * 10 = 1.0; only the first feature > 1.0
        assert n_significant_features(diagram, threshold=0.1) == 1

    def test_empty(self):
        assert n_significant_features(np.empty((0, 2))) == 0


# ---------------------------------------------------------------------------
# End-to-end feature extraction
# ---------------------------------------------------------------------------

class TestComputeTDAFeatures:
    def test_trending_low_complexity(self):
        """A clean uptrend should have lower topological complexity than noise."""
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.005, 200)))
        features = compute_tda_features(prices, window=60, dimension=3, delay=5)
        assert features["tda_complexity"] < 0.7

    def test_choppy_higher_complexity(self):
        """Random noise should have higher complexity than a clean trend."""
        rng = np.random.default_rng(42)
        # Clean trend
        trend = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.003, 200)))
        # Choppy / random walk
        choppy = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 200)))
        f_trend = compute_tda_features(trend, window=60)
        f_choppy = compute_tda_features(choppy, window=60)
        assert f_choppy["tda_complexity"] > f_trend["tda_complexity"]

    def test_constant_price_minimal_features(self):
        prices = np.full(100, 100.0)
        features = compute_tda_features(prices, window=60)
        assert features["tda_complexity"] == pytest.approx(0.0, abs=0.05)

    def test_returns_all_keys(self):
        prices = np.linspace(100, 150, 200)
        features = compute_tda_features(prices, window=60)
        for key in ["h0_entropy", "h0_max_persistence", "h0_mean_persistence",
                     "h0_n_significant", "h1_proxy", "tda_complexity"]:
            assert key in features

    def test_insufficient_data_returns_defaults(self):
        prices = np.array([100.0, 101.0])
        features = compute_tda_features(prices, window=60)
        assert features["tda_complexity"] == 0.0

    def test_complexity_bounded(self):
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 500)))
        features = compute_tda_features(prices, window=60)
        assert 0.0 <= features["tda_complexity"] <= 1.0

    def test_no_look_ahead(self):
        """Features at index 100 should not depend on data after index 100."""
        rng = np.random.default_rng(42)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 200)))
        f1 = compute_tda_features(prices[:100], window=60)
        modified = prices.copy()
        modified[100:] = 0.01
        f2 = compute_tda_features(modified[:100], window=60)
        assert f1["tda_complexity"] == f2["tda_complexity"]
