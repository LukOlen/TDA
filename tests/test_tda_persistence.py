"""Unit tests for persistent homology computation."""
import numpy as np
import pytest

from backtester.tda.persistence import vietoris_rips_persistence


def _circle_points(n: int = 20, radius: float = 1.0) -> np.ndarray:
    """Points evenly spaced on a circle in 2D."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _collinear_points(n: int = 20) -> np.ndarray:
    """Points on a line in 2D."""
    return np.column_stack([np.linspace(0, 10, n), np.zeros(n)])


def _two_clusters(n_each: int = 10, separation: float = 10.0) -> np.ndarray:
    """Two well-separated Gaussian clusters in 2D."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(0, 0.5, (n_each, 2))
    c2 = rng.normal(separation, 0.5, (n_each, 2))
    return np.vstack([c1, c2])


class TestVietorisRipsPersistence:
    def test_returns_list_of_arrays(self):
        cloud = _circle_points(10)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=1, n_subsample=None)
        assert isinstance(diagrams, list)
        assert len(diagrams) == 2  # H0 and H1

    def test_h0_shape(self):
        cloud = _circle_points(10)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=None)
        h0 = diagrams[0]
        assert h0.ndim == 2
        assert h0.shape[1] == 2  # (birth, death)
        # 10 points → 9 merges → 9 H0 features
        assert h0.shape[0] == 9

    def test_h0_births_are_zero(self):
        cloud = _circle_points(15)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=None)
        np.testing.assert_array_equal(diagrams[0][:, 0], 0.0)

    def test_h0_deaths_positive(self):
        cloud = _circle_points(15)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=None)
        assert np.all(diagrams[0][:, 1] > 0)

    def test_h0_deaths_sorted(self):
        """Single-linkage produces merges in increasing distance order."""
        cloud = _circle_points(15)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=None)
        deaths = diagrams[0][:, 1]
        assert np.all(np.diff(deaths) >= -1e-10)  # non-decreasing

    def test_two_clusters_persistent_gap(self):
        """Two separated clusters should have one very persistent H0 feature."""
        cloud = _two_clusters(10, separation=20.0)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=None)
        lifetimes = diagrams[0][:, 1] - diagrams[0][:, 0]
        # The most persistent feature should be much larger than the rest
        sorted_lt = np.sort(lifetimes)
        assert sorted_lt[-1] > 5 * sorted_lt[-2]

    def test_circle_has_h1_features(self):
        """Points on a circle should produce at least one H1 feature (loop)."""
        cloud = _circle_points(20, radius=1.0)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=1, n_subsample=None)
        h1 = diagrams[1]
        assert h1.shape[0] > 0  # at least one loop detected

    def test_collinear_fewer_or_equal_h1_than_circle(self):
        """Collinear points should produce no more H1 features than a circle."""
        line = _collinear_points(20)
        circle = _circle_points(20, radius=1.0)
        d_line = vietoris_rips_persistence(line, max_dimension=1, n_subsample=None)
        d_circle = vietoris_rips_persistence(circle, max_dimension=1, n_subsample=None)
        # The H1 proxy is an approximation; collinear should not exceed circle
        assert d_line[1].shape[0] <= d_circle[1].shape[0]

    def test_single_point_returns_empty(self):
        cloud = np.array([[1.0, 2.0]])
        diagrams = vietoris_rips_persistence(cloud, max_dimension=1, n_subsample=None)
        assert all(d.shape[0] == 0 for d in diagrams)

    def test_subsampling_reduces_points(self):
        cloud = np.random.default_rng(42).normal(0, 1, (200, 3))
        diag_full = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=None)
        diag_sub = vietoris_rips_persistence(cloud, max_dimension=0, n_subsample=50)
        # Subsampled should have fewer H0 features (49 vs 199)
        assert diag_sub[0].shape[0] < diag_full[0].shape[0]

    def test_h1_births_less_than_deaths(self):
        cloud = _circle_points(20)
        diagrams = vietoris_rips_persistence(cloud, max_dimension=1, n_subsample=None)
        h1 = diagrams[1]
        if h1.shape[0] > 0:
            assert np.all(h1[:, 1] >= h1[:, 0])

    def test_deterministic(self):
        cloud = np.random.default_rng(42).normal(0, 1, (50, 3))
        d1 = vietoris_rips_persistence(cloud, max_dimension=1, n_subsample=None)
        d2 = vietoris_rips_persistence(cloud, max_dimension=1, n_subsample=None)
        for a, b in zip(d1, d2):
            np.testing.assert_array_equal(a, b)
