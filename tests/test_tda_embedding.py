"""Unit tests for Takens time-delay embedding."""
import numpy as np
import pytest

from backtester.tda.embedding import takens_embedding


class TestTakensEmbedding:
    def test_output_shape(self):
        series = np.arange(100, dtype=float)
        cloud = takens_embedding(series, dimension=3, delay=5)
        # n_points = 100 - (3-1)*5 = 90
        assert cloud.shape == (90, 3)

    def test_output_shape_d1(self):
        series = np.arange(50, dtype=float)
        cloud = takens_embedding(series, dimension=1, delay=1)
        assert cloud.shape == (50, 1)

    def test_last_row_values(self):
        series = np.arange(20, dtype=float)
        cloud = takens_embedding(series, dimension=3, delay=2)
        # Last point: index = 20 - (3-1)*2 - 1 = 15
        # Values: series[15], series[17], series[19]
        np.testing.assert_array_equal(cloud[-1], [15.0, 17.0, 19.0])

    def test_first_row_values(self):
        series = np.arange(20, dtype=float)
        cloud = takens_embedding(series, dimension=3, delay=2)
        # First point: series[0], series[2], series[4]
        np.testing.assert_array_equal(cloud[0], [0.0, 2.0, 4.0])

    def test_delay_1(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cloud = takens_embedding(series, dimension=3, delay=1)
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=float)
        np.testing.assert_array_equal(cloud, expected)

    def test_too_short_raises(self):
        series = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="too short"):
            takens_embedding(series, dimension=3, delay=5)

    def test_dimension_zero_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            takens_embedding(np.arange(10.0), dimension=0, delay=1)

    def test_delay_zero_raises(self):
        with pytest.raises(ValueError, match="delay"):
            takens_embedding(np.arange(10.0), dimension=2, delay=0)

    def test_different_delays_differ(self):
        series = np.random.default_rng(42).normal(0, 1, 100)
        c1 = takens_embedding(series, dimension=3, delay=1)
        c2 = takens_embedding(series, dimension=3, delay=5)
        assert c1.shape != c2.shape or not np.allclose(c1[:c2.shape[0]], c2)

    def test_deterministic(self):
        series = np.random.default_rng(42).normal(0, 1, 100)
        c1 = takens_embedding(series, dimension=3, delay=5)
        c2 = takens_embedding(series, dimension=3, delay=5)
        np.testing.assert_array_equal(c1, c2)

    def test_preserves_order(self):
        """Temporal order is preserved: cloud[i] uses earlier data than cloud[i+1]."""
        series = np.arange(50, dtype=float)
        cloud = takens_embedding(series, dimension=3, delay=2)
        # Each row's first element should be strictly increasing
        assert np.all(np.diff(cloud[:, 0]) > 0)

    def test_accepts_list_input(self):
        cloud = takens_embedding([1.0, 2.0, 3.0, 4.0, 5.0], dimension=2, delay=1)
        assert cloud.shape == (4, 2)
