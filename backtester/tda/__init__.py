"""
Topological Data Analysis (TDA) for financial regime detection.

Provides a pure scipy/numpy pipeline for computing persistent homology
features from price time series:

    price window → log-returns → Takens embedding → point cloud
    → Vietoris-Rips persistence → topological features

The main entry point is :func:`compute_tda_features`.
"""
from .features import compute_tda_features

__all__ = ["compute_tda_features"]
