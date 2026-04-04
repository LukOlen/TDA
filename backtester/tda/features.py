"""
Topological feature extraction from persistence diagrams.

Converts raw (birth, death) persistence diagrams into scalar features
useful for regime detection:

  * **Persistence entropy** — Shannon entropy of normalised lifetimes.
    Low entropy = one dominant feature (clean trend).
    High entropy = many features of similar scale (complex / transitional).
  * **Max / mean persistence** — scale of the dominant topological feature.
  * **Significant feature count** — number of features above a noise floor.

The main entry point is :func:`compute_tda_features`, which runs the full
pipeline (log-returns → Takens embedding → persistence → features) on a
price window.
"""
from __future__ import annotations

import numpy as np

from .embedding import takens_embedding
from .persistence import vietoris_rips_persistence


# ── Individual feature functions ─────────────────────────────────────────

def persistence_entropy(diagram: np.ndarray) -> float:
    """
    Shannon entropy of the persistence diagram.

    H = −Σ pᵢ log₂(pᵢ)  where  pᵢ = Lᵢ / ΣLⱼ  and  Lᵢ = deathᵢ − birthᵢ.
    """
    if diagram.shape[0] == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    total = lifetimes.sum()
    if total == 0:
        return 0.0
    probs = lifetimes / total
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def max_persistence(diagram: np.ndarray) -> float:
    """Maximum lifetime (death − birth) across all features."""
    if diagram.shape[0] == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return float(lifetimes.max()) if len(lifetimes) > 0 else 0.0


def mean_persistence(diagram: np.ndarray) -> float:
    """Mean lifetime across all features."""
    if diagram.shape[0] == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return float(lifetimes.mean()) if len(lifetimes) > 0 else 0.0


def n_significant_features(diagram: np.ndarray, threshold: float = 0.1) -> int:
    """
    Count features with lifetime > *threshold* × max_lifetime.

    Features below this threshold are treated as topological noise.
    """
    if diagram.shape[0] == 0:
        return 0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    mx = lifetimes.max()
    if mx <= 0:
        return 0
    return int((lifetimes > threshold * mx).sum())


# ── End-to-end feature extraction ────────────────────────────────────────

def compute_tda_features(
    close: np.ndarray,
    window: int = 60,
    dimension: int = 3,
    delay: int = 5,
    n_subsample: int | None = 100,
) -> dict[str, float]:
    """
    Full TDA pipeline: price window → topological feature dict.

    Parameters
    ----------
    close : np.ndarray, shape (n,)
        **Closing prices** (not returns — the function computes log-returns
        internally).  Must have at least *window* elements.
    window : int
        Number of most recent prices to use.
    dimension : int
        Takens embedding dimension.
    delay : int
        Takens embedding time delay.
    n_subsample : int or None
        Max points for persistence computation.

    Returns
    -------
    dict with keys:
        h0_entropy, h0_max_persistence, h0_mean_persistence,
        h0_n_significant, h1_proxy, tda_complexity
    """
    close = np.asarray(close, dtype=float)
    if len(close) < window:
        return _default_features()

    prices = close[-window:]

    # Log-returns for stationarity
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(prices))
    # Replace any nan/inf from zero prices
    log_ret = np.nan_to_num(log_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Check we have enough data for embedding
    min_len = (dimension - 1) * delay + 1
    if len(log_ret) < min_len:
        return _default_features()

    # 1. Takens embedding
    cloud = takens_embedding(log_ret, dimension=dimension, delay=delay)

    # Degenerate case: all points identical (constant prices / zero returns)
    if cloud.ptp() < 1e-12:
        return _default_features()

    # 2. Persistent homology
    diagrams = vietoris_rips_persistence(
        cloud, max_dimension=1, n_subsample=n_subsample,
    )

    # 3. Extract features
    h0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))
    h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))

    h0_ent = persistence_entropy(h0)
    h0_max = max_persistence(h0)
    h0_mean = mean_persistence(h0)
    h0_nsig = n_significant_features(h0)

    # H0 complexity: ratio of mean to max persistence (dominance).
    # High max/mean = one dominant merge = well-structured (low complexity).
    # Low max/mean = uniform merges = disordered (high complexity).
    if h0_max > 0 and h0_mean > 0:
        dominance = h0_max / h0_mean  # typically 2–20
        h0_complexity = 1.0 / dominance  # 0–1; low when one feature dominates
    else:
        h0_complexity = 0.0

    # H1 proxy: fraction of significant loops relative to point count.
    h1_count = h1.shape[0]
    max_possible_loops = max(cloud.shape[0] * 0.1, 1)
    h1_proxy = min(h1_count / max_possible_loops, 1.0)

    # Composite complexity score
    tda_complexity = 0.7 * h0_complexity + 0.3 * h1_proxy

    return {
        "h0_entropy": h0_ent,
        "h0_max_persistence": h0_max,
        "h0_mean_persistence": h0_mean,
        "h0_n_significant": h0_nsig,
        "h1_proxy": h1_proxy,
        "tda_complexity": tda_complexity,
    }


def _default_features() -> dict[str, float]:
    """Return neutral features when there's insufficient data."""
    return {
        "h0_entropy": 0.0,
        "h0_max_persistence": 0.0,
        "h0_mean_persistence": 0.0,
        "h0_n_significant": 0,
        "h1_proxy": 0.0,
        "tda_complexity": 0.0,
    }
