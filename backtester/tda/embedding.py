"""
Takens time-delay embedding for converting 1D time series into point clouds.

Given a scalar series x(t), the Takens embedding creates d-dimensional vectors:

    X[i] = (x[i], x[i + τ], x[i + 2τ], ..., x[i + (d-1)τ])

where d is the embedding dimension and τ is the time delay.  The resulting
point cloud preserves the topology of the underlying dynamical system
(Takens' theorem, 1981) and is the input to persistent homology.
"""
from __future__ import annotations

import numpy as np


def takens_embedding(
    series: np.ndarray,
    dimension: int = 3,
    delay: int = 5,
) -> np.ndarray:
    """
    Construct a Takens time-delay embedding of a 1D time series.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        1D input signal.  For financial data, use **log-returns**
        (not raw prices) for stationarity.
    dimension : int
        Embedding dimension *d*.  Each point becomes a d-vector.
    delay : int
        Time delay *τ* between successive coordinates (in bars).

    Returns
    -------
    np.ndarray, shape (n - (d-1)*τ, d)
        Point cloud in d-dimensional space.

    Raises
    ------
    ValueError
        If the series is too short for the given dimension and delay,
        or if dimension/delay are not positive integers.
    """
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    if delay < 1:
        raise ValueError("delay must be >= 1")

    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    n_points = n - (dimension - 1) * delay

    if n_points <= 0:
        raise ValueError(
            f"Series too short ({n}) for dimension={dimension}, delay={delay}; "
            f"need at least {(dimension - 1) * delay + 1} points"
        )

    # Vectorised construction: indices[i, j] = i + j * delay
    indices = np.arange(n_points)[:, None] + np.arange(dimension) * delay
    return series[indices]
