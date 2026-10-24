"""Comparing the geometry of the latent space with a hypothesis.

Representational similarity analysis: compute the distances between groups (keys,
composers, ...) in the latent space, and test whether they follow the distances a
hypothesis predicts (the circle of fifths, years apart, ...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def fifths_distance(a: int, b: int) -> int:
    """Steps between two keys (pitch classes) around the circle of fifths: 0 for the
    same key, 1 for C and G, 6 for C and F#."""
    steps = (7 * (a - b)) % 12
    return min(steps, 12 - steps)


def centroid_distances(codes: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Euclidean distances between the average codes of each label."""
    names = sorted(set(labels), key=str)
    centroids = np.stack([codes[labels == name].mean(axis=0) for name in names])
    distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    return pd.DataFrame(distances, index=names, columns=names)


def _upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(x)), x])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def mantel(
    observed: np.ndarray,
    hypothesis: np.ndarray,
    control: np.ndarray | None = None,
    n_permutations: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Rank correlation between two distance matrices, and its one-sided p-value.

    The p-value comes from shuffling the groups of the ``hypothesis`` matrix (rows and
    columns together): how often a random relabelling matches the observed distances as
    well. With ``control``, both matrices are first corrected for it (partial
    correlation), so that the result says how well the hypothesis explains the distances
    *beyond* the control.
    """
    observed, hypothesis = np.asarray(observed, float), np.asarray(hypothesis, float)
    a = rankdata(_upper(observed))
    if control is not None:
        c = rankdata(_upper(np.asarray(control, float)))
        a = _residual(a, c)

    def correlation(h):
        b = rankdata(_upper(h))
        if control is not None:
            b = _residual(b, c)
        return np.corrcoef(a, b)[0, 1]

    statistic = correlation(hypothesis)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_permutations):
        order = rng.permutation(len(hypothesis))
        exceed += correlation(hypothesis[np.ix_(order, order)]) >= statistic
    return float(statistic), (exceed + 1) / (n_permutations + 1)
