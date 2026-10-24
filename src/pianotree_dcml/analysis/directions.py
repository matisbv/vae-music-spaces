"""Directions in the latent space, and what moving along them does to the music."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pianotree_dcml.analysis.decode import decode
from pianotree_dcml.analysis.features import bar_features, notes_table
from pianotree_dcml.model import PianoTreeVAE


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def mean_difference_direction(codes: np.ndarray, values: np.ndarray, fraction: float = 0.25
                              ) -> np.ndarray:
    """The direction from the average code of the bars with the lowest ``values`` (bottom
    ``fraction``) to that of the bars with the highest (top ``fraction``), as a unit
    vector. Bars with missing values are ignored."""
    known = ~np.isnan(values)
    codes, values = codes[known], values[known]
    low, high = np.quantile(values, [fraction, 1 - fraction])
    return unit(codes[values >= high].mean(axis=0) - codes[values <= low].mean(axis=0))


def consistency(shifts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine similarity between each bar's shift (one per row) and a direction: 1 when the
    bar moved exactly along it, 0 when its move had nothing to do with it."""
    return shifts @ unit(direction) / np.linalg.norm(shifts, axis=1)


def spread_along(codes: np.ndarray, direction: np.ndarray) -> float:
    """Standard deviation of the codes along a direction: a natural unit for moving along
    it (moving by 1 spread is a typical difference between two bars)."""
    return float((codes @ unit(direction)).std())


def move_and_measure(
    model: PianoTreeVAE,
    codes: np.ndarray,
    direction: np.ndarray,
    amounts: np.ndarray,
    bar_len: int,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Move each code by each amount along ``direction``, decode, and measure the features
    of the results. Returns one row per (bar, amount), with ``bar`` the row in ``codes``."""
    moved = codes[:, None, :] + amounts[None, :, None] * direction[None, None, :]
    decoded = decode(model, moved.reshape(-1, codes.shape[1]), batch_size)
    features = bar_features(notes_table(decoded), pd.Series([bar_len] * len(decoded)))
    features.insert(0, "amount", np.tile(amounts, len(codes)))
    features.insert(0, "bar", np.repeat(np.arange(len(codes)), len(amounts)))
    return features.reset_index(drop=True)
