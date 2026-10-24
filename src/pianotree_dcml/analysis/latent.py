"""How much of the latent space a model actually uses."""

from __future__ import annotations

import numpy as np


def kl_per_dimension(mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Average KL(q(z_j | bar) || N(0, 1)) of each latent dimension over the bars, in nats.

    It measures how much information each dimension carries: about 0 for a dimension
    the encoder ignores (it outputs the prior whatever the bar).
    """
    kl = 0.5 * (mean**2 + scale**2 - 1 - 2 * np.log(scale))
    return kl.mean(axis=0)


def active_dimensions(mean: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Whether each latent dimension is *active*: its mean code varies across bars
    (variance above ``threshold``, Burda et al., 2016). An inactive dimension gives every
    bar the same value, so it can't distinguish anything."""
    return mean.var(axis=0) > threshold


def sample_prior(n: int, z_size: int, seed: int = 0) -> np.ndarray:
    """Random latent codes drawn from the prior N(0, I)."""
    return np.random.default_rng(seed).standard_normal((n, z_size)).astype(np.float32)
