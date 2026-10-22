"""Training objective of PianoTree-VAE: a beta-weighted negative ELBO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from pianotree_dcml.representation import DUR_PAD, PITCH_PAD


def dur_bit_weights(dur_width: int) -> torch.Tensor:
    """Weights of the duration bits, most significant first: a wrong high bit is a big error."""
    return torch.tensor(([1.0, 0.6, 0.4] + [0.3] * dur_width)[:dur_width])


def vae_loss(
    pitch_logits: torch.Tensor,
    dur_logits: torch.Tensor,
    grids: torch.Tensor,
    posterior: Normal,
    beta: float = 1.0,
    free_nats: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Loss of a batch, per bar (summed over the bar, averaged over the batch).

    - ``pitch``: cross-entropy of every note token of the bar, <eos> included;
    - ``duration``: weighted cross-entropy of the duration bits of every note;
    - ``kl``: KL(q(z | bar) || N(0, I)), summed over the latent dimensions.

    ``loss = pitch + duration + beta * max(kl - free_nats, 0)``. With ``free_nats``, the
    first nats of KL are free, which helps avoid posterior collapse (Kingma et al., 2016).
    Positions after <eos> have no target and are ignored.

    The original PianoTree-VAE code averaged every term instead: over tokens for the
    reconstruction and over the 512 latent dimensions for the KL. With about 30-40 pitch
    tokens per bar, its ``beta = 0.1`` corresponds to ``beta`` of about 0.007 here.
    """
    batch = grids.size(0)
    target = grids[:, :, 1:]  # the decoder does not predict <sos>
    pitch = F.cross_entropy(
        pitch_logits.flatten(0, -2), target[..., 0].flatten(),
        ignore_index=PITCH_PAD, reduction="sum",
    ) / batch
    dur_width = dur_logits.size(-2)
    bit_losses = F.cross_entropy(
        dur_logits.flatten(0, -2), target[..., 1:].flatten(),
        ignore_index=DUR_PAD, reduction="none",
    ).view(-1, dur_width)
    duration = (bit_losses * dur_bit_weights(dur_width).to(bit_losses.device)).sum() / batch

    prior = Normal(torch.zeros_like(posterior.loc), torch.ones_like(posterior.scale))
    kl = kl_divergence(posterior, prior).sum(-1).mean()
    loss = pitch + duration + beta * torch.clamp(kl - free_nats, min=0)
    return {"loss": loss, "pitch": pitch, "duration": duration, "kl": kl}
