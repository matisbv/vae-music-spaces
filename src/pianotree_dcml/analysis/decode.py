"""From latent codes back to notes."""

from __future__ import annotations

import numpy as np
import torch

from pianotree_dcml.model import PianoTreeVAE
from pianotree_dcml.representation import Note, grid_to_notes, logits_to_grid


@torch.no_grad()
def decode(
    model: PianoTreeVAE, z: np.ndarray | torch.Tensor, batch_size: int = 256, device: str = "cpu"
) -> list[list[Note]]:
    """Decode latent codes ``(n, z_size)`` (or a single code ``(z_size,)``) into the notes
    of one bar each, without any teacher forcing."""
    model.eval().to(device)
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    bars = []
    for start in range(0, len(z), batch_size):
        pitch_logits, dur_logits = model.decode(z[start:start + batch_size].to(device))
        grids = logits_to_grid(pitch_logits.cpu().numpy(), dur_logits.cpu().numpy())
        bars += [grid_to_notes(grid) for grid in grids]
    return bars


def slerp(z1: np.ndarray, z2: np.ndarray, steps: int) -> np.ndarray:
    """``steps`` codes along the great circle from ``z1`` to ``z2``, both included.

    Spherical interpolation is the usual choice for VAEs: most of a Gaussian's mass lies
    near a sphere, and points on the straight line between two codes fall inside it,
    where the decoder has seen few codes (White, 2016).
    """
    z1, z2 = np.asarray(z1, dtype=np.float64), np.asarray(z2, dtype=np.float64)
    cos = np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2))
    omega = np.arccos(np.clip(cos, -1.0, 1.0))
    t = np.linspace(0.0, 1.0, steps)[:, None]
    if np.isclose(np.sin(omega), 0.0):
        return (1 - t) * z1 + t * z2
    return (np.sin((1 - t) * omega) * z1 + np.sin(t * omega) * z2) / np.sin(omega)


def lerp(z1: np.ndarray, z2: np.ndarray, steps: int) -> np.ndarray:
    """``steps`` codes along the straight line from ``z1`` to ``z2``, both included."""
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return (1 - t) * np.asarray(z1, dtype=np.float64) + t * np.asarray(z2, dtype=np.float64)
