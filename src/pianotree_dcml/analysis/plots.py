"""Piano-roll figures of bars."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pianotree_dcml.representation import Note, pitch_name


def plot_pianoroll(
    notes: Sequence[Note],
    bar_len: int | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    pitch_range: tuple[int, int] | None = None,
    color: str = "tab:blue",
) -> plt.Axes:
    """Draw one bar: time (16th notes) left to right, pitch bottom to top."""
    ax = ax or plt.gca()
    for onset, pitch, duration in notes:
        ax.add_patch(Rectangle((onset, pitch - 0.4), duration, 0.8, facecolor=color,
                               edgecolor="black", linewidth=0.5))
    end = bar_len or max((o + d for o, _, d in notes), default=16)
    low, high = pitch_range or _pitch_range([notes])
    ax.set_xlim(0, end)
    ax.set_ylim(low - 0.5, high + 0.5)
    ax.set_xticks(range(0, end + 1, 4))
    ax.set_xticks(range(end + 1), minor=True)
    ax.grid(axis="x", which="major", alpha=0.5)
    ax.grid(axis="x", which="minor", alpha=0.15)
    c_notes = [p for p in range(low, high + 1) if p % 12 == 0]
    ax.set_yticks(c_notes, [pitch_name(p) for p in c_notes])
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("time (16th notes)")
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_bars(
    bars: Sequence[Sequence[Note]],
    titles: Sequence[str] | None = None,
    bar_lens: Sequence[int | None] | None = None,
    ncols: int | None = None,
    color: str = "tab:blue",
    panel_size: tuple[float, float] = (3.2, 2.8),
) -> plt.Figure:
    """Several bars side by side, with a shared pitch axis so they can be compared."""
    ncols = ncols or len(bars)
    nrows = -(-len(bars) // ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharey=True,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    pitch_range = _pitch_range(bars)
    for i, ax in enumerate(axes.flat):
        if i >= len(bars):
            ax.axis("off")
            continue
        plot_pianoroll(bars[i], bar_lens[i] if bar_lens else None, ax,
                       titles[i] if titles else None, pitch_range, color)
    fig.tight_layout()
    return fig


def _pitch_range(bars: Sequence[Sequence[Note]], margin: int = 3) -> tuple[int, int]:
    pitches = [p for notes in bars for _, p, _ in notes]
    if not pitches:
        return 48, 72
    return min(pitches) - margin, max(pitches) + margin
