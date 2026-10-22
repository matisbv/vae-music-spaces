"""Note-level reconstruction metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

from pianotree_dcml.representation import Note, grid_to_notes, logits_to_grid


@dataclass
class NoteCounts:
    """Counts behind precision, recall and F1, summed over bars.

    A note is matched when a note with the same onset, pitch and duration exists in the
    other bar; for the ``pitch_*`` scores, onset and pitch are enough.
    """

    predicted: int = 0
    true: int = 0
    matched: int = 0
    matched_pitch: int = 0

    def __add__(self, other: NoteCounts) -> NoteCounts:
        return NoteCounts(self.predicted + other.predicted, self.true + other.true,
                          self.matched + other.matched, self.matched_pitch + other.matched_pitch)

    def scores(self) -> dict[str, float]:
        scores = {}
        for prefix, matched in [("note", self.matched), ("pitch", self.matched_pitch)]:
            precision = matched / self.predicted if self.predicted else 0.0
            recall = matched / self.true if self.true else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            scores |= {f"{prefix}_precision": precision, f"{prefix}_recall": recall,
                       f"{prefix}_f1": f1}
        return scores


def note_counts(predicted: list[Note], true: list[Note]) -> NoteCounts:
    def onset_pitch(notes):
        return Counter((n.onset, n.pitch) for n in notes)

    return NoteCounts(
        predicted=len(predicted),
        true=len(true),
        matched=sum((Counter(predicted) & Counter(true)).values()),
        matched_pitch=sum((onset_pitch(predicted) & onset_pitch(true)).values()),
    )


def batch_note_counts(
    pitch_logits: torch.Tensor, dur_logits: torch.Tensor, grids: torch.Tensor
) -> NoteCounts:
    """Compare the notes decoded from model outputs with the true notes of a batch."""
    predicted = logits_to_grid(pitch_logits.detach().cpu().numpy(), dur_logits.detach().cpu().numpy())
    counts = NoteCounts()
    for predicted_grid, true_grid in zip(predicted, grids.cpu().numpy()):
        counts += note_counts(grid_to_notes(predicted_grid), grid_to_notes(true_grid))
    return counts
