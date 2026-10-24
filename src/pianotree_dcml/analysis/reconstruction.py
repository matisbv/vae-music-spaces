"""How well a model plays back individual bars."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from pianotree_dcml.metrics import note_counts
from pianotree_dcml.representation import Note


def bar_scores(originals: Sequence[Sequence[Note]], reconstructions: Sequence[Sequence[Note]]
               ) -> pd.DataFrame:
    """Note-level scores of each bar: ``note_f1`` (onset, pitch and duration right) and
    ``pitch_f1`` (onset and pitch right), plus the note counts behind them."""
    rows = []
    for original, reconstruction in zip(originals, reconstructions, strict=True):
        counts = note_counts(list(reconstruction), list(original))
        scores = counts.scores()
        rows.append({"note_f1": scores["note_f1"], "pitch_f1": scores["pitch_f1"],
                     "n_true": counts.true, "n_predicted": counts.predicted,
                     "n_matched": counts.matched})
    return pd.DataFrame(rows)
