"""PyTorch dataset of DCML bars encoded as PianoTree piano-grids."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pianotree_dcml.representation import N_PITCHES, dur_width_for, duration_matrix_to_grid


class BarDataset(Dataset):
    """Bars from ``prepare``'s output, encoded as piano-grids on the fly.

    Each item is ``(grid, index)``. ``grid`` is a ``LongTensor`` of shape
    ``(seq_len, max_simu_note, 1 + dur_width)`` and ``index`` is the bar's row in
    ``self.bars``, which holds its metadata (corpus, piece, year, ...).

    Bars are padded with empty steps up to ``seq_len``. Some bars are left out:
    those longer than ``seq_len``, those with more than ``max_simu_note - 2`` notes
    starting at the same step (they don't fit in a piano-grid), and any bars excluded
    by the ``corpora`` and ``timesigs`` filters. ``self.n_excluded`` counts each
    reason.

    ``transpose=(low, high)`` transposes each bar by a random number of semitones in
    ``[low, high]`` every time it is loaded (data augmentation). The range is narrowed
    for each bar so that no note leaves the MIDI range.
    """

    def __init__(
        self,
        data_dir: Path | str,
        split: str | None,
        seq_len: int,
        max_simu_note: int = 16,
        corpora: Sequence[str] | None = None,
        timesigs: Sequence[str] | None = None,
        transpose: tuple[int, int] = (0, 0),
    ):
        data_dir = Path(data_dir)
        bars = pd.read_csv(data_dir / "bars.csv")
        notes = pd.read_csv(data_dir / "notes.csv")

        excluded = {}
        keep = pd.Series(True, index=bars.index)
        if split is not None:
            keep &= bars.split == split
        if corpora is not None:
            excluded["corpus"] = (keep & ~bars.corpus.isin(corpora)).sum()
            keep &= bars.corpus.isin(corpora)
        if timesigs is not None:
            excluded["timesig"] = (keep & ~bars.timesig.isin(timesigs)).sum()
            keep &= bars.timesig.isin(timesigs)
        too_long = bars.bar_len > seq_len
        excluded["longer than seq_len"] = (keep & too_long).sum()
        keep &= ~too_long
        max_onsets = notes.groupby(["bar_id", "onset"]).size().groupby("bar_id").max()
        too_dense = bars.bar_id.isin(max_onsets.index[max_onsets > max_simu_note - 2])
        excluded["too many simultaneous notes"] = (keep & too_dense).sum()
        keep &= ~too_dense

        self.bars = bars[keep].reset_index(drop=True)
        self.n_excluded = {reason: int(n) for reason, n in excluded.items()}
        self.seq_len = seq_len
        self.max_simu_note = max_simu_note
        self.dur_width = dur_width_for(seq_len)
        self.transpose = transpose

        # Notes of bar i are rows _start[i]:_end[i] of the arrays below.
        row = pd.Series(self.bars.index, index=self.bars.bar_id)
        notes = notes[notes.bar_id.isin(row.index)]
        rows = notes.bar_id.map(row).to_numpy()
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        self._onset = notes.onset.to_numpy()[order]
        self._pitch = notes.pitch.to_numpy()[order]
        self._duration = notes.duration.to_numpy()[order]
        self._start = np.searchsorted(rows, np.arange(len(self.bars)), side="left")
        self._end = np.searchsorted(rows, np.arange(len(self.bars)), side="right")

    def __len__(self) -> int:
        return len(self.bars)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        notes = slice(self._start[index], self._end[index])
        pitch = self._pitch[notes] + self._random_shift(self._pitch[notes])
        matrix = np.zeros((self.seq_len, N_PITCHES), dtype=np.int64)
        matrix[self._onset[notes], pitch] = self._duration[notes]
        grid = duration_matrix_to_grid(matrix, self.max_simu_note, self.dur_width)
        return torch.from_numpy(grid), index

    def _random_shift(self, pitches: np.ndarray) -> int:
        low = max(self.transpose[0], -int(pitches.min()))
        high = min(self.transpose[1], N_PITCHES - 1 - int(pitches.max()))
        if low > high:  # every shift in the range would push a note out of the MIDI range
            return 0
        # torch's generator is seeded per DataLoader worker, unlike numpy's.
        return int(torch.randint(low, high + 1, ()))
