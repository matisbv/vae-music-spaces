import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pianotree_dcml.data.dataset import BarDataset
from pianotree_dcml.representation import Note, grid_to_notes

BARS = [
    # bar_id, corpus, timesig, bar_len, split
    (0, "a", "3/4", 12, "train"),
    (1, "a", "4/4", 16, "train"),
    (2, "b", "3/4", 12, "valid"),
    (3, "b", "6/4", 24, "train"),
    (4, "b", "3/4", 12, "train"),
]
NOTES = {
    0: [(0, 60, 4), (0, 64, 4), (4, 62, 8)],
    1: [(0, 48, 16), (8, 126, 2)],
    2: [(3, 70, 1)],
    3: [(0, 60, 24)],
    4: [(0, pitch, 1) for pitch in range(40, 55)],  # 15 notes at once: too many for a grid
}


@pytest.fixture
def data_dir(tmp_path):
    bars = pd.DataFrame(BARS, columns=["bar_id", "corpus", "timesig", "bar_len", "split"])
    notes = pd.DataFrame(
        [(bar_id, *note) for bar_id, notes in NOTES.items() for note in notes],
        columns=["bar_id", "onset", "pitch", "duration"],
    )
    bars.to_csv(tmp_path / "bars.csv", index=False)
    notes.to_csv(tmp_path / "notes.csv", index=False)
    return tmp_path


def test_items_decode_to_the_bar_notes(data_dir):
    dataset = BarDataset(data_dir, split=None, seq_len=16)

    assert dataset.bars.bar_id.tolist() == [0, 1, 2]
    for index, bar_id in enumerate(dataset.bars.bar_id):
        grid, returned_index = dataset[index]
        assert returned_index == index
        assert grid.dtype == torch.int64
        assert grid.shape == (16, 16, 1 + 4)
        assert grid_to_notes(grid.numpy()) == [Note(*note) for note in NOTES[bar_id]]


def test_exclusions_are_counted(data_dir):
    dataset = BarDataset(data_dir, split="train", seq_len=16)
    assert dataset.bars.bar_id.tolist() == [0, 1]
    assert dataset.n_excluded == {"longer than seq_len": 1, "too many simultaneous notes": 1}

    roomy = BarDataset(data_dir, split="train", seq_len=24, max_simu_note=20)
    assert roomy.bars.bar_id.tolist() == [0, 1, 3, 4]


def test_filters(data_dir):
    assert BarDataset(data_dir, "valid", seq_len=16).bars.bar_id.tolist() == [2]
    by_corpus = BarDataset(data_dir, None, seq_len=24, corpora=["b"])
    assert by_corpus.bars.bar_id.tolist() == [2, 3]
    assert by_corpus.n_excluded["corpus"] == 2
    by_timesig = BarDataset(data_dir, None, seq_len=24, timesigs=["3/4", "6/4"])
    assert by_timesig.bars.bar_id.tolist() == [0, 2, 3]


def test_transposition_shifts_whole_bars_within_range(data_dir):
    dataset = BarDataset(data_dir, split=None, seq_len=16, transpose=(-3, 3))
    torch.manual_seed(0)

    shifts = set()
    for _ in range(200):
        notes = grid_to_notes(dataset[0][0].numpy())
        shift = notes[0].pitch - 60
        assert [n.pitch - shift for n in notes] == [60, 64, 62]
        shifts.add(shift)
    assert shifts == set(range(-3, 4))

    # Bar 1 contains MIDI pitch 126: it can only go up by one semitone.
    highest = {max(n.pitch for n in grid_to_notes(dataset[1][0].numpy())) for _ in range(200)}
    assert highest == {123, 124, 125, 126, 127}


def test_dataloader_batches(data_dir):
    dataset = BarDataset(data_dir, split=None, seq_len=16)
    grids, indices = next(iter(DataLoader(dataset, batch_size=3)))
    assert grids.shape == (3, 16, 16, 5)
    assert indices.tolist() == [0, 1, 2]
