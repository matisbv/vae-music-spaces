import numpy as np
import pytest
import torch

pytest.importorskip("sklearn", reason="needs the analysis extra")

from pianotree_dcml.analysis.directions import (  # noqa: E402
    consistency, mean_difference_direction, move_and_measure, spread_along, unit)
from pianotree_dcml.analysis.encode import encode_notes  # noqa: E402
from pianotree_dcml.analysis.transform import split_notes, transpose  # noqa: E402
from pianotree_dcml.model import PianoTreeVAE  # noqa: E402
from pianotree_dcml.representation import Note  # noqa: E402

TINY_MODEL = {"note_emb_size": 8, "enc_notes_hid_size": 8, "enc_time_hid_size": 8,
              "z_size": 4, "dec_emb_hid_size": 8, "dec_time_hid_size": 8,
              "dec_notes_hid_size": 8, "dec_z_in_size": 4, "dec_dur_hid_size": 4}


def test_transpose():
    notes = [Note(0, 60, 4), Note(2, 64, 1)]
    assert transpose(notes, 3) == [Note(0, 63, 4), Note(2, 67, 1)]
    assert transpose(notes, -12) == [Note(0, 48, 4), Note(2, 52, 1)]
    with pytest.raises(ValueError, match="MIDI range"):
        transpose([Note(0, 125, 1)], 3)


def test_split_notes():
    notes = [Note(0, 60, 4), Note(0, 64, 3), Note(4, 67, 1)]
    assert split_notes(notes) == [Note(0, 60, 2), Note(0, 64, 1), Note(1, 64, 2),
                                  Note(2, 60, 2), Note(4, 67, 1)]


def test_mean_difference_direction():
    rng = np.random.default_rng(0)
    values = rng.normal(size=2000)
    codes = rng.normal(size=(2000, 5))
    codes[:, 2] += 3 * values  # dimension 2 encodes the value
    values[:10] = np.nan
    direction = mean_difference_direction(codes, values)
    assert np.linalg.norm(direction) == pytest.approx(1)
    assert direction[2] > 0.95


def test_consistency_and_spread():
    direction = np.array([1.0, 0.0, 0.0])
    shifts = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, -3.0]])
    assert consistency(shifts, 5 * direction) == pytest.approx([1, 1 / np.sqrt(2), 0])
    codes = np.array([[1.0, 5, 5], [3.0, -5, 0]])
    assert spread_along(codes, direction) == pytest.approx(1.0)
    assert np.allclose(unit(np.array([3.0, 4.0])), [0.6, 0.8])


def test_encode_notes_and_move_and_measure():
    torch.manual_seed(0)
    model = PianoTreeVAE(seq_len=8, max_simu_note=6, **TINY_MODEL)
    bars = [[Note(0, 60, 4)], [Note(0, p, 1) for p in range(50, 60)]]  # 10 notes: truncated
    mean, scale = encode_notes(model, bars)
    assert mean.shape == scale.shape == (2, 4)

    amounts = np.array([-1.0, 0.0, 1.0])
    measured = move_and_measure(model, mean, unit(np.ones(4)), amounts, bar_len=8)
    assert len(measured) == 6
    assert measured.bar.tolist() == [0, 0, 0, 1, 1, 1]
    assert measured.amount.tolist() == [-1, 0, 1, -1, 0, 1]
    assert "pitch_mean" in measured
