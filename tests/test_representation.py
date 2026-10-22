import numpy as np
import pytest

from pianotree_dcml.representation import (
    DUR_PAD,
    PITCH_EOS,
    PITCH_PAD,
    PITCH_SOS,
    Note,
    dur_width_for,
    duration_matrix_to_grid,
    duration_matrix_to_notes,
    grid_to_notes,
    logits_to_grid,
    notes_to_duration_matrix,
    notes_to_grid,
    notes_to_pianoroll,
    pitch_name,
)


def random_bar(rng, seq_len, max_notes_per_onset):
    """Random valid bar: unique (onset, pitch) pairs, durations in 1..seq_len."""
    notes = []
    for onset in range(seq_len):
        n = rng.integers(0, max_notes_per_onset + 1)
        pitches = rng.choice(128, size=n, replace=False)
        durations = rng.integers(1, seq_len + 1, size=n)
        notes += [Note(onset, int(p), int(d)) for p, d in zip(pitches, durations)]
    return notes


@pytest.mark.parametrize(
    ("seq_len", "expected"), [(2, 1), (6, 3), (12, 4), (16, 4), (17, 5), (24, 5), (36, 6)]
)
def test_dur_width_for(seq_len, expected):
    assert dur_width_for(seq_len) == expected


def test_dur_width_for_rejects_tiny_bars():
    with pytest.raises(ValueError):
        dur_width_for(1)


def test_pitch_name():
    assert pitch_name(60) == "C4"
    assert pitch_name(69) == "A4"
    assert pitch_name(0) == "C-1"
    assert pitch_name(127) == "G9"


def test_grid_layout_of_c_major_triad():
    # Regression test: the 2023 pipeline stored MIDI pitch p at index p - 1, so the
    # model saw everything one semitone too low.
    grid = notes_to_grid([Note(0, 60, 4), Note(0, 64, 4), Note(0, 67, 4)], seq_len=16)

    assert grid.shape == (16, 16, 5)
    assert grid[0, :, 0].tolist() == [PITCH_SOS, 60, 64, 67, PITCH_EOS] + [PITCH_PAD] * 11
    assert grid[0, 1:4, 1:].tolist() == [[0, 0, 1, 1]] * 3  # duration 4 -> bits of 3
    assert (grid[0, 0, 1:] == DUR_PAD).all()
    assert (grid[0, 4:, 1:] == DUR_PAD).all()
    # Empty steps are just <sos> <eos>.
    assert grid[1, :3, 0].tolist() == [PITCH_SOS, PITCH_EOS, PITCH_PAD]


def test_pitches_are_sorted_within_a_step():
    grid = notes_to_grid([Note(3, 72, 1), Note(3, 48, 2), Note(3, 60, 3)], seq_len=8)
    assert grid[3, 1:4, 0].tolist() == [48, 60, 72]


@pytest.mark.parametrize("seq_len", [4, 6, 8, 12, 16, 24, 36])
def test_round_trip_notes_matrix_grid(seq_len):
    rng = np.random.default_rng(seq_len)
    max_simu_note = 16
    for _ in range(50):
        notes = random_bar(rng, seq_len, max_notes_per_onset=max_simu_note - 2)

        matrix = notes_to_duration_matrix(notes, seq_len)
        assert duration_matrix_to_notes(matrix) == sorted(notes)

        grid = duration_matrix_to_grid(matrix, max_simu_note)
        assert grid_to_notes(grid) == sorted(notes)


def test_unison_doubling_keeps_the_longest_note():
    matrix = notes_to_duration_matrix([Note(0, 60, 2), Note(0, 60, 6)], seq_len=8)
    assert duration_matrix_to_notes(matrix) == [Note(0, 60, 6)]


def test_durations_longer_than_the_bar_are_clipped():
    grid = notes_to_grid([Note(2, 60, 40)], seq_len=12)
    assert grid_to_notes(grid) == [Note(2, 60, 12)]


def test_too_many_notes_at_one_onset_raises():
    notes = [Note(0, p, 1) for p in range(40, 55)]  # 15 notes, room for 14
    with pytest.raises(ValueError, match="15 notes at step 0"):
        notes_to_grid(notes, seq_len=16, max_simu_note=16)


def test_truncate_keeps_the_lowest_pitches():
    notes = [Note(0, p, 1) for p in range(40, 55)]
    grid = notes_to_grid(notes, seq_len=16, max_simu_note=16, overflow="truncate")
    assert grid_to_notes(grid) == notes[:14]
    assert grid[0, 15, 0] == PITCH_EOS


def test_too_narrow_dur_width_raises():
    with pytest.raises(ValueError, match="dur_width"):
        notes_to_grid([Note(0, 60, 1)], seq_len=16, dur_width=3)


@pytest.mark.parametrize(
    "note", [Note(-1, 60, 1), Note(16, 60, 1), Note(0, 128, 1), Note(0, -1, 1), Note(0, 60, 0)]
)
def test_invalid_notes_raise(note):
    with pytest.raises(ValueError):
        notes_to_duration_matrix([note], seq_len=16)


def test_decoding_ignores_tokens_after_eos():
    # The model predicts every slot, including those after <eos>, which are never
    # trained. The 2023 metrics counted them as notes, which deflated precision.
    grid = notes_to_grid([Note(0, 60, 4)], seq_len=8)
    grid[0, 3] = [70, 0, 0, 1]  # garbage after <eos> at slot 2
    assert grid_to_notes(grid) == [Note(0, 60, 4)]


def test_decoding_rejects_padded_duration_on_a_note():
    grid = notes_to_grid([Note(0, 60, 4)], seq_len=8)
    grid[0, 1, 1] = DUR_PAD
    with pytest.raises(ValueError, match="duration bits"):
        grid_to_notes(grid)


def test_logits_to_grid_round_trip():
    rng = np.random.default_rng(0)
    seq_len, max_simu_note, n_pitch_tokens = 12, 16, 130
    batch = [random_bar(rng, seq_len, max_notes_per_onset=5) for _ in range(3)]
    grids = np.stack([notes_to_grid(notes, seq_len, max_simu_note) for notes in batch])
    dur_width = grids.shape[-1] - 1

    # Build one-hot "logits" that decode to the targets; padded slots get arbitrary
    # garbage, as real model outputs would.
    body = grids[:, :, 1:]
    pitch_targets = np.where(body[..., 0] == PITCH_PAD, 70, body[..., 0])
    bit_targets = np.where(body[..., 1:] == DUR_PAD, 1, body[..., 1:])
    pitch_logits = np.eye(n_pitch_tokens)[pitch_targets]
    dur_logits = np.eye(2)[bit_targets]
    assert pitch_logits.shape == (3, seq_len, max_simu_note - 1, n_pitch_tokens)
    assert dur_logits.shape == (3, seq_len, max_simu_note - 1, dur_width, 2)

    decoded = logits_to_grid(pitch_logits, dur_logits)
    assert decoded.shape == grids.shape
    assert (decoded[:, :, 0, 0] == PITCH_SOS).all()
    for notes, grid in zip(batch, decoded):
        assert grid_to_notes(grid) == sorted(notes)


def test_pianoroll():
    roll = notes_to_pianoroll([Note(0, 60, 3), Note(2, 64, 10)], seq_len=8)
    assert roll.shape == (8, 128)
    assert np.flatnonzero(roll[:, 60]).tolist() == [0, 1, 2]
    assert np.flatnonzero(roll[:, 64]).tolist() == [2, 3, 4, 5, 6, 7]  # clipped to the bar
    assert roll.sum() == 9
