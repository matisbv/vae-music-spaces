"""Symbolic representations of one bar of polyphonic music.

A bar is quantized to ``seq_len`` time steps (16th notes in this project). It can be
written in four forms:

notes
    A list of :class:`Note` ``(onset, pitch, duration)``: the onset step, the MIDI pitch
    (0-127) and the duration in steps (at least 1). This is the human-readable form.
duration matrix
    ``int`` array of shape ``(seq_len, 128)``. ``matrix[t, p]`` is the duration of the note
    with MIDI pitch ``p`` that starts at step ``t``, and 0 means no note. The data pipeline
    stores bars on disk in this compact form.
piano-grid
    ``int`` array of shape ``(seq_len, max_simu_note, 1 + dur_width)``, the input and
    target format of PianoTree-VAE. Each time step holds a sequence of note tokens
    ``<sos>, n_1, ..., n_k, <eos>, <pad>, ...`` with pitches in ascending order.
    Column 0 is the pitch token. Columns ``1:`` hold ``duration - 1`` in binary, most
    significant bit first. The special tokens use ``DUR_PAD`` in the duration columns.
piano-roll
    ``bool`` array of shape ``(seq_len, 128)`` that is True wherever a pitch sounds. It is
    lossy (repeated notes merge into one), so it is only used for plotting.

notes -> duration matrix -> piano-grid -> notes is lossless as long as a bar respects
three limits:

* at most one note per ``(onset, pitch)``. The duration matrix keeps only the longest
  note of a unison doubling;
* at most ``max_simu_note - 2`` notes per onset, since ``<sos>`` and ``<eos>`` take two
  slots;
* durations of at most ``seq_len`` steps. Longer durations are clipped.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, NamedTuple

import numpy as np

N_PITCHES = 128
PITCH_SOS = 128
PITCH_EOS = 129
PITCH_PAD = 130
DUR_PAD = 2

_PITCH_CLASS_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


class Note(NamedTuple):
    onset: int
    pitch: int
    duration: int


def dur_width_for(seq_len: int) -> int:
    """Number of bits needed to encode durations 1..seq_len (stored as duration - 1)."""
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    return math.ceil(math.log2(seq_len))


def pitch_name(pitch: int) -> str:
    """Scientific pitch name of a MIDI pitch, e.g. 60 -> 'C4'."""
    if not 0 <= pitch < N_PITCHES:
        raise ValueError(f"MIDI pitch must be in [0, 127], got {pitch}")
    return f"{_PITCH_CLASS_NAMES[pitch % 12]}{pitch // 12 - 1}"


def notes_to_duration_matrix(notes: Iterable[Note], seq_len: int) -> np.ndarray:
    """Write notes into a ``(seq_len, 128)`` duration matrix.

    If two notes share an onset and a pitch (e.g. the same note doubled in two staves),
    only the longer one is kept.
    """
    matrix = np.zeros((seq_len, N_PITCHES), dtype=np.int64)
    for onset, pitch, duration in notes:
        if not 0 <= onset < seq_len:
            raise ValueError(f"onset must be in [0, {seq_len}), got {onset}")
        if not 0 <= pitch < N_PITCHES:
            raise ValueError(f"MIDI pitch must be in [0, 127], got {pitch}")
        if duration < 1:
            raise ValueError(f"duration must be at least 1, got {duration}")
        matrix[onset, pitch] = max(matrix[onset, pitch], duration)
    return matrix


def duration_matrix_to_notes(matrix: np.ndarray) -> list[Note]:
    """Read the notes out of a duration matrix, sorted by onset, then pitch."""
    onsets, pitches = np.nonzero(matrix)
    return [Note(int(t), int(p), int(matrix[t, p])) for t, p in zip(onsets, pitches)]


def duration_matrix_to_grid(
    matrix: np.ndarray,
    max_simu_note: int = 16,
    dur_width: int | None = None,
    overflow: Literal["raise", "truncate"] = "raise",
) -> np.ndarray:
    """Encode a duration matrix as a PianoTree piano-grid.

    Durations longer than ``seq_len`` are clipped to ``seq_len``. An onset with more than
    ``max_simu_note - 2`` notes raises a ``ValueError``, unless
    ``overflow="truncate"``, which keeps the lowest pitches.
    """
    if overflow not in ("raise", "truncate"):
        raise ValueError(f"overflow must be 'raise' or 'truncate', got {overflow!r}")
    seq_len = matrix.shape[0]
    if dur_width is None:
        dur_width = dur_width_for(seq_len)
    if 2**dur_width < seq_len:
        raise ValueError(f"dur_width={dur_width} cannot encode durations up to {seq_len}")
    capacity = max_simu_note - 2

    grid = np.full((seq_len, max_simu_note, 1 + dur_width), DUR_PAD, dtype=np.int64)
    grid[:, :, 0] = PITCH_PAD
    grid[:, 0, 0] = PITCH_SOS
    for t in range(seq_len):
        pitches = np.nonzero(matrix[t])[0]
        if len(pitches) > capacity:
            if overflow == "raise":
                raise ValueError(
                    f"{len(pitches)} notes at step {t}, but max_simu_note={max_simu_note} "
                    f"leaves room for {capacity}"
                )
            pitches = pitches[:capacity]
        n = len(pitches)
        durations = np.minimum(matrix[t, pitches], seq_len)
        grid[t, 1 : n + 1, 0] = pitches
        grid[t, 1 : n + 1, 1:] = _to_bits(durations - 1, dur_width)
        grid[t, n + 1, 0] = PITCH_EOS
    return grid


def notes_to_grid(
    notes: Iterable[Note],
    seq_len: int,
    max_simu_note: int = 16,
    dur_width: int | None = None,
    overflow: Literal["raise", "truncate"] = "raise",
) -> np.ndarray:
    """Encode notes as a piano-grid (via the duration matrix)."""
    matrix = notes_to_duration_matrix(notes, seq_len)
    return duration_matrix_to_grid(matrix, max_simu_note, dur_width, overflow)


def grid_to_notes(grid: np.ndarray) -> list[Note]:
    """Decode a piano-grid into notes, sorted by onset, pitch, then duration.

    At each time step, reading stops at the first special token after ``<sos>``. Whatever
    comes after ``<eos>`` is ignored: in model outputs, those positions are never trained
    and contain garbage.
    """
    dur_width = grid.shape[-1] - 1
    weights = 1 << np.arange(dur_width - 1, -1, -1)
    notes = []
    for t in range(grid.shape[0]):
        for token in grid[t, 1:]:
            pitch = int(token[0])
            if pitch >= N_PITCHES:
                break
            if pitch < 0:
                raise ValueError(f"invalid pitch token {pitch} at step {t}")
            bits = token[1:]
            if np.any((bits != 0) & (bits != 1)):
                raise ValueError(f"invalid duration bits {bits.tolist()} at step {t}")
            notes.append(Note(t, pitch, int(bits @ weights) + 1))
    return sorted(notes)


def logits_to_grid(pitch_logits: np.ndarray, dur_logits: np.ndarray) -> np.ndarray:
    """Turn PianoTree-VAE decoder outputs into a piano-grid by taking the argmax.

    ``pitch_logits`` has shape ``(..., seq_len, max_simu_note - 1, n_pitch_tokens)`` and
    ``dur_logits`` has shape ``(..., seq_len, max_simu_note - 1, dur_width, 2)``. The
    decoder does not predict ``<sos>``, so it is prepended. Torch tensors must be moved to
    the CPU first.
    """
    pitch = np.argmax(np.asarray(pitch_logits), axis=-1)
    bits = np.argmax(np.asarray(dur_logits), axis=-1)
    body = np.concatenate([pitch[..., None], bits], axis=-1)
    sos = np.full(body.shape[:-2] + (1, body.shape[-1]), DUR_PAD, dtype=body.dtype)
    sos[..., 0] = PITCH_SOS
    return np.concatenate([sos, body], axis=-2)


def notes_to_pianoroll(notes: Iterable[Note], seq_len: int) -> np.ndarray:
    """Render notes as a boolean ``(seq_len, 128)`` piano-roll, clipped to the bar."""
    roll = np.zeros((seq_len, N_PITCHES), dtype=bool)
    for onset, pitch, duration in notes:
        roll[onset : onset + duration, pitch] = True
    return roll


def _to_bits(values: np.ndarray, width: int) -> np.ndarray:
    """Binary representation of each value, most significant bit first."""
    return (values[:, None] >> np.arange(width - 1, -1, -1)) & 1
