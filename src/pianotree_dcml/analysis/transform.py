"""Musical transformations of bars, whose effect we know exactly."""

from __future__ import annotations

from typing import Sequence

from pianotree_dcml.representation import N_PITCHES, Note


def transpose(notes: Sequence[Note], semitones: int) -> list[Note]:
    """Move every note up (or down, if negative) by ``semitones``."""
    moved = [Note(onset, pitch + semitones, duration) for onset, pitch, duration in notes]
    if any(not 0 <= n.pitch < N_PITCHES for n in moved):
        raise ValueError(f"transposing by {semitones} leaves the MIDI range")
    return moved


def split_notes(notes: Sequence[Note]) -> list[Note]:
    """Split every note of two steps or more into two shorter notes of the same pitch:
    the same harmony, played with a busier rhythm. A note of length 3 becomes 1 + 2."""
    result = []
    for onset, pitch, duration in notes:
        if duration < 2:
            result.append(Note(onset, pitch, duration))
            continue
        first = duration // 2
        result += [Note(onset, pitch, first), Note(onset + first, pitch, duration - first)]
    return sorted(result)
