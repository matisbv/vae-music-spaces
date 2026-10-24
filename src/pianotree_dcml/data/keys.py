"""Keys as written in the DCML annotations: 'C', 'Eb', 'F#' (major) or 'c', 'eb', 'f#' (minor)."""

from __future__ import annotations

_NATURAL_PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def tonic_pitch_class(key: str) -> int:
    """Pitch class (0 = C, 1 = C#/Db, ..., 11 = B) of a key's tonic."""
    letter, accidentals = key[0].lower(), key[1:]
    if letter not in _NATURAL_PITCH_CLASSES or set(accidentals) - {"#", "b"}:
        raise ValueError(f"not a key: {key!r}")
    return (_NATURAL_PITCH_CLASSES[letter] + accidentals.count("#") - accidentals.count("b")) % 12


def is_minor(key: str) -> bool:
    return key[0].islower()


def shift_to_c(key: str) -> int:
    """Semitones to transpose music in ``key`` so that its tonic becomes C (C major or
    C minor), choosing the smaller move: between 6 down and 5 up."""
    tonic = tonic_pitch_class(key)
    return -tonic if tonic <= 6 else 12 - tonic
