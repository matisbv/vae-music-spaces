"""Simple, human-readable descriptions of bars.

They serve two purposes in the analysis:

- **baselines**: if a question (e.g. "which composer wrote this bar?") can be answered
  from these features as well as from the latent code, the model hasn't learned anything
  beyond them;
- **descriptions**: when we move a bar through the latent space, these features tell
  what changed in the music (higher? denser? more dissonant?).
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd

from pianotree_dcml.representation import Note, pitch_name

PITCH_CLASSES = [re.sub(r"-?\d+$", "", pitch_name(60 + i)) for i in range(12)]

# Dissonant intervals (in semitones, ignoring octaves): those that sound tense when played
# together, i.e. the semitone, whole tone, tritone, minor and major seventh.
DISSONANT_INTERVALS = [1, 2, 6, 10, 11]

FEATURES = {
    "n_notes": "number of notes",
    "note_density": "notes per 16th-note step",
    "onset_density": "share of 16th-note steps where at least one note starts",
    "chord_size": "average number of notes starting together",
    "max_chord_size": "largest number of notes starting together",
    "pitch_mean": "average pitch, in MIDI numbers (60 = middle C): the register",
    "pitch_std": "spread of the pitches, in semitones",
    "pitch_lowest": "lowest pitch",
    "pitch_highest": "highest pitch",
    "pitch_range": "distance between the lowest and highest pitch, in semitones",
    "duration_mean": "average note length, in 16th notes",
    "duration_std": "spread of note lengths, in 16th notes",
    "sounding_fraction": "share of the bar during which at least one note sounds",
    "offbeat_onsets": "share of notes starting on an odd 16th note, between eighth-note positions",
    "n_pitch_classes": "number of distinct note names (C, C#, ..., B), ignoring the octave",
    "pitch_class_entropy": "how evenly the bar spreads over the 12 note names, in bits: "
                           "low for a few repeated notes, high for chromatic music",
    "top_line_leap": "average jump between successive highest notes, in semitones: "
                     "a smooth or a leaping melody",
    "dissonance": "share of pairs of notes starting together that are dissonant, i.e. sound "
                  "tense (a semitone, whole tone, tritone or seventh apart)",
} | {f"pc_{name}": f"share of notes that are a {name}" for name in PITCH_CLASSES}


def notes_table(bars: Sequence[Sequence[Note]]) -> pd.DataFrame:
    """Notes of a list of bars (e.g. decoded ones) as a table; ``bar`` is the position
    of their bar in the list."""
    rows = [(i, *note) for i, notes in enumerate(bars) for note in notes]
    return pd.DataFrame(rows, columns=["bar", "onset", "pitch", "duration"])


def bar_features(notes: pd.DataFrame, bar_len: pd.Series | Sequence[int]) -> pd.DataFrame:
    """Features (see ``FEATURES``) of every bar.

    ``notes`` has the columns ``bar``, ``onset``, ``pitch`` and ``duration``.
    ``bar_len`` gives the length of each bar in 16th notes, indexed like ``bar``; bars
    without notes get zero counts and missing values elsewhere.
    """
    bar_len = pd.Series(bar_len) if not isinstance(bar_len, pd.Series) else bar_len
    by_bar = notes.groupby("bar")
    chords = notes.groupby(["bar", "onset"]).pitch
    chord_sizes = chords.size()
    features = pd.DataFrame({
        "n_notes": by_bar.size(),
        "onset_density": chord_sizes.groupby(level="bar").size(),
        "chord_size": chord_sizes.groupby(level="bar").mean(),
        "max_chord_size": chord_sizes.groupby(level="bar").max(),
        "pitch_mean": by_bar.pitch.mean(),
        "pitch_std": by_bar.pitch.std(ddof=0),
        "pitch_lowest": by_bar.pitch.min(),
        "pitch_highest": by_bar.pitch.max(),
        "duration_mean": by_bar.duration.mean(),
        "duration_std": by_bar.duration.std(ddof=0),
        "offbeat_onsets": (notes.onset % 2 == 1).groupby(notes.bar).mean(),
    }).reindex(bar_len.index)
    features["n_notes"] = features.n_notes.fillna(0)
    features["note_density"] = features.n_notes / bar_len
    features["onset_density"] = features.onset_density.fillna(0) / bar_len
    features["pitch_range"] = features.pitch_highest - features.pitch_lowest
    features["sounding_fraction"] = _sounding_steps(notes, bar_len) / bar_len

    pitch_classes = (notes.pitch % 12).groupby(notes.bar).value_counts().unstack(fill_value=0)
    shares = pitch_classes.reindex(columns=range(12), fill_value=0).div(
        pitch_classes.sum(axis=1), axis=0)
    features["n_pitch_classes"] = (shares > 0).sum(axis=1).reindex(bar_len.index).fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(shares * np.log2(shares)).fillna(0).sum(axis=1)
    features["pitch_class_entropy"] = entropy.clip(lower=0)  # no "-0.0" for one-note bars
    features["top_line_leap"] = (
        chords.max().groupby(level="bar").diff().abs().groupby(level="bar").mean())
    features["dissonance"] = _dissonance(notes)
    for pc, name in enumerate(PITCH_CLASSES):
        features[f"pc_{name}"] = shares[pc]
    return features[list(FEATURES)]


def _sounding_steps(notes: pd.DataFrame, bar_len: pd.Series) -> pd.Series:
    """Number of 16th-note steps of each bar during which at least one note sounds."""
    end = np.minimum(notes.onset + notes.duration, notes.bar.map(bar_len))
    lengths = np.maximum(end - notes.onset, 0).to_numpy()
    bars = np.repeat(notes.bar.to_numpy(), lengths)
    steps = np.repeat(notes.onset.to_numpy(), lengths) + _ranges(lengths)
    sounding = pd.DataFrame({"bar": bars, "step": steps}).drop_duplicates()
    return sounding.groupby("bar").size().reindex(bar_len.index, fill_value=0)


def _ranges(lengths: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(n)`` for each ``n`` in ``lengths``."""
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.arange(lengths.sum()) - starts


def _dissonance(notes: pd.DataFrame) -> pd.Series:
    pairs = notes[["bar", "onset", "pitch"]].merge(notes[["bar", "onset", "pitch"]],
                                                   on=["bar", "onset"], suffixes=("", "_other"))
    pairs = pairs[pairs.pitch < pairs.pitch_other]
    dissonant = ((pairs.pitch_other - pairs.pitch) % 12).isin(DISSONANT_INTERVALS)
    return dissonant.groupby(pairs.bar).mean()
