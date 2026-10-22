"""Turn the DCML corpora release into a table of bars and a table of notes.

The release zip holds one TSV per facet (notes, measures, harmony labels, metadata), each
covering all corpora. ``prepare`` turns it into two CSV files:

``bars.csv``
    One row per bar: where it comes from, its metadata, the harmony label at its start,
    simple statistics, and whether it belongs to the training or validation set.
``notes.csv``
    One row per note: ``bar_id``, ``onset`` and ``duration`` in 16th-note steps within the
    bar, and MIDI ``pitch``.

How the bars are built:

1. **Bars from measures.** A bar is one score measure (``mc``), plus the following
   measures that only complete it (``mc_offset > 0``), like the second half of a bar split
   by a repeat sign. Bar numbers (``mn``) can't be used as keys: first and second endings
   share them, and so do some renumbered measures.
2. **Notes onto the grid.** Grace notes are dropped. Tied notes are merged within a bar; a
   note tied over a barline starts again at onset 0 of the next bar. Onsets and durations
   are rounded down to 16th notes (so triplets are approximated), durations are at least
   one step, and notes are cut at the end of their bar. When two notes share an onset and a
   pitch (e.g. the same note in two staves), the longest one is kept.
3. **Cleaning.** Bars are dropped when they don't fit their time signature (free-time
   cadenzas, overfull measures, lengths that aren't a whole number of 16ths), and when they
   contain no notes.
4. **Split.** Whole pieces, not bars, go to the validation set, so that the model is never
   evaluated on bars from pieces it was trained on (pieces repeat many bars).
"""

from __future__ import annotations

import zipfile
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

STEPS_PER_WHOLE_NOTE = 16

_COLUMNS = {
    "notes": ["corpus", "piece", "mc", "mn_onset", "staff", "duration", "gracenote", "tied", "midi"],
    "measures": ["corpus", "piece", "mc", "mn", "volta", "timesig", "act_dur", "mc_offset"],
    "expanded": ["corpus", "piece", "mc", "mc_onset", "localkey", "chord", "numeral", "chord_type"],
    "metadata": ["corpus", "piece", "composer", "composed_end", "annotated_key"],
}

# Coarse genre of each corpus; anything not listed is solo piano music.
CORPUS_GENRES = {
    "ABC": "String quartet",
    "corelli": "Trio sonata",
    "chopin_mazurkas": "Mazurka",
}

ERAS = [(1600, "Renaissance"), (1750, "Baroque"), (1820, "Classical"), (1900, "Romantic")]


def read_release(zip_path: Path) -> dict[str, pd.DataFrame]:
    """Read the facets of the DCML release zip that the pipeline needs."""
    with zipfile.ZipFile(zip_path) as zf:
        return {
            facet: pd.read_csv(
                zf.open(f"dcml_corpora.{facet}.tsv"), sep="\t", usecols=columns, low_memory=False
            )
            for facet, columns in _COLUMNS.items()
        }


def prepare(
    zip_path: Path, out_dir: Path, valid_fraction: float = 0.1, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``bars.csv`` and ``notes.csv`` in ``out_dir`` from the release zip."""
    tables = read_release(zip_path)
    bars, notes = build_bars_and_notes(tables, valid_fraction, seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(out_dir / "bars.csv", index=False)
    notes.to_csv(out_dir / "notes.csv", index=False)

    counts = bars.split.value_counts()
    print(f"Wrote {len(bars)} bars ({counts.get('train', 0)} train, "
          f"{counts.get('valid', 0)} valid) and {len(notes)} notes to {out_dir}")
    return bars, notes


def build_bars_and_notes(
    tables: dict[str, pd.DataFrame], valid_fraction: float = 0.1, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars, mc_to_bar = group_measures_into_bars(tables["measures"])
    notes = place_notes_in_bars(tables["notes"], mc_to_bar)
    notes = notes.merge(bars[["bar_id", "bar_len"]], on="bar_id")

    irregular = (
        (bars.bar_len % 1 != 0)
        | (bars.actual_len > bars.bar_len)
        | bars.bar_id.isin(notes.bar_id[notes.onset >= notes.bar_len])
    )
    notes = quantize(notes[~notes.bar_id.isin(bars.bar_id[irregular])])
    empty = ~irregular & ~bars.bar_id.isin(notes.bar_id)
    print(f"{len(bars)} bars: dropping {irregular.sum()} that don't fit their time signature "
          f"and {empty.sum()} without notes")

    bars = bars[~irregular & ~empty].copy()
    bars["bar_len"] = bars.bar_len.astype(int)
    bars = (
        bars.merge(piece_metadata(tables["metadata"], tables["notes"]), on=["corpus", "piece"], how="left")
        .merge(harmony_at_bar_start(tables["expanded"], mc_to_bar), on="bar_id", how="left")
        .merge(bar_statistics(notes), on="bar_id")
    )
    bars["notes_per_step"] = bars.n_notes / bars.bar_len
    bars["split"] = split_by_piece(bars, valid_fraction, seed)

    # Renumber bars 0..n-1 so that bar_id is also the row index of bars.csv.
    new_ids = pd.Series(np.arange(len(bars)), index=bars.bar_id)
    bars["bar_id"] = np.arange(len(bars))
    notes["bar_id"] = notes.bar_id.map(new_ids)
    columns = ["bar_id", "corpus", "piece", "mc", "mn", "volta", "timesig", "bar_len", "split",
               "composer", "year", "era", "century", "genre", "key", "n_staffs", "localkey",
               "chord", "numeral", "chord_type", "n_notes", "notes_per_step", "mean_pitch",
               "pitch_range", "mean_duration"]
    return bars[columns], notes[["bar_id", "onset", "pitch", "duration"]]


def group_measures_into_bars(measures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group measures (``mc``) into bars. Returns the bars and an ``mc -> bar_id`` table."""
    m = measures.sort_values(["corpus", "piece", "mc"], ignore_index=True)
    is_first = m.groupby(["corpus", "piece"]).cumcount() == 0
    starts_bar = (_in_steps(m.mc_offset) == 0) | is_first
    m["bar_id"] = starts_bar.cumsum() - 1
    m["actual_len"] = _in_steps(m.act_dur)

    bars = m.groupby("bar_id", as_index=False).agg(
        corpus=("corpus", "first"),
        piece=("piece", "first"),
        mc=("mc", "first"),
        mn=("mn", "first"),
        volta=("volta", "first"),
        timesig=("timesig", "first"),
        actual_len=("actual_len", "sum"),
    )
    bars["volta"] = bars.volta.astype("Int64")
    bars["bar_len"] = _in_steps(bars.timesig)
    return bars, m[["corpus", "piece", "mc", "bar_id"]]


def place_notes_in_bars(notes: pd.DataFrame, mc_to_bar: pd.DataFrame) -> pd.DataFrame:
    """Assign notes to bars, drop grace notes and merge tied notes within each bar.

    Onsets and durations are returned in (fractional) 16th-note steps.
    """
    notes = notes[notes.gracenote.isna()].merge(mc_to_bar, on=["corpus", "piece", "mc"])
    notes = pd.DataFrame({
        "bar_id": notes.bar_id,
        "staff": notes.staff,
        "pitch": notes.midi,
        "tied": notes.tied,
        "onset": _in_steps(notes.mn_onset),
        "duration": _in_steps(notes.duration),
    }).sort_values(["bar_id", "staff", "pitch", "onset"], ignore_index=True)

    # tied == -1 ends a tie and 0 continues one: such notes extend the previous note of the
    # same pitch in the same staff. A continuation with no such note in its bar (tied over
    # the barline) starts a new note.
    key = ["bar_id", "staff", "pitch"]
    continues = notes.tied.isin([-1, 0]) & (notes.groupby(key).cumcount() > 0)
    return (
        notes.groupby((~continues).cumsum())
        .agg(bar_id=("bar_id", "first"), pitch=("pitch", "first"),
             onset=("onset", "first"), duration=("duration", "sum"))
        .reset_index(drop=True)
    )


def quantize(notes: pd.DataFrame) -> pd.DataFrame:
    """Round onsets and durations down to whole steps, cut notes at the end of their bar and
    keep only the longest of notes sharing an onset and pitch."""
    eps = 1e-9  # guards against float error on triplets, e.g. 3 * (1/3) == 0.999...
    onset = np.floor(notes.onset + eps).astype(int)
    duration = np.maximum(np.floor(notes.duration + eps), 1).astype(int)
    notes = pd.DataFrame({
        "bar_id": notes.bar_id,
        "onset": onset,
        "pitch": notes.pitch.astype(int),
        "duration": np.minimum(duration, notes.bar_len.astype(int) - onset),
    })
    return (
        notes.sort_values("duration", ascending=False)
        .drop_duplicates(["bar_id", "onset", "pitch"])
        .sort_values(["bar_id", "onset", "pitch"], ignore_index=True)
    )


def piece_metadata(metadata: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    pieces = metadata.rename(columns={"composed_end": "year", "annotated_key": "key"})
    pieces["era"] = pieces.year.map(era)
    pieces["century"] = (pieces.year // 100 + 1).map(lambda c: f"{c}th")
    pieces["genre"] = pieces.corpus.map(CORPUS_GENRES).fillna("Piano piece")
    n_staffs = notes.groupby(["corpus", "piece"]).staff.nunique().rename("n_staffs")
    return pieces.merge(n_staffs.reset_index(), on=["corpus", "piece"], how="left")


def era(year: int) -> str:
    for end, name in ERAS:
        if year < end:
            return name
    return "Modern"


def harmony_at_bar_start(expanded: pd.DataFrame, mc_to_bar: pd.DataFrame) -> pd.DataFrame:
    """Harmony label of each bar: the first label in the bar or, if the bar has none, the
    last label before it in the same piece."""
    columns = ["localkey", "chord", "numeral", "chord_type"]
    labels = expanded.dropna(subset=["chord"]).merge(mc_to_bar, on=["corpus", "piece", "mc"])
    labels["position"] = _in_steps(labels.mc_onset)
    labels = labels.sort_values(["bar_id", "mc", "position"])

    all_bars = mc_to_bar.drop_duplicates("bar_id").set_index("bar_id")
    first = labels.groupby("bar_id")[columns].first().reindex(all_bars.index)
    last = labels.groupby("bar_id")[columns].last().reindex(all_bars.index)
    previous = last.groupby([all_bars.corpus, all_bars.piece]).ffill()
    previous = previous.groupby([all_bars.corpus, all_bars.piece]).shift(1)

    has_label = first.chord.notna()
    return first.where(has_label, previous, axis=0).reset_index()


def bar_statistics(notes: pd.DataFrame) -> pd.DataFrame:
    stats = notes.groupby("bar_id").agg(
        n_notes=("pitch", "size"),
        mean_pitch=("pitch", "mean"),
        lowest=("pitch", "min"),
        highest=("pitch", "max"),
        mean_duration=("duration", "mean"),
    )
    stats["pitch_range"] = stats.highest - stats.lowest
    return stats.drop(columns=["lowest", "highest"]).reset_index()


def split_by_piece(bars: pd.DataFrame, valid_fraction: float = 0.1, seed: int = 0) -> pd.Series:
    """Assign whole pieces to 'train' or 'valid', aiming for ``valid_fraction`` of the bars
    of each corpus. A corpus with a single piece stays entirely in 'train'.

    Each corpus is shuffled with its own seed, so adding or removing a corpus does not
    change how the others are split.
    """
    split = pd.Series("train", index=bars.index)
    for corpus, group in bars.groupby("corpus"):
        sizes = group.groupby("piece").size()
        if len(sizes) < 2:
            continue
        rng = np.random.default_rng([seed, zlib.crc32(corpus.encode())])
        sizes = sizes.iloc[rng.permutation(len(sizes))]
        # Take pieces until the target is reached, but always leave one for training.
        bars_before = sizes.cumsum() - sizes
        valid_pieces = sizes.index[bars_before < valid_fraction * sizes.sum()][: len(sizes) - 1]
        split[group.index[group.piece.isin(valid_pieces)]] = "valid"
    return split


def _in_steps(values: pd.Series) -> pd.Series:
    """Convert fractions of a whole note written as strings ('3/8') to 16th-note steps."""
    as_str = values.astype(str)
    lookup = {s: float(Fraction(s) * STEPS_PER_WHOLE_NOTE) for s in as_str.unique()}
    return as_str.map(lookup)
