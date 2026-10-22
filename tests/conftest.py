"""A tiny synthetic DCML release covering the edge cases found in the real corpus."""

import zipfile

import pandas as pd
import pytest


def note(piece, mc, mn_onset, midi, duration, staff=1, tied=None, gracenote=None, corpus="c1"):
    return dict(corpus=corpus, piece=piece, mc=mc, mn_onset=mn_onset, staff=staff,
                duration=duration, gracenote=gracenote, tied=tied, midi=midi)


def measure(piece, mc, mn, timesig="3/4", act_dur=None, mc_offset="0", volta=None, corpus="c1"):
    return dict(corpus=corpus, piece=piece, mc=mc, mn=mn, volta=volta, timesig=timesig,
                act_dur=act_dur or timesig, mc_offset=mc_offset)


def label(piece, mc, mc_onset, chord, corpus="c1"):
    return dict(corpus=corpus, piece=piece, mc=mc, mc_onset=mc_onset, localkey="I",
                chord=chord, numeral=chord.rstrip("7"), chord_type="M")


def piece(name, composer, year, key, corpus="c1"):
    return dict(corpus=corpus, piece=name, composer=composer, composed_end=year, annotated_key=key)


@pytest.fixture
def release_tables():
    """Piece p1 is in 3/4 (12 steps per bar), p2 in 2/4 (8 steps) with an anacrusis."""
    notes = [
        # p1, bar mc=1: a grace note, a tie within the bar, a unison doubling and a tie
        # into the next bar.
        note("p1", 1, "0", 60, "1/4"),
        note("p1", 1, "0", 64, "0", gracenote="grace16"),
        note("p1", 1, "0", 69, "1/8", staff=1),
        note("p1", 1, "0", 69, "1/4", staff=2),
        note("p1", 1, "1/4", 62, "1/8", tied=1),
        note("p1", 1, "3/8", 62, "1/8", tied=-1),
        note("p1", 1, "1/2", 67, "1/4", tied=1),
        # p1, bar mc=2: the end of the tie, triplets, and a note running past the bar end.
        note("p1", 2, "0", 67, "1/4", tied=-1),
        note("p1", 2, "1/4", 71, "1/12"),
        note("p1", 2, "1/3", 72, "1/12"),
        note("p1", 2, "5/12", 74, "1/12"),
        note("p1", 2, "1/2", 76, "1/2"),
        # p1, one bar split by a repeat sign into mc=3 and mc=4.
        note("p1", 3, "0", 60, "1/2"),
        note("p1", 4, "1/2", 65, "1/4"),
        # p1, first and second endings, both numbered mn=4.
        note("p1", 5, "0", 60, "1/4"),
        note("p1", 6, "0", 62, "1/4"),
        # p1, mc=7 is empty; mc=8 is a cadenza whose notes overflow the time signature.
        note("p1", 8, "0", 60, "1/4"),
        note("p1", 8, "3/4", 62, "1/4"),
        # p2: an eighth-note anacrusis, then a full bar.
        note("p2", 1, "3/8", 67, "1/8"),
        note("p2", 2, "0", 72, "1/2"),
    ]
    measures = [
        measure("p1", 1, 1),
        measure("p1", 2, 2),
        measure("p1", 3, 3, act_dur="1/2"),
        measure("p1", 4, 3, act_dur="1/4", mc_offset="1/2"),
        measure("p1", 5, 4, volta=1),
        measure("p1", 6, 4, volta=2),
        measure("p1", 7, 5),
        measure("p1", 8, 6),
        measure("p2", 1, 0, timesig="2/4", act_dur="1/8", mc_offset="3/8"),
        measure("p2", 2, 1, timesig="2/4"),
    ]
    expanded = [
        label("p1", 1, "0", "I"),
        label("p1", 1, "1/2", "V7"),
        label("p1", 3, "1/4", "IV"),
        label("p2", 2, "0", "I"),
    ]
    metadata = [
        piece("p1", "Composer A", 1790, "C"),
        piece("p2", "Composer B", 1850, "G"),
    ]
    return {
        "notes": pd.DataFrame(notes),
        "measures": pd.DataFrame(measures),
        "expanded": pd.DataFrame(expanded),
        "metadata": pd.DataFrame(metadata),
    }


@pytest.fixture
def release_zip(tmp_path, release_tables):
    path = tmp_path / "dcml_corpora.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for facet, table in release_tables.items():
            zf.writestr(f"dcml_corpora.{facet}.tsv", table.to_csv(sep="\t", index=False))
    return path
