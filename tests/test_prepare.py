import pandas as pd
import pytest

from pianotree_dcml.data.prepare import build_bars_and_notes, era, prepare, split_by_piece


@pytest.fixture
def prepared(release_tables):
    return build_bars_and_notes(release_tables)


def notes_of(prepared, piece, mc):
    bars, notes = prepared
    bar_id = bars.bar_id[(bars.piece == piece) & (bars.mc == mc)].item()
    rows = notes[notes.bar_id == bar_id]
    return list(zip(rows.onset, rows.pitch, rows.duration))


def test_bars(prepared):
    bars, _ = prepared
    # mc=4 completes the bar started by mc=3; mc=7 (empty) and mc=8 (cadenza) are dropped.
    assert bars[["piece", "mc", "bar_len"]].values.tolist() == [
        ["p1", 1, 12], ["p1", 2, 12], ["p1", 3, 12], ["p1", 5, 12], ["p1", 6, 12],
        ["p2", 1, 8], ["p2", 2, 8],
    ]
    assert bars.bar_id.tolist() == list(range(len(bars)))


def test_grace_notes_ties_and_unisons(prepared):
    assert notes_of(prepared, "p1", 1) == [
        (0, 60, 4),
        (0, 69, 4),  # unison doubling in two staves: the longer note is kept
        (4, 62, 4),  # two tied eighths merged into a quarter
        (8, 67, 4),  # the grace note is gone
    ]


def test_ties_over_barlines_triplets_and_clipping(prepared):
    assert notes_of(prepared, "p1", 2) == [
        (0, 67, 4),  # tied from the previous bar: starts again at onset 0
        (4, 71, 1),  # triplet 16ths: onsets and durations rounded down
        (5, 72, 1),
        (6, 74, 1),
        (8, 76, 4),  # a half note at step 8 of a 12-step bar is cut at the bar end
    ]


def test_bar_split_by_a_repeat_sign(prepared):
    assert notes_of(prepared, "p1", 3) == [(0, 60, 8), (8, 65, 4)]


def test_voltas_are_separate_bars(prepared):
    bars, _ = prepared
    assert bars.volta[bars.mn == 4].tolist() == [1, 2]
    assert notes_of(prepared, "p1", 5) == [(0, 60, 4)]
    assert notes_of(prepared, "p1", 6) == [(0, 62, 4)]


def test_anacrusis_notes_sit_at_the_end_of_the_bar(prepared):
    assert notes_of(prepared, "p2", 1) == [(6, 67, 2)]


def test_piece_metadata(prepared):
    bars, _ = prepared
    first = bars.groupby("piece").first()
    assert first.composer.to_dict() == {"p1": "Composer A", "p2": "Composer B"}
    assert first.year.to_dict() == {"p1": 1790, "p2": 1850}
    assert first.era.to_dict() == {"p1": "Classical", "p2": "Romantic"}
    assert first.century.to_dict() == {"p1": "18th", "p2": "19th"}
    assert first.key.to_dict() == {"p1": "C", "p2": "G"}
    assert first.n_staffs.to_dict() == {"p1": 2, "p2": 1}
    assert (bars.genre == "Piano piece").all()


def test_harmony_at_bar_start(prepared):
    bars, _ = prepared
    # A bar without a label of its own takes the last label before it in the piece.
    assert bars.chord.fillna("none").tolist() == ["I", "V7", "IV", "IV", "IV", "none", "I"]
    assert bars.numeral.tolist()[:2] == ["I", "V"]


def test_bar_statistics(prepared):
    bars, _ = prepared
    first = bars.iloc[0]
    assert first.n_notes == 4
    assert first.notes_per_step == pytest.approx(4 / 12)
    assert first.mean_pitch == pytest.approx((60 + 69 + 62 + 67) / 4)
    assert first.pitch_range == 9
    assert first.mean_duration == 4


@pytest.mark.parametrize(
    ("year", "expected"),
    [(1599, "Renaissance"), (1600, "Baroque"), (1749, "Baroque"), (1750, "Classical"),
     (1819, "Classical"), (1820, "Romantic"), (1899, "Romantic"), (1900, "Modern")],
)
def test_era(year, expected):
    assert era(year) == expected


def test_prepare_writes_csvs(release_zip, tmp_path):
    out_dir = tmp_path / "processed"
    bars, notes = prepare(release_zip, out_dir)

    written_bars = pd.read_csv(out_dir / "bars.csv")
    written_notes = pd.read_csv(out_dir / "notes.csv")
    assert written_bars.bar_id.tolist() == bars.bar_id.tolist()
    assert written_bars.columns.tolist() == bars.columns.tolist()
    assert written_notes.values.tolist() == notes.values.tolist()


def make_bars(sizes):
    """A bars table with the given number of bars per (corpus, piece)."""
    rows = [(corpus, piece) for (corpus, piece), n in sizes.items() for _ in range(n)]
    return pd.DataFrame(rows, columns=["corpus", "piece"])


def test_split_keeps_pieces_whole_and_respects_the_fraction():
    sizes = {("a", f"p{i}"): 10 + i for i in range(40)} | {("b", f"q{i}"): 20 for i in range(30)}
    bars = make_bars(sizes)
    bars["split"] = split_by_piece(bars, valid_fraction=0.1)

    assert (bars.groupby(["corpus", "piece"]).split.nunique() == 1).all()
    share = bars.groupby("corpus").split.apply(lambda s: (s == "valid").mean())
    # Whole pieces are taken until the target is reached, overshooting by less than one piece.
    assert share["a"] == pytest.approx(0.1, abs=0.05)
    assert share["b"] == pytest.approx(0.1, abs=0.035)


def test_split_does_not_depend_on_other_corpora():
    sizes = {("a", f"p{i}"): 5 for i in range(20)}
    alone = split_by_piece(make_bars(sizes))
    with_other = split_by_piece(make_bars(sizes | {("b", f"q{i}"): 5 for i in range(20)}))
    assert alone.tolist() == with_other.tolist()[: len(alone)]
    assert split_by_piece(make_bars(sizes), seed=1).tolist() != alone.tolist()


def test_split_keeps_at_least_one_training_piece_per_corpus():
    bars = make_bars({("a", "only"): 50, ("b", "p1"): 5, ("b", "p2"): 5})
    split = split_by_piece(bars, valid_fraction=0.9)
    assert (split[bars.corpus == "a"] == "train").all()
    assert set(split[bars.corpus == "b"]) == {"train", "valid"}
