import wave

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("pretty_midi", reason="needs the analysis extra")
pytest.importorskip("matplotlib", reason="needs the analysis extra")

from pianotree_dcml.analysis.audio import (  # noqa: E402
    SAMPLE_RATE, save_mp3, save_wav, synthesize, synthesize_midi, to_midi)
from pianotree_dcml.analysis.decode import decode, lerp, slerp  # noqa: E402
from pianotree_dcml.analysis.encode import (  # noqa: E402
    dataset_for, default_output_path, encode_bars, load_embeddings, save_embeddings)
from pianotree_dcml.analysis.features import FEATURES, bar_features, notes_table  # noqa: E402
from pianotree_dcml.analysis.plots import plot_bars  # noqa: E402
from pianotree_dcml.cli import main  # noqa: E402
from pianotree_dcml.config import Config  # noqa: E402
from pianotree_dcml.data.dataset import BarDataset  # noqa: E402
from pianotree_dcml.data.keys import is_minor, shift_to_c, tonic_pitch_class  # noqa: E402
from pianotree_dcml.model import PianoTreeVAE  # noqa: E402
from pianotree_dcml.representation import Note, grid_to_notes  # noqa: E402
from pianotree_dcml.train import train  # noqa: E402

TINY_MODEL = {"note_emb_size": 8, "enc_notes_hid_size": 8, "enc_time_hid_size": 8,
              "z_size": 4, "dec_emb_hid_size": 8, "dec_time_hid_size": 8,
              "dec_notes_hid_size": 8, "dec_z_in_size": 4, "dec_dur_hid_size": 4}
KEYS = ["C", "G", "f#", None, "Eb", "a"]


@pytest.fixture
def data_dir(tmp_path):
    """Six 2/4 bars, one per key in KEYS (one bar has no annotated key)."""
    rng = np.random.default_rng(0)
    bars, notes = [], []
    for bar_id, key in enumerate(KEYS):
        bars.append((bar_id, "a", f"p{bar_id}", "2/4", 8, "train" if bar_id < 4 else "valid", key))
        for onset in range(0, 8, 2):
            notes.append((bar_id, onset, int(rng.integers(55, 75)), 2))
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    pd.DataFrame(bars, columns=["bar_id", "corpus", "piece", "timesig", "bar_len", "split",
                                "key"]).to_csv(data_dir / "bars.csv", index=False)
    pd.DataFrame(notes, columns=["bar_id", "onset", "pitch", "duration"]).to_csv(
        data_dir / "notes.csv", index=False)
    return data_dir


@pytest.fixture
def checkpoint(data_dir, tmp_path):
    config = Config.from_dict({
        "data": {"data_dir": str(data_dir), "seq_len": 8, "max_simu_note": 6},
        "model": TINY_MODEL,
        "train": {"epochs": 1, "batch_size": 2},
    })
    train(config, tmp_path / "run")
    return tmp_path / "run" / "best.pt"


# Keys and transposition to C

@pytest.mark.parametrize(("key", "tonic", "minor", "shift"), [
    ("C", 0, False, 0), ("c", 0, True, 0), ("G", 7, False, 5), ("f#", 6, True, -6),
    ("Eb", 3, False, -3), ("bb", 10, True, 2), ("Db", 1, False, -1), ("B", 11, False, 1),
])
def test_keys(key, tonic, minor, shift):
    assert tonic_pitch_class(key) == tonic
    assert is_minor(key) == minor
    assert shift_to_c(key) == shift


def test_invalid_key():
    with pytest.raises(ValueError):
        tonic_pitch_class("H")


def test_transpose_to_c(data_dir):
    plain = BarDataset(data_dir, None, seq_len=8)
    to_c = BarDataset(data_dir, None, seq_len=8, transpose_to_c=True)

    assert to_c.n_excluded["no annotated key"] == 1
    assert to_c.bars.key.tolist() == [k for k in KEYS if k is not None]
    for i, bar_id in enumerate(to_c.bars.bar_id):
        original = plain.notes(int(np.flatnonzero(plain.bars.bar_id == bar_id)[0]))
        shift = shift_to_c(to_c.bars.key[i])
        assert to_c.notes(i) == [Note(o, p + shift, d) for o, p, d in original]
        assert grid_to_notes(to_c[i][0].numpy()) == to_c.notes(i)

    with pytest.raises(ValueError, match="can't be combined"):
        BarDataset(data_dir, None, seq_len=8, transpose=(-1, 1), transpose_to_c=True)


def test_notes_table_matches_notes(data_dir):
    dataset = BarDataset(data_dir, None, seq_len=8)
    table = dataset.notes_table()
    for i in range(len(dataset)):
        rows = table[table.bar == i]
        assert [Note(*r) for r in rows[["onset", "pitch", "duration"]].itertuples(index=False)] \
            == dataset.notes(i)


# Features

def test_bar_features():
    bars = [
        [Note(0, 60, 4), Note(0, 64, 4), Note(0, 66, 2), Note(3, 67, 2), Note(4, 72, 2)],
        [],
    ]
    features = bar_features(notes_table(bars), pd.Series([8, 16]))
    first = features.loc[0]

    assert list(features.columns) == list(FEATURES)
    assert first.n_notes == 5
    assert first.note_density == pytest.approx(5 / 8)
    assert first.onset_density == pytest.approx(3 / 8)
    assert first.chord_size == pytest.approx(5 / 3)
    assert first.max_chord_size == 3
    assert first.pitch_mean == pytest.approx(65.8)
    assert (first.pitch_lowest, first.pitch_highest, first.pitch_range) == (60, 72, 12)
    assert first.duration_mean == pytest.approx(2.8)
    assert first.sounding_fraction == pytest.approx(6 / 8)  # steps 0-5 sound
    assert first.offbeat_onsets == pytest.approx(1 / 5)  # the note at step 3
    assert first.n_pitch_classes == 4
    assert first.pc_C == pytest.approx(2 / 5) and first.pc_E == pytest.approx(1 / 5)
    assert first.pitch_class_entropy == pytest.approx(-(0.4 * np.log2(0.4) + 0.6 * np.log2(0.2)))
    assert first.top_line_leap == pytest.approx(3)  # highest notes 66 -> 67 -> 72
    assert first.dissonance == pytest.approx(2 / 3)  # 60-66 tritone, 64-66 whole tone

    empty = features.loc[1]
    assert (empty.n_notes, empty.note_density, empty.sounding_fraction) == (0, 0, 0)
    assert np.isnan(empty.pitch_mean)


# Audio and plots

def test_midi():
    bars = [[Note(0, 60, 4)], [Note(2, 67, 2)]]
    midi = to_midi(bars, bar_lens=[8, 8], tempo=120, gap_steps=4)
    step = 60 / 120 / 4
    notes = [(n.pitch, n.start, n.end) for n in midi.instruments[0].notes]
    assert notes == [(60, 0.0, 4 * step), (67, 14 * step, 16 * step)]  # second bar at 8 + 4


def test_synthesize_and_save_wav(tmp_path):
    audio = synthesize([[Note(0, 60, 4)], [Note(0, 72, 8)]], bar_lens=[8, 8], tempo=120)
    step = 60 / 120 / 4
    # the music, the last note's release (0.15 s) and half a second of reverb tail
    assert len(audio) == pytest.approx((16 * step + 0.15 + 0.5) * SAMPLE_RATE, abs=3)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() == pytest.approx(0.9, abs=1e-6)
    assert np.abs(audio[: int(4 * step * SAMPLE_RATE)]).max() > 0.1  # the first note sounds
    assert not synthesize([[]], bar_lens=[8]).any()  # an empty bar is silent

    save_wav(audio, tmp_path / "a.wav")
    with wave.open(str(tmp_path / "a.wav")) as f:
        assert (f.getframerate(), f.getnchannels(), f.getnframes()) == (SAMPLE_RATE, 1, len(audio))


def test_save_mp3(tmp_path):
    audio = synthesize([[Note(0, 60, 4), Note(4, 64, 4)]], bar_lens=[8])
    save_mp3(audio, tmp_path / "a.mp3")
    data = (tmp_path / "a.mp3").read_bytes()
    assert data[:2] == bytes([0xFF, 0xFB])  # the start of an MPEG audio frame
    assert 0 < len(data) < len(audio) * 2 / 2  # less than half the size of a 16-bit WAV


def test_synthesize_midi_matches_synthesize(tmp_path):
    bars = [[Note(0, 60, 4), Note(2, 67, 2)], [Note(0, 48, 8)]]
    to_midi(bars, tmp_path / "a.mid", bar_lens=[8, 8], gap_steps=2)
    direct = synthesize(bars, bar_lens=[8, 8], gap_steps=2)
    from_file = synthesize_midi(tmp_path / "a.mid")
    assert len(from_file) == pytest.approx(len(direct), abs=SAMPLE_RATE * 0.01)
    n = min(len(direct), len(from_file))
    assert np.corrcoef(direct[:n], from_file[:n])[0, 1] > 0.99


def test_plot_bars():
    fig = plot_bars([[Note(0, 60, 4)], [Note(2, 67, 2)], []], titles=["a", "b", "c"], ncols=2)
    assert len(fig.axes) == 4


# Decoding and interpolation

def test_slerp_and_lerp():
    rng = np.random.default_rng(0)
    z1, z2 = rng.normal(size=8), rng.normal(size=8)
    z2 *= np.linalg.norm(z1) / np.linalg.norm(z2)
    path = slerp(z1, z2, 5)
    assert np.allclose(path[0], z1) and np.allclose(path[-1], z2)
    assert np.allclose(np.linalg.norm(path, axis=1), np.linalg.norm(z1))  # stays on the sphere
    assert np.allclose(lerp(z1, z2, 3)[1], (z1 + z2) / 2)
    assert np.allclose(slerp(z1, 2 * z1, 3)[1], 1.5 * z1)  # parallel codes: straight line


def test_decode():
    torch.manual_seed(0)
    model = PianoTreeVAE(seq_len=8, max_simu_note=6, **TINY_MODEL)
    bars = decode(model, np.random.default_rng(0).normal(size=(5, 4)), batch_size=2)
    assert len(bars) == 5
    assert all(0 <= n.onset < 8 and 0 <= n.pitch < 128 and n.duration >= 1
               for notes in bars for n in notes)
    assert len(decode(model, np.zeros(4))) == 1


# Encoding

def test_encode_save_and_load(checkpoint):
    embeddings = encode_bars(checkpoint)
    assert len(embeddings) == len(KEYS)
    assert embeddings.mean.shape == embeddings.scale.shape == (len(KEYS), 4)
    assert (embeddings.scale > 0).all()
    np.testing.assert_array_equal(encode_bars(checkpoint).mean, embeddings.mean)

    path = default_output_path(checkpoint, transpose_to_c=False, untrained=False)
    save_embeddings(embeddings, path)
    loaded = load_embeddings(path)
    np.testing.assert_allclose(loaded.mean, embeddings.mean, atol=2e-3)
    assert loaded.bars.equals(embeddings.bars)
    assert dataset_for(loaded).bars.bar_id.tolist() == loaded.bars.bar_id.tolist()
    assert len(loaded.subset(loaded.bars.split == "valid")) == 2


def test_untrained_control_and_transpose_to_c(checkpoint):
    trained = encode_bars(checkpoint)
    untrained = encode_bars(checkpoint, untrained=True, seed=1)
    assert not np.allclose(trained.mean, untrained.mean)
    np.testing.assert_array_equal(untrained.mean, encode_bars(checkpoint, untrained=True,
                                                              seed=1).mean)
    to_c = encode_bars(checkpoint, transpose_to_c=True)
    assert len(to_c) == len(KEYS) - 1 and to_c.info["excluded"]["no annotated key"] == 1


def test_encode_command(checkpoint):
    main(["encode", str(checkpoint), "--to-c"])
    path = checkpoint.parent / "embeddings_to_c.npz"
    assert path.exists()
    assert load_embeddings(path).info["transpose_to_c"]
