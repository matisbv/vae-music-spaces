import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="needs the analysis extra")

from pianotree_dcml.analysis.latent import (  # noqa: E402
    active_dimensions, kl_per_dimension, sample_prior)
from pianotree_dcml.analysis.plots import plot_probe_results  # noqa: E402
from pianotree_dcml.analysis.probes import detector_accuracy, probe, run_probes  # noqa: E402
from pianotree_dcml.analysis.reconstruction import bar_scores  # noqa: E402
from pianotree_dcml.representation import Note  # noqa: E402


def test_kl_per_dimension_and_active_dimensions():
    rng = np.random.default_rng(0)
    mean = np.zeros((1000, 3))
    scale = np.ones((1000, 3))
    mean[:, 1] = rng.normal(size=1000)  # an informative dimension with a sure encoder
    scale[:, 1] = 0.1
    mean[:, 2] = 0.001 * rng.normal(size=1000)  # varies, but negligibly

    kl = kl_per_dimension(mean, scale)
    assert kl[0] == pytest.approx(0.0)  # outputs the prior: no information
    assert kl[1] == pytest.approx(0.5 * (1 + 0.01 - 1 - 2 * np.log(0.1)), rel=0.05)
    assert active_dimensions(mean).tolist() == [False, True, False]


def test_sample_prior():
    z = sample_prior(2000, 4, seed=1)
    assert z.shape == (2000, 4) and z.dtype == np.float32
    assert np.allclose(z.mean(axis=0), 0, atol=0.1) and np.allclose(z.std(axis=0), 1, atol=0.1)
    assert np.array_equal(z, sample_prior(2000, 4, seed=1))


def make_data(n, rng):
    """Codes of n bars from 40 pieces, where dimension 0 encodes a number and dimension 1
    a category; the other dimensions are noise."""
    x = rng.normal(size=(n, 6))
    number = 3 * x[:, 0] + 0.1 * rng.normal(size=n)
    category = np.where(x[:, 1] > 0, "minor", "major")
    pieces = rng.integers(40, size=n).astype(str)
    return x, number, category, pieces


def test_probe_finds_encoded_properties():
    rng = np.random.default_rng(0)
    x_train, number_train, category_train, _ = make_data(2000, rng)
    x_test, number_test, category_test, pieces = make_data(1000, rng)

    number = probe(x_train, number_train, x_test, number_test, pieces, "number")
    assert number.score > 0.99 and number.low <= number.score <= number.high
    assert number.chance == 0

    category = probe(x_train, category_train, x_test, category_test, pieces, "category")
    assert category.score > 0.95 and category.chance == 0.5


def test_probe_scores_noise_at_chance():
    rng = np.random.default_rng(0)
    x_train, _, _, _ = make_data(2000, rng)
    x_test, _, _, pieces = make_data(1000, rng)
    noise = probe(x_train, rng.normal(size=2000), x_test, rng.normal(size=1000), pieces, "number")
    assert abs(noise.score) < 0.02 and noise.low < 0 < noise.high
    labels = probe(x_train, rng.choice(["a", "b", "c"], 2000), x_test,
                   rng.choice(["a", "b", "c"], 1000), pieces, "category")
    assert abs(labels.score - 1 / 3) < 0.05


def test_probe_rejects_unknown_kind():
    with pytest.raises(ValueError):
        probe(np.zeros((4, 2)), np.zeros(4), np.zeros((4, 2)), np.zeros(4), np.zeros(4), "rank")


def test_run_probes_and_plot():
    rng = np.random.default_rng(0)
    x, number, category, pieces = make_data(3000, rng)
    category = pd.Series(category, dtype=object)
    category[:100] = None  # undefined for some bars
    category[100] = "rare"  # a category too rare to be scored
    test = np.arange(3000) >= 2000
    codes = {"informative": x, "noise": rng.normal(size=(3000, 6))}
    targets = {"a number": ("number", number), "a category": ("category", category)}

    results = run_probes(codes, targets, ~test, test, pieces)
    assert results[["property", "codes"]].values.tolist() == [
        ["a number", "informative"], ["a number", "noise"],
        ["a category", "informative"], ["a category", "noise"]]
    scores = results.set_index(["property", "codes"]).score
    assert scores["a number", "informative"] > 0.99 > 0.05 > abs(scores["a number", "noise"])
    assert scores["a category", "informative"] > 0.95
    assert results.chance.tolist() == [0, 0, 0.5, 0.5]

    ax = plot_probe_results(results, {"informative": "tab:blue"})
    assert len(ax.get_yticks()) == 2


def test_detector_accuracy():
    rng = np.random.default_rng(0)
    real = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    same = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    shifted = same + [5, 0, 0]  # best possible accuracy: Φ(2.5) ≈ 0.99
    assert abs(detector_accuracy(real, same) - 0.5) < 0.06
    assert detector_accuracy(real, shifted) > 0.95


def test_bar_scores():
    originals = [[Note(0, 60, 4), Note(4, 64, 4)], [Note(0, 48, 2)]]
    reconstructions = [[Note(0, 60, 4), Note(4, 64, 2)], [Note(0, 48, 2)]]
    scores = bar_scores(originals, reconstructions)
    assert scores.note_f1.tolist() == pytest.approx([0.5, 1.0])
    assert scores.pitch_f1.tolist() == pytest.approx([1.0, 1.0])
    assert scores.n_true.tolist() == [2, 1]
