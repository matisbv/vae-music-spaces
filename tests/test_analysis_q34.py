import numpy as np
import pytest

pytest.importorskip("sklearn", reason="needs the analysis extra")

from pianotree_dcml.analysis.probes import (  # noqa: E402
    cross_validated_predictions, score_predictions)
from pianotree_dcml.analysis.similarity import (  # noqa: E402
    centroid_distances, fifths_distance, mantel)


def test_fifths_distance():
    assert fifths_distance(0, 0) == 0
    assert fifths_distance(0, 7) == 1 and fifths_distance(0, 5) == 1  # C-G, C-F
    assert fifths_distance(0, 2) == 2  # C-D
    assert fifths_distance(0, 1) == 5  # C-C#: far apart on the circle
    assert fifths_distance(0, 6) == 6
    assert fifths_distance(3, 10) == fifths_distance(10, 3) == 1  # Eb-Bb


def test_centroid_distances():
    codes = np.array([[0.0, 0], [0, 2], [3, 0], [3, 4]])
    labels = np.array(["b", "b", "a", "a"])
    d = centroid_distances(codes, labels)
    assert list(d.index) == ["a", "b"]
    assert d.loc["a", "b"] == pytest.approx(np.hypot(3, 1)) and d.loc["a", "a"] == 0


def distance_matrix(points):
    points = np.asarray(points, float).reshape(len(points), -1)
    return np.linalg.norm(points[:, None] - points[None, :], axis=-1)


def test_mantel_detects_matching_geometry():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(12, 3))
    hypothesis = distance_matrix(points)
    observed = distance_matrix(points + 0.05 * rng.normal(size=(12, 3)))
    rho, p = mantel(observed, hypothesis, n_permutations=2000)
    assert rho > 0.9 and p < 0.01
    rho, p = mantel(distance_matrix(rng.normal(size=(12, 3))), hypothesis, n_permutations=2000)
    assert p > 0.05


def test_mantel_with_control():
    # The observed distances follow the control exactly; a hypothesis that only shares
    # part of the control's structure must not look significant once it is controlled for.
    height = np.arange(12)
    control = distance_matrix(height)
    observed = control.copy()
    hypothesis = control + 3 * distance_matrix(np.random.default_rng(1).normal(size=12))
    assert mantel(observed, hypothesis, n_permutations=1000)[0] > 0.3
    rho, p = mantel(observed, hypothesis, control=control, n_permutations=1000)
    assert abs(rho) < 0.2 and p > 0.05


def test_cross_validated_predictions():
    rng = np.random.default_rng(0)
    pieces = np.repeat(np.arange(30), 20)
    x = rng.normal(size=(600, 4))
    category = np.where(x[:, 0] > 0, "late", "early")
    number = 2 * x[:, 1] + 0.1 * rng.normal(size=600)

    probabilities = cross_validated_predictions(x, category, pieces, "category", n_folds=3)
    assert probabilities.shape == (600, 2) and np.allclose(probabilities.sum(axis=1), 1)
    predicted = np.array(["early", "late"])[probabilities.argmax(axis=1)]
    assert score_predictions(category, predicted, pieces, "category").score > 0.9

    values = cross_validated_predictions(x, number, pieces, "number", n_folds=3)
    result = score_predictions(number, values, pieces, "number")
    assert result.score > 0.95 and result.low <= result.score <= result.high


def test_balanced_cross_validated_predictions():
    # A rare class that is easy to recognise: without balancing, the probe favours the
    # common class; with balancing, the rare class is recognised.
    rng = np.random.default_rng(0)
    y = np.where(np.arange(1000) < 950, "common", "rare")
    x = rng.normal(size=(1000, 2))
    x[y == "rare", 0] += 1.5
    pieces = np.arange(1000) // 10
    plain = cross_validated_predictions(x, y, pieces, "category", n_folds=3)
    balanced = cross_validated_predictions(x, y, pieces, "category", n_folds=3, balanced=True)
    recall = lambda p: (p.argmax(axis=1)[y == "rare"] == 1).mean()
    assert recall(balanced) > 0.6 > recall(plain)
