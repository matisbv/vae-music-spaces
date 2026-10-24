"""Probes: how much of a property of bars can be read from their latent codes.

A probe is a simple linear model (ridge regression for numbers, logistic regression for
categories) that predicts a property of a bar (its register, its key, ...) from its code.
It is trained on bars from some pieces and tested on bars from other pieces, so a good
score means the code holds the property in an easily readable form for new music. A
linear probe is deliberately weak: we want to know what the code makes *accessible*, not
what a powerful model could dig out of it.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    """``score`` is R² for numbers (share of the bar-to-bar differences predicted: 0 is no
    better than guessing the average, 1 is perfect) or balanced accuracy for categories
    (average share of each category recognised; ``chance`` is what guessing gets).
    ``low`` and ``high`` bound a 95% interval obtained by resampling test pieces."""

    score: float
    low: float
    high: float
    chance: float


def probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pieces_test: np.ndarray,
    kind: str,
    max_train: int = 20000,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> ProbeResult:
    """Train a linear probe of a ``"number"`` or ``"category"`` on the training bars and
    score it on the test bars. At most ``max_train`` training bars are used (a random
    subset), which keeps the logistic regressions fast without changing the picture."""
    if kind not in ("number", "category"):
        raise ValueError(f"kind must be 'number' or 'category', got {kind!r}")
    rng = np.random.default_rng(seed)
    if len(x_train) > max_train:
        keep = rng.choice(len(x_train), max_train, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]

    if kind == "number":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    else:
        y_train, y_test = np.asarray(y_train).astype(str), np.asarray(y_test).astype(str)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    predictions = model.fit(x_train, y_train).predict(x_test)
    return score_predictions(y_test, predictions, pieces_test, kind, n_bootstrap, seed)


def score_predictions(
    y: np.ndarray, predictions: np.ndarray, pieces: np.ndarray, kind: str,
    n_bootstrap: int = 500, seed: int = 0,
) -> ProbeResult:
    """Score predictions (R² or balanced accuracy, see ``ProbeResult``), with a 95%
    interval from resampling whole pieces: bars of a piece resemble each other, so pieces,
    not bars, are the independent units."""
    y, predictions = np.asarray(y), np.asarray(predictions)
    if kind == "number":
        score, chance = r2_score, 0.0
    else:
        y, predictions = y.astype(str), predictions.astype(str)
        score, chance = balanced_accuracy_score, 1 / len(np.unique(y))
    rng = np.random.default_rng(seed)
    unique_pieces, piece_index = np.unique(pieces, return_inverse=True)
    bars_of = [np.flatnonzero(piece_index == p) for p in range(len(unique_pieces))]
    resampled = []
    with warnings.catch_warnings():
        # A resample can miss some categories; balanced accuracy then warns, harmlessly.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_bootstrap):
            chosen = rng.integers(len(bars_of), size=len(bars_of))
            rows = np.concatenate([bars_of[p] for p in chosen])
            if kind == "category" and len(np.unique(y[rows])) < 2:
                continue
            resampled.append(score(y[rows], predictions[rows]))
        total = score(y, predictions)
    low, high = np.percentile(resampled, [2.5, 97.5])
    return ProbeResult(float(total), float(low), float(high), chance)


def cross_validated_predictions(
    x: np.ndarray, y: np.ndarray, pieces: np.ndarray, kind: str, n_folds: int = 5,
    max_train: int = 10000, balanced: bool = False, seed: int = 0,
) -> np.ndarray:
    """Predictions of a linear probe for every bar, each made by a probe trained on the
    other pieces (pieces are split into ``n_folds`` groups). For categories, returns the
    predicted probability of each class (columns in sorted class order); for numbers, the
    predicted value. With ``balanced``, every class weighs the same in training, however
    many bars it has, so that predictions don't drift towards the largest classes."""
    y = np.asarray(y)
    classes = np.unique(y.astype(str)) if kind == "category" else None
    out = np.zeros((len(y), len(classes))) if kind == "category" else np.zeros(len(y))
    rng = np.random.default_rng(seed)
    for train, test in GroupKFold(n_splits=n_folds).split(x, y, pieces):
        if len(train) > max_train:
            train = rng.choice(train, max_train, replace=False)
        if kind == "category":
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=0.1, max_iter=2000, class_weight="balanced" if balanced else None))
            model.fit(x[train], y[train].astype(str))
            probabilities = np.zeros((len(test), len(classes)))
            probabilities[:, np.searchsorted(classes, model.classes_)] = model.predict_proba(x[test])
            out[test] = probabilities
        else:
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
            out[test] = model.fit(x[train], y[train]).predict(x[test])
    return out


def run_probes(
    codes: dict[str, np.ndarray],
    targets: dict[str, tuple[str, pd.Series]],
    train: np.ndarray,
    test: np.ndarray,
    pieces: np.ndarray,
    min_examples: int = 30,
) -> pd.DataFrame:
    """Probe several properties in several sets of codes.

    ``codes`` maps a name to codes aligned with the bars; ``targets`` maps a property name
    to ``(kind, values)``, with missing values for bars where it is undefined. Categories
    with fewer than ``min_examples`` test bars are left out. Returns one row per
    (property, codes) pair.
    """
    rows = []
    for target, (kind, values) in targets.items():
        values = pd.Series(np.asarray(values, dtype=object if kind == "category" else float))
        usable = values.notna().to_numpy()
        if kind == "category":
            counts = values[usable & test].astype(str).value_counts()
            usable = usable & values.astype(str).isin(counts[counts >= min_examples].index).to_numpy()
        y = values.to_numpy()
        for name, x in codes.items():
            result = probe(x[usable & train], y[usable & train], x[usable & test],
                           y[usable & test], pieces[usable & test], kind)
            rows.append({"property": target, "kind": kind, "codes": name} | asdict(result))
    return pd.DataFrame(rows)


def detector_accuracy(real: pd.DataFrame, generated: pd.DataFrame, seed: int = 0) -> float:
    """How well a simple classifier tells generated bars from real ones, given their
    features (5-fold cross-validated accuracy on balanced classes). 0.5 means it can't
    tell them apart at all; 1.0 means they are always recognisable."""
    n = min(len(real), len(generated))
    x = pd.concat([real.sample(n, random_state=seed), generated.sample(n, random_state=seed)])
    y = np.r_[np.zeros(n), np.ones(n)]
    classifier = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(classifier, x, y, cv=5).mean())
