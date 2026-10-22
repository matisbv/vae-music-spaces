import numpy as np
import pytest
import torch
from torch.distributions import Normal

from pianotree_dcml.losses import dur_bit_weights, vae_loss
from pianotree_dcml.metrics import NoteCounts, batch_note_counts, note_counts
from pianotree_dcml.model import N_PITCH_TOKENS
from pianotree_dcml.representation import DUR_PAD, PITCH_PAD, Note, notes_to_grid

BARS = [
    [Note(0, 60, 4), Note(0, 64, 4), Note(4, 67, 2)],
    [Note(2, 48, 6)],
]


def grids_and_perfect_logits(confidence=50.0):
    """Grids of BARS, and logits that predict them with near certainty. Positions after
    <eos> (no target) get random logits."""
    grids = torch.from_numpy(np.stack([notes_to_grid(b, seq_len=8, max_simu_note=6) for b in BARS]))
    target = grids[:, :, 1:]
    torch.manual_seed(0)
    pitch = torch.randn(*target.shape[:-1], N_PITCH_TOKENS)
    has_pitch = target[..., 0] != PITCH_PAD
    pitch[has_pitch] = confidence * torch.nn.functional.one_hot(
        target[..., 0][has_pitch], N_PITCH_TOKENS).float()
    dur = torch.randn(*target[..., 1:].shape, 2)
    has_bit = target[..., 1:] != DUR_PAD
    dur[has_bit] = confidence * torch.nn.functional.one_hot(target[..., 1:][has_bit], 2).float()
    return grids, pitch, dur


def standard_normal(batch, z_size=4):
    return Normal(torch.zeros(batch, z_size), torch.ones(batch, z_size))


def test_perfect_predictions_cost_nothing():
    grids, pitch, dur = grids_and_perfect_logits()
    losses = vae_loss(pitch, dur, grids, standard_normal(len(grids)))
    for name in ["loss", "pitch", "duration", "kl"]:
        assert losses[name].item() == pytest.approx(0.0, abs=1e-6)


def test_losses_are_summed_per_bar_and_averaged_over_the_batch():
    grids, pitch, dur = grids_and_perfect_logits(confidence=0.0)  # uniform predictions
    losses = vae_loss(pitch, dur, grids, standard_normal(len(grids)))
    # Pitch targets: one <eos> per step (8 steps x 2 bars) plus the 4 notes.
    assert losses["pitch"].item() == pytest.approx((16 + 4) * np.log(N_PITCH_TOKENS) / 2)
    weights = dur_bit_weights(3).sum().item()
    assert losses["duration"].item() == pytest.approx(4 * weights * np.log(2) / 2)


def test_kl_beta_and_free_nats():
    grids, pitch, dur = grids_and_perfect_logits()
    posterior = Normal(torch.ones(2, 4), torch.ones(2, 4))  # KL = 0.5 per dimension
    losses = vae_loss(pitch, dur, grids, posterior, beta=0.5)
    assert losses["kl"].item() == pytest.approx(2.0)
    assert losses["loss"].item() == pytest.approx(1.0, abs=1e-5)
    free = vae_loss(pitch, dur, grids, posterior, beta=0.5, free_nats=1.5)
    assert free["loss"].item() == pytest.approx(0.25, abs=1e-5)
    assert vae_loss(pitch, dur, grids, posterior, free_nats=3.0)["loss"].item() == pytest.approx(0.0, abs=1e-5)


def test_dur_bit_weights():
    assert dur_bit_weights(3).tolist() == pytest.approx([1.0, 0.6, 0.4])
    assert dur_bit_weights(5).tolist() == pytest.approx([1.0, 0.6, 0.4, 0.3, 0.3])


def test_note_counts():
    true = [Note(0, 60, 4), Note(0, 64, 4), Note(4, 67, 2)]
    predicted = [Note(0, 60, 4), Note(0, 64, 2), Note(5, 67, 2), Note(4, 72, 1)]
    counts = note_counts(predicted, true)
    assert counts == NoteCounts(predicted=4, true=3, matched=1, matched_pitch=2)
    scores = counts.scores()
    assert scores["note_precision"] == pytest.approx(1 / 4)
    assert scores["note_recall"] == pytest.approx(1 / 3)
    assert scores["pitch_f1"] == pytest.approx(2 * (2 / 4) * (2 / 3) / (2 / 4 + 2 / 3))


def test_note_counts_handle_repeated_notes_and_empty_bars():
    assert note_counts([Note(0, 60, 1)] * 2, [Note(0, 60, 1)]).matched == 1
    assert NoteCounts().scores()["note_f1"] == 0.0


def test_batch_note_counts_ignore_predictions_after_eos():
    grids, pitch, dur = grids_and_perfect_logits()  # random logits after <eos>
    counts = batch_note_counts(pitch, dur, grids)
    assert counts == NoteCounts(predicted=4, true=4, matched=4, matched_pitch=4)
