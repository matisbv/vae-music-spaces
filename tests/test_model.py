import numpy as np
import pytest
import torch

from pianotree_dcml.losses import vae_loss
from pianotree_dcml.metrics import batch_note_counts
from pianotree_dcml.model import N_PITCH_TOKENS, PianoTreeVAE
from pianotree_dcml.representation import Note, notes_to_grid

SEQ_LEN, MAX_SIMU_NOTE = 8, 8
SIZES = dict(note_emb_size=16, enc_notes_hid_size=32, enc_time_hid_size=32, z_size=8,
             dec_emb_hid_size=16, dec_time_hid_size=64, dec_notes_hid_size=64,
             dec_z_in_size=16, dec_dur_hid_size=16)


def random_grids(n_bars, seed=0):
    rng = np.random.default_rng(seed)
    grids = []
    for _ in range(n_bars):
        notes = []
        for onset in rng.choice(SEQ_LEN, size=3, replace=False):
            for pitch in rng.choice(np.arange(48, 72), size=rng.integers(1, 4), replace=False):
                notes.append(Note(int(onset), int(pitch), int(rng.integers(1, SEQ_LEN - onset + 1))))
        grids.append(notes_to_grid(notes, SEQ_LEN, MAX_SIMU_NOTE))
    return torch.from_numpy(np.stack(grids))


@pytest.fixture
def model():
    torch.manual_seed(0)
    return PianoTreeVAE(seq_len=SEQ_LEN, max_simu_note=MAX_SIMU_NOTE, **SIZES)


@pytest.mark.parametrize("teacher_forcing", [0.0, 0.5, 1.0])
def test_output_shapes(model, teacher_forcing):
    grids = random_grids(3)
    pitch, dur, posterior = model(grids, teacher_forcing, teacher_forcing)
    assert pitch.shape == (3, SEQ_LEN, MAX_SIMU_NOTE - 1, N_PITCH_TOKENS)
    assert dur.shape == (3, SEQ_LEN, MAX_SIMU_NOTE - 1, model.dur_width, 2)
    assert posterior.mean.shape == (3, SIZES["z_size"])
    assert model.encode(grids).mean.shape == (3, SIZES["z_size"])


def test_early_stopping_changes_nothing_that_is_used(model):
    """Stopping note generation once every bar has predicted <eos> must not change the
    loss or the decoded notes."""
    model.eval()
    grids = random_grids(4)
    with torch.no_grad():
        notes, lengths = model.embed_grids(grids)
        z = model.encode_embedded(notes, lengths).mean
        fast = model.decode(z, min_lengths=lengths.max(dim=0).values)
        full = model.decode(z, min_lengths=torch.full((SEQ_LEN,), MAX_SIMU_NOTE))
    posterior = model.encode(grids)
    fast_loss = vae_loss(*fast, grids, posterior)
    full_loss = vae_loss(*full, grids, posterior)
    for name in fast_loss:
        assert fast_loss[name].item() == pytest.approx(full_loss[name].item(), rel=1e-5)
    assert batch_note_counts(*fast, grids) == batch_note_counts(*full, grids)


@pytest.mark.slow
def test_model_can_memorize_a_few_bars(model):
    """End-to-end check that the whole model learns: trained with teacher forcing, it must
    then reconstruct four bars perfectly on its own."""
    grids = random_grids(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(450):
        pitch, dur, posterior = model(grids, teacher_forcing_time=1.0, teacher_forcing_notes=1.0)
        loss = vae_loss(pitch, dur, grids, posterior, beta=0.0)["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pitch, dur, _ = model(grids, sample=False)
    assert batch_note_counts(pitch, dur, grids).scores()["note_f1"] == 1.0
