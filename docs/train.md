# Instructions for training
[<< go back to the main README](../README.md)

First prepare the data (see [the data instructions](preparing_data.md)). Then train a model:

```
uv run pianotree-dcml train configs/tiny.toml --run-dir runs/tiny
```

## The two configurations

| | `configs/tiny.toml` | `configs/full.toml` |
|---|---|---|
| Data | Chopin's mazurkas (about 4,300 training bars of 3/4) | all 12 corpora, bars of up to 16 sixteenths (about 53,000 training bars) |
| Model | 2.1M parameters | 13.6M parameters (sizes of the original paper) |
| Hardware | any laptop CPU | a GPU |
| Time | about 75 s per epoch, 10 epochs, on a 2-thread laptop CPU | hours per epoch on a CPU |

`tiny` lets you follow the whole tutorial on your own machine. `full` is the model to
use for real results. If you don't have a GPU, see [Training on Kaggle or Colab](#training-on-kaggle-or-colab).

To try other settings, copy a configuration and edit it. Every option, with its default, is
documented in [`src/pianotree_dcml/config.py`](../src/pianotree_dcml/config.py).

## What a run produces

Everything goes to the `--run-dir` directory:

- `config.json`: the configuration of the run;
- `metrics.csv`: one row per epoch with the training and validation losses, the learning
  rate, and the validation precision, recall and F1 of the reconstructed notes;
- `last.pt`: the latest checkpoint. If training stops, continue it with
  `uv run pianotree-dcml train <config> --run-dir <dir> --resume`;
- `best.pt`: the checkpoint with the lowest validation loss.

Load a trained model in Python with:

```python
from pianotree_dcml.train import load_checkpoint

model, config = load_checkpoint("runs/tiny/best.pt")
```

## How the model is trained

**Objective.** PianoTree-VAE is a variational autoencoder. For each bar, the loss
(`pianotree_dcml.losses.vae_loss`) adds up three terms:

- `pitch`: how badly the decoder predicts each note's pitch and the end of each time step;
- `duration`: how badly it predicts the duration bits of each note. Errors on the most
  significant bits weigh more;
- `kl`: how far the encoder's distribution over latent codes is from a standard normal,
  weighted by `beta`.

A larger `beta` gives a smoother, more regular latent space, at the cost of less precise
reconstructions. `beta_warmup_epochs` and `free_nats` are two common ways to keep the
decoder from ignoring the latent code early in training.

**Teacher forcing.** The decoder generates a bar one note at a time, from its own previous
predictions. Early in training, those predictions are poor, so the decoder is sometimes fed
the true previous notes instead. The probability of doing so decreases over training
(`teacher_forcing_time` and `teacher_forcing_notes`), so that the model ends up generating
on its own.

**Validation.** After every epoch, the model reconstructs the validation bars without any
teacher forcing, as it would in use. A reconstructed note counts as correct if it has the
right onset, pitch and duration (`note_*` scores), or only the right onset and pitch
(`pitch_*` scores).

**Differences from the original PianoTree-VAE code.** The original code averaged the KL
term over the 512 latent dimensions. Here it is summed, as in the usual VAE objective, which
makes `beta` mean the same thing whatever the size of the model. The original `beta = 0.1`
corresponds to a `beta` of about 0.007 here. A few smaller inconsistencies in the decoder
are fixed too; see the docstring of [`model.py`](../src/pianotree_dcml/model.py).

## Training on Kaggle or Colab

Kaggle and Colab offer free GPUs, and both come with PyTorch installed. In a notebook with a
GPU enabled:

```
!git clone https://github.com/matisbv/vae-music-spaces.git
%cd vae-music-spaces
!pip install -e .
!pianotree-dcml download
!pianotree-dcml prepare
!pianotree-dcml train configs/full.toml --run-dir runs/full
```

Free sessions are time-limited. If a session ends before training does, keep the run
directory (on Kaggle, save it as a dataset; on Colab, put it in Google Drive) and continue
later with `--resume`.
