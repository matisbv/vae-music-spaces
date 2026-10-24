# VAE Music Spaces

Train a [PianoTree-VAE](https://github.com/ZZWaang/PianoTree-VAE) from scratch on the
[DCML corpora](https://github.com/DCMLab/dcml_corpora): 12 corpora of annotated classical
scores, from Corelli's trio sonatas (1680s) to Medtner's *Tales* (1920s).

PianoTree-VAE is a variational autoencoder for polyphonic music. It encodes a bar of music
as a point in a latent space and decodes it back, note by note. This repository walks
through every step:

- **Data**: download a pinned release of the corpora and turn it into 61,519 bars on a
  16th-note grid, with metadata (composer, year, era, key, harmony) for each bar.
- **Representation**: encode each bar as a *piano-grid*, the tree-shaped format the model
  reads and writes.
- **Training**: train the model on a laptop CPU (small configuration) or a GPU (full size),
  with checkpoints, resuming and validation metrics.

## Quickstart

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs the
right Python version and the exact package versions this code was tested with.

```
git clone https://github.com/matisbv/vae-music-spaces.git
cd vae-music-spaces
uv sync --extra cpu        # or --extra cuda with an NVIDIA GPU

uv run pianotree-dcml download                                  # the DCML corpora, 24 MB
uv run pianotree-dcml prepare                                   # bars and notes, ~10 s
uv run pianotree-dcml train configs/tiny.toml --run-dir runs/tiny
```

The last command trains a small model on Chopin's mazurkas. It takes about half an hour on a
laptop CPU and reconstructs about a third of the validation notes exactly. The full-size
model, trained on all corpora, needs a GPU: free ones are available on Kaggle and Colab
(see the training instructions).

## Documentation

1. [Setting up the environment](docs/environment.md), with uv or with pip (Colab, Kaggle)
2. [Preparing the data](docs/preparing_data.md): what the commands produce and how bars are
   built from the scores
3. [Training](docs/train.md): the two configurations, what a run produces, and how the model
   is trained
4. [Analysis tools](docs/analysis.md): encode the corpus into the latent space, describe and
   decode bars, and listen to the results

## What the model learned

Four notebooks look inside the trained models, comparing them with an untrained copy and with
simple measurements of the music. Each one comes with figures and sounds.

1. [What does the model remember about a bar?](notebooks/03_what_the_code_remembers.ipynb)
   Mostly a faithful picture of the notes, not their musical context.
2. [Moving through the code](notebooks/04_moving_through_the_code.ipynb): some musical
   qualities, like "higher" and "busier", are directions you can move along; others are not.
3. [Do related keys sit close together?](notebooks/05_related_keys.ipynb) One of the models
   rediscovered the circle of fifths on its own.
4. [Composers and the history of music](notebooks/06_composers_and_history.ipynb): the code
   recognises composers and dates pieces, better than simple statistics but far from perfectly.

The code is documented throughout. A good place to start reading is
[`representation.py`](src/pianotree_dcml/representation.py), which describes the data
formats, then [`model.py`](src/pianotree_dcml/model.py).

## Repository layout

```
configs/                 training configurations (tiny: CPU, full: GPU)
docs/                    step-by-step instructions
src/pianotree_dcml/
    representation.py    notes, piano-grids and piano-rolls, and the conversions between them
    data/                downloading and preparing the corpora, PyTorch dataset
    model.py             PianoTree-VAE
    losses.py            training objective
    metrics.py           note-level precision, recall and F1
    train.py             training loop, checkpoints
    analysis/            encoding, bar features, decoding, audio and plots
    cli.py               the pianotree-dcml command
notebooks/               the analysis of the trained models, with exported figures and sounds
tests/                   run them with `uv run pytest`
```

## Background

This project started as my [master's thesis](https://zenodo.org/records/19537861) (2023), which studied how the latent space of
such a model reflects changes in musical style over time. The original thesis code is
archived at [vae-music-spaces-thesis-2023](https://github.com/matisbv/vae-music-spaces-thesis-2023).
This repository is a rewrite, with several bugs of the original pipeline fixed and a more
rigorous analysis of the latent space.

## References

- Wang, Z., Zhang, Y., Zhang, Y., Jiang, J., Yang, R., Zhao, J. and Xia, G. (2020).
  PianoTree VAE: Structured Representation Learning for Polyphonic Music. *ISMIR*.
- The DCML corpora, by the [Digital and Cognitive Musicology Lab](https://www.epfl.ch/labs/dcml/)
  at EPFL.

## License

The code is released under the [MIT License](LICENSE). The model is adapted from the
[original PianoTree-VAE implementation](https://github.com/ZZWaang/PianoTree-VAE), also
under the MIT License (see [licenses/PianoTree-VAE.txt](licenses/PianoTree-VAE.txt)). The
DCML corpora are not included in this repository and are distributed by the DCML lab under
their own terms.
