# VAE Music Spaces

**Learning from VAE latent space representations of musical surface structure – A diachronic style change analysis**

Matis Brassard-Verrier

*Note: this documentation is a stub. It will be constinuously updated and improved upon as the project progresses.*

## Set up the environment
Before you start working with the project, you have to setup the environment for the code. You can do this by following the instructions [here](docs/environment.md).

## Prepare the datasets
Download the [DCML corpora](https://github.com/DCMLab/dcml_corpora) and turn them into training data:

```
uv run pianotree-dcml download
uv run pianotree-dcml prepare
```

See [the data instructions](docs/preparing_data.md) for what these commands produce and how bars are built.

## Train the model
Train a small model on your own machine:

```
uv run pianotree-dcml train configs/tiny.toml --run-dir runs/tiny
```

See [the training instructions](docs/train.md) for the full-size model, what a run
produces, and how training works.
