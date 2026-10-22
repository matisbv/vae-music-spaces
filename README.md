# VAE Music Spaces

**Learning from VAE latent space representations of musical surface structure – A diachronic style change analysis**

Matis Brassard-Verrier

*Note: this documentation is a stub. It will be constinuously updated and improved upon as the project progresses.*

## Set up the environment
Before you start working with the project, you have to setup the environment for the code. You can do this by following the instructions [here](docs/environment.md).

## Prepare the datasets
- First, within folder `data/`, clone the [dcml_corpora](https://github.com/DCMLab/dcml_corpora) corpus from the [DCMLab](https://www.epfl.ch/labs/dcml/).
	- You should now have a subfolder `data/dcml_corpora` containing the corpus.
- To prepare the datasets for training and inference, follow the instructions [here](docs/preparing_data.md).

## Train the model and use it for inference
- To train the model, follow the instructions [here](docs/train.md).
- To perform inference with a trained model, follow the instructions [here](docs/inference.md).
