# Instructions for environment setup
[<< go back to the main README](../README.md)

## With uv (recommended)

[uv](https://docs.astral.sh/uv/) installs the right Python version and the exact package
versions recorded in `uv.lock`, so your environment matches the one the code was tested with.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
2. From the project folder, create the environment, choosing a PyTorch build:
	- `uv sync --extra cpu` if you don't have an NVIDIA GPU (laptops, macOS);
	- `uv sync --extra cuda` if you have an NVIDIA GPU (Linux or Windows).
3. Check the installation by running the tests: `uv run pytest`

Run any command inside the environment with `uv run <command>`, or activate it with
`source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).

## With pip (Google Colab, Kaggle)

Hosted notebooks already come with PyTorch installed. From a clone of the repository, run:

```
pip install -e .
```

On your own machine, use `pip install -e ".[cpu]"` to install PyTorch from PyPI as well. pip
doesn't read `uv.lock`, so the package versions you get may differ from the tested ones.

The environment is now ready!
