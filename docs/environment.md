# Instructions for environment setup
[<< go back to the main README](../README.md)

You need Python 3.10 or newer. From the project folder:

1. Create and activate a virtual environment:
	- `python -m venv .venv`
	- `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
2. Install the package with its development tools: `pip install -e ".[dev]"`
3. Check the installation by running the tests: `pytest`

The environment is now ready!
