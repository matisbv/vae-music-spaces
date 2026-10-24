"""Tools to analyse a trained model and its latent space.

They need the ``analysis`` extra: ``uv sync --extra cpu --extra analysis``.

- :mod:`.encode`: latent codes of every bar, saved to a file with the bars' metadata;
- :mod:`.features`: simple, human-readable descriptions of bars (register, density,
  ...), used as baselines and to describe what the model does;
- :mod:`.decode`: from latent codes back to notes, interpolation;
- :mod:`.audio`: listen to bars, and save them as MIDI or WAV files;
- :mod:`.plots`: piano-roll figures.
"""
