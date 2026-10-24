# Tools for analysing a trained model
[<< go back to the main README](../README.md)

The analysis itself is in the notebooks:

- [What does the model remember about a bar?](../notebooks/03_what_the_code_remembers.ipynb)
  (figures and sounds in [`notebooks/outputs/q1`](../notebooks/outputs/q1))
- [Moving through the code](../notebooks/04_moving_through_the_code.ipynb)
  (figures and sounds in [`notebooks/outputs/q2`](../notebooks/outputs/q2))

Once a model is trained, these tools let you look inside it: encode every bar of the corpus
into the latent space, describe bars with simple measurements, decode latent codes back into
music, and listen to the results.

They need a few extra packages (scikit-learn, matplotlib, pretty_midi and a Jupyter kernel):

```
uv sync --extra cpu --extra analysis
```

## 1. Encode the corpus

```
uv run pianotree-dcml encode runs/full/best.pt
```

This computes the latent code of every bar the model can read and saves it next to the
checkpoint, as `runs/full/embeddings.npz`. It only runs the encoder: about 5 minutes for the
whole corpus on a laptop CPU. Two options produce the variants used in the analysis:

- `--to-c` first transposes every piece to C major or C minor (using the key annotated in
  the corpus), so that the key can't explain any result. Pieces without an annotated key are
  left out. The file is `embeddings_to_c.npz`.
- `--untrained` uses the same architecture with random, untrained weights. This is the
  **control**: anything the untrained model's codes can also tell us was not learned
  during training. The file is `embeddings_untrained.npz`.

In Python:

```python
from pianotree_dcml.analysis.encode import load_embeddings, dataset_for

embeddings = load_embeddings("runs/full/embeddings.npz")
embeddings.mean     # (bars, 512): the latent code of each bar (mean of q(z | bar))
embeddings.scale    # (bars, 512): how uncertain the encoder is about each dimension
embeddings.bars     # one row per bar: corpus, piece, composer, year, era, key, split, ...

valid = embeddings.subset(embeddings.bars.split == "valid")
dataset = dataset_for(embeddings)   # the same bars, row for row, as the model reads them
dataset.notes(0)                    # the notes of the first bar
```

## 2. Describe bars with simple measurements

[`features.py`](../src/pianotree_dcml/analysis/features.py) measures each bar in ways anyone
can understand: its register (how high it is), how many notes it has and how many start
together, how long they are, how much the melody leaps, how dissonant (tense) the notes
played together are, and which of the 12 note names it uses. `FEATURES` lists them all with
a plain description.

```python
from pianotree_dcml.analysis.features import bar_features

features = bar_features(dataset.notes_table(), dataset.bars.bar_len)
```

These measurements are the baseline of the analysis: if a question (for example, "which
composer wrote this bar?") can be answered from them as well as from the latent codes, the
model hasn't learned anything beyond them.

## 3. Decode, interpolate and listen

```python
from pianotree_dcml.train import load_checkpoint
from pianotree_dcml.analysis.decode import decode, slerp
from pianotree_dcml.analysis.audio import play, save_wav, synthesize, to_midi
from pianotree_dcml.analysis.plots import plot_bars

model, config = load_checkpoint("runs/full/best.pt")
reconstruction = decode(model, embeddings.mean[0])[0]

# A path through the latent space between two bars, in 8 steps
path = decode(model, slerp(embeddings.mean[0], embeddings.mean[100], 8))
plot_bars(path, ncols=8)

play(path, gap_steps=2)                                  # an audio player, in a notebook
save_wav(synthesize(path, gap_steps=2), "interpolation.wav")
to_midi(path, "interpolation.mid", gap_steps=2)          # for a real piano sound
```

Audio comes from a small piano-like synthesizer built into the package, so it works
anywhere, including on Colab, without installing anything. The MIDI files can be opened in
any music software for a better piano sound.
