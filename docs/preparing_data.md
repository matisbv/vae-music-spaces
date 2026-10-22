# Instructions to prepare the datasets
[<< go back to the main README](../README.md)

The model is trained on bars from the [DCML corpora](https://github.com/DCMLab/dcml_corpora):
12 corpora of annotated scores, from Corelli's trio sonatas (1680s) to Medtner's *Tales* (1920s).
Each corpus is distributed by the [DCML lab](https://www.epfl.ch/labs/dcml/) under its own terms.
This repository doesn't include the data, so check those terms before you share it.

## 1. Download the corpora

```
uv run pianotree-dcml download
```

This saves release v2.3 of the corpora (a 24 MB zip) to `data/raw/dcml_corpora.zip` and
verifies its checksum. Every user works from exactly the same data.

## 2. Build the training data

```
uv run pianotree-dcml prepare
```

This takes about 10 seconds and writes two files to `data/processed/`:

- `bars.csv`: one row per bar (61,519 in total), with:
  - `bar_id`, the bar's row number, used by `notes.csv`;
  - where the bar comes from: `corpus`, `piece`, `mc` (measure count) and `mn` (bar
    number in the score), `volta` (1st/2nd ending);
  - its time signature, as `timesig` and as `bar_len` (length in 16th notes);
  - `split`: `train` or `valid`;
  - piece metadata: `composer`, `year` (of composition), `era`, `century`, `genre`, `key`,
    `n_staffs`;
  - the harmony label at the start of the bar: `localkey`, `chord`, `numeral`,
    `chord_type`. It is missing for the 23 Beethoven sonata movements that have no
    harmony annotations;
  - simple statistics: `n_notes`, `notes_per_step`, `mean_pitch`, `pitch_range`,
    `mean_duration`.
- `notes.csv`: one row per note, with its `bar_id`, its `onset` and `duration` in 16th
  notes within the bar, and its MIDI `pitch`.

Piece names are only unique within a corpus (e.g. `op08n01` exists in two corpora), so
identify a piece by `corpus` and `piece` together.

### How bars are built

- **Grid.** Notes are placed on a grid of 16th notes. Onsets and durations are rounded down
  (triplets are approximated), and notes are cut at the end of their bar. Grace notes are
  dropped.
- **Ties.** Tied notes are merged within a bar. A note tied over a barline starts again at
  the beginning of the next bar, so every bar can be read on its own.
- **Doubled notes.** When two notes have the same onset and pitch (e.g. a note doubled in
  two staves), only the longer one is kept.
- **Split bars and endings.** A bar split in two by a repeat sign counts as one bar, while
  first and second endings are separate bars.
- **Dropped bars.** 62 bars that don't fit their time signature (mostly free-time cadenzas)
  and 158 bars without any notes are dropped.

### Train/validation split

Whole pieces go to the validation set, about 10% of the bars of each corpus. Pieces repeat
many bars, so splitting by bar would put copies of validation bars in the training set and
make validation scores look better than they are. Use `--valid-fraction` and `--seed` to
change the split.

## 3. Load the data in PyTorch

```python
from pianotree_dcml.data.dataset import BarDataset

train = BarDataset("data/processed", split="train", seq_len=16, transpose=(-5, 6))
grid, index = train[0]           # piano-grid of the first bar, and its row in train.bars
train.bars.loc[index, "composer"]
```

`BarDataset` turns each bar into a piano-grid (see `pianotree_dcml.representation`), padded
to `seq_len` steps. It leaves out bars longer than `seq_len` (`seq_len=16` covers the bars of
4/4 and shorter time signatures), and it can filter by `corpora` and `timesigs`.
`transpose=(low, high)` transposes each bar by a random number of semitones every time it is
loaded, which is a common data augmentation.
