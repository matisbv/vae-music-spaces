"""Latent codes of every bar, saved with the bars' metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pianotree_dcml.config import Config
from pianotree_dcml.data.dataset import BarDataset
from pianotree_dcml.data.keys import shift_to_c
from pianotree_dcml.model import PianoTreeVAE
from pianotree_dcml.representation import Note, notes_to_grid
from pianotree_dcml.train import build_model, load_checkpoint


@dataclass
class Embeddings:
    """Latent codes of bars: row ``i`` of ``mean`` and ``scale`` is q(z | bar) for the
    bar described by row ``i`` of ``bars`` (corpus, piece, year, split, ...)."""

    bars: pd.DataFrame
    mean: np.ndarray
    scale: np.ndarray
    info: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.bars)

    def subset(self, mask: np.ndarray | pd.Series) -> Embeddings:
        mask = np.asarray(mask, dtype=bool)
        return Embeddings(self.bars[mask].reset_index(drop=True), self.mean[mask],
                          self.scale[mask], self.info)


@torch.no_grad()
def encode_dataset(
    model: PianoTreeVAE, dataset: BarDataset, batch_size: int = 256, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of q(z | bar) for every bar of ``dataset``."""
    model.eval().to(device)
    means, scales = [], []
    for grids, _ in tqdm(DataLoader(dataset, batch_size=batch_size), desc="Encoding", leave=False):
        posterior = model.encode(grids.to(device))
        means.append(posterior.mean.cpu())
        scales.append(posterior.scale.cpu())
    return torch.cat(means).numpy(), torch.cat(scales).numpy()


@torch.no_grad()
def encode_notes(
    model: PianoTreeVAE, bars: Sequence[Sequence[Note]], batch_size: int = 256,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of q(z | bar) for bars given as notes, e.g. bars
    transformed on purpose. Onsets with more notes than a piano-grid holds keep their
    lowest notes."""
    model.eval().to(device)
    grids = torch.from_numpy(np.stack([
        notes_to_grid(notes, model.seq_len, model.max_simu_note, overflow="truncate")
        for notes in bars]))
    means, scales = [], []
    for start in range(0, len(grids), batch_size):
        posterior = model.encode(grids[start:start + batch_size].to(device))
        means.append(posterior.mean.cpu())
        scales.append(posterior.scale.cpu())
    return torch.cat(means).numpy(), torch.cat(scales).numpy()


def encode_bars(
    checkpoint: Path | str,
    data_dir: Path | str | None = None,
    transpose_to_c: bool = False,
    untrained: bool = False,
    seed: int = 0,
    batch_size: int = 256,
    device: str = "cpu",
) -> Embeddings:
    """Encode every bar the model can read (all splits).

    With ``untrained=True``, the model's weights are replaced by a random initialisation
    (seeded): the same architecture, but nothing learned. Its codes are the control that
    tells what the training added.
    """
    model, config = load_checkpoint(checkpoint, device)
    if untrained:
        torch.manual_seed(seed)
        model = build_model(config).to(device).eval()
    data_dir = str(data_dir or config.data.data_dir)
    dataset = _dataset(config, data_dir, transpose_to_c)
    mean, scale = encode_dataset(model, dataset, batch_size, device)
    info = {"checkpoint": str(checkpoint), "data_dir": data_dir,
            "transpose_to_c": transpose_to_c, "untrained": untrained,
            "seed": seed if untrained else None, "excluded": dataset.n_excluded}
    return Embeddings(dataset.bars, mean, scale, info)


def dataset_for(embeddings: Embeddings) -> BarDataset:
    """The dataset the codes were computed from, row for row: use it to listen to the
    bars (``dataset.notes(i)``) or to compute their features (``dataset.notes_table()``)."""
    checkpoint = torch.load(embeddings.info["checkpoint"], map_location="cpu", weights_only=False)
    dataset = _dataset(Config.from_dict(checkpoint["config"]), embeddings.info["data_dir"],
                       embeddings.info["transpose_to_c"])
    if not np.array_equal(dataset.bars.bar_id, embeddings.bars.bar_id):
        raise ValueError("the data has changed since these codes were computed")
    return dataset


def _dataset(config: Config, data_dir: str, transpose_to_c: bool) -> BarDataset:
    return BarDataset(data_dir, split=None, seq_len=config.data.seq_len,
                      max_simu_note=config.data.max_simu_note, transpose_to_c=transpose_to_c)


def save_embeddings(embeddings: Embeddings, path: Path | str) -> None:
    """Save the codes; the metadata is reloaded from ``bars.csv`` by ``load_embeddings``.

    Codes are stored in half precision (about 3 significant digits), which is plenty for
    analysis and halves the file size.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, bar_id=embeddings.bars.bar_id.to_numpy(), mean=embeddings.mean.astype(np.float16),
        scale=embeddings.scale.astype(np.float16), info=json.dumps(embeddings.info),
    )


def load_embeddings(path: Path | str) -> Embeddings:
    with np.load(path) as saved:
        bar_id = saved["bar_id"]
        mean, scale = saved["mean"].astype(np.float32), saved["scale"].astype(np.float32)
        info = json.loads(str(saved["info"]))
    bars = pd.read_csv(Path(info["data_dir"]) / "bars.csv")
    bars = bars.set_index("bar_id").loc[bar_id].reset_index()
    if info.get("transpose_to_c"):
        bars["shift_to_c"] = bars.key.map(shift_to_c)
    return Embeddings(bars, mean, scale, info)


def default_output_path(checkpoint: Path | str, transpose_to_c: bool, untrained: bool) -> Path:
    """``<run dir>/embeddings[_to_c][_untrained].npz``"""
    name = "embeddings" + ("_to_c" if transpose_to_c else "") + ("_untrained" if untrained else "")
    return Path(checkpoint).parent / f"{name}.npz"
