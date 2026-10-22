"""Training loop of PianoTree-VAE.

A run lives in its own directory:

- ``config.json``: the configuration of the run;
- ``metrics.csv``: losses and validation scores after every epoch;
- ``last.pt``: the latest checkpoint, used to resume training;
- ``best.pt``: the checkpoint with the lowest validation loss.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pianotree_dcml.config import Config
from pianotree_dcml.data.dataset import BarDataset
from pianotree_dcml.losses import vae_loss
from pianotree_dcml.metrics import NoteCounts, batch_note_counts
from pianotree_dcml.model import PianoTreeVAE


def train(config: Config, run_dir: Path | str, resume: bool = False) -> None:
    run_dir = Path(run_dir)
    if (run_dir / "last.pt").exists() and not resume:
        raise FileExistsError(
            f"{run_dir} already holds a run: resume it with --resume or choose another directory"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    _seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    train_set, valid_set = load_datasets(config)
    print(f"{len(train_set)} training bars, {len(valid_set)} validation bars; "
          f"left out of training: {train_set.n_excluded}")
    loader_args = dict(batch_size=config.train.batch_size, num_workers=config.train.num_workers,
                       pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)

    model = build_model(config).to(device)
    print(f"Training on {device}: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    floor = config.train.min_learning_rate / config.train.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(config.train.lr_decay**step, floor)
    )

    start_epoch, best_valid_loss = 0, math.inf
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["rng_state"])
        start_epoch, best_valid_loss = checkpoint["epoch"], checkpoint["best_valid_loss"]
        print(f"Resuming after epoch {start_epoch}")

    for epoch in range(start_epoch, config.train.epochs):
        start = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, epoch, device)
        valid_metrics = evaluate(model, valid_loader, config, device)
        row = {"epoch": epoch + 1, "seconds": round(time.time() - start),
               "learning_rate": scheduler.get_last_lr()[0]}
        row |= {f"train_{k}": v for k, v in train_metrics.items()}
        row |= {f"valid_{k}": v for k, v in valid_metrics.items()}
        _append_csv(run_dir / "metrics.csv", row)
        print(f"Epoch {epoch + 1}/{config.train.epochs}: train loss {train_metrics['loss']:.2f}, "
              f"valid loss {valid_metrics['loss']:.2f}, valid note F1 {valid_metrics['note_f1']:.3f}",
              flush=True)  # show progress live in notebooks and logs, not only in terminals

        checkpoint = {
            "config": config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "epoch": epoch + 1,
            "best_valid_loss": min(best_valid_loss, valid_metrics["loss"]),
        }
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            torch.save(checkpoint, run_dir / "best.pt")
        torch.save(checkpoint, run_dir / "last.pt")


def train_epoch(model, loader, optimizer, scheduler, config: Config, epoch: int, device) -> dict:
    model.train()
    cfg = config.train
    totals, n_bars = {}, 0
    for i, (grids, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)):
        grids = grids.to(device)
        progress = (epoch + i / len(loader)) / cfg.epochs
        pitch_logits, dur_logits, posterior = model(
            grids,
            teacher_forcing_time=teacher_forcing(progress, *cfg.teacher_forcing_time),
            teacher_forcing_notes=teacher_forcing(progress, *cfg.teacher_forcing_notes),
        )
        warmup = min(1.0, (epoch + i / len(loader)) / cfg.beta_warmup_epochs) if cfg.beta_warmup_epochs else 1.0
        losses = vae_loss(pitch_logits, dur_logits, grids, posterior,
                          beta=cfg.beta * warmup, free_nats=cfg.free_nats)

        optimizer.zero_grad()
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        _accumulate(totals, losses, len(grids))
        n_bars += len(grids)
    return {k: v / n_bars for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, config: Config, device) -> dict:
    """Losses and note scores of the free-running model (no teacher forcing), decoding
    from the mean of q(z | bar)."""
    model.eval()
    totals, counts, n_bars = {}, NoteCounts(), 0
    for grids, _ in tqdm(loader, desc="Validation", leave=False):
        grids = grids.to(device)
        pitch_logits, dur_logits, posterior = model(grids, sample=False)
        losses = vae_loss(pitch_logits, dur_logits, grids, posterior,
                          beta=config.train.beta, free_nats=config.train.free_nats)
        _accumulate(totals, losses, len(grids))
        counts += batch_note_counts(pitch_logits, dur_logits, grids)
        n_bars += len(grids)
    return {k: v / n_bars for k, v in totals.items()} | counts.scores()


def teacher_forcing(progress: float, start: float, end: float) -> float:
    """Teacher forcing probability at ``progress`` (0 to 1) through training: a sigmoid
    going from ``start`` to ``end``, halfway at mid-training."""
    return end + (start - end) / (1 + math.exp(10 * (progress - 0.5)))


def build_model(config: Config) -> PianoTreeVAE:
    return PianoTreeVAE(seq_len=config.data.seq_len, max_simu_note=config.data.max_simu_note,
                        **config.model.__dict__)


def load_datasets(config: Config) -> tuple[BarDataset, BarDataset]:
    data = config.data
    common = dict(data_dir=data.data_dir, seq_len=data.seq_len, max_simu_note=data.max_simu_note,
                  corpora=data.corpora, timesigs=data.timesigs)
    return (BarDataset(split="train", transpose=data.transpose, **common),
            BarDataset(split="valid", **common))


def load_checkpoint(path: Path | str, device: str = "cpu") -> tuple[PianoTreeVAE, Config]:
    """Load a trained model (in eval mode) and the configuration it was trained with."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = Config.from_dict(checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), config


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _accumulate(totals: dict, losses: dict, n: int) -> None:
    for name, value in losses.items():
        totals[name] = totals.get(name, 0.0) + value.item() * n


def _append_csv(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
