"""Experiment configuration, read from TOML files (see ``configs/``)."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: str = "data/processed"
    seq_len: int = 16  # bars are padded to this many 16th notes; longer bars are left out
    max_simu_note: int = 16  # note slots per time step, <sos> and <eos> included
    corpora: list[str] | None = None  # None: all corpora
    timesigs: list[str] | None = None  # None: all time signatures
    transpose: tuple[int, int] = (0, 0)  # random transposition range for training, in semitones


@dataclass
class ModelConfig:
    """Layer sizes of PianoTree-VAE (defaults: those of the original paper)."""

    note_emb_size: int = 128
    enc_notes_hid_size: int = 256
    enc_time_hid_size: int = 512
    z_size: int = 512
    dec_emb_hid_size: int = 128
    dec_time_hid_size: int = 1024
    dec_notes_hid_size: int = 512
    dec_z_in_size: int = 256
    dec_dur_hid_size: int = 64


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    lr_decay: float = 0.9999  # multiplies the learning rate after every batch...
    min_learning_rate: float = 5e-5  # ...until it reaches this value
    grad_clip: float = 1.0
    beta: float = 0.1  # weight of the KL term
    beta_warmup_epochs: float = 0.0  # beta grows linearly from 0 during these epochs
    free_nats: float = 0.0
    # Teacher forcing probabilities of the time-level and note-level decoders: they go from
    # the first to the second value over training, following a sigmoid.
    teacher_forcing_time: tuple[float, float] = (0.8, 0.0)
    teacher_forcing_notes: tuple[float, float] = (0.6, 0.0)
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"  # "auto" uses a CUDA GPU when there is one


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_toml(cls, path: Path | str) -> Config:
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))

    @classmethod
    def from_dict(cls, values: dict) -> Config:
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config section(s): {', '.join(sorted(unknown))}")
        return cls(
            data=_section(DataConfig, values.get("data", {})),
            model=_section(ModelConfig, values.get("model", {})),
            train=_section(TrainConfig, values.get("train", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _section(section_cls, values: dict):
    names = {f.name for f in fields(section_cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {section_cls.__name__} option(s): {', '.join(sorted(unknown))}")
    # TOML has no tuples: convert the pairs read as lists.
    values = {k: tuple(v) if isinstance(getattr(section_cls(), k), tuple) else v
              for k, v in values.items()}
    return section_cls(**values)
