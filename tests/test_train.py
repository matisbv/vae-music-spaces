import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pianotree_dcml.config import Config
from pianotree_dcml.train import load_checkpoint, teacher_forcing, train

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("name", ["tiny.toml", "full.toml"])
def test_shipped_configs_load(name):
    config = Config.from_toml(CONFIGS / name)
    assert isinstance(config.data.transpose, tuple)
    assert Config.from_dict(config.to_dict()) == config


def test_unknown_config_options_are_rejected():
    with pytest.raises(ValueError, match="unknown TrainConfig option"):
        Config.from_dict({"train": {"epoch": 3}})
    with pytest.raises(ValueError, match="unknown config section"):
        Config.from_dict({"optimizer": {}})


def test_teacher_forcing_schedule():
    assert teacher_forcing(0.0, 0.8, 0.0) == pytest.approx(0.8, abs=0.01)
    assert teacher_forcing(0.5, 0.8, 0.0) == pytest.approx(0.4)
    assert teacher_forcing(1.0, 0.8, 0.0) == pytest.approx(0.0, abs=0.01)


@pytest.fixture
def config(tmp_path):
    rng = np.random.default_rng(0)
    bars, notes = [], []
    for bar_id in range(12):
        bars.append((bar_id, "a", "2/4", 8, "valid" if bar_id < 4 else "train"))
        for onset in range(0, 8, 2):
            notes.append((bar_id, onset, int(rng.integers(55, 75)), 2))
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    pd.DataFrame(bars, columns=["bar_id", "corpus", "timesig", "bar_len", "split"]).to_csv(
        data_dir / "bars.csv", index=False)
    pd.DataFrame(notes, columns=["bar_id", "onset", "pitch", "duration"]).to_csv(
        data_dir / "notes.csv", index=False)
    return Config.from_dict({
        "data": {"data_dir": str(data_dir), "seq_len": 8, "max_simu_note": 6, "transpose": [-2, 2]},
        "model": {"note_emb_size": 8, "enc_notes_hid_size": 8, "enc_time_hid_size": 8,
                  "z_size": 4, "dec_emb_hid_size": 8, "dec_time_hid_size": 8,
                  "dec_notes_hid_size": 8, "dec_z_in_size": 4, "dec_dur_hid_size": 4},
        "train": {"epochs": 1, "batch_size": 4, "beta_warmup_epochs": 1},
    })


def read_metrics(run_dir):
    with open(run_dir / "metrics.csv") as f:
        return list(csv.DictReader(f))


def test_train_writes_a_run_and_resumes(config, tmp_path):
    run_dir = tmp_path / "run"
    train(config, run_dir)
    assert {p.name for p in run_dir.iterdir()} == {"config.json", "metrics.csv", "last.pt", "best.pt"}
    rows = read_metrics(run_dir)
    assert [row["epoch"] for row in rows] == ["1"]
    assert {"train_loss", "valid_loss", "valid_kl", "valid_note_f1", "valid_pitch_f1"} <= rows[0].keys()

    with pytest.raises(FileExistsError):
        train(config, run_dir)

    config.train.epochs = 2
    train(config, run_dir, resume=True)
    assert [row["epoch"] for row in read_metrics(run_dir)] == ["1", "2"]
    assert torch.load(run_dir / "last.pt", weights_only=False)["epoch"] == 2


def test_load_checkpoint(config, tmp_path):
    train(config, tmp_path / "run")
    model, loaded_config = load_checkpoint(tmp_path / "run" / "best.pt")
    assert loaded_config == config
    assert not model.training
    assert model.seq_len == 8
