"""Command-line interface: ``pianotree-dcml <command>``."""

from __future__ import annotations

import argparse
from pathlib import Path

from pianotree_dcml.data.download import DEFAULT_ZIP_PATH, download_dcml
from pianotree_dcml.data.prepare import prepare

DEFAULT_DATA_DIR = Path("data/processed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pianotree-dcml", description="Train a PianoTree-VAE on the DCML corpora."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download the DCML corpora release")
    download.add_argument("--dest", type=Path, default=DEFAULT_ZIP_PATH,
                          help=f"where to save the zip (default: {DEFAULT_ZIP_PATH})")
    download.add_argument("--force", action="store_true", help="download even if the file exists")

    prep = commands.add_parser("prepare", help="build bars.csv and notes.csv from the release")
    prep.add_argument("--zip", type=Path, default=DEFAULT_ZIP_PATH,
                      help=f"the DCML release zip (default: {DEFAULT_ZIP_PATH})")
    prep.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR,
                      help=f"output directory (default: {DEFAULT_DATA_DIR})")
    prep.add_argument("--valid-fraction", type=float, default=0.1,
                      help="share of each corpus's bars used for validation (default: 0.1)")
    prep.add_argument("--seed", type=int, default=0, help="seed of the train/valid split")

    args = parser.parse_args(argv)
    if args.command == "download":
        download_dcml(args.dest, args.force)
    elif args.command == "prepare":
        prepare(args.zip, args.out_dir, args.valid_fraction, args.seed)


if __name__ == "__main__":
    main()
