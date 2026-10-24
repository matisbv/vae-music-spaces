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

    train_cmd = commands.add_parser("train", help="train a model")
    train_cmd.add_argument("config", type=Path, help="TOML configuration (see configs/)")
    train_cmd.add_argument("--run-dir", type=Path, required=True,
                           help="directory for checkpoints and metrics, e.g. runs/tiny")
    train_cmd.add_argument("--resume", action="store_true", help="continue the run in --run-dir")

    encode = commands.add_parser(
        "encode", help="save the latent code of every bar (needs the analysis extra)")
    encode.add_argument("checkpoint", type=Path, help="a trained model, e.g. runs/full/best.pt")
    encode.add_argument("--out", type=Path, default=None,
                        help="output .npz file (default: embeddings*.npz next to the checkpoint)")
    encode.add_argument("--to-c", action="store_true",
                        help="transpose every piece to C major / C minor before encoding")
    encode.add_argument("--untrained", action="store_true",
                        help="use randomly initialised weights instead (the control model)")
    encode.add_argument("--seed", type=int, default=0, help="seed of the untrained weights")
    encode.add_argument("--device", default="cpu", help="cpu or cuda")

    args = parser.parse_args(argv)
    if args.command == "download":
        download_dcml(args.dest, args.force)
    elif args.command == "prepare":
        prepare(args.zip, args.out_dir, args.valid_fraction, args.seed)
    elif args.command == "train":
        # Imported here so that the data commands work without loading PyTorch.
        from pianotree_dcml.config import Config
        from pianotree_dcml.train import train

        train(Config.from_toml(args.config), args.run_dir, args.resume)
    elif args.command == "encode":
        from pianotree_dcml.analysis.encode import default_output_path, encode_bars, save_embeddings

        embeddings = encode_bars(args.checkpoint, transpose_to_c=args.to_c,
                                 untrained=args.untrained, seed=args.seed, device=args.device)
        out = args.out or default_output_path(args.checkpoint, args.to_c, args.untrained)
        save_embeddings(embeddings, out)
        print(f"Saved the codes of {len(embeddings)} bars to {out} "
              f"(left out: {embeddings.info['excluded']})")


if __name__ == "__main__":
    main()
