"""Download a pinned release of the DCML corpora."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

DCML_VERSION = "v2.3"
DCML_URL = (
    f"https://github.com/DCMLab/dcml_corpora/releases/download/{DCML_VERSION}/dcml_corpora.zip"
)
DCML_SHA256 = "245fd8f04f09e1cb121e8c66427ca9cba9cf8aac17beadabbc223d402346ddff"
DEFAULT_ZIP_PATH = Path("data/raw/dcml_corpora.zip")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dcml(dest: Path = DEFAULT_ZIP_PATH, force: bool = False) -> Path:
    """Download the DCML corpora release zip to ``dest`` and verify its checksum.

    Does nothing if a file with the right checksum is already there (unless ``force``).
    """
    dest = Path(dest)
    if dest.exists() and not force:
        if sha256(dest) == DCML_SHA256:
            print(f"{dest} already exists and is valid.")
            return dest
        raise RuntimeError(
            f"{dest} exists but does not match DCML corpora {DCML_VERSION}. "
            "Delete it or use --force to download it again."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading DCML corpora {DCML_VERSION} (24 MB) from {DCML_URL}")
    with urllib.request.urlopen(DCML_URL) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    if sha256(tmp) != DCML_SHA256:
        tmp.unlink()
        raise RuntimeError("Downloaded file does not match the expected checksum.")
    tmp.replace(dest)
    print(f"Saved to {dest}")
    return dest
