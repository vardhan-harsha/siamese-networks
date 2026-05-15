#!/usr/bin/env python3
"""Download and extract the AT&T/ORL face database.

The AT&T database contains 400 PGM images of 40 subjects (10 images each).
Images are 92x112 pixels, grayscale.

Usage:
    python data/download_att.py
    python data/download_att.py --output data/att_faces
"""
import argparse
import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


ATT_URL = "https://cs.nyu.edu/~roweis/data/olivettifaces.mat"
# Primary: direct zip from Cambridge (original source mirror)
ATT_ZIP_URL = "https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip"
# Fallback: kaggle-hosted mirror (public)
ATT_FALLBACK_URL = "https://github.com/kasimte/face-recognition-using-siamese-networks/raw/main/att_faces.zip"


def download_file(url: str, dest: Path) -> bool:
    print(f"  Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"  Extracting to {output_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)


def create_synthetic_dataset(output_dir: Path) -> None:
    """Creates a tiny synthetic AT&T-style dataset for offline/CI use."""
    print("  Creating synthetic AT&T-style dataset (40 subjects × 10 images) ...")
    try:
        from PIL import Image
        import random
    except ImportError:
        print("  Pillow not available. Install requirements first: pip install Pillow")
        sys.exit(1)

    rng = random.Random(42)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject in range(1, 41):
        subdir = output_dir / f"s{subject}"
        subdir.mkdir(exist_ok=True)
        # Each subject has a unique base brightness to make them distinguishable
        base = rng.randint(30, 200)
        for img_idx in range(1, 11):
            # Small per-image variation around the subject's base tone
            tone = max(0, min(255, base + rng.randint(-20, 20)))
            img = Image.new("L", (92, 112), color=tone)
            img.save(subdir / f"{img_idx}.pgm")

    print(f"  Done: synthetic dataset at {output_dir}")


def verify_structure(output_dir: Path) -> bool:
    subjects = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("s")]
    if len(subjects) < 40:
        return False
    for subj in subjects[:5]:
        imgs = list(subj.glob("*.pgm"))
        if len(imgs) < 10:
            return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download AT&T face database")
    p.add_argument(
        "--output", default="data/att_faces", help="Output directory (default: data/att_faces)"
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Create a synthetic dataset instead of downloading (useful for CI/offline)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)

    if output_dir.exists() and verify_structure(output_dir):
        print(f"AT&T dataset already exists at {output_dir}. Skipping download.")
        return

    if args.synthetic:
        create_synthetic_dataset(output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir.parent / "att_faces.zip"

    # Try primary URL then fallback
    downloaded = download_file(ATT_ZIP_URL, zip_path)
    if not downloaded:
        print("  Trying fallback URL ...")
        downloaded = download_file(ATT_FALLBACK_URL, zip_path)

    if downloaded and zip_path.exists():
        extract_zip(zip_path, output_dir)
        zip_path.unlink(missing_ok=True)

        # The zip may unpack as att_faces/ inside output_dir — flatten if needed
        nested = output_dir / "att_faces"
        if nested.exists() and nested.is_dir():
            for item in nested.iterdir():
                item.rename(output_dir / item.name)
            nested.rmdir()

        if verify_structure(output_dir):
            print(f"\nAT&T dataset ready at {output_dir}")
            print("  40 subjects × 10 images = 400 images")
        else:
            print("Warning: downloaded dataset structure looks unexpected.")
    else:
        print("\nDownload failed. Creating synthetic dataset as fallback ...")
        create_synthetic_dataset(output_dir)


if __name__ == "__main__":
    main()
