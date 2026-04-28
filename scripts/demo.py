#!/usr/bin/env python3
"""One-shot facial recognition demo.

Enroll a new identity with a single image, then verify query images against it.

Usage — verify two images:
    python scripts/demo.py --checkpoint checkpoints/best_model.pth \\
        --enroll data/att_faces/s1/1.pgm \\
        --query  data/att_faces/s1/2.pgm

Usage — gallery mode (enroll a whole folder, query a single image):
    python scripts/demo.py --checkpoint checkpoints/best_model.pth \\
        --gallery data/att_faces/s1/ \\
        --query   data/att_faces/s2/1.pgm

Usage — identify query among multiple enrolled identities:
    python scripts/demo.py --checkpoint checkpoints/best_model.pth \\
        --identities data/att_faces/ \\
        --query      data/att_faces/s3/10.pgm
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_transform
from src.evaluator import Evaluator
from src.model import build_model
from PIL import Image


IMAGE_EXTS = {".pgm", ".png", ".jpg", ".jpeg"}


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", config)
    model = build_model(
        loss_type=cfg["training"]["loss"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def embed_image(path: str, model: torch.nn.Module, transform, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    embedding_net = getattr(model, "embedding_net", model)
    with torch.no_grad():
        return embedding_net(tensor).squeeze(0)


def distance_to_similarity(distance: float, scale: float = 5.0) -> float:
    """Converts Euclidean distance to a [0,1] similarity score."""
    import math
    return math.exp(-scale * distance)


def verify(
    enroll_path: str,
    query_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> None:
    emb_enroll = embed_image(enroll_path, model, transform, device)
    emb_query = embed_image(query_path, model, transform, device)
    dist = F.pairwise_distance(emb_enroll.unsqueeze(0), emb_query.unsqueeze(0)).item()
    similarity = distance_to_similarity(dist)
    verdict = "Same person" if dist <= threshold else "Different person"
    print(f"\n  Enrolled : {enroll_path}")
    print(f"  Query    : {query_path}")
    print(f"  Distance : {dist:.4f}  |  Similarity : {similarity:.4f}")
    print(f"  Verdict  : {verdict}  (threshold={threshold:.2f})\n")


def gallery_mode(
    gallery_dir: str,
    query_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    threshold: float = 0.5,
) -> None:
    gallery_paths = [
        str(p) for p in Path(gallery_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
    ]
    if not gallery_paths:
        print(f"No images found in {gallery_dir}")
        return

    emb_query = embed_image(query_path, model, transform, device)
    emb_gallery = [embed_image(p, model, transform, device) for p in gallery_paths]

    avg_emb = torch.stack(emb_gallery).mean(0)
    avg_emb = F.normalize(avg_emb, p=2, dim=0)

    dist = F.pairwise_distance(avg_emb.unsqueeze(0), emb_query.unsqueeze(0)).item()
    similarity = distance_to_similarity(dist)
    verdict = "Same person" if dist <= threshold else "Different person"

    print(f"\n  Gallery  : {gallery_dir} ({len(gallery_paths)} images)")
    print(f"  Query    : {query_path}")
    print(f"  Distance : {dist:.4f}  |  Similarity : {similarity:.4f}")
    print(f"  Verdict  : {verdict}  (threshold={threshold:.2f})\n")


def identify_mode(
    identities_dir: str,
    query_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    top_k: int = 3,
) -> None:
    """Identify query face among multiple enrolled identities (one-shot: first image per person)."""
    identities: dict[str, torch.Tensor] = {}
    root = Path(identities_dir)
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = sorted(p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if imgs:
            emb = embed_image(str(imgs[0]), model, transform, device)
            identities[subdir.name] = emb

    if not identities:
        print(f"No identity folders found in {identities_dir}")
        return

    emb_query = embed_image(query_path, model, transform, device)
    scores = {
        name: F.pairwise_distance(emb.unsqueeze(0), emb_query.unsqueeze(0)).item()
        for name, emb in identities.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1])

    print(f"\n  Query  : {query_path}")
    print(f"  Top-{top_k} matches:")
    for i, (name, dist) in enumerate(ranked[:top_k], 1):
        sim = distance_to_similarity(dist)
        print(f"    {i}. {name:10s}  distance={dist:.4f}  similarity={sim:.4f}")
    best_name, best_dist = ranked[0]
    print(f"\n  Best match : {best_name}  (distance={best_dist:.4f})\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-shot face verification demo")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--threshold", type=float, default=0.5, help="Verification distance threshold")
    p.add_argument("--query", required=True, help="Query image path")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--enroll", help="Single enrollment image (direct verify)")
    mode.add_argument("--gallery", help="Directory of images for one identity (gallery verify)")
    mode.add_argument("--identities", help="Directory of identity folders (identify mode)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model, cfg = load_model(args.checkpoint, config, device)
    transform = get_transform(tuple(cfg["dataset"]["image_size"]), augment=False)

    print(f"Model loaded | device={device} | loss={cfg['training']['loss']}")

    if args.enroll:
        verify(args.enroll, args.query, model, transform, device, args.threshold)
    elif args.gallery:
        gallery_mode(args.gallery, args.query, model, transform, device, args.threshold)
    else:
        identify_mode(args.identities, args.query, model, transform, device)


if __name__ == "__main__":
    main()
