#!/usr/bin/env python3
"""Evaluate a trained Siamese Network checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --n-way 10
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.evaluator import Evaluator
from src.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Siamese Network")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--n-way", type=int, dest="n_way", help="Override N for N-way one-shot eval")
    p.add_argument("--n-episodes", type=int, dest="n_episodes", help="Override number of episodes")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", None)

    # Fall back to file config if not embedded in checkpoint
    if config is None:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    if args.n_way:
        config["evaluation"]["n_way"] = args.n_way
    if args.n_episodes:
        config["evaluation"]["n_episodes"] = args.n_episodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_type = config["training"]["loss"]

    model = build_model(
        loss_type=loss_type,
        embedding_dim=config["model"]["embedding_dim"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded model from {args.checkpoint} (loss={loss_type})")

    loaders = get_dataloaders(config)
    evaluator = Evaluator(model, config, device=device)

    eval_cfg = config["evaluation"]
    print("\n=== Evaluation Results ===")

    if loss_type == "contrastive":
        metrics = evaluator.evaluate_pairs(loaders["test"])
        print(f"  Verification Accuracy : {metrics['verification_accuracy']:.4f}")
        print(f"  ROC-AUC               : {metrics['roc_auc']:.4f}")
        print(f"  Optimal threshold     : {metrics['optimal_threshold']:.4f}")

    one_shot_acc = evaluator.evaluate_one_shot(
        loaders["test_samples"],
        n_way=eval_cfg["n_way"],
        n_episodes=eval_cfg["n_episodes"],
    )
    print(f"  {eval_cfg['n_way']}-way 1-shot Accuracy : {one_shot_acc:.4f}")

    # Visualizations
    if "history" in ckpt:
        evaluator.plot_loss_curves(ckpt["history"])
    evaluator.plot_embedding_tsne(loaders["test_samples"])


if __name__ == "__main__":
    main()
