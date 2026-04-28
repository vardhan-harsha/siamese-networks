#!/usr/bin/env python3
"""Train a Siamese Network for face verification.

Usage:
    python scripts/train.py
    python scripts/train.py --config config.yaml
    python scripts/train.py --loss triplet
    python scripts/train.py --config config.yaml --loss contrastive --epochs 30
"""
import argparse
import sys
from pathlib import Path

import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.evaluator import Evaluator
from src.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Siamese Network for face verification")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--loss", choices=["contrastive", "triplet"], help="Override loss type")
    p.add_argument("--epochs", type=int, help="Override number of epochs")
    p.add_argument("--batch-size", type=int, dest="batch_size", help="Override batch size")
    p.add_argument("--lr", type=float, help="Override learning rate")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.loss:
        config["training"]["loss"] = args.loss
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    print("=== Configuration ===")
    print(f"  Dataset : {config['dataset']['name']} ({config['dataset']['data_dir']})")
    print(f"  Loss    : {config['training']['loss']}")
    print(f"  Epochs  : {config['training']['epochs']}")
    print(f"  LR      : {config['training']['lr']}")
    print()

    print("Loading data...")
    loaders = get_dataloaders(config)
    print(
        f"  Train pairs/triplets : {len(loaders['train'].dataset)}\n"
        f"  Val   pairs/triplets : {len(loaders['val'].dataset)}\n"
        f"  Test  pairs/triplets : {len(loaders['test'].dataset)}\n"
    )

    trainer = Trainer(config)
    print(f"Training on device: {trainer.device}\n")
    trainer.train(loaders["train"], loaders["val"])

    print("\n=== Quick evaluation on test set ===")
    evaluator = Evaluator(trainer.model, config, device=trainer.device)

    eval_cfg = config["evaluation"]
    if config["training"]["loss"] == "contrastive":
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

    evaluator.plot_loss_curves(trainer.history)
    evaluator.plot_embedding_tsne(loaders["test_samples"])

    print("\nTraining complete. Checkpoints saved to", config["paths"]["checkpoint_dir"])


if __name__ == "__main__":
    main()
