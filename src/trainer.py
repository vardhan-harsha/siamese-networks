import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .losses import get_loss
from .model import build_model


class Trainer:
    def __init__(self, config: dict, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_cfg = config["training"]
        model_cfg = config["model"]
        self.loss_type = train_cfg["loss"]
        self.epochs = train_cfg["epochs"]
        self.patience = train_cfg["patience"]

        self.model = build_model(
            loss_type=self.loss_type,
            embedding_dim=model_cfg["embedding_dim"],
        ).to(self.device)

        self.criterion = get_loss(self.loss_type, train_cfg["margin"])
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_contrastive(self, batch: tuple) -> torch.Tensor:
        img1, img2, label = [x.to(self.device) for x in batch]
        _, _, distance = self.model(img1, img2)
        return self.criterion(distance, label)

    def _step_triplet(self, batch: tuple) -> torch.Tensor:
        anchor, positive, negative = [x.to(self.device) for x in batch]
        emb_a, emb_p, emb_n = self.model(anchor, positive, negative)
        return self.criterion(emb_a, emb_p, emb_n)

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss = 0.0
        context = torch.enable_grad if train else torch.no_grad

        with context():
            for batch in tqdm(loader, desc="train" if train else "val", leave=False):
                if train:
                    self.optimizer.zero_grad()

                if self.loss_type == "triplet":
                    loss = self._step_triplet(batch)
                else:
                    loss = self._step_contrastive(batch)

                if train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, train_loader, val_loader) -> None:
        best_val = float("inf")
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch:3d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                self.save_checkpoint("best_model.pth")
                print(f"  ✓ Saved best model (val_loss={best_val:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

        self.save_checkpoint("last_model.pth")

    def save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.history = ckpt.get("history", self.history)
        print(f"Loaded checkpoint: {path}")
