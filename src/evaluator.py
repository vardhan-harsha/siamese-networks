import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .dataset import get_transform


class Evaluator:
    def __init__(self, model: torch.nn.Module, config: dict, device: torch.device | None = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        ds_cfg = config["dataset"]
        self.transform = get_transform(tuple(ds_cfg["image_size"]), augment=False)
        results_dir = Path(config["paths"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = results_dir

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_image(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        embedding_net = getattr(self.model, "embedding_net", self.model)
        return embedding_net(tensor).squeeze(0)

    @torch.no_grad()
    def embed_batch(self, loader) -> tuple[np.ndarray, np.ndarray]:
        embeddings, labels = [], []
        embedding_net = getattr(self.model, "embedding_net", self.model)
        for batch in loader:
            # Works for both PairDataset (img1, img2, label) and TripletDataset
            img = batch[0].to(self.device)
            lbl = batch[-1] if isinstance(batch[-1], torch.Tensor) else torch.zeros(img.size(0))
            emb = embedding_net(img)
            embeddings.append(emb.cpu().numpy())
            labels.append(lbl.numpy() if isinstance(lbl, torch.Tensor) else np.array(lbl))
        return np.concatenate(embeddings), np.concatenate(labels)

    # ------------------------------------------------------------------
    # Pair verification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_pairs(self, test_loader) -> dict[str, float]:
        """Computes verification accuracy and ROC-AUC on a PairDataset loader."""
        distances, labels = [], []
        embedding_net = getattr(self.model, "embedding_net", self.model)

        for img1, img2, label in test_loader:
            img1, img2 = img1.to(self.device), img2.to(self.device)
            emb1 = embedding_net(img1)
            emb2 = embedding_net(img2)
            dist = F.pairwise_distance(emb1, emb2).cpu().numpy()
            distances.extend(dist.tolist())
            labels.extend(label.numpy().tolist())

        distances = np.array(distances)
        labels = np.array(labels)

        # y=0 same person → smaller distance should mean "same"
        # Convert to similarity scores for AUC (higher = more similar)
        similarity = -distances

        auc = roc_auc_score(1 - labels, similarity)  # 1-label: 1=same for sklearn

        # Find optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(1 - labels, similarity)
        j_scores = tpr - fpr
        best_thresh_sim = thresholds[np.argmax(j_scores)]
        best_thresh_dist = -best_thresh_sim

        # Accuracy at optimal threshold
        predictions = (distances <= best_thresh_dist).astype(int)
        ground_truth = (1 - labels).astype(int)  # 1=same person
        accuracy = (predictions == ground_truth).mean()

        self._plot_roc(fpr, tpr, auc)

        return {
            "verification_accuracy": float(accuracy),
            "roc_auc": float(auc),
            "optimal_threshold": float(best_thresh_dist),
        }

    # ------------------------------------------------------------------
    # N-way one-shot evaluation
    # ------------------------------------------------------------------

    def evaluate_one_shot(
        self,
        samples: list[tuple[str, int]],
        n_way: int = 5,
        n_episodes: int = 200,
        seed: int = 42,
    ) -> float:
        """N-way 1-shot: pick N classes, 1 support + 1 query per class, classify query."""
        rng = random.Random(seed)
        by_class: dict[int, list[str]] = {}
        for path, label in samples:
            by_class.setdefault(label, []).append(path)

        valid = [c for c, imgs in by_class.items() if len(imgs) >= 2]
        if len(valid) < n_way:
            print(f"Warning: only {len(valid)} valid classes for {n_way}-way evaluation.")
            n_way = len(valid)

        correct = 0
        for _ in range(n_episodes):
            classes = rng.sample(valid, n_way)
            support_embeddings, query_embeddings = [], []

            for cls in classes:
                imgs = by_class[cls]
                chosen = rng.sample(imgs, 2)
                support_embeddings.append(self.embed_image(chosen[0]))
                query_embeddings.append(self.embed_image(chosen[1]))

            # Classify each query against all support embeddings
            for i, query_emb in enumerate(query_embeddings):
                dists = [
                    F.pairwise_distance(query_emb.unsqueeze(0), s.unsqueeze(0)).item()
                    for s in support_embeddings
                ]
                predicted = int(np.argmin(dists))
                if predicted == i:
                    correct += 1

        accuracy = correct / (n_episodes * n_way)
        return accuracy

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def _plot_roc(self, fpr: np.ndarray, tpr: np.ndarray, auc: float) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Face Verification")
        ax.legend()
        fig.tight_layout()
        path = self.results_dir / "roc_curve.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"ROC curve saved to {path}")

    def plot_loss_curves(self, history: dict) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history["train_loss"], label="Train Loss")
        ax.plot(history["val_loss"], label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Curves")
        ax.legend()
        fig.tight_layout()
        path = self.results_dir / "loss_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Loss curves saved to {path}")

    def plot_embedding_tsne(self, samples: list[tuple[str, int]], max_samples: int = 200) -> None:
        """t-SNE visualization of embeddings coloured by identity."""
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("scikit-learn not available for t-SNE.")
            return

        rng = random.Random(0)
        subset = rng.sample(samples, min(max_samples, len(samples)))

        embs, labels = [], []
        for path, label in subset:
            embs.append(self.embed_image(path).cpu().numpy())
            labels.append(label)

        embs_2d = TSNE(n_components=2, random_state=0).fit_transform(np.array(embs))
        labels_arr = np.array(labels)

        fig, ax = plt.subplots(figsize=(8, 7))
        for lbl in np.unique(labels_arr):
            mask = labels_arr == lbl
            ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], s=20, label=str(lbl), alpha=0.7)
        ax.set_title("t-SNE of Face Embeddings")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6)
        fig.tight_layout()
        path = self.results_dir / "tsne.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"t-SNE plot saved to {path}")
