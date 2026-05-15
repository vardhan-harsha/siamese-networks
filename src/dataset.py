import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Base loader
# ---------------------------------------------------------------------------

def load_att_dataset(data_dir: str) -> list[tuple[str, int]]:
    """Returns list of (image_path, subject_id) for the AT&T face database."""
    root = Path(data_dir)
    samples: list[tuple[str, int]] = []
    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue
        try:
            subject_id = int(subject_dir.name.lstrip("s"))
        except ValueError:
            continue
        for img_file in sorted(subject_dir.iterdir()):
            if img_file.suffix.lower() in (".pgm", ".png", ".jpg", ".jpeg"):
                samples.append((str(img_file), subject_id))
    return samples


def split_by_subject(
    samples: list[tuple[str, int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Splits subjects (not images) into train/val/test to prevent identity leakage."""
    subject_ids = sorted(set(s[1] for s in samples))
    rng = random.Random(seed)
    rng.shuffle(subject_ids)

    n = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(subject_ids[:n_train])
    val_ids = set(subject_ids[n_train : n_train + n_val])
    test_ids = set(subject_ids[n_train + n_val :])

    train = [(p, l) for p, l in samples if l in train_ids]
    val = [(p, l) for p, l in samples if l in val_ids]
    test = [(p, l) for p, l in samples if l in test_ids]
    return train, val, test


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def get_transform(image_size: tuple[int, int], augment: bool = False) -> T.Compose:
    ops: list[Any] = []
    if augment:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    ops += [
        T.Resize(image_size),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ]
    return T.Compose(ops)


# ---------------------------------------------------------------------------
# Pair dataset  (for contrastive loss)
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Generates balanced positive (y=0) and negative (y=1) image pairs.

    Convention: y=0 → same person, y=1 → different person.
    This matches ContrastiveLoss where y=1 triggers the margin penalty.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        image_size: tuple[int, int] = (105, 105),
        augment: bool = False,
        pairs_per_sample: int = 4,
        seed: int = 42,
    ):
        self.transform = get_transform(image_size, augment)
        self.pairs = self._generate_pairs(samples, pairs_per_sample, seed)

    def _generate_pairs(
        self,
        samples: list[tuple[str, int]],
        pairs_per_sample: int,
        seed: int,
    ) -> list[tuple[str, str, int]]:
        rng = random.Random(seed)
        by_class: dict[int, list[str]] = {}
        for path, label in samples:
            by_class.setdefault(label, []).append(path)

        # Only subjects with ≥2 images can provide positive pairs
        valid_classes = [c for c, imgs in by_class.items() if len(imgs) >= 2]
        all_paths = [p for p, _ in samples]
        all_labels = [l for _, l in samples]

        pairs: list[tuple[str, str, int]] = []
        for path, label in samples:
            class_imgs = by_class[label]

            # Positive pairs (same person)
            other_same = [p for p in class_imgs if p != path]
            if other_same:
                for _ in range(pairs_per_sample // 2):
                    pairs.append((path, rng.choice(other_same), 0))

            # Negative pairs (different person)
            for _ in range(pairs_per_sample // 2):
                while True:
                    idx = rng.randrange(len(all_paths))
                    if all_labels[idx] != label:
                        pairs.append((path, all_paths[idx], 1))
                        break

        rng.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, p2, label = self.pairs[idx]
        img1 = self.transform(Image.open(p1).convert("L"))
        img2 = self.transform(Image.open(p2).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Triplet dataset  (for triplet loss)
# ---------------------------------------------------------------------------

class TripletDataset(Dataset):
    """Returns (anchor, positive, negative) image triplets."""

    def __init__(
        self,
        samples: list[tuple[str, int]],
        image_size: tuple[int, int] = (105, 105),
        augment: bool = False,
        triplets_per_sample: int = 4,
        seed: int = 42,
    ):
        self.transform = get_transform(image_size, augment)
        self.triplets = self._generate_triplets(samples, triplets_per_sample, seed)

    def _generate_triplets(
        self,
        samples: list[tuple[str, int]],
        triplets_per_sample: int,
        seed: int,
    ) -> list[tuple[str, str, str]]:
        rng = random.Random(seed)
        by_class: dict[int, list[str]] = {}
        for path, label in samples:
            by_class.setdefault(label, []).append(path)

        valid_classes = [c for c, imgs in by_class.items() if len(imgs) >= 2]
        all_classes = list(by_class.keys())

        triplets: list[tuple[str, str, str]] = []
        for path, label in samples:
            if label not in set(valid_classes):
                continue
            class_imgs = by_class[label]
            other_same = [p for p in class_imgs if p != path]
            if not other_same:
                continue

            negative_classes = [c for c in all_classes if c != label]
            for _ in range(triplets_per_sample):
                positive = rng.choice(other_same)
                neg_class = rng.choice(negative_classes)
                negative = rng.choice(by_class[neg_class])
                triplets.append((path, positive, negative))

        rng.shuffle(triplets)
        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, p, n = self.triplets[idx]
        anchor = self.transform(Image.open(a).convert("L"))
        positive = self.transform(Image.open(p).convert("L"))
        negative = self.transform(Image.open(n).convert("L"))
        return anchor, positive, negative


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Returns train/val/test DataLoaders based on config."""
    ds_cfg = config["dataset"]
    train_cfg = config["training"]
    loss_type = train_cfg["loss"]
    image_size = tuple(ds_cfg["image_size"])
    batch_size = train_cfg["batch_size"]

    samples = load_att_dataset(ds_cfg["data_dir"])
    train_s, val_s, test_s = split_by_subject(
        samples, ds_cfg["train_split"], ds_cfg["val_split"]
    )

    DatasetCls = TripletDataset if loss_type == "triplet" else PairDataset

    train_ds = DatasetCls(train_s, image_size=image_size, augment=True)
    val_ds = DatasetCls(val_s, image_size=image_size, augment=False)
    test_ds = DatasetCls(test_s, image_size=image_size, augment=False)

    import torch
    cuda = torch.cuda.is_available()
    workers = 2 if cuda else 0

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=cuda),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=cuda),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=cuda),
        "train_samples": train_s,
        "val_samples": val_s,
        "test_samples": test_s,
    }
