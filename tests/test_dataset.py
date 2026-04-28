import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

from src.dataset import (
    PairDataset,
    TripletDataset,
    load_att_dataset,
    split_by_subject,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_att(tmp_path):
    """Creates a minimal AT&T-style directory: 5 subjects × 4 images each."""
    for subj in range(1, 6):
        subdir = tmp_path / f"s{subj}"
        subdir.mkdir()
        for i in range(1, 5):
            img = Image.new("L", (92, 112), color=subj * 40)
            img.save(subdir / f"{i}.pgm")
    return tmp_path


@pytest.fixture
def samples(synthetic_att):
    return load_att_dataset(str(synthetic_att))


# ---------------------------------------------------------------------------
# load_att_dataset
# ---------------------------------------------------------------------------

class TestLoadATT:
    def test_correct_count(self, synthetic_att):
        s = load_att_dataset(str(synthetic_att))
        assert len(s) == 20  # 5 subjects × 4 images

    def test_labels_range(self, synthetic_att):
        s = load_att_dataset(str(synthetic_att))
        labels = {label for _, label in s}
        assert labels == {1, 2, 3, 4, 5}

    def test_paths_exist(self, synthetic_att):
        s = load_att_dataset(str(synthetic_att))
        for path, _ in s:
            assert Path(path).exists()


# ---------------------------------------------------------------------------
# split_by_subject
# ---------------------------------------------------------------------------

class TestSplitBySubject:
    def test_no_leakage(self, samples):
        train, val, test = split_by_subject(samples, 0.6, 0.2)
        train_ids = {l for _, l in train}
        val_ids = {l for _, l in val}
        test_ids = {l for _, l in test}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_samples_accounted(self, samples):
        train, val, test = split_by_subject(samples, 0.6, 0.2)
        assert len(train) + len(val) + len(test) == len(samples)


# ---------------------------------------------------------------------------
# PairDataset
# ---------------------------------------------------------------------------

class TestPairDataset:
    def test_len(self, samples):
        ds = PairDataset(samples, image_size=(64, 64), pairs_per_sample=4)
        assert len(ds) > 0

    def test_item_shapes(self, samples):
        ds = PairDataset(samples, image_size=(64, 64), pairs_per_sample=4)
        img1, img2, label = ds[0]
        assert img1.shape == (1, 64, 64)
        assert img2.shape == (1, 64, 64)
        assert label.shape == ()

    def test_labels_binary(self, samples):
        ds = PairDataset(samples, image_size=(64, 64), pairs_per_sample=4)
        for _, _, label in ds:
            assert label.item() in (0, 1)

    def test_balanced_labels(self, samples):
        ds = PairDataset(samples, image_size=(64, 64), pairs_per_sample=4)
        labels = [ds[i][2].item() for i in range(len(ds))]
        n_same = labels.count(0)
        n_diff = labels.count(1)
        # Should be roughly balanced (within 30%)
        ratio = n_same / max(n_diff, 1)
        assert 0.5 < ratio < 2.0


# ---------------------------------------------------------------------------
# TripletDataset
# ---------------------------------------------------------------------------

class TestTripletDataset:
    def test_len(self, samples):
        ds = TripletDataset(samples, image_size=(64, 64), triplets_per_sample=2)
        assert len(ds) > 0

    def test_item_shapes(self, samples):
        ds = TripletDataset(samples, image_size=(64, 64), triplets_per_sample=2)
        anchor, positive, negative = ds[0]
        assert anchor.shape == (1, 64, 64)
        assert positive.shape == (1, 64, 64)
        assert negative.shape == (1, 64, 64)

    def test_triplets_are_distinct(self, samples):
        import torch
        ds = TripletDataset(samples, image_size=(64, 64), triplets_per_sample=2)
        anchor, positive, negative = ds[0]
        # Anchor and negative should differ (different subjects)
        assert not torch.allclose(anchor, negative)
