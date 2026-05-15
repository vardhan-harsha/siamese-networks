import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from src.losses import ContrastiveLoss, TripletLoss, get_loss


MARGIN = 1.0
BATCH = 8
DIM = 128


class TestContrastiveLoss:
    def setup_method(self):
        self.loss_fn = ContrastiveLoss(margin=MARGIN)

    def test_same_pair_zero_distance_zero_loss(self):
        distance = torch.zeros(BATCH)
        label = torch.zeros(BATCH)  # y=0: same person
        loss = self.loss_fn(distance, label)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_diff_pair_at_margin_zero_loss(self):
        distance = torch.full((BATCH,), MARGIN)
        label = torch.ones(BATCH)  # y=1: different person
        loss = self.loss_fn(distance, label)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_diff_pair_beyond_margin_zero_loss(self):
        distance = torch.full((BATCH,), MARGIN + 0.5)
        label = torch.ones(BATCH)
        loss = self.loss_fn(distance, label)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_same_pair_large_distance_positive_loss(self):
        distance = torch.full((BATCH,), 2.0)
        label = torch.zeros(BATCH)
        loss = self.loss_fn(distance, label)
        assert loss.item() > 0.0

    def test_diff_pair_zero_distance_positive_loss(self):
        distance = torch.zeros(BATCH)
        label = torch.ones(BATCH)
        loss = self.loss_fn(distance, label)
        assert loss.item() > 0.0

    def test_loss_non_negative(self):
        distance = torch.rand(BATCH)
        label = torch.randint(0, 2, (BATCH,)).float()
        loss = self.loss_fn(distance, label)
        assert loss.item() >= 0.0

    def test_gradients_flow(self):
        distance = torch.rand(BATCH, requires_grad=True)
        label = torch.randint(0, 2, (BATCH,)).float()
        loss = self.loss_fn(distance, label)
        loss.backward()
        assert distance.grad is not None


class TestTripletLoss:
    def setup_method(self):
        self.loss_fn = TripletLoss(margin=MARGIN)

    def test_easy_triplet_zero_loss(self):
        # d(a,p) << d(a,n): loss should be 0
        anchor = torch.zeros(BATCH, DIM)
        positive = torch.zeros(BATCH, DIM) + 0.01
        negative = torch.ones(BATCH, DIM) * 5.0
        loss = self.loss_fn(anchor, positive, negative)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_hard_triplet_positive_loss(self):
        # d(a,p) > d(a,n): loss should be positive
        anchor = torch.zeros(BATCH, DIM)
        positive = torch.ones(BATCH, DIM) * 5.0
        negative = torch.zeros(BATCH, DIM) + 0.01
        loss = self.loss_fn(anchor, positive, negative)
        assert loss.item() > 0.0

    def test_loss_non_negative(self):
        anchor = torch.randn(BATCH, DIM)
        positive = torch.randn(BATCH, DIM)
        negative = torch.randn(BATCH, DIM)
        loss = self.loss_fn(anchor, positive, negative)
        assert loss.item() >= 0.0

    def test_gradients_flow(self):
        anchor = torch.randn(BATCH, DIM, requires_grad=True)
        positive = torch.randn(BATCH, DIM, requires_grad=True)
        negative = torch.randn(BATCH, DIM, requires_grad=True)
        loss = self.loss_fn(anchor, positive, negative)
        loss.backward()
        assert anchor.grad is not None
        assert positive.grad is not None
        assert negative.grad is not None


class TestGetLoss:
    def test_returns_contrastive(self):
        assert isinstance(get_loss("contrastive", 1.0), ContrastiveLoss)

    def test_returns_triplet(self):
        assert isinstance(get_loss("triplet", 0.3), TripletLoss)

    def test_default_returns_contrastive(self):
        assert isinstance(get_loss("unknown_type", 1.0), ContrastiveLoss)
