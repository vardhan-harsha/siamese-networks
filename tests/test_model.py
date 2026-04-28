import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from src.model import EmbeddingNet, SiameseNet, TripletNet, build_model


BATCH = 4
C, H, W = 1, 105, 105
EMB_DIM = 128


@pytest.fixture
def backbone():
    return EmbeddingNet(in_channels=C, embedding_dim=EMB_DIM)


@pytest.fixture
def dummy_imgs():
    return torch.randn(BATCH, C, H, W)


class TestEmbeddingNet:
    def test_output_shape(self, backbone, dummy_imgs):
        emb = backbone(dummy_imgs)
        assert emb.shape == (BATCH, EMB_DIM)

    def test_l2_normalized(self, backbone, dummy_imgs):
        emb = backbone(dummy_imgs)
        norms = emb.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

    def test_deterministic(self, backbone, dummy_imgs):
        backbone.eval()
        with torch.no_grad():
            e1 = backbone(dummy_imgs)
            e2 = backbone(dummy_imgs)
        assert torch.allclose(e1, e2)

    def test_different_inputs_different_embeddings(self, backbone):
        backbone.eval()
        x1 = torch.randn(2, C, H, W)
        x2 = torch.randn(2, C, H, W)
        with torch.no_grad():
            e1 = backbone(x1)
            e2 = backbone(x2)
        assert not torch.allclose(e1, e2)


class TestSiameseNet:
    def test_forward_shapes(self, backbone, dummy_imgs):
        net = SiameseNet(backbone)
        x1 = dummy_imgs
        x2 = torch.randn(BATCH, C, H, W)
        emb1, emb2, dist = net(x1, x2)
        assert emb1.shape == (BATCH, EMB_DIM)
        assert emb2.shape == (BATCH, EMB_DIM)
        assert dist.shape == (BATCH,)

    def test_distance_non_negative(self, backbone, dummy_imgs):
        net = SiameseNet(backbone)
        x2 = torch.randn(BATCH, C, H, W)
        _, _, dist = net(dummy_imgs, x2)
        assert (dist >= 0).all()

    def test_same_input_zero_distance(self, backbone, dummy_imgs):
        net = SiameseNet(backbone)
        net.eval()
        with torch.no_grad():
            _, _, dist = net(dummy_imgs, dummy_imgs)
        assert torch.allclose(dist, torch.zeros(BATCH), atol=1e-4)

    def test_get_embedding(self, backbone, dummy_imgs):
        net = SiameseNet(backbone)
        emb = net.get_embedding(dummy_imgs)
        assert emb.shape == (BATCH, EMB_DIM)


class TestTripletNet:
    def test_forward_shapes(self, backbone, dummy_imgs):
        net = TripletNet(backbone)
        anchor = dummy_imgs
        positive = torch.randn(BATCH, C, H, W)
        negative = torch.randn(BATCH, C, H, W)
        e_a, e_p, e_n = net(anchor, positive, negative)
        assert e_a.shape == (BATCH, EMB_DIM)
        assert e_p.shape == (BATCH, EMB_DIM)
        assert e_n.shape == (BATCH, EMB_DIM)

    def test_get_embedding(self, backbone, dummy_imgs):
        net = TripletNet(backbone)
        emb = net.get_embedding(dummy_imgs)
        assert emb.shape == (BATCH, EMB_DIM)


class TestBuildModel:
    def test_build_siamese(self):
        model = build_model("contrastive", EMB_DIM)
        assert isinstance(model, SiameseNet)

    def test_build_triplet(self):
        model = build_model("triplet", EMB_DIM)
        assert isinstance(model, TripletNet)

    def test_shared_weights(self):
        model = build_model("contrastive", EMB_DIM)
        x1 = torch.randn(2, C, H, W)
        x2 = torch.randn(2, C, H, W)
        emb1, emb2, _ = model(x1, x2)
        # Both images processed by same backbone
        direct_emb1 = model.embedding_net(x1)
        assert torch.allclose(emb1, direct_emb1)
