import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """Lightweight CNN that maps a face image to a 128-d L2-normalized embedding."""

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(128, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self._embedding_dim = embedding_dim
        # FC layers sized for 105x105 input; recomputed lazily on first forward
        self.fc = None
        self._conv_out_size: int | None = None

    def _build_fc(self, conv_out_size: int) -> None:
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self._embedding_dim),
        ).to(next(self.conv.parameters()).device)
        self._conv_out_size = conv_out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        if self.fc is None:
            self._build_fc(flat.size(1))
        embedding = self.fc(flat)
        return F.normalize(embedding, p=2, dim=1)


class SiameseNet(nn.Module):
    """Siamese network: passes two images through a shared EmbeddingNet."""

    def __init__(self, embedding_net: EmbeddingNet):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb1 = self.embedding_net(x1)
        emb2 = self.embedding_net(x2)
        distance = F.pairwise_distance(emb1, emb2)
        return emb1, emb2, distance

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_net(x)


class TripletNet(nn.Module):
    """Triplet network: processes (anchor, positive, negative) through shared EmbeddingNet."""

    def __init__(self, embedding_net: EmbeddingNet):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_a = self.embedding_net(anchor)
        emb_p = self.embedding_net(positive)
        emb_n = self.embedding_net(negative)
        return emb_a, emb_p, emb_n

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_net(x)


def build_model(
    loss_type: str = "contrastive",
    embedding_dim: int = 128,
    in_channels: int = 1,
) -> nn.Module:
    """Factory: returns SiameseNet for contrastive loss, TripletNet for triplet loss."""
    backbone = EmbeddingNet(in_channels=in_channels, embedding_dim=embedding_dim)
    if loss_type == "triplet":
        return TripletNet(backbone)
    return SiameseNet(backbone)
