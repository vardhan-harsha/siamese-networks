import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks.

    y=0: same person  → penalizes large distance
    y=1: different    → penalizes distance < margin
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        distance: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        same_loss = (1 - label) * 0.5 * distance.pow(2)
        diff_loss = label * 0.5 * F.relu(self.margin - distance).pow(2)
        return (same_loss + diff_loss).mean()


class TripletLoss(nn.Module):
    """Triplet loss: pulls anchor–positive together, pushes anchor–negative apart."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb_anchor: torch.Tensor,
        emb_positive: torch.Tensor,
        emb_negative: torch.Tensor,
    ) -> torch.Tensor:
        d_ap = F.pairwise_distance(emb_anchor, emb_positive)
        d_an = F.pairwise_distance(emb_anchor, emb_negative)
        loss = F.relu(d_ap - d_an + self.margin)
        return loss.mean()


def get_loss(loss_type: str, margin: float) -> nn.Module:
    if loss_type == "triplet":
        return TripletLoss(margin=margin)
    return ContrastiveLoss(margin=margin)
