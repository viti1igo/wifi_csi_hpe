from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from shared.data.skeleton import BODY_14_EDGES, SYMMETRIC_BONE_PAIRS


class SkeletonAwareLoss(nn.Module):
    def __init__(self, coordinate_weight: float = 1.0, bone_weight: float = 0.25, symmetry_weight: float = 0.05) -> None:
        super().__init__()
        self.coordinate_weight = coordinate_weight
        self.bone_weight = bone_weight
        self.symmetry_weight = symmetry_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> dict[str, torch.Tensor]:
        weights = confidence.clamp_min(0).unsqueeze(-1)
        coordinate = (F.smooth_l1_loss(prediction, target, reduction="none") * weights).sum() / weights.sum().clamp_min(1.0)
        pred_bones = torch.stack([prediction[:, b] - prediction[:, a] for a, b in BODY_14_EDGES], dim=1)
        target_bones = torch.stack([target[:, b] - target[:, a] for a, b in BODY_14_EDGES], dim=1)
        bone = F.smooth_l1_loss(pred_bones, target_bones)
        symmetry_terms = []
        for (la, lb), (ra, rb) in SYMMETRIC_BONE_PAIRS:
            left = torch.linalg.vector_norm(prediction[:, lb] - prediction[:, la], dim=-1)
            right = torch.linalg.vector_norm(prediction[:, rb] - prediction[:, ra], dim=-1)
            symmetry_terms.append((left - right).abs())
        symmetry = torch.stack(symmetry_terms, dim=1).mean()
        total = self.coordinate_weight * coordinate + self.bone_weight * bone + self.symmetry_weight * symmetry
        return {"total": total, "coordinate": coordinate, "bone": bone, "symmetry": symmetry}

