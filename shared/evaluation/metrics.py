from __future__ import annotations

import torch

from shared.data.skeleton import BODY_14_EDGES


def pose_metrics(prediction: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> dict[str, float]:
    valid = confidence > 0
    distances = torch.linalg.vector_norm(prediction - target, dim=-1)
    valid_count = valid.sum().clamp_min(1)
    nme = (distances * valid).sum() / valid_count
    result = {"nme": float(nme)}
    for threshold in (0.05, 0.10, 0.20):
        result[f"pck_{threshold:.2f}"] = float(((distances <= threshold) & valid).sum() / valid_count)
    pred_lengths = torch.stack([torch.linalg.vector_norm(prediction[:, b] - prediction[:, a], dim=-1) for a, b in BODY_14_EDGES], dim=1)
    true_lengths = torch.stack([torch.linalg.vector_norm(target[:, b] - target[:, a], dim=-1) for a, b in BODY_14_EDGES], dim=1)
    result["bone_length_error"] = float((pred_lengths - true_lengths).abs().mean())
    return result

