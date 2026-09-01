from __future__ import annotations

import argparse
from pathlib import Path

import torch

from conditions.source_only_zero_shot.models import SourceOnlyPoseModel
from shared.training import SkeletonAwareLoss
from shared.utils import deep_merge, load_yaml


def build_components(config: dict) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    model = SourceOnlyPoseModel(
        channels=tuple(config["model"]["channels"]),
        num_joints=config["data"]["num_joints"],
        dropout=config["model"]["dropout"],
    )
    criterion = SkeletonAwareLoss(
        coordinate_weight=config["loss"]["coordinate_weight"],
        bone_weight=config["loss"]["bone_weight"],
        symmetry_weight=config["loss"]["symmetry_weight"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    return model, criterion, optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--prepare-only", action="store_true", help="Build config/model only; do not train")
    args = parser.parse_args()
    base = load_yaml(args.project_root / "shared/configs/base.yaml")
    condition = load_yaml(args.project_root / "conditions/source_only_zero_shot/configs/condition.yaml")
    config = deep_merge(base, condition)
    if args.prepare_only:
        build_components(config)
        return
    raise RuntimeError("Training is intentionally locked until exploration and preprocessing are approved")


if __name__ == "__main__":
    main()
