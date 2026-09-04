from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import DataLoader

from conditions.source_only_zero_shot.training.train import build_components
from conditions.ssl_domain_adapted.training.train import build_finetuner, build_pretrainer
from shared.data.datasets import ManifestPoseDataset, load_manifest


def build_wipose_loaders(
    project_root: str | Path,
    normalizer: Callable,
    preview_batch: int = 32,
    num_workers: int = 0,
    partitions: Iterable[str] = ("train", "monitor", "test"),
) -> dict[str, DataLoader]:
    records = load_manifest(Path(project_root) / "data/manifests/wipose_split_v1.jsonl")
    generator = torch.Generator().manual_seed(31050)
    loaders = {}
    for partition in partitions:
        rows = [row for row in records if row["partition"] == partition]
        if not rows:
            raise ValueError(f"No Wi-Pose records in partition {partition!r}")
        loaders[partition] = DataLoader(
            ManifestPoseDataset(rows, normalizer),
            batch_size=preview_batch,
            shuffle=partition == "train",
            num_workers=num_workers,
            generator=generator if partition == "train" else None,
        )
    return loaders


def build_source_components(config: dict) -> tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    return build_components(config)


def build_ssl_components(config: dict) -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    pretrainer = build_pretrainer(config)
    pose_model, criterion, finetune_optimizer = build_finetuner(config)
    pretrain_optimizer = torch.optim.AdamW(
        pretrainer.parameters(),
        lr=config["pretraining"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    return pretrainer, pose_model, criterion, pretrain_optimizer, finetune_optimizer


def load_trained_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_condition: str,
    expected_checksums: dict[str, str],
    expected_config: dict | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "epoch", "condition", "model", "optimizer", "scaler", "best_monitor_nme",
        "epochs_without_improvement", "config", "split_checksum", "normalizer_checksum",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["condition"] != expected_condition:
        raise ValueError(
            f"Expected condition {expected_condition!r}, received {checkpoint['condition']!r}"
        )
    if expected_config is not None and checkpoint["config"].get("model") != expected_config.get("model"):
        raise ValueError("Checkpoint model configuration does not match the resolved configuration")
    for name, expected in expected_checksums.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"Checkpoint {name} does not match the current artefact")
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


def resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    checkpoint_path: str | Path,
    expected_condition: str,
    expected_config: dict,
    expected_checksums: dict[str, str],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "epoch", "condition", "model", "optimizer", "scaler", "best_monitor_nme",
        "epochs_without_improvement", "config", "split_checksum", "normalizer_checksum",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["condition"] != expected_condition:
        raise ValueError(f"Expected condition {expected_condition!r}, received {checkpoint['condition']!r}")
    if checkpoint["config"].get("model") != expected_config.get("model"):
        raise ValueError("Checkpoint model configuration does not match the resolved configuration")
    for name, expected in expected_checksums.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"Checkpoint {name} does not match the current artefact")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
