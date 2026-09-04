from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from shared.data.datasets import ManifestPoseDataset, load_manifest
from shared.data.preprocessing import csi_amplitude, reshape_csi_window


def _monitor(sample_id: str, seed: int, fraction: float) -> bool:
    value = int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:16], 16)
    return value / 16**16 < fraction


class BalancedSSLDataset(Dataset):
    """One WiMANS recording and one sampled Wi-Pose window per source pair."""

    def __init__(self, wipose: list[dict], wimans: list[dict], normalizer: Callable, seed: int = 31050) -> None:
        if len(wipose) < len(wimans) or not wimans:
            raise ValueError("Balanced SSL requires at least as many Wi-Pose samples as WiMANS recordings")
        self.wipose = ManifestPoseDataset(wipose, normalizer)
        self.wimans = wimans
        self.normalizer = normalizer
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        self.wipose_indices = torch.randperm(len(self.wipose), generator=generator)[: len(self.wimans)].tolist()
        self.epoch = epoch

    def __len__(self) -> int:
        return 2 * len(self.wimans)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        source_index, source = divmod(index, 2)
        if source == 0:
            sample = self.wipose[self.wipose_indices[source_index]]
            return {"csi": sample["csi"], "source": "wipose", "sample_id": sample["sample_id"]}
        record = self.wimans[source_index]
        raw = np.load(record["csi_path"], mmap_mode="r", allow_pickle=False)
        maximum = raw.shape[0] - 5
        digest = hashlib.sha256(f"{self.seed + self.epoch}:{record['sample_id']}".encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % (maximum + 1)
        csi = self.normalizer(reshape_csi_window(csi_amplitude(np.asarray(raw[offset : offset + 5]))))
        return {"csi": torch.from_numpy(csi), "source": "wimans", "sample_id": record["sample_id"]}


def build_ssl_loaders(
    project_root: str | Path,
    normalizer: Callable,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 31050,
    monitor_fraction: float = 0.10,
) -> tuple[dict[str, DataLoader], dict[str, int]]:
    root = Path(project_root)
    wipose = [row for row in load_manifest(root / "data/manifests/wipose_split_v1.jsonl") if row["partition"] == "train"]
    with (root / "data/manifests/wimans_partitions_v1.jsonl").open(encoding="utf-8") as handle:
        wimans = [json.loads(line) for line in handle if line.strip()]
    permitted = [row for row in wimans if row["partition"] == "ssl_train" and row["environment"] in {"classroom", "meeting_room"}]
    excluded = [row for row in wimans if row["environment"] == "empty_room"]
    wp_monitor = [row for row in wipose if _monitor(row["sample_id"], seed, monitor_fraction)]
    wm_monitor = [row for row in permitted if _monitor(row["sample_id"], seed, monitor_fraction)]
    wp_monitor_ids = {row["sample_id"] for row in wp_monitor}
    wm_monitor_ids = {row["sample_id"] for row in wm_monitor}
    wp_train = [row for row in wipose if row["sample_id"] not in wp_monitor_ids]
    wm_train = [row for row in permitted if row["sample_id"] not in wm_monitor_ids]
    monitor_size = min(len(wp_monitor), len(wm_monitor))
    datasets = {
        "train": BalancedSSLDataset(wp_train, wm_train, normalizer, seed),
        "monitor": BalancedSSLDataset(wp_monitor[:monitor_size], wm_monitor[:monitor_size], normalizer, seed),
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(dataset, batch_size=batch_size, shuffle=name == "train", num_workers=num_workers,
                         generator=generator if name == "train" else None)
        for name, dataset in datasets.items()
    }
    summary = {
        "wipose_train_pool": len(wp_train), "wimans_train_recordings": len(wm_train),
        "balanced_train_windows": len(datasets["train"]), "monitor_windows": len(datasets["monitor"]),
        "excluded_empty_room_recordings": len(excluded),
    }
    return loaders, summary


class MaskedReconstructionTrainer:
    def __init__(self, model, optimizer, device: torch.device, mask_ratio: float, precision: str = "auto") -> None:
        self.model, self.optimizer, self.device, self.mask_ratio = model, optimizer, device, mask_ratio
        self.cuda = device.type == "cuda"
        self.use_bf16 = self.cuda and (precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()))
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cuda and not self.use_bf16)

    def run_epoch(self, loader, training: bool) -> dict[str, float]:
        self.model.train(training)
        total, count, started = 0.0, 0, time.perf_counter()
        for batch in loader:
            csi = batch["csi"].to(self.device, non_blocking=True)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=self.dtype, enabled=self.cuda):
                reconstruction, mask = self.model(csi, self.mask_ratio)
                loss = ((reconstruction - csi).square() * mask).sum() / (mask.sum() * csi.shape[1]).clamp_min(1)
            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            total += float(loss.detach()) * csi.shape[0]
            count += csi.shape[0]
        if not count:
            raise RuntimeError("Cannot run an epoch on an empty loader")
        return {"masked_mse": total / count, "seconds": time.perf_counter() - started}


def resume_pretraining(model, optimizer, scaler, checkpoint_path, expected_config, expected_checksums):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {
        "epoch", "condition", "stage", "model", "optimizer", "scaler",
        "best_monitor_reconstruction", "epochs_without_improvement", "config",
        "source_manifest_checksum", "split_checksum", "normalizer_checksum",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Pretraining checkpoint is missing fields: {sorted(missing)}")
    if checkpoint["condition"] != "ssl_domain_adapted" or checkpoint["stage"] != "pretraining":
        raise ValueError("Checkpoint is not Condition B pretraining")
    if checkpoint["config"].get("model") != expected_config.get("model"):
        raise ValueError("Checkpoint model configuration does not match")
    for name, expected in expected_checksums.items():
        if checkpoint.get(name) != expected:
            raise ValueError(f"Checkpoint {name} does not match the current artefact")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
