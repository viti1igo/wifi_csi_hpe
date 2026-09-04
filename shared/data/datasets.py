from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .preprocessing import csi_amplitude, reshape_csi_window


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if "csi_path" not in value or "participant" not in value:
                    raise ValueError(f"Invalid manifest record at line {line_number}")
                records.append(value)
    return records


class ManifestPoseDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Dataset over an exploration-produced JSONL manifest.

    No discovery assumptions are embedded here. The later exploration milestone is
    responsible for mapping raw archives to records and the common 14-joint format.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        normalizer: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.records = records
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        path = Path(record["csi_path"])
        if path.suffix == ".mat":
            payload = loadmat(path)
            raw = payload[record.get("csi_key", "csi_serial")]
        elif path.suffix == ".npy":
            raw = np.load(path, allow_pickle=False)
        elif path.suffix == ".npz":
            payload = np.load(path, allow_pickle=False)
            raw = payload["csi"]
        else:
            raise ValueError(f"Unsupported CSI file type: {path.suffix}")
        csi = reshape_csi_window(csi_amplitude(raw))
        if self.normalizer is not None:
            csi = self.normalizer(csi)
        result: dict[str, torch.Tensor | str] = {
            "csi": torch.from_numpy(csi),
            "participant": str(record["participant"]),
            "sample_id": str(record.get("sample_id", path.stem)),
        }
        if path.suffix == ".npz" and "joints" in payload:
            result["joints"] = torch.from_numpy(payload["joints"].astype(np.float32))
            result["confidence"] = torch.from_numpy(payload["confidence"].astype(np.float32))
        elif "joints" in record:
            result["joints"] = torch.tensor(record["joints"], dtype=torch.float32)
            result["confidence"] = torch.tensor(record.get("confidence", [1.0] * 14), dtype=torch.float32)
        return result
