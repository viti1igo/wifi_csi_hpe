from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    def __init__(self, result_dir: str | Path, mode: str = "min") -> None:
        self.directory = Path(result_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")

    def _improved(self, metric: float) -> bool:
        return metric < self.best if self.mode == "min" else metric > self.best

    def save(self, state: dict[str, Any], metric: float) -> bool:
        torch.save(state, self.directory / "last.pt")
        if self._improved(metric):
            self.best = metric
            torch.save(state, self.directory / "best.pt")
            return True
        return False

