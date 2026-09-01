from __future__ import annotations

import time
from typing import Any

import torch

from shared.evaluation.metrics import pose_metrics


class Trainer:
    """Shared per-epoch trainer; final-test evaluation is intentionally separate."""

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, precision: str = "auto") -> None:
        self.model, self.optimizer, self.criterion, self.device = model, optimizer, criterion, device
        self.use_bf16 = precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported())
        self.autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=not self.use_bf16)

    def run_epoch(self, loader: Any, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        sample_count = 0
        started = time.perf_counter()
        for batch in loader:
            csi = batch["csi"].to(self.device, non_blocking=True)
            target = batch["joints"].to(self.device, non_blocking=True)
            confidence = batch["confidence"].to(self.device, non_blocking=True)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=self.autocast_dtype):
                prediction = self.model(csi)
                losses = self.criterion(prediction, target, confidence)
            if training:
                self.scaler.scale(losses["total"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            metrics = pose_metrics(prediction.detach(), target, confidence)
            values = {**{f"loss_{k}": float(v.detach()) for k, v in losses.items()}, **metrics}
            batch_size = csi.shape[0]
            sample_count += batch_size
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
        if sample_count == 0:
            raise RuntimeError("Cannot run an epoch on an empty loader")
        return {**{k: v / sample_count for k, v in totals.items()}, "seconds": time.perf_counter() - started}

