from __future__ import annotations

from pathlib import Path

import torch

from shared.models import PoseCNN


class DomainAdaptedPoseModel(PoseCNN):
    """PoseCNN whose compatible encoder weights may come from masked pretraining."""

    def load_pretrained_encoder(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("encoder", payload)
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected pretrained encoder keys: {unexpected}")
        if len(missing) == len(self.encoder.state_dict()):
            raise ValueError("No compatible pretrained encoder parameters were loaded")

