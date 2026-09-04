from __future__ import annotations

from pathlib import Path

import torch

from shared.models import PoseCNN


class DomainAdaptedPoseModel(PoseCNN):
    """PoseCNN whose compatible encoder weights may come from masked pretraining."""

    def load_pretrained_encoder(self, checkpoint: str | Path) -> list[str]:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model", payload.get("encoder", payload))
        state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")} or state
        self.encoder.load_state_dict(state, strict=True)
        return list(state)
