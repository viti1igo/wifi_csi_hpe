from __future__ import annotations

import torch
from torch import nn

from .pose_cnn import ResidualBlock


class MaskedCSIAutoencoder(nn.Module):
    """Compact masked-reconstruction pretrainer for common CSI tensors."""

    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(9, 32), ResidualBlock(32, 64), ResidualBlock(64, latent_channels),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 9, 1),
        )

    @staticmethod
    def make_mask(x: torch.Tensor, ratio: float = 0.30) -> torch.Tensor:
        if not 0.0 < ratio < 1.0:
            raise ValueError("Mask ratio must be between zero and one")
        return torch.rand_like(x[:, :1]) < ratio

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.30) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.make_mask(x, mask_ratio)
        encoded = self.encoder(x.masked_fill(mask, 0.0))
        return self.decoder(encoded), mask

