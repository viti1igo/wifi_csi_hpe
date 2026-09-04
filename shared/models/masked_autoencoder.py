from __future__ import annotations

import torch
from torch import nn

from .pose_cnn import ResidualBlock


class MaskedCSIAutoencoder(nn.Module):
    """Compact masked-reconstruction pretrainer for common CSI tensors."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 192)) -> None:
        super().__init__()
        blocks, in_channels = [], 9
        for index, out_channels in enumerate(channels):
            blocks.append(ResidualBlock(in_channels, out_channels, (1, 1) if index == 0 else (1, 2)))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[2], 3, stride=(1, 2), padding=1, output_padding=(0, 1)), nn.GELU(),
            nn.ConvTranspose2d(channels[2], channels[1], 3, stride=(1, 2), padding=1, output_padding=(0, 1)), nn.GELU(),
            nn.ConvTranspose2d(channels[1], channels[0], 3, stride=(1, 2), padding=1, output_padding=(0, 1)), nn.GELU(),
            nn.Conv2d(channels[0], 9, 1),
        )

    @staticmethod
    def make_mask(x: torch.Tensor, ratio: float = 0.30) -> torch.Tensor:
        if not 0.0 < ratio < 1.0:
            raise ValueError("Mask ratio must be between zero and one")
        return torch.rand_like(x[:, :1]) < ratio

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.30) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.make_mask(x, mask_ratio)
        encoded = self.encoder(x.masked_fill(mask, 0.0))
        return self.decoder(encoded)[..., :30], mask
