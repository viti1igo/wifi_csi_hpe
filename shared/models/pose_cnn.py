from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity() if in_channels == out_channels and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


class PoseCNN(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 192), num_joints: int = 14, dropout: float = 0.2) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 9
        for index, out_channels in enumerate(channels):
            stride = (1, 1) if index == 0 else (1, 2)
            blocks.append(ResidualBlock(in_channels, out_channels, stride))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(channels[-1], channels[-1]), nn.GELU(),
            nn.Linear(channels[-1], num_joints * 2),
        )
        self.num_joints = num_joints

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        if csi.ndim != 4 or tuple(csi.shape[1:]) != (9, 5, 30):
            raise ValueError(f"Expected [B, 9, 5, 30], received {tuple(csi.shape)}")
        return self.head(self.encoder(csi)).reshape(csi.shape[0], self.num_joints, 2)

