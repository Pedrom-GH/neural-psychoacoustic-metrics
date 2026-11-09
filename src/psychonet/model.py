from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
        )


class MetricNet(nn.Module):
    """CNN over (mel band, frame) maps.

    The time axis is averaged after the convolutional stack, since the target
    metrics are stationary. The frequency axis is kept and flattened: where
    energy sits on the critical-band scale matters for both loudness and
    sharpness.
    """

    def __init__(
        self,
        n_mels: int = 128,
        channels: tuple[int, ...] = (24, 48, 96),
        hidden: int = 256,
        dropout: float = 0.2,
        n_outputs: int = 2,
    ) -> None:
        super().__init__()
        blocks, c_in = [], 1
        for c_out in channels:
            blocks.append(ConvBlock(c_in, c_out))
            c_in = c_out
        self.encoder = nn.Sequential(*blocks)
        n_freq = n_mels // 2 ** len(channels)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * n_freq, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x).mean(dim=-1)
        return self.head(z)
