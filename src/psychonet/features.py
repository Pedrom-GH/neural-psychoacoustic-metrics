"""Calibrated log-mel front end in PyTorch."""

from __future__ import annotations

import math

import torch
from torch import nn

P_REF = 20e-6


def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, fs: int, n_mels: int, f_min: float, f_max: float) -> torch.Tensor:
    """Triangular filters with unit peak. Adjacent filters sum to one between the
    first and last centre frequency, so band powers add up to the total power."""
    freqs = torch.linspace(0.0, fs / 2, n_fft // 2 + 1, dtype=torch.float64)
    edges = mel_to_hz(
        torch.linspace(
            hz_to_mel(torch.tensor(f_min)),
            hz_to_mel(torch.tensor(f_max)),
            n_mels + 2,
            dtype=torch.float64,
        )
    )
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (freqs - lower) / (center - lower)
    falling = (upper - freqs) / (upper - center)
    return torch.clamp(torch.minimum(rising, falling), min=0.0).float()


class LogMel(nn.Module):
    """Maps pressure signals in Pa, shape (batch, samples), to band levels in
    dB SPL, shape (batch, n_mels, frames)."""

    def __init__(
        self,
        fs: int = 48000,
        n_fft: int = 4096,
        hop_length: int = 2400,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 16000.0,
        floor_db: float = -20.0,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.floor_db = floor_db
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window)
        self.register_buffer("filters", mel_filterbank(n_fft, fs, n_mels, f_min, f_max))
        self.scale = 2.0 / (n_fft * float(window.pow(2).sum()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs().pow(2) * self.scale
        bands = torch.matmul(self.filters, power)
        floor = (P_REF**2) * 10.0 ** (self.floor_db / 10.0)
        return 10.0 * torch.log10(bands + floor) - 20.0 * math.log10(P_REF)
