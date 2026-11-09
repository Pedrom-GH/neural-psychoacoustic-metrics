import numpy as np
import pytest
import torch

from psychonet.features import LogMel, mel_filterbank
from psychonet.signals import set_level

FS = 48000


def test_filterbank_partition_of_unity():
    fb = mel_filterbank(4096, FS, 128, 20.0, 16000.0)
    freqs = np.linspace(0, FS / 2, 4096 // 2 + 1)
    inner = (freqs > 60.0) & (freqs < 15000.0)
    assert fb.sum(0).numpy()[inner] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("freq", [125.0, 1000.0, 6300.0])
def test_band_powers_sum_to_tone_level(freq):
    t = np.arange(FS) / FS
    x = set_level(np.sin(2 * np.pi * freq * t), 70.0)
    levels = LogMel(fs=FS)(torch.from_numpy(x).float()[None])[0].double()
    total = 10 * torch.log10((10 ** (levels / 10)).sum(0)).mean()
    assert total.item() == pytest.approx(70.0, abs=0.5)


def test_output_shape():
    frontend = LogMel(fs=FS, n_fft=4096, hop_length=2400, n_mels=128)
    out = frontend(torch.zeros(3, FS))
    assert out.shape == (3, 128, 1 + (FS - 4096) // 2400)
