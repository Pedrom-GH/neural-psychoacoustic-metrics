"""Estimate loudness and sharpness of a calibrated recording.

Example:
    python -m psychonet.inference recording.wav --pa-per-unit 1.0
"""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from psychonet.data import Standardizer
from psychonet.features import LogMel
from psychonet.references import loudness_from_level
from psychonet.train import build_model


class MetricEstimator:
    def __init__(self, checkpoint: str | Path) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = state["config"]
        self.fs = self.cfg["audio"]["fs"]
        self.frontend = LogMel(fs=self.fs, **self.cfg["features"]).eval()
        self.model = build_model(self.cfg)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.standardizer = Standardizer.from_state_dict(state["standardizer"])

    @torch.no_grad()
    def predict_features(self, features: torch.Tensor) -> np.ndarray:
        x = self.standardizer.features(features.float()).unsqueeze(1)
        return self.standardizer.inverse_targets(self.model(x).numpy())

    @torch.no_grad()
    def predict(self, pressure: np.ndarray | torch.Tensor) -> dict[str, np.ndarray]:
        """``pressure`` in Pa, shape (samples,) or (batch, samples), sampled at ``self.fs``."""
        x = torch.as_tensor(pressure, dtype=torch.float32)
        if x.ndim == 1:
            x = x[None]
        out = self.predict_features(self.frontend(x))
        return {
            "loudness_level_phon": out[:, 0],
            "loudness_sone": loudness_from_level(out[:, 0]),
            "sharpness_acum": out[:, 1],
        }


def load_calibrated(path: str | Path, pa_per_unit: float, fs_target: int) -> np.ndarray:
    fs, x = wavfile.read(path)
    if np.issubdtype(x.dtype, np.integer):
        x = x / np.iinfo(x.dtype).max
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if fs != fs_target:
        g = gcd(fs, fs_target)
        x = resample_poly(x, fs_target // g, fs // g)
    return x * pa_per_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav")
    parser.add_argument("--checkpoint", default="checkpoints/metricnet.pt")
    parser.add_argument(
        "--pa-per-unit",
        type=float,
        required=True,
        help="pressure in Pa that corresponds to a digital full-scale value of 1.0",
    )
    args = parser.parse_args()

    estimator = MetricEstimator(args.checkpoint)
    x = load_calibrated(args.wav, args.pa_per_unit, estimator.fs)
    seg = estimator.fs
    frames = [x[i : i + seg] for i in range(0, len(x) - seg + 1, seg)] or [
        np.pad(x, (0, seg - len(x)))
    ]
    out = estimator.predict(np.stack(frames))
    print(f"segments: {len(frames)} x 1 s")
    print(f"loudness: {np.mean(out['loudness_sone']):.2f} sone")
    print(f"loudness level: {np.mean(out['loudness_level_phon']):.1f} phon")
    print(f"sharpness: {np.mean(out['sharpness_acum']):.2f} acum")


if __name__ == "__main__":
    main()
