from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from psychonet.references import loudness_level

TARGETS = ("loudness_level_phon", "sharpness_acum")

# Sharpness is a loudness-normalised quantity and becomes numerically unstable
# close to the hearing threshold, so near-inaudible examples are excluded.
MIN_LOUDNESS_SONE = 0.1


@dataclass
class Split:
    features: np.ndarray
    loudness: np.ndarray
    sharpness: np.ndarray
    family: np.ndarray
    level: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> Split:
        with np.load(path) as d:
            split = cls(
                **{k: d[k] for k in ("features", "loudness", "sharpness", "family", "level")}
            )
        keep = (
            np.isfinite(split.loudness)
            & np.isfinite(split.sharpness)
            & (split.loudness >= MIN_LOUDNESS_SONE)
        )
        return split.subset(keep)

    def subset(self, mask: np.ndarray) -> Split:
        return Split(
            self.features[mask],
            self.loudness[mask],
            self.sharpness[mask],
            self.family[mask],
            self.level[mask],
        )

    def targets(self) -> np.ndarray:
        return np.stack([loudness_level(self.loudness), self.sharpness], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.loudness)


@dataclass
class Standardizer:
    feature_mean: float
    feature_std: float
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def fit(cls, split: Split) -> Standardizer:
        feats = split.features.astype(np.float32)
        y = split.targets()
        return cls(float(feats.mean()), float(feats.std()), y.mean(0), y.std(0))

    def features(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def targets(self, y: np.ndarray) -> np.ndarray:
        return (y - self.target_mean) / self.target_std

    def inverse_targets(self, y: np.ndarray) -> np.ndarray:
        return y * self.target_std + self.target_mean

    def state_dict(self) -> dict:
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, s: dict) -> Standardizer:
        return cls(
            s["feature_mean"],
            s["feature_std"],
            np.asarray(s["target_mean"], dtype=np.float32),
            np.asarray(s["target_std"], dtype=np.float32),
        )


class MetricDataset(Dataset):
    def __init__(self, split: Split, standardizer: Standardizer) -> None:
        feats = standardizer.features(split.features.astype(np.float32))
        self.x = torch.from_numpy(feats).unsqueeze(1)
        self.y = torch.from_numpy(standardizer.targets(split.targets()).astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]
