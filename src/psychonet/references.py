"""Reference psychoacoustic metrics computed with MOSQITO."""

from __future__ import annotations

import numpy as np
from mosqito.sq_metrics import loudness_zwst, sharpness_din_from_loudness


def reference_metrics(x: np.ndarray, fs: int) -> tuple[float, float]:
    """Stationary loudness (ISO 532-1, sone) and sharpness (DIN 45692, acum)."""
    loudness, specific_loudness, _ = loudness_zwst(x, fs, field_type="free")
    sharpness = sharpness_din_from_loudness(loudness, specific_loudness, weighting="din")
    return float(loudness), float(sharpness)


def loudness_level(sone: np.ndarray) -> np.ndarray:
    """Loudness level in phon. Uses the ISO 532-1 relation below 1 sone."""
    sone = np.maximum(np.asarray(sone, dtype=float), 1e-6)
    above = 40.0 + 10.0 * np.log2(sone)
    below = 40.0 * (sone + 0.0005) ** 0.35
    return np.where(sone >= 1.0, above, below)


def loudness_from_level(phon: np.ndarray) -> np.ndarray:
    """Inverse of :func:`loudness_level`."""
    phon = np.asarray(phon, dtype=float)
    above = 2.0 ** ((phon - 40.0) / 10.0)
    below = np.maximum(phon / 40.0, 0.0) ** (1.0 / 0.35) - 0.0005
    return np.maximum(np.where(phon >= 40.0, above, below), 0.0)
