"""Synthetic stationary test signals calibrated in pascal."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

P_REF = 20e-6
F_MIN = 50.0
F_MAX = 12000.0

Generator = Callable[[np.random.Generator, int, int], np.ndarray]


def spl_to_rms(level_db: float) -> float:
    return P_REF * 10.0 ** (level_db / 20.0)


def rms_to_spl(x: np.ndarray) -> float:
    return 20.0 * np.log10(np.sqrt(np.mean(x**2)) / P_REF)


def set_level(x: np.ndarray, level_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2))
    return x * (spl_to_rms(level_db) / rms)


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _shaped_noise(rng: np.random.Generator, n: int, fs: int, gain: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    return np.fft.irfft(spectrum * gain, n)


def _freqs(n: int, fs: int) -> np.ndarray:
    return np.fft.rfftfreq(n, 1.0 / fs)


def pure_tone(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    f = _log_uniform(rng, F_MIN, F_MAX)
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))


def harmonic_complex(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    f0 = _log_uniform(rng, 80.0, 1000.0)
    slope_db_oct = rng.uniform(-12.0, 0.0)
    n_harmonics = int(min(rng.integers(2, 25), F_MAX // f0))
    t = np.arange(n) / fs
    x = np.zeros(n)
    for k in range(1, max(n_harmonics, 1) + 1):
        amplitude = 10.0 ** (slope_db_oct * np.log2(k) / 20.0)
        x += amplitude * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    return x


def white_noise(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    f = _freqs(n, fs)
    gain = ((f >= 20.0) & (f <= F_MAX)).astype(float)
    return _shaped_noise(rng, n, fs, gain)


def colored_noise(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    f = _freqs(n, fs)
    slope_db_oct = rng.uniform(-6.0, 3.0)
    gain = np.zeros_like(f)
    band = (f >= 20.0) & (f <= F_MAX)
    gain[band] = 10.0 ** (slope_db_oct * np.log2(f[band] / 1000.0) / 20.0)
    return _shaped_noise(rng, n, fs, gain)


def bandpass_noise(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    fc = _log_uniform(rng, 100.0, 10000.0)
    width_oct = rng.uniform(1.0 / 12.0, 2.0)
    lo = max(fc * 2.0 ** (-width_oct / 2), 20.0)
    hi = min(fc * 2.0 ** (width_oct / 2), F_MAX)
    f = _freqs(n, fs)
    gain = ((f >= lo) & (f <= hi)).astype(float)
    if gain.sum() < 2:
        gain[np.argmin(np.abs(f - fc))] = 1.0
    return _shaped_noise(rng, n, fs, gain)


def tone_in_noise(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    tone = set_level(pure_tone(rng, fs, n), 0.0)
    noise = set_level(colored_noise(rng, fs, n), 0.0)
    snr_db = rng.uniform(-10.0, 20.0)
    return tone * 10.0 ** (snr_db / 20.0) + noise


def amplitude_modulated_tone(rng: np.random.Generator, fs: int, n: int) -> np.ndarray:
    fc = _log_uniform(rng, 250.0, 8000.0)
    fm = _log_uniform(rng, 30.0, 150.0)
    depth = rng.uniform(0.2, 1.0)
    t = np.arange(n) / fs
    envelope = 1.0 + depth * np.sin(2 * np.pi * fm * t)
    return envelope * np.sin(2 * np.pi * fc * t)


FAMILIES: dict[str, Generator] = {
    "pure_tone": pure_tone,
    "harmonic_complex": harmonic_complex,
    "white_noise": white_noise,
    "colored_noise": colored_noise,
    "bandpass_noise": bandpass_noise,
    "tone_in_noise": tone_in_noise,
    "am_tone": amplitude_modulated_tone,
}


def fade(x: np.ndarray, fs: int, duration_s: float = 0.01) -> np.ndarray:
    n_fade = int(duration_s * fs)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_fade))
    y = x.copy()
    y[:n_fade] *= ramp
    y[-n_fade:] *= ramp[::-1]
    return y


def generate(
    family: str, level_db: float, rng: np.random.Generator, fs: int, duration_s: float
) -> np.ndarray:
    """Return a faded signal of the given family scaled to ``level_db`` dB SPL."""
    n = int(round(fs * duration_s))
    x = FAMILIES[family](rng, fs, n)
    return fade(set_level(x, level_db), fs)
