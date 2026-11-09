import numpy as np
import pytest

from psychonet.references import loudness_from_level, loudness_level, reference_metrics
from psychonet.signals import FAMILIES, P_REF, generate, rms_to_spl, set_level

FS = 48000


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_generated_signal_is_finite_and_calibrated(family):
    rng = np.random.default_rng(1)
    x = generate(family, 70.0, rng, FS, 1.0)
    assert x.shape == (FS,)
    assert np.all(np.isfinite(x))
    assert rms_to_spl(x) == pytest.approx(70.0, abs=0.2)


def test_set_level_matches_definition():
    x = set_level(np.random.default_rng(0).standard_normal(FS), 94.0)
    assert np.sqrt(np.mean(x**2)) == pytest.approx(1.0, rel=3e-3)
    assert 20 * np.log10(np.sqrt(np.mean(x**2)) / P_REF) == pytest.approx(94.0, abs=1e-6)


def test_reference_loudness_of_1khz_tone_at_40_db_is_one_sone():
    t = np.arange(FS) / FS
    x = set_level(np.sin(2 * np.pi * 1000 * t), 40.0)
    loudness, _ = reference_metrics(x, FS)
    assert loudness == pytest.approx(1.0, abs=0.05)


def test_loudness_doubles_per_10_db():
    t = np.arange(FS) / FS
    tone = np.sin(2 * np.pi * 1000 * t)
    n60, _ = reference_metrics(set_level(tone, 60.0), FS)
    n70, _ = reference_metrics(set_level(tone, 70.0), FS)
    assert n70 / n60 == pytest.approx(2.0, rel=0.05)


def test_loudness_level_round_trip():
    sone = np.array([0.05, 0.5, 1.0, 4.0, 32.0])
    assert loudness_from_level(loudness_level(sone)) == pytest.approx(sone, rel=1e-3)
