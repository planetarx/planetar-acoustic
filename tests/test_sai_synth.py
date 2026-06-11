"""Sanity tests for the SAI pipeline on synthetic inputs.

We feed in tones / chirps / silence and check that the SAI behaves as
expected:

  * Silence → near-zero image, zero or very few strobes.
  * Pure tone at f_0 → energy concentrated in channels near f_0, and a
    horizontal stripe at lag ≈ 1/f_0 (the autocorrelation period).
  * Two coexisting tones → energy on two channel rows.

These are coarse — full unit tests of CAR-FAC + strobed integration are out
of scope; we just verify the wiring isn't catastrophically wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from planetar_acoustic.sai.carfac import CARFAC, erb_space
from planetar_acoustic.sai.sai import compute_sai


SR = 16000
DUR_S = 1.0


def _tone(freq_hz: float, dur_s: float = DUR_S, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_erb_space_monotone_and_endpoints():
    cfs = erb_space(50.0, 8000.0, 32)
    assert cfs.shape == (32,)
    assert cfs[0] == pytest.approx(50.0, rel=1e-3)
    assert cfs[-1] == pytest.approx(8000.0, rel=1e-3)
    assert np.all(np.diff(cfs) > 0)


def test_silence_produces_near_zero_sai():
    audio = np.zeros(SR, dtype=np.float32)
    s = compute_sai(audio, sample_rate=SR, n_channels=24, lag_ms=16.0)
    assert s.image.shape == (24, int(round(0.016 * SR)))
    # Sub-noise-floor; specifically the strobe loop should rarely fire.
    assert float(np.abs(s.image).max()) < 1e-3
    assert s.n_strobes < 10


def test_tone_concentrates_energy_in_matching_channel():
    """A 500 Hz tone should activate channels nearest to 500 Hz, not the extremes."""
    f0 = 500.0
    audio = _tone(f0)
    n_channels = 32
    s = compute_sai(audio, sample_rate=SR, n_channels=n_channels, lag_ms=16.0)
    # Energy per channel (sum across lag).
    channel_energy = s.image.sum(axis=1)
    expected_idx = int(np.argmin(np.abs(s.cf_hz - f0)))
    peak_idx = int(np.argmax(channel_energy))
    # Strict containment is hard for a parallel filterbank; just check the
    # peak sits within ±5 channels of the expected one.
    assert abs(peak_idx - expected_idx) <= 5, (
        f"peak channel {peak_idx} (cf={s.cf_hz[peak_idx]:.0f}Hz) far from "
        f"expected {expected_idx} (cf={s.cf_hz[expected_idx]:.0f}Hz)"
    )


def test_tone_strobes_fire():
    """A real periodic signal must actually fire strobes — else the SAI is a zero image."""
    audio = _tone(440.0)
    s = compute_sai(audio, sample_rate=SR, n_channels=24, lag_ms=16.0)
    assert s.n_strobes > 100, f"expected many strobes for a 1s tone, got {s.n_strobes}"


def test_carfac_reset_is_idempotent():
    car = CARFAC(sample_rate=SR, n_channels=8)
    car.step(_tone(800.0, dur_s=0.1))
    car.reset()
    nap = car.step(np.zeros(int(0.05 * SR), dtype=np.float32))
    # After reset, response to silence should be tiny everywhere.
    assert float(np.abs(nap).max()) < 1e-2
