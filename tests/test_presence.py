"""Tests for the presence gate."""
from __future__ import annotations

import numpy as np

from planetar_acoustic.config import PresenceParams
from planetar_acoustic.detect.presence import detect_presence


SR = 16000


def _tone(freq_hz: float, dur_s: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_silence_is_not_detected():
    audio = np.zeros(SR, dtype=np.float32) + 1e-6 * np.random.default_rng(0).normal(size=SR).astype(np.float32)
    r = detect_presence(audio, SR)
    assert not r.detected


def test_in_band_tone_is_detected():
    audio = _tone(800.0, dur_s=1.0)
    r = detect_presence(audio, SR)
    assert r.detected, f"expected detection at 800Hz; got snr={r.snr_db:.1f}dB"
    assert r.snr_db > 6.0
    assert 700.0 <= r.peak_freq_hz <= 900.0


def test_out_of_band_tone_is_rejected():
    # 9 kHz tone — outside the default 100–4000 Hz vessel band.
    audio = _tone(9000.0, dur_s=1.0)
    r = detect_presence(audio, SR)
    assert not r.detected, f"out-of-band tone leaked: snr={r.snr_db:.1f}dB"


def test_short_clip_rejected_even_if_loud():
    audio = _tone(800.0, dur_s=0.2)
    r = detect_presence(audio, SR, PresenceParams(min_duration_s=0.5))
    assert not r.detected
