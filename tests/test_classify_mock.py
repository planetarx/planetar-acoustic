"""Classifier wiring tests using the deterministic mock backend (no torch needed)."""
from __future__ import annotations

import numpy as np

from planetar_acoustic.classify.sai_cnn import DEFAULT_CLASSES, SAIClassifier, sai_to_image
from planetar_acoustic.sai.sai import compute_sai


SR = 16000


def _tone(freq_hz: float, dur_s: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * dur_s), dtype=np.float32) / SR
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_mock_is_deterministic_same_input_same_output():
    sai = compute_sai(_tone(440.0), sample_rate=SR, n_channels=16, lag_ms=16.0)
    clf = SAIClassifier(model=None, classes=DEFAULT_CLASSES)
    a = clf.predict(sai)
    b = clf.predict(sai)
    assert a.class_ == b.class_
    assert a.score == b.score
    assert a.model_id == "mock"


def test_sai_to_image_shape_and_range():
    sai = compute_sai(_tone(600.0), sample_rate=SR, n_channels=24, lag_ms=16.0)
    img = sai_to_image(sai)
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0001


def test_top_k_sums_to_at_most_one():
    sai = compute_sai(_tone(220.0), sample_rate=SR, n_channels=16, lag_ms=16.0)
    clf = SAIClassifier(model=None)
    h = clf.predict(sai)
    assert 0.0 <= h.score <= 1.0
    assert sum(p for _, p in h.top_k) <= 1.0001
    assert h.class_ in DEFAULT_CLASSES
