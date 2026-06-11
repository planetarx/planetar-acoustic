"""Pragmatic CAR-FAC cochlear model.

Lyons' CAR-FAC ("Cascade of Asymmetric Resonators with Fast-Acting Compression",
*Human and Machine Hearing*, ch. 14–16) is a real-valued time-domain cochlea
model. This module implements a faithful-but-minimal version sufficient to
feed Stabilized Auditory Images into a CV classifier:

  * ERB-spaced channel centers from f_lo to f_hi
  * Per-channel 4th-order gammatone bandpass (scipy IIR design)
  * Half-wave rectification with mild compression (NAP — neural activity pattern)
  * Per-channel single-time-constant AGC (loudness normalisation)

What this is *not*:

  * Not the cascade-of-asymmetric-resonators topology Lyons describes — each
    channel is independent here (parallel filter bank). That is a known
    simplification; the cascade gives you sharper high-CF tuning and the
    characteristic asymmetric impulse response, which matters more for vowel
    discrimination than for vessel acoustics. The SAI output remains a 2-D
    image-like representation either way.
  * No multi-time-constant AGC, no outer-hair-cell nonlinearity, no efferent
    feedback loop. Add as needed; the API in `step()` is designed to allow it.

Reference implementation of full CAR-FAC: https://github.com/google/carfac
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import gammatone, sosfilt, sosfilt_zi, tf2sos

log = logging.getLogger(__name__)


def erb_space(f_lo_hz: float, f_hi_hz: float, n: int) -> np.ndarray:
    """ERB-rate-uniform channel centers from f_lo to f_hi (ascending Hz).

    Glasberg & Moore (1990): ERB(f) = 24.7 * (4.37 * f/1000 + 1).
    Spacing is uniform in ERB-rate so the cochlea places lower CFs further
    apart in Hz than higher CFs — same convention as Lyons / Slaney.
    """
    erb_lo = 21.4 * np.log10(4.37 * f_lo_hz / 1000.0 + 1.0)
    erb_hi = 21.4 * np.log10(4.37 * f_hi_hz / 1000.0 + 1.0)
    erbs = np.linspace(erb_lo, erb_hi, n)
    return (10.0 ** (erbs / 21.4) - 1.0) * 1000.0 / 4.37


@dataclass
class CARFAC:
    """Stateful cochlea: feed it audio chunks, get NAP frames out.

    Attributes set at construct time:
        sample_rate, n_channels, f_lo_hz, f_hi_hz, agc_tau_s, compress.

    State carried between `step()` calls:
        _sos   — per-channel SOS biquad cascade (immutable post-build)
        _zi    — per-channel filter state (sosfilt zi)
        _agc   — per-channel running RMS estimate for the AGC loop

    The IIR gammatone is converted to second-order-section form for numerical
    stability — the direct-form 4th-order coefficients overflow at high CFs.
    """

    sample_rate: int
    n_channels: int = 64
    f_lo_hz: float = 50.0
    f_hi_hz: float = 8000.0
    agc_tau_s: float = 0.040     # 40 ms loudness time constant
    compress: float = 0.3         # NAP compression exponent (0 = log-like, 1 = linear)

    def __post_init__(self) -> None:
        # Keep f_hi strictly below Nyquist — scipy's gammatone rejects exact
        # equality, and floating-point round-off in erb_space can push the
        # endpoint a couple of ULPs over fs/2.
        nyquist = self.sample_rate / 2.0
        f_hi = min(self.f_hi_hz, nyquist * 0.95)
        self.cf_hz = erb_space(self.f_lo_hz, f_hi, self.n_channels).astype(np.float64)
        self._sos: list[np.ndarray] = []
        self._zi: list[np.ndarray] = []
        for cf in self.cf_hz:
            b, a = gammatone(float(cf), ftype="iir", fs=self.sample_rate)
            sos = tf2sos(b.astype(np.float64), a.astype(np.float64))
            self._sos.append(sos)
            self._zi.append(sosfilt_zi(sos) * 0.0)
        # AGC pole — exp(-1/(tau*fs)) per sample.
        self._agc_alpha = float(np.exp(-1.0 / (self.agc_tau_s * self.sample_rate)))
        self._agc = np.full(self.n_channels, 1e-6, dtype=np.float64)

    def reset(self) -> None:
        for i in range(self.n_channels):
            self._zi[i][:] = 0.0
        self._agc[:] = 1e-6

    def step(self, audio: np.ndarray) -> np.ndarray:
        """Process one chunk; returns NAP of shape (n_channels, len(audio)) float32."""
        if audio.ndim != 1:
            raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
        x = audio.astype(np.float64, copy=False)
        out = np.empty((self.n_channels, x.shape[0]), dtype=np.float32)
        for i, sos in enumerate(self._sos):
            y, self._zi[i] = sosfilt(sos, x, zi=self._zi[i])
            # Half-wave rectify, then compressive AGC. Update running RMS in
            # tandem with the rectified signal so the gain tracks recent
            # energy rather than just the current sample.
            rect = np.maximum(y, 0.0)
            # Exponentially-smoothed RMS estimate (energy domain).
            energy = rect * rect
            agc_state = self._agc[i]
            agc_trace = np.empty_like(energy)
            alpha = self._agc_alpha
            for t in range(energy.shape[0]):
                agc_state = alpha * agc_state + (1.0 - alpha) * energy[t]
                agc_trace[t] = agc_state
            self._agc[i] = agc_state
            gain = 1.0 / np.sqrt(agc_trace + 1e-12)
            nap = (rect * gain) ** self.compress
            out[i] = nap.astype(np.float32, copy=False)
        return out
