"""Cheap vessel-presence gate that runs *before* the SAI / CNN stack.

The goal is a fast in-band-vs-out-of-band SNR check that filters obvious
silence and out-of-band noise without paying for a full cochlear pass. Tuned
for coastal hydrophones where vessel signatures live in ~100 Hz – 4 kHz
(propeller/cavitation broadband + shaft tonals).

Returns a `PresenceResult` carrying band SNR, peak frequency, and a boolean
`detected` flag the caller can use to skip downstream work.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from planetar_acoustic.config import PresenceParams

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresenceResult:
    detected: bool
    snr_db: float
    band_lo_hz: float
    band_hi_hz: float
    peak_freq_hz: float
    duration_s: float


def detect_presence(
    audio: np.ndarray,
    sample_rate: int,
    params: PresenceParams = PresenceParams(),
) -> PresenceResult:
    """In-band-vs-out-of-band SNR test on Welch PSD.

    SNR_dB = 10·log10( mean(PSD inside band) / mean(PSD outside band) ).
    The out-of-band region brackets the vessel band on both sides so we
    don't get fooled by a single narrowband tone bleeding from a passing
    sonar ping.
    """
    if audio.ndim != 1:
        raise ValueError(f"expected 1-D audio, got shape {audio.shape}")
    dur = audio.shape[0] / float(sample_rate)
    nperseg = min(audio.shape[0], 1 << 12)
    if nperseg < 64:
        # Clip is unusably short — return a hard "no detection" rather than
        # crash, so the caller can keep iterating.
        return PresenceResult(False, -np.inf, params.band_lo_hz, params.band_hi_hz, 0.0, dur)
    freqs, psd = welch(audio, fs=sample_rate, nperseg=nperseg, scaling="density")
    in_band = (freqs >= params.band_lo_hz) & (freqs <= params.band_hi_hz)
    # Out-of-band guard: everything else above 10 Hz (we discard DC/sub-bass).
    out_band = (freqs >= 10.0) & ~in_band
    if not in_band.any() or not out_band.any():
        return PresenceResult(False, -np.inf, params.band_lo_hz, params.band_hi_hz, 0.0, dur)
    p_in = float(psd[in_band].mean())
    p_out = float(psd[out_band].mean())
    snr_db = 10.0 * np.log10(max(p_in, 1e-20) / max(p_out, 1e-20))
    peak_idx = np.argmax(psd * in_band)
    peak_hz = float(freqs[peak_idx])
    detected = bool(
        snr_db >= params.snr_db_min
        and dur >= params.min_duration_s
    )
    log.debug(
        "presence: dur=%.2fs snr=%.1fdB peak=%.0fHz → %s",
        dur, snr_db, peak_hz, "DETECT" if detected else "skip",
    )
    return PresenceResult(
        detected=detected,
        snr_db=snr_db,
        band_lo_hz=params.band_lo_hz,
        band_hi_hz=params.band_hi_hz,
        peak_freq_hz=peak_hz,
        duration_s=dur,
    )
