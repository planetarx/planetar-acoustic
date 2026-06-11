"""Strobed temporal integration → Stabilized Auditory Image.

Algorithm (Lyons, *Human and Machine Hearing*, ch. 17):

  For each cochlear channel:
    1. Maintain a running "strobe threshold" — an exponentially decaying
       envelope that tracks recent peak NAP energy.
    2. A sample is a "strobe" if it is a local maximum AND exceeds the
       threshold; on each strobe, reset the threshold to that sample value.
    3. On every strobe at time t, extract the lag window NAP[t : t + L] and
       add it (with a per-strobe decay) into the channel's row of the SAI
       accumulator. The accumulator is then a stable image of how the
       channel's NAP looks aligned to its own peaks.

Periodic inputs (vessel tonals, propeller blade rate, cavitation lines) produce
sharp vertical structure in the SAI that is invariant to absolute phase —
which is what makes SAIs useful as input to a CV classifier.

Output shape: (n_channels, lag_samples) float32.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from planetar_acoustic.sai.carfac import CARFAC

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAI:
    image: np.ndarray            # float32 (n_channels, lag_samples)
    cf_hz: np.ndarray            # channel center freqs (n_channels,)
    sample_rate: int
    lag_samples: int
    n_strobes: int               # total strobes that contributed (sanity stat)

    @property
    def n_channels(self) -> int:
        return int(self.image.shape[0])


def _strobe_indices(
    channel: np.ndarray,
    decay: float,
    min_lag: int,
) -> np.ndarray:
    """Return strobe sample indices for a single channel.

    A strobe fires when the sample is (a) a local maximum vs. its immediate
    neighbours and (b) above the running threshold. After firing, the
    threshold is reset to the sample value; between strobes it decays
    geometrically by `decay` per sample. `min_lag` enforces a refractory
    period so we don't fire on adjacent samples of the same peak.
    """
    n = channel.shape[0]
    out: list[int] = []
    thresh = 0.0
    last = -min_lag
    for t in range(1, n - 1):
        v = channel[t]
        thresh *= decay
        if v > thresh and v > channel[t - 1] and v >= channel[t + 1] and (t - last) >= min_lag:
            out.append(t)
            thresh = v
            last = t
    return np.asarray(out, dtype=np.int64)


def _accumulate_sai(
    nap: np.ndarray,
    lag_samples: int,
    strobe_decay: float,
    integration_decay: float,
    min_lag: int,
) -> tuple[np.ndarray, int]:
    """Per-channel strobed integration. Returns (sai_image, total_strobes)."""
    n_channels, n_samples = nap.shape
    sai = np.zeros((n_channels, lag_samples), dtype=np.float32)
    total = 0
    for ch in range(n_channels):
        row = nap[ch]
        strobes = _strobe_indices(row, strobe_decay, min_lag)
        if strobes.size == 0:
            continue
        # Vectorised lag-window accumulation with per-strobe geometric decay.
        # Build a (n_strobes, lag_samples) matrix of windows (zero-pad tail).
        windows = np.zeros((strobes.size, lag_samples), dtype=np.float32)
        for k, t in enumerate(strobes):
            end = min(t + lag_samples, n_samples)
            windows[k, : end - t] = row[t:end]
        # Weight by integration_decay^age so later strobes dominate.
        ages = (strobes.size - 1) - np.arange(strobes.size)
        weights = integration_decay ** ages.astype(np.float32)
        sai[ch] = (windows * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)
        total += int(strobes.size)
    return sai, total


def compute_sai(
    audio: np.ndarray,
    sample_rate: int,
    *,
    n_channels: int = 64,
    f_lo_hz: float = 50.0,
    f_hi_hz: float = 8000.0,
    lag_ms: float = 32.0,
    strobe_decay: float = 0.997,
    integration_decay: float = 0.985,
    carfac: CARFAC | None = None,
) -> SAI:
    """One-shot SAI from a finite audio clip.

    For streaming use (continuous hydrophone feed), keep a `CARFAC` instance
    yourself and feed it chunks; the strobe + accumulate loop is cheap to
    re-run each frame because lag_samples is small (~1024 at 32 kHz, 32 ms).
    """
    if carfac is None:
        carfac = CARFAC(
            sample_rate=sample_rate,
            n_channels=n_channels,
            f_lo_hz=f_lo_hz,
            f_hi_hz=f_hi_hz,
        )
    elif carfac.sample_rate != sample_rate:
        raise ValueError(f"carfac sample_rate {carfac.sample_rate} != audio {sample_rate}")
    nap = carfac.step(audio.astype(np.float32, copy=False))
    lag_samples = int(round(lag_ms * 1e-3 * sample_rate))
    # Refractory period: 1 / highest CF (~ shortest expected period), in samples,
    # but at least 1 to keep _strobe_indices monotone.
    min_lag = max(1, int(sample_rate / float(carfac.cf_hz[-1])))
    image, n_strobes = _accumulate_sai(
        nap,
        lag_samples=lag_samples,
        strobe_decay=strobe_decay,
        integration_decay=integration_decay,
        min_lag=min_lag,
    )
    log.debug("SAI: %d channels × %d lag samples, %d strobes", image.shape[0], image.shape[1], n_strobes)
    return SAI(
        image=image,
        cf_hz=carfac.cf_hz.copy(),
        sample_rate=sample_rate,
        lag_samples=lag_samples,
        n_strobes=n_strobes,
    )
