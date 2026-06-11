from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path(os.environ.get("PLANETAR_ACOUSTIC_CACHE", "data/clips")).resolve()
DEFAULT_SAI_DIR = Path(os.environ.get("PLANETAR_ACOUSTIC_SAI", "data/sai")).resolve()
DEFAULT_BROKER = os.environ.get("PLANETAR_BROKER", "127.0.0.1:12001")

DEFAULT_SAMPLE_RATE = int(os.environ.get("PLANETAR_ACOUSTIC_SR", "32000"))
DEFAULT_CLIP_SECONDS = float(os.environ.get("PLANETAR_ACOUSTIC_CLIP_S", "4.0"))
DEFAULT_HOP_SECONDS = float(os.environ.get("PLANETAR_ACOUSTIC_HOP_S", "2.0"))


@dataclass(frozen=True)
class SAIParams:
    """Knobs for the Lyons-style SAI pipeline.

    Channels follow ERB-scale spacing from `f_lo` to `f_hi`. `lag_ms` controls
    SAI width (longer lag captures lower fundamentals down to 1000/lag_ms Hz).
    """
    n_channels: int = 64
    f_lo_hz: float = 50.0
    f_hi_hz: float = 8000.0
    lag_ms: float = 32.0          # SAI width = 32 ms (down to ~31 Hz periodicity)
    strobe_decay: float = 0.997   # per-sample decay of the running peak threshold
    integration_decay: float = 0.985  # per-strobe decay of the SAI image accumulator


@dataclass(frozen=True)
class PresenceParams:
    """Vessel-presence gate. Tuned for harbour/coastal hydrophones."""
    band_lo_hz: float = 100.0
    band_hi_hz: float = 4000.0
    snr_db_min: float = 6.0        # band SNR vs. out-of-band noise floor
    min_duration_s: float = 0.5
