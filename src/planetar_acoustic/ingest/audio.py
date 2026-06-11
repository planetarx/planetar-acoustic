"""Hydrophone audio ingest: file → mono float32 → fixed-duration clips.

Supports WAV / FLAC / OGG (anything libsndfile reads). Multi-channel input is
mixed down to mono — hydrophone arrays are handled by a separate beamformer
stage (out of scope for v0.1).

A `Clip` is the unit the rest of the pipeline operates on: a contiguous
window of audio at a known sample rate, with provenance metadata that flows
through detect / SAI / classify all the way to the bus.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Clip:
    samples: np.ndarray            # float32, mono, length = round(dur_s * sample_rate)
    sample_rate: int
    source_path: str
    start_sample: int              # offset within the source file (post-resample)
    start_ns: int                  # wall-clock anchor; file mtime if unknown
    clip_id: str                   # f"{stem}:{start_sample}" — stable across runs

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / float(self.sample_rate)


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Read a file, mix to mono, resample to `target_sr`.

    Uses polyphase resampling with a rational factor: gcd-reduces target/src
    so we never lose precision to float-rate conversion.
    """
    data, src_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if src_sr != target_sr:
        from math import gcd
        g = gcd(target_sr, src_sr)
        up, down = target_sr // g, src_sr // g
        mono = resample_poly(mono, up, down).astype(np.float32, copy=False)
        log.debug("resampled %s: %d → %d Hz (poly %d/%d)", path.name, src_sr, target_sr, up, down)
    return mono, target_sr


def iter_clips(
    path: Path,
    sample_rate: int,
    clip_s: float,
    hop_s: float,
    *,
    start_ns: int | None = None,
) -> Iterator[Clip]:
    """Yield overlapping clips of `clip_s` seconds, stepping by `hop_s`.

    The tail is yielded only if it's at least one hop long (i.e. > half a clip
    of fresh audio). Anchored to file mtime if no `start_ns` is supplied so
    bus envelopes have a meaningful timestamp without external clock infra.
    """
    if hop_s <= 0 or clip_s <= 0:
        raise ValueError("clip_s and hop_s must be > 0")
    if start_ns is None:
        start_ns = int(path.stat().st_mtime * 1e9)
    audio, sr = load_audio(path, sample_rate)
    n = audio.shape[0]
    clip_n = int(round(clip_s * sr))
    hop_n = int(round(hop_s * sr))
    if n < clip_n:
        log.warning("file %s shorter (%.2fs) than clip_s=%.2fs — yielding zero-padded clip",
                    path.name, n / sr, clip_s)
        pad = np.zeros(clip_n - n, dtype=np.float32)
        yield Clip(
            samples=np.concatenate([audio, pad]),
            sample_rate=sr,
            source_path=str(path),
            start_sample=0,
            start_ns=start_ns,
            clip_id=f"{path.stem}:0",
        )
        return
    stem = path.stem
    i = 0
    last_emitted = -hop_n
    while i + clip_n <= n:
        yield Clip(
            samples=audio[i : i + clip_n].copy(),
            sample_rate=sr,
            source_path=str(path),
            start_sample=i,
            start_ns=start_ns + int(i * 1e9 / sr),
            clip_id=f"{stem}:{i}",
        )
        last_emitted = i
        i += hop_n
    # Trailing partial: emit if >= one hop of unused audio remains.
    if n - last_emitted - clip_n >= hop_n:
        tail = audio[n - clip_n : n]
        yield Clip(
            samples=tail.copy(),
            sample_rate=sr,
            source_path=str(path),
            start_sample=n - clip_n,
            start_ns=start_ns + int((n - clip_n) * 1e9 / sr),
            clip_id=f"{stem}:{n - clip_n}",
        )
