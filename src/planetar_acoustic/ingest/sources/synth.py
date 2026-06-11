"""Deterministic synthetic hydrophone source — useful for CI, offline demo,
and proving the pipeline shape without touching the network.

Generates a configurable mix of:
  * a low cavitation-rumble tone (~70 Hz)
  * a mid propeller / shaft tone (~600 Hz)
  * pink-ish broadband background noise

Every N clips, the synth mutes the vessel tones for one clip so downstream
consumers can verify the presence gate actually skips quiet clips.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from planetar_acoustic.bus.topics import HydrophoneSite
from planetar_acoustic.ingest.audio import Clip
from planetar_acoustic.ingest.sources.base import Source, SourceClip

log = logging.getLogger(__name__)


SYNTH_SITE = HydrophoneSite(
    site_code="SYNTH-01",
    name="Synthetic Hydrophone (test)",
    lat=48.6500,
    lon=-123.5000,
    depth_m=100.0,
    network="SYNTH",
    description="Deterministic synthesizer for CI / offline demo",
)


@dataclass
class SyntheticSource(Source):
    sample_rate: int = 32000
    clip_seconds: float = 4.0
    pace_realtime: bool = True
    seed: int = 0
    site: HydrophoneSite = SYNTH_SITE
    max_clips: int | None = None     # None = run forever

    def sites(self) -> list[HydrophoneSite]:
        return [self.site]

    def stream(self) -> Iterator[SourceClip]:
        rng = np.random.default_rng(self.seed)
        n = int(round(self.clip_seconds * self.sample_rate))
        t = np.arange(n, dtype=np.float32) / self.sample_rate
        i = 0
        while self.max_clips is None or i < self.max_clips:
            quiet = (i % 5) == 4
            f_low = 70.0 + rng.normal(0, 1)
            f_mid = 600.0 + rng.normal(0, 5)
            tone_amp = 0.0 if quiet else 0.35
            audio = (
                tone_amp * np.sin(2 * np.pi * f_low * t)
                + tone_amp * 0.8 * np.sin(2 * np.pi * f_mid * t)
                + 0.025 * rng.normal(size=n).astype(np.float32)
            ).astype(np.float32)
            now_ns = time.time_ns()
            clip = Clip(
                samples=audio,
                sample_rate=self.sample_rate,
                source_path=f"synth://{self.site.site_code}",
                start_sample=i * n,
                start_ns=now_ns,
                clip_id=f"{self.site.site_code}:{now_ns}",
            )
            yield SourceClip(clip=clip, site=self.site)
            i += 1
            if self.pace_realtime:
                time.sleep(self.clip_seconds)
