"""Local archive source — replay a folder of hydrophone files.

Two modes:

  * ``pace="realtime"``   — clip-by-clip emit, sleep ``clip_seconds`` between.
                            Use this for a demo: the UI sees the timeline
                            advance at wall-clock pace.
  * ``pace="asfast"``     — emit as fast as the pipeline can process; useful
                            for offline bulk reprocessing / training data
                            generation.

The file list can include WAV / FLAC / OGG (anything libsndfile reads). Files
are processed in sorted order; the daemon loops the folder forever if
``loop=True``, which is the convenient default for a stand-up demo where
the operator wants the screen to keep moving.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from planetar_acoustic.bus.topics import HydrophoneSite
from planetar_acoustic.ingest.audio import iter_clips
from planetar_acoustic.ingest.sources.base import Source, SourceClip
from planetar_acoustic.ingest.sources.synth import SYNTH_SITE

log = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aiff", ".aif")


@dataclass
class ArchiveSource(Source):
    folder: Path
    site: HydrophoneSite = SYNTH_SITE
    sample_rate: int = 32000
    clip_seconds: float = 4.0
    hop_seconds: float = 2.0
    pace: Literal["realtime", "asfast"] = "realtime"
    loop: bool = False
    extensions: tuple[str, ...] = AUDIO_EXTS

    _files: list[Path] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self._refresh()
        if not self._files:
            log.warning("archive folder %s has no audio files matching %s",
                        self.folder, self.extensions)

    def _refresh(self) -> None:
        if not self.folder.exists():
            self._files = []
            return
        self._files = sorted(
            p for p in self.folder.rglob("*")
            if p.is_file() and p.suffix.lower() in self.extensions
        )

    def sites(self) -> list[HydrophoneSite]:
        return [self.site]

    def stream(self) -> Iterator[SourceClip]:
        once = True
        while once or self.loop:
            once = False
            self._refresh()
            for path in self._files:
                log.info("archive: replaying %s", path)
                for clip in iter_clips(
                    path,
                    sample_rate=self.sample_rate,
                    clip_s=self.clip_seconds,
                    hop_s=self.hop_seconds,
                ):
                    yield SourceClip(clip=clip, site=self.site)
                    if self.pace == "realtime":
                        time.sleep(self.hop_seconds)
            if not self.loop:
                return
            # Sleep a beat between loops so a tiny folder doesn't hammer.
            time.sleep(1.0)
