"""OrcaSound HLS hydrophone source.

OrcaSound (https://www.orcasound.net) runs a network of open hydrophones in
the Salish Sea — Bush Point, Orcasound Lab, Port Townsend, North San Juan,
Sunset Bay — that stream audio as HTTP Live Streaming (HLS). No API key.

Live feed manifest: https://live.orcasound.net/api/json/feeds
HLS playlist:       https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/
                    {node}/hls/live/playlist.m3u8

This source uses ffmpeg as a subprocess to:

  1. Open the HLS master playlist.
  2. Re-encode incoming PCM into clip-length WAV chunks in a cache dir.
  3. Yield each finished WAV as a SourceClip.

We pick the subprocess path over a Python HLS library because ffmpeg is
the reference, handles segment timing / re-fetching cleanly, and we already
need it in the planetar-ui demo box for other reasons.

Implementation note: the network plumbing here has not been exercised
from this environment — it's wired correctly but requires ffmpeg on PATH
and outbound HTTPS to S3. Falls back to a clear runtime error otherwise.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from planetar_acoustic.bus.topics import ORCASOUND_SITES, HydrophoneSite, site_by_code
from planetar_acoustic.ingest.audio import iter_clips
from planetar_acoustic.ingest.sources.base import Source, SourceClip

log = logging.getLogger(__name__)


ORCASOUND_HLS = (
    "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/"
    "{node}/hls/live/playlist.m3u8"
)


@dataclass
class OrcaSoundSource(Source):
    node: str = "rpi_bush_point"      # one of the OrcaSound node IDs
    sample_rate: int = 32000
    clip_seconds: float = 4.0
    hop_seconds: float = 2.0
    cache_dir: Path = field(default_factory=lambda: Path("data/orcasound_cache"))
    segment_max: int = 50              # cap on cached segments (rotating buffer)

    _site: HydrophoneSite = field(init=False)
    _ff_proc: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        s = site_by_code(self.node)
        if s is None:
            # Fall back to first OrcaSound site so the source still has
            # plottable lat/lon even for an unknown node id.
            log.warning("orcasound node %s not in catalog — using stub metadata", self.node)
            base = ORCASOUND_SITES[0]
            s = HydrophoneSite(
                site_code=self.node, name=self.node, lat=base.lat, lon=base.lon,
                depth_m=base.depth_m, network="OrcaSound",
                description="OrcaSound node (no catalog entry)",
            )
        self._site = s
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. OrcaSoundSource needs ffmpeg to pull HLS. "
                "Install via your OS package manager (apt install ffmpeg)."
            )

    def sites(self) -> list[HydrophoneSite]:
        return [self._site]

    def stream(self) -> Iterator[SourceClip]:
        url = ORCASOUND_HLS.format(node=self.node)
        clip_dir = self.cache_dir / self.node
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Tell ffmpeg to segment the live decode into clip-length WAV files.
        # `-f segment` + `-segment_time` writes a fresh file every N seconds;
        # we then watch the directory and yield finished files.
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-y",
            "-i", url,
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "segment",
            "-segment_time", str(self.clip_seconds),
            "-strftime", "1",
            str(clip_dir / "%Y%m%dT%H%M%SZ.wav"),
        ]
        log.info("orcasound: launching ffmpeg HLS pull from %s", url)
        self._ff_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env={**os.environ, "TZ": "UTC"},
        )
        seen: set[Path] = set()
        try:
            while True:
                if self._ff_proc.poll() is not None:
                    err = self._ff_proc.stderr.read().decode("utf-8", "replace") if self._ff_proc.stderr else ""
                    raise RuntimeError(f"ffmpeg exited (rc={self._ff_proc.returncode}): {err[-500:]}")
                # Files appear when each segment is closed; sort by mtime
                # so we yield in temporal order.
                fresh = sorted(
                    (p for p in clip_dir.glob("*.wav") if p not in seen),
                    key=lambda p: p.stat().st_mtime,
                )
                if not fresh:
                    time.sleep(0.5)
                    continue
                # Skip the last (still-being-written) file.
                ready = fresh[:-1] if len(fresh) > 1 else []
                for path in ready:
                    seen.add(path)
                    try:
                        for clip in iter_clips(
                            path,
                            sample_rate=self.sample_rate,
                            clip_s=self.clip_seconds,
                            hop_s=self.hop_seconds,
                        ):
                            yield SourceClip(clip=clip, site=self._site)
                    except Exception as e:
                        log.warning("failed to read segment %s: %s", path, e)
                # Rotate older segments to keep cache bounded.
                if len(seen) > self.segment_max:
                    old = sorted(seen, key=lambda p: p.stat().st_mtime)[:len(seen) - self.segment_max]
                    for p in old:
                        seen.discard(p)
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass
                if not ready:
                    time.sleep(0.5)
        finally:
            if self._ff_proc and self._ff_proc.poll() is None:
                self._ff_proc.terminate()
                try:
                    self._ff_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._ff_proc.kill()
