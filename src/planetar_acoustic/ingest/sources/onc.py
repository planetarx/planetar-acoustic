"""ONC Oceans 3.0 hydrophone source — NEPTUNE + VENUS cabled-array data.

Uses the public dataProductDelivery API at
https://data.oceannetworks.ca/api/ to:

  1. Request an audio data product (WAV/FLAC) for a hydrophone location and
     a time window.
  2. Poll its run status until READY.
  3. Download the resulting archive (WAV files) into a local cache.
  4. Yield each WAV file's clips through ``ingest.audio.iter_clips``.

Two operating modes:

  * **Archive mode** (`window=("YYYY-MM-DDTHH:MM:SS.000Z", "...")`): pull
    historical data — the proposal's "play back a 2025 dark-event window"
    scenario.

  * **Polled-live mode** (`window=None`, `live_interval_s=N`): every N
    seconds, request the last `clip_seconds` worth of audio. ONC's archives
    are typically posted within a few minutes of capture, so this is
    near-live but not zero-latency. Real-time push is not part of the
    Oceans 3.0 public API.

ONC docs: https://wiki.oceannetworks.ca/display/O2A/Oceans+3.0+API+Home
Register for a free token at https://data.oceannetworks.ca/Profile.

Notes on intent vs. implementation:

  This module is wired but its NETWORK side has not yet been exercised
  against the live ONC API from this environment. It will only succeed at
  runtime when the user has `ONC_TOKEN` set and outbound HTTPS to
  data.oceannetworks.ca is available. The Source interface is correct; if
  ONC changes a parameter shape, the patch is local to this file.
"""
from __future__ import annotations

import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx

from planetar_acoustic.bus.topics import HydrophoneSite, site_by_code
from planetar_acoustic.ingest.audio import iter_clips
from planetar_acoustic.ingest.sources.base import Source, SourceClip

log = logging.getLogger(__name__)

ONC_API = "https://data.oceannetworks.ca/api"
DEFAULT_DATA_PRODUCT = "AD"           # Audio Data — WAV
DEFAULT_EXTENSION = "wav"
DEFAULT_DEVICE_CATEGORY = "HYDROPHONE"


@dataclass
class ONCSource(Source):
    location_code: str                # ONC LocationCode, e.g. "FGPD", "BACAX", "CBYIP"
    token: str | None = None          # ONC API token; falls back to env ONC_TOKEN
    sample_rate: int = 32000
    clip_seconds: float = 4.0
    hop_seconds: float = 2.0
    data_product_code: str = DEFAULT_DATA_PRODUCT
    extension: str = DEFAULT_EXTENSION
    cache_dir: Path = field(default_factory=lambda: Path("data/onc_cache"))
    window: tuple[str, str] | None = None     # ("ISO8601 dateFrom", "ISO8601 dateTo")
    live_interval_s: float = 60.0              # polled-live mode lag/cadence
    request_timeout_s: float = 30.0
    deliver_timeout_s: float = 600.0           # max wait for data product to be READY

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.token is None:
            self.token = os.environ.get("ONC_TOKEN")
        if not self.token:
            raise RuntimeError(
                "ONC token not set. Register at "
                "https://data.oceannetworks.ca/Profile and export ONC_TOKEN."
            )
        site = site_by_code(self.location_code)
        if site is None:
            log.warning("location_code %s not in built-in catalog — using stub site metadata",
                        self.location_code)
            site = HydrophoneSite(
                site_code=self.location_code,
                name=self.location_code,
                lat=0.0, lon=0.0, depth_m=0.0,
                network="ONC",
                description="ONC location (no catalog entry)",
            )
        self._site = site

    def sites(self) -> list[HydrophoneSite]:
        return [self._site]

    def stream(self) -> Iterator[SourceClip]:
        if self.window is not None:
            yield from self._pull_window(*self.window)
            return
        # Polled-live: pull rolling tail forever.
        from datetime import datetime, timedelta, timezone
        while True:
            now = datetime.now(timezone.utc)
            start = now - timedelta(seconds=self.live_interval_s + self.clip_seconds)
            iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            try:
                yield from self._pull_window(iso(start), iso(now))
            except Exception as e:
                log.warning("ONC poll cycle failed (%s); retry in %.0fs",
                            e, self.live_interval_s)
            time.sleep(self.live_interval_s)

    # --- API plumbing ---------------------------------------------------

    def _pull_window(self, date_from: str, date_to: str) -> Iterator[SourceClip]:
        request_id = self._request_data_product(date_from, date_to)
        run_id = self._run_data_product(request_id)
        archives = self._download_when_ready(run_id)
        for path in archives:
            for wav in self._extract_wavs(path):
                for clip in iter_clips(
                    wav,
                    sample_rate=self.sample_rate,
                    clip_s=self.clip_seconds,
                    hop_s=self.hop_seconds,
                ):
                    yield SourceClip(clip=clip, site=self._site)

    def _request_data_product(self, date_from: str, date_to: str) -> int:
        params = {
            "method": "request",
            "token": self.token,
            "locationCode": self.location_code,
            "deviceCategoryCode": DEFAULT_DEVICE_CATEGORY,
            "dataProductCode": self.data_product_code,
            "extension": self.extension,
            "dateFrom": date_from,
            "dateTo": date_to,
            "dpo_hydrophoneDataDiversionMode": "All",
        }
        r = httpx.get(f"{ONC_API}/dataProductDelivery",
                      params=params, timeout=self.request_timeout_s)
        r.raise_for_status()
        data = r.json()
        rid = int(data.get("dpRequestId") or data.get("requestId") or 0)
        if rid == 0:
            raise RuntimeError(f"ONC request failed: {data}")
        log.info("ONC dpRequestId=%d for %s [%s..%s]",
                 rid, self.location_code, date_from, date_to)
        return rid

    def _run_data_product(self, request_id: int) -> int:
        r = httpx.get(f"{ONC_API}/dataProductDelivery",
                      params={"method": "run", "token": self.token,
                              "dpRequestId": request_id},
                      timeout=self.request_timeout_s)
        r.raise_for_status()
        data = r.json()
        runs = data if isinstance(data, list) else data.get("runs", [])
        if not runs:
            raise RuntimeError(f"ONC run() returned no runs: {data}")
        run_id = int(runs[0].get("dpRunId") or runs[0].get("runId"))
        return run_id

    def _download_when_ready(self, run_id: int) -> list[Path]:
        deadline = time.time() + self.deliver_timeout_s
        while time.time() < deadline:
            r = httpx.get(f"{ONC_API}/dataProductDelivery",
                          params={"method": "status", "token": self.token,
                                  "dpRunId": run_id},
                          timeout=self.request_timeout_s)
            r.raise_for_status()
            status = r.json()
            state = (status[0].get("status") if isinstance(status, list) else status.get("status")) or ""
            log.debug("ONC run %d status=%s", run_id, state)
            if state.lower() in ("complete", "completed", "ready"):
                return self._download_run(run_id)
            if state.lower() in ("error", "cancelled", "failed"):
                raise RuntimeError(f"ONC run {run_id} ended in {state}: {status}")
            time.sleep(5.0)
        raise TimeoutError(f"ONC run {run_id} did not finish within "
                           f"{self.deliver_timeout_s}s")

    def _download_run(self, run_id: int) -> list[Path]:
        out_dir = self.cache_dir / f"run-{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        archive_path = out_dir / "data.zip"
        with httpx.stream(
            "GET",
            f"{ONC_API}/dataProductDelivery",
            params={"method": "download", "token": self.token, "dpRunId": run_id},
            timeout=None,
        ) as r:
            r.raise_for_status()
            with archive_path.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 16):
                    fh.write(chunk)
        log.info("ONC run %d downloaded → %s (%d B)", run_id, archive_path,
                 archive_path.stat().st_size)
        return [archive_path]

    def _extract_wavs(self, archive: Path) -> list[Path]:
        if archive.suffix.lower() != ".zip":
            return [archive] if archive.suffix.lower() in (".wav", ".flac") else []
        out = archive.parent / "wav"
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            wavs: list[Path] = []
            for name in zf.namelist():
                if name.lower().endswith((".wav", ".flac")):
                    target = out / Path(name).name
                    if not target.exists():
                        target.write_bytes(zf.read(name))
                    wavs.append(target)
        return sorted(wavs)
