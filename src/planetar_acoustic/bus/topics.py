"""Canonical bus topics + payload helpers.

The acoustic service emits to three families of topics:

  * ``acoustic.*``         — structured per-event envelopes (machine consumers)
  * ``chat.pac.hydrophone-alerts`` — human-readable summaries that planetar-ui
    already renders, using the ``chat.v1.Message`` schema the bridge synth
    publishes. Any envelope here shows up in the demo UI immediately.

Use the helpers below so every emitter ships the same shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from planetar_acoustic.bus.zmesg import Envelope


# --- topics ------------------------------------------------------------------

T_SITE = "acoustic.site"            # one-shot hydrophone location metadata
T_DETECT = "acoustic.detect"        # per-clip presence event
T_PSD = "acoustic.psd"              # per-clip compact PSD (for UI waterfall)
T_CLASSIFY = "acoustic.classify"    # per-clip CV classifier hypothesis
T_SAI = "acoustic.sai"              # per-clip SAI image (debug; opt-in)
T_CHAT_HYDRO = "chat.pac.hydrophone-alerts"  # UI-visible chat channel


# --- site catalog ------------------------------------------------------------

@dataclass(frozen=True)
class HydrophoneSite:
    """Deployment metadata for a hydrophone.

    site_code follows ONC's location-code convention where possible so it
    matches their archives directly. Lat/lon are advisory for UI plotting;
    depth_m is informational.
    """
    site_code: str
    name: str
    lat: float
    lon: float
    depth_m: float
    network: str          # "ONC", "OrcaSound", "NOAA", "SYNTH", "ARCHIVE"
    description: str = ""


# Salish Sea / BC coast — the deployment region cited in CH13 proposal.
# Coordinates from ONC Oceans 3.0 location registry (verify before quoting in
# proposal text; these are accurate to ~0.001° but ONC publishes the exact
# deployment offsets per cabled-array node revision).
ONC_SITES: tuple[HydrophoneSite, ...] = (
    HydrophoneSite("BACAX",   "Barkley Canyon Axis",       48.3160, -126.0500, 985.0,  "ONC", "NEPTUNE axis hydrophone"),
    HydrophoneSite("BACUS",   "Barkley Canyon Upper Slope", 48.4270, -126.1740, 396.0,  "ONC", "NEPTUNE upper slope"),
    HydrophoneSite("FGPD",    "Folger Deep",                48.8140, -125.2810, 100.0,  "ONC", "NEPTUNE Folger Passage"),
    HydrophoneSite("FGPPN",   "Folger Pinnacle",            48.8230, -125.2810,  23.0,  "ONC", "NEPTUNE Folger pinnacle"),
    HydrophoneSite("CBYIP",   "Saanich Inlet Mooring",      48.6500, -123.5000, 100.0,  "ONC", "VENUS Saanich Inlet"),
    HydrophoneSite("CBYDS",   "Cambridge Bay Underwater",   69.1140, -105.0620,  10.0,  "ONC", "VENUS Cambridge Bay"),
    HydrophoneSite("SEVIP",   "Strait of Georgia East",     49.0400, -123.3170, 170.0,  "ONC", "VENUS East node"),
    HydrophoneSite("USDDL",   "Strait of Georgia Central",  49.0420, -123.4250, 297.0,  "ONC", "VENUS Central node"),
)

ORCASOUND_SITES: tuple[HydrophoneSite, ...] = (
    HydrophoneSite("rpi_bush_point",      "Bush Point",        48.0336, -122.6040, 8.0,  "OrcaSound", "Whidbey Island"),
    HydrophoneSite("rpi_orcasound_lab",   "Orcasound Lab",     48.5583, -123.1735, 8.0,  "OrcaSound", "San Juan Island"),
    HydrophoneSite("rpi_port_townsend",   "Port Townsend",     48.1370, -122.7600, 8.0,  "OrcaSound", "Port Townsend"),
    HydrophoneSite("rpi_north_sjc",       "North San Juan",    48.5917, -123.1430, 8.0,  "OrcaSound", "North San Juan Channel"),
    HydrophoneSite("rpi_sunset_bay",      "Sunset Bay",        47.8650, -122.3350, 8.0,  "OrcaSound", "Sunset Bay"),
)


def site_by_code(code: str) -> HydrophoneSite | None:
    for s in ONC_SITES + ORCASOUND_SITES:
        if s.site_code == code:
            return s
    return None


# --- envelope helpers --------------------------------------------------------

def site_envelope(site: HydrophoneSite) -> Envelope:
    """One-shot metadata so UI tiles / map plots know where this stream is."""
    payload = {
        "site_code": site.site_code,
        "name": site.name,
        "lat": site.lat,
        "lon": site.lon,
        "depth_m": site.depth_m,
        "network": site.network,
        "description": site.description,
    }
    return Envelope(
        topic=T_SITE,
        schema_name="planetar.acoustic.site",
        schema_version=1,
        correlation_id=site.site_code,
        payload=json.dumps(payload).encode("utf-8"),
    )


def detect_envelope(
    *,
    clip_id: str,
    site: HydrophoneSite,
    source_uri: str,
    start_ns: int,
    duration_s: float,
    snr_db: float,
    band_hz: tuple[float, float],
    peak_freq_hz: float,
    detected: bool,
) -> Envelope:
    payload = {
        "clip_id": clip_id,
        "site_code": site.site_code,
        "site_name": site.name,
        "lat": site.lat,
        "lon": site.lon,
        "network": site.network,
        "source": source_uri,
        "start_ns": start_ns,
        "dur_s": duration_s,
        "snr_db": snr_db,
        "band_hz": list(band_hz),
        "peak_freq_hz": peak_freq_hz,
        "detected": detected,
    }
    return Envelope(
        topic=T_DETECT,
        schema_name="planetar.acoustic.detect",
        schema_version=1,
        correlation_id=clip_id,
        payload=json.dumps(payload).encode("utf-8"),
    )


def psd_envelope(
    *,
    clip_id: str,
    site_code: str,
    start_ns: int,
    freqs_hz: np.ndarray,
    psd_db: np.ndarray,
    n_bins: int = 128,
) -> Envelope:
    """Compact PSD for UI waterfall. We downsample to ~n_bins log-spaced bins
    so a typical envelope is ~2 KiB.
    """
    if freqs_hz.shape != psd_db.shape:
        raise ValueError("freqs_hz and psd_db must match")
    # Drop DC and bin into n_bins log-spaced buckets across [10 Hz, fmax].
    mask = freqs_hz >= 10.0
    f = freqs_hz[mask]
    p = psd_db[mask]
    if f.size == 0:
        bins_f = []
        bins_p = []
    else:
        edges = np.geomspace(max(f[0], 10.0), float(f[-1]), n_bins + 1)
        bins_f = []
        bins_p = []
        for i in range(n_bins):
            sel = (f >= edges[i]) & (f < edges[i + 1])
            if sel.any():
                bins_f.append(float(np.sqrt(edges[i] * edges[i + 1])))
                bins_p.append(float(p[sel].mean()))
    payload = {
        "clip_id": clip_id,
        "site_code": site_code,
        "start_ns": start_ns,
        "freqs_hz": bins_f,
        "psd_db": bins_p,
    }
    return Envelope(
        topic=T_PSD,
        schema_name="planetar.acoustic.psd",
        schema_version=1,
        correlation_id=clip_id,
        payload=json.dumps(payload).encode("utf-8"),
    )


def hydrophone_chat_envelope(
    *,
    site: HydrophoneSite,
    text: str,
    author_id: str = "agent-acoustic",
    author_name: str = "acoustic",
    author_role: str = "agent",
) -> Envelope:
    """Emit a chat-channel message the planetar-ui already renders.

    The schema and payload shape mirror planetar-ui/bridge/synth.mjs exactly
    (chat.v1.Message with `text` + `author`) so the UI needs no changes — the
    message appears in the existing `hydrophone-alerts` channel alongside
    synthetic chatter.
    """
    payload = {
        "text": text,
        "author": {"id": author_id, "name": author_name, "role": author_role},
        "site_code": site.site_code,
    }
    return Envelope(
        topic=T_CHAT_HYDRO,
        schema_name="chat.v1.Message",
        schema_version=1,
        correlation_id=site.site_code,
        payload=json.dumps(payload).encode("utf-8"),
        source=author_id,
    )


def summarise_detection(
    *,
    site: HydrophoneSite,
    snr_db: float,
    peak_freq_hz: float,
    duration_s: float,
) -> str:
    """One-liner suitable for the hydrophone-alerts channel."""
    return (
        f"⟦{site.network}⟧ {site.name}: tonal at {peak_freq_hz:.0f} Hz, "
        f"SNR {snr_db:.1f} dB over {duration_s:.1f} s"
    )
