"""Network-free tests for the synth and archive sources."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from planetar_acoustic.ingest.sources.archive import ArchiveSource
from planetar_acoustic.ingest.sources.synth import SyntheticSource


def test_synth_yields_finite_clips_with_max_clips():
    src = SyntheticSource(sample_rate=16000, clip_seconds=0.25, pace_realtime=False, max_clips=3)
    clips = list(src.stream())
    assert len(clips) == 3
    for sc in clips:
        assert sc.clip.samples.shape == (4000,)
        assert sc.site.site_code == "SYNTH-01"
        assert sc.clip.sample_rate == 16000


def test_synth_includes_periodic_quiet_clip():
    src = SyntheticSource(sample_rate=16000, clip_seconds=0.1, pace_realtime=False, max_clips=5)
    clips = list(src.stream())
    rms = [float(np.sqrt(np.mean(c.clip.samples ** 2))) for c in clips]
    # Clip index 4 is the muted one.
    assert rms[4] < min(rms[:4]) * 0.5, f"expected quiet clip at idx 4: {rms}"


def test_archive_replays_files_in_sorted_order(tmp_path: Path):
    sr = 16000
    for name, freq in [("b_400.wav", 400.0), ("a_800.wav", 800.0)]:
        t = np.arange(sr, dtype=np.float32) / sr
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(str(tmp_path / name), audio, sr)
    src = ArchiveSource(
        folder=tmp_path,
        sample_rate=sr,
        clip_seconds=0.5,
        hop_seconds=0.25,
        pace="asfast",
        loop=False,
    )
    sources = [sc.clip.source_path for sc in src.stream()]
    assert sources, "archive source produced no clips"
    # Sorted order: a_800 before b_400.
    first = Path(sources[0]).name
    assert first == "a_800.wav", f"expected a_800.wav first, got {first}"


def test_archive_empty_folder_yields_nothing(tmp_path: Path):
    src = ArchiveSource(folder=tmp_path, pace="asfast", loop=False)
    assert list(src.stream()) == []


def test_psd_envelope_payload_shape():
    """The compact PSD envelope must downsample to ~n_bins log-spaced bins."""
    from planetar_acoustic.bus.topics import psd_envelope
    import json
    freqs = np.linspace(0, 8000, 2049, dtype=np.float32)
    psd = (-80.0 + 20.0 * np.sin(freqs / 500.0)).astype(np.float32)
    env = psd_envelope(
        clip_id="t", site_code="SYNTH-01", start_ns=0,
        freqs_hz=freqs, psd_db=psd, n_bins=32,
    )
    payload = json.loads(env.payload.decode("utf-8"))
    assert payload["clip_id"] == "t"
    assert len(payload["freqs_hz"]) <= 32
    assert len(payload["psd_db"]) == len(payload["freqs_hz"])
    # All freqs should be above 10 Hz floor and below original max.
    assert all(10.0 <= f <= 8000.0 for f in payload["freqs_hz"])


def test_summarise_detection_string_format():
    from planetar_acoustic.bus.topics import ONC_SITES, summarise_detection
    s = ONC_SITES[0]  # Barkley Canyon Axis
    text = summarise_detection(site=s, snr_db=12.4, peak_freq_hz=230.0, duration_s=4.0)
    assert "Barkley" in text
    assert "230" in text
    assert "12.4" in text
    assert "⟦ONC⟧" in text
