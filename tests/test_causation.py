"""Distributed-tracing wiring: derived acoustic envelopes must cite the
clip's acoustic.detect envelope via `causation_id` (dashed-UUID string),
while `correlation_id` stays the clip_id.
"""
from __future__ import annotations

import json

import numpy as np
import soundfile as sf
from click.testing import CliRunner

from planetar_acoustic.bus.topics import (
    ONC_SITES,
    detect_envelope,
    hydrophone_chat_envelope,
    psd_envelope,
)
from planetar_acoustic.bus.zmesg import Envelope, parse, uuid_str
from planetar_acoustic.cli import main

SITE = ONC_SITES[0]


def _detect(clip_id: str = "clip:0") -> Envelope:
    return detect_envelope(
        clip_id=clip_id,
        site=SITE,
        source_uri="synth:",
        start_ns=0,
        duration_s=5.0,
        snr_db=12.0,
        band_hz=(100.0, 4000.0),
        peak_freq_hz=300.0,
        detected=True,
    )


def test_psd_causation_is_detect_envelope_id():
    det = _detect()
    det_id = uuid_str(det.id)
    psd = psd_envelope(
        clip_id="clip:0",
        site_code=SITE.site_code,
        start_ns=0,
        freqs_hz=np.array([10.0, 100.0, 1000.0]),
        psd_db=np.array([-80.0, -70.0, -90.0]),
        causation_id=det_id,
    )
    assert psd.causation_id == det_id
    assert psd.correlation_id == "clip:0"
    # detect envelopes are roots of the acoustic chain
    assert det.causation_id == ""
    # and the link survives serialization
    assert parse(psd.serialize()).causation_id == det_id


def test_chat_alert_cites_the_envelope_it_announces():
    det = _detect()
    chat = hydrophone_chat_envelope(
        site=SITE, text="tonal at 300 Hz", causation_id=uuid_str(det.id),
    )
    assert parse(chat.serialize()).causation_id == uuid_str(det.id)
    assert chat.correlation_id == SITE.site_code


class _FakePublisher:
    """Captures envelopes instead of shipping them to the broker."""

    instances: list[_FakePublisher] = []

    def __init__(self) -> None:
        self.published: list[Envelope] = []
        _FakePublisher.instances.append(self)

    @classmethod
    def from_endpoint(cls, endpoint: str) -> _FakePublisher:
        return cls()

    def publish(self, env: Envelope) -> int:
        self.published.append(env)
        return len(env.serialize()) + 4

    def __enter__(self) -> _FakePublisher:
        return self

    def __exit__(self, *exc) -> None:
        pass


def test_run_wires_sai_and_classify_causation(tmp_path, monkeypatch):
    """End-to-end `run` path: acoustic.sai + acoustic.classify envelopes must
    cite the same clip's acoustic.detect envelope id."""
    sr = 16000
    t = np.arange(int(sr * 0.6), dtype=np.float32) / sr
    tone = 0.5 * np.sin(2 * np.pi * 300.0 * t)  # in-band tonal, SNR >> gate
    wav = tmp_path / "tone.wav"
    sf.write(wav, tone, sr)

    _FakePublisher.instances.clear()
    monkeypatch.setattr("planetar_acoustic.cli.Publisher", _FakePublisher)
    result = CliRunner().invoke(main, [
        "run", str(wav),
        "--sample-rate", str(sr), "--clip-s", "0.6", "--hop-s", "0.6",
        "--emit-sai",
    ])
    assert result.exit_code == 0, result.output

    (pub,) = _FakePublisher.instances
    detects = {e.correlation_id: e for e in pub.published if e.topic == "acoustic.detect"}
    derived = [e for e in pub.published if e.topic in ("acoustic.sai", "acoustic.classify")]
    assert detects and derived
    for env in derived:
        det = detects[env.correlation_id]
        assert env.causation_id == uuid_str(det.id), env.topic
        assert json.loads(env.payload)["clip_id"] == env.correlation_id
