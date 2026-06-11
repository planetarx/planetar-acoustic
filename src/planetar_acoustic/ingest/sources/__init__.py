"""Hydrophone data sources.

A Source yields ``Clip`` objects (from ``ingest.audio``) tagged with a
``HydrophoneSite``. The stream daemon doesn't care whether the bytes came
from a live HLS feed, an ONC archive product, or a folder of WAVs — it just
processes them and publishes envelopes.

Concrete sources:

  * ``synth.SyntheticSource``        — deterministic synthesizer (no network)
  * ``archive.ArchiveSource``        — local folder, wall-clock or sped-up replay
  * ``onc.ONCSource``                — Oceans 3.0 dataProductDelivery (ONC token)
  * ``orcasound.OrcaSoundSource``    — public HLS feeds (requires ffmpeg)
"""
from planetar_acoustic.ingest.sources.base import Source, SourceClip

__all__ = ["Source", "SourceClip"]
