"""planetar-acoustic command-line entry point.

Subcommands:

    ingest <files>   --   slice files into clips, print per-clip metadata
    detect <files>   --   run only the presence gate
    sai    <files>   --   compute SAI, save as .npy
    classify <files> --   ingest → presence → SAI → CV classifier
    run    <files>   --   batch: classify and publish to planetar-broker
    stream <source>  --   long-running daemon: ingest from a Source and
                          publish acoustic.* + chat.pac.hydrophone-alerts
                          envelopes continuously. This is the primary
                          mode for the planetar demo.
    sources          --   list known hydrophone sites with site codes
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from scipy.signal import welch

from planetar_acoustic.bus.publisher import Publisher
from planetar_acoustic.bus.topics import (
    ONC_SITES,
    ORCASOUND_SITES,
    detect_envelope,
    hydrophone_chat_envelope,
    psd_envelope,
    site_envelope,
    summarise_detection,
)
from planetar_acoustic.bus.zmesg import Envelope, uuid_str
from planetar_acoustic.classify.sai_cnn import DEFAULT_CLASSES, SAIClassifier
from planetar_acoustic.config import (
    DEFAULT_BROKER,
    DEFAULT_CLIP_SECONDS,
    DEFAULT_HOP_SECONDS,
    DEFAULT_SAI_DIR,
    DEFAULT_SAMPLE_RATE,
    PresenceParams,
    SAIParams,
)
from planetar_acoustic.detect.presence import detect_presence
from planetar_acoustic.ingest.audio import iter_clips
from planetar_acoustic.ingest.sources import Source, SourceClip
from planetar_acoustic.sai.sai import compute_sai

log = logging.getLogger("planetar_acoustic")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _common_clip_opts(f):
    f = click.option("--sample-rate", default=DEFAULT_SAMPLE_RATE, type=int)(f)
    f = click.option("--clip-s", default=DEFAULT_CLIP_SECONDS, type=float)(f)
    f = click.option("--hop-s", default=DEFAULT_HOP_SECONDS, type=float)(f)
    return f


@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """planetar-acoustic — hydrophone → Lyons SAI → CV classifier → bus."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_clip_opts
def ingest(inputs: tuple[str, ...], sample_rate: int, clip_s: float, hop_s: float) -> None:
    """Slice audio files into clips and print metadata."""
    if not inputs:
        raise click.UsageError("provide one or more audio file paths")
    for path in inputs:
        for clip in iter_clips(Path(path), sample_rate=sample_rate, clip_s=clip_s, hop_s=hop_s):
            click.echo(json.dumps({
                "clip_id": clip.clip_id,
                "source": clip.source_path,
                "start_ns": clip.start_ns,
                "dur_s": round(clip.duration_s, 3),
                "rms": float(np.sqrt(np.mean(clip.samples.astype(np.float64) ** 2))),
            }))


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_clip_opts
@click.option("--snr-db-min", default=PresenceParams().snr_db_min, type=float)
def detect(inputs: tuple[str, ...], sample_rate: int, clip_s: float, hop_s: float, snr_db_min: float) -> None:
    """Run the presence gate over each clip; JSON-lines result per clip."""
    if not inputs:
        raise click.UsageError("provide one or more audio file paths")
    params = PresenceParams(snr_db_min=snr_db_min)
    for path in inputs:
        for clip in iter_clips(Path(path), sample_rate=sample_rate, clip_s=clip_s, hop_s=hop_s):
            r = detect_presence(clip.samples, sample_rate=clip.sample_rate, params=params)
            out = {"clip_id": clip.clip_id, **asdict(r)}
            click.echo(json.dumps(out))


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_clip_opts
@click.option("--out", "out_dir", default=str(DEFAULT_SAI_DIR), type=click.Path())
@click.option("--lag-ms", default=SAIParams().lag_ms, type=float)
@click.option("--n-channels", default=SAIParams().n_channels, type=int)
def sai(
    inputs: tuple[str, ...],
    sample_rate: int,
    clip_s: float,
    hop_s: float,
    out_dir: str,
    lag_ms: float,
    n_channels: int,
) -> None:
    """Compute SAI per clip and save as .npy under out_dir."""
    if not inputs:
        raise click.UsageError("provide one or more audio file paths")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        for clip in iter_clips(Path(path), sample_rate=sample_rate, clip_s=clip_s, hop_s=hop_s):
            s = compute_sai(
                clip.samples,
                sample_rate=clip.sample_rate,
                n_channels=n_channels,
                lag_ms=lag_ms,
            )
            target = out_path / f"{clip.clip_id.replace(':', '_')}.npy"
            np.save(target, s.image)
            click.echo(json.dumps({
                "clip_id": clip.clip_id,
                "sai_path": str(target),
                "shape": list(s.image.shape),
                "n_strobes": s.n_strobes,
            }))


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_clip_opts
@click.option("--model", default=None, type=click.Path(),
              help="path to .pt (torch) or .onnx (onnxruntime); mock if unset")
@click.option("--classes", default=",".join(DEFAULT_CLASSES),
              help="comma-separated class labels matching the model head")
@click.option("--gate/--no-gate", default=True, help="apply presence gate before classify")
def classify(
    inputs: tuple[str, ...],
    sample_rate: int,
    clip_s: float,
    hop_s: float,
    model: str | None,
    classes: str,
    gate: bool,
) -> None:
    """ingest → presence → SAI → classify. Prints JSON-lines hypotheses."""
    if not inputs:
        raise click.UsageError("provide one or more audio file paths")
    classifier = SAIClassifier(model=model, classes=tuple(c.strip() for c in classes.split(",")))
    presence_params = PresenceParams()
    for path in inputs:
        for clip in iter_clips(Path(path), sample_rate=sample_rate, clip_s=clip_s, hop_s=hop_s):
            if gate:
                p = detect_presence(clip.samples, clip.sample_rate, presence_params)
                if not p.detected:
                    continue
            s = compute_sai(clip.samples, sample_rate=clip.sample_rate)
            h = classifier.predict(s)
            click.echo(json.dumps({
                "clip_id": clip.clip_id,
                "source": clip.source_path,
                "class": h.class_,
                "score": h.score,
                "top_k": h.top_k,
                "model_id": h.model_id,
            }))


@main.command()
@click.argument("inputs", nargs=-1, type=click.Path(exists=True))
@_common_clip_opts
@click.option("--model", default=None, type=click.Path())
@click.option("--classes", default=",".join(DEFAULT_CLASSES))
@click.option("--broker", default=DEFAULT_BROKER)
@click.option("--emit-sai/--no-emit-sai", default=False,
              help="also publish acoustic.sai envelopes carrying SAI thumbnails")
def run(
    inputs: tuple[str, ...],
    sample_rate: int,
    clip_s: float,
    hop_s: float,
    model: str | None,
    classes: str,
    broker: str,
    emit_sai: bool,
) -> None:
    """End-to-end: ingest → detect → SAI → classify → publish to planetar-broker."""
    if not inputs:
        raise click.UsageError("provide one or more audio file paths")
    classifier = SAIClassifier(model=model, classes=tuple(c.strip() for c in classes.split(",")))
    presence_params = PresenceParams()
    n_clips = n_det = n_class = 0
    try:
        with Publisher.from_endpoint(broker) as pub:
            for path in inputs:
                for clip in iter_clips(Path(path), sample_rate=sample_rate, clip_s=clip_s, hop_s=hop_s):
                    n_clips += 1
                    p = detect_presence(clip.samples, clip.sample_rate, presence_params)
                    if not p.detected:
                        continue
                    n_det += 1
                    det_payload = {
                        "clip_id": clip.clip_id,
                        "source": clip.source_path,
                        "start_ns": clip.start_ns,
                        "dur_s": clip.duration_s,
                        "snr_db": p.snr_db,
                        "band_hz": [p.band_lo_hz, p.band_hi_hz],
                        "peak_freq_hz": p.peak_freq_hz,
                    }
                    det_env = Envelope(
                        topic="acoustic.detect",
                        schema_name="planetar.acoustic.detect",
                        schema_version=1,
                        correlation_id=clip.clip_id,
                        payload=json.dumps(det_payload).encode("utf-8"),
                    )
                    pub.publish(det_env)
                    # Everything derived from this clip cites the detect
                    # envelope as its cause (correlation stays clip_id).
                    det_id = uuid_str(det_env.id)

                    s = compute_sai(clip.samples, sample_rate=clip.sample_rate)
                    if emit_sai:
                        sai_payload = {
                            "clip_id": clip.clip_id,
                            "channels": s.n_channels,
                            "lag_samples": s.lag_samples,
                            "sample_rate": s.sample_rate,
                            "sai_f16_b64": base64.b64encode(
                                s.image.astype(np.float16).tobytes()
                            ).decode("ascii"),
                        }
                        pub.publish(Envelope(
                            topic="acoustic.sai",
                            schema_name="planetar.acoustic.sai",
                            schema_version=1,
                            correlation_id=clip.clip_id,
                            causation_id=det_id,
                            payload=json.dumps(sai_payload).encode("utf-8"),
                        ))

                    h = classifier.predict(s)
                    cls_payload = {
                        "clip_id": clip.clip_id,
                        "class": h.class_,
                        "score": h.score,
                        "top_k": h.top_k,
                        "model_id": h.model_id,
                    }
                    if h.embedding is not None:
                        cls_payload["embedding_f16_b64"] = base64.b64encode(
                            h.embedding.astype(np.float16).tobytes()
                        ).decode("ascii")
                    pub.publish(Envelope(
                        topic="acoustic.classify",
                        schema_name="planetar.acoustic.classify",
                        schema_version=1,
                        correlation_id=clip.clip_id,
                        causation_id=det_id,
                        payload=json.dumps(cls_payload).encode("utf-8"),
                    ))
                    n_class += 1
    except ConnectionRefusedError as e:
        click.echo(f"broker connection refused at {broker}: {e}", err=True)
        sys.exit(2)
    click.echo(f"processed {n_clips} clips → {n_det} detections → {n_class} classifications")


@main.command()
def sources() -> None:
    """List built-in hydrophone sites (site codes for `stream --source ...`)."""
    for s in ONC_SITES + ORCASOUND_SITES:
        click.echo(
            f"{s.network:10s} {s.site_code:20s} {s.name:32s} "
            f"({s.lat:+.4f}, {s.lon:+.4f}) {s.depth_m:.0f}m  — {s.description}"
        )


def _build_source(spec: str, sample_rate: int, clip_s: float, hop_s: float, pace_realtime: bool):
    """Parse a source spec into a Source instance.

    Specs:
      synth                       — deterministic synthesizer
      archive:/path/to/folder     — local folder, sorted replay
      orcasound:<node>            — HLS pull (requires ffmpeg)
      onc:<location_code>         — Oceans 3.0 polled-live
      onc:<location_code>:<dateFrom>:<dateTo>
                                  — Oceans 3.0 archive window (ISO 8601)
    """
    kind, _, rest = spec.partition(":")
    kind = kind.lower()
    if kind == "synth":
        from planetar_acoustic.ingest.sources.synth import SyntheticSource
        return SyntheticSource(
            sample_rate=sample_rate, clip_seconds=clip_s, pace_realtime=pace_realtime,
        )
    if kind == "archive":
        if not rest:
            raise click.UsageError("archive source requires a folder: archive:/path/to/folder")
        from planetar_acoustic.ingest.sources.archive import ArchiveSource
        return ArchiveSource(
            folder=Path(rest),
            sample_rate=sample_rate,
            clip_seconds=clip_s,
            hop_seconds=hop_s,
            pace="realtime" if pace_realtime else "asfast",
            loop=True,
        )
    if kind == "orcasound":
        from planetar_acoustic.ingest.sources.orcasound import OrcaSoundSource
        node = rest or "rpi_bush_point"
        return OrcaSoundSource(
            node=node, sample_rate=sample_rate, clip_seconds=clip_s, hop_seconds=hop_s,
        )
    if kind == "onc":
        from planetar_acoustic.ingest.sources.onc import ONCSource
        parts = rest.split(":")
        location_code = parts[0]
        window = None
        if len(parts) >= 3:
            window = (parts[1], parts[2])
        return ONCSource(
            location_code=location_code,
            sample_rate=sample_rate,
            clip_seconds=clip_s,
            hop_seconds=hop_s,
            window=window,
        )
    raise click.UsageError(f"unknown source kind: {kind}")


@main.command()
@click.option("--source", "source_spec", default="synth",
              help="source spec: synth | archive:DIR | orcasound:NODE | onc:CODE[:FROM:TO]")
@_common_clip_opts
@click.option("--broker", default=DEFAULT_BROKER)
@click.option("--snr-db-min", default=PresenceParams().snr_db_min, type=float,
              help="presence-gate threshold; chat messages also use this")
@click.option("--classify/--no-classify", default=False,
              help="run the SAI + CV classifier per detection (off by default in stream)")
@click.option("--model", default=None, type=click.Path(),
              help="classifier model path (only used with --classify)")
@click.option("--classes", default=",".join(DEFAULT_CLASSES))
@click.option("--realtime/--asfast", default=True,
              help="pace the source at wall-clock (default) vs. as fast as possible")
@click.option("--emit-psd/--no-emit-psd", default=True,
              help="publish compact PSD envelopes for UI waterfall")
@click.option("--chat-snr-db-min", default=10.0, type=float,
              help="post a hydrophone-alerts chat message when SNR exceeds this")
def stream(
    source_spec: str,
    sample_rate: int,
    clip_s: float,
    hop_s: float,
    broker: str,
    snr_db_min: float,
    classify: bool,
    model: str | None,
    classes: str,
    realtime: bool,
    emit_psd: bool,
    chat_snr_db_min: float,
) -> None:
    """Long-running ingest daemon: source → broker.

    Picks a Source (synth / archive / orcasound / onc), connects to
    planetar-broker, and publishes:

      acoustic.site                 — once per site at startup
      acoustic.detect               — every clip
      acoustic.psd                  — every clip (compact, ~2 KiB)
      chat.pac.hydrophone-alerts    — when SNR exceeds --chat-snr-db-min
      acoustic.classify             — only if --classify is set
    """
    presence_params = PresenceParams(snr_db_min=snr_db_min)
    src = _build_source(source_spec, sample_rate, clip_s, hop_s, pace_realtime=realtime)
    classifier = None
    if classify:
        classifier = SAIClassifier(
            model=model,
            classes=tuple(c.strip() for c in classes.split(",")),
        )

    log.info("stream: source=%s broker=%s classify=%s", source_spec, broker, classify)
    n_clips = n_det = n_chat = n_class = 0
    pub: Publisher | None = None
    try:
        pub = Publisher.from_endpoint(broker)
        pub.connect()
        # Announce sites once.
        for site in src.sites():
            pub.publish(site_envelope(site))
            log.info("announced site: %s (%s)", site.site_code, site.name)

        for sclip in src.stream():
            n_clips += 1
            clip, site = sclip.clip, sclip.site
            p = detect_presence(clip.samples, clip.sample_rate, presence_params)
            det_env = detect_envelope(
                clip_id=clip.clip_id,
                site=site,
                source_uri=clip.source_path,
                start_ns=clip.start_ns,
                duration_s=clip.duration_s,
                snr_db=p.snr_db,
                band_hz=(p.band_lo_hz, p.band_hi_hz),
                peak_freq_hz=p.peak_freq_hz,
                detected=p.detected,
            )
            pub.publish(det_env)
            # Everything derived from this clip cites the detect envelope as
            # its cause (correlation stays clip_id).
            det_id = uuid_str(det_env.id)
            if p.detected:
                n_det += 1

            if emit_psd:
                nperseg = min(clip.samples.shape[0], 1 << 12)
                if nperseg >= 64:
                    f, pxx = welch(clip.samples, fs=clip.sample_rate, nperseg=nperseg,
                                   scaling="density")
                    psd_db = 10.0 * np.log10(np.maximum(pxx, 1e-20))
                    pub.publish(psd_envelope(
                        clip_id=clip.clip_id,
                        site_code=site.site_code,
                        start_ns=clip.start_ns,
                        freqs_hz=f.astype(np.float32),
                        psd_db=psd_db.astype(np.float32),
                        causation_id=det_id,
                    ))

            # Classify before the chat alert so the alert can cite the
            # classification it announces; without --classify it cites the
            # detect envelope instead.
            chat_causation = det_id
            if classifier is not None and p.detected:
                s = compute_sai(clip.samples, sample_rate=clip.sample_rate)
                h = classifier.predict(s)
                cls_payload = {
                    "clip_id": clip.clip_id,
                    "site_code": site.site_code,
                    "class": h.class_,
                    "score": h.score,
                    "top_k": h.top_k,
                    "model_id": h.model_id,
                }
                cls_env = Envelope(
                    topic="acoustic.classify",
                    schema_name="planetar.acoustic.classify",
                    schema_version=1,
                    correlation_id=clip.clip_id,
                    causation_id=det_id,
                    payload=json.dumps(cls_payload).encode("utf-8"),
                )
                pub.publish(cls_env)
                n_class += 1
                chat_causation = uuid_str(cls_env.id)

            if p.detected and p.snr_db >= chat_snr_db_min:
                text = summarise_detection(
                    site=site,
                    snr_db=p.snr_db,
                    peak_freq_hz=p.peak_freq_hz,
                    duration_s=clip.duration_s,
                )
                pub.publish(hydrophone_chat_envelope(
                    site=site, text=text, causation_id=chat_causation,
                ))
                n_chat += 1

            if n_clips % 10 == 0:
                log.info("stream: clips=%d det=%d chat=%d class=%d",
                         n_clips, n_det, n_chat, n_class)
    except KeyboardInterrupt:
        click.echo(f"\nstopped: clips={n_clips} det={n_det} chat={n_chat} class={n_class}")
    except ConnectionRefusedError as e:
        click.echo(f"broker connection refused at {broker}: {e}", err=True)
        sys.exit(2)
    finally:
        if pub is not None:
            pub.close()


if __name__ == "__main__":
    main()
