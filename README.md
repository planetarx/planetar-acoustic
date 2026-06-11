# planetar-acoustic

Underwater-acoustic vessel-identification microservice for **planetarx** / `zdefence`.

Takes hydrophone audio, runs it through a Lyons-style cochlear model to produce
**Stabilized Auditory Images (SAIs)**, feeds those SAIs into a computer-vision
classifier, and publishes per-clip hypotheses to `planetar-broker` as native
`zmesg` envelopes.

Built for the CH13 v2 proposal flagship demo (dark-vessel re-identification over
the Salish Sea, fused with AIS / SAR / EO / RF on a single bus).

## Why SAI

A Stabilized Auditory Image (Dick Lyons, *Human and Machine Hearing*, 2017) is
a 2-D, image-like representation of sound produced by:

1. A **CAR-FAC** cochlear cascade (Cascade of Asymmetric Resonators with
   Fast-Acting Compression) that mimics the basilar membrane — bandpass + AGC.
2. A **NAP** (Neural Activity Pattern) — half-wave rectified, compressed
   channel outputs.
3. **Strobed Temporal Integration** — for each channel, lock to local energy
   peaks ("strobes") and average lag-aligned windows. The result is stable for
   periodic / quasi-periodic sounds, which is exactly the kind of signature
   vessels produce (propeller blade rate, shaft rate, cavitation lines).

The SAI is a 2-D image (cochlear-channel × time-lag), so any image classifier
— ResNet, ViT, a custom CNN — can ingest it. That is the bridge between
psychoacoustic feature extraction and the CV stack the rest of `planetarx` uses
for SAR / EO.

References:

- R. F. Lyon, *Human and Machine Hearing: Extracting Meaning from Sound*,
  Cambridge UP, 2017. Chapters 17–19 cover SAI; Ch. 14–16 cover CAR-FAC.
- T. C. Walters, R. F. Lyon, et al., "Sound-ranking with stabilized auditory
  images," 2011.
- Google CARFAC reference: <https://github.com/google/carfac>.

## Layout

```
planetar-acoustic/
├── src/planetar_acoustic/
│   ├── ingest/         # WAV/FLAC reader, resampler, clip slicer
│   ├── sai/            # CAR-FAC cochlea + strobed temporal integration → SAI
│   ├── detect/         # cheap presence gate (band-energy + spectral shape)
│   ├── classify/       # CV classifier over SAI images (torch / ONNX / mock)
│   ├── bus/            # zmesg encoder + TCP publisher (mirrors planetar-sat)
│   ├── cli.py          # `planetar-acoustic ingest|sai|detect|classify|run`
│   └── config.py
├── tests/
└── docs/architecture.md
```

## Quickstart

```bash
make install                                   # editable install + dev deps
make install-ml                                # adds torch / torchvision / onnxruntime
.venv/bin/planetar-acoustic sai clip.wav --out out.npy
.venv/bin/planetar-acoustic run data/clips/*.wav --broker 127.0.0.1:12001
```

### Primary mode: `stream`

`stream` is the long-running ingest daemon — it picks a Source, connects to
planetar-broker, and continuously publishes envelopes the planetar-ui already
renders:

```bash
.venv/bin/planetar-acoustic sources                            # list site codes
.venv/bin/planetar-acoustic stream --source synth              # offline / CI
.venv/bin/planetar-acoustic stream --source archive:/path/to/wavs  --realtime
.venv/bin/planetar-acoustic stream --source onc:FGPD           # ONC Folger Deep polled-live (needs ONC_TOKEN)
.venv/bin/planetar-acoustic stream --source onc:CBYIP:2025-03-14T08:00:00.000Z:2025-03-14T09:00:00.000Z   # archive window
.venv/bin/planetar-acoustic stream --source orcasound:rpi_bush_point   # OrcaSound HLS (needs ffmpeg)
.venv/bin/planetar-acoustic stream --source synth --classify --model my.pt  # also run CV classifier
```

The batch-mode `run` subcommand still exists for one-shot file processing,
and `sai`/`classify`/`detect`/`ingest` remain for tooling.

## Bus topics

| Topic                          | Schema                                                                  | Notes |
|--------------------------------|-------------------------------------------------------------------------|-------|
| `acoustic.site`                | `{site_code, name, lat, lon, depth_m, network, description}`            | Emitted once per source at startup so UI tiles know where this stream is. |
| `acoustic.detect`              | `{clip_id, site_code, lat, lon, source, start_ns, dur_s, snr_db, band_hz, peak_freq_hz, detected}` | Every clip. `correlation_id = clip_id`. |
| `acoustic.psd`                 | `{clip_id, site_code, start_ns, freqs_hz[~128], psd_db[~128]}`          | Compact log-spaced PSD (~2 KiB) for UI waterfall. |
| `acoustic.classify`            | `{clip_id, site_code, class, score, top_k, model_id}`                   | Only when `--classify` is set. |
| `acoustic.sai`                 | `{clip_id, channels, lag_samples, sample_rate, sai_f16_b64}`            | Debug only, opt-in. |
| `chat.pac.hydrophone-alerts`   | `chat.v1.Message` — `{text, author:{id,name,role}, site_code}`          | One-liner per detection (when SNR ≥ `--chat-snr-db-min`). Renders in the **existing** planetar-ui channel. |

Wire framing: 4-byte **big-endian** length prefix (network byte order — broker
calls `ntohl`), then a little-endian `zmesg` envelope. The envelope is the
same shape as `~/github/sness23/zmesg/zmesg.h`.

## Data sources

### Built-in (`--source ...`)

| Source spec                  | Network needed?  | What it does                                                      |
|------------------------------|------------------|-------------------------------------------------------------------|
| `synth`                      | no               | Deterministic synthesizer — CI / offline demo.                    |
| `archive:DIR`                | no               | Replay a folder of WAV/FLAC at wall-clock or as-fast-as-possible. |
| `onc:CODE`                   | yes (HTTPS + `ONC_TOKEN`) | ONC Oceans 3.0 — polled-live (60 s cadence).             |
| `onc:CODE:FROM:TO`           | yes              | ONC archive window in ISO 8601.                                   |
| `orcasound:NODE`             | yes (HTTPS + `ffmpeg`) | OrcaSound HLS live feed.                                    |

### Site catalog (`planetar-acoustic sources`)

**ONC (Salish Sea / BC coast)** — matches what the CH13 proposal cites:
`BACAX`, `BACUS` (Barkley Canyon, NEPTUNE), `FGPD`, `FGPPN` (Folger Passage),
`CBYIP` (Saanich Inlet, VENUS), `CBYDS` (Cambridge Bay), `SEVIP` / `USDDL`
(Strait of Georgia, VENUS).

**OrcaSound (no auth required)**: `rpi_bush_point`, `rpi_orcasound_lab`,
`rpi_port_townsend`, `rpi_north_sjc`, `rpi_sunset_bay`.

### Historic / training corpora (replay via `archive:`)

- **DeepShip** (Esmaeilpour et al., 2021) — 47 h labeled commercial vessels.
- **ShipsEar** (Santos-Domínguez et al., 2016) — 90 min labeled coastal.
- **NOAA SanctSound** — multi-sanctuary multi-year hydrophone archive.
- **OOI Cabled Array hydrophones** (Axial Seamount, Oregon margin).
- **WAB Watkins Marine Mammal Sound Database** — false-positive controls.
- **DCLDE bioacoustic challenge** corpora — annotated detection sets.

Drop any of these into a folder, point `archive:` at it, and the daemon
replays at wall-clock pace into the broker.

See `~/github/planetarx/planetar/05-DATASETS.md` for the full modality list.

## How this appears in planetar-ui

`chat.pac.hydrophone-alerts` is **already** a recognized channel in
`planetar-ui/bridge/synth.mjs`, with the `chat.v1.Message` schema. Real
envelopes from this service appear in that channel alongside synthetic
chatter — no UI changes needed for the demo.

For richer surfaces (per-site map markers, PSD waterfall tiles, classifier
hypothesis annotations) the UI consumes `acoustic.site` + `acoustic.psd` +
`acoustic.classify` envelopes, all sharing `correlation_id = clip_id` so the
view can join across them.

## Status

v0.2 — ingest-first. The primary value is getting hydrophone data into the
planetarx bus and visible in the UI; the SAI + CV classifier stack is
optional infrastructure for later work.

- ONC and OrcaSound source classes are wired but their **network paths have
  not been exercised** from this environment. The Source interface is
  correct; the live demo wiring needs `ONC_TOKEN` set and outbound HTTPS.
- CAR-FAC is a parallel filter bank in SOS form, not the full Lyons
  cascade-of-asymmetric-resonators topology — flagged in `sai/carfac.py`.
- The mock classifier carries `model_id="mock"` so downstream consumers can
  filter it out in production.

No fake numbers. Any benchmark printed by this service is measured.

## Licensing

Licensed under **AGPL-3.0** (see [`LICENSE`](LICENSE)). **Commercial licenses**
(for use without AGPL obligations) are available — contact `sness@sness.net`.
