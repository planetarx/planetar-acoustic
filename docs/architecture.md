# planetar-acoustic — architecture

```
hydrophone files / stream
        │
        ▼
┌────────────────┐
│ ingest/audio   │  WAV/FLAC → mono float32 → resample → clip slicer
└────────┬───────┘
         │ Clip (samples, sr, clip_id, start_ns)
         ▼
┌────────────────┐
│ detect/        │  Welch PSD, in-band/out-of-band SNR check.
│   presence     │  Cheap gate — fires before SAI is even computed.
└────────┬───────┘
         │ PresenceResult.detected
         ▼ (if detected)
┌────────────────┐
│ sai/carfac     │  ERB-spaced gammatone bank + per-channel AGC → NAP
│ sai/sai        │  Strobed temporal integration → SAI image (C × L)
└────────┬───────┘
         │ SAI(image, cf_hz, sample_rate, lag_samples)
         ▼
┌────────────────┐
│ classify/      │  SAI → 3-channel image tensor → CNN (torch / onnx / mock)
│   sai_cnn      │  Returns class, score, top-K, embedding, model_id
└────────┬───────┘
         │ Hypothesis
         ▼
┌────────────────┐
│ bus/           │  zmesg.Envelope → length-prefixed TCP frame
│   publisher    │  planetar-broker @ 127.0.0.1:12001
└────────────────┘
         │
         ▼
   broker fans out to
   subscribers (planetar-ui, planetar-fusion, …)
```

## Bus envelopes emitted

| Topic | When | Payload |
|---|---|---|
| `acoustic.detect` | Every clip that passes the presence gate | `{clip_id, source, start_ns, dur_s, snr_db, band_hz, peak_freq_hz}` |
| `acoustic.sai` | If `--emit-sai` set (debug / UI heatmap) | `{clip_id, channels, lag_samples, sample_rate, sai_f16_b64}` |
| `acoustic.classify` | Every classified clip | `{clip_id, class, score, top_k, model_id, embedding_f16_b64?}` |

All three share the same `correlation_id = clip_id` so downstream services
(fusion, UI) can join them.

## Why SAI vs. spectrogram

A log-mel spectrogram is the obvious image-like representation of audio, and
ResNet / VGG models trained on AudioSet expect it. Why bother with SAI?

1. **Periodicity is the dominant cue for vessel ID.** Propeller blade rate,
   shaft rate, cavitation lines — they're all periodic. SAI is the
   representation that *makes periodicity local*: a stable horizontal stripe
   at lag = 1/f. Spectrograms encode periodicity diffusely via harmonic
   stacks; SAI encodes it directly.

2. **Phase / latency invariance.** The strobing step is lock-to-peak, so
   absolute phase is discarded and the image is stable under time shifts
   shorter than the analysis window. Useful when the same vessel signature
   appears at unpredictable offsets across passes.

3. **Domain transfer.** SAIs look enough like spectrogram-grade 2-D images
   that ImageNet-pretrained backbones work as a starting point — see
   Walters, Lyon et al. 2011 "Sound-ranking with stabilized auditory images,"
   which uses SAIs as Bag-of-Features inputs to large-scale audio search.
   Today the equivalent move is fine-tuning a vision backbone.

The pipeline is structured so swapping SAI for log-mel is a one-file change
(`sai/sai.py` becomes `sai/mel.py`); the rest of the stack treats it as
"some 2-D feature."

## Streaming vs. file-mode

This v0.1 reads files. For continuous hydrophone streams (ONC OPeNDAP, RTSP,
raw UDP) keep one persistent `CARFAC` instance per stream and feed it
chunks; the strobe + integrate loop is cheap to re-run each frame because
lag windows are short. Streaming wrapper lives in `ingest/stream.py` (not in
v0.1 — add when needed).

## Calibration & honest claims

- **No published latency numbers yet.** When we measure end-to-end clip →
  envelope-on-bus time, it'll be recorded in `docs/benchmarks.md` with the
  setup. No fake "10 ms!" headlines.
- **Mock classifier is mock.** When `--model` is unset, the classifier ships
  with `model_id="mock"` so anyone subscribing to `acoustic.classify`
  envelopes can filter out mock outputs in production.
- **Class list is configurable.** Default (`cargo, passenger, tanker, tug,
  fishing, background`) matches DeepShip + a background-class control; the
  actual labels must match whatever head was trained.

## Outstanding (v0.2+)

- Streaming hydrophone ingest (ONC + RTSP).
- Multi-time-constant AGC in CAR-FAC (Lyons HMH §15.4).
- True cascade topology instead of parallel filter bank — matters for
  high-CF tuning sharpness.
- Per-vessel re-ID via embedding distance, not just class prediction.
- WAB marine-mammal corpus integration for false-positive controls.
