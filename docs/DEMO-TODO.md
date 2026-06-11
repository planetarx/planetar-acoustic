# planetar-acoustic — what's left to do for the demo

**As of 2026-05-15.** Goal: real hydrophone audio flowing through `planetar-acoustic`
into `planetar-broker` and visible in `planetar-ui`, in time for the CH13 flagship
demo (submission target 2026-05-29).

This is an action list, ordered by priority. Items P1–P3 are demo-blocking; P4 is
release hygiene; P5 is post-submission.

## Where things stand

- ✅ Service builds, 22/22 tests pass, end-to-end verified against the live broker
  with the `synth` source (2026-05-14).
- ✅ Repo public at <https://github.com/sness23/planetar-acoustic>.
- ✅ MBARI source path verified from this machine — real ocean audio pulled and run
  through `detect` (2026-05-15).
- ⚠️ ONC and OrcaSound source code is wired but **not working end-to-end** — see P2, P3.

---

## P1 — Get real audio into the demo (do this first)

The one source that works today with **no account and no fixes**: MBARI Pacific
Ocean Sound (Monterey Bay, public AWS Open Data, busy shipping lanes).

- [ ] **Pull an MBARI demo clip.** A 24 h file is ~518 MB; a ranged download grabs
      a usable slice in seconds:
      ```bash
      cd ~/github/planetarx/planetar-acoustic
      aws s3api get-object --no-sign-request \
        --bucket pacific-sound-2khz \
        --key 2022/01/MARS-20220101T000000Z-2kHz.wav \
        --range bytes=0-33554431 \
        data/clips/mbari_demo.wav      # first 32 MB ≈ 90 min at 2 kHz
      ```
      (Browse keys with `aws s3 ls --no-sign-request s3://pacific-sound-2khz/2022/`.
      Pick a window with known ship traffic if you want a guaranteed detection.)
- [ ] **Start the broker** (`~/github/planetarx/planetar-broker`) if not already up.
- [ ] **Stream it into the bus:**
      ```bash
      .venv/bin/planetar-acoustic stream --source archive:data/clips --realtime
      ```
- [ ] **Confirm it lands in planetar-ui** — open the UI, check the
      `chat.pac.hydrophone-alerts` channel for detection one-liners.

Outcome: a reproducible, no-credentials demo. Good enough to show on its own.

## P2 — Salish Sea data (the geography the proposal actually claims)

MBARI is Monterey Bay. The proposal is about the Salish Sea / Victoria. To demo on
home water, use Ocean Networks Canada (VENUS / NEPTUNE) — the sites the CH13 docs
already cite (Saanich Inlet, Barkley Canyon, Folger Passage).

- [ ] **Register a free ONC account** at <https://data.oceannetworks.ca> and
      generate an API token (Profile → Web Services API).
- [ ] **Export it:** `export ONC_TOKEN=<token>` (add to shell profile or a
      `.env` the service reads).
- [ ] **Test the ONC source** against one site, archive window first (deterministic):
      ```bash
      .venv/bin/planetar-acoustic stream \
        --source onc:CBYIP:2025-03-14T08:00:00.000Z:2025-03-14T09:00:00.000Z
      ```
      Then polled-live: `--source onc:FGPD`.
- [ ] If the ONC API calls fail, the bug is in
      `src/planetar_acoustic/ingest/sources/onc.py` — the data-product request /
      poll / download path has never been exercised against the live API. Budget
      time to debug it, don't assume it works.

## P3 — Fix the two known-broken bits

Found 2026-05-15 while verifying sources:

- [ ] **OrcaSound live source is broken.** The HLS URL hardcoded in
      `src/planetar_acoustic/ingest/sources/orcasound.py`
      (`s3-us-west-2.amazonaws.com/streaming-orcasound-net/<node>/hls/live/playlist.m3u8`)
      returns **403 Forbidden**, and the streaming bucket no longer allows public
      listing. OrcaSound has changed how they deliver live audio. Either find the
      current stream URL (check <https://github.com/orcasound/orcadata>) and update
      the source, or drop OrcaSound as a demo source.
- [ ] **OrcaSound archive is cold storage.** Files in `archive-orcasound-net` are
      **GLACIER** class — they need a restore request (hours) before download, so
      `archive:` replay of OrcaSound data is not grab-and-go. Note this if anyone
      plans to use it.
- [ ] **Update `README.md`** — the "Data sources" table currently lists ONC and
      OrcaSound without these caveats. Mark OrcaSound as needing-work and call out
      MBARI as the verified path.

## P4 — Open-source / licensing hygiene

The org decision (2026-05-15) is that the six `planetar-*` repos ship under
**AGPL-3.0**. planetar-acoustic does not yet match:

- [ ] **`pyproject.toml` says `license = { text = "MIT" }`** — change to AGPL-3.0
      (`license = { text = "AGPL-3.0-or-later" }` or the SPDX form).
- [ ] **Add a `LICENSE` file** — the full AGPL-3.0 text. The repo has none.
- [ ] Confirm consistency with the sibling repos (broker, ui, ais, sat, eo) so all
      six carry the same license.
- [ ] Run `gitleaks` over the repo before relying on it as a public artifact (the
      other public repos were scanned clean).

## P5 — Later / optional (post-submission or if time allows)

- [ ] **VTUAD dataset** for training the SAI→CV classifier — ~49 h of ONC
      Strait-of-Georgia clips, AIS-labeled into Cargo/Tanker/Tug/Passenger/Background.
      Same water as the proposal, ground-truth labels. Needs an IEEE DataPort
      subscription: <https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data>.
- [ ] **Historic corpora** (DeepShip, ShipsEar, NOAA SanctSound) — replay via
      `archive:`; useful for the classifier, not needed for the ingest demo.
- [ ] **Real CV classifier** — the current classifier is a hash-deterministic mock
      (`model_id="mock"`). A real model plugs in via `--model path.pt|.onnx`. Only
      needed if the demo is supposed to show vessel *type*, not just presence.

---

## TL;DR for the next demo

1. `aws s3api get-object` an MBARI slice → `stream --source archive:` → done. No account.
2. Register an ONC account so you can also demo on Salish Sea water.
3. Don't rely on OrcaSound until the live URL is fixed.
4. Fix the license (MIT → AGPL-3.0) before anyone audits the public repo.
