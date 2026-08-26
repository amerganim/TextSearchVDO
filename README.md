# TextSearchVDO

Local-first search over long CCTV / IP-camera footage. The goal is to ask
plain-language questions — *when did my son go outside?* — and get back the
timestamp and the frame.

Everything runs on the user's own machine. No footage leaves the device.

**Phase 0 (this repo, working):** turn a folder of recordings into a scrubable
timeline where the small fraction of the day that contains activity is
obvious. No models involved. On the two synthetic test clips it discards ~75%
of the recording, and it runs at **35–140× realtime on a CPU/iGPU laptop with
no discrete GPU** — roughly 10 minutes to trawl a 24-hour day.

## Architecture

The design is **index → retrieve → verify**, not summarise-then-search. A
summary of 24h of hallway footage will never contain "father took his
medicine at 08:47" unless something forced a model to look at that moment and
write it down. Summaries are a by-product (a daily digest), never the search
substrate.

Cheap stages gate expensive ones. CCTV is 90–97% static, so nothing expensive
should ever see a frame a cheap filter could have discarded:

```
Stage 0  Motion segmentation              <- Phase 0, this repo
Stage 1  Detection + tracking (YOLO/ByteTrack)
Stage 2  Identity (face + re-ID) and user-drawn zones
Stage 3  Frame embeddings, VLM captions on person crops, audio ASR
Stage 4  Hybrid retrieval + local LLM answers with timestamps
```

Stage 0 itself is two tiers:

**Tier A — packet scan, no decoding.** In an inter-frame codec a P-frame
encodes only what changed, so packet size tracks motion. Demuxing is bounded
by disk speed, so this triages a whole night cheaply. Two encoder artefacts
have to be removed first, and both were found the hard way:

- *GOP sawtooth.* A P-frame's cost depends on its distance from the last
  keyframe. With a fixed GOP this puts a strong periodic swing in raw sizes —
  larger than a walking person. Each packet is scored against the median size
  **for its own position in the GOP**, which flattens it whatever the GOP
  length is.
- *Local all-intra stretches.* Encoders fall back to emitting nothing but
  keyframes when a scene gets noisy enough to trip scene-cut detection on
  every frame — heavy IR noise does this, and so do spliced NVR exports. Such
  a stretch has no inter-frame signal at all, and scoring it reads as "idle",
  silently losing whatever happened there. All-intra is detected **per bin**,
  not per file, and blind bins are passed to Tier B untriaged.

The baseline is a rolling median/MAD, not a global one: a camera's noise floor
changes hugely across a recording (IR cutover, rain, auto-exposure), and one
global baseline lets the noisier half hide its own activity behind a threshold
set by the quieter half.

Tier A is tuned for **recall over precision** — a false candidate costs decode
time, a miss is unrecoverable.

**Tier B — decode and subtract.** Only Tier A's candidates, at ~3 fps and 320px
wide. MOG2 background subtraction, shadows excluded, morphological opening and
a minimum blob area to kill sensor speckle, and explicit IR day/night flip
detection (a whole-frame luminance step is not motion — the model is rebuilt
rather than allowed to treat the new illumination as a moving object).

Segments then come from hysteresis + gap merging + pre/post roll, so one
person walking through is one event rather than twenty.

## Usage

```bash
python -m tsv ingest /path/to/footage
```

```bash
python -m tsv serve
```

Then open http://127.0.0.1:8000. `python -m tsv stats` prints what is indexed.

Ingest is idempotent — re-running over a growing NVR export directory only
processes what changed.

## Setup

```bash
py -3.14 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
```

PyAV bundles ffmpeg, so no separate ffmpeg install is needed. Verified on
Python 3.14 (av 18, OpenCV 5, numpy 2.5).

## Tests

```bash
.venv/Scripts/python -m pytest
```

`tools/make_synthetic.py` generates CCTV-like clips with known activity
windows and writes a `.truth.json` beside each, which is what lets the
pipeline be tested for correctness on a machine with no cameras attached. The
scene deliberately includes sensor noise and a textured background — on a
clean synthetic scene the encoder compresses idle frames to almost nothing and
Tier A looks far better than it will in reality.

```bash
.venv/Scripts/python tools/make_synthetic.py --out footage/ch01_20260826120000.mp4 --duration 120 --activity "18-26,55-61,95-104"
```

## Before trusting this on real footage

Every threshold in `tsv/config.py` is a guess until it has run against real
recordings. The synthetic clips prove correctness, not calibration. Expect to
tune, in roughly this order:

1. **Night/IR scenes.** In testing, a dark subject on a dark IR background
   scored ~10× lower than the same subject in daylight — close enough to
   `SegmentConfig.open_frac` to matter. A per-camera or per-illumination
   threshold is likely needed.
2. **Rain, snow, foliage, headlights.** All produce sustained real motion.
   `min_blob_area_frac` is the first knob.
3. **Timestamp accuracy.** `probe.py` reads the recorder's filename first and
   container metadata second, because NVRs rewrite `creation_time` on export.
   If neither is present it falls back to mtime and says so — check
   `videos.ts_source` before trusting a timeline.
4. **Clock sync across cameras**, before it silently corrupts cross-camera
   reasoning in later phases.

Build a gold set of 50–100 real queries early. Without one there is no way to
tell whether a threshold change helped.

## Layout

```
tsv/motion/packets.py    Tier A - packet-size triage
tsv/motion/decode.py     Tier B - MOG2 scoring, IR flip handling
tsv/motion/segments.py   activity trace -> segments
tsv/probe.py             container metadata + recording-time recovery
tsv/ingest.py            orchestration, idempotency
tsv/api.py               FastAPI: timeline, segments, thumbnails, ranged media
tsv/config.py            every tunable, grouped by stage
web/                     timeline UI
```

The UI is a web app on localhost rather than a native window on purpose: the
planned Android companion becomes a thin client against these same endpoints
instead of a second implementation.
