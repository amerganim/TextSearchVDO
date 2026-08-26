# TextSearchVDO

Local-first search over long CCTV / IP-camera footage. The goal is to ask
plain-language questions — *when did my son go outside?* — and get back the
timestamp and the frame.

Everything runs on the user's own machine. No footage leaves the device.

**Phase 0 (working):** turn a folder of recordings into a scrubable timeline
where the small fraction of the day that contains activity is obvious. No
models involved. On the synthetic test clips it discards ~75% of the
recording, and runs at **35–140× realtime on a CPU/iGPU laptop with no
discrete GPU** — roughly 10 minutes to trawl a 24-hour day.

**Phase 1 (working):** detect and track objects inside those segments, so the
timeline can say *two people and a dog* rather than *something moved*. Runs
only over what Phase 0 kept.

## Architecture

The design is **index → retrieve → verify**, not summarise-then-search. A
summary of 24h of hallway footage will never contain "father took his
medicine at 08:47" unless something forced a model to look at that moment and
write it down. Summaries are a by-product (a daily digest), never the search
substrate.

Cheap stages gate expensive ones. CCTV is 90–97% static, so nothing expensive
should ever see a frame a cheap filter could have discarded:

```
Stage 0  Motion segmentation                        <- done
Stage 1  Detection + tracking (YOLO / ByteTrack)    <- done
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

## Stage 1: detection and tracking

Runs only over the segments Phase 0 found, which is the point of the cascade —
the detector never sees the ~73% of a recording that held nothing.

**Runtime is not torch.** Neither ultralytics nor torch is a runtime
dependency; at inference this is an ONNX graph plus numpy.
`tools/export_model.py` uses them once, from a throwaway environment, to
produce the ONNX file. The backend abstraction sits over *runtimes* (ONNX
Runtime, OpenVINO) rather than ONNX Runtime execution providers, because
`onnxruntime-openvino` has no wheel for Python 3.14 — the Intel iGPU path has
to go through OpenVINO directly. Which backend was chosen is reported, never
silently decided.

**Tracking is ByteTrack** with a constant-velocity Kalman filter. The filter
matters more here than in a typical benchmark: frames are sampled a few per
second rather than at 25 fps, so a walking person can move most of their own
width between samples and raw IoU between consecutive detections collapses.
Association runs per class, and the second pass over low-confidence detections
is what keeps someone walking behind furniture as one tracklet instead of
three.

## Usage

```bash
python -m tsv ingest /path/to/footage
```

```bash
python -m tsv analyze
```

```bash
python -m tsv serve
```

Then open http://127.0.0.1:8000. `python -m tsv stats` prints what is indexed,
and `python -m tsv bench` times the detector on every backend that will load
it.

Both `ingest` and `analyze` are idempotent — re-running over a growing NVR
export directory only processes what changed. Pass `--force` to redo work.

### Getting a detector

```bash
py -3.14 -m venv .venv-export && .venv-export/Scripts/python -m pip install ultralytics onnx onnxslim
```

```bash
.venv-export/Scripts/python tools/export_model.py --out data/models
```

The export venv can be deleted afterwards; only `data/models/yolo11n.onnx`
(~10 MB) is needed to run. Two export flags are not the default everywhere and
both matter: `nms=False`, because `tsv.models.detect` does its own class-aware
NMS and a graph with NMS baked in has a different output layout entirely; and
`dynamic=False`, because OpenVINO compiles static shapes far better.

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
5. **Detection thresholds and sample rate.** `DetectConfig.detect_fps` trades
   cost against tracking stability linearly, and `decode_width` decides whether
   distant figures survive at all. Both want real footage to settle.

Build a gold set of 50–100 real queries early. Without one there is no way to
tell whether a threshold change helped.

## Layout

```
tsv/motion/packets.py    Tier A - packet-size triage
tsv/motion/decode.py     Tier B - MOG2 scoring, IR flip handling
tsv/motion/segments.py   activity trace -> segments
tsv/frames.py            window sampling, shared by both analysis stages
tsv/probe.py             container metadata + recording-time recovery
tsv/ingest.py            Phase 0 orchestration, idempotency
tsv/models/backend.py    runtime selection (ONNX Runtime / OpenVINO)
tsv/models/detect.py     YOLO pre/post-processing, no torch
tsv/track/bytetrack.py   two-stage association
tsv/track/kalman.py      constant-velocity motion model
tsv/boxes.py             box geometry, IoU, class-aware NMS
tsv/analyze.py           Phase 1 orchestration
tsv/api.py               FastAPI: timeline, segments, objects, ranged media
tsv/config.py            every tunable, grouped by stage
web/                     timeline UI
```

The UI is a web app on localhost rather than a native window on purpose: the
planned Android companion becomes a thin client against these same endpoints
instead of a second implementation.
