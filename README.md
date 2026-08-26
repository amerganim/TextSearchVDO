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

**Phase 2 (working):** zones and identity. A door line drawn once turns a
trajectory into a crossing with a direction and a timestamp; naming a person
once lets matching name them everywhere else. *When did Rafi cross the front
door* is now a query.

**Phase 3 (working):** search by text. CLIP embeddings over frames and object
crops, a word index over everything the earlier phases learned, and structured
filters, fused into one ranked answer.

**Phase 4 (in progress):** answers, not just results. *When did Rafi go out
the front door yesterday* returns two timestamps and the clips to play, rather
than forty segments to scroll. Captioning and audio are not built yet.

## The app

Double-click **TextSearchVDO.bat**, or:

```bash
.venv/Scripts/python -m tsv app
```

A native window opens. Drop a video on it, wait while it reads it, then type
what you are looking for. Results come back as frames with times; clicking one
plays from just before that moment. If nothing matches, it says so rather than
handing back the closest three frames in the library.

The window is the same server the CLI runs, in a WebView2 frame. That keeps
one implementation rather than two — the window, a plain browser and the
planned Android client all talk to the same endpoints. What the window adds is
a native file dialog that returns real paths, so **a video is indexed where it
already lives** rather than being uploaded to the machine it is already on. A
browser can still be used (`python -m tsv serve`); there it uploads, because
there is no path to point at.

The full timeline, zone editor and people tools are behind **Advanced** in the
corner, or at `/advanced`.

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
Stage 2  Identity (face) and user-drawn zones       <- done
Stage 3  CLIP embeddings + hybrid retrieval         <- done
Stage 4  Question answering                         <- done
         VLM captions, audio ASR                    <- not started
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

The baseline is rolling, not global: a camera's noise floor changes hugely
across a recording (IR cutover, rain, auto-exposure), and one global baseline
lets the noisier half hide its own activity behind a threshold set by the
quieter half. It is also taken at a **low quantile rather than the median**,
because motion only ever pushes bins upward — on a short file where someone is
on screen half the time, the median sits inside the activity and hides exactly
what raised it. Plenty of NVRs write one-minute files, so that case is normal
rather than exotic.

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

Measured on the baseline machine (i5-1235U, Intel UHD graphics), yolo11n at
640px, one frame at a time:

| backend | per frame | |
|---|---:|---|
| `openvino:GPU` | 33 ms | the iGPU is worth having |
| `onnxruntime:CPU` | 45 ms | |
| `openvino:CPU` | 53 ms | |

The best backend turns out to be a property of the **model**, not just the
machine. On a small graph the iGPU spends longer moving data and launching
kernels than computing, and loses outright:

| model | iGPU | CPU | |
|---|---:|---:|---|
| yolo11n @640 | 33 ms | 45 ms | iGPU wins |
| SCRFD det_500m @640 | 54 ms | 24 ms | CPU wins |
| ArcFace mbf @112 | 24 ms | 9 ms | CPU wins |
| CLIP image @224 | 82 ms | 51 ms | CPU wins |
| CLIP text @77 tok | 34 ms | 36 ms | a wash |

It is not about model size — CLIP's image encoder is the largest graph here
and still loses on the iGPU. Only a sustained, convolution-heavy workload
keeps that GPU busy enough to pay back the cost of getting data to it.
Compile time compounds it: the iGPU takes 4–6 seconds to compile a graph the
CPU loads in half a second, which a user feels directly on their first search.

So the detector runs on the iGPU while the face and CLIP stacks run on the
CPU, in the same pass, chosen per model.

Two things came out of measuring rather than assuming. OpenVINO's CPU path
*loses* to ONNX Runtime's, so it sits last in the preference order rather than
ahead of it on the strength of being Intel's own runtime. And OpenVINO must be
given `PERFORMANCE_HINT: LATENCY` — the throughput hint optimises for
concurrent streams, and this pipeline submits one frame and blocks, which
measured 8× slower per frame.

`openvino` is an optional dependency; without it everything falls back to
ONNX Runtime on the CPU.

**Tracking is ByteTrack** with a constant-velocity Kalman filter. The filter
matters more here than in a typical benchmark: frames are sampled a few per
second rather than at 25 fps, so a walking person can move most of their own
width between samples and raw IoU between consecutive detections collapses.
Association runs per class, and the second pass over low-confidence detections
is what keeps someone walking behind furniture as one tracklet instead of
three.

## Stage 2: zones and identity

**Zones.** A *line* counts crossings and their direction; a *region* counts
entries and how long someone stayed. Two geometry decisions carry the feature:
the anchor is the **bottom centre** of a box rather than its centre, because a
person is where their feet are and the centre puts someone "in the kitchen"
while they are still leaning through the doorway; and crossings are tested on
the **segment between consecutive samples** rather than by watching which side
a point is on, because at a few frames a second there is usually no sample on
the line at all.

Events are derived from stored detections and **never touch video**. Zones get
redrawn constantly, and moving a door line two pixels must not mean re-running
the detector over a week of footage.

**Identity.** SCRFD finds faces, ArcFace embeds them, and naming one tracklet
puts its vector in a gallery that names the rest. Three rules keep it from
confidently mislabelling people:

- a **margin**, not just a threshold - two similar people both scoring just
  over the line is exactly when to say "I don't know";
- **best score per identity**, so someone with twenty enrolled examples cannot
  out-rank a better match by having more chances;
- matching **never feeds its own output back** into the gallery, and never
  overwrites a manual label.

Faces are embedded from the best few crops of each tracklet, not every frame:
the face stack would otherwise cost more than the detector that found the
person. Alignment is not optional - measured on a real photograph, the *same*
face scores 0.99 against itself when warped onto ArcFace's landmark template
and only 0.68 from a plain box crop, while two different people score 0.00.

## Stage 3: search

Three signals, kept separate because they fail in different places.

**Semantic** (CLIP) finds what *looks* like the query - clothing, posture,
scene. It cannot tell you a person's name. **Lexical** (FTS5) finds what is
*named* like the query - an object class, an enrolled person, a drawn zone -
exact where it applies and silent where it does not. **Structured** filters
(a person, a zone, a day, a camera) are not ranked at all: they are
constraints, and that is what makes *when did Rafi go out the front door*
exact rather than merely likely.

The two ranked signals are fused with reciprocal rank fusion rather than by
adding scores. A cosine similarity and a BM25 score share no scale, and
normalising them needs recalibrating whenever either model changes; RRF reads
only the ordering.

Both scene frames and object crops are embedded. A person in a red jacket is a
property of the crop, not of the whole frame, so searching scene vectors alone
misses exactly the queries this is for.

The tokenizer is pure Python rather than a dependency, and its equivalence is
proven rather than assumed: the export dumps the reference vocabulary and
reference token ids, and the tests assert an exact match on both.

A query that matches nothing returns nothing. It never falls back to browsing,
which would answer a different question than the one asked.

```bash
python -m tsv search "a person carrying a box" --limit 10
```

```bash
python -m tsv search --who Rafi --zone "front door"
```

### Where the similarity floor lives

The library has **no** default similarity floor. Absolute CLIP scores are not
calibrated, and a caller that can see the scores should choose its own, so
`--min-similarity` exists and the score is always reported.

The **app** does apply one (`ClipConfig.min_similarity`, currently `0.20`),
because it cannot afford not to: its whole promise is an answer or an honest
nothing, and a search box that returns the closest three frames for *a snowy
mountain* reads as broken. The number is shown to the user as a percentage on
every result and is one line to change. It comes from measurement but on few
samples — a matching query scored 0.22–0.27 and an unrelated one 0.15 — so
expect to move it once real footage has been through it. Queries do slip
through: *"a helicopter landing"* scores 0.22 against a person walking.

## Stage 4: asking questions

Search ranks moments; this answers questions.

The parser is **entity grounded, not general language understanding**, and
that is what lets it work without a model. The vocabulary is closed and
already in the database: the people who were enrolled, the zones that were
drawn, the object classes the detector knows, the cameras that exist. Matching
against that is reliable in a way that parsing arbitrary English is not.

Whatever is left after the known entities are lifted out is not discarded — it
becomes the CLIP query, which handles the open-vocabulary half ("in a red
jacket"). So everything the index knows *exactly* is answered exactly, and
only the genuinely fuzzy remainder is left to a similarity score.

```bash
python -m tsv ask "when did Rafi go out the front door yesterday"
```

```
  understood: identity=Rafi, zone=front door, event=go out

  2 times, from Wed 01 Apr, 09:00:07 to 09:00:19.
    Wed 01 Apr 09:00:07  Rafi (cross out) at front door
    Wed 01 Apr 09:00:19  Rafi (cross out) at front door
```

It understands *when / who / how many / how long / did*, relative days
("yesterday", "last night", "on wednesday"), times of day, directions through
a zone, object classes, and people.

**A question about someone it has never heard of is refused, not answered.**
This is the one failure that would make the feature untrustworthy: ask about
someone who was never enrolled and the zone and direction still match, so a
naive implementation hands back *somebody else's* movements with a confident
timestamp. Instead it says who it does know, and offers ranked matches
separately.

## Usage

```bash
python -m tsv ingest /path/to/footage
```

```bash
python -m tsv analyze
```

```bash
python -m tsv zones add --camera ch01 --name "front door" --kind line --points 0.5,0.0 0.5,1.0
```

```bash
python -m tsv people name --tracklet 1 --name "Rafi" && python -m tsv people assign
```

```bash
python -m tsv search "someone at the front door" --reindex
```

```bash
python -m tsv ask "how many times did Rafi go out yesterday"
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

For semantic search, from the same environment:

```bash
.venv-export/Scripts/python -m pip install transformers && .venv-export/Scripts/python tools/export_clip.py --out data/models
```

That writes both CLIP encoders (~600 MB as fp32; quantising is an obvious
future win), the BPE merge table so the runtime can tokenise without
transformers, and the reference vocabulary and token ids the tests check
against.

For face recognition, from the same environment:

```bash
.venv-export/Scripts/python -m pip install insightface && .venv-export/Scripts/python tools/fetch_face_models.py --out data/models
```

That fetches SCRFD and ArcFace as plain ONNX (~16 MB for the small pack).
insightface is used only to download and unpack; nothing imports it at
runtime. Face models are optional - everything through Phase 1 works without
them.

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

**The synthetic clips validate Phase 0, not Phase 1.** Their moving subject is
a rectangle, so a real detector correctly finds nothing in them — running
`analyze` over synthetic footage reports zero tracklets, and that is the right
answer rather than a failure. Phase 1's coordinate maths, association and
bookkeeping are covered by unit tests and by an end-to-end run against a
threshold-based stand-in detector; whether detection is any *good* is a
question only real footage can answer.

`tests/test_model_integration.py` checks the exported graph against what the
decoder assumes — static input shape, raw predictions rather than baked-in
NMS, 4 + 80 attributes — and skips entirely when no model has been exported.

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
5. **Noisy scenes weaken Tier A.** The packet-size signal is *relative*, so
   heavy sensor noise shrinks it. On a deliberately noisy test clip the gap
   between idle and active fell to about 3 sigma with the two populations
   touching - no threshold could have separated them. The baseline is taken at
   a low quantile rather than the median for the related reason that motion
   only ever pushes bins upward, so on a short file where someone is on screen
   half the time the median sits inside the activity and hides it.
6. **Identity thresholds.** `DEFAULT_THRESHOLDS` in `identity.py` are guesses.
   On clean frontal faces the separation is enormous (0.998 against 0.034 in
   testing), but CCTV faces are small, angled and often lit by IR, and that
   margin will shrink.
7. **Search relevance.** CLIP similarity is not calibrated in absolute terms:
   in testing a matching query scored 0.22–0.27 and an unrelated one 0.15, but
   that gap moves with the camera and with how the query is phrased. Pick
   `--min-similarity` from real numbers.
8. **Question phrasing.** The parser knows a fixed list of ways to say "went
   out" and "came in". It covers the obvious ones, but it is a list, not an
   understanding of English — expect to add phrasings that real use throws up.
9. **Detection thresholds and sample rate.** `DetectConfig.detect_fps` trades
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
tsv/models/face.py       SCRFD + ArcFace, alignment
tsv/zones.py             zone geometry, crossings, dwell
tsv/events.py            event derivation, no video access
tsv/identity.py          gallery, matching, enrolment
tsv/models/clip.py       CLIP image/text encoders
tsv/models/tokenizer.py  CLIP byte-level BPE, pure Python
tsv/search.py            filters, lexical, semantic, rank fusion
tsv/query.py             question parsing and answering
tsv/importer.py          one call: ingest, analyse, index
tsv/jobs.py              background work with progress
tsv/desktop.py           the native window
web/app.html             the simple app
web/index.html           the advanced timeline UI
tsv/api.py               FastAPI: timeline, segments, objects, ranged media
tsv/config.py            every tunable, grouped by stage
web/                     timeline UI
```

The UI is a web app on localhost rather than a native window on purpose: the
planned Android companion becomes a thin client against these same endpoints
instead of a second implementation.
