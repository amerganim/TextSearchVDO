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
than forty segments to scroll. Optional Florence-2 captioning describes what
each person is doing, so *red bag* or *medicine bottle* become searchable
words. Audio transcription is not built yet.

## Getting started

Double-click **setup.bat** once, then **TextSearchVDO.vbs**. Or from a prompt:

```bash
py -3.14 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python -m tsv setup
```

`setup` fetches or builds all four models — object detection, faces, semantic
search and descriptions — skipping whatever is already there. It is the only
step that needs the internet, downloads roughly a gigabyte, and takes a few
minutes.

It builds a throwaway `.venv-export` to do it. That environment carries torch,
and it exists so the *running* application does not: some model sets have to be
exported rather than downloaded, because the CLIP ONNX on the hub is a single
fused graph rather than the separate encoders this runtime uses. Pass `--clean`
to delete it afterwards.

### Which detector, and why it decides whether you can sell this

```bash
.venv/Scripts/python -m tsv setup --detector yolox-tiny
```

Ultralytics YOLO11 is **AGPL-3.0**, whose obligations reach any application
distributed with it. YOLOX is Apache-2.0, publishes ONNX graphs directly — so
it needs no export step and no torch at all — and measured on real footage it
holds up: over one eight-minute recording YOLO11n found five people and a
spurious bird, YOLOX-tiny found seven people and no bird, twelve seconds
against fourteen. On a single frame both put the same person in the same box
to within a few pixels, at 0.918 and 0.880.

```bash
.venv/Scripts/python -m tsv hardware
```

reports what this machine can run, what is running now, and which models block
a commercial release.

### Hearing as well as seeing

```bash
.venv/Scripts/python -m tsv listen
```

Speech is transcribed with faster-whisper &mdash; ctranslate2, no torch &mdash;
and lands in the same word index as object labels, names and descriptions, so
a spoken word ranks by the same fusion as everything else. It runs as part of
every import once the model is installed.

Audio earns its place by being a *different* signal: a doorbell, a knock,
breaking glass or an alarm has no visual signature at all, and a camera
pointed at the hallway still hears the front door.

### Checking that it works

```bash
.venv/Scripts/python -m tsv listen --file "some video with talking.mp4"
```

One file, nothing indexed, nothing changed — the fastest way to see whether
transcription works on your machine and your audio. It prints the model, the
device it chose, how long it took, and every line it kept. If your recording
is silent it says so rather than inventing something.

### What it costs

Measured on a laptop (i5-1235U, int8, CPU), with the silence gate off so the
whole file is genuinely decoded — the honest wall-to-wall-speech figure:

| model | size | speed | an hour of speech |
|---|---:|---:|---:|
| tiny | 78 MB | 25.0× realtime | 2.4 min |
| **base** | 148 MB | 17.8× realtime | 3.4 min |
| small | 486 MB | 8.0× realtime | 7.5 min |

Real footage is mostly silence, and the gate skips it: base ran at **112×
realtime** over an actual recording.

**More CPU cores buy almost nothing here.** Two threads gave 19.4× and all
twelve gave 20.6× — six percent for six times the cores. Whisper's decoder is
autoregressive and this is not a core-bound workload. The hardware that does
change the answer is a discrete GPU, which the app now takes automatically
(`float16` on CUDA, `int8` on CPU) instead of the CPU it used to hard-code.

Two gates keep invented text out of the index, and both are load-bearing. The
footage this was built against has audio tracks that are **digitally silent**
&mdash; peak 0.0000, about &minus;123&nbsp;dBFS. With voice-activity detection
on, Whisper correctly returns nothing. With it off, it returns five identical
lines of `"Hey!"` conjured out of the silence, which without the second filter
would be indexed as something somebody said.

### The face models, and what measuring them showed

InsightFace's SCRFD + ArcFace are research-only. OpenCV's **YuNet (MIT)** and
**SFace (Apache-2.0)** replace them and install nothing new, since OpenCV is
already a dependency:

```bash
.venv/Scripts/python tools/fetch_face_models_permissive.py --out data/models
```

Detection came out a wash — 20.4% against 18.4% over 98 person crops, median
17–21 pixel faces either way — which is the right answer: the swap costs no
accuracy and removes the restriction.

Recognition could not be validated, and that finding is worth more than the
models. At the face sizes in this footage, same-person and different-person
pairs score **identically** (0.306 against 0.306); separation only appears
above about 30 pixels. Recognition wants roughly 112 pixels of face. So naming
people works on a camera that sees faces at head height and close range, and
will not work on a wide overview shot — whichever models are installed.

```bash
.venv/Scripts/python -m tsv setup --check
```

tells you what is present without changing anything.

## The app

Double-click **TextSearchVDO.bat**, or:

```bash
.venv/Scripts/python -m tsv app
```

Double-click **TextSearchVDO** — the shortcut with the blue play icon.
`setup.bat` makes it, and `python -m tsv shortcut --desktop --start-menu`
makes more. It carries an icon because the files it sits next to cannot:
Windows draws every `.vbs` with the same generic script glyph, so the
thing to double-click looked identical to the thing not to.

One warning for anyone editing the launchers: `.bat`, `.cmd` and `.vbs`
files must keep **CRLF** line endings. `cmd.exe` cannot find a label in a
file saved with plain LF — `goto :elevated` falls straight through and the
script does nothing while reporting nothing. `.gitattributes` pins them so
a clone cannot undo it.

It points at **TextSearchVDO.vbs**, not the `.bat`. Both start the same
app, but a `.bat` is run by `cmd.exe`, and `cmd.exe` shows a console window
before anything else happens &mdash; no amount of `pythonw` suppresses that
from inside a batch file. The `.vbs` has no console to begin with. The `.bat`
is kept for when something is wrong and you want to see the error.

A native window opens. Drop a video on it, wait while it reads it, then type
what you are looking for. Results come back as frames with times; clicking one
plays from just before that moment. If nothing matches, it says so rather than
handing back the closest three frames in the library.

**Videos** is the library. Everything indexed stays until you remove it, so
you can search across recordings rather than one at a time &mdash; but a
search covers the *newest* one by default, and a strip under the search box
says which. One click widens it to everything. That is deliberate: a
persistent archive is the point of a tool like this, and the fix for "why is
it finding my old videos" is to say what is in scope, not to throw the
library away every launch.

**Places** is where you draw a line across a doorway, or a shape around an
area. It matters more than it sounds: a direction is only measurable against
a drawn line, so until one exists *when did he go outside* has nothing to
compare a track to — and the app now says exactly that instead of returning an
empty screen. Saving one reports how many crossings it already found in the
videos you have, which is the honest test of whether it is in the right place.

**People** appears in the corner once a recording contains any, and carries
the number still unnamed. Naming one sighting adds its face to a gallery and
finds the same person everywhere else, which is what turns *a person went out*
into *Rafi went out* — so it is one name, not one name per appearance. Where
the face is too small or turned away, the app names that sighting and says
plainly that the others will not be found, rather than promising a match that
never arrives. Every name it worked out for itself carries **Not them**.

The window is the same server the CLI runs, in a WebView2 frame. That keeps
one implementation rather than two — the window, a plain browser and the
planned Android client all talk to the same endpoints. What the window adds is
a native file dialog that returns real paths, so **a video is indexed where it
already lives** rather than being uploaded to the machine it is already on. A
browser can still be used (`python -m tsv serve`); there it uploads, because
there is no path to point at.

**Advanced**, in the corner or at `/advanced`, is the operator view rather
than a better search: a 24-hour timeline per camera showing where the activity
was and — as important — where nothing was recorded at all, filters for
browsing by camera, day and object class, and a readout of what the question
parser actually grounded. It has a way back to the app.

### Sideways video

Phones write recordings whose frames are stored on their side, with no
rotation for a player to read. A detector trained on upright scenes finds
almost nothing in one — measured here, a clip went from 9 detections at
0.60 to 14 at 0.81 once turned, and only then contained the bed that made
*"anyone sleeping?"* answerable.

So the app works it out: the four right angles are tried on about six
frames and the one the detector is most confident about wins. An
alternative has to beat the stored orientation by a clear margin, because
turning a recording that was already upright is worse than leaving a
sideways one alone. Nothing to configure.

## Using it from a phone

```bash
.venv/Scripts/python -m tsv share
```

Click **Phone** in the app, then **Start**: a QR code to point a camera at,
the address, and a six-digit code. No terminal, which matters when the
computer is not yours. `python -m tsv share` does the same from a command
line.

On the phone, join the same WiFi — or plug it in and turn on USB
tethering — scan the code or open the address, then type the six digits.
Pairing lasts 30 days and survives restarts.

**Windows Firewall will block this until told not to**, and it does so
silently: the server binds, the address prints, the QR scans, and the
phone then sits there forever. Both `tsv share` and the Phone panel check
for it and say so.

The fix is one double-click on **allow-phone.bat**. It asks Windows for
administrator rights itself — say yes to the prompt — and adds a single
rule scoped to `LocalSubnet`, so the port is open to your own network
and nothing else. To undo it later:

```powershell
Remove-NetFirewallRule -DisplayName "TextSearchVDO"
```

The rule covers every profile, because a phone hotspot, a USB tether and
a home router are three different ones to Windows and it marks new ones
Public.

Sharing also warns when Windows has the network marked **Public** — which
is what it calls one you have not said you trust. A private *address* and
a trusted *network* are different questions, and only the second is about
who else can see the traffic.

Tapping a result plays a **clip** of that moment rather than the whole
recording: measured on real footage, 0.82 MB and 0.01 seconds against
124 MB. Nothing is re-encoded, so it costs about what reading the file
costs. *Whole recording* is one click away when what happened next is the
point.

Phones record **HEVC**, and phones play it, so the clip keeps whatever
the camera used. Desktop Firefox does not play HEVC — there the browser
says so and the clip comes back as H.264 instead, at 1.69 MB and 0.84
seconds. The browser asks rather than the server guessing from a user
agent, and it asks up front where it can, falling back once if playback
fails anyway (`canPlayType` says "probably" and means "possibly").
Asking for H.264 from a recording that already is H.264 costs nothing.

**No Android app is needed, and one would not have helped.** The page is
already responsive, already streams video by HTTP range request, and already
uploads through a file picker. It carries a web app manifest, so *Add to Home
Screen* gives it its own icon and a full-screen launch — on iOS as well as
Android, which a native Android app would not have covered. One codebase, no
store review.

Only one thing needs to be running. The pairing code is kept with the
index rather than in memory, so the desktop window and `tsv share` always
show the same one — and `tsv share` refuses to start on a port already in
use rather than lingering beside the real server printing a code nothing
is checking.

`python -m tsv devices` lists what has been paired; `--revoke <id>` removes
one, and it is refused on its very next request.

### Sending a video from the phone

Uploads are chunked and resumable, which matters more than it sounds. A phone
recording is gigabytes; at the ~12 MB/s a laptop manages over WiFi, four of
them is five minutes of holding one connection open, and everything a phone
does normally &mdash; locking the screen, switching apps, walking towards the
door &mdash; ends it. Under the old single-POST upload the reward was starting
again from zero.

Now only one chunk has to survive at a time, retried with backoff, and the
**server** says where to carry on. Picking the same video again after a
failure resumes from the byte it reached, including after a reload &mdash; the
partial data is found by name and size, which is all a phone still knows once
its page has gone. **Stop** in the progress strip pauses rather than cancels.

Two guards worth knowing about: an upload is refused up front if there is not
enough disk space, and refused at the end if fewer bytes arrived than were
promised. The second matters because a short file otherwise reaches the
demuxer and is reported as a corrupt recording, which sends somebody to look
at their camera instead of their WiFi.

### What sharing does and does not protect

Sharing is **off by default** and is a constructor argument rather than a
runtime toggle — whether strangers can read your footage is not the sort of
thing that should be flippable by a request.

With it on, a device that has not paired gets the pairing page and nothing
else: not the index, not a thumbnail, not the application's own JavaScript.
Loopback is exempt, because somebody sitting at the machine can open the
database in a text editor and a login would protect nothing.

What it does **not** do is encrypt. There is no TLS, because the alternative
is asking people to trust a self-signed certificate on a phone, which is
unpleasant enough that most would give up. Somebody already on your WiFi,
running the right tools, can see the traffic. **Your network is the security
boundary** — which is why `tsv share` refuses to start when it cannot find a
private address, and warns loudly if the machine has a public one.

## Architecture

**[How it works, step by step, with diagrams](docs/HOW-IT-WORKS.md)** — the
full pipeline, which model does what, where each one runs, and why question
parsing needs no LLM.

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
         VLM captions (Florence-2, optional)        <- done
         Audio transcription                        <- not started
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
python -m tsv caption
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

For captioning - describing what people are doing, so actions and objects
become searchable - from the same environment:

```bash
.venv-export/Scripts/python tools/fetch_caption_model.py --out data/models
```

### A bigger caption model, where the CPU can spare it

```bash
.venv-export/Scripts/python tools/fetch_caption_model.py --model large --out data/models
.venv/Scripts/python -m tsv caption --large --force
```

Florence-2-large (1.5 GB) names what base only gestures at. On the same
crop, base gave *"a mannequin standing on a chair"* where large gave *"a
woman standing on a set of stairs... a pink hat... holding a metal pole"*.
It costs about three times as long — roughly 15 seconds a crop against 5
— and this is already the slowest stage, so base stays the default. Both
install side by side and `--large` picks between them.

That fetches Florence-2-base-ft as four int8 ONNX graphs (~275 MB) plus its
task prompts, tokenised once so the runtime needs no tokenizer. Once it is
there, captioning runs as the last stage of every import. It costs about six
seconds per person on a CPU, essentially all of it in the vision encoder — but
a description that has not been written is a search that fails silently, and
nobody remembers to press a button before searching. Motion, objects and
search are all live long before it starts, and the progress bar weights it
honestly rather than appearing to stall at the end.

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

## Setup, in detail

`python -m tsv setup` runs everything below; these are the individual steps it
performs, for when one of them needs doing by hand.

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
