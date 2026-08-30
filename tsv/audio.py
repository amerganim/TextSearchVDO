"""What the recording heard.

Every answer this app gives so far comes from pixels. Audio is a different
signal, not a better version of the same one, and that is why it earns its
place: a doorbell, a knock, breaking glass or a smoke alarm has no visual
signature at all, and a camera pointed at the hallway still hears the front
door. Speech carries the rest - names, and the medicine case in particular,
where *did you take your tablets* is said far more often than it is legible
from a crop.

**No torch.** faster-whisper runs on ctranslate2, which is the only reason
this is acceptable here: the runtime stays ONNX Runtime, numpy and now one
more native library, rather than growing a two-gigabyte tensor framework.

**What it costs.** Measured on the baseline machine (i5-1235U, whisper-base,
int8, CPU) over 500 seconds of audio, with voice-activity detection off so
the whole file is genuinely decoded:

    tiny    25.0x realtime     2.4 minutes per hour of speech
    base    17.8x realtime     3.4 minutes per hour
    small    8.0x realtime     7.5 minutes per hour

With the gate on and real footage, which is mostly silence, base ran at 112x.

Threads barely matter: 2 gave 19.4x and all 12 gave 20.6x - six times the
cores for six percent. The decoder is autoregressive and this is not a
core-bound workload, so a larger CPU is the wrong thing to spend on. The
headroom is a discrete GPU, which `pick_device` takes when there is one.

**Gated the same way video is.** Stage 0 refuses to decode video where the
packet sizes say nothing moved; this refuses to transcribe where there is no
voice. That is not an optimisation, it is what keeps invented text out of the
index - and it was worth confirming rather than assuming.

The footage this was built against carries audio streams that are digitally
silent: peak amplitude 0.0000, around -123 dBFS across half an hour. Run with
voice-activity detection on, Whisper returns nothing from them, correctly.
Run with it off, it returns:

    480.0-482.0  logprob -0.73  no_speech 0.69   "Hey!"
    482.0-484.0  logprob -0.73  no_speech 0.69   "Hey!"
    486.0-488.0  logprob -0.73  no_speech 0.69   "Hey!"

Five identical lines conjured out of silence, which without the second filter
below would be indexed as something somebody said. Both gates are load-bearing
and both earn their thresholds from that: the model's own no-speech estimate
already flagged those at 0.69, above the 0.6 ceiling here.

**Where it lands.** Transcripts become part of a segment's document in the
word index, next to object labels, names, zones and captions. So they are
searchable through the machinery that already exists, and a spoken word ranks
by the same fusion as everything else rather than through a parallel path.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Whisper wants 16 kHz mono. Cameras record whatever they record - the sample
# footage here is 8 kHz - so resampling is not optional.
SAMPLE_RATE = 16_000

# Below this the model is guessing. Whisper reports a per-segment average log
# probability, and the low end of it is reliably where "thank you for
# watching" and other training-set residue appears; dropping those matters
# more here than recall, because a hallucinated line is indexed as fact.
MIN_LOGPROB = -1.0

# A segment that is mostly silence, by Whisper's own estimate. Same reasoning.
MAX_NO_SPEECH = 0.6


@dataclass
class Utterance:
    """One stretch of speech, in seconds from the start of the video."""

    t_start: float
    t_end: float
    text: str
    logprob: float
    no_speech: float


@dataclass
class Heard:
    """What came back from one file, and enough to explain a silent result.

    A bare list of utterances cannot distinguish the three ways of hearing
    nothing, and they mean completely different things to somebody deciding
    whether the feature is working: a recording with no microphone, a
    recording where nobody spoke, and a recording full of speech that the
    model could not hold well enough to be worth indexing. Reported as one
    empty list, all three look like the app is broken.
    """

    utterances: list["Utterance"] = field(default_factory=list)
    # Segments the confidence floor rejected. The difference between "nobody
    # spoke" and "somebody spoke and this model cannot follow them", which for
    # Bengali here is the difference between a wrong answer and a fixable one.
    discarded: int = 0
    language: str = ""
    language_probability: float = 0.0

    @property
    def unclear(self) -> bool:
        """Speech was there; none of it survived."""
        return not self.utterances and self.discarded > 0


@dataclass
class TranscribeSummary:
    videos: int = 0
    utterances: int = 0
    seconds_of_speech: float = 0.0
    skipped_silent: int = 0
    skipped_no_audio: int = 0
    # Had speech in it, and none of it was clear enough to keep. Separate from
    # silent because the remedy is different: silence is nothing to fix, this
    # is a model that cannot hold the language being spoken.
    skipped_unclear: int = 0
    failed: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    samples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "videos": self.videos,
            "utterances": self.utterances,
            "seconds_of_speech": round(self.seconds_of_speech, 1),
            "skipped_silent": self.skipped_silent,
            "skipped_no_audio": self.skipped_no_audio,
            "skipped_unclear": self.skipped_unclear,
            "failed": self.failed,
            "elapsed": round(self.elapsed, 1),
            "samples": self.samples[:5],
        }


def has_audio(path: Path) -> bool:
    """Whether the file carries an audio stream at all.

    Most IP cameras do not record sound, and several record a silent stream
    rather than none. Checking costs a demux and saves loading a model.
    """
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:  # noqa: BLE001 - an unreadable file is not audio
        return False


def pick_device(preference: str = "auto") -> tuple[str, str]:
    """(device, compute type) for this machine.

    ctranslate2 ships with CUDA compiled in, so the question is whether there
    is a card to use rather than whether the library can. On one there is real
    headroom - float16 on a discrete GPU is several times what an int8 CPU
    path manages - and hard-coding CPU meant a machine with a 4090 in it
    transcribed no faster than a laptop.

    Same rule as everywhere else here: what is reported has to be what was
    tried. `get_cuda_device_count` counts visible devices, and a count without
    a working driver is still a failure at model load, so the caller falls
    back rather than trusting this.
    """
    if preference == "cpu":
        return "cpu", "int8"
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:      # noqa: BLE001 - no CUDA is an answer, not an error
        pass
    return "cpu", "int8"


def load_model(cfg):
    """The transcription model, or None when it is not installed.

    Absent is a normal state, like every other optional model here: the app
    keeps working and says which part is missing rather than failing at the
    moment somebody searches for a spoken word.
    """
    directory = cfg.audio_model_dir
    if not directory.is_dir():
        return None
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    device, compute = pick_device(cfg.audio.device)
    if cfg.audio.compute_type:
        compute = cfg.audio.compute_type

    try:
        return WhisperModel(
            str(directory), device=device, compute_type=compute,
            cpu_threads=cfg.audio.threads or 0,
        )
    except Exception:      # noqa: BLE001 - a listed GPU that will not load
        if device == "cpu":
            raise
        # The card was counted but the model would not compile on it. CPU
        # always works, and a slower transcript beats none.
        return WhisperModel(
            str(directory), device="cpu", compute_type="int8",
            cpu_threads=cfg.audio.threads or 0,
        )


def transcribe_file(
    model,
    path: Path,
    language: str | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Heard:
    """Speech in one file, as timed utterances.

    Voice activity detection does the gating, so silence costs a fraction of
    what transcribing it would. What comes back is filtered again on the
    model's own confidence, because the failure mode of an unfiltered
    transcript is not a gap - it is a plausible sentence that was never said.
    """
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 700},
        condition_on_previous_text=False,   # stops one hallucination seeding more
        beam_size=1,
    )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    heard = Heard(
        language=str(getattr(info, "language", "") or ""),
        language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
    )
    out: list[Utterance] = heard.utterances
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue
        logprob = float(getattr(segment, "avg_logprob", 0.0) or 0.0)
        no_speech = float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
        if logprob < MIN_LOGPROB or no_speech > MAX_NO_SPEECH:
            # Counted rather than forgotten: this is the only evidence that
            # speech was present at all when nothing survives.
            heard.discarded += 1
            continue
        out.append(Utterance(
            t_start=float(segment.start), t_end=float(segment.end),
            text=text, logprob=logprob, no_speech=no_speech,
        ))
        if on_progress and duration:
            on_progress(min(float(segment.end) / duration, 1.0))
    return heard


def store_utterances(
    conn: sqlite3.Connection, video_id: int, utterances: list[Utterance]
) -> int:
    """Replace this video's transcript, attaching each line to a segment.

    An utterance belongs to whichever segment covers the moment it was said.
    Speech during a stretch the motion pass discarded has no segment to hang
    on, and is stored with none rather than dropped: it is still true that
    something was said then, and the timeline can show it even though nothing
    moved.
    """
    conn.execute("DELETE FROM utterances WHERE video_id = ?", (video_id,))
    row = conn.execute(
        "SELECT start_ts FROM videos WHERE id = ?", (video_id,)
    ).fetchone()
    if row is None:
        return 0
    start_ts = float(row["start_ts"])

    stored = 0
    for utterance in utterances:
        segment = conn.execute(
            """SELECT id FROM segments
               WHERE video_id = ? AND t_start <= ? AND t_end >= ?
               ORDER BY t_start LIMIT 1""",
            (video_id, utterance.t_end, utterance.t_start),
        ).fetchone()
        conn.execute(
            """INSERT INTO utterances(video_id, segment_id, t_start, t_end,
                                      ts_start, ts_end, text, confidence)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                video_id,
                int(segment["id"]) if segment else None,
                utterance.t_start, utterance.t_end,
                start_ts + utterance.t_start, start_ts + utterance.t_end,
                utterance.text, utterance.logprob,
            ),
        )
        stored += 1
    conn.commit()
    return stored


def pending_videos(conn: sqlite3.Connection, force: bool = False) -> list[sqlite3.Row]:
    """Videos with no transcript yet."""
    where = "" if force else """
        WHERE v.id NOT IN (SELECT DISTINCT video_id FROM utterances)
          AND (v.transcribed_at IS NULL)"""
    return conn.execute(
        f"SELECT v.id, v.path, v.duration FROM videos v {where} ORDER BY v.start_ts"
    ).fetchall()


def transcribe_videos(
    conn: sqlite3.Connection,
    cfg,
    force: bool = False,
    limit: int | None = None,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> TranscribeSummary:
    """Transcribe whatever has not been done, and index it for search."""
    started = time.time()
    summary = TranscribeSummary()

    rows = pending_videos(conn, force=force)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        summary.elapsed = time.time() - started
        return summary

    model = load_model(cfg)
    if model is None:
        summary.failed.append(
            "no transcription model. Fetch it with: python -m tsv setup --only audio"
        )
        summary.elapsed = time.time() - started
        return summary

    for index, row in enumerate(rows):
        path = Path(row["path"])
        if not path.is_file():
            summary.failed.append(f"{path.name}: file has moved")
            continue
        if not has_audio(path):
            summary.skipped_no_audio += 1
            conn.execute(
                "UPDATE videos SET transcribed_at = ? WHERE id = ?",
                (time.time(), int(row["id"])),
            )
            continue

        try:
            heard = transcribe_file(
                model, path, language=cfg.audio.language,
                on_progress=(
                    lambda frac, i=index: on_progress(i, len(rows), frac)
                    if on_progress else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            summary.failed.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue

        utterances = heard.utterances
        store_utterances(conn, int(row["id"]), utterances)
        conn.execute(
            "UPDATE videos SET transcribed_at = ? WHERE id = ?",
            (time.time(), int(row["id"])),
        )
        conn.commit()

        summary.videos += 1
        summary.utterances += len(utterances)
        summary.seconds_of_speech += sum(u.t_end - u.t_start for u in utterances)
        if heard.unclear:
            summary.skipped_unclear += 1
        elif not utterances:
            summary.skipped_silent += 1
        for utterance in utterances[:2]:
            if len(summary.samples) < 5:
                summary.samples.append(utterance.text)

    summary.elapsed = time.time() - started
    return summary
