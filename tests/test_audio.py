"""Transcription: the gates, the storage, and where it lands in search.

The model itself is not tested here - that would be testing Whisper. What is
tested is everything around it, because that is where this can go wrong in a
way nobody notices: a hallucinated line stored as something somebody said is
indistinguishable, once indexed, from a true one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tsv import db
from tsv.audio import (
    MAX_NO_SPEECH, MIN_LOGPROB, Utterance, has_audio, load_model,
    pending_videos, store_utterances, transcribe_file, transcribe_videos,
)
from tsv.config import DEFAULT
from tsv.search import lexical_ranking, rebuild_text_index, segment_document


class StubSegment:
    def __init__(self, start, end, text, avg_logprob=-0.2, no_speech_prob=0.1):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob = avg_logprob, no_speech_prob


class StubInfo:
    duration = 100.0
    language = "en"


class StubModel:
    """Stands in for Whisper, returning whatever the test wants."""

    def __init__(self, segments):
        self._segments = segments
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        return iter(self._segments), StubInfo()


@pytest.fixture
def conn(tmp_path):
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    connection = db.open_db(cfg.db_path)
    connection.execute("INSERT INTO cameras(id, name) VALUES (1, 'ch01')")
    connection.execute(
        """INSERT INTO videos(id, camera_id, path, start_ts, ts_source, duration)
           VALUES (1, 1, 'a.mp4', 1000.0, 'test', 100.0)"""
    )
    connection.execute(
        """INSERT INTO segments(id, video_id, camera_id, t_start, t_end,
                                ts_start, ts_end, activity_score, peak_offset)
           VALUES (1, 1, 1, 10.0, 20.0, 1010.0, 1020.0, 0.5, 12.0)"""
    )
    connection.commit()
    return connection


# ---------- the gates ----------

def test_voice_activity_detection_is_always_on():
    """The gate that stops silence being transcribed at all.

    Not an optimisation. The footage this was built against has audio tracks
    that are digitally silent, and Whisper run over them without this returns
    five identical lines of "Hey!" - text conjured out of nothing, which would
    be indexed as something somebody said.
    """
    model = StubModel([])
    transcribe_file(model, Path("a.mp4"))
    assert model.calls[0]["vad_filter"] is True
    assert model.calls[0]["condition_on_previous_text"] is False


def test_low_confidence_lines_are_dropped():
    """The second gate, for what survives the first.

    A transcript's failure mode is not a gap, it is a fluent sentence that was
    never spoken. Whisper rates its own output and the low end is reliably
    where training-set residue appears.
    """
    model = StubModel([
        StubSegment(0, 2, "this one is fine"),
        StubSegment(2, 4, "invented", avg_logprob=MIN_LOGPROB - 0.1),
        StubSegment(4, 6, "mostly silence", no_speech_prob=MAX_NO_SPEECH + 0.1),
        StubSegment(6, 8, "   "),
    ])
    kept = transcribe_file(model, Path("a.mp4"))
    assert [u.text for u in kept] == ["this one is fine"]


def test_the_hallucination_this_was_measured_against_is_rejected():
    """The exact output silence produced, at the exact numbers it produced."""
    model = StubModel([
        StubSegment(480, 482, "Hey!", avg_logprob=-0.73, no_speech_prob=0.69),
        StubSegment(482, 484, "Hey!", avg_logprob=-0.73, no_speech_prob=0.69),
    ])
    assert transcribe_file(model, Path("a.mp4")) == []


# ---------- storage ----------

def test_an_utterance_is_attached_to_the_segment_it_falls_in(conn):
    stored = store_utterances(conn, 1, [Utterance(12.0, 14.0, "hello", -0.2, 0.1)])
    assert stored == 1
    row = conn.execute("SELECT segment_id, ts_start FROM utterances").fetchone()
    assert row["segment_id"] == 1
    assert row["ts_start"] == pytest.approx(1012.0)


def test_speech_outside_every_segment_is_kept_with_no_segment(conn):
    """Somebody can talk while nothing moves.

    The motion pass discards that stretch, so there is no segment to hang the
    line on - but it is still true that something was said then, and dropping
    it would lose exactly the case audio exists to catch.
    """
    store_utterances(conn, 1, [Utterance(80.0, 82.0, "who is there", -0.2, 0.1)])
    row = conn.execute("SELECT segment_id, text FROM utterances").fetchone()
    assert row["segment_id"] is None
    assert row["text"] == "who is there"


def test_re_transcribing_replaces_rather_than_duplicates(conn):
    store_utterances(conn, 1, [Utterance(12.0, 13.0, "first", -0.2, 0.1)])
    store_utterances(conn, 1, [Utterance(12.0, 13.0, "second", -0.2, 0.1)])
    rows = conn.execute("SELECT text FROM utterances").fetchall()
    assert [r["text"] for r in rows] == ["second"]


def test_removing_a_video_takes_its_transcript_with_it(conn):
    store_utterances(conn, 1, [Utterance(12.0, 13.0, "hello", -0.2, 0.1)])
    conn.execute("DELETE FROM videos WHERE id = 1")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM utterances").fetchone()["c"] == 0


# ---------- search ----------

def test_what_was_said_becomes_searchable(conn):
    """The whole point: a spoken word ranks through the machinery that exists.

    No parallel path - a transcript joins the same document as labels, names
    and captions, so it fuses with the other signals rather than competing.
    """
    store_utterances(conn, 1, [Utterance(12.0, 14.0, "did you take your tablets", -0.2, 0.1)])
    assert "tablets" in segment_document(conn, 1)

    rebuild_text_index(conn)
    assert [sid for sid, _ in lexical_ranking(conn, "tablets")] == [1]
    assert lexical_ranking(conn, "helicopter") == []


# ---------- bookkeeping ----------

def test_a_silent_video_is_not_retried_every_run(conn, tmp_path):
    """Marked as listened to, not as having speech.

    Without the distinction a recording with no audio is transcribed again on
    every import, forever, and always finds the same nothing.
    """
    assert [r["id"] for r in pending_videos(conn)] == [1]
    conn.execute("UPDATE videos SET transcribed_at = 123.0 WHERE id = 1")
    conn.commit()
    assert pending_videos(conn) == []
    assert [r["id"] for r in pending_videos(conn, force=True)] == [1]


def test_a_missing_model_is_reported_not_raised(conn, tmp_path):
    """Every optional model here degrades rather than stops."""
    cfg = dataclasses.replace(
        DEFAULT, data_dir=tmp_path,
        model_dir_override=tmp_path / "no-models-here",
    )
    assert load_model(cfg) is None
    summary = transcribe_videos(conn, cfg)
    assert summary.videos == 0
    assert summary.failed and "model" in summary.failed[0]


def test_a_file_with_no_audio_stream_is_recognised(tmp_path):
    """Most cameras record none, and checking costs one demux."""
    fake = tmp_path / "not-a-video.mp4"
    fake.write_bytes(b"nonsense")
    assert has_audio(fake) is False


# ---------- where it runs ----------

def test_a_machine_with_no_gpu_falls_back_to_int8_on_the_cpu():
    """The common case, and it must never raise.

    `pick_device` is called before any model exists, on machines this has
    never seen. A capability probe that throws is worse than one that shrugs.
    """
    from tsv.audio import pick_device

    device, compute = pick_device("cpu")
    assert (device, compute) == ("cpu", "int8")

    device, compute = pick_device("auto")
    assert device in {"cpu", "cuda"}
    assert compute in {"int8", "float16"}


def test_a_gpu_is_taken_when_one_is_visible(monkeypatch):
    """Hard-coding the CPU meant a workstation transcribed at laptop speed.

    Measured on the baseline machine, cores are nearly worthless here - 2
    threads gave 19.4x realtime and 12 gave 20.6x - so a discrete GPU is the
    only hardware that changes the answer.
    """
    import ctranslate2

    from tsv.audio import pick_device

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    assert pick_device("auto") == ("cuda", "float16")

    # Still overridable, because a counted device is not a working one.
    assert pick_device("cpu") == ("cpu", "int8")


def test_counting_a_gpu_that_will_not_load_falls_back(monkeypatch, tmp_path):
    """Listed is not usable - the rule the backend layer already lives by.

    A CUDA device with a broken driver is counted and then fails at model
    load. A slower transcript beats no transcript.
    """
    import ctranslate2

    import tsv.audio as audio

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)

    directory = tmp_path / "whisper-base"
    directory.mkdir()
    (directory / "model.bin").write_bytes(b"stub")

    attempts = []

    class StubWhisper:
        def __init__(self, path, device, compute_type, cpu_threads=0):
            attempts.append(device)
            if device == "cuda":
                raise RuntimeError("no usable CUDA driver")

    import sys
    import types

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = StubWhisper
    monkeypatch.setitem(sys.modules, "faster_whisper", module)

    cfg = dataclasses.replace(
        DEFAULT, data_dir=tmp_path, model_dir_override=tmp_path
    )
    assert audio.load_model(cfg) is not None
    assert attempts == ["cuda", "cpu"]
