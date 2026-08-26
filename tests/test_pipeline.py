"""End-to-end checks against clips whose activity windows are known.

These are the tests that would have caught the two bugs that actually
happened: a GOP sawtooth swamping the motion signal, and a stretch of
all-intra frames reading as idle because the encoder stopped emitting
P-frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tsv import db
from tsv.config import DEFAULT
from tsv.ingest import ingest_file
from tsv.motion.decode import trace_windows
from tsv.motion.packets import scan
from tsv.motion.segments import activity_to_segments

# Tier A pads generously and Tier B trims to a few sample periods, so a couple
# of seconds either side of truth is a pass.
TOLERANCE = 2.5


def _segments_for(clip: dict):
    path = Path(clip["path"])
    duration = clip["duration"]
    packet_scan = scan(path, DEFAULT.tier_a, duration)
    trace = trace_windows(path, packet_scan.candidates, DEFAULT.tier_b)
    period = trace.sample_period or (1.0 / DEFAULT.tier_b.sample_fps)
    segments = activity_to_segments(
        trace.times, trace.scores, DEFAULT.segments, period, duration=duration
    )
    return packet_scan, trace, segments


def test_tier_a_prefilter_keeps_every_real_window(day_clip):
    """A miss here is unrecoverable - Tier B never sees what Tier A drops."""
    packet_scan, _, _ = _segments_for(day_clip)
    for start, end in day_clip["activity"]:
        covered = any(a <= start and b >= end for a, b in packet_scan.candidates)
        assert covered, f"window {start}-{end} not in {packet_scan.candidates}"


def test_tier_a_actually_discards_something(day_clip):
    """A prefilter that keeps everything is just overhead."""
    packet_scan, _, _ = _segments_for(day_clip)
    assert packet_scan.candidate_seconds < 0.9 * packet_scan.duration


def test_gop_position_normalisation_flattens_the_sawtooth(day_clip):
    """Idle bins must not alternate with the keyframe interval.

    Before normalisation the idle floor swung ~12% between bins that held a
    keyframe and bins that did not, which is larger than a walking person.
    """
    packet_scan, _, _ = _segments_for(day_clip)
    bins = packet_scan.bins
    active = np.zeros(bins.size, dtype=bool)
    for start, end in day_clip["activity"]:
        active[int(start) : int(end) + 1] = True
    idle = bins[(~active) & (bins > 0)]
    assert idle.std() / idle.mean() < 0.06


def test_known_windows_are_recovered(day_clip):
    _, _, segments = _segments_for(day_clip)
    assert len(segments) == len(day_clip["activity"])
    for (start, end), segment in zip(day_clip["activity"], segments):
        assert abs(segment.t_start - start) <= TOLERANCE
        assert abs(segment.t_end - end) <= TOLERANCE


def test_idle_footage_produces_no_segments(idle_clip):
    _, _, segments = _segments_for(idle_clip)
    assert segments == []


def test_ingest_is_idempotent(day_clip, tmp_path):
    import dataclasses

    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    conn = db.open_db(cfg.db_path)
    path = Path(day_clip["path"])

    first = ingest_file(conn, path, cfg)
    assert first.status == "ingested"
    assert first.n_segments > 0

    second = ingest_file(conn, path, cfg)
    assert second.status == "skipped"

    forced = ingest_file(conn, path, cfg, force=True)
    assert forced.status == "ingested"
    assert forced.n_segments == first.n_segments

    # Re-ingesting must replace rows, never accumulate duplicates.
    n_videos = conn.execute("SELECT COUNT(*) c FROM videos").fetchone()["c"]
    n_segments = conn.execute("SELECT COUNT(*) c FROM segments").fetchone()["c"]
    assert n_videos == 1
    assert n_segments == first.n_segments


def test_ingest_reports_a_real_reduction(day_clip, tmp_path):
    import dataclasses

    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    conn = db.open_db(cfg.db_path)
    result = ingest_file(conn, Path(day_clip["path"]), cfg)
    assert 0.3 < result.compression < 0.99


def test_a_broken_file_fails_without_stopping_the_run(tmp_path):
    import dataclasses

    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    conn = db.open_db(cfg.db_path)
    junk = tmp_path / "ch01_20260101000000.mp4"
    junk.write_bytes(b"this is not a video" * 100)

    result = ingest_file(conn, junk, cfg)
    assert result.status == "failed"
    assert result.note
