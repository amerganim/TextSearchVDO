"""Zone events derived from stored tracklets.

The property that matters most here: recomputing touches no video. Zones are
edited constantly, and moving a door line must not mean re-running detection
over a week of footage.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tsv import db
from tsv.analyze import analyze_all
from tsv.config import DEFAULT
from tsv.events import create_zone, delete_zone, list_zones, recompute_events
from tsv.ingest import ingest_file
from tests.test_analyze import BrightBoxDetector


@pytest.fixture
def analyzed(day_clip, tmp_path):
    cfg = dataclasses.replace(
        DEFAULT,
        data_dir=tmp_path,
        detect=dataclasses.replace(DEFAULT.detect, detect_fps=6.0, decode_width=320),
    )
    conn = db.open_db(cfg.db_path)
    ingest_file(conn, Path(day_clip["path"]), cfg)
    analyze_all(conn, cfg, detector=BrightBoxDetector())
    camera_id = int(conn.execute("SELECT id FROM cameras").fetchone()["id"])
    return conn, cfg, camera_id


def _mid_line(conn, camera_id):
    """A vertical line down the middle of frame; the subject walks through it."""
    return create_zone(conn, camera_id, "middle", "line", [(0.5, 0.0), (0.5, 1.0)])


def test_zone_round_trips_through_the_database(analyzed):
    conn, _, camera_id = analyzed
    made = _mid_line(conn, camera_id)
    listed = list_zones(conn, camera_id)
    assert len(listed) == 1
    assert listed[0].id == made.id
    assert listed[0].name == "middle"
    assert listed[0].kind == "line"
    assert listed[0].points == [(0.5, 0.0), (0.5, 1.0)]


def test_invalid_zones_are_refused_before_insert(analyzed):
    conn, _, camera_id = analyzed
    with pytest.raises(ValueError):
        create_zone(conn, camera_id, "bad", "line", [(0, 0), (1, 1), (0.5, 0.5)])
    assert list_zones(conn, camera_id) == []


def test_a_walker_crossing_the_line_generates_events(analyzed):
    conn, _, camera_id = analyzed
    _mid_line(conn, camera_id)
    summary = recompute_events(conn)

    assert summary.n_zones == 1
    assert summary.n_events > 0
    # The synthetic subject always walks left to right, so every crossing is
    # the same direction.
    assert set(summary.by_kind) == {"cross_out"}


def test_events_carry_wall_clock_and_the_object_label(analyzed):
    conn, _, camera_id = analyzed
    _mid_line(conn, camera_id)
    recompute_events(conn)

    row = conn.execute(
        """SELECT e.ts, e.t, e.label, v.start_ts
           FROM events e JOIN videos v ON v.id = e.video_id LIMIT 1"""
    ).fetchone()
    assert row["label"] == "person"
    assert abs(row["ts"] - (row["start_ts"] + row["t"])) < 1e-6


def test_recomputing_is_idempotent(analyzed):
    conn, _, camera_id = analyzed
    _mid_line(conn, camera_id)
    first = recompute_events(conn)
    second = recompute_events(conn)
    assert first.n_events == second.n_events
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == first.n_events


def test_moving_a_zone_changes_the_events_without_touching_video(analyzed, monkeypatch):
    """Editing a zone must never reopen a video file."""
    conn, _, camera_id = analyzed
    _mid_line(conn, camera_id)

    import av
    def explode(*a, **kw):
        raise AssertionError("recompute must not open the video")
    monkeypatch.setattr(av, "open", explode)

    before = recompute_events(conn).n_events

    # A line off in the corner that nothing crosses.
    conn.execute("DELETE FROM zones")
    create_zone(conn, camera_id, "corner", "line", [(0.0, 0.0), (0.02, 0.02)])
    after = recompute_events(conn).n_events

    assert before > 0
    assert after == 0


def test_deleting_a_zone_cascades_its_events(analyzed):
    conn, _, camera_id = analyzed
    zone = _mid_line(conn, camera_id)
    recompute_events(conn)
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] > 0

    assert delete_zone(conn, zone.id)
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == 0


def test_no_zones_means_no_events(analyzed):
    conn, _, _ = analyzed
    summary = recompute_events(conn)
    assert summary.n_zones == 0
    assert summary.n_events == 0


def test_a_region_covering_the_whole_frame_catches_every_tracklet(analyzed):
    conn, _, camera_id = analyzed
    create_zone(conn, camera_id, "everywhere", "region",
                [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    summary = recompute_events(conn, hysteresis=1)

    n_tracklets = conn.execute("SELECT COUNT(*) c FROM tracklets").fetchone()["c"]
    assert summary.n_tracklets == n_tracklets
    assert summary.by_kind.get("enter", 0) == n_tracklets
    assert summary.by_kind.get("dwell", 0) == n_tracklets


def test_zones_on_another_camera_are_left_alone(analyzed):
    conn, _, camera_id = analyzed
    other = db.get_or_create_camera(conn, "ch99")
    _mid_line(conn, camera_id)
    create_zone(conn, other, "elsewhere", "line", [(0.5, 0.0), (0.5, 1.0)])

    recompute_events(conn)
    total = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]

    # Recomputing just the other camera must not delete this camera's events.
    recompute_events(conn, camera_id=other)
    assert conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"] == total
