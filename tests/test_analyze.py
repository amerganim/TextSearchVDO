"""Phase 1 end to end, with a stand-in detector.

The synthetic clips contain a bright rectangle moving on a known trajectory,
so a threshold-and-contour "detector" produces genuinely moving boxes. That
exercises association, tracklet aggregation, crops and the database writes
without needing model weights - the parts most likely to be wrong are the
bookkeeping around the network, not the network.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tsv import db
from tsv.analyze import analyze_all, analyze_video
from tsv.config import DEFAULT
from tsv.ingest import ingest_file
from tsv.models.detect import Detection
from tsv.track.bytetrack import TrackerConfig


class BrightBoxDetector:
    """Finds the synthetic subject: the brightest blob in the frame."""

    info = "stub:none"

    def __init__(self, cls: int = 0, threshold: int = 180, min_area: int = 60) -> None:
        self.cls, self.threshold, self.min_area = cls, threshold, min_area

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= self.min_area:
                out.append(Detection(float(x), float(y), float(x + w), float(y + h), 0.9, self.cls))
        return out


@pytest.fixture
def indexed(day_clip, tmp_path):
    cfg = dataclasses.replace(
        DEFAULT,
        data_dir=tmp_path,
        detect=dataclasses.replace(DEFAULT.detect, detect_fps=6.0, decode_width=320),
    )
    conn = db.open_db(cfg.db_path)
    result = ingest_file(conn, Path(day_clip["path"]), cfg)
    assert result.status == "ingested"
    return conn, cfg, result


def _analyze(conn, cfg, **kw):
    video_id = int(conn.execute("SELECT id FROM videos").fetchone()["id"])
    return analyze_video(
        conn, video_id, BrightBoxDetector(), cfg,
        tracker_config=TrackerConfig(min_hits=2, max_age=8), **kw,
    )


def test_analysis_finds_one_tracklet_per_activity_window(indexed, day_clip):
    conn, cfg, ingested = indexed
    result = _analyze(conn, cfg)

    assert result.status == "analyzed"
    # The clip has one moving subject per window, so one tracklet per segment.
    assert result.n_tracklets == len(day_clip["activity"])
    assert result.n_detections > result.n_tracklets


def test_tracklet_times_land_inside_their_segment(indexed):
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    rows = conn.execute(
        """SELECT t.t_start, t.t_end, s.t_start AS s_start, s.t_end AS s_end
           FROM tracklets t JOIN segments s ON s.id = t.segment_id"""
    ).fetchall()
    assert rows
    for r in rows:
        assert r["s_start"] <= r["t_start"] <= r["t_end"] <= r["s_end"]


def test_wall_clock_is_derived_from_the_video_start(indexed):
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    row = conn.execute(
        """SELECT t.ts_start, t.t_start, v.start_ts
           FROM tracklets t JOIN videos v ON v.id = t.video_id LIMIT 1"""
    ).fetchone()
    assert abs(row["ts_start"] - (row["start_ts"] + row["t_start"])) < 1e-6


def test_boxes_are_stored_normalised(indexed):
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    for table in ("tracklets", "detections"):
        row = conn.execute(
            f"SELECT MIN(x1) a, MIN(y1) b, MAX(x2) c, MAX(y2) d FROM {table}"
        ).fetchone()
        assert 0.0 <= row["a"] and 0.0 <= row["b"]
        assert row["c"] <= 1.0 and row["d"] <= 1.0


def test_direction_of_travel_is_recorded(indexed):
    """The synthetic subject always walks left to right."""
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    rows = conn.execute("SELECT x_start, x_end FROM tracklets").fetchall()
    assert rows
    assert all(r["x_end"] > r["x_start"] for r in rows)


def test_segment_labels_are_denormalised_for_the_timeline(indexed):
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    rows = conn.execute(
        "SELECT labels, n_tracklets, analyzed_at FROM segments WHERE n_tracklets > 0"
    ).fetchall()
    assert rows
    for r in rows:
        assert r["analyzed_at"] is not None
        assert json.loads(r["labels"])["person"] >= 1


def test_crops_are_written_for_each_tracklet(indexed):
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    paths = [r["thumb_path"] for r in conn.execute("SELECT thumb_path FROM tracklets")]
    assert paths and all(p and Path(p).is_file() for p in paths)
    for p in paths:
        assert Path(p).read_bytes()[:2] == b"\xff\xd8"   # JPEG magic


def test_analysis_is_skipped_on_a_second_run(indexed):
    conn, cfg, _ = indexed
    first = _analyze(conn, cfg)
    assert first.status == "analyzed"

    again = _analyze(conn, cfg)
    assert again.status == "skipped"

    forced = _analyze(conn, cfg, force=True)
    assert forced.status == "analyzed"
    assert forced.n_tracklets == first.n_tracklets

    # Re-analysis replaces rows rather than accumulating them.
    total = conn.execute("SELECT COUNT(*) c FROM tracklets").fetchone()["c"]
    assert total == first.n_tracklets


def test_detections_cascade_away_with_their_tracklet(indexed):
    conn, cfg, _ = indexed
    _analyze(conn, cfg)
    assert conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"] > 0
    conn.execute("DELETE FROM tracklets")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"] == 0


def test_analyze_all_reports_a_backend_and_labels(indexed):
    conn, cfg, _ = indexed
    summary = analyze_all(conn, cfg, detector=BrightBoxDetector())
    assert summary.backend == "stub:none"
    assert summary.analyzed
    assert summary.labels["person"] >= 1


def test_missing_source_file_fails_cleanly(indexed, tmp_path):
    conn, cfg, _ = indexed
    conn.execute("UPDATE videos SET path = ?", (str(tmp_path / "gone.mp4"),))
    conn.commit()
    assert _analyze(conn, cfg).status == "failed"


# ---------- vectors are written, and labelled ----------

class StubClip:
    """A CLIP stand-in: fixed-width vectors, no model files needed."""

    info = "stub:clip"

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_image(None)


def test_analysis_writes_embeddings_and_records_which_model_made_them(indexed):
    """A regression test for a bug that produced no tracklets at all.

    `_write_tracklets` is module level and has no `cfg`. Reaching for
    `cfg.clip.name` inside it raised NameError, which `analyze_video`'s caller
    caught and filed as "this video failed" - so with a CLIP model loaded,
    every analysed file silently yielded nothing. Every existing test passed,
    because they all ran with `clip=None` and never reached the line.

    So this asserts both halves: that vectors are written when an embedder is
    present, and that each one carries the name of what produced it.
    """
    conn, cfg, _ = indexed
    result = _analyze(conn, cfg, clip=StubClip())

    assert result.status != "failed", result.note
    assert result.n_tracklets > 0, "analysis produced no tracklets"

    vectors = conn.execute(
        "SELECT model, COUNT(*) AS n FROM tracklet_embeddings "
        "WHERE kind = 'clip' GROUP BY model"
    ).fetchall()
    assert vectors, "an embedder was supplied but nothing was stored"
    assert [row["model"] for row in vectors] == [cfg.clip.name]

    scene = conn.execute(
        "SELECT model FROM segment_embeddings WHERE kind = 'clip'"
    ).fetchall()
    assert scene and all(row["model"] == cfg.clip.name for row in scene)
