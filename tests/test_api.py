from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tsv import db
from tsv.api import create_app
from tsv.config import DEFAULT
from tsv.ingest import ingest_file


@pytest.fixture(scope="module")
def client(day_clip, tmp_path_factory) -> TestClient:
    data_dir = tmp_path_factory.mktemp("data")
    cfg = dataclasses.replace(DEFAULT, data_dir=data_dir)
    conn = db.open_db(cfg.db_path)
    ingest_file(conn, Path(day_clip["path"]), cfg)
    conn.close()
    return TestClient(create_app(cfg))


def test_summary_reports_the_reduction(client: TestClient):
    body = client.get("/api/summary").json()
    assert body["n_videos"] == 1
    assert body["n_segments"] > 0
    assert 0.0 < body["reduction"] < 1.0


def test_cameras_and_days_are_listed(client: TestClient):
    cameras = client.get("/api/cameras").json()
    assert [c["name"] for c in cameras] == ["ch09"]

    days = client.get("/api/days").json()
    assert days[0]["day"] == "2026-01-01"


def test_timeline_buckets_a_full_day(client: TestClient):
    body = client.get("/api/timeline?day=2026-01-01").json()
    assert len(body["activity"]) == 24 * 60
    assert len(body["coverage"]) == 24 * 60
    assert body["segments"]

    # The clip starts at 08:00, so coverage belongs to that minute and no other.
    covered = [i for i, v in enumerate(body["coverage"]) if v > 0]
    assert covered and all(8 * 60 <= i <= 8 * 60 + 1 for i in covered)


def test_timeline_rejects_a_malformed_day(client: TestClient):
    assert client.get("/api/timeline?day=nonsense").status_code == 400


def test_timeline_of_an_empty_day_is_flat_not_an_error(client: TestClient):
    body = client.get("/api/timeline?day=2019-05-05").json()
    assert body["segments"] == []
    assert sum(body["activity"]) == 0


def test_thumbnails_are_served(client: TestClient):
    segment_id = client.get("/api/segments").json()[0]["id"]
    response = client.get(f"/api/thumb/{segment_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic


def test_media_supports_range_requests(client: TestClient):
    """Without 206 the browser downloads the whole recording to seek."""
    video_id = client.get("/api/segments").json()[0]["video_id"]
    response = client.get(f"/api/media/{video_id}", headers={"Range": "bytes=0-1023"})
    assert response.status_code == 206
    assert response.headers["accept-ranges"] == "bytes"
    assert len(response.content) == 1024


def test_missing_resources_return_404(client: TestClient):
    assert client.get("/api/thumb/999999").status_code == 404
    assert client.get("/api/media/999999").status_code == 404


def test_static_route_refuses_path_traversal(client: TestClient):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/..%2F..%2Ftsv%2Fdb.py").status_code == 404


# ---------- Phase 1 ----------


@pytest.fixture(scope="module")
def analyzed_client(day_clip, tmp_path_factory) -> TestClient:
    """A client whose index has been through detection and tracking."""
    from tsv.analyze import analyze_all
    from tests.test_analyze import BrightBoxDetector

    data_dir = tmp_path_factory.mktemp("data-analyzed")
    cfg = dataclasses.replace(
        DEFAULT,
        data_dir=data_dir,
        detect=dataclasses.replace(DEFAULT.detect, detect_fps=6.0, decode_width=320),
    )
    conn = db.open_db(cfg.db_path)
    ingest_file(conn, Path(day_clip["path"]), cfg)
    analyze_all(conn, cfg, detector=BrightBoxDetector())
    conn.close()
    return TestClient(create_app(cfg))


def test_summary_reports_analysis_progress(analyzed_client: TestClient):
    body = analyzed_client.get("/api/summary").json()
    assert body["n_analyzed"] == body["n_segments"]
    assert body["n_tracklets"] > 0
    assert body["top_labels"][0]["label"] == "person"


def test_labels_endpoint_counts_object_kinds(analyzed_client: TestClient):
    labels = analyzed_client.get("/api/labels").json()
    assert labels and labels[0]["label"] == "person"
    assert labels[0]["n"] >= 1


def test_objects_endpoint_returns_tracklets_in_time_order(analyzed_client: TestClient):
    objects = analyzed_client.get("/api/objects").json()
    assert objects
    starts = [o["ts_start"] for o in objects]
    assert starts == sorted(starts)
    assert {"label", "x_start", "x_end", "max_score", "segment_id"} <= set(objects[0])


def test_objects_can_be_filtered_by_label(analyzed_client: TestClient):
    assert analyzed_client.get("/api/objects?label=person").json()
    assert analyzed_client.get("/api/objects?label=giraffe").json() == []


def test_objects_can_be_filtered_by_score(analyzed_client: TestClient):
    assert analyzed_client.get("/api/objects?min_score=0.5").json()
    assert analyzed_client.get("/api/objects?min_score=0.999").json() == []


def test_objects_rejects_a_malformed_day(analyzed_client: TestClient):
    assert analyzed_client.get("/api/objects?day=nonsense").status_code == 400


def test_timeline_segments_carry_their_labels(analyzed_client: TestClient):
    day = analyzed_client.get("/api/days").json()[0]["day"]
    body = analyzed_client.get(f"/api/timeline?day={day}").json()
    labelled = [s for s in body["segments"] if s["labels"]]
    assert labelled
    assert json.loads(labelled[0]["labels"])["person"] >= 1


def test_tracklet_crops_are_served(analyzed_client: TestClient):
    tracklet_id = analyzed_client.get("/api/objects").json()[0]["id"]
    response = analyzed_client.get(f"/api/crop/{tracklet_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_missing_crop_returns_404(analyzed_client: TestClient):
    assert analyzed_client.get("/api/crop/999999").status_code == 404
