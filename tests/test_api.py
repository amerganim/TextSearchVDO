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


# ---------- Phase 2: zones ----------


@pytest.fixture
def zone_client(day_clip, tmp_path) -> TestClient:
    """A fresh analysed index per test, since zone tests mutate it."""
    from tsv.analyze import analyze_all
    from tests.test_analyze import BrightBoxDetector

    cfg = dataclasses.replace(
        DEFAULT,
        data_dir=tmp_path,
        detect=dataclasses.replace(DEFAULT.detect, detect_fps=6.0, decode_width=320),
    )
    conn = db.open_db(cfg.db_path)
    ingest_file(conn, Path(day_clip["path"]), cfg)
    analyze_all(conn, cfg, detector=BrightBoxDetector())
    conn.close()
    return TestClient(create_app(cfg))


def _camera_id(client: TestClient) -> int:
    return client.get("/api/cameras").json()[0]["id"]


def _add_line(client: TestClient, name="middle"):
    return client.post("/api/zones", json={
        "camera_id": _camera_id(client), "name": name, "kind": "line",
        "points": [[0.5, 0.0], [0.5, 1.0]],
    })


def test_creating_a_zone_returns_it_with_events_already_computed(zone_client):
    response = _add_line(zone_client)
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "middle"
    assert body["kind"] == "line"
    # Events are available immediately - no video decoding is involved.
    assert body["n_events"] > 0


def test_zones_are_listed_and_filtered_by_camera(zone_client):
    _add_line(zone_client)
    assert len(zone_client.get("/api/zones").json()) == 1
    camera_id = _camera_id(zone_client)
    assert len(zone_client.get(f"/api/zones?camera_id={camera_id}").json()) == 1
    assert zone_client.get("/api/zones?camera_id=9999").json() == []


def test_a_malformed_zone_is_rejected(zone_client):
    bad = zone_client.post("/api/zones", json={
        "camera_id": _camera_id(zone_client), "name": "bad", "kind": "line",
        "points": [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
    })
    assert bad.status_code == 400

    off_frame = zone_client.post("/api/zones", json={
        "camera_id": _camera_id(zone_client), "name": "off", "kind": "line",
        "points": [[0.1, 0.1], [4.0, 0.5]],
    })
    assert off_frame.status_code == 400

    wrong_kind = zone_client.post("/api/zones", json={
        "camera_id": _camera_id(zone_client), "name": "x", "kind": "triangle",
        "points": [[0.1, 0.1], [0.5, 0.5]],
    })
    assert wrong_kind.status_code == 422


def test_duplicate_zone_names_on_one_camera_conflict(zone_client):
    assert _add_line(zone_client).status_code == 200
    assert _add_line(zone_client).status_code == 409


def test_events_are_listed_and_filterable(zone_client):
    _add_line(zone_client)
    events = zone_client.get("/api/events").json()
    assert events
    assert {"zone_name", "kind", "ts", "label"} <= set(events[0])
    assert [e["ts"] for e in events] == sorted(e["ts"] for e in events)

    assert zone_client.get("/api/events?kind=cross_out").json()
    assert zone_client.get("/api/events?kind=nonsense").json() == []
    assert zone_client.get("/api/events?label=person").json()
    assert zone_client.get("/api/events?label=giraffe").json() == []


def test_events_reject_a_malformed_day(zone_client):
    assert zone_client.get("/api/events?day=nonsense").status_code == 400


def test_deleting_a_zone_removes_its_events(zone_client):
    zone_id = _add_line(zone_client).json()["id"]
    assert zone_client.get("/api/events").json()

    assert zone_client.delete(f"/api/zones/{zone_id}").status_code == 200
    assert zone_client.get("/api/events").json() == []
    assert zone_client.delete(f"/api/zones/{zone_id}").status_code == 404


def test_recompute_is_reported(zone_client):
    _add_line(zone_client)
    body = zone_client.post("/api/zones/recompute").json()
    assert body["n_zones"] == 1
    assert body["n_events"] > 0
    assert "cross_out" in body["by_kind"]


def test_a_still_frame_is_served_for_drawing_on(zone_client):
    camera_id = _camera_id(zone_client)
    response = zone_client.get(f"/api/frame/{camera_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"


def test_a_frame_for_an_unknown_camera_is_404(zone_client):
    assert zone_client.get("/api/frame/9999").status_code == 404


# ---------- Phase 2: identity ----------


def _first_tracklet(client: TestClient) -> int:
    return client.get("/api/objects").json()[0]["id"]


def test_enrolling_names_a_tracklet(zone_client):
    tracklet_id = _first_tracklet(zone_client)
    response = zone_client.post("/api/identities/enroll",
                                json={"tracklet_id": tracklet_id, "name": "Rafi"})
    assert response.status_code == 200
    assert response.json()["name"] == "Rafi"

    people = zone_client.get("/api/identities").json()
    assert people[0]["name"] == "Rafi"
    assert people[0]["n_sightings"] == 1


def test_enrolling_a_tracklet_with_no_embedding_says_so(zone_client):
    """The sighting is still labelled; it just teaches the gallery nothing."""
    body = zone_client.post("/api/identities/enroll",
                            json={"tracklet_id": _first_tracklet(zone_client),
                                  "name": "Rafi"}).json()
    assert body["examples_added"] == 0
    assert "no embeddings" in body["note"]


def test_enrolling_an_unknown_tracklet_is_404(zone_client):
    response = zone_client.post("/api/identities/enroll",
                                json={"tracklet_id": 999999, "name": "Nobody"})
    assert response.status_code == 404


def test_enrolling_without_a_name_is_rejected(zone_client):
    response = zone_client.post("/api/identities/enroll",
                                json={"tracklet_id": _first_tracklet(zone_client), "name": ""})
    assert response.status_code == 422


def test_objects_carry_the_identity_name(zone_client):
    tracklet_id = _first_tracklet(zone_client)
    zone_client.post("/api/identities/enroll", json={"tracklet_id": tracklet_id, "name": "Rafi"})

    named = [o for o in zone_client.get("/api/objects").json() if o["id"] == tracklet_id][0]
    assert named["identity_name"] == "Rafi"


def test_events_carry_the_identity_and_can_be_filtered_by_it(zone_client):
    """The point of the whole phase: "when did Rafi cross the front door"."""
    _add_line(zone_client, "front door")
    events = zone_client.get("/api/events").json()
    assert events
    tracklet_id = events[0]["tracklet_id"]
    zone_client.post("/api/identities/enroll", json={"tracklet_id": tracklet_id, "name": "Rafi"})

    named = zone_client.get("/api/events?identity=Rafi").json()
    assert named
    assert all(e["identity_name"] == "Rafi" for e in named)
    assert zone_client.get("/api/events?identity=Nobody").json() == []


def test_summary_counts_the_people_left_to_name(zone_client):
    """What the People button in the app is built from.

    Without a count of the unnamed there is nothing to put on the button, and
    naming - the feature the whole product is sold on - had no way into the UI
    at all.
    """
    body = zone_client.get("/api/summary").json()
    assert body["n_people"] >= 1
    assert body["n_people_unnamed"] == body["n_people"]
    assert "faces_ready" in body

    tracklet_id = _first_tracklet(zone_client)
    zone_client.post("/api/identities/enroll",
                     json={"tracklet_id": tracklet_id, "name": "Rafi"})

    after = zone_client.get("/api/summary").json()
    assert after["n_people_unnamed"] == body["n_people_unnamed"] - 1


def test_the_people_panel_can_be_built_from_the_endpoints_it_calls(zone_client):
    """The exact three requests the panel makes on open.

    It shows only sightings with a crop, so a person row without a thumbnail
    would render a blank square asking somebody to name nothing.
    """
    people = zone_client.get("/api/objects?label=person&limit=200").json()
    assert people and all(p["label"] == "person" for p in people)

    croppable = [p for p in people if p["thumb_path"]]
    assert croppable, "no person crop to name"
    assert zone_client.get(f"/api/crop/{croppable[0]['id']}").status_code == 200
    assert zone_client.get("/api/identities").json() == []


def test_naming_a_sighting_with_no_face_says_so_rather_than_claiming_a_match(zone_client):
    """The honest half of enrolment.

    A tracklet with no stored face embedding teaches the gallery nothing, so
    the name covers that one sighting. Reporting it as a success would promise
    matching that will never happen.
    """
    tracklet_id = _first_tracklet(zone_client)
    body = zone_client.post(
        "/api/identities/enroll", json={"tracklet_id": tracklet_id, "name": "Rafi"}
    ).json()

    assert body["examples_added"] == 0
    assert body["kinds"] == []
    assert body["note"]
    assert zone_client.post("/api/identities/assign?kind=face").json()["assigned"] == 0


def test_a_named_sighting_can_be_rejected_through_the_api(zone_client):
    """The undo the app offers next to every guessed name."""
    tracklet_id = _first_tracklet(zone_client)
    zone_client.post("/api/identities/enroll",
                     json={"tracklet_id": tracklet_id, "name": "Rafi"})

    assert zone_client.delete(f"/api/objects/{tracklet_id}/identity").json() == {"cleared": True}
    named = [o for o in zone_client.get("/api/objects").json() if o["id"] == tracklet_id][0]
    assert named["identity_name"] is None
    assert zone_client.delete("/api/objects/999999/identity").status_code == 404


def test_deleting_an_identity_unnames_its_sightings(zone_client):
    tracklet_id = _first_tracklet(zone_client)
    identity_id = zone_client.post(
        "/api/identities/enroll", json={"tracklet_id": tracklet_id, "name": "Rafi"}
    ).json()["identity_id"]

    assert zone_client.delete(f"/api/identities/{identity_id}").status_code == 200
    assert zone_client.get("/api/identities").json() == []
    named = [o for o in zone_client.get("/api/objects").json() if o["id"] == tracklet_id][0]
    assert named["identity_name"] is None
    assert zone_client.delete(f"/api/identities/{identity_id}").status_code == 404


def test_assign_reports_its_decisions(zone_client):
    body = zone_client.post("/api/identities/assign?kind=face").json()
    assert body["kind"] == "face"
    assert body["assigned"] == 0          # nothing enrolled with embeddings yet
    assert "ambiguous" in body


def test_assign_rejects_an_unknown_kind(zone_client):
    assert zone_client.post("/api/identities/assign?kind=vibes").status_code == 422


def test_summary_counts_identities(zone_client):
    zone_client.post("/api/identities/enroll",
                     json={"tracklet_id": _first_tracklet(zone_client), "name": "Rafi"})
    body = zone_client.get("/api/summary").json()
    assert body["n_identities"] == 1
    assert body["n_named"] == 1


# ---------- Phase 3: search ----------


def test_search_needs_no_models_to_match_words(zone_client):
    zone_client.post("/api/search/reindex")
    body = zone_client.get("/api/search?q=person").json()
    assert body["n"] > 0
    assert all("lexical" in r["sources"] for r in body["results"])


def test_reindex_reports_what_it_indexed(zone_client):
    body = zone_client.post("/api/search/reindex").json()
    assert body["indexed"] >= 1


def test_search_results_are_playable(zone_client):
    zone_client.post("/api/search/reindex")
    hit = zone_client.get("/api/search?q=person").json()["results"][0]
    assert {"segment_id", "video_id", "t_start", "ts_start"} <= set(hit)
    assert zone_client.get(f"/api/media/{hit['video_id']}").status_code in (200, 206)


def test_search_can_be_narrowed_by_filters(zone_client):
    zone_client.post("/api/search/reindex")
    assert zone_client.get("/api/search?q=person&camera_id=9999").json()["n"] == 0
    assert zone_client.get("/api/search?q=person&label=giraffe").json()["n"] == 0
    assert zone_client.get("/api/search?q=person&label=person").json()["n"] > 0


def test_search_by_person_after_enrolment(zone_client):
    tracklet_id = _first_tracklet(zone_client)
    zone_client.post("/api/identities/enroll", json={"tracklet_id": tracklet_id, "name": "Rafi"})
    zone_client.post("/api/search/reindex")

    assert zone_client.get("/api/search?identity=Rafi").json()["n"] > 0
    assert zone_client.get("/api/search?identity=Nobody").json()["n"] == 0


def test_a_query_matching_nothing_returns_nothing(zone_client):
    """It must not fall back to browsing and answer a different question."""
    zone_client.post("/api/search/reindex")
    assert zone_client.get("/api/search?q=helicopter").json()["n"] == 0


def test_an_empty_query_browses(zone_client):
    body = zone_client.get("/api/search").json()
    assert body["n"] > 0
    assert all(r["sources"] == ["filter"] for r in body["results"])


def test_search_rejects_a_malformed_day(zone_client):
    assert zone_client.get("/api/search?q=person&day=nonsense").status_code == 400


def test_search_survives_hostile_query_text(zone_client):
    """Users type quotes and FTS operators; nothing may 500."""
    zone_client.post("/api/search/reindex")
    for hostile in ("person*", "-person", "a AND OR", chr(34), "^%$#"):
        assert zone_client.get("/api/search", params={"q": hostile}).status_code == 200


def test_summary_reports_whether_semantic_search_is_available(zone_client):
    body = zone_client.get("/api/summary").json()
    assert "semantic_ready" in body
    assert "n_embedded" in body


def test_concurrent_requests_do_not_corrupt_the_connection(zone_client):
    """FastAPI runs sync endpoints in a threadpool.

    A single sqlite3 connection shared across those threads raises
    "bad parameter or other API misuse" under concurrent execute() calls, even
    with check_same_thread=False. A real page requests several thumbnails at
    once, so this is the normal case, not a stress test.
    """
    import concurrent.futures

    zone_client.post("/api/search/reindex")
    paths = [
        "/api/summary", "/api/cameras", "/api/days",
        "/api/objects", "/api/labels", "/api/zones",
        "/api/events", "/api/identities", "/api/search?q=person",
    ] * 6

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        codes = list(pool.map(lambda p: zone_client.get(p).status_code, paths))

    assert set(codes) == {200}, f"unexpected statuses: {sorted(set(codes))}"


def test_concurrent_thumbnail_requests(zone_client):
    import concurrent.futures

    segments = zone_client.get("/api/segments").json()
    ids = [s["id"] for s in segments] * 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        codes = list(pool.map(lambda i: zone_client.get(f"/api/thumb/{i}").status_code, ids))
    assert set(codes) <= {200, 404}
    assert 500 not in codes


def test_static_assets_are_not_cached(zone_client):
    """A stale stylesheet after an update looks like a broken app, and there is
    nothing to save by caching over localhost."""
    response = zone_client.get("/static/simple.css")
    assert response.status_code == 200
    assert "no-store" in response.headers.get("cache-control", "")


def test_the_simple_app_is_served_at_the_root(zone_client):
    body = zone_client.get("/").text
    assert "Add a video to get started" in body or 'id="q"' in body
    assert zone_client.get("/advanced").status_code == 200


# ---------- captioning from the app ----------


def test_summary_reports_how_much_captioning_is_outstanding(zone_client):
    """The button says how much work it is, so it must be countable."""
    body = zone_client.get("/api/summary").json()
    assert "caption_ready" in body
    assert "n_captioned" in body
    assert "n_to_caption" in body
    # Nothing captioned yet, and the fixture has person tracklets.
    assert body["n_captioned"] == 0
    assert body["n_to_caption"] >= 1


def test_captioning_without_the_model_explains_itself(zone_client, monkeypatch):
    """A 400 with instructions beats a 500 with a traceback."""
    import tsv.config

    monkeypatch.setattr(
        tsv.config.Config, "has_caption_model", property(lambda self: False)
    )
    response = zone_client.post("/api/caption")
    assert response.status_code == 400
    assert "fetch_caption_model" in response.json()["detail"]
