"""Receiving a video in pieces.

The thing being defended is that an interrupted transfer costs seconds rather
than starting again - and, just as much, that a transfer which resumed wrongly
is caught here rather than surfacing later as a video that will not play.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time

import pytest
from fastapi.testclient import TestClient

from tsv import db, uploads
from tsv.api import create_app
from tsv.config import DEFAULT


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(DEFAULT, data_dir=tmp_path)


@pytest.fixture
def conn(cfg):
    return db.open_db(cfg.db_path)


@pytest.fixture
def client(cfg):
    db.open_db(cfg.db_path).close()
    return TestClient(create_app(cfg))


def payload(size: int, seed: int = 7) -> bytes:
    return bytes((i * seed + 13) % 251 for i in range(size))


# ---------- resuming ----------

def test_an_interrupted_upload_carries_on_rather_than_restarting(conn, cfg):
    data = payload(1000)
    upload = uploads.begin(conn, cfg.data_dir, "clip.mp4", len(data))
    upload = uploads.write_chunk(conn, upload, 0, data[:400])
    assert upload.received == 400

    # The phone reloads and knows nothing but the file it picked again.
    resumed = uploads.begin(conn, cfg.data_dir, "clip.mp4", len(data))
    assert resumed.id == upload.id
    assert resumed.received == 400, "would have started from zero"


def test_the_offset_is_read_from_disk_not_from_a_counter(conn, cfg):
    """A stored count can disagree with the file after a crash.

    That disagreement is the dangerous one: the client resumes at an offset
    the server believes, writes from there, and produces a file with a hole in
    it. The failure surfaces much later, in the demuxer, looking like a
    corrupt recording rather than a bad transfer.
    """
    data = payload(1000)
    upload = uploads.begin(conn, cfg.data_dir, "clip.mp4", len(data))
    uploads.write_chunk(conn, upload, 0, data[:400])

    # Something outside this process truncates the partial file.
    with upload.path.open("r+b") as handle:
        handle.truncate(150)

    assert uploads.get(conn, upload.id).received == 150


def test_a_chunk_at_the_wrong_offset_is_answered_not_applied(conn, cfg):
    """A retry whose first reply was lost must not duplicate the data."""
    data = payload(1000)
    upload = uploads.begin(conn, cfg.data_dir, "clip.mp4", len(data))
    upload = uploads.write_chunk(conn, upload, 0, data[:400])

    replayed = uploads.write_chunk(conn, upload, 0, data[:400])
    assert replayed.received == 400, "the chunk was written twice"

    ahead = uploads.write_chunk(conn, upload, 900, data[900:])
    assert ahead.received == 400, "a gap was written into the file"


def test_a_completed_upload_is_byte_for_byte_what_was_sent(conn, cfg):
    data = payload(4096)
    upload = uploads.begin(conn, cfg.data_dir, "clip.mp4", len(data))
    for start in range(0, len(data), 512):
        upload = uploads.write_chunk(conn, upload, start, data[start:start + 512])

    assert upload.complete
    staged = uploads.finish(conn, cfg.data_dir, upload)
    assert staged.read_bytes() == data
    assert hashlib.blake2b(staged.read_bytes()).digest() == hashlib.blake2b(data).digest()


def test_finishing_short_is_refused(conn, cfg):
    """Better a refusal than an import of a truncated video.

    A short file reaches the demuxer and is reported as a corrupt recording,
    which sends somebody to look at their camera instead of their WiFi.
    """
    data = payload(1000)
    upload = uploads.begin(conn, cfg.data_dir, "clip.mp4", len(data))
    upload = uploads.write_chunk(conn, upload, 0, data[:600])

    with pytest.raises(ValueError, match="expected"):
        uploads.finish(conn, cfg.data_dir, upload)


def test_two_uploads_of_the_same_name_do_not_share_a_file(conn, cfg):
    """Two people, one router, the same "video.mp4"."""
    first = uploads.begin(conn, cfg.data_dir, "video.mp4", 100)
    second = uploads.begin(conn, cfg.data_dir, "video.mp4", 200)   # different size
    assert first.id != second.id
    assert first.path != second.path


# ---------- housekeeping ----------

def test_abandoned_uploads_are_swept(conn, cfg):
    """These are whole videos in a temp directory - the largest thing this
    application could ever leave behind."""
    upload = uploads.begin(conn, cfg.data_dir, "forgotten.mp4", 1000)
    uploads.write_chunk(conn, upload, 0, payload(500))
    assert upload.path.is_file()

    conn.execute(
        "UPDATE uploads SET touched_at = ? WHERE id = ?",
        (time.time() - uploads.STALE_SECONDS - 60, upload.id),
    )
    conn.commit()

    assert uploads.sweep(conn) == 1
    assert not upload.path.is_file()
    assert uploads.pending(conn) == []


def test_a_recent_upload_is_left_alone(conn, cfg):
    uploads.begin(conn, cfg.data_dir, "in-progress.mp4", 1000)
    assert uploads.sweep(conn) == 0
    assert len(uploads.pending(conn)) == 1


def test_a_row_whose_data_vanished_starts_over_rather_than_resuming(conn, cfg):
    """A cleaned temp directory must not make resuming impossible.

    The row alone is not enough to resume from - without its partial file
    there is nothing to append to, and handing back an offset would tell the
    client to skip bytes that were never stored.
    """
    upload = uploads.begin(conn, cfg.data_dir, "clip.mp4", 1000)
    uploads.write_chunk(conn, upload, 0, payload(400))
    upload.path.unlink()

    fresh = uploads.begin(conn, cfg.data_dir, "clip.mp4", 1000)
    assert fresh.received == 0, "resumed into a file that is not there"
    assert fresh.path.is_file()


def test_there_is_a_disk_space_check(cfg):
    ok, why = uploads.space_for(cfg.data_dir, 1024)
    assert ok and why == ""

    ok, why = uploads.space_for(cfg.data_dir, 1 << 50)     # a petabyte
    assert not ok
    assert "GB" in why


# ---------- over HTTP ----------

def test_the_endpoints_resume_the_way_the_library_does(client):
    data = payload(3000, 11)

    begun = client.post(
        "/api/upload/begin", json={"name": "phone.mp4", "size": len(data)}
    ).json()
    assert begun["offset"] == 0
    assert begun["chunk_bytes"] > 0

    sent = client.put(
        f"/api/upload/{begun['id']}", content=data[:1000],
        headers={"X-Upload-Offset": "0"},
    ).json()
    assert sent["offset"] == 1000 and not sent["complete"]

    again = client.post(
        "/api/upload/begin", json={"name": "phone.mp4", "size": len(data)}
    ).json()
    assert again["id"] == begun["id"] and again["offset"] == 1000

    assert client.get(f"/api/upload/{begun['id']}").json()["offset"] == 1000


def test_finishing_early_is_refused_over_http(client):
    data = payload(3000, 11)
    begun = client.post(
        "/api/upload/begin", json={"name": "phone.mp4", "size": len(data)}
    ).json()
    client.put(f"/api/upload/{begun['id']}", content=data[:1000],
               headers={"X-Upload-Offset": "0"})

    response = client.post(f"/api/upload/{begun['id']}/finish")
    assert response.status_code == 409


def test_a_chunk_running_past_the_declared_size_is_refused(client):
    begun = client.post(
        "/api/upload/begin", json={"name": "phone.mp4", "size": 100}
    ).json()
    response = client.put(
        f"/api/upload/{begun['id']}", content=payload(500),
        headers={"X-Upload-Offset": "0"},
    )
    assert response.status_code == 400


def test_an_upload_can_be_called_off(client):
    begun = client.post(
        "/api/upload/begin", json={"name": "phone.mp4", "size": 100}
    ).json()
    assert client.delete(f"/api/upload/{begun['id']}").status_code == 200
    assert client.get(f"/api/upload/{begun['id']}").status_code == 404


def test_an_upload_needs_a_name_and_a_size(client):
    assert client.post("/api/upload/begin", json={"name": "", "size": 10}).status_code == 400
    assert client.post("/api/upload/begin", json={"name": "a.mp4", "size": 0}).status_code == 400


def test_asking_for_more_room_than_exists_is_refused_before_any_bytes(client):
    response = client.post(
        "/api/upload/begin", json={"name": "huge.mp4", "size": 1 << 50}
    )
    assert response.status_code == 507
    assert "GB" in response.json()["detail"]
