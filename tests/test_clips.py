"""Cutting a few seconds out, instead of sending the whole recording.

Tapping a result used to hand the browser the entire file. Measured on real
footage: 0.68 MB and 0.02 seconds against 124 MB, because nothing is
re-encoded. What these tests defend is that the small file is still a correct
one - the failure mode of a stream copy is a clip that plays but lies about
what it is.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import av
import pytest
from fastapi.testclient import TestClient

from tsv import db
from tsv.api import create_app
from tsv.clips import MAX_SECONDS, cut
from tsv.config import DEFAULT
from tsv.ingest import ingest_file


@pytest.fixture(scope="module")
def clip_client(day_clip, tmp_path_factory) -> TestClient:
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path_factory.mktemp("clips"))
    conn = db.open_db(cfg.db_path)
    ingest_file(conn, Path(day_clip["path"]), cfg)
    conn.close()
    return TestClient(create_app(cfg))


def probe(data: bytes):
    with av.open(io.BytesIO(data)) as container:
        video = container.streams.video[0]
        times = [float(f.pts * video.time_base) for f in container.decode(video)]
    return times


# ---------- the cut itself ----------

def test_a_clip_starts_at_zero_rather_than_where_it_came_from(day_clip):
    """Copied packets keep the timestamps they had in the source.

    Left alone, a ten second clip taken from twenty seconds in declares itself
    thirty seconds long: the player draws a scrubber across the whole
    recording, starts near the end of it, and the part that exists is a
    sliver. It plays, and it lies about what it is.
    """
    data = cut(Path(day_clip["path"]), at=20.0, seconds=6.0)
    times = probe(data)

    assert times, "no frames came back"
    assert times[0] < 1.0, f"clip begins at {times[0]:.1f}s instead of the start"
    assert times[-1] < 12.0, "clip is longer than it should be"


def test_a_clip_is_far_smaller_than_the_recording(day_clip):
    data = cut(Path(day_clip["path"]), at=15.0, seconds=4.0)
    whole = Path(day_clip["path"]).stat().st_size
    assert len(data) < whole


def test_a_clip_covers_the_moment_asked_for(day_clip):
    """It may start earlier - a stream copy has to begin on a keyframe - but
    it must not start later, or the moment is missing from it."""
    data = cut(Path(day_clip["path"]), at=20.0, seconds=8.0, lead_in=2.0)
    times = probe(data)
    assert times[-1] >= 1.0, "too short to contain anything"


def test_the_length_is_capped(day_clip):
    """A request for ten minutes is somebody using this as a download
    endpoint. The whole file is already available from /api/media."""
    data = cut(Path(day_clip["path"]), at=0.0, seconds=MAX_SECONDS * 10)
    assert probe(data)[-1] <= MAX_SECONDS + 5


def test_a_missing_recording_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        cut(tmp_path / "gone.mp4", at=1.0)


def test_something_that_is_not_video_is_refused(tmp_path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"nonsense" * 100)
    with pytest.raises((ValueError, av.FFmpegError, OSError)):
        cut(junk, at=1.0)


# ---------- over HTTP ----------

def test_the_endpoint_returns_a_playable_clip(clip_client):
    response = clip_client.get("/api/clip/1?t=20&seconds=6")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert int(response.headers["content-length"]) == len(response.content)
    assert probe(response.content), "the bytes are not a video"


def test_a_moment_past_the_end_gives_the_end_not_three_frames(clip_client):
    """Unclamped, seeking past the end lands on the last keyframe and returns
    whatever follows it - which was a 0.3 second clip."""
    response = clip_client.get("/api/clip/1?t=99999&seconds=6")
    assert response.status_code == 200
    assert probe(response.content)[-1] > 0.5


def test_an_unknown_recording_is_404(clip_client):
    assert clip_client.get("/api/clip/9999?t=1").status_code == 404


def test_an_absurd_length_is_rejected_by_the_endpoint(clip_client):
    assert clip_client.get("/api/clip/1?t=1&seconds=600").status_code == 422
    assert clip_client.get("/api/clip/1?t=1&seconds=0").status_code == 422


# ---------- the H.264 fallback ----------
#
# Phones record HEVC and play HEVC, so copying is right by default. Desktop
# Firefox does not play it, and canPlayType is optimistic enough that a
# browser can claim HEVC and then fail on a real file - so a client has to be
# able to ask for something it is certain of.

def test_asking_for_web_safe_output_from_an_h264_source_costs_nothing(day_clip):
    """The sample footage is already H.264, so the flag must fall through to
    the copy path rather than re-encoding for no reason."""
    from tsv.clips import source_codec

    assert source_codec(Path(day_clip["path"])) == "h264"

    copied = cut(Path(day_clip["path"]), at=15.0, seconds=6.0)
    safe = cut(Path(day_clip["path"]), at=15.0, seconds=6.0, web_safe=True)
    assert safe == copied, "an H.264 source was needlessly re-encoded"


def test_a_frame_rate_an_encoder_would_reject_is_cleaned_up():
    """Real recordings report absurd rates.

    This project's phone footage claims 26996000/1799749 - which is 15 fps to
    six decimal places, and which becomes the encoder's time base denominator
    where libx264 rejects it with a bare "Invalid argument" naming nothing.
    """
    from fractions import Fraction

    from tsv.clips import _clean_rate

    assert _clean_rate(Fraction(26996000, 1799749)) == Fraction(15, 1)

    # NTSC rates are spelled this way on purpose and must survive intact.
    assert _clean_rate(Fraction(30000, 1001)) == Fraction(30000, 1001)
    assert _clean_rate(Fraction(24000, 1001)) == Fraction(24000, 1001)

    # Nonsense falls back rather than raising.
    for bad in (None, 0, -5, 100000):
        assert _clean_rate(bad) == Fraction(30, 1)


def test_the_endpoint_accepts_the_h264_flag(clip_client):
    plain = clip_client.get("/api/clip/1?t=20&seconds=6")
    forced = clip_client.get("/api/clip/1?t=20&seconds=6&h264=1")
    assert plain.status_code == forced.status_code == 200
    assert probe(forced.content), "the fallback did not produce a video"


def test_web_safe_covers_what_a_browser_can_decode():
    """The list is what decides whether a re-encode happens at all."""
    from tsv.clips import WEB_SAFE

    assert "h264" in WEB_SAFE
    assert "hevc" not in WEB_SAFE, "HEVC would never be converted"
