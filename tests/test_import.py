"""Importing a video the way the app does it, and reporting progress."""

from __future__ import annotations

import dataclasses
import shutil
import time
from pathlib import Path

import pytest

from tsv import db
from tsv.config import DEFAULT
from tsv.importer import import_videos, stage_video
from tsv.jobs import JobRunner


@pytest.fixture
def cfg(tmp_path):
    return dataclasses.replace(
        DEFAULT,
        data_dir=tmp_path / "data",
        detect=dataclasses.replace(DEFAULT.detect, detect_fps=4.0, decode_width=320),
    )


# ---------- jobs ----------

def test_a_job_reports_progress_and_finishes():
    runner = JobRunner()

    def work(report):
        report.stage("first", 0.5, "starting")
        report.step(1.0)
        report.stage("second", 0.5)
        report.step(1.0)
        return {"ok": True}

    job = runner.submit("test", "demo", work)
    for _ in range(200):
        if job.status in ("done", "failed"):
            break
        time.sleep(0.02)

    assert job.status == "done"
    assert job.progress == 1.0
    assert job.result == {"ok": True}


def test_a_failing_job_surfaces_its_error_rather_than_hanging():
    runner = JobRunner()

    def work(report):
        raise RuntimeError("the disk fell over")

    job = runner.submit("test", "demo", work)
    for _ in range(200):
        if job.status in ("done", "failed"):
            break
        time.sleep(0.02)

    assert job.status == "failed"
    assert "the disk fell over" in job.error
    assert job.finished is not None


def test_progress_never_exceeds_one_however_it_is_driven():
    runner = JobRunner()

    def work(report):
        report.stage("only", 1.0)
        for _ in range(5):
            report.step(9.0)          # nonsense input
        return {}

    job = runner.submit("test", "demo", work)
    for _ in range(200):
        if job.status == "done":
            break
        time.sleep(0.02)
    assert job.progress == 1.0


def test_jobs_are_listed_newest_first():
    runner = JobRunner()
    for i in range(3):
        runner.submit("test", f"job{i}", lambda report: {})
    for _ in range(200):
        if not runner.active():
            break
        time.sleep(0.02)
    assert [j.title for j in runner.all()] == ["job2", "job1", "job0"]


# ---------- importing ----------

def test_import_makes_a_video_searchable_in_one_call(day_clip, cfg):
    conn = db.open_db(cfg.db_path)
    result = import_videos(conn, Path(day_clip["path"]), cfg)

    assert result.files == 1
    assert result.segments > 0
    assert result.duration > 0
    assert 0.0 < result.reduction < 1.0

    # The word index is built as part of the import, not as a separate step
    # the user has to know about.
    indexed = conn.execute("SELECT COUNT(*) c FROM segment_text").fetchone()["c"]
    assert indexed > 0


def test_import_reports_each_stage(day_clip, cfg):
    conn = db.open_db(cfg.db_path)
    runner = JobRunner()
    seen: list[str] = []

    class Watcher:
        def __init__(self, reporter):
            self._r = reporter

        def stage(self, name, share, message=""):
            seen.append(name)
            self._r.stage(name, share, message)

        def step(self, fraction, message=""):
            self._r.step(fraction, message)

        def say(self, message):
            self._r.say(message)

    job = runner.submit(
        "import", "clip",
        lambda report: import_videos(conn, Path(day_clip["path"]), cfg,
                                     report=Watcher(report)).as_dict(),
    )
    for _ in range(3000):
        if job.status in ("done", "failed"):
            break
        time.sleep(0.02)

    assert job.status == "done", job.error
    assert seen == ["Finding activity", "Recognising objects", "Building the index"]
    assert job.result["segments"] > 0


def test_importing_the_same_file_twice_skips_it(day_clip, cfg):
    conn = db.open_db(cfg.db_path)
    import_videos(conn, Path(day_clip["path"]), cfg)
    again = import_videos(conn, Path(day_clip["path"]), cfg)
    assert again.skipped == 1
    assert again.files == 0


def test_importing_a_folder_takes_every_video(day_clip, idle_clip, cfg, tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    for clip in (day_clip, idle_clip):
        shutil.copy2(clip["path"], folder / Path(clip["path"]).name)

    conn = db.open_db(cfg.db_path)
    result = import_videos(conn, folder, cfg)
    assert result.files == 2


def test_importing_something_that_is_not_video_is_refused_clearly(cfg, tmp_path):
    conn = db.open_db(cfg.db_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no video files"):
        import_videos(conn, empty, cfg)


def test_a_corrupt_file_is_reported_not_fatal(cfg, tmp_path):
    conn = db.open_db(cfg.db_path)
    junk = tmp_path / "ch01_20260101000000.mp4"
    junk.write_bytes(b"not a video" * 200)

    result = import_videos(conn, junk, cfg)
    assert result.files == 0
    assert result.failed


def test_staging_a_file_does_not_clobber_an_existing_one(tmp_path):
    staging = tmp_path / "incoming"
    staging.mkdir()
    (staging / "clip.mp4").write_bytes(b"original")

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"new")
    landed = stage_video(source, staging)

    assert landed.name != "clip.mp4"
    assert (staging / "clip.mp4").read_bytes() == b"original"
    assert landed.read_bytes() == b"new"


def test_a_finished_job_has_a_finish_time_the_moment_it_reports_done():
    """Status and finish time must land together.

    A poller that sees "done" with no finish time shows an elapsed counter
    that keeps climbing after the work stopped.
    """
    runner = JobRunner()
    for outcome in (lambda report: {}, lambda report: (_ for _ in ()).throw(RuntimeError("x"))):
        job = runner.submit("test", "demo", outcome)
        for _ in range(300):
            if job.status in ("done", "failed"):
                assert job.finished is not None, job.status
                break
            time.sleep(0.01)
        else:
            raise AssertionError("job never finished")
