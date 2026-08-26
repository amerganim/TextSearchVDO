"""One call that takes a video file and makes it searchable.

The CLI exposes ingest, analyze and reindex separately because they are
separate concerns and each is worth running alone. Someone who has just
dragged a file onto a window does not care: they want the video searchable,
and they want to see it happening.

Videos are indexed **where they already are**. Copying a night of 1080p
footage into an application folder would double the disk it occupies for no
benefit, and people keep recordings where they keep them on purpose. Only
files arriving over HTTP - a browser upload, where there is no path to point
at - get written to a staging directory.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tsv.analyze import analyze_all
from tsv.config import Config
from tsv.events import recompute_events
from tsv.ingest import ingest_path
from tsv.jobs import Reporter
from tsv.probe import iter_videos
from tsv.search import rebuild_text_index

# Roughly what each stage costs relative to the others, from the measured
# rates: the packet scan runs at hundreds of times realtime, detection at
# about fifteen frames a second. Detection dominates and the bar should say so.
STAGE_SHARES = {"ingest": 0.15, "analyze": 0.75, "index": 0.10}


@dataclass
class ImportResult:
    files: int = 0
    segments: int = 0
    tracklets: int = 0
    duration: float = 0.0
    active: float = 0.0
    faces: int = 0
    skipped: int = 0
    failed: list[str] | None = None

    @property
    def reduction(self) -> float:
        return (1.0 - self.active / self.duration) if self.duration else 0.0

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "segments": self.segments,
            "tracklets": self.tracklets,
            "duration": round(self.duration, 1),
            "active": round(self.active, 1),
            "reduction": round(self.reduction, 4),
            "faces": self.faces,
            "skipped": self.skipped,
            "failed": self.failed or [],
        }


def stage_video(source: Path, staging_dir: Path) -> Path:
    """Copy an uploaded file into the staging area, without clobbering."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / source.name
    counter = 1
    while target.exists():
        target = staging_dir / f"{source.stem}-{counter}{source.suffix}"
        counter += 1
    shutil.move(str(source), target)
    return target


def import_videos(
    conn: sqlite3.Connection,
    path: Path,
    cfg: Config,
    report: Reporter | None = None,
    force: bool = False,
) -> ImportResult:
    """Ingest, analyse and index a file or folder, reporting progress."""
    result = ImportResult(failed=[])
    videos = iter_videos(path)
    if not videos:
        raise ValueError(f"no video files found in {path}")

    # ---- motion segmentation ----
    if report:
        report.stage("Finding activity", STAGE_SHARES["ingest"],
                     f"scanning {len(videos)} file(s)")
    done = 0

    def on_ingest(item) -> None:
        nonlocal done
        done += 1
        if item.status == "failed":
            result.failed.append(f"{item.path.name}: {item.note}")
        elif item.status == "skipped":
            result.skipped += 1
        else:
            result.files += 1
            result.segments += item.n_segments
            result.duration += item.duration
            result.active += item.active_seconds
        if report:
            report.step(done / len(videos), f"{item.path.name}: {item.status}")

    ingest_path(conn, path, cfg, force=force, on_result=on_ingest)

    # ---- detection, faces, embeddings ----
    pending = conn.execute(
        "SELECT COUNT(DISTINCT video_id) AS n FROM segments WHERE analyzed_at IS NULL"
    ).fetchone()["n"]

    if report:
        report.stage(
            "Recognising objects", STAGE_SHARES["analyze"],
            f"{pending} file(s) to analyse" if pending else "nothing new to analyse",
        )

    # The bar is driven by seconds of footage examined across every pending
    # file, so it moves continuously rather than once per file or per segment.
    total_seconds = conn.execute(
        "SELECT COALESCE(SUM(t_end - t_start), 0) AS s FROM segments "
        "WHERE analyzed_at IS NULL"
    ).fetchone()["s"] or 1.0

    if pending:
        seen = 0
        seconds_before = 0.0

        def on_progress(done: float, total: float) -> None:
            if not report:
                return
            overall = min(1.0, (seconds_before + done) / total_seconds)
            report.step(
                overall,
                f"{int(seconds_before + done)}s of {int(total_seconds)}s examined",
            )

        def on_analyze(item) -> None:
            nonlocal seen, seconds_before
            seen += 1
            seconds_before += item.analysed_seconds
            if item.status == "analyzed":
                result.tracklets += item.n_tracklets
                result.faces += item.n_faces
            elif item.status == "failed":
                result.failed.append(f"{item.path.name}: {item.note}")
            if report:
                report.say(f"{item.path.name}: {item.n_tracklets} object(s) found")

        try:
            analyze_all(conn, cfg, force=force, on_result=on_analyze,
                        on_progress=on_progress)
        except FileNotFoundError as exc:
            # No detector present. Motion segmentation still worked, and the
            # timeline is usable; say so rather than failing the whole import.
            result.failed.append(f"detection skipped: {exc}")

    # ---- zones and the word index ----
    if report:
        report.stage("Building the index", STAGE_SHARES["index"], "")
    recompute_events(conn)
    if report:
        report.step(0.5, "")
    rebuild_text_index(conn)
    if report:
        report.step(1.0, "ready")

    return result
