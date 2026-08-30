"""Ingest orchestration: a folder of video files becomes a searchable timeline.

Ingest is idempotent. A file already in the index at the same size and mtime
is skipped; one that changed has its segments replaced. Re-running over a
growing NVR export directory is therefore cheap and safe.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tsv import db, probe
from tsv.config import Config
from tsv.motion.decode import trace_windows
from tsv.motion.packets import scan
from tsv.motion.segments import Segment, activity_to_segments
from tsv.thumbs import extract_thumbs


@dataclass
class IngestResult:
    path: Path
    status: str  # ingested | skipped | duplicate | failed
    duration: float = 0.0
    active_seconds: float = 0.0
    n_segments: int = 0
    candidate_seconds: float = 0.0
    elapsed: float = 0.0
    ts_source: str = ""
    note: str = ""

    @property
    def compression(self) -> float:
        """How much of the recording a reviewer no longer has to watch."""
        if self.duration <= 0:
            return 0.0
        return 1.0 - (self.active_seconds / self.duration)


@dataclass
class IngestSummary:
    results: list[IngestResult] = field(default_factory=list)

    @property
    def ingested(self) -> list[IngestResult]:
        return [r for r in self.results if r.status == "ingested"]

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.ingested)

    @property
    def total_active(self) -> float:
        return sum(r.active_seconds for r in self.ingested)

    @property
    def total_segments(self) -> int:
        return sum(r.n_segments for r in self.ingested)


def _same_recording(conn: sqlite3.Connection, info: probe.VideoInfo) -> str | None:
    """The path this recording is already indexed under, if any.

    Matched on content rather than path. Uploading a file the browser has
    already sent lands it under a fresh name in the staging directory, and
    without this every repeat upload became another library entry with its
    own copy of the same hours.
    """
    if not info.fingerprint:
        return None
    row = conn.execute(
        "SELECT path FROM videos WHERE fingerprint = ? AND path <> ? "
        "AND ingested_at IS NOT NULL LIMIT 1",
        (info.fingerprint, str(info.path)),
    ).fetchone()
    return str(row["path"]) if row else None


def _already_ingested(conn: sqlite3.Connection, info: probe.VideoInfo) -> int | None:
    row = conn.execute(
        "SELECT id, mtime, size_bytes, ingested_at FROM videos WHERE path = ?",
        (str(info.path),),
    ).fetchone()
    if row is None:
        return None
    unchanged = (
        row["ingested_at"] is not None
        and row["size_bytes"] == info.size_bytes
        and abs((row["mtime"] or 0) - info.mtime) < 1e-6
    )
    return int(row["id"]) if unchanged else None


def _even_windows(duration: float, window: float) -> list[Segment]:
    """Cut a whole file into equal pieces, for when nothing stood out.

    One segment spanning everything would technically work and be useless:
    every search returns the same result, ranking has a single candidate to
    order, and "when did this happen" is answered with the length of the
    recording. Windows give search something to choose between and a moment
    to point at.
    """
    window = max(1.0, float(window))
    if duration <= window:
        return [Segment(0.0, duration, 0.0, duration / 2.0)]

    edges = []
    start = 0.0
    while start < duration - 0.01:
        end = min(start + window, duration)
        # Fold a final sliver into the one before it rather than leaving a
        # one second segment nobody wants to see in a result list.
        if duration - end < window * 0.4:
            end = duration
        edges.append(Segment(start, end, 0.0, (start + end) / 2.0))
        start = end
    return edges


def ingest_file(
    conn: sqlite3.Connection,
    path: Path,
    cfg: Config,
    force: bool = False,
) -> IngestResult:
    started = time.time()
    try:
        info = probe.probe(path)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        return IngestResult(path=path, status="failed", note=f"{type(exc).__name__}: {exc}")

    if not force and _already_ingested(conn, info) is not None:
        return IngestResult(path=path, status="skipped", ts_source=info.ts_source)

    if not force:
        seen = _same_recording(conn, info)
        if seen is not None:
            return IngestResult(
                path=path,
                status="duplicate",
                ts_source=info.ts_source,
                note=f"already indexed as {Path(seen).name}",
            )

    try:
        packet_scan = scan(path, cfg.tier_a, info.duration)
        duration = packet_scan.duration or info.duration
        trace = trace_windows(path, packet_scan.candidates, cfg.tier_b)
        sample_period = trace.sample_period or (1.0 / cfg.tier_b.sample_fps)
        segments = activity_to_segments(
            trace.times, trace.scores, cfg.segments, sample_period, duration=duration
        )

        # Nothing stood out. On a short file that is far more likely to mean
        # the motion gate had no contrast to work with - a handheld camera,
        # where everything moves and nothing is background - than that the
        # recording is empty. Looking properly costs seconds at this length,
        # and not looking costs the user a video they cannot search at all.
        whole_file = False
        if not segments and duration <= cfg.segments.whole_file_under_seconds:
            segments = _even_windows(
                duration, cfg.segments.whole_file_window_seconds
            )
            whole_file = True
    except Exception as exc:  # noqa: BLE001
        return IngestResult(path=path, status="failed", note=f"{type(exc).__name__}: {exc}")

    camera_id = db.get_or_create_camera(conn, info.camera)

    # Replaces any previous ingest of this path; segments cascade away with it.
    conn.execute("DELETE FROM videos WHERE path = ?", (str(path),))
    cur = conn.execute(
        """INSERT INTO videos(camera_id, path, start_ts, ts_source, duration, fps,
                              width, height, codec, size_bytes, mtime, ingested_at,
                              active_seconds, fingerprint)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            camera_id, str(path), info.start_ts, info.ts_source, duration, info.fps,
            info.width, info.height, info.codec, info.size_bytes, info.mtime,
            time.time(), sum(s.duration for s in segments), info.fingerprint,
        ),
    )
    video_id = int(cur.lastrowid)

    thumb_dir = cfg.thumb_dir / str(video_id)
    thumb_paths = [thumb_dir / f"{i:05d}.jpg" for i in range(len(segments))]
    written = extract_thumbs(
        path,
        [s.peak_offset for s in segments],
        thumb_paths,
        width=cfg.thumb_width,
        quality=cfg.thumb_quality,
    )

    conn.executemany(
        """INSERT INTO segments(video_id, camera_id, t_start, t_end, ts_start, ts_end,
                                activity_score, peak_offset, thumb_path)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [
            (
                video_id, camera_id, s.t_start, s.t_end,
                info.start_ts + s.t_start, info.start_ts + s.t_end,
                s.activity_score, s.peak_offset,
                str(thumb) if thumb else None,
            )
            for s, thumb in zip(segments, written)
        ],
    )
    conn.commit()

    note = ""
    if packet_scan.all_intra:
        note = "all-intra stream: packet prefilter skipped, decoded in full"
    elif info.ts_source == "mtime-duration":
        note = "start time guessed from file mtime; timeline may be off"

    note = ""
    if whole_file:
        note = (
            "no activity stood out, so the whole file was indexed - usual for "
            "a handheld recording, where nothing is background"
        )

    return IngestResult(
        path=path,
        status="ingested",
        duration=duration,
        active_seconds=sum(s.duration for s in segments),
        n_segments=len(segments),
        candidate_seconds=packet_scan.candidate_seconds,
        elapsed=time.time() - started,
        ts_source=info.ts_source,
        note=note,
    )


def ingest_path(
    conn: sqlite3.Connection,
    root: Path,
    cfg: Config,
    force: bool = False,
    on_result: Callable[[IngestResult], None] | None = None,
) -> IngestSummary:
    summary = IngestSummary()
    for path in probe.iter_videos(root):
        result = ingest_file(conn, path, cfg, force=force)
        summary.results.append(result)
        if on_result:
            on_result(result)
    return summary
