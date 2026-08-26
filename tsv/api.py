"""Local HTTP API and static host for the timeline UI.

Deliberately a web app served over localhost rather than a native window: the
Android companion planned for a later phase is then a thin client against
these same endpoints rather than a second implementation.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse

from tsv import db
from tsv.config import DEFAULT, Config

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# One bucket per minute across a 24h day.
TIMELINE_BUCKETS = 24 * 60


def _day_bounds(day: date) -> tuple[float, float]:
    """Local-midnight bounds for a day, as unix timestamps."""
    start = datetime.combine(day, dtime.min)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _spread(
    buckets: list[float],
    bucket_seconds: float,
    day_start: float,
    span_start: float,
    span_end: float,
    value: float | None = None,
) -> None:
    """Accumulate a time span into per-minute buckets.

    With `value` given the buckets hold the maximum of that value; otherwise
    they hold covered seconds.
    """
    lo = max(span_start, day_start)
    hi = min(span_end, day_start + len(buckets) * bucket_seconds)
    if hi <= lo:
        return
    first = int((lo - day_start) // bucket_seconds)
    last = min(len(buckets) - 1, int((hi - day_start - 1e-9) // bucket_seconds))
    for i in range(max(0, first), last + 1):
        b_lo = day_start + i * bucket_seconds
        b_hi = b_lo + bucket_seconds
        overlap = min(hi, b_hi) - max(lo, b_lo)
        if overlap <= 0:
            continue
        if value is None:
            buckets[i] += overlap
        else:
            buckets[i] = max(buckets[i], value)


def create_app(cfg: Config = DEFAULT) -> FastAPI:
    app = FastAPI(title="TextSearchVDO", version="0.1.0")
    conn: sqlite3.Connection = db.open_db(cfg.db_path)

    def _camera_filter(camera_id: int | None) -> tuple[str, list]:
        return ("AND camera_id = ?", [camera_id]) if camera_id else ("", [])

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/static/{name}")
    def static(name: str) -> FileResponse:
        # Flat directory, no nesting: reject anything with path structure.
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(404)
        path = WEB_DIR / name
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/api/summary")
    def summary() -> dict:
        row = conn.execute(
            """SELECT COUNT(*) AS n_videos,
                      COALESCE(SUM(duration), 0) AS duration,
                      COALESCE(SUM(active_seconds), 0) AS active
               FROM videos"""
        ).fetchone()
        n_segments = conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()["n"]
        duration = float(row["duration"])
        active = float(row["active"])
        return {
            "n_videos": row["n_videos"],
            "n_segments": n_segments,
            "duration": duration,
            "active_seconds": active,
            "reduction": (1.0 - active / duration) if duration else 0.0,
        }

    @app.get("/api/cameras")
    def cameras() -> list[dict]:
        rows = conn.execute(
            """SELECT c.id, c.name,
                      COUNT(DISTINCT v.id) AS n_videos,
                      COALESCE(SUM(v.duration), 0) AS duration
               FROM cameras c LEFT JOIN videos v ON v.camera_id = c.id
               GROUP BY c.id ORDER BY c.name"""
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/days")
    def days(camera_id: int | None = None) -> list[dict]:
        where, params = _camera_filter(camera_id)
        rows = conn.execute(
            f"""SELECT date(ts_start, 'unixepoch', 'localtime') AS day,
                       COUNT(*) AS n_segments,
                       COALESCE(SUM(ts_end - ts_start), 0) AS active
                FROM segments WHERE 1=1 {where}
                GROUP BY day ORDER BY day DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/timeline")
    def timeline(day: str, camera_id: int | None = None) -> dict:
        try:
            target = date.fromisoformat(day)
        except ValueError:
            raise HTTPException(400, "day must be YYYY-MM-DD") from None
        day_start, day_end = _day_bounds(target)
        bucket_seconds = 86400 / TIMELINE_BUCKETS
        where, params = _camera_filter(camera_id)

        activity = [0.0] * TIMELINE_BUCKETS
        coverage = [0.0] * TIMELINE_BUCKETS

        seg_rows = conn.execute(
            f"""SELECT id, video_id, camera_id, t_start, t_end, ts_start, ts_end,
                       activity_score, peak_offset, thumb_path
                FROM segments
                WHERE ts_end > ? AND ts_start < ? {where}
                ORDER BY ts_start""",
            [day_start, day_end, *params],
        ).fetchall()
        for r in seg_rows:
            _spread(activity, bucket_seconds, day_start, r["ts_start"], r["ts_end"],
                    value=float(r["activity_score"]))

        vid_rows = conn.execute(
            f"""SELECT start_ts, duration FROM videos
                WHERE start_ts + duration > ? AND start_ts < ? {where}""",
            [day_start, day_end, *params],
        ).fetchall()
        for r in vid_rows:
            _spread(coverage, bucket_seconds, day_start, r["start_ts"],
                    r["start_ts"] + r["duration"])

        return {
            "day": day,
            "day_start": day_start,
            "bucket_seconds": bucket_seconds,
            # Recorded seconds per bucket: lets the UI show "camera was
            # offline" differently from "camera saw nothing".
            "coverage": coverage,
            "activity": activity,
            "segments": [dict(r) for r in seg_rows],
        }

    @app.get("/api/segments")
    def segments(
        day: str | None = None,
        camera_id: int | None = None,
        limit: int = Query(200, le=2000),
        offset: int = 0,
    ) -> list[dict]:
        where, params = _camera_filter(camera_id)
        time_clause = ""
        if day:
            try:
                start, end = _day_bounds(date.fromisoformat(day))
            except ValueError:
                raise HTTPException(400, "day must be YYYY-MM-DD") from None
            time_clause = "AND ts_end > ? AND ts_start < ?"
            params = [*params, start, end]
        rows = conn.execute(
            f"""SELECT s.*, v.path, v.start_ts AS video_start
                FROM segments s JOIN videos v ON v.id = s.video_id
                WHERE 1=1 {where} {time_clause}
                ORDER BY s.ts_start LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/thumb/{segment_id}")
    def thumb(segment_id: int) -> FileResponse:
        row = conn.execute(
            "SELECT thumb_path FROM segments WHERE id = ?", (segment_id,)
        ).fetchone()
        if row is None or not row["thumb_path"]:
            raise HTTPException(404)
        path = Path(row["thumb_path"])
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/media/{video_id}")
    def media(video_id: int) -> FileResponse:
        """Serve the source file. FileResponse honours Range, which is what
        lets the browser seek instead of downloading the whole recording."""
        row = conn.execute("SELECT path FROM videos WHERE id = ?", (video_id,)).fetchone()
        if row is None:
            raise HTTPException(404)
        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(410, "source file has moved or been deleted")
        return FileResponse(path)

    return app


app = create_app()
