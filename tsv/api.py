"""Local HTTP API and static host for the timeline UI.

Deliberately a web app served over localhost rather than a native window: the
Android companion planned for a later phase is then a thin client against
these same endpoints rather than a second implementation.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

from tsv import db
from tsv.config import DEFAULT, Config
from tsv.events import create_zone, delete_zone, list_zones, recompute_events
from tsv.identity import (
    assign_identities, delete_identity, enroll_tracklet, list_identities,
)
from tsv.models.clip import build_clip
from tsv.search import SearchFilters, rebuild_text_index, search as run_search


class EnrollIn(BaseModel):
    tracklet_id: int
    name: str = Field(min_length=1, max_length=64)


class ZoneIn(BaseModel):
    camera_id: int
    name: str = Field(min_length=1, max_length=64)
    kind: str = Field(pattern="^(region|line)$")
    points: list[tuple[float, float]] = Field(min_length=2, max_length=64)

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
    # One handle per worker thread; see db.ThreadLocalConnection.
    conn = db.open_threadlocal(cfg.db_path)

    # The text encoder is loaded once, lazily: it is only needed when someone
    # actually types something, and loading it costs a second or two.
    text_encoder: dict[str, object] = {}

    def _query_vector(text: str):
        if not text.strip() or not cfg.has_clip_models:
            return None
        if "clip" not in text_encoder:
            text_encoder["clip"] = build_clip(
                cfg.model_dir, cfg.clip.image_file, cfg.clip.text_file,
                crop_mode=cfg.clip.crop_mode, force_backend=cfg.clip.force_backend,
            )
        clip = text_encoder["clip"]
        return clip.embed_text(text) if clip is not None else None

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
        counts = conn.execute(
            """SELECT COUNT(*) AS n_segments,
                      SUM(CASE WHEN analyzed_at IS NOT NULL THEN 1 ELSE 0 END) AS n_analyzed
               FROM segments"""
        ).fetchone()
        n_tracklets = conn.execute("SELECT COUNT(*) AS n FROM tracklets").fetchone()["n"]
        top = conn.execute(
            """SELECT label, COUNT(*) AS n FROM tracklets
               GROUP BY label ORDER BY n DESC LIMIT 8"""
        ).fetchall()
        duration = float(row["duration"])
        active = float(row["active"])
        return {
            "n_videos": row["n_videos"],
            "n_segments": counts["n_segments"],
            "n_analyzed": counts["n_analyzed"] or 0,
            "n_tracklets": n_tracklets,
            "n_identities": conn.execute(
                "SELECT COUNT(*) AS n FROM identities"
            ).fetchone()["n"],
            "n_named": conn.execute(
                "SELECT COUNT(*) AS n FROM tracklets WHERE identity_id IS NOT NULL"
            ).fetchone()["n"],
            "n_embedded": conn.execute(
                "SELECT COUNT(*) AS n FROM segment_embeddings WHERE kind = 'clip'"
            ).fetchone()["n"],
            "semantic_ready": cfg.has_clip_models,
            "duration": duration,
            "active_seconds": active,
            "reduction": (1.0 - active / duration) if duration else 0.0,
            "top_labels": [dict(r) for r in top],
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
                       activity_score, peak_offset, thumb_path, labels, n_tracklets,
                       analyzed_at
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

    @app.get("/api/labels")
    def labels(day: str | None = None, camera_id: int | None = None) -> list[dict]:
        """Which object kinds appear, and how often. Drives the UI filter."""
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
            f"""SELECT label, COUNT(*) AS n FROM tracklets
                WHERE 1=1 {where} {time_clause}
                GROUP BY label ORDER BY n DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/objects")
    def objects(
        day: str | None = None,
        label: str | None = None,
        camera_id: int | None = None,
        segment_id: int | None = None,
        min_score: float = 0.0,
        limit: int = Query(300, le=2000),
        offset: int = 0,
    ) -> list[dict]:
        """Tracked objects in time order, for the object view."""
        clauses: list[str] = []
        params: list = []
        if camera_id:
            clauses.append("AND t.camera_id = ?")
            params.append(camera_id)
        if segment_id:
            clauses.append("AND t.segment_id = ?")
            params.append(segment_id)
        if day:
            try:
                start, end = _day_bounds(date.fromisoformat(day))
            except ValueError:
                raise HTTPException(400, "day must be YYYY-MM-DD") from None
            clauses.append("AND t.ts_end > ? AND t.ts_start < ?")
            params += [start, end]
        if label:
            clauses.append("AND t.label = ?")
            params.append(label)
        if min_score > 0:
            clauses.append("AND t.max_score >= ?")
            params.append(min_score)

        rows = conn.execute(
            f"""SELECT t.*, s.t_start AS segment_start, v.path,
                       i.name AS identity_name
                FROM tracklets t
                JOIN segments s ON s.id = t.segment_id
                JOIN videos v ON v.id = t.video_id
                LEFT JOIN identities i ON i.id = t.identity_id
                WHERE 1=1 {" ".join(clauses)}
                ORDER BY t.ts_start LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/search")
    def search_endpoint(
        q: str = "",
        day: str | None = None,
        camera_id: int | None = None,
        identity: str | None = None,
        zone: str | None = None,
        label: str | None = None,
        event_kind: str | None = None,
        semantic: bool = True,
        min_similarity: float | None = None,
        limit: int = Query(40, le=200),
    ) -> dict:
        """Search by text, by filters, or by both."""
        if day:
            try:
                date.fromisoformat(day)
            except ValueError:
                raise HTTPException(400, "day must be YYYY-MM-DD") from None

        filters = SearchFilters(
            day=day, camera_id=camera_id, identity=identity,
            zone=zone, label=label, event_kind=event_kind,
        )
        vector = _query_vector(q) if semantic else None
        hits = run_search(conn, text=q, query_vector=vector, filters=filters,
                          limit=limit, min_similarity=min_similarity)

        return {
            "query": q,
            "semantic": vector is not None,
            "n": len(hits),
            "results": [
                {
                    "segment_id": h.segment_id, "score": h.score,
                    "ts_start": h.ts_start, "ts_end": h.ts_end,
                    "video_id": h.video_id, "camera_id": h.camera_id,
                    "t_start": h.t_start, "t_end": h.t_end,
                    "labels": h.labels, "sources": h.sources,
                    "semantic_score": h.semantic_score,
                    "tracklet_id": h.best_tracklet_id,
                }
                for h in hits
            ],
        }

    @app.post("/api/search/reindex")
    def reindex() -> dict:
        return {"indexed": rebuild_text_index(conn)}

    @app.get("/api/identities")
    def identities() -> list[dict]:
        return list_identities(conn)

    @app.post("/api/identities/enroll")
    def enroll(body: EnrollIn) -> dict:
        exists = conn.execute(
            "SELECT id FROM tracklets WHERE id = ?", (body.tracklet_id,)
        ).fetchone()
        if exists is None:
            raise HTTPException(404, "no such tracklet")
        try:
            identity, added = enroll_tracklet(conn, body.tracklet_id, body.name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {
            "identity_id": identity.id, "name": identity.name,
            "examples_added": added,
            # Zero means the tracklet had no embedding yet - the sighting is
            # still labelled, but it teaches the gallery nothing.
            "note": "" if added else "no embeddings stored for this tracklet yet",
        }

    @app.delete("/api/identities/{identity_id}")
    def remove_identity(identity_id: int) -> dict:
        if not delete_identity(conn, identity_id):
            raise HTTPException(404)
        return {"deleted": identity_id}

    @app.post("/api/identities/assign")
    def assign(
        kind: str = Query("face", pattern="^(face|body)$"),
        threshold: float | None = None,
        margin: float | None = None,
        reassign: bool = False,
    ) -> dict:
        summary = assign_identities(conn, kind, threshold, margin, reassign)
        return {
            "kind": kind,
            "considered": summary.n_considered,
            "assigned": summary.n_assigned,
            "ambiguous": summary.n_ambiguous,
            "below_threshold": summary.n_below_threshold,
            "by_name": summary.by_name,
        }

    @app.get("/api/zones")
    def zones(camera_id: int | None = None) -> list[dict]:
        return [
            {"id": z.id, "camera_id": z.camera_id, "name": z.name,
             "kind": z.kind, "points": z.points}
            for z in list_zones(conn, camera_id)
        ]

    @app.post("/api/zones")
    def add_zone(zone: ZoneIn) -> dict:
        try:
            made = create_zone(conn, zone.camera_id, zone.name, zone.kind, zone.points)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"a zone named {zone.name!r} already exists") from None
        # Events for the new zone are available immediately; this costs no
        # video decoding, only the boxes already in the index.
        summary = recompute_events(conn, made.camera_id)
        return {
            "id": made.id, "camera_id": made.camera_id, "name": made.name,
            "kind": made.kind, "points": made.points, "n_events": summary.n_events,
        }

    @app.delete("/api/zones/{zone_id}")
    def remove_zone(zone_id: int) -> dict:
        if not delete_zone(conn, zone_id):
            raise HTTPException(404)
        return {"deleted": zone_id}

    @app.post("/api/zones/recompute")
    def recompute(camera_id: int | None = None) -> dict:
        summary = recompute_events(conn, camera_id)
        return {
            "n_zones": summary.n_zones,
            "n_tracklets": summary.n_tracklets,
            "n_events": summary.n_events,
            "by_kind": summary.by_kind,
            "elapsed": summary.elapsed,
        }

    @app.get("/api/events")
    def events(
        day: str | None = None,
        zone_id: int | None = None,
        kind: str | None = None,
        label: str | None = None,
        camera_id: int | None = None,
        identity: str | None = None,
        limit: int = Query(300, le=2000),
        offset: int = 0,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
        if identity:
            clauses.append("AND i.name = ?")
            params.append(identity)
        for column, value in (
            ("e.zone_id", zone_id), ("e.kind", kind),
            ("e.label", label), ("e.camera_id", camera_id),
        ):
            if value is not None:
                clauses.append(f"AND {column} = ?")
                params.append(value)
        if day:
            try:
                start, end = _day_bounds(date.fromisoformat(day))
            except ValueError:
                raise HTTPException(400, "day must be YYYY-MM-DD") from None
            clauses.append("AND e.ts >= ? AND e.ts < ?")
            params += [start, end]

        rows = conn.execute(
            f"""SELECT e.*, z.name AS zone_name, t.thumb_path IS NOT NULL AS has_crop,
                       s.t_start AS segment_start, i.name AS identity_name,
                       t.identity_score
                FROM events e
                JOIN zones z ON z.id = e.zone_id
                JOIN tracklets t ON t.id = e.tracklet_id
                JOIN segments s ON s.id = e.segment_id
                LEFT JOIN identities i ON i.id = t.identity_id
                WHERE 1=1 {" ".join(clauses)}
                ORDER BY e.ts LIMIT ? OFFSET ?""",
            [*params, limit, offset],
        ).fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/frame/{camera_id}")
    def frame(camera_id: int, width: int = Query(960, le=1920)) -> Response:
        """A representative still from a camera, to draw zones on.

        The busiest segment's peak moment: a frame with something in it makes
        the scene easier to read than an empty corridor would.
        """
        row = conn.execute(
            """SELECT v.path, s.peak_offset
               FROM segments s JOIN videos v ON v.id = s.video_id
               WHERE s.camera_id = ?
               ORDER BY s.activity_score DESC LIMIT 1""",
            (camera_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "no indexed footage for this camera")

        path = Path(row["path"])
        if not path.is_file():
            raise HTTPException(410, "source file has moved or been deleted")

        from tsv.frames import sample_windows

        offset = float(row["peak_offset"])
        for sample in sample_windows(
            path, [(offset, offset + 1.0)], fps=1.0, width=width, pixel_format="rgb24"
        ):
            ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(sample.frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if ok:
                return Response(encoded.tobytes(), media_type="image/jpeg")
        raise HTTPException(500, "could not decode a frame")

    @app.get("/api/crop/{tracklet_id}")
    def crop(tracklet_id: int) -> FileResponse:
        row = conn.execute(
            "SELECT thumb_path FROM tracklets WHERE id = ?", (tracklet_id,)
        ).fetchone()
        if row is None or not row["thumb_path"]:
            raise HTTPException(404)
        path = Path(row["thumb_path"])
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

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
