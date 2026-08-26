"""Deriving zone events from stored tracklets.

This pass touches no video at all - every box it needs is already in the
`detections` table. That is deliberate: zones are drawn and redrawn by the
user, and moving a door line two pixels must not mean re-running the detector
over a week of footage. Recomputing every event for a camera is a few seconds
of SQL and arithmetic.

Because of that, events are always rebuilt wholesale for the zones in scope
rather than patched incrementally. There is no partial state to get wrong.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from tsv.zones import DEFAULT_HYSTERESIS, Zone, events_for_zones


@dataclass
class EventSummary:
    n_zones: int = 0
    n_tracklets: int = 0
    n_events: int = 0
    elapsed: float = 0.0
    by_kind: dict[str, int] = field(default_factory=dict)


def list_zones(conn: sqlite3.Connection, camera_id: int | None = None) -> list[Zone]:
    where, params = ("WHERE camera_id = ?", [camera_id]) if camera_id else ("", [])
    rows = conn.execute(f"SELECT * FROM zones {where} ORDER BY name", params).fetchall()
    return [Zone.from_row(r) for r in rows]


def create_zone(
    conn: sqlite3.Connection,
    camera_id: int,
    name: str,
    kind: str,
    points: Sequence[Sequence[float]],
) -> Zone:
    zone = Zone(id=0, camera_id=camera_id, name=name, kind=kind,
                points=[(float(x), float(y)) for x, y in points])
    zone.validate()
    cur = conn.execute(
        """INSERT INTO zones(camera_id, name, kind, points, created_at)
           VALUES (?,?,?,?,?)""",
        (camera_id, name, kind, json.dumps(zone.points), time.time()),
    )
    conn.commit()
    zone.id = int(cur.lastrowid)
    return zone


def delete_zone(conn: sqlite3.Connection, zone_id: int) -> bool:
    cur = conn.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
    conn.commit()
    return cur.rowcount > 0


def recompute_events(
    conn: sqlite3.Connection,
    camera_id: int | None = None,
    hysteresis: int = DEFAULT_HYSTERESIS,
) -> EventSummary:
    """Rebuild every event for the zones on one camera, or all cameras."""
    started = time.time()
    zones = list_zones(conn, camera_id)
    summary = EventSummary(n_zones=len(zones))
    if not zones:
        conn.execute(
            "DELETE FROM events" + (" WHERE camera_id = ?" if camera_id else ""),
            [camera_id] if camera_id else [],
        )
        conn.commit()
        summary.elapsed = time.time() - started
        return summary

    by_camera: dict[int, list[Zone]] = {}
    for zone in zones:
        by_camera.setdefault(zone.camera_id, []).append(zone)

    conn.execute(
        "DELETE FROM events WHERE zone_id IN (%s)" % ",".join("?" * len(zones)),
        [z.id for z in zones],
    )

    where, params = ("WHERE t.camera_id = ?", [camera_id]) if camera_id else ("", [])
    tracklets = conn.execute(
        f"""SELECT t.id, t.camera_id, t.video_id, t.segment_id, t.label, v.start_ts
            FROM tracklets t JOIN videos v ON v.id = t.video_id
            {where} ORDER BY t.ts_start""",
        params,
    ).fetchall()

    rows: list[tuple] = []
    for tracklet in tracklets:
        zones_here = by_camera.get(int(tracklet["camera_id"]))
        if not zones_here:
            continue

        detections = conn.execute(
            "SELECT t, x1, y1, x2, y2 FROM detections WHERE tracklet_id = ? ORDER BY t",
            (tracklet["id"],),
        ).fetchall()
        if not detections:
            continue

        summary.n_tracklets += 1
        times = [float(d["t"]) for d in detections]
        boxes = np.array(
            [[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float64
        )

        for event in events_for_zones(zones_here, times, boxes, hysteresis):
            rows.append((
                tracklet["id"], event.zone_id, tracklet["segment_id"], tracklet["video_id"],
                tracklet["camera_id"], tracklet["label"], event.kind,
                event.t, tracklet["start_ts"] + event.t, event.duration,
            ))
            summary.by_kind[event.kind] = summary.by_kind.get(event.kind, 0) + 1

    conn.executemany(
        """INSERT INTO events(tracklet_id, zone_id, segment_id, video_id, camera_id,
                              label, kind, t, ts, duration)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()

    summary.n_events = len(rows)
    summary.elapsed = time.time() - started
    return summary
