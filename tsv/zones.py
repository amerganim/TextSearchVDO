"""User-drawn zones, and the events a tracklet generates against them.

This is where "when did my son go outside" stops being a similarity search and
becomes an exact query: a door line the user draws once, plus a trajectory,
gives a crossing with a direction and a timestamp.

Two decisions here matter more than the geometry:

**The anchor is the bottom centre of the box, not its centre.** A person's
feet are where they actually are; their head is a metre and a half away in the
wrong direction. Using the box centre puts someone "in the kitchen" while they
are still in the hallway leaning through the doorway.

**Crossings are tested on the segment between consecutive samples**, not by
watching which side a point is on. Detection runs at a few frames a second, so
a person can be fully on one side in one sample and fully across in the next -
there is often no sample *on* the line at all.

Coordinates are normalised 0..1 throughout, so a zone survives the camera
being reconfigured to a different resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

ZoneKind = Literal["region", "line"]

# Consecutive samples required inside (or outside) a region before the change
# is believed. Someone standing on a threshold otherwise generates a burst of
# enter/exit pairs.
DEFAULT_HYSTERESIS = 2


@dataclass
class Zone:
    id: int
    camera_id: int
    name: str
    kind: ZoneKind
    # Region: >= 3 vertices. Line: exactly 2 points, a -> b.
    points: list[tuple[float, float]]

    @staticmethod
    def from_row(row) -> "Zone":
        return Zone(
            id=int(row["id"]),
            camera_id=int(row["camera_id"]),
            name=row["name"],
            kind=row["kind"],
            points=[tuple(p) for p in json.loads(row["points"])],
        )

    def validate(self) -> None:
        if self.kind == "line" and len(self.points) != 2:
            raise ValueError("a line zone needs exactly 2 points")
        if self.kind == "region" and len(self.points) < 3:
            raise ValueError("a region zone needs at least 3 points")
        for x, y in self.points:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"zone points must be normalised 0..1, got ({x}, {y})")


@dataclass
class ZoneEvent:
    zone_id: int
    zone_name: str
    kind: Literal["enter", "exit", "cross_in", "cross_out", "dwell"]
    t: float
    duration: float = 0.0


def anchor_of(box: Sequence[float]) -> tuple[float, float]:
    """Ground-contact point of a box: bottom centre."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def anchors_of(boxes: np.ndarray) -> np.ndarray:
    """Vectorised `anchor_of` over an (n, 4) array."""
    if boxes.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    out = np.empty((len(boxes), 2), dtype=np.float64)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    out[:, 1] = boxes[:, 3]
    return out


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Ray casting. Points exactly on an edge may fall either way."""
    x, y = point[0], point[1]
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # Does the edge straddle the horizontal ray through y?
        if (y1 > y) != (y2 > y):
            t = (y - y1) / (y2 - y1)
            if x < x1 + t * (x2 - x1):
                inside = not inside
    return inside


def side_of_line(
    point: Sequence[float], a: Sequence[float], b: Sequence[float]
) -> float:
    """Signed area of the triangle (a, b, point).

    Positive on the left of a->b, negative on the right, zero when collinear.
    """
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def segments_intersect(
    p1: Sequence[float], p2: Sequence[float],
    a: Sequence[float], b: Sequence[float],
) -> bool:
    """Proper intersection of segment p1->p2 with segment a->b.

    Collinear overlap counts as no crossing: a track running exactly along a
    door line has not gone through it.
    """
    d1 = side_of_line(p1, a, b)
    d2 = side_of_line(p2, a, b)
    d3 = side_of_line(a, p1, p2)
    d4 = side_of_line(b, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _region_events(
    zone: Zone, times: Sequence[float], anchors: np.ndarray, hysteresis: int
) -> list[ZoneEvent]:
    inside_flags = [point_in_polygon(p, zone.points) for p in anchors]

    events: list[ZoneEvent] = []
    state = False          # believed to be inside
    run_value = None
    run_length = 0
    entered_at: float | None = None

    for index, (t, flag) in enumerate(zip(times, inside_flags)):
        if flag == run_value:
            run_length += 1
        else:
            run_value, run_length = flag, 1

        if run_length < hysteresis or flag == state:
            continue

        # The change is believed; date it from the start of the run rather
        # than from the sample that confirmed it, so a two-sample hysteresis
        # does not report someone arriving half a second late.
        confirmed_t = times[max(0, index - run_length + 1)]
        state = flag
        if flag:
            entered_at = confirmed_t
            events.append(ZoneEvent(zone.id, zone.name, "enter", confirmed_t))
        else:
            events.append(ZoneEvent(zone.id, zone.name, "exit", confirmed_t))
            if entered_at is not None:
                events.append(
                    ZoneEvent(zone.id, zone.name, "dwell", entered_at, confirmed_t - entered_at)
                )
            entered_at = None

    # Still inside when the tracklet ended.
    if state and entered_at is not None and times:
        events.append(
            ZoneEvent(zone.id, zone.name, "dwell", entered_at, times[-1] - entered_at)
        )
    return events


def _line_events(zone: Zone, times: Sequence[float], anchors: np.ndarray) -> list[ZoneEvent]:
    a, b = zone.points
    events: list[ZoneEvent] = []
    for i in range(1, len(anchors)):
        p1, p2 = anchors[i - 1], anchors[i]
        if not segments_intersect(p1, p2, a, b):
            continue
        # Direction is the side the track ended up on. Left of a->b is "in";
        # the user picks which way round that is by how they draw the line.
        kind = "cross_in" if side_of_line(p2, a, b) > 0 else "cross_out"
        events.append(ZoneEvent(zone.id, zone.name, kind, times[i]))
    return events


def events_for_track(
    zone: Zone,
    times: Sequence[float],
    boxes: np.ndarray,
    hysteresis: int = DEFAULT_HYSTERESIS,
) -> list[ZoneEvent]:
    """Every event one tracklet generates against one zone.

    `boxes` are normalised (n, 4) in the same order as `times`.
    """
    if len(times) == 0 or boxes.size == 0:
        return []
    anchors = anchors_of(np.asarray(boxes, dtype=np.float64))
    if zone.kind == "line":
        return _line_events(zone, times, anchors)
    return _region_events(zone, times, anchors, max(1, hysteresis))


def events_for_zones(
    zones: Iterable[Zone],
    times: Sequence[float],
    boxes: np.ndarray,
    hysteresis: int = DEFAULT_HYSTERESIS,
) -> list[ZoneEvent]:
    out: list[ZoneEvent] = []
    for zone in zones:
        out.extend(events_for_track(zone, times, boxes, hysteresis))
    return sorted(out, key=lambda e: (e.t, e.zone_id))
