"""Zone geometry and the events a trajectory generates against it.

Scenarios are the ones a front door actually produces: someone walking out,
someone walking back in, someone loitering on the threshold, someone crossing
so fast that no sample lands on the line.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsv.zones import (
    Zone, ZoneEvent, anchor_of, anchors_of, events_for_track, events_for_zones,
    point_in_polygon, segments_intersect, side_of_line,
)

# A doorway across the middle of frame, drawn left to right. Left of a->b
# (i.e. the lower half, in screen coordinates) counts as inbound.
DOOR = Zone(id=1, camera_id=1, name="front door", kind="line",
            points=[(0.0, 0.5), (1.0, 0.5)])

# The right-hand half of the frame.
KITCHEN = Zone(id=2, camera_id=1, name="kitchen", kind="region",
               points=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])


def _boxes(points, w=0.08, h=0.30):
    """Boxes whose bottom centre sits at each given point."""
    out = []
    for x, y in points:
        out.append([x - w / 2, y - h, x + w / 2, y])
    return np.array(out, dtype=np.float64)


def _times(n, step=0.25):
    return [i * step for i in range(n)]


# ---------- primitives ----------

def test_anchor_is_the_bottom_centre_not_the_middle():
    """A person is where their feet are, not where their head is."""
    assert anchor_of([0.2, 0.1, 0.4, 0.9]) == (0.30000000000000004, 0.9)


def test_anchors_of_matches_the_scalar_version():
    boxes = np.array([[0.1, 0.2, 0.3, 0.8], [0.5, 0.0, 0.7, 0.4]])
    assert np.allclose(anchors_of(boxes), [[0.2, 0.8], [0.6, 0.4]])
    assert anchors_of(np.empty((0, 4))).shape == (0, 2)


def test_point_in_polygon_basics():
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((1.5, 0.5), square)
    assert not point_in_polygon((0.5, -0.1), square)


def test_point_in_concave_polygon():
    """An L-shaped room: the notch must not count as inside."""
    shape = [(0, 0), (1, 0), (1, 0.4), (0.4, 0.4), (0.4, 1), (0, 1)]
    assert point_in_polygon((0.2, 0.2), shape)
    assert point_in_polygon((0.8, 0.2), shape)
    assert not point_in_polygon((0.8, 0.8), shape)   # the notch


def test_side_of_line_signs():
    a, b = (0.0, 0.0), (1.0, 0.0)
    assert side_of_line((0.5, 1.0), a, b) > 0
    assert side_of_line((0.5, -1.0), a, b) < 0
    assert side_of_line((0.5, 0.0), a, b) == 0


def test_segments_intersect():
    assert segments_intersect((0, 0), (1, 1), (0, 1), (1, 0))
    assert not segments_intersect((0, 0), (0.4, 0.4), (0, 1), (1, 0))


def test_collinear_travel_along_a_line_is_not_a_crossing():
    """Walking along the doorway is not walking through it."""
    assert not segments_intersect((0.1, 0.5), (0.9, 0.5), (0.0, 0.5), (1.0, 0.5))


# ---------- line crossings ----------

def test_walking_out_produces_one_outbound_crossing():
    path = [(0.5, 0.9), (0.5, 0.7), (0.5, 0.3), (0.5, 0.1)]
    events = events_for_track(DOOR, _times(4), _boxes(path))
    assert [e.kind for e in events] == ["cross_out"]


def test_walking_in_produces_one_inbound_crossing():
    path = [(0.5, 0.1), (0.5, 0.3), (0.5, 0.7), (0.5, 0.9)]
    events = events_for_track(DOOR, _times(4), _boxes(path))
    assert [e.kind for e in events] == ["cross_in"]


def test_a_fast_crossing_with_no_sample_on_the_line_is_still_caught():
    """At 4 fps a person is simply on one side, then the other."""
    path = [(0.5, 0.95), (0.5, 0.05)]
    events = events_for_track(DOOR, _times(2), _boxes(path))
    assert [e.kind for e in events] == ["cross_out"]


def test_going_out_and_coming_back_gives_two_crossings_in_order():
    path = [(0.5, 0.9), (0.5, 0.2), (0.5, 0.9)]
    events = events_for_track(DOOR, _times(3), _boxes(path))
    assert [e.kind for e in events] == ["cross_out", "cross_in"]
    assert events[0].t < events[1].t


def test_approaching_without_crossing_produces_nothing():
    path = [(0.5, 0.95), (0.5, 0.8), (0.5, 0.6), (0.5, 0.8)]
    assert events_for_track(DOOR, _times(4), _boxes(path)) == []


def test_crossing_timestamps_come_from_the_sample_after_the_line():
    path = [(0.5, 0.9), (0.5, 0.1)]
    event = events_for_track(DOOR, [10.0, 10.25], _boxes(path))[0]
    assert event.t == 10.25


# ---------- regions ----------

def test_entering_and_leaving_a_region():
    path = [(0.1, 0.5), (0.2, 0.5), (0.6, 0.5), (0.7, 0.5), (0.2, 0.5), (0.1, 0.5)]
    events = events_for_track(KITCHEN, _times(6), _boxes(path), hysteresis=2)
    assert [e.kind for e in events] == ["enter", "exit", "dwell"]


def test_dwell_duration_spans_the_visit():
    path = [(0.1, 0.5), (0.6, 0.5), (0.7, 0.5), (0.8, 0.5), (0.1, 0.5), (0.1, 0.5)]
    events = events_for_track(KITCHEN, _times(6, step=1.0), _boxes(path), hysteresis=2)
    dwell = [e for e in events if e.kind == "dwell"][0]
    assert dwell.duration == pytest.approx(3.0)


def test_hysteresis_suppresses_threshold_flicker():
    """Someone standing in a doorway wobbling across the boundary."""
    path = [(0.49, 0.5), (0.51, 0.5), (0.49, 0.5), (0.51, 0.5), (0.49, 0.5)]
    events = events_for_track(KITCHEN, _times(5), _boxes(path), hysteresis=3)
    assert events == []


def test_a_track_still_inside_at_the_end_still_reports_a_dwell():
    path = [(0.1, 0.5), (0.6, 0.5), (0.7, 0.5), (0.8, 0.5)]
    events = events_for_track(KITCHEN, _times(4, step=1.0), _boxes(path), hysteresis=2)
    kinds = [e.kind for e in events]
    assert "enter" in kinds and "dwell" in kinds
    assert "exit" not in kinds


def test_a_track_entirely_outside_produces_nothing():
    path = [(0.1, 0.5), (0.2, 0.5), (0.3, 0.5)]
    assert events_for_track(KITCHEN, _times(3), _boxes(path)) == []


def test_the_anchor_choice_actually_matters():
    """Leaning through a doorway: head over the line, feet outside.

    With the box centre as anchor this would report an entry; with the feet it
    correctly reports nothing.
    """
    # Bottom centre at x=0.45, outside; the box still reaches across to 0.60.
    boxes = np.array([[0.30, 0.2, 0.60, 0.6]] * 3)
    assert anchors_of(boxes)[0][0] == pytest.approx(0.45)
    assert events_for_track(KITCHEN, _times(3), boxes) == []

    # Shifting the feet across the boundary does register.
    inside = np.array([[0.45, 0.2, 0.75, 0.6]] * 3)
    assert events_for_track(KITCHEN, _times(3), inside, hysteresis=1) != []


# ---------- multiple zones ----------

def test_events_from_several_zones_come_back_in_time_order():
    path = [(0.1, 0.9), (0.6, 0.9), (0.6, 0.1)]
    events = events_for_zones([DOOR, KITCHEN], _times(3), _boxes(path), hysteresis=1)
    assert [e.t for e in events] == sorted(e.t for e in events)
    assert {e.zone_name for e in events} == {"front door", "kitchen"}


def test_empty_track_is_safe():
    assert events_for_track(DOOR, [], np.empty((0, 4))) == []
    assert events_for_zones([DOOR, KITCHEN], [], np.empty((0, 4))) == []


# ---------- validation ----------

def test_zone_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Zone(1, 1, "bad", "line", [(0, 0), (1, 1), (0.5, 0.5)]).validate()
    with pytest.raises(ValueError):
        Zone(1, 1, "bad", "region", [(0, 0), (1, 1)]).validate()
    with pytest.raises(ValueError):
        Zone(1, 1, "bad", "line", [(0, 0), (2.5, 1)]).validate()
    Zone(1, 1, "ok", "line", [(0, 0), (1, 1)]).validate()
