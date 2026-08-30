"""Which way up a recording is stored.

Two clips from a phone produced almost nothing: one person in 34 seconds, and
in the other, nothing at all. Not the motion gate and not the models - the
frames come out of the file lying on their side, with no rotation anywhere for
ffmpeg to apply, so the app was faithfully analysing a sideways room.

Detectors do not cope with that; everything they were trained on stands up.
Measured on those clips, the same detector at each right angle:

    as stored        9 detections, best 0.60
    rotated 90 cw   14 detections, best 0.81, and finds a bed

The bed is the point. The question that failed was "anyone sleeping?".
"""

from __future__ import annotations

import numpy as np
import pytest

from tsv.orientation import MARGIN, Orientation, apply, detect


class Box:
    """A detection, as far as these tests care."""

    def __init__(self, score: float) -> None:
        self.score = score


class UprightOnly:
    """A detector that only sees things in portrait-shaped frames.

    Stands in for the real one's behaviour: a person lying across a sideways
    frame is not a shape it was trained on.
    """

    def __init__(self, wants_tall: bool = True) -> None:
        self.wants_tall = wants_tall

    def detect(self, frame):
        height, width = frame.shape[:2]
        tall = height > width
        return [Box(0.9), Box(0.8)] if tall == self.wants_tall else [Box(0.2)]


def test_turning_a_frame_is_reversible_and_shaped_right():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert apply(frame, 0).shape == (100, 200, 3)
    assert apply(frame, 90).shape == (200, 100, 3)
    assert apply(frame, 180).shape == (100, 200, 3)
    assert apply(frame, 270).shape == (200, 100, 3)


def test_turning_by_nothing_changes_nothing():
    frame = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
    assert np.array_equal(apply(frame, 0), frame)
    assert np.array_equal(apply(apply(frame, 90), 270), frame)


def test_a_sideways_recording_is_recognised(monkeypatch, tmp_path):
    """The case this exists for: nothing is found until the frame is turned."""
    wide = np.zeros((108, 192, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "tsv.frames.sample_windows",
        lambda *a, **k: [type("S", (), {"frame": wide})() for _ in range(4)],
    )
    found = detect(tmp_path / "clip.mp4", UprightOnly(), duration=30.0)
    assert found.turned
    assert found.degrees in (90, 270)
    assert found.score > found.baseline
    assert "off upright" in found.describe()


def test_an_upright_recording_is_left_alone(monkeypatch, tmp_path):
    """Turning a recording that was already right is worse than leaving a
    sideways one alone: it breaks something that worked."""
    tall = np.zeros((192, 108, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "tsv.frames.sample_windows",
        lambda *a, **k: [type("S", (), {"frame": tall})() for _ in range(4)],
    )
    found = detect(tmp_path / "clip.mp4", UprightOnly(), duration=30.0)
    assert not found.turned
    assert found.degrees == 0
    assert found.describe() == "upright"


def test_an_ambiguous_recording_stays_as_it_is(monkeypatch, tmp_path):
    """A detector equally happy either way must not cause a coin toss."""
    square = np.zeros((128, 128, 3), dtype=np.uint8)

    class Indifferent:
        def detect(self, frame):
            return [Box(0.5)]

    monkeypatch.setattr(
        "tsv.frames.sample_windows",
        lambda *a, **k: [type("S", (), {"frame": square})() for _ in range(4)],
    )
    assert detect(tmp_path / "clip.mp4", Indifferent(), duration=30.0).degrees == 0


def test_a_small_improvement_is_not_enough(monkeypatch, tmp_path):
    """Only a clear margin is believed, so noise cannot turn a recording."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    class BarelyBetter:
        def detect(self, f):
            tall = f.shape[0] > f.shape[1]
            # Just under the margin required.
            return [Box(0.5 * (MARGIN - 0.1) if tall else 0.5)]

    monkeypatch.setattr(
        "tsv.frames.sample_windows",
        lambda *a, **k: [type("S", (), {"frame": frame})() for _ in range(4)],
    )
    assert detect(tmp_path / "clip.mp4", BarelyBetter(), duration=30.0).degrees == 0


def test_a_recording_with_no_readable_frames_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr("tsv.frames.sample_windows", lambda *a, **k: [])
    found = detect(tmp_path / "clip.mp4", UprightOnly(), duration=30.0)
    assert found == Orientation(0, 0.0, 0.0, 0)


def test_frames_come_out_turned_when_asked():
    """Everything that looks at pixels has to agree which way up they are, so
    the rotation is applied where frames are read rather than by each caller -
    one of them disagreeing means boxes drawn against a different picture."""
    import inspect

    from tsv import frames

    source = inspect.getsource(frames.sample_windows)
    assert "rotation" in inspect.signature(frames.sample_windows).parameters
    assert "turn_upright" in source
