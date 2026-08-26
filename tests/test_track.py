"""Tracker behaviour, driven by synthetic trajectories with known identity.

Every scenario here is one a hallway camera produces daily: someone walking
across frame, two people crossing, a person disappearing behind furniture.
"""

from __future__ import annotations

import numpy as np

from tsv.track.bytetrack import ByteTracker, TrackerConfig
from tsv.track.kalman import KalmanFilter

CFG = TrackerConfig(min_hits=2, max_age=6)


def _walk(x0: float, step: float, n: int, y: float = 100.0, w: float = 40.0, h: float = 90.0):
    """A box moving right at a constant rate."""
    return [
        np.array([[x0 + i * step, y, x0 + i * step + w, y + h]], dtype=np.float32)
        for i in range(n)
    ]


def _run(frames, scores=None, classes=None, cfg=CFG):
    tracker = ByteTracker(cfg)
    for i, boxes in enumerate(frames):
        n = len(boxes)
        tracker.update(
            boxes,
            np.full(n, 0.9, dtype=np.float32) if scores is None else np.asarray(scores[i]),
            np.zeros(n, dtype=np.int64) if classes is None else np.asarray(classes[i]),
            i,
        )
    return tracker


# ---------- Kalman ----------

def test_kalman_predicts_constant_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 0.5, 80.0]))
    for x in (110.0, 120.0, 130.0):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([x, 100.0, 0.5, 80.0]))
    predicted, _ = kf.predict(mean, cov)
    # Having seen +10 per step, the next prediction should lead the last
    # observation rather than sit on it.
    assert predicted[0] > 132.0


def test_kalman_covariance_stays_finite():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([50.0, 50.0, 0.4, 60.0]))
    for _ in range(200):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([50.0, 50.0, 0.4, 60.0]))
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))


# ---------- association ----------

def test_a_single_walker_is_one_tracklet():
    tracker = _run(_walk(0, 12, 10))
    finished = tracker.close()
    assert len(finished) == 1
    assert finished[0].hits == 10


def test_track_ids_are_stable_across_frames():
    tracker = ByteTracker(CFG)
    seen = set()
    for i, boxes in enumerate(_walk(0, 12, 8)):
        for track in tracker.update(boxes, np.array([0.9]), np.array([0]), i):
            seen.add(track.track_id)
    assert len(seen) == 1


def test_two_separated_people_get_separate_tracks():
    frames = []
    for i in range(8):
        frames.append(np.array([
            [i * 10, 100, i * 10 + 40, 190],
            [600 - i * 10, 300, 640 - i * 10, 390],
        ], dtype=np.float32))
    assert len(_run(frames).close()) == 2


def test_a_brief_gap_does_not_split_the_tracklet():
    """Someone stepping behind a pillar for two samples is still one person."""
    frames = _walk(0, 10, 4) + [np.empty((0, 4), np.float32)] * 2 + _walk(60, 10, 4)
    finished = _run(frames).close()
    assert len(finished) == 1


def test_a_long_absence_does_close_the_track():
    frames = _walk(0, 10, 4) + [np.empty((0, 4), np.float32)] * 10 + _walk(300, 10, 4)
    finished = _run(frames).close()
    assert len(finished) == 2


def test_low_confidence_detections_sustain_a_track_but_never_start_one():
    """The whole reason ByteTrack exists."""
    frames = _walk(0, 10, 8)
    scores = [[0.9], [0.9], [0.2], [0.2], [0.2], [0.9], [0.9], [0.9]]
    finished = _run(frames, scores=scores).close()
    assert len(finished) == 1
    assert finished[0].hits == 8

    # The same weak detections with no confident start produce nothing.
    only_low = _run(_walk(0, 10, 6), scores=[[0.2]] * 6).close()
    assert only_low == []


def test_classes_do_not_swap_identity():
    """A dog crossing a person must not inherit the person's track."""
    frames, classes = [], []
    for i in range(8):
        frames.append(np.array([
            [i * 20, 100, i * 20 + 40, 190],
            [140 - i * 20, 100, 180 - i * 20, 190],
        ], dtype=np.float32))
        classes.append([0, 16])   # person, dog

    finished = _run(frames, classes=classes).close()
    by_class = sorted(t.cls for t in finished)
    assert by_class == [0, 16]
    assert all(t.hits == 8 for t in finished)


def test_single_frame_blip_is_discarded():
    frames = [np.array([[0, 0, 40, 90]], dtype=np.float32)] + [np.empty((0, 4), np.float32)] * 8
    assert _run(frames, cfg=TrackerConfig(min_hits=3, max_age=4)).close() == []


def test_observations_are_recorded_for_every_hit():
    finished = _run(_walk(0, 10, 6)).close()
    assert len(finished[0].observations) == 6
    frames = [o[0] for o in finished[0].observations]
    assert frames == sorted(frames)


def test_empty_updates_are_safe():
    tracker = ByteTracker(CFG)
    for i in range(5):
        assert tracker.update(
            np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64), i
        ) == []
    assert tracker.close() == []
