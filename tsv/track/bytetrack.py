"""ByteTrack association.

The idea worth having: do not throw away low-confidence detections. Match the
confident ones first, then give every track that found no partner a second
chance against the leftovers. A person walking behind a chair drops to a low
score for a few frames rather than vanishing, and that second pass is what
keeps them one tracklet instead of three.

Tracks are matched per class - a dog passing a person must not inherit the
person's identity - and each class is associated independently.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from tsv.boxes import iou_matrix, xyxy_to_xywh
from tsv.track.kalman import KalmanFilter


@dataclass
class TrackerConfig:
    # Detections at or above this start and sustain tracks.
    high_threshold: float = 0.45
    # Detections between low and high are only used to continue existing
    # tracks, never to create them.
    low_threshold: float = 0.12
    # Minimum IoU to associate. The first pass is deliberately loose, because
    # sparse sampling means a true match often overlaps only partially. The
    # second pass is *stricter*: a low-confidence detection is weaker evidence,
    # so it has to line up better before it is allowed to continue a track.
    min_iou: float = 0.25
    second_min_iou: float = 0.45
    # How many consecutive missed samples before a track is closed. At 4 fps
    # this is about four seconds of occlusion.
    max_age: int = 16
    # Detections needed before a track is reported at all; suppresses
    # one-frame false positives.
    min_hits: int = 3


@dataclass
class Track:
    track_id: int
    cls: int
    mean: np.ndarray
    covariance: np.ndarray
    score: float
    start_frame: int
    frame_index: int
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    scores: list[float] = field(default_factory=list)
    # (frame_index, x1, y1, x2, y2) for each observed sample.
    observations: list[tuple[int, float, float, float, float]] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        """Current estimate as (x1, y1, x2, y2)."""
        cx, cy, aspect, height = self.mean[:4]
        width = aspect * height
        return np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            dtype=np.float32,
        )


def _to_measurement(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = xyxy_to_xywh(box.reshape(1, 4))[0]
    height = max(float(h), 1e-3)
    return np.array([cx, cy, float(w) / height, height], dtype=np.float64)


def _associate(
    track_boxes: np.ndarray, det_boxes: np.ndarray, min_iou: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching on IoU distance.

    Returns (matches, unmatched_track_indices, unmatched_detection_indices).
    """
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

    ious = iou_matrix(track_boxes, det_boxes)
    cost = 1.0 - ious
    rows, cols = linear_sum_assignment(cost)

    matches = []
    matched_tracks, matched_dets = set(), set()
    for r, c in zip(rows, cols):
        # The assignment is globally optimal but still allows terrible pairs
        # when counts differ; reject them on the original IoU.
        if ious[r, c] >= min_iou:
            matches.append((int(r), int(c)))
            matched_tracks.add(int(r))
            matched_dets.add(int(c))

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.kalman = KalmanFilter()
        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._ids = itertools.count(1)
        self._frame = -1

    def update(
        self, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, frame_index: int
    ) -> list[Track]:
        """Advance the tracker by one sampled frame."""
        self._frame = frame_index
        cfg = self.config

        for track in self.tracks:
            track.mean, track.covariance = self.kalman.predict(track.mean, track.covariance)
            track.age += 1
            track.time_since_update += 1

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)

        present = set(classes.tolist()) | {t.cls for t in self.tracks}
        for cls in sorted(present):
            self._update_class(cls, boxes, scores, classes, frame_index)

        # Retire anything that has been missing too long.
        alive = []
        for track in self.tracks:
            if track.time_since_update > cfg.max_age:
                if track.hits >= cfg.min_hits:
                    self.finished.append(track)
            else:
                alive.append(track)
        self.tracks = alive

        return [t for t in self.tracks if t.time_since_update == 0 and t.hits >= cfg.min_hits]

    def _update_class(
        self,
        cls: int,
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        frame_index: int,
    ) -> None:
        cfg = self.config
        det_mask = classes == cls
        cls_boxes, cls_scores = boxes[det_mask], scores[det_mask]

        high = cls_scores >= cfg.high_threshold
        low = (cls_scores >= cfg.low_threshold) & ~high

        tracks = [t for t in self.tracks if t.cls == cls]
        if not tracks and not high.any():
            return

        track_boxes = np.array([t.box for t in tracks], dtype=np.float32).reshape(-1, 4)

        # First pass: confident detections against every track.
        matches, unmatched_tracks, unmatched_high = _associate(
            track_boxes, cls_boxes[high], cfg.min_iou
        )
        high_idx = np.flatnonzero(high)
        for t_i, d_i in matches:
            self._apply(tracks[t_i], cls_boxes[high_idx[d_i]], float(cls_scores[high_idx[d_i]]), frame_index)

        # Second pass: whatever is left, against the low-confidence leftovers.
        # This is the whole point of ByteTrack.
        if unmatched_tracks and low.any():
            remaining = [tracks[i] for i in unmatched_tracks]
            remaining_boxes = np.array([t.box for t in remaining], dtype=np.float32).reshape(-1, 4)
            low_idx = np.flatnonzero(low)
            second, still_unmatched, _ = _associate(
                remaining_boxes, cls_boxes[low], cfg.second_min_iou
            )
            for t_i, d_i in second:
                self._apply(remaining[t_i], cls_boxes[low_idx[d_i]], float(cls_scores[low_idx[d_i]]), frame_index)

        # New tracks come only from confident, unmatched detections.
        for d_i in unmatched_high:
            box = cls_boxes[high_idx[d_i]]
            score = float(cls_scores[high_idx[d_i]])
            mean, covariance = self.kalman.initiate(_to_measurement(box))
            track = Track(
                track_id=next(self._ids), cls=int(cls), mean=mean, covariance=covariance,
                score=score, start_frame=frame_index, frame_index=frame_index,
            )
            track.scores.append(score)
            track.observations.append((frame_index, *map(float, box)))
            self.tracks.append(track)

    def _apply(self, track: Track, box: np.ndarray, score: float, frame_index: int) -> None:
        track.mean, track.covariance = self.kalman.update(
            track.mean, track.covariance, _to_measurement(box)
        )
        track.hits += 1
        track.time_since_update = 0
        track.score = score
        track.frame_index = frame_index
        track.scores.append(score)
        track.observations.append((frame_index, *map(float, box)))

    def close(self) -> list[Track]:
        """Flush every surviving track. Call once the clip is exhausted."""
        for track in self.tracks:
            if track.hits >= self.config.min_hits:
                self.finished.append(track)
        self.tracks = []
        return self.finished
