"""Phase 1: turn motion segments into tracked objects.

Runs only over segments Phase 0 already found, which is the whole point of the
cascade - the detector never sees the 73% of the recording that held nothing.

One container open serves an entire video: segments are handed to the sampler
as ordered windows, so the file is walked forwards once rather than reopened
per segment. The tracker is reset at each window boundary, because two
segments can be hours apart and carrying identities across that gap would
invent continuity that does not exist.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from tsv.config import Config
from tsv.frames import sample_windows
from tsv.models.detect import CCTV_CLASSES, COCO_CLASSES, Detector
from tsv.track.bytetrack import ByteTracker, Track, TrackerConfig


@dataclass
class AnalyzeResult:
    video_id: int
    path: Path
    status: str  # analyzed | skipped | failed
    n_segments: int = 0
    n_tracklets: int = 0
    n_detections: int = 0
    frames: int = 0
    elapsed: float = 0.0
    labels: Counter = field(default_factory=Counter)
    note: str = ""

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed else 0.0


@dataclass
class AnalyzeSummary:
    results: list[AnalyzeResult] = field(default_factory=list)
    backend: str = ""

    @property
    def analyzed(self) -> list[AnalyzeResult]:
        return [r for r in self.results if r.status == "analyzed"]

    @property
    def total_tracklets(self) -> int:
        return sum(r.n_tracklets for r in self.analyzed)

    @property
    def total_frames(self) -> int:
        return sum(r.frames for r in self.analyzed)

    @property
    def labels(self) -> Counter:
        total: Counter = Counter()
        for r in self.analyzed:
            total.update(r.labels)
        return total


def _crop_bytes(frame_rgb: np.ndarray, box: tuple[float, float, float, float], width: int) -> bytes | None:
    """A JPEG of one detection, padded slightly so the subject is not clipped."""
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.shape[1] > width:
        scale = width / crop.shape[1]
        crop = cv2.resize(
            crop, (width, max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return encoded.tobytes() if ok else None


def _write_tracklets(
    conn: sqlite3.Connection,
    segment: sqlite3.Row,
    video: sqlite3.Row,
    tracks: list[Track],
    sample_times: dict[int, float],
    frame_w: int,
    frame_h: int,
    crops: dict[int, bytes],
    crop_dir: Path,
) -> tuple[int, int, Counter]:
    labels: Counter = Counter()
    n_detections = 0

    for track in tracks:
        if not track.observations:
            continue
        boxes = np.array([o[1:] for o in track.observations], dtype=np.float32)
        frames = [o[0] for o in track.observations]
        times = [sample_times[f] for f in frames]

        first, last = boxes[0], boxes[-1]
        label = COCO_CLASSES[track.cls] if track.cls < len(COCO_CLASSES) else str(track.cls)
        labels[label] += 1

        thumb_path = None
        if track.track_id in crops:
            crop_dir.mkdir(parents=True, exist_ok=True)
            out = crop_dir / f"{segment['id']:07d}_{track.track_id:04d}.jpg"
            out.write_bytes(crops[track.track_id])
            thumb_path = str(out)

        cur = conn.execute(
            """INSERT INTO tracklets(
                   segment_id, video_id, camera_id, cls, label,
                   t_start, t_end, ts_start, ts_end,
                   n_detections, mean_score, max_score,
                   x_start, y_start, x_end, y_end, x1, y1, x2, y2, thumb_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                segment["id"], video["id"], video["camera_id"], track.cls, label,
                times[0], times[-1],
                video["start_ts"] + times[0], video["start_ts"] + times[-1],
                len(track.observations),
                float(np.mean(track.scores)), float(np.max(track.scores)),
                float((first[0] + first[2]) / 2 / frame_w), float((first[1] + first[3]) / 2 / frame_h),
                float((last[0] + last[2]) / 2 / frame_w), float((last[1] + last[3]) / 2 / frame_h),
                float(boxes[:, 0].min() / frame_w), float(boxes[:, 1].min() / frame_h),
                float(boxes[:, 2].max() / frame_w), float(boxes[:, 3].max() / frame_h),
                thumb_path,
            ),
        )
        tracklet_id = int(cur.lastrowid)

        conn.executemany(
            """INSERT INTO detections(tracklet_id, t, ts, x1, y1, x2, y2, score)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    tracklet_id, t, video["start_ts"] + t,
                    float(b[0] / frame_w), float(b[1] / frame_h),
                    float(b[2] / frame_w), float(b[3] / frame_h),
                    float(s),
                )
                for t, b, s in zip(times, boxes, track.scores)
            ],
        )
        n_detections += len(track.observations)

    conn.execute(
        "UPDATE segments SET analyzed_at = ?, n_tracklets = ?, labels = ? WHERE id = ?",
        (time.time(), len(tracks), json.dumps(dict(labels)) if labels else None, segment["id"]),
    )
    return len(tracks), n_detections, labels


def analyze_video(
    conn: sqlite3.Connection,
    video_id: int,
    detector: Detector,
    cfg: Config,
    force: bool = False,
    tracker_config: TrackerConfig | None = None,
) -> AnalyzeResult:
    started = time.time()
    video = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if video is None:
        return AnalyzeResult(video_id, Path(""), "failed", note="no such video")

    path = Path(video["path"])
    where = "" if force else " AND analyzed_at IS NULL"
    segments = conn.execute(
        f"SELECT * FROM segments WHERE video_id = ?{where} ORDER BY t_start", (video_id,)
    ).fetchall()

    if not segments:
        return AnalyzeResult(video_id, path, "skipped")
    if not path.is_file():
        return AnalyzeResult(video_id, path, "failed", note="source file missing")

    if force:
        conn.execute(
            "DELETE FROM tracklets WHERE segment_id IN "
            "(SELECT id FROM segments WHERE video_id = ?)",
            (video_id,),
        )

    windows = [(float(s["t_start"]), float(s["t_end"])) for s in segments]
    tracker = ByteTracker(tracker_config)
    sample_times: dict[int, float] = {}
    crops: dict[int, bytes] = {}
    best_score: dict[int, float] = {}
    current_window = 0
    frame_w = frame_h = 0
    frames = 0
    n_tracklets = n_detections = 0
    labels: Counter = Counter()

    def flush(window_index: int) -> None:
        nonlocal tracker, sample_times, crops, best_score
        nonlocal n_tracklets, n_detections, labels
        if frame_w and frame_h:
            added, dets, found = _write_tracklets(
                conn, segments[window_index], video, tracker.close(),
                sample_times, frame_w, frame_h, crops, cfg.crop_dir,
            )
            n_tracklets += added
            n_detections += dets
            labels.update(found)
        tracker = ByteTracker(tracker_config)
        sample_times, crops, best_score = {}, {}, {}

    try:
        for sample in sample_windows(
            path, windows, cfg.detect.detect_fps,
            width=cfg.detect.decode_width, pixel_format="rgb24",
        ):
            if sample.window_index != current_window:
                flush(current_window)
                current_window = sample.window_index

            frame_h, frame_w = sample.frame.shape[:2]
            index = len(sample_times)
            sample_times[index] = sample.t
            frames += 1

            detections = detector.detect(sample.frame)
            if detections:
                boxes = np.array([d.as_array() for d in detections], dtype=np.float32)
                scores = np.array([d.score for d in detections], dtype=np.float32)
                classes = np.array([d.cls for d in detections], dtype=np.int64)
            else:
                boxes = np.empty((0, 4), np.float32)
                scores = np.empty(0, np.float32)
                classes = np.empty(0, np.int64)

            tracker.update(boxes, scores, classes, index)

            # Keep the best-looking crop of each track as it goes past, rather
            # than decoding the file a second time to fetch them afterwards.
            for track in tracker.tracks:
                if track.time_since_update != 0:
                    continue
                if track.score > best_score.get(track.track_id, 0.0):
                    encoded = _crop_bytes(
                        sample.frame, tuple(track.observations[-1][1:]), cfg.detect.crop_width
                    )
                    if encoded:
                        best_score[track.track_id] = track.score
                        crops[track.track_id] = encoded

        flush(current_window)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        conn.rollback()
        return AnalyzeResult(video_id, path, "failed", note=f"{type(exc).__name__}: {exc}")

    conn.commit()
    return AnalyzeResult(
        video_id=video_id, path=path, status="analyzed",
        n_segments=len(segments), n_tracklets=n_tracklets, n_detections=n_detections,
        frames=frames, elapsed=time.time() - started, labels=labels,
    )


def analyze_all(
    conn: sqlite3.Connection,
    cfg: Config,
    force: bool = False,
    detector: Detector | None = None,
    on_result: Callable[[AnalyzeResult], None] | None = None,
) -> AnalyzeSummary:
    classes = frozenset(cfg.detect.classes) if cfg.detect.classes else CCTV_CLASSES
    detector = detector or Detector(
        cfg.detect_model_path,
        size=cfg.detect.input_size,
        conf_threshold=cfg.detect.conf_threshold,
        iou_threshold=cfg.detect.iou_threshold,
        keep_classes=classes,
        force_backend=cfg.detect.force_backend,
    )

    summary = AnalyzeSummary(backend=detector.info)
    where = "" if force else " WHERE id IN (SELECT video_id FROM segments WHERE analyzed_at IS NULL)"
    rows = conn.execute(f"SELECT id FROM videos{where} ORDER BY start_ts").fetchall()

    for row in rows:
        result = analyze_video(conn, int(row["id"]), detector, cfg, force=force)
        summary.results.append(result)
        if on_result:
            on_result(result)
    return summary
