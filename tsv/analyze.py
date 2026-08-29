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
from tsv.identity import aggregate, store_tracklet_embedding
from tsv.models.detect import CCTV_CLASSES, COCO_CLASSES, Detector
from tsv.models.clip import ClipEmbedder, build_clip
from tsv.models.face import FacePipeline
from tsv.track.bytetrack import ByteTracker, Track, TrackerConfig

# Faces are only looked for on people. A face detector run over a dog or a
# parked car is pure cost.
PERSON_CLASS = COCO_CLASSES.index("person")


@dataclass
class AnalyzeResult:
    video_id: int
    path: Path
    status: str  # analyzed | skipped | failed
    n_segments: int = 0
    n_tracklets: int = 0
    n_detections: int = 0
    n_faces: int = 0
    n_embedded: int = 0
    # Seconds of footage this pass examined, for driving a progress bar.
    analysed_seconds: float = 0.0
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
    def total_faces(self) -> int:
        return sum(r.n_faces for r in self.analyzed)

    @property
    def total_embedded(self) -> int:
        return sum(r.n_embedded for r in self.analyzed)

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


def _remember_face_crop(
    face_crops: dict[int, list[tuple[float, np.ndarray]]],
    track: Track,
    frame_rgb: np.ndarray,
    limit: int,
) -> None:
    """Keep the best few crops of a person for the face pass at flush time.

    Bounded by detection score, so a tracklet that lasts two minutes costs the
    same memory as one that lasts two seconds.
    """
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = track.observations[-1][1:]
    # Faces sit above the box's centre and the detector often clips the top of
    # the head, so pad upward more generously than sideways.
    pad_x = (x2 - x1) * 0.10
    pad_y = (y2 - y1) * 0.10
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y * 1.6))
    x2 = min(width, int(x2 + pad_x))
    y2 = min(height, int(y2 + pad_y))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return

    kept = face_crops.setdefault(track.track_id, [])
    if len(kept) < limit:
        kept.append((track.score, frame_rgb[y1:y2, x1:x2].copy()))
        kept.sort(key=lambda pair: pair[0])
    elif track.score > kept[0][0]:
        kept[0] = (track.score, frame_rgb[y1:y2, x1:x2].copy())
        kept.sort(key=lambda pair: pair[0])


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
    face_crops: dict[int, list[tuple[float, np.ndarray]]] | None = None,
    face: FacePipeline | None = None,
    min_face_px: int = 24,
    clip: ClipEmbedder | None = None,
    clip_crops: dict[int, np.ndarray] | None = None,
    scene_frame: np.ndarray | None = None,
    # Which weights produced the vectors written here. Passed in rather than
    # read from a config this function does not have - the reason being that
    # reaching for `cfg` here raised NameError, the caller logged it as a bad
    # file, and every analysed video silently produced nothing.
    face_model: str | None = None,
    clip_model: str | None = None,
) -> tuple[int, int, int, int, Counter]:
    labels: Counter = Counter()
    n_detections = 0
    n_faces = 0
    n_embedded = 0

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

        if face is not None and face_crops and track.cls == PERSON_CLASS:
            vectors = []
            for _, crop in face_crops.get(track.track_id, []):
                found = face.best_face_in(crop, min_size=min_face_px)
                if found is not None:
                    vectors.append(found[1])
            if vectors:
                # One vector per tracklet: the person is the same throughout,
                # so averaging their views is strictly better evidence than
                # any single frame, which may be a blink or a turn.
                store_tracklet_embedding(
                    conn, tracklet_id, "face", aggregate(vectors),
                    n_samples=len(vectors), model=face_model,
                )
                n_faces += 1

        if clip is not None and clip_crops is not None:
            crop = clip_crops.get(track.track_id)
            if crop is not None and crop.size:
                store_tracklet_embedding(
                    conn, tracklet_id, "clip", clip.embed_image(crop), model=clip_model
                )
                n_embedded += 1

    if clip is not None and scene_frame is not None and scene_frame.size:
        # A scene-level vector as well as per-object ones: "a car in the
        # driveway" is a property of the frame, not of any one crop.
        vector = clip.embed_image(scene_frame)
        conn.execute(
            """INSERT INTO segment_embeddings(segment_id, kind, dim, vector, model)
               VALUES (?,?,?,?,?)
               ON CONFLICT(segment_id, kind) DO UPDATE SET
                   dim = excluded.dim, vector = excluded.vector,
                   model = excluded.model""",
            (segment["id"], "clip", len(vector), vector.astype(np.float32).tobytes(),
             clip_model),
        )
        n_embedded += 1

    conn.execute(
        "UPDATE segments SET analyzed_at = ?, n_tracklets = ?, labels = ? WHERE id = ?",
        (time.time(), len(tracks), json.dumps(dict(labels)) if labels else None, segment["id"]),
    )
    return len(tracks), n_detections, n_faces, n_embedded, labels


def analyze_video(
    conn: sqlite3.Connection,
    video_id: int,
    detector: Detector,
    cfg: Config,
    force: bool = False,
    tracker_config: TrackerConfig | None = None,
    face: FacePipeline | None = None,
    clip: ClipEmbedder | None = None,
    on_progress: Callable[[float, float], None] | None = None,
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
    # Progress is measured in seconds of footage examined, not segments. A
    # long recording often yields only two or three segments, and a bar that
    # moves twice in four minutes is indistinguishable from a hung one.
    window_seconds = [end - start for start, end in windows]
    total_seconds = sum(window_seconds) or 1.0
    seconds_before = [sum(window_seconds[:i]) for i in range(len(windows))]
    tracker = ByteTracker(tracker_config)
    sample_times: dict[int, float] = {}
    crops: dict[int, bytes] = {}
    best_score: dict[int, float] = {}
    # The best few person crops per track, held for the face pass at flush.
    face_crops: dict[int, list[tuple[float, np.ndarray]]] = {}
    clip_crops: dict[int, np.ndarray] = {}
    # The sampled frame nearest this segment's peak, for the scene vector.
    scene_frame: np.ndarray | None = None
    scene_gap = float("inf")
    current_window = 0
    frame_w = frame_h = 0
    frames = 0
    n_tracklets = n_detections = n_faces = n_embedded = 0
    labels: Counter = Counter()

    def flush(window_index: int) -> None:
        nonlocal tracker, sample_times, crops, best_score, face_crops, clip_crops
        nonlocal scene_frame, scene_gap
        nonlocal n_tracklets, n_detections, n_faces, n_embedded, labels
        if frame_w and frame_h:
            added, dets, faces_found, embedded, found = _write_tracklets(
                conn, segments[window_index], video, tracker.close(),
                sample_times, frame_w, frame_h, crops, cfg.crop_dir,
                face_crops=face_crops, face=face, min_face_px=cfg.face.min_face_px,
                face_model=cfg.face.name, clip_model=cfg.clip.name,
                clip=clip, clip_crops=clip_crops, scene_frame=scene_frame,
            )
            n_tracklets += added
            n_detections += dets
            n_faces += faces_found
            n_embedded += embedded
            labels.update(found)
        tracker = ByteTracker(tracker_config)
        sample_times, crops, best_score, face_crops, clip_crops = {}, {}, {}, {}, {}
        scene_frame, scene_gap = None, float("inf")

    try:
        for sample in sample_windows(
            path, windows, cfg.detect.detect_fps,
            width=cfg.detect.decode_width, pixel_format="rgb24",
        ):
            if sample.window_index != current_window:
                flush(current_window)
                current_window = sample.window_index

            if on_progress and frames % 8 == 0:
                done = seconds_before[sample.window_index] + max(
                    0.0, sample.t - windows[sample.window_index][0]
                )
                on_progress(min(done, total_seconds), total_seconds)

            frame_h, frame_w = sample.frame.shape[:2]
            if clip is not None:
                gap = abs(sample.t - float(segments[current_window]["peak_offset"]))
                if gap < scene_gap:
                    scene_gap, scene_frame = gap, sample.frame.copy()
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
                        if clip is not None:
                            x1, y1, x2, y2 = (int(v) for v in track.observations[-1][1:])
                            region = sample.frame[max(0, y1):y2, max(0, x1):x2]
                            if region.size:
                                clip_crops[track.track_id] = region.copy()

                if face is not None and track.cls == PERSON_CLASS:
                    _remember_face_crop(
                        face_crops, track, sample.frame, cfg.face.max_faces_per_tracklet
                    )

        flush(current_window)
        if on_progress:
            on_progress(total_seconds, total_seconds)
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
        conn.rollback()
        return AnalyzeResult(video_id, path, "failed", note=f"{type(exc).__name__}: {exc}")

    conn.commit()
    return AnalyzeResult(
        video_id=video_id, path=path, status="analyzed",
        n_segments=len(segments), n_tracklets=n_tracklets, n_detections=n_detections,
        n_faces=n_faces, n_embedded=n_embedded, analysed_seconds=total_seconds,
        frames=frames, elapsed=time.time() - started, labels=labels,
    )


def build_face_pipeline(cfg: Config) -> FacePipeline | None:
    """A face pipeline if the models are present, else None.

    Face models are optional: everything up to Phase 1 works without them, and
    a missing model should degrade the run rather than stop it.
    """
    if not cfg.has_face_models:
        return None

    if cfg.face.stack == "opencv":
        # YuNet and SFace, the permissively licensed pair. Same interface, and
        # on this footage a better detector; see models/face_opencv.py.
        from tsv.models.face_opencv import OpenCVFacePipeline

        return OpenCVFacePipeline(
            cfg.face_detector_path,
            cfg.face_embedder_path,
            conf_threshold=cfg.face.conf_threshold,
        )

    from tsv.models.face import ArcFaceEmbedder, SCRFDDetector

    return FacePipeline(
        SCRFDDetector(
            cfg.face_detector_path,
            size=cfg.face.det_size,
            conf_threshold=cfg.face.conf_threshold,
            force_backend=cfg.face.force_backend,
        ),
        ArcFaceEmbedder(cfg.face_embedder_path, force_backend=cfg.face.force_backend),
    )


def analyze_all(
    conn: sqlite3.Connection,
    cfg: Config,
    force: bool = False,
    detector: Detector | None = None,
    on_result: Callable[[AnalyzeResult], None] | None = None,
    face: FacePipeline | None = None,
    with_faces: bool = True,
    clip: ClipEmbedder | None = None,
    with_clip: bool = True,
    on_progress: Callable[[float, float], None] | None = None,
) -> AnalyzeSummary:
    classes = frozenset(cfg.detect.classes) if cfg.detect.classes else CCTV_CLASSES
    detector = detector or Detector(
        cfg.detect_model_path,
        size=cfg.detect.input_size,
        conf_threshold=cfg.detect.conf_threshold,
        iou_threshold=cfg.detect.iou_threshold,
        keep_classes=classes,
        force_backend=cfg.detect.force_backend,
        family=cfg.detect.family,
    )
    if face is None and with_faces:
        face = build_face_pipeline(cfg)
    if clip is None and with_clip:
        clip = build_clip(
            cfg.model_dir, cfg.clip.image_file, cfg.clip.text_file,
            crop_mode=cfg.clip.crop_mode, force_backend=cfg.clip.force_backend,
        )

    summary = AnalyzeSummary(backend=detector.info)
    if face is not None:
        summary.backend += f" | {face.info}"
    if clip is not None:
        summary.backend += f" | {clip.info}"
    where = "" if force else " WHERE id IN (SELECT video_id FROM segments WHERE analyzed_at IS NULL)"
    rows = conn.execute(f"SELECT id FROM videos{where} ORDER BY start_ts").fetchall()

    for row in rows:
        result = analyze_video(
            conn, int(row["id"]), detector, cfg, force=force, face=face, clip=clip,
            on_progress=on_progress,
        )
        summary.results.append(result)
        if on_result:
            on_result(result)
    return summary
