"""Face detection (SCRFD) and recognition (ArcFace), on our own backend layer.

As with the detector, neither insightface nor torch is a runtime dependency:
these are plain ONNX graphs plus the arithmetic around them. That arithmetic
is where the mistakes live, so it is written out rather than imported.

Two parts are easy to get subtly wrong and quietly ruin recognition:

**Alignment.** ArcFace is trained on faces warped so the eyes, nose and mouth
corners sit at fixed pixel positions. Feeding it a plain crop of the detector's
box produces embeddings that look plausible, cluster badly, and match the wrong
people. The five landmarks SCRFD returns exist for this and must be used.

**Colour and scale.** Both networks want RGB scaled to roughly [-1, 1], not
BGR and not 0..1. A channel swap does not crash; it just degrades accuracy in a
way no test would notice without ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tsv.boxes import nms
from tsv.models.backend import CPU_FIRST_PREFERENCE, Backend, load_model

# Where ArcFace expects the five landmarks to land in a 112x112 crop. These
# are the reference positions the model was trained against; they are not
# adjustable.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],   # right eye
        [73.5318, 51.5014],   # left eye
        [56.0252, 71.7366],   # nose tip
        [41.5493, 92.3655],   # right mouth corner
        [70.7299, 92.2041],   # left mouth corner
    ],
    dtype=np.float32,
)

SCRFD_STRIDES = (8, 16, 32)
SCRFD_MEAN, SCRFD_STD = 127.5, 128.0
ARCFACE_MEAN, ARCFACE_STD = 127.5, 127.5


@dataclass
class Face:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    # Five landmarks in source-image pixels, or None if the graph has no
    # keypoint head - in which case recognition cannot be aligned.
    landmarks: np.ndarray | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


def _distance_to_box(centres: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """SCRFD predicts distances from an anchor centre to each box edge."""
    x1 = centres[:, 0] - distances[:, 0]
    y1 = centres[:, 1] - distances[:, 1]
    x2 = centres[:, 0] + distances[:, 2]
    y2 = centres[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance_to_points(centres: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Landmarks are offsets from the same anchor centre, as (x, y) pairs."""
    points = []
    for i in range(0, distances.shape[1], 2):
        points.append(centres[:, 0] + distances[:, i])
        points.append(centres[:, 1] + distances[:, i + 1])
    return np.stack(points, axis=-1)


def _anchor_centres(height: int, width: int, stride: int, n_anchors: int) -> np.ndarray:
    grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
    centres = (grid * stride).reshape(-1, 2)
    if n_anchors > 1:
        centres = np.stack([centres] * n_anchors, axis=1).reshape(-1, 2)
    return centres


def resize_to_fit(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Scale to fit a square, pad bottom-right. Returns (canvas, scale).

    SCRFD pads to the bottom right rather than centring, so undoing it is a
    plain division with no offset.
    """
    src_h, src_w = image.shape[:2]
    scale = min(size / src_h, size / src_w)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((size, size, 3), dtype=image.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas, scale


def align_face(image_rgb: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp a face so its landmarks sit where ArcFace expects them."""
    template = ARCFACE_TEMPLATE.copy()
    if size != 112:
        template = template * (size / 112.0)

    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32).reshape(5, 2), template, method=cv2.LMEDS
    )
    if matrix is None:
        raise ValueError("could not fit an alignment transform to these landmarks")
    return cv2.warpAffine(image_rgb, matrix, (size, size), borderValue=0)


class SCRFDDetector:
    """Anchor-based face detector.

    Handles both SCRFD variants: with a keypoint head (9 outputs) and without
    (6). The anchor count per location is derived from the output shape rather
    than hard-coded, because it differs between the released models.
    """

    def __init__(
        self,
        model_path: Path,
        size: int = 640,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.4,
        backend: Backend | None = None,
        force_backend: str | None = None,
    ) -> None:
        self.backend = backend or load_model(
            model_path, preference=CPU_FIRST_PREFERENCE, force=force_backend
        )
        self.size = size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @property
    def info(self) -> str:
        return str(self.backend.info)

    def detect(self, image_rgb: np.ndarray) -> list[Face]:
        canvas, scale = resize_to_fit(image_rgb, self.size)

        blob = (canvas.astype(np.float32) - SCRFD_MEAN) / SCRFD_STD
        blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None, ...])
        outputs = self.backend.run({self.backend.input_names[0]: blob})

        n_levels = len(SCRFD_STRIDES)
        has_landmarks = len(outputs) >= n_levels * 3

        boxes: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        points: list[np.ndarray] = []

        for level, stride in enumerate(SCRFD_STRIDES):
            level_scores = np.asarray(outputs[level]).reshape(-1)
            level_boxes = np.asarray(outputs[level + n_levels]).reshape(-1, 4) * stride

            grid_h, grid_w = self.size // stride, self.size // stride
            n_anchors = max(1, len(level_scores) // (grid_h * grid_w))
            centres = _anchor_centres(grid_h, grid_w, stride, n_anchors)

            keep = np.flatnonzero(level_scores >= self.conf_threshold)
            if keep.size == 0:
                continue

            boxes.append(_distance_to_box(centres, level_boxes)[keep])
            scores.append(level_scores[keep])
            if has_landmarks:
                level_points = np.asarray(outputs[level + n_levels * 2]).reshape(-1, 10) * stride
                points.append(_distance_to_points(centres, level_points)[keep])

        if not boxes:
            return []

        all_boxes = np.concatenate(boxes) / scale
        all_scores = np.concatenate(scores)
        all_points = (np.concatenate(points) / scale) if points else None

        # One class, so plain NMS rather than the class-aware variant.
        order = nms(all_boxes, all_scores, self.iou_threshold)

        height, width = image_rgb.shape[:2]
        faces = []
        for i in order:
            box = np.clip(all_boxes[i], [0, 0, 0, 0], [width, height, width, height])
            landmarks = all_points[i].reshape(5, 2) if all_points is not None else None
            faces.append(Face(*map(float, box), float(all_scores[i]), landmarks))
        return faces


class ArcFaceEmbedder:
    """512-dimensional face embeddings from aligned crops."""

    def __init__(
        self,
        model_path: Path,
        size: int = 112,
        backend: Backend | None = None,
        force_backend: str | None = None,
    ) -> None:
        self.backend = backend or load_model(
            model_path, preference=CPU_FIRST_PREFERENCE, force=force_backend
        )
        self.size = size

    @property
    def info(self) -> str:
        return str(self.backend.info)

    def embed_aligned(self, aligned_rgb: np.ndarray) -> np.ndarray:
        blob = (aligned_rgb.astype(np.float32) - ARCFACE_MEAN) / ARCFACE_STD
        blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None, ...])
        vector = np.asarray(
            self.backend.run({self.backend.input_names[0]: blob})[0]
        ).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else vector

    def embed(self, image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        return self.embed_aligned(align_face(image_rgb, landmarks, self.size))


class FacePipeline:
    """Detect the faces in a frame and embed each one."""

    def __init__(self, detector: SCRFDDetector, embedder: ArcFaceEmbedder) -> None:
        self.detector = detector
        self.embedder = embedder

    @property
    def info(self) -> str:
        return f"scrfd={self.detector.info} arcface={self.embedder.info}"

    def faces_in(self, image_rgb: np.ndarray) -> list[tuple[Face, np.ndarray]]:
        out = []
        for face in self.detector.detect(image_rgb):
            if face.landmarks is None:
                continue
            try:
                out.append((face, self.embedder.embed(image_rgb, face.landmarks)))
            except ValueError:
                continue
        return out

    def best_face_in(
        self, image_rgb: np.ndarray, min_size: int = 24
    ) -> tuple[Face, np.ndarray] | None:
        """The largest usable face, or None.

        Size is a better proxy for usefulness than detector confidence here: a
        confident twelve-pixel face carries almost no identity information.
        """
        candidates = [
            (face, vector)
            for face, vector in self.faces_in(image_rgb)
            if min(face.width, face.height) >= min_size
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0].width * pair[0].height)
