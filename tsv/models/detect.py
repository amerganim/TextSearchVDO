"""YOLO detection: letterboxing in, boxes in source pixels out.

Deliberately does not depend on `ultralytics` or torch. Those are export-time
tools; at runtime this is an ONNX graph plus about a hundred lines of numpy,
which is what keeps the install small enough to ship to a user who just wants
to search their own cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tsv.boxes import batched_nms, clip_boxes, xywh_to_xyxy
from tsv.models.backend import Backend, load_model

# COCO, in the order every YOLO export uses.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# What a home or shop camera actually cares about. Everything else in COCO is
# noise indoors, and filtering early keeps the tracker's matching problem
# small. Override per deployment rather than editing this.
CCTV_CLASSES: frozenset[str] = frozenset({
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "cat", "dog", "bird",
    "backpack", "umbrella", "handbag", "suitcase", "bottle", "cup", "bowl",
    "chair", "laptop", "cell phone", "book", "scissors",
})


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def label(self) -> str:
        return COCO_CLASSES[self.cls] if self.cls < len(COCO_CLASSES) else str(self.cls)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int


def letterbox(
    image: np.ndarray, size: int = 640, pad_value: int = 114
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize preserving aspect ratio and pad to a square.

    Aspect ratio must be preserved: CCTV frames are wide, and squashing them
    to a square makes standing people short and wide, which is exactly the
    shape the detector was not trained on.
    """
    src_h, src_w = image.shape[:2]
    scale = min(size / src_h, size / src_w)
    new_w, new_h = round(src_w * scale), round(src_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return padded, LetterboxMeta(scale, float(left), float(top), src_w, src_h)


def to_input_tensor(letterboxed: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB to NCHW float32 in 0..1."""
    tensor = letterboxed.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...])


def _as_predictions(raw: np.ndarray, n_attributes: int = 4 + len(COCO_CLASSES)) -> np.ndarray:
    """Normalise the export's layout to (n_anchors, 4 + n_classes).

    Exports differ: most emit (1, 84, 8400), some (1, 8400, 84). Orienting by
    the known attribute count rather than by "anchors are the bigger axis",
    because the latter silently transposes any output that happens to carry
    fewer anchors than attributes - a frame with a single detection, or a
    model run at a small input size.
    """
    preds = raw[0] if raw.ndim == 3 else raw
    if preds.shape[1] == n_attributes:
        return preds
    if preds.shape[0] == n_attributes:
        return preds.T
    # Unknown class count: fall back to the size heuristic.
    return preds.T if preds.shape[0] < preds.shape[1] else preds


def decode(
    raw: np.ndarray,
    meta: LetterboxMeta,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    keep_classes: frozenset[str] | None = None,
    max_detections: int = 300,
) -> list[Detection]:
    preds = _as_predictions(np.asarray(raw))
    if preds.size == 0:
        return []

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    if class_scores.shape[1] == 0:
        return []

    best_cls = np.argmax(class_scores, axis=1)
    best_score = class_scores[np.arange(len(class_scores)), best_cls]

    mask = best_score >= conf_threshold
    if keep_classes is not None:
        allowed = np.array(
            [i for i, name in enumerate(COCO_CLASSES) if name in keep_classes],
            dtype=np.int64,
        )
        mask &= np.isin(best_cls, allowed)
    if not mask.any():
        return []

    boxes = xywh_to_xyxy(boxes_xywh[mask])
    scores = best_score[mask].astype(np.float32)
    classes = best_cls[mask].astype(np.int64)

    # Undo the letterbox before suppression, so the IoU threshold means the
    # same thing it would on the source frame.
    boxes[:, [0, 2]] -= meta.pad_x
    boxes[:, [1, 3]] -= meta.pad_y
    boxes /= meta.scale
    boxes = clip_boxes(boxes, meta.src_w, meta.src_h)

    keep = batched_nms(boxes, scores, classes, iou_threshold)[:max_detections]
    return [
        Detection(
            float(boxes[i, 0]), float(boxes[i, 1]),
            float(boxes[i, 2]), float(boxes[i, 3]),
            float(scores[i]), int(classes[i]),
        )
        for i in keep
    ]


class Detector:
    """A loaded YOLO ONNX graph plus its pre- and post-processing."""

    def __init__(
        self,
        model_path: Path,
        size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        keep_classes: frozenset[str] | None = CCTV_CLASSES,
        backend: Backend | None = None,
        force_backend: str | None = None,
    ) -> None:
        self.backend = backend or load_model(model_path, force=force_backend)
        self.size = size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.keep_classes = keep_classes

    @property
    def info(self) -> str:
        return str(self.backend.info)

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        padded, meta = letterbox(frame_rgb, self.size)
        tensor = to_input_tensor(padded)
        outputs = self.backend.run({self.backend.input_names[0]: tensor})
        return decode(
            outputs[0], meta,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            keep_classes=self.keep_classes,
        )
