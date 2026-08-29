"""Object detection: letterboxing in, boxes in source pixels out.

Deliberately does not depend on `ultralytics` or torch. Those are export-time
tools; at runtime this is an ONNX graph plus about two hundred lines of numpy,
which is what keeps the install small enough to ship to a user who just wants
to search their own cameras.

**Two model families, and they agree about almost nothing.** Supporting YOLOX
alongside YOLO11 is not a matter of one extra column. Measured against a real
frame, with a person YOLO11 scores at 0.916:

    BGR 0..255, corner pad   0.886   <- YOLOX as trained
    RGB 0..255, corner pad   0.825
    BGR 0..255, centre pad   0.859
    anything scaled to 0..1  0.000

Every difference in that table is a silent failure rather than an error: feed
YOLOX a 0..1 tensor and it returns an empty frame, which looks exactly like a
frame with nothing in it. So preprocessing is a property of the family, and
the family is chosen explicitly rather than inferred from whatever happens to
load. YOLOX matters because it is Apache-2.0 and YOLO11 is AGPL-3.0, which is
the difference between something that can be sold and something that cannot.
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


@dataclass(frozen=True)
class Family:
    """How one lineage of detector wants its input, and returns its output."""

    key: str
    # Per anchor. YOLO11 emits 4 box values plus a score per class; YOLOX adds
    # an objectness column in front of the classes.
    n_attributes: int
    has_objectness: bool
    # Channel order and range the weights were trained on. Getting either
    # wrong costs accuracy silently, or everything at once.
    bgr: bool
    scale_to_unit: bool
    # Where the padding goes. YOLOX pads bottom-right, which its own
    # preprocessing does and its coordinates assume.
    pad: str
    # True when the graph emits raw grid offsets and log-space sizes that
    # still need the anchor decode below. YOLOX's published ONNX does; the
    # Ultralytics export bakes decoding into the graph.
    grid_decode: bool
    strides: tuple[int, ...] = (8, 16, 32)


YOLO11 = Family(
    key="yolo11",
    n_attributes=4 + len(COCO_CLASSES),
    has_objectness=False,
    bgr=False,
    scale_to_unit=True,
    pad="center",
    grid_decode=False,
)

YOLOX = Family(
    key="yolox",
    n_attributes=5 + len(COCO_CLASSES),
    has_objectness=True,
    bgr=True,
    scale_to_unit=False,
    pad="corner",
    grid_decode=True,
)

FAMILIES = {family.key: family for family in (YOLO11, YOLOX)}


def family_for(model_path: Path | str, override: str | None = None) -> Family:
    """Which family a model file belongs to.

    From the filename, because preprocessing has to be decided before the
    graph has ever run - there is no output to inspect yet. `Detector` checks
    the guess against the real output afterwards and raises if they disagree,
    so a misnamed file fails loudly instead of returning empty frames.
    """
    if override:
        if override not in FAMILIES:
            raise ValueError(f"unknown detector family {override!r}")
        return FAMILIES[override]
    name = Path(model_path).name.lower()
    return YOLOX if "yolox" in name else YOLO11


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
    image: np.ndarray, size: int = 640, pad_value: int = 114, pad: str = "center"
) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize preserving aspect ratio and pad to a square.

    Aspect ratio must be preserved: CCTV frames are wide, and squashing them
    to a square makes standing people short and wide, which is exactly the
    shape the detector was not trained on.

    `pad` is "center" or "corner". YOLOX pads bottom-right and its published
    weights were trained that way; centring instead cost about three points of
    confidence when measured, which is the kind of loss nothing reports.
    """
    src_h, src_w = image.shape[:2]
    scale = min(size / src_h, size / src_w)
    new_w, new_h = round(src_w * scale), round(src_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if pad == "corner":
        left = top = 0
        right, bottom = size - new_w, size - new_h
    else:
        pad_x = (size - new_w) / 2.0
        pad_y = (size - new_h) / 2.0
        left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
        top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value),
    )
    return padded, LetterboxMeta(scale, float(left), float(top), src_w, src_h)


def to_input_tensor(letterboxed: np.ndarray, family: Family = YOLO11) -> np.ndarray:
    """HWC uint8 RGB frame to the NCHW float32 tensor this family expects.

    The frame arriving here is always RGB, because that is what the decoder
    upstream produces. Channel order and range then depend entirely on what
    the weights were trained on, and both failures are silent: a YOLOX graph
    handed a 0..1 tensor detects nothing at all.
    """
    frame = letterboxed[:, :, ::-1] if family.bgr else letterboxed
    tensor = frame.astype(np.float32)
    if family.scale_to_unit:
        tensor /= 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...])


def anchor_grid(size: int, strides: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres and their strides, for a graph that did not decode.

    One row per anchor, in the order the feature maps are concatenated:
    stride 8 first, then 16, then 32. At 416 that is 52x52 + 26x26 + 13x13 =
    3549 rows, which is exactly what the published YOLOX graph emits.
    """
    grids, expanded = [], []
    for stride in strides:
        cells = size // stride
        ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing="ij")
        grid = np.stack((xs, ys), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded.append(np.full((grid.shape[0], 1), stride))
    return (
        np.concatenate(grids).astype(np.float32),
        np.concatenate(expanded).astype(np.float32),
    )


def _as_predictions(raw: np.ndarray, n_attributes: int = 4 + len(COCO_CLASSES)) -> np.ndarray:  # noqa: E501
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
    family: Family = YOLO11,
    input_size: int | None = None,
) -> list[Detection]:
    preds = _as_predictions(np.asarray(raw), family.n_attributes)
    if preds.size == 0:
        return []

    boxes_xywh = preds[:, :4]
    if family.grid_decode:
        # Raw offsets into log space. Without this every box lands within a
        # few pixels of the origin, which suppresses to nothing rather than
        # reporting anything wrong.
        size = input_size or int(round(max(meta.src_w, meta.src_h) * meta.scale))
        grid, strides = anchor_grid(size, family.strides)
        if grid.shape[0] != boxes_xywh.shape[0]:
            raise ValueError(
                f"{family.key}: graph returned {boxes_xywh.shape[0]} anchors, "
                f"a {size}px input implies {grid.shape[0]}"
            )
        boxes_xywh = np.column_stack((
            (boxes_xywh[:, :2] + grid) * strides,
            np.exp(boxes_xywh[:, 2:4]) * strides,
        ))

    if family.has_objectness:
        # A YOLOX class column is conditional on there being an object at all,
        # so the two multiply. Taking the class score alone puts confident
        # nonsense on every empty patch of wall.
        objectness = preds[:, 4:5]
        class_scores = preds[:, 5:] * objectness
    else:
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
    """A loaded detection graph plus the pre- and post-processing it wants."""

    def __init__(
        self,
        model_path: Path,
        size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        keep_classes: frozenset[str] | None = CCTV_CLASSES,
        backend: Backend | None = None,
        force_backend: str | None = None,
        family: str | None = None,
    ) -> None:
        self.backend = backend or load_model(model_path, force=force_backend)
        self.family = family_for(model_path, family)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.keep_classes = keep_classes
        self.size = self._graph_size(default=size)
        self._checked_output = False

    def _graph_size(self, default: int) -> int:
        """The input size the graph itself declares, where it declares one.

        These exports are static: YOLOX-tiny is 416, YOLO11 is 640. Feeding
        the configured 640 to a 416 graph is a hard failure at inference, and
        a config default is the wrong place to hold a fact the file already
        knows.
        """
        shape = getattr(self.backend, "input_shape", ()) or ()
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            return int(shape[2])
        return default

    @property
    def info(self) -> str:
        return f"{self.backend.info} {self.family.key}@{self.size}"

    def _check_output(self, raw: np.ndarray) -> None:
        """Confirm the family guess against what the graph actually returned.

        The guess comes from the filename, because preprocessing has to be
        chosen before anything runs. If it was wrong the frame was already
        preprocessed for the wrong model, and every result from here is
        meaningless - so this raises rather than carrying on with a plausible
        empty list.
        """
        self._checked_output = True
        preds = np.asarray(raw)
        attributes = [d for d in preds.shape if d not in (1, 0)]
        if not attributes:
            return
        if self.family.n_attributes not in attributes:
            other = next(
                (f.key for f in FAMILIES.values() if f.n_attributes in attributes),
                None,
            )
            hint = f" - this looks like {other}" if other else ""
            raise ValueError(
                f"detector loaded as {self.family.key}, which expects "
                f"{self.family.n_attributes} values per anchor, but the graph "
                f"returned {attributes}{hint}. Pass family= explicitly."
            )

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        padded, meta = letterbox(frame_rgb, self.size, pad=self.family.pad)
        tensor = to_input_tensor(padded, self.family)
        outputs = self.backend.run({self.backend.input_names[0]: tensor})
        if not self._checked_output:
            self._check_output(outputs[0])
        return decode(
            outputs[0], meta,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            keep_classes=self.keep_classes,
            family=self.family,
            input_size=self.size,
        )
