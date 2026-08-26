"""Box geometry shared by the detector and the tracker.

Boxes are `(x1, y1, x2, y2)` throughout, in pixels of the source frame, with
y increasing downward. Normalisation to 0..1 happens only at the database
boundary, so nothing in the pipeline has to remember which convention it is
holding.
"""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Centre-form `(cx, cy, w, h)` to corner-form. YOLO emits centre-form."""
    out = np.empty_like(boxes, dtype=np.float32)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2.0
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2.0
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def area(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. Returns shape (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    overlap = wh[..., 0] * wh[..., 1]

    union = area(a)[:, None] + area(b)[None, :] - overlap
    # An empty box against an empty box is 0 overlap, not a divide-by-zero.
    return np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def clip_boxes(boxes: np.ndarray, width: float, height: float) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = np.clip(out[:, 0], 0, width)
    out[:, 1] = np.clip(out[:, 1], 0, height)
    out[:, 2] = np.clip(out[:, 2], 0, width)
    out[:, 3] = np.clip(out[:, 3], 0, height)
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns indices into `boxes`."""
    if boxes.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        ious = iou_matrix(boxes[best : best + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return keep


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float
) -> list[int]:
    """NMS applied per class.

    A person standing in front of a car should not suppress the car, so boxes
    are offset into disjoint coordinate bands by class and suppressed in one
    pass.
    """
    if boxes.size == 0:
        return []
    span = float(boxes.max()) + 1.0
    offset = classes.astype(np.float32)[:, None] * span
    return nms(boxes + offset, scores, iou_threshold)
