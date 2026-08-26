from __future__ import annotations

import numpy as np

from tsv.boxes import (
    area, batched_nms, clip_boxes, iou_matrix, nms, xywh_to_xyxy, xyxy_to_xywh,
)


def test_xywh_xyxy_roundtrip():
    boxes = np.array([[50, 60, 20, 40], [10, 10, 4, 4]], dtype=np.float32)
    assert np.allclose(xyxy_to_xywh(xywh_to_xyxy(boxes)), boxes)


def test_xywh_to_xyxy_is_centre_form():
    out = xywh_to_xyxy(np.array([[10, 10, 4, 6]], dtype=np.float32))
    assert np.allclose(out, [[8, 7, 12, 13]])


def test_iou_of_identical_boxes_is_one():
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(box, box)[0, 0] == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[5, 0, 15, 10]], dtype=np.float32)
    # intersection 50, union 150
    assert abs(iou_matrix(a, b)[0, 0] - 50 / 150) < 1e-6


def test_iou_of_degenerate_boxes_does_not_divide_by_zero():
    zero = np.array([[5, 5, 5, 5]], dtype=np.float32)
    assert iou_matrix(zero, zero)[0, 0] == 0.0


def test_iou_matrix_shapes():
    a = np.random.rand(3, 4).astype(np.float32) * 10
    b = np.random.rand(5, 4).astype(np.float32) * 10
    assert iou_matrix(a, b).shape == (3, 5)
    assert iou_matrix(np.empty((0, 4), np.float32), b).shape == (0, 5)


def test_area_of_inverted_box_is_zero():
    assert area(np.array([[10, 10, 5, 5]], dtype=np.float32))[0] == 0.0


def test_clip_keeps_boxes_inside_the_frame():
    boxes = np.array([[-5, -5, 100, 100]], dtype=np.float32)
    assert np.allclose(clip_boxes(boxes, 60, 40), [[0, 0, 60, 40]])


def test_nms_drops_the_weaker_duplicate():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, 0.5)
    assert keep == [0, 2]


def test_nms_keeps_both_when_overlap_is_below_threshold():
    boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(nms(boxes, scores, 0.5)) == [0, 1]


def test_batched_nms_does_not_suppress_across_classes():
    """A person in front of a car must not delete the car."""
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)
    same = batched_nms(boxes, scores, np.array([0, 0]), 0.5)
    different = batched_nms(boxes, scores, np.array([0, 2]), 0.5)
    assert len(same) == 1
    assert len(different) == 2


def test_nms_on_empty_input():
    assert nms(np.empty((0, 4), np.float32), np.empty(0, np.float32), 0.5) == []
