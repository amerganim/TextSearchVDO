"""Detector pre- and post-processing, tested without any model weights.

The correctness risk in a detector integration is almost never the network -
it is the coordinate maths around it. A letterbox that is not undone exactly
puts every box slightly wrong, which then quietly poisons tracking, zones and
every downstream phase.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsv.models.detect import (
    COCO_CLASSES, Detection, decode, letterbox, to_input_tensor, _as_predictions,
)


def _frame(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def test_letterbox_output_is_square_and_padded():
    padded, meta = letterbox(_frame(1280, 720), size=640)
    assert padded.shape == (640, 640, 3)
    assert meta.pad_y > 0 and meta.pad_x == 0     # wide frame pads top/bottom
    assert abs(meta.scale - 640 / 1280) < 1e-6


def test_letterbox_preserves_aspect_ratio():
    """Squashing a wide frame would make standing people short and wide."""
    padded, meta = letterbox(_frame(1600, 400), size=640)
    content_h = 400 * meta.scale
    content_w = 1600 * meta.scale
    assert abs(content_w / content_h - 1600 / 400) < 1e-6
    assert padded.shape[0] == padded.shape[1] == 640


def test_letterbox_of_an_already_square_frame_adds_no_padding():
    _, meta = letterbox(_frame(500, 500), size=640)
    assert meta.pad_x == 0 and meta.pad_y == 0


def test_input_tensor_layout_and_range():
    padded, _ = letterbox(_frame(1280, 720), size=640)
    tensor = to_input_tensor(padded)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0


@pytest.mark.parametrize("src", [(1280, 720), (640, 480), (720, 1280), (500, 500)])
def test_letterbox_inverse_maps_a_known_box_back_exactly(src):
    """The round trip is what every downstream coordinate depends on."""
    width, height = src
    _, meta = letterbox(_frame(width, height), size=640)

    truth = np.array([width * 0.25, height * 0.4, width * 0.55, height * 0.9])
    # Forward: source pixels into letterboxed space.
    fwd = truth * meta.scale
    fwd[[0, 2]] += meta.pad_x
    fwd[[1, 3]] += meta.pad_y

    # A single-anchor prediction placing a box exactly there.
    cx, cy = (fwd[0] + fwd[2]) / 2, (fwd[1] + fwd[3]) / 2
    w, h = fwd[2] - fwd[0], fwd[3] - fwd[1]
    raw = np.zeros((1, 4 + 80, 1), dtype=np.float32)
    raw[0, :4, 0] = [cx, cy, w, h]
    raw[0, 4 + 0, 0] = 0.9        # class 0 = person

    got = decode(raw, meta, conf_threshold=0.5, keep_classes=None)
    assert len(got) == 1
    assert np.allclose(got[0].as_array(), truth, atol=0.75)


def test_decode_transposed_layout_is_handled():
    """Some exports emit (1, anchors, 84) instead of (1, 84, anchors)."""
    _, meta = letterbox(_frame(640, 640), size=640)
    raw = np.zeros((1, 84, 1), dtype=np.float32)
    raw[0, :4, 0] = [320, 320, 100, 200]
    raw[0, 4, 0] = 0.9
    straight = decode(raw, meta, conf_threshold=0.5, keep_classes=None)
    flipped = decode(np.transpose(raw, (0, 2, 1)), meta, conf_threshold=0.5, keep_classes=None)
    assert len(straight) == len(flipped) == 1
    assert np.allclose(straight[0].as_array(), flipped[0].as_array())


def test_as_predictions_orients_by_attribute_count():
    assert _as_predictions(np.zeros((1, 84, 8400))).shape == (8400, 84)
    assert _as_predictions(np.zeros((1, 8400, 84))).shape == (8400, 84)
    # A frame with fewer detections than attributes must not be transposed.
    assert _as_predictions(np.zeros((1, 84, 1))).shape == (1, 84)
    assert _as_predictions(np.zeros((1, 84, 3))).shape == (3, 84)


def test_confidence_threshold_filters():
    _, meta = letterbox(_frame(640, 640), size=640)
    raw = np.zeros((1, 84, 2), dtype=np.float32)
    raw[0, :4, 0] = [100, 100, 40, 80]; raw[0, 4, 0] = 0.9
    raw[0, :4, 1] = [400, 400, 40, 80]; raw[0, 4, 1] = 0.1
    assert len(decode(raw, meta, conf_threshold=0.5, keep_classes=None)) == 1
    assert len(decode(raw, meta, conf_threshold=0.05, keep_classes=None)) == 2


def test_class_filter_keeps_only_requested_labels():
    _, meta = letterbox(_frame(640, 640), size=640)
    raw = np.zeros((1, 84, 2), dtype=np.float32)
    raw[0, :4, 0] = [100, 100, 40, 80]
    raw[0, 4 + COCO_CLASSES.index("person"), 0] = 0.9
    raw[0, :4, 1] = [400, 400, 40, 80]
    raw[0, 4 + COCO_CLASSES.index("toaster"), 1] = 0.9

    got = decode(raw, meta, conf_threshold=0.5, keep_classes=frozenset({"person"}))
    assert [d.label for d in got] == ["person"]


def test_boxes_are_clipped_to_the_frame():
    """A detection hanging off the edge must not produce negative pixels."""
    _, meta = letterbox(_frame(640, 480), size=640)
    raw = np.zeros((1, 84, 1), dtype=np.float32)
    raw[0, :4, 0] = [10, 10, 400, 400]   # centred near the corner, huge
    raw[0, 4, 0] = 0.9
    got = decode(raw, meta, conf_threshold=0.5, keep_classes=None)[0]
    assert got.x1 >= 0 and got.y1 >= 0
    assert got.x2 <= 640 and got.y2 <= 480


def test_decode_of_empty_output():
    _, meta = letterbox(_frame(640, 640), size=640)
    assert decode(np.zeros((1, 84, 0), dtype=np.float32), meta) == []


def test_detection_label_lookup():
    assert Detection(0, 0, 1, 1, 0.9, 0).label == "person"
