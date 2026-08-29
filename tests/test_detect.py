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


# ---------- model families ----------
#
# YOLOX matters because it is Apache-2.0 where YOLO11 is AGPL-3.0, which is
# the difference between something that can be sold and something that cannot.
# Almost every difference between the two families fails *silently* - a wrong
# channel order or value range returns an empty frame, which is
# indistinguishable from a frame with nothing in it.

from tsv.models.detect import (  # noqa: E402
    FAMILIES, YOLO11, YOLOX, Detector, anchor_grid, family_for,
)


def test_the_two_families_disagree_about_everything_that_matters():
    """Documents why this is not a one-column change."""
    assert YOLOX.n_attributes == YOLO11.n_attributes + 1     # objectness
    assert YOLOX.has_objectness and not YOLO11.has_objectness
    assert YOLOX.bgr and not YOLO11.bgr
    assert YOLO11.scale_to_unit and not YOLOX.scale_to_unit
    assert YOLOX.pad == "corner" and YOLO11.pad == "center"
    assert YOLOX.grid_decode and not YOLO11.grid_decode


def test_the_family_is_read_from_the_filename():
    """It has to be: preprocessing is chosen before the graph has ever run."""
    assert family_for("yolox_tiny.onnx") is YOLOX
    assert family_for("data/models/yolox_s.onnx") is YOLOX
    assert family_for("yolo11n.onnx") is YOLO11
    assert family_for("something_unknown.onnx") is YOLO11
    assert family_for("yolo11n.onnx", "yolox") is YOLOX
    with pytest.raises(ValueError):
        family_for("yolo11n.onnx", "nonsense")


def test_corner_padding_puts_the_image_where_yolox_expects_it():
    padded, meta = letterbox(_frame(1280, 720), size=640, pad="corner")
    assert padded.shape == (640, 640, 3)
    assert meta.pad_x == 0 and meta.pad_y == 0
    # The bottom strip is padding, the top row is image.
    assert (padded[-1] == 114).all()


def test_the_input_tensor_follows_the_family_not_the_caller():
    """Both of these are silent failures when wrong.

    A YOLOX graph handed a 0..1 tensor detects nothing whatsoever - measured,
    0.000 against 0.886 for the same frame correctly prepared.
    """
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[..., 0] = 200        # red channel, in an RGB frame

    yolo = to_input_tensor(frame, YOLO11)
    assert yolo.max() <= 1.0
    assert yolo[0, 0].max() > 0.5, "RGB order lost: red should be channel 0"

    yolox = to_input_tensor(frame, YOLOX)
    assert yolox.max() > 1.0, "YOLOX weights expect 0..255, not 0..1"
    assert yolox[0, 2].max() > 128, "BGR order lost: red should be channel 2"


def test_the_anchor_grid_matches_the_published_graph():
    """3549 rows at 416px is exactly what YOLOX-tiny returns."""
    grid, strides = anchor_grid(416, (8, 16, 32))
    assert grid.shape == (3549, 2)
    assert strides.shape == (3549, 1)
    assert set(np.unique(strides)) == {8, 16, 32}
    # Stride 8 first, and its first cell is the origin.
    assert strides[0] == 8 and tuple(grid[0]) == (0.0, 0.0)
    assert strides[-1] == 32

    at_640, _ = anchor_grid(640, (8, 16, 32))
    assert at_640.shape == (8400, 2)      # what YOLO11 emits at 640


def test_grid_decoding_turns_offsets_into_pixels():
    """Without it every box lands near the origin and suppresses to nothing."""
    n = 3549
    raw = np.zeros((1, n, 5 + len(COCO_CLASSES)), dtype=np.float32)
    grid, strides = anchor_grid(416, (8, 16, 32))

    # One object at the centre of the last stride-32 cell, half a cell wide.
    target = n - 1
    raw[0, target, :2] = 0.5
    raw[0, target, 2:4] = np.log(0.5)
    raw[0, target, 4] = 0.9            # objectness
    raw[0, target, 5] = 0.9            # person

    meta = letterbox(_frame(416, 416), size=416, pad="corner")[1]
    found = decode(raw, meta, conf_threshold=0.5, family=YOLOX, input_size=416)

    assert len(found) == 1
    assert found[0].label == "person"
    assert found[0].score == pytest.approx(0.81, abs=0.01)   # 0.9 * 0.9

    expected_cx = (grid[target, 0] + 0.5) * strides[target, 0]
    assert (found[0].x1 + found[0].x2) / 2 == pytest.approx(expected_cx, abs=1.0)
    assert (found[0].x2 - found[0].x1) == pytest.approx(0.5 * 32, abs=1.0)


def test_objectness_gates_the_class_score():
    """A class column alone puts confident nonsense on every empty wall."""
    n = 3549
    raw = np.zeros((1, n, 5 + len(COCO_CLASSES)), dtype=np.float32)
    raw[0, 0, 4] = 0.02        # nothing here
    raw[0, 0, 5] = 0.99        # but "person" if you ignore that
    meta = letterbox(_frame(416, 416), size=416, pad="corner")[1]

    assert decode(raw, meta, conf_threshold=0.25, family=YOLOX, input_size=416) == []


def test_an_anchor_count_that_cannot_be_a_grid_is_refused():
    """Better a loud error than boxes computed against the wrong strides."""
    raw = np.zeros((1, 999, 5 + len(COCO_CLASSES)), dtype=np.float32)
    meta = letterbox(_frame(416, 416), size=416, pad="corner")[1]
    with pytest.raises(ValueError, match="anchors"):
        decode(raw, meta, family=YOLOX, input_size=416)
