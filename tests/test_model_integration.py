"""Checks against the real exported ONNX graph.

Skipped when no model has been exported, so the suite still runs on a clean
checkout. What these catch is an export whose *shape* disagrees with what
`tsv.models.detect` assumes - a graph with NMS baked in, a dynamic axis, or a
different class count - which would otherwise surface as silently empty or
nonsensical detections rather than an error.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsv.config import DEFAULT
from tsv.models.backend import load_model
from tsv.models.detect import COCO_CLASSES, Detector, _as_predictions

MODEL = DEFAULT.detect_model_path

pytestmark = pytest.mark.skipif(
    not MODEL.is_file(), reason=f"no exported model at {MODEL}"
)


@pytest.fixture(scope="module")
def backend():
    return load_model(MODEL, force="onnxruntime:CPU")


def test_a_backend_loads_and_reports_itself(backend):
    assert str(backend.info) == "onnxruntime:CPU"
    assert len(backend.input_names) == 1


def test_input_shape_is_static_and_square(backend):
    """`dynamic=False` at export; OpenVINO compiles static shapes far better."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert shape[0] == 1
    assert shape[1] == 3
    assert shape[2] == shape[3] == DEFAULT.detect.input_size
    assert all(isinstance(d, int) for d in shape), f"dynamic axis in {shape}"


def test_output_is_raw_predictions_not_nms_output(backend):
    """`nms=False` at export.

    A graph with NMS baked in emits a handful of rows shaped (n, 6); ours must
    emit one column per anchor with 4 + 80 attributes, because the class-aware
    NMS lives in Python where it can be tuned per deployment.
    """
    size = DEFAULT.detect.input_size
    dummy = np.zeros((1, 3, size, size), dtype=np.float32)
    outputs = backend.run({backend.input_names[0]: dummy})

    assert len(outputs) == 1, "an NMS graph usually emits several outputs"
    preds = _as_predictions(np.asarray(outputs[0]))
    assert preds.shape[1] == 4 + len(COCO_CLASSES)
    assert preds.shape[0] > 1000, "too few anchors to be a raw YOLO head"


def test_a_synthetic_person_shape_is_detected():
    """A crude figure on a plain ground should register as *something*.

    Deliberately weak: this asserts the graph, the preprocessing and the
    decode agree well enough to produce boxes at all. Whether the detector is
    any good is a question for real footage, not a unit test.
    """
    detector = Detector(MODEL, conf_threshold=0.10, keep_classes=None, force_backend="onnxruntime:CPU")

    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    # Head, torso, legs - a stick figure with human proportions.
    frame[150:190, 300:340] = 40
    frame[190:300, 285:355] = 60
    frame[300:400, 295:315] = 45
    frame[300:400, 325:345] = 45

    detections = detector.detect(frame)
    for det in detections:
        assert 0 <= det.x1 < det.x2 <= 640
        assert 0 <= det.y1 < det.y2 <= 480
        assert 0.0 <= det.score <= 1.0


def test_detections_from_noise_are_not_wildly_out_of_frame():
    """Random input must still produce in-bounds, well-formed boxes."""
    detector = Detector(MODEL, conf_threshold=0.05, keep_classes=None, force_backend="onnxruntime:CPU")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    for det in detector.detect(frame):
        assert 0 <= det.x1 <= det.x2 <= 1280
        assert 0 <= det.y1 <= det.y2 <= 720
        assert det.cls < len(COCO_CLASSES)
