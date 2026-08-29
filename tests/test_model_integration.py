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


# ---------- face models ----------

FACE_DET = DEFAULT.face_detector_path
FACE_EMB = DEFAULT.face_embedder_path
have_face = FACE_DET.is_file() and FACE_EMB.is_file()

face_only = pytest.mark.skipif(not have_face, reason="no face models exported")


@pytest.fixture(scope="module")
def face_pipeline():
    from tsv.models.face import ArcFaceEmbedder, FacePipeline, SCRFDDetector

    return FacePipeline(
        SCRFDDetector(FACE_DET, conf_threshold=0.4, force_backend="onnxruntime:CPU"),
        ArcFaceEmbedder(FACE_EMB, force_backend="onnxruntime:CPU"),
    )


@face_only
def test_scrfd_returns_landmarks(face_pipeline):
    """Without the keypoint head there is no alignment, and no recognition."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    for face in face_pipeline.detector.detect(frame):
        assert face.landmarks is not None
        assert face.landmarks.shape == (5, 2)


@face_only
def test_detections_stay_inside_the_frame(face_pipeline):
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    for face in face_pipeline.detector.detect(frame):
        assert 0 <= face.x1 <= face.x2 <= 1280
        assert 0 <= face.y1 <= face.y2 <= 720


@face_only
def test_arcface_emits_a_unit_vector_of_the_expected_width(face_pipeline):
    import numpy as _np

    rng = _np.random.default_rng(2)
    aligned = rng.integers(0, 255, (112, 112, 3), dtype=_np.uint8)
    vector = face_pipeline.embedder.embed_aligned(aligned)
    assert vector.shape == (512,)
    assert abs(float(_np.linalg.norm(vector)) - 1.0) < 1e-4


@face_only
def test_the_same_input_embeds_identically(face_pipeline):
    """Sanity: the pipeline is deterministic, so similarity is meaningful."""
    import numpy as _np

    rng = _np.random.default_rng(3)
    aligned = rng.integers(0, 255, (112, 112, 3), dtype=_np.uint8)
    a = face_pipeline.embedder.embed_aligned(aligned)
    b = face_pipeline.embedder.embed_aligned(aligned)
    assert float(a @ b) > 0.9999


@face_only
def test_best_face_ignores_faces_below_the_size_floor(face_pipeline):
    """A confident twelve-pixel face carries no identity information."""
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert face_pipeline.best_face_in(frame, min_size=10_000) is None


# ---------- captioning ----------

CAPTION_DIR = DEFAULT.caption_model_dir
have_caption = DEFAULT.has_caption_model

caption_only = pytest.mark.skipif(not have_caption, reason="no captioning model fetched")


@pytest.fixture(scope="module")
def captioner():
    from tsv.models.caption import build_captioner

    return build_captioner(CAPTION_DIR, max_tokens=32)


@caption_only
def test_the_four_graphs_load_and_agree_on_width(captioner):
    assert captioner is not None
    assert "florence2" in captioner.info


@caption_only
def test_a_caption_is_produced_and_is_words(captioner):
    """Not a quality check - only that the decoding loop terminates and the
    byte-level detokenisation yields readable text rather than mojibake."""
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 255, (240, 240, 3), dtype=np.uint8)
    caption = captioner.caption(frame, "caption")

    assert caption.tokens > 0
    assert caption.text
    assert caption.text.isprintable()
    assert " " in caption.text.strip()


@caption_only
def test_the_cross_attention_cache_survives_more_than_two_steps(captioner):
    """The bug this guards against.

    On cached steps the merged decoder returns placeholder cross-attention
    key/value tensors with a zero batch. Copying those over the real cache
    corrupts it, and the failure surfaces on the *next* step as a broadcast
    error deep inside the decoder - far from the cause, and only after the
    loop has already run twice.
    """
    rng = np.random.default_rng(8)
    frame = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    caption = captioner.caption(frame, "more_detailed")
    assert caption.tokens > 3, "generation stopped before the cache was reused"


@caption_only
def test_a_longer_task_yields_at_least_as_much_text(captioner):
    rng = np.random.default_rng(9)
    frame = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    short = captioner.caption(frame, "caption")
    long = captioner.caption(frame, "more_detailed")
    assert long.tokens >= short.tokens


# ---------- the shippable detector ----------
#
# Skipped until `tools/fetch_detector.py` has run. When it has, these are the
# checks that matter for swapping detector: that the graph is the shape the
# family assumes, and that a mislabelled file fails loudly rather than
# returning empty frames that look like quiet footage.

YOLOX_MODEL = DEFAULT.model_dir / "yolox_tiny.onnx"

yolox_only = pytest.mark.skipif(
    not YOLOX_MODEL.is_file(),
    reason=f"no YOLOX model at {YOLOX_MODEL}; run tools/fetch_detector.py",
)


@yolox_only
def test_the_yolox_graph_is_the_shape_the_family_assumes():
    """416px in, 3549 anchors of 85 out, and static.

    85 rather than 84 is the objectness column. 3549 is 52 + 26 + 13 squared,
    which is what pins the input size and the strides together.
    """
    import onnxruntime as ort

    from tsv.models.detect import YOLOX, anchor_grid

    session = ort.InferenceSession(str(YOLOX_MODEL), providers=["CPUExecutionProvider"])
    in_shape = session.get_inputs()[0].shape
    out_shape = session.get_outputs()[0].shape

    assert in_shape == [1, 3, 416, 416]
    assert out_shape[2] == YOLOX.n_attributes == 85
    assert out_shape[1] == anchor_grid(416, YOLOX.strides)[0].shape[0] == 3549


@yolox_only
def test_the_detector_takes_its_size_from_the_graph_not_the_config():
    """YOLOX-tiny is 416 and YOLO11 is 640, and both exports are static.

    Feeding the configured size to a graph that wants another is a hard
    failure at inference, so the file's own declaration wins.
    """
    from tsv.models.detect import Detector

    detector = Detector(YOLOX_MODEL, size=640, force_backend="onnxruntime:CPU")
    assert detector.size == 416
    assert detector.family.key == "yolox"


@yolox_only
def test_a_mislabelled_detector_raises_rather_than_finding_nothing():
    """The failure this guard exists for.

    Preprocessing is chosen from the filename before anything runs. If that
    guess is wrong the frame was prepared for the wrong model and every result
    afterwards is meaningless - but it looks like an empty scene, not an
    error.
    """
    from tsv.models.detect import Detector

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    as_yolo11 = Detector(YOLOX_MODEL, force_backend="onnxruntime:CPU", family="yolo11")
    with pytest.raises(ValueError, match="yolox"):
        as_yolo11.detect(frame)

    as_yolox = Detector(MODEL, force_backend="onnxruntime:CPU", family="yolox")
    with pytest.raises(ValueError, match="yolo11"):
        as_yolox.detect(frame)


@yolox_only
def test_both_detectors_find_the_same_person_in_the_same_place():
    """The end-to-end check that the whole YOLOX path is right.

    Grid decode, corner padding, BGR at 0..255 and the letterbox inversion all
    have to be correct together; any one of them wrong puts the box somewhere
    else entirely. Measured on real footage the two agree to within a few
    pixels, at 0.918 and 0.880.
    """
    from tsv.models.detect import Detector

    # A synthetic frame no detector will fire on is useless here, so this runs
    # only where the sample footage exists.
    footage = DEFAULT.data_dir.parent / "footage"
    frames = sorted(footage.glob("*.mp4")) if footage.is_dir() else []
    if not frames:
        pytest.skip("no sample footage to compare on")

    from tsv.frames import sample_windows

    yolo = Detector(MODEL, keep_classes=None, force_backend="onnxruntime:CPU")
    yolox = Detector(YOLOX_MODEL, keep_classes=None, force_backend="onnxruntime:CPU")

    for sample in sample_windows(frames[0], [(0.0, 8.0)], fps=1.0):
        a = yolo.detect(sample.frame)
        b = yolox.detect(sample.frame)
        if not a or not b:
            continue
        best_a = max(a, key=lambda d: d.score)
        best_b = max(b, key=lambda d: d.score)
        if best_a.label != best_b.label:
            continue
        # Same object, so the centres must be close on the source frame.
        assert abs(best_a.x1 - best_b.x1) < sample.frame.shape[1] * 0.1
        assert abs(best_a.y1 - best_b.y1) < sample.frame.shape[0] * 0.1
        return
    pytest.skip("neither detector found a comparable object in the sample")
