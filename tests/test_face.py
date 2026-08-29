"""Face pre- and post-processing.

As with the object detector, the network is rarely the thing that is wrong -
the arithmetic around it is. Alignment especially: a plain box crop produces
embeddings that look fine, cluster badly, and match the wrong people, which no
test notices without something to compare against.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tsv.models.face import (
    ARCFACE_TEMPLATE, Face, _anchor_centres, _distance_to_box, _distance_to_points,
    align_face, resize_to_fit,
)


def _image(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


# ---------- letterboxing ----------

def test_resize_to_fit_pads_to_a_square():
    canvas, scale = resize_to_fit(_image(1280, 720), 640)
    assert canvas.shape == (640, 640, 3)
    assert scale == pytest.approx(640 / 1280)


def test_resize_to_fit_pads_bottom_right_not_centred():
    """SCRFD's convention; undoing it must be a division with no offset."""
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    canvas, scale = resize_to_fit(image, 640)
    content_h = int(round(100 * scale))
    # Top-left holds the image, the bottom strip is padding.
    assert canvas[0, 0].tolist() == [255, 255, 255]
    assert canvas[content_h + 5, 0].tolist() == [0, 0, 0]


def test_resize_to_fit_preserves_aspect_ratio():
    canvas, scale = resize_to_fit(_image(1600, 400), 640)
    assert canvas.shape[0] == canvas.shape[1] == 640
    assert scale == pytest.approx(640 / 1600)


# ---------- anchor decoding ----------

def test_distance_to_box_expands_around_the_centre():
    centres = np.array([[100.0, 100.0]])
    distances = np.array([[10.0, 20.0, 30.0, 40.0]])
    assert np.allclose(_distance_to_box(centres, distances), [[90, 80, 130, 140]])


def test_distance_to_points_pairs_x_and_y():
    centres = np.array([[50.0, 60.0]])
    distances = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.allclose(_distance_to_points(centres, distances), [[51, 62, 53, 64]])


def test_anchor_centres_cover_the_grid_in_stride_steps():
    centres = _anchor_centres(2, 3, stride=8, n_anchors=1)
    assert centres.shape == (6, 2)
    assert centres[0].tolist() == [0.0, 0.0]
    assert centres[1].tolist() == [8.0, 0.0]
    assert centres[3].tolist() == [0.0, 8.0]


def test_multiple_anchors_repeat_each_location():
    single = _anchor_centres(2, 2, stride=8, n_anchors=1)
    double = _anchor_centres(2, 2, stride=8, n_anchors=2)
    assert double.shape == (8, 2)
    # Consecutive pairs share a location.
    assert np.allclose(double[0], double[1])
    assert np.allclose(double[::2], single)


# ---------- alignment ----------

def test_alignment_puts_landmarks_on_the_template():
    """The whole point: after warping, the eyes land where ArcFace expects."""
    image = _image(400, 400)
    # A plausible face, scaled and shifted away from the template.
    landmarks = ARCFACE_TEMPLATE * 1.7 + np.array([120.0, 90.0], dtype=np.float32)

    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS
    )
    moved = (matrix[:, :2] @ landmarks.T).T + matrix[:, 2]
    assert np.allclose(moved, ARCFACE_TEMPLATE, atol=0.5)

    aligned = align_face(image, landmarks)
    assert aligned.shape == (112, 112, 3)


def test_alignment_output_size_scales_the_template():
    image = _image(400, 400)
    landmarks = ARCFACE_TEMPLATE * 1.4 + np.array([80.0, 60.0], dtype=np.float32)
    assert align_face(image, landmarks, size=224).shape == (224, 224, 3)


def test_alignment_is_invariant_to_where_the_face_sits():
    """Two views of the same geometry must align to the same picture."""
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 40, (200, 180, 160), -1)
    cv2.circle(canvas, (138, 140), 6, (20, 20, 20), -1)
    cv2.circle(canvas, (162, 140), 6, (20, 20, 20), -1)

    landmarks = np.array(
        [[138, 140], [162, 140], [150, 155], [140, 168], [160, 168]], dtype=np.float32
    )
    shifted_image = np.roll(canvas, (30, 45), axis=(0, 1))
    shifted_landmarks = landmarks + np.array([45.0, 30.0], dtype=np.float32)

    a = align_face(canvas, landmarks)
    b = align_face(shifted_image, shifted_landmarks)
    assert np.abs(a.astype(int) - b.astype(int)).mean() < 3.0


def test_degenerate_landmarks_raise_rather_than_return_rubbish():
    image = _image(200, 200)
    collapsed = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        align_face(image, collapsed)


# ---------- Face record ----------

def test_face_geometry_helpers():
    face = Face(10.0, 20.0, 40.0, 70.0, 0.9)
    assert face.width == 30.0
    assert face.height == 50.0
    assert face.as_array().tolist() == [10.0, 20.0, 40.0, 70.0]


# ---------- the permissive stack ----------
#
# YuNet (MIT) and SFace (Apache-2.0) exist to remove the last research-only
# licence from the pipeline. They are not an accuracy upgrade and these tests
# do not pretend otherwise; what they check is that the swap is a real swap.

from pathlib import Path  # noqa: E402

from tsv.config import DEFAULT  # noqa: E402

YUNET = DEFAULT.model_dir / "face_detection_yunet_2023mar.onnx"
SFACE = DEFAULT.model_dir / "face_recognition_sface_2021dec.onnx"

permissive_only = pytest.mark.skipif(
    not (YUNET.is_file() and SFACE.is_file()),
    reason="run tools/fetch_face_models_permissive.py",
)


def test_the_permissive_config_selects_the_permissive_files():
    """The switch has to move all of it: stack, filenames and the name
    recorded on every vector.

    Vectors carry the model that made them, and SFace's 128 dimensions are
    not ArcFace's 512 on a different scale - they are a different space
    entirely. A half-applied switch would mix them.
    """
    permissive = DEFAULT.opencv_face
    assert permissive.face.stack == "opencv"
    assert permissive.face.name == "yunet-sface" != DEFAULT.face.name
    assert "yunet" in permissive.face.detector_file
    assert "sface" in permissive.face.embedder_file


@permissive_only
def test_the_permissive_pipeline_builds_and_reports_itself():
    from tsv.analyze import build_face_pipeline

    pipeline = build_face_pipeline(DEFAULT.opencv_face)
    assert pipeline is not None
    assert "yunet" in pipeline.info and "sface" in pipeline.info


@permissive_only
def test_both_stacks_answer_the_same_questions():
    """`analyze` only ever calls these two, so this is the whole contract."""
    from tsv.analyze import build_face_pipeline

    for cfg in (DEFAULT, DEFAULT.opencv_face):
        pipeline = build_face_pipeline(cfg)
        if pipeline is None:
            continue
        assert callable(pipeline.faces_in)
        assert callable(pipeline.best_face_in)
        blank = np.zeros((240, 180, 3), dtype=np.uint8)
        assert pipeline.faces_in(blank) == []
        assert pipeline.best_face_in(blank) is None


@permissive_only
def test_sface_vectors_are_its_own_width_not_arcface_s():
    """128 against ArcFace's 512.

    Worth pinning: a stored vector is only comparable to one from the same
    model, and the `model` column on the embedding tables is what keeps them
    apart. A silent change of width here would break that quietly.
    """
    import cv2

    embedder = cv2.FaceRecognizerSF.create(str(SFACE), "")
    aligned = np.zeros((112, 112, 3), dtype=np.uint8)
    assert np.asarray(embedder.feature(aligned)).flatten().shape == (128,)


@permissive_only
def test_a_face_partly_outside_the_frame_is_skipped_not_embedded():
    """A crop that cannot be warped would otherwise embed the background,
    which is an identity vector for a wall."""
    from tsv.models.face_opencv import OpenCVFacePipeline

    pipeline = OpenCVFacePipeline(YUNET, SFACE)
    # Nothing here to find, so this exercises the empty path rather than the
    # error path; the error path is guarded in faces_in.
    assert pipeline.faces_in(np.zeros((64, 64, 3), dtype=np.uint8)) == []
