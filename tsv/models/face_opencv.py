"""A face stack that can be shipped: YuNet detection, SFace recognition.

The incumbent - InsightFace's SCRFD plus ArcFace - is published for
non-commercial research, which makes it the last licence blocker between this
project and a product. These two are the permissive replacement: YuNet is MIT,
SFace is Apache-2.0, and both are distributed by OpenCV, which is already a
dependency here. Nothing new is installed to use them.

**Detection is a wash, and the licence is the whole reason to switch.** An
early fourteen-frame sample suggested YuNet was far better; a hundred-crop
sample says otherwise, and the larger number is the one to believe:

    on 112 full frames     SCRFD 17.9%   YuNet 12.5%
    on 98 person crops     SCRFD 20.4%   YuNet 18.4%

Median face found is 17-21 pixels either way. So this buys no accuracy - it
costs none either, which is the point: the swap is free, and it removes the
last research-only licence from the stack.

**Recognition cannot be judged on this footage at all.** Split by face size,
SFace's ability to tell one person from another:

    all faces (median 18px)   same-person 0.306  different 0.306   +0.000
    faces >= 30px             same-person 0.646  different 0.215   +0.431

At the sizes this footage actually contains there is no separation
whatsoever: same-person pairs and different-person pairs score identically,
so any threshold names strangers as often as it names the right person. The
>= 30px row rests on a single same-person pair, so it indicates rather than
proves. What it points at is that the limit here is the camera rather than
the model: recognition wants roughly 112 pixels of face, and 18 pixels
upscaled carries no identity to recover. No face model, permissive or not,
fixes that.

OpenCV's own wrappers are used rather than a hand-written decode. YuNet has
twelve output heads across three strides, and SFace's alignment expects the
landmarks in YuNet's own order; re-implementing both to gain nothing would be
a good way to introduce a silent misalignment.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tsv.models.face import Face

# Cosine similarity above which OpenCV considers two SFace vectors the same
# person. Their number, not a guess of ours - and a different scale from
# ArcFace's, which is why thresholds cannot be shared between the two stacks.
SFACE_SAME_PERSON = 0.363

# Detection runs on a frame scaled to fit this, not on the full 1080p one.
# Measured at 89 ms a frame full-size against 30 ms for SCRFD at 640; the
# cost is resolution the detector does not use.
DETECT_SIZE = 640


class OpenCVFacePipeline:
    """YuNet and SFace behind the same interface as the InsightFace stack.

    Kept as one class rather than a detector and an embedder that happen to be
    paired, because they genuinely are paired: SFace aligns using YuNet's
    fifteen-column row directly, in YuNet's landmark order. Splitting them
    would mean converting to a neutral landmark format and back, which is
    exactly where an alignment bug would hide.
    """

    def __init__(
        self,
        detector_path: Path,
        embedder_path: Path,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (DETECT_SIZE, DETECT_SIZE),
            score_threshold=conf_threshold, nms_threshold=nms_threshold, top_k=50,
        )
        self._embedder = cv2.FaceRecognizerSF.create(str(embedder_path), "")
        self._detector_name = Path(detector_path).stem
        self._embedder_name = Path(embedder_path).stem

    @property
    def info(self) -> str:
        return f"yunet={self._detector_name} sface={self._embedder_name}"

    def _detect_rows(self, image_bgr: np.ndarray) -> np.ndarray:
        """YuNet's raw rows, in the coordinates of the image passed in.

        Detection happens on a scaled copy and the rows are scaled back, so
        alignment still crops from full resolution. At these face sizes every
        pixel of the crop is worth having.
        """
        height, width = image_bgr.shape[:2]
        scale = min(DETECT_SIZE / max(height, width), 1.0)
        if scale < 1.0:
            small = cv2.resize(
                image_bgr, (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            small = image_bgr

        self._detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self._detector.detect(small)
        if faces is None or not len(faces):
            return np.empty((0, 15), dtype=np.float32)

        rows = np.asarray(faces, dtype=np.float32).copy()
        if scale < 1.0:
            # Everything but the trailing confidence is a coordinate.
            rows[:, :14] /= scale
        return rows

    def faces_in(self, image_rgb: np.ndarray) -> list[tuple[Face, np.ndarray]]:
        image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        out: list[tuple[Face, np.ndarray]] = []

        for row in self._detect_rows(image_bgr):
            x, y, w, h = (float(v) for v in row[:4])
            try:
                aligned = self._embedder.alignCrop(image_bgr, row.reshape(1, -1))
                vector = np.asarray(
                    self._embedder.feature(aligned), dtype=np.float32
                ).flatten()
            except cv2.error:
                # A face partly outside the frame cannot be warped. Skipping it
                # is right: a half-cropped alignment embeds the background.
                continue
            if vector.size == 0:
                continue

            face = Face(
                x1=x, y1=y, x2=x + w, y2=y + h,
                score=float(row[14]),
                landmarks=row[4:14].reshape(5, 2).astype(np.float32),
            )
            out.append((face, vector))
        return out

    def best_face_in(
        self, image_rgb: np.ndarray, min_size: int = 24
    ) -> tuple[Face, np.ndarray] | None:
        """The largest usable face, or None.

        Size rather than confidence, for the reason the numbers at the top of
        this file make concrete: a confident eighteen-pixel face carries no
        identity at all.
        """
        candidates = [
            (face, vector)
            for face, vector in self.faces_in(image_rgb)
            if min(face.width, face.height) >= min_size
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0].width * pair[0].height)
