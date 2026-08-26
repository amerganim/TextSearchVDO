"""CLIP image and text encoders: the "search by text" half of the product.

Both encoders are ONNX graphs on the shared backend layer, and the tokenizer
is pure Python, so nothing here imports transformers or torch at runtime.

**Preprocessing diverges from CLIP's own on purpose.** The reference pipeline
resizes the shortest side then centre-crops to a square. On the crops this
system produces that is actively wrong: a standing person is roughly 1:2.5, so
a centre crop throws away their head and their feet and keeps a band of torso.
Padding to a square instead keeps the whole subject, at the cost of some grey
border the model never saw in training. Queries here are about people, what
they carry and what they wear, and losing the head to preserve training
fidelity is the worse trade. `crop_mode="center"` restores the original
behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import cv2
import numpy as np

from tsv.models.backend import CPU_FIRST_PREFERENCE, Backend, load_model
from tsv.models.tokenizer import ClipTokenizer, load_tokenizer

# CLIP's channel statistics. Not adjustable; the model was trained with these.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

CropMode = Literal["pad", "center"]


def preprocess_image(
    image_rgb: np.ndarray, size: int = 224, crop_mode: CropMode = "pad"
) -> np.ndarray:
    """One RGB frame to a normalised NCHW batch of 1."""
    if crop_mode == "center":
        h, w = image_rgb.shape[:2]
        scale = size / min(h, w)
        resized = cv2.resize(
            image_rgb, (max(size, int(round(w * scale))), max(size, int(round(h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        top = (resized.shape[0] - size) // 2
        left = (resized.shape[1] - size) // 2
        square = resized[top : top + size, left : left + size]
    else:
        h, w = image_rgb.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        # Mid grey rather than black: a black border reads as a dark region and
        # drags night-time queries toward every padded crop.
        square = np.full((size, size, 3), 124, dtype=np.uint8)
        top, left = (size - new_h) // 2, (size - new_w) // 2
        square[top : top + new_h, left : left + new_w] = resized

    tensor = square.astype(np.float32) / 255.0
    tensor = (tensor - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def pick_embedding(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """The projected embedding among a graph's outputs.

    Exports vary in how many tensors they expose: calling CLIP's
    `get_*_features` emits both the raw per-token hidden states and the
    projected vector, in that order, so taking the first output silently
    yields a 38400-long sequence rather than a 512-long embedding. The
    embedding is the one shaped (batch, dim), so select it by rank rather than
    by position.
    """
    arrays = [np.asarray(o) for o in outputs]
    flat = [a for a in arrays if a.ndim == 2]
    if flat:
        # If several qualify, the projection is the narrowest.
        return min(flat, key=lambda a: a.shape[-1])
    return arrays[0]


@dataclass
class ClipPaths:
    image_model: Path
    text_model: Path
    model_dir: Path

    @property
    def complete(self) -> bool:
        return self.image_model.is_file() and self.text_model.is_file()


class ClipImageEncoder:
    def __init__(
        self,
        model_path: Path,
        size: int = 224,
        crop_mode: CropMode = "pad",
        backend: Backend | None = None,
        force_backend: str | None = None,
    ) -> None:
        self.backend = backend or load_model(
            model_path, preference=CPU_FIRST_PREFERENCE, force=force_backend
        )
        self.size = size
        self.crop_mode = crop_mode

    @property
    def info(self) -> str:
        return str(self.backend.info)

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        tensor = preprocess_image(image_rgb, self.size, self.crop_mode)
        outputs = self.backend.run({self.backend.input_names[0]: tensor})
        return _unit(pick_embedding(outputs))


class ClipTextEncoder:
    def __init__(
        self,
        model_path: Path,
        tokenizer: ClipTokenizer,
        backend: Backend | None = None,
        force_backend: str | None = None,
    ) -> None:
        self.backend = backend or load_model(
            model_path, preference=CPU_FIRST_PREFERENCE, force=force_backend
        )
        self.tokenizer = tokenizer

    @property
    def info(self) -> str:
        return str(self.backend.info)

    def embed(self, text: str) -> np.ndarray:
        ids = self.tokenizer.tokenize(text).astype(np.int64)
        outputs = self.backend.run({self.backend.input_names[0]: ids})
        return _unit(pick_embedding(outputs))

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        """One row per text. The graph is exported with a batch of one, so
        these run in a loop rather than as a batch."""
        return np.stack([self.embed(t) for t in texts]) if texts else np.empty((0, 0), np.float32)


class ClipEmbedder:
    """Both encoders, sharing a tokenizer."""

    def __init__(self, image: ClipImageEncoder, text: ClipTextEncoder) -> None:
        self.image = image
        self.text = text

    @property
    def info(self) -> str:
        return f"clip-image={self.image.info} clip-text={self.text.info}"

    @property
    def dim(self) -> int:
        return int(self.text.embed("a photo").shape[0])

    def embed_image(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.image.embed(image_rgb)

    def embed_text(self, text: str) -> np.ndarray:
        return self.text.embed(text)


def build_clip(
    model_dir: Path,
    image_file: str = "clip_image.onnx",
    text_file: str = "clip_text.onnx",
    crop_mode: CropMode = "pad",
    force_backend: str | None = None,
) -> ClipEmbedder | None:
    """A CLIP embedder if the graphs and merge table are present, else None."""
    paths = ClipPaths(model_dir / image_file, model_dir / text_file, model_dir)
    if not paths.complete:
        return None
    try:
        tokenizer = load_tokenizer(model_dir)
    except FileNotFoundError:
        return None

    return ClipEmbedder(
        ClipImageEncoder(paths.image_model, crop_mode=crop_mode, force_backend=force_backend),
        ClipTextEncoder(paths.text_model, tokenizer, force_backend=force_backend),
    )
