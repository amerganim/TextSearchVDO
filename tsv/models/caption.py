"""Florence-2 captioning: describing what a person is doing.

This is the piece that answers the *"when did father take his medicine"* class
of question. Detection knows a person is present; identity knows who; zones
know where. None of them can say what someone is holding or doing, and that is
what a caption adds.

Florence-2-base-ft is 230M parameters and purpose-built for describing an
image on demand, which is why it is here rather than a general 3-4B
vision-language model: on a CPU-only machine the large ones caption slowly
enough to make a day of footage an overnight job at best.

**No generation library is used.** ONNX Runtime executes graphs; it does not
run a decoding loop, so the loop is written out here - four graphs, a KV
cache, and greedy sampling. The same reasoning as everywhere else in this
project: the arithmetic around a model is where the mistakes live, so it is
visible rather than imported.

**No tokenizer library either.** Florence-2's prompts are fixed task strings,
so encoding is a dictionary lookup done once, and decoding is the reverse map
plus GPT-2's byte-level convention. That is a hundred lines against a
dependency tree, and it keeps captioning optional rather than infectious.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from tsv.models.backend import CPU_FIRST_PREFERENCE, Backend, load_model

# Florence-2 was trained at this size, with ImageNet statistics.
IMAGE_SIZE = 768
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The task prompts. Florence-2 is steered entirely by these; there is no free
# instruction text.
TASKS = {
    "caption": "<CAPTION>",
    "detailed": "<DETAILED_CAPTION>",
    "more_detailed": "<MORE_DETAILED_CAPTION>",
}

# The shape of the decoder's key-value cache, which differs per model size:
# base is 6 layers of 12 heads, large is 12 of 16. These are the fallbacks
# for a graph that will not declare itself; the real values are read from the
# decoder's own inputs at load time. Hardcoding them is what kept the large
# model from working - it loaded, ran, and produced nothing usable, because
# every cache tensor was the wrong shape.
DECODER_LAYERS = 6
NUM_HEADS = 12
HEAD_DIM = 64

BOS_ID, PAD_ID, EOS_ID = 0, 1, 2
# Florence-2 starts decoding from the EOS token, as BART does.
DECODER_START_ID = 2


@lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Reverse of GPT-2's byte-to-unicode map, for turning tokens into bytes."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    mapped = printable[:]
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {chr(c): b for b, c in zip(printable, mapped)}


class Vocabulary:
    """Just enough of a tokenizer to prompt the model and read it back.

    Prompting needs no encoder at all: Florence-2 expands each task token into
    a fixed English sentence, and those were tokenised once at fetch time into
    prompts.json. Reading the answer back is the reverse vocabulary plus GPT-2
    byte decoding.
    """

    def __init__(self, token_to_id: dict[str, int], prompts: dict[str, dict]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.prompts = prompts

    @classmethod
    def load(cls, model_dir: Path) -> "Vocabulary":
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        prompts = json.loads((model_dir / "prompts.json").read_text(encoding="utf-8"))
        return cls(vocab, prompts)

    def prompt_ids(self, task_token: str) -> list[int]:
        entry = self.prompts.get(task_token)
        if entry is None:
            raise KeyError(
                f"{task_token!r} was not tokenised at fetch time; "
                f"have {sorted(self.prompts)}"
            )
        return list(entry["ids"])



    def decode(self, ids: list[int]) -> str:
        byte_decoder = _byte_decoder()
        pieces: list[str] = []
        for token_id in ids:
            if token_id in (BOS_ID, PAD_ID, EOS_ID):
                continue
            token = self.id_to_token.get(token_id)
            if token is None or (token.startswith("<") and token.endswith(">")):
                continue
            pieces.append(token)

        raw = "".join(pieces)
        try:
            text = bytearray(byte_decoder.get(ch, ord(ch)) for ch in raw).decode(
                "utf-8", errors="replace"
            )
        except ValueError:
            text = raw
        return text.strip()


def preprocess(image_rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize to the trained square and normalise.

    Squashed rather than letterboxed, deliberately: Florence-2 was trained on
    whole images resized this way, and a grey border is a thing it has never
    seen. The crops fed to it are padded to a sane aspect ratio before they
    arrive - see `crop_for_caption`.
    """
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(np.transpose(tensor, (2, 0, 1))[None, ...])


def crop_for_caption(
    frame_rgb: np.ndarray,
    box: tuple[float, float, float, float],
    context: float = 0.35,
    min_side: int = 96,
) -> np.ndarray | None:
    """A person crop with room around them, squared off.

    Context matters more here than for a face: what someone is holding, and
    what is on the table next to them, is usually just outside the detector's
    box. Squaring the crop before the model squashes it to 768 keeps a
    standing person from being flattened.
    """
    height, width = frame_rgb.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 8 or box_h < 8:
        return None

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(box_w, box_h) * (1 + context)
    half = side / 2

    left = int(max(0, cx - half))
    top = int(max(0, cy - half))
    right = int(min(width, cx + half))
    bottom = int(min(height, cy + half))
    if right - left < min_side or bottom - top < min_side:
        # Too small to describe; upscaling invents detail rather than finding it.
        return None
    return frame_rgb[top:bottom, left:right]


@dataclass
class Caption:
    text: str
    task: str
    tokens: int


class Florence2Captioner:
    """Four ONNX graphs and a greedy decoding loop."""

    def __init__(
        self,
        model_dir: Path,
        max_tokens: int = 48,
        force_backend: str | None = None,
    ) -> None:
        self.dir = model_dir
        self.max_tokens = max_tokens
        self.vocab = Vocabulary.load(model_dir)

        def graph(name: str) -> Backend:
            return load_model(
                model_dir / f"{name}.onnx",
                preference=CPU_FIRST_PREFERENCE,
                force=force_backend,
            )

        self.vision = graph("vision_encoder")
        self.embed = graph("embed_tokens")
        self.encoder = graph("encoder_model")
        self.decoder = graph("decoder_model_merged")

    @property
    def info(self) -> str:
        return f"florence2={self.vision.info}"

    # ---- the pieces ----

    def _image_features(self, image_rgb: np.ndarray) -> np.ndarray:
        pixel_values = preprocess(image_rgb)
        return np.asarray(self.vision.run({"pixel_values": pixel_values})[0])

    def _embed(self, ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.embed.run({"input_ids": ids.astype(np.int64)})[0])

    def _encode(self, image_features: np.ndarray, prompt_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        text_embeds = self._embed(np.array([prompt_ids], dtype=np.int64))
        # Image tokens first, then the task prompt - the order Florence-2 was
        # trained with.
        inputs_embeds = np.concatenate([image_features, text_embeds], axis=1)
        attention_mask = np.ones(inputs_embeds.shape[:2], dtype=np.int64)
        hidden = self.encoder.run(
            {"attention_mask": attention_mask, "inputs_embeds": inputs_embeds}
        )[0]
        return np.asarray(hidden), attention_mask

    @property
    def cache_shape(self) -> tuple[int, int, int]:
        """(layers, heads, head dim), read from the decoder graph itself.

        Every one of these is declared on the graph's own `past_key_values`
        inputs, so asking is both easier and more honest than keeping a table
        of model sizes that has to be updated whenever a new one appears -
        which is precisely how the large model came to be listed as available
        while not working.
        """
        if getattr(self, "_cache_shape", None) is None:
            self._cache_shape = self._read_cache_shape()
        return self._cache_shape

    def _read_cache_shape(self) -> tuple[int, int, int]:
        session = getattr(self.decoder, "_session", None)
        if session is None:
            return DECODER_LAYERS, NUM_HEADS, HEAD_DIM

        layers, heads, dim = set(), None, None
        for spec in session.get_inputs():
            if "past_key_values" not in spec.name:
                continue
            parts = spec.name.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                layers.add(int(parts[1]))
            shape = list(spec.shape)
            # [batch, heads, sequence, head_dim] - the two fixed axes are the
            # ones worth reading; the others are symbolic.
            if len(shape) == 4:
                if isinstance(shape[1], int):
                    heads = shape[1]
                if isinstance(shape[3], int):
                    dim = shape[3]

        if not layers or heads is None or dim is None:
            return DECODER_LAYERS, NUM_HEADS, HEAD_DIM
        return len(layers), int(heads), int(dim)

    def _empty_cache(self, batch: int) -> dict[str, np.ndarray]:
        layers, heads, dim = self.cache_shape
        empty = np.zeros((batch, heads, 0, dim), dtype=np.float32)
        cache = {}
        for layer in range(layers):
            for side in ("decoder", "encoder"):
                for kind in ("key", "value"):
                    cache[f"past_key_values.{layer}.{side}.{kind}"] = empty
        return cache

    def _step(
        self,
        token_ids: np.ndarray,
        encoder_hidden: np.ndarray,
        encoder_mask: np.ndarray,
        cache: dict[str, np.ndarray],
        first: bool,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        inputs = {
            "encoder_attention_mask": encoder_mask,
            "encoder_hidden_states": encoder_hidden,
            "inputs_embeds": self._embed(token_ids),
            "use_cache_branch": np.array([not first], dtype=bool),
            **cache,
        }
        outputs = self.decoder.run(inputs)
        logits = np.asarray(outputs[0])

        updated = dict(cache)
        for name, value in zip(self.decoder_output_names[1:], outputs[1:]):
            key = name.replace("present", "past_key_values")
            # Only the self-attention cache grows. Cross-attention keys and
            # values depend on the encoder output alone, so the graph computes
            # them once on the first pass and returns *placeholders* -
            # zero-batch tensors - on every cached step afterwards. Copying
            # those back over the real ones corrupts the cache, and the failure
            # surfaces one step later as a broadcast error deep inside the
            # decoder rather than anywhere near here.
            if first or ".decoder." in key:
                updated[key] = np.asarray(value)
        return logits, updated

    @property
    def decoder_output_names(self) -> list[str]:
        if not hasattr(self, "_decoder_out_names"):
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(self.dir / "decoder_model_merged.onnx"),
                providers=["CPUExecutionProvider"],
            )
            self._decoder_out_names = [o.name for o in session.get_outputs()]
        return self._decoder_out_names

    # ---- the loop ----

    def caption(self, image_rgb: np.ndarray, task: str = "detailed") -> Caption:
        prompt = TASKS.get(task, TASKS["detailed"])
        prompt_ids = self.vocab.prompt_ids(prompt)

        image_features = self._image_features(image_rgb)
        encoder_hidden, encoder_mask = self._encode(image_features, prompt_ids)

        cache = self._empty_cache(encoder_hidden.shape[0])
        generated: list[int] = []
        next_ids = np.array([[DECODER_START_ID]], dtype=np.int64)
        first = True

        for _ in range(self.max_tokens):
            logits, cache = self._step(
                next_ids, encoder_hidden, encoder_mask, cache, first
            )
            first = False
            token = int(np.argmax(logits[0, -1]))
            if token == EOS_ID:
                break
            generated.append(token)
            next_ids = np.array([[token]], dtype=np.int64)

        return Caption(
            text=self.vocab.decode(generated), task=task, tokens=len(generated)
        )


def build_captioner(
    model_dir: Path, max_tokens: int = 48, force_backend: str | None = None
) -> Florence2Captioner | None:
    """A captioner if the graphs are present, else None.

    `model_dir` is the directory holding the graphs themselves. Accepts the
    parent too, for convenience at a prompt, but the caller is expected to
    pass `Config.caption_model_dir`.
    """
    needed = ("vision_encoder", "embed_tokens", "encoder_model", "decoder_model_merged")

    def complete(directory: Path) -> bool:
        return all((directory / f"{n}.onnx").is_file() for n in needed) and all(
            (directory / n).is_file() for n in ("vocab.json", "prompts.json")
        )

    for candidate in (model_dir, model_dir / "florence2"):
        if complete(candidate):
            return Florence2Captioner(
                candidate, max_tokens=max_tokens, force_backend=force_backend
            )
    return None
