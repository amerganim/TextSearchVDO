"""CLIP's byte-level BPE tokenizer, in pure Python.

Written out rather than imported so the runtime stays ONNX plus numpy. It is
about a hundred lines, it never changes, and correctness is verifiable: the
export tool dumps token ids produced by the reference implementation, and the
tests assert this one reproduces them exactly.

The one deliberate divergence is the pre-tokenisation regex. The reference uses
the `regex` module's Unicode property classes (\\p{L}, \\p{N}); the standard
library's `re` has no equivalent, so those are approximated. For Latin-script
queries - which is what a search box over English captions receives - the two
agree, and the golden-pair test covers the cases that matter.
"""

from __future__ import annotations

import gzip
import html
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

# CLIP's fixed context width. Every prompt is padded or truncated to this.
CONTEXT_LENGTH = 77

START_TOKEN = "<|startoftext|>"
END_TOKEN = "<|endoftext|>"

# \p{L}+ -> [^\W\d_]+ (letters), \p{N} -> \d (one digit at a time, as CLIP
# does), and the punctuation class becomes "not space, not letter, not digit".
_PATTERN = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d"
    r"|[^\W\d_]+|\d|[^\s\w]+|_+",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _byte_encoder() -> dict[int, str]:
    """Reversible map from bytes to printable unicode characters.

    BPE operates on text, but the input is bytes; the bytes that are not
    printable get moved into an unused block so nothing is lost.
    """
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
    return {b: chr(c) for b, c in zip(printable, mapped)}


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


def clean_text(text: str) -> str:
    """Unescape entities and collapse whitespace, as CLIP does."""
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip().lower()


class ClipTokenizer:
    def __init__(self, merges: list[tuple[str, str]]) -> None:
        vocab = list(_byte_encoder().values())
        vocab += [v + "</w>" for v in vocab]
        vocab += ["".join(merge) for merge in merges]
        vocab += [START_TOKEN, END_TOKEN]

        self.encoder = {token: i for i, token in enumerate(vocab)}
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}
        self.cache = {START_TOKEN: START_TOKEN, END_TOKEN: END_TOKEN}
        self.start_id = self.encoder[START_TOKEN]
        self.end_id = self.encoder[END_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    @classmethod
    def from_merges_file(cls, path: Path) -> "ClipTokenizer":
        """Load CLIP's `bpe_simple_vocab_16e6.txt.gz`, or a plain-text copy."""
        raw = (
            gzip.open(path, "rt", encoding="utf-8").read()
            if path.suffix == ".gz"
            else path.read_text(encoding="utf-8")
        )
        lines = raw.split("\n")
        # Line 0 is a version banner; the vocabulary is 49152 entries, of which
        # 256 single bytes, 256 byte+</w>, and 2 specials are not merges.
        lines = lines[1 : 49152 - 256 - 2 + 1]
        return cls([tuple(line.split()) for line in lines if line])  # type: ignore[misc]

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            first, second = best

            merged: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    merged.extend(word[i:])
                    break
                merged.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1

            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        """Token ids for one string, without the start/end markers."""
        byte_encoder = _byte_encoder()
        ids: list[int] = []
        for token in _PATTERN.findall(clean_text(text)):
            token = "".join(byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[piece] for piece in self.bpe(token).split(" "))
        return ids

    def tokenize(
        self, texts: str | list[str], context_length: int = CONTEXT_LENGTH
    ) -> np.ndarray:
        """A padded (n, context_length) int32 batch ready for the text encoder."""
        if isinstance(texts, str):
            texts = [texts]

        out = np.zeros((len(texts), context_length), dtype=np.int32)
        for row, text in enumerate(texts):
            ids = [self.start_id, *self.encode(text), self.end_id]
            if len(ids) > context_length:
                # Truncate, but keep the end marker: the text encoder reads the
                # feature at the end-token position, so losing it loses the
                # entire embedding rather than just the tail of the prompt.
                ids = ids[:context_length]
                ids[-1] = self.end_id
            out[row, : len(ids)] = ids
        return out


def load_tokenizer(model_dir: Path) -> ClipTokenizer:
    """Find CLIP's merges alongside the exported models."""
    for name in ("bpe_simple_vocab_16e6.txt.gz", "clip_merges.txt"):
        path = model_dir / name
        if path.is_file():
            return ClipTokenizer.from_merges_file(path)
    raise FileNotFoundError(
        f"no CLIP merges file in {model_dir}; run tools/export_clip.py"
    )


def load_golden(path: Path) -> list[tuple[str, list[int]]]:
    """Reference (text, token ids) pairs dumped at export time."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(item["text"], item["ids"]) for item in data]
