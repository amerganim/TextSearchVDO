"""CLIP's tokenizer, checked against the reference implementation.

Re-implementing a tokenizer is only safe if equivalence is provable. The
export tool dumps both the reference vocabulary and reference token ids for a
set of probe strings; these tests assert this implementation reproduces them
exactly. Without the exported files they skip, so a clean checkout still runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tsv.config import DEFAULT
from tsv.models.tokenizer import (
    CONTEXT_LENGTH, ClipTokenizer, clean_text, load_golden,
)

MERGES = DEFAULT.model_dir / "clip_merges.txt"
VOCAB = DEFAULT.model_dir / "clip_vocab.json"
GOLDEN = DEFAULT.model_dir / "clip_golden.json"

pytestmark = pytest.mark.skipif(
    not MERGES.is_file(), reason=f"no CLIP merges at {MERGES}; run tools/export_clip.py"
)


@pytest.fixture(scope="module")
def tokenizer() -> ClipTokenizer:
    return ClipTokenizer.from_merges_file(MERGES)


def test_vocabulary_matches_the_reference_exactly(tokenizer: ClipTokenizer):
    """The vocabulary is derived, not shipped, so it has to be verified."""
    if not VOCAB.is_file():
        pytest.skip("no reference vocabulary exported")
    reference = json.loads(VOCAB.read_text(encoding="utf-8"))
    assert tokenizer.vocab_size == len(reference)
    disagreeing = [t for t, i in reference.items() if tokenizer.encoder.get(t) != i]
    assert disagreeing == []


def test_token_ids_match_the_reference_for_every_probe(tokenizer: ClipTokenizer):
    if not GOLDEN.is_file():
        pytest.skip("no golden pairs exported")
    for text, expected in load_golden(GOLDEN):
        assert tokenizer.encode(text) == expected, f"diverged on {text!r}"


def test_specials_bracket_the_sequence(tokenizer: ClipTokenizer):
    ids = tokenizer.tokenize("a person walking")[0]
    assert ids[0] == tokenizer.start_id
    body = [int(i) for i in ids if i != 0]
    assert body[-1] == tokenizer.end_id


def test_output_is_padded_to_the_context_width(tokenizer: ClipTokenizer):
    ids = tokenizer.tokenize("short")
    assert ids.shape == (1, CONTEXT_LENGTH)
    assert ids.dtype == np.int32
    assert ids[0, -1] == 0


def test_a_long_prompt_is_truncated_but_keeps_its_end_marker(tokenizer: ClipTokenizer):
    """The text encoder reads the feature at the end token; losing it would
    lose the entire embedding, not just the tail of the prompt."""
    ids = tokenizer.tokenize("a person carrying a large cardboard box " * 20)[0]
    assert len(ids) == CONTEXT_LENGTH
    assert ids[-1] == tokenizer.end_id


def test_batching_several_prompts(tokenizer: ClipTokenizer):
    ids = tokenizer.tokenize(["a dog", "a cat", "a person"])
    assert ids.shape == (3, CONTEXT_LENGTH)
    assert not np.array_equal(ids[0], ids[1])


def test_empty_and_blank_prompts_are_just_the_markers(tokenizer: ClipTokenizer):
    for text in ("", "   ", "\t\n"):
        ids = [int(i) for i in tokenizer.tokenize(text)[0] if i != 0]
        assert ids == [tokenizer.start_id, tokenizer.end_id]


def test_case_and_whitespace_are_normalised(tokenizer: ClipTokenizer):
    assert tokenizer.encode("A  PERSON   walking") == tokenizer.encode("a person walking")


def test_clean_text_collapses_and_unescapes():
    assert clean_text("  A &amp;  B  ") == "a & b"
    assert clean_text("x\n\ty") == "x y"


def test_encoding_is_deterministic(tokenizer: ClipTokenizer):
    """The BPE cache must not change results on a second call."""
    text = "two people talking near a doorway"
    assert tokenizer.encode(text) == tokenizer.encode(text)
