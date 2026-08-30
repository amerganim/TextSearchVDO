"""Captioning: the arithmetic around the model, and the pass that uses it.

The model itself is checked by the integration tests, which skip when it has
not been fetched. What is checked here is everything else - cropping, byte
decoding, the prompt table, and the resumable pass over stored tracklets -
because that is where the mistakes live.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

from tsv import db
from tsv.config import DEFAULT
from tsv.models.caption import (
    TASKS, Caption, Vocabulary, _byte_decoder, build_captioner, crop_for_caption,
    preprocess,
)


def _frame(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


# ---------- preprocessing ----------

def test_preprocess_produces_the_trained_shape():
    tensor = preprocess(_frame(400, 900))
    assert tensor.shape == (1, 3, 768, 768)
    assert tensor.dtype == np.float32


def test_preprocess_normalises_rather_than_just_scaling():
    """ImageNet statistics, not 0..1: a plain rescale silently degrades it."""
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    tensor = preprocess(flat)
    assert abs(float(tensor.mean())) < 0.6
    assert float(tensor.std()) < 0.5


# ---------- cropping ----------

def test_crop_is_squared_and_padded_with_context():
    """A tall person crop squashed to a square loses their proportions, and
    what they are holding is usually just outside the detector's box."""
    frame = _frame(1000, 1000)
    crop = crop_for_caption(frame, (400, 200, 500, 700), context=0.35)
    assert crop is not None
    height, width = crop.shape[:2]
    # Square-ish, and wider than the original box.
    assert abs(height - width) <= 2
    assert width > 100


def test_crop_is_clipped_to_the_frame():
    frame = _frame(300, 300)
    crop = crop_for_caption(frame, (10, 10, 120, 290))
    assert crop is not None
    assert crop.shape[0] <= 300 and crop.shape[1] <= 300


def test_a_tiny_box_is_refused_rather_than_upscaled():
    """Upscaling a 20-pixel person invents detail instead of finding it."""
    frame = _frame(1000, 1000)
    assert crop_for_caption(frame, (10, 10, 30, 40), min_side=96) is None


def test_a_degenerate_box_is_refused():
    frame = _frame(200, 200)
    assert crop_for_caption(frame, (50, 50, 52, 52)) is None


# ---------- vocabulary ----------

@pytest.fixture
def vocabulary(tmp_path) -> Vocabulary:
    (tmp_path / "vocab.json").write_text(
        json.dumps({"Hello": 10, "Ġworld": 11, "!": 12, "<s>": 0}), encoding="utf-8"
    )
    (tmp_path / "prompts.json").write_text(
        json.dumps({"<CAPTION>": {"text": "What does the image describe?", "ids": [0, 1, 2]}}),
        encoding="utf-8",
    )
    return Vocabulary.load(tmp_path)


def test_prompt_ids_come_from_the_precomputed_table(vocabulary):
    assert vocabulary.prompt_ids("<CAPTION>") == [0, 1, 2]


def test_an_untokenised_task_says_so_rather_than_guessing(vocabulary):
    with pytest.raises(KeyError, match="fetch time"):
        vocabulary.prompt_ids("<OCR>")


def test_decoding_handles_the_byte_level_space_convention(vocabulary):
    """GPT-2 encodes a leading space as U+0120, not a space."""
    assert vocabulary.decode([10, 11, 12]) == "Hello world!"


def test_decoding_drops_control_tokens(vocabulary):
    assert vocabulary.decode([0, 10, 0]) == "Hello"


def test_byte_decoder_round_trips_every_byte():
    decoder = _byte_decoder()
    assert len(decoder) == 256
    assert len(set(decoder.values())) == 256


# ---------- the pass ----------

@pytest.fixture
def indexed(tmp_path):
    """One video, two person tracklets and a car, none captioned."""
    cfg = dataclasses.replace(DEFAULT, data_dir=tmp_path)
    conn = db.open_db(cfg.db_path)
    conn.execute("INSERT INTO cameras(id, name) VALUES (1,'ch01')")
    conn.execute(
        """INSERT INTO videos(id, camera_id, path, start_ts, ts_source, duration)
           VALUES (1,1,'missing.mp4',1000.0,'test',60)"""
    )
    conn.execute(
        """INSERT INTO segments(id, video_id, camera_id, t_start, t_end,
                                ts_start, ts_end, activity_score, peak_offset)
           VALUES (1,1,1,0,30,1000,1030,0.1,5)"""
    )
    for tid, label in ((1, "person"), (2, "person"), (3, "car")):
        conn.execute(
            """INSERT INTO tracklets(id, segment_id, video_id, camera_id, cls, label,
                                     t_start, t_end, ts_start, ts_end,
                                     n_detections, mean_score, max_score,
                                     x1, y1, x2, y2)
               VALUES (?,1,1,1,0,?,0,10,1000,1010,5,0.9,0.95,0.2,0.1,0.5,0.9)""",
            (tid, label),
        )
    conn.commit()
    return conn, cfg


def test_only_the_configured_classes_are_considered(indexed):
    from tsv.captioning import pending_tracklets

    conn, cfg = indexed
    rows = pending_tracklets(conn, cfg)
    assert {r["label"] for r in rows} == {"person"}
    assert len(rows) == 2


def test_already_captioned_tracklets_are_skipped(indexed):
    """The pass is resumable: an interrupted run costs one tracklet."""
    from tsv.captioning import pending_tracklets

    conn, cfg = indexed
    conn.execute("UPDATE tracklets SET caption = 'a person walking' WHERE id = 1")
    conn.commit()

    assert [r["id"] for r in pending_tracklets(conn, cfg)] == [2]
    assert len(pending_tracklets(conn, cfg, force=True)) == 2


def test_a_missing_video_is_counted_not_fatal(indexed):
    from tsv.captioning import caption_tracklets

    conn, cfg = indexed

    class Stub:
        def caption(self, image, task):
            raise AssertionError("should never be reached")

    summary = caption_tracklets(conn, cfg, captioner=Stub())
    assert summary.considered == 2
    assert summary.failed == 2
    assert summary.captioned == 0


def test_captions_reach_the_word_index(indexed):
    """The point of the whole feature: words no other stage produces."""
    from tsv.search import lexical_ranking, rebuild_text_index, segment_document

    conn, cfg = indexed
    conn.execute(
        "UPDATE tracklets SET caption = ? WHERE id = 1",
        ("A man taking a pill from an orange medicine bottle",),
    )
    conn.commit()

    body = segment_document(conn, 1)
    assert "medicine" in body

    rebuild_text_index(conn)
    assert [sid for sid, _ in lexical_ranking(conn, "medicine")] == [1]
    assert lexical_ranking(conn, "helicopter") == []


def test_captioning_is_on_by_default_but_only_where_the_model_is():
    """It used to be opt-in, and the cost is the reason - about six seconds an
    image.

    That was still the wrong trade. An undescribed sighting turns "carrying a
    bag" into an empty screen indistinguishable from the moment not being in
    the video, and nobody remembers to press a button before searching. It
    runs last, so everything else is answering questions while it works, and
    an installation without the weights is unaffected: the importer ands this
    with `has_caption_model`.
    """
    assert DEFAULT.caption.enabled is True


def test_build_captioner_returns_none_when_the_model_is_absent(tmp_path):
    assert build_captioner(tmp_path) is None


def test_the_task_table_covers_the_prompts_that_were_fetched():
    assert set(TASKS.values()) == {
        "<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"
    }


def test_the_outstanding_count_falls_as_tracklets_are_described(indexed):
    """What the app's button counts down.

    Written against the database rather than through the API, because the
    quantity being checked is bookkeeping - whether a described tracklet stops
    being offered as work - and not whether the model writes good text.
    """
    from tsv.captioning import pending_tracklets

    conn, cfg = indexed
    assert len(pending_tracklets(conn, cfg)) == 2

    conn.execute("UPDATE tracklets SET caption = 'a person on the stairs' WHERE id = 1")
    conn.commit()
    assert len(pending_tracklets(conn, cfg)) == 1

    outstanding = conn.execute(
        "SELECT COUNT(*) c FROM tracklets WHERE caption IS NULL AND label = 'person'"
    ).fetchone()["c"]
    assert outstanding == 1


# ---------- switching model size ----------

def test_the_cache_shape_is_read_from_the_graph_not_hardcoded():
    """This is what kept the large model from working.

    base is 6 layers of 12 heads, large is 12 of 16. With those as module
    constants the large graph loaded, ran, and produced nothing usable -
    every key-value tensor was the wrong shape. Asking the graph is both
    easier and survives the next model.
    """
    source = (
        Path(__file__).resolve().parent.parent / "tsv" / "models" / "caption.py"
    ).read_text(encoding="utf-8")
    assert "_read_cache_shape" in source
    assert "past_key_values" in source
    # The constants remain, but only as a fallback for a graph that will not
    # declare itself.
    assert "self.cache_shape" in source


@pytest.mark.skipif(
    not (DEFAULT.model_dir / "florence2-large" / "vision_encoder.onnx").is_file(),
    reason="run tools/fetch_caption_model.py --model large",
)
def test_base_and_large_report_different_cache_shapes():
    from tsv.models.caption import build_captioner

    base = build_captioner(DEFAULT.model_dir / "florence2")
    large = build_captioner(DEFAULT.model_dir / "florence2-large")
    assert base is not None and large is not None
    assert base.cache_shape == (6, 12, 64)
    assert large.cache_shape == (12, 16, 64)
    assert base.cache_shape != large.cache_shape


def test_the_two_sizes_live_in_separate_folders():
    """So both can be installed and switched between without re-downloading."""
    assert DEFAULT.caption.model_dir == "florence2"
    assert DEFAULT.large_captions.caption.model_dir == "florence2-large"
    assert DEFAULT.large_captions.caption.name != DEFAULT.caption.name


def test_an_empty_caption_falls_back_rather_than_being_stored():
    """Stored as-is it is worse than a failure.

    Florence-2 large returns an empty string for some crops under
    more_detailed while answering the same image under detailed. Written to
    the database that tracklet counts as captioned, is never retried, and can
    never be found by anything it contains.
    """
    from tsv.captioning import _FALLBACK_TASKS, _describe

    class Stub:
        def __init__(self):
            self.asked = []

        def caption(self, crop, task):
            self.asked.append(task)
            text = "" if task == "more_detailed" else "a person on the stairs"
            return Caption(text=text, task=task, tokens=len(text.split()))

    stub = Stub()
    result = _describe(stub, object(), "more_detailed")
    assert result is not None
    assert result.text == "a person on the stairs"
    assert stub.asked[0] == "more_detailed", "the wanted task must be tried first"
    assert len(stub.asked) > 1, "it gave up after one empty answer"


def test_a_model_that_never_answers_is_a_failure_not_an_empty_caption():
    from tsv.captioning import _describe

    class Silent:
        def caption(self, crop, task):
            return Caption(text="   ", task=task, tokens=0)

    assert _describe(Silent(), object(), "more_detailed") is None
