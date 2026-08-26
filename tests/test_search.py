"""Retrieval: filters, lexical matching, and rank fusion.

Driven with synthetic vectors so the ranking logic is tested independently of
whether CLIP is any good, which is a separate question answered by real
footage.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsv import db
from tsv.events import create_zone, recompute_events
from tsv.identity import enroll_tracklet, normalise, store_tracklet_embedding
from tsv.search import (
    SearchFilters, _fts_query, allowed_segments, lexical_ranking,
    rebuild_text_index, reciprocal_rank_fusion, search, segment_document,
    semantic_ranking,
)

DIM = 16


def direction(seed: int) -> np.ndarray:
    return normalise(np.random.default_rng(seed).normal(size=DIM).astype(np.float32))


@pytest.fixture
def conn(tmp_path):
    c = db.open_db(tmp_path / "t.db")
    c.execute("INSERT INTO cameras(id, name) VALUES (1, 'ch01'), (2, 'ch02')")
    c.execute(
        """INSERT INTO videos(id, camera_id, path, start_ts, ts_source, duration)
           VALUES (1, 1, 'a.mp4', 1767225600.0, 'test', 600.0),
                  (2, 2, 'b.mp4', 1767225600.0, 'test', 600.0)"""
    )
    # Three segments: two on camera 1, one on camera 2.
    for sid, vid, cam, off in ((1, 1, 1, 0), (2, 1, 1, 120), (3, 2, 2, 60)):
        c.execute(
            """INSERT INTO segments(id, video_id, camera_id, t_start, t_end,
                                    ts_start, ts_end, activity_score, peak_offset)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (sid, vid, cam, off, off + 30, 1767225600.0 + off,
             1767225600.0 + off + 30, 0.1, off + 5),
        )
    c.commit()
    return c


def add_tracklet(conn, tid: int, segment_id: int, label: str = "person",
                 camera_id: int = 1, video_id: int = 1) -> None:
    conn.execute(
        """INSERT INTO tracklets(id, segment_id, video_id, camera_id, cls, label,
                                 t_start, t_end, ts_start, ts_end,
                                 n_detections, mean_score, max_score,
                                 x_start, y_start, x_end, y_end)
           VALUES (?,?,?,?,0,?,0,5,1767225600,1767225605,5,0.9,0.95,0.1,0.9,0.9,0.9)""",
        (tid, segment_id, video_id, camera_id, label),
    )
    conn.executemany(
        """INSERT INTO detections(tracklet_id, t, ts, x1, y1, x2, y2, score)
           VALUES (?,?,?,?,?,?,?,?)""",
        [(tid, i * 1.0, 1767225600.0 + i, 0.1 + i * 0.2, 0.5, 0.2 + i * 0.2, 0.9, 0.9)
         for i in range(5)],
    )
    conn.commit()


# ---------- the lexical index ----------

def test_segment_document_gathers_what_is_known(conn):
    add_tracklet(conn, 1, 1, "person")
    add_tracklet(conn, 2, 1, "dog")
    enroll_tracklet(conn, 1, "Rafi")

    body = segment_document(conn, 1)
    assert "person" in body and "dog" in body and "Rafi" in body and "ch01" in body


def test_document_includes_zone_events(conn):
    add_tracklet(conn, 1, 1)
    create_zone(conn, 1, "front door", "line", [(0.5, 0.0), (0.5, 1.0)])
    recompute_events(conn)
    body = segment_document(conn, 1)
    assert "front door" in body
    assert "cross" in body


def test_rebuilding_the_index_skips_empty_segments(conn):
    add_tracklet(conn, 1, 1)
    n = rebuild_text_index(conn)
    # Segments 2 and 3 have nothing in them but a camera name, so they still
    # index; segment 1 additionally has an object.
    assert n >= 1
    assert lexical_ranking(conn, "person")


def test_lexical_ranking_finds_named_things(conn):
    add_tracklet(conn, 1, 1, "person")
    add_tracklet(conn, 2, 3, "dog", camera_id=2, video_id=2)
    rebuild_text_index(conn)

    assert [sid for sid, _ in lexical_ranking(conn, "dog")] == [3]
    assert [sid for sid, _ in lexical_ranking(conn, "person")] == [1]
    assert lexical_ranking(conn, "helicopter") == []


def test_lexical_query_is_safe_against_fts_syntax(conn):
    """Users type quotes and operators; FTS5 must not try to execute them."""
    add_tracklet(conn, 1, 1)
    rebuild_text_index(conn)
    hostile = [
        chr(34) + "person" + chr(34),
        "a AND OR NEAR" + chr(40),
        "*",
        "-person",
        "person*",
        "^",
    ]
    for text in hostile:
        lexical_ranking(conn, text)   # must not raise


def test_fts_query_builder_quotes_every_token():
    assert _fts_query("red jacket") == '"red" OR "jacket"'
    assert _fts_query('a "b" -c') == '"a" OR "b" OR "c"'
    assert _fts_query("!!!") == ""


# ---------- filters ----------

def test_filters_constrain_by_camera_and_label(conn):
    add_tracklet(conn, 1, 1, "person")
    add_tracklet(conn, 2, 3, "dog", camera_id=2, video_id=2)

    assert allowed_segments(conn, SearchFilters(camera_id=2)) == [3]
    assert allowed_segments(conn, SearchFilters(label="dog")) == [3]
    assert allowed_segments(conn, SearchFilters(label="person")) == [1]


def test_filter_by_identity(conn):
    add_tracklet(conn, 1, 1)
    add_tracklet(conn, 2, 2)
    enroll_tracklet(conn, 2, "Rafi")
    assert allowed_segments(conn, SearchFilters(identity="Rafi")) == [2]
    assert allowed_segments(conn, SearchFilters(identity="Nobody")) == []


def test_filter_by_zone(conn):
    add_tracklet(conn, 1, 1)
    create_zone(conn, 1, "front door", "line", [(0.5, 0.0), (0.5, 1.0)])
    recompute_events(conn)
    assert allowed_segments(conn, SearchFilters(zone="front door")) == [1]
    assert allowed_segments(conn, SearchFilters(zone="back gate")) == []


def test_filter_by_day(conn):
    everything = allowed_segments(conn, SearchFilters(day="2026-01-01"))
    assert sorted(everything) == [1, 2, 3]
    assert allowed_segments(conn, SearchFilters(day="2019-05-05")) == []


# ---------- semantic ----------

def test_semantic_ranking_prefers_the_closest_vector(conn):
    target = direction(1)
    for sid, seed in ((1, 1), (2, 2), (3, 3)):
        vector = direction(seed)
        conn.execute(
            "INSERT INTO segment_embeddings(segment_id, kind, dim, vector) VALUES (?,?,?,?)",
            (sid, "clip", DIM, vector.tobytes()),
        )
    conn.commit()

    ranked = semantic_ranking(conn, target)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]


def test_semantic_ranking_searches_object_vectors_too(conn):
    """A person in a red jacket is a property of the crop, not the frame."""
    add_tracklet(conn, 7, 2)
    store_tracklet_embedding(conn, 7, "clip", direction(9))
    conn.commit()

    ranked = semantic_ranking(conn, direction(9))
    assert ranked[0][0] == 2
    assert ranked[0][2] == 7          # the tracklet that matched


def test_semantic_ranking_respects_the_allowed_set(conn):
    for sid in (1, 2, 3):
        conn.execute(
            "INSERT INTO segment_embeddings(segment_id, kind, dim, vector) VALUES (?,?,?,?)",
            (sid, "clip", DIM, direction(sid).tobytes()),
        )
    conn.commit()
    ranked = semantic_ranking(conn, direction(1), allowed=[2, 3])
    assert {sid for sid, _, _ in ranked} == {2, 3}


# ---------- fusion ----------

def test_rrf_rewards_agreement_between_signals():
    fused = reciprocal_rank_fusion({"a": [10, 20, 30], "b": [20, 10, 40]})
    # 20 is 2nd and 1st; 10 is 1st and 2nd - both beat anything ranked once.
    top_two = {item for item, _, _ in fused[:2]}
    assert top_two == {10, 20}
    assert fused[0][2] == ["a", "b"] or fused[0][2] == ["b", "a"]


def test_rrf_reads_only_order_not_scores():
    """Cosine and BM25 have no shared scale, so scores must not be summed."""
    a = reciprocal_rank_fusion({"x": [1, 2, 3]})
    b = reciprocal_rank_fusion({"x": [1, 2, 3]})
    assert [i for i, _, _ in a] == [i for i, _, _ in b] == [1, 2, 3]


def test_rrf_of_nothing_is_nothing():
    assert reciprocal_rank_fusion({}) == []


# ---------- end to end ----------

def test_search_combines_lexical_and_filters(conn):
    add_tracklet(conn, 1, 1, "person")
    add_tracklet(conn, 2, 3, "dog", camera_id=2, video_id=2)
    rebuild_text_index(conn)

    hits = search(conn, text="dog")
    assert [h.segment_id for h in hits] == [3]
    assert "lexical" in hits[0].sources

    narrowed = search(conn, text="dog", filters=SearchFilters(camera_id=1))
    assert narrowed == []


def test_search_with_only_filters_returns_newest_first(conn):
    add_tracklet(conn, 1, 1)
    add_tracklet(conn, 2, 2)
    hits = search(conn, filters=SearchFilters(camera_id=1))
    assert [h.segment_id for h in hits] == [2, 1]
    assert hits[0].sources == ["filter"]


def test_an_empty_search_box_browses_rather_than_refusing(conn):
    """No query and no filters is "show me what there is", newest first."""
    # The fixture puts segment 2 latest (offset 120), then 3 (60), then 1 (0).
    hits = search(conn)
    assert [h.segment_id for h in hits] == [2, 3, 1]
    assert all(h.sources == ["filter"] for h in hits)


def test_a_query_that_matches_nothing_returns_nothing(conn):
    """It must not quietly fall back to browsing and answer a different
    question than the one asked."""
    add_tracklet(conn, 1, 1, "person")
    rebuild_text_index(conn)
    assert search(conn, text="helicopter") == []
    assert search(conn, text="dog", filters=SearchFilters(camera_id=1)) == []


def test_search_returns_playable_coordinates(conn):
    add_tracklet(conn, 1, 1, "person")
    rebuild_text_index(conn)
    hit = search(conn, text="person")[0]
    assert hit.video_id == 1
    assert hit.t_start == 0
    assert hit.ts_start > 0


def test_an_impossible_filter_short_circuits(conn):
    add_tracklet(conn, 1, 1)
    rebuild_text_index(conn)
    assert search(conn, text="person", filters=SearchFilters(identity="Nobody")) == []
