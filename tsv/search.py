"""Retrieval: turning a typed question into moments.

Three signals, deliberately kept separate because they fail in different
places:

**Semantic** (CLIP) finds things that *look* like the query. It handles
"someone in a red jacket" and clothing, posture, scene. It cannot tell you a
person's name and is vague about counts.

**Lexical** (FTS5) finds things that are *named* like the query - an object
class, a person the user enrolled, a zone they drew. It is exact where it
applies and silent where it does not.

**Structured** filters - a person, a zone, a day, a camera - are not ranked at
all. They are constraints, and applying them as constraints rather than as
ranking signal is what makes "when did Rafi go out the front door" exact
rather than merely likely.

The two ranked signals are fused with reciprocal rank fusion rather than by
adding scores. A cosine similarity and a BM25 score have no common scale, and
any attempt to normalise them has to be recalibrated whenever either model
changes; RRF only reads the ordering, so it never needs tuning.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Iterable, Sequence

import numpy as np

from tsv.identity import from_blob, normalise

# Rank-fusion constant. 60 is the value the RRF paper settled on and it is not
# sensitive: it only controls how sharply the top of each list is favoured.
RRF_K = 60


@dataclass
class SearchHit:
    segment_id: int
    score: float
    ts_start: float
    ts_end: float
    video_id: int
    camera_id: int
    t_start: float
    t_end: float
    thumb_path: str | None = None
    labels: str | None = None
    # Which signals put this here, for the UI to explain itself.
    sources: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    best_tracklet_id: int | None = None


@dataclass
class SearchFilters:
    day: str | None = None
    camera_id: int | None = None
    identity: str | None = None
    zone: str | None = None
    label: str | None = None
    event_kind: str | None = None

    @property
    def active(self) -> bool:
        return any(
            v is not None
            for v in (self.day, self.camera_id, self.identity, self.zone,
                      self.label, self.event_kind)
        )


def _day_bounds(day: str) -> tuple[float, float]:
    start = datetime.combine(date.fromisoformat(day), dtime.min)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


# ---------- the lexical index ----------

def segment_document(conn: sqlite3.Connection, segment_id: int) -> str:
    """Everything already known about a segment, as words.

    Built from the index rather than from pixels: object labels, the people
    identified in it, the zones it touched and what it did there.
    """
    parts: list[str] = []

    camera = conn.execute(
        """SELECT c.name FROM segments s JOIN cameras c ON c.id = s.camera_id
           WHERE s.id = ?""",
        (segment_id,),
    ).fetchone()
    if camera:
        parts.append(camera["name"])

    for row in conn.execute(
        """SELECT t.label, COUNT(*) AS n, i.name AS who
           FROM tracklets t LEFT JOIN identities i ON i.id = t.identity_id
           WHERE t.segment_id = ? GROUP BY t.label, i.name""",
        (segment_id,),
    ):
        parts.extend([row["label"]] * min(int(row["n"]), 3))
        if row["who"]:
            parts.append(row["who"])

    for row in conn.execute(
        """SELECT DISTINCT z.name, e.kind FROM events e
           JOIN zones z ON z.id = e.zone_id WHERE e.segment_id = ?""",
        (segment_id,),
    ):
        parts.append(row["name"])
        parts.append(row["kind"].replace("_", " "))

    return " ".join(parts)


def rebuild_text_index(conn: sqlite3.Connection) -> int:
    """Rebuild the lexical index from what the other phases have stored.

    Cheap and total, for the same reason zone events are: the inputs change
    whenever someone is named or a zone is redrawn, and a partial update is
    more ways to be wrong than it is worth.
    """
    conn.execute("DELETE FROM segment_text")
    rows = conn.execute("SELECT id FROM segments").fetchall()
    documents = [
        (segment_document(conn, int(r["id"])), int(r["id"])) for r in rows
    ]
    documents = [(body, sid) for body, sid in documents if body.strip()]
    conn.executemany(
        "INSERT INTO segment_text(body, segment_id) VALUES (?,?)", documents
    )
    conn.commit()
    return len(documents)


# ---------- individual signals ----------

def _filter_clause(filters: SearchFilters) -> tuple[str, list]:
    """SQL constraining segments, applied to every signal alike."""
    clauses: list[str] = []
    params: list = []

    if filters.day:
        start, end = _day_bounds(filters.day)
        clauses.append("s.ts_end > ? AND s.ts_start < ?")
        params += [start, end]
    if filters.camera_id:
        clauses.append("s.camera_id = ?")
        params.append(filters.camera_id)
    if filters.label:
        clauses.append(
            "EXISTS (SELECT 1 FROM tracklets t WHERE t.segment_id = s.id AND t.label = ?)"
        )
        params.append(filters.label)
    if filters.identity:
        clauses.append(
            "EXISTS (SELECT 1 FROM tracklets t JOIN identities i ON i.id = t.identity_id "
            "WHERE t.segment_id = s.id AND i.name = ?)"
        )
        params.append(filters.identity)
    if filters.zone:
        clauses.append(
            "EXISTS (SELECT 1 FROM events e JOIN zones z ON z.id = e.zone_id "
            "WHERE e.segment_id = s.id AND z.name = ?)"
        )
        params.append(filters.zone)
    if filters.event_kind:
        clauses.append(
            "EXISTS (SELECT 1 FROM events e WHERE e.segment_id = s.id AND e.kind = ?)"
        )
        params.append(filters.event_kind)

    return (" AND ".join(clauses) if clauses else "1=1"), params


def allowed_segments(conn: sqlite3.Connection, filters: SearchFilters) -> list[int]:
    where, params = _filter_clause(filters)
    rows = conn.execute(f"SELECT s.id FROM segments s WHERE {where}", params).fetchall()
    return [int(r["id"]) for r in rows]


def semantic_ranking(
    conn: sqlite3.Connection,
    query_vector: np.ndarray,
    allowed: Sequence[int] | None = None,
    limit: int = 200,
) -> list[tuple[int, float, int | None]]:
    """Segments by CLIP similarity, as (segment_id, score, tracklet_id).

    Both scene vectors and per-object vectors are searched. A person in a red
    jacket is a property of the crop, not of the whole frame, so searching only
    scene embeddings misses exactly the queries this is for; a segment takes
    the best score from either source.
    """
    query = normalise(query_vector)
    allow = set(allowed) if allowed is not None else None
    best: dict[int, tuple[float, int | None]] = {}

    for row in conn.execute(
        "SELECT segment_id, dim, vector FROM segment_embeddings WHERE kind = 'clip'"
    ):
        sid = int(row["segment_id"])
        if allow is not None and sid not in allow:
            continue
        score = float(from_blob(row["vector"], row["dim"]) @ query)
        if score > best.get(sid, (-2.0, None))[0]:
            best[sid] = (score, None)

    for row in conn.execute(
        """SELECT te.tracklet_id, te.dim, te.vector, t.segment_id
           FROM tracklet_embeddings te JOIN tracklets t ON t.id = te.tracklet_id
           WHERE te.kind = 'clip'"""
    ):
        sid = int(row["segment_id"])
        if allow is not None and sid not in allow:
            continue
        score = float(from_blob(row["vector"], row["dim"]) @ query)
        if score > best.get(sid, (-2.0, None))[0]:
            best[sid] = (score, int(row["tracklet_id"]))

    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
    return [(sid, score, tid) for sid, (score, tid) in ranked]


def _fts_query(text: str) -> str:
    """A safe FTS5 query from free text.

    Every token is quoted and OR-ed. Quoting keeps FTS5 operators the user
    happened to type (NEAR, *, -, ") from being executed or raising, and OR
    rather than AND because a typed phrase is a description, not a conjunction
    that must hold in full.
    """
    words = [w for w in "".join(c if c.isalnum() else " " for c in text).split() if w]
    return " OR ".join(f'"{w}"' for w in words)


def lexical_ranking(
    conn: sqlite3.Connection,
    text: str,
    allowed: Sequence[int] | None = None,
    limit: int = 200,
) -> list[tuple[int, float]]:
    match = _fts_query(text)
    if not match:
        return []
    rows = conn.execute(
        "SELECT segment_id, rank FROM segment_text WHERE segment_text MATCH ? "
        "ORDER BY rank LIMIT ?",
        (match, limit * 4),
    ).fetchall()

    allow = set(allowed) if allowed is not None else None
    out = []
    for row in rows:
        sid = int(row["segment_id"])
        if allow is not None and sid not in allow:
            continue
        # FTS5 rank is negative, better being more negative.
        out.append((sid, -float(row["rank"])))
        if len(out) >= limit:
            break
    return out


# ---------- fusion ----------

def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[int]], k: int = RRF_K
) -> list[tuple[int, float, list[str]]]:
    """Merge ranked id lists by rank alone.

    Reads only the ordering, never the scores, so a cosine similarity and a
    BM25 score can be combined without inventing a shared scale.
    """
    scores: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for name, ids in rankings.items():
        for position, item in enumerate(ids):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position + 1)
            sources.setdefault(item, []).append(name)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(item, score, sources[item]) for item, score in ranked]


def search(
    conn: sqlite3.Connection,
    text: str = "",
    query_vector: np.ndarray | None = None,
    filters: SearchFilters | None = None,
    limit: int = 40,
    min_similarity: float | None = None,
) -> list[SearchHit]:
    """Rank segments for a query, under whatever filters are in force.

    `min_similarity` discards weak semantic matches outright. It is off by
    default and deliberately has no baked-in value: CLIP's cosine scores are
    not calibrated in absolute terms, the useful floor differs by camera and
    by how the query is phrased, and a number guessed here would quietly
    decide what the user is allowed to find. Ranking is reported either way,
    so a caller can pick a floor by looking at real scores.
    """
    filters = filters or SearchFilters()
    allowed = allowed_segments(conn, filters) if filters.active else None
    if allowed is not None and not allowed:
        return []

    semantic = (
        semantic_ranking(conn, query_vector, allowed)
        if query_vector is not None
        else []
    )
    if min_similarity is not None:
        semantic = [row for row in semantic if row[1] >= min_similarity]
    lexical = lexical_ranking(conn, text, allowed) if text.strip() else []

    asked_for_something = bool(text.strip()) or query_vector is not None

    if not semantic and not lexical:
        if asked_for_something:
            # The user named something specific and nothing matched. Falling
            # back to browsing would answer a question they did not ask, and
            # answer it confidently.
            return []
        # No query at all: browsing under whatever filters are set. Newest
        # first is the only ordering that means anything.
        rows = conn.execute(
            f"""SELECT s.id FROM segments s WHERE {_filter_clause(filters)[0]}
                ORDER BY s.ts_start DESC LIMIT ?""",
            [*_filter_clause(filters)[1], limit],
        ).fetchall()
        order = [(int(r["id"]), 0.0, ["filter"]) for r in rows]
    else:
        rankings: dict[str, Sequence[int]] = {}
        if semantic:
            rankings["semantic"] = [sid for sid, _, _ in semantic]
        if lexical:
            rankings["lexical"] = [sid for sid, _ in lexical]
        order = reciprocal_rank_fusion(rankings)[:limit]

    semantic_by_id = {sid: (score, tid) for sid, score, tid in semantic}
    hits: list[SearchHit] = []
    for segment_id, score, sources in order[:limit]:
        row = conn.execute(
            """SELECT id, video_id, camera_id, t_start, t_end, ts_start, ts_end,
                      thumb_path, labels
               FROM segments WHERE id = ?""",
            (segment_id,),
        ).fetchone()
        if row is None:
            continue
        sem = semantic_by_id.get(segment_id)
        hits.append(
            SearchHit(
                segment_id=segment_id, score=score,
                ts_start=row["ts_start"], ts_end=row["ts_end"],
                video_id=row["video_id"], camera_id=row["camera_id"],
                t_start=row["t_start"], t_end=row["t_end"],
                thumb_path=row["thumb_path"], labels=row["labels"],
                sources=sources,
                semantic_score=sem[0] if sem else None,
                best_tracklet_id=sem[1] if sem else None,
            )
        )
    return hits
