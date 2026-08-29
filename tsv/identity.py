"""Naming people, and deciding which tracklet is whom.

The gallery is small by construction - a household is a handful of people and
maybe a few hundred confirmed vectors - so matching is brute-force cosine
similarity in numpy. An approximate index would add a dependency and a tuning
surface to save microseconds on a problem that does not have them.

Two rules keep this from confidently mislabelling people, which is the failure
mode that would make the whole feature untrustworthy:

**A margin, not just a threshold.** A tracklet is only named when its best
match beats the runner-up by a clear gap. Two similar-looking people both
scoring 0.71 against a 0.70 threshold is exactly when a system should say "I
don't know" rather than guess.

**Face and body vectors never meet.** They live in different spaces; comparing
them produces a number, and the number is meaningless. `kind` is carried
everywhere and matching is always within one kind.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

EmbeddingKind = Literal["face", "body"]

# A face vector is far more discriminative than an appearance one, so the bar
# for believing an appearance match is higher. Both are guesses until they
# have been run against real footage with real people in it.
# Nothing currently writes a "body" vector, so appearance matching finds an
# empty gallery and does nothing. That is deliberate, and the tempting fix is
# wrong: the analysis pass does store a CLIP embedding of every person crop,
# but CLIP is a *semantic* space, not a re-identification one. Measured on
# real footage, two crops of the same person score a median 0.818 against each
# other - and a person against a bird scores 0.805, a bird beating the
# person-to-person floor of 0.683. There is no threshold that separates them.
# Wiring "body" to those vectors makes every person in a recording match the
# first one named, which is not identification, it is a confident wrong answer
# to every later question about that person. A real body-ReID model would go
# here; until then this stays empty on purpose.
STORED_KIND: dict[str, str] = {"face": "face", "body": "body"}

DEFAULT_THRESHOLDS: dict[str, float] = {"face": 0.42, "body": 0.62}
DEFAULT_MARGINS: dict[str, float] = {"face": 0.06, "body": 0.10}


@dataclass
class Identity:
    id: int
    name: str
    notes: str | None = None


@dataclass
class Match:
    identity_id: int | None
    name: str | None
    score: float
    runner_up: float
    kind: str
    # Why no identity was assigned, when identity_id is None.
    reason: str = ""

    @property
    def margin(self) -> float:
        return self.score - self.runner_up


@dataclass
class AssignSummary:
    n_considered: int = 0
    n_assigned: int = 0
    n_ambiguous: int = 0
    n_below_threshold: int = 0
    by_name: dict[str, int] = field(default_factory=dict)


# ---------- vectors ----------

def normalise(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, so a dot product is a cosine similarity."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        # A zero vector has no direction; leave it alone rather than divide by
        # nearly nothing and manufacture a unit vector pointing anywhere.
        return vector
    return vector / norm


def aggregate(vectors: Sequence[np.ndarray], weights: Sequence[float] | None = None) -> np.ndarray:
    """Combine a tracklet's per-detection vectors into one.

    Normalise first, then average, then normalise again: averaging raw vectors
    lets whichever detection happened to have the largest magnitude dominate,
    which is usually just the one closest to the camera.
    """
    if not len(vectors):
        raise ValueError("no vectors to aggregate")
    stacked = np.stack([normalise(v) for v in vectors])
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        if len(w) != len(stacked):
            raise ValueError("weights and vectors must be the same length")
        stacked = stacked * np.clip(w, 0.0, None)
    return normalise(stacked.mean(axis=0))


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).reshape(-1).tobytes()


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


# ---------- gallery ----------

def create_identity(conn: sqlite3.Connection, name: str, notes: str | None = None) -> Identity:
    name = name.strip()
    if not name:
        raise ValueError("an identity needs a name")
    cur = conn.execute(
        "INSERT INTO identities(name, notes, created_at) VALUES (?,?,?)",
        (name, notes, time.time()),
    )
    conn.commit()
    return Identity(int(cur.lastrowid), name, notes)


def get_or_create_identity(conn: sqlite3.Connection, name: str) -> Identity:
    row = conn.execute("SELECT * FROM identities WHERE name = ?", (name.strip(),)).fetchone()
    if row:
        return Identity(int(row["id"]), row["name"], row["notes"])
    return create_identity(conn, name)


def list_identities(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """SELECT i.id, i.name, i.notes,
                  (SELECT COUNT(*) FROM identity_embeddings e WHERE e.identity_id = i.id)
                      AS n_examples,
                  (SELECT COUNT(*) FROM tracklets t WHERE t.identity_id = i.id)
                      AS n_sightings
           FROM identities i ORDER BY i.name"""
    ).fetchall()
    return [dict(r) for r in rows]


def delete_identity(conn: sqlite3.Connection, identity_id: int) -> bool:
    conn.execute(
        "UPDATE tracklets SET identity_id = NULL, identity_score = NULL, "
        "identity_source = NULL WHERE identity_id = ?",
        (identity_id,),
    )
    cur = conn.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
    conn.commit()
    return cur.rowcount > 0


def unname_tracklet(conn: sqlite3.Connection, tracklet_id: int) -> bool:
    """Take a name off one sighting, and out of the gallery with it.

    Needed because matching now guesses from appearance where no face was
    clear, and a guess nobody can overrule is worse than no guess. Removing
    the gallery example matters as much as clearing the label: leaving it
    would keep teaching the mistake to every later match.
    """
    conn.execute(
        "DELETE FROM identity_embeddings WHERE tracklet_id = ?", (tracklet_id,)
    )
    cur = conn.execute(
        "UPDATE tracklets SET identity_id = NULL, identity_score = NULL, "
        "identity_source = NULL WHERE id = ? AND identity_id IS NOT NULL",
        (tracklet_id,),
    )
    # A name with nothing left under it is not a person the index knows, it is
    # a stray row that goes on being offered as "known so far" and can never
    # match anything. Rejecting the only sighting of someone undoes the whole
    # enrolment, not half of it.
    conn.execute(
        """DELETE FROM identities WHERE id NOT IN (
               SELECT identity_id FROM identity_embeddings
               UNION SELECT identity_id FROM tracklets WHERE identity_id IS NOT NULL
           )"""
    )
    conn.commit()
    return cur.rowcount > 0


def store_tracklet_embedding(
    conn: sqlite3.Connection,
    tracklet_id: int,
    kind: EmbeddingKind,
    vector: np.ndarray,
    n_samples: int = 1,
    model: str | None = None,
) -> None:
    """Store one vector, recording which weights produced it.

    `model` is not decoration. A vector is only comparable to another from the
    same model, and the numbers do not say which one that was, so an index
    whose model changed has to be able to tell its old vectors from its new
    ones rather than averaging nonsense across both.
    """
    vector = normalise(vector)
    conn.execute(
        """INSERT INTO tracklet_embeddings(tracklet_id, kind, dim, vector, n_samples, model)
           VALUES (?,?,?,?,?,?)
           ON CONFLICT(tracklet_id, kind) DO UPDATE SET
               dim = excluded.dim, vector = excluded.vector,
               n_samples = excluded.n_samples, model = excluded.model""",
        (tracklet_id, kind, len(vector), to_blob(vector), n_samples, model),
    )


def enroll_tracklet(
    conn: sqlite3.Connection,
    tracklet_id: int,
    name: str,
    kinds: Iterable[EmbeddingKind] = ("face", "body"),
) -> tuple[Identity, list[str]]:
    """Name a tracklet, and add its embeddings to that identity's gallery.

    This is the only way a vector enters the gallery: a person confirmed it.
    Assignments made by matching never feed themselves back in, which would
    let one mistake drift the gallery away from the actual person.

    Returns which kinds were learned, not just how many. The caller has to
    tell the user something different in each case: a face carries across
    days, an appearance vector only holds while the clothes do.
    """
    identity = get_or_create_identity(conn, name)
    added: list[str] = []
    for kind in kinds:
        stored = STORED_KIND[kind]
        row = conn.execute(
            "SELECT dim, vector, model FROM tracklet_embeddings "
            "WHERE tracklet_id = ? AND kind = ?",
            (tracklet_id, stored),
        ).fetchone()
        if row is None:
            continue
        conn.execute(
            """INSERT INTO identity_embeddings(identity_id, tracklet_id, kind, dim, vector,
                                               created_at, model)
               VALUES (?,?,?,?,?,?,?)""",
            (identity.id, tracklet_id, stored, row["dim"], row["vector"],
             time.time(), row["model"]),
        )
        added.append(kind)

    conn.execute(
        "UPDATE tracklets SET identity_id = ?, identity_score = 1.0, identity_source = 'manual' "
        "WHERE id = ?",
        (identity.id, tracklet_id),
    )
    conn.commit()
    return identity, added


def load_gallery(
    conn: sqlite3.Connection,
    kind: EmbeddingKind,
    model: str | None = None,
) -> tuple[np.ndarray, list[int], dict[int, str]]:
    """Every confirmed vector of one kind, as a matrix.

    Restricted to one model when given. Faces enrolled against different
    weights are not a bigger gallery, they are two galleries in a trench coat,
    and matching across them is meaningless in a way no error would reveal.
    """
    rows = conn.execute(
        """SELECT e.identity_id, e.dim, e.vector, i.name
           FROM identity_embeddings e JOIN identities i ON i.id = e.identity_id
           WHERE e.kind = ? AND (? IS NULL OR e.model = ?)
           ORDER BY e.identity_id""",
        (STORED_KIND[kind], model, model),
    ).fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32), [], {}

    vectors = np.stack([normalise(from_blob(r["vector"], r["dim"])) for r in rows])
    owners = [int(r["identity_id"]) for r in rows]
    names = {int(r["identity_id"]): r["name"] for r in rows}
    return vectors, owners, names


# ---------- matching ----------

def match_vector(
    vector: np.ndarray,
    gallery: np.ndarray,
    owners: Sequence[int],
    names: dict[int, str],
    kind: EmbeddingKind,
    threshold: float | None = None,
    margin: float | None = None,
) -> Match:
    """Best identity for one vector, or none with a reason."""
    threshold = DEFAULT_THRESHOLDS[kind] if threshold is None else threshold
    margin = DEFAULT_MARGINS[kind] if margin is None else margin

    if gallery.size == 0:
        return Match(None, None, 0.0, 0.0, kind, "gallery is empty")

    similarities = gallery @ normalise(vector)

    # Best score per identity, not per vector: an identity with twenty
    # enrolled examples must not out-rank one with two just by having more
    # chances to score well.
    best_per_identity: dict[int, float] = {}
    for owner, score in zip(owners, similarities.tolist()):
        if score > best_per_identity.get(owner, -1.0):
            best_per_identity[owner] = score

    ranked = sorted(best_per_identity.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score < threshold:
        return Match(None, None, top_score, runner_up, kind, "below threshold")
    if top_score - runner_up < margin:
        return Match(None, None, top_score, runner_up, kind, "too close to call")
    return Match(top_id, names.get(top_id), top_score, runner_up, kind)


def assign_identities(
    conn: sqlite3.Connection,
    kind: EmbeddingKind = "face",
    threshold: float | None = None,
    margin: float | None = None,
    reassign: bool = False,
    model: str | None = None,
) -> AssignSummary:
    """Name every tracklet that has an embedding and a confident match.

    Manual enrolments are never overwritten: a person's own labelling outranks
    anything matching decides, and `reassign` only revisits automatic ones.

    Both sides are held to one `model` when given. Comparing a vector from one
    set of weights against a gallery built from another is not a weaker match,
    it is an arbitrary number, and the thresholds below would treat it as
    evidence.
    """
    gallery, owners, names = load_gallery(conn, kind, model=model)
    summary = AssignSummary()
    if gallery.size == 0:
        return summary

    where = "WHERE te.kind = ? AND (? IS NULL OR te.model = ?) AND (t.identity_id IS NULL"
    where += " OR t.identity_source = 'auto')" if reassign else ")"
    # Appearance vectors describe whatever was cropped, so without this a car
    # can score above the threshold against a person's gallery and be given
    # their name. Faces need no such guard - they are only ever computed on
    # people in the first place.
    if kind == "body":
        where += " AND t.label = 'person'"
    rows = conn.execute(
        f"""SELECT t.id, te.dim, te.vector
            FROM tracklets t JOIN tracklet_embeddings te ON te.tracklet_id = t.id
            {where}""",
        (STORED_KIND[kind], model, model),
    ).fetchall()

    updates = []
    for row in rows:
        summary.n_considered += 1
        match = match_vector(
            from_blob(row["vector"], row["dim"]), gallery, owners, names,
            kind, threshold, margin,
        )
        if match.identity_id is None:
            if match.reason == "too close to call":
                summary.n_ambiguous += 1
            else:
                summary.n_below_threshold += 1
            continue
        updates.append((match.identity_id, match.score, int(row["id"])))
        summary.n_assigned += 1
        if match.name:
            summary.by_name[match.name] = summary.by_name.get(match.name, 0) + 1

    conn.executemany(
        "UPDATE tracklets SET identity_id = ?, identity_score = ?, identity_source = 'auto' "
        "WHERE id = ?",
        updates,
    )
    conn.commit()
    return summary
