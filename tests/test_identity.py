"""Identity matching, driven by synthetic embeddings with known structure.

Real embedders come later; what is tested here is the part that decides who
someone is, which is where a mistake becomes a confidently wrong answer rather
than a missing one.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsv import db
from tsv.identity import (
    aggregate, assign_identities, create_identity, delete_identity, enroll_tracklet,
    unname_tracklet,
    from_blob, get_or_create_identity, list_identities, load_gallery, match_vector,
    normalise, store_tracklet_embedding, to_blob,
)

DIM = 32
RNG = np.random.default_rng(11)


def person_vector(seed: int, jitter: float = 0.0) -> np.ndarray:
    """A stable direction per person, optionally perturbed."""
    rng = np.random.default_rng(1000 + seed)
    base = rng.normal(size=DIM).astype(np.float32)
    if jitter:
        base = base + RNG.normal(scale=jitter, size=DIM).astype(np.float32)
    return normalise(base)


@pytest.fixture
def conn(tmp_path):
    connection = db.open_db(tmp_path / "t.db")
    connection.execute("INSERT INTO cameras(id, name) VALUES (1, 'ch01')")
    connection.execute(
        """INSERT INTO videos(id, camera_id, path, start_ts, ts_source, duration)
           VALUES (1, 1, 'x.mp4', 1000.0, 'test', 60.0)"""
    )
    connection.execute(
        """INSERT INTO segments(id, video_id, camera_id, t_start, t_end,
                                ts_start, ts_end, activity_score, peak_offset)
           VALUES (1, 1, 1, 0, 60, 1000, 1060, 0.1, 5)"""
    )
    connection.commit()
    return connection


def add_tracklet(conn, tracklet_id: int, vector: np.ndarray, kind: str = "face") -> int:
    conn.execute(
        """INSERT INTO tracklets(id, segment_id, video_id, camera_id, cls, label,
                                 t_start, t_end, ts_start, ts_end,
                                 n_detections, mean_score, max_score)
           VALUES (?,1,1,1,0,'person',0,5,1000,1005,5,0.9,0.95)""",
        (tracklet_id,),
    )
    store_tracklet_embedding(conn, tracklet_id, kind, vector, n_samples=5)
    conn.commit()
    return tracklet_id


# ---------- vector maths ----------

def test_normalise_gives_unit_length():
    v = normalise(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.allclose(v, [0.6, 0.8])


def test_normalise_leaves_a_zero_vector_alone():
    """Dividing by nearly nothing would invent a direction."""
    v = normalise(np.zeros(8))
    assert np.allclose(v, 0.0)
    assert np.all(np.isfinite(v))


def test_aggregate_normalises_before_averaging():
    """Otherwise the longest vector - usually just the closest detection -
    dominates the mean."""
    small = np.array([1.0, 0.0])
    huge = np.array([0.0, 50.0])
    combined = aggregate([small, huge])
    # Equal contribution means 45 degrees, not almost straight up.
    assert np.allclose(combined, [0.7071, 0.7071], atol=1e-3)


def test_aggregate_honours_weights():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    weighted = aggregate([a, b], weights=[0.9, 0.1])
    assert weighted[0] > weighted[1]


def test_aggregate_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        aggregate([np.array([1.0, 0.0])], weights=[0.5, 0.5])
    with pytest.raises(ValueError):
        aggregate([])


def test_blob_round_trip():
    v = normalise(RNG.normal(size=DIM))
    assert np.allclose(from_blob(to_blob(v), DIM), v, atol=1e-6)


# ---------- gallery ----------

def test_identities_are_created_and_reused(conn):
    first = create_identity(conn, "Rafi")
    again = get_or_create_identity(conn, "Rafi")
    assert first.id == again.id
    assert len(list_identities(conn)) == 1


def test_identity_needs_a_name(conn):
    with pytest.raises(ValueError):
        create_identity(conn, "   ")


def test_enrolling_a_tracklet_fills_the_gallery(conn):
    add_tracklet(conn, 1, person_vector(1))
    identity, added = enroll_tracklet(conn, 1, "Rafi")

    assert added == ["face"]
    listed = list_identities(conn)[0]
    assert listed["name"] == "Rafi"
    assert listed["n_examples"] == 1
    assert listed["n_sightings"] == 1

    row = conn.execute("SELECT identity_source, identity_score FROM tracklets WHERE id=1").fetchone()
    assert row["identity_source"] == "manual"
    assert row["identity_score"] == 1.0


def test_gallery_loads_as_a_matrix(conn):
    add_tracklet(conn, 1, person_vector(1))
    add_tracklet(conn, 2, person_vector(2))
    enroll_tracklet(conn, 1, "Rafi")
    enroll_tracklet(conn, 2, "Mira")

    vectors, owners, names = load_gallery(conn, "face")
    assert vectors.shape == (2, DIM)
    assert len(set(owners)) == 2
    assert set(names.values()) == {"Rafi", "Mira"}
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_gallery_of_another_kind_is_empty(conn):
    """Face and body vectors live in different spaces and never mix."""
    add_tracklet(conn, 1, person_vector(1), kind="face")
    enroll_tracklet(conn, 1, "Rafi")
    assert load_gallery(conn, "body")[0].size == 0


def test_a_clip_embedding_is_never_used_as_an_identity(conn):
    """CLIP describes what something is, not which one it is.

    The analysis pass stores a CLIP vector for every person crop, and reaching
    for it as the missing "body" vector is the obvious shortcut. Measured on
    real footage it does not work at all: same-person pairs score a median
    0.818, a person against a *bird* scores 0.805, and the person-to-person
    floor is 0.683 - the distributions sit on top of each other. Used as an
    identity it names every person in a recording after the first one named.
    """
    add_tracklet(conn, 1, person_vector(1), kind="clip")
    identity, added = enroll_tracklet(conn, 1, "Rafi")

    assert added == [], "a CLIP vector became an identity example"
    assert load_gallery(conn, "body")[0].size == 0

    add_tracklet(conn, 2, person_vector(1) + 0.01 * RNG.normal(size=DIM), kind="clip")
    assert assign_identities(conn, "body").n_assigned == 0


def test_appearance_matching_would_never_name_a_car(conn):
    """The guard that has to hold if a body model is ever added.

    An appearance vector describes whatever was cropped, and nothing in the
    maths stops a vehicle scoring above the threshold against a person's
    gallery. Being told a car is your son is worse than being told nothing.
    """
    add_tracklet(conn, 1, person_vector(1), kind="body")
    enroll_tracklet(conn, 1, "Rafi")

    add_tracklet(conn, 2, person_vector(1) + 0.01 * RNG.normal(size=DIM), kind="body")
    conn.execute("UPDATE tracklets SET label = 'car' WHERE id = 2")
    conn.commit()

    assert assign_identities(conn, "body").n_assigned == 0
    conn.execute("UPDATE tracklets SET label = 'person' WHERE id = 2")
    conn.commit()
    assert assign_identities(conn, "body").n_assigned == 1


def test_a_wrong_name_can_be_taken_off_a_sighting(conn):
    """And must leave the gallery with it.

    A rejected example that stays in the gallery keeps teaching the mistake to
    every later match.
    """
    add_tracklet(conn, 1, person_vector(1))
    enroll_tracklet(conn, 1, "Rafi")

    assert unname_tracklet(conn, 1) is True
    row = conn.execute(
        "SELECT identity_id, identity_source FROM tracklets WHERE id = 1"
    ).fetchone()
    assert row["identity_id"] is None and row["identity_source"] is None
    assert load_gallery(conn, "face")[0].size == 0
    # Rafi had one sighting and it has just been rejected, so nothing is left
    # of him. A name with no examples and no sightings can never match, but
    # would go on being listed as somebody the index knows.
    assert list_identities(conn) == []
    assert unname_tracklet(conn, 1) is False


def test_rejecting_one_sighting_leaves_the_others_named(conn):
    """The name survives as long as anything still carries it."""
    add_tracklet(conn, 1, person_vector(1))
    add_tracklet(conn, 2, person_vector(1))
    enroll_tracklet(conn, 1, "Rafi")
    enroll_tracklet(conn, 2, "Rafi")

    assert unname_tracklet(conn, 2) is True
    listed = list_identities(conn)
    assert len(listed) == 1 and listed[0]["n_sightings"] == 1


def test_a_gallery_never_mixes_two_sets_of_face_weights(conn):
    """Enrolling under one model and matching under another is not a weak
    match, it is an arbitrary number that the thresholds treat as evidence.

    This becomes reachable the moment somebody with a better machine picks a
    larger face model: the faces they already named were embedded by the old
    one.
    """
    add_tracklet(conn, 1, person_vector(1))
    conn.execute("UPDATE tracklet_embeddings SET model = 'buffalo_s' WHERE tracklet_id = 1")
    enroll_tracklet(conn, 1, "Rafi")

    assert load_gallery(conn, "face", model="buffalo_s")[0].shape == (1, DIM)
    assert load_gallery(conn, "face", model="buffalo_l")[0].size == 0

    add_tracklet(conn, 2, person_vector(1) + 0.01 * RNG.normal(size=DIM))
    conn.execute("UPDATE tracklet_embeddings SET model = 'buffalo_l' WHERE tracklet_id = 2")
    conn.commit()

    # The near-identical vector is only a match under its own weights.
    assert assign_identities(conn, "face", model="buffalo_l").n_assigned == 0
    assert assign_identities(conn, "face", model="buffalo_s").n_assigned == 0


# ---------- matching ----------

def _gallery(conn):
    return load_gallery(conn, "face")


def test_a_close_vector_matches_its_person(conn):
    add_tracklet(conn, 1, person_vector(1))
    add_tracklet(conn, 2, person_vector(2))
    enroll_tracklet(conn, 1, "Rafi")
    enroll_tracklet(conn, 2, "Mira")

    match = match_vector(person_vector(1, jitter=0.15), *_gallery(conn), "face")
    assert match.name == "Rafi"
    assert match.score > match.runner_up


def test_a_stranger_is_not_named(conn):
    add_tracklet(conn, 1, person_vector(1))
    enroll_tracklet(conn, 1, "Rafi")

    match = match_vector(person_vector(99), *_gallery(conn), "face", threshold=0.9)
    assert match.identity_id is None
    assert match.reason == "below threshold"


def test_an_ambiguous_match_is_refused_not_guessed(conn):
    """Two people scoring almost the same is when to say "I don't know"."""
    base = person_vector(1)
    nearly = normalise(base + 0.02 * RNG.normal(size=DIM).astype(np.float32))
    add_tracklet(conn, 1, base)
    add_tracklet(conn, 2, nearly)
    enroll_tracklet(conn, 1, "Rafi")
    enroll_tracklet(conn, 2, "Twin")

    match = match_vector(base, *_gallery(conn), "face", threshold=0.1, margin=0.5)
    assert match.identity_id is None
    assert match.reason == "too close to call"
    assert match.margin < 0.5


def test_an_empty_gallery_says_so(conn):
    match = match_vector(person_vector(1), *_gallery(conn), "face")
    assert match.identity_id is None
    assert match.reason == "gallery is empty"


def test_more_examples_do_not_win_by_weight_of_numbers(conn):
    """An identity with many enrolments must not out-rank a better match."""
    for i in range(6):
        add_tracklet(conn, 10 + i, person_vector(2, jitter=0.3))
        enroll_tracklet(conn, 10 + i, "Loud")
    add_tracklet(conn, 1, person_vector(1))
    enroll_tracklet(conn, 1, "Quiet")

    match = match_vector(person_vector(1, jitter=0.05), *_gallery(conn), "face",
                         threshold=0.1, margin=0.0)
    assert match.name == "Quiet"


# ---------- assignment ----------

def test_assignment_names_matching_tracklets(conn):
    add_tracklet(conn, 1, person_vector(1))
    enroll_tracklet(conn, 1, "Rafi")
    add_tracklet(conn, 2, person_vector(1, jitter=0.1))

    summary = assign_identities(conn, "face", threshold=0.3, margin=0.0)
    assert summary.n_assigned == 1
    assert summary.by_name == {"Rafi": 1}

    row = conn.execute("SELECT identity_id, identity_source FROM tracklets WHERE id=2").fetchone()
    assert row["identity_id"] is not None
    assert row["identity_source"] == "auto"


def test_assignment_never_overwrites_a_manual_label(conn):
    add_tracklet(conn, 1, person_vector(1))
    add_tracklet(conn, 2, person_vector(2))
    enroll_tracklet(conn, 1, "Rafi")
    enroll_tracklet(conn, 2, "Mira")

    # Deliberately mislabel 2 by hand, then let matching run.
    enroll_tracklet(conn, 2, "Rafi")
    assign_identities(conn, "face", threshold=0.0, margin=0.0, reassign=True)

    row = conn.execute(
        """SELECT i.name, t.identity_source FROM tracklets t
           JOIN identities i ON i.id = t.identity_id WHERE t.id = 2"""
    ).fetchone()
    assert row["name"] == "Rafi"
    assert row["identity_source"] == "manual"


def test_ambiguous_and_weak_matches_are_counted_separately(conn):
    add_tracklet(conn, 1, person_vector(1))
    enroll_tracklet(conn, 1, "Rafi")
    add_tracklet(conn, 2, person_vector(50))

    summary = assign_identities(conn, "face", threshold=0.99, margin=0.0)
    assert summary.n_assigned == 0
    assert summary.n_below_threshold == 1
    assert summary.n_ambiguous == 0


def test_assignment_with_an_empty_gallery_does_nothing(conn):
    add_tracklet(conn, 1, person_vector(1))
    summary = assign_identities(conn, "face")
    assert summary.n_considered == 0
    assert summary.n_assigned == 0


def test_deleting_an_identity_unassigns_its_sightings(conn):
    add_tracklet(conn, 1, person_vector(1))
    identity, _ = enroll_tracklet(conn, 1, "Rafi")
    assert delete_identity(conn, identity.id)

    row = conn.execute("SELECT identity_id, identity_score FROM tracklets WHERE id=1").fetchone()
    assert row["identity_id"] is None
    assert row["identity_score"] is None
    assert conn.execute("SELECT COUNT(*) c FROM identity_embeddings").fetchone()["c"] == 0


def test_storing_an_embedding_twice_replaces_it(conn):
    add_tracklet(conn, 1, person_vector(1))
    store_tracklet_embedding(conn, 1, "face", person_vector(2), n_samples=9)
    conn.commit()

    rows = conn.execute("SELECT n_samples FROM tracklet_embeddings WHERE tracklet_id=1").fetchall()
    assert len(rows) == 1
    assert rows[0]["n_samples"] == 9
