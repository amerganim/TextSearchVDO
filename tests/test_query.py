"""Question parsing and answering.

The parser is grounded in the index's own vocabulary, so the tests set up a
small world - one person, two zones, a couple of object classes - and ask it
the questions a household actually asks.
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime, timedelta

import pytest

from tsv import db
from tsv.query import TIME_OF_DAY, answer, parse

TODAY = date(2026, 4, 10)


def _ts(day_offset: int, hour: int, minute: int = 0) -> float:
    when = datetime.combine(TODAY - timedelta(days=day_offset), dtime(hour, minute))
    return when.timestamp()


@pytest.fixture
def world(tmp_path):
    """One camera, one door, one room, one named person."""
    conn = db.open_db(tmp_path / "t.db")
    conn.execute("INSERT INTO cameras(id, name) VALUES (1, 'hallway')")
    conn.execute(
        """INSERT INTO videos(id, camera_id, path, start_ts, ts_source, duration)
           VALUES (1, 1, 'a.mp4', ?, 'test', 86400)""",
        (_ts(1, 0),),
    )
    conn.execute("INSERT INTO identities(id, name) VALUES (1, 'Rafi'), (2, 'Mira')")
    conn.execute(
        """INSERT INTO zones(id, camera_id, name, kind, points)
           VALUES (1, 1, 'front door', 'line', '[[0.5,0],[0.5,1]]'),
                  (2, 1, 'kitchen', 'region', '[[0,0],[1,0],[1,1],[0,1]]')"""
    )

    # Rafi goes out twice yesterday morning and comes back once in the evening;
    # Mira is in the kitchen. A dog also wanders through.
    rows = [
        (1, 1, "person", "cross_out", _ts(1, 8, 15), 1, 0.0),
        (2, 1, "person", "cross_out", _ts(1, 9, 30), 1, 0.0),
        (3, 1, "person", "cross_in", _ts(1, 19, 5), 1, 0.0),
        (4, 2, "person", "dwell", _ts(1, 12, 30), 2, 900.0),
        (5, 1, "dog", "cross_out", _ts(1, 7, 0), None, 0.0),
    ]
    for tid, zone_id, label, kind, ts, identity, duration in rows:
        conn.execute(
            """INSERT INTO segments(id, video_id, camera_id, t_start, t_end,
                                    ts_start, ts_end, activity_score, peak_offset)
               VALUES (?,1,1,0,10,?,?,0.1,5)""",
            (tid, ts, ts + 10),
        )
        conn.execute(
            """INSERT INTO tracklets(id, segment_id, video_id, camera_id, cls, label,
                                     t_start, t_end, ts_start, ts_end,
                                     n_detections, mean_score, max_score, identity_id)
               VALUES (?,?,1,1,0,?,0,10,?,?,5,0.9,0.95,?)""",
            (tid, tid, label, ts, ts + 10, identity),
        )
        conn.execute(
            """INSERT INTO events(tracklet_id, zone_id, segment_id, video_id, camera_id,
                                  label, kind, t, ts, duration)
               VALUES (?,?,?,1,1,?,?,0,?,?)""",
            (tid, zone_id, tid, label, kind, ts, duration),
        )
    conn.commit()
    return conn


# ---------- parsing ----------

def test_a_full_question_grounds_every_part(world):
    plan = parse(world, "when did Rafi go out the front door yesterday", today=TODAY)
    assert plan.intent == "when"
    assert plan.filters.identity == "Rafi"
    assert plan.filters.zone == "front door"
    assert plan.filters.event_kind == "cross_out"
    assert plan.filters.day == (TODAY - timedelta(days=1)).isoformat()
    assert plan.grounded


def test_direction_words_map_to_crossings(world):
    for phrase, kind in (
        ("go out", "cross_out"), ("left", "cross_out"), ("came home", "cross_in"),
        ("entered", "cross_in"), ("arrived", "cross_in"),
    ):
        plan = parse(world, f"when did Rafi {phrase} the front door", today=TODAY)
        assert plan.filters.event_kind == kind, phrase


def test_direction_against_a_region_uses_enter_and_exit(world):
    """A region has no sides to cross; it is entered and left."""
    plan = parse(world, "when did Mira go into the kitchen", today=TODAY)
    assert plan.filters.zone == "kitchen"
    assert plan.filters.event_kind in ("enter", "cross_in")


def test_longest_entity_name_wins(world):
    """A zone called 'front door' must not be shadowed by the word 'door'."""
    world.execute(
        "INSERT INTO zones(id, camera_id, name, kind, points) "
        "VALUES (3, 1, 'door', 'line', '[[0,0],[1,1]]')"
    )
    world.commit()
    plan = parse(world, "when did Rafi use the front door", today=TODAY)
    assert plan.filters.zone == "front door"


def test_relative_days(world):
    for phrase, offset in (("today", 0), ("yesterday", 1), ("last night", 1)):
        plan = parse(world, f"when did Rafi go out {phrase}", today=TODAY)
        assert plan.filters.day == (TODAY - timedelta(days=offset)).isoformat(), phrase


def test_an_explicit_date_is_honoured(world):
    plan = parse(world, "who was here on 2026-03-02", today=TODAY)
    assert plan.filters.day == "2026-03-02"


def test_a_weekday_resolves_to_the_most_recent_one(world):
    plan = parse(world, "when did Rafi go out on wednesday", today=TODAY)
    assert plan.filters.day is not None
    assert date.fromisoformat(plan.filters.day) < TODAY
    assert date.fromisoformat(plan.filters.day).weekday() == 2


def test_time_of_day_bands(world):
    plan = parse(world, "when did Rafi go out in the morning", today=TODAY)
    assert plan.time_of_day == TIME_OF_DAY["morning"]


def test_object_classes_are_recognised(world):
    plan = parse(world, "when was the dog here", today=TODAY)
    assert plan.filters.label == "dog"


def test_intents(world):
    assert parse(world, "who came in yesterday", today=TODAY).intent == "who"
    assert parse(world, "how many times did Rafi go out", today=TODAY).intent == "how_many"
    assert parse(world, "how long was Mira in the kitchen", today=TODAY).intent == "how_long"
    assert parse(world, "did Rafi go out yesterday", today=TODAY).intent == "did"


def test_unmatched_words_become_the_semantic_query(world):
    """The open-vocabulary half is kept, not thrown away."""
    plan = parse(world, "when did Rafi go out wearing a red jacket", today=TODAY)
    assert plan.filters.identity == "Rafi"
    assert "red" in plan.semantic_text and "jacket" in plan.semantic_text
    assert "rafi" not in plan.semantic_text


def test_a_question_about_nothing_known_is_not_grounded(world):
    plan = parse(world, "a helicopter landing on the roof", today=TODAY)
    assert not plan.grounded
    assert "helicopter" in plan.semantic_text


# ---------- answering ----------

def test_when_lists_the_times(world):
    plan = parse(world, "when did Rafi go out the front door yesterday", today=TODAY)
    result = answer(world, plan)
    assert result.found
    assert len(result.rows) == 2
    assert result.exact
    assert "2 times" in result.headline


def test_a_single_occurrence_reads_as_one_time(world):
    plan = parse(world, "when did Rafi come in yesterday", today=TODAY)
    result = answer(world, plan)
    assert len(result.rows) == 1
    assert "19:05" in result.headline


def test_how_many_counts(world):
    plan = parse(world, "how many times did Rafi go out yesterday", today=TODAY)
    result = answer(world, plan)
    assert result.headline.startswith("2 times")


def test_how_long_totals_the_dwell(world):
    plan = parse(world, "how long was Mira in the kitchen yesterday", today=TODAY)
    result = answer(world, plan)
    assert "15.0 minutes" in result.headline


def test_who_names_the_people(world):
    plan = parse(world, "who went out the front door yesterday", today=TODAY)
    result = answer(world, plan)
    assert "Rafi" in result.headline


def test_did_answers_yes_with_evidence(world):
    plan = parse(world, "did Rafi go out yesterday", today=TODAY)
    result = answer(world, plan)
    assert result.headline.startswith("Yes")
    assert result.rows


def test_a_question_with_no_matching_record_says_so_plainly(world):
    plan = parse(world, "when did Mira go out the front door yesterday", today=TODAY)
    result = answer(world, plan)
    assert not result.found
    assert "No record" in result.headline
    assert "Mira" in result.headline


def test_time_of_day_narrows_the_answer(world):
    morning = answer(world, parse(world, "when did Rafi go out yesterday morning", today=TODAY))
    evening = answer(world, parse(world, "when did Rafi go out yesterday evening", today=TODAY))
    assert len(morning.rows) == 2
    assert not evening.found


def test_answers_carry_coordinates_to_play_from(world):
    plan = parse(world, "when did Rafi go out yesterday", today=TODAY)
    row = answer(world, plan).rows[0]
    assert row.video_id == 1
    assert row.segment_id > 0
    assert row.ts > 0


def test_the_dog_is_not_confused_with_a_person(world):
    plan = parse(world, "when did the dog go out yesterday", today=TODAY)
    result = answer(world, plan)
    assert len(result.rows) == 1
    assert result.rows[0].label == "dog"


# ---------- unknown names ----------

def test_an_unknown_name_is_refused_not_answered_with_someone_else(world):
    """The failure this guards against: asking about a person the index has
    never heard of, and getting a different person's movements back because
    the zone and direction still matched."""
    # Tarek was never enrolled; the zone and direction still match, which is
    # exactly how the wrong answer used to get through.
    plan = parse(world, "when did Tarek go out the front door yesterday", today=TODAY)
    assert plan.unknown_names == ["Tarek"]

    result = answer(world, plan)
    assert not result.found
    assert "do not know anyone called Tarek" in result.headline
    assert "Rafi" in result.headline          # tells you who it does know


def test_a_known_name_is_not_flagged_as_unknown(world):
    plan = parse(world, "when did Rafi go out yesterday", today=TODAY)
    assert plan.unknown_names == []
    assert answer(world, plan).found


def test_the_first_word_is_not_mistaken_for_a_name(world):
    """"When did..." starts with a capital and is not a person."""
    plan = parse(world, "When did Rafi go out yesterday", today=TODAY)
    assert plan.unknown_names == []


def test_lowercase_leftovers_are_not_treated_as_names(world):
    plan = parse(world, "when did Rafi go out wearing a red jacket", today=TODAY)
    assert plan.unknown_names == []
    assert "jacket" in plan.semantic_text


def test_counting_words_do_not_leak_into_the_semantic_query(world):
    plan = parse(world, "how many times did Rafi go out yesterday", today=TODAY)
    assert plan.semantic_text == ""


def test_ask_refuses_the_unknown_name_but_still_offers_matches(world):
    from tsv.query import ask
    from tsv.search import rebuild_text_index

    rebuild_text_index(world)

    result = ask(world, "when did Tarek go out the front door yesterday", today=TODAY)
    assert result.answer is not None
    assert not result.answer.found
    assert "do not know anyone called Tarek" in result.answer.headline


def test_ask_answers_a_grounded_question_exactly(world):
    from tsv.query import ask

    result = ask(world, "when did Rafi go out the front door yesterday", today=TODAY)
    assert result.mode == "answer"
    assert len(result.answer.rows) == 2


def test_ask_falls_back_to_ranking_when_nothing_is_grounded(world):
    from tsv.query import ask
    from tsv.search import rebuild_text_index

    rebuild_text_index(world)
    # "person" would be grounded - it is a known object label. Something the
    # index has no word for is not.
    result = ask(world, "a helicopter on the roof", today=TODAY)
    assert result.mode in ("search", "empty")
    assert not result.plan.grounded
