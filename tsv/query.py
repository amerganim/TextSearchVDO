"""Turning a typed question into a query plan, and a plan into an answer.

Search ranks moments; this answers questions. "When did Rafi go out the front
door yesterday" should return two timestamps, not forty segments to scroll.

The parsing here is **entity grounded, not general language understanding**,
and that is what makes it work without a model. The vocabulary is closed and
already in the database: the people the user enrolled, the zones they drew,
the object classes the detector knows, the cameras they have. Matching against
that is reliable in a way that parsing arbitrary English is not. Whatever is
left over after the known entities are lifted out is not discarded - it
becomes the semantic query, where CLIP handles the open-vocabulary half
("in a red jacket", "carrying a box").

The split matters: everything the index knows *exactly* is answered exactly,
and only the genuinely fuzzy remainder is left to a similarity score.
"""

from __future__ import annotations

import calendar
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from typing import Literal

from tsv.search import SearchFilters

Intent = Literal["when", "who", "how_long", "how_many", "did", "what", "find"]

# Phrases that imply a direction through a zone. Ordered longest-first at use
# so "went out" is not shadowed by "went".
EXIT_PHRASES = (
    "go out", "goes out", "went out", "going out",
    "go outside", "went outside", "going outside",
    "walk out", "walked out", "step out", "stepped out",
    "head out", "headed out", "get out", "got out",
    "leave", "leaves", "left", "leaving",
    "exit", "exits", "exited", "depart", "departed", "out of",
)
ENTER_PHRASES = (
    "come in", "comes in", "came in", "coming in",
    "come inside", "came inside", "go inside", "went inside",
    "go into", "goes into", "went into", "going into",
    "go in", "goes in", "went in", "going in",
    "walk in", "walked in", "step in", "stepped in",
    "get in", "got in", "get home", "got home",
    "come home", "came home", "arrive", "arrives", "arrived",
    "enter", "enters", "entered", "return", "returned",
)
DWELL_PHRASES = ("stay", "stayed", "staying", "was in", "were in", "spend", "spent", "linger")

# Rough local clock bands. Deliberately coarse: someone asking about "the
# evening" does not have a precise boundary in mind either.
TIME_OF_DAY = {
    "morning": (6, 12),
    "afternoon": (12, 18),
    "evening": (18, 22),
    "night": (22, 6),
    "midnight": (23, 1),
    "lunchtime": (11, 14),
    "overnight": (22, 6),
}

_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}


@dataclass
class QueryPlan:
    question: str
    intent: Intent = "find"
    filters: SearchFilters = field(default_factory=SearchFilters)
    semantic_text: str = ""
    # (kind, value) for each entity lifted out, so the UI can show its working.
    matched: list[tuple[str, str]] = field(default_factory=list)
    time_of_day: tuple[int, int] | None = None
    # Capitalised words that look like names but match nobody enrolled. These
    # are not a caveat, they are a refusal: a question about someone the index
    # has never heard of must not be answered with somebody else's movements.
    unknown_names: list[str] = field(default_factory=list)
    # Set when a phrase named a day that holds no footage.
    note: str = ""

    @property
    def grounded(self) -> bool:
        return bool(self.matched) or self.filters.active


def _known(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Every name the index could possibly be asked about."""
    def column(sql: str) -> list[str]:
        return [str(r[0]) for r in conn.execute(sql) if r[0]]

    return {
        "identity": column("SELECT name FROM identities"),
        "zone": column("SELECT DISTINCT name FROM zones"),
        "label": column("SELECT DISTINCT label FROM tracklets"),
        "camera": column("SELECT name FROM cameras"),
    }


def _lift(text: str, phrase: str) -> tuple[str, bool]:
    """Remove `phrase` from `text` if present, on word boundaries."""
    pattern = re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", re.IGNORECASE)
    if not pattern.search(text):
        return text, False
    return pattern.sub(" ", text, count=1), True


def _parse_day(text: str, today: date) -> tuple[str, str | None, str]:
    """Pull a date out of the text. Returns (remaining, iso_day, matched)."""
    explicit = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if explicit:
        return text.replace(explicit.group(1), " "), explicit.group(1), explicit.group(1)

    for phrase, delta in (("today", 0), ("yesterday", 1), ("last night", 1),
                          ("tonight", 0), ("this morning", 0), ("this afternoon", 0),
                          ("this evening", 0)):
        remaining, hit = _lift(text, phrase)
        if hit:
            return remaining, (today - timedelta(days=delta)).isoformat(), phrase

    # "on monday" / "last tuesday" - the most recent one that has passed.
    for name, index in _WEEKDAYS.items():
        remaining, hit = _lift(text, name)
        if hit:
            back = (today.weekday() - index) % 7 or 7
            return remaining, (today - timedelta(days=back)).isoformat(), name

    return text, None, ""


def _parse_intent(text: str) -> Intent:
    lowered = text.strip().lower()
    if lowered.startswith("how long") or "how long" in lowered:
        return "how_long"
    if lowered.startswith("how many") or "how many" in lowered:
        return "how_many"
    if lowered.startswith("when"):
        return "when"
    if lowered.startswith("who"):
        return "who"
    if lowered.startswith(("did ", "was ", "were ", "has ", "have ")):
        return "did"
    if lowered.startswith("what"):
        return "what"
    return "find"


def parse(conn: sqlite3.Connection, question: str, today: date | None = None) -> QueryPlan:
    """Ground a question against what this index actually contains."""
    today = today or date.today()
    plan = QueryPlan(question=question, intent=_parse_intent(question))
    remaining = question

    remaining, day, day_phrase = _parse_day(remaining, today)
    if day:
        plan.filters.day = day
        plan.matched.append(("day", day_phrase))

    for word, band in TIME_OF_DAY.items():
        remaining, hit = _lift(remaining, word)
        if hit:
            plan.time_of_day = band
            plan.matched.append(("time of day", word))
            break

    # Longest names first, so "front door" is not shadowed by a zone "door".
    known = _known(conn)
    for kind in ("identity", "zone", "camera", "label"):
        for name in sorted(known[kind], key=len, reverse=True):
            remaining, hit = _lift(remaining, name)
            if not hit:
                continue
            if kind == "identity":
                plan.filters.identity = name
            elif kind == "zone":
                plan.filters.zone = name
            elif kind == "label":
                plan.filters.label = name
            else:
                row = conn.execute(
                    "SELECT id FROM cameras WHERE name = ?", (name,)
                ).fetchone()
                if row:
                    plan.filters.camera_id = int(row["id"])
            plan.matched.append((kind, name))
            break

    # Direction only means something against a zone.
    for phrases, line_kind, region_kind in (
        (EXIT_PHRASES, "cross_out", "exit"),
        (ENTER_PHRASES, "cross_in", "enter"),
        (DWELL_PHRASES, "dwell", "dwell"),
    ):
        matched_phrase = None
        for phrase in sorted(phrases, key=len, reverse=True):
            remaining, hit = _lift(remaining, phrase)
            if hit:
                matched_phrase = phrase
                break
        if matched_phrase:
            kind = line_kind
            if plan.filters.zone:
                row = conn.execute(
                    "SELECT kind FROM zones WHERE name = ? LIMIT 1", (plan.filters.zone,)
                ).fetchone()
                if row and row["kind"] == "region":
                    kind = region_kind
            plan.filters.event_kind = kind
            plan.matched.append(("event", matched_phrase))
            break

    # Whatever survives is the open-vocabulary part.
    stop = {
        "when", "did", "does", "do", "who", "what", "how", "long", "many",
        "the", "a", "an", "was", "were", "is", "are", "at", "in", "on", "to",
        "of", "my", "our", "his", "her", "their", "there", "and", "for",
        "time", "times", "ever", "last", "any", "some", "been", "go", "went",
    }
    words = [w for w in re.findall(r"[\w']+", remaining.lower()) if w not in stop]
    plan.semantic_text = " ".join(words)

    # A capitalised leftover, in a database that knows some names, is almost
    # always a person this index has never been introduced to.
    if known["identity"]:
        tokens = re.findall(r"[A-Za-z']+", question)
        lowered_known = {n.lower() for n in known["identity"]}
        survivors = set(words)
        plan.unknown_names = [
            token for i, token in enumerate(tokens)
            if i > 0                                   # not the sentence's first word
            and token[:1].isupper()
            and token.lower() in survivors
            and token.lower() not in lowered_known
        ]
    return plan


# ---------- answering ----------

@dataclass
class AnswerRow:
    ts: float
    label: str
    who: str | None
    zone: str | None
    kind: str | None
    duration: float
    video_id: int
    t: float
    segment_id: int


@dataclass
class Answer:
    plan: QueryPlan
    headline: str
    rows: list[AnswerRow] = field(default_factory=list)
    # True when the answer rests on exact index facts rather than similarity.
    exact: bool = True

    @property
    def found(self) -> bool:
        return bool(self.rows)


def _within_band(ts: float, band: tuple[int, int] | None) -> bool:
    if band is None:
        return True
    hour = datetime.fromtimestamp(ts).hour
    start, end = band
    return start <= hour < end if start < end else (hour >= start or hour < end)


def _event_rows(conn: sqlite3.Connection, plan: QueryPlan, limit: int) -> list[AnswerRow]:
    clauses, params = [], []
    if plan.filters.identity:
        clauses.append("AND i.name = ?")
        params.append(plan.filters.identity)
    if plan.filters.zone:
        clauses.append("AND z.name = ?")
        params.append(plan.filters.zone)
    if plan.filters.event_kind:
        clauses.append("AND e.kind = ?")
        params.append(plan.filters.event_kind)
    if plan.filters.label:
        clauses.append("AND e.label = ?")
        params.append(plan.filters.label)
    if plan.filters.camera_id:
        clauses.append("AND e.camera_id = ?")
        params.append(plan.filters.camera_id)
    if plan.filters.day:
        start = datetime.combine(date.fromisoformat(plan.filters.day), dtime.min)
        clauses.append("AND e.ts >= ? AND e.ts < ?")
        params += [start.timestamp(), (start + timedelta(days=1)).timestamp()]

    rows = conn.execute(
        f"""SELECT e.ts, e.t, e.label, e.kind, e.duration, e.video_id, e.segment_id,
                   z.name AS zone, i.name AS who
            FROM events e
            JOIN zones z ON z.id = e.zone_id
            JOIN tracklets t ON t.id = e.tracklet_id
            LEFT JOIN identities i ON i.id = t.identity_id
            WHERE 1=1 {" ".join(clauses)}
            ORDER BY e.ts LIMIT ?""",
        [*params, limit * 4],
    ).fetchall()

    return [
        AnswerRow(
            ts=r["ts"], label=r["label"], who=r["who"], zone=r["zone"],
            kind=r["kind"], duration=r["duration"] or 0.0,
            video_id=r["video_id"], t=r["t"], segment_id=r["segment_id"],
        )
        for r in rows if _within_band(r["ts"], plan.time_of_day)
    ][:limit]


def _sighting_rows(conn: sqlite3.Connection, plan: QueryPlan, limit: int) -> list[AnswerRow]:
    clauses, params = [], []
    if plan.filters.identity:
        clauses.append("AND i.name = ?")
        params.append(plan.filters.identity)
    if plan.filters.label:
        clauses.append("AND t.label = ?")
        params.append(plan.filters.label)
    if plan.filters.camera_id:
        clauses.append("AND t.camera_id = ?")
        params.append(plan.filters.camera_id)
    if plan.filters.day:
        start = datetime.combine(date.fromisoformat(plan.filters.day), dtime.min)
        clauses.append("AND t.ts_start >= ? AND t.ts_start < ?")
        params += [start.timestamp(), (start + timedelta(days=1)).timestamp()]

    rows = conn.execute(
        f"""SELECT t.ts_start AS ts, t.t_start AS t, t.label, t.video_id,
                   t.segment_id, i.name AS who, t.ts_end - t.ts_start AS duration
            FROM tracklets t LEFT JOIN identities i ON i.id = t.identity_id
            WHERE 1=1 {" ".join(clauses)}
            ORDER BY t.ts_start LIMIT ?""",
        [*params, limit * 4],
    ).fetchall()

    return [
        AnswerRow(
            ts=r["ts"], label=r["label"], who=r["who"], zone=None, kind=None,
            duration=r["duration"] or 0.0, video_id=r["video_id"],
            t=r["t"], segment_id=r["segment_id"],
        )
        for r in rows if _within_band(r["ts"], plan.time_of_day)
    ][:limit]


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%a %d %b, %H:%M:%S")


def _describe_subject(plan: QueryPlan) -> str:
    if plan.filters.identity:
        return plan.filters.identity
    if plan.filters.label:
        return f"a {plan.filters.label}"
    return "anything"


def answer(conn: sqlite3.Connection, plan: QueryPlan, limit: int = 20) -> Answer:
    """Answer a grounded plan from index facts alone.

    Only exact facts are used here. An ungrounded question - one where nothing
    matched a known person, zone, object or date - is not answered; it is
    handed back for ranked search instead, because guessing at it would mean
    inventing precision the index does not have.
    """
    if plan.unknown_names:
        names = ", ".join(sorted(set(plan.unknown_names)))
        known_names = [
            r["name"] for r in conn.execute("SELECT name FROM identities ORDER BY name")
        ]
        return Answer(
            plan,
            f"I do not know anyone called {names}. "
            f"Enrolled: {', '.join(known_names)}.",
            [],
            exact=True,
        )

    wants_events = bool(plan.filters.zone or plan.filters.event_kind)
    rows = (
        _event_rows(conn, plan, limit)
        if wants_events
        else _sighting_rows(conn, plan, limit)
    )
    subject = _describe_subject(plan)

    if not rows:
        where = f" at {plan.filters.zone}" if plan.filters.zone else ""
        when = f" on {plan.filters.day}" if plan.filters.day else ""
        return Answer(plan, f"No record of {subject}{where}{when}.", [], exact=True)

    if plan.intent == "how_many":
        headline = f"{len(rows)} time{'s' if len(rows) != 1 else ''}"
        if plan.filters.zone:
            headline += f" at {plan.filters.zone}"
        return Answer(plan, headline, rows)

    if plan.intent == "how_long":
        total = sum(r.duration for r in rows)
        longest = max(rows, key=lambda r: r.duration)
        return Answer(
            plan,
            f"{subject} for {total / 60:.1f} minutes across {len(rows)} "
            f"visit{'s' if len(rows) != 1 else ''}; longest "
            f"{longest.duration:.0f}s at {_clock(longest.ts)}.",
            rows,
        )

    if plan.intent == "who":
        names = sorted({r.who for r in rows if r.who})
        anonymous = sum(1 for r in rows if not r.who)
        if not names:
            return Answer(plan, f"{len(rows)} sighting(s), nobody recognised.", rows)
        listed = ", ".join(names)
        tail = f", plus {anonymous} unrecognised" if anonymous else ""
        return Answer(plan, f"{listed}{tail}.", rows)

    if plan.intent == "did":
        first = rows[0]
        return Answer(plan, f"Yes - {len(rows)} time(s), first at {_when(first.ts)}.", rows)

    # "when", and anything else that landed here.
    if len(rows) == 1:
        return Answer(plan, f"{_when(rows[0].ts)}.", rows)
    return Answer(
        plan,
        f"{len(rows)} times, from {_when(rows[0].ts)} to {_clock(rows[-1].ts)}.",
        rows,
    )


@dataclass
class AskResult:
    """What a typed question produced: an exact answer, ranked hits, or both."""

    plan: QueryPlan
    answer: Answer | None = None
    hits: list = field(default_factory=list)
    # Set when part of the question could not be answered exactly and was left
    # to similarity instead. Surfaced rather than swallowed: a filter that
    # silently ignores "in a red jacket" gives a confident, wrong-shaped answer.
    caveat: str = ""

    @property
    def mode(self) -> str:
        if self.answer and self.answer.found:
            return "answer"
        return "search" if self.hits else "empty"


def ask(
    conn: sqlite3.Connection,
    question: str,
    embed_text=None,
    today: date | None = None,
    limit: int = 20,
    min_similarity: float | None = None,
    model: str | None = None,
) -> AskResult:
    """Answer a question, ranking as a fallback.

    `embed_text` is an optional callable turning a string into a CLIP vector;
    without it the fallback is word matching alone.
    """
    from tsv.search import search as ranked_search

    plan = parse(conn, question, today=today)
    result = AskResult(plan=plan)

    if plan.grounded or plan.unknown_names:
        result.answer = answer(conn, plan, limit=limit)
        if plan.semantic_text and not plan.unknown_names:
            result.caveat = (
                f"'{plan.semantic_text}' is not something the index knows exactly, "
                f"so it did not narrow this answer."
            )

    needs_ranking = not plan.grounded or (result.answer and not result.answer.found)
    if needs_ranking or plan.semantic_text:
        vector = embed_text(plan.semantic_text or question) if embed_text else None
        result.hits = ranked_search(
            conn,
            text=plan.semantic_text or question,
            query_vector=vector,
            filters=plan.filters if plan.grounded else SearchFilters(),
            limit=limit,
            min_similarity=min_similarity,
            model=model,
        )
    return result
