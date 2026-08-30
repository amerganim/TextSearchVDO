"""SQLite index.

Phase 0 only fills cameras/videos/segments, but the schema is laid out so the
later phases (tracklets, identities, zones, captions, embeddings) attach to
`segments` without a migration of the existing tables.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 11

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    tz          TEXT NOT NULL DEFAULT 'UTC'
);

CREATE TABLE IF NOT EXISTS videos (
    id          INTEGER PRIMARY KEY,
    camera_id   INTEGER NOT NULL REFERENCES cameras(id),
    path        TEXT NOT NULL UNIQUE,
    -- Wall-clock start of the recording, unix seconds. Everything the user
    -- ever sees is derived from this, so a wrong value here is worse than a
    -- missing one; see probe.py for how confidently it was determined.
    start_ts    REAL NOT NULL,
    ts_source   TEXT NOT NULL,
    duration    REAL NOT NULL,
    fps         REAL,
    width       INTEGER,
    height      INTEGER,
    codec       TEXT,
    size_bytes  INTEGER,
    mtime       REAL,
    -- Ingest bookkeeping.
    ingested_at REAL,
    active_seconds REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    camera_id   INTEGER NOT NULL REFERENCES cameras(id),
    -- Offsets within the video file, seconds.
    t_start     REAL NOT NULL,
    t_end       REAL NOT NULL,
    -- Absolute wall clock, unix seconds. Denormalised so the timeline can be
    -- queried by time range without joining videos.
    ts_start    REAL NOT NULL,
    ts_end      REAL NOT NULL,
    activity_score REAL NOT NULL,
    peak_offset REAL NOT NULL,
    thumb_path  TEXT
);

-- Phase 1. One row per tracked object within a segment: "a person was here,
-- from this second to that one, moving this way".
CREATE TABLE IF NOT EXISTS tracklets (
    id          INTEGER PRIMARY KEY,
    segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    camera_id   INTEGER NOT NULL REFERENCES cameras(id),
    cls         INTEGER NOT NULL,
    label       TEXT NOT NULL,
    t_start     REAL NOT NULL,
    t_end       REAL NOT NULL,
    ts_start    REAL NOT NULL,
    ts_end      REAL NOT NULL,
    n_detections INTEGER NOT NULL,
    mean_score  REAL NOT NULL,
    max_score   REAL NOT NULL,
    -- Normalised 0..1 centre at first and last sighting. Phase 2 reads these
    -- to decide "entered" versus "left" against a user-drawn line.
    x_start     REAL, y_start REAL, x_end REAL, y_end REAL,
    -- Normalised union of every box in the tracklet.
    x1          REAL, y1 REAL, x2 REAL, y2 REAL,
    thumb_path  TEXT
);

-- Transfers in progress. How much has arrived is deliberately NOT stored
-- here: it is os.stat on the partial file, because a counter can disagree
-- with the disk after a crash and the disagreement produces a file with a
-- hole in it that only fails later, in the demuxer, looking like a corrupt
-- video rather than a bad upload.
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    size        INTEGER NOT NULL,
    path        TEXT NOT NULL,
    started_at  REAL NOT NULL,
    touched_at  REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_upload_resume ON uploads(name, size, finished_at);

-- Phones that have been let in. A cookie names a row here, so revoking is a
-- DELETE and takes effect on the very next request rather than whenever a
-- token would have expired.
CREATE TABLE IF NOT EXISTS devices (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    user_agent  TEXT,
    address     TEXT,
    paired_at   REAL NOT NULL,
    last_seen   REAL
);

-- What was said, and when. Separate from segments because speech does not
-- respect them: somebody can talk during a stretch the motion pass discarded,
-- and that is still worth finding. segment_id is therefore nullable.
CREATE TABLE IF NOT EXISTS utterances (
    id          INTEGER PRIMARY KEY,
    video_id    INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_id  INTEGER REFERENCES segments(id) ON DELETE SET NULL,
    t_start     REAL NOT NULL,
    t_end       REAL NOT NULL,
    ts_start    REAL NOT NULL,
    ts_end      REAL NOT NULL,
    text        TEXT NOT NULL,
    -- The model's own average log probability. Kept so a later pass can
    -- raise the bar without re-transcribing everything.
    confidence  REAL
);
CREATE INDEX IF NOT EXISTS idx_utterance_video ON utterances(video_id, t_start);
CREATE INDEX IF NOT EXISTS idx_utterance_segment ON utterances(segment_id);

-- Individual sightings. Kept because Phase 2 identity and Phase 3 captioning
-- both need to crop the actual pixels at a specific instant.
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY,
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id) ON DELETE CASCADE,
    t           REAL NOT NULL,
    ts          REAL NOT NULL,
    x1          REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
    score       REAL NOT NULL
);

-- Phase 2. Zones are drawn once by the user, in normalised coordinates so
-- they survive the camera being reconfigured to a different resolution.
CREATE TABLE IF NOT EXISTS zones (
    id          INTEGER PRIMARY KEY,
    camera_id   INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('region', 'line')),
    -- JSON [[x, y], ...]; a line has exactly two points, a region at least 3.
    points      TEXT NOT NULL,
    created_at  REAL,
    UNIQUE (camera_id, name)
);

-- What a tracklet did against a zone. Derived from the stored detections
-- rather than during analysis: zones get edited, and recomputing must not
-- require re-running the detector or touching the video again.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id) ON DELETE CASCADE,
    zone_id     INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    segment_id  INTEGER NOT NULL,
    video_id    INTEGER NOT NULL,
    camera_id   INTEGER NOT NULL,
    label       TEXT NOT NULL,
    kind        TEXT NOT NULL,
    t           REAL NOT NULL,
    ts          REAL NOT NULL,
    duration    REAL NOT NULL DEFAULT 0
);

-- Phase 2 identity. A person the user has named.
CREATE TABLE IF NOT EXISTS identities (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    notes       TEXT,
    created_at  REAL
);

-- The gallery: embeddings the user has confirmed belong to an identity.
-- `kind` separates face vectors from body/appearance ones; they live in
-- different spaces and must never be compared to each other.
CREATE TABLE IF NOT EXISTS identity_embeddings (
    id          INTEGER PRIMARY KEY,
    identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    tracklet_id INTEGER REFERENCES tracklets(id) ON DELETE SET NULL,
    kind        TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  REAL
);

-- One aggregated embedding per tracklet per kind, so matching does not have
-- to re-read every detection.
CREATE TABLE IF NOT EXISTS tracklet_embeddings (
    tracklet_id INTEGER NOT NULL REFERENCES tracklets(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    n_samples   INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (tracklet_id, kind)
);

-- Phase 3. A CLIP vector per segment, from its peak frame: scene-level
-- meaning, for queries like "a car in the driveway".
CREATE TABLE IF NOT EXISTS segment_embeddings (
    segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    PRIMARY KEY (segment_id, kind)
);

-- Lexical half of retrieval. One row per segment, holding everything already
-- known about it in words: object labels, who was there, which zones it
-- touched. Semantic search finds things that look right; this finds things
-- that are named right, and the two fail in different places.
CREATE VIRTUAL TABLE IF NOT EXISTS segment_text USING fts5(
    body,
    segment_id UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE INDEX IF NOT EXISTS idx_segments_ts ON segments(ts_start, ts_end);
CREATE INDEX IF NOT EXISTS idx_identity_emb ON identity_embeddings(identity_id, kind);
CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind, ts);
CREATE INDEX IF NOT EXISTS idx_events_tracklet ON events(tracklet_id);
CREATE INDEX IF NOT EXISTS idx_tracklets_segment ON tracklets(segment_id);
CREATE INDEX IF NOT EXISTS idx_tracklets_ts ON tracklets(ts_start, ts_end);
CREATE INDEX IF NOT EXISTS idx_tracklets_label ON tracklets(label, ts_start);
CREATE INDEX IF NOT EXISTS idx_detections_tracklet ON detections(tracklet_id);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_segments_camera_ts ON segments(camera_id, ts_start);
CREATE INDEX IF NOT EXISTS idx_videos_camera_ts ON videos(camera_id, start_ts);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """Add a column if it is not already there.

    SQLite has no ADD COLUMN IF NOT EXISTS, and every table in SCHEMA is
    CREATE IF NOT EXISTS, so new *columns* on existing tables are the only
    part of a schema bump that needs handling by hand.
    """
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def _backfill_fingerprints(conn: sqlite3.Connection) -> None:
    """Fingerprint videos indexed before there was such a column.

    Without this the duplicate check only protects libraries built from
    scratch, and an existing one keeps accepting copies of what it already
    holds. Two megabytes are read per file, and a file that has since moved
    or been deleted is left alone rather than treated as an error.
    """
    rows = conn.execute(
        "SELECT id, path FROM videos WHERE fingerprint IS NULL"
    ).fetchall()
    if not rows:
        return

    from tsv.probe import fingerprint

    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            continue
        try:
            conn.execute(
                "UPDATE videos SET fingerprint = ? WHERE id = ?",
                (fingerprint(path), int(row["id"])),
            )
        except OSError:
            continue
    conn.commit()


# What produced the vectors in an index built before they were labelled.
# There was one choice per kind at the time, so this is a record, not a guess.
_ORIGINAL_MODELS = {"clip": "clip-vit-b-32", "face": "buffalo_s"}


def _backfill_embedding_models(conn: sqlite3.Connection) -> None:
    for table in ("segment_embeddings", "tracklet_embeddings", "identity_embeddings"):
        for kind, name in _ORIGINAL_MODELS.items():
            conn.execute(
                f"UPDATE {table} SET model = ? WHERE model IS NULL AND kind = ?",
                (name, kind),
            )
    conn.commit()


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)

    # Denormalised onto segments so the timeline can label a segment without
    # joining tracklets for every row on screen.
    ensure_column(conn, "segments", "analyzed_at", "REAL")
    ensure_column(conn, "segments", "n_tracklets", "INTEGER")
    ensure_column(conn, "segments", "labels", "TEXT")

    # Who this tracklet is, once identity has run. Kept on the tracklet rather
    # than in a join table because a tracklet is one continuous sighting of
    # one object - it cannot be two people.
    ensure_column(conn, "tracklets", "identity_id", "INTEGER REFERENCES identities(id)")
    ensure_column(conn, "tracklets", "identity_score", "REAL")
    ensure_column(conn, "tracklets", "identity_source", "TEXT")

    # Phase 4. What a vision-language model said this person was doing. Stored
    # on the tracklet because that is the unit captioned - one description per
    # continuous sighting, not per frame, since the vision encoder costs about
    # six seconds an image and dominates everything else.
    # Which recording this file is, independent of where it sits or what it
    # is called. Uploading the same video a second time used to make a second
    # library entry, and the totals on screen then counted it twice.
    ensure_column(conn, "videos", "fingerprint", "TEXT")

    # When speech was last read out of this file. Separate from "has
    # utterances", because a silent recording is transcribed, finds nothing,
    # and must not be retried on every run.
    ensure_column(conn, "videos", "transcribed_at", "REAL")

    # Which weights produced each vector. A stored embedding is only
    # comparable to one from the same model, and nothing about the numbers
    # says which that was: CLIP ViT-B/32 and ViT-B/16 are both 512-wide and
    # completely incompatible, so mixing them raises no error and quietly
    # returns nonsense. Rows written before this column existed came from the
    # only models that were on offer, which is what the backfill records.
    ensure_column(conn, "segment_embeddings", "model", "TEXT")
    ensure_column(conn, "tracklet_embeddings", "model", "TEXT")
    ensure_column(conn, "identity_embeddings", "model", "TEXT")

    ensure_column(conn, "tracklets", "caption", "TEXT")
    ensure_column(conn, "tracklets", "caption_task", "TEXT")
    ensure_column(conn, "tracklets", "captioned_at", "REAL")

    _backfill_fingerprints(conn)
    _backfill_embedding_models(conn)

    row = conn.execute("SELECT version FROM schema_info").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
    else:
        conn.execute("UPDATE schema_info SET version = ?", (SCHEMA_VERSION,))
    conn.commit()


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = connect(db_path)
    init(conn)
    return conn


class ThreadLocalConnection:
    """A connection facade handing each thread its own SQLite handle.

    FastAPI runs synchronous endpoints in a worker threadpool, so a page that
    asks for several thumbnails at once hits the database from several threads
    at the same moment. A single `sqlite3.Connection` is not safe for that even
    with `check_same_thread=False`: that flag only silences the ownership
    check, and concurrent `execute()` calls on one handle raise
    `InterfaceError: bad parameter or other API misuse`.

    One handle per thread avoids it outright, and WAL mode means the readers
    do not block each other.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._local = threading.local()
        # Initialise the schema once, on the connection that opened the file.
        init(self._handle())

    def _handle(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = connect(self._path)
            self._local.conn = conn
        return conn

    def execute(self, *args, **kwargs) -> sqlite3.Cursor:
        return self._handle().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs) -> sqlite3.Cursor:
        return self._handle().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs) -> sqlite3.Cursor:
        return self._handle().executescript(*args, **kwargs)

    def cursor(self) -> sqlite3.Cursor:
        return self._handle().cursor()

    def commit(self) -> None:
        self._handle().commit()

    def rollback(self) -> None:
        self._handle().rollback()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def open_threadlocal(db_path: Path) -> ThreadLocalConnection:
    """A database handle safe to share with a threaded server."""
    return ThreadLocalConnection(db_path)


def get_or_create_camera(conn: sqlite3.Connection, name: str, tz: str = "UTC") -> int:
    row = conn.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute("INSERT INTO cameras(name, tz) VALUES (?, ?)", (name, tz))
    conn.commit()
    return int(cur.lastrowid)
