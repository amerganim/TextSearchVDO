"""SQLite index.

Phase 0 only fills cameras/videos/segments, but the schema is laid out so the
later phases (tracklets, identities, zones, captions, embeddings) attach to
`segments` without a migration of the existing tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2

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

CREATE INDEX IF NOT EXISTS idx_segments_ts ON segments(ts_start, ts_end);
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


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)

    # Denormalised onto segments so the timeline can label a segment without
    # joining tracklets for every row on screen.
    ensure_column(conn, "segments", "analyzed_at", "REAL")
    ensure_column(conn, "segments", "n_tracklets", "INTEGER")
    ensure_column(conn, "segments", "labels", "TEXT")

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


def get_or_create_camera(conn: sqlite3.Connection, name: str, tz: str = "UTC") -> int:
    row = conn.execute("SELECT id FROM cameras WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute("INSERT INTO cameras(name, tz) VALUES (?, ?)", (name, tz))
    conn.commit()
    return int(cur.lastrowid)
