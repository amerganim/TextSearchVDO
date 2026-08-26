"""SQLite index.

Phase 0 only fills cameras/videos/segments, but the schema is laid out so the
later phases (tracklets, identities, zones, captions, embeddings) attach to
`segments` without a migration of the existing tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

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

CREATE INDEX IF NOT EXISTS idx_segments_ts ON segments(ts_start, ts_end);
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


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT version FROM schema_info").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_info(version) VALUES (?)", (SCHEMA_VERSION,))
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
