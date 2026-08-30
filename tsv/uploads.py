"""Receiving a video from a phone, in pieces, so a dropped WiFi costs seconds.

The single-POST upload this replaces was fine for a file dragged from the
desktop and useless for the case it was actually written for. A phone
recording is gigabytes; over WiFi at the ~12 MB/s this laptop manages, four of
them is five minutes of holding a connection open. Anything that interrupts it
- the screen locking, walking out of range, the browser reclaiming memory -
starts the whole transfer again from zero.

**The offset is the file on disk, never a counter.** `os.stat` on the partial
file is the only source of truth about how much arrived. A number tracked
alongside it would be a number that can disagree with reality after a crash,
and the disagreement would be silent: the client resumes at an offset the
server believes, writes from there, and produces a file with a hole in it that
only fails much later, in the demuxer, as something that reads like a corrupt
video rather than a bad upload.

**Resuming is keyed by what the file is, not by who is asking.** A phone that
reloads the page has lost every variable it had; what it still has is the file
the user picked again. Name and size together identify it well enough to find
the partial data and carry on, which is why `find_resumable` takes those and
not a session id.

**Nothing is trusted about length.** The declared size is checked against free
disk space before a byte is accepted, and against what actually arrived before
the file is handed to the importer.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# What the client sends per request. Small enough that losing one is cheap,
# large enough that the per-request overhead disappears: at 12 MB/s this is
# under a second of work, and a 4 GB video is about 500 of them.
CHUNK_BYTES = 8 << 20

# Room to spare beyond the declared size, for the staged copy and the index.
# Refusing early beats filling somebody's disk and failing at 97%.
HEADROOM_BYTES = 512 << 20

# An upload nobody has touched in this long is abandoned. Phones get put in
# pockets; a few days is patient enough.
STALE_SECONDS = 3 * 86400


@dataclass
class Upload:
    id: int
    name: str
    size: int
    received: int
    path: Path

    @property
    def complete(self) -> bool:
        return self.received >= self.size

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "size": self.size,
            "offset": self.received,
            "complete": self.complete,
            "chunk_bytes": CHUNK_BYTES,
        }


def _incoming(data_dir: Path) -> Path:
    directory = data_dir / "incoming"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _part_path(data_dir: Path, upload_id: int, name: str) -> Path:
    # The id is in the filename so two people uploading "video.mp4" at the
    # same time cannot write into each other's file.
    return _incoming(data_dir) / f".part-{upload_id}-{Path(name).name}"


def _row_to_upload(row: sqlite3.Row) -> Upload:
    path = Path(row["path"])
    # Stat, not a stored count. See the note at the top of this file.
    received = path.stat().st_size if path.is_file() else 0
    return Upload(
        id=int(row["id"]), name=row["name"], size=int(row["size"]),
        received=received, path=path,
    )


def space_for(data_dir: Path, size: int) -> tuple[bool, str]:
    """Whether there is room, and what to say if not."""
    try:
        free = shutil.disk_usage(_incoming(data_dir)).free
    except OSError:
        return True, ""        # cannot tell; let the write fail honestly
    needed = size + HEADROOM_BYTES
    if free >= needed:
        return True, ""
    return False, (
        f"Not enough room: this needs about {needed / 1e9:.1f} GB free "
        f"and there is {free / 1e9:.1f} GB."
    )


def begin(conn: sqlite3.Connection, data_dir: Path, name: str, size: int) -> Upload:
    """Start receiving, or pick up where an earlier attempt stopped."""
    existing = find_resumable(conn, name, size)
    if existing is not None:
        return existing

    now = time.time()
    cur = conn.execute(
        """INSERT INTO uploads(name, size, path, started_at, touched_at)
           VALUES (?,?,?,?,?)""",
        (Path(name).name, int(size), "", now, now),
    )
    upload_id = int(cur.lastrowid)
    path = _part_path(data_dir, upload_id, name)
    conn.execute("UPDATE uploads SET path = ? WHERE id = ?", (str(path), upload_id))
    conn.commit()

    path.touch()
    return Upload(upload_id, Path(name).name, int(size), 0, path)


def find_resumable(
    conn: sqlite3.Connection, name: str, size: int
) -> Upload | None:
    """An unfinished upload of the same file, if there is one.

    Matched on name and size because that is all a phone still knows after a
    reload. Two genuinely different videos with the same name *and* the same
    byte count is not a collision worth engineering around - and if it did
    happen, the resumed file fails its size check or the demuxer, rather than
    being quietly indexed as the wrong thing.
    """
    row = conn.execute(
        """SELECT id, name, size, path FROM uploads
           WHERE name = ? AND size = ? AND finished_at IS NULL
           ORDER BY id DESC LIMIT 1""",
        (Path(name).name, int(size)),
    ).fetchone()
    if row is None:
        return None

    upload = _row_to_upload(row)
    if not upload.path.is_file():
        # The row outlived its data - a cleaned temp directory, most likely.
        conn.execute("DELETE FROM uploads WHERE id = ?", (upload.id,))
        conn.commit()
        return None
    return upload


def get(conn: sqlite3.Connection, upload_id: int) -> Upload | None:
    row = conn.execute(
        "SELECT id, name, size, path FROM uploads WHERE id = ?", (upload_id,)
    ).fetchone()
    return _row_to_upload(row) if row is not None else None


def write_chunk(
    conn: sqlite3.Connection, upload: Upload, offset: int, data: bytes
) -> Upload:
    """Append one chunk, if it starts exactly where the file ends.

    A mismatch is answered with the real offset rather than an error: the
    common cause is a chunk that arrived and was written while the reply was
    lost, so the client retried something already stored. Telling it where the
    file actually ends lets it carry on instead of starting again.
    """
    if offset != upload.received:
        return upload

    with upload.path.open("ab") as handle:
        handle.write(data)

    conn.execute(
        "UPDATE uploads SET touched_at = ? WHERE id = ?", (time.time(), upload.id)
    )
    conn.commit()
    return Upload(
        upload.id, upload.name, upload.size,
        upload.path.stat().st_size, upload.path,
    )


def finish(conn: sqlite3.Connection, data_dir: Path, upload: Upload) -> Path:
    """Turn a completed upload into a file the importer can take.

    Raises if the bytes on disk do not match what was promised. A short file
    would reach the demuxer and be reported as a corrupt video, which sends
    somebody looking at their camera instead of their WiFi.
    """
    actual = upload.path.stat().st_size if upload.path.is_file() else 0
    if actual != upload.size:
        raise ValueError(
            f"upload is {actual} bytes, expected {upload.size}"
        )

    from tsv.importer import stage_video

    final = stage_video(upload.path, _incoming(data_dir), name=upload.name)
    conn.execute(
        "UPDATE uploads SET finished_at = ?, path = ? WHERE id = ?",
        (time.time(), str(final), upload.id),
    )
    conn.commit()
    return final


def abandon(conn: sqlite3.Connection, upload_id: int) -> bool:
    upload = get(conn, upload_id)
    if upload is None:
        return False
    upload.path.unlink(missing_ok=True)
    conn.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    return True


def sweep(conn: sqlite3.Connection, older_than: float = STALE_SECONDS) -> int:
    """Delete partial uploads nobody came back for.

    These are whole videos sitting in a temporary directory; left alone they
    are the largest thing this application would ever leak onto a disk.
    """
    cutoff = time.time() - older_than
    rows = conn.execute(
        "SELECT id, name, size, path FROM uploads "
        "WHERE finished_at IS NULL AND touched_at < ?",
        (cutoff,),
    ).fetchall()

    removed = 0
    for row in rows:
        Path(row["path"]).unlink(missing_ok=True)
        conn.execute("DELETE FROM uploads WHERE id = ?", (int(row["id"]),))
        removed += 1
    conn.commit()
    return removed


def pending(conn: sqlite3.Connection) -> list[Upload]:
    rows = conn.execute(
        "SELECT id, name, size, path FROM uploads "
        "WHERE finished_at IS NULL ORDER BY id DESC"
    ).fetchall()
    return [_row_to_upload(row) for row in rows]
