"""Read a video file's shape and, harder, work out when it was recorded.

Wall-clock accuracy is load-bearing for this whole product: every answer the
app gives is a timestamp. A file whose start time is wrong by an hour produces
confidently wrong answers, so each guess is recorded with the source that
produced it and surfaced in the UI.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av

# Ordered most-specific first; the first pattern that yields a sane datetime
# wins. Groups are always (Y, M, D, h, m, s).
_FILENAME_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "filename:ymd-hms",
        re.compile(
            r"(?P<Y>\d{4})[-_.](?P<M>\d{2})[-_.](?P<D>\d{2})"
            r"[ _T-]+(?P<h>\d{2})[-_.:](?P<m>\d{2})[-_.:](?P<s>\d{2})"
        ),
    ),
    (
        "filename:ymd_hms",
        re.compile(
            r"(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})[ _T-]+(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})"
        ),
    ),
    (
        "filename:ymdhms",
        re.compile(
            r"(?<!\d)(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})(?!\d)"
        ),
    ),
]

_CAMERA_PATTERNS = [
    # Not \b: an underscore is a word character, so "_ch1" has no boundary.
    re.compile(r"(?<![A-Za-z0-9])(?P<name>(?:ch|cam|camera|channel)[\s_-]?\d+)", re.IGNORECASE),
]

_VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".dav", ".h264", ".flv"}


@dataclass
class VideoInfo:
    path: Path
    start_ts: float
    ts_source: str
    duration: float
    fps: float | None
    width: int | None
    height: int | None
    codec: str | None
    size_bytes: int
    mtime: float
    camera: str
    # Identifies the recording rather than the file. Two copies of one video
    # under different names have the same fingerprint, which is how a second
    # upload of something already indexed is recognised as a duplicate.
    fingerprint: str = ""


def _parse_filename_ts(name: str) -> tuple[float, str] | None:
    for source, pattern in _FILENAME_PATTERNS:
        for match in pattern.finditer(name):
            try:
                dt = datetime(
                    int(match["Y"]),
                    int(match["M"]),
                    int(match["D"]),
                    int(match["h"]),
                    int(match["m"]),
                    int(match["s"]),
                )
            except ValueError:
                continue  # e.g. month 99 - keep scanning the same filename
            if 2000 <= dt.year <= 2100:
                # CCTV filenames carry local wall clock; a naive datetime's
                # .timestamp() interprets it in the local zone, which is what
                # we want.
                return dt.timestamp(), source
    return None


def _parse_metadata_ts(container: "av.container.InputContainer") -> tuple[float, str] | None:
    raw = container.metadata.get("creation_time")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.timestamp(), "metadata:creation_time"


def guess_camera(path: Path) -> str:
    """Camera name from the filename, else the containing directory."""
    for pattern in _CAMERA_PATTERNS:
        match = pattern.search(path.name)
        if match:
            return re.sub(r"[\s_-]+", "", match["name"]).lower()
    parent = path.parent.name
    return parent.lower() if parent else "default"


def iter_videos(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _VIDEO_SUFFIXES and p.is_file())


# Enough of the file to identify it, and no more. A recording can be many
# gigabytes and hashing all of it would cost more than the motion scan that
# follows; the first and last megabyte plus the exact byte count is specific
# enough that two different videos colliding is not a practical concern,
# while two copies of the same one always agree.
_FINGERPRINT_BYTES = 1 << 20


def fingerprint(path: Path) -> str:
    """A cheap content identity for a video file."""
    size = path.stat().st_size
    digest = hashlib.blake2b(str(size).encode(), digest_size=16)
    with path.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_BYTES))
        if size > _FINGERPRINT_BYTES * 2:
            handle.seek(-_FINGERPRINT_BYTES, os.SEEK_END)
            digest.update(handle.read(_FINGERPRINT_BYTES))
    return digest.hexdigest()


def probe(path: Path) -> VideoInfo:
    stat = path.stat()
    duration = 0.0
    fps = width = height = codec = None
    meta_ts = None

    with av.open(str(path)) as container:
        meta_ts = _parse_metadata_ts(container)
        if container.duration is not None:
            duration = container.duration / av.time_base
        if container.streams.video:
            stream = container.streams.video[0]
            if not duration and stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            fps = float(stream.average_rate) if stream.average_rate else None
            width, height = stream.codec_context.width, stream.codec_context.height
            codec = stream.codec_context.name

    # Filename beats container metadata: NVRs routinely rewrite or drop
    # creation_time when exporting, but the name is generated by the recorder.
    resolved = _parse_filename_ts(path.name) or meta_ts
    if resolved is None:
        # Last resort. mtime is when writing *finished*, so back off by the
        # duration to approximate the start.
        resolved = (stat.st_mtime - duration, "mtime-duration")

    start_ts, ts_source = resolved
    return VideoInfo(
        path=path,
        start_ts=start_ts,
        ts_source=ts_source,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
        size_bytes=stat.st_size,
        mtime=stat.st_mtime,
        camera=guess_camera(path),
        fingerprint=fingerprint(path),
    )
