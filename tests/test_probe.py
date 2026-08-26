from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tsv.probe import _parse_filename_ts, guess_camera


def _iso(name: str) -> str | None:
    parsed = _parse_filename_ts(name)
    return datetime.fromtimestamp(parsed[0]).isoformat() if parsed else None


@pytest.mark.parametrize(
    "name",
    [
        "ch01_20260826123000.mp4",
        "20260826_123000.mp4",
        "2026-08-26_12-30-00.mp4",
        "2026-08-26 12.30.00.mkv",
        "NVR_ch1_main_20260826123000_20260826133000.mp4",
        "cam3-20260826-123000.mp4",
    ],
)
def test_recorder_filename_layouts_all_parse(name: str):
    assert _iso(name) == "2026-08-26T12:30:00"


@pytest.mark.parametrize(
    "name",
    [
        "random_clip.mp4",
        "ch02_20261399999999.mp4",   # month 13, day 99
        "movie_1080p.mp4",
        "IMG_0042.mp4",
    ],
)
def test_names_without_a_real_timestamp_are_refused(name: str):
    """Better to fall back to mtime than to invent a confident wrong time."""
    assert _parse_filename_ts(name) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ch01_x.mp4", "ch01"),
        ("NVR_ch1_main_x.mp4", "ch1"),
        ("cam3-x.mp4", "cam3"),
        ("Channel 12 x.mp4", "channel12"),
        ("arch1val.mp4", "default"),   # must not match "ch1" inside a word
    ],
)
def test_camera_name_detection(name: str, expected: str):
    assert guess_camera(Path(name)) == expected


def test_camera_falls_back_to_the_containing_folder():
    assert guess_camera(Path("front_door/clip.mp4")) == "front_door"
