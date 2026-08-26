from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.make_synthetic import build  # noqa: E402

# Small and short: these exist to prove correctness, not to be realistic. The
# realistic tuning pass happens against real footage.
CLIP_W, CLIP_H, CLIP_FPS = 320, 240, 15


@pytest.fixture(scope="session")
def day_clip(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A daytime clip with three known activity windows."""
    out = tmp_path_factory.mktemp("footage") / "ch09_20260101080000.mp4"
    return build(
        out_path=out,
        duration=30.0,
        activity=[(6.0, 10.0), (20.0, 25.0)],
        ir_flip_at=None,
        fps=CLIP_FPS,
        width=CLIP_W,
        height=CLIP_H,
        seed=3,
    )


@pytest.fixture(scope="session")
def idle_clip(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """No activity at all - the false-positive guard."""
    out = tmp_path_factory.mktemp("footage") / "ch08_20260101090000.mp4"
    return build(
        out_path=out,
        duration=25.0,
        activity=[],
        ir_flip_at=None,
        fps=CLIP_FPS,
        width=CLIP_W,
        height=CLIP_H,
        seed=5,
    )
