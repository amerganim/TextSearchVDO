"""Sampling frames out of candidate windows.

Both analysis stages walk a video the same way - seek to a window, decode
forward, keep roughly N frames a second - and differ only in the pixel format
and size they want. Phase 0 wants small grey frames for background
subtraction; Phase 1 wants larger RGB ones for detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass
class Sample:
    t: float
    frame: np.ndarray
    window_index: int
    # True on the first sample of each window, so consumers that hold state
    # across frames know to reset it.
    window_start: bool


def output_size(src_w: int, src_h: int, target_width: int | None) -> tuple[int, int]:
    """Even dimensions at or below `target_width`, preserving aspect ratio."""
    if not target_width or target_width >= src_w:
        width = src_w
    else:
        width = target_width
    height = max(2, int(round(src_h * width / src_w)))
    return max(2, width - width % 2), height - height % 2


def sample_windows(
    path: Path,
    windows: list[tuple[float, float]],
    fps: float,
    width: int | None = None,
    pixel_format: str = "rgb24",
    rotation: int = 0,
) -> Iterator[Sample]:
    """Yield frames at ~`fps` from each window, in order.

    Frames decoded between the seek point and the window start are skipped
    rather than yielded: seeking lands on the preceding keyframe, which can be
    seconds earlier.

    `rotation` turns each frame upright, in degrees clockwise off upright as
    stored. Applied here rather than by each caller because everything that
    looks at pixels - detection, crops, faces, captions, thumbnails - has to
    agree about which way up the picture is, and one of them disagreeing
    means boxes drawn against a differently-oriented frame.
    """
    if not windows:
        return

    with av.open(str(path)) as container:
        if not container.streams.video:
            return
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        if time_base is None:
            return

        src_w = stream.codec_context.width or (width or 640)
        src_h = stream.codec_context.height or (width or 640)
        out_w, out_h = output_size(src_w, src_h, width)
        period = 1.0 / fps if fps > 0 else 0.0

        for window_index, (start, end) in enumerate(windows):
            try:
                container.seek(int(start / time_base), stream=stream, backward=True)
            except av.FFmpegError:
                continue

            next_sample = start
            first = True
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * time_base)
                if t > end:
                    break
                if t < next_sample:
                    continue
                next_sample = t + period

                array = frame.reformat(
                    width=out_w, height=out_h, format=pixel_format
                ).to_ndarray()
                if rotation:
                    from tsv.orientation import apply as turn_upright

                    array = turn_upright(array, rotation)
                yield Sample(t=t, frame=array, window_index=window_index, window_start=first)
                first = False
