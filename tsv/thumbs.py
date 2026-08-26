"""Segment thumbnails.

One container open serves every thumbnail for a file: reopening per segment
dominates the cost once a night of footage yields a few hundred of them.
"""

from __future__ import annotations

from pathlib import Path

import av
from PIL import Image


def extract_thumbs(
    video_path: Path,
    offsets: list[float],
    out_paths: list[Path],
    width: int = 320,
    quality: int = 80,
) -> list[Path | None]:
    if len(offsets) != len(out_paths):
        raise ValueError("offsets and out_paths must be the same length")
    results: list[Path | None] = [None] * len(offsets)
    if not offsets:
        return results

    # Ascending offsets keep the seeks moving forward through the file.
    order = sorted(range(len(offsets)), key=lambda i: offsets[i])

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            return results
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        if time_base is None:
            return results

        src_w = stream.codec_context.width or width
        src_h = stream.codec_context.height or width
        out_w = min(width, src_w)
        out_h = max(2, int(round(src_h * out_w / src_w)))
        out_h -= out_h % 2

        for i in order:
            offset = max(0.0, offsets[i])
            out_path = out_paths[i]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                container.seek(int(offset / time_base), stream=stream, backward=True)
            except av.FFmpegError:
                continue

            chosen = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                chosen = frame
                if float(frame.pts * time_base) >= offset - 1e-3:
                    break
            if chosen is None:
                continue

            array = chosen.reformat(width=out_w, height=out_h, format="rgb24").to_ndarray()
            Image.fromarray(array).save(out_path, format="JPEG", quality=quality)
            results[i] = out_path

    return results
