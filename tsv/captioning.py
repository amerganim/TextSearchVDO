"""Captioning stored tracklets.

A separate pass from analysis, deliberately. Captioning costs about six
seconds an image on a CPU - the vision encoder is the whole of it, and the
decoder is 25ms a token - so a day of footage is the better part of an hour.
That is a background job somebody chooses to run, not something to make them
wait through while a video imports.

Being a separate pass also means it is resumable: it captions what has no
caption yet, so an interrupted run costs nothing but the tracklet it was on.

Unlike zone events, this one *does* need the video: a caption describes pixels,
and only the boxes were kept.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tsv.config import Config
from tsv.frames import sample_windows
from tsv.models.caption import Florence2Captioner, build_captioner, crop_for_caption


@dataclass
class CaptionSummary:
    considered: int = 0
    captioned: int = 0
    skipped_small: int = 0
    failed: int = 0
    elapsed: float = 0.0
    samples: list[str] = field(default_factory=list)

    @property
    def per_caption(self) -> float:
        return self.elapsed / self.captioned if self.captioned else 0.0

    def as_dict(self) -> dict:
        return {
            "considered": self.considered,
            "captioned": self.captioned,
            "skipped_small": self.skipped_small,
            "failed": self.failed,
            "elapsed": round(self.elapsed, 1),
            "per_caption": round(self.per_caption, 2),
            "samples": self.samples[:5],
        }


def pending_tracklets(conn: sqlite3.Connection, cfg: Config, force: bool = False) -> list:
    """Tracklets that want a caption, newest first.

    Only the classes worth describing, and only those without one already,
    unless the caller insists.
    """
    labels = list(cfg.caption.labels)
    placeholders = ",".join("?" * len(labels)) or "''"
    where = "" if force else " AND t.caption IS NULL"
    return conn.execute(
        f"""SELECT t.id, t.video_id, t.t_start, t.t_end, t.label,
                   t.x1, t.y1, t.x2, t.y2, v.path
            FROM tracklets t JOIN videos v ON v.id = t.video_id
            WHERE t.label IN ({placeholders}){where}
            ORDER BY t.ts_start DESC""",
        labels,
    ).fetchall()


def caption_tracklets(
    conn: sqlite3.Connection,
    cfg: Config,
    captioner: Florence2Captioner | None = None,
    force: bool = False,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> CaptionSummary:
    """Describe what each person was doing, one caption per tracklet."""
    summary = CaptionSummary()
    started = time.time()

    captioner = captioner or build_captioner(
        cfg.caption_model_dir,
        max_tokens=cfg.caption.max_tokens,
        force_backend=cfg.caption.force_backend,
    )
    if captioner is None:
        raise FileNotFoundError(
            f"no captioning model in {cfg.caption_model_dir}; "
            f"run tools/fetch_caption_model.py"
        )

    rows = pending_tracklets(conn, cfg, force=force)
    if limit:
        rows = rows[:limit]
    summary.considered = len(rows)

    # Grouped by video so the file is opened once and walked forwards, the same
    # reason analysis batches its segments.
    by_video: dict[str, list] = {}
    for row in rows:
        by_video.setdefault(row["path"], []).append(row)

    done = 0
    for path_str, tracklets in by_video.items():
        path = Path(path_str)
        if not path.is_file():
            summary.failed += len(tracklets)
            continue

        tracklets.sort(key=lambda r: r["t_start"])
        # One frame per tracklet, taken at its midpoint: the moment a person is
        # most likely to be fully in shot rather than entering or leaving.
        windows = []
        for row in tracklets:
            middle = (float(row["t_start"]) + float(row["t_end"])) / 2.0
            windows.append((middle, middle + 0.5))

        wanted = {i: tracklets[i] for i in range(len(tracklets))}
        seen: set[int] = set()

        for sample in sample_windows(path, windows, fps=1.0, width=None,
                                     pixel_format="rgb24"):
            index = sample.window_index
            if index in seen or index not in wanted:
                continue
            seen.add(index)
            row = wanted[index]

            height, width = sample.frame.shape[:2]
            box = (row["x1"] * width, row["y1"] * height,
                   row["x2"] * width, row["y2"] * height)
            crop = crop_for_caption(
                sample.frame, box,
                context=cfg.caption.context,
                min_side=cfg.caption.min_crop_px,
            )
            if crop is None:
                summary.skipped_small += 1
                done += 1
                if on_progress:
                    on_progress(done, summary.considered)
                continue

            try:
                caption = captioner.caption(crop, cfg.caption.task)
            except Exception:  # noqa: BLE001 - one bad crop must not stop the run
                summary.failed += 1
                done += 1
                if on_progress:
                    on_progress(done, summary.considered)
                continue

            conn.execute(
                "UPDATE tracklets SET caption = ?, caption_task = ?, captioned_at = ? "
                "WHERE id = ?",
                (caption.text, caption.task, time.time(), row["id"]),
            )
            summary.captioned += 1
            if len(summary.samples) < 8 and caption.text:
                summary.samples.append(caption.text)
            done += 1
            if on_progress:
                on_progress(done, summary.considered)

        conn.commit()

    summary.elapsed = time.time() - started
    return summary
