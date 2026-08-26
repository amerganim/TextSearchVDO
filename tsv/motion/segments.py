"""Turn a per-sample activity trace into segments a person would agree with.

Three things separate this from a plain threshold:
  * hysteresis, so a score hovering at the threshold doesn't shred one event
    into twenty;
  * gap merging, so someone who stands still for two seconds mid-frame stays
    a single event;
  * pre/post roll, so playback starts before the interesting thing happens.
"""

from __future__ import annotations

from dataclasses import dataclass

from tsv.config import SegmentConfig


@dataclass
class Segment:
    t_start: float
    t_end: float
    activity_score: float
    peak_offset: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def merge_windows(
    windows: list[tuple[float, float]], gap: float = 0.0
) -> list[tuple[float, float]]:
    """Merge overlapping windows, and those separated by less than `gap`."""
    if not windows:
        return []
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def activity_to_segments(
    times: list[float],
    scores: list[float],
    cfg: SegmentConfig,
    sample_period: float,
    duration: float | None = None,
) -> list[Segment]:
    """Collapse an activity trace into segments.

    `times`/`scores` are parallel and must be sorted by time. Gaps in `times`
    (Tier A skipped a stretch) are treated as silence rather than interpolated.
    """
    if not times:
        return []
    if len(times) != len(scores):
        raise ValueError("times and scores must be the same length")

    runs: list[tuple[float, float]] = []
    start: float | None = None
    last_active: float = 0.0

    for t, score in zip(times, scores):
        if start is None:
            if score >= cfg.open_frac:
                start, last_active = t, t
        else:
            if score >= cfg.close_frac:
                last_active = t
            else:
                runs.append((start, last_active + sample_period))
                start = None

    if start is not None:
        runs.append((start, last_active + sample_period))

    runs = merge_windows(runs, gap=cfg.merge_gap_seconds)
    runs = [(a, b) for a, b in runs if b - a >= cfg.min_duration_seconds]

    # Score each run from the samples it contains, before roll is added, so
    # padding never dilutes the score used for ranking.
    scored: list[Segment] = []
    for a, b in runs:
        inside = [(t, s) for t, s in zip(times, scores) if a <= t < b]
        if not inside:
            continue
        peak_t = max(inside, key=lambda ts: ts[1])[0]
        mean = sum(s for _, s in inside) / len(inside)
        scored.append(Segment(t_start=a, t_end=b, activity_score=mean, peak_offset=peak_t))

    # Apply roll, then re-merge: padding can make neighbours overlap.
    padded = [
        (max(0.0, s.t_start - cfg.pre_roll_seconds), s.t_end + cfg.post_roll_seconds)
        for s in scored
    ]
    if duration is not None:
        padded = [(a, min(b, duration)) for a, b in padded]
    merged = merge_windows(padded, gap=0.0)

    # Re-attach scores to the (possibly fused) padded windows.
    out: list[Segment] = []
    for a, b in merged:
        members = [s for s in scored if s.t_start < b and s.t_end > a]
        if not members:
            continue
        best = max(members, key=lambda s: s.activity_score)
        out.append(
            Segment(
                t_start=a,
                t_end=b,
                activity_score=max(s.activity_score for s in members),
                peak_offset=best.peak_offset,
            )
        )
    return out
