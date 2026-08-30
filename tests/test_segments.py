from __future__ import annotations

from pathlib import Path

import pytest

from tsv.config import SegmentConfig
from tsv.motion.segments import activity_to_segments, merge_windows

CFG = SegmentConfig(
    open_frac=0.5,
    close_frac=0.2,
    merge_gap_seconds=1.0,
    min_duration_seconds=0.5,
    pre_roll_seconds=0.5,
    post_roll_seconds=0.5,
)
PERIOD = 0.5


def _trace(scores: list[float]) -> tuple[list[float], list[float]]:
    return [i * PERIOD for i in range(len(scores))], scores


def test_merge_windows_joins_overlapping_and_near():
    assert merge_windows([(0, 1), (0.5, 2)], gap=0) == [(0, 2)]
    assert merge_windows([(0, 1), (1.5, 2)], gap=1.0) == [(0, 2)]
    assert merge_windows([(0, 1), (5, 6)], gap=1.0) == [(0, 1), (5, 6)]
    assert merge_windows([], gap=1.0) == []


def test_empty_trace_yields_nothing():
    assert activity_to_segments([], [], CFG, PERIOD) == []


def test_idle_trace_yields_nothing():
    times, scores = _trace([0.0] * 20)
    assert activity_to_segments(times, scores, CFG, PERIOD) == []


def test_single_burst_is_one_segment_with_roll():
    times, scores = _trace([0, 0, 0.9, 0.9, 0.9, 0, 0, 0])
    segments = activity_to_segments(times, scores, CFG, PERIOD, duration=4.0)
    assert len(segments) == 1
    # Burst spans samples at 1.0-2.0s, closing at 2.5s, then 0.5s of roll.
    assert segments[0].t_start == 0.5
    assert segments[0].t_end == 3.0


def test_hysteresis_keeps_a_wobbling_burst_whole():
    """A score dipping below `open` but above `close` must not split."""
    times, scores = _trace([0, 0.9, 0.3, 0.9, 0.3, 0.9, 0, 0])
    segments = activity_to_segments(times, scores, CFG, PERIOD)
    assert len(segments) == 1


def test_short_gap_is_bridged_but_long_gap_is_not():
    short = _trace([0.9, 0.9, 0, 0.9, 0.9])          # 0.5s gap  < 1.0s
    assert len(activity_to_segments(*short, CFG, PERIOD)) == 1

    long = _trace([0.9, 0, 0, 0, 0, 0, 0, 0.9, 0.9])  # 3s gap    > 1.0s
    assert len(activity_to_segments(*long, CFG, PERIOD)) == 2


def test_below_min_duration_is_discarded():
    tight = SegmentConfig(
        open_frac=0.5, close_frac=0.2, merge_gap_seconds=0.0,
        min_duration_seconds=2.0, pre_roll_seconds=0, post_roll_seconds=0,
    )
    times, scores = _trace([0, 0.9, 0, 0, 0, 0])  # a single 0.5s spike
    assert activity_to_segments(times, scores, tight, PERIOD) == []


def test_roll_is_clamped_to_the_recording():
    times, scores = _trace([0.9, 0.9])
    segments = activity_to_segments(times, scores, CFG, PERIOD, duration=1.0)
    assert segments[0].t_start == 0.0        # cannot roll before the file
    assert segments[0].t_end == 1.0          # nor past its end


def test_peak_offset_points_at_the_busiest_moment():
    times, scores = _trace([0, 0.6, 0.95, 0.6, 0])
    segment = activity_to_segments(times, scores, CFG, PERIOD)[0]
    assert segment.peak_offset == 1.0        # index 2 * 0.5s


def test_mismatched_inputs_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        activity_to_segments([0.0, 1.0], [0.5], CFG, PERIOD)


# ---------- when the motion gate has nothing to work with ----------

def test_a_recording_where_nothing_stands_out_is_still_indexed(tmp_path, day_clip):
    """"Nothing stood out" has two meanings and only one is "nothing happened".

    The gate assumes what a fixed camera gives it: activity is rare and stands
    out against a quiet baseline. A phone held in the hand breaks that - every
    P-frame costs about the same and there is no contrast to find. A real 34
    second clip peaked at 1.06x its own baseline against the 1.08 needed, so
    it produced no segments, no objects, and nothing to search at all. The
    same file analysed in full had a person in it at 0.81.
    """
    import dataclasses

    from tsv import db
    from tsv.config import DEFAULT
    from tsv.ingest import ingest_file

    cfg = dataclasses.replace(
        DEFAULT,
        data_dir=tmp_path,
        # A threshold nothing can clear, so the gate finds nothing at all.
        tier_a=dataclasses.replace(DEFAULT.tier_a, z_threshold=1e9, min_ratio=1e9),
    )
    conn = db.open_db(cfg.db_path)
    result = ingest_file(conn, Path(day_clip["path"]), cfg)

    assert result.status == "ingested"
    assert result.n_segments > 0, "the recording was silently discarded"
    assert result.active_seconds > 0
    assert "whole file" in result.note


def test_the_fallback_is_bounded_by_length(tmp_path, day_clip):
    """A static camera overnight genuinely has nothing in it, and decoding
    eight hours to confirm that is the cost this stage exists to avoid."""
    import dataclasses

    from tsv import db
    from tsv.config import DEFAULT
    from tsv.ingest import ingest_file

    cfg = dataclasses.replace(
        DEFAULT,
        data_dir=tmp_path,
        tier_a=dataclasses.replace(DEFAULT.tier_a, z_threshold=1e9, min_ratio=1e9),
        segments=dataclasses.replace(DEFAULT.segments, whole_file_under_seconds=1.0),
    )
    conn = db.open_db(cfg.db_path)
    result = ingest_file(conn, Path(day_clip["path"]), cfg)
    assert result.n_segments == 0, "a long empty recording was decoded anyway"


def test_the_fallback_is_cut_into_windows():
    """One segment spanning everything is technically indexed and useless:
    every search returns the same thing and "when" is answered with the
    length of the recording."""
    from tsv.ingest import _even_windows

    windows = _even_windows(34.6, 15.0)
    assert len(windows) == 2
    assert windows[0].t_start == 0.0
    assert windows[-1].t_end == pytest.approx(34.6)
    # Contiguous, with no gap or overlap between them.
    for earlier, later in zip(windows, windows[1:]):
        assert earlier.t_end == pytest.approx(later.t_start)
    # Each one points somewhere inside itself.
    for window in windows:
        assert window.t_start <= window.peak_offset <= window.t_end


def test_a_short_recording_stays_one_window():
    from tsv.ingest import _even_windows

    assert len(_even_windows(8.0, 15.0)) == 1


def test_no_window_is_left_as_a_sliver():
    """A one second segment at the end is something nobody wants to see in a
    result list."""
    from tsv.ingest import _even_windows

    for duration in (16.0, 31.0, 46.0, 100.0):
        windows = _even_windows(duration, 15.0)
        assert min(w.t_end - w.t_start for w in windows) > 5.0, duration
