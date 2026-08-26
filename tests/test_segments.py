from __future__ import annotations

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
