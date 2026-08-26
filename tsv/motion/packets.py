"""Tier A: find candidate windows from compressed packet sizes, no decoding.

In inter-frame codecs a P-frame encodes only what changed, so its packet size
tracks motion closely. Demuxing is bounded by disk speed rather than CPU,
which is what makes a whole night of footage cheap to triage on a laptop with
no GPU.

Two encoder artefacts have to be removed first, or they swamp the motion
signal:

  * keyframes are large and periodic - excluded outright;
  * a P-frame's cost depends on how far it sits from the last keyframe (the
    frames just after a refresh have little residual to code). With a fixed
    GOP this puts a strong periodic sawtooth in the raw sizes. Each packet is
    therefore scored against the median size *for its own position in the
    GOP*, which flattens the artefact whatever the GOP length happens to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

from tsv.config import TierAConfig
from tsv.motion.segments import merge_windows

# Below this many samples a GOP position has no reliable median of its own.
_MIN_SAMPLES_PER_POSITION = 5
# GOP positions beyond this are lumped together; guards against a stream with
# no keyframes at all producing a million distinct positions.
_MAX_GOP_POSITION = 600


@dataclass
class PacketScan:
    duration: float
    bin_seconds: float
    bins: np.ndarray = field(repr=False)
    keyframe_ratio: float
    all_intra: bool
    baseline: float
    threshold: float
    candidates: list[tuple[float, float]]
    # Seconds that carried no inter-frame signal and were passed through to
    # Tier B without triage.
    blind_seconds: float = 0.0

    @property
    def candidate_seconds(self) -> float:
        return sum(b - a for a, b in self.candidates)


def _demux(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Return (times, sizes, gop_positions, keyframe_times, n_packets, last_t)."""
    times: list[float] = []
    sizes: list[int] = []
    positions: list[int] = []
    key_times: list[float] = []
    n_packets = 0
    last_t = 0.0
    gop_pos = 0

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        time_base = stream.time_base

        for packet in container.demux(stream):
            # The demuxer emits a final empty flush packet; it has no timing.
            if packet.pts is None or packet.size == 0 or time_base is None:
                continue
            t = float(packet.pts * time_base)
            last_t = max(last_t, t)
            n_packets += 1
            if packet.is_keyframe:
                key_times.append(t)
                gop_pos = 0
                continue
            gop_pos += 1
            times.append(t)
            sizes.append(packet.size)
            positions.append(min(gop_pos, _MAX_GOP_POSITION))

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(sizes, dtype=np.float64),
        np.asarray(positions, dtype=np.int32),
        np.asarray(key_times, dtype=np.float64),
        n_packets,
        last_t,
    )


def _normalise_by_gop_position(sizes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Divide each packet size by the median size at its GOP position."""
    if sizes.size == 0:
        return sizes
    global_median = float(np.median(sizes)) or 1.0
    expected = np.full(sizes.shape, global_median, dtype=np.float64)

    for pos in np.unique(positions):
        mask = positions == pos
        if int(mask.sum()) >= _MIN_SAMPLES_PER_POSITION:
            median = float(np.median(sizes[mask]))
            if median > 0:
                expected[mask] = median

    return sizes / expected


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average with edge replication, so ends are not dragged to zero."""
    if window <= 1 or values.size < window:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: values.size]


def _grow_regions(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Every run of `low` that contains at least one `high` bin, filled in."""
    out = np.zeros_like(high)
    n = high.size
    i = 0
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j]:
            j += 1
        if high[i:j].any():
            out[i:j] = True
        i = j
    return out


def _idle_level(values: np.ndarray, cfg: TierAConfig) -> tuple[float, float]:
    """Estimate the idle level and its spread from a set of bins.

    A low quantile, not the median. Motion only ever *increases* a P-frame's
    size, so the idle floor lives at the bottom of the distribution. The
    median is only a good estimate of it when activity is a small minority,
    and that assumption breaks exactly where it matters: short NVR files where
    someone is on screen for half the clip, which then set a baseline above
    their own idle level and hide the activity that raised it.

    The spread is measured against the lower half alone, for the same reason.
    """
    if values.size == 0:
        return 0.0, 1.0
    baseline = float(np.quantile(values, cfg.idle_quantile))
    lower = values[values <= np.median(values)]
    if lower.size == 0:
        lower = values
    mad = float(np.median(np.abs(lower - baseline)))
    spread = max(1.4826 * mad, cfg.min_rel_mad * baseline)
    return baseline, spread


def _local_baseline(
    bins: np.ndarray, valid: np.ndarray, cfg: TierAConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling median and MAD of the bin series.

    Computed on non-overlapping blocks and then interpolated, which is O(n)
    and smooth enough at this resolution; a true sliding median over a day of
    one-second bins costs more memory than the rest of the pipeline combined.
    """
    n = bins.size
    block = max(1, int(cfg.baseline_window_seconds / cfg.bin_seconds))
    # Always get at least three blocks, however short the clip. Plenty of NVRs
    # write one-minute files, and with a single block there is no local
    # structure to exploit and no neighbour to sanity-check against.
    block = min(block, max(1, n // 3))
    n_blocks = (n + block - 1) // block

    if n_blocks < 3:
        populated = bins[valid]
        level = _idle_level(populated, cfg)
        return np.full(n, level[0]), np.full(n, level[1])

    centres = np.empty(n_blocks)
    medians = np.empty(n_blocks)
    mads = np.empty(n_blocks)
    for i in range(n_blocks):
        lo, hi = i * block, min((i + 1) * block, n)
        centres[i] = (lo + hi - 1) / 2.0
        chunk = bins[lo:hi][valid[lo:hi]]
        if chunk.size == 0:
            medians[i] = np.nan
            mads[i] = np.nan
            continue
        medians[i], mads[i] = _idle_level(chunk, cfg)

    # Blocks with no data borrow from their neighbours.
    if np.isnan(medians).any():
        good = ~np.isnan(medians)
        if not good.any():
            return np.zeros(n), np.ones(n)
        medians = np.interp(np.arange(n_blocks), np.flatnonzero(good), medians[good])
        mads = np.interp(np.arange(n_blocks), np.flatnonzero(good), mads[good])

    # A block that happens to be mostly activity would raise its own baseline
    # and hide itself; a median-of-three across blocks pulls such a block back
    # towards its quieter neighbours.
    if n_blocks >= 3:
        stacked = np.vstack([
            np.concatenate(([medians[0]], medians[:-1])),
            medians,
            np.concatenate((medians[1:], [medians[-1]])),
        ])
        medians = np.median(stacked, axis=0)

    index = np.arange(n, dtype=np.float64)
    baseline = np.interp(index, centres, medians)
    spread = np.interp(index, centres, mads)
    return baseline, np.maximum(spread, cfg.min_rel_mad * baseline)


def scan(path: Path, cfg: TierAConfig, duration_hint: float = 0.0) -> PacketScan:
    times, sizes, positions, key_times, n_packets, last_t = _demux(path)

    duration = max(last_t, duration_hint)
    keyframe_ratio = (key_times.size / n_packets) if n_packets else 1.0
    all_intra = keyframe_ratio >= cfg.all_intra_keyframe_ratio
    n_bins = int(duration / cfg.bin_seconds) + 1

    def _whole_file(bins: np.ndarray) -> PacketScan:
        return PacketScan(
            duration=duration,
            bin_seconds=cfg.bin_seconds,
            bins=bins,
            keyframe_ratio=keyframe_ratio,
            all_intra=all_intra,
            baseline=0.0,
            threshold=0.0,
            candidates=[(0.0, duration)] if duration > 0 else [],
        )

    # All-intra streams (MJPEG, GOP=1) carry no inter-frame signal, and a clip
    # too short to establish a baseline can't be triaged safely. In both cases
    # hand the whole file to Tier B rather than guess.
    if all_intra or n_bins < 10 or sizes.size == 0:
        return _whole_file(np.zeros(max(n_bins, 1), dtype=np.float64))

    relative = _normalise_by_gop_position(sizes, positions)

    # Mean normalised residual per bin; ~1.0 for an idle scene by construction.
    indices = np.clip((times / cfg.bin_seconds).astype(np.int64), 0, n_bins - 1)
    totals = np.bincount(indices, weights=relative, minlength=n_bins)
    counts = np.bincount(indices, minlength=n_bins)
    bins = np.divide(totals, counts, out=np.zeros(n_bins), where=counts > 0)

    # Whether a stretch is all-intra is a *local* property, not a property of
    # the file. Encoders fall back to emitting nothing but keyframes when a
    # scene gets noisy enough to trip scene-cut detection on every frame -
    # heavy IR noise and rain both do it - and NVR exports routinely splice
    # together segments encoded with different settings. A bin with no
    # P-frames carries no motion signal, so scoring it would silently read as
    # "idle" and lose whatever happened there.
    key_indices = np.clip((key_times / cfg.bin_seconds).astype(np.int64), 0, n_bins - 1)
    key_counts = np.bincount(key_indices, minlength=n_bins)
    totals_per_bin = key_counts + counts
    blind = (totals_per_bin > 0) & (
        key_counts >= cfg.all_intra_keyframe_ratio * np.maximum(totals_per_bin, 1)
    )
    scorable = (counts > 0) & ~blind

    if int(scorable.sum()) < 10:
        return _whole_file(bins)

    baseline_curve, spread_curve = _local_baseline(bins, scorable, cfg)
    threshold_curve = np.maximum(
        baseline_curve + cfg.z_threshold * spread_curve,
        baseline_curve * cfg.min_ratio,
    )
    baseline = float(np.median(baseline_curve))
    threshold = float(np.median(threshold_curve))

    # Hysteresis, for the same reason Tier B has it: activity ramps in and out
    # rather than starting at full strength, so thresholding on one level
    # alone clips the quiet head and tail off every burst. Bins over the high
    # threshold seed a region, which then grows through neighbours over a
    # lower one.
    # Seeds come from the raw series so a genuine spike is not averaged away,
    # while the region grows over the smoothed one so a burst arriving as an
    # alternating series stays a single stretch. Blind and empty bins are held
    # at baseline so smoothing cannot drag their neighbours down.
    smoothed = _smooth(np.where(scorable, bins, baseline_curve), cfg.smooth_bins)
    high = (bins > threshold_curve) & scorable
    low = (smoothed > baseline_curve + 0.5 * cfg.z_threshold * spread_curve) & scorable
    grown = _grow_regions(high, low)

    # Blind bins are handed to Tier B unconditionally rather than guessed at.
    hot = np.flatnonzero(grown | blind)
    windows = [
        (
            max(0.0, i * cfg.bin_seconds - cfg.pad_seconds),
            min(duration, (i + 1) * cfg.bin_seconds + cfg.pad_seconds),
        )
        for i in hot.tolist()
    ]

    return PacketScan(
        duration=duration,
        bin_seconds=cfg.bin_seconds,
        bins=bins,
        keyframe_ratio=keyframe_ratio,
        all_intra=all_intra,
        baseline=baseline,
        threshold=threshold,
        candidates=merge_windows(windows, gap=0.0),
        blind_seconds=float(blind.sum()) * cfg.bin_seconds,
    )
