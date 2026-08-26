"""Tunables for the Phase 0 ingest pipeline.

Every threshold here is a guess until it has been run against real footage.
They are grouped and named so they can be swept from a script later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TierAConfig:
    """Packet-size prefilter: finds candidate windows without decoding."""

    bin_seconds: float = 1.0
    # Candidates are picked by robust z-score (median/MAD) of mean P-frame
    # size. Tier B re-checks everything this stage passes, so a false positive
    # only costs decode time whereas a miss is unrecoverable - hence a
    # deliberately generous threshold.
    z_threshold: float = 3.0
    # Floor on the MAD, as a fraction of baseline. Without it a very stable
    # stream has MAD ~= 0 and every bin scores as infinitely anomalous.
    min_rel_mad: float = 0.03
    # A bin must also be this much above baseline, so trivial fluctuation on a
    # near-static stream cannot pass on z-score alone.
    min_ratio: float = 1.08
    # The baseline is measured over a rolling window rather than the whole
    # file. A camera's noise floor changes hugely across a recording - IR
    # cutover at dusk is the obvious one, but rain and auto-exposure do it too
    # - and a single global baseline lets the noisier half hide its own
    # activity behind a threshold set by the quieter half.
    baseline_window_seconds: float = 120.0
    # Bins are smoothed over this many neighbours before thresholding. The
    # motion premium on a P-frame is itself GOP-position dependent - frames
    # just after a keyframe stay cheap even while something is moving - so a
    # single burst arrives as an alternating series that breaks contiguity at
    # one-second resolution. Tier A only has to decide *whether* to decode a
    # stretch, so trading that resolution away costs nothing.
    smooth_bins: int = 3
    # Candidate windows are padded before being handed to Tier B, so segment
    # edges are found by the accurate stage rather than the cheap one.
    pad_seconds: float = 2.0
    # Above this fraction of keyframes the stream is effectively all-intra
    # (MJPEG, or a camera with GOP=1) and packet sizes carry no motion signal.
    all_intra_keyframe_ratio: float = 0.9


@dataclass(frozen=True)
class TierBConfig:
    """Decode-and-subtract stage: accurate scoring on candidate windows."""

    sample_fps: float = 3.0
    width: int = 320
    mog2_history: int = 120
    mog2_var_threshold: float = 24.0
    # Morphological opening kernel, kills single-pixel sensor noise.
    open_kernel: int = 3
    # Blobs smaller than this fraction of the frame are ignored entirely.
    min_blob_area_frac: float = 0.0008
    # A frame scores above this fraction of changed pixels to count as active.
    active_area_frac: float = 0.0025
    # Mean-luminance jump that indicates an IR day/night switch rather than
    # motion. On a 0-255 scale.
    ir_flip_luma_delta: float = 18.0
    # Frames to discard after an IR flip while MOG2 relearns the background.
    ir_flip_cooldown_frames: int = 8
    # Frames used to prime MOG2 before its output is trusted.
    warmup_frames: int = 10


@dataclass(frozen=True)
class SegmentConfig:
    """Turning a per-sample activity score into human-meaningful segments."""

    # Hysteresis: open a segment at `open_frac`, keep it alive until activity
    # drops below `close_frac`. Prevents flicker on borderline motion.
    open_frac: float = 0.0035
    close_frac: float = 0.0015
    # Bridge gaps shorter than this: someone pausing mid-frame is one event.
    merge_gap_seconds: float = 3.0
    # Discard anything shorter than this: usually a bird or a noise spike.
    min_duration_seconds: float = 1.0
    # Context around each segment so playback starts before the action.
    pre_roll_seconds: float = 1.5
    post_roll_seconds: float = 1.5


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    tier_a: TierAConfig = field(default_factory=TierAConfig)
    tier_b: TierBConfig = field(default_factory=TierBConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    thumb_width: int = 320
    thumb_quality: int = 80

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def thumb_dir(self) -> Path:
        return self.data_dir / "thumbs"


DEFAULT = Config()
