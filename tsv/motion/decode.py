"""Tier B: decode the candidate windows and score them properly.

Only the windows Tier A flagged reach this stage, and even those are decoded
at a few frames a second and a few hundred pixels wide. That is what keeps a
night of 1080p footage tractable on an integrated GPU.

The two things that generate false activity on real cameras are handled here:
IR day/night switching (a whole-frame luminance step that looks like enormous
motion) and sensor noise (single-pixel speckle, removed by an opening and a
minimum blob area).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from tsv.config import TierBConfig
from tsv.frames import sample_windows


@dataclass
class ActivityTrace:
    times: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    ir_flips: list[float] = field(default_factory=list)
    frames_decoded: int = 0
    frames_scored: int = 0

    @property
    def sample_period(self) -> float:
        if len(self.times) < 2:
            return 0.0
        deltas = np.diff(np.asarray(self.times))
        return float(np.median(deltas)) if deltas.size else 0.0


class _Scorer:
    """MOG2 background subtraction with noise and IR-flip suppression."""

    def __init__(self, cfg: TierBConfig, total_pixels: int) -> None:
        self.cfg = cfg
        self.total_pixels = total_pixels
        self.min_blob_px = max(1, int(cfg.min_blob_area_frac * total_pixels))
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel)
        )
        self._reset()

    def _reset(self) -> None:
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=self.cfg.mog2_history,
            varThreshold=self.cfg.mog2_var_threshold,
            detectShadows=True,
        )
        self.warmup_left = self.cfg.warmup_frames
        self.cooldown_left = 0
        self.prev_luma: float | None = None

    def start_window(self) -> None:
        self._reset()

    def score(self, gray: np.ndarray) -> tuple[float, bool]:
        """Return (foreground fraction, is_ir_flip)."""
        luma = float(gray.mean())
        flipped = (
            self.prev_luma is not None
            and abs(luma - self.prev_luma) > self.cfg.ir_flip_luma_delta
        )
        self.prev_luma = luma

        if flipped:
            # The entire frame changed at once. Rebuild the model rather than
            # let it treat the new illumination as a moving object.
            self._reset()
            self.prev_luma = luma
            self.cooldown_left = self.cfg.ir_flip_cooldown_frames
            self.bg.apply(gray)
            return 0.0, True

        mask = self.bg.apply(gray)

        if self.warmup_left > 0:
            self.warmup_left -= 1
            return 0.0, False
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return 0.0, False

        # 255 is foreground, 127 is MOG2's shadow class - shadows are not
        # motion and counting them doubles the apparent size of every person.
        fg = (mask == 255).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        area = sum(
            int(stats[i, cv2.CC_STAT_AREA])
            for i in range(1, n_labels)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_px
        )
        return area / self.total_pixels, False


def trace_windows(
    path: Path,
    windows: list[tuple[float, float]],
    cfg: TierBConfig,
) -> ActivityTrace:
    trace = ActivityTrace()
    if not windows:
        return trace

    scorer: _Scorer | None = None
    for sample in sample_windows(
        path, windows, cfg.sample_fps, width=cfg.width, pixel_format="gray"
    ):
        gray = sample.frame
        if scorer is None:
            scorer = _Scorer(cfg, gray.shape[0] * gray.shape[1])
        if sample.window_start:
            # Windows can be hours apart, so each starts from a fresh
            # background model rather than one built on a scene that no
            # longer exists.
            scorer.start_window()

        score, flipped = scorer.score(gray)
        if flipped:
            trace.ir_flips.append(sample.t)
        trace.times.append(sample.t)
        trace.scores.append(score)
        trace.frames_scored += 1
    trace.frames_decoded = trace.frames_scored

    order = np.argsort(np.asarray(trace.times), kind="stable")
    trace.times = [trace.times[i] for i in order]
    trace.scores = [trace.scores[i] for i in order]
    return trace
