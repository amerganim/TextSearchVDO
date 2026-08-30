"""Which way up a recording actually is.

A phone recorded two clips here and the app found almost nothing in either.
Not the motion gate this time, and not the models: the frames come out of the
file lying on their side. There is no rotation to read - no `rotate` tag, no
display matrix, nothing ffmpeg would apply on its way past - so the app was
faithfully analysing a sideways room.

Detectors do not cope with that. Everything they were trained on stands up.
Measured on those two clips, twelve frames each, the same detector at each of
the four right angles:

    video 8, as stored       9 detections, best 0.60
    video 8, rotated 90 cw  14 detections, best 0.81, and finds a bed
    video 9, as stored       8 detections, best 0.72
    video 9, rotated 90 cw  17 detections, best 0.94

The bed is the whole point: the question that failed was "anyone sleeping?",
and the bed only exists once the frame is the right way up.

**So it is measured rather than read.** Try the four right angles on a handful
of frames, keep the one the detector is most confident about. Rotating a
picture does not create objects in it, so the orientation that yields the most
confident detections is upright - and the cost is four passes over about six
frames, once per recording.

**Only when it is clearly better.** A recording that genuinely is the right way
up must not be turned by noise, so an alternative has to beat what is already
there by a clear margin before it is believed. Ambiguous stays as it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# The four ways a camera can be held. Keys are degrees clockwise, which is
# what has to be undone to make the picture upright.
ROTATIONS: dict[int, int | None] = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# Frames to look at. Enough that one odd moment cannot decide it, few enough
# that this stays a rounding error against analysing the recording.
SAMPLE_FRAMES = 6

# How much better an alternative must be before it is believed. Rotating a
# picture that was already upright is far worse than leaving a sideways one
# alone: it breaks a recording that worked.
MARGIN = 1.35


@dataclass
class Orientation:
    degrees: int
    score: float
    baseline: float
    detections: int

    @property
    def turned(self) -> bool:
        return self.degrees != 0

    def describe(self) -> str:
        if not self.turned:
            return "upright"
        return (
            f"stored {self.degrees} degrees off upright "
            f"({self.score:.1f} against {self.baseline:.1f} as stored)"
        )


def apply(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Turn a frame upright, given how far off it was stored."""
    code = ROTATIONS.get(int(degrees) % 360)
    if code is None:
        return frame
    return np.ascontiguousarray(cv2.rotate(frame, code))


def _score(detector, frame: np.ndarray) -> tuple[float, int]:
    """Total confidence the detector places in a frame, and how many objects.

    Summed rather than maximised: one lucky box at a bad angle should not
    outweigh a whole scene resolving at a good one.
    """
    found = detector.detect(frame)
    return sum(d.score for d in found), len(found)


def detect(
    path: Path,
    detector,
    duration: float,
    frames: int = SAMPLE_FRAMES,
    margin: float = MARGIN,
) -> Orientation:
    """How far off upright this recording is stored, in degrees clockwise."""
    from tsv.frames import sample_windows

    samples = [
        sample.frame
        for sample in sample_windows(
            path, [(0.0, max(duration, 1.0))],
            fps=max(frames / max(duration, 1.0), 0.05), width=960,
        )
    ][:frames]
    if not samples:
        return Orientation(0, 0.0, 0.0, 0)

    scores: dict[int, tuple[float, int]] = {}
    for degrees in ROTATIONS:
        total, count = 0.0, 0
        for frame in samples:
            one, many = _score(detector, apply(frame, degrees))
            total += one
            count += many
        scores[degrees] = (total, count)

    baseline = scores[0][0]
    best_degrees, (best_score, best_count) = max(
        scores.items(), key=lambda kv: kv[1][0]
    )

    # Believed only on a clear margin. `baseline or 1e-9` so a recording where
    # nothing at all is found as stored can still be turned - that is exactly
    # the case this exists for.
    if best_degrees != 0 and best_score < baseline * margin:
        best_degrees, best_score, best_count = 0, baseline, scores[0][1]

    return Orientation(best_degrees, best_score, baseline, best_count)
