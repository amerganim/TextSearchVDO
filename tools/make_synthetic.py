"""Generate CCTV-like clips with known ground truth.

Real footage is the only thing that will tune the thresholds, but synthetic
clips with known activity windows are what let the pipeline be tested for
*correctness* in CI, on a machine with no cameras attached.

The scene deliberately includes sensor noise and a textured background: on a
perfectly clean synthetic scene x264 compresses idle frames to almost nothing
and the Tier A prefilter looks far better than it will on real footage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np


def _background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """A textured static scene: gradient walls, a floor, a few fixed objects."""
    ys, xs = np.mgrid[0:height, 0:width]
    base = (60 + 40 * (ys / height) + 25 * np.sin(xs / 55.0)).astype(np.float32)
    scene = np.repeat(base[:, :, None], 3, axis=2)
    scene[:, :, 0] *= 0.92
    scene[:, :, 2] *= 1.08
    # Fixed furniture, so the encoder has real detail to hold onto.
    scene[int(height * 0.62) :, :, :] *= 0.75
    scene[int(height * 0.30) : int(height * 0.55), int(width * 0.08) : int(width * 0.26)] = 110
    scene[int(height * 0.40) : int(height * 0.70), int(width * 0.70) : int(width * 0.94)] = 85
    # Static high-frequency texture (carpet/brick), fixed across all frames.
    scene += rng.normal(0, 6, size=scene.shape).astype(np.float32)
    return np.clip(scene, 0, 255)


def build(
    out_path: Path,
    duration: float,
    activity: list[tuple[float, float]],
    ir_flip_at: float | None,
    fps: int,
    width: int,
    height: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    scene = _background(width, height, rng)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        # GOP and CRF chosen to look like a mid-range IP camera.
        stream.options = {"crf": "23", "g": str(fps * 2), "preset": "veryfast", "tune": "zerolatency"}

        n_frames = int(duration * fps)
        for i in range(n_frames):
            t = i / fps
            frame = scene.copy()

            night = ir_flip_at is not None and t >= ir_flip_at
            if night:
                # IR mode: monochrome, darker, and much noisier - the exact
                # combination that fools naive frame differencing.
                grey = frame.mean(axis=2, keepdims=True)
                frame = np.repeat(grey, 3, axis=2) * 0.55
                frame += rng.normal(0, 9, size=frame.shape).astype(np.float32)

            # Per-frame sensor noise.
            frame += rng.normal(0, 2.5, size=frame.shape).astype(np.float32)

            for start, end in activity:
                if start <= t < end:
                    progress = (t - start) / max(end - start, 1e-6)
                    cx = int(width * (0.1 + 0.8 * progress))
                    cy = int(height * (0.45 + 0.06 * np.sin(progress * 9)))
                    hw, hh = int(width * 0.045), int(height * 0.20)
                    x0, x1 = max(0, cx - hw), min(width, cx + hw)
                    y0, y1 = max(0, cy - hh), min(height, cy + hh)
                    frame[y0:y1, x0:x1] = 40 if night else 215

            av_frame = av.VideoFrame.from_ndarray(
                np.clip(frame, 0, 255).astype(np.uint8), format="rgb24"
            )
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

    truth = {
        "path": str(out_path),
        "duration": duration,
        "fps": fps,
        "activity": [list(w) for w in activity],
        "ir_flip_at": ir_flip_at,
    }
    out_path.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("footage/ch01_20260826120000.mp4"))
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument(
        "--activity",
        default="18-26,55-61,95-104",
        help="comma-separated start-end windows in seconds",
    )
    ap.add_argument("--ir-flip-at", type=float, default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    windows = []
    for chunk in filter(None, args.activity.split(",")):
        start, _, end = chunk.partition("-")
        windows.append((float(start), float(end)))

    truth = build(
        args.out, args.duration, windows, args.ir_flip_at,
        args.fps, args.width, args.height, args.seed,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB), activity={truth['activity']}")


if __name__ == "__main__":
    main()
