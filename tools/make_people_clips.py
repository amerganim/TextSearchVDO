"""Build clips containing two real, different people walking across frame.

The synthetic generator's subject is a rectangle, which no face detector will
ever look at. To prove identity end to end we need footage with actual faces,
so this cuts two real people out of the sample photographs and composites them
onto a moving background.
"""

from __future__ import annotations

import sys
from pathlib import Path

import av
import cv2
import numpy as np

ASSETS = Path(".venv-export/Lib/site-packages/ultralytics/assets")
W, H, FPS = 960, 540, 12


def person_cutout(image_path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    x1, y1, x2, y2 = box
    return cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def background(rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.mgrid[0:H, 0:W]
    scene = (70 + 45 * (ys / H) + 20 * np.sin(xs / 40.0)).astype(np.float32)
    scene = np.repeat(scene[:, :, None], 3, axis=2)
    scene[int(H * 0.7):, :, :] *= 0.7
    scene += rng.normal(0, 5, scene.shape).astype(np.float32)
    return np.clip(scene, 0, 255)


def build(out_path: Path, person: np.ndarray, duration: float, windows: list[tuple[float, float]],
          seed: int, target_h: int = 380) -> None:
    rng = np.random.default_rng(seed)
    scene = background(rng)

    scale = target_h / person.shape[0]
    sprite = cv2.resize(person, (int(person.shape[1] * scale), target_h),
                        interpolation=cv2.INTER_AREA)
    sh, sw = sprite.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(out_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = {"crf": "20", "g": str(FPS * 2), "preset": "veryfast",
                          "tune": "zerolatency"}

        for i in range(int(duration * FPS)):
            t = i / FPS
            # Light per-frame sensor noise. Real cameras denoise before the
            # encoder sees the signal; heavy full-frame noise swamps the
            # packet-size motion signal Tier A depends on.
            frame = scene.copy() + rng.normal(0, 0.5, scene.shape).astype(np.float32)

            for start, end in windows:
                if start <= t < end:
                    progress = (t - start) / max(end - start, 1e-6)
                    x = int((W - sw) * (0.05 + 0.9 * progress))
                    y = int(H * 0.95 - sh)
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(W, x + sw), min(H, y + sh)
                    frame[y0:y1, x0:x1] = sprite[: y1 - y0, : x1 - x0]

            av_frame = av.VideoFrame.from_ndarray(
                np.clip(frame, 0, 255).astype(np.uint8), format="rgb24"
            )
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) windows={windows}")


def main() -> int:
    if not ASSETS.is_dir():
        print(f"sample photographs not found at {ASSETS}")
        return 1

    # Two different real people, from the two sample photographs.
    person_a = person_cutout(ASSETS / "zidane.jpg", (749, 41, 1149, 711))
    person_b = person_cutout(ASSETS / "bus.jpg", (49, 398, 243, 904))

    out = Path("footage-people")
    build(out / "ch10_20260401090000.mp4", person_a, 26.0, [(4.0, 10.0), (16.0, 22.0)], seed=1)
    build(out / "ch10_20260401100000.mp4", person_b, 16.0, [(4.0, 11.0)], seed=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
