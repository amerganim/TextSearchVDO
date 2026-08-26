"""Fetch the SCRFD and ArcFace ONNX graphs from the InsightFace model zoo.

Run from the throwaway export environment:

    .venv-export/Scripts/python -m pip install insightface
    .venv-export/Scripts/python tools/fetch_face_models.py --out data/models

insightface is used only to download and unpack; nothing imports it at
runtime. The files it fetches are plain ONNX, and `tsv.models.face` does its
own pre- and post-processing against them.

`buffalo_s` is the default rather than `buffalo_l`: the small pack is about
16 MB against roughly 180 MB, and the baseline machine has no discrete GPU.
Pass --pack buffalo_l on hardware that can afford the accuracy.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# What each pack calls its detector and recogniser.
PACKS = {
    "buffalo_s": ("det_500m.onnx", "w600k_mbf.onnx"),
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", choices=sorted(PACKS), default="buffalo_s")
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    args = ap.parse_args()

    from insightface.utils import storage

    print(f"fetching {args.pack} (downloads on first run)")
    directory = Path(storage.ensure_available("models", args.pack))
    print(f"unpacked to {directory}")

    args.out.mkdir(parents=True, exist_ok=True)
    wanted = PACKS[args.pack]
    copied = []

    for name in wanted:
        source = directory / name
        if not source.is_file():
            available = sorted(p.name for p in directory.glob("*.onnx"))
            print(f"\n{name} is not in the pack. It contains: {available}")
            return 1
        target = args.out / name
        shutil.copy2(source, target)
        copied.append(target)
        print(f"  {target}  ({target.stat().st_size / 1e6:.1f} MB)")

    print(f"\ncopied {len(copied)} model(s) into {args.out}")
    print("detector: ", wanted[0])
    print("recogniser:", wanted[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
