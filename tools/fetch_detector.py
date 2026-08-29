"""Download a YOLOX detector, already exported to ONNX.

    .venv/Scripts/python tools/fetch_detector.py --out data/models

Unlike every other model this project uses, this one needs no export step and
no torch: YOLOX publishes ONNX graphs in its own GitHub releases, so this is a
download and a checksum. That is the whole point of preferring it - the
detector was the reason `export_model.py` had to exist, and with YOLOX the
throwaway export environment is no longer needed for it at all.

The other reason is licensing. YOLOX is Apache-2.0. Ultralytics YOLO11 is
AGPL-3.0, whose obligations extend to any application distributed with it,
which makes it the single biggest obstacle to shipping this as a product.
`tsv.catalogue` records both.

Sizes are the model's own: YOLOX-tiny runs at 416px and YOLOX-s at 640, and
the graphs are static, so the runtime reads the size from the file rather
than being told.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

# key -> (filename, url, input size, approximate MB)
MODELS: dict[str, tuple[str, str, int, int]] = {
    "yolox-nano": ("yolox_nano.onnx", f"{RELEASE}/yolox_nano.onnx", 416, 4),
    "yolox-tiny": ("yolox_tiny.onnx", f"{RELEASE}/yolox_tiny.onnx", 416, 20),
    "yolox-s": ("yolox_s.onnx", f"{RELEASE}/yolox_s.onnx", 640, 35),
    "yolox-m": ("yolox_m.onnx", f"{RELEASE}/yolox_m.onnx", 640, 97),
}


def download(url: str, target: Path) -> None:
    """Fetch to a temporary name, then rename.

    A half-written file with the final name is worse than no file: it looks
    installed, loads as a corrupt graph, and the error names ONNX rather than
    the interrupted download that caused it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        digest = hashlib.blake2b(digest_size=8)
        with partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:5.1%}  {done / 1e6:6.1f} MB",
                          end="", flush=True)
    print()
    partial.replace(target)
    print(f"  {target.name}  {target.stat().st_size / 1e6:.1f} MB  "
          f"blake2b:{digest.hexdigest()}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="yolox-tiny", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    ap.add_argument("--force", action="store_true", help="re-download if present")
    args = ap.parse_args()

    filename, url, size, approx_mb = MODELS[args.model]
    target = args.out / filename

    if target.is_file() and not args.force:
        print(f"{target} already present ({target.stat().st_size / 1e6:.1f} MB)")
        return 0

    print(f"fetching {args.model} (~{approx_mb} MB, {size}px input, Apache-2.0)")
    try:
        download(url, target)
    except Exception as exc:  # noqa: BLE001 - the message is the whole output
        print(f"could not fetch {url}: {type(exc).__name__}: {exc}")
        return 1

    # Loading it here means a broken download is reported now rather than in
    # the middle of somebody's first import.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        out_shape = session.get_outputs()[0].shape
        print(f"  loads: input {shape}, output {out_shape}")
        if len(out_shape) == 3 and out_shape[2] != 85:
            print(f"  WARNING: expected 85 values per anchor, got {out_shape[2]}")
            return 1
    except ImportError:
        print("  (onnxruntime not importable here, skipping the load check)")
    except Exception as exc:  # noqa: BLE001
        print(f"  downloaded but will not load: {exc}")
        return 1

    print(f"\nset the detector to {filename} to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
