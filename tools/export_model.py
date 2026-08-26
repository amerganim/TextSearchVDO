"""Export a YOLO checkpoint to ONNX.

Run this from the throwaway export environment, not the runtime one:

    py -3.14 -m venv .venv-export
    .venv-export/Scripts/python -m pip install ultralytics onnx
    .venv-export/Scripts/python tools/export_model.py --out data/models

ultralytics and torch are export-time tools only. Keeping them out of the
runtime environment is deliberate: at inference TextSearchVDO is an ONNX graph
plus numpy, which is what makes it small enough to ship to someone who just
wants to search their own cameras.

Two export flags matter and are not the defaults everywhere:

  nms=False     - the graph must emit raw predictions. tsv.models.detect does
                  its own class-aware NMS, and a graph with NMS baked in has a
                  completely different output layout.
  dynamic=False - a fixed input shape. OpenVINO compiles static shapes far
                  better, and the pipeline always feeds exactly one letterboxed
                  frame at a fixed size.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n.pt", help="ultralytics checkpoint name or path")
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from ultralytics import YOLO

    print(f"loading {args.model} (downloads the checkpoint on first run)")
    model = YOLO(args.model)

    print(f"exporting to ONNX at {args.imgsz}px, opset {args.opset}")
    produced = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=False,
            nms=False,
            simplify=True,
            verbose=False,
        )
    )

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / produced.name
    shutil.move(str(produced), target)

    size_mb = target.stat().st_size / 1e6
    print(f"\nwrote {target} ({size_mb:.1f} MB)")
    print(f"classes: {len(model.names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
