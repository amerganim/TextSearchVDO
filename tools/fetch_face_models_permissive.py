"""Download YuNet and SFace: the face stack that can be shipped.

    .venv/Scripts/python tools/fetch_face_models_permissive.py --out data/models

Like the detector, this is a download rather than an export - no torch, no
throwaway environment. Both come from OpenCV's model zoo, mirrored on Hugging
Face because the GitHub copies are Git LFS pointers rather than files.

    YuNet   face detection    MIT
    SFace   face recognition  Apache-2.0

That is the point of them. The InsightFace `buffalo` packs this replaces are
published for non-commercial research, which makes them the last licence
blocker before this can be sold.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/opencv"

MODELS: tuple[tuple[str, str, str, int], ...] = (
    (
        "face_detection_yunet_2023mar.onnx",
        f"{HF}/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "YuNet face detection (MIT)",
        1,
    ),
    (
        "face_recognition_sface_2021dec.onnx",
        f"{HF}/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "SFace face recognition (Apache-2.0)",
        37,
    ),
)


def download(url: str, target: Path) -> None:
    """Fetch to a temporary name, then rename.

    A half-written file with the final name looks installed and fails later as
    a corrupt graph, blaming ONNX for an interrupted download.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with partial.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:5.1%}  {done / 1e6:5.1f} MB",
                          end="", flush=True)
    if total:
        print()
    partial.replace(target)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for filename, url, title, approx_mb in MODELS:
        target = args.out / filename
        if target.is_file() and not args.force:
            print(f"{filename} already present")
            continue
        print(f"fetching {title}, ~{approx_mb} MB")
        try:
            download(url, target)
        except Exception as exc:  # noqa: BLE001 - the message is the output
            print(f"could not fetch {url}: {type(exc).__name__}: {exc}")
            return 1
        print(f"  {filename}  {target.stat().st_size / 1e6:.1f} MB")

    # Load them here so a broken download is reported now, not during an
    # import. OpenCV's own constructors are what the runtime uses.
    try:
        import cv2

        detector = args.out / MODELS[0][0]
        embedder = args.out / MODELS[1][0]
        cv2.FaceDetectorYN.create(str(detector), "", (320, 320))
        cv2.FaceRecognizerSF.create(str(embedder), "")
        print("\nboth load.")
    except Exception as exc:  # noqa: BLE001
        print(f"\ndownloaded but will not load: {exc}")
        return 1

    print("select them with:  python -m tsv setup --faces opencv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
