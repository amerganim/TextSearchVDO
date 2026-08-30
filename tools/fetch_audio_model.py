"""Download a Whisper model in ctranslate2 form, for transcription.

    .venv/Scripts/python tools/fetch_audio_model.py --out data/models

A download rather than an export, and no torch: faster-whisper runs on
ctranslate2, and the Systran conversions on Hugging Face are already in that
format. The runtime stays ONNX Runtime, numpy and ctranslate2.

Sizes, and what each buys on the kind of audio a camera records - which is to
say noisy, far from the microphone, and often 8 kHz:

    tiny    75 MB   fastest, and drops or invents words on poor audio
    base   145 MB   the default here: the smallest that is usually worth
                    trusting on a room recording
    small  484 MB   noticeably better on accents and distance, several times
                    the cost

All are MIT, as Whisper itself is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS: dict[str, tuple[str, int]] = {
    "tiny": ("Systran/faster-whisper-tiny", 75),
    "base": ("Systran/faster-whisper-base", 145),
    "small": ("Systran/faster-whisper-small", 484),
    # For languages base cannot hold. Measured here on Bengali, where
    # base transliterates and small fails outright - see tsv/catalogue.py.
    "medium": ("Systran/faster-whisper-medium", 1530),
}

# What ctranslate2 needs on disk. Fetching only these avoids pulling the
# repository's extras.
FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="base", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    repo, approx_mb = MODELS[args.model]
    target = args.out / f"whisper-{args.model}"

    if (target / "model.bin").is_file() and not args.force:
        print(f"{target} already present")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is needed to fetch this.")
        print("  .venv/Scripts/python -m pip install huggingface_hub")
        return 1

    print(f"fetching {repo} (~{approx_mb} MB)")
    target.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            allow_patterns=list(FILES),
        )
    except Exception as exc:  # noqa: BLE001 - the message is the whole output
        print(f"could not fetch {repo}: {type(exc).__name__}: {exc}")
        return 1

    missing = [name for name in ("model.bin", "config.json") if not (target / name).is_file()]
    if missing:
        print(f"downloaded but incomplete, missing: {', '.join(missing)}")
        return 1

    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"  {target}  {size / 1e6:.0f} MB")

    # Load it now so a broken download is reported here rather than in the
    # middle of somebody's first import.
    try:
        from faster_whisper import WhisperModel

        WhisperModel(str(target), device="cpu", compute_type="int8")
        print("  loads.")
    except ImportError:
        print("  (faster-whisper not installed here, skipping the load check)")
    except Exception as exc:  # noqa: BLE001
        print(f"  downloaded but will not load: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
