"""Fetch Florence-2 as ONNX, for captioning person crops.

    .venv-export/Scripts/python tools/fetch_caption_model.py --out data/models

Florence-2-base-ft is 230M parameters and built for exactly this job -
describing an image or a region on demand - which is why it is here rather
than a general 3-4B vision-language model. On a CPU-only machine the large
ones caption at a rate that makes a day of footage an overnight job at best.

Four graphs make up one caption:

    vision_encoder      pixels          -> image features
    embed_tokens        prompt ids      -> token embeddings
    encoder_model       both, joined    -> encoder hidden states
    decoder_model_merged                -> next-token logits, with KV cache

The int8 weights are the default. They are roughly a quarter the size of fp32
and faster on CPU, which is the machine this has to run on; pass --precision
fp32 where accuracy matters more than time.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = "onnx-community/Florence-2-base-ft"

# base is the default because it is what a CPU can keep up with. large is
# roughly five times the weights and names things base only gestures at - a
# real crop that base called "a large black object" is the reason somebody
# would want it. It goes in its own directory so both can be installed and
# switched between without re-downloading.
MODELS = {
    "base": ("onnx-community/Florence-2-base-ft", "florence2", 275),
    "large": ("onnx-community/Florence-2-large-ft", "florence2-large", 1540),
}

# Each graph, and the file suffix per precision. The decoder is "quantized"
# rather than "int8" in this repo; the others follow the usual naming.
GRAPHS = {
    "vision_encoder": {"int8": "vision_encoder_int8.onnx", "fp32": "vision_encoder.onnx"},
    "embed_tokens": {"int8": "embed_tokens_int8.onnx", "fp32": "embed_tokens.onnx"},
    "encoder_model": {"int8": "encoder_model_int8.onnx", "fp32": "encoder_model.onnx"},
    "decoder_model_merged": {
        "int8": "decoder_model_merged_quantized.onnx",
        "fp32": "decoder_model_merged.onnx",
    },
}

# Vocabulary and preprocessing settings. `tokenizer.json` is fetched too, but
# only so the vocabulary can be read; nothing at runtime imports a tokenizer
# library. Florence-2 prompts are fixed task strings, so encoding is a lookup
# and decoding is the reverse map.
SIDECARS = ("vocab.json", "tokenizer_config.json", "config.json", "preprocessor_config.json")

# Florence-2 is steered by task tokens, but the processor expands each one into
# an English sentence before tokenising - '<CAPTION>' is not a vocabulary entry.
# Those sentences are read from the model's own processing source rather than
# copied here, then tokenised once and written to prompts.json. The runtime
# then needs no tokenizer at all: prompting is a lookup and reading the answer
# back is the reverse vocabulary plus GPT-2 byte decoding.
WANTED_TASKS = ("<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>")


def _write_prompts(repo: str, target: Path) -> None:
    """Tokenise the task prompts once, here, so the runtime never has to."""
    import re

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    source_repo = repo.replace("onnx-community/", "microsoft/")
    processing = Path(hf_hub_download(source_repo, "processing_florence2.py"))
    src = processing.read_text(encoding="utf-8")

    block = re.search(r"task_prompts_without_inputs\s*=\s*\{(.*?)\n\s*\}", src, re.S)
    if block is None:
        raise RuntimeError("could not find the task prompts in processing_florence2.py")
    found = dict(re.findall(r"'(<[A-Z_]+>)'\s*:\s*'([^']*)'", block.group(1)))

    tokenizer = AutoTokenizer.from_pretrained(source_repo)
    prompts = {}
    for task in WANTED_TASKS:
        sentence = found.get(task)
        if sentence is None:
            continue
        prompts[task] = {
            "text": sentence,
            "ids": tokenizer(sentence, return_tensors=None)["input_ids"],
        }
        print(f'  {task:24} {len(prompts[task]["ids"]):3} tokens  {sentence!r}')

    (target / "prompts.json").write_text(json.dumps(prompts, indent=1), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    ap.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--model", default="base", choices=sorted(MODELS),
                    help="base is the CPU-affordable one; large describes more")
    ap.add_argument("--repo", default=None,
                    help="override the repository for --model")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    repo, folder, approx_mb = MODELS[args.model]
    if args.repo:
        repo = args.repo
    args.repo = repo
    target = args.out / folder
    print(f"{args.model}: about {approx_mb} MB")
    target.mkdir(parents=True, exist_ok=True)
    print(f"fetching {args.repo} ({args.precision})")

    total = 0.0
    for name, per_precision in GRAPHS.items():
        remote = f"onnx/{per_precision[args.precision]}"
        local = hf_hub_download(args.repo, remote)
        out = target / f"{name}.onnx"
        shutil.copy2(local, out)
        size = out.stat().st_size / 1e6
        total += size
        print(f"  {out.name:28} {size:7.1f} MB")

    for name in SIDECARS:
        try:
            local = hf_hub_download(args.repo, name)
        except Exception as exc:  # noqa: BLE001 - optional sidecars
            print(f"  {name:28} unavailable ({type(exc).__name__})")
            continue
        shutil.copy2(local, target / name)

    _write_prompts(args.repo, target)

    # A compact vocabulary the runtime can read without a tokenizer library.
    vocab_path = target / "vocab.json"
    if vocab_path.is_file():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        print(f"  vocabulary                  {len(vocab)} tokens")

    print(f"\n{total:.0f} MB into {target}")
    print("captioning is optional; nothing else needs these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
