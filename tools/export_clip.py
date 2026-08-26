"""Export CLIP's image and text encoders to ONNX.

Run from the throwaway export environment:

    .venv-export/Scripts/python -m pip install transformers
    .venv-export/Scripts/python tools/export_clip.py --out data/models

Alongside the two graphs it writes:

  clip_merges.txt      the BPE merge table, so the runtime tokenizer can be
                       built without transformers
  clip_vocab.json      the reference vocabulary, used only by the tests to
                       prove our constructed one is identical
  clip_golden.json     (text, token ids) pairs from the reference tokenizer,
                       so the pure-Python implementation can be checked
                       against it exactly

The two encoders are exported separately because they are used at completely
different rates: images are embedded once at ingest, text once per query.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROBE_TEXTS = [
    "a person walking",
    "a man in a red jacket",
    "two people talking near a doorway",
    "a dog running across the garden",
    "someone carrying a cardboard box",
    "a car parked in the driveway at night",
    "PERSON  in   a   HAT",
    "a person's bicycle, left outside",
    "don't walk 3 dogs",
    "café naïve résumé",
    "a" * 400,
    "",
    "   ",
    "100 200 300",
    "<|startoftext|> already tokenised <|endoftext|>",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--out", type=Path, default=Path("data/models"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    from transformers import CLIPModel, CLIPTokenizer

    print(f"loading {args.model} (downloads on first run)")
    model = CLIPModel.from_pretrained(args.model).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)

    image_size = model.config.vision_config.image_size
    context = model.config.text_config.max_position_embeddings

    # Project explicitly rather than calling get_*_features. Those return a
    # tuple in transformers 5.x, which torch.onnx.export flattens into several
    # graph outputs - the first being the raw hidden states, not the embedding
    # anyone wants. Doing the two steps by hand yields exactly one output.
    class ImageEncoder(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, pixel_values):
            pooled = self.clip.vision_model(pixel_values=pixel_values)[1]
            return self.clip.visual_projection(pooled)

    class TextEncoder(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, input_ids):
            pooled = self.clip.text_model(input_ids=input_ids)[1]
            return self.clip.text_projection(pooled)

    with torch.no_grad():
        image_path = args.out / "clip_image.onnx"
        torch.onnx.export(
            ImageEncoder(model),
            torch.zeros(1, 3, image_size, image_size),
            str(image_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=args.opset,
            dynamo=False,
        )

        text_path = args.out / "clip_text.onnx"
        torch.onnx.export(
            TextEncoder(model),
            torch.zeros(1, context, dtype=torch.int64),
            str(text_path),
            input_names=["input_ids"],
            output_names=["text_embeds"],
            opset_version=args.opset,
            dynamo=False,
        )

    # The merge table, so the runtime can tokenise without transformers.
    merges_src = Path(tokenizer.vocab_files_names["merges_file"])
    for candidate in (
        Path(tokenizer.name_or_path) / merges_src.name,
        *[Path(p) for p in getattr(tokenizer, "init_kwargs", {}).values()
          if isinstance(p, str) and p.endswith("merges.txt")],
    ):
        if candidate.is_file():
            shutil.copy2(candidate, args.out / "clip_merges.txt")
            break
    else:
        saved = args.out / "_tokenizer"
        tokenizer.save_pretrained(saved)
        shutil.copy2(saved / "merges.txt", args.out / "clip_merges.txt")
        shutil.copy2(saved / "vocab.json", args.out / "clip_vocab.json")
        shutil.rmtree(saved, ignore_errors=True)

    if not (args.out / "clip_vocab.json").is_file():
        (args.out / "clip_vocab.json").write_text(
            json.dumps(tokenizer.get_vocab()), encoding="utf-8"
        )

    # Golden pairs, so the pure-Python tokenizer can be proven equivalent.
    golden = [
        {"text": text, "ids": tokenizer(text, add_special_tokens=False)["input_ids"]}
        for text in PROBE_TEXTS
    ]
    (args.out / "clip_golden.json").write_text(json.dumps(golden, indent=1), encoding="utf-8")

    print(f"\nimage encoder : {image_path} ({image_path.stat().st_size / 1e6:.1f} MB)")
    print(f"text encoder  : {text_path} ({text_path.stat().st_size / 1e6:.1f} MB)")
    print(f"input size    : {image_size}px, context {context}")
    print(f"embedding dim : {model.config.projection_dim}")
    print(f"golden pairs  : {len(golden)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
