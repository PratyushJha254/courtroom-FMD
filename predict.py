"""Predict labels for a test file and write a submission csv (id, label).

With several test files (the same claims with independently sampled arguments), the
per-file predictions are combined by soft voting (mean probability) or hard voting.

Example:
    python predict.py --config configs/modernbert_debate_gca_softbce.yaml \
        --checkpoint checkpoints/gca_softbce.pt \
        --test data/test_aug_0.5.json --output submissions/submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fmd.data import FMDDataset, item_text, load_json, load_texts
from fmd.engine import predict_probs
from fmd.models import build_model, load_checkpoint
from fmd.utils import get_device, load_config


def to_labels(preds):
    # 1 = misinformation
    return np.where(preds == 1, "False", "True")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, help="defaults to predict.max_length, then data.max_length")
    p.add_argument("--vote", choices=["soft", "hard"], default="soft")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--replace", help="json with re-generated items carrying an `id`; their text replaces the originals")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-probs", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    max_length = args.max_length or cfg.get("predict", {}).get("max_length") or cfg["data"]["max_length"]
    use_arguments = cfg["data"].get("use_arguments", True)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    model = load_checkpoint(build_model(cfg["model"]), args.checkpoint).to(device)
    replacements = {it["id"]: item_text(it, use_arguments) for it in load_json(args.replace)} if args.replace else {}

    all_probs = []
    for path in args.test:
        texts = load_texts([path], use_arguments)
        for i, text in replacements.items():
            texts[i] = text
        loader = DataLoader(FMDDataset(texts, tokenizer, max_length), batch_size=args.batch_size)
        all_probs.append(predict_probs(model, loader, device, desc=Path(path).name))
    all_probs = np.stack(all_probs)

    if args.vote == "soft":
        probs = all_probs.mean(axis=0)
        preds = (probs >= args.threshold).astype(int)
    else:
        votes = (all_probs >= args.threshold).sum(axis=0)
        preds = (2 * votes > len(args.test)).astype(int)
        probs = all_probs.mean(axis=0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"id": range(len(preds)), "label": to_labels(preds)})
    if args.save_probs:
        out["prob"] = probs
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} predictions to {args.output} (max_length={max_length}, {len(args.test)} file(s))")


if __name__ == "__main__":
    main()
