"""Evaluate a checkpoint on the validation split of its config.

Example:
    python evaluate.py --config configs/modernbert_debate_gca_softbce.yaml \
        --checkpoint checkpoints/gca_softbce.pt --sweep
"""
import argparse
import json

import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoTokenizer

from fmd.data import build_loaders
from fmd.engine import classification_metrics, predict_probs
from fmd.models import build_model, load_checkpoint
from fmd.utils import get_device, load_config


def best_macro_f1_threshold(probs, targets):
    grid = np.arange(0.01, 1.0, 0.01)
    scores = [f1_score(targets, (probs >= t).astype(int), average="macro") for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--max-length", type=int, help="defaults to data.max_length in the config")
    p.add_argument("--sweep", action="store_true", help="also report the threshold that maximises macro-F1")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    train_loader, val_loader = build_loaders(cfg, tokenizer, args.max_length)
    loader = val_loader if args.split == "val" else train_loader

    model = load_checkpoint(build_model(cfg["model"]), args.checkpoint).to(device)
    probs = predict_probs(model, loader, device, desc=f"evaluating ({args.split})")
    targets = np.array(loader.dataset.labels)

    metrics = classification_metrics(probs, targets)
    if args.sweep:
        metrics["best_threshold"], metrics["best_macro_f1"] = best_macro_f1_threshold(probs, targets)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
