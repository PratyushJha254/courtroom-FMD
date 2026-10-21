"""Train the classifier described by a config file.

Example:
    python train.py --config configs/modernbert_debate_gca.yaml
"""
import argparse

from transformers import AutoTokenizer

from fmd.data import build_loaders
from fmd.engine import fit
from fmd.models import build_model, load_checkpoint
from fmd.utils import get_device, load_config, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", help="overrides train.output in the config")
    p.add_argument("--init-from", help="overrides train.init_from in the config")
    args = p.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    if tcfg.get("seed") is not None:
        set_seed(tcfg["seed"])
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder"])
    train_loader, val_loader = build_loaders(cfg, tokenizer)
    print(f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val samples")

    model = build_model(cfg["model"])
    init_from = args.init_from or tcfg.get("init_from")
    if init_from:
        load_checkpoint(model, init_from)
        print(f"initialised from {init_from}")
    if tcfg.get("unfreeze_last_n"):
        model.unfreeze_last_layers(tcfg["unfreeze_last_n"])
    model.to(device)

    fit(model, train_loader, val_loader, tcfg, device, args.output or tcfg["output"], bool(init_from))


if __name__ == "__main__":
    main()
