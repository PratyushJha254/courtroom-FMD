import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

from .losses import build_loss


def build_optimizer(model, cfg):
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    groups = [{"params": head, "lr": cfg["lr"]}]
    if enc:
        groups.append({"params": enc, "lr": cfg.get("encoder_lr", cfg["lr"])})

    name = cfg.get("optimizer", "adamw")
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.01))
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def predict_probs(model, loader, device, desc=None):
    model.eval()
    probs = []
    for batch in tqdm(loader, desc=desc, disable=desc is None):
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs.append(torch.sigmoid(logits).squeeze(1).cpu())
    return torch.cat(probs).numpy()


def classification_metrics(probs, targets, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    targets = np.asarray(targets).astype(int)
    return {
        "accuracy": accuracy_score(targets, preds),
        "macro_f1": f1_score(targets, preds, average="macro"),
        "auc": roc_auc_score(targets, probs),
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, probs, targets = 0.0, [], []
    for batch in loader:
        y = batch["targets"].to(device).unsqueeze(1)
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        total += criterion(logits, y).item()
        probs.append(torch.sigmoid(logits).squeeze(1).cpu())
        targets.append(y.squeeze(1).cpu())
    probs, targets = torch.cat(probs).numpy(), torch.cat(targets).numpy()
    out = classification_metrics(probs, targets)
    out["loss"] = total / len(loader)
    return out


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None, desc=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=desc):
        y = batch["targets"].to(device).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)), y)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def fit(model, train_loader, val_loader, cfg, device, output, start_from_checkpoint=False):
    """Train with early stopping on validation AUC (or loss) and keep the best weights at `output`.

    When fine-tuning from a checkpoint, the score of that checkpoint is the one to beat,
    so the saved file is only replaced by a strictly better model.
    """
    criterion = build_loss(cfg.get("loss", "bce"), cfg.get("label_smoothing", 0.1))
    optimizer = build_optimizer(model, cfg)
    monitor = cfg.get("monitor", "auc")
    better = (lambda a, b: a > b) if monitor == "auc" else (lambda a, b: a < b)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    best = -math.inf if monitor == "auc" else math.inf
    if start_from_checkpoint:
        best = evaluate(model, val_loader, criterion, device)[monitor]
        torch.save(model.state_dict(), output)
        print(f"initial val {monitor}: {best:.4f}")

    saved, bad_epochs = False, 0
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg.get("grad_clip"),
            desc=f"epoch {epoch}/{cfg['epochs']}",
        )
        val = evaluate(model, val_loader, criterion, device)
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val['loss']:.4f} | "
            f"Val Acc: {val['accuracy']:.4f} | Val AUC: {val['auc']:.4f}"
        )

        if better(val[monitor], best):
            best, bad_epochs, saved = val[monitor], 0, True
            torch.save(model.state_dict(), output)
        else:
            bad_epochs += 1
            if bad_epochs == cfg["patience"]:
                print("Early stopping.")
                break

    if saved:
        print(f"best val {monitor}: {best:.4f}, saved to {output}")
    else:
        print(f"no epoch beat the starting val {monitor}; {output} holds the starting weights")
    return best
