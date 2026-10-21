import torch.nn as nn


class SoftBCEWithLogitsLoss(nn.Module):
    """BCE with label smoothing towards 0.5: y' = y(1 - eps) + 0.5 eps."""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce(logits, targets * (1.0 - self.smoothing) + 0.5 * self.smoothing)


def build_loss(name, smoothing=0.1):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "soft_bce":
        return SoftBCEWithLogitsLoss(smoothing)
    raise ValueError(f"unknown loss: {name}")
