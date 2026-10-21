import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# padding="same" with the kernel-size-2 convolution triggers a harmless warning on every call
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")


class BaselineAttention(nn.Module):
    """Attention pooling without a context vector, alpha = softmax(tanh(W h_t)) (Kamal et al., 2023)."""

    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, 1)

    def forward(self, x):
        alpha = F.softmax(torch.tanh(self.W(x)), dim=1)
        return (x * alpha).sum(dim=1)


class GlobalContextAttention(nn.Module):
    """Attention pooling with a learned global context vector u_w (Yang et al., 2016)."""

    def __init__(self, hidden_size, attention_dim=None):
        super().__init__()
        attention_dim = attention_dim or hidden_size
        self.W = nn.Linear(hidden_size, attention_dim)
        self.u_w = nn.Parameter(torch.empty(attention_dim))
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        scores = torch.tanh(self.W(x)) @ self.u_w
        alpha = F.softmax(scores, dim=1)
        return (x * alpha.unsqueeze(-1)).sum(dim=1)


class MultiHeadGlobalContextAttention(nn.Module):
    """One context vector per head; the pooled vectors of all heads are concatenated."""

    def __init__(self, hidden_size, num_heads=3, attention_dim=None):
        super().__init__()
        attention_dim = attention_dim or hidden_size
        self.W = nn.Linear(hidden_size, attention_dim)
        self.u_w = nn.Parameter(torch.empty(num_heads, attention_dim))
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        scores = torch.einsum("bsd,hd->bhs", torch.tanh(self.W(x)), self.u_w)
        alpha = F.softmax(scores, dim=-1)
        return torch.matmul(alpha, x).flatten(1)


ATTENTION = {
    "baseline": BaselineAttention,
    "gca": GlobalContextAttention,
    "multihead_gca": MultiHeadGlobalContextAttention,
}


class MultiChannelClassifier(nn.Module):
    """Frozen encoder -> 3 x (Conv1D -> MaxPool -> BiGRU -> attention pooling) -> MLP -> logit.

    The three channels use Conv1D kernel sizes 1, 2 and 3.
    """

    def __init__(
        self,
        encoder="answerdotai/ModernBERT-base",
        attention="gca",
        filters=128,
        gru_units=64,
        fc_hidden=64,
        pool_kernel=2,
        dropout=0.2,
        gru_dropout=0.0,
        num_heads=3,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.conv1 = nn.Conv1d(hidden, filters, kernel_size=1, padding="same")
        self.conv2 = nn.Conv1d(hidden, filters, kernel_size=2, padding="same")
        self.conv3 = nn.Conv1d(hidden, filters, kernel_size=3, padding="same")
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)
        self.dropout = nn.Dropout(dropout)
        self.gru_dropout = gru_dropout

        self.bigru1 = nn.GRU(filters, gru_units, batch_first=True, bidirectional=True)
        self.bigru2 = nn.GRU(filters, gru_units, batch_first=True, bidirectional=True)
        self.bigru3 = nn.GRU(filters, gru_units, batch_first=True, bidirectional=True)

        attn_cls = ATTENTION[attention]
        attn_kwargs = {"num_heads": num_heads} if attention == "multihead_gca" else {}
        self.attn1 = attn_cls(2 * gru_units, **attn_kwargs)
        self.attn2 = attn_cls(2 * gru_units, **attn_kwargs)
        self.attn3 = attn_cls(2 * gru_units, **attn_kwargs)

        pooled = 2 * gru_units * (num_heads if attention == "multihead_gca" else 1)
        self.fc1 = nn.Linear(3 * pooled, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

    def _channel(self, x, conv, gru, attn):
        c = self.dropout(self.pool(F.relu(conv(x))))
        g, _ = gru(c.permute(0, 2, 1))
        if self.gru_dropout:
            g = F.dropout(g, p=self.gru_dropout, training=self.training)
        return attn(g)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x.permute(0, 2, 1)
        merged = torch.cat(
            [
                self._channel(x, self.conv1, self.bigru1, self.attn1),
                self._channel(x, self.conv2, self.bigru2, self.attn2),
                self._channel(x, self.conv3, self.bigru3, self.attn3),
            ],
            dim=1,
        )
        return self.fc2(F.relu(self.fc1(merged)))

    def unfreeze_last_layers(self, n):
        layers = encoder_layers(self.encoder)
        for p in layers[-n:].parameters():
            p.requires_grad = True
        return layers[-n:]


def encoder_layers(encoder):
    if hasattr(encoder, "layers"):  # ModernBERT
        return encoder.layers
    return encoder.encoder.layer  # BERT / RoBERTa / DeBERTa


def build_model(cfg):
    return MultiChannelClassifier(**cfg)


def load_checkpoint(model, path, device="cpu"):
    """Load a state dict saved by this repo or by the original notebooks."""
    state = torch.load(path, map_location=device)
    # older RoBERTa checkpoints store the encoder under `roberta.`
    state = {("encoder." + k[len("roberta."):] if k.startswith("roberta.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing + unexpected if not k.startswith("encoder.")]
    if bad:
        raise RuntimeError(f"checkpoint does not match the model config: {bad[:10]}")
    if missing or unexpected:
        # harmless for a frozen encoder (pretrained weights are used), not for a fine-tuned one
        print(f"warning: {len(missing)} missing / {len(unexpected)} unexpected encoder keys in {path}")
    return model
