import json

import torch
from torch.utils.data import DataLoader, Dataset

PROMPT_PREFIX = (
    "You are a financial misinformation detector.\n"
    "Please check whether the following information is true or false "
    "and output the answer [true/false].\n\n\n"
)
QUESTION_KEY = "Open-ended Verifiable Question"
ANSWER_KEY = "Ground-True Answer"
ARGUMENTS_MARKER = "\n\n[PROSECUTION]"


def strip_prefix(text):
    return text[len(PROMPT_PREFIX):] if text.startswith(PROMPT_PREFIX) else text


def label_from_answer(answer):
    """1 = misinformation (false), 0 = true, -1 = unknown."""
    answer = str(answer).lower()
    if "false" in answer:
        return 1
    if "true" in answer:
        return 0
    return -1


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_items(paths):
    items = []
    for path in paths:
        items.extend(load_json(path))
    return items


def item_text(item, use_arguments=True):
    text = strip_prefix(item.get(QUESTION_KEY, ""))
    if not use_arguments:
        text = text.split(ARGUMENTS_MARKER)[0]
    return text


def load_texts(paths, use_arguments=True):
    return [item_text(item, use_arguments) for item in load_items(paths)]


def load_labelled(paths, use_arguments=True):
    texts, labels = [], []
    for item in load_items(paths):
        y = label_from_answer(item.get(ANSWER_KEY, ""))
        if y != -1:
            texts.append(item_text(item, use_arguments))
            labels.append(y)
    return texts, labels


def split_train_val(texts, labels, val_fraction=0.2, method="ordered", seed=42):
    """`ordered` keeps the file order and takes the tail as validation (as in the paper)."""
    if method == "ordered":
        cut = int(len(texts) * (1 - val_fraction))
        return texts[:cut], texts[cut:], labels[:cut], labels[cut:]
    if method == "stratified":
        from sklearn.model_selection import train_test_split

        tr_x, va_x, tr_y, va_y = train_test_split(
            texts, labels, test_size=val_fraction, random_state=seed, stratify=labels
        )
        return tr_x, va_x, tr_y, va_y
    raise ValueError(f"unknown split method: {method}")


class FMDDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["targets"] = torch.tensor(float(self.labels[idx]))
        return item


def build_loaders(cfg, tokenizer, max_length=None):
    """Train and validation loaders for a config, using the split described in its `data` section."""
    data = cfg["data"]
    texts, labels = load_labelled(data["train_files"], data.get("use_arguments", True))
    tr_x, va_x, tr_y, va_y = split_train_val(
        texts, labels, data.get("val_fraction", 0.2), data.get("split", "ordered"), data.get("split_seed", 42)
    )
    max_length = max_length or data["max_length"]
    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(FMDDataset(tr_x, tokenizer, max_length, tr_y), batch_size=bs, shuffle=True)
    val_loader = DataLoader(FMDDataset(va_x, tokenizer, max_length, va_y), batch_size=bs)
    return train_loader, val_loader
