from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import DEFAULT_HF_MODEL_REPO, ID2LABEL, MODEL_DIR
from .text import make_model_input, normalize_text

COMMON_ASPECTS = [
    "food",
    "service",
    "staff",
    "price",
    "menu",
    "atmosphere",
    "delivery",
    "quality",
    "taste",
    "portion",
    "location",
    "cleanliness",
    "wait time",
    "value",
    "drinks",
]


def suggest_aspects(text: str, limit: int = 8) -> list[str]:
    lower = normalize_text(text).lower()
    found = [aspect for aspect in COMMON_ASPECTS if re.search(rf"\b{re.escape(aspect)}\b", lower)]
    if found:
        return found[:limit]
    tokens = re.findall(r"[A-Za-z][A-Za-z-]{2,}", lower)
    stop = {
        "the",
        "and",
        "but",
        "was",
        "were",
        "for",
        "with",
        "this",
        "that",
        "very",
        "really",
        "have",
        "had",
        "not",
        "our",
        "their",
    }
    candidates = []
    for token in tokens:
        if token not in stop and token not in candidates:
            candidates.append(token)
    return candidates[: min(limit, len(candidates))]


class TransformerAspectSentiment:
    def __init__(self, model_path: str | Path | None = None):
        model_path = model_path or os.getenv("MODEL_REPO_ID") or _local_or_hub_model()
        self.model_path = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.inference_mode()
    def predict_one(self, review: str, aspect: str) -> dict:
        encoded = self.tokenizer(
            make_model_input(review, aspect),
            truncation=True,
            max_length=192,
            return_tensors="pt",
        )
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        logits = self.model(**encoded).logits.detach().cpu().numpy()[0]
        probs = _softmax(logits)
        label_id = int(np.argmax(probs))
        return {
            "aspect": normalize_text(aspect),
            "sentiment": self.model.config.id2label.get(label_id, ID2LABEL.get(label_id, str(label_id))),
            "confidence": float(probs[label_id]),
            "scores": {
                self.model.config.id2label.get(idx, ID2LABEL.get(idx, str(idx))): float(score)
                for idx, score in enumerate(probs)
            },
        }

    def predict_many(self, review: str, aspects: list[str]) -> list[dict]:
        return [self.predict_one(review, aspect) for aspect in aspects if normalize_text(aspect)]


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


def _local_or_hub_model() -> str:
    local_path = MODEL_DIR / "transformer-final"
    if local_path.exists():
        return str(local_path)
    return DEFAULT_HF_MODEL_REPO
