from __future__ import annotations

import re


def normalize_text(value: str) -> str:
    """Normalize review/aspect text while preserving sentiment cues."""
    value = str(value or "")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def make_model_input(text: str, aspect: str) -> str:
    return f"review: {normalize_text(text)} [SEP] aspect: {normalize_text(aspect)}"
