"""Helpers for parsing structured model output."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_payload(text: str) -> dict[str, Any]:
    """Extract a JSON object from plain text or fenced model output."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        loaded = json.loads(match.group(0))

    if not isinstance(loaded, dict):
        raise ValueError("Expected a JSON object from model output.")
    return loaded


def clamp_float(
    value: Any,
    default: float | None = None,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, numeric))
