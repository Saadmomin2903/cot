"""
Text normalization helpers.

The pipeline sometimes receives text as a list of strings (e.g., chunked content,
lines, or multi-block ingestor outputs). Downstream processors expect a string.
These helpers coerce common input shapes into a single, readable string.
"""

from __future__ import annotations

from typing import Any, Iterable


def to_text(value: Any, *, joiner: str = "\n") -> str:
    """
    Coerce value into a string suitable for NLP processing.

    Rules:
    - str -> returned as-is
    - list/tuple -> join each element (recursively) with `joiner`
    - dict -> try common keys; else stringified
    - None -> ""
    - other -> str(value)
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            t = to_text(item, joiner=joiner)
            if t:
                parts.append(t)
        return joiner.join(parts)

    if isinstance(value, dict):
        # Common shapes in this codebase
        for key in ("text", "final_text", "cleaned_text", "content"):
            if key in value:
                return to_text(value.get(key), joiner=joiner)
        return str(value)

    return str(value)


def is_text_like(value: Any) -> bool:
    """Best-effort check for values we can reasonably coerce to text."""
    return value is None or isinstance(value, (str, list, tuple, dict))


