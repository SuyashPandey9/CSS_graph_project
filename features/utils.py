"""Shared utilities for feature calculations."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z]+")


def tokenize(text: str) -> set[str]:
    """Tokenize text into a lowercase word set."""

    return {t.lower() for t in _WORD_RE.findall(text)}


def clamp(value: float, *, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """Clamp value into [min_value, max_value]."""

    return max(min_value, min(max_value, value))
