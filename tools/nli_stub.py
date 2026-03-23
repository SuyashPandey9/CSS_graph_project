"""Rule-based NLI stub for V3."""

from __future__ import annotations

import re

from tools.base import BaseNLI

_WORD_RE = re.compile(r"[A-Za-z]+")
_STOPWORDS = {
    "the",
    "and",
    "or",
    "of",
    "to",
    "a",
    "in",
    "is",
    "are",
    "for",
    "on",
    "with",
    "by",
    "from",
}


class NLIStub(BaseNLI):
    """Returns 1.0 if keyword overlap is high, else 0.2."""

    def __init__(self, threshold: float = 0.3) -> None:
        self._threshold = threshold

    def score(self, premise: str, hypothesis: str) -> float:
        premise_tokens = {
            t.lower() for t in _WORD_RE.findall(premise) if t.lower() not in _STOPWORDS
        }
        hypothesis_tokens = {
            t.lower() for t in _WORD_RE.findall(hypothesis) if t.lower() not in _STOPWORDS
        }
        if not premise_tokens or not hypothesis_tokens:
            return 0.2
        overlap = premise_tokens.intersection(hypothesis_tokens)
        ratio = len(overlap) / max(len(hypothesis_tokens), 1)
        return 1.0 if ratio >= self._threshold else 0.2
