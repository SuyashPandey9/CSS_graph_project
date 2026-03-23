"""Rule-based contradiction stub for V3."""

from __future__ import annotations

from tools.base import BaseContradiction


class ContradictionStub(BaseContradiction):
    """Always returns 0.0 for contradiction."""

    def score(self, text_a: str, text_b: str) -> float:
        return 0.0
