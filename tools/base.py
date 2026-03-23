"""Abstract base classes for frozen tools in V3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseParser(ABC):
    """Parses text into structured tokens."""

    @abstractmethod
    def extract(self, text: str) -> dict:
        """Extract structured information from text."""


class BaseEmbedder(ABC):
    """Embeds text into a deterministic vector representation."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return a vector embedding for the text."""


class BaseNLI(ABC):
    """Natural language inference scoring stub."""

    @abstractmethod
    def score(self, premise: str, hypothesis: str) -> float:
        """Return an NLI-style score in [0, 1]."""


class BaseContradiction(ABC):
    """Contradiction scoring stub."""

    @abstractmethod
    def score(self, text_a: str, text_b: str) -> float:
        """Return a contradiction score in [0, 1]."""
