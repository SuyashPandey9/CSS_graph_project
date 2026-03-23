"""Deterministic TF-IDF embedder stub for V3."""

from __future__ import annotations

from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from tools.base import BaseEmbedder


class EmbedderStub(BaseEmbedder):
    """Embeds text using a TF-IDF vectorizer fit on the corpus."""

    def __init__(self, corpus_texts: list[str]) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True)
        self._vectorizer.fit(corpus_texts)

    def embed(self, text: str) -> List[float]:
        vector = self._vectorizer.transform([text]).toarray()[0]
        return vector.tolist()
