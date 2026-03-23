"""Neural embedder using sentence-transformers for V3.

Uses the same embedding model as Traditional RAG for fair comparison.
"""

from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from tools.base import BaseEmbedder

# Same model as Traditional RAG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class NeuralEmbedder(BaseEmbedder):
    """Embeds text using sentence-transformers (semantic embeddings).
    
    Unlike TF-IDF, this captures semantic similarity:
    - "solar panels" ≈ "photovoltaic cells"
    - "renewable energy" ≈ "green power"
    """

    def __init__(self) -> None:
        """Load the pre-trained sentence transformer model."""
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, text: str) -> List[float]:
        """Embed text into a dense vector."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently in a batch."""
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()
