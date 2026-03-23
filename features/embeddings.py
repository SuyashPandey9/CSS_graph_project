"""Shared embedding utilities for CSS features.

Provides centralized access to embeddings for semantic features.
Includes caching to avoid recomputing embeddings for the same text.
"""

from __future__ import annotations

from typing import Dict, List

# Cache the embedder instance
_EMBEDDER = None

# Embedding cache: text -> embedding vector
_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def get_shared_embedder():
    """Return a cached embedder instance for semantic calculations.
    
    Uses the same NeuralEmbedder as FrozenState for consistency.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        from tools.neural_embedder import NeuralEmbedder
        _EMBEDDER = NeuralEmbedder()
    return _EMBEDDER


def get_embedding(text: str) -> List[float]:
    """Get embedding for text, using cache to avoid recomputation.
    
    Args:
        text: The text to embed
        
    Returns:
        The embedding vector
    """
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]
    
    embedder = get_shared_embedder()
    embedding = embedder.embed(text)
    _EMBEDDING_CACHE[text] = embedding
    return embedding


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts efficiently.
    
    Uses cache for previously computed embeddings and batches new ones.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors (same order as input)
    """
    if not texts:
        return []
    
    # Check which texts need embedding
    results: List[List[float] | None] = [None] * len(texts)
    texts_to_embed: List[str] = []
    indices_to_embed: List[int] = []
    
    for i, text in enumerate(texts):
        if text in _EMBEDDING_CACHE:
            results[i] = _EMBEDDING_CACHE[text]
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)
    
    # Batch embed uncached texts
    if texts_to_embed:
        embedder = get_shared_embedder()
        new_embeddings = embedder.embed_batch(texts_to_embed)
        
        for idx, text, embedding in zip(indices_to_embed, texts_to_embed, new_embeddings):
            _EMBEDDING_CACHE[text] = embedding
            results[idx] = embedding
    
    return results  # type: ignore


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Returns a value in [-1, 1], where 1 means identical direction.
    Note: NeuralEmbedder already normalizes embeddings, so this is just dot product.
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    # Since embeddings are normalized, cosine similarity = dot product
    dot = sum(a * b for a, b in zip(vec1, vec2))
    return dot


def clear_embedding_cache() -> None:
    """Clear the embedding cache (for testing or memory cleanup)."""
    global _EMBEDDING_CACHE
    _EMBEDDING_CACHE = {}


def clear_embedder_cache() -> None:
    """Clear the cached embedder (for testing or memory cleanup)."""
    global _EMBEDDER
    _EMBEDDER = None
    clear_embedding_cache()
