"""Relevance feature for CSS.

Measures semantic similarity between query and graph nodes using embeddings.
Optimized with batch embedding and caching.
"""

from __future__ import annotations

from core.types import Graph
from features.embeddings import cosine_similarity, get_embedding, get_embeddings_batch
from features.utils import clamp


def relevance(graph: Graph, query: str) -> float:
    """Maximum semantic similarity between query and any node.
    
    Uses sentence-transformers embeddings for semantic matching:
    - "solar power" ≈ "photovoltaic energy" (high similarity)
    - Captures meaning, not just word overlap
    
    Optimized: Uses batch embedding and caching for speed.
    
    Args:
        graph: The context graph to evaluate
        query: The user's query
    
    Returns:
        Highest cosine similarity [0, 1] between query and any node
    """

    if not query or not graph.nodes:
        return 0.0

    # Get query embedding (cached)
    query_vec = get_embedding(query)
    
    # Batch embed all nodes at once (with caching)
    node_texts = [node.text for node in graph.nodes if node.text]
    if not node_texts:
        return 0.0
    
    node_embeddings = get_embeddings_batch(node_texts)
    
    # Find max similarity
    best = 0.0
    for node_vec in node_embeddings:
        sim = cosine_similarity(query_vec, node_vec)
        sim = max(0.0, sim)  # Clamp negative similarities
        best = max(best, sim)

    return clamp(best)
