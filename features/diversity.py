"""Diversity feature for CSS.

Measures topical diversity across graph nodes using embedding-based distance.
Optimized with batch embedding and caching.
"""

from __future__ import annotations

from core.types import Graph
from features.embeddings import cosine_similarity, get_embeddings_batch
from features.utils import clamp


def diversity(graph: Graph, query: str) -> float:
    """Measure topical diversity as average pairwise embedding distance.
    
    Higher values indicate nodes cover different topics/concepts.
    Lower values indicate nodes are semantically similar (redundant).
    
    Uses: 1 - average_cosine_similarity between all node pairs
    
    Optimized: Uses batch embedding with caching.
    
    Args:
        graph: The context graph to evaluate
        query: The query (unused, but kept for consistent API)
    
    Returns:
        Diversity score [0, 1], higher = more diverse topics
    """

    if len(graph.nodes) < 2:
        # Single node or empty: neutral diversity
        return 0.5

    # Batch embed all nodes (with caching)
    node_texts = [node.text for node in graph.nodes if node.text]
    if len(node_texts) < 2:
        return 0.5
    
    embeddings = get_embeddings_batch(node_texts)
    
    # Compute average pairwise similarity
    total_sim = 0.0
    pair_count = 0
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            total_sim += max(0.0, sim)  # Clamp negative similarities
            pair_count += 1
    
    if pair_count == 0:
        return 0.5
    
    avg_similarity = total_sim / pair_count
    
    # Diversity = 1 - similarity (high similarity = low diversity)
    diversity_score = 1.0 - avg_similarity
    
    return clamp(diversity_score)
