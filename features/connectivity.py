"""Connectivity feature for CSS.

Measures structural graph connectivity (edge density).
This is a graph-theoretic metric, not semantic coherence.
"""

from __future__ import annotations

from core.types import Graph
from features.utils import clamp


def connectivity(graph: Graph, query: str) -> float:
    """Estimate structural connectivity based on edge count.
    
    Measures: edges / (n-1), where n-1 is minimum edges for a connected graph.
    
    Args:
        graph: The context graph to evaluate
        query: The query (unused, but kept for consistent API)
    
    Returns:
        0.5 for single node or empty graph (neutral)
        Higher values for more connected graphs
    """

    node_count = len(graph.nodes)
    if node_count <= 1:
        return 0.5

    # Fix 1.4: Use n*(n-1)/2 (max possible edges), consistent with edge_builder.py
    max_edges = node_count * (node_count - 1) / 2
    if max_edges == 0:
        return 0.5
    score = len(graph.edges) / max_edges
    return clamp(score)
