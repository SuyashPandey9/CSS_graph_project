"""Connectivity feature for CSS (formerly coherence).

Measures structural graph connectivity, not semantic coherence.
Renamed to avoid confusion with NLP coherence concepts.
"""

from __future__ import annotations

from core.types import Graph
from features.utils import clamp


def connectivity(graph: Graph, query: str) -> float:
    """Estimate structural connectivity based on edge count.
    
    Measures: edges / (n-1), where n-1 is minimum edges for a connected graph.
    
    Returns:
        0.5 for single node or empty graph (neutral)
        Higher values for more connected graphs
    """

    node_count = len(graph.nodes)
    if node_count <= 1:
        return 0.5

    max_edges = node_count - 1
    score = len(graph.edges) / max_edges
    return clamp(score)


# Backward compatibility alias (deprecated)
coherence = connectivity
