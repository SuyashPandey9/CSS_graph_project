"""Density feature for CSS."""

from __future__ import annotations

from core.types import Graph
from features.utils import clamp


def density(graph: Graph, query: str) -> float:
    """Graph density based on directed edge ratio."""

    node_count = len(graph.nodes)
    if node_count <= 1:
        return 0.0

    max_edges = node_count * (node_count - 1)
    score = len(graph.edges) / max_edges
    return clamp(score)
