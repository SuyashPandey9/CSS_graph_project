"""User reuse feature for CSS."""

from __future__ import annotations

from core.types import Graph
from features.utils import clamp


def user_reuse(graph: Graph, query: str, user_graph: Graph) -> float:
    """Compute overlap between graph and user_graph node ids."""

    graph_ids = {node.id for node in graph.nodes}
    user_ids = {node.id for node in user_graph.nodes}
    if not graph_ids:
        return 0.0
    overlap = graph_ids.intersection(user_ids)
    score = len(overlap) / len(graph_ids)
    return clamp(score)
