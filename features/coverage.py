"""Coverage feature for CSS."""

from __future__ import annotations

from core.types import Graph
from features.utils import clamp, tokenize


def coverage(graph: Graph, query: str) -> float:
    """Proportion of query tokens covered by the graph."""

    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    graph_tokens: set[str] = set()
    for node in graph.nodes:
        graph_tokens.update(tokenize(node.text))

    covered = query_tokens.intersection(graph_tokens)
    score = len(covered) / len(query_tokens)
    return clamp(score)
