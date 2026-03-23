"""Novelty feature for CSS."""

from __future__ import annotations

from core.types import Graph
from features.utils import clamp, tokenize


def novelty(graph: Graph, query: str) -> float:
    """Fraction of graph tokens not present in the query."""

    query_tokens = tokenize(query)
    graph_tokens: set[str] = set()
    for node in graph.nodes:
        graph_tokens.update(tokenize(node.text))

    if not graph_tokens:
        return 0.0

    novel = graph_tokens.difference(query_tokens)
    score = len(novel) / len(graph_tokens)
    return clamp(score)
