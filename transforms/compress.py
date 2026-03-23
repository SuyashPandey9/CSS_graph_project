"""Compress operation for V3 graph transformation engine."""

from __future__ import annotations

from core.types import Graph, Node
from features.utils import tokenize
from transforms.utils import jaccard_similarity


def compress(graph: Graph, *, threshold: float = 0.5) -> Graph:
    """Merge nodes with high token overlap."""

    if not graph.nodes:
        return graph

    merged_nodes: list[Node] = []
    merged_ids: set[str] = set()

    for node in sorted(graph.nodes, key=lambda n: n.id):
        if node.id in merged_ids:
            continue
        node_tokens = tokenize(node.text)
        target = None
        for existing in merged_nodes:
            existing_tokens = tokenize(existing.text)
            if jaccard_similarity(node_tokens, existing_tokens) >= threshold:
                target = existing
                break
        if target:
            target.text = f"{target.text} {node.text}"
            merged_ids.add(node.id)
        else:
            merged_nodes.append(Node(**node.model_dump()))

    keep_ids = {node.id for node in merged_nodes}
    new_edges = [
        edge
        for edge in graph.edges
        if edge.source in keep_ids and edge.target in keep_ids
    ]
    return Graph(nodes=merged_nodes, edges=new_edges)
