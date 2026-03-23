"""Phase 5 verification script for graph transformations."""

from __future__ import annotations

from core.frozen_state import FrozenState
from core.types import Graph, Node
from transforms.compress import compress
from transforms.expand import expand
from transforms.prune import prune
from transforms.stop import stop


def main() -> None:
    """Apply expand, prune, compress, and stop to a test graph."""

    state = FrozenState.build()
    graph = Graph(nodes=[Node(id="seed", text="Solar energy basics")])
    print(f"Initial graph nodes: {len(graph.nodes)}")

    graph = expand(graph, state, top_k=2)
    print(f"After expand nodes: {len(graph.nodes)}")
    for node in graph.nodes:
        print(f"- {node.id}: {node.text[:60]}")

    graph = prune(graph, state, min_nodes=1)
    print(f"After prune nodes: {len(graph.nodes)}")
    for node in graph.nodes:
        print(f"- {node.id}: {node.text[:60]}")

    graph = compress(graph, threshold=0.4)
    print(f"After compress nodes: {len(graph.nodes)}")
    for node in graph.nodes:
        print(f"- {node.id}: {node.text[:60]}")

    graph = stop(graph, state)
    print(f"After stop nodes: {len(graph.nodes)}")


if __name__ == "__main__":
    main()
