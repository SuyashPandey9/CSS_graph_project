"""Phase 2 verification script for core data structures and corpus loading."""

from __future__ import annotations

from core.types import Edge, Graph, Node
from storage.corpus_store import load_fixed_corpus


def main() -> None:
    """Load corpus, build a graph, serialize/deserialize, and print results."""

    corpus = load_fixed_corpus()
    print(f"Corpus loaded. Documents: {len(corpus.documents)}")

    graph = Graph()
    node_a = Node(id="n1", text="Solar energy basics")
    node_b = Node(id="n2", text="Wind energy basics")
    node_c = Node(id="n3", text="Hydropower basics")
    graph.add_node(node_a)
    graph.add_node(node_b)
    graph.add_node(node_c)

    graph.edges.append(Edge(source="n1", target="n2", relation="related_to"))
    graph.edges.append(Edge(source="n2", target="n3", relation="related_to"))

    serialized = graph.model_dump_json()
    restored = Graph.model_validate_json(serialized)

    print("Graph nodes:")
    for node in restored.get_nodes():
        print(f"- {node.id}: {node.text}")

    print("Graph edges:")
    for edge in restored.edges:
        print(f"- {edge.source} -> {edge.target} ({edge.relation})")


if __name__ == "__main__":
    main()
