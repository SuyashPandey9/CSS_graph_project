"""Phase 9 verification script for simple GraphRAG baseline."""

from __future__ import annotations

from evaluation.rag_baselines.graph_rag_simple import SimpleGraphRAG


def main() -> None:
    """Build the graph, run a query, and print the response."""

    graph_rag = SimpleGraphRAG()
    query = "How does solar energy generate electricity?"

    result = graph_rag.answer(query, top_k=5)
    print("GraphRAG answer:")
    print(result["answer"] or "(empty response)")
    print("\nGraph metadata:")
    print(f"- Nodes used: {len(result['nodes_used'])}")
    print(f"- Documents used: {len(result['documents_used'])}")
    print(f"- Context chunks: {result['context_size']}")
    print(f"- Latency: {result['latency_seconds']:.2f}s")


if __name__ == "__main__":
    main()
