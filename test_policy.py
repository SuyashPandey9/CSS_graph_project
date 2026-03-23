"""Phase 6 verification script for the greedy optimization loop."""

from __future__ import annotations

from core.types import Graph, Node
from policy.optimizer import optimize


def main() -> None:
    """Run the optimization loop and print the final graph."""

    user_graph = Graph(
        nodes=[Node(id="u1", text="Solar energy basics")],
        edges=[],
    )
    query = "Explain how solar power generates electricity."

    final_graph = optimize(query, user_graph, max_steps=5)

    print("Final graph nodes:")
    for node in final_graph.nodes:
        print(f"- {node.id}: {node.text[:60]}")


if __name__ == "__main__":
    main()
