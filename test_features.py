"""Phase 4 verification script for CSS feature framework."""

from __future__ import annotations

from core.types import Edge, Graph, Node
from css.calculator import compute_css, compute_css_final
from features.connectivity import connectivity
from features.coverage import coverage
from features.density import density
from features.diversity import diversity
from features.novelty import novelty
from features.relevance import relevance
from features.user_reuse import user_reuse


def main() -> None:
    """Create test graphs and compute all features."""

    graph = Graph(
        nodes=[
            Node(id="n1", text="Solar energy uses sunlight to generate electricity."),
            Node(id="n2", text="Wind turbines convert moving air into power."),
        ],
        edges=[Edge(source="n1", target="n2", relation="related_to")],
    )

    user_graph = Graph(
        nodes=[
            Node(id="n2", text="Wind energy basics"),
            Node(id="n3", text="Hydropower basics"),
        ]
    )

    query = "How does solar power generate electricity?"

    feature_values = {
        "coverage": coverage(graph, query),
        "relevance": relevance(graph, query),
        "novelty": novelty(graph, query),
        "connectivity": connectivity(graph, query),
        "density": density(graph, query),
        "diversity": diversity(graph, query),
        "user_reuse": user_reuse(graph, query, user_graph),
    }

    print("Feature values:")
    for name, value in feature_values.items():
        print(f"- {name}: {value:.3f}")

    css = compute_css(graph, query, user_graph)
    css_final = compute_css_final(graph, query, user_graph)

    print(f"CSS score: {css['css_score']:.3f}")
    print(f"CSS final: {css_final['css_final']:.3f}")


if __name__ == "__main__":
    main()
