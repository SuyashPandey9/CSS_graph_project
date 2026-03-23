"""V3 CLI integration: run optimization loop and generate final answer."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from core.types import Graph
from css.calculator import compute_css_final
from llm.llm_client import generate_text
from policy.optimizer import optimize
from tools.contradiction_flagger import build_contradiction_annotation


def _load_user_graph() -> Graph:
    """Load default user graph from data/user_graph.json."""

    path = Path(__file__).resolve().parent.parent / "data" / "user_graph.json"
    if not path.exists():
        raise FileNotFoundError("data/user_graph.json not found.")
    return Graph.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    """Run V3 workflow from CLI."""

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m app.v3_cli \"Your query?\"")

    query = sys.argv[1]
    user_graph = _load_user_graph()

    start_time = time.time()
    optimized = optimize(query, user_graph, max_steps=5)
    scores = compute_css_final(optimized, query, user_graph)

    context = "\n".join(f"- {node.text}" for node in optimized.nodes)
    
    # Fix A2: Inject contradiction/qualification warnings if detected
    contradiction_annotation = build_contradiction_annotation(optimized)
    
    prompt = (
        "Use the following graph context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
    )
    
    if contradiction_annotation:
        prompt += f"\n{contradiction_annotation}\n"

    answer = generate_text(prompt).text.strip()
    elapsed = time.time() - start_time

    print("=== V3 Answer ===")
    print(answer or "(empty response)")
    print("\n=== Metrics ===")
    print(f"CSS final: {scores['css_final']:.3f}")
    print(f"Nodes in G*: {len(optimized.nodes)}")
    print(f"Latency: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
