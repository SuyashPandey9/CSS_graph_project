"""Batch 3 verification: CSS feature changes (updated after audit)."""

import sys


def test_weights_updated():
    """WEIGHTS should match final spec (no answer_density)."""
    from css.calculator import WEIGHTS, DEFAULT_TOKEN_BUDGET
    
    expected = {
        "query_relevance": 3.0,
        "subquery_coverage": 2.5,
        "context_cohesion": 1.5,
        "graph_connectivity": 1.0,
        "coverage": 0.5,
        "token_efficiency": 0.5,
        "redundancy_penalty": -1.5,
    }
    
    for key, val in expected.items():
        assert key in WEIGHTS, f"Missing weight: {key}"
        assert WEIGHTS[key] == val, f"Weight {key}: expected {val}, got {WEIGHTS[key]}"
    
    assert "confidence" not in WEIGHTS
    assert "answer_density" not in WEIGHTS, "answer_density should be removed per spec"
    assert DEFAULT_TOKEN_BUDGET == 2000
    
    print(f"  WEIGHTS: {WEIGHTS}")
    print("PASS: Weights match final specification")


def test_token_efficiency_simplified():
    """Fix 1.5: Simplified token_efficiency."""
    from css.calculator import compute_token_efficiency
    from core.types import Graph, Node
    
    g1 = Graph(nodes=[Node(id="a", text="short text")])
    assert compute_token_efficiency(g1, budget=2000) == 1.0
    
    big_text = " ".join(["word"] * 4000)
    g2 = Graph(nodes=[Node(id="b", text=big_text)])
    eff = compute_token_efficiency(g2, budget=2000)
    assert eff < 1.0 and eff > 0.0
    
    print(f"  Under budget: 1.0, Over budget (2x): {eff:.3f}")
    print("PASS: Fix 1.5 - Token efficiency simplified")


def test_subquery_coverage():
    """Fix 1.8: subquery_coverage feature."""
    from css.calculator import compute_subquery_coverage
    from core.types import Graph, Node
    
    g = Graph(nodes=[
        Node(id="a", text="The company shall indemnify the licensee for all losses."),
        Node(id="b", text="Termination may occur upon thirty days notice."),
    ])
    
    score = compute_subquery_coverage(g, "What are the indemnification obligations?")
    assert 0.0 <= score <= 1.0
    
    print(f"  subquery_coverage score: {score:.3f}")
    print("PASS: Fix 1.8 - subquery_coverage computes")


def test_full_css_pipeline():
    """Full CSS pipeline with spec-compliant features."""
    from css.calculator import compute_css_final
    from core.types import Graph, Node, Edge
    
    g = Graph(
        nodes=[
            Node(id="a", text="Solar panels use photovoltaic cells to generate electricity from sunlight."),
            Node(id="b", text="The efficiency of solar cells depends on the quality of silicon used."),
        ],
        edges=[Edge(source="a", target="b", relation="semantic:0.7", weight=0.5)]
    )
    
    result = compute_css_final(g, "How do solar panels generate electricity?", Graph())
    
    assert "subquery_coverage" in result
    assert "answer_density" not in result, "answer_density should not be computed"
    assert "query_relevance" in result
    assert "css_final" in result
    
    print(f"  query_relevance: {result['query_relevance']:.3f}")
    print(f"  subquery_coverage: {result['subquery_coverage']:.3f}")
    print(f"  css_final: {result['css_final']:.4f}")
    print("PASS: Full CSS pipeline (spec-compliant)")


if __name__ == "__main__":
    tests = [
        test_weights_updated,
        test_token_efficiency_simplified,
        test_subquery_coverage,
        test_full_css_pipeline,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\n--- {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n=== BATCH 3 RESULTS: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed > 0 else 0)
