"""Batch 1 verification: Removed broken CSS components."""

from css.calculator import compute_css, compute_css_final, WEIGHTS
from core.types import Graph, Node, Edge
from features.connectivity import connectivity


def test_confidence_removed():
    """Fix 1.1: confidence should not be in WEIGHTS."""
    assert "confidence" not in WEIGHTS, f"confidence still in WEIGHTS: {WEIGHTS}"
    print("PASS: Fix 1.1 - confidence removed from WEIGHTS")


def test_css_final_no_multipliers():
    """Fix 1.2 + 1.3: CSS_final should equal CSS (no Semantic/Contradict multipliers)."""
    g = Graph(nodes=[Node(id="n1", text="Solar energy generates electricity")])
    result = compute_css_final(g, "solar energy", Graph())
    
    assert result["css_final"] == result["css_score"], \
        f"css_final ({result['css_final']}) != css_score ({result['css_score']})"
    assert result["semantic"] == 1.0, f"semantic should be 1.0 stub, got {result['semantic']}"
    assert result["contradict"] == 0.0, f"contradict should be 0.0 stub, got {result['contradict']}"
    print("PASS: Fix 1.2+1.3 - CSS_final = CSS (no multipliers)")


def test_connectivity_denominator():
    """Fix 1.4: connectivity should use n*(n-1)/2 denominator."""
    # 3 nodes, 1 edge → max_edges = 3*2/2 = 3 → score = 1/3 = 0.333
    g = Graph(
        nodes=[Node(id="a", text="x"), Node(id="b", text="y"), Node(id="c", text="z")],
        edges=[Edge(source="a", target="b", relation="test")]
    )
    score = connectivity(g, "")
    expected = 1.0 / 3.0  # 1 edge / 3 max edges
    assert abs(score - expected) < 0.01, \
        f"connectivity score {score} != expected {expected} (using n*(n-1)/2)"
    print(f"PASS: Fix 1.4 - connectivity uses n*(n-1)/2 (score={score:.3f})")


def test_css_still_computes():
    """Sanity: CSS still produces reasonable scores."""
    g = Graph(
        nodes=[
            Node(id="n1", text="Solar panels convert sunlight to electricity"),
            Node(id="n2", text="Photovoltaic cells absorb photons from the sun"),
        ],
        edges=[Edge(source="n1", target="n2", relation="related")]
    )
    result = compute_css_final(g, "How do solar panels work?", Graph())
    
    css = result["css_final"]
    assert 0.0 < css < 1.0, f"CSS score out of range: {css}"
    print(f"PASS: CSS still produces valid score ({css:.4f})")
    print(f"  Features: { {k: f'{v:.3f}' for k, v in result.items() if isinstance(v, float)} }")


if __name__ == "__main__":
    test_confidence_removed()
    test_css_final_no_multipliers()
    test_connectivity_denominator()
    test_css_still_computes()
    print("\n=== ALL BATCH 1 TESTS PASSED ===")
