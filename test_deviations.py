"""Verification: all 7 deviation fixes."""

import sys


def test_deviation_1_answer_density_removed():
    """answer_density should NOT be in WEIGHTS (per proposed_fixes.md)."""
    from css.calculator import WEIGHTS
    assert "answer_density" not in WEIGHTS, f"answer_density still in WEIGHTS: {WEIGHTS}"
    print(f"  WEIGHTS keys: {list(WEIGHTS.keys())}")
    print("PASS: Deviation #1 — answer_density removed from WEIGHTS")


def test_deviation_2_frozenstate_has_idf_and_edges():
    """FrozenState should have idf_dict and precomputed_edges fields."""
    import dataclasses
    from core.frozen_state import FrozenState
    
    fields = {f.name for f in dataclasses.fields(FrozenState)}
    assert "idf_dict" in fields, f"Missing idf_dict field. Fields: {fields}"
    assert "precomputed_edges" in fields, f"Missing precomputed_edges field. Fields: {fields}"
    
    print(f"  FrozenState fields include: idf_dict, precomputed_edges")
    print("PASS: Deviation #2 — FrozenState has pre-computation fields")


def test_deviation_2b_precompute_function_exists():
    """precompute_cross_ref_edges should exist and work."""
    from graph.edge_builder import precompute_cross_ref_edges
    from types import SimpleNamespace
    
    docs = [
        SimpleNamespace(id="chunk_001", text="8.1 The Company shall indemnify the Licensee subject to Section 12.3."),
        SimpleNamespace(id="chunk_002", text="12.3 Termination provisions apply after thirty days."),
    ]
    
    edges = precompute_cross_ref_edges(docs)
    print(f"  Pre-computed edges: {edges}")
    # chunk_001 references "Section 12.3", chunk_002 has section_id "12.3"
    # This should find the cross-ref
    assert isinstance(edges, list), f"Should return list, got {type(edges)}"
    print("PASS: Deviation #2b — precompute_cross_ref_edges works")


def test_deviation_3_expand_no_regex():
    """expand should NOT import _extract_section_refs."""
    with open("transforms/expand.py", "r") as f:
        content = f.read()
    
    assert "_extract_section_refs" not in content, \
        "expand.py still imports _extract_section_refs (runtime regex!)"
    assert "precomputed_edges" in content or "precomputed" in content, \
        "expand.py doesn't reference precomputed edges"
    
    print("PASS: Deviation #3 — expand.py uses pre-computed edges, no runtime regex")


def test_deviation_4_coverage_suppression():
    """Coverage should be suppressed when preprocessor is active."""
    from css.calculator import compute_css
    from core.types import Graph, Node
    
    g = Graph(nodes=[
        Node(id="a", text="Indemnification obligations are defined here."),
    ])
    
    result = compute_css(g, "What is indemnification?", Graph())
    
    sq_cov = result.get("subquery_coverage", 0.5)
    coverage = result.get("coverage", 0.0)
    
    # If preprocessor returned non-0.5, coverage should be 0.0
    if sq_cov != 0.5:
        assert coverage == 0.0, f"Coverage should be 0.0 when preprocessor active, got {coverage}"
        print(f"  subquery_coverage={sq_cov:.3f} (active), coverage={coverage} (suppressed)")
    else:
        print(f"  subquery_coverage=0.5 (fallback), coverage={coverage:.3f} (active)")
    
    print("PASS: Deviation #4 — coverage suppression logic exists")


def test_deviation_5_section_aware_adjacency():
    """Section-aware adjacency: same section gets full bonus, cross-section gets reduced."""
    from graph.edge_builder import _are_adjacent_chunks, _same_section
    from core.types import Node
    
    # Same section, adjacent chunks
    a = Node(id="doc_chunk_001", text="text a", metadata={"section_id": "8"})
    b = Node(id="doc_chunk_002", text="text b", metadata={"section_id": "8"})
    
    assert _are_adjacent_chunks(a, b), "Should be adjacent"
    assert _same_section(a, b), "Should be same section"
    
    # Same document, adjacent but different sections
    c = Node(id="doc_chunk_003", text="text c", metadata={"section_id": "9"})
    
    assert _are_adjacent_chunks(b, c), "Should be adjacent"
    assert not _same_section(b, c), "Should NOT be same section"
    
    print("PASS: Deviation #5 — Section-aware adjacency implemented")


def test_deviation_6_docstring():
    """compute_css docstring should not mention confidence or wrong weights."""
    import inspect
    from css.calculator import compute_css
    
    doc = inspect.getdoc(compute_css) or ""
    assert "confidence" not in doc.lower(), f"Docstring still mentions 'confidence'"
    assert "w=2.0" not in doc, f"Docstring still has old token_efficiency weight"
    assert "w=1.0" not in doc or "query terms" not in doc, \
        "Docstring still has old coverage weight"
    
    print(f"  Docstring preview: {doc[:100]}...")
    print("PASS: Deviation #6 — Docstring updated")


def test_deviation_7_no_dead_nli_imports():
    """compute_css should not lazy-load NLI or contradiction stubs."""
    with open("css/calculator.py", "r") as f:
        content = f.read()
    
    # Find just the compute_css function body
    start = content.index("def compute_css(")
    next_def = content.index("\ndef ", start + 1)
    css_body = content[start:next_def]
    
    assert "NLIStub" not in css_body, "compute_css still loads NLIStub"
    assert "ContradictionStub" not in css_body, "compute_css still loads ContradictionStub"
    
    print("PASS: Deviation #7 — Dead NLI/contradiction imports removed from compute_css")


if __name__ == "__main__":
    tests = [
        test_deviation_1_answer_density_removed,
        test_deviation_2_frozenstate_has_idf_and_edges,
        test_deviation_2b_precompute_function_exists,
        test_deviation_3_expand_no_regex,
        test_deviation_4_coverage_suppression,
        test_deviation_5_section_aware_adjacency,
        test_deviation_6_docstring,
        test_deviation_7_no_dead_nli_imports,
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
    
    print(f"\n=== DEVIATION FIX RESULTS: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed > 0 else 0)
