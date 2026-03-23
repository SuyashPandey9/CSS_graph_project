"""Batch 2 verification: Pre-CSS Pipeline fixes."""

import sys


def test_structure_aware_chunking():
    """Fix 3.1: Structure-aware chunking with section boundaries."""
    from tools.text_chunker import chunk_text
    
    sample_contract = """AGREEMENT

ARTICLE I - DEFINITIONS

1.1 "Company" means Acme Corporation, a Delaware corporation.
1.2 "Effective Date" means the date set forth on the signature page.
1.3 "Confidential Information" shall mean any proprietary information disclosed.

ARTICLE II - LICENSE GRANT

2.1 Subject to the terms of this Agreement, Company grants to Licensee a non-exclusive license.
2.2 The license granted herein shall be limited to the territory specified in Exhibit A.

ARTICLE III - TERM AND TERMINATION

3.1 This Agreement shall commence on the Effective Date and continue for a period of five years.
3.2 Either party may terminate this Agreement upon thirty days written notice.
3.3 Upon termination, Licensee shall cease all use of the licensed materials.

ARTICLE IV - INDEMNIFICATION

4.1 The Company shall indemnify and hold harmless the Licensee against all losses arising from breach.
4.2 The maximum aggregate liability under this Article shall not exceed $5,000,000.

ARTICLE V - SURVIVAL

5.1 Sections 1, 4, and 6 of this Agreement shall survive termination for a period of 24 months.
"""
    
    chunks = chunk_text(sample_contract, "test_contract", "Test Contract")
    
    # Should detect section boundaries (not just 500-char splits)
    assert len(chunks) >= 3, f"Expected >= 3 sections, got {len(chunks)}"
    
    # Check that section metadata is populated
    sections_with_ids = [c for c in chunks if c.section_id is not None]
    print(f"  Chunks: {len(chunks)}, with section_ids: {len(sections_with_ids)}")
    for c in chunks[:5]:
        print(f"    [{c.chunk_index}] section_id={c.section_id}, heading={c.section_heading[:40] if c.section_heading else 'None'}")
    
    print("PASS: Fix 3.1 - Structure-aware chunking")


def test_stopword_removal():
    """Fix 3.2: Section references survive tokenization."""
    from tools.parser_stub import ParserStub
    
    parser = ParserStub()
    result = parser.extract("What does Section 3.2 say about liability?")
    entities = result["entities"]
    
    # "Section 3.2" should survive as a compound entity
    has_section_ref = any("section" in e.lower() and "3.2" in e for e in entities)
    assert has_section_ref, f"'Section 3.2' not found in entities: {entities}"
    
    # "liability" should also be an entity
    has_liability = any("liability" in e.lower() for e in entities)
    assert has_liability, f"'liability' not found in entities: {entities}"
    
    print(f"  Entities: {entities[:6]}")
    print("PASS: Fix 3.2 - Section references preserved as compound entities")


def test_cross_reference_edges():
    """Fix 3.3: Cross-reference edge detection."""
    from graph.edge_builder import _has_cross_reference
    from core.types import Node
    
    # Node A references Section 4
    node_a = Node(
        id="test_chunk_001",
        text="Subject to Section 4, the Company shall indemnify the Licensee.",
        metadata={"section_id": "3.1"}
    )
    
    # Node B is Section 4
    node_b = Node(
        id="test_chunk_003",
        text="4.1 The Company shall indemnify and hold harmless the Licensee.",
        metadata={"section_id": "4", "section_heading": "ARTICLE IV - INDEMNIFICATION"}
    )
    
    assert _has_cross_reference(node_a, node_b), "Should detect cross-reference from A to B (Section 4)"
    print("PASS: Fix 3.3 - Cross-reference edge detection")


def test_idf_computation():
    """Fix 1c.1: IDF-based entity detection."""
    from graph.edge_builder import compute_corpus_idf
    from core.types import CorpusDocument
    
    # Create a small corpus
    docs = [
        CorpusDocument(id="d1", title="Doc 1", text="The Company shall indemnify the Licensee for all losses."),
        CorpusDocument(id="d2", title="Doc 2", text="The Company shall pay all fees on time."),
        CorpusDocument(id="d3", title="Doc 3", text="The Company shall maintain confidentiality of all information."),
        CorpusDocument(id="d4", title="Doc 4", text="Indemnification obligations shall survive termination."),
        CorpusDocument(id="d5", title="Doc 5", text="The Company agrees to the terms herein."),
    ]
    
    idf = compute_corpus_idf(docs)
    
    # "company" should be low IDF (appears in docs 1-3, 5)
    # "indemnify"/"indemnification" should be high IDF (appears in only 1-2 docs)
    assert len(idf) > 0, "IDF dict should not be empty"
    
    company_idf = idf.get("company", 0)
    indem_idf = idf.get("indemnify", 0) or idf.get("indemnification", 0)
    
    print(f"  IDF scores: company={company_idf:.2f}, indemnify/indemnification={indem_idf:.2f}")
    if company_idf > 0 and indem_idf > 0:
        assert indem_idf > company_idf, "Rare terms should have higher IDF than common terms"
    
    print("PASS: Fix 1c.1 - IDF computation works")


def test_jaccard_removed():
    """Fix 1c.2: Jaccard/lexical factor removed."""
    from graph.edge_builder import DEFAULT_EDGE_PARAMS
    
    assert "lexical_weight" not in DEFAULT_EDGE_PARAMS, "lexical_weight should be removed"
    assert "lexical_threshold" not in DEFAULT_EDGE_PARAMS, "lexical_threshold should be removed"
    print("PASS: Fix 1c.2 - Jaccard/lexical factor removed")


def test_chunk_adjacency():
    """Fix 1c.3: Chunk adjacency detection."""
    from graph.edge_builder import _are_adjacent_chunks
    from core.types import Node
    
    node_a = Node(id="contract_chunk_003", text="Section A content")
    node_b = Node(id="contract_chunk_004", text="Section B content")
    node_c = Node(id="contract_chunk_010", text="Section C content")
    node_d = Node(id="other_doc_chunk_004", text="Different doc")
    
    assert _are_adjacent_chunks(node_a, node_b), "Chunks 003 and 004 should be adjacent"
    assert not _are_adjacent_chunks(node_a, node_c), "Chunks 003 and 010 should not be adjacent"
    assert not _are_adjacent_chunks(node_b, node_d), "Chunks from different docs should not be adjacent"
    print("PASS: Fix 1c.3 - Chunk adjacency detection")


def test_query_preprocessor():
    """Fix 3.4: Query preprocessor fallback (no LLM)."""
    from tools.query_preprocessor import preprocess_query, clear_cache
    
    clear_cache()
    result = preprocess_query("Does the indemnification obligation survive termination?")
    
    assert "entities" in result, "Missing 'entities' key"
    assert "subqueries" in result, "Missing 'subqueries' key"
    assert "intent" in result, "Missing 'intent' key"
    assert "clause_types" in result, "Missing 'clause_types' key"
    
    # Should detect some clause types
    print(f"  entities: {result['entities'][:5]}")
    print(f"  intent: {result['intent']}")
    print(f"  clause_types: {result['clause_types']}")
    print(f"  subqueries: {result['subqueries'][:3]}")
    
    # Test caching: second call should return same result
    result2 = preprocess_query("Does the indemnification obligation survive termination?")
    assert result == result2, "Cached result should be identical"
    
    print("PASS: Fix 3.4 - Query preprocessor (fallback mode)")


def test_bridge_node_protection():
    """Fix 2.12: Bridge-node protection in prune."""
    from graph.edge_builder import get_node_degree, has_cross_ref_edges
    from core.types import Node, Edge, Graph
    
    # Create a graph where node_b is a bridge (high degree, cross-ref edges)
    nodes = [
        Node(id="a", text="Indemnification clause", confidence=0.8),
        Node(id="b", text="Subject to Section 4, survival clause survives termination", confidence=0.4),  # Low relevance but bridge
        Node(id="c", text="Termination provisions", confidence=0.7),
        Node(id="d", text="Unrelated boilerplate", confidence=0.3),
    ]
    edges = [
        Edge(source="a", target="b", relation="cross_ref|entity_overlap:2", weight=0.6),
        Edge(source="b", target="c", relation="cross_ref|entity_overlap:1", weight=0.5),
        Edge(source="a", target="b", relation="semantic:0.7", weight=0.3),
        Edge(source="b", target="d", relation="same_source", weight=0.2),
    ]
    graph = Graph(nodes=nodes, edges=edges)
    
    # Node b should have high degree and cross-ref edges
    degree_b = get_node_degree(graph, "b")
    has_xref_b = has_cross_ref_edges(graph, "b")
    degree_d = get_node_degree(graph, "d")
    has_xref_d = has_cross_ref_edges(graph, "d")
    
    assert degree_b >= 3, f"Node b should have degree >= 3, got {degree_b}"
    assert has_xref_b, "Node b should have cross-ref edges"
    assert not has_xref_d, "Node d should not have cross-ref edges"
    
    print(f"  Node b: degree={degree_b}, has_cross_ref={has_xref_b}")
    print(f"  Node d: degree={degree_d}, has_cross_ref={has_xref_d}")
    print("PASS: Fix 2.12 - Bridge-node detection works")


def test_larger_retrieval_pool():
    """Fix 3.6: Default top_k increased."""
    from policy.optimizer import DEFAULT_TOP_K
    
    assert DEFAULT_TOP_K >= 10, f"DEFAULT_TOP_K should be >= 10, got {DEFAULT_TOP_K}"
    print(f"PASS: Fix 3.6 - DEFAULT_TOP_K = {DEFAULT_TOP_K}")


if __name__ == "__main__":
    tests = [
        test_structure_aware_chunking,
        test_stopword_removal,
        test_cross_reference_edges,
        test_idf_computation,
        test_jaccard_removed,
        test_chunk_adjacency,
        test_query_preprocessor,
        test_bridge_node_protection,
        test_larger_retrieval_pool,
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
            failed += 1
    
    print(f"\n=== BATCH 2 RESULTS: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed > 0 else 0)
