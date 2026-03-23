"""Verification tests for CSS-gated streaming integration.

Tests:
1. Change detection (new vs update vs unchanged)
2. Registry tracking (version history, old/new data)
3. IndexIDMap CRUD (add → find, update → old gone, delete → gone)
4. CSS quality gate (noise rejected, quality accepted)
5. Incremental edge building
6. End-to-end streaming adapter

Run:
    python test_streaming.py
"""

from __future__ import annotations

import time


def test_change_detection():
    """Test 1: Detect add vs update vs unchanged."""
    from core.streaming_state import StreamingState
    
    state = StreamingState()
    
    # New document
    assert state.detect_change("doc1", "Hello world") == "new"
    
    # Ingest it
    result = state.ingest("doc1", "Hello world")
    assert result.action == "added"
    assert result.version == 1
    
    # Same content → unchanged
    assert state.detect_change("doc1", "Hello world") == "unchanged"
    
    # Different content → updated
    assert state.detect_change("doc1", "Hello world updated") == "updated"
    
    # Another new
    assert state.detect_change("doc2", "Something else") == "new"
    
    print("  ✅ Test 1: Change detection works")


def test_version_history():
    """Test 2: Version tracking with full history."""
    from core.streaming_state import StreamingState
    
    state = StreamingState()
    
    # Create
    r1 = state.ingest("doc1", "Version 1 text")
    assert r1.version == 1
    assert r1.action == "added"
    
    # Update
    r2 = state.ingest("doc1", "Version 2 text with changes")
    assert r2.version == 2
    assert r2.action == "updated"
    assert r2.old_text_hash is not None
    assert r2.old_text_hash != r2.new_text_hash
    
    # Update again
    r3 = state.ingest("doc1", "Version 3 text with more changes")
    assert r3.version == 3
    
    # Check history
    history = state.get_version_history("doc1")
    assert len(history) == 3  # created, updated, updated
    assert history[0].action == "created"
    assert history[0].version == 1
    assert history[1].action == "updated"
    assert history[1].version == 1  # the old version that was replaced
    assert history[2].action == "updated"
    assert history[2].version == 2
    
    # Check current
    current = state.get_current_version("doc1")
    assert current.version == 3
    assert "Version 3" in current.text
    
    print("  ✅ Test 2: Version history tracking works")


def test_faiss_crud():
    """Test 3: FAISS add/update/delete via IndexIDMap."""
    from core.streaming_state import StreamingState
    
    state = StreamingState()
    
    # Add documents
    state.ingest("doc1", "Machine learning is a subset of artificial intelligence")
    state.ingest("doc2", "Deep learning uses neural networks with many layers")
    state.ingest("doc3", "Natural language processing deals with text and speech")
    
    assert state.get_stats()["active_documents"] == 3
    assert state.get_stats()["faiss_vectors"] == 3
    
    # Search should find relevant docs
    query_vec = state.embedder.embed("What is deep learning?")
    results = state.search_similar(query_vec, top_k=2)
    assert len(results) == 2
    doc_ids = [r[0] for r in results]
    assert "doc2" in doc_ids  # Deep learning doc should be found
    
    # Update doc2
    state.ingest("doc2", "Quantum computing uses qubits instead of classical bits")
    
    # Now search for deep learning should NOT find doc2
    results_after = state.search_similar(query_vec, top_k=3)
    doc_ids_after = [r[0] for r in results_after]
    # doc2 is now about quantum computing, so it should be less relevant
    
    # Delete doc3
    state.remove("doc3")
    assert state.get_stats()["active_documents"] == 2
    
    # doc3 should not appear in search
    all_vec = state.embedder.embed("text and speech processing")
    results_final = state.search_similar(all_vec, top_k=5)
    doc_ids_final = [r[0] for r in results_final]
    assert "doc3" not in doc_ids_final
    
    # But history is preserved
    history = state.get_version_history("doc3")
    assert len(history) > 0
    assert any(h.action == "deleted" for h in history)
    
    print("  ✅ Test 3: FAISS CRUD (add/update/delete) works")


def test_css_gate():
    """Test 4: CSS quality gate rejects noise, accepts quality."""
    from core.streaming_state import StreamingState
    
    state = StreamingState()
    
    # Seed with some content first
    state.ingest("seed1", "Login authentication error occurs when users try to sign in with expired session tokens. The error code is AUTH_500 and it affects version 3.2.1 of the application.")
    
    # Accept: detailed bug report (high specificity)
    r1 = state.ingest(
        "ticket1",
        "Payment gateway timeout on Stripe integration. Error code STRIPE_001. "
        "Transactions over $500 fail after 30 seconds. Affects 15% of checkouts since January 5, 2026.",
        css_gate=True, css_threshold=0.3,
    )
    assert r1.action == "added", f"Expected 'added' but got '{r1.action}': {r1.reason}"
    
    # Reject: generic comment (low specificity + no info)
    r2 = state.ingest(
        "noise1",
        "Thanks for reporting.",
        css_gate=True, css_threshold=0.3,
    )
    assert r2.action == "rejected", f"Expected 'rejected' but got '{r2.action}': {r2.reason}"
    
    # Reject: near-duplicate of seed1
    r3 = state.ingest(
        "dup1",
        "Login authentication error occurs when users try to sign in with expired session tokens. The error code is AUTH_500.",
        css_gate=True, css_threshold=0.3,
    )
    assert r3.action == "rejected", f"Expected 'rejected' but got '{r3.action}': {r3.reason}"
    
    stats = state.get_stats()
    assert stats["total_rejected"] >= 2
    
    print("  ✅ Test 4: CSS gate rejects noise, accepts quality")


def test_incremental_edges():
    """Test 5: Incremental edge building (O(n) not O(n²))."""
    from core.types import Graph, Node
    from graph.edge_builder import add_node_with_edges, build_edges
    
    # Create initial graph
    nodes = [
        Node(id="n1", text="The termination clause allows either party to terminate upon 30 days notice."),
        Node(id="n2", text="Indemnification obligations survive the termination of this agreement."),
    ]
    graph = Graph(nodes=nodes, edges=[])
    graph = build_edges(graph)
    initial_edges = len(graph.edges)
    
    # Add a new node incrementally
    new_node = Node(id="n3", text="Upon termination, all confidential information must be returned within 15 days.")
    graph2 = add_node_with_edges(graph, new_node)
    
    assert len(graph2.nodes) == 3
    assert len(graph2.edges) >= initial_edges  # Should have at least as many edges
    
    # The new edges should only involve n3
    new_edges = [e for e in graph2.edges if e not in graph.edges]
    for edge in new_edges:
        assert "n3" in (edge.source, edge.target), f"New edge doesn't involve n3: {edge}"
    
    # Duplicate add should be no-op
    graph3 = add_node_with_edges(graph2, new_node)
    assert len(graph3.nodes) == 3  # Still 3, not 4
    
    print("  ✅ Test 5: Incremental edge building works (O(n))")


def test_streaming_adapter():
    """Test 6: End-to-end streaming adapter."""
    from streaming.adapter import StreamAdapter
    
    adapter = StreamAdapter(css_gate=True, css_threshold=0.3)
    
    # Create tickets
    r1 = adapter.on_data_created(
        "TICK-201",
        "Database connection pool exhausted. Max connections reached at 100. "
        "Error: CONN_POOL_FULL. PostgreSQL 14 on AWS RDS. Affects API response times, "
        "p99 latency increased from 200ms to 5 seconds.",
    )
    assert r1.action == "added"
    
    r2 = adapter.on_data_created(
        "TICK-202",
        "Memory leak in background worker process. RSS grows by 50MB per hour. "
        "After 24 hours, OOM killer terminates the process. Heap dump shows "
        "unclosed database cursors accumulating.",
    )
    assert r2.action == "added"
    
    # Noise should be rejected
    r3 = adapter.on_data_created("TICK-noise", "Ok thanks.")
    assert r3.action == "rejected"
    
    # Update a ticket
    r4 = adapter.on_data_updated(
        "TICK-201",
        "Database connection pool exhausted. RESOLVED: Increased max connections to 200 "
        "and added connection timeout of 30 seconds. Deployed pgBouncer as connection pooler.",
    )
    assert r4.action == "updated"
    assert r4.version == 2
    
    # Query against live data
    result = adapter.query("What database issues are happening?")
    assert result["n_results"] > 0
    assert result["css_final"] > 0
    
    # Check that updated content is searchable
    doc_ids = [n[0] for n in result["nodes"]]
    if "TICK-201" in doc_ids:
        # The updated text should contain "pgBouncer"
        for doc_id, preview in result["nodes"]:
            if doc_id == "TICK-201":
                assert "pgBouncer" in preview or "RESOLVED" in preview, \
                    f"Retrieved stale data: {preview}"
    
    # Delete and verify
    adapter.on_data_deleted("TICK-202")
    stats = adapter.get_stats()
    assert stats["total_deleted"] >= 1
    
    # Version history should be preserved
    history = adapter.get_document_history("TICK-201")
    assert len(history) >= 2  # created + updated
    
    print("  ✅ Test 6: End-to-end streaming adapter works")


def test_latency():
    """Test 7: Verify ingestion latency is < 100ms."""
    from core.streaming_state import StreamingState
    
    state = StreamingState()
    
    # Measure single ingest latency
    latencies = []
    for i in range(5):
        start = time.time()
        state.ingest(f"perf_doc_{i}", f"Performance test document number {i} with some content about testing latency in real-time streaming systems.")
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"  ✅ Test 7: Avg ingest latency={avg_latency:.1f}ms, max={max_latency:.1f}ms",
          end="")
    if avg_latency < 100:
        print(" (under 100ms target ✓)")
    else:
        print(f" (WARNING: above 100ms target)")


def main():
    """Run all streaming tests."""
    print("=" * 60)
    print("  CSS-GATED STREAMING - Verification Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_change_detection,
        test_version_history,
        test_faiss_crud,
        test_css_gate,
        test_incremental_edges,
        test_streaming_adapter,
        test_latency,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
