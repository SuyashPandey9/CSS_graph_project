"""Interactive demo: CSS-gated streaming with a simulated ticketing system.

Simulates a support ticketing system where:
1. Tickets are created with bug reports
2. Comments and status updates arrive as streaming data
3. CSS gate filters out noise (generic comments, status changes)
4. User queries are answered against live data

Run:
    python streaming/demo.py
"""

from __future__ import annotations

import time

from streaming.adapter import StreamAdapter


# Simulated ticket events (mix of useful and noisy data)
TICKET_EVENTS = [
    # Useful: detailed bug report
    ("create", "TICK-101", "Login button not working on Chrome browser. User clicks submit and gets HTTP 500 error. Stack trace shows NullPointerException in AuthController.java line 42. Affects version 3.2.1."),
    
    # Noise: generic acknowledgment
    ("create", "TICK-101-comment-1", "Thanks for reporting this issue. We will investigate."),
    
    # Noise: status update
    ("create", "TICK-101-status-1", "Status changed from Open to In Progress"),
    
    # Useful: duplicate bug but from different source
    ("create", "TICK-102", "Login page returns 500 error on Chrome when clicking the sign-in button. Started happening after the v3.2.1 deployment yesterday. Multiple users affected."),
    
    # Noise: request for info
    ("create", "TICK-101-comment-2", "Can you please share a screenshot of the error?"),
    
    # Very useful: root cause and fix
    ("create", "TICK-101-comment-3", "Root cause identified: session token validation was failing due to expired JWT signing key. Fixed by rotating the signing key and updating the AuthMiddleware to handle key rotation gracefully. Deployed fix in v3.2.2."),
    
    # Useful: different issue
    ("create", "TICK-103", "Payment processing timeout on Stripe integration. Transactions over $500 are failing with gateway timeout after 30 seconds. Error code: STRIPE_TIMEOUT_001. Affects checkout flow."),
    
    # Noise: closing
    ("create", "TICK-101-status-2", "Status changed from In Progress to Resolved"),
    
    # Update: ticket text changes
    ("update", "TICK-103", "Payment processing timeout on Stripe integration. Transactions over $500 are failing with gateway timeout after 30 seconds. Error code: STRIPE_TIMEOUT_001. UPDATE: Also affects PayPal transactions. Workaround: reduce timeout to 15 seconds."),
    
    # Noise: thank you
    ("create", "TICK-101-comment-4", "Thanks, closing this ticket."),
    
    # Useful: new related issue
    ("create", "TICK-104", "Users unable to reset password. The reset email is sent but the reset link returns 404. The link format changed in v3.2.1. URL pattern: /auth/reset?token=xxx should be /api/auth/reset?token=xxx."),
    
    # Delete: old resolved ticket
    ("delete", "TICK-101-status-1", ""),
    ("delete", "TICK-101-status-2", ""),
]


def run_demo():
    """Run the streaming demo."""
    print("=" * 70)
    print("  V3 CSS-GATED STREAMING DEMO")
    print("  Simulating a support ticketing system")
    print("=" * 70)
    
    # Create adapter with CSS gating enabled
    adapter = StreamAdapter(css_gate=True, css_threshold=0.3)
    
    # Process events
    print(f"\n📥 Processing {len(TICKET_EVENTS)} ticket events...\n")
    
    for i, (event_type, doc_id, text) in enumerate(TICKET_EVENTS, 1):
        if event_type == "create":
            result = adapter.on_data_created(doc_id, text, source="ticket_system")
            status = "✅" if result.action == "added" else "❌"
            print(f"  {i:2d}. [{status} {result.action:>10}] {doc_id:<25} "
                  f"({result.latency_ms:.1f}ms) {result.reason[:50]}")
        
        elif event_type == "update":
            result = adapter.on_data_updated(doc_id, text, source="ticket_system")
            print(f"  {i:2d}. [🔄 {result.action:>10}] {doc_id:<25} "
                  f"({result.latency_ms:.1f}ms) {result.reason[:50]}")
        
        elif event_type == "delete":
            removed = adapter.on_data_deleted(doc_id)
            status = "🗑️" if removed else "⚠️"
            print(f"  {i:2d}. [{status} {'deleted' if removed else 'not found':>10}] {doc_id}")
    
    # Print stats
    stats = adapter.get_stats()
    print(f"\n{'─' * 50}")
    print(f"📊 Ingestion Stats:")
    print(f"   Added:     {stats['total_ingested']}")
    print(f"   Updated:   {stats['total_updated']}")
    print(f"   Rejected:  {stats['total_rejected']} (CSS gate filtered)")
    print(f"   Deleted:   {stats['total_deleted']}")
    print(f"   Unchanged: {stats['total_unchanged']}")
    print(f"   Active:    {stats['active_documents']} documents in index")
    print(f"   FAISS:     {stats['faiss_vectors']} vectors")
    
    # Query against live data
    print(f"\n{'=' * 70}")
    print("  🔍 QUERYING LIVE DATA")
    print(f"{'=' * 70}\n")
    
    queries = [
        "What causes login 500 errors?",
        "How do I fix payment timeout issues?",
        "What changed in version 3.2.1?",
    ]
    
    for query in queries:
        print(f"  Q: \"{query}\"")
        result = adapter.query(query, top_k=3)
        
        print(f"  CSS Score: {result['css_final']:.4f}  |  Latency: {result['latency_ms']:.1f}ms")
        print(f"  Retrieved {result['n_results']} chunks:")
        for doc_id, preview in result["nodes"]:
            print(f"    • [{doc_id}] {preview}...")
        print()
    
    # Show version history for updated ticket
    print(f"{'=' * 70}")
    print("  📜 VERSION HISTORY: TICK-103")
    print(f"{'=' * 70}\n")
    
    history = adapter.get_document_history("TICK-103")
    for entry in history:
        print(f"  v{entry.version} [{entry.action}] {entry.timestamp}")
        print(f"    {entry.text_preview}...")
        print()


if __name__ == "__main__":
    run_demo()
