"""Streaming Adapter for CSS-gated live data ingestion.

Connects external data event streams (ticketing, CRM, logs, etc.)
to V3's CSS-gated streaming pipeline.

Usage:
    adapter = StreamAdapter()
    adapter.load_initial_data(corpus)
    
    # Live events
    adapter.on_data_created("TICK-101", "Login error on Chrome...")
    adapter.on_data_updated("TICK-101", "Fixed: session token expired")
    adapter.on_data_deleted("TICK-101")
    
    # Query against live data
    result = adapter.query("What causes login errors?")
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from core.streaming_state import IngestResult, StreamingState
from core.types import Corpus, Graph, Node


class StreamAdapter:
    """Adapts external data events to V3's CSS-gated streaming pipeline."""
    
    def __init__(
        self,
        css_gate: bool = True,
        css_threshold: float = 0.3,
    ):
        """Initialize the streaming adapter.
        
        Args:
            css_gate: Whether to use CSS quality gating on incoming data
            css_threshold: Minimum CSS score to accept a chunk (0-1)
        """
        self._state = StreamingState()
        self._css_gate = css_gate
        self._css_threshold = css_threshold
        self._recent_queries: List[str] = []
        self._event_log: List[Dict] = []
    
    @property
    def state(self) -> StreamingState:
        """Access the underlying StreamingState (for advanced usage)."""
        return self._state
    
    def load_initial_data(self, corpus: Corpus) -> None:
        """Load initial corpus data (no CSS gating for bulk load)."""
        self._state.load_corpus(corpus)
    
    # ================================================================
    # Data Event Handlers
    # ================================================================
    
    def on_data_created(
        self, doc_id: str, text: str, source: str = "", title: str = ""
    ) -> IngestResult:
        """Handle: new data created.
        
        Args:
            doc_id: Unique document ID
            text: Document text content
            source: Source system identifier
            title: Document title
        
        Returns:
            IngestResult with action taken
        """
        result = self._state.ingest(
            doc_id=doc_id,
            text=text,
            source=source,
            title=title,
            css_gate=self._css_gate,
            css_threshold=self._css_threshold,
            recent_queries=self._recent_queries,
        )
        
        self._log_event("created", doc_id, result)
        return result
    
    def on_data_updated(
        self, doc_id: str, new_text: str, source: str = "", title: str = ""
    ) -> IngestResult:
        """Handle: existing data updated.
        
        Automatically detects if content actually changed.
        If unchanged, returns early. If changed, re-embeds and replaces.
        
        Args:
            doc_id: Document ID to update
            new_text: Updated text content
        
        Returns:
            IngestResult with action (updated/unchanged/rejected)
        """
        result = self._state.ingest(
            doc_id=doc_id,
            text=new_text,
            source=source,
            title=title,
            css_gate=self._css_gate,
            css_threshold=self._css_threshold,
            recent_queries=self._recent_queries,
        )
        
        self._log_event("updated", doc_id, result)
        return result
    
    def on_data_deleted(self, doc_id: str) -> bool:
        """Handle: data deleted/archived.
        
        Removes from FAISS index but preserves version history.
        
        Args:
            doc_id: Document ID to remove
        
        Returns:
            True if removed, False if not found
        """
        removed = self._state.remove(doc_id)
        
        self._log_event("deleted", doc_id, None if not removed else "removed")
        return removed
    
    # ================================================================
    # Query Interface
    # ================================================================
    
    def query(self, query: str, top_k: int = 5) -> Dict:
        """Query against the live streaming index.
        
        Uses the standard V3 pipeline (search → build graph → optimize → CSS).
        
        Args:
            query: User's question
            top_k: Number of initial chunks to retrieve
        
        Returns:
            Dict with graph, nodes, and CSS scores
        """
        # Track query for CSS relevance scoring of future ingestions
        self._recent_queries.append(query)
        self._recent_queries = self._recent_queries[-20:]  # Keep last 20
        
        start = time.time()
        
        # Embed query
        query_vec = self._state.embedder.embed(query)
        
        # Retrieve similar chunks from live index
        results = self._state.search_similar(query_vec, top_k=top_k)
        
        # Build graph from results
        nodes = []
        for doc_id, score in results:
            entry = self._state.get_current_version(doc_id)
            if entry:
                nodes.append(Node(
                    id=doc_id,
                    text=entry.text,
                    metadata={
                        "title": entry.title,
                        "source": entry.source,
                        "version": entry.version,
                    },
                    confidence=score,
                    embedding=entry.embedding,
                ))
        
        # Build edges between retrieved nodes
        from graph.edge_builder import build_edges
        graph = build_edges(Graph(nodes=nodes, edges=[]), embedder=self._state.embedder)
        
        # Compute CSS score
        from css.calculator import compute_css_final
        css_result = compute_css_final(graph, query, Graph(), embedder=self._state.embedder)
        
        latency = (time.time() - start) * 1000
        
        return {
            "graph": graph,
            "nodes": [(n.id, n.text[:80]) for n in nodes],
            "css_score": css_result.get("css_score", 0),
            "css_final": css_result.get("css_final", 0),
            "latency_ms": latency,
            "n_results": len(results),
        }
    
    # ================================================================
    # Utilities
    # ================================================================
    
    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        return {
            **self._state.get_stats(),
            "recent_queries": len(self._recent_queries),
            "total_events": len(self._event_log),
        }
    
    def get_event_log(self, last_n: int = 20) -> List[Dict]:
        """Get recent event log entries."""
        return self._event_log[-last_n:]
    
    def get_document_history(self, doc_id: str) -> List:
        """Get version history for a specific document."""
        return self._state.get_version_history(doc_id)
    
    def _log_event(self, event_type: str, doc_id: str, result) -> None:
        """Log an event for debugging/auditing."""
        self._event_log.append({
            "event": event_type,
            "doc_id": doc_id,
            "result": str(result) if result else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
