"""StreamingState: Mutable corpus state for CSS-gated live data ingestion.

Extends the FrozenState interface (same search_similar, get_doc_embedding)
but adds CRUD operations for live data updates.

Key differences from FrozenState:
    - Uses FAISS IndexIDMap (supports add_with_ids + remove_ids)
    - Maintains a metadata registry tracking versions and change history
    - Supports ingest(), remove(), detect_change() for streaming data

All downstream modules (expand, optimizer, greedy_policy, features)
call search_similar() which has an identical interface — zero changes needed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.types import Corpus, CorpusDocument
from tools.neural_embedder import NeuralEmbedder

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class VersionEntry:
    """A single version snapshot of a document."""
    version: int
    text_hash: str
    timestamp: str
    action: str  # "created", "updated", "deleted"
    text_preview: str = ""  # first 100 chars for debugging


@dataclass
class RegistryEntry:
    """Metadata registry entry for a tracked document."""
    doc_id: str
    text: str
    text_hash: str
    embedding: List[float]
    faiss_id: int
    version: int
    created_at: str
    updated_at: str
    source: str = ""
    title: str = ""
    is_active: bool = True
    history: List[VersionEntry] = field(default_factory=list)


@dataclass 
class IngestResult:
    """Result of an ingest operation."""
    doc_id: str
    action: str  # "added", "updated", "unchanged", "rejected"
    version: int
    old_text_hash: Optional[str] = None
    new_text_hash: Optional[str] = None
    css_score: Optional[float] = None
    reason: str = ""
    latency_ms: float = 0.0


class StreamingState:
    """Mutable corpus state that supports CSS-gated live data ingestion.
    
    Exposes the same interface as FrozenState:
        - search_similar(query_vec, top_k, exclude_ids) -> List[(doc_id, score)]
        - get_doc_embedding(doc_id) -> List[float]
        - corpus: Corpus
        - embedder: BaseEmbedder
        - corpus_embeddings: Dict[str, List[float]]
    
    Plus new streaming methods:
        - ingest(doc_id, text) -> IngestResult
        - remove(doc_id) -> bool
        - detect_change(doc_id, text) -> str
        - get_version_history(doc_id) -> List[VersionEntry]
    """
    
    def __init__(self, embedder: NeuralEmbedder = None, dim: int = 384):
        """Initialize an empty streaming state.
        
        Args:
            embedder: Embedder instance (creates one if None)
            dim: Embedding dimension (384 for MiniLM)
        """
        self.embedder = embedder or NeuralEmbedder()
        self._dim = dim
        
        # Registry: doc_id → RegistryEntry
        self._registry: Dict[str, RegistryEntry] = {}
        
        # FAISS IndexIDMap for add/remove by ID
        if FAISS_AVAILABLE:
            base_index = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIDMap(base_index)
        else:
            self.faiss_index = None
        
        # ID mappings
        self._next_faiss_id: int = 0
        self._doc_id_to_faiss_id: Dict[str, int] = {}
        self._faiss_id_to_doc_id: Dict[int, str] = {}
        
        # Corpus-compatible attributes (for downstream module compatibility)
        self.corpus_embeddings: Dict[str, List[float]] = {}
        self._documents: Dict[str, CorpusDocument] = {}
        
        # Statistics
        self._stats = {
            "total_ingested": 0,
            "total_updated": 0,
            "total_rejected": 0,
            "total_deleted": 0,
            "total_unchanged": 0,
        }
    
    # ================================================================
    # FrozenState-compatible interface (downstream modules use these)
    # ================================================================
    
    @property
    def corpus(self) -> Corpus:
        """Return a Corpus object (compatible with FrozenState.corpus)."""
        return Corpus(documents=list(self._documents.values()))
    
    @property
    def doc_id_list(self) -> List[str]:
        """List of active document IDs (compatible with FrozenState.doc_id_list)."""
        return [doc_id for doc_id, entry in self._registry.items() if entry.is_active]
    
    def get_doc_embedding(self, doc_id: str) -> List[float]:
        """Get pre-computed embedding for a document."""
        return self.corpus_embeddings.get(doc_id, [])
    
    def search_similar(
        self, query_vec: List[float], top_k: int = 5, exclude_ids: set = None
    ) -> List[Tuple[str, float]]:
        """Fast similarity search using FAISS IndexIDMap.
        
        IDENTICAL interface to FrozenState.search_similar().
        
        Returns:
            List of (doc_id, score) tuples
        """
        exclude_ids = exclude_ids or set()
        
        if self.faiss_index is not None and FAISS_AVAILABLE and self.faiss_index.ntotal > 0:
            query_arr = np.array([query_vec], dtype=np.float32)
            faiss.normalize_L2(query_arr)
            
            search_k = min(top_k + len(exclude_ids) + 10, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_arr, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                doc_id = self._faiss_id_to_doc_id.get(int(idx))
                if doc_id and doc_id not in exclude_ids:
                    # Check the doc is still active
                    entry = self._registry.get(doc_id)
                    if entry and entry.is_active:
                        results.append((doc_id, float(score)))
                        if len(results) >= top_k:
                            break
            return results
        else:
            # Fallback: linear search
            from transforms.utils import cosine_similarity
            scored = []
            for doc_id, doc_vec in self.corpus_embeddings.items():
                if doc_id in exclude_ids:
                    continue
                entry = self._registry.get(doc_id)
                if entry and not entry.is_active:
                    continue
                score = cosine_similarity(query_vec, doc_vec)
                scored.append((doc_id, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
    
    # ================================================================
    # Streaming CRUD methods (new capability)
    # ================================================================
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute SHA-256 hash of text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    
    def detect_change(self, doc_id: str, text: str) -> str:
        """Detect what kind of change this represents.
        
        Returns:
            "new" - doc_id not seen before
            "updated" - doc_id exists but text changed
            "unchanged" - doc_id exists with same text
        """
        if doc_id not in self._registry:
            return "new"
        
        entry = self._registry[doc_id]
        new_hash = self._hash_text(text)
        
        if entry.text_hash == new_hash:
            return "unchanged"
        else:
            return "updated"
    
    def ingest(
        self, doc_id: str, text: str, source: str = "", title: str = "",
        css_gate: bool = False, css_threshold: float = 0.3,
        recent_queries: List[str] = None,
    ) -> IngestResult:
        """Add or update a single document in the streaming index.
        
        This is the primary entry point for live data.
        
        Args:
            doc_id: Unique document identifier
            text: Document text content
            source: Source identifier (e.g., "ticket_system")
            title: Document title
            css_gate: If True, evaluate CSS quality before accepting
            css_threshold: Minimum CSS score to accept (when css_gate=True)
            recent_queries: Recent user queries for CSS relevance scoring
        
        Returns:
            IngestResult with action taken and metadata
        """
        start = time.time()
        new_hash = self._hash_text(text)
        change_type = self.detect_change(doc_id, text)
        now = datetime.now().isoformat()
        
        # Case 1: Unchanged — skip
        if change_type == "unchanged":
            self._stats["total_unchanged"] += 1
            return IngestResult(
                doc_id=doc_id,
                action="unchanged",
                version=self._registry[doc_id].version,
                reason="Content unchanged (same hash)",
                latency_ms=(time.time() - start) * 1000,
            )
        
        # Embed the new text
        embedding = self.embedder.embed(text)
        
        # CSS quality gate (optional)
        if css_gate:
            from core.ingestion_gate import evaluate_for_ingestion
            decision = evaluate_for_ingestion(
                new_text=text,
                new_embedding=embedding,
                streaming_state=self,
                recent_queries=recent_queries,
                threshold=css_threshold,
            )
            if not decision["accept"]:
                self._stats["total_rejected"] += 1
                return IngestResult(
                    doc_id=doc_id,
                    action="rejected",
                    version=0,
                    new_text_hash=new_hash,
                    css_score=decision.get("score", 0),
                    reason=decision.get("reason", "Below CSS threshold"),
                    latency_ms=(time.time() - start) * 1000,
                )
        
        # Case 2: Update existing document
        if change_type == "updated":
            old_entry = self._registry[doc_id]
            old_hash = old_entry.text_hash
            
            # Archive old version
            old_entry.history.append(VersionEntry(
                version=old_entry.version,
                text_hash=old_hash,
                timestamp=old_entry.updated_at,
                action="updated",
                text_preview=old_entry.text[:100],
            ))
            
            # Remove old vector from FAISS
            self._remove_from_faiss(doc_id)
            
            # Add new vector
            faiss_id = self._add_to_faiss(doc_id, embedding)
            
            # Update registry
            new_version = old_entry.version + 1
            old_entry.text = text
            old_entry.text_hash = new_hash
            old_entry.embedding = embedding
            old_entry.faiss_id = faiss_id
            old_entry.version = new_version
            old_entry.updated_at = now
            
            # Update corpus document
            self._documents[doc_id] = CorpusDocument(
                id=doc_id, title=title or old_entry.title,
                text=text, source=source or old_entry.source,
            )
            self.corpus_embeddings[doc_id] = embedding
            
            self._stats["total_updated"] += 1
            
            return IngestResult(
                doc_id=doc_id,
                action="updated",
                version=new_version,
                old_text_hash=old_hash,
                new_text_hash=new_hash,
                reason=f"Updated from v{new_version - 1} to v{new_version}",
                latency_ms=(time.time() - start) * 1000,
            )
        
        # Case 3: New document
        faiss_id = self._add_to_faiss(doc_id, embedding)
        
        entry = RegistryEntry(
            doc_id=doc_id,
            text=text,
            text_hash=new_hash,
            embedding=embedding,
            faiss_id=faiss_id,
            version=1,
            created_at=now,
            updated_at=now,
            source=source,
            title=title,
            is_active=True,
            history=[VersionEntry(
                version=1, text_hash=new_hash,
                timestamp=now, action="created",
                text_preview=text[:100],
            )],
        )
        
        self._registry[doc_id] = entry
        self._documents[doc_id] = CorpusDocument(
            id=doc_id, title=title, text=text, source=source,
        )
        self.corpus_embeddings[doc_id] = embedding
        
        self._stats["total_ingested"] += 1
        
        return IngestResult(
            doc_id=doc_id,
            action="added",
            version=1,
            new_text_hash=new_hash,
            reason="New document added",
            latency_ms=(time.time() - start) * 1000,
        )
    
    def remove(self, doc_id: str) -> bool:
        """Remove a document from the active index.
        
        The document is soft-deleted (marked inactive) and removed from FAISS.
        Version history is preserved.
        
        Args:
            doc_id: Document to remove
        
        Returns:
            True if removed, False if not found
        """
        if doc_id not in self._registry:
            return False
        
        entry = self._registry[doc_id]
        
        # Archive the deletion
        entry.history.append(VersionEntry(
            version=entry.version,
            text_hash=entry.text_hash,
            timestamp=datetime.now().isoformat(),
            action="deleted",
            text_preview=entry.text[:100],
        ))
        
        # Remove from FAISS
        self._remove_from_faiss(doc_id)
        
        # Mark inactive (keep in registry for history)
        entry.is_active = False
        
        # Remove from active corpus
        self._documents.pop(doc_id, None)
        self.corpus_embeddings.pop(doc_id, None)
        
        self._stats["total_deleted"] += 1
        return True
    
    def get_version_history(self, doc_id: str) -> List[VersionEntry]:
        """Get full version history for a document."""
        entry = self._registry.get(doc_id)
        if not entry:
            return []
        return list(entry.history)
    
    def get_current_version(self, doc_id: str) -> Optional[RegistryEntry]:
        """Get current registry entry for a document."""
        return self._registry.get(doc_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            **self._stats,
            "active_documents": sum(1 for e in self._registry.values() if e.is_active),
            "total_tracked": len(self._registry),
            "faiss_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
        }
    
    # ================================================================
    # Bulk loading (for initial corpus setup)
    # ================================================================
    
    def load_corpus(self, corpus: Corpus) -> None:
        """Bulk load an entire corpus (for initial setup).
        
        This is equivalent to calling ingest() for each document,
        but more efficient (batch embedding).
        
        Args:
            corpus: Corpus to load
        """
        print(f"[StreamingState] Loading {len(corpus.documents)} documents...")
        
        for i, doc in enumerate(corpus.documents):
            self.ingest(
                doc_id=doc.id,
                text=doc.text,
                source=doc.source or "",
                title=doc.title,
                css_gate=False,  # No quality gate for initial bulk load
            )
            if (i + 1) % 100 == 0:
                print(f"[StreamingState] Loaded {i + 1}/{len(corpus.documents)}")
        
        stats = self.get_stats()
        print(f"[StreamingState] Done. {stats['active_documents']} active, "
              f"{stats['faiss_vectors']} vectors in FAISS.")
    
    @classmethod
    def from_corpus(cls, corpus: Corpus = None, *, use_chunking: bool = True) -> "StreamingState":
        """Create a StreamingState from a corpus (mirrors FrozenState.build).
        
        Args:
            corpus: Corpus to load (loads active corpus if None)
            use_chunking: If True, chunk documents (same as FrozenState)
        
        Returns:
            StreamingState pre-loaded with corpus data
        """
        from storage.corpus_store import load_active_corpus
        
        corpus = corpus or load_active_corpus()
        
        if use_chunking:
            from tools.text_chunker import chunk_corpus_documents
            print(f"[StreamingState] Chunking {len(corpus.documents)} documents...")
            chunks = chunk_corpus_documents(corpus.documents)
            chunked_docs = [
                CorpusDocument(
                    id=chunk.id,
                    title=f"{chunk.source_title} (chunk {chunk.chunk_index})",
                    text=chunk.text,
                    source=chunk.source_doc_id,
                )
                for chunk in chunks
            ]
            corpus = Corpus(documents=chunked_docs)
            print(f"[StreamingState] Created {len(chunked_docs)} chunks.")
        
        state = cls()
        state.load_corpus(corpus)
        return state
    
    # ================================================================
    # Internal FAISS helpers
    # ================================================================
    
    def _add_to_faiss(self, doc_id: str, embedding: List[float]) -> int:
        """Add a vector to the FAISS index with a tracked numeric ID."""
        faiss_id = self._next_faiss_id
        self._next_faiss_id += 1
        
        if self.faiss_index is not None:
            arr = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(arr)
            self.faiss_index.add_with_ids(arr, np.array([faiss_id], dtype=np.int64))
        
        self._doc_id_to_faiss_id[doc_id] = faiss_id
        self._faiss_id_to_doc_id[faiss_id] = doc_id
        
        return faiss_id
    
    def _remove_from_faiss(self, doc_id: str) -> None:
        """Remove a vector from the FAISS index by doc_id."""
        faiss_id = self._doc_id_to_faiss_id.get(doc_id)
        if faiss_id is None:
            return
        
        if self.faiss_index is not None:
            self.faiss_index.remove_ids(np.array([faiss_id], dtype=np.int64))
        
        del self._doc_id_to_faiss_id[doc_id]
        del self._faiss_id_to_doc_id[faiss_id]


# Module-level cache (mirrors get_shared_state pattern)
_CACHED_STREAMING: StreamingState | None = None


def get_streaming_state(
    *, corpus: Corpus | None = None, refresh: bool = False
) -> StreamingState:
    """Return a cached StreamingState instance."""
    global _CACHED_STREAMING
    if refresh or _CACHED_STREAMING is None:
        _CACHED_STREAMING = StreamingState.from_corpus(corpus)
    return _CACHED_STREAMING


def clear_streaming_state() -> None:
    """Clear the cached StreamingState."""
    global _CACHED_STREAMING
    _CACHED_STREAMING = None
