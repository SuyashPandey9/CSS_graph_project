"""FrozenState holds immutable tools and corpus for deterministic behavior.

Now includes FAISS index, pre-computed IDF, and cross-reference edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.types import Corpus
from storage.corpus_store import load_active_corpus
from tools.base import BaseEmbedder
from tools.contradiction_stub import ContradictionStub
from tools.neural_embedder import NeuralEmbedder
from tools.nli_stub import NLIStub
from tools.parser_stub import ParserStub

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class FrozenState:
    """Container for frozen, deterministic tools and corpus.
    
    Includes:
    - Pre-computed corpus embeddings
    - FAISS index for O(log n) similarity search (or ChromaDB when chroma_store set)
    - Pre-computed IDF dictionary for entity overlap (Fix 1c.1)
    - Pre-computed cross-reference edges (Fix 3.3 + 2.10)
    """

    corpus: Corpus
    parser: ParserStub
    embedder: BaseEmbedder
    nli: NLIStub
    contradiction: ContradictionStub
    corpus_embeddings: Dict[str, List[float]]
    faiss_index: Optional[object]  # faiss.IndexFlatIP
    doc_id_list: List[str]  # Maps FAISS index position to doc_id
    idf_dict: Dict[str, float]  # Pre-computed IDF per term (Fix 1c.1)
    precomputed_edges: List[object]  # Pre-computed cross-ref edges (Fix 3.3)
    chroma_store: Optional[object] = None  # Chroma vector store when using persistent ChromaDB

    @classmethod
    def build(cls, corpus: Corpus | None = None, *, use_chunking: bool = True) -> "FrozenState":
        """Initialize tools and load the active or provided corpus.
        
        Pre-computes embeddings and builds FAISS index for fast retrieval.
        
        Args:
            corpus: Corpus to use (loads active corpus if None)
            use_chunking: If True, chunk documents same as Traditional RAG (500 chars)
        """
        from core.types import CorpusDocument
        
        corpus = corpus or load_active_corpus()
        embedder = NeuralEmbedder()
        
        # Optionally chunk the corpus to match Traditional RAG
        if use_chunking:
            from tools.text_chunker import chunk_corpus_documents
            print(f"[FrozenState] Chunking {len(corpus.documents)} documents (500 chars each)...")
            chunks = chunk_corpus_documents(corpus.documents)
            # Convert chunks to CorpusDocument objects
            chunked_docs = [
                CorpusDocument(
                    id=chunk.id,
                    title=f"{chunk.source_title} (chunk {chunk.chunk_index})",
                    text=chunk.text,
                    source=chunk.source_doc_id,
                )
                for chunk in chunks
            ]
            # Replace corpus with chunked version
            corpus = Corpus(documents=chunked_docs)
            print(f"[FrozenState] Created {len(chunked_docs)} chunks from documents.")
        
        # Pre-compute all corpus embeddings
        print(f"[FrozenState] Pre-computing embeddings for {len(corpus.documents)} items...")
        corpus_embeddings: Dict[str, List[float]] = {}
        embedding_matrix = []
        doc_id_list = []
        
        for i, doc in enumerate(corpus.documents):
            vec = embedder.embed(doc.text)
            corpus_embeddings[doc.id] = vec
            embedding_matrix.append(vec)
            doc_id_list.append(doc.id)
            if (i + 1) % 100 == 0:
                print(f"[FrozenState] Embedded {i + 1}/{len(corpus.documents)} items")
        
        print(f"[FrozenState] Done embedding all items.")
        
        # Build FAISS index
        faiss_index = None
        if FAISS_AVAILABLE and embedding_matrix:
            print(f"[FrozenState] Building FAISS index...")
            dim = len(embedding_matrix[0])
            # Use Inner Product (cosine similarity for normalized vectors)
            faiss_index = faiss.IndexFlatIP(dim)
            # Normalize vectors for cosine similarity
            matrix = np.array(embedding_matrix, dtype=np.float32)
            faiss.normalize_L2(matrix)
            faiss_index.add(matrix)
            print(f"[FrozenState] FAISS index built with {faiss_index.ntotal} vectors.")
        elif not FAISS_AVAILABLE:
            print(f"[FrozenState] FAISS not available, using linear search fallback.")
        
        # Pre-compute IDF dictionary (Fix 1c.1)
        print(f"[FrozenState] Computing corpus IDF...")
        from graph.edge_builder import compute_corpus_idf
        idf_dict = compute_corpus_idf(corpus.documents)
        print(f"[FrozenState] IDF computed for {len(idf_dict)} terms.")
        
        # Pre-compute cross-reference edges (Fix 3.3 + 2.10)
        print(f"[FrozenState] Pre-computing cross-reference edges...")
        from graph.edge_builder import precompute_cross_ref_edges
        precomputed_edges = precompute_cross_ref_edges(corpus.documents)
        print(f"[FrozenState] Found {len(precomputed_edges)} cross-reference edges.")
        
        return cls(
            corpus=corpus,
            parser=ParserStub(),
            embedder=embedder,
            nli=NLIStub(),
            contradiction=ContradictionStub(),
            corpus_embeddings=corpus_embeddings,
            faiss_index=faiss_index,
            doc_id_list=doc_id_list,
            idf_dict=idf_dict,
            precomputed_edges=precomputed_edges,
            chroma_store=None,
        )

    @classmethod
    def build_from_chroma(
        cls,
        chroma_path: str,
        collection_name: str = "cuad_contracts",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "FrozenState":
        """Build FrozenState from a persistent ChromaDB (510-contract CUAD store).

        Uses Chroma for similarity search instead of FAISS. Corpus is built from
        Chroma's stored documents for graph nodes.
        """
        from pathlib import Path
        from core.types import CorpusDocument
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        p = Path(chroma_path)
        if not p.exists():
            raise FileNotFoundError(f"ChromaDB path not found: {chroma_path}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        chroma = Chroma(
            persist_directory=str(p),
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        embedder = NeuralEmbedder()

        # Build corpus from Chroma's stored documents
        result = chroma._collection.get(include=["documents", "metadatas"])
        ids = result["ids"]
        documents_raw = result["documents"]
        metadatas = result.get("metadatas") or [{}] * len(ids)

        corpus_docs = []
        for i, (doc_id, text) in enumerate(zip(ids, documents_raw)):
            if not text:
                continue
            meta = metadatas[i] if i < len(metadatas) else {}
            corpus_docs.append(
                CorpusDocument(
                    id=doc_id,
                    title=meta.get("title", f"doc_{i}"),
                    text=text,
                    source=meta.get("id", ""),
                )
            )
        corpus = Corpus(documents=corpus_docs)
        print(f"[FrozenState] Loaded {len(corpus_docs)} chunks from ChromaDB at {chroma_path}")

        # Sparse embeddings for graph nodes (optimizer uses top-k only)
        corpus_embeddings: Dict[str, List[float]] = {}

        # IDF and edges
        from graph.edge_builder import compute_corpus_idf, precompute_cross_ref_edges
        idf_dict = compute_corpus_idf(corpus.documents)
        # Cross-ref precomputation is O(n²) - infeasible for 100k+ chunks.
        # Use sampled precomputation: group by parent doc (source), run O(m²) per group
        # where m = chunks per doc (~20-50). Total ~510 * 50² = 1.3M pairs, not 5.5B.
        MAX_DOCS_FOR_FULL_CROSS_REF = 15000
        if len(corpus.documents) <= MAX_DOCS_FOR_FULL_CROSS_REF:
            precomputed_edges = precompute_cross_ref_edges(corpus.documents)
            print(f"[FrozenState] Found {len(precomputed_edges)} cross-reference edges.")
        else:
            # Sampled: only compute edges within same parent document (source)
            from collections import defaultdict
            by_source: dict = defaultdict(list)
            for doc in corpus.documents:
                src = doc.source or doc.id.split("_chunk_")[0] if "_chunk_" in doc.id else doc.id
                by_source[src].append(doc)
            precomputed_edges = []
            for docs in by_source.values():
                if len(docs) <= 1:
                    continue
                precomputed_edges.extend(precompute_cross_ref_edges(docs))
            print(f"[FrozenState] Sampled cross-ref: {len(precomputed_edges)} edges (within-doc only, {len(by_source)} docs)")

        return cls(
            corpus=corpus,
            parser=ParserStub(),
            embedder=embedder,
            nli=NLIStub(),
            contradiction=ContradictionStub(),
            corpus_embeddings=corpus_embeddings,
            faiss_index=None,
            doc_id_list=[],
            idf_dict=idf_dict,
            precomputed_edges=precomputed_edges,
            chroma_store=chroma,
        )

    def get_doc_embedding(self, doc_id: str) -> List[float]:
        """Get pre-computed embedding for a document."""
        return self.corpus_embeddings.get(doc_id, [])
    
    def search_similar(self, query_vec: List[float], top_k: int = 5, exclude_ids: set = None) -> List[tuple]:
        """Fast similarity search using FAISS, ChromaDB, or fallback.
        
        Returns: List of (doc_id, score) tuples (score higher = more similar)
        """
        exclude_ids = exclude_ids or set()

        if self.chroma_store is not None:
            # Use persistent ChromaDB for retrieval (query returns ids directly)
            search_k = min(top_k + len(exclude_ids) + 20, 100)
            import numpy as np
            query_arr = np.array([query_vec], dtype=np.float32)
            results = self.chroma_store._collection.query(
                query_embeddings=query_arr.tolist(),
                n_results=search_k,
                include=["documents", "metadatas", "distances"],
            )
            ids = results["ids"][0]
            distances = results["distances"][0]
            out = []
            for doc_id, dist in zip(ids, distances):
                if doc_id in exclude_ids:
                    continue
                # Chroma returns distance (lower=better). Convert to similarity in [0,1]
                # for Node confidence. Use 1/(1+dist) or clamp -dist to [0,1]
                raw = float(-dist)  # negate so higher = more similar
                score = max(0.0, min(1.0, raw))  # clamp for Node validation
                out.append((doc_id, score))
                if len(out) >= top_k:
                    break
            return out
        
        if self.faiss_index is not None and FAISS_AVAILABLE:
            # FAISS search - O(log n)
            query_arr = np.array([query_vec], dtype=np.float32)
            faiss.normalize_L2(query_arr)
            
            # Search more than top_k to account for exclusions
            search_k = min(top_k + len(exclude_ids) + 10, len(self.doc_id_list))
            scores, indices = self.faiss_index.search(query_arr, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:  # Invalid index
                    continue
                doc_id = self.doc_id_list[idx]
                if doc_id not in exclude_ids:
                    results.append((doc_id, float(score)))
                    if len(results) >= top_k:
                        break
            return results
        else:
            # Fallback: linear search - O(n)
            from transforms.utils import cosine_similarity
            scored = []
            for doc_id, doc_vec in self.corpus_embeddings.items():
                if doc_id in exclude_ids:
                    continue
                score = cosine_similarity(query_vec, doc_vec)
                scored.append((doc_id, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]


_CACHED_STATE: FrozenState | None = None
_CACHED_CHROMA_PATH: str | None = None


def get_shared_state(
    *,
    corpus: Corpus | None = None,
    chroma_path: str | None = None,
    refresh: bool = False,
) -> FrozenState:
    """Return a cached FrozenState to avoid reloading models per query.

    Args:
        corpus: Corpus to use (ignored if chroma_path set)
        chroma_path: Path to persistent ChromaDB. When set, uses Chroma for retrieval.
        refresh: Force rebuild of state
    """
    global _CACHED_STATE, _CACHED_CHROMA_PATH
    if refresh or _CACHED_STATE is None:
        if chroma_path:
            _CACHED_STATE = FrozenState.build_from_chroma(chroma_path)
            _CACHED_CHROMA_PATH = chroma_path
        else:
            _CACHED_STATE = FrozenState.build(corpus)
            _CACHED_CHROMA_PATH = None
    elif chroma_path and chroma_path != _CACHED_CHROMA_PATH:
        _CACHED_STATE = FrozenState.build_from_chroma(chroma_path)
        _CACHED_CHROMA_PATH = chroma_path
    return _CACHED_STATE


def clear_shared_state() -> None:
    """Clear the cached FrozenState."""
    global _CACHED_STATE
    _CACHED_STATE = None
