"""FrozenState holds immutable tools and corpus for deterministic behavior.

Now includes FAISS index, pre-computed IDF, and cross-reference edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
    - FAISS index for O(log n) similarity search (or ChromaDB / Pinecone when set)
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
    pinecone_index: Optional[object] = None  # Pinecone Index when using persistent Pinecone store

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

    @classmethod
    def build_from_pinecone(
        cls,
        index_name: str,
        corpus_cache_path: str | None = None,
        idf_cache_path: str | None = None,
        cross_ref_cache_path: str | None = None,
    ) -> "FrozenState":
        """Build FrozenState backed by a persistent Pinecone index.

        Uses Pinecone for similarity search. Corpus text is loaded from the
        JSON cache written by scripts/load_cuad_to_pinecone.py.

        Args:
            index_name: Pinecone index name (must already exist).
            corpus_cache_path: Path to pinecone_corpus_cache.json (id → {title, text, source}).
                               Defaults to data/pinecone_corpus_cache.json.
            idf_cache_path: Path to idf_dict.json. Defaults to data/idf_dict.json.
            cross_ref_cache_path: Path to cross_ref_edges.json. Defaults to data/cross_ref_edges.json.
        """
        import os
        from pinecone import Pinecone

        # Load .env / .env.txt via project's config module
        try:
            from config.settings import _load_env
            _load_env()
        except Exception:
            pass

        data_dir = Path(__file__).resolve().parent.parent / "data"
        corpus_cache_path = corpus_cache_path or str(data_dir / "pinecone_corpus_cache.json")
        idf_cache_path = idf_cache_path or str(data_dir / "idf_dict.json")
        cross_ref_cache_path = cross_ref_cache_path or str(data_dir / "cross_ref_edges.json")

        # Connect to Pinecone
        api_key = os.environ.get("PINECONE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY environment variable not set.")
        pc = Pinecone(api_key=api_key)
        pinecone_index = pc.Index(index_name)
        stats = pinecone_index.describe_index_stats()
        print(f"[FrozenState] Connected to Pinecone index '{index_name}' "
              f"({stats.total_vector_count} vectors).")

        # Load corpus from JSON cache
        corpus_cache_file = Path(corpus_cache_path)
        if not corpus_cache_file.exists():
            raise FileNotFoundError(
                f"Corpus cache not found: {corpus_cache_path}\n"
                f"Run: python -m scripts.load_cuad_to_pinecone first."
            )
        corpus_cache = json.load(corpus_cache_file.open(encoding="utf-8"))
        from core.types import CorpusDocument
        corpus_docs = [
            CorpusDocument(
                id=doc_id,
                title=entry["title"],
                text=entry["text"],
                source=entry.get("source", ""),
            )
            for doc_id, entry in corpus_cache.items()
        ]
        corpus = Corpus(documents=corpus_docs)
        print(f"[FrozenState] Loaded {len(corpus_docs)} documents from corpus cache.")

        # Load IDF dict from cache
        idf_dict: Dict[str, float] = {}
        idf_file = Path(idf_cache_path)
        if idf_file.exists():
            idf_dict = json.load(idf_file.open(encoding="utf-8"))
            print(f"[FrozenState] Loaded IDF dict ({len(idf_dict)} terms) from cache.")
        else:
            print(f"[FrozenState] IDF cache not found, computing from corpus...")
            from graph.edge_builder import compute_corpus_idf
            idf_dict = compute_corpus_idf(corpus.documents)

        # Load cross-ref edges from cache
        precomputed_edges = []
        cr_file = Path(cross_ref_cache_path)
        if cr_file.exists():
            raw_edges = json.load(cr_file.open(encoding="utf-8"))
            # Stored as [[src, tgt, type], ...] — convert back to tuples
            precomputed_edges = [tuple(e) for e in raw_edges]
            print(f"[FrozenState] Loaded {len(precomputed_edges)} cross-ref edges from cache.")
        else:
            print(f"[FrozenState] Cross-ref cache not found, skipping precomputation.")

        embedder = NeuralEmbedder()

        return cls(
            corpus=corpus,
            parser=ParserStub(),
            embedder=embedder,
            nli=NLIStub(),
            contradiction=ContradictionStub(),
            corpus_embeddings={},  # Not pre-loaded; Pinecone used for similarity search
            faiss_index=None,
            doc_id_list=[],
            idf_dict=idf_dict,
            precomputed_edges=precomputed_edges,
            chroma_store=None,
            pinecone_index=pinecone_index,
        )

    def get_doc_embedding(self, doc_id: str) -> List[float]:
        """Get pre-computed embedding for a document."""
        return self.corpus_embeddings.get(doc_id, [])
    
    def search_similar(self, query_vec: List[float], top_k: int = 5, exclude_ids: set = None,
                       filter_metadata: dict = None) -> List[tuple]:
        """Fast similarity search using Pinecone, FAISS, ChromaDB, or fallback.

        Args:
            filter_metadata: Optional Pinecone metadata filter, e.g.
                             {"contract": {"$eq": "ENERGOUSCORP_..."}}
        Returns: List of (doc_id, score) tuples (score higher = more similar)
        """
        exclude_ids = exclude_ids or set()

        if self.pinecone_index is not None:
            # Use Pinecone for similarity search
            search_k = min(top_k + len(exclude_ids) + 20, 1000)
            query_kwargs = {
                "vector": query_vec,
                "top_k": search_k,
                "include_metadata": False,
            }
            if filter_metadata:
                query_kwargs["filter"] = filter_metadata
            results = self.pinecone_index.query(**query_kwargs)
            out = []
            for match in results.matches:
                if match.id in exclude_ids:
                    continue
                out.append((match.id, float(match.score)))
                if len(out) >= top_k:
                    break
            return out

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
_CACHED_PINECONE_INDEX: str | None = None


def get_shared_state(
    *,
    corpus: Corpus | None = None,
    chroma_path: str | None = None,
    pinecone_index: str | None = None,
    refresh: bool = False,
) -> FrozenState:
    """Return a cached FrozenState to avoid reloading models per query.

    Args:
        corpus: Corpus to use (ignored if chroma_path or pinecone_index set).
        chroma_path: Path to persistent ChromaDB. When set, uses Chroma for retrieval.
        pinecone_index: Pinecone index name. When set, uses Pinecone for retrieval.
                        Takes precedence over chroma_path.
        refresh: Force rebuild of state.
    """
    global _CACHED_STATE, _CACHED_CHROMA_PATH, _CACHED_PINECONE_INDEX

    if pinecone_index:
        if refresh or _CACHED_STATE is None or _CACHED_PINECONE_INDEX != pinecone_index:
            _CACHED_STATE = FrozenState.build_from_pinecone(pinecone_index)
            _CACHED_PINECONE_INDEX = pinecone_index
            _CACHED_CHROMA_PATH = None
        return _CACHED_STATE

    if refresh or _CACHED_STATE is None:
        if chroma_path:
            _CACHED_STATE = FrozenState.build_from_chroma(chroma_path)
            _CACHED_CHROMA_PATH = chroma_path
        else:
            _CACHED_STATE = FrozenState.build(corpus)
            _CACHED_CHROMA_PATH = None
        _CACHED_PINECONE_INDEX = None
    elif chroma_path and chroma_path != _CACHED_CHROMA_PATH:
        _CACHED_STATE = FrozenState.build_from_chroma(chroma_path)
        _CACHED_CHROMA_PATH = chroma_path
        _CACHED_PINECONE_INDEX = None
    return _CACHED_STATE


def clear_shared_state() -> None:
    """Clear the cached FrozenState."""
    global _CACHED_STATE
    _CACHED_STATE = None
