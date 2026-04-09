"""Traditional RAG baseline using LangChain with neural embeddings.

Uses sentence-transformers for embeddings and ChromaDB (or Pinecone) for
vector storage. Same fixed corpus and Gemini LLM as V3 for fair comparison.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from core.types import Corpus
from llm.llm_client import generate_text
from storage.corpus_store import corpus_to_documents, load_active_corpus

# Shared embedding model name - used by both Traditional RAG and V3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default corpus cache written by scripts/load_cuad_to_pinecone.py
_DEFAULT_CORPUS_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "pinecone_corpus_cache.json"


def _build_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    documents: List[dict],
    pre_chunked: bool = False,
) -> Chroma:
    """Build a Chroma vector store from the corpus documents.

    Fix E3: Skip re-chunking when corpus is already chunked (pre_chunked=True).
    When V3 and TRAG both use the same pre-chunked corpus, TRAG must NOT apply
    RecursiveCharacterTextSplitter again — that would give TRAG smaller chunks
    than V3, making the comparison unfair.
    """
    texts = [f"{doc['title']}\n{doc['text']}" for doc in documents]
    metadatas = [{"id": doc["id"], "title": doc["title"]} for doc in documents]

    if pre_chunked:
        # Corpus already chunked — use as-is for fair comparison with V3
        split_texts = texts
        split_metadatas = metadatas
    else:
        # Fresh corpus (raw documents) — apply standard chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_texts: List[str] = []
        split_metadatas: List[dict] = []
        for text, meta in zip(texts, metadatas):
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                split_texts.append(chunk)
                split_metadatas.append({**meta, "chunk_idx": i})

    # Create in-memory Chroma store
    vector_store = Chroma.from_texts(
        texts=split_texts,
        embedding=embedding_model,
        metadatas=split_metadatas,
    )
    return vector_store


class TraditionalRAG:
    """Standard RAG: embed corpus, vector search, LLM generation."""

    def __init__(
        self,
        corpus: Corpus | None = None,
        chroma_path: str | None = None,
        collection_name: str = "cuad_contracts",
        pre_chunked: bool = False,
        pinecone_index_name: str | None = None,
        corpus_cache_path: str | None = None,
    ) -> None:
        """Initialize embeddings and vector store.

        Args:
            corpus: Corpus to build vector store from (ignored if chroma_path or
                    pinecone_index_name set).
            chroma_path: Path to persistent ChromaDB directory.
            collection_name: Chroma collection name when using chroma_path.
            pre_chunked: If True, corpus is already chunked — skip re-splitting.
                         Set True when sharing the same pre-chunked corpus as V3
                         to ensure fair comparison (Fix E3).
            pinecone_index_name: Pinecone index name. When set, uses Pinecone for
                                 retrieval instead of ChromaDB or in-memory Chroma.
                                 Takes precedence over chroma_path.
            corpus_cache_path: Path to pinecone_corpus_cache.json (needed when
                                pinecone_index_name is set to look up chunk text).
                                Defaults to data/pinecone_corpus_cache.json.
        """
        self._pinecone_index = None
        self._corpus_cache: dict = {}
        self._vector_store = None
        self._embedding_model = None

        if pinecone_index_name:
            # --- Pinecone backend ---
            # Load .env / .env.txt via project's config module
            try:
                from config.settings import _load_env
                _load_env()
            except Exception:
                pass
            api_key = os.environ.get("PINECONE_API_KEY", "")
            if not api_key:
                raise EnvironmentError("PINECONE_API_KEY environment variable not set.")
            from pinecone import Pinecone
            pc = Pinecone(api_key=api_key)
            self._pinecone_index = pc.Index(pinecone_index_name)
            # Load corpus cache for text lookup
            cache_file = Path(corpus_cache_path or _DEFAULT_CORPUS_CACHE)
            if cache_file.exists():
                self._corpus_cache = json.loads(cache_file.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(
                    f"Corpus cache not found: {cache_file}\n"
                    f"Run: python -m scripts.load_cuad_to_pinecone first."
                )
            # Still need embedding model for encoding queries
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        elif chroma_path:
            p = Path(chroma_path)
            if not p.exists():
                raise FileNotFoundError(f"ChromaDB path not found: {chroma_path}")
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._vector_store = Chroma(
                persist_directory=str(p),
                collection_name=collection_name,
                embedding_function=self._embedding_model,
            )
        else:
            self._embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            corpus = corpus or load_active_corpus()
            self._documents = corpus_to_documents(corpus)
            self._vector_store = _build_vector_store(
                self._embedding_model, self._documents, pre_chunked=pre_chunked
            )

    def retrieve(self, query: str, k: int = 5, filter_metadata: dict = None) -> List[Tuple[str, float, dict]]:
        """Retrieve top-k relevant chunks with scores and metadata.

        Default k=5 for TRAG (stronger baseline than k=3).
        Routes to Pinecone when pinecone_index is set, otherwise Chroma.
        """
        if self._pinecone_index is not None:
            # Encode query and search Pinecone
            query_vec = self._embedding_model.embed_query(query)
            query_kwargs = {
                "vector": query_vec,
                "top_k": k,
                "include_metadata": True,
            }
            if filter_metadata:
                query_kwargs["filter"] = filter_metadata
            results = self._pinecone_index.query(**query_kwargs)
            out = []
            for match in results.matches:
                doc_id = match.id
                entry = self._corpus_cache.get(doc_id, {})
                text = entry.get("text", "")
                title = entry.get("title", doc_id)
                meta = {"id": doc_id, "title": title}
                out.append((text, float(match.score), meta))
            return out

        results = self._vector_store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score, doc.metadata) for doc, score in results]

    def answer(self, query: str, k: int = 5, filter_metadata: dict = None) -> dict:
        """Full RAG pipeline: retrieve + generate answer."""
        start_time = time.time()

        # Retrieve relevant chunks
        retrieved = self.retrieve(query, k=k, filter_metadata=filter_metadata)
        context_chunks = [chunk for chunk, _, _ in retrieved]
        retrieved_doc_ids = [meta.get("id") for _, _, meta in retrieved if meta.get("id")]
        context = "\n\n---\n\n".join(context_chunks)
        
        # Build prompt
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        # Generate with Gemini
        answer_text = generate_text(prompt)
        
        latency = time.time() - start_time
        
        return {
            "answer": answer_text.text.strip(),
            "retrieved_chunks": context_chunks,
            "retrieval_scores": [score for _, score, _ in retrieved],
            "retrieved_doc_ids": retrieved_doc_ids,
            "latency_seconds": latency,
            "prompt_length": len(prompt),
        }

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """Return the embedding model for use by other systems."""
        return self._embedding_model
