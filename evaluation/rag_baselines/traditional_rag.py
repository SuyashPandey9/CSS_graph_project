"""Traditional RAG baseline using LangChain with neural embeddings.

Uses sentence-transformers for embeddings and ChromaDB for vector storage.
Same fixed corpus and Gemini LLM as V3 for fair comparison.
"""

from __future__ import annotations

import time
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from core.types import Corpus
from llm.llm_client import generate_text
from storage.corpus_store import corpus_to_documents, load_active_corpus

# Shared embedding model name - used by both Traditional RAG and V3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _build_vector_store(
    embedding_model: HuggingFaceEmbeddings,
    documents: List[dict],
) -> Chroma:
    """Build a Chroma vector store from the corpus documents."""
    # Combine title + text for each document
    texts = [f"{doc['title']}\n{doc['text']}" for doc in documents]
    metadatas = [{"id": doc["id"], "title": doc["title"]} for doc in documents]
    
    # Optional: split into smaller chunks for better retrieval
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
    ) -> None:
        """Initialize embeddings and vector store.

        Args:
            corpus: Corpus to build vector store from (ignored if chroma_path set)
            chroma_path: Path to persistent ChromaDB directory. When set, loads
                         existing ChromaDB instead of building from corpus.
            collection_name: Chroma collection name when using chroma_path.
        """
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        if chroma_path:
            from pathlib import Path
            p = Path(chroma_path)
            if not p.exists():
                raise FileNotFoundError(f"ChromaDB path not found: {chroma_path}")
            self._vector_store = Chroma(
                persist_directory=str(p),
                collection_name=collection_name,
                embedding_function=self._embedding_model,
            )
        else:
            corpus = corpus or load_active_corpus()
            self._documents = corpus_to_documents(corpus)
            self._vector_store = _build_vector_store(self._embedding_model, self._documents)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float, dict]]:
        """Retrieve top-k relevant chunks with scores and metadata."""
        results = self._vector_store.similarity_search_with_score(query, k=k)
        return [(doc.page_content, score, doc.metadata) for doc, score in results]

    def answer(self, query: str, k: int = 3) -> dict:
        """Full RAG pipeline: retrieve + generate answer."""
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, k=k)
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
