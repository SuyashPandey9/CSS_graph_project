"""Load the entire CUAD dataset into a persistent ChromaDB vector store.

Usage:
    python -m scripts.load_cuad_to_chromadb
    python -m scripts.load_cuad_to_chromadb --cuad-path "C:/path/to/CUAD_v1/full_contract_txt"
    python -m scripts.load_cuad_to_chromadb --output-dir ./chroma_cuad_db
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.cuad_loader import DEFAULT_CUAD_PATH, load_cuad_corpus
from storage.corpus_store import corpus_to_documents

# Match Traditional RAG embedding model for consistency
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CHROMA_DIR = "chroma_cuad_db"


def load_cuad_into_chromadb(
    cuad_path: str | None = None,
    output_dir: str = DEFAULT_CHROMA_DIR,
    max_chunks_per_contract: int = 50,
) -> Chroma:
    """Load entire CUAD dataset into a persistent ChromaDB.

    Args:
        cuad_path: Path to CUAD full_contract_txt folder (default from cuad_loader)
        output_dir: Directory to persist ChromaDB (created if missing)
        max_chunks_per_contract: Max chunks per contract (higher = more content)

    Returns:
        Chroma vector store (persisted to output_dir)
    """
    cuad_path = cuad_path or DEFAULT_CUAD_PATH
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading entire CUAD corpus...")
    corpus = load_cuad_corpus(
        contracts_dir=cuad_path,
        max_contracts=None,  # Load ALL contracts
        max_chunks_per_contract=max_chunks_per_contract,
        chunk_by="sections",
    )

    documents = corpus_to_documents(corpus)
    print(f"[2/4] Loaded {len(documents)} document chunks. Initializing embeddings...")

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Combine title + text for each document
    texts = [f"{doc['title']}\n{doc['text']}" for doc in documents]
    metadatas = [{"id": doc["id"], "title": doc["title"]} for doc in documents]

    # Split into smaller chunks for better retrieval (same as Traditional RAG)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_texts: list[str] = []
    split_metadatas: list[dict] = []
    for text, meta in zip(texts, metadatas):
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            split_texts.append(chunk)
            split_metadatas.append({**meta, "chunk_idx": i})

    print(f"[3/4] Created {len(split_texts)} chunks. Building ChromaDB (this may take a while)...")

    vector_store = Chroma.from_texts(
        texts=split_texts,
        embedding=embedding_model,
        metadatas=split_metadatas,
        persist_directory=str(output_path),
        collection_name="cuad_contracts",
    )

    print(f"[4/4] Done. ChromaDB persisted to: {output_path.absolute()}")
    return vector_store


def main() -> None:
    parser = argparse.ArgumentParser(description="Load entire CUAD dataset into ChromaDB")
    parser.add_argument(
        "--cuad-path",
        default=None,
        help=f"Path to CUAD full_contract_txt folder (default: {DEFAULT_CUAD_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_CHROMA_DIR,
        help=f"ChromaDB output directory (default: {DEFAULT_CHROMA_DIR})",
    )
    parser.add_argument(
        "--max-chunks-per-contract",
        type=int,
        default=50,
        help="Max chunks per contract (default: 50)",
    )
    args = parser.parse_args()

    load_cuad_into_chromadb(
        cuad_path=args.cuad_path,
        output_dir=args.output_dir,
        max_chunks_per_contract=args.max_chunks_per_contract,
    )


if __name__ == "__main__":
    main()
