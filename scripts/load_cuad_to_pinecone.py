"""Load entire CUAD dataset into Pinecone vector store.

One-time ingestion script. After running, both V3 and Traditional RAG can
query the same persistent Pinecone index without re-embedding.

Also caches to disk (so FrozenState can load them without re-computing):
  - data/pinecone_corpus_cache.json   id → {title, text, source}
  - data/idf_dict.json                term → IDF float
  - data/cross_ref_edges.json         [[src, tgt, type], ...] per-contract only

Usage:
    python -m scripts.load_cuad_to_pinecone
    python -m scripts.load_cuad_to_pinecone --index-name cuad-contracts --max-chunks 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_INDEX_NAME = "cuad-contracts"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 100  # Pinecone upsert batch limit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_CACHE_PATH = DATA_DIR / "pinecone_corpus_cache.json"
IDF_CACHE_PATH = DATA_DIR / "idf_dict.json"
CROSS_REF_CACHE_PATH = DATA_DIR / "cross_ref_edges.json"


def _get_pinecone_api_key() -> str:
    # Load .env / .env.txt via project's config module
    try:
        from config.settings import _load_env
        _load_env()
    except Exception:
        pass
    key = os.environ.get("PINECONE_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "PINECONE_API_KEY environment variable not set. "
            "Add it to your .env.txt file or set it:\n"
            '  PowerShell: $env:PINECONE_API_KEY = "pc-xxxxx"\n'
            '  CMD:        set PINECONE_API_KEY=pc-xxxxx'
        )
    return key


def _sanitize_id(doc_id: str) -> str:
    """Make ID ASCII-safe for Pinecone (requires ASCII vector IDs).

    Replaces non-ASCII chars (é, ü, etc.) with underscores.
    ASCII-only IDs pass through unchanged.
    """
    return re.sub(r'[^\x00-\x7F]', '_', doc_id)


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using HuggingFace sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vecs]


def _build_corpus_cache(documents) -> Dict[str, dict]:
    """Build sanitized_id → {title, text, source} mapping.

    IDs are sanitized to ASCII for Pinecone compatibility.
    """
    return {
        _sanitize_id(doc.id): {"title": doc.title, "text": doc.text, "source": doc.source or ""}
        for doc in documents
    }


def load_cuad_into_pinecone(
    index_name: str = DEFAULT_INDEX_NAME,
    max_chunks_per_contract: int = 20,
    skip_if_exists: bool = True,
) -> None:
    """Embed all CUAD contracts and upsert into Pinecone.

    Args:
        index_name: Pinecone index name (created if it doesn't exist).
        max_chunks_per_contract: Chunks per contract (20 × 510 ≈ 10K vectors).
        skip_if_exists: If corpus cache already exists, skip ingestion (idempotent).
    """
    from pinecone import Pinecone, ServerlessSpec

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load CUAD corpus ---
    print("[1/6] Loading CUAD corpus...")
    from data.cuad_loader import load_cuad_corpus
    corpus = load_cuad_corpus(
        max_contracts=None,
        max_chunks_per_contract=max_chunks_per_contract,
        chunk_by="sections",
    )
    documents = corpus.documents
    print(f"      Loaded {len(documents)} chunks from CUAD.")

    # --- Cache corpus to disk ---
    print("[2/6] Caching corpus metadata to disk...")
    corpus_cache = _build_corpus_cache(documents)
    CORPUS_CACHE_PATH.write_text(
        json.dumps(corpus_cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      Saved {len(corpus_cache)} entries → {CORPUS_CACHE_PATH}")

    # --- Compute and cache IDF dict ---
    print("[3/6] Computing corpus-wide IDF dict...")
    from graph.edge_builder import compute_corpus_idf
    idf_dict = compute_corpus_idf(documents)
    IDF_CACHE_PATH.write_text(
        json.dumps(idf_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      IDF computed for {len(idf_dict)} terms → {IDF_CACHE_PATH}")

    # --- Compute and cache cross-ref edges (within-contract only) ---
    print("[4/6] Pre-computing cross-reference edges (within-contract)...")
    from collections import defaultdict
    from graph.edge_builder import precompute_cross_ref_edges

    by_source: dict = defaultdict(list)
    for doc in documents:
        # Group by parent contract (source = original .txt path or doc id prefix)
        src_key = doc.source or doc.id.split("_chunk_")[0]
        by_source[src_key].append(doc)

    all_edges = []
    for group_docs in by_source.values():
        if len(group_docs) < 2:
            continue
        edges = precompute_cross_ref_edges(group_docs)
        for edge in edges:
            if isinstance(edge, tuple):
                # Sanitize IDs in edges for Pinecone consistency
                sanitized = [_sanitize_id(str(e)) if i < 2 else str(e) for i, e in enumerate(edge)]
                all_edges.append(sanitized)
            elif hasattr(edge, "source"):
                all_edges.append([
                    _sanitize_id(edge.source),
                    _sanitize_id(edge.target),
                    getattr(edge, "relation", "cross_ref"),
                ])

    CROSS_REF_CACHE_PATH.write_text(
        json.dumps(all_edges, ensure_ascii=False), encoding="utf-8"
    )
    print(f"      Found {len(all_edges)} cross-ref edges → {CROSS_REF_CACHE_PATH}")

    # --- Check if already ingested ---
    pc = Pinecone(api_key=_get_pinecone_api_key())
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if skip_if_exists and index_name in existing_indexes:
        stats = pc.Index(index_name).describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"[5/6] Pinecone index '{index_name}' already has "
                  f"{stats.total_vector_count} vectors — skipping upsert.")
            print("[6/6] Done (used existing index).")
            return

    # --- Create Pinecone index if needed ---
    print(f"[5/6] Setting up Pinecone index '{index_name}' (dim={EMBEDDING_DIM}, cosine)...")
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"      Created new index '{index_name}'.")
    else:
        print(f"      Using existing empty index '{index_name}'.")

    index = pc.Index(index_name)

    # --- Embed and upsert in batches ---
    print(f"[6/6] Embedding {len(documents)} chunks and upserting to Pinecone...")
    texts = [f"{doc.title}\n{doc.text}" for doc in documents]
    all_vecs = _embed_texts(texts)

    upserted = 0
    batch_vectors = []
    for doc, vec in zip(documents, all_vecs):
        # Sanitize ID for Pinecone (ASCII-only requirement)
        safe_id = _sanitize_id(doc.id)
        # Extract contract name from doc ID (everything before _chunk_)
        contract_name = doc.id.rsplit("_chunk_", 1)[0] if "_chunk_" in doc.id else doc.id
        batch_vectors.append({
            "id": safe_id,
            "values": vec,
            "metadata": {
                "title": doc.title[:200],  # Pinecone metadata 40KB limit per vector
                "source": (doc.source or "")[:200],
                "contract": _sanitize_id(contract_name),  # ASCII-safe contract identifier
            },
        })
        if len(batch_vectors) >= BATCH_SIZE:
            index.upsert(vectors=batch_vectors)
            upserted += len(batch_vectors)
            batch_vectors = []
            if upserted % 1000 == 0:
                print(f"      Upserted {upserted}/{len(documents)} vectors...")

    if batch_vectors:
        index.upsert(vectors=batch_vectors)
        upserted += len(batch_vectors)

    stats = index.describe_index_stats()
    print(f"\n{'='*60}")
    print(f"Pinecone ingestion complete.")
    print(f"  Index: {index_name}")
    print(f"  Vectors upserted: {upserted}")
    print(f"  Index total vectors: {stats.total_vector_count}")
    print(f"  Corpus cache: {CORPUS_CACHE_PATH}")
    print(f"  IDF cache:    {IDF_CACHE_PATH}")
    print(f"  Cross-ref cache: {CROSS_REF_CACHE_PATH}")
    print(f"{'='*60}")
    print(f"\nNext step:")
    print(f"  python -m evaluation.batch_evaluation \\")
    print(f"    --gt data/ground_truth_cuad_v2.json \\")
    print(f"    --pinecone {index_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load CUAD into Pinecone + cache IDF/cross-ref edges"
    )
    parser.add_argument(
        "--index-name", default=DEFAULT_INDEX_NAME,
        help=f"Pinecone index name (default: {DEFAULT_INDEX_NAME})"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=20,
        help="Max chunks per contract (default: 20). 20×510≈10K vectors fits free tier."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-upsert even if index already has vectors."
    )
    args = parser.parse_args()

    load_cuad_into_pinecone(
        index_name=args.index_name,
        max_chunks_per_contract=args.max_chunks,
        skip_if_exists=not args.force,
    )


if __name__ == "__main__":
    main()
