"""Phase 8 verification script for Traditional RAG baseline."""

from __future__ import annotations

from evaluation.rag_baselines.traditional_rag import TraditionalRAG


def main() -> None:
    """Test Traditional RAG retrieval and generation."""

    print("Initializing Traditional RAG (loading embeddings model)...")
    rag = TraditionalRAG()
    print("Traditional RAG initialized.\n")

    query = "How do solar panels generate electricity from sunlight?"
    print(f"Query: {query}\n")

    # Test retrieval
    print("--- Retrieval Results ---")
    retrieved = rag.retrieve(query, k=3)
    for i, (chunk, score) in enumerate(retrieved, 1):
        preview = chunk[:100].replace("\n", " ") + "..."
        print(f"{i}. [score={score:.4f}] {preview}")

    print("\n--- Generating Answer ---")
    result = rag.answer(query, k=3)

    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nMetrics:")
    print(f"  Latency: {result['latency_seconds']:.2f}s")
    print(f"  Prompt length: {result['prompt_length']} chars")
    print(f"  Chunks retrieved: {len(result['retrieved_chunks'])}")


if __name__ == "__main__":
    main()
