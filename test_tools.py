"""Phase 3 verification script for frozen tools."""

from __future__ import annotations

from core.frozen_state import FrozenState


def main() -> None:
    """Initialize FrozenState and print deterministic outputs."""

    state = FrozenState.build()
    query = "Explain how solar panels generate electricity from sunlight."

    parsed = state.parser.extract(query)
    embedding = state.embedder.embed(query)
    nli_score = state.nli.score(
        premise=state.corpus.documents[0].text, hypothesis=query
    )
    contradiction_score = state.contradiction.score(
        text_a=state.corpus.documents[0].text, text_b=query
    )

    print("Parser output:")
    print(parsed)
    print(f"Embedding dimension: {len(embedding)}")
    print(f"NLI score: {nli_score}")
    print(f"Contradiction score: {contradiction_score}")

    # Determinism check
    parsed_2 = state.parser.extract(query)
    embedding_2 = state.embedder.embed(query)
    nli_score_2 = state.nli.score(
        premise=state.corpus.documents[0].text, hypothesis=query
    )
    contradiction_score_2 = state.contradiction.score(
        text_a=state.corpus.documents[0].text, text_b=query
    )

    deterministic = (
        parsed == parsed_2
        and embedding == embedding_2
        and nli_score == nli_score_2
        and contradiction_score == contradiction_score_2
    )
    print(f"Deterministic outputs: {deterministic}")


if __name__ == "__main__":
    main()
