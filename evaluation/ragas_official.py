"""Official RAGAS evaluation helpers."""

from __future__ import annotations

from typing import Iterable

import os


_RAGAS_LLM = None
_RAGAS_EMBEDDINGS = None

_METRIC_NAME_MAP = {
    "answer_relevancy": "ragas_answer_relevancy",
    "faithfulness": "ragas_faithfulness",
    "context_precision": "ragas_context_precision",
    "context_recall": "ragas_context_recall",
}


def _get_ragas_clients():
    """Create and cache LLM + embeddings for RAGAS metrics."""
    global _RAGAS_LLM, _RAGAS_EMBEDDINGS

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for official RAGAS metrics.")

    if _RAGAS_LLM is None or _RAGAS_EMBEDDINGS is None:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        _RAGAS_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        _RAGAS_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

    return _RAGAS_LLM, _RAGAS_EMBEDDINGS


def _extract_scores(result: object) -> dict[str, float]:
    """Extract per-row metric scores from a RAGAS EvaluationResult."""
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        row = df.iloc[0].to_dict()
        return {
            _METRIC_NAME_MAP[name]: float(row.get(name, 0.0) or 0.0)
            for name in _METRIC_NAME_MAP
        }

    if hasattr(result, "scores"):
        scores = getattr(result, "scores")
        if isinstance(scores, dict):
            extracted: dict[str, float] = {}
            for name, out_name in _METRIC_NAME_MAP.items():
                raw = scores.get(name, 0.0)
                if isinstance(raw, list):
                    raw = raw[0] if raw else 0.0
                extracted[out_name] = float(raw or 0.0)
            return extracted

    raise ValueError("Unexpected RAGAS result format; unable to extract scores.")


def compute_ragas_official_metrics(
    query: str,
    answer: str,
    contexts: Iterable[str],
    ground_truth: str,
) -> dict[str, float]:
    """Compute official RAGAS metrics for a single query/answer pair."""
    if not ground_truth or not ground_truth.strip():
        raise ValueError("Ground truth is required for RAGAS context_recall.")

    context_list = [c for c in contexts if c]
    if not context_list:
        context_list = [""]

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, embeddings = _get_ragas_clients()

    dataset = Dataset.from_dict(
        {
            "question": [query],
            "answer": [answer],
            "contexts": [context_list],
            "ground_truth": [ground_truth],
        }
    )

    result = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    return _extract_scores(result)
