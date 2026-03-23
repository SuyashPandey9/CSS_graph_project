"""Immutable corpus loader for V3."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from core.types import Corpus, CorpusDocument


_ACTIVE_CORPUS: Corpus | None = None


def load_fixed_corpus() -> Corpus:
    """Load the fixed corpus from data/fixed_corpus.json."""

    corpus_path = Path(__file__).resolve().parent.parent / "data" / "fixed_corpus.json"
    if not corpus_path.exists():
        raise FileNotFoundError("fixed_corpus.json not found in data/ directory.")

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = [CorpusDocument(**item) for item in raw.get("documents", [])]
    return Corpus(documents=documents)


def load_active_corpus() -> Corpus:
    """Return the active corpus if set; otherwise load the fixed corpus."""

    return _ACTIVE_CORPUS or load_fixed_corpus()


def set_active_corpus(corpus: Corpus) -> None:
    """Set the active in-memory corpus for the current session."""

    global _ACTIVE_CORPUS
    _ACTIVE_CORPUS = corpus


def clear_active_corpus() -> None:
    """Clear the active in-memory corpus."""

    global _ACTIVE_CORPUS
    _ACTIVE_CORPUS = None


def corpus_to_documents(corpus: Corpus) -> list[dict]:
    """Normalize a Corpus into a list of dicts for baseline use."""

    return [
        {
            "id": doc.id,
            "title": doc.title,
            "text": doc.text,
            "source": doc.source,
        }
        for doc in corpus.documents
    ]


def load_corpus_from_csv(csv_path: Path) -> Corpus:
    """Load a CSV file into a Corpus with flexible column handling."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    content = csv_path.read_text(encoding="utf-8-sig")
    sample = content[:4096]
    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        has_header = False

    documents: list[CorpusDocument] = []
    seen_ids: set[str] = set()

    def _unique_id(base_id: str, idx: int) -> str:
        candidate = base_id or f"row_{idx}"
        if candidate not in seen_ids:
            return candidate
        suffix = 1
        while f"{candidate}_{suffix}" in seen_ids:
            suffix += 1
        return f"{candidate}_{suffix}"

    def _row_to_doc(row: dict, idx: int) -> CorpusDocument | None:
        values = {k: ("" if v is None else str(v).strip()) for k, v in row.items()}
        if not any(values.values()):
            return None
        doc_id = values.get("id") or values.get("doc_id") or values.get("document_id")
        title = values.get("title") or values.get("name") or values.get("header") or f"Row {idx}"
        text = values.get("text") or values.get("content") or values.get("body")
        if not text:
            ordered_values = [v for v in values.values() if v]
            text = " | ".join(ordered_values) if ordered_values else title
        unique_id = _unique_id(doc_id or f"row_{idx}", idx)
        seen_ids.add(unique_id)
        return CorpusDocument(
            id=unique_id,
            title=title,
            text=text,
            source=csv_path.name,
        )

    if has_header:
        reader = csv.DictReader(StringIO(content))
        for idx, row in enumerate(reader, start=1):
            doc = _row_to_doc(row, idx)
            if doc:
                documents.append(doc)
    else:
        reader = csv.reader(StringIO(content))
        for idx, row in enumerate(reader, start=1):
            row_dict = {f"col_{i + 1}": value for i, value in enumerate(row)}
            doc = _row_to_doc(row_dict, idx)
            if doc:
                documents.append(doc)

    if not documents:
        raise ValueError("CSV did not contain any usable rows.")

    return Corpus(documents=documents)
