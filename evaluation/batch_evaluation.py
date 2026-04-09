"""Batch evaluation harness for V3 vs Traditional RAG.

Runs all ground truth queries through both systems, computes metrics,
and saves structured results for statistical analysis.

Usage:
    python -m evaluation.batch_evaluation
    python -m evaluation.batch_evaluation --gt data/ground_truth.json --out results/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import sys
from datetime import datetime
from pathlib import Path


# --------------- retry helper for transient network errors ---------------
def _retry(fn, *args, max_retries=3, backoff=15, **kwargs):
    """Call *fn* with retries on transient network / API errors."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            transient = any(k in msg for k in [
                "connection error", "timeout", "max retries", "name resolution",
                "getaddrinfo", "apiconnectionerror", "rate limit",
            ])
            if transient and attempt < max_retries:
                wait = backoff * attempt
                print(f"    [Network error on attempt {attempt}/{max_retries}: {exc}. "
                      f"Retrying in {wait}s...]")
                time.sleep(wait)
            else:
                raise

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Corpus, Graph, Node
from core.frozen_state import FrozenState, clear_shared_state, get_shared_state
from css.calculator import compute_css_final
from data.cuad_loader import load_cuad_corpus
from data.ground_truth_loader import (
    compute_information_unit_coverage,
    load_ground_truth,
    validate_ground_truth,
)
from evaluation.cost_tracker import ComparisonCostTracker, count_tokens_approx
from evaluation.metrics_calculator import (
    answer_completeness,
    answer_length,
    clause_coverage,
    context_length,
    context_relevance_score,
    faithfulness_score,
    ragas_aspect_coverage,
    relevance_score,
    run_cost_tokens,
    section_diversity,
    token_count,
    token_efficiency,
)
from evaluation.ragas_official import compute_ragas_official_metrics
from evaluation.rag_baselines.traditional_rag import TraditionalRAG
from llm.llm_client import generate_text
from policy.optimizer import optimize
from storage.corpus_store import clear_active_corpus, set_active_corpus
from tools.contradiction_flagger import build_contradiction_annotation


# --------------- contract-scoped retrieval helpers ---------------

def _sanitize_id(doc_id: str) -> str:
    """Make ID ASCII-safe for Pinecone (mirrors load_cuad_to_pinecone)."""
    return re.sub(r'[^\x00-\x7F]', '_', doc_id)


def _build_contract_filter(contract_file: str) -> dict:
    """Build a Pinecone metadata filter for a specific contract.

    Args:
        contract_file: e.g. "ENERGOUSCORP_03_16_2017-EX-10.24-STRATEGIC ALLIANCE AGREEMENT.txt"
    Returns:
        Pinecone filter dict, e.g. {"contract": {"$eq": "ENERGOUSCORP_..."}}
    """
    contract_stem = contract_file.replace(".txt", "")
    return {"contract": {"$eq": _sanitize_id(contract_stem)}}


# --------------- NaN recovery helpers ---------------

_RAGAS_KEYS = ["ragas_context_recall", "ragas_faithfulness",
               "ragas_context_precision", "ragas_answer_relevancy"]


def _has_nan_ragas(metrics: dict) -> bool:
    """Check if any RAGAS metric is NaN or missing."""
    for k in _RAGAS_KEYS:
        v = metrics.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return True
    return False


def _retry_nan_ragas(all_results: list[dict], max_retries: int = 2) -> int:
    """Scan results for NaN RAGAS scores and retry evaluation.

    Only re-runs the RAGAS scoring — answers/context are already computed.
    Returns number of entries successfully recovered.
    """
    fixed = 0
    for entry in all_results:
        for system in ["v3", "trag"]:
            metrics = entry[system]["metrics"]
            if not _has_nan_ragas(metrics):
                continue

            query = entry["query"]
            answer = entry[system]["answer"]
            context = entry[system].get("context", "")
            ground_truth = entry["ground_truth_answer"]

            if not context:
                print(f"    [Skip] No stored context for {entry['annotation_id']} ({system})")
                continue

            print(f"  Retrying RAGAS for {entry['annotation_id']} ({system})...")
            for attempt in range(1, max_retries + 1):
                try:
                    ragas_scores = compute_ragas_official_metrics(
                        query=query, answer=answer,
                        contexts=[context], ground_truth=ground_truth,
                    )
                    metrics.update(ragas_scores)
                    fixed += 1
                    print(f"    Recovered on attempt {attempt}")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        print(f"    Attempt {attempt} failed: {e}. Retrying in 15s...")
                        time.sleep(15 * attempt)
                    else:
                        print(f"    Still failing after {max_retries} attempts: {e}")
    return fixed


def _incremental_save(all_results, annotations, output_dir, gt_path,
                      full_corpus, chroma_path, pinecone_index, validation):
    """Save partial results after every query so progress is never lost."""
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    partial_file = out_dir / "eval_partial_progress.json"
    n_contracts = len({a["contract_file"] for a in annotations})
    output = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "gt_file": str(gt_path) if gt_path else None,
            "n_annotations": len(annotations),
            "n_contracts": n_contracts,
            "completed": len(all_results),
            "full_corpus": full_corpus,
            "chroma_path": chroma_path,
            "pinecone_index": pinecone_index,
            "trap_distribution": validation["by_trap_type"],
        },
        "results": all_results,
    }
    partial_file.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


def _build_user_graph(corpus: Corpus, *, max_nodes: int = 10) -> Graph:
    """Build a simple user graph from the corpus documents."""
    nodes: list[Node] = []
    for doc in corpus.documents[:max_nodes]:
        nodes.append(Node(id=doc.id, text=doc.text, metadata={"title": doc.title}))
    return Graph(nodes=nodes, edges=[])


def _build_v3_prompt(query: str, graph: Graph) -> str:
    """Build V3 final answer prompt with contradiction annotations."""
    context = "\n".join(f"- {node.text}" for node in graph.nodes)
    contradiction_annotation = build_contradiction_annotation(graph)
    prompt = (
        "Use the following graph context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
    )
    if contradiction_annotation:
        prompt += f"\n{contradiction_annotation}\n"
    return prompt


def run_single_v3(query: str, user_graph: Graph, corpus: Corpus,
                  state: FrozenState, filter_metadata: dict = None) -> dict:
    """Run V3 pipeline for a single query. Returns raw result dict."""
    start = time.time()
    optimized = optimize(query, user_graph, max_steps=3, state=state,
                         filter_metadata=filter_metadata)
    prompt = _build_v3_prompt(query, optimized)

    try:
        answer_text = generate_text(prompt).text.strip()
    except RuntimeError as exc:
        answer_text = f"(error: {exc})"
    latency = time.time() - start

    corpus_ids = {doc.id for doc in corpus.documents}
    retrieved_ids = [n.id for n in optimized.nodes if n.id in corpus_ids]
    context_text = "\n".join(n.text for n in optimized.nodes)

    scores = compute_css_final(optimized, query, user_graph, embedder=state.embedder)

    return {
        "system": "V3",
        "answer": answer_text,
        "prompt": prompt,
        "context": context_text,
        "latency_seconds": latency,
        "retrieved_doc_ids": retrieved_ids,
        "css_final": scores["css_final"],
        "prompt_tokens": count_tokens_approx(prompt),
        "completion_tokens": count_tokens_approx(answer_text),
    }


def run_single_trag(query: str, rag: TraditionalRAG, filter_metadata: dict = None) -> dict:
    """Run Traditional RAG for a single query. Returns raw result dict."""
    try:
        result = rag.answer(query, k=3, filter_metadata=filter_metadata)
    except RuntimeError as exc:
        return {
            "system": "Traditional RAG",
            "answer": f"(error: {exc})",
            "prompt": "", "context": "",
            "latency_seconds": 0.0, "retrieved_doc_ids": [],
            "prompt_tokens": 0, "completion_tokens": 0,
        }

    context_text = "\n\n---\n\n".join(result["retrieved_chunks"])
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return {
        "system": "Traditional RAG",
        "answer": result["answer"],
        "prompt": prompt,
        "context": context_text,
        "latency_seconds": result["latency_seconds"],
        "retrieved_doc_ids": result.get("retrieved_doc_ids", []),
        "prompt_tokens": count_tokens_approx(prompt),
        "completion_tokens": count_tokens_approx(result["answer"]),
    }


def compute_all_metrics(query: str, result: dict, ground_truth: str,
                        annotation: dict) -> dict:
    """Compute all evaluation metrics for a single (system, query) pair.

    Args:
        query: The evaluation query
        result: Output from run_single_v3 or run_single_trag
        ground_truth: The human-annotated correct answer
        annotation: The full annotation dict (for information units)

    Returns:
        Dict of metric_name → float
    """
    answer = result["answer"]
    context = result["context"]
    prompt = result["prompt"]

    metrics: dict[str, float] = {}

    # --- Embedding-based (no API cost) ---
    metrics["relevance"] = relevance_score(query, answer)
    metrics["faithfulness_embed"] = faithfulness_score(answer, context)
    metrics["context_relevance"] = context_relevance_score(query, context)
    metrics["completeness_heuristic"] = answer_completeness(query, answer)
    metrics["token_efficiency"] = token_efficiency(context, answer)

    # --- Length metrics ---
    metrics["answer_tokens"] = float(answer_length(answer))
    metrics["context_tokens"] = float(context_length(context))
    metrics["prompt_tokens"] = float(result.get("prompt_tokens", 0))
    metrics["completion_tokens"] = float(result.get("completion_tokens", 0))
    metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]
    metrics["latency_seconds"] = result["latency_seconds"]

    # --- Information Unit Coverage (checklist, no API cost) ---
    iu_result = compute_information_unit_coverage(answer, annotation)
    metrics["info_unit_coverage"] = iu_result["score"]
    metrics["info_units_found"] = float(iu_result["covered_units"])
    metrics["info_units_total"] = float(iu_result["total_units"])

    # --- Official RAGAS metrics (LLM-based, uses OpenAI API) ---
    print(f"    Computing RAGAS metrics...")
    try:
        ragas_scores = compute_ragas_official_metrics(
            query=query,
            answer=answer,
            contexts=[context],
            ground_truth=ground_truth,
        )
        metrics.update(ragas_scores)
    except Exception as ragas_err:
        # Fallback: use custom LLM-based implementations from metrics_calculator
        # (same logic as official RAGAS, but via direct OpenAI calls — no ragas lib)
        print(f"    [RAGAS lib failed: {ragas_err}. Using custom LLM metrics.]")
        from evaluation.metrics_calculator import (
            ragas_answer_relevancy,
            ragas_context_precision,
            ragas_context_recall,
            ragas_faithfulness,
        )
        metrics["ragas_context_recall"] = ragas_context_recall(ground_truth, context)
        metrics["ragas_faithfulness"] = ragas_faithfulness(answer, context)
        metrics["ragas_context_precision"] = ragas_context_precision(query, context)
        metrics["ragas_answer_relevancy"] = ragas_answer_relevancy(query, answer)

    metrics["ragas_aspect_coverage"] = ragas_aspect_coverage(query, answer)

    # --- Domain-specific (no API cost) ---
    clauses = annotation.get("clauses_combined", [])
    metrics["clause_coverage"] = clause_coverage(
        context, clauses, ground_truth
    )
    metrics["section_diversity"] = float(
        section_diversity(result.get("retrieved_doc_ids", []))
    )

    # --- CSS (V3 only) ---
    if "css_final" in result:
        metrics["css_final"] = result["css_final"]

    return metrics


def _load_corpus_for_contract(contract_file: str) -> Corpus:
    """Load a single CUAD contract as a corpus."""
    from data.cuad_loader import DEFAULT_CUAD_PATH
    
    contract_path = Path(DEFAULT_CUAD_PATH) / contract_file
    if not contract_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_path}")
    
    text = contract_path.read_text(encoding="utf-8", errors="ignore")
    
    # Chunk the contract
    from data.cuad_loader import _chunk_by_sections
    chunks = _chunk_by_sections(text, max_chunks=20)
    
    from core.types import CorpusDocument
    documents = []
    contract_name = contract_path.stem
    for j, chunk in enumerate(chunks):
        doc_id = f"{contract_name}_chunk_{j:03d}"
        section_name = chunk.get("section", f"chunk_{j}")
        documents.append(CorpusDocument(
            id=doc_id,
            title=f"{contract_name[:50]} - {section_name[:30]}",
            text=chunk["text"],
            source=str(contract_path),
        ))
    
    return Corpus(documents=documents)


def _load_full_cuad_corpus() -> Corpus:
    """Load entire CUAD dataset (all 510 contracts) for full-dataset evaluation."""
    print("Loading full CUAD corpus (510 contracts)...")
    corpus = load_cuad_corpus(
        max_contracts=None,
        max_chunks_per_contract=20,
        chunk_by="sections",
    )
    print(f"  Loaded {len(corpus.documents)} chunks from {510} contracts")
    return corpus


def run_batch_evaluation(
    gt_path: str | None = None,
    output_dir: str | None = None,
    full_corpus: bool = False,
    chroma_path: str | None = None,
    pinecone_index: str | None = None,
    case_studies: bool = False,
) -> dict:
    """Run the full batch evaluation.

    1. Loads ground truth annotations
    2. For each annotation, runs V3 + TRAG (per-contract or full corpus)
    3. Computes all metrics
    4. Saves results to JSON for statistical analysis

    Args:
        gt_path: Path to ground_truth.json
        output_dir: Output directory for results
        full_corpus: If True, load entire 510-contract CUAD dataset for retrieval.
        chroma_path: Path to persistent ChromaDB (e.g. chroma_cuad_db). When set,
                     both V3 and Traditional RAG use the pre-loaded 510-contract store.
                     Overrides full_corpus (ChromaDB implies full corpus).
        pinecone_index: Pinecone index name (e.g. "cuad-contracts"). When set,
                        both V3 and Traditional RAG query the same Pinecone index.
                        Takes precedence over chroma_path and full_corpus.

    Returns:
        Summary dict with aggregate results
    """
    # Validate ground truth first
    validation = validate_ground_truth(gt_path)
    if not validation["valid"]:
        print(f"ERROR: Ground truth validation failed!")
        print(f"  {validation.get('error', 'No valid annotations')}")
        return validation

    annotations = load_ground_truth(gt_path)
    print(f"\n{'='*70}")
    print(f"BATCH EVALUATION: V3 vs Traditional RAG")
    if pinecone_index:
        print(f"  MODE: Pinecone index '{pinecone_index}' (full CUAD corpus)")
    elif chroma_path:
        print(f"  MODE: ChromaDB at {chroma_path} (510 contracts)")
    elif full_corpus:
        print(f"  MODE: Full 510-contract CUAD corpus")
    print(f"{'='*70}")
    print(f"Annotations: {len(annotations)}")
    print(f"Trap types: {validation['by_trap_type']}")
    print(f"Contracts: {validation['n_contracts']}")
    print(f"{'='*70}\n")

    all_results: list[dict] = []

    if pinecone_index:
        # --- Pinecone mode: both V3 and TRAG query the same Pinecone index ---
        try:
            state = get_shared_state(pinecone_index=pinecone_index, refresh=True)
            corpus = state.corpus
            user_graph = _build_user_graph(corpus)
            rag = TraditionalRAG(pinecone_index_name=pinecone_index)
        except Exception as e:
            print(f"ERROR loading Pinecone index '{pinecone_index}': {e}")
            return {"valid": False, "error": str(e)}

        for i, ann in enumerate(annotations):
            contract_file = ann["contract_file"]
            query = ann["query"]
            ground_truth = ann["ground_truth_answer"]
            # Build contract-scoped Pinecone filter
            contract_filter = _build_contract_filter(contract_file)
            print(f"\n--- Query {i+1}/{len(annotations)} [{contract_file[:50]}] ---")
            print(f"  Query: {query[:80]}...")
            print(f"  Trap: {ann['trap_type']} | Difficulty: {ann.get('difficulty', 'unknown')}")
            print(f"  Contract filter: {contract_filter['contract']['$eq'][:50]}...")

            print(f"  Running V3...")
            v3_result = _retry(run_single_v3, query, user_graph, corpus, state,
                               filter_metadata=contract_filter)
            print(f"    V3 answer: {v3_result['answer'][:100]}...")
            v3_metrics = compute_all_metrics(query, v3_result, ground_truth, ann)

            print(f"  Running Traditional RAG (k=5)...")
            trag_result = _retry(run_single_trag, query, rag,
                                 filter_metadata=contract_filter)
            print(f"    TRAG answer: {trag_result['answer'][:100]}...")
            trag_metrics = compute_all_metrics(query, trag_result, ground_truth, ann)

            entry = {
                "annotation_id": ann["id"],
                "contract_file": contract_file,
                "query": query,
                "trap_type": ann["trap_type"],
                "difficulty": ann.get("difficulty", "unknown"),
                "ground_truth_answer": ground_truth,
                "v3": {
                    "answer": v3_result["answer"],
                    "context": v3_result["context"],
                    "metrics": v3_metrics,
                    "retrieved_doc_ids": v3_result["retrieved_doc_ids"],
                },
                "trag": {
                    "answer": trag_result["answer"],
                    "context": trag_result["context"],
                    "metrics": trag_metrics,
                    "retrieved_doc_ids": trag_result["retrieved_doc_ids"],
                },
            }
            all_results.append(entry)

            _print_quick_comparison(v3_metrics, trag_metrics)

            # --- incremental save so progress is never lost ---
            _incremental_save(all_results, annotations, output_dir, gt_path,
                              full_corpus, chroma_path, pinecone_index, validation)

        clear_shared_state()

    elif chroma_path:
        # Use pre-loaded ChromaDB (510 contracts)
        try:
            state = get_shared_state(chroma_path=chroma_path, refresh=True)
            corpus = state.corpus
            user_graph = _build_user_graph(corpus)
            rag = TraditionalRAG(chroma_path=chroma_path)
        except Exception as e:
            print(f"ERROR loading ChromaDB from {chroma_path}: {e}")
            return {"valid": False, "error": str(e)}

        for i, ann in enumerate(annotations):
            contract_file = ann["contract_file"]
            query = ann["query"]
            ground_truth = ann["ground_truth_answer"]
            print(f"\n--- Query {i+1}/{len(annotations)} [{contract_file[:50]}...] ---")
            print(f"  Query: {query[:80]}...")
            print(f"  Trap: {ann['trap_type']} | Difficulty: {ann.get('difficulty', 'unknown')}")

            print(f"  Running V3...")
            v3_result = run_single_v3(query, user_graph, corpus, state)
            print(f"    V3 answer: {v3_result['answer'][:100]}...")
            v3_metrics = compute_all_metrics(query, v3_result, ground_truth, ann)

            print(f"  Running Traditional RAG...")
            trag_result = run_single_trag(query, rag)
            print(f"    TRAG answer: {trag_result['answer'][:100]}...")
            trag_metrics = compute_all_metrics(query, trag_result, ground_truth, ann)

            entry = {
                "annotation_id": ann["id"],
                "contract_file": contract_file,
                "query": query,
                "trap_type": ann["trap_type"],
                "difficulty": ann.get("difficulty", "unknown"),
                "ground_truth_answer": ground_truth,
                "v3": {
                    "answer": v3_result["answer"],
                    "metrics": v3_metrics,
                    "retrieved_doc_ids": v3_result["retrieved_doc_ids"],
                },
                "trag": {
                    "answer": trag_result["answer"],
                    "metrics": trag_metrics,
                    "retrieved_doc_ids": trag_result["retrieved_doc_ids"],
                },
            }
            all_results.append(entry)

            _print_quick_comparison(v3_metrics, trag_metrics)

        clear_shared_state()
    elif full_corpus:
        # Load entire 510-contract dataset once; run all queries against it
        try:
            corpus = _load_full_cuad_corpus()
            set_active_corpus(corpus)
            state = get_shared_state(corpus=corpus, refresh=True)
            user_graph = _build_user_graph(corpus)
            rag = TraditionalRAG(corpus=corpus)
        except Exception as e:
            print(f"ERROR loading full corpus: {e}")
            return {"valid": False, "error": str(e)}

        for i, ann in enumerate(annotations):
            contract_file = ann["contract_file"]
            query = ann["query"]
            ground_truth = ann["ground_truth_answer"]
            print(f"\n--- Query {i+1}/{len(annotations)} [{contract_file[:50]}...] ---")
            print(f"  Query: {query[:80]}...")
            print(f"  Trap: {ann['trap_type']} | Difficulty: {ann.get('difficulty', 'unknown')}")

            print(f"  Running V3...")
            v3_result = run_single_v3(query, user_graph, corpus, state)
            print(f"    V3 answer: {v3_result['answer'][:100]}...")
            v3_metrics = compute_all_metrics(query, v3_result, ground_truth, ann)

            print(f"  Running Traditional RAG...")
            trag_result = run_single_trag(query, rag)
            print(f"    TRAG answer: {trag_result['answer'][:100]}...")
            trag_metrics = compute_all_metrics(query, trag_result, ground_truth, ann)

            entry = {
                "annotation_id": ann["id"],
                "contract_file": contract_file,
                "query": query,
                "trap_type": ann["trap_type"],
                "difficulty": ann.get("difficulty", "unknown"),
                "ground_truth_answer": ground_truth,
                "v3": {
                    "answer": v3_result["answer"],
                    "metrics": v3_metrics,
                    "retrieved_doc_ids": v3_result["retrieved_doc_ids"],
                },
                "trag": {
                    "answer": trag_result["answer"],
                    "metrics": trag_metrics,
                    "retrieved_doc_ids": trag_result["retrieved_doc_ids"],
                },
            }
            all_results.append(entry)

            _print_quick_comparison(v3_metrics, trag_metrics)

        clear_active_corpus()
        clear_shared_state()
    else:
        # Original per-contract loading
        by_contract: dict[str, list[dict]] = {}
        for ann in annotations:
            cf = ann["contract_file"]
            by_contract.setdefault(cf, []).append(ann)

        contract_count = 0
        for contract_file, contract_annotations in by_contract.items():
            contract_count += 1
            print(f"\n--- Contract {contract_count}/{len(by_contract)}: {contract_file[:60]} ---")
            
            try:
                corpus = _load_corpus_for_contract(contract_file)
                set_active_corpus(corpus)
                state = get_shared_state(corpus=corpus, refresh=True)
                user_graph = _build_user_graph(corpus)
                rag = TraditionalRAG(corpus=corpus)
            except Exception as e:
                print(f"  ERROR loading contract: {e}")
                for ann in contract_annotations:
                    all_results.append({
                        "annotation_id": ann["id"],
                        "contract_file": contract_file,
                        "query": ann["query"],
                        "trap_type": ann["trap_type"],
                        "error": str(e),
                    })
                continue

            for ann in contract_annotations:
                query = ann["query"]
                ground_truth = ann["ground_truth_answer"]
                print(f"\n  Query: {query[:80]}...")
                print(f"  Trap: {ann['trap_type']} | Difficulty: {ann.get('difficulty', 'unknown')}")

                # Run V3
                print(f"  Running V3...")
                v3_result = run_single_v3(query, user_graph, corpus, state)
                print(f"    V3 answer: {v3_result['answer'][:100]}...")
                v3_metrics = compute_all_metrics(query, v3_result, ground_truth, ann)

                # Run Traditional RAG
                print(f"  Running Traditional RAG...")
                trag_result = run_single_trag(query, rag)
                print(f"    TRAG answer: {trag_result['answer'][:100]}...")
                trag_metrics = compute_all_metrics(query, trag_result, ground_truth, ann)

                # Store paired result
                entry = {
                    "annotation_id": ann["id"],
                    "contract_file": contract_file,
                    "query": query,
                    "trap_type": ann["trap_type"],
                    "difficulty": ann.get("difficulty", "unknown"),
                    "ground_truth_answer": ground_truth,
                    "v3": {
                        "answer": v3_result["answer"],
                        "metrics": v3_metrics,
                        "retrieved_doc_ids": v3_result["retrieved_doc_ids"],
                    },
                    "trag": {
                        "answer": trag_result["answer"],
                        "metrics": trag_metrics,
                        "retrieved_doc_ids": trag_result["retrieved_doc_ids"],
                    },
                }
                all_results.append(entry)

                # Print quick comparison
                print(f"  --- Quick Comparison ---")
                for metric_name in ["ragas_context_recall", "ragas_faithfulness",
                                    "ragas_context_precision", "info_unit_coverage",
                                    "latency_seconds", "total_tokens"]:
                    v3_val = v3_metrics.get(metric_name, 0.0)
                    trag_val = trag_metrics.get(metric_name, 0.0)
                    winner = "V3" if v3_val > trag_val else "TRAG" if trag_val > v3_val else "TIE"
                    print(f"    {metric_name:30s}  V3={v3_val:.3f}  TRAG={trag_val:.3f}  [{winner}]")

            # Clean up after each contract
            clear_active_corpus()
            clear_shared_state()

    # Save results
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gt_stem = Path(gt_path or "ground_truth").stem if gt_path else "ground_truth"
    results_file = out_dir / f"eval_{gt_stem}_{timestamp}.json"
    
    n_contracts = len({a["contract_file"] for a in annotations})
    output = {
        "metadata": {
            "timestamp": timestamp,
            "gt_file": str(gt_path) if gt_path else None,
            "n_annotations": len(annotations),
            "n_contracts": n_contracts,
            "full_corpus": full_corpus,
            "chroma_path": chroma_path,
            "pinecone_index": pinecone_index,
            "trap_distribution": validation["by_trap_type"],
        },
        "results": all_results,
    }
    
    # --- NaN recovery: retry failed RAGAS scores before final save ---
    nan_entries = sum(1 for r in all_results
                      if _has_nan_ragas(r.get("v3", {}).get("metrics", {}))
                      or _has_nan_ragas(r.get("trag", {}).get("metrics", {})))
    if nan_entries > 0:
        print(f"\n[NaN Recovery] Found {nan_entries} entries with NaN RAGAS scores. Retrying...")
        fixed = _retry_nan_ragas(all_results)
        print(f"[NaN Recovery] Fixed {fixed} entries.\n")

    results_file.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

    # Print aggregate summary
    _print_summary(all_results)

    # --- Case studies: max-delta query per trap type ---
    if case_studies:
        _dump_case_studies(all_results, out_dir, timestamp)

    return output


_LOWER_IS_BETTER = {"latency_seconds", "total_tokens", "context_tokens", "prompt_tokens"}


def _print_metric_line(name: str, v3_val: float, trag_val: float) -> None:
    """Print a single metric comparison line."""
    if name in _LOWER_IS_BETTER:
        winner = "TRAG" if v3_val > trag_val else "V3" if trag_val > v3_val else "TIE"
    else:
        winner = "V3" if v3_val > trag_val else "TRAG" if trag_val > v3_val else "TIE"
    print(f"    {name:30s}  V3={v3_val:.3f}  TRAG={trag_val:.3f}  [{winner}]")


def _print_quick_comparison(v3_metrics: dict, trag_metrics: dict) -> None:
    """Print per-query comparison grouped by metric category."""
    print(f"  --- RAGAS Official (paper-ready) ---")
    for m in ["ragas_context_recall", "ragas_faithfulness", "ragas_context_precision"]:
        _print_metric_line(m, v3_metrics.get(m, 0.0), trag_metrics.get(m, 0.0))

    print(f"  --- Domain-Specific (custom, supporting) ---")
    for m in ["info_unit_coverage", "clause_coverage", "section_diversity"]:
        _print_metric_line(m, v3_metrics.get(m, 0.0), trag_metrics.get(m, 0.0))

    print(f"  --- Efficiency ---")
    for m in ["latency_seconds", "total_tokens"]:
        _print_metric_line(m, v3_metrics.get(m, 0.0), trag_metrics.get(m, 0.0))


def _print_summary(results: list[dict]) -> None:
    """Print aggregate summary of evaluation results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to summarize.")
        return

    def _agg(metric: str, subset: list[dict]) -> tuple[float, float]:
        v3_raw = [r["v3"]["metrics"].get(metric, 0.0) for r in subset]
        tr_raw = [r["trag"]["metrics"].get(metric, 0.0) for r in subset]
        # Filter NaN values to prevent poisoning the average
        v3 = [x for x in v3_raw if isinstance(x, (int, float)) and not math.isnan(x)]
        tr = [x for x in tr_raw if isinstance(x, (int, float)) and not math.isnan(x)]
        return (
            sum(v3) / len(v3) if v3 else 0.0,
            sum(tr) / len(tr) if tr else 0.0,
        )

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({len(valid)} queries)")
    print(f"{'='*70}")
    print(f"{'Metric':35s}  {'V3 Mean':>10s}  {'TRAG Mean':>10s}  {'Delta':>8s}")

    # --- RAGAS Official ---
    print(f"\n  RAGAS Official (primary, paper Table 1)")
    print(f"  {'-'*65}")
    for metric in ["ragas_context_recall", "ragas_faithfulness",
                    "ragas_context_precision", "ragas_answer_relevancy",
                    "ragas_aspect_coverage"]:
        v3_mean, trag_mean = _agg(metric, valid)
        delta = v3_mean - trag_mean
        print(f"  {metric:33s}  {v3_mean:10.3f}  {trag_mean:10.3f}  {delta:+8.3f}")

    # --- Domain-Specific ---
    print(f"\n  Domain-Specific (custom, paper Table 2)")
    print(f"  {'-'*65}")
    for metric in ["info_unit_coverage", "clause_coverage", "section_diversity"]:
        v3_mean, trag_mean = _agg(metric, valid)
        delta = v3_mean - trag_mean
        print(f"  {metric:33s}  {v3_mean:10.3f}  {trag_mean:10.3f}  {delta:+8.3f}")

    # --- Efficiency ---
    print(f"\n  Efficiency (paper Table 3)")
    print(f"  {'-'*65}")
    for metric in ["latency_seconds", "total_tokens", "context_tokens"]:
        v3_mean, trag_mean = _agg(metric, valid)
        delta = v3_mean - trag_mean
        print(f"  {metric:33s}  {v3_mean:10.3f}  {trag_mean:10.3f}  {delta:+8.3f}")

    # --- Per trap type breakdown ---
    print(f"\n{'='*70}")
    print(f"PER TRAP TYPE BREAKDOWN")
    print(f"{'='*70}")
    for trap in ["trap_a", "trap_b", "trap_c"]:
        trap_results = [r for r in valid if r.get("trap_type") == trap]
        if not trap_results:
            continue

        print(f"\n  {trap.upper()} ({len(trap_results)} queries):")
        print(f"  {'Metric':33s}  {'V3':>7s}  {'TRAG':>7s}")
        print(f"  {'-'*52}")
        for metric in ["ragas_context_recall", "ragas_faithfulness",
                        "info_unit_coverage", "clause_coverage",
                        "section_diversity"]:
            v3_mean, trag_mean = _agg(metric, trap_results)
            print(f"  {metric:33s}  {v3_mean:7.3f}  {trag_mean:7.3f}")


def _dump_case_studies(results: list[dict], out_dir: Path, timestamp: str) -> None:
    """Write one detailed case study per trap type (max context_recall delta).

    For each trap type, pick the query where V3 beat TRAG by the widest margin
    on ``ragas_context_recall``. Dump the query, ground truth, V3 graph nodes
    + retrieved IDs, TRAG chunks + retrieved IDs, and side-by-side metrics.
    """
    valid = [r for r in results if "error" not in r]
    if not valid:
        return

    case_studies = []
    for trap in ["trap_a", "trap_b", "trap_c"]:
        trap_results = [r for r in valid if r.get("trap_type") == trap]
        if not trap_results:
            continue

        # Find max-delta query on context_recall
        best = max(
            trap_results,
            key=lambda r: (
                r["v3"]["metrics"].get("ragas_context_recall", 0.0)
                - r["trag"]["metrics"].get("ragas_context_recall", 0.0)
            ),
        )

        v3m = best["v3"]["metrics"]
        tm = best["trag"]["metrics"]
        delta = v3m.get("ragas_context_recall", 0) - tm.get("ragas_context_recall", 0)

        study = {
            "trap_type": trap,
            "annotation_id": best["annotation_id"],
            "contract_file": best["contract_file"],
            "query": best["query"],
            "ground_truth_answer": best["ground_truth_answer"][:500] + "...",
            "context_recall_delta": round(delta, 4),
            "v3": {
                "answer_preview": best["v3"]["answer"][:300],
                "retrieved_doc_ids": best["v3"]["retrieved_doc_ids"],
                "key_metrics": {
                    k: round(v3m.get(k, 0.0), 4)
                    for k in [
                        "ragas_context_recall", "ragas_faithfulness",
                        "info_unit_coverage", "clause_coverage",
                        "section_diversity", "context_tokens",
                    ]
                },
            },
            "trag": {
                "answer_preview": best["trag"]["answer"][:300],
                "retrieved_doc_ids": best["trag"]["retrieved_doc_ids"],
                "key_metrics": {
                    k: round(tm.get(k, 0.0), 4)
                    for k in [
                        "ragas_context_recall", "ragas_faithfulness",
                        "info_unit_coverage", "clause_coverage",
                        "section_diversity", "context_tokens",
                    ]
                },
            },
            "why_v3_won": (
                f"V3 retrieved {len(best['v3']['retrieved_doc_ids'])} unique chunks "
                f"(section_diversity={v3m.get('section_diversity', 0):.0f}) vs "
                f"TRAG's {len(best['trag']['retrieved_doc_ids'])} "
                f"(section_diversity={tm.get('section_diversity', 0):.0f}). "
                f"Clause coverage: V3={v3m.get('clause_coverage', 0):.2f} vs "
                f"TRAG={tm.get('clause_coverage', 0):.2f}."
            ),
        }
        case_studies.append(study)

    # Write to file
    case_file = out_dir / f"case_studies_{timestamp}.json"
    case_file.write_text(
        json.dumps(case_studies, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nCase studies (1 per trap type) saved to: {case_file}")

    # Also print a summary
    for cs in case_studies:
        print(f"\n  [{cs['trap_type'].upper()}] {cs['annotation_id']}")
        print(f"    Query: {cs['query'][:80]}...")
        print(f"    context_recall delta: {cs['context_recall_delta']:+.3f}")
        print(f"    {cs['why_v3_won']}")


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation: V3 vs TRAG")
    parser.add_argument("--gt", type=str, default=None,
                        help="Path to ground_truth.json (default: data/ground_truth.json)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for results (default: results/)")
    parser.add_argument("--full-corpus", action="store_true",
                        help="Load entire 510-contract CUAD dataset for retrieval (default: per-contract)")
    parser.add_argument("--chroma", type=str, default=None,
                        help="Path to ChromaDB (e.g. chroma_cuad_db). Uses pre-loaded 510-contract store.")
    parser.add_argument("--pinecone", type=str, default=None, metavar="INDEX_NAME",
                        help="Pinecone index name (e.g. cuad-contracts). Both V3 and TRAG "
                             "query the same index. Run scripts/load_cuad_to_pinecone.py first.")
    parser.add_argument("--case-studies", action="store_true",
                        help="Dump detailed retrieval traces for best V3 query per trap type.")
    parser.add_argument("--all-gt", action="store_true",
                        help="Run evaluation for all 3 ground truth files (ground_truth.json, _2, _3)")
    args = parser.parse_args()

    if args.all_gt:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        gt_files = [
            data_dir / "ground_truth.json",
            data_dir / "ground_truth_2.json",
            data_dir / "ground_truth_3.json",
        ]
        for gt_file in gt_files:
            if gt_file.exists():
                print(f"\n{'#'*70}\n# Running evaluation for {gt_file.name}\n{'#'*70}")
                run_batch_evaluation(
                    gt_path=str(gt_file),
                    output_dir=args.out,
                    full_corpus=args.full_corpus or bool(args.chroma),
                    chroma_path=args.chroma,
                    pinecone_index=args.pinecone,
                    case_studies=args.case_studies,
                )
            else:
                print(f"Warning: {gt_file} not found, skipping")

        if not any(p.exists() for p in gt_files):
            print("ERROR: No ground truth files found")
    else:
        run_batch_evaluation(
            gt_path=args.gt,
            output_dir=args.out,
            full_corpus=args.full_corpus,
            chroma_path=args.chroma,
            pinecone_index=args.pinecone,
            case_studies=args.case_studies,
        )


if __name__ == "__main__":
    main()
