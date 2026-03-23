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
import time
import sys
from datetime import datetime
from pathlib import Path

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
    context_length,
    context_relevance_score,
    faithfulness_score,
    ragas_aspect_coverage,
    relevance_score,
    run_cost_tokens,
    token_count,
    token_efficiency,
)
from evaluation.ragas_official import compute_ragas_official_metrics
from evaluation.rag_baselines.traditional_rag import TraditionalRAG
from llm.llm_client import generate_text
from policy.optimizer import optimize
from storage.corpus_store import clear_active_corpus, set_active_corpus
from tools.contradiction_flagger import build_contradiction_annotation


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
                  state: FrozenState) -> dict:
    """Run V3 pipeline for a single query. Returns raw result dict."""
    start = time.time()
    optimized = optimize(query, user_graph, max_steps=3, state=state)
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


def run_single_trag(query: str, rag: TraditionalRAG) -> dict:
    """Run Traditional RAG for a single query. Returns raw result dict."""
    try:
        result = rag.answer(query, k=3)
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
    ragas_scores = compute_ragas_official_metrics(
        query=query,
        answer=answer,
        contexts=[context],
        ground_truth=ground_truth,
    )
    metrics.update(ragas_scores)
    metrics["ragas_aspect_coverage"] = ragas_aspect_coverage(query, answer)

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
    if chroma_path:
        print(f"  MODE: ChromaDB at {chroma_path} (510 contracts)")
    elif full_corpus:
        print(f"  MODE: Full 510-contract CUAD corpus")
    print(f"{'='*70}")
    print(f"Annotations: {len(annotations)}")
    print(f"Trap types: {validation['by_trap_type']}")
    print(f"Contracts: {validation['n_contracts']}")
    print(f"{'='*70}\n")

    all_results: list[dict] = []

    if chroma_path:
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

            print(f"  --- Quick Comparison ---")
            for metric_name in ["ragas_context_recall", "ragas_faithfulness",
                                "ragas_context_precision", "info_unit_coverage",
                                "latency_seconds", "total_tokens"]:
                v3_val = v3_metrics.get(metric_name, 0.0)
                trag_val = trag_metrics.get(metric_name, 0.0)
                winner = "V3" if v3_val > trag_val else "TRAG" if trag_val > v3_val else "TIE"
                print(f"    {metric_name:30s}  V3={v3_val:.3f}  TRAG={trag_val:.3f}  [{winner}]")

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

            print(f"  --- Quick Comparison ---")
            for metric_name in ["ragas_context_recall", "ragas_faithfulness",
                                "ragas_context_precision", "info_unit_coverage",
                                "latency_seconds", "total_tokens"]:
                v3_val = v3_metrics.get(metric_name, 0.0)
                trag_val = trag_metrics.get(metric_name, 0.0)
                winner = "V3" if v3_val > trag_val else "TRAG" if trag_val > v3_val else "TIE"
                print(f"    {metric_name:30s}  V3={v3_val:.3f}  TRAG={trag_val:.3f}  [{winner}]")

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
            "trap_distribution": validation["by_trap_type"],
        },
        "results": all_results,
    }
    
    results_file.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

    # Print aggregate summary
    _print_summary(all_results)

    return output


def _print_summary(results: list[dict]) -> None:
    """Print aggregate summary of evaluation results."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("No valid results to summarize.")
        return

    key_metrics = [
        "ragas_context_recall", "ragas_faithfulness", "ragas_context_precision",
        "ragas_answer_relevancy", "ragas_aspect_coverage",
        "info_unit_coverage", "latency_seconds", "total_tokens",
    ]

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({len(valid)} queries)")
    print(f"{'='*70}")
    print(f"{'Metric':35s}  {'V3 Mean':>10s}  {'TRAG Mean':>10s}  {'Delta':>8s}")
    print(f"{'-'*70}")

    for metric in key_metrics:
        v3_vals = [r["v3"]["metrics"].get(metric, 0.0) for r in valid]
        trag_vals = [r["trag"]["metrics"].get(metric, 0.0) for r in valid]
        
        v3_mean = sum(v3_vals) / len(v3_vals) if v3_vals else 0.0
        trag_mean = sum(trag_vals) / len(trag_vals) if trag_vals else 0.0
        delta = v3_mean - trag_mean
        
        print(f"{metric:35s}  {v3_mean:10.3f}  {trag_mean:10.3f}  {delta:+8.3f}")

    # Per trap type
    for trap in ["trap_a", "trap_b", "trap_c"]:
        trap_results = [r for r in valid if r.get("trap_type") == trap]
        if not trap_results:
            continue
        
        print(f"\n  {trap.upper()} ({len(trap_results)} queries):")
        for metric in ["ragas_context_recall", "info_unit_coverage"]:
            v3_vals = [r["v3"]["metrics"].get(metric, 0.0) for r in trap_results]
            trag_vals = [r["trag"]["metrics"].get(metric, 0.0) for r in trap_results]
            v3_mean = sum(v3_vals) / len(v3_vals)
            trag_mean = sum(trag_vals) / len(trag_vals)
            print(f"    {metric:30s}  V3={v3_mean:.3f}  TRAG={trag_mean:.3f}")


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
        )


if __name__ == "__main__":
    main()
