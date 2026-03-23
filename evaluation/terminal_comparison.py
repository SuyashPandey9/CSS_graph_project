"""Terminal comparison of V3 vs Traditional RAG.

Fix F2: Simplified to only compare V3 and Traditional RAG.
With comprehensive cost tracking (initial, run, total) for both systems.
"""

from __future__ import annotations

import time
from pathlib import Path

from core.types import Corpus, Graph, Node
from css.calculator import compute_css_final
from evaluation.cost_tracker import ComparisonCostTracker, count_tokens_approx
from evaluation.metrics_calculator import (
    answer_completeness,
    answer_length,
    context_length,
    context_relevance_score,
    corpus_token_count,
    faithfulness_score,
    llm_judge_score,
    ragas_answer_relevancy,
    ragas_aspect_coverage,
    ragas_context_precision,
    ragas_context_recall,
    ragas_faithfulness,
    relevance_score,
    run_cost_tokens,
    token_efficiency,
)
from evaluation.rag_baselines.traditional_rag import TraditionalRAG
from llm.llm_client import generate_text
from policy.optimizer import optimize
from core.frozen_state import FrozenState, clear_shared_state, get_shared_state
from tools.contradiction_flagger import build_contradiction_annotation
from storage.corpus_store import (
    clear_active_corpus,
    load_corpus_from_csv,
    load_fixed_corpus,
    set_active_corpus,
)



def _build_user_graph(corpus: Corpus, *, max_nodes: int = 10) -> Graph:
    """Build a simple user graph from the corpus documents."""

    nodes: list[Node] = []
    for doc in corpus.documents[:max_nodes]:
        nodes.append(Node(id=doc.id, text=doc.text, metadata={"title": doc.title}))
    return Graph(nodes=nodes, edges=[])


def _build_v3_prompt(query: str, graph: Graph) -> str:
    """Build V3 final answer prompt.
    
    Fix A2: Includes contradiction/qualification annotations when detected.
    The contradiction flagger scans the optimized graph for legal override
    indicators (e.g., 'notwithstanding', 'except as provided') and injects
    a warning so the LLM pays attention to exceptions and qualifications.
    """

    context = "\n".join(f"- {node.text}" for node in graph.nodes)
    
    # Fix A2: Inject contradiction/qualification warnings if detected
    contradiction_annotation = build_contradiction_annotation(graph)
    
    prompt = (
        "Use the following graph context to answer the question.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
    )
    
    if contradiction_annotation:
        prompt += f"\n{contradiction_annotation}\n"
    
    return prompt


def run_v3(query: str, *, user_graph: Graph, corpus: Corpus, state: FrozenState,
           cost_tracker=None) -> dict:
    """Run the full V3 pipeline for a query."""

    start = time.time()
    optimized = optimize(query, user_graph, max_steps=3, state=state)
    prompt = _build_v3_prompt(query, optimized)
    
    # Track optimization tokens (CSS calculations don't use LLM)
    if cost_tracker:
        cost_tracker.add_run(count_tokens_approx(prompt), "prompt")
    
    try:
        answer_text = generate_text(prompt).text.strip()
        if cost_tracker:
            cost_tracker.add_run(count_tokens_approx(answer_text), "completion")
    except RuntimeError as exc:
        answer_text = f"(error: {exc})"
    latency = time.time() - start

    corpus_ids = {doc.id for doc in corpus.documents}
    retrieved_ids = [node.id for node in optimized.nodes if node.id in corpus_ids]

    scores = compute_css_final(optimized, query, user_graph, embedder=state.embedder)
    
    # Check if V3 flagged low confidence (context may be insufficient)
    low_confidence = any(n.metadata.get("low_confidence", False) for n in optimized.nodes)
    if low_confidence:
        answer_text = "[Low confidence - context may be insufficient] " + answer_text
    
    if cost_tracker:
        cost_tracker.record_query()
    
    return {
        "answer": answer_text,
        "prompt": prompt,
        "latency_seconds": latency,
        "retrieved_doc_ids": retrieved_ids,
        "css_final": scores["css_final"],
        "low_confidence": low_confidence,
    }


def run_traditional_rag(query: str, rag: TraditionalRAG, cost_tracker=None) -> dict:
    """Run Traditional RAG for a query."""

    try:
        result = rag.answer(query, k=3)
    except RuntimeError as exc:
        return {
            "answer": f"(error: {exc})",
            "prompt": "",
            "latency_seconds": 0.0,
            "retrieved_doc_ids": [],
        }
    context = "\n\n---\n\n".join(result["retrieved_chunks"])
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    
    if cost_tracker:
        cost_tracker.add_run(count_tokens_approx(prompt), "prompt")
        cost_tracker.add_run(count_tokens_approx(result["answer"]), "completion")
        cost_tracker.record_query()
    
    return {
        "answer": result["answer"],
        "prompt": prompt,
        "latency_seconds": result["latency_seconds"],
        "retrieved_doc_ids": result.get("retrieved_doc_ids", []),
    }



def _prompt_for_dataset() -> Corpus:
    """Prompt user to load a CSV dataset or fall back to the fixed corpus."""

    while True:
        raw = input("Enter CSV path (or 'cuad' for legal contracts, Enter for default): ").strip()
        if not raw:
            clear_active_corpus()
            return load_fixed_corpus()
        
        # CUAD shortcut
        if raw.lower() == "cuad":
            from data.cuad_loader import load_cuad_corpus
            print("[CUAD] Loading legal contracts dataset...")
            corpus = load_cuad_corpus(max_contracts=30)  # Load 30 contracts
            set_active_corpus(corpus)
            return corpus
        
        path = Path(raw).expanduser()
        try:
            corpus = load_corpus_from_csv(path)
        except Exception as exc:
            print(f"Could not load CSV: {exc}")
            continue
        set_active_corpus(corpus)
        return corpus


def _prompt_for_query() -> str:
    """Prompt the user for a query string."""

    while True:
        query = input("Type your query: ").strip()
        if query:
            return query
        print("Query cannot be empty.")


def main() -> None:
    """Run all systems on a user-provided query and print comparison metrics."""

    corpus = _prompt_for_dataset()
    
    # Initialize cost tracking (Fix F2: V3 vs TRAG only)
    cost_comparison = ComparisonCostTracker()
    v3_costs = cost_comparison.add_system("V3", model="gpt-4o-mini")
    trad_costs = cost_comparison.add_system("Traditional RAG", model="gpt-4o-mini")
    
    # Track initial costs
    corpus_tokens = corpus_token_count(corpus)
    v3_costs.add_initial(0, "embeddings_local")  # Local embeddings = 0 LLM tokens
    trad_costs.add_initial(0, "embeddings_local")  # Local embeddings = 0 LLM tokens
    
    try:
        user_graph = _build_user_graph(corpus)
        state = get_shared_state(corpus=corpus, refresh=True)
        rag = TraditionalRAG(corpus=corpus)

        while True:
            print("\nChoose an option:")
            print("  1) Ask a query")
            print("  2) Show cost comparison")
            print("  3) Exit")
            choice = input("Selection: ").strip()
            
            if choice == "3":
                break
            if choice == "2":
                print(cost_comparison.comparison_table(n_queries=10))
                continue
            if choice != "1":
                print("Invalid selection.")
                continue

            query = _prompt_for_query()

            systems = {
                "V3": lambda q: run_v3(q, user_graph=user_graph, corpus=corpus, state=state, cost_tracker=v3_costs),
                "Traditional RAG": lambda q: run_traditional_rag(q, rag, cost_tracker=trad_costs),
            }

            print("\n" + "=" * 80)
            print(f"Query: {query}")
            print("=" * 80)
            for name, runner in systems.items():
                print(f"\n{name} is starting...")
                result = runner(query)
                
                # Extract context from prompt for metrics
                context = result["prompt"]
                answer = result["answer"]
                
                # Compute all metrics
                relevance = relevance_score(query, answer)
                faithfulness = faithfulness_score(answer, context)
                ctx_relevance = context_relevance_score(query, context)
                completeness = answer_completeness(query, answer)
                efficiency = token_efficiency(context, answer)
                latency = result["latency_seconds"]
                run_cost = run_cost_tokens(context, answer)
                ans_len = answer_length(answer)
                ctx_len = context_length(context)

                print(f"\n{name}")
                print("  Answer:")
                print(f"  {answer}")
                if "css_final" in result:
                    print(f"  CSS Final:           {result['css_final']:.3f}")
                print(f"  Relevance Score:     {relevance:.2f}")
                print(f"  Faithfulness:        {faithfulness:.2f}")
                print(f"  Context Relevance:   {ctx_relevance:.2f}")
                print(f"  Answer Completeness: {completeness:.2f}")
                print(f"  Token Efficiency:    {efficiency:.2f}")
                print(f"  Latency (s):         {latency:.2f}")
                print(f"  Answer Length:       {ans_len} tokens")
                print(f"  Context Length:      {ctx_len} tokens")
                print(f"  Run Cost:            {run_cost} tokens")
                
                # LLM-as-Judge (standard RAG metric)
                llm_judge = llm_judge_score(query, answer, context)
                if "error" not in llm_judge:
                    print(f"  --- LLM-as-Judge (GPT) ---")
                    print(f"  Correctness:         {llm_judge['correctness']:.2f}")
                    print(f"  Completeness (GPT):  {llm_judge['completeness']:.2f}")
                    print(f"  Conciseness:         {llm_judge['conciseness']:.2f}")
                    print(f"  Overall Quality:     {llm_judge['overall']:.2f}")
                else:
                    print(f"  LLM-as-Judge:        {llm_judge.get('error', 'unavailable')}")
                
                # Official RAGAS Metrics (Academic Standard)
                print(f"  --- RAGAS Official Metrics ---")
                r_relevancy = ragas_answer_relevancy(query, answer)
                r_faithful = ragas_faithfulness(answer, context)
                r_ctx_prec = ragas_context_precision(query, context)
                r_coverage = ragas_aspect_coverage(query, answer)
                print(f"  Answer Relevancy:    {r_relevancy:.2f}")
                print(f"  Faithfulness:        {r_faithful:.2f}")
                print(f"  Context Precision:   {r_ctx_prec:.2f}")
                print(f"  Aspect Coverage:     {r_coverage:.2f}")
                # Context Recall requires ground truth — only shown in batch evaluation
                
                print(f"{name} has been executed.")
                input("Press Enter to run the next model...")
            
            # Show cost comparison after each query round
            print(cost_comparison.comparison_table(n_queries=10))
            
    finally:
        clear_active_corpus()
        clear_shared_state()


if __name__ == "__main__":
    main()
