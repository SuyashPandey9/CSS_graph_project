"""Edge Weight Optimizer for the Learning Loop.

Optimizes edge-building parameters in graph/edge_builder.py to maximize
graph connectivity quality and answer relevance.

Focuses on:
- Continuity: same-source chunks staying connected
- Similarity: semantically related chunks being linked
- Entity overlap: legal-term-sharing chunks being connected
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from core.frozen_state import FrozenState, get_shared_state
from core.types import Graph
from css.calculator import compute_css_final
from data.cuad_eval_dataset import (
    CUAD_EVAL_QUERIES,
    compute_answer_keyword_coverage,
    compute_entity_retrieval_coverage,
)
from graph.edge_builder import DEFAULT_EDGE_PARAMS
from learning.legal_features import compute_all_legal_features
from learning.results_logger import ResultsLogger


# Search bounds for edge parameters
EDGE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "entity_overlap_weight":  (0.1, 0.8),
    "entity_overlap_max_count": (1, 5),
    "same_source_weight":     (0.0, 0.6),
    "semantic_weight":        (0.1, 0.6),
    "semantic_threshold":     (0.3, 0.8),
    "lexical_weight":         (0.0, 0.4),
    "lexical_threshold":      (0.05, 0.3),
    "edge_threshold":         (0.05, 0.4),
}


def _sample_edge_params(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Sample random edge parameters."""
    return {k: random.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _evaluate_edge_params(
    edge_params: Dict[str, float],
    css_weights: Dict[str, float],
    queries: List[Dict],
    state: FrozenState,
    n_queries: int = 10,
    selected_features: List[str] = None,
) -> Tuple[float, List[Dict]]:
    """Evaluate edge parameters by building graphs with them and scoring.
    
    This runs a modified pipeline where:
    1. Initial retrieval is the same (FAISS)
    2. Edge building uses the trial edge_params
    3. CSS evaluation uses the (already optimized) css_weights
    4. Answer quality is measured
    
    Args:
        edge_params: Edge weight configuration to test
        css_weights: CSS weights (from Phase 2 optimization)
        queries: Evaluation queries
        state: FrozenState
        n_queries: Number of queries
        selected_features: Feature set for extra CSS features
    
    Returns:
        (mean_reward, per_query_scores)
    """
    from core.types import Node
    from core.query_graph import get_query_entities
    from graph.edge_builder import build_edges
    from evaluation.metrics_calculator import context_relevance_score
    
    selected_queries = queries[:n_queries]
    per_query_scores: List[Dict] = []
    user_graph = Graph(nodes=[], edges=[])
    
    for q_entry in selected_queries:
        query = q_entry["query"]
        
        try:
            # Step 1: Retrieve chunks (same as normal pipeline)
            query_vec = state.embedder.embed(query)
            results = state.search_similar(query_vec, top_k=6)
            
            doc_map = {doc.id: doc for doc in state.corpus.documents}
            nodes = []
            for doc_id, score in results:
                if doc_id in doc_map:
                    doc = doc_map[doc_id]
                    nodes.append(Node(
                        id=doc_id,
                        text=doc.text,
                        metadata={"title": doc.title, "source": doc.source},
                        confidence=score,
                        embedding=state.corpus_embeddings.get(doc_id),
                    ))
            
            # Step 2: Build edges with trial edge_params
            raw_graph = Graph(nodes=nodes, edges=[])
            graph_with_edges = build_edges(
                raw_graph, embedder=state.embedder, edge_params=edge_params
            )
            
            # Step 3: Compute extra features
            extra_features = None
            if selected_features:
                legal_feats = compute_all_legal_features(graph_with_edges, query)
                extra_features = {
                    k: v for k, v in legal_feats.items()
                    if k in selected_features
                }
            
            # Step 4: Evaluate with CSS using optimized weights
            css_result = compute_css_final(
                graph_with_edges, query, user_graph,
                embedder=state.embedder,
                weights_override=css_weights,
                extra_features=extra_features,
            )
            
            # Step 5: Compute reward
            context = " ".join(node.text for node in graph_with_edges.nodes)
            entity_cov = compute_entity_retrieval_coverage(context, q_entry)
            ctx_rel = context_relevance_score(query, context)
            kw_cov = compute_answer_keyword_coverage(context, q_entry)
            
            # Graph quality bonus: reward well-connected graphs
            n_edges = len(graph_with_edges.edges)
            n_nodes = len(graph_with_edges.nodes)
            edge_density = n_edges / max(1, n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            
            # Reward includes graph quality
            reward = (
                0.30 * ctx_rel +
                0.30 * entity_cov +
                0.20 * kw_cov +
                0.10 * edge_density +
                0.10 * css_result.get("graph_connectivity", 0)
            )
            
            per_query_scores.append({
                "query": query,
                "category": q_entry["category"],
                "css_final": css_result.get("css_final", 0),
                "graph_connectivity": css_result.get("graph_connectivity", 0),
                "edge_count": n_edges,
                "edge_density": edge_density,
                "context_relevance": ctx_rel,
                "entity_coverage": entity_cov,
                "keyword_coverage": kw_cov,
                "reward": reward,
            })
        except Exception as e:
            per_query_scores.append({
                "query": query,
                "error": str(e),
                "reward": 0.0,
            })
    
    valid_scores = [s["reward"] for s in per_query_scores if "error" not in s]
    mean_reward = sum(valid_scores) / max(1, len(valid_scores))
    
    return mean_reward, per_query_scores


def optimize_edge_weights(
    css_weights: Dict[str, float],
    n_trials: int = 50,
    n_queries: int = 10,
    selected_features: List[str] = None,
    state: FrozenState = None,
    logger: ResultsLogger = None,
) -> Dict[str, float]:
    """Optimize edge-building parameters.
    
    Uses the same optimization strategy as CSS weights.
    Edge optimization happens AFTER CSS weight optimization,
    using the already-optimized CSS weights.
    
    Args:
        css_weights: Optimized CSS weights from Phase 2
        n_trials: Number of trials
        n_queries: Number of eval queries per trial
        selected_features: Feature set from Phase 1
        state: FrozenState
        logger: Results logger
    
    Returns:
        Best edge parameter configuration
    """
    state = state or get_shared_state()
    logger = logger or ResultsLogger()
    queries = CUAD_EVAL_QUERIES
    
    print(f"\n{'='*60}")
    print(f"EDGE WEIGHT OPTIMIZATION")
    print(f"Trials: {n_trials}  |  Queries: {min(n_queries, len(queries))}  |  Params: {len(EDGE_BOUNDS)}")
    print(f"{'='*60}\n")
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return _optimize_with_optuna(
            css_weights, queries, state, logger,
            n_trials, n_queries, selected_features
        )
    except ImportError:
        print("[EdgeOptimizer] Optuna not installed, using random search.")
        return _optimize_random_search(
            css_weights, queries, state, logger,
            n_trials, n_queries, selected_features
        )


def _optimize_with_optuna(
    css_weights, queries, state, logger, n_trials, n_queries, selected_features
) -> Dict[str, float]:
    """Optuna-based edge parameter optimization."""
    import optuna
    
    best_params = dict(DEFAULT_EDGE_PARAMS)
    best_reward = -1.0
    
    def objective(trial):
        nonlocal best_params, best_reward
        
        params = {}
        for name, (lo, hi) in EDGE_BOUNDS.items():
            if name == "entity_overlap_max_count":
                params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = trial.suggest_float(name, lo, hi)
        
        reward, per_query = _evaluate_edge_params(
            params, css_weights, queries, state, n_queries, selected_features
        )
        
        logger.log_trial(
            trial_id=trial.number,
            phase="edge_weights",
            params=params,
            per_query_scores=per_query,
            mean_reward=reward,
        )
        
        if reward > best_reward:
            best_reward = reward
            best_params = dict(params)
            print(f"  ✓ Trial {trial.number}: NEW BEST reward={reward:.4f}")
        else:
            print(f"    Trial {trial.number}: reward={reward:.4f}")
        
        return reward
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    logger.log_best("edge_weights", best_params, best_reward)
    
    print(f"\n  Best edge params: {best_params}")
    print(f"  Best reward: {best_reward:.4f}\n")
    
    return best_params


def _optimize_random_search(
    css_weights, queries, state, logger, n_trials, n_queries, selected_features
) -> Dict[str, float]:
    """Random search fallback."""
    best_params = dict(DEFAULT_EDGE_PARAMS)
    best_reward = -1.0
    
    # Trial 0: evaluate baseline (current defaults)
    print(f"  Trial 0: testing baseline (default edge params)")
    baseline_reward, baseline_scores = _evaluate_edge_params(
        DEFAULT_EDGE_PARAMS, css_weights, queries, state, n_queries, selected_features
    )
    best_reward = baseline_reward
    print(f"  ✓ Baseline reward: {baseline_reward:.4f}")
    
    logger.log_trial(
        trial_id=0,
        phase="edge_weights",
        params=dict(DEFAULT_EDGE_PARAMS),
        per_query_scores=baseline_scores,
        mean_reward=baseline_reward,
        extra={"is_baseline": True},
    )
    
    for trial_id in range(1, n_trials):
        params = _sample_edge_params(EDGE_BOUNDS)
        
        reward, per_query = _evaluate_edge_params(
            params, css_weights, queries, state, n_queries, selected_features
        )
        
        logger.log_trial(
            trial_id=trial_id,
            phase="edge_weights",
            params=params,
            per_query_scores=per_query,
            mean_reward=reward,
        )
        
        if reward > best_reward:
            best_reward = reward
            best_params = dict(params)
            print(f"  ✓ Trial {trial_id}: NEW BEST reward={reward:.4f}")
        else:
            print(f"    Trial {trial_id}: reward={reward:.4f}")
    
    logger.log_best("edge_weights", best_params, best_reward)
    
    print(f"\n  Best edge params: {best_params}")
    print(f"  Best reward: {best_reward:.4f}\n")
    
    return best_params
