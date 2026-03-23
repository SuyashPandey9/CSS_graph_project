"""CSS Weight Optimizer for the Learning Loop.

Optimizes the WEIGHTS dict in css/calculator.py using either
Optuna (Bayesian optimization) or a fallback random search.

The optimizer tests different weight configurations by running
the full V3 pipeline on CUAD evaluation queries and measuring
answer quality via embedding-based metrics.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

from core.frozen_state import FrozenState, get_shared_state
from core.types import Graph
from css.calculator import compute_css_final
from data.cuad_eval_dataset import (
    CUAD_EVAL_QUERIES,
    compute_answer_keyword_coverage,
    compute_entity_retrieval_coverage,
)
from learning.legal_features import compute_all_legal_features
from learning.results_logger import ResultsLogger


# Search bounds for each CSS weight
WEIGHT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "query_relevance":    (0.5, 5.0),
    "context_cohesion":   (0.0, 3.0),
    "graph_connectivity": (0.5, 5.0),
    "token_efficiency":   (0.5, 4.0),
    "coverage":           (0.0, 3.0),
    "confidence":         (0.0, 3.0),
    "redundancy_penalty": (-3.0, 0.0),
}

# Bounds for new legal features (added if selected by feature discovery)
LEGAL_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "legal_entity_density":   (0.0, 3.0),
    "clause_coverage":        (0.0, 3.0),
    "cross_reference_density": (0.0, 3.0),
    "section_diversity":      (0.0, 3.0),
    "answer_specificity":     (0.0, 3.0),
}


def _sample_weights(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Sample random weights within bounds."""
    return {k: random.uniform(lo, hi) for k, (lo, hi) in bounds.items()}


def _evaluate_weights(
    weights: Dict[str, float],
    queries: List[Dict],
    state: FrozenState,
    n_queries: int = 10,
    selected_features: List[str] = None,
) -> Tuple[float, List[Dict]]:
    """Run full V3 pipeline with given weights and compute reward.
    
    Args:
        weights: CSS weight dict to test
        queries: Evaluation queries
        state: FrozenState
        n_queries: Number of queries
        selected_features: Feature set from discovery phase
    
    Returns:
        (mean_reward, per_query_scores)
    """
    from policy.optimizer import optimize
    from evaluation.metrics_calculator import (
        relevance_score,
        context_relevance_score,
    )
    
    selected_queries = queries[:n_queries]
    per_query_scores: List[Dict] = []
    user_graph = Graph(nodes=[], edges=[])
    
    for q_entry in selected_queries:
        query = q_entry["query"]
        
        try:
            # Run V3 optimization
            best_graph = optimize(query, user_graph, max_steps=3, state=state)
            
            # Compute legal features if needed
            extra_features = None
            if selected_features:
                legal_feats = compute_all_legal_features(best_graph, query)
                extra_features = {
                    k: v for k, v in legal_feats.items() 
                    if k in selected_features
                }
            
            # Compute CSS with trial weights
            css_result = compute_css_final(
                best_graph, query, user_graph,
                embedder=state.embedder,
                weights_override=weights,
                extra_features=extra_features,
            )
            
            # Build context text
            context = " ".join(node.text for node in best_graph.nodes)
            
            # Compute reward metrics (embedding-based, free)
            entity_cov = compute_entity_retrieval_coverage(context, q_entry)
            ctx_rel = context_relevance_score(query, context)
            kw_cov = compute_answer_keyword_coverage(context, q_entry)
            
            # Reward = weighted combination
            reward = (0.35 * ctx_rel + 
                     0.35 * entity_cov + 
                     0.30 * kw_cov)
            
            per_query_scores.append({
                "query": query,
                "category": q_entry["category"],
                "css_score": css_result.get("css_score", 0),
                "css_final": css_result.get("css_final", 0),
                "context_relevance": ctx_rel,
                "entity_coverage": entity_cov,
                "keyword_coverage": kw_cov,
                "reward": reward,
                "n_nodes": len(best_graph.nodes),
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


def optimize_weights_optuna(
    n_trials: int = 50,
    n_queries: int = 10,
    selected_features: List[str] = None,
    state: FrozenState = None,
    logger: ResultsLogger = None,
) -> Dict[str, float]:
    """Optimize CSS weights using Optuna Bayesian optimization.
    
    Falls back to random search if Optuna is not installed.
    
    Args:
        n_trials: Number of optimization trials
        n_queries: Number of eval queries per trial
        selected_features: Feature set from discovery phase
        state: FrozenState
        logger: Results logger
    
    Returns:
        Best weight configuration found
    """
    state = state or get_shared_state()
    logger = logger or ResultsLogger()
    queries = CUAD_EVAL_QUERIES
    
    # Determine which features to optimize weights for
    bounds = dict(WEIGHT_BOUNDS)
    if selected_features:
        for feat in selected_features:
            if feat in LEGAL_FEATURE_BOUNDS:
                bounds[feat] = LEGAL_FEATURE_BOUNDS[feat]
    
    print(f"\n{'='*60}")
    print(f"CSS WEIGHT OPTIMIZATION")
    print(f"Trials: {n_trials}  |  Queries: {min(n_queries, len(queries))}  |  Params: {len(bounds)}")
    print(f"{'='*60}\n")
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return _optimize_with_optuna(
            bounds, queries, state, logger, n_trials, n_queries, selected_features
        )
    except ImportError:
        print("[WeightOptimizer] Optuna not installed, using random search fallback.")
        return _optimize_random_search(
            bounds, queries, state, logger, n_trials, n_queries, selected_features
        )


def _optimize_with_optuna(
    bounds, queries, state, logger, n_trials, n_queries, selected_features
) -> Dict[str, float]:
    """Use Optuna TPE sampler for Bayesian optimization."""
    import optuna
    
    best_weights = {}
    best_reward = -1.0
    
    def objective(trial):
        nonlocal best_weights, best_reward
        
        weights = {}
        for name, (lo, hi) in bounds.items():
            weights[name] = trial.suggest_float(name, lo, hi)
        
        reward, per_query = _evaluate_weights(
            weights, queries, state, n_queries, selected_features
        )
        
        logger.log_trial(
            trial_id=trial.number,
            phase="css_weights",
            params=weights,
            per_query_scores=per_query,
            mean_reward=reward,
        )
        
        if reward > best_reward:
            best_reward = reward
            best_weights = dict(weights)
            print(f"  ✓ Trial {trial.number}: NEW BEST reward={reward:.4f}")
        else:
            print(f"    Trial {trial.number}: reward={reward:.4f}")
        
        return reward
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials)
    
    logger.log_best("css_weights", best_weights, best_reward)
    
    print(f"\n  Best weights: {best_weights}")
    print(f"  Best reward: {best_reward:.4f}\n")
    
    return best_weights


def _optimize_random_search(
    bounds, queries, state, logger, n_trials, n_queries, selected_features
) -> Dict[str, float]:
    """Fallback: random search over weight space."""
    best_weights = {}
    best_reward = -1.0
    
    for trial_id in range(n_trials):
        weights = _sample_weights(bounds)
        
        reward, per_query = _evaluate_weights(
            weights, queries, state, n_queries, selected_features
        )
        
        logger.log_trial(
            trial_id=trial_id,
            phase="css_weights",
            params=weights,
            per_query_scores=per_query,
            mean_reward=reward,
        )
        
        if reward > best_reward:
            best_reward = reward
            best_weights = dict(weights)
            print(f"  ✓ Trial {trial_id}: NEW BEST reward={reward:.4f}")
        else:
            print(f"    Trial {trial_id}: reward={reward:.4f}")
    
    logger.log_best("css_weights", best_weights, best_reward)
    
    print(f"\n  Best weights: {best_weights}")
    print(f"  Best reward: {best_reward:.4f}\n")
    
    return best_weights
