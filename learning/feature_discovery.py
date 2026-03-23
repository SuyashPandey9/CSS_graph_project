"""Feature Discovery for CSS Learning Loop.

Implements greedy forward selection with leave-one-out ablation to identify
which CSS features matter most for legal document retrieval on CUAD.

Algorithm:
    1. Start with empty feature set S = {}
    2. For each candidate feature: add to S, evaluate, record reward
    3. Add the feature with highest marginal reward to S
    4. Repeat until adding features doesn't improve reward by > ε
    5. Ablation: test removing each feature from final S
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

from core.frozen_state import FrozenState, get_shared_state
from core.types import Graph
from css.calculator import WEIGHTS, compute_css_final
from data.cuad_eval_dataset import (
    CUAD_EVAL_QUERIES,
    compute_entity_retrieval_coverage,
)
from learning.legal_features import LEGAL_FEATURES, compute_all_legal_features
from learning.results_logger import ResultsLogger


# All candidate features: existing CSS features + new legal features
EXISTING_FEATURE_NAMES = [
    "query_relevance",
    "context_cohesion",
    "graph_connectivity",
    "token_efficiency",
    "coverage",
    "confidence",
    "redundancy",
]

NEW_FEATURE_NAMES = list(LEGAL_FEATURES.keys())

ALL_CANDIDATE_FEATURES = EXISTING_FEATURE_NAMES + NEW_FEATURE_NAMES

# Minimum improvement to add a feature
EPSILON = 0.005


def _evaluate_feature_set(
    feature_set: List[str],
    queries: List[Dict],
    state: FrozenState,
    n_queries: int = 10,
) -> Tuple[float, List[Dict]]:
    """Evaluate a feature set by running the full pipeline on eval queries.
    
    Uses embedding-based metrics (free, no API calls) as primary signal.
    
    Args:
        feature_set: List of feature names to include
        queries: Evaluation queries
        state: FrozenState with corpus and embedder
        n_queries: Number of queries to evaluate
    
    Returns:
        (mean_reward, per_query_scores)
    """
    from policy.optimizer import optimize
    from evaluation.metrics_calculator import (
        relevance_score,
        faithfulness_score,
        context_relevance_score,
    )
    
    # Build weights dict: only include selected features, zero out others
    trial_weights = {}
    for feature in ALL_CANDIDATE_FEATURES:
        # Use default weights for selected existing features
        if feature in feature_set and feature in EXISTING_FEATURE_NAMES:
            weight_key = feature + "_penalty" if feature == "redundancy" else feature
            trial_weights[weight_key] = WEIGHTS.get(weight_key, WEIGHTS.get(feature, 1.0))
        elif feature in feature_set and feature in NEW_FEATURE_NAMES:
            trial_weights[feature] = 1.0  # Default weight for new features
        # Features not in feature_set are omitted (weight = 0)
    
    selected_queries = queries[:n_queries]
    per_query_scores = []
    
    user_graph = Graph(nodes=[], edges=[])
    
    for q_entry in selected_queries:
        query = q_entry["query"]
        
        try:
            # Compute legal features for extra_features
            extra = {}
            for fname in feature_set:
                if fname in NEW_FEATURE_NAMES:
                    # Will be computed during CSS calculation via extra_features
                    extra[fname] = None  # placeholder, computed below
            
            # Run optimization to get best graph
            best_graph = optimize(
                query, user_graph, max_steps=3, state=state,
            )
            
            # Compute extra legal features on the result graph
            if any(f in NEW_FEATURE_NAMES for f in feature_set):
                legal_feats = compute_all_legal_features(best_graph, query)
                extra = {k: v for k, v in legal_feats.items() if k in feature_set}
            
            # Build context text from graph
            context = " ".join(node.text for node in best_graph.nodes)
            
            # Compute reward using embedding-based metrics (free)
            entity_cov = compute_entity_retrieval_coverage(context, q_entry)
            ctx_rel = context_relevance_score(query, context)
            
            # Reward = weighted combo of metrics
            reward = 0.5 * ctx_rel + 0.5 * entity_cov
            
            per_query_scores.append({
                "query": query,
                "category": q_entry["category"],
                "context_relevance": ctx_rel,
                "entity_coverage": entity_cov,
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


def greedy_forward_selection(
    n_queries: int = 10,
    state: FrozenState = None,
    logger: ResultsLogger = None,
) -> List[str]:
    """Greedy forward selection of CSS features.
    
    Starts empty, adds the feature with highest marginal reward each round.
    
    Args:
        n_queries: Number of eval queries per evaluation
        state: FrozenState (creates one if None)
        logger: Optional results logger
    
    Returns:
        Selected feature set (ordered by selection order)
    """
    state = state or get_shared_state()
    logger = logger or ResultsLogger()
    queries = CUAD_EVAL_QUERIES
    
    selected: List[str] = []
    remaining = list(ALL_CANDIDATE_FEATURES)
    best_reward = 0.0
    trial_id = 0
    
    print(f"\n{'='*60}")
    print(f"FEATURE DISCOVERY - Greedy Forward Selection")
    print(f"Candidates: {len(remaining)}  |  Queries: {min(n_queries, len(queries))}")
    print(f"{'='*60}\n")
    
    while remaining:
        round_results: List[Tuple[str, float]] = []
        
        for candidate in remaining:
            trial_features = selected + [candidate]
            trial_id += 1
            
            print(f"  Trial {trial_id}: testing +{candidate} (set={len(trial_features)} features)")
            
            reward, per_query = _evaluate_feature_set(
                trial_features, queries, state, n_queries
            )
            
            round_results.append((candidate, reward))
            
            logger.log_trial(
                trial_id=trial_id,
                phase="feature_discovery",
                params={"added_feature": candidate, "feature_set": trial_features},
                per_query_scores=per_query,
                mean_reward=reward,
            )
        
        # Pick best feature this round
        round_results.sort(key=lambda x: x[1], reverse=True)
        best_candidate, best_candidate_reward = round_results[0]
        
        # Check if improvement exceeds epsilon
        improvement = best_candidate_reward - best_reward
        
        print(f"\n  Best: +{best_candidate} → reward={best_candidate_reward:.4f} "
              f"(Δ={improvement:+.4f})")
        
        if improvement < EPSILON and selected:
            print(f"  STOPPING: improvement {improvement:.4f} < ε={EPSILON}")
            break
        
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_reward = best_candidate_reward
        
        print(f"  Selected: {selected}\n")
    
    # Ablation: test removing each feature
    print(f"\n{'='*60}")
    print(f"ABLATION - Testing removal of each feature")
    print(f"{'='*60}\n")
    
    final_set = list(selected)
    
    for feature in selected:
        ablated = [f for f in selected if f != feature]
        trial_id += 1
        
        print(f"  Trial {trial_id}: testing -{feature}")
        
        ablated_reward, per_query = _evaluate_feature_set(
            ablated, queries, state, n_queries
        )
        
        drop = best_reward - ablated_reward
        print(f"    Without {feature}: reward={ablated_reward:.4f} (drop={drop:+.4f})")
        
        logger.log_trial(
            trial_id=trial_id,
            phase="feature_ablation",
            params={"removed_feature": feature, "feature_set": ablated},
            per_query_scores=per_query,
            mean_reward=ablated_reward,
            extra={"reward_drop": drop},
        )
        
        # If removing the feature doesn't hurt, drop it
        if drop <= 0:
            print(f"    → REMOVING {feature} (no contribution)")
            final_set.remove(feature)
    
    logger.log_best("feature_discovery", {"selected_features": final_set}, best_reward)
    
    print(f"\n{'='*60}")
    print(f"FINAL FEATURE SET: {final_set}")
    print(f"{'='*60}\n")
    
    return final_set
