"""CSS score calculator for V3.

UPDATED (Batch 1 + Batch 3):
  Batch 1: Removed confidence, Semantic/Contradict multipliers
  Batch 3: Updated weights, added subquery_coverage + answer_density,
           simplified token_efficiency, increased budget to 2000

CSS = σ(Σ wᵢ × Fᵢ)
"""

from __future__ import annotations

import math
from typing import List, Dict

from core.types import Graph


# Default token budget B
DEFAULT_TOKEN_BUDGET = 2000  # Fix 2.5: Increased from 1000 for multi-hop legal context

# Feature weights — final configuration (Batch 3 + audit fixes)
WEIGHTS = {
    "query_relevance": 3.0,       # Primary signal: avg cosine(query, node)
    "subquery_coverage": 3.0,     # Fix 1.8 + SC2: Equal to relevance — completeness matters
    "context_cohesion": 1.5,      # Pairwise node similarity
    "graph_connectivity": 1.0,    # Fix 1.6: Quality-aware (sum of edge weights)
    "coverage": 0.5,              # Fix 1.7: Fallback — suppressed when preprocessor active
    "token_efficiency": 0.5,      # Fix 1.5: Constraint, not signal
    "redundancy_penalty": -1.5,   # Penalize near-duplicate content
    # answer_density REMOVED per proposed_fixes.md: correlated with query_relevance
}


def sigmoid(x: float) -> float:
    """Logistic sigmoid: σ(x) = 1 / (1 + e^(-x))"""
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def compute_query_relevance(graph: Graph, query: str, embedder=None) -> float:
    """Average cosine similarity between query and all nodes.
    
    This is the MOST IMPORTANT metric - measures if context is relevant.
    """
    if not graph.nodes or not query:
        return 0.0
    
    if embedder is None:
        from tools.neural_embedder import NeuralEmbedder
        embedder = NeuralEmbedder()
    
    query_vec = embedder.embed(query)
    
    total_sim = 0.0
    count = 0
    for node in graph.nodes:
        if node.embedding:
            node_vec = node.embedding
        else:
            node_vec = embedder.embed(node.text)
        
        # Cosine similarity
        sim = _cosine_similarity(query_vec, node_vec)
        total_sim += sim
        count += 1
    
    return total_sim / max(1, count)


def compute_context_cohesion(graph: Graph, embedder=None) -> float:
    """Average pairwise similarity between nodes.
    
    High cohesion = nodes are semantically related = coherent context.
    """
    if len(graph.nodes) < 2:
        return 1.0  # Single node is perfectly cohesive
    
    if embedder is None:
        from tools.neural_embedder import NeuralEmbedder
        embedder = NeuralEmbedder()
    
    # Get all embeddings
    embeddings = []
    for node in graph.nodes:
        if node.embedding:
            embeddings.append(node.embedding)
        else:
            embeddings.append(embedder.embed(node.text))
    
    # Pairwise similarities (sample if too many nodes)
    total_sim = 0.0
    count = 0
    max_pairs = 50  # Limit computation
    
    for i in range(len(embeddings)):
        for j in range(i + 1, min(len(embeddings), i + 10)):  # Limit pairs per node
            if count >= max_pairs:
                break
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            total_sim += sim
            count += 1
        if count >= max_pairs:
            break
    
    return total_sim / max(1, count)


def compute_redundancy(graph: Graph, embedder=None) -> float:
    """Detect duplicate/highly similar content.
    
    Returns high value if many nodes are near-duplicates.
    """
    if len(graph.nodes) < 2:
        return 0.0
    
    if embedder is None:
        from tools.neural_embedder import NeuralEmbedder
        embedder = NeuralEmbedder()
    
    # Get embeddings
    embeddings = []
    for node in graph.nodes:
        if node.embedding:
            embeddings.append(node.embedding)
        else:
            embeddings.append(embedder.embed(node.text))
    
    # Count near-duplicate pairs (similarity > 0.9)
    duplicate_count = 0
    total_pairs = 0
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim > 0.9:  # Near-duplicate threshold
                duplicate_count += 1
            total_pairs += 1
            if total_pairs >= 50:  # Limit computation
                break
        if total_pairs >= 50:
            break
    
    return duplicate_count / max(1, total_pairs)


def compute_token_efficiency(graph: Graph, budget: int = DEFAULT_TOKEN_BUDGET) -> float:
    """Token efficiency: simple constraint function.
    
    Fix 1.5: Simplified to min(1.0, budget / tokens).
    1.0 when under budget, decreases linearly when over budget.
    """
    tokens = graph.token_count()
    if tokens == 0:
        return 1.0
    return min(1.0, budget / tokens)


def compute_coverage(graph: Graph, query: str) -> float:
    """What fraction of important query terms appear in context?"""
    if not query or not graph.nodes:
        return 0.0
    
    # Extract important words from query (simple: words > 3 chars)
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    if not query_words:
        return 1.0
    
    # Check which appear in graph
    graph_text = " ".join(node.text.lower() for node in graph.nodes)
    found = sum(1 for w in query_words if w in graph_text)
    
    return found / len(query_words)


def compute_subquery_coverage(graph: Graph, query: str, embedder=None) -> float:
    """Fix 1.8 + Fix SC1: Continuous subquery coverage scoring.
    
    UPDATED: Changed from binary threshold (>0.5 → covered) to continuous
    max-similarity scoring. This makes the metric sensitive to pruning:
    removing a node that was the best match for a subquery now LOWERS the score,
    even if another node barely passes the old threshold.
    
    Formula: 0.7 × mean(max_sim per subquery) + 0.3 × entity_fraction
    
    Uses the query preprocessor to get subqueries and entities.
    Returns -1.0 sentinel if preprocessor is unavailable (not 0.5, which
    could be a legitimate score).
    """
    if not graph.nodes or not query:
        return 0.0
    
    # Fix C2: Pass LLM client for intelligent subquery decomposition
    try:
        from tools.query_preprocessor import preprocess_query
        from llm.preprocessor_client import get_preprocessor_client
        llm_client = get_preprocessor_client()
        pp_output = preprocess_query(query, llm_client=llm_client)
    except Exception:
        return -1.0  # Sentinel: preprocessor unavailable (was 0.5, which is ambiguous)
    
    if embedder is None:
        from tools.neural_embedder import NeuralEmbedder
        embedder = NeuralEmbedder()
    
    # 1. Subquery coverage: CONTINUOUS scoring + DIVERSITY bonus
    #    Fix SC1: Use actual best-match similarity (not binary threshold).
    #    Fix SC2: Add diversity bonus — reward having DIFFERENT nodes be the
    #    best match for different subqueries. This prevents the optimizer from
    #    pruning nodes that provide unique coverage for specific subqueries,
    #    which is critical for "list all X" type queries (Trap C).
    subqueries = pp_output.get("subqueries", [])
    if subqueries:
        subquery_sims = []
        best_node_per_sq = []  # Track which node is the best match per subquery

        for sq in subqueries:
            sq_vec = embedder.embed(sq)
            max_sim = 0.0
            best_node_id = None
            for node in graph.nodes:
                node_vec = node.embedding if node.embedding else embedder.embed(node.text)
                sim = _cosine_similarity(sq_vec, node_vec)
                if sim > max_sim:
                    max_sim = sim
                    best_node_id = node.id
            subquery_sims.append(max(0.0, max_sim))
            best_node_per_sq.append(best_node_id)

        # Mean coverage quality
        mean_coverage = sum(subquery_sims) / len(subquery_sims)

        # Diversity bonus: fraction of subqueries served by DISTINCT nodes.
        # If all subqueries are best-matched by the SAME node, diversity = 1/n.
        # If each subquery has a different best node, diversity = 1.0.
        # This rewards keeping nodes that uniquely serve different subqueries.
        unique_best_nodes = len(set(n for n in best_node_per_sq if n is not None))
        diversity = unique_best_nodes / len(subqueries) if subqueries else 0.5

        # Blend: 60% coverage quality + 40% diversity
        subquery_fraction = 0.6 * mean_coverage + 0.4 * diversity
    else:
        subquery_fraction = 0.5
    
    # 2. Entity fraction: how many entities appear in the graph text?
    entities = pp_output.get("entities", [])
    if entities:
        graph_text = " ".join(n.text.lower() for n in graph.nodes)
        found = sum(1 for e in entities if e.lower() in graph_text)
        entity_fraction = found / len(entities)
    else:
        entity_fraction = 0.5
    
    return 0.7 * subquery_fraction + 0.3 * entity_fraction


def compute_answer_density(graph: Graph, query: str, embedder=None) -> float:
    """Fix 1.10: Fraction of nodes directly relevant to the query.
    
    count(nodes with cosine(query, node) > 0.5) / total_nodes
    
    Provides a gradient for pruning: if only 3/7 nodes are directly
    relevant, the optimizer knows there's room to prune.
    """
    if not graph.nodes or not query:
        return 0.0
    
    if embedder is None:
        from tools.neural_embedder import NeuralEmbedder
        embedder = NeuralEmbedder()
    
    query_vec = embedder.embed(query)
    relevant_count = 0
    
    for node in graph.nodes:
        node_vec = node.embedding if node.embedding else embedder.embed(node.text)
        sim = _cosine_similarity(query_vec, node_vec)
        if sim > 0.5:
            relevant_count += 1
    
    return relevant_count / len(graph.nodes)


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot / (norm1 * norm2)


def semantic_score(graph: Graph, query: str, nli) -> float:
    """Semantic(G,x) = f_NLI(G, query)"""
    if not graph.nodes or not query:
        return 0.5
    
    premise = " ".join(node.text for node in graph.nodes if node.text)
    if not premise:
        return 0.5
    
    return nli.score(premise, query)


def contradiction_score(graph: Graph, contradiction_fn) -> float:
    """Contradict(G) = average contradiction across edges."""
    if not graph.edges:
        return 0.0
    
    total = 0.0
    for edge in graph.edges:
        source_text = next((n.text for n in graph.nodes if n.id == edge.source), "")
        target_text = next((n.text for n in graph.nodes if n.id == edge.target), "")
        
        if source_text and target_text:
            score = contradiction_fn.score(source_text, target_text)
            total += score * edge.weight
    
    return total / len(graph.edges)


def compute_css(graph: Graph, query: str, user_graph: Graph, *,
                embedder=None, nli=None, contradiction_fn=None, 
                budget: int = DEFAULT_TOKEN_BUDGET,
                weights_override: dict = None,
                extra_features: dict = None) -> dict:
    """Compute CSS score.
    
    CSS = σ(Σ wᵢ × Fᵢ)
    
    Features: query_relevance (3.0), subquery_coverage (2.5),
    context_cohesion (1.5), graph_connectivity (1.0),
    coverage (0.5 fallback), token_efficiency (0.5),
    redundancy_penalty (-1.5)
    """
    # Use overridden weights if provided (for learning loop optimization)
    weights = weights_override if weights_override is not None else WEIGHTS
    
    # Lazy load embedder only (NLI/contradiction stubs are unused — Fix 1.2+1.3)
    if embedder is None:
        from tools.neural_embedder import NeuralEmbedder
        embedder = NeuralEmbedder()

    # Compute features
    values = {
        "query_relevance": compute_query_relevance(graph, query, embedder),
        "subquery_coverage": compute_subquery_coverage(graph, query, embedder),
        "context_cohesion": compute_context_cohesion(graph, embedder),
        "token_efficiency": compute_token_efficiency(graph, budget),
        "redundancy": compute_redundancy(graph, embedder),
    }
    
    # Fix 1.7 + Fix SC1: Suppress coverage when preprocessor is active.
    # The sentinel -1.0 means "preprocessor unavailable" — use fallback coverage.
    # Any other value (including 0.5, which is now a legitimate score) means
    # the preprocessor is active — suppress the crude coverage feature.
    sq_cov = values["subquery_coverage"]
    if sq_cov < 0:  # Sentinel: preprocessor unavailable
        values["subquery_coverage"] = 0.5  # Neutral fallback for weighted sum
        values["coverage"] = compute_coverage(graph, query)
    else:
        values["coverage"] = 0.0  # Suppressed: subquery_coverage is active
    
    # Compute graph connectivity (V3's graph structure advantage)
    from graph.edge_builder import graph_connectivity_score
    values["graph_connectivity"] = graph_connectivity_score(graph)
    
    # Include extra features from feature discovery (if provided)
    if extra_features:
        values.update(extra_features)
    
    # Weighted sum - only include features that have weights
    weighted_sum = 0.0
    total_positive_weight = 0.0
    for feature_name, weight in weights.items():
        # Map weight keys to feature value keys
        feature_key = feature_name.replace("_penalty", "")  # e.g. "redundancy_penalty" → "redundancy"
        if feature_key in values:
            weighted_sum += weight * values[feature_key]
            if weight > 0:
                total_positive_weight += weight
    
    # Normalize and sigmoid
    # Use dynamic normalization based on total positive weights
    norm_factor = max(total_positive_weight, 1.0)
    normalized = (weighted_sum / norm_factor) * 2 - 1  # Map to [-1, 1] range
    css_score = sigmoid(normalized * 3)  # Scale factor for sensitivity
    
    values["weighted_sum"] = weighted_sum
    values["css_score"] = css_score
    return values


def compute_css_final(graph: Graph, query: str, user_graph: Graph, *,
                      embedder=None, nli=None, contradiction_fn=None, 
                      budget: int = DEFAULT_TOKEN_BUDGET,
                      weights_override: dict = None,
                      extra_features: dict = None) -> dict:
    """Compute final CSS score.
    
    Fix 1.2 + 1.3: Multipliers removed. CSS_final = CSS.
    Semantic and Contradict stubs were non-functional:
      - Semantic was a crude word-overlap gate (1.0 or 0.2), not real NLI
      - Contradict always returned 0.0
    When real NLI is implemented, add them back as FEATURES (not multipliers).
    """
    values = compute_css(graph, query, user_graph, embedder=embedder, 
                         budget=budget,
                         weights_override=weights_override,
                         extra_features=extra_features)
    
    # Fix 1.2 + 1.3: CSS_final = CSS (no multipliers)
    values["css_final"] = values["css_score"]
    
    # Keep stubs zeroed for backward compatibility of return dict
    values["semantic"] = 1.0
    values["contradict"] = 0.0
    
    return values
