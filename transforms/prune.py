"""Graph pruning transform for V3.

UPDATED (Fix 2.12 + Fix B1 + Fix SC3): Three layers of node protection.
  1. Type-based bridge protection (Fix B1):
     Protects nodes with structural edges (cross_ref or defined_term).
  2. High-degree bridge protection (Fix 2.12):
     Protects connector nodes with degree >= 3 and cross_ref edges.
  3. Subquery coverage protection (Fix SC3 — NEW):
     Protects nodes that are the SOLE good match for a decomposed subquery.
     This prevents pruning away nodes that provide unique coverage for
     "list all X" type queries (Trap C).

V3 Spec: T_prune(G, s) removes the least query-relevant node.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from core.frozen_state import FrozenState
from core.types import Graph, Node, Edge


# Structural edge types that indicate author-intended document connections.
_STRUCTURAL_EDGE_TYPES = {"cross_ref", "defined_term"}

# Minimum similarity for a node to "cover" a subquery
_SUBQUERY_COVERAGE_THRESHOLD = 0.35


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _has_structural_edges(graph: Graph, node_id: str) -> bool:
    """Check if a node has any structural edges (cross_ref or defined_term)."""
    for edge in graph.edges:
        if edge.source == node_id or edge.target == node_id:
            for edge_type in _STRUCTURAL_EDGE_TYPES:
                if edge_type in edge.relation:
                    return True
    return False


def _get_subquery_critical_nodes(graph: Graph, query: str, 
                                  embedder) -> Set[str]:
    """Find nodes that are the SOLE good match for a decomposed subquery.
    
    Fix SC3: A node is "subquery-critical" if:
    1. It's the best match for at least one subquery
    2. No OTHER node in the graph has similarity > threshold for that subquery
    
    These nodes provide unique information coverage — pruning them means
    losing an entire aspect of the answer.
    
    Returns:
        Set of node IDs that should be protected.
    """
    try:
        from tools.query_preprocessor import preprocess_query
        from llm.preprocessor_client import get_preprocessor_client
        llm_client = get_preprocessor_client()
        pp_output = preprocess_query(query, llm_client=llm_client)
    except Exception:
        return set()  # Can't determine critical nodes without preprocessor
    
    subqueries = pp_output.get("subqueries", [])
    if not subqueries:
        return set()
    
    # For each subquery, find all nodes with similarity > threshold
    critical_nodes: Set[str] = set()
    
    for sq in subqueries:
        sq_vec = embedder.embed(sq)
        
        # Score all nodes against this subquery
        node_scores: List[tuple] = []  # (node_id, similarity)
        for node in graph.nodes:
            node_vec = node.embedding if node.embedding else embedder.embed(node.text)
            sim = _cosine_similarity(sq_vec, node_vec)
            node_scores.append((node.id, sim))
        
        # Sort by similarity (highest first)
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not node_scores:
            continue
        
        best_id, best_sim = node_scores[0]
        
        # If best match is below threshold, no node covers this subquery well
        if best_sim < _SUBQUERY_COVERAGE_THRESHOLD:
            continue
        
        # Check if there's a viable backup (second-best above threshold)
        has_backup = False
        if len(node_scores) > 1:
            second_id, second_sim = node_scores[1]
            # Backup must be at least 80% as good as the best
            has_backup = second_sim >= _SUBQUERY_COVERAGE_THRESHOLD and \
                         second_sim >= best_sim * 0.8
        
        # If no good backup exists, the best node is critical
        if not has_backup:
            critical_nodes.add(best_id)
    
    return critical_nodes


def prune(graph: Graph, state: FrozenState, min_nodes: int = 3, 
          query: str = None) -> Graph:
    """Remove the least relevant node, with three layers of protection.
    
    Protection layers:
    1. Fix B1: Structural edges (cross_ref, defined_term) → +0.5 boost
    2. Fix 2.12: High-degree bridges (degree >= 3 + cross_ref) → +0.3 boost  
    3. Fix SC3: Subquery-critical nodes (sole match for a subquery) → +0.4 boost
    
    Args:
        graph: Current graph
        state: FrozenState with tools
        min_nodes: Minimum nodes to keep
        query: Query for relevance scoring
    """
    if len(graph.nodes) <= min_nodes:
        return Graph(nodes=list(graph.nodes), edges=list(graph.edges))
    
    from graph.edge_builder import get_node_degree, has_cross_ref_edges
    
    # Get query embedding for relevance scoring
    query_vec = None
    if query:
        query_vec = state.embedder.embed(query)
    
    # Fix SC3: Find subquery-critical nodes
    critical_for_subquery: Set[str] = set()
    if query:
        critical_for_subquery = _get_subquery_critical_nodes(
            graph, query, state.embedder
        )
    
    # Score each node: relevance to query + protection boosts
    candidates = []
    for node in graph.nodes:
        # Relevance score
        if query_vec:
            node_vec = node.embedding if node.embedding else state.embedder.embed(node.text)
            relevance = _cosine_similarity(query_vec, node_vec)
        else:
            relevance = node.confidence
        
        effective_score = relevance
        
        # Protection Layer 1: Structural edges (Fix B1)
        if _has_structural_edges(graph, node.id):
            effective_score += 0.5
        
        # Protection Layer 2: High-degree bridges (Fix 2.12)
        degree = get_node_degree(graph, node.id)
        if degree >= 3 and has_cross_ref_edges(graph, node.id):
            effective_score += 0.3
        
        # Protection Layer 3: Subquery-critical nodes (Fix SC3)
        if node.id in critical_for_subquery:
            effective_score += 0.4
        
        candidates.append((node, effective_score))
    
    # Remove the node with lowest effective score
    candidates.sort(key=lambda x: x[1])
    node_to_remove = candidates[0][0]
    
    # Build new graph without the removed node
    remaining_nodes = [n for n in graph.nodes if n.id != node_to_remove.id]
    remaining_edges = [
        e for e in graph.edges
        if e.source != node_to_remove.id and e.target != node_to_remove.id
    ]
    
    return Graph(nodes=remaining_nodes, edges=remaining_edges)
