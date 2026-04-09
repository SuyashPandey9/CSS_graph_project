"""Graph expansion transform for V3.

UPDATED:
  Fix 3.5: Edge-traversal expand — follows pre-computed cross-ref edges.
  Fix E1:  Ranked expansion — collects ALL candidate targets, scores each
           by cosine(query, candidate_embedding), picks top_k by score.
           Prevents wasted steps from irrelevant first-found edges.
"""

from __future__ import annotations
import math
from typing import List, Tuple

from core.frozen_state import FrozenState
from core.types import Graph, Node


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Cosine similarity between two vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0


def expand(graph: Graph, state: FrozenState, top_k: int = 2,
           query: str = None) -> Graph:
    """Add new nodes via edge-traversal or FAISS similarity.

    Fix 3.5: Follows pre-computed cross-reference edges from FrozenState.
    Fix E1:  Collects ALL cross-ref candidates first, then scores by
             cosine(query_vec, candidate_embedding) and picks the best top_k.
             Uses pre-computed embeddings — no extra LLM/embedding calls.
    Falls back to FAISS similarity search when no cross-ref candidates found.
    """
    doc_map = {doc.id: doc for doc in state.corpus.documents}
    existing_ids = {node.id for node in graph.nodes}
    precomputed = getattr(state, 'precomputed_edges', None) or []

    # --- Phase 1: Collect ALL candidate expansion targets (no early break) ---
    # Key change from old code: we no longer break at top_k here.
    # We gather everything, score it, then pick the best.
    candidate_ids = set()  # deduplicate

    # Source 1: Pre-computed cross-ref edges from FrozenState
    for node in graph.nodes:
        for edge in precomputed:
            if isinstance(edge, tuple) and len(edge) >= 2:
                src, tgt = edge[0], edge[1]
            elif hasattr(edge, 'source'):
                src, tgt = edge.source, edge.target
            else:
                continue

            if src == node.id and tgt not in existing_ids and tgt in doc_map:
                candidate_ids.add(tgt)
            elif tgt == node.id and src not in existing_ids and src in doc_map:
                candidate_ids.add(src)

    # Source 2: Cross-ref edges already embedded in the graph
    for edge in graph.edges:
        if "cross_ref" not in edge.relation:
            continue
        for target_id in [edge.source, edge.target]:
            if target_id not in existing_ids and target_id in doc_map:
                candidate_ids.add(target_id)

    # --- Phase 2: Score all candidates, pick top_k by query relevance ---
    # Fix E1: rank by cosine(query_vec, candidate_embedding)
    scored_candidates: List[Tuple[float, str]] = []

    if candidate_ids:
        query_vec = state.embedder.embed(query) if query else None

        for cand_id in candidate_ids:
            emb = state.corpus_embeddings.get(cand_id)
            if query_vec and emb:
                score = _cosine_similarity(query_vec, emb)
            else:
                score = 0.85  # default when no query/embedding available
            scored_candidates.append((score, cand_id))

        # Sort descending by relevance score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        if scored_candidates:
            print(f"[expand] Ranked {len(scored_candidates)} cross-ref candidates, "
                  f"picking top {top_k}. Scores: "
                  f"{[round(s, 3) for s, _ in scored_candidates[:5]]}")

    # Build new nodes from top_k candidates
    new_nodes = []
    for score, cand_id in scored_candidates[:top_k]:
        doc = doc_map[cand_id]
        new_node = Node(
            id=cand_id,
            text=doc.text,
            metadata={"source": "cross_ref_expand", "title": doc.title,
                      "expand_score": round(score, 4)},
            confidence=min(score + 0.1, 1.0),  # slight boost for structural link
            embedding=state.corpus_embeddings.get(cand_id),
        )
        new_nodes.append(new_node)
        existing_ids.add(cand_id)

    # --- Phase 3: FAISS fallback if cross-ref didn't fill top_k slots ---
    still_needed = top_k - len(new_nodes)
    if still_needed > 0:
        fallback_query = query
        if not fallback_query and graph.nodes:
            fallback_query = graph.nodes[0].text  # legacy fallback

        if fallback_query:
            q_vec = state.embedder.embed(fallback_query)
            results = state.search_similar(
                q_vec, top_k=still_needed, exclude_ids=existing_ids
            )
            for doc_id, sim_score in results:
                if doc_id in existing_ids or doc_id not in doc_map:
                    continue
                doc = doc_map[doc_id]
                new_node = Node(
                    id=doc_id,
                    text=doc.text,
                    metadata={"source": "similarity_expand", "title": doc.title,
                              "expand_score": round(sim_score * 0.9, 4)},
                    confidence=sim_score * 0.9,
                    embedding=state.corpus_embeddings.get(doc_id),
                )
                new_nodes.append(new_node)
                existing_ids.add(doc_id)
    
    # Build new graph with all nodes
    all_nodes = list(graph.nodes) + new_nodes
    
    # Rebuild edges for the expanded graph — pass pre-computed data from FrozenState
    from graph.edge_builder import build_edges
    expanded_graph = Graph(nodes=all_nodes, edges=[])
    return build_edges(
        expanded_graph, 
        embedder=state.embedder,
        idf_dict=getattr(state, 'idf_dict', None),
        precomputed_edges=precomputed,
    )
