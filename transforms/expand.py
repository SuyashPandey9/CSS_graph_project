"""Graph expansion transform for V3.

UPDATED (Fix 3.5 — Deviation fix): Edge-traversal expand.
  - Follows EXISTING cross-reference edges in the graph (no runtime regex)
  - Falls back to query-based FAISS if no cross-ref edges exist
  - Prevents redundant retrievals via seen_ids tracking
"""

from __future__ import annotations

from core.frozen_state import FrozenState
from core.types import Graph, Node


def expand(graph: Graph, state: FrozenState, top_k: int = 2, 
           query: str = None) -> Graph:
    """Add new nodes via edge-traversal or FAISS similarity.
    
    Fix 3.5: Follows pre-computed cross-reference edges from FrozenState
    to retrieve structurally-connected chunks. No regex at query time.
    Falls back to FAISS if no cross-ref expansions are found.
    """
    doc_map = {doc.id: doc for doc in state.corpus.documents}
    existing_ids = {node.id for node in graph.nodes}
    new_nodes = []
    
    # Fix 3.5: Follow pre-computed cross-ref edges from FrozenState
    precomputed = getattr(state, 'precomputed_edges', None) or []
    
    # Build a lookup: for each existing node, find its cross-ref targets
    for node in graph.nodes:
        if len(new_nodes) >= top_k:
            break
        
        for edge in precomputed:
            if len(new_nodes) >= top_k:
                break
            
            # edge is (source_id, target_id, type)
            if isinstance(edge, tuple) and len(edge) >= 2:
                src, tgt = edge[0], edge[1]
            elif hasattr(edge, 'source'):
                src, tgt = edge.source, edge.target
            else:
                continue
            
            # If this node is the source, retrieve the target (or vice versa)
            target_id = None
            if src == node.id and tgt not in existing_ids:
                target_id = tgt
            elif tgt == node.id and src not in existing_ids:
                target_id = src
            
            if target_id and target_id in doc_map:
                doc = doc_map[target_id]
                new_node = Node(
                    id=target_id,
                    text=doc.text,
                    metadata={"source": "cross_ref_expand", "title": doc.title},
                    confidence=0.85,
                    embedding=state.corpus_embeddings.get(target_id),
                )
                new_nodes.append(new_node)
                existing_ids.add(target_id)
    
    # Also follow cross-ref edges already in the graph (from build_edges)
    if len(new_nodes) < top_k:
        for edge in graph.edges:
            if len(new_nodes) >= top_k:
                break
            if "cross_ref" not in edge.relation:
                continue
            
            # Find the target that's NOT in the graph yet
            for target_id in [edge.source, edge.target]:
                if target_id not in existing_ids and target_id in doc_map:
                    doc = doc_map[target_id]
                    new_node = Node(
                        id=target_id,
                        text=doc.text,
                        metadata={"source": "cross_ref_expand", "title": doc.title},
                        confidence=0.85,
                        embedding=state.corpus_embeddings.get(target_id),
                    )
                    new_nodes.append(new_node)
                    existing_ids.add(target_id)
                    if len(new_nodes) >= top_k:
                        break
    
    # Fallback: FAISS similarity search (original behavior)
    if len(new_nodes) < top_k and query:
        query_vec = state.embedder.embed(query)
        results = state.search_similar(query_vec, top_k=top_k - len(new_nodes),
                                        exclude_ids=existing_ids)
        for doc_id, score in results:
            if doc_id in existing_ids:
                continue
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                new_node = Node(
                    id=doc_id,
                    text=doc.text,
                    metadata={"source": "similarity_expand", "title": doc.title},
                    confidence=score * 0.9,
                    embedding=state.corpus_embeddings.get(doc_id),
                )
                new_nodes.append(new_node)
                existing_ids.add(doc_id)
    elif len(new_nodes) < top_k and not query:
        # Legacy fallback: no query, use first node text
        if graph.nodes:
            query_text = graph.nodes[0].text
            query_vec = state.embedder.embed(query_text)
            results = state.search_similar(query_vec, top_k=top_k - len(new_nodes),
                                            exclude_ids=existing_ids)
            for doc_id, score in results:
                if doc_id in existing_ids:
                    continue
                if doc_id in doc_map:
                    doc = doc_map[doc_id]
                    new_node = Node(
                        id=doc_id,
                        text=doc.text,
                        metadata={"source": "similarity_expand", "title": doc.title},
                        confidence=score,
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
