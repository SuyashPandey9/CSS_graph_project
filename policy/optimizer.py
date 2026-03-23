"""Greedy optimization loop for V3.

UPDATED (Batch 2):
  Fix 3.6: Increased top_k from 6 → 10 for larger initial retrieval pool.
  Integrated query preprocessor for better entity-based retrieval.
  Uses IDF-aware edge building.

V3 Spec: Finite-horizon optimization π* = argmax_π E_π[Σ R(s_t, a_t)]
"""

from __future__ import annotations

from core.frozen_state import FrozenState, get_shared_state
from core.query_graph import build_query_graph
from core.types import Graph
from css.calculator import compute_css_final
from policy.greedy_policy import select_action
from transforms.expand import expand


# Minimum improvement to accept a new graph
IMPROVEMENT_THRESHOLD = 0.03

# If relevance is already this high, don't expand more
HIGH_RELEVANCE_THRESHOLD = 0.75

# Minimum CSS to trust the answer (below this, flag as low confidence)
MIN_CSS_THRESHOLD = 0.30

# Minimum steps before allowing early stopping
MIN_STEPS_BEFORE_EARLY_STOP = 3

# Fix 3.6: Increased from 6 → 10
DEFAULT_TOP_K = 10


def _build_initial_retrieval_graph(query: str, state: FrozenState, top_k: int = None) -> Graph:
    """Build G₀ using ENTITY-BASED retrieval (V3's key differentiator).
    
    Fix 3.6: Retrieves top_k=10 chunks (up from 6) using preprocessor
    entities when available for better multi-hop coverage.
    
    Args:
        query: User's query
        state: FrozenState with FAISS index
        top_k: Total number of chunks to retrieve (default: 10)
    
    Returns:
        Graph with document nodes from entity-based retrieval
    """
    if top_k is None:
        top_k = DEFAULT_TOP_K
    
    from core.types import Node
    from core.query_graph import get_query_entities
    
    doc_map = {doc.id: doc for doc in state.corpus.documents}
    seen_ids = set()
    nodes = []
    
    # Try to get entities from query preprocessor (Fix 3.4 + Fix C2)
    # Fix C2: Pass LLM client for intelligent subquery decomposition
    try:
        from tools.query_preprocessor import preprocess_query
        from llm.preprocessor_client import get_preprocessor_client
        llm_client = get_preprocessor_client()
        preprocessor_output = preprocess_query(query, llm_client=llm_client)
        entities = preprocessor_output.get("entities", [])
        domain_terms = preprocessor_output.get("domain_terms", [])
        # Combine entities + domain_terms for broader retrieval
        all_search_terms = entities + [t for t in domain_terms if t not in entities]
    except Exception:
        # Fallback to rule-based parser
        all_search_terms = get_query_entities(query, parser=state.parser)
        preprocessor_output = None
    
    # Step 1: Retrieve using full query (like Trad RAG)
    per_entity_k = max(2, top_k // (len(all_search_terms) + 1)) if all_search_terms else top_k
    
    query_vec = state.embedder.embed(query)
    query_results = state.search_similar(query_vec, top_k=per_entity_k, exclude_ids=seen_ids)
    for doc_id, score in query_results:
        if doc_id not in seen_ids and doc_id in doc_map:
            doc = doc_map[doc_id]
            node = Node(
                id=doc_id,
                text=doc.text,
                metadata={"source": "query_match", "title": doc.title},
                confidence=score,
                embedding=state.corpus_embeddings.get(doc_id),
            )
            nodes.append(node)
            seen_ids.add(doc_id)
    
    # Step 2: Retrieve for each entity/domain_term (multi-hop)
    if all_search_terms:
        for term in all_search_terms[:6]:
            term_vec = state.embedder.embed(term)
            term_results = state.search_similar(term_vec, top_k=per_entity_k, exclude_ids=seen_ids)
            for doc_id, score in term_results:
                if doc_id not in seen_ids and doc_id in doc_map:
                    doc = doc_map[doc_id]
                    node = Node(
                        id=doc_id,
                        text=doc.text,
                        metadata={"source": f"entity:{term}", "title": doc.title},
                        confidence=score * 0.9,
                        embedding=state.corpus_embeddings.get(doc_id),
                    )
                    nodes.append(node)
                    seen_ids.add(doc_id)
                    if len(nodes) >= top_k:
                        break
            if len(nodes) >= top_k:
                break
    
    # Fallback: no entities, just use query similarity
    if not nodes:
        query_vec = state.embedder.embed(query)
        results = state.search_similar(query_vec, top_k=top_k, exclude_ids=set())
        for doc_id, score in results:
            if doc_id in doc_map:
                doc = doc_map[doc_id]
                node = Node(
                    id=doc_id,
                    text=doc.text,
                    metadata={"source": "query_fallback", "title": doc.title},
                    confidence=score,
                    embedding=state.corpus_embeddings.get(doc_id),
                )
                nodes.append(node)
    
    # Build weighted edges between chunks
    from graph.edge_builder import build_edges
    initial_graph = Graph(nodes=nodes, edges=[])
    
    # Use IDF dict if available in state
    idf_dict = getattr(state, 'idf_dict', None)
    precomputed_edges = getattr(state, 'precomputed_edges', None)
    graph_with_edges = build_edges(
        initial_graph, embedder=state.embedder,
        idf_dict=idf_dict, precomputed_edges=precomputed_edges
    )
    
    return graph_with_edges


def optimize(
    query: str,
    user_graph: Graph,
    *,
    max_steps: int = 5,
    state: FrozenState | None = None,
    initial_graph: Graph | None = None,
) -> Graph:
    """Run the greedy optimization loop with early stopping.
    
    V3 Spec:
        - G₀ = initial_graph (from parsed query if not provided)
        - For t = 1..T: a_t, G_{t+1} = select_action(G_t)
        - Return G* when action = stop or t = T
    
    Args:
        query: User's query string
        user_graph: User context graph (for personalization)
        max_steps: Maximum optimization steps
        state: FrozenState (created if not provided)
        initial_graph: G₀ (built from query if not provided)
    """
    state = state or get_shared_state()
    
    if initial_graph is not None:
        current = Graph(nodes=list(initial_graph.nodes), edges=list(initial_graph.edges))
    else:
        current = _build_initial_retrieval_graph(query, state)
    
    # Track best graph seen
    best_graph = current
    best_css = 0.0
    
    # Get shared embedder to avoid reloading
    embedder = state.embedder
    
    for step in range(1, max_steps + 1):
        # Compute current CSS/relevance
        current_scores = compute_css_final(current, query, user_graph, embedder=embedder)
        current_css = current_scores["css_final"]
        current_relevance = current_scores.get("query_relevance", 0)
        
        if step == 1:
            best_css = current_css
            best_graph = current
            print(f"Step 0: initial retrieval, nodes={len(current.nodes)}, css={current_css:.3f}, relevance={current_relevance:.3f}")
        
        # Fix 3.6: Adjusted threshold for larger graphs
        skip_expand = len(current.nodes) >= 8  # Was 6, now 8 for larger initial pool
        action, next_graph = select_action(
            current, state, query, user_graph, 
            skip_expand=skip_expand,
            embedder=embedder
        )
        
        # Compute CSS for new graph
        scores = compute_css_final(next_graph, query, user_graph, embedder=embedder)
        next_css = scores["css_final"]
        
        # Logging
        print(
            f"Step {step}: action={action}, nodes={len(next_graph.nodes)}, "
            f"css_final={next_css:.3f}, relevance={scores.get('query_relevance', 0):.3f}"
        )
        
        # Check if improvement is meaningful
        improvement = next_css - current_css
        
        if next_css > best_css:
            best_graph = next_graph
            best_css = next_css
        
        # Early stopping
        if step > MIN_STEPS_BEFORE_EARLY_STOP and improvement < IMPROVEMENT_THRESHOLD:
            print(f"Early stopping: improvement {improvement:.4f} < threshold {IMPROVEMENT_THRESHOLD}")
            break
        
        current = next_graph
        
        if action == "stop":
            break
    
    # Flag low-confidence results
    if best_css < MIN_CSS_THRESHOLD:
        print(f"[V3 Warning] Low CSS ({best_css:.3f} < {MIN_CSS_THRESHOLD}) - context may be insufficient")
        for node in best_graph.nodes:
            node.metadata["low_confidence"] = True
    
    return best_graph
