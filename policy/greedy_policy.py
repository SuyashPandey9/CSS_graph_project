"""Greedy policy for selecting graph transformation actions.

UPDATED: Supports skip_expand for high-relevance contexts, shared embedder.

V3 Spec: Select action a_t that maximizes CSS_final.
"""

from __future__ import annotations

from typing import Callable, Optional

from core.frozen_state import FrozenState
from core.types import Graph
from css.calculator import compute_css_final
from transforms.compress import compress
from transforms.expand import expand
from transforms.prune import prune
from transforms.stop import stop


ActionFn = Callable[[Graph, FrozenState], Graph]


def select_action(
    graph: Graph, 
    state: FrozenState, 
    query: str, 
    user_graph: Graph,
    *,
    skip_expand: bool = False,
    embedder=None,
) -> tuple[str, Graph]:
    """Select the action that maximizes CSS_final.
    
    V3 Spec:
        a_t = argmax_a CSS_final(T_op(G_t, a), x, U)
    
    Args:
        graph: Current graph
        state: FrozenState with tools
        query: User's query
        user_graph: User context graph
        skip_expand: If True, don't consider expand action (for high-relevance contexts)
        embedder: Shared embedder to avoid reloading
    """
    # Build candidate graphs
    candidates: list[tuple[str, Graph]] = []
    
    # Only add expand if not skipped
    if not skip_expand:
        candidates.append(("expand", expand(graph, state, top_k=2, query=query)))
    
    # Always consider prune, compress, stop
    candidates.extend([
        ("prune", prune(graph, state, min_nodes=3, query=query)),  # Pass query for relevance
        ("compress", compress(graph, threshold=0.6)),  # FIXED: Increased from 0.4 to prevent over-merging
        ("stop", stop(graph, state)),
    ])

    # Find best action
    best_action = "stop"
    best_graph = graph
    best_score = -1.0
    
    for name, candidate in candidates:
        scores = compute_css_final(
            candidate, query=query, user_graph=user_graph, 
            embedder=embedder
        )
        css_final = scores["css_final"]
        
        if css_final > best_score:
            best_score = css_final
            best_action = name
            best_graph = candidate

    return best_action, best_graph
