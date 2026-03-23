"""CSS Ingestion Gate for streaming data.

Evaluates incoming chunks using CSS features to decide whether
they should be added to the retrieval index.

Quality signals used (no query required):
    - redundancy: Is this a near-duplicate of existing indexed content?
    - answer_specificity: Does this chunk contain actionable information?
    - token_efficiency: Is this chunk too short/empty to be useful?

Optional query-aware signals (when recent queries available):
    - query_relevance: Does this improve CSS for recent queries?
    - coverage: Does this cover entities from recent queries?
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.types import Graph, Node


def _compute_redundancy(new_embedding: List[float], existing_embeddings: Dict[str, List[float]],
                        threshold: float = 0.92) -> float:
    """Check if the new chunk is a near-duplicate of anything already indexed.
    
    Args:
        new_embedding: Embedding of the candidate chunk
        existing_embeddings: Dict of doc_id → embedding for indexed chunks
        threshold: Cosine similarity above which chunks are considered redundant
    
    Returns:
        Max similarity to any existing chunk (0-1). High = redundant.
    """
    if not existing_embeddings:
        return 0.0
    
    from transforms.utils import cosine_similarity
    
    max_sim = 0.0
    for doc_id, existing_emb in existing_embeddings.items():
        sim = cosine_similarity(new_embedding, existing_emb)
        if sim > max_sim:
            max_sim = sim
    
    return max_sim


def _compute_specificity(text: str) -> float:
    """Score how much actionable/specific information is in the text.
    
    Looks for numbers, dates, money, time periods, named entities.
    Low specificity = generic/empty content ("Thanks!", "Any update?")
    
    Returns:
        Score [0, 1] where higher = more specific
    """
    if not text or len(text.strip()) < 10:
        return 0.0
    
    signals = 0
    
    # Numbers and percentages
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
    signals += min(len(numbers), 5)
    
    # Dates
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})\b', text)
    signals += min(len(dates), 3) * 2
    
    # Dollar amounts
    money = re.findall(r'\$[\d,]+(?:\.\d{2})?', text)
    signals += min(len(money), 3) * 2
    
    # Time periods
    periods = re.findall(r'\b\d+\s*(?:days?|months?|years?|weeks?|hours?)\b', text, re.IGNORECASE)
    signals += min(len(periods), 3) * 2
    
    # Word count bonus (longer text = likely more info)
    words = len(text.split())
    if words >= 20:
        signals += 2
    elif words >= 10:
        signals += 1
    
    # Normalize
    return min(1.0, signals / 8.0)


def _compute_token_efficiency(text: str, min_tokens: int = 5) -> float:
    """Score whether this chunk has enough tokens to be useful.
    
    Very short chunks (status updates, single words) waste retrieval slots.
    
    Returns:
        Score [0, 1] where higher = better token count
    """
    words = len(text.split())
    
    if words < min_tokens:
        return 0.0
    elif words < 15:
        return 0.3
    elif words < 30:
        return 0.6
    else:
        return 1.0


def evaluate_for_ingestion(
    new_text: str,
    new_embedding: List[float],
    streaming_state: Any,  # StreamingState (avoid circular import)
    recent_queries: List[str] = None,
    threshold: float = 0.3,
) -> Dict:
    """Evaluate whether a new chunk should be accepted into the index.
    
    Uses CSS features as a quality gate. No LLM calls required.
    
    Args:
        new_text: The candidate text to evaluate
        new_embedding: Pre-computed embedding of the text
        streaming_state: StreamingState with existing indexed data
        recent_queries: Recent user queries for relevance scoring
        threshold: Minimum quality score to accept (0-1)
    
    Returns:
        Dict with:
            accept: bool - whether to add this chunk
            score: float - composite quality score
            reason: str - explanation
            features: dict - individual feature scores
    """
    features = {}
    reasons = []
    
    # 1. Token efficiency - is this chunk long enough?
    token_eff = _compute_token_efficiency(new_text)
    features["token_efficiency"] = token_eff
    
    if token_eff == 0.0:
        return {
            "accept": False,
            "score": 0.0,
            "reason": "Too short (fewer than 5 words)",
            "features": features,
        }
    
    # 2. Specificity - does it contain actionable information?
    specificity = _compute_specificity(new_text)
    features["answer_specificity"] = specificity
    
    # 3. Redundancy - is it a near-duplicate?
    redundancy = _compute_redundancy(new_embedding, streaming_state.corpus_embeddings)
    features["redundancy"] = redundancy
    
    if redundancy > 0.95:
        return {
            "accept": False,
            "score": 0.0,
            "reason": f"Near-duplicate of existing content (similarity={redundancy:.3f})",
            "features": features,
        }
    
    # 4. Query relevance (optional - if recent queries available)
    query_relevance = 0.5  # neutral default
    if recent_queries and streaming_state.embedder:
        from transforms.utils import cosine_similarity
        
        max_rel = 0.0
        for query in recent_queries[-5:]:
            query_emb = streaming_state.embedder.embed(query)
            rel = cosine_similarity(new_embedding, query_emb)
            if rel > max_rel:
                max_rel = rel
        query_relevance = max_rel
    features["query_relevance"] = query_relevance
    
    # Composite quality score
    # Weights: specificity and non-redundancy matter most
    score = (
        0.20 * token_eff +
        0.25 * specificity +
        0.30 * (1.0 - redundancy) +  # Invert: low redundancy = good
        0.25 * query_relevance
    )
    features["composite_score"] = score
    
    # Decision
    accept = score >= threshold
    
    if not accept:
        if redundancy > 0.8:
            reasons.append(f"high redundancy ({redundancy:.2f})")
        if specificity < 0.2:
            reasons.append(f"low specificity ({specificity:.2f})")
        if token_eff < 0.3:
            reasons.append(f"low token efficiency")
        reason = f"Below threshold ({score:.3f} < {threshold}): " + ", ".join(reasons)
    else:
        reason = f"Accepted (score={score:.3f})"
    
    return {
        "accept": accept,
        "score": score,
        "reason": reason,
        "features": features,
    }
