"""CUAD Evaluation Dataset for CSS Learning Loop.

Provides labeled query/answer pairs across CUAD's legal clause categories.
Used as ground truth for measuring answer quality during weight optimization.

Categories covered (from CUAD's 41 clause types):
    Termination, Indemnification, Liability, Confidentiality, IP Rights,
    Payment, Governing Law, Assignment, Non-Compete, Dispute Resolution
"""

from __future__ import annotations

from typing import Dict, List


# Evaluation queries grouped by CUAD clause category.
# Each entry provides a query, expected entities in relevant context,
# difficulty level, and answer keywords to check for.
CUAD_EVAL_QUERIES: List[Dict] = [
    # --- Termination ---
    {
        "query": "What are the termination conditions in this contract?",
        "category": "termination",
        "expected_entities": ["termination", "terminate", "breach", "expiration"],
        "answer_keywords": ["terminate", "breach", "notice", "days"],
        "difficulty": "single_hop",
    },
    {
        "query": "Under what circumstances can either party terminate the agreement early?",
        "category": "termination",
        "expected_entities": ["termination", "terminate", "breach"],
        "answer_keywords": ["terminate", "cause", "breach", "notice"],
        "difficulty": "multi_hop",
    },
    # --- Indemnification ---
    {
        "query": "What are the indemnification obligations of each party?",
        "category": "indemnification",
        "expected_entities": ["indemnification", "indemnify", "liability", "damages"],
        "answer_keywords": ["indemnify", "hold harmless", "claims", "damages"],
        "difficulty": "single_hop",
    },
    # --- Liability ---
    {
        "query": "Are there any limitations on liability in this agreement?",
        "category": "liability",
        "expected_entities": ["liability", "damages", "indemnification"],
        "answer_keywords": ["liability", "limit", "damages", "consequential"],
        "difficulty": "single_hop",
    },
    # --- Confidentiality ---
    {
        "query": "What information is considered confidential under this contract?",
        "category": "confidentiality",
        "expected_entities": ["confidential", "confidentiality", "proprietary"],
        "answer_keywords": ["confidential", "proprietary", "disclose", "information"],
        "difficulty": "single_hop",
    },
    {
        "query": "How long do the confidentiality obligations last after the agreement ends?",
        "category": "confidentiality",
        "expected_entities": ["confidential", "confidentiality", "termination", "expiration"],
        "answer_keywords": ["confidential", "years", "survive", "termination"],
        "difficulty": "multi_hop",
    },
    # --- IP Rights ---
    {
        "query": "Who owns the intellectual property created during the contract?",
        "category": "ip_rights",
        "expected_entities": ["intellectual property", "license", "rights", "proprietary"],
        "answer_keywords": ["intellectual property", "ownership", "rights", "license"],
        "difficulty": "single_hop",
    },
    # --- Payment ---
    {
        "query": "What are the payment terms and schedule in this agreement?",
        "category": "payment",
        "expected_entities": ["payment", "fee", "compensation", "royalty"],
        "answer_keywords": ["payment", "fee", "days", "invoice"],
        "difficulty": "single_hop",
    },
    # --- Governing Law ---
    {
        "query": "Which jurisdiction's laws govern this contract?",
        "category": "governing_law",
        "expected_entities": ["governing law", "jurisdiction"],
        "answer_keywords": ["governing law", "state", "jurisdiction", "laws"],
        "difficulty": "single_hop",
    },
    # --- Assignment ---
    {
        "query": "Can this agreement be assigned to a third party?",
        "category": "assignment",
        "expected_entities": ["assignment", "sublicense"],
        "answer_keywords": ["assign", "consent", "transfer", "third party"],
        "difficulty": "single_hop",
    },
    # --- Non-Compete ---
    {
        "query": "Does this contract include any non-compete or exclusivity clauses?",
        "category": "non_compete",
        "expected_entities": ["non-compete", "exclusivity"],
        "answer_keywords": ["non-compete", "exclusive", "compete", "restrict"],
        "difficulty": "single_hop",
    },
    # --- Dispute Resolution ---
    {
        "query": "How are disputes resolved under this agreement?",
        "category": "dispute_resolution",
        "expected_entities": ["arbitration", "dispute", "resolution"],
        "answer_keywords": ["arbitration", "dispute", "mediation", "court"],
        "difficulty": "single_hop",
    },
    # --- Multi-hop queries ---
    {
        "query": "If a party breaches confidentiality, what are the remedies and can the other party terminate?",
        "category": "multi_hop_breach",
        "expected_entities": ["confidential", "breach", "termination", "damages"],
        "answer_keywords": ["breach", "confidential", "terminate", "remedy", "damages"],
        "difficulty": "multi_hop",
    },
    {
        "query": "What happens to intellectual property rights after the contract is terminated?",
        "category": "multi_hop_ip_term",
        "expected_entities": ["intellectual property", "rights", "termination"],
        "answer_keywords": ["intellectual property", "terminate", "survive", "rights"],
        "difficulty": "multi_hop",
    },
    {
        "query": "Does the indemnification obligation survive contract termination and what are the liability caps?",
        "category": "multi_hop_indem_term",
        "expected_entities": ["indemnification", "termination", "liability"],
        "answer_keywords": ["indemnify", "survive", "termination", "liability", "limit"],
        "difficulty": "multi_hop",
    },
]


def get_eval_queries(difficulty: str = None, category: str = None) -> List[Dict]:
    """Return filtered evaluation queries.
    
    Args:
        difficulty: Filter by "single_hop" or "multi_hop" (None = all)
        category: Filter by CUAD category (None = all)
    
    Returns:
        Filtered list of evaluation query dicts
    """
    queries = CUAD_EVAL_QUERIES
    
    if difficulty:
        queries = [q for q in queries if q["difficulty"] == difficulty]
    
    if category:
        queries = [q for q in queries if q["category"] == category]
    
    return queries


def get_all_categories() -> List[str]:
    """Return all unique categories."""
    return sorted(set(q["category"] for q in CUAD_EVAL_QUERIES))


def compute_answer_keyword_coverage(answer: str, query_entry: Dict) -> float:
    """Check what fraction of expected answer keywords appear in the answer.
    
    This is a cheap, API-free proxy for answer quality.
    
    Args:
        answer: Generated answer text
        query_entry: A dict from CUAD_EVAL_QUERIES
    
    Returns:
        Fraction of expected keywords found [0, 1]
    """
    if not answer:
        return 0.0
    
    answer_lower = answer.lower()
    keywords = query_entry.get("answer_keywords", [])
    
    if not keywords:
        return 1.0
    
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords)


def compute_entity_retrieval_coverage(retrieved_text: str, query_entry: Dict) -> float:
    """Check what fraction of expected entities appear in retrieved context.
    
    This measures retrieval quality without API calls.
    
    Args:
        retrieved_text: The concatenated retrieved context 
        query_entry: A dict from CUAD_EVAL_QUERIES
    
    Returns:
        Fraction of expected entities found [0, 1]
    """
    if not retrieved_text:
        return 0.0
    
    text_lower = retrieved_text.lower()
    entities = query_entry.get("expected_entities", [])
    
    if not entities:
        return 1.0
    
    found = sum(1 for ent in entities if ent.lower() in text_lower)
    return found / len(entities)
