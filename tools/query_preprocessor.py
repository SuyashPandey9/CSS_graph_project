"""LLM-based query preprocessor for V3 (Fix 3.4).

Decomposes a user query into structured components using a single LLM call:
  - entities: Key concepts
  - domain_terms: Multi-word domain-specific phrases
  - subqueries: Decomposed questions
  - intent: Query type classification
  - clause_types: Relevant clause categories (for legal domain)

Results are cached per unique query string for determinism.
Uses temperature=0 + structured JSON output for maximum consistency.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


# Cache: normalized query → preprocessor output
_CACHE: Dict[str, dict] = {}

# System prompt for the LLM preprocessor
PREPROCESSOR_PROMPT = """You are a query preprocessor for a legal document retrieval system.

Given a user query, output EXACTLY this JSON structure:
{
  "entities": ["list of 3-6 key legal concepts, most specific first"],
  "domain_terms": ["list of 2-4 multi-word legal phrases as they appear in contracts"],
  "subqueries": ["list of 3-5 decomposed questions that together answer the query"],
  "intent": "one of: factual, multi_hop_factual, comparison, yes_no, definition",
  "clause_types": ["list from: termination, indemnification, liability, confidentiality, ip_rights, non_compete, warranty, force_majeure, insurance, governing_law, assignment, change_of_control, survival, payment, representations"]
}

Rules:
- entities must be legal concepts, not common words
- subqueries must preserve relationships between concepts
- domain_terms should be multi-word phrases as they appear in contracts
- Output ONLY valid JSON. No explanation, no commentary."""


def _normalize_query(query: str) -> str:
    """Normalize query for cache key."""
    return query.strip().lower()


def _parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response, handling potential formatting issues."""
    # Try direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try extracting JSON from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object in text
    match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def _validate_output(output: dict) -> dict:
    """Ensure output has all required fields with correct types."""
    defaults = {
        "entities": [],
        "domain_terms": [],
        "subqueries": [],
        "intent": "factual",
        "clause_types": [],
    }
    
    result = {}
    for key, default in defaults.items():
        val = output.get(key, default)
        if isinstance(default, list) and not isinstance(val, list):
            val = [val] if val else default
        elif isinstance(default, str) and not isinstance(val, str):
            val = str(val) if val else default
        result[key] = val
    
    return result


def _fallback_preprocess(query: str) -> dict:
    """Rule-based fallback when LLM is unavailable.
    
    Uses the parser_stub to extract basic entities and generates
    simple subqueries. Better than nothing, but misses relationships.
    """
    from tools.parser_stub import ParserStub
    parser = ParserStub()
    parsed = parser.extract(query)
    
    entities = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    subqueries = parsed.get("subqueries", [])
    
    # Infer intent from query structure
    query_lower = query.lower()
    if query_lower.startswith(("does ", "is ", "are ", "can ", "will ")):
        intent = "yes_no"
    elif " and " in query_lower or " or " in query_lower:
        intent = "multi_hop_factual"
    elif "what is" in query_lower or "define" in query_lower:
        intent = "definition"
    elif "compare" in query_lower or "difference" in query_lower:
        intent = "comparison"
    else:
        intent = "factual"
    
    # Infer clause types from entities
    clause_mapping = {
        "termination": ["termination", "terminate", "expiration"],
        "indemnification": ["indemnification", "indemnify", "indemnity"],
        "liability": ["liability", "damages", "cap"],
        "confidentiality": ["confidential", "confidentiality", "secret"],
        "ip_rights": ["intellectual property", "patent", "copyright", "trademark"],
        "non_compete": ["non-compete", "non-solicitation", "restrictive"],
        "warranty": ["warranty", "warranties", "representation"],
        "survival": ["survival", "survive", "surviving"],
        "payment": ["payment", "compensation", "fee", "royalty"],
    }
    
    clause_types = []
    for clause, keywords in clause_mapping.items():
        if any(kw in query_lower for kw in keywords):
            clause_types.append(clause)
    
    return {
        "entities": entities[:6],
        "domain_terms": [e for e in entities if " " in e][:4],
        "subqueries": subqueries[:5],
        "intent": intent,
        "clause_types": clause_types,
    }


def preprocess_query(query: str, llm_client=None) -> dict:
    """Preprocess a query using LLM (or fallback to rule-based).
    
    Results are cached per unique query string for determinism.
    
    Args:
        query: The raw user query
        llm_client: Optional LLM client with a .call(prompt, system) method.
                    If None, uses rule-based fallback.
    
    Returns:
        Dict with keys: entities, domain_terms, subqueries, intent, clause_types
    """
    cache_key = _normalize_query(query)
    
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    if llm_client is not None:
        try:
            response = llm_client.call(
                prompt=query,
                system=PREPROCESSOR_PROMPT,
                temperature=0,
            )
            
            parsed = _parse_llm_response(response)
            if parsed:
                result = _validate_output(parsed)
                _CACHE[cache_key] = result
                return result
        except Exception as e:
            print(f"[Preprocessor] LLM call failed, using fallback: {e}")
    
    # Fallback to rule-based preprocessing
    result = _fallback_preprocess(query)
    _CACHE[cache_key] = result
    return result


def clear_cache():
    """Clear the preprocessor cache (for testing)."""
    _CACHE.clear()
