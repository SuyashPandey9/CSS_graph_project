"""Legal domain features for CSS learning loop.

UPDATED (Batch 4):
  - Removed `legal_entity_density` (now covered by IDF-based entity overlap in edge_builder)
  - Removed `cross_reference_density` (now covered by cross-ref edges in edge_builder)
  - Kept: `clause_coverage`, `section_diversity`, `answer_specificity`

These are domain-specific addons used via config.yaml.
"""

from __future__ import annotations

from typing import Dict, List, Set

from core.types import Graph


# Legal entity groups for clause-type detection
CLAUSE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "termination": ["termination", "terminate", "expiration", "expire", "renewal"],
    "indemnification": ["indemnification", "indemnify", "hold harmless", "indemnities"],
    "liability": ["liability", "liable", "damages", "consequential", "limitation of liability"],
    "confidentiality": ["confidential", "confidentiality", "proprietary", "non-disclosure", "nda"],
    "ip_rights": ["intellectual property", "patent", "copyright", "trademark", "trade secret"],
    "payment": ["payment", "fee", "royalty", "compensation", "invoice", "remittance"],
    "governing_law": ["governing law", "jurisdiction", "venue", "applicable law"],
    "assignment": ["assignment", "assign", "sublicense", "transfer"],
    "non_compete": ["non-compete", "non compete", "exclusivity", "exclusive", "restrictive covenant"],
    "dispute_resolution": ["arbitration", "mediation", "dispute", "resolution", "litigation"],
    "warranty": ["warranty", "warranties", "warrant", "representation"],
    "force_majeure": ["force majeure", "act of god", "unforeseeable"],
    "insurance": ["insurance", "insure", "coverage", "policy"],
}


def legal_entity_density(graph: Graph, query: str) -> float:
    """Fraction of recognized legal entities found across all nodes.
    
    Higher values mean the context is rich in legal terminology,
    which is important for legal QA tasks.
    
    Args:
        graph: The context graph
        query: The user's query (unused, kept for consistent API)
    
    Returns:
        Score [0, 1] - fraction of all known legal entities present
    """
    if not graph.nodes:
        return 0.0
    
    all_text = " ".join(node.text.lower() for node in graph.nodes)
    
    # Flatten all keywords from all clause types
    all_entities: Set[str] = set()
    for keywords in CLAUSE_TYPE_KEYWORDS.values():
        all_entities.update(kw.lower() for kw in keywords)
    
    if not all_entities:
        return 0.0
    
    found = sum(1 for entity in all_entities if entity in all_text)
    
    # Normalize: having 20%+ of all entities is very dense legal text
    raw = found / len(all_entities)
    return min(1.0, raw * 3.0)  # Scale up since most chunks won't have >30% of all entities


def clause_coverage(graph: Graph, query: str) -> float:
    """Does the retrieved context cover the clause type requested by the query?
    
    Detects the target clause type from the query, then checks if the
    retrieved context contains relevant keywords for that clause type.
    
    Args:
        graph: The context graph
        query: The user's query
    
    Returns:
        Score [0, 1] - how well the context covers the target clause
    """
    if not graph.nodes or not query:
        return 0.0
    
    # Detect target clause type from query
    query_lower = query.lower()
    target_types: List[str] = []
    
    for clause_type, keywords in CLAUSE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                target_types.append(clause_type)
                break
    
    if not target_types:
        return 0.5  # Can't determine target clause → neutral
    
    # Check how many target clause keywords appear in context
    all_text = " ".join(node.text.lower() for node in graph.nodes)
    
    total_coverage = 0.0
    for clause_type in target_types:
        keywords = CLAUSE_TYPE_KEYWORDS[clause_type]
        found = sum(1 for kw in keywords if kw.lower() in all_text)
        total_coverage += found / len(keywords)
    
    return min(1.0, total_coverage / len(target_types))


def cross_reference_density(graph: Graph, query: str) -> float:
    """How many chunks reference the same legal terms?
    
    Measures the density of cross-references between nodes.
    High cross-reference density means chunks are discussing related
    legal concepts, improving multi-hop reasoning capability.
    
    Args:
        graph: The context graph
        query: The user's query (unused)
    
    Returns:
        Score [0, 1] - density of shared legal terms across chunks
    """
    if len(graph.nodes) < 2:
        return 0.5
    
    # Extract entity sets per node
    node_entity_sets: List[Set[str]] = []
    for node in graph.nodes:
        text_lower = node.text.lower()
        entities = set()
        for keywords in CLAUSE_TYPE_KEYWORDS.values():
            for kw in keywords:
                if kw in text_lower:
                    entities.add(kw)
        node_entity_sets.append(entities)
    
    # Count pairs that share at least one entity
    shared_pairs = 0
    total_pairs = 0
    
    for i in range(len(node_entity_sets)):
        for j in range(i + 1, len(node_entity_sets)):
            total_pairs += 1
            overlap = node_entity_sets[i] & node_entity_sets[j]
            if overlap:
                shared_pairs += 1
    
    if total_pairs == 0:
        return 0.5
    
    return shared_pairs / total_pairs


def section_diversity(graph: Graph, query: str) -> float:
    """Are chunks from different contract sections?
    
    Measures whether retrieved chunks come from diverse parts of
    the source documents. Higher diversity means broader coverage.
    
    Args:
        graph: The context graph
        query: The user's query (unused)
    
    Returns:
        Score [0, 1] - fraction of unique source sections
    """
    if len(graph.nodes) < 2:
        return 0.5
    
    # Extract source identifiers from node metadata or IDs
    sources: Set[str] = set()
    for node in graph.nodes:
        # Try metadata source
        source = node.metadata.get("source", "")
        if not source:
            # Try extracting from ID (e.g., "contract_name_chunk_003")
            if "_chunk_" in node.id:
                source = node.id.rsplit("_chunk_", 1)[0]
            else:
                source = node.id
        sources.add(source)
    
    # Ratio of unique sources to total nodes
    source_ratio = len(sources) / len(graph.nodes)
    return min(1.0, source_ratio)


def answer_specificity(graph: Graph, query: str) -> float:
    """Does the context enable a specific (non-generic) answer?
    
    Measures the presence of specific details like numbers, dates,
    percentages, and named parties — indicators that the context
    contains actionable information rather than generic text.
    
    Args:
        graph: The context graph
        query: The user's query (unused)
    
    Returns:
        Score [0, 1] - specificity of the context
    """
    import re
    
    if not graph.nodes:
        return 0.0
    
    all_text = " ".join(node.text for node in graph.nodes)
    
    specificity_signals = 0
    
    # Numbers and percentages
    numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', all_text)
    specificity_signals += min(len(numbers), 5)
    
    # Dates (various formats)
    dates = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+ \d{1,2},? \d{4})\b', all_text)
    specificity_signals += min(len(dates), 3) * 2
    
    # Dollar amounts
    money = re.findall(r'\$[\d,]+(?:\.\d{2})?', all_text)
    specificity_signals += min(len(money), 3) * 2
    
    # Time periods (e.g., "thirty (30) days", "12 months")
    periods = re.findall(r'\b\d+\s*(?:days?|months?|years?|weeks?)\b', all_text, re.IGNORECASE)
    specificity_signals += min(len(periods), 3) * 2
    
    # Normalize: 10+ signals is highly specific
    return min(1.0, specificity_signals / 10.0)


# Registry of legal domain features for CSS addons
# Batch 4: Removed legal_entity_density (→ IDF in edge_builder)
#          Removed cross_reference_density (→ cross_ref edges in edge_builder)
LEGAL_FEATURES = {
    "clause_coverage": clause_coverage,
    "section_diversity": section_diversity,
    "answer_specificity": answer_specificity,
}


def compute_all_legal_features(graph: Graph, query: str) -> Dict[str, float]:
    """Compute all legal domain features for a graph/query pair.
    
    Args:
        graph: The context graph
        query: The user's query
    
    Returns:
        Dict of {feature_name: value}
    """
    return {name: fn(graph, query) for name, fn in LEGAL_FEATURES.items()}
