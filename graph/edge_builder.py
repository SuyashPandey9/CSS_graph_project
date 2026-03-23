"""Edge builder for V3 graph relationships.

UPDATED (Batch 2): Major overhaul.
  Fix 3.3 + 2.10: Cross-reference and defined-term edge detection (unified)
  Fix 1c.1: IDF-based entity overlap (replaces hardcoded LEGAL_ENTITIES)
  Fix 1c.2: Jaccard/lexical overlap removed (weight → 0)
  Fix 1c.3: Chunk adjacency factor added
  Fix 1c.5: Semantic similarity embedding fallback (compute on-the-fly)

Edge factors after fixes:
  cross_reference > entity_overlap (IDF) > semantic > chunk_adjacency > same_source
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from core.types import Edge, Graph, Node


# Fix 3.3 + 2.10: Cross-reference patterns (unified indexing pass)
# Fix A1: Added override/exception trigger words (notwithstanding, except as provided, etc.)
CROSS_REF_PATTERNS = [
    re.compile(r'(?:subject to|pursuant to|as (?:defined|described|set forth) in|see|under|per|'
               r'notwithstanding|except as (?:provided|set forth|described) in|'
               r'without limiting|provided,? however,? (?:that )?(?:pursuant to|under)|'
               r'in accordance with|as set forth in|as specified in|'
               r'in addition to)\s+'
               r'(?:Section|Article|Exhibit|Schedule|Appendix)\s+[\dIVXLCA-Z]+(?:\.\d+)*',
               re.IGNORECASE),
    re.compile(r'Section\s+\d+(?:\.\d+)*', re.IGNORECASE),
    re.compile(r'Article\s+[IVXLC]+', re.IGNORECASE),
    re.compile(r'Exhibit\s+[A-Z]', re.IGNORECASE),
    re.compile(r'Schedule\s+\d+', re.IGNORECASE),
    re.compile(r'Appendix\s+[A-Z]', re.IGNORECASE),
    re.compile(r'\bherein\b|\bhereinafter\b|\bhereunder\b', re.IGNORECASE),
]

# Fix 2.10: Defined-term patterns
DEFINED_TERM_PATTERN = re.compile(
    r'"([^"]{3,50})"\s+(?:means|shall mean|refers to|is defined as)',
    re.IGNORECASE
)

# Default edge weight parameters (updated for new factors)
DEFAULT_EDGE_PARAMS = {
    "cross_ref_weight": 0.4,           # Fix 3.3: Highest weight — author-intent connection
    "entity_overlap_weight": 0.35,     # Fix 1c.1: IDF-based entity overlap
    "entity_overlap_max_count": 3,     # number of shared entities for max score
    "same_source_weight": 0.2,         # Reduced (adjacency is more precise)
    "semantic_weight": 0.3,            # Embedding similarity
    "semantic_threshold": 0.5,         # min cosine sim to count
    "adjacency_weight": 0.25,          # Fix 1c.3: Chunk adjacency bonus
    # "lexical_weight": 0.0,           # Fix 1c.2: REMOVED (semantic is superior)
    "edge_threshold": 0.15,            # min total weight to create any edge
}


# ──────────────────────────────────────────────
# Fix 1c.1: IDF-based entity detection
# ──────────────────────────────────────────────

def _extract_candidate_entities(text: str) -> Set[str]:
    """Extract candidate entity terms from text.
    
    Uses simple heuristics: multi-word capitalized phrases and
    non-stopword terms longer than 3 characters.
    """
    # Find capitalized multi-word phrases (e.g., "Acme Corporation", "Effective Date")
    cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    
    # Also extract individual significant words (lowercased, > 3 chars, not common)
    _COMMON = {"the", "and", "for", "that", "this", "with", "from", "have", "been",
               "shall", "will", "would", "could", "should", "such", "each", "upon",
               "into", "under", "between", "through", "during", "before", "after",
               "other", "more", "most", "some", "only", "also", "than", "very"}
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    significant = {w for w in words if w not in _COMMON}
    
    # Combine: lowercased phrases + individual words
    entities = {phrase.lower() for phrase in cap_phrases}
    entities.update(significant)
    return entities


def compute_corpus_idf(chunks: list) -> Dict[str, float]:
    """Compute IDF (Inverse Document Frequency) for all terms across the corpus.
    
    Args:
        chunks: List of objects with .text attribute (CorpusDocument or TextChunk)
    
    Returns:
        Dict mapping term → IDF score. Higher = more specific/rare.
    """
    n_docs = len(chunks)
    if n_docs == 0:
        return {}
    
    # Count document frequency for each term
    df: Counter = Counter()
    for chunk in chunks:
        chunk_entities = _extract_candidate_entities(chunk.text)
        for entity in chunk_entities:
            df[entity] += 1
    
    # Compute IDF: log(N / df)
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log(n_docs / freq) if freq > 0 else 0.0
    
    return idf


def extract_entities_idf(text: str, idf_dict: Dict[str, float],
                          min_idf: float = 1.0) -> Set[str]:
    """Extract entities from text, filtered by IDF score.
    
    Only returns terms with IDF >= min_idf (i.e., somewhat rare in the corpus).
    
    Args:
        text: Text to extract entities from
        idf_dict: Pre-computed IDF dictionary from compute_corpus_idf()
        min_idf: Minimum IDF score to consider a term an entity
    
    Returns:
        Set of entity terms that are "specific" (high IDF)
    """
    candidates = _extract_candidate_entities(text)
    return {term for term in candidates if idf_dict.get(term, 0.0) >= min_idf}


# Backward compatibility: old function name, now uses IDF if available
def extract_entities_from_text(text: str, idf_dict: Optional[Dict[str, float]] = None) -> Set[str]:
    """Extract entities from text.
    
    If idf_dict is provided, uses IDF-based filtering (Fix 1c.1).
    Otherwise falls back to extracting all candidate entities.
    """
    if idf_dict:
        return extract_entities_idf(text, idf_dict)
    return _extract_candidate_entities(text)


# ──────────────────────────────────────────────
# Fix 3.3: Cross-reference detection
# ──────────────────────────────────────────────

def _extract_section_refs(text: str) -> Set[str]:
    """Extract section/article/exhibit references from text."""
    refs = set()
    for pattern in CROSS_REF_PATTERNS:
        for match in pattern.finditer(text):
            # Normalize: extract just the section identifier
            ref_text = match.group(0).strip()
            refs.add(ref_text.lower())
    return refs


def _extract_defined_terms(text: str) -> Set[str]:
    """Extract defined terms ("X" means ...) from text."""
    terms = set()
    for match in DEFINED_TERM_PATTERN.finditer(text):
        terms.add(match.group(1).lower())
    return terms


def _has_cross_reference(node_a: Node, node_b: Node, 
                         precomputed_pairs: Optional[Set[tuple]] = None) -> bool:
    """Check if either node references the other's section.
    
    If precomputed_pairs is provided (from FrozenState), uses O(1) lookup.
    Otherwise falls back to runtime regex (slower).
    """
    # Fast path: use pre-computed lookup
    if precomputed_pairs is not None:
        return (node_a.id, node_b.id) in precomputed_pairs or \
               (node_b.id, node_a.id) in precomputed_pairs
    
    # Slow path: runtime regex (fallback)
    refs_a = _extract_section_refs(node_a.text)
    refs_b = _extract_section_refs(node_b.text)
    
    section_id_b = node_b.metadata.get("section_id", "")
    section_id_a = node_a.metadata.get("section_id", "")
    
    if section_id_b:
        for ref in refs_a:
            if section_id_b in ref:
                return True
    
    if section_id_a:
        for ref in refs_b:
            if section_id_a in ref:
                return True
    
    # Defined term links
    defined_a = _extract_defined_terms(node_a.text)
    defined_b = _extract_defined_terms(node_b.text)
    
    text_b_lower = node_b.text.lower()
    text_a_lower = node_a.text.lower()
    
    for term in defined_a:
        if term in text_b_lower:
            return True
    for term in defined_b:
        if term in text_a_lower:
            return True
    
    return False


# ──────────────────────────────────────────────
# Fix 1c.3: Chunk adjacency detection
# ──────────────────────────────────────────────

def _get_chunk_index(node: Node) -> Optional[int]:
    """Extract chunk index from node ID (e.g., 'doc_chunk_003' → 3)."""
    match = re.search(r'chunk_(\d+)', node.id)
    if match:
        return int(match.group(1))
    return None


def _get_base_doc_id(node: Node) -> str:
    """Extract base document ID from node (before _chunk_ suffix)."""
    if "_chunk_" in node.id:
        return node.id.rsplit("_chunk_", 1)[0]
    return node.metadata.get("source", node.id)


def _are_adjacent_chunks(node_a: Node, node_b: Node) -> bool:
    """Check if two nodes are adjacent chunks from the same document.
    
    Fix 2.8: Section-aware adjacency.
    Returns 'same_section' or 'cross_section' or False.
    """
    base_a = _get_base_doc_id(node_a)
    base_b = _get_base_doc_id(node_b)
    
    if base_a != base_b:
        return False
    
    idx_a = _get_chunk_index(node_a)
    idx_b = _get_chunk_index(node_b)
    
    if idx_a is not None and idx_b is not None:
        return abs(idx_a - idx_b) == 1
    
    return False


def _same_section(node_a: Node, node_b: Node) -> bool:
    """Check if two adjacent nodes are in the same section (Fix 2.8)."""
    sec_a = node_a.metadata.get("section_id") or node_a.metadata.get("section_heading")
    sec_b = node_b.metadata.get("section_id") or node_b.metadata.get("section_heading")
    if sec_a and sec_b:
        return sec_a == sec_b
    return True  # If no section info, assume same section


# ──────────────────────────────────────────────
# Main edge computation
# ──────────────────────────────────────────────

def compute_edge_weight(
    node_a: Node,
    node_b: Node,
    embedder=None,
    edge_params: Optional[Dict[str, float]] = None,
    idf_dict: Optional[Dict[str, float]] = None,
    precomputed_pairs: Optional[Set[tuple]] = None,
) -> Tuple[float, str]:
    """Compute weighted edge score between two nodes.
    
    Args:
        node_a, node_b: Nodes to compare
        embedder: Embedder for semantic similarity (+ Fix 1c.5 fallback)
        edge_params: Override DEFAULT_EDGE_PARAMS
        idf_dict: Pre-computed IDF dictionary for entity detection
        precomputed_pairs: Set of (src_id, tgt_id) tuples for O(1) cross-ref lookup
    
    Returns:
        Tuple of (weight, edge_type_string)
    """
    params = edge_params if edge_params is not None else DEFAULT_EDGE_PARAMS
    weight = 0.0
    edge_types = []
    
    # 1. Cross-reference (Fix 3.3) — highest priority, uses O(1) lookup when available
    cross_ref_weight = params.get("cross_ref_weight", 0.4)
    if _has_cross_reference(node_a, node_b, precomputed_pairs=precomputed_pairs):
        weight += cross_ref_weight
        edge_types.append("cross_ref")
    
    # 2. Entity Overlap (Fix 1c.1 — IDF-based)
    entities_a = extract_entities_from_text(node_a.text, idf_dict)
    entities_b = extract_entities_from_text(node_b.text, idf_dict)
    overlap = entities_a & entities_b
    
    if overlap:
        max_count = params.get("entity_overlap_max_count", 3)
        entity_weight = params.get("entity_overlap_weight", 0.35)
        
        if idf_dict:
            # Weight by IDF: sum of IDF scores for shared entities, normalized
            idf_sum = sum(idf_dict.get(e, 1.0) for e in overlap)
            max_idf = max(idf_dict.get(e, 1.0) for e in overlap) * max_count
            entity_score = min(idf_sum / max(max_idf, 1.0), 1.0) * entity_weight
        else:
            # Fallback: simple count-based
            entity_score = min(len(overlap) / max_count, 1.0) * entity_weight
        
        weight += entity_score
        edge_types.append(f"entity_overlap:{len(overlap)}")
    
    # 3. Same Source Document
    base_a = _get_base_doc_id(node_a)
    base_b = _get_base_doc_id(node_b)
    
    if base_a and base_b and base_a == base_b:
        weight += params.get("same_source_weight", 0.2)
        edge_types.append("same_source")
    
    # 4. Semantic Similarity (Fix 1c.5 — embedding fallback)
    sem_threshold = params.get("semantic_threshold", 0.5)
    sem_weight = params.get("semantic_weight", 0.3)
    
    if embedder:
        # Fix 1c.5: Compute embeddings on-the-fly if missing
        emb_a = node_a.embedding
        emb_b = node_b.embedding
        
        if not emb_a:
            emb_a = embedder.embed(node_a.text)
        if not emb_b:
            emb_b = embedder.embed(node_b.text)
        
        if emb_a and emb_b:
            from transforms.utils import cosine_similarity
            sim = cosine_similarity(emb_a, emb_b)
            if sim > sem_threshold:
                sem_score = (sim - sem_threshold) / (1.0 - sem_threshold) * sem_weight
                weight += max(0, sem_score)
                edge_types.append(f"semantic:{sim:.2f}")
    
    # 5. Chunk Adjacency (Fix 1c.3 + Fix 2.8: section-aware)
    adjacency_weight = params.get("adjacency_weight", 0.25)
    if _are_adjacent_chunks(node_a, node_b):
        if _same_section(node_a, node_b):
            weight += adjacency_weight  # Full bonus within same section
        else:
            weight += adjacency_weight * 0.5  # Reduced bonus across sections (Fix 2.8)
        edge_types.append("adjacent")
    
    # Fix 1c.2: Jaccard/lexical overlap REMOVED (semantic is superior)
    
    edge_type = "|".join(edge_types) if edge_types else "none"
    return weight, edge_type


def build_edges(graph: Graph, embedder=None, threshold: float = 0.15,
                edge_params: Optional[Dict[str, float]] = None,
                idf_dict: Optional[Dict[str, float]] = None,
                precomputed_edges: Optional[list] = None) -> Graph:
    """Build weighted edges between nodes in the graph.
    
    Args:
        graph: Graph with nodes
        embedder: Optional embedder for semantic similarity
        threshold: Minimum weight to create an edge
        edge_params: Override DEFAULT_EDGE_PARAMS
        idf_dict: Pre-computed IDF dictionary for entity detection
        precomputed_edges: Pre-computed cross-ref edges from FrozenState
    
    Returns:
        Graph with edges added
    """
    params = edge_params if edge_params is not None else DEFAULT_EDGE_PARAMS
    effective_threshold = params.get("edge_threshold", threshold)
    
    nodes = graph.nodes
    edges = list(graph.edges)  # Keep existing edges
    
    # Build pre-computed cross-ref pair set for O(1) lookup
    precomputed_pairs = None
    if precomputed_edges:
        precomputed_pairs = set()
        for edge in precomputed_edges:
            if hasattr(edge, 'source'):
                precomputed_pairs.add((edge.source, edge.target))
            elif isinstance(edge, tuple) and len(edge) >= 2:
                precomputed_pairs.add((edge[0], edge[1]))
    
    # Compare all pairs
    edge_count = 0
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i+1:]:
            weight, edge_type = compute_edge_weight(
                node_a, node_b, embedder, edge_params=params, idf_dict=idf_dict,
                precomputed_pairs=precomputed_pairs
            )
            
            if weight >= effective_threshold:
                edge = Edge(
                    source=node_a.id,
                    target=node_b.id,
                    relation=edge_type,
                    weight=min(weight, 1.0),  # Clamp to [0, 1] for Edge model
                )
                edges.append(edge)
                edge_count += 1
    
    return Graph(nodes=nodes, edges=edges)


def add_node_with_edges(
    graph: Graph,
    new_node: Node,
    embedder=None,
    edge_params: Optional[Dict[str, float]] = None,
    idf_dict: Optional[Dict[str, float]] = None,
) -> Graph:
    """Add a single node to the graph and compute only its edges.
    
    O(n) where n = existing nodes, instead of O(n²) for rebuilding all edges.
    """
    params = edge_params if edge_params is not None else DEFAULT_EDGE_PARAMS
    threshold = params.get("edge_threshold", 0.15)
    
    if any(n.id == new_node.id for n in graph.nodes):
        return graph
    
    new_nodes = list(graph.nodes) + [new_node]
    new_edges = list(graph.edges)
    
    for existing_node in graph.nodes:
        weight, edge_type = compute_edge_weight(
            new_node, existing_node, embedder, edge_params=params, idf_dict=idf_dict
        )
        if weight >= threshold:
            new_edges.append(Edge(
                source=new_node.id,
                target=existing_node.id,
                relation=edge_type,
                weight=weight,
            ))
    
    return Graph(nodes=new_nodes, edges=new_edges)


def get_connected_nodes(graph: Graph, node_id: str, min_weight: float = 0.3) -> List[str]:
    """Get nodes connected to a given node by edges above threshold."""
    connected = []
    for edge in graph.edges:
        if edge.weight >= min_weight:
            if edge.source == node_id:
                connected.append(edge.target)
            elif edge.target == node_id:
                connected.append(edge.source)
    return connected


def get_node_degree(graph: Graph, node_id: str) -> int:
    """Get the number of edges connected to a node."""
    degree = 0
    for edge in graph.edges:
        if edge.source == node_id or edge.target == node_id:
            degree += 1
    return degree


def has_cross_ref_edges(graph: Graph, node_id: str) -> bool:
    """Check if a node has any cross-reference edges."""
    for edge in graph.edges:
        if (edge.source == node_id or edge.target == node_id) and "cross_ref" in edge.relation:
            return True
    return False


def graph_connectivity_score(graph: Graph) -> float:
    """Compute how well-connected the graph is (0-1).
    
    Quality-aware: uses sum of edge weights, not just count.
    """
    if len(graph.nodes) <= 1:
        return 1.0
    
    max_edges = len(graph.nodes) * (len(graph.nodes) - 1) / 2
    if max_edges == 0:
        return 1.0
    
    # Quality-aware: sum of edge weights (not count)
    total_weight = sum(edge.weight for edge in graph.edges)
    connectivity = min(1.0, total_weight / max_edges)
    
    return connectivity
def precompute_cross_ref_edges(documents: list):
    """Pre-compute cross-reference and defined-term edges at indexing time.
    
    Called by FrozenState.build() during offline indexing.
    Returns list of (source_id, target_id, edge_type) tuples.
    """
    edges = []
    
    # Pre-extract section refs and defined terms for all docs
    doc_refs = {}
    doc_defined_terms = {}
    doc_section_ids = {}
    
    for doc in documents:
        doc_refs[doc.id] = _extract_section_refs(doc.text)
        doc_defined_terms[doc.id] = _extract_defined_terms(doc.text)
        if '_chunk_' in doc.id:
            match = re.match(r'(?:Section\s+)?(\d+(?:\.\d+)*)', doc.text.strip())
            doc_section_ids[doc.id] = match.group(1) if match else ''
        else:
            doc_section_ids[doc.id] = ''
    
    # Compare all pairs for cross-references
    doc_list = list(documents)
    for i in range(len(doc_list)):
        for j in range(i + 1, len(doc_list)):
            doc_a = doc_list[i]
            doc_b = doc_list[j]
            
            # A references B's section
            section_id_b = doc_section_ids[doc_b.id]
            if section_id_b:
                for ref in doc_refs[doc_a.id]:
                    if section_id_b in ref:
                        edges.append((doc_a.id, doc_b.id, 'cross_ref'))
                        break
            
            # B references A's section
            section_id_a = doc_section_ids[doc_a.id]
            if section_id_a:
                for ref in doc_refs[doc_b.id]:
                    if section_id_a in ref:
                        edges.append((doc_b.id, doc_a.id, 'cross_ref'))
                        break
            
            # Defined term links
            text_b_lower = doc_b.text.lower()
            text_a_lower = doc_a.text.lower()
            
            for term in doc_defined_terms[doc_a.id]:
                if term in text_b_lower:
                    edges.append((doc_a.id, doc_b.id, 'defined_term'))
                    break
            
            for term in doc_defined_terms[doc_b.id]:
                if term in text_a_lower:
                    edges.append((doc_b.id, doc_a.id, 'defined_term'))
                    break
    
    return edges

