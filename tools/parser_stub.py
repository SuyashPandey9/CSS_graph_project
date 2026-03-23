"""Deterministic rule-based parser for V3.

V3 Spec: f_parse(x) → (E_x, R_x, Q(x))
    - E_x: entities (nouns/concepts)
    - R_x: relationships (verbs/connections)
    - Q(x): subqueries (decomposed questions)

Fix 3.2: Removed structural terms from stopwords. Added regex for section references.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from tools.base import BaseParser

# Updated regex to preserve hyphenated compound terms like "non-compete"
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

# Fix 3.2: Section reference patterns — extracted as compound entities BEFORE tokenization
_SECTION_REF_PATTERNS = [
    re.compile(r'Section\s+\d+(?:\.\d+)*', re.IGNORECASE),     # Section 3.2, Section 12.3.1
    re.compile(r'Article\s+[IVXLC]+', re.IGNORECASE),           # Article V, Article XII
    re.compile(r'Exhibit\s+[A-Z]', re.IGNORECASE),              # Exhibit A, Exhibit B
    re.compile(r'Schedule\s+\d+', re.IGNORECASE),               # Schedule 1, Schedule 2
    re.compile(r'Appendix\s+[A-Z]', re.IGNORECASE),             # Appendix A
]

# Legal domain compound terms to detect and keep together
_COMPOUND_TERMS = {
    "non-compete", "non-disclosure", "non-solicitation", "non-exclusive",
    "intellectual property", "good faith", "best efforts", "force majeure",
    "governing law", "confidentiality agreement", "material breach",
    "termination rights", "indemnification clause", "liability cap",
}

_STOPWORDS = {
    # Standard stopwords
    "the", "and", "or", "of", "to", "a", "an", "in", "is", "are", "for",
    "on", "with", "by", "from", "this", "that", "these", "those", "it",
    "its", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must",
    "shall", "can", "not", "no", "yes", "what", "which", "who", "whom",
    "how", "when", "where", "why", "if", "then", "so", "than", "too",
    "very", "just", "also", "more", "most", "some", "any", "all", "each",
    "every", "both", "few", "many", "much", "such", "own", "same", "other",
    # Query/action words (not useful entities for retrieval)
    "explain", "describe", "tell", "show", "give", "provide", "list",
    "define", "discuss", "analyze", "compare", "contrast", "identify",
    "find", "get", "make", "help", "need", "want", "know", "understand",
    # Generic nouns (too vague for retrieval)
    "connection", "relationship", "thing", "things", "way", "ways",
    "type", "types", "kind", "kinds", "form", "forms", "part", "parts",
    "example", "examples", "case", "cases", "use", "uses", "time", "times",
    "information", "details", "data", "stuff", "item", "items",
    # Common adjectives
    "common", "main", "major", "minor", "different", "various", "certain",
    "specific", "general", "overall", "key", "important", "relevant",
    # Pronouns and possessives (not useful for retrieval)
    "their", "your", "our", "my", "his", "her", "them", "us",
    # Fix 3.2: REMOVED structural terms that were incorrectly in stopwords:
    # "section", "paragraph", "article", "clause", "subsection", "provision"
    # These are now preserved so queries like "What does Section 3.2 say?" work.
    "outlined", "stated", "described", "mentioned", "specified",
}

# Common relationship indicators
_RELATION_WORDS = {
    "associated", "linked", "related", "connected", "correlated",
    "affects", "causes", "leads", "influences", "impacts",
    "higher", "lower", "increased", "decreased", "between",
}


class ParserStub(BaseParser):
    """Extracts entities, relationships, and subqueries from text.
    
    V3 Spec: f_parse(x) → (E_x, R_x, Q(x))
    """

    def extract(self, text: str) -> dict:
        """Parse text into entities, relationships, and subqueries.
        
        Returns:
            dict with keys:
                - entities: List[str] - extracted concepts/nouns
                - relations: List[Tuple[str, str, str]] - (entity1, relation, entity2)
                - subqueries: List[str] - decomposed questions
                - nouns: List[str] - (legacy, for backward compatibility)
                - verbs: List[str] - (legacy, for backward compatibility)
                - tokens: List[str] - all non-stopword tokens
        """
        tokens = [t.lower() for t in _WORD_RE.findall(text)]
        tokens = [t for t in tokens if t not in _STOPWORDS]
        
        # First: detect compound legal terms in the original text
        text_lower = text.lower()
        compound_entities = []
        for term in _COMPOUND_TERMS:
            if term in text_lower:
                compound_entities.append(term)
        
        # Fix 3.2: Detect section references as compound entities
        section_refs = []
        for pattern in _SECTION_REF_PATTERNS:
            for match in pattern.finditer(text):
                section_refs.append(match.group(0))
        
        # Extract verbs (words ending in -ing, -ed, or in relation words)
        verbs = [t for t in tokens if t.endswith("ing") or t.endswith("ed") or t in _RELATION_WORDS]
        
        # Extract nouns/entities (non-verb content words)
        nouns = [t for t in tokens if t not in verbs and len(t) > 2]
        
        # Combine section refs + compound terms + single-word entities
        # Section refs first (highest priority), then compound terms, then single words
        all_entities = section_refs + compound_entities + nouns
        
        # Remove duplicates while preserving order
        entities = list(dict.fromkeys(all_entities))
        
        # Extract relationships (entity pairs connected by relation words)
        relations = self._extract_relations(text, entities)
        
        # Generate subqueries
        subqueries = self._generate_subqueries(text, entities, relations)
        
        return {
            "entities": entities,
            "relations": relations,
            "subqueries": subqueries,
            # Legacy compatibility
            "nouns": nouns,
            "verbs": verbs,
            "tokens": tokens,
        }
    
    def _extract_relations(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities.
        
        Simple heuristic: if two entities appear near a relation word, link them.
        """
        relations = []
        text_lower = text.lower()
        
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Find relation word between entities
                for rel_word in _RELATION_WORDS:
                    if rel_word in text_lower:
                        # Check if both entities are in the text
                        if e1 in text_lower and e2 in text_lower:
                            relations.append((e1, rel_word, e2))
                            break  # One relation per entity pair
        
        return relations
    
    def _generate_subqueries(self, text: str, entities: List[str], 
                             relations: List[Tuple[str, str, str]]) -> List[str]:
        """Generate subqueries from the parsed components.
        
        V3 Spec: Q(x) = decomposed questions that the graph should answer.
        """
        subqueries = []
        
        # For each entity, create a "what is X" subquery
        for entity in entities[:5]:  # Limit to avoid explosion
            subqueries.append(f"What is {entity}?")
        
        # For each relationship, create a relationship subquery
        for e1, rel, e2 in relations[:3]:  # Limit to avoid explosion
            subqueries.append(f"How is {e1} {rel} to {e2}?")
        
        # Add the original query as a subquery
        if text.strip():
            subqueries.append(text.strip())
        
        return subqueries
