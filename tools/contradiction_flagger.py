"""Contradiction flagging for V3 context (Fix 2.11).

NOT a CSS feature — metadata annotation on retrieved context.
Scans final context for legal contradiction indicators and attaches
warning annotations for the LLM to consider.

Indicators: "notwithstanding", "subject to", "except as provided",
            "unless otherwise", "provided however", etc.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from core.types import Graph


# Contradiction/qualification indicator patterns
CONTRADICTION_PATTERNS = [
    re.compile(r'\bnotwithstanding\s+(?:the\s+)?(?:foregoing|above|anything)', re.IGNORECASE),
    re.compile(r'\bsubject\s+to\b', re.IGNORECASE),
    re.compile(r'\bexcept\s+as\s+(?:provided|set forth|described|otherwise)', re.IGNORECASE),
    re.compile(r'\bunless\s+otherwise\s+(?:specified|agreed|provided|stated)', re.IGNORECASE),
    re.compile(r'\bprovided,?\s+however\b', re.IGNORECASE),
    re.compile(r'\bnotwithstanding\b', re.IGNORECASE),
    re.compile(r'\bexcluding\b', re.IGNORECASE),
    re.compile(r'\bin\s+no\s+event\s+shall\b', re.IGNORECASE),
    re.compile(r'\bto\s+the\s+extent\s+(?:that|permitted)\b', re.IGNORECASE),
]


def detect_contradiction_flags(graph: Graph) -> List[Tuple[str, str, str]]:
    """Scan graph nodes for contradiction/qualification indicators.
    
    Args:
        graph: Final optimized graph
    
    Returns:
        List of (node_id, indicator_text, context_snippet) tuples
    """
    flags = []
    
    for node in graph.nodes:
        for pattern in CONTRADICTION_PATTERNS:
            matches = pattern.finditer(node.text)
            for match in matches:
                # Extract a short context window around the match
                start = max(0, match.start() - 30)
                end = min(len(node.text), match.end() + 50)
                snippet = node.text[start:end].strip()
                
                flags.append((node.id, match.group(0), snippet))
    
    return flags


def build_contradiction_annotation(graph: Graph) -> str:
    """Build a warning annotation string for the LLM prompt.
    
    If no contradiction indicators found, returns empty string.
    If indicators found, returns a formatted warning block.
    
    Args:
        graph: Final optimized graph
    
    Returns:
        Warning annotation string (or empty string)
    """
    flags = detect_contradiction_flags(graph)
    
    if not flags:
        return ""
    
    # Deduplicate by indicator text
    seen_indicators = set()
    unique_flags = []
    for node_id, indicator, snippet in flags:
        if indicator.lower() not in seen_indicators:
            seen_indicators.add(indicator.lower())
            unique_flags.append((node_id, indicator, snippet))
    
    # Build annotation
    lines = [
        "⚠️ POTENTIAL CONFLICT: The retrieved context contains provisions that may "
        "override or qualify each other. Pay attention to exceptions and qualifications "
        "when synthesizing your answer.",
        "",
        "Detected qualification indicators:",
    ]
    
    for node_id, indicator, snippet in unique_flags[:5]:  # Limit to 5 flags
        lines.append(f'  - "{indicator}" in {node_id}: "...{snippet}..."')
    
    return "\n".join(lines)
