"""Shared text chunking utilities for V3 and Traditional RAG.

UPDATED (Fix 3.1): Structure-aware chunking.
  - Detects section boundaries via regex cascade
  - Prepends section headings to each chunk
  - Preserves section_id and section_heading metadata
  - Increased overlap from 50 → 100 chars
  - Falls back to character splitting at ~1000 chars if no sections found
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Shared chunking configuration
CHUNK_SIZE = 500           # For character fallback
CHUNK_OVERLAP = 100        # Fix 3.1: Increased from 50 → 100
MAX_SECTION_SIZE = 1000    # Fix 3.1: Max section size before sub-splitting
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Fix 3.1: Section boundary patterns (priority cascade)
SECTION_PATTERNS = [
    # Priority 1: Explicit section markers (highest confidence)
    re.compile(r'\n(?=ARTICLE\s+[IVXLC]+\.?\s)', re.MULTILINE),         # ARTICLE I, ARTICLE II
    re.compile(r'\n(?=Section\s+\d+(?:\.\d+)*\.?\s)', re.MULTILINE),    # Section 1., Section 1.2
    re.compile(r'\n(?=\d+\.\d+\.?\s+[A-Z])', re.MULTILINE),            # 1.1 Definitions
    re.compile(r'\n(?=\d+\.\s+[A-Z][A-Z\s]+)', re.MULTILINE),          # 1. DEFINITIONS
    # Priority 2: Typographic markers (medium confidence)
    re.compile(r'\n(?=[A-Z][A-Z\s]{10,}(?:\n|\.))', re.MULTILINE),     # ALL CAPS HEADERS (10+ chars)
]

# Pattern to extract section identifiers from heading text
SECTION_ID_PATTERNS = [
    re.compile(r'ARTICLE\s+([IVXLC]+)', re.IGNORECASE),
    re.compile(r'Section\s+(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'^(\d+(?:\.\d+)*)\.\s', re.MULTILINE),
]

MIN_SECTIONS_THRESHOLD = 3  # Minimum splits to consider pattern successful


@dataclass
class TextChunk:
    """A chunk of text with metadata.
    
    Fix 3.1: Added section_id, section_heading for structure-aware chunking.
    """
    id: str
    text: str
    source_doc_id: str
    source_title: str
    chunk_index: int
    section_id: Optional[str] = None
    section_heading: Optional[str] = None


def _extract_section_id(heading: str) -> Optional[str]:
    """Extract a section identifier (e.g., '3.2', 'VIII') from a heading string."""
    for pattern in SECTION_ID_PATTERNS:
        match = pattern.search(heading)
        if match:
            return match.group(1)
    return None


def _split_by_sections(text: str) -> Optional[List[dict]]:
    """Try to split text by section boundaries using regex cascade.
    
    Returns list of {text, heading, section_id} dicts, or None if no good splits found.
    """
    for pattern in SECTION_PATTERNS:
        parts = pattern.split(text)
        # Filter out tiny fragments
        sections = []
        for part in parts:
            part = part.strip()
            if len(part) > 50:  # Skip very tiny fragments
                first_line = part.split('\n')[0].strip()
                section_id = _extract_section_id(first_line)
                sections.append({
                    "text": part,
                    "heading": first_line[:100],
                    "section_id": section_id,
                })
        
        if len(sections) >= MIN_SECTIONS_THRESHOLD:
            return sections
    
    return None  # No pattern produced enough sections


def _subsplit_large_section(section_text: str, max_size: int = MAX_SECTION_SIZE) -> List[str]:
    """Sub-split a section that exceeds max_size using character splitting."""
    if len(section_text) <= max_size:
        return [section_text]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    return splitter.split_text(section_text)


def create_chunker() -> RecursiveCharacterTextSplitter:
    """Create the standard text splitter (fallback when no sections found)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )


def chunk_text(text: str, doc_id: str, doc_title: str) -> List[TextChunk]:
    """Split a document's text into chunks.
    
    Fix 3.1: Structure-aware chunking.
    1. Try to split by section boundaries (regex cascade)
    2. Sub-split large sections at ~1000 chars
    3. Prepend section heading to each chunk
    4. Fall back to character splitting if no sections found
    
    Args:
        text: The full document text
        doc_id: Unique identifier for the source document
        doc_title: Title of the source document
    
    Returns:
        List of TextChunk objects with section metadata
    """
    chunks = []
    
    # Step 1: Try structure-aware splitting
    sections = _split_by_sections(text)
    
    if sections:
        # Structure-aware path
        chunk_idx = 0
        for section in sections:
            heading = section["heading"]
            section_id = section["section_id"]
            section_text = section["text"]
            
            # Sub-split large sections
            sub_parts = _subsplit_large_section(section_text)
            
            for sub_part in sub_parts:
                # Prepend heading to chunk text if it's a sub-split
                # (the first sub-part already starts with the heading)
                if sub_part != section_text and heading:
                    chunk_content = f"[{heading}]\n{sub_part}"
                else:
                    chunk_content = sub_part
                
                chunk = TextChunk(
                    id=f"{doc_id}_chunk_{chunk_idx:03d}",
                    text=chunk_content,
                    source_doc_id=doc_id,
                    source_title=doc_title,
                    chunk_index=chunk_idx,
                    section_id=section_id,
                    section_heading=heading,
                )
                chunks.append(chunk)
                chunk_idx += 1
    else:
        # Fallback: character splitting
        splitter = create_chunker()
        raw_chunks = splitter.split_text(text)
        
        for i, chunk_text_str in enumerate(raw_chunks):
            chunk = TextChunk(
                id=f"{doc_id}_chunk_{i:03d}",
                text=chunk_text_str,
                source_doc_id=doc_id,
                source_title=doc_title,
                chunk_index=i,
                section_id=None,
                section_heading=None,
            )
            chunks.append(chunk)
    
    return chunks


def chunk_corpus_documents(documents: list) -> List[TextChunk]:
    """Chunk all documents in a corpus.
    
    Args:
        documents: List of CorpusDocument objects (with id, title, text)
    
    Returns:
        List of all TextChunks from all documents
    """
    all_chunks = []
    
    for doc in documents:
        # Combine title + text for each document
        full_text = f"{doc.title}\n{doc.text}"
        doc_chunks = chunk_text(full_text, doc.id, doc.title)
        all_chunks.extend(doc_chunks)
    
    return all_chunks
