"""CUAD Dataset Loader for V3 RAG System.

Loads legal contracts from CUAD (Contract Understanding Atticus Dataset)
and chunks them into sections for use with V3.

Usage:
    from data.cuad_loader import load_cuad_corpus
    corpus = load_cuad_corpus("C:/path/to/CUAD_v1/full_contract_txt", max_contracts=50)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from core.types import Corpus, CorpusDocument


# Default path to CUAD contracts
DEFAULT_CUAD_PATH = r"C:\Users\suyas\Downloads\archive (1)\CUAD_v1\full_contract_txt"


def load_cuad_corpus(
    contracts_dir: str = DEFAULT_CUAD_PATH,
    max_contracts: Optional[int] = 50,
    max_chunks_per_contract: int = 20,
    chunk_by: str = "sections",  # "sections" or "paragraphs"
) -> Corpus:
    """Load CUAD contracts as a Corpus.
    
    Args:
        contracts_dir: Path to full_contract_txt folder
        max_contracts: Maximum number of contracts to load (None = all contracts)
        max_chunks_per_contract: Max chunks per contract to avoid huge corpus
        chunk_by: "sections" (by headers) or "paragraphs" (by line breaks)
    
    Returns:
        Corpus with contract chunks as documents
    """
    contracts_path = Path(contracts_dir)
    
    if not contracts_path.exists():
        raise FileNotFoundError(f"CUAD contracts folder not found: {contracts_dir}")
    
    txt_files = sorted(contracts_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {contracts_dir}")
    
    limit = len(txt_files) if max_contracts is None else min(max_contracts, len(txt_files))
    print(f"[CUAD] Found {len(txt_files)} contracts, loading {limit}...")
    
    documents: List[CorpusDocument] = []
    
    for i, txt_file in enumerate(txt_files[:limit]):
        try:
            # Extract contract name from filename
            contract_name = txt_file.stem
            
            # Read contract text
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            
            # Chunk the contract
            if chunk_by == "sections":
                chunks = _chunk_by_sections(text, max_chunks_per_contract)
            else:
                chunks = _chunk_by_paragraphs(text, max_chunks_per_contract)
            
            # Create documents for each chunk
            for j, chunk in enumerate(chunks):
                doc_id = f"{contract_name}_chunk_{j:03d}"
                section_name = chunk.get("section", f"chunk_{j}")
                doc = CorpusDocument(
                    id=doc_id,
                    title=f"{contract_name[:50]} - {section_name[:30]}",
                    text=chunk["text"],
                    source=str(txt_file),
                )
                documents.append(doc)
            
            if (i + 1) % 10 == 0:
                print(f"[CUAD] Loaded {i + 1}/{limit} contracts...")
                
        except Exception as e:
            print(f"[CUAD] Warning: Failed to load {txt_file.name}: {e}")
            continue
    
    print(f"[CUAD] Created corpus with {len(documents)} document chunks from {limit} contracts")
    
    return Corpus(documents=documents)


def _chunk_by_sections(text: str, max_chunks: int) -> List[dict]:
    """Chunk contract by section headers.
    
    Looks for patterns like:
    - "ARTICLE I", "ARTICLE II", etc.
    - "Section 1.", "Section 2.", etc.
    - "1. DEFINITIONS", "2. LICENSE GRANT", etc.
    - All-caps headers like "DEFINITIONS", "TERM AND TERMINATION"
    """
    # Split by common section patterns
    section_patterns = [
        r'\n(?=ARTICLE\s+[IVXLC]+\.?)',           # ARTICLE I, ARTICLE II
        r'\n(?=Section\s+\d+\.)',                  # Section 1., Section 2.
        r'\n(?=\d+\.\s+[A-Z][A-Z\s]+)',            # 1. DEFINITIONS
        r'\n(?=[A-Z][A-Z\s]{10,}(?:\n|\.))',       # ALL CAPS HEADERS (10+ chars)
    ]
    
    # Try each pattern to find good splits
    chunks = []
    
    for pattern in section_patterns:
        parts = re.split(pattern, text)
        if len(parts) > 5:  # Found reasonable sections
            for part in parts:
                part = part.strip()
                if len(part) > 100:  # Skip tiny chunks
                    # Extract section name from first line
                    first_line = part.split('\n')[0][:100]
                    chunks.append({
                        "text": part[:2000],  # Limit chunk size
                        "section": first_line.strip(),
                    })
            break
    
    # Fallback: if no good sections found, chunk by paragraphs
    if len(chunks) < 3:
        return _chunk_by_paragraphs(text, max_chunks)
    
    return chunks[:max_chunks]


def _chunk_by_paragraphs(text: str, max_chunks: int, min_chunk_size: int = 200) -> List[dict]:
    """Chunk contract by paragraphs (double newlines)."""
    # Split by double newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Add to current chunk
        if len(current_chunk) + len(para) < 1500:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            # Save current chunk and start new one
            if len(current_chunk) >= min_chunk_size:
                chunks.append({
                    "text": current_chunk,
                    "section": current_chunk.split('\n')[0][:50],
                })
            current_chunk = para
    
    # Don't forget the last chunk
    if len(current_chunk) >= min_chunk_size:
        chunks.append({
            "text": current_chunk,
            "section": current_chunk.split('\n')[0][:50],
        })
    
    return chunks[:max_chunks]


def load_cuad_sample(n_contracts: int = 10) -> Corpus:
    """Quick helper to load a small sample for testing."""
    return load_cuad_corpus(max_contracts=n_contracts)


# For command-line testing
if __name__ == "__main__":
    corpus = load_cuad_sample(5)
    print(f"\nSample documents:")
    for doc in corpus.documents[:3]:
        print(f"  - {doc.id}: {doc.text[:100]}...")
