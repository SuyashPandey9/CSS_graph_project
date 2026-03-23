"""Scan all 510 CUAD contracts for trap patterns and rank by trap richness.

Outputs a ranked list of contracts with trap counts for selecting best 48 queries.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

DEFAULT_CUAD_PATH = r"C:\Users\suyas\Downloads\archive (1)\CUAD_v1\full_contract_txt"


def scan_contract(path: Path) -> dict:
    """Scan a single contract for trap patterns. Returns counts and sample locations."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text_lower = text.lower()
    
    result = {
        "path": str(path),
        "name": path.name,
        "trap_a_count": 0,
        "trap_b_count": 0,
        "trap_c_count": 0,
        "trap_a_matches": [],
        "trap_b_definitions": [],
        "trap_c_sections": 0,
        "total_score": 0,
    }
    
    # Trap A: notwithstanding, except as, subject to, provided however, unless otherwise
    trap_a_patterns = [
        r'\bnotwithstanding\b',
        r'\bexcept\s+as\b',
        r'\bsubject\s+to\b',
        r'\bprovided\s+however\b',
        r'\bunless\s+otherwise\b',
    ]
    for pat in trap_a_patterns:
        matches = list(re.finditer(pat, text_lower, re.I))
        result["trap_a_count"] += len(matches)
        if matches and len(result["trap_a_matches"]) < 3:
            for m in matches[:1]:
                start = max(0, m.start() - 50)
                end = min(len(text), m.end() + 100)
                result["trap_a_matches"].append(text[start:end].replace("\n", " ")[:150])
    
    # Trap B: definitions - "X" means, "X" shall mean, Section 1., Article 1
    # Count defined terms (quoted terms with means/shall mean)
    def_pattern = r'"([^"]+)"\s+(?:means|shall\s+mean)'
    def_matches = re.findall(def_pattern, text, re.I)
    result["trap_b_definitions"] = list(set(def_matches))[:15]  # unique, max 15
    result["trap_b_count"] = len(result["trap_b_definitions"])
    
    # Also count Article 1 / Section 1.XX as definition indicators
    art1 = len(re.findall(r'\b(?:article\s+1|section\s+1\.\d+)', text_lower))
    if art1 > 5 and result["trap_b_count"] < art1:
        result["trap_b_count"] = max(result["trap_b_count"], min(art1, 30))
    
    # Trap C: scattered - look for numbered lists, "list all", multiple sections
    # Count section references (Section X.Y, Section X.Y.Z)
    section_refs = re.findall(r'\bsection\s+\d+(?:\.\d+)*(?:\([a-z]\))?', text_lower)
    unique_sections = len(set(section_refs))
    result["trap_c_sections"] = unique_sections
    result["trap_c_count"] = min(unique_sections // 3, 20)  # rough: 3+ sections = 1 trap_c potential
    
    # Total score: weighted sum
    result["total_score"] = (
        result["trap_a_count"] * 2 +  # notwithstanding etc are strong
        result["trap_b_count"] * 1.5 +
        result["trap_c_count"] * 1
    )
    
    return result


def main():
    cuad_path = Path(DEFAULT_CUAD_PATH)
    if not cuad_path.exists():
        print(f"ERROR: CUAD path not found: {cuad_path}")
        return
    
    files = sorted(cuad_path.glob("*.txt"))
    print(f"Scanning {len(files)} contracts...")
    
    results = []
    for i, f in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  Scanned {i + 1}/{len(files)}...")
        try:
            r = scan_contract(f)
            results.append(r)
        except Exception as e:
            print(f"  Error {f.name}: {e}")
    
    # Sort by total score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)
    
    # Output top 100 for manual/automated query selection
    print("\n" + "=" * 80)
    print("TOP 100 CONTRACTS BY TRAP RICHNESS (for selecting best 48 queries)")
    print("=" * 80)
    
    for i, r in enumerate(results[:100], 1):
        print(f"\n{i:3d}. {r['name'][:70]}")
        print(f"     Score: {r['total_score']:.1f}  |  Trap A: {r['trap_a_count']}  Trap B: {r['trap_b_count']}  Trap C: {r['trap_c_sections']} sections")
        if r["trap_b_definitions"][:3]:
            print(f"     Sample definitions: {r['trap_b_definitions'][:3]}")
    
    # Save full results for scripted query generation
    import json
    out_path = Path(__file__).parent.parent / "data" / "cuad_trap_scan_results.json"
    out_path.parent.mkdir(exist_ok=True)
    # Make serializable
    for r in results:
        r["path"] = str(r["path"])
    out_path.write_text(json.dumps(results[:200], indent=2, default=str), encoding="utf-8")
    print(f"\n\nFull results saved to: {out_path}")
    print(f"Total contracts scanned: {len(results)}")


if __name__ == "__main__":
    main()
