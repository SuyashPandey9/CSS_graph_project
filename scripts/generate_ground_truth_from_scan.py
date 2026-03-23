"""Generate 48 ground truth entries from scan results - best from entire 510-contract dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

CUAD_PATH = Path(r"C:\Users\suyas\Downloads\archive (1)\CUAD_v1\full_contract_txt")
SCAN_RESULTS = Path(__file__).parent.parent / "data" / "cuad_trap_scan_results.json"


def get_short_id(name: str, trap: str, idx: int) -> str:
    """Create short contract ID from filename."""
    # e.g. BERKELEYLIGHTS,INC_06_26_2020-EX-10.12 -> berkeley
    base = name.replace(".txt", "")[:40]
    base = re.sub(r'[^a-zA-Z0-9]', '_', base).strip("_").lower()
    base = base[:15] if len(base) > 15 else base
    return f"{base}_{trap}_{idx:02d}"


def extract_trap_a_content(text: str) -> Optional[dict]:
    """Extract a notwithstanding clause with context for Trap A."""
    # Find "notwithstanding" with section reference
    pattern = r'([^.]{0,80}(?:notwithstanding|except as|subject to)[^.]{0,200}\.)'
    matches = list(re.finditer(pattern, text, re.I | re.DOTALL))
    if not matches:
        return None
    m = matches[0]
    snippet = m.group(1).replace("\n", " ").strip()[:400]
    # Try to extract section ref
    section_match = re.search(r'[Ss]ection\s+\d+(?:\.\d+)*(?:\([a-z]\))?', snippet)
    section = section_match.group(0) if section_match else "relevant section"
    return {"snippet": snippet, "section": section}


def extract_trap_b_content(text: str, definitions: list) -> Optional[dict]:
    """Extract a definition for Trap B query."""
    if not definitions:
        return None
    term = definitions[0]
    if term in ("[***]", "including", "1", "to the extent"):
        term = definitions[1] if len(definitions) > 1 else None
    if not term:
        return None
    # Find definition in text
    escaped = re.escape(term)
    pattern = rf'"{escaped}"\s+(?:means|shall\s+mean)\s+([^.]{{0,300}}\.)'
    match = re.search(pattern, text)
    if match:
        def_text = match.group(1).replace("\n", " ").strip()[:350]
        return {"term": term, "definition": def_text}
    # Try without quotes
    pattern = rf'\b{re.escape(term)}\b\s+(?:means|shall\s+mean)\s+([^.]{{0,300}}\.)'
    match = re.search(pattern, text, re.I)
    if match:
        return {"term": term, "definition": match.group(1).replace("\n", " ").strip()[:350]}
    return {"term": term, "definition": f"Defined in the Agreement."}


def extract_trap_c_content(text: str) -> Optional[dict]:
    """Extract multiple sections for Trap C (scattered)."""
    sections = re.findall(r'\b(?:Section|Article)\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)', text, re.I)
    unique = list(dict.fromkeys(sections))[:5]
    if len(unique) >= 3:
        return {"sections": unique}
    return None


def generate_entry(contract_name: str, trap_type: str, idx: int, text: str, scan_data: dict) -> Optional[dict]:
    """Generate one ground truth entry."""
    short_id = get_short_id(contract_name, trap_type.replace("trap_", ""), idx)
    
    if trap_type == "trap_a":
        content = extract_trap_a_content(text)
        if not content:
            return None
        query = f"What are the key obligations or exceptions in the {content['section']} of this agreement?"
        gt = f"Per {content['section']}, {content['snippet'][:300]}..."
        relevant = [content["section"]]
        info_units = ["Main obligation described", "Exception or qualification present", "Section reference included"]
        
    elif trap_type == "trap_b":
        defs = scan_data.get("trap_b_definitions", [])
        content = extract_trap_b_content(text, defs)
        if not content:
            return None
        term = content["term"]
        query = f"What restrictions or rights apply in connection with {term} under this agreement?"
        gt = f"'{term}' is defined as: {content['definition']} This definition applies wherever {term} is referenced in the Agreement."
        relevant = [f"Definition of {term}"]
        info_units = [f"{term} defined", "Definition applies throughout", "Usage context"]
        
    elif trap_type == "trap_c":
        content = extract_trap_c_content(text)
        if not content:
            return None
        secs = content["sections"]
        query = "List all material obligations, rights, or conditions that span multiple sections of this agreement."
        gt = f"The Agreement addresses these topics across multiple sections including Section {secs[0]}, Section {secs[1]}, and Section {secs[2]}. A complete answer requires aggregating information from these and potentially other sections."
        relevant = [f"Section {s}" for s in secs]
        info_units = [f"Section {s} relevant" for s in secs[:3]]
    else:
        return None
    
    return {
        "id": short_id,
        "contract_file": contract_name,
        "query": query,
        "trap_type": trap_type,
        "difficulty": "multi_hop",
        "ground_truth_answer": gt,
        "relevant_sections": relevant,
        "information_units": info_units,
        "notes": f"Selected from full 510-contract CUAD dataset. {trap_type}.",
    }


def main():
    scan = json.loads(SCAN_RESULTS.read_text(encoding="utf-8"))
    
    # Select 16 per trap type - by that trap's count
    by_a = sorted(scan, key=lambda x: x["trap_a_count"], reverse=True)[:20]
    by_b = sorted(scan, key=lambda x: x["trap_b_count"], reverse=True)[:20]
    by_c = sorted(scan, key=lambda x: x["trap_c_sections"], reverse=True)[:20]
    
    entries = []
    seen_contracts = set()
    
    def add_from_list(candidates: list, trap: str, max_per_trap: int = 16):
        count = 0
        for c in candidates:
            if count >= max_per_trap:
                break
            name = c["name"]
            if name in seen_contracts and len(entries) > 32:
                continue  # Prefer diversity in later picks
            path = Path(c["path"])
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                entry = generate_entry(name, trap, count + 1, text, c)
                if entry:
                    entries.append(entry)
                    seen_contracts.add(name)
                    count += 1
            except Exception as e:
                print(f"  Skip {name}: {e}")
        return count
    
    print("Generating Trap A entries...")
    add_from_list(by_a, "trap_a")
    print("Generating Trap B entries...")
    add_from_list(by_b, "trap_b")
    print("Generating Trap C entries...")
    add_from_list(by_c, "trap_c")
    
    # If we're short, fill from next best
    while len(entries) < 48:
        remaining = 48 - len(entries)
        for c in scan:
            if c["name"] in seen_contracts:
                continue
            path = Path(c["path"])
            if not path.exists():
                continue
            trap = "trap_a" if len([e for e in entries if e["trap_type"]=="trap_a"]) < 16 else \
                   "trap_b" if len([e for e in entries if e["trap_type"]=="trap_b"]) < 16 else "trap_c"
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                entry = generate_entry(c["name"], trap, len(entries) + 1, text, c)
                if entry:
                    entries.append(entry)
                    seen_contracts.add(c["name"])
                    remaining -= 1
                    if remaining <= 0:
                        break
            except Exception:
                pass
    
    # Build final ground_truth structure
    output = {
        "_schema_version": "1.0",
        "_description": "Ground truth from full 510-contract CUAD dataset. Best 48 queries.",
        "_trap_types": {
            "trap_a": "Invisible Exception",
            "trap_b": "Distant Definition",
            "trap_c": "Scattered Components",
        },
        "_source": "Full CUAD dataset (510 contracts) - scripted selection",
        "annotations": entries[:48],
    }
    
    out_path = Path(__file__).parent.parent / "data" / "ground_truth_full_dataset.json"
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nGenerated {len(entries[:48])} entries. Saved to {out_path}")
    print(f"  Trap A: {len([e for e in entries if e['trap_type']=='trap_a'])}")
    print(f"  Trap B: {len([e for e in entries if e['trap_type']=='trap_b'])}")
    print(f"  Trap C: {len([e for e in entries if e['trap_type']=='trap_c'])}")
    print(f"  Unique contracts: {len(seen_contracts)}")


if __name__ == "__main__":
    main()
