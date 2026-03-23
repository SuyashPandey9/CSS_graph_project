"""Generate additional ground truth files (ground_truth_2.json, ground_truth_3.json).

Each file has 48 queries from different contract batches in the 510-contract CUAD dataset.
Uses the same extraction logic as generate_ground_truth_from_scan.py but selects from
different rank ranges to avoid overlap with existing ground_truth.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

SCAN_RESULTS = Path(__file__).parent.parent / "data" / "cuad_trap_scan_results.json"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_short_id(name: str, trap: str, idx: int, batch_suffix: str) -> str:
    """Create short contract ID from filename with batch suffix."""
    base = name.replace(".txt", "")[:40]
    base = re.sub(r"[^a-zA-Z0-9]", "_", base).strip("_").lower()
    base = base[:12] if len(base) > 12 else base
    return f"{base}_{trap}_{idx:02d}_{batch_suffix}"


def extract_trap_a_content(text: str) -> Optional[dict]:
    """Extract a notwithstanding clause with context for Trap A."""
    pattern = r"([^.]{0,80}(?:notwithstanding|except as|subject to)[^.]{0,200}\.)"
    matches = list(re.finditer(pattern, text, re.I | re.DOTALL))
    if not matches:
        return None
    m = matches[0]
    snippet = m.group(1).replace("\n", " ").strip()[:400]
    section_match = re.search(r"[Ss]ection\s+\d+(?:\.\d+)*(?:\([a-z]\))?", snippet)
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
    escaped = re.escape(term)
    pattern = rf'"{escaped}"\s+(?:means|shall\s+mean)\s+([^.]{{0,300}}\.)'
    match = re.search(pattern, text)
    if match:
        def_text = match.group(1).replace("\n", " ").strip()[:350]
        return {"term": term, "definition": def_text}
    pattern = rf"\b{re.escape(term)}\b\s+(?:means|shall\s+mean)\s+([^.]{{0,300}}\.)"
    match = re.search(pattern, text, re.I)
    if match:
        return {"term": term, "definition": match.group(1).replace("\n", " ").strip()[:350]}
    return {"term": term, "definition": "Defined in the Agreement."}


def extract_trap_c_content(text: str) -> Optional[dict]:
    """Extract multiple sections for Trap C (scattered)."""
    sections = re.findall(r"\b(?:Section|Article)\s+(\d+(?:\.\d+)*(?:\([a-z]\))?)", text, re.I)
    unique = list(dict.fromkeys(sections))[:5]
    if len(unique) >= 3:
        return {"sections": unique}
    return None


def generate_entry(
    contract_name: str,
    trap_type: str,
    idx: int,
    text: str,
    scan_data: dict,
    batch_suffix: str,
) -> Optional[dict]:
    """Generate one ground truth entry."""
    short_id = get_short_id(contract_name, trap_type.replace("trap_", ""), idx, batch_suffix)

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
        "notes": f"Ground truth batch {batch_suffix}. {trap_type}.",
    }


def load_excluded_contracts(gt_files: list[Path]) -> set[str]:
    """Load contract filenames from existing ground truth files to exclude."""
    excluded = set()
    for p in gt_files:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            for ann in data.get("annotations", []):
                excluded.add(ann["contract_file"])
    return excluded


def generate_batch(
    batch_num: int,
    excluded: set[str],
    scan: list,
) -> list[dict]:
    """Generate 48 entries for a batch, excluding already-used contracts."""
    batch_suffix = str(batch_num)
    # Rank ranges: batch 2 = 21-70, batch 3 = 71-120
    start = 20 + (batch_num - 2) * 50
    end = start + 50

    by_a = sorted(scan, key=lambda x: x["trap_a_count"], reverse=True)[start:end]
    by_b = sorted(scan, key=lambda x: x["trap_b_count"], reverse=True)[start:end]
    by_c = sorted(scan, key=lambda x: x["trap_c_sections"], reverse=True)[start:end]

    entries = []
    seen_contracts = set()

    def add_from_list(candidates: list, trap: str, max_per_trap: int = 16):
        count = 0
        for c in candidates:
            if count >= max_per_trap:
                break
            name = c["name"]
            if name in excluded or name in seen_contracts:
                continue
            path = Path(c["path"])
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                entry = generate_entry(name, trap, count + 1, text, c, batch_suffix)
                if entry:
                    entries.append(entry)
                    seen_contracts.add(name)
                    count += 1
            except Exception as e:
                print(f"  Skip {name}: {e}")
        return count

    print(f"  Trap A (ranks {start+1}-{end})...")
    add_from_list(by_a, "trap_a")
    print(f"  Trap B (ranks {start+1}-{end})...")
    add_from_list(by_b, "trap_b")
    print(f"  Trap C (ranks {start+1}-{end})...")
    add_from_list(by_c, "trap_c")

    # Fill shortfall from remaining scan
    while len(entries) < 48:
        for c in scan:
            if c["name"] in excluded or c["name"] in seen_contracts:
                continue
            path = Path(c["path"])
            if not path.exists():
                continue
            trap = (
                "trap_a"
                if len([e for e in entries if e["trap_type"] == "trap_a"]) < 16
                else "trap_b"
                if len([e for e in entries if e["trap_type"] == "trap_b"]) < 16
                else "trap_c"
            )
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                entry = generate_entry(c["name"], trap, len(entries) + 1, text, c, batch_suffix)
                if entry:
                    entries.append(entry)
                    seen_contracts.add(c["name"])
                    break
            except Exception:
                pass
        else:
            break

    return entries[:48]


def main():
    scan = json.loads(SCAN_RESULTS.read_text(encoding="utf-8"))

    # Exclude contracts from ground_truth.json
    excluded = load_excluded_contracts([DATA_DIR / "ground_truth.json"])
    print(f"Excluding {len(excluded)} contracts from ground_truth.json")

    for batch_num in [2, 3]:
        print(f"\n--- Generating ground_truth_{batch_num}.json ---")
        entries = generate_batch(batch_num, excluded, scan)

        if len(entries) < 48:
            print(f"  WARNING: Only {len(entries)} entries (target 48)")

        output = {
            "_schema_version": "1.0",
            "_description": f"Ground truth batch {batch_num} - 48 queries from full 510-contract CUAD dataset.",
            "_trap_types": {
                "trap_a": "Invisible Exception",
                "trap_b": "Distant Definition",
                "trap_c": "Scattered Components",
            },
            "_source": f"Full CUAD dataset - batch {batch_num} (different contracts from ground_truth.json)",
            "annotations": entries,
        }

        out_path = DATA_DIR / f"ground_truth_{batch_num}.json"
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"  Saved {len(entries)} entries to {out_path}")
        print(f"  Trap A: {len([e for e in entries if e['trap_type']=='trap_a'])}")
        print(f"  Trap B: {len([e for e in entries if e['trap_type']=='trap_b'])}")
        print(f"  Trap C: {len([e for e in entries if e['trap_type']=='trap_c'])}")

        # Add this batch's contracts to exclusion for next batch
        excluded.update(a["contract_file"] for a in entries)

    print("\nDone. Created ground_truth_2.json and ground_truth_3.json")


if __name__ == "__main__":
    main()
