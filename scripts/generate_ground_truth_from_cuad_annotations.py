"""Generate high-quality ground truth from CUADv1.json professional annotations.

Creates multi-hop queries that combine related CUAD clause types, forcing
retrieval systems to find information from multiple contract sections.

Usage:
    python -m scripts.generate_ground_truth_from_cuad_annotations
    python -m scripts.generate_ground_truth_from_cuad_annotations --cuad-json "path/to/CUAD_v1.json"
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Contract selection: 8 contracts with rich multi-hop potential
# Selected by: (1) trap scan density, (2) CUAD clause coverage, (3) industry diversity
# ---------------------------------------------------------------------------

SELECTED_CONTRACTS = [
    {
        "cuad_title_prefix": "ENERGOUSCORP",
        "short_id": "energous",
        "industry": "tech/energy",
        "agreement_type": "Strategic Alliance",
    },
    {
        "cuad_title_prefix": "DovaPharmaceuticalsInc",
        "short_id": "dova",
        "industry": "pharma",
        "agreement_type": "Promotion Agreement",
    },
    {
        "cuad_title_prefix": "BERKELEYLIGHTS",
        "short_id": "berkeley",
        "industry": "biotech",
        "agreement_type": "Collaboration Agreement",
    },
    {
        "cuad_title_prefix": "DigitalCinemaDestinationsCorp",
        "short_id": "digitalcinema",
        "industry": "entertainment",
        "agreement_type": "Affiliate Agreement",
    },
    {
        "cuad_title_prefix": "JOINTCORP",
        "short_id": "jointcorp",
        "industry": "retail/franchise",
        "agreement_type": "Franchise Agreement",
    },
    {
        "cuad_title_prefix": "GOOSEHEADINSURANCE",
        "short_id": "goosehead",
        "industry": "insurance/franchise",
        "agreement_type": "Franchise Agreement",
    },
    {
        "cuad_title_prefix": "RevolutionMedicinesInc",
        "short_id": "revmed",
        "industry": "pharma",
        "agreement_type": "Development Agreement",
    },
    {
        "cuad_title_prefix": "PhasebioPharmaceuticalsInc",
        "short_id": "phasebio",
        "industry": "pharma",
        "agreement_type": "Development Agreement",
    },
]


# ---------------------------------------------------------------------------
# Multi-hop query templates per trap type
# Each template combines 2-3 CUAD clause types into one question
# ---------------------------------------------------------------------------

TRAP_A_TEMPLATES = [
    # Template 1: Liability cap vs uncapped
    {
        "clauses_needed": ["Cap On Liability", "Uncapped Liability"],
        "query_template": "Is liability capped or uncapped under this agreement? What are the specific limitations, and are there any exceptions where liability remains uncapped?",
        "fallback_clauses": ["Cap On Liability", "Competitive Restriction Exception"],
        "fallback_query": "What are the liability limitations in this agreement, and are there any exceptions or carve-outs to those limitations?",
    },
    # Template 2: Non-compete + exception
    {
        "clauses_needed": ["Non-Compete", "Competitive Restriction Exception"],
        "query_template": "What competitive restrictions apply to the parties under this agreement, and what are the specific exceptions or carve-outs to those restrictions?",
        "fallback_clauses": ["Exclusivity", "Competitive Restriction Exception"],
        "fallback_query": "What exclusivity obligations exist in this agreement, and are there exceptions or situations where exclusivity does not apply?",
    },
]

TRAP_B_TEMPLATES = [
    # Template 1: Change of Control + Anti-Assignment
    {
        "clauses_needed": ["Change Of Control", "Anti-Assignment"],
        "query_template": "What happens in the event of a Change of Control, and what are the restrictions on assigning this agreement to third parties?",
        "fallback_clauses": ["Change Of Control", "License Grant"],
        "fallback_query": "How does a Change of Control affect the rights and licenses granted under this agreement?",
    },
    # Template 2: License Grant + IP Ownership + Non-Transferable
    {
        "clauses_needed": ["License Grant", "Ip Ownership Assignment"],
        "query_template": "What licenses are granted under this agreement, and how is intellectual property ownership allocated between the parties?",
        "fallback_clauses": ["License Grant", "Non-Transferable License"],
        "fallback_query": "What licenses are granted under this agreement, and what restrictions exist on transferring or sublicensing those rights?",
    },
]

TRAP_C_TEMPLATES = [
    # Template 1: Financial obligations (scattered across sections)
    {
        "clauses_needed": ["Revenue/Profit Sharing", "Minimum Commitment", "Audit Rights"],
        "query_template": "List all financial obligations and payment-related provisions in this agreement, including any revenue sharing, minimum commitments, and audit rights.",
        "fallback_clauses": ["Revenue/Profit Sharing", "Audit Rights"],
        "fallback_query": "What are the financial reporting and payment obligations in this agreement, including any audit rights?",
    },
    # Template 2: Post-termination + Liquidated Damages
    {
        "clauses_needed": ["Post-Termination Services", "Liquidated Damages", "Termination For Convenience"],
        "query_template": "What happens after this agreement terminates? Describe all post-termination obligations, any termination fees or liquidated damages, and the conditions for termination for convenience.",
        "fallback_clauses": ["Post-Termination Services", "Termination For Convenience"],
        "fallback_query": "What are the termination provisions in this agreement, including any post-termination obligations and conditions for termination for convenience?",
    },
]


def extract_clause_name(question: str) -> str:
    """Extract the clause type name from a CUAD question."""
    m = re.search(r'"([^"]+)"', question)
    return m.group(1) if m else ""


def get_contract_annotations(cuad_entry: dict) -> Dict[str, List[dict]]:
    """Extract all answered clause annotations from a CUAD entry.

    Returns dict mapping clause_name -> list of answer dicts with text and start.
    """
    annotations = {}
    para = cuad_entry["paragraphs"][0]
    context = para["context"]

    for qa in para["qas"]:
        clause = extract_clause_name(qa["question"])
        if not clause:
            continue

        answers = []
        for ans in qa.get("answers", []):
            text = ans.get("text", "").strip()
            if text:
                answers.append({
                    "text": text,
                    "answer_start": ans["answer_start"],
                })

        if answers:
            annotations[clause] = answers

    return annotations


def find_section_for_offset(context: str, offset: int) -> str:
    """Try to identify the section header nearest to a character offset."""
    # Look backwards from offset for section patterns
    search_start = max(0, offset - 500)
    preceding = context[search_start:offset]

    patterns = [
        r"(?:Section|SECTION)\s+(\d+(?:\.\d+)*)",
        r"(?:Article|ARTICLE)\s+([IVXLCDM]+|\d+)",
        r"(\d+(?:\.\d+)+)\s+[A-Z]",
    ]
    best_match = None
    best_pos = -1
    for pat in patterns:
        for m in re.finditer(pat, preceding):
            if m.start() > best_pos:
                best_pos = m.start()
                best_match = m.group(0).strip()

    return best_match or f"offset_{offset}"


def build_annotation_entry(
    contract: dict,
    cuad_entry: dict,
    annotations: Dict[str, List[dict]],
    template: dict,
    trap_type: str,
    seq: int,
) -> Optional[dict]:
    """Build one ground truth annotation from a template + CUAD annotations."""
    context = cuad_entry["paragraphs"][0]["context"]

    # Try primary clauses first, then fallback
    clauses_to_use = template["clauses_needed"]
    query = template["query_template"]

    available = [c for c in clauses_to_use if c in annotations]
    if len(available) < 2:
        # Try fallback
        fb_clauses = template.get("fallback_clauses", [])
        fb_available = [c for c in fb_clauses if c in annotations]
        if len(fb_available) >= 2:
            clauses_to_use = fb_clauses
            available = fb_available
            query = template.get("fallback_query", query)
        else:
            return None  # Can't build this query for this contract

    # Build ground truth answer from CUAD spans
    gt_parts = []
    info_units = []
    relevant_sections = []

    for clause_name in clauses_to_use:
        if clause_name not in annotations:
            continue
        ans_list = annotations[clause_name]
        # Take the first (primary) answer span
        ans = ans_list[0]
        span_text = ans["text"]

        # Truncate very long spans to 500 chars for ground truth
        if len(span_text) > 500:
            span_text = span_text[:500] + "..."

        section = find_section_for_offset(context, ans["answer_start"])
        relevant_sections.append(section)

        gt_parts.append(f"[{clause_name}]: {span_text}")
        info_units.append(f"{clause_name} clause content is present")

        # Add specific information units from the span
        # Look for key legal indicators in the span
        if re.search(r"notwithstanding|except|subject to|provided however", span_text, re.I):
            info_units.append(f"Exception or qualification in {clause_name} is identified")
        if re.search(r"\d+%|\$[\d,]+|\d+ days|\d+ months|\d+ years", span_text):
            info_units.append(f"Specific numeric terms in {clause_name} are stated")
        if re.search(r'"[A-Z][^"]*"', span_text):
            info_units.append(f"Defined term referenced in {clause_name} is included")

    if not gt_parts:
        return None

    ground_truth_answer = "\n\n".join(gt_parts)

    entry_id = f"{contract['short_id']}_{trap_type[-1]}_{seq:02d}"

    # Find the actual .txt filename
    title = cuad_entry["title"]
    # CUAD titles map to .txt filenames (with .txt extension)
    txt_filename = title + ".txt"

    return {
        "id": entry_id,
        "contract_file": txt_filename,
        "query": query,
        "trap_type": trap_type,
        "difficulty": "multi_hop",
        "ground_truth_answer": ground_truth_answer,
        "relevant_sections": relevant_sections,
        "information_units": info_units,
        "clauses_combined": [c for c in clauses_to_use if c in annotations],
        "notes": f"Auto-generated from CUADv1.json annotations. {contract['industry']} / {contract['agreement_type']}.",
    }


def generate_ground_truth(cuad_path: str, output_path: str) -> dict:
    """Main generation function."""
    print(f"Loading CUADv1.json from {cuad_path}...")
    with open(cuad_path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    print(f"Loaded {len(cuad['data'])} contracts.")

    # Build lookup by title prefix
    cuad_lookup = {}
    for entry in cuad["data"]:
        cuad_lookup[entry["title"]] = entry

    all_annotations = []
    stats = {"trap_a": 0, "trap_b": 0, "trap_c": 0, "total": 0}

    for contract in SELECTED_CONTRACTS:
        prefix = contract["cuad_title_prefix"]

        # Find matching CUAD entry
        matched_entry = None
        for title, entry in cuad_lookup.items():
            if title.lower().startswith(prefix.lower()):
                matched_entry = entry
                break

        if not matched_entry:
            print(f"  WARNING: No CUAD entry found for {prefix}")
            continue

        print(f"\nProcessing: {matched_entry['title'][:70]}...")
        annotations = get_contract_annotations(matched_entry)
        print(f"  Found {len(annotations)} answered clause types")

        # Generate 2 Trap A queries
        seq = 1
        for template in TRAP_A_TEMPLATES:
            entry = build_annotation_entry(
                contract, matched_entry, annotations, template, "trap_a", seq
            )
            if entry:
                all_annotations.append(entry)
                stats["trap_a"] += 1
                stats["total"] += 1
                print(f"  + {entry['id']}: {entry['query'][:60]}... ({len(entry['clauses_combined'])} clauses)")
                seq += 1

        # Generate 2 Trap B queries
        seq = 1
        for template in TRAP_B_TEMPLATES:
            entry = build_annotation_entry(
                contract, matched_entry, annotations, template, "trap_b", seq
            )
            if entry:
                all_annotations.append(entry)
                stats["trap_b"] += 1
                stats["total"] += 1
                print(f"  + {entry['id']}: {entry['query'][:60]}... ({len(entry['clauses_combined'])} clauses)")
                seq += 1

        # Generate 2 Trap C queries
        seq = 1
        for template in TRAP_C_TEMPLATES:
            entry = build_annotation_entry(
                contract, matched_entry, annotations, template, "trap_c", seq
            )
            if entry:
                all_annotations.append(entry)
                stats["trap_c"] += 1
                stats["total"] += 1
                print(f"  + {entry['id']}: {entry['query'][:60]}... ({len(entry['clauses_combined'])} clauses)")
                seq += 1

    # Build output
    output = {
        "_schema_version": "2.0",
        "_description": f"Ground truth from CUADv1.json professional annotations. {stats['total']} multi-hop queries across {len(SELECTED_CONTRACTS)} contracts.",
        "_trap_types": {
            "trap_a": "Invisible Exception — requires finding both a clause and its override/exception",
            "trap_b": "Distant Definition — requires finding a defined term AND the clause that uses it",
            "trap_c": "Scattered Components — requires assembling information from 3+ sections",
        },
        "_generation_method": "Automated from CUADv1.json exact answer spans. Each query combines 2-3 related CUAD clause types.",
        "_stats": stats,
        "annotations": all_annotations,
    }

    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generated {stats['total']} ground truth entries:")
    print(f"  Trap A (Invisible Exception): {stats['trap_a']}")
    print(f"  Trap B (Distant Definition):  {stats['trap_b']}")
    print(f"  Trap C (Scattered Components): {stats['trap_c']}")
    print(f"Output: {output_file}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth from CUAD annotations")
    parser.add_argument(
        "--cuad-json",
        default=r"C:\Users\suyas\Downloads\archive (1)\CUAD_v1\CUAD_v1.json",
        help="Path to CUADv1.json",
    )
    parser.add_argument(
        "--output",
        default="data/ground_truth_cuad_v2.json",
        help="Output path for ground truth JSON",
    )
    args = parser.parse_args()

    generate_ground_truth(args.cuad_json, args.output)
