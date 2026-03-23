"""Ground truth loader for V3 evaluation.

Loads human-annotated ground truth from data/ground_truth.json.
Used by the batch evaluation harness to compute RAGAS Context Recall
and the Information Units checklist metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


_GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"


def load_ground_truth(path: str | Path | None = None) -> List[Dict]:
    """Load ground truth annotations from JSON.
    
    Args:
        path: Path to ground_truth.json. Defaults to data/ground_truth.json.
    
    Returns:
        List of annotation dicts, each containing:
          - id: unique identifier
          - contract_file: filename of the CUAD contract
          - query: the evaluation query
          - trap_type: "trap_a", "trap_b", or "trap_c"
          - difficulty: "single_hop" or "multi_hop"
          - ground_truth_answer: the correct answer (human-written)
          - relevant_sections: list of section references
          - information_units: list of atomic facts the answer must contain
          - notes: optional notes
    
    Raises:
        FileNotFoundError: if ground_truth.json doesn't exist
        ValueError: if file has no valid annotations
    """
    gt_path = Path(path) if path else _GT_PATH
    
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {gt_path}\n"
            "Please create annotations following the instructions in data/ANNOTATION_INSTRUCTIONS.md"
        )
    
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", [])
    
    # Filter out template/placeholder entries
    valid = []
    for ann in annotations:
        # Skip entries that still have REPLACE placeholders
        if "REPLACE" in ann.get("ground_truth_answer", "REPLACE"):
            continue
        if "REPLACE" in ann.get("query", "REPLACE"):
            continue
        if "REPLACE" in ann.get("contract_file", "REPLACE"):
            continue
        valid.append(ann)
    
    if not valid:
        raise ValueError(
            f"No valid annotations found in {gt_path}. "
            "All entries still contain REPLACE placeholders. "
            "Please fill in the annotations following the instructions."
        )
    
    return valid


def get_annotations_by_trap(trap_type: str, path: str | Path | None = None) -> List[Dict]:
    """Get annotations filtered by trap type.
    
    Args:
        trap_type: "trap_a", "trap_b", or "trap_c"
        path: Optional path to ground_truth.json
    """
    all_gt = load_ground_truth(path)
    return [a for a in all_gt if a.get("trap_type") == trap_type]


def get_annotations_by_contract(contract_file: str, path: str | Path | None = None) -> List[Dict]:
    """Get annotations filtered by contract filename."""
    all_gt = load_ground_truth(path)
    return [a for a in all_gt if a.get("contract_file") == contract_file]


def compute_information_unit_coverage(answer: str, annotation: Dict) -> Dict:
    """Compute how many information units from ground truth are in the answer.
    
    This is the "checklist" metric: each information unit is a discrete fact
    that the answer should contain. The score is the fraction found.
    
    Args:
        answer: The generated answer text
        annotation: A ground truth annotation dict
    
    Returns:
        Dict with:
          - score: fraction of information units covered [0, 1]
          - total_units: total number of information units
          - covered_units: number found in the answer
          - missing_units: list of units NOT found
          - found_units: list of units found
    """
    info_units = annotation.get("information_units", [])
    
    if not info_units:
        return {"score": 1.0, "total_units": 0, "covered_units": 0,
                "missing_units": [], "found_units": []}
    
    answer_lower = answer.lower()
    found = []
    missing = []
    
    for unit in info_units:
        # Check if the key concepts from the information unit appear in the answer
        # We use keyword extraction rather than exact match
        unit_words = set(w.lower() for w in unit.split() if len(w) > 3)
        
        if not unit_words:
            found.append(unit)
            continue
        
        # Require at least 60% of significant words to match
        matches = sum(1 for w in unit_words if w in answer_lower)
        coverage = matches / len(unit_words) if unit_words else 0
        
        if coverage >= 0.6:
            found.append(unit)
        else:
            missing.append(unit)
    
    total = len(info_units)
    covered = len(found)
    
    return {
        "score": covered / total if total > 0 else 1.0,
        "total_units": total,
        "covered_units": covered,
        "missing_units": missing,
        "found_units": found,
    }


def validate_ground_truth(path: str | Path | None = None) -> Dict:
    """Validate the ground truth file and report statistics.
    
    Returns:
        Dict with validation results and statistics.
    """
    gt_path = Path(path) if path else _GT_PATH
    
    if not gt_path.exists():
        return {"valid": False, "error": "File not found"}
    
    try:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"valid": False, "error": f"Invalid JSON: {e}"}
    
    annotations = data.get("annotations", [])
    
    total = len(annotations)
    valid = 0
    by_trap = {"trap_a": 0, "trap_b": 0, "trap_c": 0}
    by_contract = {}
    issues = []
    
    for i, ann in enumerate(annotations):
        is_placeholder = (
            "REPLACE" in ann.get("ground_truth_answer", "REPLACE") or
            "REPLACE" in ann.get("query", "REPLACE") or
            "REPLACE" in ann.get("contract_file", "REPLACE")
        )
        
        if is_placeholder:
            continue
        
        valid += 1
        trap = ann.get("trap_type", "unknown")
        by_trap[trap] = by_trap.get(trap, 0) + 1
        
        contract = ann.get("contract_file", "unknown")
        by_contract[contract] = by_contract.get(contract, 0) + 1
        
        # Check quality
        if len(ann.get("ground_truth_answer", "")) < 50:
            issues.append(f"Entry {ann.get('id', i)}: Ground truth answer is very short")
        if not ann.get("information_units"):
            issues.append(f"Entry {ann.get('id', i)}: No information units defined")
        if not ann.get("relevant_sections"):
            issues.append(f"Entry {ann.get('id', i)}: No relevant sections listed")
    
    return {
        "valid": valid > 0,
        "total_entries": total,
        "valid_entries": valid,
        "placeholder_entries": total - valid,
        "by_trap_type": by_trap,
        "by_contract": by_contract,
        "n_contracts": len(by_contract),
        "issues": issues,
        "recommendation": (
            f"You have {valid} annotations. "
            f"Minimum recommended: 30 (for statistical significance). "
            f"Target: 48 (8 contracts × 6 queries)."
        ),
    }


if __name__ == "__main__":
    """Quick validation when run directly."""
    result = validate_ground_truth()
    print("=== Ground Truth Validation ===")
    for key, value in result.items():
        if key == "issues":
            print(f"\nIssues ({len(value)}):")
            for issue in value[:10]:
                print(f"  - {issue}")
        elif key == "by_contract":
            print(f"\nContracts: {len(value)}")
            for contract, count in value.items():
                print(f"  - {contract}: {count} queries")
        else:
            print(f"{key}: {value}")
