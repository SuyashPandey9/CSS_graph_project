"""Batch 4 verification: Domain-specific fixes."""

import sys
import yaml


def test_legal_config():
    """Config.yaml has legal domain settings."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    assert config.get("domain") == "legal", f"Domain should be 'legal', got {config.get('domain')}"
    assert "legal" in config, "Missing 'legal' section"
    
    legal = config["legal"]
    assert legal["css_weights"]["context_cohesion"] == 0.5
    assert legal["css_weights"]["clause_coverage"] == 1.0
    assert legal["css_weights"]["answer_specificity"] == 2.0
    assert legal["token_budget"] == 2000
    assert legal["redundancy_threshold"] == 0.95
    
    print(f"  Legal CSS overrides: {legal['css_weights']}")
    print(f"  Redundancy threshold: {legal['redundancy_threshold']}")
    print("PASS: Legal domain config")


def test_legal_features_cleaned():
    """legal_entity_density and cross_reference_density removed from registry."""
    from learning.legal_features import LEGAL_FEATURES
    
    assert "legal_entity_density" not in LEGAL_FEATURES, "legal_entity_density should be removed"
    assert "cross_reference_density" not in LEGAL_FEATURES, "cross_reference_density should be removed"
    assert "clause_coverage" in LEGAL_FEATURES, "clause_coverage should be kept"
    assert "section_diversity" in LEGAL_FEATURES, "section_diversity should be kept"
    assert "answer_specificity" in LEGAL_FEATURES, "answer_specificity should be kept"
    
    print(f"  LEGAL_FEATURES keys: {list(LEGAL_FEATURES.keys())}")
    print("PASS: Legal features cleaned up")


def test_legal_features_compute():
    """Legal features still compute correctly."""
    from learning.legal_features import compute_all_legal_features
    from core.types import Graph, Node
    
    g = Graph(nodes=[
        Node(id="a", text="The Company shall indemnify the Licensee for damages not exceeding $5,000,000 within 30 days."),
        Node(id="b", text="Termination may occur upon thirty (30) days written notice to the other party."),
    ])
    
    features = compute_all_legal_features(g, "What is the indemnification liability cap?")
    
    assert "clause_coverage" in features
    assert "section_diversity" in features
    assert "answer_specificity" in features
    
    for name, val in features.items():
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"
        print(f"    {name}: {val:.3f}")
    
    print("PASS: Legal features compute correctly")


def test_contradiction_flagging():
    """Fix 2.11: Contradiction flagging as metadata annotation."""
    from tools.contradiction_flagger import detect_contradiction_flags, build_contradiction_annotation
    from core.types import Graph, Node
    
    g = Graph(nodes=[
        Node(id="sec8", text="The Company shall indemnify the Licensee against all losses."),
        Node(id="sec8_4", text="Notwithstanding the foregoing, aggregate liability shall not exceed $5,000,000."),
        Node(id="exhibitB", text="Excluding claims arising from willful misconduct, subject to Section 8.4."),
    ])
    
    flags = detect_contradiction_flags(g)
    assert len(flags) > 0, "Should detect contradiction indicators"
    
    annotation = build_contradiction_annotation(g)
    assert "POTENTIAL CONFLICT" in annotation
    assert "notwithstanding" in annotation.lower() or "subject to" in annotation.lower() or "excluding" in annotation.lower()
    
    print(f"  Detected {len(flags)} flags")
    print(f"  Annotation preview: {annotation[:200]}...")
    print("PASS: Fix 2.11 - Contradiction flagging")


def test_no_contradiction_when_clean():
    """Contradiction flagger returns empty for clean text."""
    from tools.contradiction_flagger import build_contradiction_annotation
    from core.types import Graph, Node
    
    g = Graph(nodes=[
        Node(id="a", text="The Company shall pay all fees on time."),
        Node(id="b", text="Licensee agrees to the terms."),
    ])
    
    annotation = build_contradiction_annotation(g)
    assert annotation == "", f"Should return empty for clean text, got: {annotation}"
    print("PASS: No false contradiction flags")


if __name__ == "__main__":
    tests = [
        test_legal_config,
        test_legal_features_cleaned,
        test_legal_features_compute,
        test_contradiction_flagging,
        test_no_contradiction_when_clean,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\n--- {test.__name__} ---")
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n=== BATCH 4 RESULTS: {passed} passed, {failed} failed ===")
    sys.exit(1 if failed > 0 else 0)
