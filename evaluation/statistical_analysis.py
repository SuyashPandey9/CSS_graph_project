"""Statistical analysis for V3 vs Traditional RAG evaluation results.

Computes paired statistical tests, effect sizes, and confidence intervals
from batch evaluation results. Designed for research paper reporting.

Usage:
    python -m evaluation.statistical_analysis results/eval_results_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================================
# CORE STATISTICAL FUNCTIONS (no external dependencies)
# ============================================================================

def _mean(values: List[float]) -> float:
    """Compute mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float], ddof: int = 1) -> float:
    """Compute standard deviation (sample by default)."""
    if len(values) <= ddof:
        return 0.0
    m = _mean(values)
    ss = sum((x - m) ** 2 for x in values)
    return math.sqrt(ss / (len(values) - ddof))


def _t_cdf_approx(t_stat: float, df: int) -> float:
    """Approximate two-tailed p-value for t-distribution.
    
    Uses the approximation from Abramowitz & Stegun (1964) for
    the cumulative distribution function of the t-distribution.
    Accurate to ~4 decimal places for df >= 5.
    """
    if df <= 0:
        return 1.0
    
    # Convert t to approximate normal z for large df
    x = abs(t_stat)
    
    # Approximation using Beta distribution relation
    a = df / 2.0
    b = 0.5
    x2 = x * x
    p_val = df / (df + x2)
    
    # Regularized incomplete beta function approximation
    # Using continued fraction expansion for Ix(a, b)
    if p_val < (a + 1) / (a + b + 2):
        beta_val = _beta_cf(p_val, a, b)
    else:
        beta_val = 1.0 - _beta_cf(1.0 - p_val, b, a)
    
    # Two-tailed p-value
    return beta_val


def _beta_cf(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Evaluate regularized incomplete beta function using continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    
    # Compute the log of the prefix
    try:
        ln_prefix = (
            a * math.log(x) + b * math.log(1 - x)
            - math.log(a)
            - _log_beta(a, b)
        )
        prefix = math.exp(ln_prefix)
    except (ValueError, OverflowError):
        return 0.5  # Fallback
    
    # Lentz's continued fraction method
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    
    for m in range(1, max_iter + 1):
        # Even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c
        
        # Odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = d * c
        f *= delta
        
        if abs(delta - 1.0) < 1e-8:
            break
    
    return prefix * f


def _log_beta(a: float, b: float) -> float:
    """Compute log of Beta function: log(B(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def paired_t_test(v3_values: List[float], trag_values: List[float]) -> Dict:
    """Paired t-test for comparing V3 vs TRAG on the same queries.
    
    H0: There is no difference between V3 and TRAG means.
    H1: V3 and TRAG means are different (two-tailed).
    
    Returns:
        Dict with t_statistic, p_value, df, significant (at α=0.05)
    """
    n = len(v3_values)
    if n != len(trag_values) or n < 2:
        return {"error": "Need paired data with n >= 2", "significant": False}
    
    # Compute differences
    diffs = [v3 - trag for v3, trag in zip(v3_values, trag_values)]
    
    d_mean = _mean(diffs)
    d_std = _std(diffs)
    
    if d_std == 0:
        return {
            "t_statistic": float("inf") if d_mean != 0 else 0.0,
            "p_value": 0.0 if d_mean != 0 else 1.0,
            "df": n - 1,
            "mean_diff": d_mean,
            "significant": d_mean != 0,
        }
    
    t_stat = d_mean / (d_std / math.sqrt(n))
    df = n - 1
    p_value = _t_cdf_approx(t_stat, df)
    
    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "df": df,
        "mean_diff": round(d_mean, 4),
        "std_diff": round(d_std, 4),
        "significant": p_value < 0.05,
    }


def wilcoxon_signed_rank(v3_values: List[float], trag_values: List[float]) -> Dict:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Use this when data may not be normally distributed (common with small N).
    
    Returns:
        Dict with W_statistic, approximate p_value, significant
    """
    n = len(v3_values)
    if n != len(trag_values) or n < 5:
        return {"error": "Need paired data with n >= 5", "significant": False}
    
    # Compute differences, remove zeros
    diffs = [(v3 - trag, i) for i, (v3, trag) in enumerate(zip(v3_values, trag_values))]
    nonzero_diffs = [(d, i) for d, i in diffs if abs(d) > 1e-10]
    
    if len(nonzero_diffs) < 3:
        return {"W_statistic": 0, "p_value": 1.0, "significant": False,
                "note": "Too few non-zero differences"}
    
    nr = len(nonzero_diffs)
    
    # Rank by absolute difference
    ranked = sorted(nonzero_diffs, key=lambda x: abs(x[0]))
    
    # Assign ranks (handle ties by averaging)
    ranks = []
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and abs(abs(ranked[j][0]) - abs(ranked[i][0])) < 1e-10:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks.append((ranked[k][0], avg_rank))
        i = j
    
    # Sum of positive ranks and negative ranks
    w_plus = sum(rank for diff, rank in ranks if diff > 0)
    w_minus = sum(rank for diff, rank in ranks if diff < 0)
    
    W = min(w_plus, w_minus)
    
    # Normal approximation for p-value (valid for n >= 10, approximate for smaller)
    mean_w = nr * (nr + 1) / 4.0
    
    # Correction for ties
    std_w = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0)
    
    if std_w == 0:
        return {"W_statistic": W, "p_value": 1.0, "significant": False}
    
    z = (W - mean_w) / std_w
    
    # Approximate p-value using normal distribution
    p_value = 2.0 * _normal_cdf(-abs(z))
    
    return {
        "W_statistic": round(W, 4),
        "W_plus": round(w_plus, 4),
        "W_minus": round(w_minus, 4),
        "z_approx": round(z, 4),
        "p_value": round(p_value, 6),
        "n_nonzero": nr,
        "significant": p_value < 0.05,
    }


def _normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cohens_d(v3_values: List[float], trag_values: List[float]) -> Dict:
    """Compute Cohen's d effect size for paired data.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    """
    n = len(v3_values)
    if n != len(trag_values) or n < 2:
        return {"error": "Need paired data with n >= 2"}
    
    diffs = [v3 - trag for v3, trag in zip(v3_values, trag_values)]
    d_mean = _mean(diffs)
    d_std = _std(diffs)
    
    if d_std == 0:
        d = float("inf") if d_mean != 0 else 0.0
    else:
        d = d_mean / d_std
    
    # Interpretation
    abs_d = abs(d) if d != float("inf") else float("inf")
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"
    
    return {
        "cohens_d": round(d, 4) if d != float("inf") else "inf",
        "interpretation": interpretation,
        "mean_diff": round(d_mean, 4),
        "pooled_std": round(d_std, 4),
    }


def confidence_interval(v3_values: List[float], trag_values: List[float],
                        confidence: float = 0.95) -> Dict:
    """Compute confidence interval for the mean difference (V3 - TRAG).
    
    Uses t-distribution critical values.
    """
    n = len(v3_values)
    if n != len(trag_values) or n < 2:
        return {"error": "Need paired data with n >= 2"}
    
    diffs = [v3 - trag for v3, trag in zip(v3_values, trag_values)]
    d_mean = _mean(diffs)
    d_std = _std(diffs)
    
    # Approximate t critical value for common confidence levels
    df = n - 1
    alpha = 1 - confidence
    
    # t critical values (two-tailed) - approximation for common levels
    if confidence == 0.95:
        if df >= 120:
            t_crit = 1.96
        elif df >= 60:
            t_crit = 2.00
        elif df >= 30:
            t_crit = 2.04
        elif df >= 20:
            t_crit = 2.086
        elif df >= 10:
            t_crit = 2.228
        elif df >= 5:
            t_crit = 2.571
        else:
            t_crit = 2.776
    elif confidence == 0.99:
        if df >= 120:
            t_crit = 2.576
        elif df >= 30:
            t_crit = 2.750
        else:
            t_crit = 3.250
    else:
        t_crit = 2.0  # Rough fallback

    se = d_std / math.sqrt(n) if n > 0 else 0
    margin = t_crit * se
    
    return {
        "mean_diff": round(d_mean, 4),
        "ci_lower": round(d_mean - margin, 4),
        "ci_upper": round(d_mean + margin, 4),
        "confidence": confidence,
        "standard_error": round(se, 4),
        "n": n,
    }


def bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict:
    """Apply Bonferroni correction for multiple comparisons.
    
    When testing multiple metrics, the chance of at least one false positive
    increases. Bonferroni corrects this by dividing α by the number of tests.
    """
    n_tests = len(p_values)
    adjusted_alpha = alpha / n_tests if n_tests > 0 else alpha
    
    results = {}
    for metric, p in p_values.items():
        adjusted_p = min(p * n_tests, 1.0)  # Adjusted p-value
        results[metric] = {
            "original_p": round(p, 6),
            "adjusted_p": round(adjusted_p, 6),
            "significant_original": p < alpha,
            "significant_corrected": adjusted_p < alpha,
        }
    
    return {
        "n_tests": n_tests,
        "original_alpha": alpha,
        "corrected_alpha": round(adjusted_alpha, 6),
        "metrics": results,
    }


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

def analyze_results(results_path: str) -> Dict:
    """Run full statistical analysis on batch evaluation results.
    
    Args:
        results_path: Path to eval_results_*.json from batch_evaluation.py
    
    Returns:
        Comprehensive analysis dict ready for paper reporting.
    """
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    results = data.get("results", [])
    
    # Filter out errored entries
    valid = [r for r in results if "error" not in r]
    
    if len(valid) < 5:
        return {"error": f"Only {len(valid)} valid results. Need at least 5 for analysis."}
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS")
    print(f"{'='*70}")
    print(f"Valid results: {len(valid)}")
    
    # Key metrics to analyze
    key_metrics = [
        "ragas_context_recall",
        "ragas_faithfulness",
        "ragas_context_precision",
        "ragas_answer_relevancy",
        "ragas_aspect_coverage",
        "info_unit_coverage",
        "latency_seconds",
        "total_tokens",
    ]
    
    analysis: dict = {
        "n_samples": len(valid),
        "metrics": {},
        "by_trap_type": {},
    }
    
    p_values_for_bonferroni: dict[str, float] = {}
    
    # Overall analysis for each metric
    print(f"\n{'Metric':35s}  {'V3 M':>7s}  {'TRAG M':>7s}  {'t':>7s}  {'p':>8s}  {'d':>6s}  {'Sig?':>5s}")
    print(f"{'-'*80}")
    
    for metric in key_metrics:
        v3_vals = [r["v3"]["metrics"].get(metric, 0.0) for r in valid]
        trag_vals = [r["trag"]["metrics"].get(metric, 0.0) for r in valid]
        
        # Skip metrics where all values are 0 or -1 (not computed)
        if all(v == 0.0 or v == -1.0 for v in v3_vals + trag_vals):
            continue
        
        t_result = paired_t_test(v3_vals, trag_vals)
        w_result = wilcoxon_signed_rank(v3_vals, trag_vals)
        d_result = cohens_d(v3_vals, trag_vals)
        ci_result = confidence_interval(v3_vals, trag_vals)
        
        v3_mean = _mean(v3_vals)
        trag_mean = _mean(trag_vals)
        
        sig = "YES" if t_result.get("significant", False) else "no"
        d_val = d_result.get("cohens_d", "?")
        t_val = t_result.get("t_statistic", "?")
        p_val = t_result.get("p_value", 1.0)
        
        t_str = f"{t_val:7.3f}" if isinstance(t_val, (int, float)) else f"{t_val:>7s}"
        d_str = f"{d_val:6.3f}" if isinstance(d_val, (int, float)) else f"{d_val:>6s}"
        p_str = f"{p_val:8.4f}" if isinstance(p_val, (int, float)) else f"{p_val:>8s}"
        
        print(f"{metric:35s}  {v3_mean:7.3f}  {trag_mean:7.3f}  {t_str}  {p_str}  {d_str}  {sig:>5s}")
        
        analysis["metrics"][metric] = {
            "v3_mean": round(v3_mean, 4),
            "v3_std": round(_std(v3_vals), 4),
            "trag_mean": round(trag_mean, 4),
            "trag_std": round(_std(trag_vals), 4),
            "paired_t_test": t_result,
            "wilcoxon": w_result,
            "cohens_d": d_result,
            "confidence_interval_95": ci_result,
        }
        
        if isinstance(p_val, (int, float)):
            p_values_for_bonferroni[metric] = p_val
    
    # Bonferroni correction
    if p_values_for_bonferroni:
        bonferroni = bonferroni_correction(p_values_for_bonferroni)
        analysis["bonferroni_correction"] = bonferroni
        
        print(f"\n--- Bonferroni Correction ({bonferroni['n_tests']} tests, alpha={bonferroni['corrected_alpha']:.4f}) ---")
        for metric, info in bonferroni["metrics"].items():
            status = "SIGNIFICANT" if info["significant_corrected"] else "not significant"
            print(f"  {metric:35s}  p_adj={info['adjusted_p']:.4f}  [{status}]")
    
    # Per trap-type analysis
    for trap in ["trap_a", "trap_b", "trap_c"]:
        trap_results = [r for r in valid if r.get("trap_type") == trap]
        if len(trap_results) < 3:
            continue
        
        trap_analysis = {"n_samples": len(trap_results), "metrics": {}}
        
        print(f"\n--- {trap.upper()} ({len(trap_results)} samples) ---")
        
        for metric in ["ragas_context_recall", "ragas_faithfulness", "info_unit_coverage"]:
            v3_vals = [r["v3"]["metrics"].get(metric, 0.0) for r in trap_results]
            trag_vals = [r["trag"]["metrics"].get(metric, 0.0) for r in trap_results]
            
            t_result = paired_t_test(v3_vals, trag_vals)
            d_result = cohens_d(v3_vals, trag_vals)
            
            v3_mean = _mean(v3_vals)
            trag_mean = _mean(trag_vals)
            sig = "YES" if t_result.get("significant", False) else "no"
            
            print(f"  {metric:30s}  V3={v3_mean:.3f}  TRAG={trag_mean:.3f}  d={d_result.get('cohens_d', '?')}  [{sig}]")
            
            trap_analysis["metrics"][metric] = {
                "v3_mean": round(v3_mean, 4),
                "trag_mean": round(trag_mean, 4),
                "paired_t_test": t_result,
                "cohens_d": d_result,
            }
        
        analysis["by_trap_type"][trap] = trap_analysis
    
    # Save analysis
    out_path = Path(results_path).parent / f"statistical_analysis_{Path(results_path).stem.replace('eval_results_', '')}.json"
    out_path.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    print(f"\nAnalysis saved to: {out_path}")
    
    return analysis


def generate_latex_table(analysis: Dict) -> str:
    """Generate a LaTeX table from the analysis results for paper inclusion."""
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{V3 vs Traditional RAG: Paired Comparison Results}",
        r"\label{tab:v3_vs_trag}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Metric & V3 ($\mu$) & TRAG ($\mu$) & $\Delta$ & $t$ & $p$ & Cohen's $d$ \\",
        r"\midrule",
    ]
    
    metrics = analysis.get("metrics", {})
    for metric_name, data in metrics.items():
        v3_m = data.get("v3_mean", 0)
        trag_m = data.get("trag_mean", 0)
        delta = v3_m - trag_m
        t_stat = data.get("paired_t_test", {}).get("t_statistic", "—")
        p_val = data.get("paired_t_test", {}).get("p_value", "—")
        d_val = data.get("cohens_d", {}).get("cohens_d", "—")
        
        # Format metric name for LaTeX
        name = metric_name.replace("_", r"\_")
        
        # Bold if significant
        sig = data.get("paired_t_test", {}).get("significant", False)
        if sig and isinstance(p_val, (int, float)):
            p_str = f"\\textbf{{{p_val:.4f}}}"
        elif isinstance(p_val, (int, float)):
            p_str = f"{p_val:.4f}"
        else:
            p_str = str(p_val)
        
        lines.append(
            f"{name} & {v3_m:.3f} & {trag_m:.3f} & {delta:+.3f} & "
            f"{t_stat} & {p_str} & {d_val} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\\\\\footnotesize{{$n = {analysis.get('n_samples', '?')}$ paired observations. "
        r"Bonferroni-corrected significance at $\alpha = 0.05$.}}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis of V3 vs TRAG results")
    parser.add_argument("results_file", type=str,
                        help="Path to eval_results_*.json from batch evaluation")
    parser.add_argument("--latex", action="store_true",
                        help="Also generate LaTeX table")
    args = parser.parse_args()
    
    analysis = analyze_results(args.results_file)
    
    if args.latex and "error" not in analysis:
        latex = generate_latex_table(analysis)
        print(f"\n{'='*70}")
        print("LATEX TABLE:")
        print(f"{'='*70}")
        print(latex)
        
        # Save to file
        latex_path = Path(args.results_file).parent / "results_table.tex"
        latex_path.write_text(latex, encoding="utf-8")
        print(f"\nSaved to: {latex_path}")


if __name__ == "__main__":
    main()
