#!/usr/bin/env python3
"""
Evaluation Script for Hybrid Pattern + AI Scanner
===================================================
Runs the full test suite and produces an honest evaluation report.

Metrics:
    - Pattern Engine: Recall per CWE, False Positive Rate
    - AI Refinement: Precision improvement, FP reduction
    - System-Level: End-to-end P/R/F1 per CWE
    
Usage:
    python evaluate_hybrid.py
    python evaluate_hybrid.py --no-ai
    python evaluate_hybrid.py --json
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

# Ensure backend is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.hybrid_scanner.pattern_engine import PatternMatchingEngine
from app.hybrid_scanner.models import VulnLabel, Verdict
from app.hybrid_scanner.test_suite import ALL_TEST_CASES, TestCase


@dataclass
class CWEMetrics:
    cwe: str
    total_vulnerable: int = 0
    total_safe: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    false_positives: int = 0
    true_negatives: int = 0

    @property
    def recall(self) -> float:
        if self.total_vulnerable == 0:
            return 0.0
        return self.true_positives / self.total_vulnerable

    @property
    def precision(self) -> float:
        detected = self.true_positives + self.false_positives
        if detected == 0:
            return 1.0
        return self.true_positives / detected

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def fpr(self) -> float:
        if self.total_safe == 0:
            return 0.0
        return self.false_positives / self.total_safe


def evaluate_pattern_engine() -> Dict[str, CWEMetrics]:
    """
    Evaluate the pattern matching engine (Phase 1 only).
    
    For each test case:
        - Vulnerable code should produce at least one finding with the correct CWE
        - Safe code should produce NO finding with that CWE
    """
    engine = PatternMatchingEngine()
    metrics: Dict[str, CWEMetrics] = {}

    for cwe, tests in ALL_TEST_CASES.items():
        m = CWEMetrics(cwe=cwe)

        for tc in tests:
            findings = engine.scan(
                code=tc.code,
                language=tc.language,
                filename=f"{tc.id}.{'py' if tc.language == 'python' else 'js'}"
            )

            # Check if any finding matches the expected CWE
            matched_cwe = any(f.cwe.value == cwe for f in findings)

            if tc.is_vulnerable:
                m.total_vulnerable += 1
                if matched_cwe:
                    m.true_positives += 1
                else:
                    m.false_negatives += 1
            else:
                m.total_safe += 1
                if matched_cwe:
                    m.false_positives += 1
                else:
                    m.true_negatives += 1

        metrics[cwe] = m

    return metrics


def print_evaluation_report(metrics: Dict[str, CWEMetrics]):
    """Print a detailed evaluation report to stdout."""
    print("=" * 80)
    print("HYBRID SCANNER: PATTERN ENGINE EVALUATION REPORT")
    print("=" * 80)
    print()

    # Rule counts
    engine = PatternMatchingEngine()
    rule_counts = engine.get_rule_count()
    total_rules = sum(rule_counts.values())
    print(f"Pattern rules loaded: {total_rules}")
    for cwe, count in sorted(rule_counts.items()):
        print(f"  {cwe}: {count} rules")
    print()

    # Per-CWE results
    print(f"{'CWE':<10} {'Vuln':<6} {'Safe':<6} {'TP':<5} {'FN':<5} {'FP':<5} {'TN':<5} "
          f"{'Recall':<8} {'Prec':<8} {'F1':<8} {'FPR':<8} {'Verdict'}")
    print("-" * 100)

    total_tp = 0
    total_fn = 0
    total_fp = 0
    total_tn = 0
    pass_count = 0

    for cwe in sorted(metrics.keys()):
        m = metrics[cwe]
        total_tp += m.true_positives
        total_fn += m.false_negatives
        total_fp += m.false_positives
        total_tn += m.true_negatives

        verdict = "PASS" if m.recall >= 0.95 else ("PARTIAL" if m.recall >= 0.75 else "FAIL")
        if verdict == "PASS":
            pass_count += 1

        print(f"{cwe:<10} {m.total_vulnerable:<6} {m.total_safe:<6} "
              f"{m.true_positives:<5} {m.false_negatives:<5} "
              f"{m.false_positives:<5} {m.true_negatives:<5} "
              f"{m.recall:<8.1%} {m.precision:<8.1%} {m.f1:<8.1%} "
              f"{m.fpr:<8.1%} {verdict}")

    # Overall
    print("-" * 100)
    total_vuln = total_tp + total_fn
    total_safe = total_fp + total_tn
    recall_all = total_tp / total_vuln if total_vuln else 0
    precision_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    f1_all = 2 * precision_all * recall_all / (precision_all + recall_all) if (precision_all + recall_all) else 0
    fpr_all = total_fp / total_safe if total_safe else 0

    print(f"{'TOTAL':<10} {total_vuln:<6} {total_safe:<6} "
          f"{total_tp:<5} {total_fn:<5} {total_fp:<5} {total_tn:<5} "
          f"{recall_all:<8.1%} {precision_all:<8.1%} {f1_all:<8.1%} {fpr_all:<8.1%}")

    # Detailed failure analysis
    print()
    print("=" * 80)
    print("FAILURE ANALYSIS")
    print("=" * 80)

    engine = PatternMatchingEngine()
    has_failures = False

    for cwe, tests in sorted(ALL_TEST_CASES.items()):
        m = metrics[cwe]
        if m.false_negatives == 0 and m.false_positives == 0:
            continue

        has_failures = True
        print(f"\n--- {cwe} ---")

        if m.false_negatives > 0:
            print(f"  FALSE NEGATIVES (missed vulnerabilities):")
            for tc in tests:
                if not tc.is_vulnerable:
                    continue
                findings = engine.scan(tc.code, tc.language, f"{tc.id}.test")
                cwe_matched = any(f.cwe.value == cwe for f in findings)
                if not cwe_matched:
                    print(f"    [MISS] {tc.id}: {tc.description}")
                    # Show what WAS detected
                    if findings:
                        for ff in findings:
                            print(f"           Found: {ff.cwe.value} at line {ff.line} ({ff.rule_id})")
                    else:
                        print(f"           No findings at all")

        if m.false_positives > 0:
            print(f"  FALSE POSITIVES (safe code flagged):")
            for tc in tests:
                if tc.is_vulnerable:
                    continue
                findings = engine.scan(tc.code, tc.language, f"{tc.id}.test")
                cwe_matched = any(f.cwe.value == cwe for f in findings)
                if cwe_matched:
                    print(f"    [FP] {tc.id}: {tc.description}")
                    for ff in findings:
                        if ff.cwe.value == cwe:
                            print(f"         Rule: {ff.rule_id} at line {ff.line}")

    if not has_failures:
        print("  No failures to analyze!")

    # Final verdict
    print()
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"  CWEs passing (recall >= 95%): {pass_count}/{len(metrics)}")
    print(f"  Overall recall: {recall_all:.1%}")
    print(f"  Overall precision: {precision_all:.1%}")
    print(f"  Overall F1: {f1_all:.1%}")
    print()

    if recall_all >= 0.95:
        print("  [PASS] Pattern engine meets recall target (>= 95%)")
    elif recall_all >= 0.80:
        print("  [PARTIAL] Pattern engine close to target but needs improvement")
    else:
        print("  [FAIL] Pattern engine needs significant rule improvements")

    print()
    print("  NOTE: False positives are EXPECTED and ACCEPTABLE in Phase 1.")
    print("  The AI refinement layer (Phase 2) reduces false positives.")
    print("  Pattern engine's primary job is HIGH RECALL.")
    print()

    # Honest limitations
    print("=" * 80)
    print("HONEST LIMITATIONS")
    print("=" * 80)
    print("""
  1. Pattern matching is REGEX-BASED, not semantic analysis.
     It cannot understand data flow across function boundaries.

  2. The AI model was trained on synthetic data and currently shows
     0% detection on realistic OWASP test cases (see previous evaluation).
     AI refinement is ADVISORY and may not improve precision in practice.

  3. TypeScript transpilation uses regex stripping, not a full compiler.
     Complex TS patterns may not be fully processed.

  4. This scanner covers 6 CWEs, NOT the full OWASP Top 10.
     It explicitly does NOT cover:
     - Access control (requires auth context)
     - Authentication logic (requires flow analysis)
     - Dependency vulnerabilities (requires SCA tooling)
     - Business logic flaws (requires domain knowledge)

  5. Cross-file data flow is NOT supported.
     Taint propagation across modules/files is out of scope.
""")

    return {
        "total_rules": total_rules,
        "per_cwe": {cwe: {
            "recall": m.recall, "precision": m.precision, "f1": m.f1,
            "tp": m.true_positives, "fn": m.false_negatives,
            "fp": m.false_positives, "tn": m.true_negatives,
        } for cwe, m in metrics.items()},
        "overall": {
            "recall": recall_all, "precision": precision_all, "f1": f1_all,
            "fpr": fpr_all,
        },
        "pass_count": pass_count,
        "total_cwes": len(metrics),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="JSON output only")
    args = parser.parse_args()

    t0 = time.time()
    metrics = evaluate_pattern_engine()
    elapsed = time.time() - t0

    if args.json:
        result = {}
        for cwe, m in metrics.items():
            result[cwe] = {
                "recall": m.recall, "precision": m.precision, "f1": m.f1,
                "tp": m.true_positives, "fn": m.false_negatives,
                "fp": m.false_positives, "tn": m.true_negatives,
            }
        print(json.dumps(result, indent=2))
    else:
        report_data = print_evaluation_report(metrics)
        print(f"\nEvaluation completed in {elapsed:.2f}s")
