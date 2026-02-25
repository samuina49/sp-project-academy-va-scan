#!/usr/bin/env python3
"""
CLI for the Hybrid Pattern + AI Vulnerability Scanner
======================================================
Usage:
    python -m app.hybrid_scanner.cli scan --file path/to/code.py
    python -m app.hybrid_scanner.cli scan --code "import os; os.system(cmd)"
    python -m app.hybrid_scanner.cli scan --dir ./src
    python -m app.hybrid_scanner.cli status
    python -m app.hybrid_scanner.cli rules

Or from backend directory:
    python hybrid_scan_cli.py scan --file app.py
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure backend is on path
_backend_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_backend_root))


def cmd_scan(args):
    """Run a vulnerability scan."""
    from app.hybrid_scanner.pipeline import HybridPipeline

    pipeline = HybridPipeline(
        ai_enabled=args.ai,
        threshold=args.threshold,
        per_snippet_refinement=args.per_snippet,
    )

    results = []

    if args.file:
        result = pipeline.scan_file(args.file)
        results.append(result)
    elif args.dir:
        results = pipeline.scan_directory(args.dir, recursive=not args.no_recursive)
    elif args.code:
        result = pipeline.scan_code(
            code=args.code,
            language=args.language,
            filename="<stdin>",
        )
        results.append(result)
    else:
        # Read from stdin
        code = sys.stdin.read()
        result = pipeline.scan_code(
            code=code,
            language=args.language or "python",
            filename="<stdin>",
        )
        results.append(result)

    # Output
    if args.json:
        output = [r.summary for r in results]
        print(json.dumps(output, indent=2, default=str))
    else:
        _print_results(results)


def cmd_status(args):
    """Show pipeline status."""
    from app.hybrid_scanner.pipeline import HybridPipeline

    pipeline = HybridPipeline(ai_enabled=True)
    status = pipeline.status
    print(json.dumps(status, indent=2, default=str))


def cmd_rules(args):
    """List all pattern rules."""
    from app.hybrid_scanner.pattern_engine import RULES

    print(f"Total rules: {len(RULES)}")
    print(f"{'='*80}")

    # Group by CWE
    by_cwe = {}
    for rule in RULES:
        key = rule.cwe.value
        if key not in by_cwe:
            by_cwe[key] = []
        by_cwe[key].append(rule)

    for cwe in sorted(by_cwe.keys()):
        rules = by_cwe[cwe]
        print(f"\n{cwe} ({len(rules)} rules)")
        print(f"{'-'*60}")
        for r in rules:
            langs = ", ".join(r.languages)
            negs = f" [has {len(r.negative_patterns)} negative patterns]" if r.negative_patterns else ""
            print(f"  {r.rule_id:30s} [{r.severity.value:8s}] ({langs}){negs}")
            print(f"    {r.title}")


def _print_results(results):
    """Pretty-print scan results to terminal."""
    from app.hybrid_scanner.models import Verdict, Severity

    total_files = len(results)
    total_findings = 0
    total_confirmed = 0
    total_fps = 0

    for result in results:
        if result.errors:
            for err in result.errors:
                print(f"  WARNING: {err}")

        # Only show files with findings
        active_findings = [
            rf for rf in result.refined_findings
            if rf.verdict in (Verdict.VULNERABLE, Verdict.LIKELY_VULNERABLE)
        ]

        if not active_findings:
            continue

        print(f"\n{'='*80}")
        print(f"FILE: {result.file}")
        print(f"Language: {result.original_language} | "
              f"Candidates: {result.total_candidates} | "
              f"Confirmed: {result.confirmed_vulns} | "
              f"FP filtered: {result.false_positives} | "
              f"AI: {'ON' if result.ai_available else 'OFF'}")
        print(f"{'='*80}")

        for rf in active_findings:
            pf = rf.pattern_finding
            sev_indicator = {
                Severity.CRITICAL: "[!!]",
                Severity.HIGH: "[! ]",
                Severity.MEDIUM: "[- ]",
                Severity.LOW: "[  ]",
                Severity.INFO: "[  ]",
            }.get(pf.severity, "[  ]")

            verdict_str = {
                Verdict.VULNERABLE: "CONFIRMED",
                Verdict.LIKELY_VULNERABLE: "LIKELY",
                Verdict.FALSE_POSITIVE: "FP",
                Verdict.SAFE: "SAFE",
            }.get(rf.verdict, "?")

            ai_str = f"AI={rf.ai_score:.2f}" if rf.ai_available else "AI=N/A"

            print(f"\n  {sev_indicator} Line {pf.line}: {pf.message}")
            print(f"      CWE: {pf.cwe.value} | Rule: {pf.rule_id} "
                  f"| {pf.severity.value} | {verdict_str} | {ai_str}")
            print(f"      {pf.explanation}")
            print(f"      Code: {pf.code_snippet.strip()[:120]}")

        total_findings += len(active_findings)
        total_confirmed += result.confirmed_vulns
        total_fps += result.false_positives

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {total_files} file(s) scanned | "
          f"{total_findings} vulnerabilities found | "
          f"{total_fps} false positives filtered by AI")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Pattern + AI Vulnerability Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_scan_cli.py scan --file app.py
  python hybrid_scan_cli.py scan --code "os.system(user_input)"
  python hybrid_scan_cli.py scan --dir ./src --json
  python hybrid_scan_cli.py scan --file app.py --no-ai
  python hybrid_scan_cli.py rules
  python hybrid_scan_cli.py status
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # scan
    scan_parser = subparsers.add_parser("scan", help="Scan code for vulnerabilities")
    scan_parser.add_argument("--file", "-f", help="Path to source file")
    scan_parser.add_argument("--dir", "-d", help="Path to directory")
    scan_parser.add_argument("--code", "-c", help="Inline code string")
    scan_parser.add_argument("--language", "-l", help="Language (auto-detected)")
    scan_parser.add_argument("--json", action="store_true", help="JSON output")
    scan_parser.add_argument("--no-ai", dest="ai", action="store_false",
                             default=True, help="Disable AI refinement")
    scan_parser.add_argument("--threshold", "-t", type=float, default=0.7,
                             help="AI confidence threshold (default: 0.7)")
    scan_parser.add_argument("--per-snippet", action="store_true",
                             help="Run AI per finding (slower, more precise)")
    scan_parser.add_argument("--no-recursive", action="store_true",
                             help="Don't recurse into subdirectories")

    # status
    subparsers.add_parser("status", help="Show pipeline status")

    # rules
    subparsers.add_parser("rules", help="List all pattern rules")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "rules":
        cmd_rules(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
