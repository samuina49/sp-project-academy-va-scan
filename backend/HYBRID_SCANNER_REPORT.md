# Hybrid Vulnerability Scanner — Technical Report

## Executive Summary

A hybrid vulnerability detection system combining **deterministic pattern matching** (Phase 1, high recall) with **ML-based refinement** (Phase 2, high precision) for Python, JavaScript, and TypeScript source code.

**Evaluation Results (61 test cases):**

| Metric | Score |
|--------|-------|
| Overall Recall | **100.0%** |
| Overall Precision | **100.0%** |
| Overall F1 | **100.0%** |
| CWEs Passing (≥95% recall) | **6/6** |
| Pattern Rules | **52** |
| Test Cases | **61** (42 vulnerable, 19 safe) |

---

## Architecture

### Design Philosophy

> "Security vulnerability detection is not a pure ML problem."

The system uses pattern matching as the **primary detector** and AI as an **advisory refinement layer**. This reflects the reality that:
- Pattern matching provides deterministic, explainable results
- The ML model was trained on synthetic data and cannot generalize to real-world code
- False negatives in security scanning are worse than false positives

### Pipeline Flow

```
Source Code → Language Detection → TS Transpilation → Pattern Matching → AI Refinement → Results
```

1. **Phase 0 — Preprocessing**: Detect language (extension/shebang/heuristics), transpile TypeScript to JavaScript
2. **Phase 1 — Pattern Matching**: 52 regex rules across 6 CWEs, negative patterns for validation detection, line-by-line + multiline scanning
3. **Phase 2 — AI Refinement**: Existing GNN+BiLSTM+Metrics model scores each candidate, verdict = Pattern ∧ AI

### Final Verdict Logic

- Pattern detects + AI confirms (score ≥ 0.7) → **VULNERABLE**
- Pattern detects + AI rejects → **FALSE_POSITIVE**
- Pattern detects + AI unavailable → **LIKELY_VULNERABLE** (never silent-drops)

---

## Coverage Scope

### Supported CWEs

| CWE | Name | Rules | Test Cases (Vuln/Safe) |
|-----|------|-------|----------------------|
| CWE-89 | SQL Injection | 13 | 8 / 5 |
| CWE-77 | Command Injection | 9 | 8 / 3 |
| CWE-22 | Path Traversal | 10 | 6 / 2 |
| CWE-502 | Insecure Deserialization | 7 | 6 / 3 |
| CWE-918 | SSRF | 7 | 6 / 2 |
| CWE-798 | Hardcoded Secrets | 6 | 8 / 4 |

### Explicitly NOT Covered

- **Access control** (requires architectural/runtime context)
- **Authentication logic** (requires flow analysis)
- **Dependency vulnerabilities** (requires SCA tooling like Snyk/Dependabot)
- **Business logic flaws** (requires domain knowledge)
- **Cross-file data flow** (taint propagation across modules)

This scanner does **NOT** claim OWASP Top 10 coverage.

---

## Per-CWE Evaluation Results

| CWE | Recall | Precision | F1 | FPR | Verdict |
|-----|--------|-----------|-----|-----|---------|
| CWE-22 | 100.0% | 100.0% | 100.0% | 0.0% | PASS |
| CWE-502 | 100.0% | 100.0% | 100.0% | 0.0% | PASS |
| CWE-77 | 100.0% | 100.0% | 100.0% | 0.0% | PASS |
| CWE-798 | 100.0% | 100.0% | 100.0% | 0.0% | PASS |
| CWE-89 | 100.0% | 100.0% | 100.0% | 0.0% | PASS |
| CWE-918 | 100.0% | 100.0% | 100.0% | 0.0% | PASS |

---

## Module Structure

```
backend/app/hybrid_scanner/
├── __init__.py          # Package exports
├── models.py            # Data models (PatternFinding, RefinedFinding, PipelineResult)
├── pattern_engine.py    # 52 regex rules across 6 CWEs
├── language_detect.py   # Language detection (extension, shebang, heuristics)
├── transpiler.py        # TypeScript → JavaScript transpilation
├── ai_refiner.py        # AI refinement bridge to HybridPredictor
├── pipeline.py          # Full orchestration pipeline
├── cli.py               # CLI interface (scan/status/rules)
└── test_suite.py        # 61 realistic test cases

backend/app/api/v1/hybrid_pipeline_scan.py  # FastAPI endpoint
backend/hybrid_scan_cli.py                   # CLI entry point
backend/evaluate_hybrid.py                   # Evaluation script
```

---

## API Endpoints

### POST `/api/v1/hybrid-scan/code`

Scan source code for vulnerabilities.

**Request:**
```json
{
  "code": "import os; os.system(user_input)",
  "language": "python",
  "filename": "app.py",
  "use_ai": true,
  "ai_threshold": 0.7
}
```

**Response:**
```json
{
  "scan_id": "uuid",
  "findings": [...],
  "total_candidates": 5,
  "confirmed_vulnerabilities": 3,
  "filtered_false_positives": 2,
  "scan_time_ms": 45.2
}
```

### GET `/api/v1/hybrid-scan/status`

Check scanner health and configuration.

---

## CLI Usage

```bash
# Scan a file
python hybrid_scan_cli.py scan --file vulnerable_app.py

# Scan a directory
python hybrid_scan_cli.py scan --dir ./src/

# Scan inline code
python hybrid_scan_cli.py scan --code "os.system(user_input)"

# JSON output (for CI/CD)
python hybrid_scan_cli.py scan --file app.py --json

# Pattern-only (no AI)
python hybrid_scan_cli.py scan --file app.py --no-ai

# List all rules
python hybrid_scan_cli.py rules

# Check status
python hybrid_scan_cli.py status
```

---

## Honest Limitations

1. **Regex-based, not semantic**: Cannot understand data flow across function boundaries
2. **AI model limitations**: Trained on synthetic data, shows 0% detection on realistic OWASP test cases — AI refinement is advisory only
3. **TS transpilation**: Uses regex stripping fallback when `tsc` isn't available; complex TS patterns may not fully process
4. **Single-file scope**: Cross-file taint propagation is not supported
5. **Test suite bias**: 100% on our 61 test cases does not guarantee 100% on all real-world code — real code is far more diverse
6. **No runtime context**: Cannot detect vulnerabilities that depend on configuration, authentication state, or runtime behavior

---

## How to Run Evaluation

```bash
cd backend
python evaluate_hybrid.py
```

This runs all 61 test cases through the pattern engine and reports per-CWE recall, precision, F1, and false positive rate with failure analysis.

---

*Generated: Hybrid Scanner v1.0 — Pattern Engine with AI Refinement*
