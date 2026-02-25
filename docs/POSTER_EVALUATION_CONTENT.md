# Poster Content — Hybrid Vulnerability Scanner

> Ready-to-use text blocks for academic poster. All numbers verified and reproducible.

---

## TITLE

**Hybrid Pattern + AI Vulnerability Scanner for Python & JavaScript**

*Static Analysis with Rule-Based Detection and Neural Network Advisory*

---

## ABSTRACT (≤100 words)

We present a hybrid vulnerability scanner that combines high-recall pattern matching (52 regex rules across 6 CWEs) with an advisory AI model (GNN+BiLSTM, 2.6M parameters). Evaluated against the OWASP Top 10 (2021) using 40 realistic code samples, the system achieves **65% overall detection** with **100% precision** (zero false positives). It attains **100% detection** on deserialization and SSRF, **88%** on injection flaws, and **80%** on hardcoded secrets. Categories requiring architectural analysis (Insecure Design, Logging) are honestly reported as out of scope. The scanner processes code in under 2 seconds per file.

---

## SYSTEM ARCHITECTURE (for diagram)

```
Input Code (Python / JavaScript / TypeScript)
        │
        ▼
┌─────────────────────────────┐
│  Phase 1: Pattern Engine    │
│  52 regex rules × 6 CWEs   │
│  Wide-context ±10 lines     │
│  Negative-pattern filtering │
└──────────┬──────────────────┘
           │ Candidates
           ▼
┌─────────────────────────────┐
│  Phase 2: AI Refiner        │
│  GNN + BiLSTM (2.6M params) │
│  Confidence scoring ≥ 0.7   │
│  Advisory verdicts          │
└──────────┬──────────────────┘
           │ Final verdicts
           ▼
┌─────────────────────────────┐
│  Output: Findings Report    │
│  CWE · OWASP · Severity    │
│  Fix recommendations        │
└─────────────────────────────┘
```

---

## KEY NUMBERS (for poster callouts)

| Metric | Value |
|--------|-------|
| Pattern rules | 52 |
| CWEs covered | 6 |
| OWASP categories with detections | 8 / 10 |
| Total test samples | 40 |
| True positives | 26 |
| False positives | 0 |
| Overall detection rate | 65.0% |
| Precision | 100% |
| Best category (A08 Deserialization) | 100% |
| Best category (A10 SSRF) | 100% |
| Injection detection (A03) | 88% |
| Scan time per sample (pattern only) | < 5 ms |
| Scan time per sample (with AI) | ~1.7 s |
| AI model parameters | 2.6M |
| Languages supported | Python, JavaScript, TypeScript |

---

## OWASP TOP 10 RESULTS TABLE (for poster)

| # | OWASP Category | Det. Rate | Bar |
|---|----------------|-----------|-----|
| A01 | Broken Access Control | 60% | ██████░░░░ |
| A02 | Cryptographic Failures | 80% | ████████░░ |
| A03 | Injection | **88%** | █████████░ |
| A04 | Insecure Design | 0% | ░░░░░░░░░░ |
| A05 | Security Misconfiguration | 33% | ███░░░░░░░ |
| A06 | Vulnerable Components | 50% | █████░░░░░ |
| A07 | Auth Failures | 33% | ███░░░░░░░ |
| A08 | Data Integrity (Deser.) | **100%** | ██████████ |
| A09 | Logging & Monitoring | 0% | ░░░░░░░░░░ |
| A10 | SSRF | **100%** | ██████████ |

---

## STRENGTHS SECTION

- **100% detection** on Deserialization (A08) and SSRF (A10) — critical vulnerability classes
- **88% detection** on Injection (A03) — SQL injection and command injection across multiple patterns
- **80% detection** on Cryptographic Failures (A02) — hardcoded API keys, passwords, AWS credentials
- **Zero false positives** across all 40 test samples
- **Sub-second pattern matching** — suitable for CI/CD integration
- Detects vulnerabilities across **Python, JavaScript, and TypeScript**

---

## LIMITATIONS SECTION (Honest Assessment)

- Pattern matching is **regex-based**, not semantic — cannot track cross-function data flow
- AI model trained on **synthetic data** — serves as advisory, not primary detector
- **6 out of 25+ CWEs** covered — scanner does not claim full OWASP Top 10 coverage
- Not detected: Insecure Design (A04), Logging failures (A09) — require architectural context
- Not detected: Vulnerable components (A06) — requires SCA tooling (npm audit, pip-audit)
- Single-file analysis only — **no cross-file taint tracking**
- Evaluation uses **crafted test samples** — real-world rates may differ

---

## EXAMPLE VULNERABILITY (for poster sidebar)

### SQL Injection (CWE-89)

**Vulnerable:**
```python
cursor.execute(f"SELECT * FROM products WHERE name LIKE '%{query}%'")
```

**Detected by:** `SQLI_PY_EXEC_FSTRING` — Pattern Engine  
**Severity:** HIGH  
**Risk:** Attacker injects `' OR '1'='1` → dumps entire database

**Fixed:**
```python
cursor.execute("SELECT * FROM products WHERE name LIKE ?", (f"%{query}%",))
```

---

## FUTURE WORK

1. Add rules for `eval()`, weak hashing (MD5/SHA1), debug mode detection
2. Implement cross-file taint analysis using code property graphs
3. Retrain AI model on real-world labeled datasets (CodeQL, SARD)
4. Integrate SCA scanning for dependency vulnerabilities (A06)
5. Add DAST-style runtime validation for configuration issues (A05)

---

## CONCLUSION (≤50 words)

The hybrid scanner provides reliable, high-precision detection for injection, deserialization, SSRF, and hardcoded secrets — four of the most exploitable vulnerability classes. It honestly acknowledges limitations in architectural and component-level analysis. With zero false positives and sub-second scan times, it is suitable as a first-line defense in development workflows.

---

*All results reproducible via `backend/owasp_evaluation.py`*
