"""
API Endpoint for the Hybrid Pattern + AI Vulnerability Scanner
===============================================================
POST /api/v1/hybrid-scan/code
POST /api/v1/hybrid-scan/status
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Hard False-Positive Filter — bare import / require declarations are never
# directly exploitable.  Applied as the very last step before JSON response.
# ---------------------------------------------------------------------------
_IMPORT_ONLY_RE = re.compile(
    r"""^[\s]*(?:
        import\s+[\w.,\s*{}]+(?:\s+from\s+['"][^'"]+['"])?  # Python import / ES6 import
        |from\s+[\w.]+\s+import\s+[\w.,*\s()]+              # Python from-import
        |(?:[\w\s,{}*]+\s*=\s*)?require\s*\(\s*['"][^'"]+['"]\s*\)  # CommonJS require
    )\s*;?\s*$""",
    re.VERBOSE | re.IGNORECASE,
)


def _is_import_snippet(snippet: str) -> bool:
    """Return True when every non-blank line of *snippet* is a bare import/require."""
    lines = [ln for ln in snippet.splitlines() if ln.strip()]
    if not lines:
        return False
    return all(_IMPORT_ONLY_RE.match(ln) for ln in lines)


def _filter_fp_imports(findings: list, code: str = "") -> list:
    """
    Drop FindingResponse objects that point to a bare import/require declaration.

    Two-pass check:
    1. code_snippet — if every non-blank line is an import/require, drop it.
    2. actual source line — if the specific line in the original code is an
       import/require declaration, drop it.  This catches cases where the
       snippet includes non-import context lines (e.g. the line below the
       import), which made check 1 fail.
    """
    code_lines = code.splitlines() if code else []
    out = []
    for f in findings:
        # Pass 1: full snippet check
        snippet = (f.code_snippet or "").strip()
        if _is_import_snippet(snippet):
            logger.info(
                "[FP-FILTER] Dropped (snippet) — line %s: %s",
                f.line, snippet[:80],
            )
            continue
        # Pass 2: actual source-line check
        if code_lines and 1 <= f.line <= len(code_lines):
            actual = code_lines[f.line - 1].strip()
            if actual and _IMPORT_ONLY_RE.match(actual):
                logger.info(
                    "[FP-FILTER] Dropped (source line) — line %s: %s",
                    f.line, actual[:80],
                )
                continue
        out.append(f)
    return out

from app.hybrid_scanner.pipeline import HybridPipeline
from app.hybrid_scanner.models import Verdict

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hybrid-scan", tags=["hybrid-scanner"])

# ─── Singleton pipeline (lazy init) ─────────────────────────────────

_pipeline: Optional[HybridPipeline] = None


def _get_pipeline() -> HybridPipeline:
    """Lazy-initialize the hybrid pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridPipeline(
            ai_enabled=True,
            threshold=0.7,
            per_snippet_refinement=False,
        )
    return _pipeline


# ─── Request / Response Models ───────────────────────────────────────

class HybridScanRequest(BaseModel):
    code: str = Field(..., description="Source code to scan", min_length=1)
    language: Optional[str] = Field(
        None, description="Language (auto-detected if omitted): python, javascript, typescript"
    )
    filename: Optional[str] = Field(None, description="Optional filename")
    threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="AI confidence threshold (default 0.7)"
    )
    ai_enabled: Optional[bool] = Field(
        None, description="Enable/disable AI refinement"
    )


class FindingResponse(BaseModel):
    line: int
    end_line: int
    cwe: str
    rule_id: str
    severity: str
    confidence: str
    message: str
    explanation: str
    code_snippet: str
    verdict: str
    ai_score: float
    ai_available: bool
    language: str


class HybridScanResponse(BaseModel):
    scan_id: str
    timestamp: str
    file: str
    language: str
    original_language: str
    total_candidates: int
    confirmed_vulnerabilities: int
    false_positives_filtered: int
    ai_available: bool
    scan_duration_ms: float
    findings: list[FindingResponse]
    errors: list[str]


class StatusResponse(BaseModel):
    pattern_engine: dict
    ai_refiner: dict
    supported_cwes: list[str]
    supported_languages: list[str]
    threshold: float


# ─── Endpoints ───────────────────────────────────────────────────────

@router.post("/code", response_model=HybridScanResponse)
async def scan_code(request: HybridScanRequest):
    """
    Scan source code using the hybrid Pattern + AI pipeline.
    
    Phase 1: Pattern matching (deterministic, high recall)
    Phase 2: AI refinement (ML model, high precision, optional)
    
    Final verdict = Pattern Match AND AI Confirmation.
    """
    try:
        pipeline = _get_pipeline()
        
        # Override threshold/enabled if requested
        if request.threshold is not None:
            pipeline.ai_refiner.threshold = request.threshold
        if request.ai_enabled is not None:
            pipeline.ai_refiner.enabled = request.ai_enabled
        
        result = pipeline.scan_code(
            code=request.code,
            language=request.language,
            filename=request.filename,
        )
        
        # Convert refined findings to response format
        findings = []
        for rf in result.refined_findings:
            if rf.verdict in (Verdict.VULNERABLE, Verdict.LIKELY_VULNERABLE):
                pf = rf.pattern_finding
                findings.append(FindingResponse(
                    line=pf.line,
                    end_line=pf.end_line,
                    cwe=pf.cwe.value,
                    rule_id=pf.rule_id,
                    severity=pf.severity.value,
                    confidence=pf.confidence,
                    message=pf.message,
                    explanation=pf.explanation,
                    code_snippet=pf.code_snippet,
                    verdict=rf.verdict.value,
                    ai_score=round(rf.ai_score, 4),
                    ai_available=rf.ai_available,
                    language=pf.language,
                ))
        
        # ── Hard False-Positive Filter ──────────────────────────────────────
        # Pass request.code so the filter can cross-check the actual source
        # line at finding.line, not only the ±1-context snippet.
        before = len(findings)
        findings = _filter_fp_imports(findings, request.code)
        fp_extra = before - len(findings)
        if fp_extra:
            logger.info("[FP-FILTER] Removed %d import false positive(s)", fp_extra)

        return HybridScanResponse(
            scan_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            file=result.file,
            language=result.language,
            original_language=result.original_language,
            total_candidates=result.total_candidates,
            confirmed_vulnerabilities=len(findings),
            false_positives_filtered=result.false_positives + fp_extra,
            ai_available=result.ai_available,
            scan_duration_ms=round(result.scan_duration_ms, 1),
            findings=findings,
            errors=result.errors,
        )
    
    except Exception as e:
        logger.error(f"[HybridScan] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the hybrid scanner pipeline status."""
    pipeline = _get_pipeline()
    status = pipeline.status
    return StatusResponse(**status)
