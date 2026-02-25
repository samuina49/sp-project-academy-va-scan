"""
Data models for the hybrid vulnerability detection pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CWECategory(str, Enum):
    """Supported CWE vulnerability classes."""
    SQL_INJECTION = "CWE-89"
    COMMAND_INJECTION = "CWE-77"
    PATH_TRAVERSAL = "CWE-22"
    INSECURE_DESERIALIZATION = "CWE-502"
    SSRF = "CWE-918"
    HARDCODED_SECRETS = "CWE-798"


class VulnLabel(str, Enum):
    """Classification labels from the pattern engine."""
    VULN_CANDIDATE = "VULN_CANDIDATE"
    SAFE = "SAFE"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class Verdict(str, Enum):
    VULNERABLE = "VULNERABLE"          # Pattern + AI agree
    LIKELY_VULNERABLE = "LIKELY_VULN"  # Pattern says yes; AI unavailable
    FALSE_POSITIVE = "FALSE_POSITIVE"  # Pattern says yes; AI says no
    SAFE = "SAFE"                      # Pattern says safe


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PatternFinding:
    """A single finding from the pattern-matching engine (Phase 1)."""
    file: str
    line: int
    end_line: int
    cwe: CWECategory
    rule_id: str
    confidence: str          # "high", "medium", "low"
    severity: Severity
    code_snippet: str
    message: str
    explanation: str         # Human-readable *why* this triggers
    label: VulnLabel = VulnLabel.VULN_CANDIDATE
    language: str = "python"
    negative_matched: bool = False  # True if a negative pattern cancelled this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "cwe": self.cwe.value,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "code_snippet": self.code_snippet,
            "message": self.message,
            "explanation": self.explanation,
            "label": self.label.value,
            "language": self.language,
        }


@dataclass
class RefinedFinding:
    """A finding after AI refinement (Phase 2)."""
    pattern_finding: PatternFinding
    ai_score: float                    # Probability 0..1 from AI model
    ai_available: bool = True          # False → AI was skipped
    verdict: Verdict = Verdict.LIKELY_VULNERABLE
    ai_features_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.pattern_finding.to_dict()
        d["ai_score"] = round(self.ai_score, 4)
        d["ai_available"] = self.ai_available
        d["verdict"] = self.verdict.value
        d["ai_features_used"] = self.ai_features_used
        return d


@dataclass
class PipelineResult:
    """Full result from one pipeline run."""
    file: str
    language: str
    original_language: str       # Before transpilation
    pattern_findings: List[PatternFinding] = field(default_factory=list)
    refined_findings: List[RefinedFinding] = field(default_factory=list)
    total_candidates: int = 0
    confirmed_vulns: int = 0
    false_positives: int = 0
    ai_available: bool = False
    scan_duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "language": self.language,
            "original_language": self.original_language,
            "total_pattern_candidates": self.total_candidates,
            "confirmed_vulnerabilities": self.confirmed_vulns,
            "false_positives_filtered": self.false_positives,
            "ai_available": self.ai_available,
            "scan_duration_ms": round(self.scan_duration_ms, 1),
            "findings": [f.to_dict() for f in self.refined_findings
                         if f.verdict in (Verdict.VULNERABLE, Verdict.LIKELY_VULNERABLE)],
        }
