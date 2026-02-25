"""
Hybrid Pattern-Matching + AI Vulnerability Detection System

Architecture:
    Source Code → Language Detection → [TS→JS Transpilation] → Pattern Matching
    → VULN_CANDIDATE only → Feature Extraction → GNN+LSTM Model → Final Verdict

Supported Languages: Python, JavaScript, TypeScript (→ transpiled to JS)
Supported CWEs: CWE-89, CWE-77, CWE-22, CWE-502, CWE-918, CWE-798

Design Principles:
    - Pattern matching provides HIGH RECALL (catch everything)
    - AI refinement provides HIGH PRECISION (reduce false positives)
    - Final verdict = Pattern Match ∧ AI Confirmation
    - AI output is ADVISORY, never authoritative alone
"""

from app.hybrid_scanner.pattern_engine import PatternMatchingEngine
from app.hybrid_scanner.language_detect import LanguageDetector
from app.hybrid_scanner.transpiler import TypeScriptTranspiler
from app.hybrid_scanner.ai_refiner import AIRefiner
from app.hybrid_scanner.pipeline import HybridPipeline
from app.hybrid_scanner.models import (
    PatternFinding,
    RefinedFinding,
    PipelineResult,
    CWECategory,
    VulnLabel,
)

__all__ = [
    "PatternMatchingEngine",
    "LanguageDetector",
    "TypeScriptTranspiler",
    "AIRefiner",
    "HybridPipeline",
    "PatternFinding",
    "RefinedFinding",
    "PipelineResult",
    "CWECategory",
    "VulnLabel",
]
