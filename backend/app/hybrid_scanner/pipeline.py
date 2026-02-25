"""
Hybrid Scanning Pipeline
=========================
Orchestrates the complete flow:

    Source Code
       |
    Language Detection
       |
    TS -> JS Transpilation (if needed)
       |
    Pattern Matching Engine (Phase 1 - High Recall)
       |
    VULN_CANDIDATE only
       |
    AI Refinement (Phase 2 - High Precision)
       |
    Final Classification with Verdicts

Design principles:
    - Pattern matching is ALWAYS run (deterministic, fast)
    - AI is OPTIONAL and ADVISORY
    - Final verdict = Pattern AND AI
    - If AI unavailable, findings are LIKELY_VULNERABLE (not rejected)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from app.hybrid_scanner.models import (
    PatternFinding,
    RefinedFinding,
    PipelineResult,
    Verdict,
    VulnLabel,
)
from app.hybrid_scanner.pattern_engine import PatternMatchingEngine
from app.hybrid_scanner.language_detect import LanguageDetector
from app.hybrid_scanner.transpiler import TypeScriptTranspiler
from app.hybrid_scanner.ai_refiner import AIRefiner

logger = logging.getLogger(__name__)


class HybridPipeline:
    """
    Complete hybrid vulnerability detection pipeline.
    
    Usage:
        pipeline = HybridPipeline(ai_enabled=True, threshold=0.7)
        result = pipeline.scan_code(code, language="python", filename="app.py")
        
        # result.summary → dict with all findings
        # result.refined_findings → list of RefinedFinding
    """

    def __init__(
        self,
        ai_enabled: bool = True,
        threshold: float = 0.7,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        device: Optional[str] = None,
        per_snippet_refinement: bool = False,
    ):
        """
        Args:
            ai_enabled: Enable AI refinement (Phase 2)
            threshold: AI confidence threshold for confirming a vuln (0-1)
            model_path: Path to .pt model checkpoint
            vocab_path: Path to vocabulary file
            device: "cuda" or "cpu"
            per_snippet_refinement: If True, run AI per finding (slower, more precise)
        """
        self.threshold = threshold
        self.per_snippet = per_snippet_refinement
        
        # Phase 1: Pattern matching (always available)
        self.pattern_engine = PatternMatchingEngine()
        
        # Language detection
        self.lang_detector = LanguageDetector()
        
        # TypeScript transpiler
        self.transpiler = TypeScriptTranspiler(use_tsc=True)
        
        # Phase 2: AI refinement (optional)
        self.ai_refiner = AIRefiner(
            model_path=model_path,
            vocab_path=vocab_path,
            threshold=threshold,
            enabled=ai_enabled,
            device=device,
        )
        
        logger.info(
            f"[HybridPipeline] Initialized: "
            f"pattern_rules={sum(self.pattern_engine.get_rule_count().values())}, "
            f"ai_enabled={ai_enabled}, ai_available={self.ai_refiner.available}, "
            f"threshold={threshold}"
        )

    def scan_code(
        self,
        code: str,
        language: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> PipelineResult:
        """
        Scan source code through the full hybrid pipeline.
        
        Args:
            code: Source code string
            language: Programming language (auto-detected if None)
            filename: Optional filename
            
        Returns:
            PipelineResult with all findings and metadata
        """
        t0 = time.time()
        errors: List[str] = []
        
        # ─── Step 1: Language Detection ───
        if language:
            detected_lang = LanguageDetector.normalize(language)
        else:
            detected_lang = self.lang_detector.detect(code=code, filename=filename)
        
        if not detected_lang:
            detected_lang = "python"  # Conservative default
            errors.append("Language auto-detection failed; defaulting to python")
        
        if not LanguageDetector.is_supported(detected_lang):
            return PipelineResult(
                file=filename or "input",
                language=detected_lang,
                original_language=detected_lang,
                errors=[f"Unsupported language: {detected_lang}"],
                scan_duration_ms=(time.time() - t0) * 1000,
            )
        
        original_language = detected_lang
        scan_code = code
        
        # ─── Step 2: TypeScript → JavaScript Transpilation ───
        if detected_lang == "typescript":
            try:
                scan_code = self.transpiler.transpile(code, filename or "input.ts")
                detected_lang = "javascript"
                logger.info("[HybridPipeline] TypeScript transpiled to JavaScript for scanning")
            except Exception as e:
                errors.append(f"TS transpilation failed: {e}; scanning raw TS")
                detected_lang = "javascript"  # Try scanning as JS anyway
        
        # ─── Step 3: Pattern Matching (Phase 1) ───
        pattern_findings = self.pattern_engine.scan(
            code=scan_code,
            language=detected_lang,
            filename=filename or "input",
        )
        
        # Filter to VULN_CANDIDATE only
        candidates = [f for f in pattern_findings if f.label == VulnLabel.VULN_CANDIDATE]
        
        logger.info(
            f"[HybridPipeline] Phase 1 complete: "
            f"{len(candidates)} candidates from {len(pattern_findings)} matches"
        )
        
        # ─── Step 4: AI Refinement (Phase 2) ───
        if candidates and self.ai_refiner.enabled:
            if self.per_snippet:
                refined = self.ai_refiner.refine_per_snippet(
                    candidates, scan_code, detected_lang
                )
            else:
                refined = self.ai_refiner.refine(
                    candidates, scan_code, detected_lang
                )
        else:
            # No AI → all candidates are LIKELY_VULNERABLE
            refined = [
                RefinedFinding(
                    pattern_finding=f,
                    ai_score=0.0,
                    ai_available=False,
                    verdict=Verdict.LIKELY_VULNERABLE,
                )
                for f in candidates
            ]
        
        # ─── Step 5: Tally results ───
        confirmed = sum(1 for r in refined
                        if r.verdict in (Verdict.VULNERABLE, Verdict.LIKELY_VULNERABLE))
        fps = sum(1 for r in refined if r.verdict == Verdict.FALSE_POSITIVE)
        
        elapsed_ms = (time.time() - t0) * 1000
        
        result = PipelineResult(
            file=filename or "input",
            language=detected_lang,
            original_language=original_language,
            pattern_findings=candidates,
            refined_findings=refined,
            total_candidates=len(candidates),
            confirmed_vulns=confirmed,
            false_positives=fps,
            ai_available=self.ai_refiner.available,
            scan_duration_ms=elapsed_ms,
            errors=errors,
        )
        
        logger.info(
            f"[HybridPipeline] Scan complete in {elapsed_ms:.0f}ms: "
            f"{confirmed} confirmed, {fps} filtered, "
            f"AI={'ON' if self.ai_refiner.available else 'OFF'}"
        )
        
        return result

    def scan_file(self, file_path: Union[str, Path]) -> PipelineResult:
        """
        Scan a file from disk.
        
        Args:
            file_path: Path to the source file
            
        Returns:
            PipelineResult
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return PipelineResult(
                file=str(file_path),
                language="unknown",
                original_language="unknown",
                errors=[f"File not found: {file_path}"],
            )
        
        code = file_path.read_text(encoding="utf-8", errors="replace")
        language = self.lang_detector.detect(filename=str(file_path))
        
        return self.scan_code(
            code=code,
            language=language,
            filename=str(file_path.name),
        )

    def scan_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
    ) -> List[PipelineResult]:
        """
        Scan all supported files in a directory.
        
        Args:
            directory: Directory path
            recursive: Scan subdirectories
            
        Returns:
            List of PipelineResult, one per file
        """
        directory = Path(directory)
        if not directory.is_dir():
            return [PipelineResult(
                file=str(directory),
                language="unknown",
                original_language="unknown",
                errors=[f"Not a directory: {directory}"],
            )]
        
        supported_exts = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
        skip_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv",
                     "dist", "build", ".next", ".tox", "env"}
        
        results: List[PipelineResult] = []
        glob_fn = directory.rglob if recursive else directory.glob
        
        for fpath in sorted(glob_fn("*")):
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in supported_exts:
                continue
            # Skip ignored directories
            if any(part in skip_dirs for part in fpath.parts):
                continue
            
            result = self.scan_file(fpath)
            results.append(result)
        
        return results

    @property
    def status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "pattern_engine": {
                "rules_by_cwe": self.pattern_engine.get_rule_count(),
                "total_rules": sum(self.pattern_engine.get_rule_count().values()),
            },
            "ai_refiner": self.ai_refiner.status,
            "supported_cwes": [c.value for c in
                               sorted(set(r.cwe for r in self.pattern_engine.rules),
                                      key=lambda x: x.value)],
            "supported_languages": ["python", "javascript", "typescript"],
            "threshold": self.threshold,
        }
