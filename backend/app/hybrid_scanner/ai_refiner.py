"""
AI Refinement Module (Phase 2) — High Precision
=================================================
Takes VULN_CANDIDATE findings from the pattern engine and uses the
trained HybridVulnerabilityModel (GNN + BiLSTM + Metrics) to either
confirm or reject each finding.

DESIGN CONSTRAINTS:
    - AI output is ADVISORY, not authoritative
    - Final verdict = Pattern Match AND AI Confirmation
    - AI is NEVER used as a standalone scanner
    - Threshold is configurable (default 0.7)
    - If AI fails/unavailable, findings remain LIKELY_VULNERABLE (not rejected)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.hybrid_scanner.models import (
    PatternFinding,
    RefinedFinding,
    Verdict,
)

logger = logging.getLogger(__name__)

# Resolve paths regardless of cwd
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL = _BACKEND_ROOT / "models" / "best_model.pt"
_DEFAULT_VOCAB = _BACKEND_ROOT / "data" / "processed_graphs" / "vocabulary.pkl"


class AIRefiner:
    """
    Phase 2: Use the pre-trained Hybrid ML model to refine pattern findings.
    
    Only processes items labeled VULN_CANDIDATE from Phase 1.
    Produces a probability score [0,1] and decides:
        score >= threshold → VULNERABLE  (confirmed by AI)
        score <  threshold → FALSE_POSITIVE (filtered by AI)
    
    If model is unavailable, all candidates stay LIKELY_VULNERABLE.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        threshold: float = 0.7,
        enabled: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to trained .pt checkpoint
            vocab_path: Path to vocabulary pickle
            threshold: Confidence threshold for confirming a vulnerability
            enabled: If False, skip AI entirely
            device: "cuda" or "cpu"
        """
        self.threshold = threshold
        self.enabled = enabled
        self._predictor = None
        self._load_error: Optional[str] = None
        
        model_path = model_path or str(_DEFAULT_MODEL)
        vocab_path = vocab_path or str(_DEFAULT_VOCAB)
        
        if not enabled:
            self._load_error = "AI refinement disabled by configuration"
            return
        
        # Try to load the model
        try:
            from app.ml.inference.hybrid_predictor import HybridPredictor
            
            if not Path(model_path).exists():
                self._load_error = f"Model not found: {model_path}"
                logger.warning(f"[AIRefiner] {self._load_error}")
                return
            
            self._predictor = HybridPredictor(
                model_path=model_path,
                vocab_path=vocab_path,
                device=device,
            )
            logger.info("[AIRefiner] Model loaded successfully")
            
        except Exception as e:
            self._load_error = f"Model load failed: {e}"
            logger.warning(f"[AIRefiner] {self._load_error}")

    @property
    def available(self) -> bool:
        """Check if AI model is loaded and ready."""
        return self._predictor is not None

    @property
    def status(self) -> Dict[str, Any]:
        """Get AI refiner status."""
        info = {
            "enabled": self.enabled,
            "available": self.available,
            "threshold": self.threshold,
        }
        if self._load_error:
            info["error"] = self._load_error
        if self._predictor:
            try:
                model_info = self._predictor.get_model_info()
                info["model"] = model_info
            except Exception:
                pass
        return info

    def refine(
        self,
        findings: List[PatternFinding],
        full_code: str,
        language: str = "python",
    ) -> List[RefinedFinding]:
        """
        Refine pattern-matched findings using the AI model.
        
        The model receives the FULL code context (not just the snippet),
        because the GNN/LSTM needs complete AST structure.
        
        Args:
            findings: List of VULN_CANDIDATE findings from Phase 1
            full_code: Complete source code string
            language: Programming language
            
        Returns:
            List of RefinedFinding with AI scores and verdicts
        """
        refined: List[RefinedFinding] = []
        
        if not findings:
            return refined
        
        # If AI is not available, mark everything as LIKELY_VULNERABLE
        if not self.available:
            for f in findings:
                refined.append(RefinedFinding(
                    pattern_finding=f,
                    ai_score=0.0,
                    ai_available=False,
                    verdict=Verdict.LIKELY_VULNERABLE,
                    ai_features_used=[],
                ))
            return refined
        
        # Run AI prediction on the FULL code once (it's the same code for all findings)
        ai_result = self._predict_safe(full_code, language)
        
        score = ai_result.get("raw_score", 0.0)
        had_error = "error" in ai_result
        features_used = []
        
        if not had_error:
            features_used = ["gnn_graph", "lstm_tokens", "code_metrics"]
        
        for finding in findings:
            if had_error:
                # AI failed for this run → keep as LIKELY_VULNERABLE
                verdict = Verdict.LIKELY_VULNERABLE
            elif score >= self.threshold:
                verdict = Verdict.VULNERABLE
            else:
                verdict = Verdict.FALSE_POSITIVE
            
            refined.append(RefinedFinding(
                pattern_finding=finding,
                ai_score=score,
                ai_available=not had_error,
                verdict=verdict,
                ai_features_used=features_used,
            ))
        
        return refined

    def refine_per_snippet(
        self,
        findings: List[PatternFinding],
        full_code: str,
        language: str = "python",
    ) -> List[RefinedFinding]:
        """
        Refine each finding individually by extracting a focused code window
        around the vulnerable line. More expensive but more precise.
        
        Falls back to full-code mode if snippets are too small for AST parsing.
        
        Args:
            findings: List of VULN_CANDIDATE findings
            full_code: Complete source code
            language: Programming language
            
        Returns:
            List of RefinedFinding with per-finding AI scores
        """
        refined: List[RefinedFinding] = []
        
        if not findings:
            return refined
        
        if not self.available:
            for f in findings:
                refined.append(RefinedFinding(
                    pattern_finding=f,
                    ai_score=0.0,
                    ai_available=False,
                    verdict=Verdict.LIKELY_VULNERABLE,
                    ai_features_used=[],
                ))
            return refined
        
        lines = full_code.split("\n")
        
        for finding in findings:
            # Extract a window of ~30 lines around the finding
            center = finding.line - 1  # 0-indexed
            start = max(0, center - 15)
            end = min(len(lines), center + 15)
            snippet_code = "\n".join(lines[start:end])
            
            # Need at least a few lines for meaningful analysis
            if len(snippet_code.strip()) < 20:
                snippet_code = full_code
            
            ai_result = self._predict_safe(snippet_code, language)
            score = ai_result.get("raw_score", 0.0)
            had_error = "error" in ai_result
            
            if had_error:
                verdict = Verdict.LIKELY_VULNERABLE
                features_used = []
            elif score >= self.threshold:
                verdict = Verdict.VULNERABLE
                features_used = ["gnn_graph", "lstm_tokens", "code_metrics"]
            else:
                verdict = Verdict.FALSE_POSITIVE
                features_used = ["gnn_graph", "lstm_tokens", "code_metrics"]
            
            refined.append(RefinedFinding(
                pattern_finding=finding,
                ai_score=score,
                ai_available=not had_error,
                verdict=verdict,
                ai_features_used=features_used,
            ))
        
        return refined

    def _predict_safe(self, code: str, language: str) -> Dict:
        """Run prediction with error handling."""
        try:
            return self._predictor.predict(
                code=code,
                language=language,
                return_confidence=False,
            )
        except Exception as e:
            logger.error(f"[AIRefiner] Prediction error: {e}")
            return {"vulnerable": False, "confidence": 0.0, "raw_score": 0.0, "error": str(e)}
