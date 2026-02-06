"""
Hybrid Vulnerability Scanner - Combines Pattern Matching + ML Model

This orchestrator routes scanning through two detection layers:
1. Pattern Matching (Semgrep/Bandit) - Fast, high-precision
2. ML Model (GNN+LSTM) - Catches novel patterns

Results are combined via ensemble learning for maximum coverage.
"""

from pathlib import Path
from typing import List, Optional, Dict
import os

from app.scanners.scanner_orchestrator import ScannerOrchestrator
from app.models.scan_models import FileScanResult, VulnerabilityFinding
from app.core.config import settings

# Conditional ML imports
try:
    from app.ml.inference.hybrid_predictor import HybridPredictor
    from app.ml.inference.ensemble import EnsembleCombiner
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class HybridScanner:
    """
    Hybrid vulnerability scanner combining pattern-matching and ML
    """
    
    def __init__(self, ml_model_path: Optional[str] = None, ml_vocab_path: Optional[str] = None, ml_enabled: bool = None):
        """
        Initialize hybrid scanner
        
        Args:
            ml_model_path: Path to ML model checkpoint
            ml_vocab_path: Path to vocabulary file
            ml_enabled: Whether to use ML model (default from settings)
        """
        # Pattern matching scanner (always available)
        self.pattern_scanner = ScannerOrchestrator()
        
        # ML components (optional)
        self.ml_predictor = None
        self.ensemble = None
        
        # Determine if ML should be enabled
        if ml_enabled is None:
            ml_enabled = getattr(settings, 'ML_ENABLED', False)
        
        # Initialize ML if available and enabled
        if ml_enabled and ML_AVAILABLE:
            model_path = ml_model_path or getattr(settings, 'ML_MODEL_PATH', None)
            vocab_path = ml_vocab_path or getattr(settings, 'ML_VOCAB_PATH', None)
            
            if model_path and vocab_path and os.path.exists(model_path) and os.path.exists(vocab_path):
                try:
                    self.ml_predictor = HybridPredictor(
                        model_path=model_path,
                        vocab_path=vocab_path,
                        device='cpu'  # Use CPU for production
                    )
                    
                    # Initialize ensemble combiner
                    ml_weight = getattr(settings, 'ML_WEIGHT', 0.3)
                    self.ensemble = EnsembleCombiner(ml_weight=ml_weight)
                    
                    print(f"[HybridScanner] ML Model loaded successfully")
                    print(f"  Model: {model_path}")
                    print(f"  Vocab: {vocab_path}")
                except Exception as e:
                    print(f"WARNING: ML Model failed to load: {e}")
                    self.ml_predictor = None
            else:
                if not model_path:
                    print(f"WARNING: ML_MODEL_PATH not configured")
                elif not vocab_path:
                    print(f"WARNING: ML_VOCAB_PATH not configured")
                elif not os.path.exists(model_path):
                    print(f"WARNING: ML Model not found at {model_path}")
                elif not os.path.exists(vocab_path):
                    print(f"WARNING: Vocabulary not found at {vocab_path}")
        
        self.ml_enabled = self.ml_predictor is not None
    
    def scan_file(
        self,
        file_path: Path,
        language: Optional[str] = None
    ) -> Optional[FileScanResult]:
        """
        Scan file with hybrid approach
        
        Args:
            file_path: Path to file
            language: Programming language (auto-detected if None)
            
        Returns:
            FileScanResult with combined findings
        """
        # Stage 1: Pattern Matching (always runs)
        pattern_result = self.pattern_scanner.scan_file(file_path, language)
        
        if not pattern_result:
            return None
        
        # If ML is disabled or unavailable, return pattern results
        if not self.ml_enabled:
            return pattern_result
        
        # Stage 2: ML Prediction (optional enhancement)
        try:
            # Read code
            code = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Debug: Log pattern matching results
            print(f"[HYBRID] Pattern results: {len(pattern_result.findings)} findings")
            for f in pattern_result.findings:
                print(f"  - [{f.tool}] {f.rule_id}: {f.message[:50]}...")
            
            # Get ML prediction
            ml_prediction = self.ml_predictor.predict(
                code=code,
                language=pattern_result.language,
                return_confidence=True
            )
            
            print(f"[HYBRID] ML prediction: vulnerable={ml_prediction.get('vulnerable')}, confidence={ml_prediction.get('confidence', 0):.2%}")
            
            # Stage 3: Ensemble Combination
            if self.ensemble and ml_prediction:
                # Convert to format expected by ensemble
                # Include all pattern-based findings (semgrep, bandit, AND simple_scanner)
                semgrep_results = [
                    self._finding_to_dict(f) 
                    for f in pattern_result.findings 
                    if f.tool in ('semgrep', 'simple_scanner')  # Include simple_scanner as semgrep-like
                ]
                
                bandit_results = [
                    self._finding_to_dict(f)
                    for f in pattern_result.findings
                    if f.tool == 'bandit'
                ]
                
                # Debug: Check semgrep and bandit results before combining
                print(f"[HYBRID] Pattern findings (semgrep/simple): {len(semgrep_results)}")
                print(f"[HYBRID] Bandit findings: {len(bandit_results)}")
                
                # Combine results
                combined_findings = self.ensemble.combine(
                    semgrep_results=semgrep_results,
                    bandit_results=bandit_results,
                    ml_prediction=ml_prediction,
                    code=code
                )
                
                print(f"[HYBRID] Combined findings: {len(combined_findings)}")
                
                # Convert back to VulnerabilityFinding format
                enhanced_findings = [
                    self._ensemble_to_finding(f)
                    for f in combined_findings
                ]
                
                pattern_result.findings = enhanced_findings
        
        except Exception as e:
            # ML prediction failed, fall back to pattern-only results
            print(f"WARNING: ML prediction failed: {e}")
        
        return pattern_result
    
    def scan_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> List[FileScanResult]:
        """
        Scan directory with hybrid approach (Pattern + ML)
        
        Args:
            directory: Directory to scan
            recursive: Scan subdirectories
            
        Returns:
            List of FileScanResults with ML-enhanced detection
        """
        # Get all files to scan
        supported_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx'}
        files_to_scan = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Skip ignored directories
                dirs[:] = [d for d in dirs if d not in getattr(settings, 'IGNORED_DIRS', [])]
                for file in files:
                    if Path(file).suffix.lower() in supported_extensions:
                        files_to_scan.append(Path(root) / file)
        else:
            for file in directory.iterdir():
                if file.is_file() and file.suffix.lower() in supported_extensions:
                    files_to_scan.append(file)
        
        results = []
        for file_path in files_to_scan:
            try:
                # Detect language
                ext = file_path.suffix.lower()
                if ext == '.py':
                    language = 'python'
                elif ext in {'.js', '.jsx'}:
                    language = 'javascript'
                elif ext in {'.ts', '.tsx'}:
                    language = 'typescript'
                else:
                    continue  # Skip unsupported files
                
                # Use scan_file for each file (Pattern + ML if enabled)
                result = self.scan_file(file_path, language)
                if result:  # Only add if scan succeeded
                    # Add source code to result for ZIP scan
                    try:
                        source_code = file_path.read_text(encoding='utf-8', errors='ignore')
                        # Limit source code size (max 100KB per file)
                        if len(source_code) > 100000:
                            source_code = source_code[:100000] + '\n// ... (truncated)'
                        result.source_code = source_code
                    except Exception:
                        result.source_code = None
                    results.append(result)
                
            except Exception as e:
                print(f"[HYBRID] Error scanning {file_path}: {e}")
                import traceback
                traceback.print_exc()
                # Still add empty result for the file
                results.append(FileScanResult(
                    file_path=str(file_path),
                    language='unknown',
                    findings=[]
                ))
        
        print(f"[HYBRID] scan_directory completed: {len(results)} files, {sum(len(r.findings) for r in results)} findings")
        return results
    
    def _finding_to_dict(self, finding: VulnerabilityFinding) -> Dict:
        """Convert VulnerabilityFinding to dict for ensemble"""
        return {
            'check_id': finding.rule_id,
            'start': {'line': finding.start_line},
            'end': {'line': finding.end_line},
            'extra': {
                'severity': finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity),
                'message': finding.message
            },
            'test_id': finding.rule_id if finding.tool == 'bandit' else None,
            'line_number': finding.start_line,
            'issue_severity': finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity),
            'issue_text': finding.message,
            'code': finding.code_snippet
        }
    
    def _ensemble_to_finding(self, ensemble_finding) -> VulnerabilityFinding:
        """Convert ensemble Finding to VulnerabilityFinding"""
        from app.models.scan_models import SeverityEnum
        
        # Map severity (preserve CRITICAL severity!)
        severity_map = {
            'CRITICAL': SeverityEnum.CRITICAL,
            'HIGH': SeverityEnum.HIGH,
            'MEDIUM': SeverityEnum.MEDIUM,
            'LOW': SeverityEnum.LOW,
            'INFO': SeverityEnum.INFO
        }
        
        severity_str = ensemble_finding.severity.value if hasattr(ensemble_finding.severity, 'value') else str(ensemble_finding.severity)
        severity = severity_map.get(severity_str, SeverityEnum.MEDIUM)
        
        # Determine tool
        if ensemble_finding.semgrep_rule:
            tool = 'semgrep'
            rule_id = ensemble_finding.semgrep_rule
        elif ensemble_finding.bandit_test:
            tool = 'bandit'
            rule_id = ensemble_finding.bandit_test
        elif hasattr(ensemble_finding, 'ml_probability') and ensemble_finding.ml_probability:
            tool = 'ml'
            rule_id = 'ai-detection'
        else:
            tool = 'hybrid'
            rule_id = 'combined'
        
        return VulnerabilityFinding(
            tool=tool,
            rule_id=rule_id,
            severity=severity,
            message=ensemble_finding.explanation,
            start_line=ensemble_finding.line,
            end_line=ensemble_finding.line,
            code_snippet=ensemble_finding.code_snippet,
            cwe_id=ensemble_finding.cwe_id
        )
    
    @staticmethod
    def get_scanner_status() -> Dict:
        """Get status of all scanner components"""
        status = {
            'pattern_matching': {
                'bandit': ScannerOrchestrator.get_available_scanners()['bandit'],
                'semgrep': ScannerOrchestrator.get_available_scanners()['semgrep']
            },
            'ml_available': ML_AVAILABLE,
            'ml_enabled': False
        }
        
        if ML_AVAILABLE:
            ml_path = getattr(settings, 'ML_MODEL_PATH', None)
            if ml_path:
                status['ml_enabled'] = os.path.exists(ml_path)
                status['ml_model_path'] = ml_path
        
        return status
