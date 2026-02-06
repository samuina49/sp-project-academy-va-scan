"""
Ensemble Layer for Hybrid Vulnerability Detection

Combines results from:
1. Pattern Matching: Semgrep + Bandit  
2. Deep Learning: GNN + LSTM

Provides unified results with confidence scoring and deduplication.
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ...core.owasp_mapper import OWASPMapper


class DetectionSource(Enum):
    """Source of vulnerability detection"""
    SEMGREP = "semgrep"
    BANDIT = "bandit"
    ML_MODEL = "ml"
    HYBRID = "hybrid"


class Severity(Enum):
    """Unified severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class Finding:
    """Unified vulnerability finding"""
    line: int
    vulnerability_type: str
    severity: Severity
    confidence: float  # 0.0 - 1.0
    sources: List[DetectionSource]
    code_snippet: str
    explanation: str
    remediation: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    
    # Source-specific details
    semgrep_rule: Optional[str] = None
    bandit_test: Optional[str] = None
    ml_probability: Optional[float] = None


class EnsembleCombiner:
    """
    Combine pattern matching and ML results into unified findings
    """
    
    def __init__(self, ml_weight: float = 0.4):
        """
        Initialize ensemble combiner
        
        Args:
            ml_weight: Weight for ML predictions (0.0-1.0)
                      Pattern matching gets (1 - ml_weight)
        """
        self.ml_weight = ml_weight
        self.pattern_weight = 1.0 - ml_weight
        
        # Initialize OWASP mapper
        self.owasp_mapper = OWASPMapper()
    
    def combine(
        self,
        semgrep_results: List[Dict],
        bandit_results: List[Dict],
        ml_prediction: Dict,
        code: str
    ) -> List[Finding]:
        """
        Combine all detection sources
        
        Args:
            semgrep_results: Results from Semgrep
            bandit_results: Results from Bandit
            ml_prediction: Prediction from ML model
            code: Source code analyzed
            
        Returns:
            List of unified findings
        """
        findings = []
        
        # Convert Semgrep findings
        for result in semgrep_results:
            finding = self._convert_semgrep(result, code)
            findings.append(finding)
        
        # Convert Bandit findings
        for result in bandit_results:
            finding = self._convert_bandit(result, code)
            findings.append(finding)
        
        # Add ML prediction (with sanity checks)
        if ml_prediction.get('vulnerable'):
            # Filter out false positives: plain import statements
            code_stripped = code.strip()
            is_safe_import = (
                code_stripped.startswith('import ') or 
                code_stripped.startswith('from ') and ' import ' in code_stripped
            ) and not any([
                'eval(' in code,
                'exec(' in code,
                'os.system(' in code,
                '__import__(' in code,
                'subprocess' in code
            ])
            
            # Only add ML finding if it's not a safe import statement
            if not is_safe_import:
                finding = self._convert_ml(ml_prediction, code)
                findings.append(finding)
            else:
                print(f"[ENSEMBLE] Filtered ML false positive: {code_stripped[:60]}...")
        
        # Deduplicate and merge overlapping findings
        findings = self._deduplicate(findings)
        
        # Enhance confidence scores
        findings = self._calculate_confidence(findings, ml_prediction)
        
        # Sort by severity and confidence
        findings.sort(key=lambda f: (
            self._severity_priority(f.severity),
            -f.confidence
        ))
        
        return findings
    
    def _convert_semgrep(self, result: Dict, code: str) -> Finding:
        """Convert Semgrep result to unified format"""
        rule_id = result.get('check_id', '')
        return Finding(
            line=result.get('start', {}).get('line', 0),
            vulnerability_type=self._extract_vuln_type(rule_id),
            severity=self._map_semgrep_severity(result.get('extra', {}).get('severity', 'INFO')),
            confidence=0.85,  # Semgrep is generally high confidence
            sources=[DetectionSource.SEMGREP],
            code_snippet=self._extract_snippet(code, result.get('start', {}).get('line', 0)),
            explanation=result.get('extra', {}).get('message', ''),
            remediation=None,
            semgrep_rule=rule_id,
            cwe_id=self._extract_cwe(result),
            owasp_category=self.owasp_mapper.map_semgrep_rule(rule_id)
        )
    
    def _convert_bandit(self, result: Dict, code: str) -> Finding:
        """Convert Bandit result to unified format"""
        test_id = result.get('test_id', '')
        test_name = result.get('test_name', 'Unknown')
        return Finding(
            line=result.get('line_number', 0),
            vulnerability_type=test_name,
            severity=self._map_bandit_severity(result.get('issue_severity', 'LOW')),
            confidence=self._map_bandit_confidence(result.get('issue_confidence', 'LOW')),
            sources=[DetectionSource.BANDIT],
            code_snippet=self._extract_snippet(code, result.get('line_number', 0)),
            explanation=result.get('issue_text', ''),
            remediation=None,
            bandit_test=test_id,
            cwe_id=result.get('issue_cwe', {}).get('id') if isinstance(result.get('issue_cwe'), dict) else None,
            owasp_category=self.owasp_mapper.map_bandit_test(test_id)
        )
    
    def _convert_ml(self, prediction: Dict, code: str) -> Finding:
        """Convert ML prediction to unified format"""
        vuln_type = "AI-Detected Vulnerability"
        return Finding(
            line=1,  # ML doesn't have specific line numbers yet
            vulnerability_type=vuln_type,
            severity=self._map_ml_severity(prediction.get('confidence', 0.5)),
            confidence=prediction.get('confidence', 0.0),
            sources=[DetectionSource.ML_MODEL],
            code_snippet=code[:200],  # First 200 chars
            explanation=f"AI model detected potential vulnerability with {prediction.get('confidence', 0):.0%} confidence",
            remediation=None,
            ml_probability=prediction.get('probabilities', {}).get('vulnerable', 0.0),
            owasp_category=self.owasp_mapper.map_vulnerability_type(vuln_type)
        )
    
    def _deduplicate(self, findings: List[Finding]) -> List[Finding]:
        """
        Deduplicate findings that are likely the same issue
        
        Merge findings if:
        - Same line number (±2 lines)
        - Similar vulnerability type
        """
        if not findings:
            return []
        
        deduplicated = []
        processed_lines: Set[int] = set()
        
        for finding in findings:
            # Check if we've already processed a nearby line
            nearby_processed = any(
                abs(finding.line - pl) <= 2 
                for pl in processed_lines
            )
            
            if nearby_processed:
                # Merge with existing finding
                for existing in deduplicated:
                    if abs(existing.line - finding.line) <= 2:
                        # Merge sources
                        existing.sources.extend(finding.sources)
                        existing.sources = list(set(existing.sources))
                        
                        # Update severity (use higher)
                        if self._severity_priority(finding.severity) < self._severity_priority(existing.severity):
                            existing.severity = finding.severity
                        
                        # Keep higher confidence
                        existing.confidence = max(existing.confidence, finding.confidence)
                        
                        # Add detection details
                        if finding.semgrep_rule:
                            existing.semgrep_rule = finding.semgrep_rule
                        if finding.bandit_test:
                            existing.bandit_test = finding.bandit_test
                        if finding.ml_probability:
                            existing.ml_probability = finding.ml_probability
                        
                        break
            else:
                deduplicated.append(finding)
                processed_lines.add(finding.line)
        
        return deduplicated
    
    def _calculate_confidence(
        self,
        findings: List[Finding],
        ml_prediction: Dict
    ) -> List[Finding]:
        """
        Recalculate confidence scores based on agreement
        
        Higher confidence when multiple sources agree
        """
        for finding in findings:
            num_sources = len(finding.sources)
            
            # Base confidence
            base = finding.confidence
            
            # Bonus for multiple sources
            if num_sources == 2:
                finding.confidence = min(base + 0.10, 1.0)
            elif num_sources >= 3:
                finding.confidence = min(base + 0.15, 1.0)
                finding.sources.append(DetectionSource.HYBRID)
            
            # Bonus if ML agrees with pattern matching
            if DetectionSource.ML_MODEL in finding.sources and num_sources > 1:
                finding.confidence = min(finding.confidence + 0.05, 1.0)
        
        return findings
    
    def _extract_snippet(self, code: str, line: int, context: int = 1) -> str:
        """Extract code snippet around line"""
        lines = code.split('\n')
        start = max(0, line - context - 1)
        end = min(len(lines), line + context)
        snippet_lines = lines[start:end]
        return '\n'.join(snippet_lines)
    
    def _extract_vuln_type(self, check_id: str) -> str:
        """Extract human-readable vulnerability type from Semgrep rule ID"""
        # Convert semgrep.rule-id to readable format
        parts = check_id.split('.')
        if len(parts) > 0:
            return parts[-1].replace('-', ' ').title()
        return "Unknown Vulnerability"
    
    def _extract_cwe(self, result: Dict) -> Optional[str]:
        """Extract CWE ID from Semgrep result"""
        metadata = result.get('extra', {}).get('metadata', {})
        cwe = metadata.get('cwe')
        if cwe and isinstance(cwe, list):
            return cwe[0] if cwe else None
        return str(cwe) if cwe else None
    
    def _map_semgrep_severity(self, severity: str) -> Severity:
        """Map Semgrep severity to unified scale (supports both Semgrep and SimplePatternScanner formats)"""
        severity_upper = severity.upper()
        
        # Semgrep format (ERROR/WARNING/INFO)
        semgrep_mapping = {
            'ERROR': Severity.HIGH,
            'WARNING': Severity.MEDIUM,
            'INFO': Severity.LOW
        }
        
        # SimplePatternScanner format (CRITICAL/HIGH/MEDIUM/LOW/INFO)
        simple_scanner_mapping = {
            'CRITICAL': Severity.CRITICAL,
            'HIGH': Severity.HIGH,
            'MEDIUM': Severity.MEDIUM,
            'LOW': Severity.LOW,
            'INFO': Severity.INFO
        }
        
        # Try SimpleScanner format first (more specific), then Semgrep format
        if severity_upper in simple_scanner_mapping:
            return simple_scanner_mapping[severity_upper]
        elif severity_upper in semgrep_mapping:
            return semgrep_mapping[severity_upper]
        else:
            # Unknown severity - default to MEDIUM to avoid underestimating risks
            print(f"[ENSEMBLE] WARNING: Unknown severity '{severity}', defaulting to MEDIUM")
            return Severity.MEDIUM
    
    def _map_bandit_severity(self, severity: str) -> Severity:
        """Map Bandit severity to unified scale"""
        mapping = {
            'CRITICAL': Severity.CRITICAL,
            'HIGH': Severity.HIGH,
            'MEDIUM': Severity.MEDIUM,
            'LOW': Severity.LOW,
            'INFO': Severity.INFO
        }
        # Default to MEDIUM instead of LOW to avoid underestimating risks
        return mapping.get(severity.upper(), Severity.MEDIUM)
    
    def _map_bandit_confidence(self, confidence: str) -> float:
        """Map Bandit confidence to 0-1 scale"""
        mapping = {
            'HIGH': 0.9,
            'MEDIUM': 0.7,
            'LOW': 0.5
        }
        return mapping.get(confidence.upper(), 0.5)
    
    def _map_ml_severity(self, confidence: float) -> Severity:
        """Map ML confidence to severity"""
        if confidence >= 0.9:
            return Severity.HIGH
        elif confidence >= 0.7:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _severity_priority(self, severity: Severity) -> int:
        """Get priority number for severity (lower = more severe)"""
        priorities = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4
        }
        return priorities.get(severity, 99)


# Example usage
if __name__ == '__main__':
    # Sample results
    semgrep = [
        {
            'check_id': 'python.lang.security.sql-injection',
            'start': {'line': 15},
            'extra': {'severity': 'ERROR', 'message': 'SQL injection detected'}
        }
    ]
    
    bandit = [
        {
            'line_number': 15,
            'test_name': 'SQL Injection',
            'issue_severity': 'HIGH',
            'issue_confidence': 'HIGH',
            'test_id': 'B608',
            'issue_text': 'Possible SQL injection'
        }
    ]
    
    ml = {
        'vulnerable': True,
        'confidence': 0.89,
        'probabilities': {'safe': 0.11, 'vulnerable': 0.89}
    }
    
    code = 'query = f"SELECT * FROM users WHERE id={user_id}"'
    
    # Combine
    combiner = EnsembleCombiner()
    findings = combiner.combine(semgrep, bandit, ml, code)
    
    print(f"Found {len(findings)} unique findings")
    for f in findings:
        print(f"\nLine {f.line}: {f.vulnerability_type}")
        print(f"  Severity: {f.severity.value}")
        print(f"  Confidence: {f.confidence:.0%}")
        print(f"  Sources: {[s.value for s in f.sources]}")
