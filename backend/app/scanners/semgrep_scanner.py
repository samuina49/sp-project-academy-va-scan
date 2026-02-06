"""
Semgrep scanner integration for JavaScript/TypeScript vulnerability detection.
"""
import subprocess
import json
from pathlib import Path
from typing import List, Optional
import time

from app.models.scan_models import VulnerabilityFinding, FileScanResult, SeverityEnum


class SemgrepScanner:
    """Semgrep static analysis scanner for JavaScript/TypeScript"""
    
    def __init__(self, rules_path: Optional[str] = None, timeout: int = 120):
        """
        Initialize Semgrep scanner.
        
        Args:
            rules_path: Optional path to custom Semgrep rules directory
            timeout: Scan timeout in seconds
        """
        # Use custom rules if available, otherwise use default
        if rules_path is None:
            custom_rules = Path(__file__).parent.parent.parent / "rules" / "semgrep"
            if custom_rules.exists():
                self.rules_path = str(custom_rules)
                print(f"OK: Using custom Semgrep rules from: {self.rules_path}")
            else:
                self.rules_path = "auto"  # Use Semgrep registry
                print("ℹ Using Semgrep registry rules")
        else:
            self.rules_path = rules_path
        
        self.timeout = timeout
        self._check_available()
    
    def _check_available(self) -> bool:
        """Check if Semgrep is installed and available"""
        try:
            subprocess.run(
                ["semgrep", "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def scan_file(self, file_path: Path, language: str = "javascript") -> FileScanResult:
        """
        Scan a single JavaScript/TypeScript file with Semgrep.
        
        Args:
            file_path: Path to source file
            language: Language (javascript or typescript)
            
        Returns:
            FileScanResult with findings
        """
        start_time = time.time()
        
        # Choose appropriate ruleset based on language
        if self.rules_path:
            config = self.rules_path
        else:
            # Use Semgrep's default security rules
            config = "p/security-audit"
        
        # Build Semgrep command
        # Use absolute path to executable in venv/Scripts
        import sys
        import os
        semgrep_cmd = os.path.join(os.path.dirname(sys.executable), "semgrep.exe" if os.name == 'nt' else "semgrep")
        
        cmd = [
            semgrep_cmd,
            "--config", config,
            "--json",  # JSON output format
            "--quiet",  # Suppress progress messages
            str(file_path)
        ]
        
        try:
            # Run Semgrep
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse JSON output
            findings = self._parse_semgrep_output(result.stdout, file_path, language)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return FileScanResult(
                file_path=str(file_path),
                language=language,
                findings=findings,
                scan_duration_ms=duration_ms
            )
            
        except subprocess.TimeoutExpired:
            duration_ms = (time.time() - start_time) * 1000
            return FileScanResult(
                file_path=str(file_path),
                language=language,
                findings=[],
                scan_duration_ms=duration_ms
            )
        except Exception as e:
            # Re-raise to trigger fallback to SimplePatternScanner
            raise e
    
    def _parse_semgrep_output(
        self, 
        output: str, 
        file_path: Path,
        language: str
    ) -> List[VulnerabilityFinding]:
        """
        Parse Semgrep JSON output into VulnerabilityFinding objects.
        
        Args:
            output: Semgrep JSON output
            file_path: Path to the scanned file
            language: Programming language
            
        Returns:
            List of findings
        """
        findings = []
        
        try:
            data = json.loads(output)
            results = data.get("results", [])
            
            for result in results:
                # Map Semgrep severity to our SeverityEnum
                severity_str = result.get("extra", {}).get("severity", "WARNING").upper()
                severity_map = {
                    "ERROR": SeverityEnum.HIGH,
                    "WARNING": SeverityEnum.MEDIUM,
                    "INFO": SeverityEnum.LOW
                }
                severity = severity_map.get(severity_str, SeverityEnum.MEDIUM)
                
                # Extract CWE if available
                cwe_id = None
                metadata = result.get("extra", {}).get("metadata", {})
                cwe_list = metadata.get("cwe", [])
                if cwe_list and isinstance(cwe_list, list):
                    cwe_id = cwe_list[0] if cwe_list else None
                
                finding = VulnerabilityFinding(
                    tool="semgrep",
                    rule_id=result.get("check_id", ""),
                    severity=severity,
                    message=result.get("extra", {}).get("message", ""),
                    start_line=result.get("start", {}).get("line", 1),
                    end_line=result.get("end", {}).get("line", 1),
                    code_snippet=result.get("extra", {}).get("lines", "").strip(),
                    cwe_id=cwe_id
                )
                findings.append(finding)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty list
            pass
        except Exception:
            # Catch any other parsing errors
            pass
        
        return findings
    
    @staticmethod
    def is_available() -> bool:
        """Check if Semgrep is available in the system"""
        try:
            import sys
            import os
            semgrep_cmd = os.path.join(os.path.dirname(sys.executable), "semgrep.exe" if os.name == 'nt' else "semgrep")
            subprocess.run(
                [semgrep_cmd, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
