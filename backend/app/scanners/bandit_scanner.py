"""
Bandit scanner integration for Python vulnerability detection.
"""
import subprocess
import json
from pathlib import Path
from typing import List, Optional
import time

from app.models.scan_models import VulnerabilityFinding, FileScanResult, SeverityEnum


class BanditScanner:
    """Bandit static analysis scanner for Python"""
    
    def __init__(self, config_path: Optional[str] = None, timeout: int = 120):
        """
        Initialize Bandit scanner.
        
        Args:
            config_path: Optional path to Bandit configuration file
            timeout: Scan timeout in seconds
        """
        self.config_path = config_path
        self.timeout = timeout
        self._check_available()
    
    def _check_available(self) -> bool:
        """Check if Bandit is installed and available"""
        try:
            import sys
            import os
            bandit_cmd = os.path.join(os.path.dirname(sys.executable), "bandit.exe" if os.name == 'nt' else "bandit")
            subprocess.run(
                [bandit_cmd, "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def scan_file(self, file_path: Path) -> FileScanResult:
        """
        Scan a single Python file with Bandit.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            FileScanResult with findings
        """
        start_time = time.time()
        
        # Build Bandit command
        # Use absolute path to executable in venv/Scripts
        import sys
        import os
        bandit_cmd = os.path.join(os.path.dirname(sys.executable), "bandit.exe" if os.name == 'nt' else "bandit")
        
        cmd = [
            bandit_cmd,
            "-f", "json",  # JSON output format
            "-ll",  # Report only medium and high severity
            str(file_path)
        ]
        
        if self.config_path:
            cmd.extend(["-c", self.config_path])
        
        try:
            # Run Bandit
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Parse JSON output
            findings = self._parse_bandit_output(result.stdout, file_path)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return FileScanResult(
                file_path=str(file_path),
                language="python",
                findings=findings,
                scan_duration_ms=duration_ms
            )
            
        except subprocess.TimeoutExpired:
            duration_ms = (time.time() - start_time) * 1000
            return FileScanResult(
                file_path=str(file_path),
                language="python",
                findings=[],
                scan_duration_ms=duration_ms
            )
        except Exception as e:
            # Re-raise to trigger fallback to SimplePatternScanner
            raise e
    
    def _parse_bandit_output(
        self, 
        output: str, 
        file_path: Path
    ) -> List[VulnerabilityFinding]:
        """
        Parse Bandit JSON output into VulnerabilityFinding objects.
        
        Args:
            output: Bandit JSON output
            file_path: Path to the scanned file
            
        Returns:
            List of findings
        """
        findings = []
        
        try:
            data = json.loads(output)
            results = data.get("results", [])
            
            for result in results:
                # Map Bandit severity to our SeverityEnum
                severity_map = {
                    "LOW": SeverityEnum.LOW,
                    "MEDIUM": SeverityEnum.MEDIUM,
                    "HIGH": SeverityEnum.HIGH
                }
                severity = severity_map.get(
                    result.get("issue_severity", "MEDIUM"),
                    SeverityEnum.MEDIUM
                )
                
                finding = VulnerabilityFinding(
                    tool="bandit",
                    rule_id=result.get("test_id", ""),
                    severity=severity,
                    message=result.get("issue_text", ""),
                    start_line=result.get("line_number", 1),
                    end_line=result.get("line_number", 1),
                    code_snippet=result.get("code", "").strip(),
                    cwe_id=result.get("cwe", {}).get("id") if isinstance(result.get("cwe"), dict) else None
                )
                findings.append(finding)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Bandit JSON parsing failed: {e}")
            pass
        except (KeyError, AttributeError, TypeError) as e:
            logger.warning(f"Bandit result parsing error: {e}")
            pass
        
        return findings
    
    @staticmethod
    def is_available() -> bool:
        """Check if Bandit is available in the system"""
        try:
            subprocess.run(
                ["bandit", "--version"],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
