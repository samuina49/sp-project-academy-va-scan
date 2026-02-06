"""
Scanner orchestrator to route scans to appropriate tools.
"""
from pathlib import Path
from typing import List, Optional
import os

from app.scanners.bandit_scanner import BanditScanner
from app.scanners.semgrep_scanner import SemgrepScanner
from app.models.scan_models import FileScanResult
from app.core.config import settings
from app.core.security import detect_language, should_ignore_path


class ScannerOrchestrator:
    """Orchestrates scanning across multiple tools based on language"""
    
    def __init__(self):
        """Initialize scanners"""
        self.bandit = BanditScanner(
            config_path=settings.BANDIT_CONFIG_PATH or None,
            timeout=settings.SCAN_TIMEOUT_SECONDS
        )
        self.semgrep = SemgrepScanner(
            rules_path=settings.SEMGREP_RULES_PATH or None,
            timeout=settings.SCAN_TIMEOUT_SECONDS
        )
    
    def scan_file(self, file_path: Path, language: Optional[str] = None) -> Optional[FileScanResult]:
        """
        Scan a single file with the appropriate tool.
        
        Args:
            file_path: Path to the file to scan
            language: Language override (auto-detected if None)
            
        Returns:
            FileScanResult or None if language not supported
        """
        # Import SimplePatternScanner - always use for credential detection
        from app.scanners.simple_scanner import SimplePatternScanner
        simple_scanner = SimplePatternScanner()
        
        # Detect language if not provided
        if not language:
            language = detect_language(file_path.name)
        
        if not language:
            return None
        
        # Start with pattern scanner results (for credentials, etc.)
        pattern_result = simple_scanner.scan_file(file_path)
        pattern_findings = pattern_result.findings if pattern_result else []
        
        # Route to specialized scanner for additional checks
        if language == "python":
            # Try Bandit for additional checks
            try:
                bandit_result = self.bandit.scan_file(file_path)
                if bandit_result and bandit_result.findings:
                    # Merge findings, avoiding duplicates
                    pattern_result = self._merge_findings(pattern_result or bandit_result, bandit_result)
            except Exception as e:
                print(f"[Scanner] Bandit failed: {e}")

        elif language in ["javascript", "typescript"]:
            # Try Semgrep for additional checks
            try:
                semgrep_result = self.semgrep.scan_file(file_path, language)
                if semgrep_result and semgrep_result.findings:
                    # Merge findings, avoiding duplicates
                    pattern_result = self._merge_findings(pattern_result or semgrep_result, semgrep_result)
            except Exception as e:
                print(f"[Scanner] Semgrep failed: {e}")
        
        return pattern_result
    
    def _merge_findings(self, base_result: FileScanResult, new_result: FileScanResult) -> FileScanResult:
        """Merge findings from two results, avoiding duplicates based on line and CWE"""
        existing_keys = set()
        for f in base_result.findings:
            # Use start_line (from SimpleScanner) or line_number (from Bandit/Semgrep)
            line = getattr(f, 'start_line', getattr(f, 'line_number', 0))
            cwe = getattr(f, 'cwe_id', 'UNKNOWN')
            key = (line, cwe)
            existing_keys.add(key)
        
        for finding in new_result.findings:
            # Use start_line (from SimpleScanner) or line_number (from Bandit/Semgrep)
            line = getattr(finding, 'start_line', getattr(finding, 'line_number', 0))
            cwe = getattr(finding, 'cwe_id', 'UNKNOWN')
            key = (line, cwe)
            if key not in existing_keys:
                base_result.findings.append(finding)
                existing_keys.add(key)
        
        return base_result
    
    def scan_directory(
        self, 
        directory: Path, 
        recursive: bool = True
    ) -> List[FileScanResult]:
        """
        Scan all supported files in a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to scan subdirectories
            
        Returns:
            List of FileScanResults
        """
        results = []
        
        # Get all files
        if recursive:
            files = self._get_files_recursive(directory)
        else:
            files = [f for f in directory.iterdir() if f.is_file()]
        
        # Scan each file
        for file_path in files:
            # Skip ignored paths
            if should_ignore_path(file_path, settings.IGNORED_DIRS):
                continue
            
            # Detect language
            language = detect_language(file_path.name)
            if not language:
                continue
            
            # Scan file
            result = self.scan_file(file_path, language)
            if result:
                results.append(result)
        
        return results
    
    def _get_files_recursive(self, directory: Path) -> List[Path]:
        """
        Recursively get all files in a directory.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of file paths
        """
        files = []
        
        try:
            for entry in directory.iterdir():
                # Skip ignored directories
                if entry.is_dir():
                    if entry.name not in settings.IGNORED_DIRS:
                        files.extend(self._get_files_recursive(entry))
                else:
                    files.append(entry)
        except PermissionError:
            # Skip directories we don't have permission to read
            pass
        
        return files
    
    @staticmethod
    def get_available_scanners() -> dict:
        """
        Check which scanners are available.
        
        Returns:
            Dictionary with scanner availability status
        """
        return {
            "bandit": BanditScanner.is_available(),
            "semgrep": SemgrepScanner.is_available()
        }
