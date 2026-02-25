"""
Scanner orchestrator to route scans to appropriate tools.
"""
from pathlib import Path
from typing import List, Optional
import os
import re

from app.scanners.bandit_scanner import BanditScanner
from app.scanners.semgrep_scanner import SemgrepScanner
from app.models.scan_models import FileScanResult
from app.core.config import settings
from app.core.security import detect_language, should_ignore_path


# ---------------------------------------------------------------------------
# Tier-1 False-Positive Filter: pure import / require statements are never
# vulnerabilities on their own — only *use* of dangerous APIs matters.
# ---------------------------------------------------------------------------

# Python: `import x`, `from x import y`
_PY_IMPORT_RE = re.compile(
    r'^\s*(import\s+[\w.,\s]+|from\s+[\w.]+\s+import\s+[\w.,*\s()]+)\s*(#.*)?$'
)
# JS/TS CommonJS: `const x = require('y')`, `require('y')`
_JS_REQUIRE_RE = re.compile(
    r'''^\s*(?:[\w\s,{}]*=\s*)?require\s*\(\s*['"][^'"]+['"]\s*\)\s*;?\s*$'''
)
# JS/TS ES6: `import x from 'y'`, `import { x } from 'y'`, `import 'y'`
_JS_IMPORT_RE = re.compile(
    r"^\s*import\s+(?:[\w*{}\s,]+\s+from\s+)?['\"][^'\"]+['\"]\s*;?\s*$"
)

# Bandit rule IDs that solely warn about risky-import presence (not misuse)
_IMPORT_ONLY_BANDIT_RULES = frozenset({
    'B301',  # pickle.loads/Unpickler (import flag)
    'B401',  # import telnetlib
    'B402',  # import ftplib
    'B403',  # import pickle
    'B404',  # import subprocess
    'B405',  # import xml.etree
    'B406',  # import xml.sax
    'B407',  # import xml.expat
    'B408',  # import xml.minidom
    'B409',  # import xml.pulldom
    'B410',  # import lxml
    'B411',  # import xmlrpc
    'B412',  # import httpoxy
})


def is_false_positive_import(finding) -> bool:
    """
    Return True when a finding is a false-positive triggered by a bare
    import / require statement rather than actual misuse of the module.

    Covers:
    1. Bandit rules that only flag *importing* (not calling) risky modules.
    2. Python `import x` / `from x import y` snippets with no dangerous call.
    3. JS/TS CommonJS `require('x')` lines.
    4. JS/TS ES6 `import x from 'y'` lines.
    """
    rule_id = getattr(finding, 'rule_id', '') or ''
    if rule_id.upper() in _IMPORT_ONLY_BANDIT_RULES:
        return True

    snippet = getattr(finding, 'code_snippet', '') or ''
    non_empty = [ln for ln in snippet.splitlines() if ln.strip()]
    if not non_empty:
        return False

    def _line_is_import(line: str) -> bool:
        return bool(
            _PY_IMPORT_RE.match(line)
            or _JS_REQUIRE_RE.match(line)
            or _JS_IMPORT_RE.match(line)
        )

    if all(_line_is_import(ln) for ln in non_empty):
        return True

    return False


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
        
        # ── Tier-1 False-Positive Filter ──────────────────────────────────────
        if pattern_result:
            before = len(pattern_result.findings)
            pattern_result.findings = [
                f for f in pattern_result.findings
                if not is_false_positive_import(f)
            ]
            removed = before - len(pattern_result.findings)
            if removed:
                print(f"[Scanner] Filtered {removed} import false-positive(s)")

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
