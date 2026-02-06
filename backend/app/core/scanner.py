"""
Vulnerability Scanner Core Module

This module provides the core VulnerabilityScanner class used by the hybrid scan endpoint.
It bridges the gap between raw code input and file-based scanners (Semgrep, Bandit).
"""
import uuid
import os
from typing import Dict, Any, List

from app.scanners.semgrep_scanner import SemgrepScanner
from app.scanners.bandit_scanner import BanditScanner
from app.core.temp_manager import create_temp_file, cleanup_temp_file
from app.core.config import settings

class VulnerabilityScanner:
    """
    Scanner wrapper that handles temporary file creation and routing 
    to specific security tools.
    """
    
    def __init__(self):
        self.semgrep = SemgrepScanner(
            rules_path=settings.SEMGREP_RULES_PATH or None,
            timeout=settings.SCAN_TIMEOUT_SECONDS
        )
        self.bandit = BanditScanner(
            config_path=settings.BANDIT_CONFIG_PATH or None,
            timeout=settings.SCAN_TIMEOUT_SECONDS
        )

    def run_semgrep(self, code: str, language: str) -> Dict[str, Any]:
        """
        Run Semgrep on raw code string
        """
        if language not in ['javascript', 'typescript', 'python', 'java', 'go']:
            # Semgrep supports many, but let's stick to what we support in UI
            pass
            
        extension_map = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'jsx': '.jsx',
            'tsx': '.tsx'
        }
        ext = extension_map.get(language, '.txt')
        
        temp_file = None
        try:
            temp_file = create_temp_file(
                content=code,
                suffix=ext,
                prefix=f"semgrep_{uuid.uuid4()}_"
            )
            
            # Scan
            result = self.semgrep.scan_file(temp_file, language)
            
            # Convert to expected dictionary format
            return {
                'results': result.findings if result else []
            }
            
        except Exception as e:
            print(f"Error in run_semgrep: {e}")
            return {'results': []}
        finally:
            if temp_file:
                cleanup_temp_file(temp_file)

    def run_bandit(self, code: str) -> Dict[str, Any]:
        """
        Run Bandit on raw python code string
        """
        temp_file = None
        try:
            temp_file = create_temp_file(
                content=code,
                suffix='.py',
                prefix=f"bandit_{uuid.uuid4()}_"
            )
            
            # Scan
            result = self.bandit.scan_file(temp_file)
            
            # Convert to expected dictionary format
            return {
                'results': result.findings if result else []
            }
            
        except Exception as e:
            print(f"Error in run_bandit: {e}")
            return {'results': []}
        finally:
            if temp_file:
                cleanup_temp_file(temp_file)
