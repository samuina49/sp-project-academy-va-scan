"""
Script 1: Preprocessing & Transpilation Module
Production-Ready Code for Senior Project: AI-based Vulnerability Scanner

Purpose: Recursively scan project directories, transpile TypeScript to JavaScript,
         and preprocess code for feature extraction.

Author: Senior Project - AI-based Vulnerability Scanner
Date: 2026-01-25
"""

import os
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Handles preprocessing and transpilation for Python, JavaScript, and TypeScript files.
    
    Key Features:
    - Recursive project scanning
    - TypeScript to JavaScript transpilation using tsc
    - Code cleaning (remove comments, whitespace)
    - File mapping preservation
    """
    
    SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts'}
    
    def __init__(self, tsc_path: str = 'tsc'):
        """
        Initialize preprocessing pipeline.
        
        Args:
            tsc_path: Path to TypeScript compiler (default: 'tsc' in PATH)
        """
        self.tsc_path = tsc_path
        self.temp_dir = tempfile.mkdtemp(prefix='vuln_scanner_')
        self.file_mappings: Dict[str, str] = {}  # temporary_path -> original_path
        
        # Verify tsc is available
        self._verify_tsc()
    
    def _verify_tsc(self) -> bool:
        """Verify TypeScript compiler is available"""
        try:
            result = subprocess.run(
                [self.tsc_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"TypeScript compiler found: {result.stdout.strip()}")
                return True
            else:
                logger.warning("TypeScript compiler not found. .ts files will be skipped.")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"TypeScript compiler check failed: {e}")
            return False
    
    def process_project(self, directory_path: str) -> List[Dict[str, str]]:
        """
        Recursively scan directory and process all supported files.
        
        Args:
            directory_path: Root directory to scan
            
        Returns:
            List of processed file information:
            [{
                'original_path': str,
                'processed_path': str,
                'language': str,  # 'python', 'javascript', 'typescript'
                'cleaned_code': str
            }]
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        processed_files = []
        
        # Recursively find all supported files
        for file_path in self._scan_directory(directory):
            try:
                processed_file = self._process_file(file_path)
                if processed_file:
                    processed_files.append(processed_file)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_files)} files from {directory_path}")
        return processed_files
    
    def _scan_directory(self, directory: Path) -> List[Path]:
        """
        Recursively scan directory for supported files.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of file paths
        """
        files = []
        
        # Directories to skip
        skip_dirs = {
            'node_modules', 'venv', '.venv', '__pycache__', 
            '.git', 'dist', 'build', 'coverage', '.next'
        }
        
        for item in directory.rglob('*'):
            # Skip directories in skip list
            if any(skip_dir in item.parts for skip_dir in skip_dirs):
                continue
            
            # Check if file has supported extension
            if item.is_file() and item.suffix in self.SUPPORTED_EXTENSIONS:
                files.append(item)
        
        return files
    
    def _process_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """
        Process a single file (transpile if .ts, clean code).
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with processed file information
        """
        language = self._detect_language(file_path)
        
        # Read original file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_code = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None
        
        # Handle TypeScript transpilation
        if language == 'typescript':
            transpiled_code = self._transpile_typescript(file_path, original_code)
            if transpiled_code is None:
                logger.warning(f"Failed to transpile {file_path}, skipping")
                return None
            code_to_clean = transpiled_code
            final_language = 'javascript'  # After transpilation
        else:
            code_to_clean = original_code
            final_language = language
        
        # Clean the code
        cleaned_code = self._clean_code(code_to_clean, final_language)
        
        return {
            'original_path': str(file_path),
            'language': final_language,
            'cleaned_code': cleaned_code,
            'original_code': original_code
        }
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript'
        }
        return extension_map.get(file_path.suffix, 'unknown')
    
    def _transpile_typescript(self, file_path: Path, code: str) -> Optional[str]:
        """
        Transpile TypeScript to JavaScript using tsc.
        
        This addresses data scarcity issues mentioned in thesis by converting
        TypeScript samples to JavaScript for unified analysis.
        
        Args:
            file_path: Original TypeScript file path
            code: TypeScript source code
            
        Returns:
            Transpiled JavaScript code or None if failed
        """
        # Create temporary TypeScript file
        temp_ts_file = Path(self.temp_dir) / f"{file_path.stem}_temp.ts"
        temp_js_file = temp_ts_file.with_suffix('.js')
        
        try:
            # Write TypeScript code to temporary file
            with open(temp_ts_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Run TypeScript compiler
            result = subprocess.run(
                [
                    self.tsc_path,
                    str(temp_ts_file),
                    '--target', 'ES2020',
                    '--module', 'commonjs',
                    '--outDir', str(self.temp_dir),
                    '--skipLibCheck',
                    '--allowJs',
                    '--noEmit', 'false'
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if transpilation succeeded
            if temp_js_file.exists():
                with open(temp_js_file, 'r', encoding='utf-8') as f:
                    transpiled_code = f.read()
                
                # Store mapping
                self.file_mappings[str(temp_js_file)] = str(file_path)
                
                logger.info(f"Successfully transpiled: {file_path.name}")
                return transpiled_code
            else:
                logger.warning(f"Transpilation failed for {file_path}: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Transpilation timeout for {file_path}")
            return None
        except Exception as e:
            logger.error(f"Transpilation error for {file_path}: {e}")
            return None
        finally:
            # Cleanup temporary files
            if temp_ts_file.exists():
                temp_ts_file.unlink()
            if temp_js_file.exists():
                temp_js_file.unlink()
    
    def _clean_code(self, code: str, language: str) -> str:
        """
        Clean code by removing comments and excessive whitespace.
        
        Preprocessing step as per thesis methodology.
        
        Args:
            code: Source code
            language: Programming language ('python' or 'javascript')
            
        Returns:
            Cleaned code
        """
        if language == 'python':
            cleaned = self._clean_python(code)
        elif language in ['javascript', 'typescript']:
            cleaned = self._clean_javascript(code)
        else:
            cleaned = code
        
        # Remove excessive blank lines
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def _clean_python(self, code: str) -> str:
        """Remove Python comments and docstrings"""
        lines = []
        in_multiline_string = False
        multiline_char = None
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Handle multiline strings/docstrings
            if '"""' in line or "'''" in line:
                if '"""' in line:
                    multiline_char = '"""'
                else:
                    multiline_char = "'''"
                
                count = line.count(multiline_char)
                if count == 2:  # Single-line docstring
                    # Remove it
                    line = re.sub(r'""".*?"""', '', line)
                    line = re.sub(r"'''.*?'''", '', line)
                elif count == 1:
                    in_multiline_string = not in_multiline_string
                    if not in_multiline_string:
                        continue
                    else:
                        continue
            
            # Skip if in multiline string
            if in_multiline_string:
                continue
            
            # Remove inline comments
            if '#' in line:
                # Don't remove # inside strings
                if not ('"' in line or "'" in line):
                    line = line.split('#')[0]
            
            if line.strip():
                lines.append(line.rstrip())
        
        return '\n'.join(lines)
    
    def _clean_javascript(self, code: str) -> str:
        """Remove JavaScript/TypeScript comments"""
        # Remove single-line comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        return code
    
    def cleanup(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Example Usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PreprocessingPipeline()
    
    # Process a project directory
    project_dir = "./sample_project"
    
    try:
        processed_files = pipeline.process_project(project_dir)
        
        # Display results
        print(f"\nProcessed {len(processed_files)} files:\n")
        for file_info in processed_files[:5]:  # Show first 5
            print(f"File: {file_info['original_path']}")
            print(f"Language: {file_info['language']}")
            print(f"Code length: {len(file_info['cleaned_code'])} characters")
            print("-" * 80)
    
    finally:
        # Always cleanup
        pipeline.cleanup()
