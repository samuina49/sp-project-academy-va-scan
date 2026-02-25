"""
Code Metrics & Taint Analysis Feature Extractor
=================================================
Extracts vulnerability-specific features from code that can help distinguish
safe from vulnerable code better than pure structural features.

Features:
1. Complexity Metrics: Cyclomatic complexity, nesting depth, LOC
2. Taint Analysis: User input → dangerous function flows
3. Security Patterns: SQL injection, XSS, command injection patterns

Author: Senior Project - AI-based Vulnerability Scanner
Date: 2026-02-07
"""

import re
import ast
from typing import Dict, List, Tuple
import numpy as np


class CodeMetricsExtractor:
    """Extract code complexity and security-relevant features"""
    
    # Dangerous function patterns by vulnerability type
    DANGEROUS_FUNCTIONS = {
        'sql_injection': [
            'execute', 'executemany', 'raw', 'cursor.execute',
            'query', 'filter', 'SELECT', 'INSERT', 'UPDATE', 'DELETE'
        ],
        'command_injection': [
            'os.system', 'subprocess', 'exec', 'eval', 'compile',
            'os.popen', 'commands.', 'shell=True'
        ],
        'xss': [
            'render_template', 'render', 'innerHTML', 'document.write',
            'Response', 'HttpResponse'
        ],
        'path_traversal': [
            'open', 'file', 'read', 'write', 'os.path.join',
            'pathlib', 'Path'
        ]
    }
    
    # User input sources
    USER_INPUT_SOURCES = [
        'request.', 'request[', 'input(', 'argv', 'args.',
        'params', 'query', 'cookies', 'headers',
        'GET', 'POST', 'json()', 'form'
    ]
    
    # Sanitization/validation functions
    SANITIZATION_FUNCTIONS = [
        'escape', 'sanitize', 'validate', 'clean',
        'strip', 'filter', 'check', 'verify',
        'parameterized', 'prepared', 'bind'
    ]
    
    def __init__(self):
        """Initialize extractor"""
        pass
    
    def extract_all_features(self, code: str, language: str = 'python') -> np.ndarray:
        """
        Extract all features from code.
        
        Args:
            code: Source code string
            language: Programming language ('python' or 'javascript')
            
        Returns:
            Feature vector of shape (20,)
        """
        features = []
        
        # 1. Complexity Metrics (6 features)
        complexity = self.extract_complexity_metrics(code, language)
        features.extend([
            complexity['cyclomatic_complexity'],
            complexity['max_nesting_depth'],
            complexity['lines_of_code'],
            complexity['num_functions'],
            complexity['num_variables'],
            complexity['num_conditionals']
        ])
        
        # 2. Taint Analysis (8 features)
        taint = self.extract_taint_features(code)
        features.extend([
            taint['has_user_input'],
            taint['has_sql_danger'],
            taint['has_command_danger'],
            taint['has_xss_danger'],
            taint['has_path_danger'],
            taint['has_sanitization'],
            taint['input_to_danger_distance'],
            taint['sanitization_coverage']
        ])
        
        # 3. Security Patterns (6 features)
        patterns = self.extract_security_patterns(code)
        features.extend([
            patterns['has_string_concat_sql'],
            patterns['has_dynamic_eval'],
            patterns['has_shell_execution'],
            patterns['has_file_operations'],
            patterns['has_network_operations'],
            patterns['complexity_risk_score']
        ])
        
        return np.array(features, dtype=np.float32)
    
    def extract_complexity_metrics(self, code: str, language: str) -> Dict[str, float]:
        """
        Extract code complexity metrics.
        
        Returns:
            Dict with complexity metrics
        """
        lines = code.split('\n')
        
        # Lines of code (non-empty, non-comment)
        loc = sum(1 for line in lines 
                  if line.strip() and not line.strip().startswith('#') 
                  and not line.strip().startswith('//'))
        
        # Cyclomatic complexity (approximate by counting decision points)
        complexity = 1  # Base complexity
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('else:')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('and ')
        complexity += code.count('or ')
        complexity += code.count('? ') if language == 'javascript' else 0
        
        # Max nesting depth
        max_depth = 0
        current_depth = 0
        for line in lines:
            indent = len(line) - len(line.lstrip())
            if language == 'python':
                current_depth = indent // 4  # 4 spaces = 1 level
            else:
                current_depth = line.count('{') - line.count('}')
            max_depth = max(max_depth, current_depth)
        
        # Count functions
        if language == 'python':
            num_functions = code.count('def ') + code.count('async def ')
        else:
            num_functions = code.count('function ') + code.count('=>')
        
        # Count variables (approximate)
        if language == 'python':
            num_variables = code.count(' = ') + code.count('self.')
        else:
            num_variables = code.count('var ') + code.count('let ') + code.count('const ')
        
        # Count conditionals
        num_conditionals = code.count('if ') + code.count('elif ') + code.count('else')
        
        return {
            'cyclomatic_complexity': min(complexity / 10.0, 10.0),  # Normalize to 0-10
            'max_nesting_depth': min(max_depth / 5.0, 5.0),  # Normalize to 0-5
            'lines_of_code': min(loc / 100.0, 10.0),  # Normalize to 0-10
            'num_functions': min(num_functions / 5.0, 5.0),
            'num_variables': min(num_variables / 10.0, 5.0),
            'num_conditionals': min(num_conditionals / 10.0, 5.0)
        }
    
    def extract_taint_features(self, code: str) -> Dict[str, float]:
        """
        Extract taint analysis features (data flow from input to dangerous sinks).
        
        Returns:
            Dict with taint features
        """
        code_lower = code.lower()
        
        # Check for user input sources
        has_user_input = float(any(source in code for source in self.USER_INPUT_SOURCES))
        
        # Check for dangerous function categories
        has_sql_danger = float(any(func in code for func in self.DANGEROUS_FUNCTIONS['sql_injection']))
        has_command_danger = float(any(func in code for func in self.DANGEROUS_FUNCTIONS['command_injection']))
        has_xss_danger = float(any(func in code for func in self.DANGEROUS_FUNCTIONS['xss']))
        has_path_danger = float(any(func in code for func in self.DANGEROUS_FUNCTIONS['path_traversal']))
        
        # Check for sanitization
        has_sanitization = float(any(func in code_lower for func in self.SANITIZATION_FUNCTIONS))
        
        # Estimate input-to-danger distance (line distance)
        input_to_danger_distance = self._calculate_taint_distance(code)
        
        # Sanitization coverage (are dangerous operations protected?)
        total_dangers = has_sql_danger + has_command_danger + has_xss_danger + has_path_danger
        if total_dangers > 0 and has_sanitization:
            sanitization_coverage = 0.5  # Partial protection assumed
        elif total_dangers > 0 and not has_sanitization:
            sanitization_coverage = 0.0  # No protection
        else:
            sanitization_coverage = 1.0  # No dangers or fully protected
        
        return {
            'has_user_input': has_user_input,
            'has_sql_danger': has_sql_danger,
            'has_command_danger': has_command_danger,
            'has_xss_danger': has_xss_danger,
            'has_path_danger': has_path_danger,
            'has_sanitization': has_sanitization,
            'input_to_danger_distance': input_to_danger_distance,
            'sanitization_coverage': sanitization_coverage
        }
    
    def _calculate_taint_distance(self, code: str) -> float:
        """
        Calculate approximate distance from input to dangerous function.
        Lower distance = higher risk.
        
        Returns:
            Normalized distance (0 = very close, 1 = far apart)
        """
        lines = code.split('\n')
        
        # Find line numbers of inputs and dangers
        input_lines = []
        danger_lines = []
        
        for i, line in enumerate(lines):
            if any(source in line for source in self.USER_INPUT_SOURCES):
                input_lines.append(i)
            
            all_dangers = []
            for danger_list in self.DANGEROUS_FUNCTIONS.values():
                all_dangers.extend(danger_list)
            
            if any(danger in line for danger in all_dangers):
                danger_lines.append(i)
        
        # Calculate minimum distance
        if not input_lines or not danger_lines:
            return 1.0  # No taint flow
        
        min_distance = float('inf')
        for inp in input_lines:
            for dan in danger_lines:
                distance = abs(dan - inp)
                min_distance = min(min_distance, distance)
        
        # Normalize: 0 lines = 0.0 (very dangerous), 20+ lines = 1.0 (safer)
        normalized_distance = min(min_distance / 20.0, 1.0)
        
        return normalized_distance
    
    def extract_security_patterns(self, code: str) -> Dict[str, float]:
        """
        Extract specific security vulnerability patterns.
        
        Returns:
            Dict with security pattern features
        """
        # SQL injection patterns (string concatenation in SQL)
        has_string_concat_sql = float(
            bool(re.search(r'(SELECT|INSERT|UPDATE|DELETE).*[\+%]', code, re.IGNORECASE)) or
            bool(re.search(r'f["\'].*SELECT', code, re.IGNORECASE)) or
            bool(re.search(r'\.format\(.*\).*execute', code, re.IGNORECASE))
        )
        
        # Dynamic code execution
        has_dynamic_eval = float(
            'eval(' in code or 'exec(' in code or 
            'Function(' in code or '__import__' in code
        )
        
        # Shell execution
        has_shell_execution = float(
            'os.system' in code or 'subprocess' in code or
            'shell=True' in code or 'os.popen' in code
        )
        
        # File operations
        has_file_operations = float(
            'open(' in code or 'file(' in code or
            'read(' in code or 'write(' in code
        )
        
        # Network operations
        has_network_operations = float(
            'requests.' in code or 'urllib' in code or
            'http' in code.lower() or 'socket' in code
        )
        
        # Overall complexity risk score
        risk_score = (
            has_string_concat_sql * 2.0 +
            has_dynamic_eval * 3.0 +
            has_shell_execution * 2.5 +
            has_file_operations * 1.0 +
            has_network_operations * 0.5
        ) / 9.0  # Normalize to 0-1
        
        return {
            'has_string_concat_sql': has_string_concat_sql,
            'has_dynamic_eval': has_dynamic_eval,
            'has_shell_execution': has_shell_execution,
            'has_file_operations': has_file_operations,
            'has_network_operations': has_network_operations,
            'complexity_risk_score': min(risk_score, 1.0)
        }


# Test the extractor
if __name__ == "__main__":
    extractor = CodeMetricsExtractor()
    
    # Test with vulnerable code
    vulnerable_code = """
def login(request):
    username = request.GET['username']
    password = request.GET['password']
    query = "SELECT * FROM users WHERE username='" + username + "' AND password='" + password + "'"
    cursor.execute(query)
    return result
"""
    
    # Test with safe code
    safe_code = """
def login(request):
    username = request.GET['username']
    password = request.GET['password']
    query = "SELECT * FROM users WHERE username=? AND password=?"
    cursor.execute(query, (username, password))
    return result
"""
    
    print("Testing Code Metrics Extractor")
    print("=" * 60)
    
    vuln_features = extractor.extract_all_features(vulnerable_code, 'python')
    safe_features = extractor.extract_all_features(safe_code, 'python')
    
    print(f"\nVulnerable code features shape: {vuln_features.shape}")
    print(f"Safe code features shape: {safe_features.shape}")
    
    print(f"\nFeature difference (L2 distance): {np.linalg.norm(vuln_features - safe_features):.4f}")
    
    if np.linalg.norm(vuln_features - safe_features) > 0.5:
        print("✅ Features can distinguish vulnerable from safe code!")
    else:
        print("⚠️ Features might not be distinctive enough")
    
    print(f"\nVulnerable features:\n{vuln_features}")
    print(f"\nSafe features:\n{safe_features}")
