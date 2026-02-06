"""
Input Validation & Sanitization Module
Prevents injection attacks on the scanner itself

Features:
- Code input validation
- Path traversal prevention
- Size limits enforcement
- Character encoding validation
"""

import re
from typing import Optional, Tuple
from pydantic import validator, Field
from fastapi import HTTPException
import html
import logging

logger = logging.getLogger(__name__)

# Configuration
MAX_CODE_SIZE = 1_000_000  # 1MB max code size
MAX_CODE_LINES = 10_000    # 10k lines max
MAX_FILENAME_LENGTH = 255
ALLOWED_LANGUAGES = {"python", "javascript", "typescript", "java", "c", "cpp", "go", "rust"}

# Dangerous patterns that might indicate attack attempts
DANGEROUS_PATTERNS = [
    # Path traversal
    r'\.\.[/\\]',
    r'\.\.%2[fF]',
    r'\.\.%5[cC]',
    
    # Null byte injection
    r'\x00',
    r'%00',
    
    # Shell command injection in filenames
    r'[|;&$`]',
    r'\$\(',
    r'`.*`',
]

# Compiled patterns for efficiency
_dangerous_regex = [re.compile(p) for p in DANGEROUS_PATTERNS]


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(message)


def validate_code_input(code: str, language: str) -> Tuple[bool, Optional[str]]:
    """
    Validate code input for security issues.
    
    Args:
        code: Source code string
        language: Programming language
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty input
    if not code or not code.strip():
        return False, "Code cannot be empty"
    
    # Check size limits
    if len(code) > MAX_CODE_SIZE:
        return False, f"Code exceeds maximum size of {MAX_CODE_SIZE // 1024}KB"
    
    # Check line count
    line_count = code.count('\n') + 1
    if line_count > MAX_CODE_LINES:
        return False, f"Code exceeds maximum of {MAX_CODE_LINES} lines"
    
    # Validate language
    if language.lower() not in ALLOWED_LANGUAGES:
        return False, f"Unsupported language: {language}. Allowed: {', '.join(ALLOWED_LANGUAGES)}"
    
    # Check for valid UTF-8 encoding
    try:
        code.encode('utf-8')
    except UnicodeEncodeError:
        return False, "Invalid character encoding. Please use UTF-8"
    
    # Check for null bytes (potential attack)
    if '\x00' in code:
        logger.warning("Null byte detected in code input - potential attack")
        return False, "Invalid characters in code"
    
    return True, None


def validate_filename(filename: str) -> Tuple[bool, Optional[str]]:
    """
    Validate filename for path traversal and injection attacks.
    
    Args:
        filename: The filename to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename:
        return False, "Filename cannot be empty"
    
    # Check length
    if len(filename) > MAX_FILENAME_LENGTH:
        return False, f"Filename too long (max {MAX_FILENAME_LENGTH} characters)"
    
    # Check for dangerous patterns
    for pattern in _dangerous_regex:
        if pattern.search(filename):
            logger.warning(f"Dangerous pattern detected in filename: {filename}")
            return False, "Invalid filename - contains dangerous characters"
    
    # Check for path separators (prevent path traversal)
    if '/' in filename or '\\' in filename:
        return False, "Filename cannot contain path separators"
    
    # Whitelist allowed characters
    if not re.match(r'^[\w\-. ]+$', filename):
        return False, "Filename contains invalid characters"
    
    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate URL for SSRF prevention.
    
    Args:
        url: The URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"
    
    # Must start with http:// or https://
    if not url.startswith(('http://', 'https://')):
        return False, "URL must start with http:// or https://"
    
    # Block internal/private IP ranges
    blocked_patterns = [
        r'localhost',
        r'127\.\d+\.\d+\.\d+',
        r'10\.\d+\.\d+\.\d+',
        r'172\.(1[6-9]|2\d|3[01])\.\d+\.\d+',
        r'192\.168\.\d+\.\d+',
        r'169\.254\.\d+\.\d+',
        r'\[::1\]',
        r'\[fe80:',
        r'0\.0\.0\.0',
    ]
    
    for pattern in blocked_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            logger.warning(f"Blocked internal URL access attempt: {url}")
            return False, "Access to internal/private URLs is not allowed"
    
    return True, None


def sanitize_output(text: str) -> str:
    """
    Sanitize text for safe output (prevent XSS in responses).
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # HTML escape
    return html.escape(text)


def sanitize_code_for_display(code: str) -> str:
    """
    Sanitize code for safe display in HTML context.
    Preserves code structure while preventing XSS.
    
    Args:
        code: Source code to sanitize
        
    Returns:
        Sanitized code safe for HTML display
    """
    if not code:
        return ""
    
    # HTML escape special characters
    code = html.escape(code)
    
    return code


class SecureCodeInput:
    """
    Pydantic-compatible validator for code input.
    
    Usage:
        class ScanRequest(BaseModel):
            code: str
            language: str
            
            _validate_code = validator('code', allow_reuse=True)(SecureCodeInput.validate)
    """
    
    @classmethod
    def validate(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Code cannot be empty")
        
        if len(v) > MAX_CODE_SIZE:
            raise ValueError(f"Code exceeds maximum size of {MAX_CODE_SIZE // 1024}KB")
        
        if '\x00' in v:
            raise ValueError("Invalid characters in code")
        
        return v


def create_validation_middleware():
    """
    Create a FastAPI middleware for input validation.
    
    Returns middleware function.
    """
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class ValidationMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Check content length
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_CODE_SIZE * 2:
                raise HTTPException(
                    status_code=413,
                    detail="Request body too large"
                )
            
            return await call_next(request)
    
    return ValidationMiddleware
