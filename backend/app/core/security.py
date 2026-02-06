"""
Security utilities for safe file handling and validation.
"""
import os
import zipfile
from pathlib import Path
from typing import Optional
import mimetypes


class SecurityError(Exception):
    """Raised when a security violation is detected"""
    pass


def validate_zip_file(zip_path: Path, max_size_bytes: int) -> None:
    """
    Validate ZIP file before extraction.
    
    Args:
        zip_path: Path to ZIP file
        max_size_bytes: Maximum allowed size in bytes
        
    Raises:
        SecurityError: If validation fails
    """
    # Check file size
    file_size = os.path.getsize(zip_path)
    if file_size > max_size_bytes:
        raise SecurityError(
            f"ZIP file too large: {file_size} bytes (max: {max_size_bytes})"
        )
    
    # Check if it's a valid ZIP
    if not zipfile.is_zipfile(zip_path):
        raise SecurityError("Invalid ZIP file")


def safe_extract_zip(
    zip_path: Path,
    extract_to: Path,
    max_files: int = 500
) -> None:
    """
    Safely extract ZIP file with Zip Slip prevention.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Destination directory
        max_files: Maximum number of files allowed
        
    Raises:
        SecurityError: If security violation detected
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        
        # Check file count
        if len(members) > max_files:
            raise SecurityError(
                f"Too many files in ZIP: {len(members)} (max: {max_files})"
            )
        
        # Validate each member for Zip Slip
        for member in members:
            member_path = Path(extract_to) / member
            
            # Resolve to absolute path and check if it's within extract_to
            try:
                member_path = member_path.resolve()
                extract_to_resolved = Path(extract_to).resolve()
                
                # Check if the resolved path is within the target directory
                if not str(member_path).startswith(str(extract_to_resolved)):
                    raise SecurityError(
                        f"Zip Slip attack detected: {member} attempts to write outside extraction directory"
                    )
            except (OSError, RuntimeError) as e:
                raise SecurityError(f"Invalid path in ZIP: {member}") from e
        
        # All checks passed, extract safely
        zip_ref.extractall(extract_to)


def validate_file_path(file_path: Path, base_dir: Path) -> Path:
    """
    Validate that a file path is within the allowed base directory.
    Prevents path traversal attacks.
    
    Args:
        file_path: File path to validate
        base_dir: Base directory that must contain the file
        
    Returns:
        Resolved file path
        
    Raises:
        SecurityError: If path is outside base directory
    """
    try:
        resolved_file = file_path.resolve()
        resolved_base = base_dir.resolve()
        
        # Check if file is within base directory
        if not str(resolved_file).startswith(str(resolved_base)):
            raise SecurityError(
                f"Path traversal detected: {file_path} is outside {base_dir}"
            )
        
        return resolved_file
    except (OSError, RuntimeError) as e:
        raise SecurityError(f"Invalid file path: {file_path}") from e


def detect_language(filename: str, content: Optional[str] = None) -> Optional[str]:
    """
    Detect programming language from filename and optionally content.
    
    Args:
        filename: Name of the file
        content: File content (optional)
        
    Returns:
        Language identifier (python, javascript, typescript) or None
    """
    extension = Path(filename).suffix.lower()
    
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
    }
    
    return language_map.get(extension)


def is_safe_filename(filename: str) -> bool:
    """
    Check if filename is safe (no path traversal, no special chars).
    
    Args:
        filename: Filename to check
        
    Returns:
        True if safe, False otherwise
    """
    # Check for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return False
    
    # Check for null bytes
    if '\x00' in filename:
        return False
    
    # Check for overly long names
    if len(filename) > 255:
        return False
    
    return True


def should_ignore_path(path: Path, ignored_dirs: list[str]) -> bool:
    """
    Check if a path should be ignored during scanning.
    
    Args:
        path: Path to check
        ignored_dirs: List of directory names to ignore
        
    Returns:
        True if path should be ignored, False otherwise
    """
    parts = path.parts
    
    # Check if any part of the path matches ignored directories
    for part in parts:
        if part in ignored_dirs:
            return True
    
    return False
