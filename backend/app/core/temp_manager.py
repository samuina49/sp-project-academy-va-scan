"""
Temporary directory management with automatic cleanup.
"""
import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
import uuid


@contextmanager
def temporary_directory(prefix: str = "scan_") -> Generator[Path, None, None]:
    """
    Create a temporary directory that is automatically cleaned up.
    
    Args:
        prefix: Prefix for the temporary directory name
        
    Yields:
        Path to the temporary directory
        
    Example:
        with temporary_directory() as temp_dir:
            # Use temp_dir for scanning
            scan_file(temp_dir / "code.py")
        # temp_dir is automatically deleted
    """
    # Create unique temporary directory
    temp_dir = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        yield temp_dir
    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_temp_file(content: str, suffix: str = ".py", prefix: str = "scan_") -> Path:
    """
    Create a temporary file with the given content.
    
    Args:
        content: Content to write to the file
        suffix: File extension
        prefix: Prefix for the filename
        
    Returns:
        Path to the created temporary file
        
    Note:
        Caller is responsible for cleanup
    """
    temp_dir = Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file = temp_dir / f"{prefix}{uuid.uuid4().hex}{suffix}"
    temp_file.write_text(content, encoding='utf-8')
    
    return temp_file


def cleanup_temp_file(file_path: Path) -> None:
    """
    Safely delete a temporary file.
    
    Args:
        file_path: Path to the file to delete
    """
    try:
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
    except Exception:
        # Ignore errors during cleanup
        pass
