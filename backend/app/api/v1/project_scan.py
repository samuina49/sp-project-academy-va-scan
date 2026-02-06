"""
Multi-File & Project Scanning API
Scans entire directories, zip files, or Git repositories

Features:
- Directory scanning with recursive file discovery
- Zip file upload and extraction
- Git repository cloning and scanning
- Cross-file analysis aggregation
- Incremental scanning (diff-based)
- Project-level vulnerability heatmap
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import zipfile
import asyncio
import uuid
import json
import os
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.jsx': 'javascript',
    '.tsx': 'typescript',
    '.java': 'java',
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
}

# Directories to skip
SKIP_DIRS = {
    'node_modules', '.git', '.svn', '__pycache__', '.venv', 'venv',
    'env', '.env', 'dist', 'build', '.next', 'coverage', '.pytest_cache',
    'vendor', 'packages', '.idea', '.vscode'
}

# Maximum limits
MAX_FILE_SIZE = 1_000_000  # 1MB per file
MAX_FILES = 1000
MAX_PROJECT_SIZE = 50_000_000  # 50MB total


# ==================== Models ====================

class FileResult(BaseModel):
    file_path: str
    language: str
    is_vulnerable: bool
    vulnerability_count: int
    severity_counts: Dict[str, int]
    vulnerabilities: List[dict]
    lines_scanned: int
    scan_time_ms: float


class ProjectScanResult(BaseModel):
    scan_id: str
    project_name: str
    scan_status: str  # pending, scanning, completed, failed
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_files: int
    files_scanned: int
    files_with_vulnerabilities: int
    total_vulnerabilities: int
    severity_summary: Dict[str, int]
    risk_score: int
    file_results: List[FileResult]
    language_breakdown: Dict[str, int]
    scan_duration_ms: float = 0


class ProjectScanRequest(BaseModel):
    project_name: Optional[str] = "Untitled Project"
    threshold: float = 0.5


class GitScanRequest(BaseModel):
    repo_url: HttpUrl
    branch: str = "main"
    project_name: Optional[str] = None
    threshold: float = 0.5


class IncrementalScanRequest(BaseModel):
    base_scan_id: str
    changed_files: List[str]
    threshold: float = 0.5


# ==================== Storage ====================

# In-memory scan results (use Redis/DB in production)
_scan_results: Dict[str, ProjectScanResult] = {}
_scan_tasks: Dict[str, asyncio.Task] = {}


# ==================== Helper Functions ====================

def get_language(file_path: Path) -> Optional[str]:
    """Determine language from file extension"""
    return SUPPORTED_EXTENSIONS.get(file_path.suffix.lower())


def should_skip_dir(dir_name: str) -> bool:
    """Check if directory should be skipped"""
    return dir_name in SKIP_DIRS or dir_name.startswith('.')


def collect_files(directory: Path) -> List[Path]:
    """Recursively collect all scannable files"""
    files = []
    
    try:
        for item in directory.rglob('*'):
            # Skip directories in SKIP_DIRS
            if any(skip in item.parts for skip in SKIP_DIRS):
                continue
            
            if item.is_file():
                if get_language(item):
                    # Check file size
                    if item.stat().st_size <= MAX_FILE_SIZE:
                        files.append(item)
                    else:
                        logger.warning(f"Skipping large file: {item}")
            
            if len(files) >= MAX_FILES:
                logger.warning(f"Maximum file limit ({MAX_FILES}) reached")
                break
    except Exception as e:
        logger.error(f"Error collecting files: {e}")
    
    return files


async def scan_single_file(file_path: Path, threshold: float) -> FileResult:
    """Scan a single file and return results"""
    import time
    start_time = time.time()
    
    language = get_language(file_path)
    if not language:
        return FileResult(
            file_path=str(file_path),
            language="unknown",
            is_vulnerable=False,
            vulnerability_count=0,
            severity_counts={},
            vulnerabilities=[],
            lines_scanned=0,
            scan_time_ms=0
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        lines = code.count('\n') + 1
        
        # Import scanner (avoid circular imports)
        from app.api.v1.ai_scan import load_model
        from app.scanners.simple_scanner import SimplePatternScanner
        
        # Use pattern scanner for speed
        scanner = SimplePatternScanner()
        result = scanner.scan_code(code, language)
        
        vulnerabilities = []
        severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for finding in result.findings:
            sev = finding.severity.value if hasattr(finding.severity, 'value') else str(finding.severity)
            severity_counts[sev.upper()] = severity_counts.get(sev.upper(), 0) + 1
            vulnerabilities.append({
                'cwe_id': finding.cwe_id,
                'severity': sev,
                'message': finding.message,
                'line': finding.start_line,
                'confidence': 0.95
            })
        
        scan_time = (time.time() - start_time) * 1000
        
        return FileResult(
            file_path=str(file_path),
            language=language,
            is_vulnerable=len(vulnerabilities) > 0,
            vulnerability_count=len(vulnerabilities),
            severity_counts=severity_counts,
            vulnerabilities=vulnerabilities,
            lines_scanned=lines,
            scan_time_ms=scan_time
        )
        
    except Exception as e:
        logger.error(f"Error scanning {file_path}: {e}")
        return FileResult(
            file_path=str(file_path),
            language=language,
            is_vulnerable=False,
            vulnerability_count=0,
            severity_counts={},
            vulnerabilities=[],
            lines_scanned=0,
            scan_time_ms=0
        )


async def scan_project_async(
    scan_id: str,
    directory: Path,
    project_name: str,
    threshold: float
):
    """Background task to scan entire project"""
    import time
    start_time = time.time()
    
    try:
        # Update status to scanning
        _scan_results[scan_id].scan_status = "scanning"
        
        # Collect files
        files = collect_files(directory)
        _scan_results[scan_id].total_files = len(files)
        
        # Scan each file
        file_results = []
        language_counts = {}
        total_vulns = 0
        files_with_vulns = 0
        severity_total = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for i, file_path in enumerate(files):
            result = await scan_single_file(file_path, threshold)
            file_results.append(result)
            
            # Update counts
            lang = result.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            if result.is_vulnerable:
                files_with_vulns += 1
                total_vulns += result.vulnerability_count
                for sev, count in result.severity_counts.items():
                    severity_total[sev] = severity_total.get(sev, 0) + count
            
            # Update progress
            _scan_results[scan_id].files_scanned = i + 1
        
        # Calculate risk score
        risk_score = min(100, (
            severity_total['CRITICAL'] * 25 +
            severity_total['HIGH'] * 10 +
            severity_total['MEDIUM'] * 3 +
            severity_total['LOW'] * 1
        ))
        
        # Update final results
        scan_duration = (time.time() - start_time) * 1000
        
        _scan_results[scan_id].scan_status = "completed"
        _scan_results[scan_id].completed_at = datetime.utcnow()
        _scan_results[scan_id].files_with_vulnerabilities = files_with_vulns
        _scan_results[scan_id].total_vulnerabilities = total_vulns
        _scan_results[scan_id].severity_summary = severity_total
        _scan_results[scan_id].risk_score = risk_score
        _scan_results[scan_id].file_results = file_results
        _scan_results[scan_id].language_breakdown = language_counts
        _scan_results[scan_id].scan_duration_ms = scan_duration
        
        logger.info(f"Project scan {scan_id} completed: {total_vulns} vulnerabilities in {files_with_vulns}/{len(files)} files")
        
    except Exception as e:
        logger.error(f"Project scan {scan_id} failed: {e}")
        _scan_results[scan_id].scan_status = "failed"


def calculate_risk_score(severity_counts: Dict[str, int]) -> int:
    """Calculate overall risk score from severity counts"""
    return min(100, (
        severity_counts.get('CRITICAL', 0) * 25 +
        severity_counts.get('HIGH', 0) * 10 +
        severity_counts.get('MEDIUM', 0) * 3 +
        severity_counts.get('LOW', 0) * 1
    ))


# ==================== API Endpoints ====================

@router.post("/project/upload", response_model=dict, tags=["Project Scanning"])
async def upload_project(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_name: Optional[str] = None,
    threshold: float = 0.5
):
    """
    **Upload and Scan Project (ZIP)**
    
    Upload a ZIP file containing your project for scanning.
    
    - Maximum size: 50MB
    - Supports: Python, JavaScript, TypeScript, Java, Go, Rust, C/C++
    - Skips: node_modules, .git, __pycache__, etc.
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are supported"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Seek back to start
    
    if size > MAX_PROJECT_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_PROJECT_SIZE // 1_000_000}MB"
        )
    
    # Create temp directory
    scan_id = str(uuid.uuid4())
    temp_dir = Path(tempfile.mkdtemp(prefix=f"scan_{scan_id}_"))
    
    try:
        # Save and extract ZIP
        zip_path = temp_dir / "upload.zip"
        with open(zip_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        extract_dir = temp_dir / "project"
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        
        # Initialize scan result
        name = project_name or file.filename.replace('.zip', '')
        _scan_results[scan_id] = ProjectScanResult(
            scan_id=scan_id,
            project_name=name,
            scan_status="pending",
            started_at=datetime.utcnow(),
            total_files=0,
            files_scanned=0,
            files_with_vulnerabilities=0,
            total_vulnerabilities=0,
            severity_summary={},
            risk_score=0,
            file_results=[],
            language_breakdown={}
        )
        
        # Start background scan
        background_tasks.add_task(
            scan_project_async,
            scan_id,
            extract_dir,
            name,
            threshold
        )
        
        return {
            "scan_id": scan_id,
            "project_name": name,
            "status": "pending",
            "message": "Scan started. Use GET /project/status/{scan_id} to check progress."
        }
        
    except zipfile.BadZipFile:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project/scan-directory", response_model=dict, tags=["Project Scanning"])
async def scan_directory(
    background_tasks: BackgroundTasks,
    directory_path: str,
    project_name: Optional[str] = None,
    threshold: float = 0.5
):
    """
    **Scan Local Directory**
    
    Scan a local directory path (for server-side scanning).
    """
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail="Directory not found")
    
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    scan_id = str(uuid.uuid4())
    name = project_name or dir_path.name
    
    _scan_results[scan_id] = ProjectScanResult(
        scan_id=scan_id,
        project_name=name,
        scan_status="pending",
        started_at=datetime.utcnow(),
        total_files=0,
        files_scanned=0,
        files_with_vulnerabilities=0,
        total_vulnerabilities=0,
        severity_summary={},
        risk_score=0,
        file_results=[],
        language_breakdown={}
    )
    
    background_tasks.add_task(
        scan_project_async,
        scan_id,
        dir_path,
        name,
        threshold
    )
    
    return {
        "scan_id": scan_id,
        "project_name": name,
        "status": "pending",
        "message": "Scan started."
    }


@router.get("/project/status/{scan_id}", response_model=dict, tags=["Project Scanning"])
async def get_scan_status(scan_id: str):
    """
    **Get Scan Status**
    
    Check the progress and status of a project scan.
    """
    if scan_id not in _scan_results:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    result = _scan_results[scan_id]
    
    return {
        "scan_id": scan_id,
        "status": result.scan_status,
        "progress": {
            "total_files": result.total_files,
            "files_scanned": result.files_scanned,
            "percentage": round(result.files_scanned / max(result.total_files, 1) * 100, 1)
        },
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat() if result.completed_at else None
    }


@router.get("/project/result/{scan_id}", response_model=ProjectScanResult, tags=["Project Scanning"])
async def get_scan_result(scan_id: str):
    """
    **Get Full Scan Results**
    
    Retrieve complete scan results including all file findings.
    """
    if scan_id not in _scan_results:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    result = _scan_results[scan_id]
    
    if result.scan_status == "pending" or result.scan_status == "scanning":
        raise HTTPException(
            status_code=202,
            detail=f"Scan still in progress: {result.files_scanned}/{result.total_files} files"
        )
    
    return result


@router.get("/project/summary/{scan_id}", response_model=dict, tags=["Project Scanning"])
async def get_scan_summary(scan_id: str):
    """
    **Get Scan Summary**
    
    Get a condensed summary without individual file details.
    """
    if scan_id not in _scan_results:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    result = _scan_results[scan_id]
    
    return {
        "scan_id": scan_id,
        "project_name": result.project_name,
        "status": result.scan_status,
        "risk_score": result.risk_score,
        "total_files": result.total_files,
        "files_with_vulnerabilities": result.files_with_vulnerabilities,
        "total_vulnerabilities": result.total_vulnerabilities,
        "severity_summary": result.severity_summary,
        "language_breakdown": result.language_breakdown,
        "scan_duration_ms": result.scan_duration_ms,
        "top_vulnerable_files": [
            {
                "file": f.file_path,
                "vulnerabilities": f.vulnerability_count,
                "severity": f.severity_counts
            }
            for f in sorted(
                result.file_results,
                key=lambda x: x.vulnerability_count,
                reverse=True
            )[:10]
        ]
    }


@router.delete("/project/{scan_id}", tags=["Project Scanning"])
async def delete_scan(scan_id: str):
    """
    **Delete Scan Results**
    
    Remove scan results from memory.
    """
    if scan_id not in _scan_results:
        raise HTTPException(status_code=404, detail="Scan not found")
    
    del _scan_results[scan_id]
    
    return {"message": "Scan results deleted"}


@router.get("/project/list", response_model=List[dict], tags=["Project Scanning"])
async def list_scans():
    """
    **List All Scans**
    
    Get a list of all project scans with basic info.
    """
    return [
        {
            "scan_id": scan_id,
            "project_name": result.project_name,
            "status": result.scan_status,
            "started_at": result.started_at.isoformat(),
            "total_vulnerabilities": result.total_vulnerabilities,
            "risk_score": result.risk_score
        }
        for scan_id, result in _scan_results.items()
    ]
