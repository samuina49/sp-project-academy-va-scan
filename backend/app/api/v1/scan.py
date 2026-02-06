"""
Scan API endpoints for vulnerability detection.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import StreamingResponse
from pathlib import Path
import uuid
import time

from app.models.scan_models import (
    CodeScanRequest,
    CodeScanResponse,
    ZipScanResponse,
    FileScanResult
)
from app.scanners.scanner_orchestrator import ScannerOrchestrator
from app.scanners.hybrid_orchestrator import HybridScanner
from app.core.temp_manager import temporary_directory, create_temp_file, cleanup_temp_file
from app.core.security import (
    validate_zip_file,
    safe_extract_zip,
    SecurityError,
    detect_language
)
from app.core.config import settings
from app.utils.excel_report import generate_excel_report


router = APIRouter(prefix="/scan", tags=["scanning"])

# Initialize scanner for /code endpoint (uses pattern matching for speed)
ml_enabled = getattr(settings, 'ML_ENABLED', False)
if ml_enabled:
    scanner = HybridScanner()
    print("[OK] /api/v1/scan/code → Hybrid Scanner (Pattern + ML)")
else:
    scanner = ScannerOrchestrator()
    print("[OK] /api/v1/scan/code → Pattern Scanner (fast mode)")

# ZIP scanning always uses Pattern Scanner (reliable and fast)
print("[OK] /api/v1/scan/zip → Pattern Scanner (Semgrep + Bandit) ✓")
print("[OK] /api/v1/ml-scan → Hybrid AI Scanner (GNN+LSTM + Pattern) ✓")


@router.post("/code", response_model=CodeScanResponse)
async def scan_code(request: CodeScanRequest):
    """
    Scan a single pasted source code file.
    
    Args:
        request: Code scan request with code, language, and optional filename
        
    Returns:
        CodeScanResponse with vulnerability findings
    """
    scan_id = str(uuid.uuid4())
    
    # Determine file extension based on language
    extension_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts"
    }
    extension = extension_map.get(request.language, ".txt")
    
    # Use provided filename or generate one
    filename = request.filename or f"code{extension}"
    
    print(f"[SINGLE FILE] Received code scan: {filename} ({request.language})")
    print(f"[SINGLE FILE] Code size: {len(request.code)} bytes")
    
    # Create temporary file with code
    temp_file = None
    try:
        temp_file = create_temp_file(
            content=request.code,
            suffix=extension,
            prefix=f"scan_{scan_id}_"
        )
        
        print(f"[SINGLE FILE] Starting Hybrid Scanner (Pattern + ML)...")
        print(f"[SINGLE FILE] ML Enabled: {ml_enabled}")
        
        # Scan the file using global scanner (HybridScanner if ML enabled)
        result = scanner.scan_file(temp_file, language=request.language)
        
        if not result:
            print(f"[SINGLE FILE] ERROR: Language '{request.language}' is not supported")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language '{request.language}' is not supported"
            )
        
        # Update file path to use original filename
        result.file_path = filename
        
        print(f"[SINGLE FILE] Scan complete: found {len(result.findings)} issues")
        
        return CodeScanResponse(
            scan_id=scan_id,
            file_result=result,
            total_findings=len(result.findings),
            success=True
        )
        
    except SecurityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file:
            cleanup_temp_file(temp_file)


@router.post("/hybrid", response_model=CodeScanResponse)
async def scan_code_hybrid(request: CodeScanRequest):
    """
    Hybrid scan endpoint - uses same scanner as /code but compatible with HybridResultsPanel.
    Currently uses pattern-matching scanner (Semgrep/Bandit).
    """
   # Use the same logic as scan_code
    return await scan_code(request)


@router.post("/zip", response_model=ZipScanResponse)
async def scan_zip(file: UploadFile = File(...)):
    """
    Scan an uploaded ZIP project.
    
    Args:
        file: Uploaded ZIP file
        
    Returns:
        ZipScanResponse with findings grouped by file
    """
    print(f"[ZIP SCAN] Received ZIP file: {file.filename}")
    scan_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Validate file is a ZIP
    if not file.filename.endswith('.zip'):
        print(f"[ZIP SCAN] ERROR: File is not a ZIP: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a ZIP archive"
        )
    
    with temporary_directory(prefix=f"zip_scan_{scan_id}_") as temp_dir:
        try:
            # Save uploaded file
            zip_path = temp_dir / "upload.zip"
            content = await file.read()
            print(f"[ZIP SCAN] ZIP size: {len(content)} bytes")
            zip_path.write_bytes(content)
            
            # Validate ZIP
            print(f"[ZIP SCAN] Validating ZIP file...")
            validate_zip_file(zip_path, settings.max_zip_size_bytes)
            
            # Extract ZIP safely
            extract_dir = temp_dir / "extracted"
            extract_dir.mkdir()
            print(f"[ZIP SCAN] Extracting ZIP to {extract_dir}...")
            safe_extract_zip(
                zip_path=zip_path,
                extract_to=extract_dir,
                max_files=settings.MAX_FILE_COUNT
            )
            
            # Scan all files using global Hybrid Scanner (Pattern + ML if enabled)
            print(f"[ZIP SCAN] Starting Hybrid Scanner (Pattern + ML)...")
            # Ensure we use HybridScanner for ZIP (always use ML if available)
            zip_scanner = HybridScanner()  # Fresh instance with ML_ENABLED from settings
            print(f"[ZIP SCAN] ML Enabled: {zip_scanner.ml_enabled}")
            results = zip_scanner.scan_directory(extract_dir, recursive=True)
            print(f"[ZIP SCAN] Scanned {len(results)} files, found {sum(len(r.findings) for r in results)} issues")
            
            # Make file paths relative to extract_dir for cleaner output
            for result in results:
                try:
                    rel_path = Path(result.file_path).relative_to(extract_dir)
                    result.file_path = str(rel_path)
                except ValueError:
                    # If relative_to fails, keep original path
                    pass
            
            duration_ms = (time.time() - start_time) * 1000
            total_findings = sum(len(r.findings) for r in results)
            
            return ZipScanResponse(
                scan_id=scan_id,
                file_results=results,
                total_files_scanned=len(results),
                total_findings=total_findings,
                scan_duration_ms=duration_ms,
                success=True
            )
            
        except SecurityError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Scan failed: {str(e)}"
            )


@router.post("/code/export/excel")
async def export_code_scan_excel(request: CodeScanRequest):
    """
    Scan code and export results as Excel file
    
    Args:
        request: Code scan request
        
    Returns:
        Excel file download
    """
    print(f"[EXCEL EXPORT] Starting export...")
    scan_id = str(uuid.uuid4())
    
    # Determine file extension
    extension_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts"
    }
    extension = extension_map.get(request.language, ".txt")
    filename = request.filename or f"code{extension}"
    print(f"[EXCEL EXPORT] Filename: {filename}")
    
    # Create temporary file
    temp_file = None
    try:
        temp_file = create_temp_file(
            content=request.code,
            suffix=extension,
            prefix=f"scan_{scan_id}_"
        )
        
        # Scan the file
        print(f"[EXCEL EXPORT] Scanning file...")
        orchestrator = ScannerOrchestrator()
        result = orchestrator.scan_file(temp_file, language=request.language)
        print(f"[EXCEL EXPORT] Scan complete. Result: {result is not None}")
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Language '{request.language}' is not supported"
            )
        
        result.file_path = filename
        
        # Generate Excel report
        print(f"[EXCEL EXPORT] Generating Excel...")
        metadata = {
            'scan_id': scan_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': 1,
            'language': request.language
        }
        
        excel_file = generate_excel_report([result], metadata)
        print(f"[EXCEL EXPORT] Excel generated successfully")

        
        # Return as download
        return StreamingResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename=vulnerability_report_{scan_id}.xlsx"
            }
        )
        
    except SecurityError as e:
        print(f"[EXCEL EXPORT] Security error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        print(f"[EXCEL EXPORT] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )
    finally:
        if temp_file:
            cleanup_temp_file(temp_file)
