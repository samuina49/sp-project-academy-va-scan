"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class SeverityEnum(str, Enum):
    """Vulnerability severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class DetectionSourceEnum(str, Enum):
    """Source of vulnerability detection"""
    SEMGREP = "semgrep"
    BANDIT = "bandit"
    ML_MODEL = "ml"
    HYBRID = "hybrid"


class VulnerabilityFinding(BaseModel):
    """Single vulnerability finding"""
    tool: str = Field(..., description="Scanner tool name (bandit, semgrep)")
    rule_id: str = Field(..., description="Rule identifier")
    severity: SeverityEnum = Field(..., description="Severity level")
    message: str = Field(..., description="Vulnerability description")
    start_line: int = Field(..., description="Starting line number (1-indexed)")
    end_line: int = Field(..., description="Ending line number (1-indexed)")
    code_snippet: Optional[str] = Field(None, description="Code snippet if available")
    cwe_id: Optional[str] = Field(None, description="CWE identifier if available")


class FileScanResult(BaseModel):
    """Scan result for a single file"""
    file_path: str = Field(..., description="Relative file path")
    language: str = Field(..., description="Programming language")
    findings: List[VulnerabilityFinding] = Field(default_factory=list)
    scan_duration_ms: Optional[float] = Field(None, description="Scan duration in milliseconds")
    source_code: Optional[str] = Field(None, description="Full source code of the file")


class CodeScanRequest(BaseModel):
    """Request to scan pasted code"""
    code: str = Field(..., description="Source code to scan", min_length=1)
    language: Literal["python", "javascript", "typescript"] = Field(
        ..., 
        description="Programming language"
    )
    filename: Optional[str] = Field(
        None, 
        description="Optional filename (used for language detection override)"
    )


class CodeScanResponse(BaseModel):
    """Response from code scan"""
    scan_id: str = Field(..., description="Unique scan identifier")
    file_result: FileScanResult
    total_findings: int = Field(..., description="Total number of findings")
    success: bool = Field(True, description="Scan completed successfully")
    error: Optional[str] = Field(None, description="Error message if scan failed")


class ZipScanResponse(BaseModel):
    """Response from ZIP project scan"""
    scan_id: str = Field(..., description="Unique scan identifier")
    file_results: List[FileScanResult] = Field(default_factory=list)
    total_files_scanned: int = Field(..., description="Number of files scanned")
    total_findings: int = Field(..., description="Total number of findings")
    scan_duration_ms: Optional[float] = Field(None, description="Total scan duration")
    success: bool = Field(True, description="Scan completed successfully")
    error: Optional[str] = Field(None, description="Error message if scan failed")


class HybridFinding(BaseModel):
    """Enhanced finding from hybrid detection"""
    line: int = Field(..., description="Line number")
    vulnerability_type: str = Field(..., description="Vulnerability type")
    severity: SeverityEnum = Field(..., description="Severity level")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    sources: List[DetectionSourceEnum] = Field(..., description="Detection sources")
    code_snippet: str = Field(..., description="Code snippet")
    explanation: str = Field(..., description="Explanation")
    remediation: Optional[str] = Field(None, description="Remediation advice")
    cwe_id: Optional[str] = Field(None, description="CWE identifier")
    owasp_category: Optional[str] = Field(None, description="OWASP Top 10 category")
    semgrep_rule: Optional[str] = Field(None, description="Semgrep rule ID")
    bandit_test: Optional[str] = Field(None, description="Bandit test ID")
    ml_probability: Optional[float] = Field(None, description="ML probability")


class HybridScanResponse(BaseModel):
    """Response from hybrid scan (Pattern + ML)"""
    scan_id: str = Field(..., description="Unique scan identifier")
    timestamp: str = Field(..., description="Scan timestamp")
    code_language: str = Field(..., description="Programming language")
    scan_type: str = Field(default="hybrid", description="Scan type")
    summary: dict = Field(..., description="Scan summary")
    findings: List[HybridFinding] = Field(default_factory=list)
    success: bool = Field(True, description="Scan completed successfully")
    error: Optional[str] = Field(None, description="Error message if scan failed")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    bandit_available: bool = Field(..., description="Bandit scanner available")
    semgrep_available: bool = Field(..., description="Semgrep scanner available")
    ml_model_available: bool = Field(False, description="ML model available")
