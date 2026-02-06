"""
Infrastructure Security API Endpoints
Dockerfile, Kubernetes, and Secret scanning
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from ...scanners.infrastructure import (
    get_infrastructure_scanner,
    get_dockerfile_scanner,
    get_secret_scanner,
    InfraType,
    InfraFinding
)

router = APIRouter()


# ==================== Request/Response Models ====================

class InfraScanRequest(BaseModel):
    content: str
    filename: Optional[str] = None
    infra_type: Optional[str] = None  # dockerfile, docker-compose, kubernetes


class InfraScanResult(BaseModel):
    scan_id: str
    infra_type: str
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    findings: List[Dict]
    scan_time_ms: int


class SecretScanResult(BaseModel):
    scan_id: str
    total_secrets: int
    secret_types: List[str]
    findings: List[Dict]
    scan_time_ms: int


# ==================== Endpoints ====================

@router.post("/infrastructure/scan", response_model=InfraScanResult, tags=["Infrastructure Security"])
async def scan_infrastructure(request: InfraScanRequest):
    """
    **Scan Infrastructure File**
    
    Analyzes Dockerfiles, docker-compose.yml, and Kubernetes manifests
    for security misconfigurations.
    
    Detects:
    - Privileged containers
    - Root user usage
    - Exposed sensitive ports
    - Hardcoded secrets
    - Missing security controls
    """
    start_time = datetime.now()
    
    scanner = get_infrastructure_scanner()
    
    # Detect or use provided type
    if request.infra_type:
        infra_type = InfraType(request.infra_type.lower().replace('-', '_'))
    else:
        infra_type = scanner.detect_type(request.filename or "", request.content)
    
    findings = scanner.scan(request.content, infra_type)
    
    # Count by severity
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
    
    scan_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return InfraScanResult(
        scan_id=f"infra_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        infra_type=infra_type.value,
        total_findings=len(findings),
        critical_count=severity_counts['CRITICAL'],
        high_count=severity_counts['HIGH'],
        medium_count=severity_counts['MEDIUM'],
        low_count=severity_counts['LOW'],
        findings=[{
            'line': f.line,
            'type': f.type,
            'severity': f.severity,
            'message': f.message,
            'rule_id': f.rule_id,
            'remediation': f.remediation
        } for f in findings],
        scan_time_ms=int(scan_time)
    )


@router.post("/infrastructure/scan/dockerfile", response_model=InfraScanResult, tags=["Infrastructure Security"])
async def scan_dockerfile(file: UploadFile = File(...)):
    """
    **Scan Dockerfile**
    
    Upload a Dockerfile for security analysis.
    
    Checks for:
    - Insecure base images (latest tag)
    - Root user
    - Piping curl to shell
    - Hardcoded secrets
    - Missing HEALTHCHECK
    - Permissive file permissions
    """
    content = await file.read()
    content_str = content.decode('utf-8')
    
    start_time = datetime.now()
    
    scanner = get_dockerfile_scanner()
    findings = scanner.scan_dockerfile(content_str)
    
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
    
    scan_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return InfraScanResult(
        scan_id=f"docker_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        infra_type="dockerfile",
        total_findings=len(findings),
        critical_count=severity_counts['CRITICAL'],
        high_count=severity_counts['HIGH'],
        medium_count=severity_counts['MEDIUM'],
        low_count=severity_counts['LOW'],
        findings=[{
            'line': f.line,
            'type': f.type,
            'severity': f.severity,
            'message': f.message,
            'rule_id': f.rule_id,
            'remediation': f.remediation
        } for f in findings],
        scan_time_ms=int(scan_time)
    )


@router.post("/infrastructure/scan/kubernetes", response_model=InfraScanResult, tags=["Infrastructure Security"])
async def scan_kubernetes(file: UploadFile = File(...)):
    """
    **Scan Kubernetes Manifest**
    
    Upload a Kubernetes YAML file for security analysis.
    
    Checks for:
    - Privileged pods
    - Root containers
    - Host network/PID/IPC sharing
    - Dangerous capabilities
    - Missing resource limits
    - Hardcoded secrets
    """
    content = await file.read()
    content_str = content.decode('utf-8')
    
    start_time = datetime.now()
    
    scanner = get_infrastructure_scanner()
    findings = scanner.scan(content_str, InfraType.KUBERNETES)
    
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for f in findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
    
    scan_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return InfraScanResult(
        scan_id=f"k8s_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        infra_type="kubernetes",
        total_findings=len(findings),
        critical_count=severity_counts['CRITICAL'],
        high_count=severity_counts['HIGH'],
        medium_count=severity_counts['MEDIUM'],
        low_count=severity_counts['LOW'],
        findings=[{
            'line': f.line,
            'type': f.type,
            'severity': f.severity,
            'message': f.message,
            'rule_id': f.rule_id,
            'remediation': f.remediation
        } for f in findings],
        scan_time_ms=int(scan_time)
    )


@router.post("/infrastructure/secrets", response_model=SecretScanResult, tags=["Infrastructure Security"])
async def scan_for_secrets(request: InfraScanRequest):
    """
    **Scan for Secrets**
    
    Detect hardcoded secrets, API keys, tokens, and credentials in code.
    
    Detects:
    - API keys
    - Passwords
    - Private keys
    - AWS credentials
    - GitHub tokens
    - Database connection strings
    - OAuth tokens
    """
    start_time = datetime.now()
    
    scanner = get_secret_scanner()
    findings = scanner.scan_file(request.content, request.filename or "unknown")
    
    secret_types = list(set(f.type for f in findings))
    
    scan_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return SecretScanResult(
        scan_id=f"secret_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        total_secrets=len(findings),
        secret_types=secret_types,
        findings=[{
            'line': f.line,
            'type': f.type,
            'severity': f.severity,
            'message': f.message,
            'remediation': f.remediation
        } for f in findings],
        scan_time_ms=int(scan_time)
    )


@router.post("/infrastructure/secrets/upload", response_model=SecretScanResult, tags=["Infrastructure Security"])
async def scan_file_for_secrets(file: UploadFile = File(...)):
    """
    **Upload File to Scan for Secrets**
    
    Upload any file to scan for hardcoded secrets.
    """
    content = await file.read()
    
    try:
        content_str = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File appears to be binary, cannot scan")
    
    start_time = datetime.now()
    
    scanner = get_secret_scanner()
    findings = scanner.scan_file(content_str, file.filename or "unknown")
    
    secret_types = list(set(f.type for f in findings))
    
    scan_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return SecretScanResult(
        scan_id=f"secret_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        total_secrets=len(findings),
        secret_types=secret_types,
        findings=[{
            'line': f.line,
            'type': f.type,
            'severity': f.severity,
            'message': f.message,
            'remediation': f.remediation
        } for f in findings],
        scan_time_ms=int(scan_time)
    )


@router.get("/infrastructure/rules", tags=["Infrastructure Security"])
async def get_infrastructure_rules():
    """
    **Get Infrastructure Security Rules**
    
    Returns all infrastructure security rules used by the scanner.
    """
    from ...scanners.infrastructure import (
        DOCKERFILE_RULES,
        COMPOSE_RULES,
        KUBERNETES_RULES,
        SECRET_PATTERNS
    )
    
    return {
        "dockerfile_rules": len(DOCKERFILE_RULES),
        "compose_rules": len(COMPOSE_RULES),
        "kubernetes_rules": len(KUBERNETES_RULES),
        "secret_patterns": len(SECRET_PATTERNS),
        "categories": {
            "dockerfile": [r["id"] for r in DOCKERFILE_RULES],
            "docker_compose": [r["id"] for r in COMPOSE_RULES],
            "kubernetes": [r["id"] for r in KUBERNETES_RULES],
            "secrets": [r["id"] for r in SECRET_PATTERNS]
        }
    }
