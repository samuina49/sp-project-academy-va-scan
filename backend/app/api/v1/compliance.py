"""
Compliance & Reporting API Endpoints
SARIF output, compliance reports, and export functionality
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from ...utils.compliance import (
    get_sarif_generator,
    get_compliance_checker,
    ComplianceReport,
    OWASP_ASVS_MAPPING,
    PCI_DSS_MAPPING,
    CWE_TO_OWASP_2021
)

router = APIRouter()


# ==================== Request/Response Models ====================

class SARIFRequest(BaseModel):
    findings: List[Dict]
    source_file: Optional[str] = None


class ComplianceReportRequest(BaseModel):
    scan_id: str
    findings: List[Dict]
    frameworks: List[str] = ["OWASP_ASVS", "PCI_DSS"]


class FindingComplianceInfo(BaseModel):
    finding: Dict
    compliance_mappings: Dict


# ==================== Endpoints ====================

@router.post("/compliance/sarif", tags=["Compliance"])
async def generate_sarif_report(request: SARIFRequest):
    """
    **Generate SARIF Report**
    
    Generates a SARIF (Static Analysis Results Interchange Format) report
    from scan findings. SARIF is widely supported by security tools and
    CI/CD platforms like GitHub Code Scanning.
    
    Response format follows SARIF v2.1.0 specification.
    """
    generator = get_sarif_generator()
    sarif = generator.generate(request.findings, request.source_file)
    
    return JSONResponse(
        content=sarif,
        media_type="application/sarif+json",
        headers={"Content-Disposition": "attachment; filename=scan-results.sarif"}
    )


@router.post("/compliance/report", response_model=ComplianceReport, tags=["Compliance"])
async def generate_compliance_report(request: ComplianceReportRequest):
    """
    **Generate Compliance Report**
    
    Analyzes findings against compliance frameworks and generates a report.
    
    Supported frameworks:
    - OWASP_ASVS: OWASP Application Security Verification Standard
    - PCI_DSS: Payment Card Industry Data Security Standard
    - OWASP_TOP_10: OWASP Top 10 2021
    """
    checker = get_compliance_checker()
    report = checker.generate_compliance_report(
        request.scan_id,
        request.findings,
        request.frameworks
    )
    return report


@router.post("/compliance/check-finding", response_model=FindingComplianceInfo, tags=["Compliance"])
async def check_finding_compliance(finding: Dict):
    """
    **Check Single Finding Against Compliance**
    
    Returns compliance mappings for a single vulnerability finding.
    """
    checker = get_compliance_checker()
    mappings = checker.get_compliance_info(finding)
    
    return FindingComplianceInfo(
        finding=finding,
        compliance_mappings={k: v.dict() for k, v in mappings.items()}
    )


@router.get("/compliance/frameworks", tags=["Compliance"])
async def get_supported_frameworks():
    """
    **Get Supported Compliance Frameworks**
    
    Returns information about supported compliance frameworks.
    """
    return {
        "frameworks": [
            {
                "id": "OWASP_ASVS",
                "name": "OWASP Application Security Verification Standard",
                "version": "4.0.3",
                "description": "Provides a basis for testing application technical security controls",
                "url": "https://owasp.org/www-project-application-security-verification-standard/"
            },
            {
                "id": "PCI_DSS",
                "name": "Payment Card Industry Data Security Standard",
                "version": "4.0",
                "description": "Security standard for organizations that handle cardholder data",
                "url": "https://www.pcisecuritystandards.org/"
            },
            {
                "id": "OWASP_TOP_10",
                "name": "OWASP Top 10",
                "version": "2021",
                "description": "Top 10 web application security risks",
                "url": "https://owasp.org/Top10/"
            }
        ]
    }


@router.get("/compliance/owasp-asvs", tags=["Compliance"])
@router.get("/compliance/asvs-requirements", tags=["Compliance"])
async def get_asvs_requirements():
    """
    **Get OWASP ASVS Requirements**
    
    Returns the complete ASVS requirement mapping used by the scanner.
    """
    return {
        "version": "4.0.3",
        "mappings": OWASP_ASVS_MAPPING
    }


@router.get("/compliance/pci-dss", tags=["Compliance"])
@router.get("/compliance/pci-requirements", tags=["Compliance"])
async def get_pci_requirements():
    """
    **Get PCI-DSS Requirements**
    
    Returns the PCI-DSS requirement mapping used by the scanner.
    """
    return {
        "version": "4.0",
        "mappings": PCI_DSS_MAPPING
    }


@router.get("/compliance/cwe-owasp-mapping", tags=["Compliance"])
async def get_cwe_owasp_mapping():
    """
    **Get CWE to OWASP Top 10 Mapping**
    
    Returns the mapping between CWE IDs and OWASP Top 10 2021 categories.
    """
    return {
        "version": "2021",
        "mappings": CWE_TO_OWASP_2021
    }


@router.post("/compliance/export/json", tags=["Compliance"])
async def export_json_report(request: ComplianceReportRequest):
    """
    **Export JSON Report**
    
    Exports a comprehensive JSON report including findings and compliance info.
    """
    checker = get_compliance_checker()
    report = checker.generate_compliance_report(
        request.scan_id,
        request.findings,
        request.frameworks
    )
    
    export_data = {
        "report_metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "scan_id": request.scan_id,
            "tool": "AI Vulnerability Scanner",
            "version": "1.0.0"
        },
        "summary": {
            "total_findings": report.total_findings,
            "compliance_score": report.compliance_score,
            "frameworks_checked": report.frameworks_checked
        },
        "findings": request.findings,
        "compliance_report": report.dict(),
        "recommendations": report.recommendations
    }
    
    return JSONResponse(
        content=export_data,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=vuln-report-{request.scan_id}.json"}
    )


@router.post("/compliance/export/csv", tags=["Compliance"])
async def export_csv_report(request: ComplianceReportRequest):
    """
    **Export CSV Report**
    
    Exports findings as CSV for spreadsheet analysis.
    """
    import csv
    import io
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        "Line", "Type", "Severity", "CWE", "OWASP", "Message",
        "ASVS Requirement", "PCI-DSS Requirement"
    ])
    
    checker = get_compliance_checker()
    
    for finding in request.findings:
        mappings = checker.get_compliance_info(finding)
        
        asvs_req = ""
        pci_req = ""
        
        if "OWASP_ASVS" in mappings:
            asvs_req = mappings["OWASP_ASVS"].requirement_id
        if "PCI_DSS" in mappings:
            pci_req = mappings["PCI_DSS"].requirement_id
        
        writer.writerow([
            finding.get("line", ""),
            finding.get("type", ""),
            finding.get("severity", ""),
            finding.get("cwe", ""),
            finding.get("owasp", ""),
            finding.get("message", ""),
            asvs_req,
            pci_req
        ])
    
    csv_content = output.getvalue()
    
    return JSONResponse(
        content={"csv": csv_content},
        headers={"Content-Disposition": f"attachment; filename=vuln-report-{request.scan_id}.csv"}
    )


@router.get("/compliance/score", tags=["Compliance"])
async def calculate_compliance_score(
    critical: int = Query(0, ge=0),
    high: int = Query(0, ge=0),
    medium: int = Query(0, ge=0),
    low: int = Query(0, ge=0)
):
    """
    **Calculate Compliance Score**
    
    Calculates a weighted compliance score based on finding counts.
    
    Scoring:
    - Critical: -25 points each
    - High: -15 points each
    - Medium: -5 points each
    - Low: -1 point each
    
    Score is normalized to 0-100 range.
    """
    base_score = 100
    
    deductions = (
        critical * 25 +
        high * 15 +
        medium * 5 +
        low * 1
    )
    
    score = max(0, base_score - deductions)
    
    # Determine grade
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "score": score,
        "grade": grade,
        "breakdown": {
            "critical": {"count": critical, "deduction": critical * 25},
            "high": {"count": high, "deduction": high * 15},
            "medium": {"count": medium, "deduction": medium * 5},
            "low": {"count": low, "deduction": low * 1}
        },
        "recommendations": [
            "Address all critical vulnerabilities immediately",
            "Prioritize high severity findings",
            "Review and fix medium severity issues",
            "Track low severity issues for future sprints"
        ] if score < 100 else ["No vulnerabilities detected - maintain secure coding practices"]
    }
