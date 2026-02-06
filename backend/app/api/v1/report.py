"""
PDF Report Generation for Vulnerability Scanner
Generates professional security assessment reports

Dependencies:
    pip install reportlab

Usage:
    POST /api/v1/report/pdf
    Body: { "scan_results": {...}, "metadata": {...} }
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from io import BytesIO
import logging

# PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, ListFlowable, ListItem
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not installed. PDF export disabled. Install with: pip install reportlab")

router = APIRouter()
logger = logging.getLogger(__name__)


class VulnerabilityItem(BaseModel):
    cwe_id: str
    severity: str
    message: str
    line: int
    confidence: float = 0.0


class ReportMetadata(BaseModel):
    title: str = "Security Assessment Report"
    project_name: str = "Web Application"
    scan_date: str = ""
    scanned_by: str = "AI Vulnerability Scanner"
    language: str = "Python"


class PDFReportRequest(BaseModel):
    code: str
    vulnerabilities: List[VulnerabilityItem]
    metadata: Optional[ReportMetadata] = None
    risk_score: int = 0
    summary: str = ""


def get_severity_color(severity: str):
    """Get color for severity level"""
    severity = severity.upper()
    colors_map = {
        'CRITICAL': colors.Color(0.8, 0.1, 0.1),  # Dark Red
        'HIGH': colors.Color(0.9, 0.4, 0.1),      # Orange
        'MEDIUM': colors.Color(0.9, 0.7, 0.1),    # Yellow
        'LOW': colors.Color(0.4, 0.4, 0.4),       # Gray
    }
    return colors_map.get(severity, colors.gray)


def generate_pdf_report(request: PDFReportRequest) -> BytesIO:
    """Generate a professional PDF security report"""
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2c5282')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor('#4a5568')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    # Build content
    story = []
    
    # Title
    metadata = request.metadata or ReportMetadata()
    story.append(Paragraph(metadata.title, title_style))
    story.append(Spacer(1, 10))
    
    # Report Info Table
    scan_date = metadata.scan_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_data = [
        ["Project:", metadata.project_name],
        ["Scan Date:", scan_date],
        ["Scanner:", metadata.scanned_by],
        ["Language:", metadata.language],
    ]
    
    info_table = Table(info_data, colWidths=[100, 300])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    # Risk Score Section
    story.append(Paragraph("Executive Summary", heading_style))
    
    risk_score = request.risk_score
    risk_level = "CRITICAL" if risk_score >= 75 else "HIGH" if risk_score >= 50 else "MEDIUM" if risk_score >= 25 else "LOW"
    risk_color = get_severity_color(risk_level)
    
    # Risk Score Display
    risk_data = [[
        f"Risk Score: {risk_score}/100",
        f"Risk Level: {risk_level}"
    ]]
    risk_table = Table(risk_data, colWidths=[200, 200])
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('TEXTCOLOR', (0, 0), (-1, -1), risk_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f7fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 15))
    
    # Summary text
    if request.summary:
        story.append(Paragraph(request.summary, body_style))
    
    # Vulnerability Statistics
    story.append(Paragraph("Vulnerability Statistics", heading_style))
    
    # Count by severity
    severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    for vuln in request.vulnerabilities:
        sev = vuln.severity.upper()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    stats_data = [
        ["Severity", "Count"],
        ["CRITICAL", str(severity_counts.get('CRITICAL', 0))],
        ["HIGH", str(severity_counts.get('HIGH', 0))],
        ["MEDIUM", str(severity_counts.get('MEDIUM', 0))],
        ["LOW", str(severity_counts.get('LOW', 0))],
        ["Total", str(len(request.vulnerabilities))]
    ]
    
    stats_table = Table(stats_data, colWidths=[150, 100])
    stats_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#feb2b2')),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor('#fbd38d')),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor('#faf089')),
        ('BACKGROUND', (0, 4), (0, 4), colors.HexColor('#cbd5e0')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#e2e8f0')),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a0aec0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 20))
    
    # Detailed Findings
    story.append(Paragraph("Detailed Findings", heading_style))
    
    if not request.vulnerabilities:
        story.append(Paragraph("✓ No vulnerabilities detected.", body_style))
    else:
        for i, vuln in enumerate(request.vulnerabilities, 1):
            # Finding header
            sev_color = get_severity_color(vuln.severity)
            finding_header = f"<font color='#{sev_color.hexval()[2:]}'>[{vuln.severity.upper()}]</font> Finding #{i}: {vuln.cwe_id}"
            story.append(Paragraph(finding_header, subheading_style))
            
            # Finding details
            finding_data = [
                ["CWE ID:", vuln.cwe_id],
                ["Severity:", vuln.severity.upper()],
                ["Line:", str(vuln.line)],
                ["Confidence:", f"{vuln.confidence:.0%}"],
            ]
            
            finding_table = Table(finding_data, colWidths=[80, 350])
            finding_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#718096')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(finding_table)
            
            # Message
            story.append(Paragraph(f"<b>Description:</b> {vuln.message}", body_style))
            story.append(Spacer(1, 10))
    
    # Code Section (truncated)
    story.append(PageBreak())
    story.append(Paragraph("Scanned Code", heading_style))
    
    code_lines = request.code.split('\n')[:50]  # First 50 lines
    code_text = '\n'.join(f"{i+1:3d} | {line}" for i, line in enumerate(code_lines))
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=8,
        leftIndent=10,
        fontName='Courier',
        backColor=colors.HexColor('#f7fafc'),
        borderPadding=10,
    )
    
    # Wrap code in preformatted style
    code_para = Paragraph(f"<pre>{code_text}</pre>", code_style)
    story.append(code_para)
    
    if len(request.code.split('\n')) > 50:
        story.append(Paragraph("<i>... (code truncated for brevity)</i>", body_style))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#a0aec0'),
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        f"Generated by AI Vulnerability Scanner | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer


@router.post("/report/pdf", tags=["Reports"])
async def generate_pdf(request: PDFReportRequest):
    """
    **Generate PDF Security Report**
    
    Creates a professional PDF report with:
    - Executive Summary with Risk Score
    - Vulnerability Statistics by Severity
    - Detailed Findings with CWE IDs
    - Scanned Code (first 50 lines)
    
    Returns a downloadable PDF file.
    """
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="PDF generation not available. Install reportlab: pip install reportlab"
        )
    
    try:
        pdf_buffer = generate_pdf_report(request)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"security_report_{timestamp}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@router.get("/report/pdf/status", tags=["Reports"])
async def pdf_status():
    """Check if PDF generation is available"""
    return {
        "available": REPORTLAB_AVAILABLE,
        "message": "PDF generation ready" if REPORTLAB_AVAILABLE else "Install reportlab: pip install reportlab"
    }
