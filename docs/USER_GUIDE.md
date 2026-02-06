# AI-Based Vulnerability Scanner - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Using the Web Interface](#using-the-web-interface)
5. [API Reference](#api-reference)
6. [CI/CD Integration](#cicd-integration)
7. [Understanding Results](#understanding-results)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Introduction

The AI-Based Vulnerability Scanner is a hybrid security analysis tool that combines:
- **Machine Learning** (GNN+LSTM architecture) for intelligent pattern detection
- **Pattern Matching** for precise vulnerability identification

### Key Features
- ✅ Supports **Python**, **JavaScript**, and **TypeScript**
- ✅ Covers **OWASP Top 10** vulnerabilities
- ✅ Provides **line-level precision** for vulnerability location
- ✅ Offers **remediation guidance** with secure code examples
- ✅ Includes **Explainable AI** to understand why code was flagged
- ✅ Supports **CI/CD integration** for DevSecOps workflows

### OWASP Top 10 Coverage
| Category | Description | Detection Status |
|----------|-------------|-----------------|
| A01 | Broken Access Control | ✅ Supported |
| A02 | Cryptographic Failures | ✅ Supported |
| A03 | Injection (SQL, XSS, Command) | ✅ Supported |
| A04 | Insecure Design | ✅ Supported |
| A05 | Security Misconfiguration | ✅ Supported |
| A06 | Vulnerable Components | ⚠️ Limited |
| A07 | Authentication Failures | ✅ Supported |
| A08 | Software Integrity Failures | ✅ Supported |
| A09 | Logging Failures | ✅ Supported |
| A10 | SSRF | ✅ Supported |

---

## Quick Start

### 1. Start the Backend Server
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 2. Start the Frontend
```bash
cd frontend
npm install
npm run dev
```

### 3. Open the Scanner
Navigate to `http://localhost:3000` in your browser.

### 4. Scan Your Code
1. Select the programming language
2. Paste your code in the editor
3. Click "Scan Code"
4. Review the results

---

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- pip (Python package manager)
- npm (Node package manager)

### Backend Setup
```bash
# Clone the repository
git clone <repository-url>
cd AI-BASED-VULNERABILITY-SCANNER

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Install ML dependencies (for full ML support)
pip install -r requirements-ml.txt
```

### Frontend Setup
```bash
cd frontend
npm install
```

### Optional: Install PDF Support
```bash
pip install reportlab
```

### Optional: Install Performance Testing
```bash
pip install locust
```

---

## Using the Web Interface

### Scanner Page (`/`)
1. **Select Language**: Choose Python, JavaScript, or TypeScript
2. **Enter Code**: Paste or type code in the Monaco editor
3. **Adjust Threshold** (Optional): Set ML confidence threshold (default: 0.5)
4. **Scan**: Click "Scan Code" button

### Report Page (`/report`)
After scanning, you'll see:
- **Risk Score**: 0-100 overall security score
- **Summary Cards**: Critical/High/Medium/Low counts
- **Vulnerability List**: Expandable cards with details
- **Code View**: Highlighted vulnerable lines

### Features
- **Filter by Severity**: Click severity badges to filter
- **Export to JSON**: Download results for compliance
- **Export to PDF**: Generate professional report
- **Feedback**: Mark findings as Confirmed or False Positive

---

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### Health Check
```
GET /health
```
Returns server status.

#### ML Scan
```
POST /ml-scan
Content-Type: application/json

{
  "code": "string",
  "language": "python|javascript|typescript",
  "threshold": 0.5
}
```

**Response:**
```json
{
  "is_vulnerable": true,
  "confidence": 0.85,
  "vulnerabilities": [
    {
      "cwe_id": "CWE-89",
      "severity": "CRITICAL",
      "message": "SQL Injection detected",
      "line": 5,
      "confidence": 0.95
    }
  ],
  "model_analysis": {
    "gnn_contribution": 0.6,
    "lstm_contribution": 0.4
  },
  "explanation": "SQL injection pattern detected"
}
```

#### Explainable AI
```
POST /explain
Content-Type: application/json

{
  "code": "string",
  "language": "python"
}
```

Returns token importance scores and feature contributions.

#### Feedback
```
POST /feedback
Content-Type: application/json

{
  "scan_id": "string",
  "vulnerability_index": 0,
  "feedback_type": "confirm|false_positive",
  "code_snippet": "string",
  "cwe_id": "CWE-89"
}
```

#### PDF Report
```
POST /report/pdf
Content-Type: application/json

{
  "code": "string",
  "vulnerabilities": [...],
  "metadata": {
    "title": "Security Report",
    "project_name": "My App"
  },
  "risk_score": 75
}
```

Returns PDF file download.

---

## CI/CD Integration

### Command Line Scanner
```bash
python backend/scripts/cicd_scanner.py ./src \
    --threshold 0.5 \
    --fail-on critical,high \
    --json report.json
```

**Exit Codes:**
- `0`: Scan passed (no critical/high vulnerabilities)
- `1`: Scan failed (vulnerabilities found)
- `2`: Scanner error

### GitHub Actions
```yaml
name: Security Scan
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Start Scanner
        run: |
          cd backend
          pip install -r requirements.txt
          uvicorn app.main:app --host 0.0.0.0 --port 8000 &
          sleep 10
      
      - name: Run Scan
        run: |
          python backend/scripts/cicd_scanner.py ./src \
            --fail-on critical,high \
            --json security-report.json
      
      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: security-report
          path: security-report.json
```

### GitLab CI
```yaml
security_scan:
  stage: test
  script:
    - pip install -r backend/requirements.txt
    - cd backend && uvicorn app.main:app &
    - sleep 10
    - python scripts/cicd_scanner.py ../src --fail-on critical
  artifacts:
    paths:
      - security-report.json
```

---

## Understanding Results

### Severity Levels

| Level | Color | Description |
|-------|-------|-------------|
| **CRITICAL** | 🔴 Red | Immediate exploitation risk. Fix immediately. |
| **HIGH** | 🟠 Orange | Serious vulnerability. Fix before deployment. |
| **MEDIUM** | 🟡 Yellow | Moderate risk. Should be addressed. |
| **LOW** | ⚪ Gray | Minor issue or informational. |

### CWE Categories
- **CWE-89**: SQL Injection
- **CWE-78**: OS Command Injection
- **CWE-79**: Cross-Site Scripting (XSS)
- **CWE-94**: Code Injection
- **CWE-22**: Path Traversal
- **CWE-502**: Insecure Deserialization
- **CWE-327**: Weak Cryptography
- **CWE-798**: Hardcoded Credentials
- **CWE-918**: Server-Side Request Forgery (SSRF)

### Model Analysis
- **GNN Contribution**: How much the structural analysis (AST patterns) contributed
- **LSTM Contribution**: How much the sequential analysis (token patterns) contributed
- Higher GNN = structural vulnerability (dangerous function calls)
- Higher LSTM = sequential vulnerability (data flow issues)

---

## Troubleshooting

### Backend Won't Start
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Model Loading Error
```bash
# Ensure model files exist
ls backend/ml/models/
# Should show: hybrid_model_best.pth, vocab.json

# Retrain if needed
cd backend
python training/train.py
```

### Frontend Connection Error
1. Check backend is running on port 8000
2. Verify CORS settings in `backend/app/core/config.py`
3. Check browser console for errors

### PDF Export Not Working
```bash
pip install reportlab
```

### Performance Issues
- Reduce code size (scan smaller chunks)
- Lower concurrent users
- Check system resources (CPU, RAM)

---

## FAQ

### Q: What languages are supported?
**A:** Python, JavaScript, and TypeScript. More languages can be added by extending the pattern scanner.

### Q: How accurate is the detection?
**A:** Based on our testing:
- 72% accuracy on synthetic test suite
- 100% OWASP Top 10 coverage on pattern-based detection
- ML model provides additional confidence scoring

### Q: Can I use this in production?
**A:** This is designed as an academic/research tool. For production use:
- Review and customize patterns for your codebase
- Test with your specific vulnerability types
- Consider combining with established tools (Semgrep, SonarQube)

### Q: How do I add custom patterns?
**A:** Edit `backend/app/scanners/simple_scanner.py` and add patterns to the `VULNERABILITY_PATTERNS` list.

### Q: Is my code sent anywhere?
**A:** No. All processing happens locally. No code is transmitted to external servers.

### Q: How do I contribute feedback?
**A:** Use the Confirm/False Positive buttons on findings. This data is stored locally for future model retraining.

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review API documentation at `http://localhost:8000/docs`
3. Open an issue on the repository

---

*AI-Based Vulnerability Scanner - Senior Project 2026*
