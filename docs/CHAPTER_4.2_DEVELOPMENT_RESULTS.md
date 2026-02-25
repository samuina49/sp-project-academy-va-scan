# บทที่ 4: ผลการดำเนินงาน

## 4.2 ผลการพัฒนาระบบ

การพัฒนาระบบตรวจสอบช่องโหว่ความปลอดภัยของเว็บแอปพลิเคชันด้วยปัญญาประดิษฐ์แบ่งออกเป็น 5 ส่วนหลัก ดังนี้

---

### 4.2.1 ส่วนติดต่อผู้ใช้ (Frontend Web Interface)

ระบบได้รับการพัฒนาส่วนติดต่อผู้ใช้ด้วย **Next.js 14** และ **TypeScript** โดยมีความสามารถหลักดังนี้:

#### คุณสมบัติหลัก:
- **Code Editor แบบ Real-time** - ใช้ Monaco Editor (ตัวเดียวกับ VS Code)
  - รองรับ Syntax Highlighting สำหรับ Python, JavaScript, TypeScript
  - Line numbering และ Code folding
  - Auto-completion พื้นฐาน

- **การอัพโหลดไฟล์หลายรูปแบบ**
  - อัพโหลดไฟล์เดี่ยว (.py, .js, .ts)
  - อัพโหลด ZIP สำหรับโปรเจกต์ทั้งหมด (สูงสุด 50 MB)
  - วิเคราะห์ไฟล์หลายไฟล์พร้อมกัน

- **Dashboard การแสดงผล**
  - สรุปจำนวนช่องโหว่แยกตามระดับความรุนแรง (Critical, High, Medium, Low)
  - กราฟแสดงการกระจายช่องโหว่ตาม OWASP Top 10
  - แสดงรายการไฟล์ที่มีปัญหา

- **Panel แสดงผลการสแกน**
  - แสดงช่องโหว่พร้อมตำแหน่งบรรทัด (Line-level precision)
  - แสดงรหัส CWE (Common Weakness Enumeration)
  - คำอธิบายช่องโหว่และความเสี่ยง
  - **คำแนะนำในการแก้ไข (Remediation Advice)** พร้อมตัวอย่าง Code ที่ปลอดภัย

#### เทคโนโลยีที่ใช้:
```
Frontend Stack:
├── Next.js 14 (App Router)
├── TypeScript
├── TailwindCSS (Styling)
├── Monaco Editor (Code Editor)
├── Recharts (Data Visualization)
├── Axios (API Communication)
└── React Icons (UI Icons)
```

**📸 รูปภาพที่ควรใส่:**
- **ภาพที่ 4-1**: หน้าจอหลักของระบบ (Main Dashboard)
- **ภาพที่ 4-2**: หน้าจอ Code Editor พร้อมตัวอย่างโค้ด
- **ภาพที่ 4-3**: หน้าจอแสดงผลการสแกน (Results Panel) พร้อมช่องโหว่ที่พบ
- **ภาพที่ 4-4**: หน้าจอการอัพโหลดไฟล์ ZIP

---

### 4.2.2 ส่วนระบบ Backend และ REST API

ระบบ Backend พัฒนาด้วย **FastAPI** (Python 3.10+) ซึ่งเป็น Modern Web Framework ที่มีประสิทธิภาพสูง

#### API Endpoints หลัก:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/scan/code` | POST | สแกนโค้ดที่ Copy-Paste เข้ามา |
| `/api/v1/scan/zip` | POST | สแกนไฟล์ ZIP ของโปรเจกต์ |
| `/api/v1/scan/hybrid` | POST | สแกนแบบ Hybrid (Pattern + ML) |
| `/api/v1/health` | GET | ตรวจสอบสถานะระบบ |
| `/api/v1/export/pdf` | POST | Export รายงานเป็น PDF |
| `/api/v1/export/sarif` | POST | Export รายงานเป็น SARIF format |

#### สถาปัตยกรรม Backend:

```
backend/
├── app/
│   ├── api/v1/           # API Endpoints
│   │   ├── scan.py       # Scan endpoints
│   │   ├── export.py     # Export endpoints
│   │   └── health.py     # Health check
│   │
│   ├── core/             # Core components
│   │   ├── config.py     # Configuration
│   │   ├── security.py   # Security utilities (ZIP validation, Path traversal protection)
│   │   └── temp_manager.py  # Temporary file management
│   │
│   ├── scanners/         # Scanner modules
│   │   ├── bandit_scanner.py    # Python scanner
│   │   ├── semgrep_scanner.py   # JS/TS scanner
│   │   └── scanner_orchestrator.py  # Scanner coordinator
│   │
│   ├── hybrid_scanner/   # Hybrid detection
│   │   ├── hybrid_detector.py   # Main hybrid logic
│   │   ├── pattern_matcher.py   # Pattern-based detection
│   │   └── ml_detector.py       # ML-based detection
│   │
│   ├── ml/               # Machine Learning
│   │   ├── model.py      # Model architecture
│   │   ├── inference.py  # Model inference
│   │   └── preprocessing.py  # Data preprocessing
│   │
│   └── models/           # Data models
│       └── schemas.py    # Pydantic schemas
```

#### ระบบความปลอดภัย (Security Layer):
- **ZIP Slip Protection** - ป้องกันการแตก ZIP ที่มี Path Traversal
- **Path Validation** - ตรวจสอบเส้นทางไฟล์ไม่ให้เข้าถึงพื้นที่นอกขอบเขต
- **File Size Limiting** - จำกัดขนาดไฟล์สูงสุด 50 MB
- **Temp File Management** - ลบไฟล์ชั่วคราวอัตโนมัติหลังสแกนเสร็จ
- **Rate Limiting** - จำกัดจำนวน requests ต่อนาที (60 requests/min)

#### Performance Optimization:
- **Async I/O** - ใช้ async/await สำหรับการทำงานแบบ non-blocking
- **Connection Pooling** - ใช้ connection pool สำหรับ database
- **Caching** - Cache ผลลัพธ์ของไฟล์ที่เคยสแกน (based on file hash)
- **Background Tasks** - ใช้ FastAPI Background Tasks สำหรับการทำงานหนัก

**📸 รูปภาพที่ควรใส่:**
- **ภาพที่ 4-5**: สถาปัตยกรรมระบบ Backend (Backend Architecture Diagram)
- **ภาพที่ 4-6**: API Documentation หน้าจอ Swagger UI (FastAPI Auto-generated Docs)
- **ภาพที่ 4-7**: Flowchart การทำงานของ API

---

### 4.2.3 ระบบ Hybrid Scanner

ระบบ Hybrid Scanner เป็นหัวใจสำคัญของโครงงาน ซึ่งผสมผสานระหว่าง **Pattern-Matching** และ **Machine Learning**

#### ส่วนประกอบหลัก:

##### 1. Pattern-Matching Engine (น้ำหนัก 60%)

**1.1 Semgrep Scanner**
- เวอร์ชัน: 1.99.0
- รองรับภาษา: JavaScript, TypeScript, JSX, TSX
- จำนวนกฎ: 120+ custom rules
- คุณสมบัติ:
  - วิเคราะห์ AST (Abstract Syntax Tree)
  - ตรวจจับรูปแบบ Taint Analysis
  - รองรับ Metavariables และ Ellipsis pattern

**1.2 Bandit Scanner**
- เวอร์ชัน: 1.8.0
- เฉพาะภาษา: Python
- จำนวนกฎ: 60+ security checks
- คุณสมบัติ:
  - Built-in CWE mapping
  - Confidence scoring
  - Custom plugin support

**1.3 Simple Pattern Matcher**
- ภาษาที่รองรับ: Python, JavaScript, TypeScript
- จำนวนกฎ: 40+ regex patterns
- จุดเด่น: รวดเร็ว สำหรับรูปแบบง่ายๆ

##### 2. Machine Learning Engine (น้ำหนัก 40%)

**Architecture: Hybrid GNN + BiLSTM**

```
Input: Source Code
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
Tokenization      AST Generation        Feature Extraction
    │                     │                     │
    ▼                     ▼                     ▼
Token Embeddings    Graph Construction    Static Features
(Vocab: 3,336)      (Nodes + Edges)      (CFG, DFG)
    │                     │                     │
    ▼                     ▼                     │
┌─────────────┐    ┌─────────────┐            │
│   BiLSTM    │    │     GNN     │            │
│  (2 layers) │    │  (3 layers) │            │
│  Hidden:128 │    │  Hidden:128 │            │
│  Dropout:0.2│    │  Heads: 4   │            │
└──────┬──────┘    └──────┬──────┘            │
       │                  │                    │
       └────────┬─────────┴────────────────────┘
                │
                ▼
         Feature Fusion Layer
         (Concatenation)
                │
                ▼
         Fully Connected Layers
         (128 → 64 → 32 → 1)
         ReLU + Dropout(0.2)
                │
                ▼
         Output: Binary Classification
         - Vulnerable (1) or Safe (0)
         - Confidence Score (0.0-1.0)
```

**Model Specifications:**
- Total Parameters: **1,905,409** (1.9M)
- Vocabulary Size: **3,336 tokens**
- Input Max Length: **512 tokens**
- Training Time: ~8 hours (on NVIDIA RTX 3060)
- Inference Time: **< 0.5 seconds per file**

##### 3. Result Fusion Strategy

การรวมผลจาก 2 ส่วน:

```python
# Weighted Fusion
final_score = (pattern_confidence × 0.6) + (ml_confidence × 0.4)

# Decision Threshold
if final_score >= 0.65:
    verdict = "Vulnerable"
elif final_score >= 0.35:
    verdict = "Suspicious (Manual Review Recommended)"
else:
    verdict = "Safe"
```

**Deduplication Strategy:**
- ใช้ (file_path, line_number, vulnerability_type) เป็น unique key
- ถ้าทั้ง Pattern และ ML ตรวจพบที่ตำแหน่งเดียวกัน → รวมเป็น 1 รายการ
- ใช้ confidence สูงสุดจากทั้ง 2 วิธี
- รวม description จากทั้ง 2 แหล่ง

#### Hybrid Detection Workflow:

```
1. File Input → Validate file type
2. Duplicate file to 2 pipelines ↓
   ├── Pattern-Matching Pipeline
   │   ├── Language Detection
   │   ├── Select Scanner (Bandit/Semgrep)
   │   ├── Execute Scan
   │   └── Parse Results
   │
   └── ML Pipeline
       ├── Tokenization
       ├── AST Generation
       ├── Feature Extraction
       ├── Model Inference
       └── Post-processing
3. Merge Results (Deduplication + Fusion)
4. Sort by Severity + Confidence
5. Return Final Report
```

**📸 รูปภาพที่ควรใส่:**
- **ภาพที่ 4-8**: Hybrid Scanner Architecture (แผนภาพการทำงานของ Hybrid System)
- **ภาพที่ 4-9**: Flowchart การทำงานของ Hybrid Detection
- **ภาพที่ 4-10**: ตัวอย่างการ Fusion ผลจาก Pattern + ML

---

### 4.2.4 โมเดล Machine Learning

#### Dataset และการเตรียมข้อมูล

##### 1. แหล่งที่มาของข้อมูล (Multi-Source Approach)

```
Total Samples: 3,117
├── Big-Vul Dataset:           850 samples (Real CVEs)
├── SARD Patterns:             420 samples (Reference patterns)
├── GitHub Security Advisory:  680 samples (Advisory-inspired)
├── Generated Patterns:        850 samples (Synthetic with variations)
└── Safe Code Samples:         317 samples (Secure implementations)
```

##### 2. การแบ่งข้อมูล (Fingerprint-Based Splitting)

**Innovation:** ใช้ Fingerprint-based splitting เพื่อป้องกัน Data Leakage

```python
# Fingerprint Calculation
fingerprint = hash(code_structure + variable_pattern + ast_signature)
```

**Dataset Splits:**
```
Training Set:     2,491 samples (80%)
Validation Set:     319 samples (10%)
Test Set:           307 samples (10%)

Unique Fingerprints: 3,009 (96.5% unique)
→ ป้องกัน Data Leakage ได้ดีมาก
```

##### 3. Class Distribution

```
Vulnerable: 1,840 samples (59%)
Safe:       1,277 samples (41%)

→ Balanced และสะท้อนสถานการณ์จริง
```

##### 4. Vulnerability Types Coverage

| Vulnerability Type | จำนวน Samples |
|-------------------|-------------|
| SQL Injection | 380 |
| Command Injection | 310 |
| XSS (Cross-Site Scripting) | 285 |
| Path Traversal | 270 |
| Hardcoded Credentials | 195 |
| Insecure Deserialization | 160 |
| SSRF | 145 |
| XML External Entity (XXE) | 95 |

#### Training Configuration

```yaml
Optimizer: AdamW
Learning Rate: 0.0005 (with cosine annealing)
Batch Size: 64
Epochs: 100
Early Stopping: Patience 15 epochs
Loss Function: BCEWithLogitsLoss
Weight Decay: 0.0001
Gradient Clipping: max_norm=1.0
LR Scheduler: CosineAnnealingLR
```

#### Training Hardware

```
GPU: NVIDIA GeForce RTX 3060 (12GB VRAM)
RAM: 32GB DDR4
Training Time: ~8 hours
```

**📸 รูปภาพที่ควรใส่:**
- **ภาพที่ 4-11**: Model Architecture Diagram (แผนภาพโครงสร้างโมเดล GNN+LSTM)
- **ภาพที่ 4-12**: Training Curves (Loss และ Accuracy ตลอด 100 epochs)

---

### 4.2.5 ระบบรายงานและการแสดงผล

#### รูปแบบการรายงาน

##### 1. JSON Report (Default)

```json
{
  "scan_id": "unique-scan-id",
  "timestamp": "2026-02-22T10:30:00Z",
  "summary": {
    "total_files": 15,
    "total_vulnerabilities": 23,
    "critical": 3,
    "high": 8,
    "medium": 9,
    "low": 3
  },
  "vulnerabilities": [
    {
      "file": "app/auth.py",
      "line": 45,
      "severity": "CRITICAL",
      "type": "sql_injection",
      "cwe": "CWE-89",
      "owasp": "A03:2021 - Injection",
      "confidence": "HIGH (95%)",
      "description": "Unsanitized user input in SQL query",
      "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
      "remediation": "Use parameterized queries instead...",
      "references": [
        "https://owasp.org/www-community/attacks/SQL_Injection"
      ]
    }
  ],
  "scanning_time": "3.42s",
  "scanner_versions": {
    "semgrep": "1.99.0",
    "bandit": "1.8.0",
    "ml_model": "v2.1.0"
  }
}
```

##### 2. PDF Report

รายงาน PDF แบบมืออาชีพประกอบด้วย:
- **Executive Summary** - สรุปผลการสแกนภาพรวม
- **Vulnerability Breakdown** - แยกรายละเอียดตามหมวดหมู่
- **OWASP Top 10 Mapping** - แมพช่องโหว่กับ OWASP
- **Risk Assessment** - ประเมินความเสี่ยงโดยรวม
- **Detailed Findings** - รายละเอียดช่องโหว่แต่ละรายการ
- **Remediation Guidelines** - แนวทางการแก้ไข
- **Code Snippets** - ส่วนโค้ดที่มีปัญหา
- **Appendix** - CWE mapping และ References

##### 3. SARIF Format (Static Analysis Results Interchange Format)

รองรับการ integrate กับ:
- GitHub Security Tab
- Azure DevOps
- GitLab Security Dashboard
- Visual Studio Code

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "AI-VA-Scanner",
        "version": "2.1.0"
      }
    },
    "results": [...]
  }]
}
```

##### 4. CSV Export

สำหรับ:
- การ Import เข้า Excel/Google Sheets
- การวิเคราะห์ใน Power BI, Tableau
- การรวมรายงานหลายครั้ง

#### Dashboard Visualization

##### 1. Severity Distribution Chart
- Pie Chart แสดงสัดส่วน Critical/High/Medium/Low
- Color-coded: Red (Critical), Orange (High), Yellow (Medium), Blue (Low)

##### 2. OWASP Top 10 Coverage Chart
- Bar Chart แสดงจำนวนช่องโหว่แต่ละประเภท
- แกน X: OWASP Categories
- แกน Y: จำนวนช่องโหว่

##### 3. File-Level Heatmap
- แสดงไฟล์ที่มีช่องโหว่มากที่สุด
- สีเข้มขึ้นตามจำนวนช่องโหว่

##### 4. Timeline View
- แสดงประวัติการสแกน
- เปรียบเทียบผลระหว่างการสแกนแต่ละครั้ง
- Track การปรับปรุงแก้ไข

##### 5. Risk Score Calculation

```python
risk_score = (
    (critical_count × 10) +
    (high_count × 5) +
    (medium_count × 2) +
    (low_count × 1)
) / total_lines_of_code × 100

Risk Level:
- 0-20:   Low Risk (Green)
- 21-50:  Medium Risk (Yellow)
- 51-80:  High Risk (Orange)
- 81-100: Critical Risk (Red)
```

**📸 รูปภาพที่ควรใส่:**
- **ภาพที่ 4-13**: Dashboard หลักแสดง Summary และ Charts
- **ภาพที่ 4-14**: ตัวอย่าง PDF Report (หน้าแรก Executive Summary)
- **ภาพที่ 4-15**: ตัวอย่างรายละเอียดช่องโหว่ใน PDF Report
- **ภาพที่ 4-16**: OWASP Top 10 Distribution Chart
- **ภาพที่ 4-17**: File-Level Heatmap

---

## สรุปภาพรวมระบบที่พัฒนา

ระบบที่ถูกพัฒนาขึ้นเป็นส่วนหนึ่งของระบบตรวจสอบช่องโหว่ความปลอดภัยของเว็บแอปพลิเคชันด้วยปัญญาประดิษฐ์ ซึ่งมีหน้าจอหลักดังภาพที่ 4-1 โดยประกอบด้วยส่วนสำคัญ 5 ส่วน ได้แก่

1. **Web Interface** - ส่วนติดต่อผู้ใช้แบบ Web Application พัฒนาด้วย Next.js
2. **Backend API** - ระบบ REST API พัฒนาด้วย FastAPI พร้อม Security Layer
3. **Hybrid Scanner** - ระบบสแกนแบบผสมผสานระหว่าง Pattern-Matching และ Machine Learning
4. **ML Model** - โมเดล Deep Learning (GNN+LSTM)
5. **Reporting System** - ระบบสร้างรายงานหลายรูปแบบ (JSON, PDF, SARIF, CSV)

---

## 📊 สรุปตารางรูปภาพทั้งหมดที่แนะนำ (17 ภาพ)

| รหัสภาพ | ชื่อภาพ | ส่วนที่ |
|---------|---------|---------|
| ภาพที่ 4-1 | หน้าจอหลักของระบบ (Main Dashboard) | 4.2.1 |
| ภาพที่ 4-2 | หน้าจอ Code Editor | 4.2.1 |
| ภาพที่ 4-3 | หน้าจอแสดงผลการสแกน (Results Panel) | 4.2.1 |
| ภาพที่ 4-4 | หน้าจอการอัพโหลดไฟล์ ZIP | 4.2.1 |
| ภาพที่ 4-5 | สถาปัตยกรรมระบบ Backend | 4.2.2 |
| ภาพที่ 4-6 | API Documentation (Swagger UI) | 4.2.2 |
| ภาพที่ 4-7 | Flowchart การทำงานของ API | 4.2.2 |
| ภาพที่ 4-8 | Hybrid Scanner Architecture | 4.2.3 |
| ภาพที่ 4-9 | Flowchart การทำงานของ Hybrid Detection | 4.2.3 |
| ภาพที่ 4-10 | ตัวอย่างการ Fusion ผลจาก Pattern + ML | 4.2.3 |
| ภาพที่ 4-11 | Model Architecture (GNN+LSTM) | 4.2.4 |
| ภาพที่ 4-12 | Training Curves (Loss และ Accuracy) | 4.2.4 |
| ภาพที่ 4-13 | Dashboard แสดง Summary และ Charts | 4.2.5 |
| ภาพที่ 4-14 | ตัวอย่าง PDF Report (Executive Summary) | 4.2.5 |
| ภาพที่ 4-15 | รายละเอียดช่องโหว่ใน PDF Report | 4.2.5 |
| ภาพที่ 4-16 | OWASP Top 10 Distribution Chart | 4.2.5 |
| ภาพที่ 4-17 | File-Level Heatmap | 4.2.5 |

---

## 💡 คำแนะนำในการถ่าย/สร้างรูปภาพ

### 1. Screenshots ของระบบจริง (ถ่าย)
- **ภาพที่ 4-1 ถึง 4-4:** เปิดระบบและจับภาพหน้าจอจริง
  - ใช้ Snipping Tool หรือ Screenshot tool
  - ครอบตัดให้สวยงามและชัดเจน
  - Highlight ส่วนสำคัญด้วยกรอบหรือลูกศร

- **ภาพที่ 4-6:** Swagger UI
  - เข้า `http://localhost:8000/docs`
  - จับภาพหน้า API documentation

### 2. Diagrams และ Flowcharts (วาด/สร้าง)
- **เครื่องมือแนะนำ:**
  - **Draw.io** (https://app.diagrams.net/) - ฟรี, ใช้งานง่าย
  - **Lucidchart** - Professional diagrams
  - **Figma** - UI/UX และ diagrams
  - **PowerPoint** - SmartArt สำหรับ flowcharts

- **ภาพที่ควรสร้าง:**
  - ภาพที่ 4-5, 4-7, 4-8, 4-9, 4-11: Architecture diagrams
  - ใช้สีสันที่สอดคล้องกัน
  - มี legend อธิบายสัญลักษณ์
  - Font ขนาดอ่านง่าย (อย่างน้อย 14pt)

### 3. Charts และ Graphs (Generate จาก Code)
- **Python Matplotlib/Seaborn:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Training Curves
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Training Progress')
plt.legend()
plt.savefig('training_curves.png', dpi=300)
```

- **Confusion Matrix:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300)
```

### 4. คุณภาพรูปภาพ
- **Resolution:** อย่างน้อย 1920x1080 (Full HD)
- **DPI:** 300 DPI สำหรับการพิมพ์
- **Format:** PNG สำหรับ screenshots, SVG สำหรับ diagrams
- **File Size:** ไม่เกิน 2 MB ต่อรูป (ถ้าใส่ใน LaTeX/Word)

### 5. การอ้างอิงในเนื้อหา
```latex
% ในเอกสาร LaTeX
ดังแสดงในภาพที่ 4-1 หน้าจอหลักของระบบประกอบด้วย...

\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/main_dashboard.png}
\caption{หน้าจอหลักของระบบ}
\label{fig:main_dashboard}
\end{figure}
```

---

## 📝 Next Steps

1. **เริ่มจาก Screenshots ง่ายๆ:**
   - เปิดระบบและจับภาพหน้าจอหลัก
   - ภาพที่ 4-1, 4-2, 4-3 ลำดับนี้

2. **สร้าง Diagrams:**
   - เริ่มจาก Architecture Diagram (ภาพที่ 4-5, 4-8)
   - ใช้ Draw.io หรือ PowerPoint

3. **Generate Charts จาก Code:**
   - รัน script Python เพื่อ generate graphs
   - Training curves, Confusion Matrix, Performance Charts

4. **จัดเรียงใส่เอกสาร:**
   - ใส่รูปภาพตามลำดับส่วน
   - ใส่ Caption และอ้างอิงในเนื้อหา
   - ตรวจสอบความสวยงามและความชัดเจน

