# เอกสารเตรียมนำเสนอโปสเตอร์
## AI-Based Vulnerability Scanner for Web Applications
### สำหรับฝึกซ้อมการนำเสนอ — Science Exhibition Day

---

## ส่วนที่ 1: เปิดการนำเสนอ (Opening - ~30 วินาที)

**สคริปต์:**
> "สวัสดีครับ โปรเจคของผมคือระบบ AI สำหรับตรวจสอบช่องโหว่ความปลอดภัยในโค้ดของเว็บแอปพลิเคชัน  
> ปัญหาที่เราแก้คือ นักพัฒนาเขียนโค้ดที่มีช่องโหว่โดยไม่รู้ตัว เช่น SQL Injection, XSS, หรือ Command Injection  
> ระบบของเราตรวจสอบโค้ดแบบอัตโนมัติโดยใช้ทั้ง Pattern Matching และ Deep Learning รวมกัน  
> ซึ่งทำให้ได้ความแม่นยำสูงกว่าการใช้วิธีเดียว"

---

## ส่วนที่ 2: ภาพรวมระบบ (System Overview - ~1 นาที)

### จุดที่ต้องพูด

**ระบบแบ่งเป็น 3 ชั้น:**

1. **Frontend (Next.js ที่ Port 3000)**
   - ผู้ใช้วางโค้ดลงใน Monaco Editor (editor เดียวกับ VS Code)
   - กด Scan → ระบบส่งโค้ดไปที่ Backend ผ่าน REST API
   - ผลลัพธ์กลับมาแสดงเป็นรายการช่องโหว่ พร้อมบรรทัด, ประเภท CWE, และวิธีแก้ไข

2. **Backend (FastAPI ที่ Port 8000)**
   - รับโค้ด → ตรวจสอบ Input → ส่งเข้า Scanner Engine
   - มีระบบ Security: Rate Limiting (10 req/นาที), Input Validation, JWT Auth

3. **Scanner Engine (สองเฟส)**
   - **Phase 1:** Pattern Matching (Bandit + Semgrep + Custom Rules)
   - **Phase 2:** ML Model (GNN + BiLSTM)
   - **Phase 3:** Hybrid Combiner รวมผลลัพธ์

**แผนภาพง่าย ๆ สำหรับชี้บนโปสเตอร์:**
```
[ผู้ใช้วางโค้ด] → [Frontend] → [FastAPI Backend]
                                      ↓
                        ┌─────────────┴─────────────┐
                        ↓                           ↓
               [Pattern Scanner]           [ML Model (AI)]
               Bandit/Semgrep              GNN + BiLSTM
                        ↓                           ↓
                        └─────────────┬─────────────┘
                                      ↓
                              [Hybrid Combiner]
                                      ↓
                           [ผลลัพธ์ + คำแนะนำ]
```

---

## ส่วนที่ 3: Phase 1 — Pattern Matching ทำงานอย่างไร (~2 นาที)

### จุดที่ต้องพูด

**สคริปต์:**
> "Phase แรกคือ Pattern Matching — ทำงานแบบเร็ว ใช้เวลาน้อยกว่า 1 วินาที"

**เครื่องมือที่ใช้:**

| เครื่องมือ | ใช้กับภาษา | หน้าที่ |
|-----------|-----------|--------|
| **Bandit** | Python เท่านั้น | ค้นหา insecure functions เช่น `os.system()`, `eval()`, `pickle.load()` |
| **Semgrep** | JavaScript / TypeScript | ค้นหา patterns เช่น `innerHTML`, `document.write()`, SQL concatenation |
| **Simple Pattern Scanner** | ทุกภาษา | Custom regex 180+ rules ตาม OWASP Top 10 |

**TypeScript จัดการอย่างไร:**
> "TypeScript จะถูกแปลงเป็น JavaScript ก่อน แล้วค่อยส่งเข้า Semgrep"

**ตัวอย่างที่จะตรวจจับได้ทันที (อธิบายได้เลย):**
```python
# SQL Injection — Bandit + Semgrep จับได้ทันที
query = "SELECT * FROM users WHERE id = " + user_input
cursor.execute(query)

# Command Injection
os.system("ping " + ip_address)

# Hardcoded Password
password = "admin123"
```

**ขั้นตอน Pattern Scanner:**
1. Parse โค้ดเป็น AST (Abstract Syntax Tree)
2. จับคู่ตาม Pattern Rules 180+ แบบ
3. แมพผลลัพธ์ออกมาเป็น → OWASP Category + CWE Number + บรรทัดที่พบ + ระดับความรุนแรง

---

## ส่วนที่ 4: Phase 2 — ML Model ทำงานอย่างไร (~3 นาที)

### จุดที่ต้องพูด

**สคริปต์:**
> "Phase สองคือ AI Model ของเรา ซึ่งเป็น Hybrid ระหว่าง GNN กับ BiLSTM"
> "ทำไมต้องใช้สองโมเดลรวมกัน? เพราะโค้ดมีสองมิติ: **โครงสร้าง** กับ **ลำดับ**"

---

### Branch 1: GNN (Graph Neural Network) — วิเคราะห์โครงสร้าง

**คำอธิบาย:**
> "GNN มองโค้ดเป็น **กราฟ** ไม่ใช่ข้อความ"

**ขั้นตอน:**
1. **AST Parsing** — แปลงโค้ดเป็น Abstract Syntax Tree
   - แต่ละ node = คำสั่ง, ตัวแปร, ฟังก์ชัน
   - แต่ละ edge = ความสัมพันธ์ (เรียกใช้, ส่งค่า, ควบคุม)

2. **Graph Construction** — สร้าง Graph จาก AST
   - Node Features: ประเภทของ node (64 dimensions)
   - Edges: Control flow + Data dependencies

3. **Graph Attention Network (GAT)**
   - 3 Layers ของ GATConv
   - 4 Attention Heads
   - Hidden Dimension: 128
   - GAT จะ "ให้ความสนใจ" กับ node ที่เชื่อมต่อกันสำคัญกว่า

4. **Global Pooling** — รวม Graph ทั้งหมดเป็น Vector เดียว

**ทำไม GNN ถึงดีกับโค้ด:**
> "เช่น ถ้า user_input ถูกส่งผ่านหลาย function แล้วไปถึง database query — GNN จะเห็นเส้นทางการไหลของข้อมูลนี้ผ่าน graph เลย"

---

### Branch 2: BiLSTM — วิเคราะห์ลำดับ Token

**คำอธิบาย:**
> "BiLSTM มองโค้ดเหมือนประโยค — อ่านไปข้างหน้าและย้อนกลับมาด้วย"

**ขั้นตอน:**
1. **Tokenization** — แยกโค้ดเป็น tokens (คำสั่ง, ชื่อตัวแปร, วงเล็บ)
   - Vocabulary: 3,336 tokens
   - Max Length: 256 tokens

2. **Embedding Layer** — แปลงแต่ละ token เป็น Vector (256 dimensions)

3. **Bidirectional LSTM** — อ่านลำดับ token
   - **Forward**: อ่านโค้ดจากบนลงล่าง → เข้าใจบริบทซ้าย
   - **Backward**: อ่านโค้ดจากล่างขึ้นบน → เข้าใจบริบทขวา
   - 2 Layers, Hidden Dimension: 128

**ทำไม BiLSTM ถึงดี:**
> "BiLSTM จะจำรูปแบบที่เคยเห็นในช่องโหว่ เช่น รูปแบบ `execute(`, string concatenation ก่อนหน้า → น่าสงสัยว่าเป็น SQL Injection"

---

### การรวม: Attention Fusion

**สคริปต์:**
> "ผลลัพธ์จาก GNN และ BiLSTM ถูกรวมกันด้วย Attention Mechanism"

```
[GNN Vector] + [BiLSTM Vector]
         ↓
   Attention Fusion (4 Heads)
         ↓
   FC Layers: 128 → 64 → 32 → 1
         ↓
   Output: 0 = Safe, 1 = Vulnerable
   + Confidence Score (0.0 - 1.0)
```

**Attention ทำอะไร:**
> "Attention เลือกว่าจะเชื่อ GNN กี่เปอร์เซ็นต์ และเชื่อ LSTM กี่เปอร์เซ็นต์ สำหรับโค้ดแต่ละชิ้น"
> "ถ้าโค้ดซับซ้อนเชิงโครงสร้าง → เชื่อ GNN มากกว่า"
> "ถ้าโค้ดมี pattern ชัด → เชื่อ LSTM มากกว่า"

---

## ส่วนที่ 5: Phase 3 — Hybrid Combiner (~1 นาที)

### จุดที่ต้องพูด

**สคริปต์:**
> "เมื่อได้ผลจากทั้งสอง phase แล้ว Hybrid Combiner จะรวมผลด้วยน้ำหนัก"

**สูตรการรวมผล:**
```
น้ำหนัก Pattern Matching = 70%
น้ำหนัก ML Model         = 30%

ถ้า Pattern เจอช่องโหว่ → เพิ่ม Confidence Score
ถ้า ML เจอด้วย           → Confidence สูงขึ้นอีก
ถ้ามีแค่ ML อย่างเดียว   → Confidence ปานกลาง
```

**การลด False Positive:**
> "ระบบมีตัวกรอง Import Filter — ถ้าบรรทัดที่ตรวจพบเป็นแค่ `import os` ไม่ใช่การใช้งานจริง จะกรองออกอัตโนมัติ"

---

## ส่วนที่ 6: ผลลัพธ์ที่ผู้ใช้เห็น (~1 นาที)

### จุดที่ต้องพูด

**สิ่งที่แสดงในผลการสแกน:**
- ❌ **ประเภทช่องโหว่**: เช่น SQL Injection, XSS
- 📍 **บรรทัดที่พบ**: Line 42
- 🔴 **ระดับความรุนแรง**: Critical / High / Medium / Low
- 📋 **CWE Number**: เช่น CWE-89 (SQL Injection)
- 🛡️ **OWASP Category**: เช่น A03: Injection
- ✅ **วิธีแก้ไข**: คำแนะนำพร้อมตัวอย่างโค้ดที่ปลอดภัย

**Export options:**
- PDF Report
- JSON
- SARIF (สำหรับ CI/CD)

---

## ส่วนที่ 7: ประสิทธิภาพของระบบ (Numbers to Quote - ~2 นาที)

### จุดที่ต้องพูด

> "ทดสอบบน Test Set 307 samples และ 1,335 samples (large validation)"

**ตัวเลขจาก Model Test Set (307 samples):**

| ตัวชี้วัด | ค่า |
|---------|-----|
| F1 Score | **99.58%** |
| Accuracy | **99.37%** |
| Precision | **99.2%** |
| Recall | **99.9%** |
| Data Leakage | **0%** |

**ตัวเลขจาก Large Validation (1,335 samples):**

| | |
|---|---|
| True Negative (Safe ถูก) | 720 |
| False Positive (แจ้งผิด) | **1 ครั้ง** |
| False Negative (พลาด) | 121 |
| True Positive (จับได้) | 493 |
| **Precision** | **99.80%** (แจ้งผิดแค่ 1 ครั้งใน 721) |
| **Accuracy** | **90.86%** |

**OWASP Top 10 Coverage:**
- 8 ใน 10 ประเภท: **Precision 100%**
- A03 (Injection): Recall **97.6%**
- ครอบคลุมทั้ง 10 ประเภท

**Ablation Study (เปรียบเทียบ):**

| โมเดล | Accuracy | F1 |
|------|---------|-----|
| GNN เท่านั้น | 78.2% | 75.8% |
| LSTM เท่านั้น | 82.1% | 79.3% |
| **Hybrid (GNN+LSTM)** | **90.86%** | **88.4%** |

> "ตรงนี้สำคัญมาก — การรวม GNN กับ LSTM ทำให้ดีขึ้นอย่างมีนัยสำคัญ"

---

## ส่วนที่ 8: สิ่งที่ทำให้โปรเจคนี้พิเศษ (Unique Selling Points - ~1 นาที)

### จุดที่ต้องพูด

1. **Hybrid Approach** — Pattern + ML ไม่ใช่แค่อย่างใดอย่างหนึ่ง
   - Pattern: เร็ว, แน่นอน, รู้จักรูปแบบที่รู้อยู่แล้ว
   - ML: ตรวจจับ pattern ใหม่, เข้าใจบริบท

2. **Zero Data Leakage** — ใช้ Fingerprint-based Splitting
   - ตรวจสอบแล้วว่า Training/Test ไม่ซ้อนกัน
   - 96.5% unique fingerprints

3. **Line-Level Detection** — บอกบรรทัดที่มีปัญหาได้เลย ไม่ใช่แค่บอกว่าไฟล์นี้มีช่องโหว่

4. **Multi-Platform** — Web UI + VS Code Extension + CLI + REST API

5. **Actionable Remediation** — ไม่ใช่แค่บอกช่องโหว่ แต่มีโค้ดตัวอย่างที่แก้ถูกต้องแล้ว

---

## ส่วนที่ 9: คำถามที่น่าถาม Q&A Preparation

### คำถามที่คาดว่าจะถาม

---

**Q1: ทำไมต้องใช้ GNN? ใช้แค่ LSTM ไม่พอหรือ?**

> A: LSTM อ่านโค้ดเป็นลำดับ แต่ไม่เห็น "เส้นทางของข้อมูล" (Taint Flow)  
> ตัวอย่าง: ถ้า user_input → function_a() → function_b() → SQL query  
> LSTM อาจพลาดเพราะอ่านทีละบรรทัด แต่ GNN เห็น Data Flow Graph ทั้งหมด  
> และผลการทดสอบยืนยัน: GNN อย่างเดียว 78.2%, LSTM อย่างเดียว 82.1%, รวมกัน 90.86%

---

**Q2: ต่างจาก SonarQube / Bandit ยังไง?**

> A: เครื่องมือเหล่านั้นใช้ Rule-based เท่านั้น จะพลาดช่องโหว่ที่ไม่ตรงกับ rule ที่เขียนไว้  
> ระบบของเราเพิ่ม ML ที่เรียนรู้จาก dataset จริง ทำให้อาจตรวจจับ pattern ใหม่ ๆ ได้  
> นอกจากนี้เรา integrate ทั้งสองแบบเข้าด้วยกัน ทำให้ทั้งเร็วและครอบคลุมกว่า

---

**Q3: False Positive ต่ำแค่ไหน?**

> A: จาก Large Validation 1,335 samples — มี False Positive แค่ 1 ครั้ง  
> คิดเป็น Precision 99.80% — แปลว่าถ้ามันบอกว่ามีช่องโหว่ โอกาส 99.8% ที่มันถูก

---

**Q4: Dataset มาจากไหน? จะ Overfit ไหม?**

> A: Dataset มาจาก 4 แหล่ง: Big-Vul (CVE จริง), SARD, GitHub Advisory, และ Generated Patterns  
> เราใช้ Fingerprint-based Splitting เพื่อป้องกัน Data Leakage  
> ตรวจสอบแล้วว่า Training และ Test Set ไม่มีตัวอย่างเดียวกันเลย (0% overlap)  
> Unique Fingerprints 96.5% หมายความว่าโค้ดแต่ละชิ้นแตกต่างกันจริง ๆ

---

**Q5: ทำงานได้กี่ภาษา?**

> A: 3 ภาษาหลัก — Python, JavaScript, TypeScript  
> TypeScript จะถูกแปลงเป็น JavaScript ก่อนสแกน  
> อนาคตสามารถขยายได้เพราะ Semgrep รองรับหลายภาษา

---

**Q6: ระบบใช้เวลาสแกนนานแค่ไหน?**

> A: Pattern Phase: < 1 วินาที  
> ML Phase: < 3 วินาที  
> Hybrid รวม: < 4 วินาที/ไฟล์  
> ทดสอบด้วย Locust: รองรับ 50 concurrent users โดย response time < 500ms

---

**Q7: Attention Fusion คืออะไร? ทำงานอย่างไร?**

> A: Attention Fusion เป็น layer ที่เรียนรู้ว่าจะเชื่อผลจาก GNN และ LSTM ในสัดส่วนเท่าไหร่  
> มี 4 Attention Heads แต่ละ Head มองจากมุมต่างกัน  
> เหมือนนักวิเคราะห์ 4 คน แต่ละคนเน้นมุมมองต่างกัน แล้วรวมความเห็น  
> Output คือ weighted combination ที่เหมาะสมกับโค้ดแต่ละชิ้น

---

**Q8: ทำไมใช้ AdamW? ทำไมไม่ใช้ Adam ธรรมดา?**

> A: AdamW เพิ่ม Weight Decay (L2 Regularization) เพื่อลด Overfitting  
> ใน code ML มักมีโครงสร้างซ้ำกันมาก AdamW ช่วยให้โมเดล generalize ได้ดีขึ้น  
> Weight Decay ที่ใช้: 0.001

---

## ส่วนที่ 10: สรุป (Closing - ~30 วินาที)

**สคริปต์:**
> "สรุปคือ ระบบของเราตรวจสอบช่องโหว่ความปลอดภัยด้วยสองกลไก:  
> Pattern Matching ที่เร็วและแน่นอน กับ AI Model ที่เข้าใจโครงสร้างและลำดับของโค้ด  
> รวมกันเป็น Hybrid System ที่ได้ Precision 99.80% บน large validation set  
> ครอบคลุม OWASP Top 10 ทั้งหมด และมี Line-Level Detection  
> มีทั้ง Web UI, VS Code Extension, CLI, และ REST API สำหรับ CI/CD  
> ขอบคุณครับ มีคำถามไหมครับ?"

---

## ข้อมูลสำคัญสรุปท้าย (Cheat Sheet)

| หัวข้อ | ค่า/รายละเอียด |
|-------|--------------|
| **โมเดล** | GNN (GAT 3 layers) + BiLSTM (2 layers) + Attention Fusion |
| **Parameters** | 1,905,409 (~1.9M) |
| **Vocabulary** | 3,336 tokens |
| **F1 Score** | 99.58% (test set 307 samples) |
| **Accuracy** | 99.37% (test set) / 90.86% (large validation) |
| **Precision** | 99.80% (large validation, FP=1 ใน 721) |
| **Dataset** | 3,117 samples, 4 แหล่ง, 0% Data Leakage |
| **ภาษาที่รองรับ** | Python, JavaScript, TypeScript |
| **Rules** | 180+ OWASP patterns (Bandit + Semgrep + Custom) |
| **OWASP Coverage** | ครบ 10 ประเภท, 8/10 Precision 100% |
| **Speed** | < 4 วินาที/ไฟล์ |
| **Platforms** | Web UI, VS Code Extension, CLI, REST API |
| **Stack** | Next.js 14 + FastAPI + PyTorch + PyTorch Geometric |
