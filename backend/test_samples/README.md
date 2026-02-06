# 🧪 Test Samples - ตัวอย่างโค้ดมีช่องโหว่

ไฟล์เหล่านี้เป็นตัวอย่างโค้ดที่มีช่องโหว่จงใจ เพื่อใช้ทดสอบว่า Scanner ตรวจจับได้ถูกต้อง

## 📁 ไฟล์ทดสอบ

### 1. [`python_vulnerabilities.py`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/test_samples/python_vulnerabilities.py)
**Python vulnerabilities** - ประมาณ 12-14 findings

ช่องโหว่ที่ควรเจอ:
- ✅ Code Injection (eval, exec) - CRITICAL
- ✅ Command Injection (os.system) - HIGH
- ✅ SQL Injection - HIGH
- ✅ Path Traversal - HIGH  
- ✅ Hardcoded Passwords - MEDIUM
- ✅ Weak Crypto (MD5, SHA1) - MEDIUM/LOW
- ✅ Unsafe Deserialization (pickle) - HIGH
- ✅ Weak Random - MEDIUM

### 2. [`javascript_vulnerabilities.js`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/test_samples/javascript_vulnerabilities.js)
**JavaScript/TypeScript vulnerabilities** - ประมาณ 13-15 findings

ช่องโหว่ที่ควรเจอ (ขึ้นกับ Semgrep rules):
- ✅ Code Injection (eval, Function) - CRITICAL
- ✅ Command Injection (exec) - HIGH
- ✅ SQL Injection - HIGH
- ✅ XSS (innerHTML) - HIGH
- ✅ Path Traversal - HIGH
- ✅ Hardcoded Credentials - MEDIUM
- ✅ Weak Random - MEDIUM
- ✅ Prototype Pollution - HIGH
- ✅ ReDoS - MEDIUM

## 🎯 วิธีทดสอบ

### ทดสอบผ่าน Frontend (แนะนำ)

1. เปิด http://localhost:3000
2. Copy โค้ดจากไฟล์ตัวอย่าง
3. Paste ลงใน Code Editor
4. เลือกภาษาให้ถูกต้อง (Python/JavaScript/TypeScript)
5. กด **"Scan Code"**
6. ดูผลลัพธ์ว่าเจอช่องโหว่ตรงบรรทัดที่ระบุไว้หรือไม่

### ทดสอบผ่าน API โดยตรง

```bash
# Python
curl -X POST http://localhost:8000/api/v1/scan/code \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "code": "eval(input())\nos.system('rm -rf /')",
  "language": "python"
}
EOF

# JavaScript
curl -X POST http://localhost:8000/api/v1/scan/code \
  -H "Content-Type: application/json" \
  -d @- << 'EOF'
{
  "code": "eval(userInput)\ndocument.innerHTML = data",
  "language": "javascript"
}
EOF
```

## ✅ การตรวจสอบผลลัพธ์

**ควรเช็ค:**
1. ✅ **จำนวน findings** - ตรงกับที่คาดหวังไหม?
2. ✅ **Severity levels** - ถูกต้องไหม? (CRITICAL, HIGH, MEDIUM, LOW)
3. ✅ **Line numbers** - ตรงกับบรรทัดที่มีช่องโหว่จริงไหม?
4. ✅ **Vulnerability types** - ระบุประเภทช่องโหว่ถูกต้องไหม?
5. ✅ **CWE IDs** - มี CWE mapping ไหม?

## 📊 ผลลัพธ์ที่คาดหวัง

### Python
```json
{
  "scan_id": "...",
  "total_findings": 12-14,
  "file_result": {
    "findings": [
      {
        "rule_id": "code_injection_via_eval()",
        "severity": "CRITICAL",
        "start_line": 11,
        "cwe_id": "CWE-94"
      },
      // ... more findings
    ]
  }
}
```

### JavaScript (ขึ้นกับ Semgrep rules)
```json
{
  "scan_id": "...",
  "total_findings": 10-15,
  "file_result": {
    "findings": [
      {
        "tool": "semgrep",
        "rule_id": "javascript.lang.security.audit.eval-detected",
        "severity": "HIGH",
        "start_line": 10
      },
      // ... more findings
    ]
  }
}
```

## 🚨 หมายเหตุ

- **Python** ใช้ SimplePatternScanner (pattern matching) - ควรเจอทุกช่องโหว่
- **JavaScript/TypeScript** ใช้ Semgrep - จำนวน findings ขึ้นกับ rules ที่ติดตั้งไว้
- บางช่องโหว่อาจเจอซ้ำถ้ามีหลาย pattern ตรง
- ถ้าเจอน้อยกว่าที่คาดหวัง → ตรวจสอบว่า scanner ทำงานถูกต้องหรือไม่

## 💡 Tips

1. **ทดสอบทีละไฟล์** เพื่อดูว่าแต่ละภาษาทำงานไหม
2. **ดูที่ line numbers** ให้ละเอียดว่าตรงกับ comment หรือไม่
3. **ทดสอบ Export Excel** เพื่อดูรายงานแบบเต็ม
4. **เปรียบเทียบ severity** ว่าตรงกับความเสี่ยงจริงไหม
