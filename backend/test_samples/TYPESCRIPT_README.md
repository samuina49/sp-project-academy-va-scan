# 🔷 TypeScript Vulnerability Test Cases

ตัวอย่างโค้ด TypeScript ที่มีช่องโหว่ สำหรับทดสอบ Scanner

---

## 📁 ไฟล์ทดสอบ

### [`typescript_vulnerabilities.ts`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/test_samples/typescript_vulnerabilities.ts)

**TypeScript vulnerabilities** - ประมาณ 13-15 findings

---

## 🎯 ช่องโหว่ที่ควรเจอ

### 1. **Code Injection** - CRITICAL
- ✅ `eval(userCode)` - บรรทัด 10
- ✅ `new Function(userCode)()` - บรรทัด 11

### 2. **XSS** - HIGH
- ✅ `innerHTML = userInput` - บรรทัด 18
- ✅ `outerHTML = dangerousData` - บรรทัด 19
- ✅ `document.write(userContent)` - บรรทัด 20

### 3. **Command Injection** - HIGH
- ✅ `exec(userCommand)` - บรรทัด 27
- ✅ `spawn('sh', ['-c', userInput])` - บรรทัด 28

### 4. **SQL Injection** - HIGH
- ✅ String concatenation - บรรทัด 33
- ✅ Template literals - บรรทัด 34

### 5. **Hardcoded Credentials** - MEDIUM
- ✅ `password = "admin123"` - บรรทัด 40
- ✅ `apiKey = "sk_live_..."` - บรรทัด 41
- ✅ `dbPassword = "P@ssw0rd!"` - บรรทัด 42

### 6. **Weak Random** - MEDIUM
- ✅ `Math.random()` - บรรทัด 48, 49

### 7. **Type Coercion** - LOW
- ✅ `==` instead of `===` - บรรทัด 55

---

## 🧪 วิธีทดสอบ

### ทาง Frontend

1. เปิด http://localhost:3000
2. Copy โค้ดจาก [`typescript_vulnerabilities.ts`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/test_samples/typescript_vulnerabilities.ts)
3. Paste ลงใน Code Editor
4. **เลือก "TypeScript"**
5. กด "Scan Code"
6. ดูผลลัพธ์

### ทาง API

```bash
curl -X POST http://localhost:8000/api/v1/scan/code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "eval(userInput)\ndocument.innerHTML = data\nMath.random()",
    "language": "typescript"
  }'
```

### PowerShell Test

```powershell
$code = @"
eval(userInput)
document.getElementById('output').innerHTML = data
Math.random()
const password = 'admin123'
"@

$payload = @{code=$code; language="typescript"} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/v1/scan/code" `
  -Method POST `
  -Body $payload `
  -ContentType "application/json"
```

---

## ✅ ผลลัพธ์ที่คาดหวัง

```json
{
  "scan_id": "...",
  "total_findings": 13-15,
  "file_result": {
    "language": "typescript",
    "findings": [
      {
        "rule_id": "code_injection_via_eval()",
        "severity": "CRITICAL",
        "start_line": 10,
        "message": "eval() allows arbitrary code execution..."
      },
      {
        "rule_id": "xss_via_innerhtml",
        "severity": "HIGH",
        "start_line": 18,
        "message": "innerHTML can execute script tags..."
      },
      // ... more findings
    ]
  }
}
```

---

## 📊 Patterns ที่รองรับ

TypeScript ใช้ patterns เดียวกับ JavaScript:

| Pattern | Severity | CWE |
|---------|----------|-----|
| `eval()` | CRITICAL | CWE-94 |
| `new Function()` | CRITICAL | CWE-94 |
| `.innerHTML =` | HIGH | CWE-79 |
| `.outerHTML =` | HIGH | CWE-79 |
| `document.write()` | HIGH | CWE-79 |
| `exec()` / `spawn()` | HIGH | CWE-78 |
| SQL concatenation | HIGH | CWE-89 |
| `password = "..."` | MEDIUM | CWE-798 |
| `Math.random()` | MEDIUM | CWE-338 |
| `==` operator | LOW | CWE-1023 |

---

## 💡 Tips

1. **TypeScript-specific issues** (type assertions, `any`) - ต้อง advanced scanner
2. **ตอนนี้ใช้ SimplePatternScanner** - เจอ runtime vulnerabilities
3. **Future:** เพิ่ม TypeScript-specific patterns (type safety bypasses)

---

## 🔗 ไฟล์ที่เกี่ยวข้อง

- Test file: [`typescript_vulnerabilities.ts`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/test_samples/typescript_vulnerabilities.ts)
- Pattern scanner: [`simple_scanner.py`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/app/scanners/simple_scanner.py)
- JavaScript tests: [`javascript_vulnerabilities.js`](file:///c:/Users/samui/OneDrive/Desktop/Project%20University%20Final%20and%20Last/AI-BASED%20VULNERABILITY%20SCANNER%20FOR%20WEB%20APPLICATIONS/backend/test_samples/javascript_vulnerabilities.js)
