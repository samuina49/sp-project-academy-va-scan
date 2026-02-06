/**
 * ตัวอย่างโค้ด JavaScript/TypeScript ที่มีช่องโหว่แต่ละประเภท
 * ใช้สำหรับทดสอบว่า Scanner เจอช่องโหว่ได้ครบถ้วนและตรงบรรทัดไหม
 */

// =============================================================================
// 1. CODE INJECTION (CRITICAL) - บรรทัด 10-11
// =============================================================================
const userInput = req.query.code;
eval(userInput);  // ⚠️ CRITICAL: Code Injection via eval()
new Function(userInput)();  // ⚠️ CRITICAL: Code Injection via Function constructor


// =============================================================================
// 2. COMMAND INJECTION (HIGH) - บรรทัด 16-18
// =============================================================================
const { exec } = require('child_process');
const cmd = req.query.command;
exec(cmd);  // ⚠️ HIGH: Command Injection


// =============================================================================
// 3. SQL INJECTION (HIGH) - บรรทัด 23-26
// =============================================================================
const username = req.body.username;
const query = "SELECT * FROM users WHERE username = '" + username + "'";  // ⚠️ HIGH: SQL Injection
db.query(`SELECT * FROM users WHERE id = ${userId}`);  // ⚠️ HIGH: SQL Injection (Template Literal)


// =============================================================================
// 4. XSS (HIGH) - บรรทัด 31-33
// =============================================================================
document.getElementById('output').innerHTML = userInput;  // ⚠️ HIGH: XSS via innerHTML
element.outerHTML = dangerousContent;  // ⚠️ HIGH: XSS via outerHTML


// =============================================================================
// 5. PATH TRAVERSAL (HIGH) - บรรทัด 38-39
// =============================================================================
const fs = require('fs');
fs.readFile(req.query.file, 'utf8');  // ⚠️ HIGH: Path Traversal


// =============================================================================
// 6. HARDCODED CREDENTIALS (MEDIUM) - บรรทัด 44-46
// =============================================================================
const password = "admin123";  // ⚠️ MEDIUM: Hardcoded Password
const apiKey = "sk_live_1234567890";  // ⚠️ MEDIUM: Hardcoded API Key
const dbPassword = "P@ssw0rd!";  // ⚠️ MEDIUM: Hardcoded Password


// =============================================================================
// 7. INSECURE RANDOM (MEDIUM) - บรรทัด 51-52
// =============================================================================
const token = Math.random().toString(36);  // ⚠️ MEDIUM: Weak Random for Security
const sessionId = Math.random();  // ⚠️ MEDIUM: Insecure Random


// =============================================================================
// 8. PROTOTYPE POLLUTION (HIGH) - บรรทัด 57-58
// =============================================================================
Object.assign({}, JSON.parse(userInput));  // ⚠️ HIGH: Potential Prototype Pollution


// =============================================================================
// 9. REGEX DOS (MEDIUM) - บรรทัด 63-64
// =============================================================================
const regex = new RegExp('^(a+)+$');  // ⚠️ MEDIUM: ReDoS vulnerability


// =============================================================================
// 10. INSECURE COMPARISON (MEDIUM) - บรรทัด 69-70
// =============================================================================
if (userPassword == storedPassword) {  // ⚠️ MEDIUM: Use === for comparison
    console.log("Login success");
}


// =============================================================================
// สรุปช่องโหว่ที่ควรเจอ (ขึ้นกับ Semgrep rules):
// =============================================================================
// ✓ Code Injection (eval, Function) - CRITICAL - บรรทัด 10, 11
// ✓ Command Injection (exec) - HIGH - บรรทัด 18
// ✓ SQL Injection - HIGH - บรรทัด 25, 26
// ✓ XSS (innerHTML, outerHTML) - HIGH - บรรทัด 32, 33
// ✓ Path Traversal - HIGH - บรรทัด 39
// ✓ Hardcoded Credentials - MEDIUM - บรรทัด 44, 45, 46
// ✓ Weak Random - MEDIUM - บรรทัด 51, 52
// ✓ Prototype Pollution - HIGH - บรรทัด 58
// ✓ ReDoS - MEDIUM - บรรทัด 64
// ✓ Insecure Comparison - MEDIUM - บรรทัด 69
//
// รวมทั้งหมด: ประมาณ 13-15 findings (ขึ้นกับ Semgrep rules ที่มี)
// =============================================================================
