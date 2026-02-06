/**
 * TypeScript Vulnerability Test Cases
 * ตัวอย่างโค้ด TypeScript ที่มีช่องโหว่ - สำหรับทดสอบ Scanner
 */

// =============================================================================
// 1. CODE INJECTION (CRITICAL) - บรรทัด 10-12
// =============================================================================
const userCode: string = getUserInput();
eval(userCode);  // ⚠️ CRITICAL: Code Injection via eval()
new Function(userCode)();  // ⚠️ CRITICAL: Code Injection via Function constructor


// =============================================================================
// 2. XSS VULNERABILITIES (HIGH) - บรรทัด 17-21
// =============================================================================
const element = document.getElementById('output');
element!.innerHTML = userInput;  // ⚠️ HIGH: XSS via innerHTML
element!.outerHTML = dangerousData;  // ⚠️ HIGH: XSS via outerHTML
document.write(userContent);  // ⚠️ HIGH: XSS via document.write


// =============================================================================
// 3. COMMAND INJECTION (HIGH) - บรรทัด 26-28
// =============================================================================
import { exec, spawn } from 'child_process';
exec(userCommand);  // ⚠️ HIGH: Command Injection
spawn('sh', ['-c', userInput]);  // ⚠️ HIGH: Command Injection


// =============================================================================
// 4. SQL INJECTION (HIGH) - บรรทัด 33-35
// =============================================================================
const query: string = "SELECT * FROM users WHERE id = " + userId;  // ⚠️ HIGH: SQL Injection
const sql = `DELETE FROM users WHERE name = ${userName}`;  // ⚠️ HIGH: SQL Injection via template


// =============================================================================
// 5. HARDCODED CREDENTIALS (MEDIUM) - บรรทัด 40-43
// =============================================================================
const password: string = "admin123";  // ⚠️ MEDIUM: Hardcoded Password
const apiKey: string = "sk_live_1234567890abcdef";  // ⚠️ MEDIUM: Hardcoded API Key
const dbPassword = "P@ssw0rd!";  // ⚠️ MEDIUM: Hardcoded Password


// =============================================================================
// 6. WEAK RANDOM (MEDIUM) - บรรทัด 48-50
// =============================================================================
const token: string = Math.random().toString(36);  // ⚠️ MEDIUM: Weak Random
const sessionId = Math.random() * 1000000;  // ⚠️ MEDIUM: Insecure Random


// =============================================================================
// 7. TYPE COERCION (LOW) - บรรทัด 55-58
// =============================================================================
if (userPassword == storedPassword) {  // ⚠️ LOW: Use === instead of ==
    console.log("Access granted");
}


// =============================================================================
// 8. PROTOTYPE POLLUTION (HIGH) - TypeScript specific
// =============================================================================
interface User {
    name: string;
    role: string;
}

const user: User = JSON.parse(untrustedInput);  // ⚠️ Potential prototype pollution
Object.assign({}, JSON.parse(userInput));  // ⚠️ Dangerous with untrusted input


// =============================================================================
// 9. TYPE ASSERTIONS (MEDIUM) - TypeScript specific
// =============================================================================
const data = userInput as any;  // ⚠️ MEDIUM: Type assertion bypassing type safety
const unsafeData = <any>externalData;  // ⚠️ MEDIUM: Unsafe type casting


// =============================================================================
// 10. UNSAFE PROPERTY ACCESS
// =============================================================================
const propName = userInput;
const value = obj[propName];  // ⚠️ Potential property injection


// =============================================================================
// สรุปช่องโหว่ที่ควรเจอ:
// =============================================================================
// ✓ Code Injection (eval, Function) - CRITICAL - บรรทัด 10, 11
// ✓ XSS (innerHTML, outerHTML, document.write) - HIGH - บรรทัด 18, 19, 20
// ✓ Command Injection (exec, spawn) - HIGH - บรรทัด 27, 28
// ✓ SQL Injection - HIGH - บรรทัด 33, 34
// ✓ Hardcoded Credentials - MEDIUM - บรรทัด 40, 41, 42
// ✓ Weak Random - MEDIUM - บรรทัด 48, 49
// ✓ Type Coercion (==) - LOW - บรรทัด 55
//
// รวมทั้งหมด: ประมาณ 13-15 findings
// (ขึ้นกับว่า pattern scanner รองรับ pattern ไหนบ้าง)
// =============================================================================
