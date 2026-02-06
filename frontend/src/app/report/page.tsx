'use client';

import { useEffect, useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import type { FileScanResult, VulnerabilityFinding } from '@/types/api';
import { useTheme } from '@/components/ThemeProvider';
import { useI18n } from '@/lib/i18n';

// Dynamic import Monaco Editor for SSR compatibility
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

interface DisplayVuln {
  id: number;
  severity: string;
  line_number: number;
  vulnerability_type: string;
  cwe_id?: string;
  owasp_category?: string;
  description: string;
  code_snippet?: string;
  confidence: number;
  recommendation: string;
  secure_example?: string;
  vulnerable_example?: string;
  feedback?: 'confirmed' | 'false_positive' | null;
  file_path?: string;
}

// OWASP Category mapping
const OWASP_CATEGORIES: Record<string, string> = {
  'CWE-78': 'A03:2021 - Injection',
  'CWE-79': 'A03:2021 - Injection',
  'CWE-89': 'A03:2021 - Injection',
  'CWE-94': 'A03:2021 - Injection',
  'CWE-943': 'A03:2021 - Injection',
  'CWE-1336': 'A03:2021 - Injection',
  'CWE-117': 'A03:2021 - Injection',
  'CWE-22': 'A01:2021 - Broken Access Control',
  'CWE-942': 'A01:2021 - Broken Access Control',
  'CWE-327': 'A02:2021 - Cryptographic Failures',
  'CWE-326': 'A02:2021 - Cryptographic Failures',
  'CWE-295': 'A02:2021 - Cryptographic Failures',
  'CWE-489': 'A05:2021 - Security Misconfiguration',
  'CWE-798': 'A07:2021 - Auth Failures',
  'CWE-347': 'A07:2021 - Auth Failures',
  'CWE-312': 'A07:2021 - Auth Failures',
  'CWE-502': 'A08:2021 - Data Integrity Failures',
  'CWE-918': 'A10:2021 - SSRF',
  'CWE-532': 'A09:2021 - Logging Failures',
  'CWE-390': 'A09:2021 - Logging Failures',
  'CWE-754': 'A09:2021 - Logging Failures',
  'CWE-362': 'A04:2021 - Insecure Design',
  'CWE-1078': 'A04:2021 - Insecure Design',
  'CWE-862': 'A04:2021 - Insecure Design',
  'CWE-1035': 'A06:2021 - Vulnerable Components',
};

// Remediation database with secure examples - bilingual
const REMEDIATION_DB: Record<string, { 
  recommendation: { en: string; th: string }; 
  vulnerable: string; 
  secure: string 
}> = {
  'CWE-78': {
    recommendation: {
      en: 'Use subprocess.run() with a list of arguments instead of os.system(). Never pass user input directly to shell commands.',
      th: 'ใช้ subprocess.run() พร้อมรายการอาร์กิวเมนต์แทน os.system() อย่าส่ง input จากผู้ใช้ไปยังคำสั่ง shell โดยตรง'
    },
    vulnerable: 'os.system("ls " + user_input)',
    secure: 'subprocess.run(["ls", user_input], shell=False)',
  },
  'CWE-89': {
    recommendation: {
      en: 'Use parameterized queries or prepared statements. Never concatenate user input directly into SQL queries.',
      th: 'ใช้ parameterized queries หรือ prepared statements อย่านำ input จากผู้ใช้มาต่อกับ SQL query โดยตรง'
    },
    vulnerable: 'cursor.execute("SELECT * FROM users WHERE id=" + user_id)',
    secure: 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
  },
  'CWE-79': {
    recommendation: {
      en: 'Sanitize all user input before rendering. Use textContent instead of innerHTML, or use a sanitization library like DOMPurify.',
      th: 'กรอง input จากผู้ใช้ทั้งหมดก่อนแสดงผล ใช้ textContent แทน innerHTML หรือใช้ไลบรารีกรองข้อมูลเช่น DOMPurify'
    },
    vulnerable: 'element.innerHTML = userInput;',
    secure: 'element.textContent = userInput;\n// or: element.innerHTML = DOMPurify.sanitize(userInput);',
  },
  'CWE-94': {
    recommendation: {
      en: 'Never use eval() or exec() with user input. Use safer alternatives like ast.literal_eval() for data parsing.',
      th: 'อย่าใช้ eval() หรือ exec() กับ input จากผู้ใช้ ใช้ทางเลือกที่ปลอดภัยกว่าเช่น ast.literal_eval() สำหรับแยกวิเคราะห์ข้อมูล'
    },
    vulnerable: 'eval(user_input)',
    secure: 'import ast\nresult = ast.literal_eval(user_input)  # Only for literals',
  },
  'CWE-502': {
    recommendation: {
      en: 'Never deserialize untrusted data with pickle. Use yaml.safe_load() instead of yaml.load(). Validate and sanitize input.',
      th: 'อย่า deserialize ข้อมูลที่ไม่น่าเชื่อถือด้วย pickle ใช้ yaml.safe_load() แทน yaml.load() ตรวจสอบและกรอง input'
    },
    vulnerable: 'data = pickle.loads(user_data)\n# or: config = yaml.load(file)',
    secure: 'import json\ndata = json.loads(user_data)  # Safer alternative\n# or: config = yaml.safe_load(file)',
  },
  'CWE-918': {
    recommendation: {
      en: 'Validate and whitelist URLs before making requests. Never allow user-controlled URLs without validation.',
      th: 'ตรวจสอบและกำหนด whitelist ของ URL ก่อนส่งคำขอ อย่าอนุญาตให้ผู้ใช้ควบคุม URL โดยไม่ตรวจสอบ'
    },
    vulnerable: 'requests.get(user_url)',
    secure: 'ALLOWED_HOSTS = ["api.example.com"]\nif urlparse(user_url).netloc in ALLOWED_HOSTS:\n    requests.get(user_url)',
  },
  'CWE-327': {
    recommendation: {
      en: 'Use strong cryptographic algorithms. Replace MD5/SHA1 with SHA-256 or stronger. Use bcrypt/argon2 for passwords.',
      th: 'ใช้อัลกอริทึมการเข้ารหัสที่แข็งแกร่ง เปลี่ยน MD5/SHA1 เป็น SHA-256 หรือสูงกว่า ใช้ bcrypt/argon2 สำหรับรหัสผ่าน'
    },
    vulnerable: 'hashlib.md5(password.encode()).hexdigest()',
    secure: 'import bcrypt\nhashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())',
  },
  'CWE-798': {
    recommendation: {
      en: 'Never hardcode credentials in source code. Use environment variables or a secrets manager.',
      th: 'อย่าเขียนข้อมูลรับรองตัวตนลงในซอร์สโค้ดโดยตรง ใช้ environment variables หรือ secrets manager'
    },
    vulnerable: 'password = "mysecretpassword123"',
    secure: 'import os\npassword = os.environ.get("DB_PASSWORD")',
  },
  'CWE-489': {
    recommendation: {
      en: 'Disable debug mode in production. Use environment-based configuration.',
      th: 'ปิด debug mode ใน production ใช้การตั้งค่าตาม environment'
    },
    vulnerable: 'app.run(debug=True)',
    secure: 'app.run(debug=os.environ.get("DEBUG", "false").lower() == "true")',
  },
  'CWE-347': {
    recommendation: {
      en: 'Always verify JWT signatures. Never use the "none" algorithm. Set explicit algorithms.',
      th: 'ตรวจสอบลายเซ็น JWT เสมอ อย่าใช้ algorithm "none" กำหนด algorithms อย่างชัดเจน'
    },
    vulnerable: 'jwt.decode(token, verify=False)',
    secure: 'jwt.decode(token, SECRET_KEY, algorithms=["HS256"])',
  },
  'CWE-532': {
    recommendation: {
      en: 'Never log sensitive data like passwords, tokens, or API keys. Use structured logging with data masking.',
      th: 'อย่าบันทึก log ข้อมูลที่ละเอียดอ่อนเช่น รหัสผ่าน, token, หรือ API keys ใช้ structured logging พร้อม data masking'
    },
    vulnerable: 'logger.info(f"User password: {password}")',
    secure: 'logger.info(f"User {username} authenticated successfully")',
  },
  'CWE-942': {
    recommendation: {
      en: 'Restrict CORS to specific trusted origins. Never use wildcard (*) in production.',
      th: 'จำกัด CORS ให้เฉพาะ origin ที่เชื่อถือได้ อย่าใช้ wildcard (*) ใน production'
    },
    vulnerable: 'Access-Control-Allow-Origin: *',
    secure: 'Access-Control-Allow-Origin: https://yourdomain.com',
  },
  'CWE-200': {
    recommendation: {
      en: 'Never log sensitive information like tokens, passwords, or user data. Remove console.log statements in production or use a proper logging library with data masking.',
      th: 'อย่า log ข้อมูลที่ละเอียดอ่อน เช่น token, password, หรือข้อมูลผู้ใช้ ลบ console.log ออกใน production หรือใช้ logging library ที่มีการปกปิดข้อมูล'
    },
    vulnerable: 'console.log("Token:", token);\nconsole.log("User data:", userData);',
    secure: '// Use environment check\nif (process.env.NODE_ENV === "development") {\n  console.log("Debug info");\n}\n// Or use proper logger\nlogger.debug("User authenticated", { userId: user.id });',
  },
  'CWE-1321': {
    recommendation: {
      en: 'Avoid using JSON.parse() on untrusted input without validation. Use schema validation libraries like Zod or Joi.',
      th: 'หลีกเลี่ยงการใช้ JSON.parse() กับ input ที่ไม่น่าเชื่อถือโดยไม่ตรวจสอบ ใช้ library ตรวจสอบ schema เช่น Zod หรือ Joi'
    },
    vulnerable: 'const data = JSON.parse(userInput);',
    secure: 'import { z } from "zod";\nconst schema = z.object({ name: z.string() });\nconst data = schema.parse(JSON.parse(userInput));',
  },
  'CWE-22': {
    recommendation: {
      en: 'Validate and sanitize file paths. Use path.resolve() and ensure the resolved path is within allowed directories.',
      th: 'ตรวจสอบและกรอง path ของไฟล์ ใช้ path.resolve() และตรวจสอบว่า path อยู่ในโฟลเดอร์ที่อนุญาต'
    },
    vulnerable: 'fs.readFile("/data/" + userInput)',
    secure: 'const safePath = path.resolve("/data", userInput);\nif (!safePath.startsWith("/data/")) throw new Error("Invalid path");\nfs.readFile(safePath);',
  },
  'CWE-295': {
    recommendation: {
      en: 'Never disable SSL/TLS certificate verification. Use proper certificate management.',
      th: 'อย่าปิดการตรวจสอบ SSL/TLS certificate ใช้การจัดการ certificate ที่ถูกต้อง'
    },
    vulnerable: 'requests.get(url, verify=False)\n// or: rejectUnauthorized: false',
    secure: 'requests.get(url, verify=True)\n// or: Use proper CA certificates',
  },
  'CWE-312': {
    recommendation: {
      en: 'Never store sensitive data in localStorage/sessionStorage. Use secure HTTP-only cookies or encrypted storage.',
      th: 'อย่าเก็บข้อมูลสำคัญใน localStorage/sessionStorage ใช้ HTTP-only cookies หรือ encrypted storage'
    },
    vulnerable: 'localStorage.setItem("token", authToken);',
    secure: '// Use HTTP-only cookies set by server\n// Or encrypt before storing\nconst encrypted = encrypt(data, key);',
  },
  'CWE-1035': {
    recommendation: {
      en: 'Regularly audit dependencies with npm audit or pip-audit. Keep packages updated and remove unused ones.',
      th: 'ตรวจสอบ dependencies ด้วย npm audit หรือ pip-audit เป็นประจำ อัพเดท packages และลบที่ไม่ใช้ออก'
    },
    vulnerable: '// Outdated package with known vulnerabilities',
    secure: 'npm audit fix\n# or\npip-audit --fix',
  },
};

const SEVERITY_CONFIG = {
  critical: { bg: 'bg-red-50 dark:bg-red-950/50', border: 'border-red-500', text: 'text-red-800 dark:text-red-300', badge: 'bg-red-100 dark:bg-red-900/50', icon: '🔴', color: '#ef4444' },
  high: { bg: 'bg-orange-50 dark:bg-orange-950/50', border: 'border-orange-500', text: 'text-orange-800 dark:text-orange-300', badge: 'bg-orange-100 dark:bg-orange-900/50', icon: '🟠', color: '#f97316' },
  medium: { bg: 'bg-yellow-50 dark:bg-yellow-950/50', border: 'border-yellow-500', text: 'text-yellow-800 dark:text-yellow-300', badge: 'bg-yellow-100 dark:bg-yellow-900/50', icon: '🟡', color: '#eab308' },
  low: { bg: 'bg-blue-50 dark:bg-blue-950/50', border: 'border-blue-500', text: 'text-blue-800 dark:text-blue-300', badge: 'bg-blue-100 dark:bg-blue-900/50', icon: '🔵', color: '#3b82f6' },
  info: { bg: 'bg-gray-50 dark:bg-gray-800/50', border: 'border-gray-400 dark:border-gray-600', text: 'text-gray-700 dark:text-gray-300', badge: 'bg-gray-100 dark:bg-gray-800', icon: 'ℹ️', color: '#6b7280' },
};

export default function ReportPage() {
  const router = useRouter();
  const { theme } = useTheme();
  const { language: uiLanguage } = useI18n();
  const [results, setResults] = useState<FileScanResult[]>([]);
  const [scannedCode, setScannedCode] = useState<string>('');
  const [language, setLanguage] = useState<string>('python');
  const [loading, setLoading] = useState(true);
  const [selectedVuln, setSelectedVuln] = useState<number | null>(null);
  const [filterSeverity, setFilterSeverity] = useState<string>('all');
  const [feedbackState, setFeedbackState] = useState<Record<number, 'confirmed' | 'false_positive' | null>>({});
  const [feedbackMessage, setFeedbackMessage] = useState<string | null>(null);
  const [selectedFileIndex, setSelectedFileIndex] = useState<number>(0);
  const [isZipScan, setIsZipScan] = useState<boolean>(false);
  const editorRef = useRef<any>(null);

  // Translation object
  const t = {
    report: {
      title: uiLanguage === 'th' ? 'รายงานผลการสแกน' : 'Security Report',
      loadingResults: uiLanguage === 'th' ? 'กำลังโหลดผลลัพธ์...' : 'Loading results...',
      scanCompleted: uiLanguage === 'th' ? 'สแกนเสร็จสิ้น' : 'Scan Completed',
      issuesFound: uiLanguage === 'th' ? 'ปัญหาที่พบ' : 'issues found',
      issueFound: uiLanguage === 'th' ? 'ปัญหาที่พบ' : 'issue found',
      exportJson: uiLanguage === 'th' ? 'ส่งออก JSON' : 'Export JSON',
      newScan: uiLanguage === 'th' ? 'สแกนใหม่' : 'New Scan',
      riskScore: uiLanguage === 'th' ? 'ระดับความเสี่ยง' : 'Risk Score',
      highRisk: uiLanguage === 'th' ? 'ความเสี่ยงสูง' : 'High Risk',
      mediumRisk: uiLanguage === 'th' ? 'ความเสี่ยงปานกลาง' : 'Medium Risk',
      lowRisk: uiLanguage === 'th' ? 'ความเสี่ยงต่ำ' : 'Low Risk',
      basedOnFindings: uiLanguage === 'th' ? 'จากผลการตรวจจับ' : 'Based on',
      findings: uiLanguage === 'th' ? 'ปัญหา' : 'findings',
      critical: uiLanguage === 'th' ? 'วิกฤต' : 'Critical',
      high: uiLanguage === 'th' ? 'สูง' : 'High',
      medium: uiLanguage === 'th' ? 'ปานกลาง' : 'Medium',
      lowInfo: uiLanguage === 'th' ? 'ต่ำ/ข้อมูล' : 'Low/Info',
      filterBy: uiLanguage === 'th' ? 'กรองตาม:' : 'Filter by:',
      clearFilter: uiLanguage === 'th' ? 'ล้างตัวกรอง' : 'Clear Filter',
      noVulnerabilities: uiLanguage === 'th' ? 'ไม่พบช่องโหว่' : 'No Vulnerabilities Found',
      codeSecure: uiLanguage === 'th' ? 'โค้ดของคุณดูปลอดภัย ไม่พบช่องโหว่ที่เป็นที่รู้จัก' : 'Your code appears to be secure. No known vulnerabilities were detected.',
      scanAnother: uiLanguage === 'th' ? 'สแกนโค้ดอื่น' : 'Scan Another Code',
      scannedCode: uiLanguage === 'th' ? 'โค้ดที่สแกน' : 'Scanned Code',
      lines: uiLanguage === 'th' ? 'บรรทัด' : 'lines',
      line: uiLanguage === 'th' ? 'บรรทัด' : 'Line',
      recommendation: uiLanguage === 'th' ? 'คำแนะนำ' : 'Recommendation',
      vulnerable: uiLanguage === 'th' ? 'โค้ดที่มีช่องโหว่' : 'Vulnerable Code',
      secure: uiLanguage === 'th' ? 'โค้ดที่ปลอดภัย' : 'Secure Code',
      wasAccurate: uiLanguage === 'th' ? 'การตรวจจับนี้ถูกต้องหรือไม่?' : 'Was this detection accurate?',
      confirmed: uiLanguage === 'th' ? 'ยืนยันแล้ว' : 'Confirmed',
      markedFalsePositive: uiLanguage === 'th' ? 'ทำเครื่องหมายเป็นผลบวกลวง' : 'Marked as False Positive',
      confirm: uiLanguage === 'th' ? 'ยืนยัน' : 'Confirm',
      falsePositive: uiLanguage === 'th' ? 'ผลบวกลวง' : 'False Positive',
    }
  };

  useEffect(() => {
    const savedResults = sessionStorage.getItem('scanResults');
    const savedCode = sessionStorage.getItem('scannedCode');
    const savedLanguage = sessionStorage.getItem('scannedLanguage');

    if (!savedResults) {
      router.push('/scan');
      return;
    }

    try {
      const parsed = JSON.parse(savedResults);
      if (savedCode) setScannedCode(savedCode);
      if (savedLanguage) setLanguage(savedLanguage);
      
      // Handle different API response formats
      if (parsed.findings && Array.isArray(parsed.findings)) {
        // HybridScanResponse format - convert findings to FileScanResult
        console.log('[REPORT] Hybrid scan results:', parsed.findings.length, 'findings');
        const converted: FileScanResult = {
          file_path: 'code_input',
          language: parsed.code_language || savedLanguage || 'unknown',
          findings: parsed.findings.map((f: any) => ({
            tool: f.sources?.join(',') || 'hybrid',
            rule_id: f.vulnerability_type || f.semgrep_rule || 'security-issue',
            severity: f.severity || 'MEDIUM',
            message: f.explanation || 'Potential vulnerability detected',
            start_line: f.line || 1,
            end_line: f.line || 1,
            code_snippet: f.code_snippet,
            cwe_id: f.cwe_id,
            owasp_category: f.owasp_category,
          })),
        };
        setResults([converted]);
      } else if (parsed.vulnerabilities) {
        const converted: FileScanResult = {
          file_path: 'code_input',
          language: parsed.language || savedLanguage || 'unknown',
          findings: parsed.vulnerabilities.map((v: any) => ({
            tool: 'ml',
            rule_id: v.cwe_id || 'security-issue',
            severity: v.severity || 'MEDIUM',
            message: v.message || 'Potential vulnerability detected',
            start_line: v.line || 1,
            end_line: v.line || 1,
            code_snippet: v.code_snippet,
            cwe_id: v.cwe_id,
          })),
        };
        setResults([converted]);
      } else if (parsed.file_results) {
        // ZIP scan response - file_results is an array
        console.log('[REPORT] ZIP scan results:', parsed.file_results);
        
        // Sort files by findings count (most findings first)
        const sortedResults = [...parsed.file_results].sort((a: FileScanResult, b: FileScanResult) => 
          (b.findings?.length || 0) - (a.findings?.length || 0)
        );
        
        setResults(sortedResults);
        setIsZipScan(true);
        
        // Find first file with findings and set as default
        const firstWithFindings = sortedResults.findIndex((f: FileScanResult) => (f.findings?.length || 0) > 0);
        if (firstWithFindings >= 0) {
          setSelectedFileIndex(firstWithFindings);
        }
        
        // Build combined code view from all file findings
        const combinedCode = sortedResults.map((file: FileScanResult) => {
          const fileName = file.file_path || 'unknown';
          const snippets = file.findings?.map((f: VulnerabilityFinding) => f.code_snippet).filter(Boolean) || [];
          return `// ========== ${fileName} ==========\n${snippets.join('\n') || '// No code snippets available'}`;
        }).join('\n\n');
        setScannedCode(combinedCode || savedCode || '// Multiple files scanned');
      } else if (parsed.file_result) {
        setResults([parsed.file_result]);
      } else if (parsed.results) {
        setResults(parsed.results);
      } else {
        setResults([]);
      }
    } catch (error) {
      console.error('Failed to parse results:', error);
      router.push('/scan');
    } finally {
      setLoading(false);
    }
  }, [router]);

  const handleNewScan = () => {
    sessionStorage.removeItem('scanResults');
    sessionStorage.removeItem('scannedCode');
    sessionStorage.removeItem('scannedLanguage');
    router.push('/scan');
  };

  // Map findings to display format with remediation
  const mapFinding = (f: VulnerabilityFinding, index: number, filePath?: string): DisplayVuln => {
    const cweId = f.cwe_id || 'CWE-UNKNOWN';
    const remediationData = REMEDIATION_DB[cweId];
    const defaultRecommendation = uiLanguage === 'th' 
      ? `ตรวจสอบและแก้ไขปัญหาความปลอดภัยที่บรรทัด ${f.start_line} ดูข้อมูลเพิ่มเติมได้ที่ฐานข้อมูล CWE`
      : `Review and fix the security issue at line ${f.start_line}. Consult the CWE database for more information.`;
    
    const remediation = remediationData ? {
      recommendation: remediationData.recommendation[uiLanguage],
      vulnerable: remediationData.vulnerable,
      secure: remediationData.secure,
    } : {
      recommendation: defaultRecommendation,
      vulnerable: f.code_snippet || '',
      secure: '// Apply appropriate security fix based on the vulnerability type',
    };

    return {
      id: index,
      severity: f.severity?.toLowerCase() || 'medium',
      line_number: f.start_line || 1,
      vulnerability_type: f.rule_id?.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) || 'Security Issue',
      cwe_id: cweId,
      owasp_category: OWASP_CATEGORIES[cweId] || 'Unknown Category',
      description: f.message || 'Potential security vulnerability detected',
      code_snippet: f.code_snippet,
      confidence: 0.85,
      recommendation: remediation.recommendation,
      vulnerable_example: remediation.vulnerable,
      secure_example: remediation.secure,
      file_path: filePath, // Track which file this finding belongs to
    };
  };

  // Map findings with file path context
  const allVulns: DisplayVuln[] = results.flatMap((r, fileIdx) => 
    (r.findings || []).map((f, findingIdx) => ({
      ...mapFinding(f, fileIdx * 1000 + findingIdx, r.file_path),
    }))
  );
  
  // Filter vulnerabilities by selected file (for ZIP scan) and severity
  const fileFilteredVulns = isZipScan && results.length > 1
    ? allVulns.filter(v => v.file_path === results[selectedFileIndex]?.file_path)
    : allVulns;
  
  const filteredVulns = filterSeverity === 'all' 
    ? fileFilteredVulns 
    : fileFilteredVulns.filter(v => v.severity === filterSeverity);

  const totalVulns = allVulns.length;
  const criticalCount = allVulns.filter(v => v.severity === 'critical').length;
  const highCount = allVulns.filter(v => v.severity === 'high').length;
  const mediumCount = allVulns.filter(v => v.severity === 'medium').length;
  const lowCount = allVulns.filter(v => v.severity === 'low').length;
  const infoCount = allVulns.filter(v => v.severity === 'info').length;

  // Calculate risk score (0-100) with balanced weights
  // Critical vulnerabilities have much higher impact
  // Weights: Critical=35, High=18, Medium=4, Low=1.5, Info=0.5
  const riskScore = Math.min(100, Math.round(
    (criticalCount * 35 + highCount * 18 + mediumCount * 4 + lowCount * 1.5 + infoCount * 0.5)
  ));

  // Navigate to line in editor
  const navigateToLine = (lineNumber: number, vulnId: number) => {
    setSelectedVuln(vulnId);
    if (editorRef.current) {
      editorRef.current.revealLineInCenter(lineNumber);
      editorRef.current.setPosition({ lineNumber, column: 1 });
    }
  };

  // Apply decorations to editor
  const applyDecorations = (editor: any, vulns: DisplayVuln[]) => {
    if (!editor) return;
    
    const decorations = vulns.map(vuln => {
      const severityConfig = SEVERITY_CONFIG[vuln.severity as keyof typeof SEVERITY_CONFIG] || SEVERITY_CONFIG.medium;
      return {
        range: { startLineNumber: vuln.line_number, startColumn: 1, endLineNumber: vuln.line_number, endColumn: 1 },
        options: {
          isWholeLine: true,
          className: `vuln-line-${vuln.severity}`,
          glyphMarginClassName: `vuln-glyph-${vuln.severity}`,
          overviewRuler: { color: severityConfig.color, position: 1 },
        },
      };
    });
    editor.deltaDecorations([], decorations);
  };

  // Re-apply decorations when file changes
  useEffect(() => {
    if (editorRef.current && isZipScan) {
      // Get findings for current file - use REAL line numbers from source
      const currentFileFindings = results[selectedFileIndex]?.findings || [];
      const mappedVulns = currentFileFindings.map((f, idx) => ({
        ...mapFinding(f, idx, results[selectedFileIndex]?.file_path),
        // Use the actual line number from the finding
        line_number: f.start_line || 1,
      }));
      applyDecorations(editorRef.current, mappedVulns);
    }
  }, [selectedFileIndex, isZipScan, results]);

  // Handle editor mount
  const handleEditorMount = (editor: any) => {
    editorRef.current = editor;
    
    // Add line decorations for vulnerabilities
    if (isZipScan) {
      const currentFileFindings = results[selectedFileIndex]?.findings || [];
      const mappedVulns = currentFileFindings.map((f, idx) => ({
        ...mapFinding(f, idx, results[selectedFileIndex]?.file_path),
        line_number: f.start_line || 1, // Use actual line number
      }));
      applyDecorations(editor, mappedVulns);
    } else {
      applyDecorations(editor, allVulns);
    }
  };

  // Export to JSON
  const exportJSON = () => {
    const exportData = {
      scan_date: new Date().toISOString(),
      total_vulnerabilities: totalVulns,
      risk_score: riskScore,
      summary: { critical: criticalCount, high: highCount, medium: mediumCount, low: lowCount, info: infoCount },
      vulnerabilities: allVulns.map(v => ({ ...v, feedback: feedbackState[v.id] || null })),
      scanned_code: scannedCode,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vulnerability-report-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  // Submit feedback for a vulnerability
  const submitFeedback = async (vulnId: number, feedbackType: 'confirmed' | 'false_positive') => {
    const vuln = allVulns.find(v => v.id === vulnId);
    if (!vuln) return;

    // Get the code snippet for this line
    const lines = scannedCode.split('\n');
    const lineIndex = vuln.line_number - 1;
    const codeContext = lines.slice(Math.max(0, lineIndex - 2), lineIndex + 3).join('\n');

    try {
      const response = await fetch('http://localhost:8000/api/v1/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          cwe_id: vuln.cwe_id,
          severity: vuln.severity,
          line_number: vuln.line_number,
          code_snippet: codeContext,
          description: vuln.description,
          feedback_type: feedbackType,
          language: language,
          is_vulnerable: feedbackType === 'confirmed',
        }),
      });

      if (response.ok) {
        setFeedbackState(prev => ({ ...prev, [vulnId]: feedbackType }));
        setFeedbackMessage(feedbackType === 'false_positive' 
          ? '✓ Marked as false positive. Thank you for improving our model!' 
          : '✓ Confirmed as vulnerability. Thank you for your feedback!');
        setTimeout(() => setFeedbackMessage(null), 3000);
      }
    } catch (error) {
      console.error('Failed to submit feedback:', error);
      // Still update UI even if API fails (for demo purposes)
      setFeedbackState(prev => ({ ...prev, [vulnId]: feedbackType }));
      setFeedbackMessage('Feedback saved locally (API unavailable)');
      setTimeout(() => setFeedbackMessage(null), 3000);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-bg-secondary dark:bg-slate-900">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          className="w-12 h-12 border-4 border-primary-200 dark:border-primary-800 border-t-primary-600 dark:border-t-primary-400 rounded-full"
        />
        <p className="mt-4 text-text-secondary dark:text-slate-400">{t.report.loadingResults}</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-secondary dark:bg-slate-900 transition-colors duration-300">
      {/* Header */}
      <div className="bg-white dark:bg-slate-800 border-b border-border-light dark:border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-text-primary dark:text-white flex items-center gap-2">
                <span className="text-primary-600 dark:text-primary-400">🛡️</span> {t.report.title}
              </h1>
              <p className="text-sm text-text-muted dark:text-slate-400 mt-1">
                {t.report.scanCompleted} • {totalVulns} {totalVulns !== 1 ? t.report.issuesFound : t.report.issueFound}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={exportJSON}
                className="btn-secondary flex items-center gap-2 text-sm"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                {t.report.exportJson}
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleNewScan}
                className="btn-primary flex items-center gap-2 text-sm"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                {t.report.newScan}
              </motion.button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Executive Summary */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-1 md:grid-cols-6 gap-4 mb-6"
        >
          {/* Risk Score */}
          <div className="md:col-span-2 card dark:bg-slate-800 dark:border-slate-700 p-6 flex items-center gap-6">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center text-2xl font-bold text-white
              ${riskScore >= 70 ? 'bg-red-500' : riskScore >= 40 ? 'bg-yellow-500' : 'bg-green-500'}`}>
              {riskScore}
            </div>
            <div>
              <div className="text-sm text-text-muted dark:text-slate-400 uppercase tracking-wide">{t.report.riskScore}</div>
              <div className={`text-xl font-bold ${riskScore >= 70 ? 'text-red-600 dark:text-red-400' : riskScore >= 40 ? 'text-yellow-600 dark:text-yellow-400' : 'text-green-600 dark:text-green-400'}`}>
                {riskScore >= 70 ? t.report.highRisk : riskScore >= 40 ? t.report.mediumRisk : t.report.lowRisk}
              </div>
              <div className="text-xs text-text-muted dark:text-slate-500 mt-1">
                {t.report.basedOnFindings} {totalVulns} {totalVulns !== 1 ? t.report.findings.toLowerCase() : t.report.findings.toLowerCase()}
              </div>
            </div>
          </div>

          {/* Severity Counts */}
          <div className="card dark:bg-slate-800 dark:border-slate-700 p-4 text-center border-l-4 border-red-500 cursor-pointer hover:bg-red-50 dark:hover:bg-red-950/50 transition"
               onClick={() => setFilterSeverity(filterSeverity === 'critical' ? 'all' : 'critical')}>
            <div className="text-2xl font-bold text-red-600 dark:text-red-400">{criticalCount}</div>
            <div className="text-xs text-text-muted dark:text-slate-400">{t.report.critical}</div>
          </div>
          <div className="card dark:bg-slate-800 dark:border-slate-700 p-4 text-center border-l-4 border-orange-500 cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-950/50 transition"
               onClick={() => setFilterSeverity(filterSeverity === 'high' ? 'all' : 'high')}>
            <div className="text-2xl font-bold text-orange-600 dark:text-orange-400">{highCount}</div>
            <div className="text-xs text-text-muted dark:text-slate-400">{t.report.high}</div>
          </div>
          <div className="card dark:bg-slate-800 dark:border-slate-700 p-4 text-center border-l-4 border-yellow-500 cursor-pointer hover:bg-yellow-50 dark:hover:bg-yellow-950/50 transition"
               onClick={() => setFilterSeverity(filterSeverity === 'medium' ? 'all' : 'medium')}>
            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{mediumCount}</div>
            <div className="text-xs text-text-muted dark:text-slate-400">{t.report.medium}</div>
          </div>
          <div className="card dark:bg-slate-800 dark:border-slate-700 p-4 text-center border-l-4 border-blue-500 cursor-pointer hover:bg-blue-50 dark:hover:bg-blue-950/50 transition"
               onClick={() => setFilterSeverity(filterSeverity === 'low' ? 'all' : 'low')}>
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{lowCount + infoCount}</div>
            <div className="text-xs text-text-muted dark:text-slate-400">{t.report.lowInfo}</div>
          </div>
        </motion.div>

        {/* Filter indicator */}
        {filterSeverity !== 'all' && (
          <div className="mb-4 flex items-center gap-2">
            <span className="text-sm text-text-secondary dark:text-slate-400">{t.report.filterBy}</span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${SEVERITY_CONFIG[filterSeverity as keyof typeof SEVERITY_CONFIG]?.badge} ${SEVERITY_CONFIG[filterSeverity as keyof typeof SEVERITY_CONFIG]?.text}`}>
              {filterSeverity.toUpperCase()}
            </span>
            <button onClick={() => setFilterSeverity('all')} className="text-sm text-primary-600 dark:text-primary-400 hover:underline">
              {t.report.clearFilter}
            </button>
          </div>
        )}

        {/* No Vulnerabilities */}
        {totalVulns === 0 && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card dark:bg-slate-800 dark:border-slate-700 p-12 text-center"
          >
            <div className="w-24 h-24 mx-auto mb-6 flex items-center justify-center bg-green-100 dark:bg-green-900/50 rounded-full">
              <svg className="w-12 h-12 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-text-primary dark:text-white mb-2">{t.report.noVulnerabilities}</h2>
            <p className="text-text-secondary dark:text-slate-400 mb-6 max-w-md mx-auto">
              {t.report.codeSecure}
            </p>
            <Link href="/">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="btn-primary"
              >
                {t.report.scanAnother}
              </motion.button>
            </Link>
          </motion.div>
        )}

        {/* Split View: Code + Vulnerabilities */}
        {totalVulns > 0 && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-6"
          >
            {/* Left: Code Editor */}
            <div className="card dark:bg-slate-800 dark:border-slate-700 overflow-hidden">
              <div className="bg-gray-100 dark:bg-slate-700 px-4 py-3 border-b border-border-light dark:border-slate-600 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-text-primary dark:text-white">📄 {t.report.scannedCode}</span>
                  {isZipScan && results.length > 1 ? (
                    <select
                      value={selectedFileIndex}
                      onChange={(e) => setSelectedFileIndex(Number(e.target.value))}
                      className="text-xs px-2 py-1 bg-gray-200 dark:bg-slate-600 rounded text-text-primary dark:text-white border-none focus:ring-2 focus:ring-primary-500"
                    >
                      {results.map((r, idx) => (
                        <option key={idx} value={idx}>
                          {r.file_path} ({r.findings?.length || 0} issues)
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span className="text-xs px-2 py-0.5 bg-gray-200 dark:bg-slate-600 rounded text-text-muted dark:text-slate-300">{language}</span>
                  )}
                </div>
                <span className="text-xs text-text-muted dark:text-slate-400">
                  {isZipScan ? `${results.length} ${uiLanguage === 'th' ? 'ไฟล์' : 'files'}` : `${scannedCode.split('\n').length} ${t.report.lines}`}
                </span>
              </div>
              <div className="h-[600px]">
                <MonacoEditor
                  height="100%"
                  language={isZipScan && results[selectedFileIndex]?.language ? results[selectedFileIndex].language : (language === 'typescript' ? 'typescript' : language === 'javascript' ? 'javascript' : 'python')}
                  value={isZipScan 
                    ? (results[selectedFileIndex]?.source_code || `// Source code not available for: ${results[selectedFileIndex]?.file_path || 'unknown'}`)
                    : (scannedCode || '// No code available')}
                  theme={theme === 'dark' ? 'vs-dark' : 'vs-light'}
                  options={{
                    readOnly: true,
                    minimap: { enabled: true },
                    fontSize: 13,
                    lineNumbers: 'on',
                    scrollBeyondLastLine: false,
                    wordWrap: 'on',
                    glyphMargin: true,
                    folding: true,
                    lineDecorationsWidth: 10,
                  }}
                  onMount={handleEditorMount}
                />
              </div>
            </div>

            {/* Right: Vulnerability List */}
            <div className="space-y-4 max-h-[650px] overflow-y-auto pr-2">
              <div className="flex items-center justify-between sticky top-0 bg-bg-secondary dark:bg-slate-900 py-2 z-10">
                <h2 className="text-lg font-semibold text-text-primary dark:text-white">
                  🔍 {t.report.findings} ({filteredVulns.length})
                </h2>
              </div>

              {filteredVulns.map((vuln, index) => {
                const severityKey = vuln.severity as keyof typeof SEVERITY_CONFIG;
                const config = SEVERITY_CONFIG[severityKey] || SEVERITY_CONFIG.medium;
                const isSelected = selectedVuln === vuln.id;

                return (
                  <motion.div
                    key={vuln.id}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.03 }}
                    className={`card dark:bg-slate-800 dark:border-slate-700 overflow-hidden border-l-4 ${config.border} ${isSelected ? 'ring-2 ring-primary-500' : ''}`}
                  >
                    {/* Header - Clickable to navigate */}
                    <button
                      onClick={() => navigateToLine(vuln.line_number, vuln.id)}
                      className="w-full p-4 text-left hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
                            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${config.badge} ${config.text}`}>
                              {config.icon} {vuln.severity.toUpperCase()}
                            </span>
                            <span className="text-xs px-2 py-0.5 bg-gray-100 dark:bg-slate-700 rounded text-text-muted dark:text-slate-300 font-mono">
                              {t.report.line} {vuln.line_number}
                            </span>
                            {vuln.cwe_id && vuln.cwe_id !== 'CWE-UNKNOWN' && (
                              <a 
                                href={`https://cwe.mitre.org/data/definitions/${vuln.cwe_id.replace('CWE-', '')}.html`}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                className="text-xs px-2 py-0.5 bg-indigo-50 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 rounded hover:bg-indigo-100 dark:hover:bg-indigo-900 transition"
                              >
                                {vuln.cwe_id} ↗
                              </a>
                            )}
                          </div>
                          <h3 className="font-semibold text-text-primary dark:text-white text-sm">
                            {vuln.vulnerability_type}
                          </h3>
                          {vuln.owasp_category && (
                            <span className="text-xs text-text-muted dark:text-slate-400">{vuln.owasp_category}</span>
                          )}
                        </div>
                        <svg className="w-5 h-5 text-primary-500 dark:text-primary-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </div>
                    </button>

                    {/* Details - Always visible */}
                    <div className={`${config.bg} border-t ${config.border.replace('border-', 'border-t-')} p-4 space-y-4`}>
                      {/* Description */}
                      <div>
                        <p className="text-sm text-text-primary dark:text-slate-200">{vuln.description}</p>
                      </div>

                      {/* Recommendation */}
                      <div>
                        <h4 className="text-xs font-semibold text-text-secondary dark:text-slate-400 uppercase tracking-wide mb-2">
                          💡 {t.report.recommendation}
                        </h4>
                        <p className="text-sm text-text-primary dark:text-slate-200 bg-white dark:bg-slate-800 p-3 rounded-lg border border-green-200 dark:border-green-800">
                          {vuln.recommendation}
                        </p>
                      </div>

                      {/* Code Comparison */}
                      {vuln.vulnerable_example && vuln.secure_example && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div>
                            <h4 className="text-xs font-semibold text-red-600 dark:text-red-400 uppercase tracking-wide mb-2">
                              ❌ {t.report.vulnerable}
                            </h4>
                            <pre className="text-xs bg-red-50 dark:bg-red-950/50 p-3 rounded-lg border border-red-200 dark:border-red-800 overflow-x-auto font-mono dark:text-red-200">
                              <code>{vuln.vulnerable_example}</code>
                            </pre>
                          </div>
                          <div>
                            <h4 className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase tracking-wide mb-2">
                              ✅ {t.report.secure}
                            </h4>
                            <pre className="text-xs bg-green-50 dark:bg-green-950/50 p-3 rounded-lg border border-green-200 dark:border-green-800 overflow-x-auto font-mono dark:text-green-200">
                              <code>{vuln.secure_example}</code>
                            </pre>
                          </div>
                        </div>
                      )}

                      {/* Feedback Buttons */}
                      <div className="border-t border-gray-200 dark:border-slate-600 pt-3 mt-3">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-text-muted dark:text-slate-400">{t.report.wasAccurate}</span>
                          {feedbackState[vuln.id] ? (
                            <span className={`text-xs px-3 py-1.5 rounded-full font-medium ${
                              feedbackState[vuln.id] === 'confirmed' 
                                ? 'bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300' 
                                : 'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300'
                            }`}>
                              {feedbackState[vuln.id] === 'confirmed' ? `✓ ${t.report.confirmed}` : `⚠ ${t.report.markedFalsePositive}`}
                            </span>
                          ) : (
                            <div className="flex gap-2">
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => submitFeedback(vuln.id, 'confirmed')}
                                className="text-xs px-3 py-1.5 bg-green-100 dark:bg-green-900/50 text-green-700 dark:text-green-300 rounded-full hover:bg-green-200 dark:hover:bg-green-900 transition font-medium"
                              >
                                ✓ {t.report.confirm}
                              </motion.button>
                              <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => submitFeedback(vuln.id, 'false_positive')}
                                className="text-xs px-3 py-1.5 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-700 dark:text-yellow-300 rounded-full hover:bg-yellow-200 dark:hover:bg-yellow-900 transition font-medium"
                              >
                                ✗ {t.report.falsePositive}
                              </motion.button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </div>

      {/* Feedback Toast Message */}
      <AnimatePresence>
        {feedbackMessage && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-6 right-6 bg-gray-900 text-white px-6 py-3 rounded-lg shadow-lg z-50"
          >
            {feedbackMessage}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Custom CSS for Monaco decorations */}
      <style jsx global>{`
        .vuln-line-critical { background-color: rgba(239, 68, 68, 0.15) !important; }
        .vuln-line-high { background-color: rgba(249, 115, 22, 0.15) !important; }
        .vuln-line-medium { background-color: rgba(234, 179, 8, 0.15) !important; }
        .vuln-line-low { background-color: rgba(59, 130, 246, 0.1) !important; }
        .vuln-glyph-critical { background-color: #ef4444; border-radius: 50%; margin-left: 3px; }
        .vuln-glyph-high { background-color: #f97316; border-radius: 50%; margin-left: 3px; }
        .vuln-glyph-medium { background-color: #eab308; border-radius: 50%; margin-left: 3px; }
        .vuln-glyph-low { background-color: #3b82f6; border-radius: 50%; margin-left: 3px; }
      `}</style>
    </div>
  );
}
