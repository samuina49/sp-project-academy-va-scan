"""
Simple Pattern-Based Scanner
Fallback when Bandit/Semgrep not available
"""
import re
from typing import List
from pathlib import Path

from app.models.scan_models import VulnerabilityFinding, FileScanResult, SeverityEnum


class SimplePatternScanner:
    """Lightweight pattern-based vulnerability scanner"""
    
    PATTERNS = [
        # ========== PYTHON PATTERNS ==========
        # Command Injection
        {
            'pattern': r'os\.system\s*\([\'\"].*[\'\"]',
            'title': 'Command Injection via os.system()',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-78',
            'description': 'Potential command injection vulnerability. Avoid using os.system() with user input.',
            'languages': ['python']
        },
        # Command Injection with f-string
        {
            'pattern': r'os\.system\s*\(\s*f[\'\"].*\{.*\}',
            'title': 'Command Injection via os.system() with f-string',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-78',
            'description': 'Command injection vulnerability! User input in os.system() f-string allows arbitrary command execution.',
            'languages': ['python']
        },
        # Command Injection with string concatenation
        {
            'pattern': r'os\.system\s*\([^\)]*\+',
            'title': 'Command Injection via os.system() with concatenation',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-78',
            'description': 'Command injection vulnerability! String concatenation in os.system() allows arbitrary command execution.',
            'languages': ['python']
        },
        # Command Injection with .format()
        {
            'pattern': r'os\.system\s*\([\'\"].*[\'\"]\.format\s*\(',
            'title': 'Command Injection via os.system() with .format()',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-78',
            'description': 'Command injection vulnerability! .format() in os.system() allows arbitrary command execution.',
            'languages': ['python']
        },
        # Subprocess with shell=True (very dangerous)
        {
            'pattern': r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True',
            'title': 'Command Injection via subprocess with shell=True',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-78',
            'description': 'subprocess with shell=True is extremely dangerous. Avoid using with user input.',
            'languages': ['python']
        },
        {
            'pattern': r'subprocess\.(call|run|Popen)\s*\(',
            'title': 'Command Execution via subprocess',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-78',
            'description': 'Subprocess execution detected. Ensure input is properly sanitized.',
            'languages': ['python']
        },
        {
            'pattern': r'eval\s*\(',
            'title': 'Code Injection via eval()',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-94',
            'description': 'eval() allows arbitrary code execution. Never use with untrusted input.',
            'languages': ['python', 'javascript', 'typescript']
        },
        {
            'pattern': r'exec\s*\(',
            'title': 'Code Injection via exec()',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-94',
            'description': 'exec() allows arbitrary code execution. Extremely dangerous with user input.',
            'languages': ['python']  # Only Python - JS/TS exec() is handled separately
        },
        
        # SQL Injection
        {
            'pattern': r'(execute|cursor\.execute)\s*\(\s*[\'\"].*%s.*[\'\"]',
            'title': 'Potential SQL Injection',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query uses string formatting. Use parameterized queries instead.',
            'languages': ['python'],
            'negative_patterns': [r'execute.*%s.*,\s*\(']  # Exclude parameterized
        },
        {
            'pattern': r'(execute|cursor\.execute)\s*\(\s*[\'\"].*\s*\+\s*\w+',
            'title': 'SQL Injection via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query uses string concatenation. Extremely vulnerable to injection.',
            'languages': ['python']
        },
        # SQL Injection - String concatenation with SQL keywords (Python)
        {
            'pattern': r'[\'\"]SELECT\s+.*[\'\"]?\s*\+\s*\w+',
            'title': 'SQL Injection via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query built using string concatenation. Use parameterized queries with execute(sql, params).',
            'languages': ['python']
        },
        {
            'pattern': r'[\'\"]INSERT\s+INTO.*[\'\"]?\s*\+\s*\w+',
            'title': 'SQL Injection via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL INSERT uses string concatenation. Use parameterized queries.',
            'languages': ['python']
        },
        {
            'pattern': r'[\'\"]UPDATE\s+.*[\'\"]?\s*\+\s*\w+',
            'title': 'SQL Injection via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL UPDATE uses string concatenation. Use parameterized queries.',
            'languages': ['python']
        },
        {
            'pattern': r'[\'\"]DELETE\s+FROM.*[\'\"]?\s*\+\s*\w+',
            'title': 'SQL Injection via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL DELETE uses string concatenation. Use parameterized queries.',
            'languages': ['python']
        },
        # SQL Injection - f-string format (Python)
        {
            'pattern': r'f[\'\"]SELECT\s+.*\{.*\}',
            'title': 'SQL Injection via f-string',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query built using f-string. Use parameterized queries instead.',
            'languages': ['python']
        },
        {
            'pattern': r'f[\'\"]INSERT\s+INTO.*\{.*\}',
            'title': 'SQL Injection via f-string',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL INSERT uses f-string. Use parameterized queries.',
            'languages': ['python']
        },
        {
            'pattern': r'f[\'\"]UPDATE\s+.*\{.*\}',
            'title': 'SQL Injection via f-string',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL UPDATE uses f-string. Use parameterized queries.',
            'languages': ['python']
        },
        {
            'pattern': r'f[\'\"]DELETE\s+FROM.*\{.*\}',
            'title': 'SQL Injection via f-string',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL DELETE uses f-string. Use parameterized queries.',
            'languages': ['python']
        },
        # SQL Injection - .format() method (Python)
        {
            'pattern': r'[\'\"]SELECT\s+.*[\'\"]\.format\s*\(',
            'title': 'SQL Injection via .format()',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query uses .format(). Use parameterized queries instead.',
            'languages': ['python']
        },
        
        # Path Traversal
        {
            'pattern': r'open\s*\(\s*[\'\"]\.\./.*[\'\"]',
            'title': 'Path Traversal Attempt',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-22',
            'description': 'Path contains directory traversal patterns (../). Validate and sanitize file paths.',
            'languages': ['python']
        },
        # Path Traversal with string concatenation (CRITICAL - user input in file path)
        {
            'pattern': r'open\s*\([^\)]*\+',
            'title': 'Path Traversal via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-22',
            'description': 'File path built using string concatenation. User input can lead to path traversal attacks.',
            'languages': ['python']
        },
        # Path Traversal with f-string (user input in file path)
        {
            'pattern': r'f[\'\"].*/(uploads|files|data|static|public|tmp|temp)/\{',
            'title': 'Path Traversal via f-string',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-22',
            'description': 'User input in file path can lead to path traversal. Validate and sanitize the path.',
            'languages': ['python']
        },
        # Path Traversal - open() with f-string
        {
            'pattern': r'open\s*\(\s*f[\'\"].*\{',
            'title': 'Path Traversal via open() with f-string',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-22',
            'description': 'User input in open() file path allows path traversal attacks.',
            'languages': ['python']
        },
        # Path Traversal - open() with variable
        {
            'pattern': r'open\s*\(\s*\w+\s*[,\)]',
            'title': 'Potential Path Traversal',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-22',
            'description': 'Opening file with variable path. Ensure path is validated to prevent traversal.',
            'languages': ['python']
        },
        
        # Information Disclosure - Debug print statements
        {
            'pattern': r'print\s*\(\s*f?[\'\"].*DEBUG.*[\'\"]',
            'title': 'Information Disclosure via Debug Output',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-200',
            'description': 'Debug print statements can leak sensitive information in production.',
            'languages': ['python']
        },
        {
            'pattern': r'print\s*\(\s*f[\'\"].*\{.*\}.*[\'\"]',
            'title': 'Potential Information Disclosure',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-200',
            'description': 'Print statement with dynamic content may leak sensitive data. Remove in production.',
            'languages': ['python']
        },
        {
            'pattern': r'console\.log\s*\(.*password|secret|token|key',
            'title': 'Information Disclosure via console.log',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-200',
            'description': 'Logging sensitive information to console. Remove in production.',
            'languages': ['javascript', 'typescript']
        },
        
        # Hardcoded Credentials - Enhanced patterns to catch more variations
        {
            'pattern': r'(DB_USER|DB_USERNAME|DATABASE_USER|db_user|db_username|admin_user|ADMIN_USER)\s*=\s*[\'"]([^\'"]{3,})[\'"]',
            'title': 'Hardcoded Database Username',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-798',
            'description': 'Database username hardcoded in source code. Use environment variables instead.',
            'languages': ['python', 'javascript', 'typescript']
        },
        {
            'pattern': r'(username|user|USERNAME|USER)\s*=\s*[\'"](?!.*\{)[\'"]*(admin|root|administrator|sa)[\'"]',
            'title': 'Hardcoded Admin Username',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-798',
            'description': 'Hardcoded admin/root username detected. Use environment variables instead.',
            'languages': ['python', 'javascript', 'typescript']
        },
        {
            'pattern': r'(password|passwd|pwd|pass|PASSWORD|PASSWD|PWD)\s*=\s*[\'\"]([^\'\"]{3,})[\'\"]',
            'title': 'Hardcoded Password Detected',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-798',
            'description': 'Password hardcoded in source code. Use environment variables or secure vaults instead.',
            'languages': ['python', 'javascript', 'typescript'],
            'negative_patterns': [r'(password|pass|pwd)\s*=\s*[\'\"][\'\"]']  # Exclude empty strings
        },
        {
            'pattern': r'(apiKey|api_key|API_KEY|apikey|APIKEY|api_secret|API_SECRET)\s*=\s*[\'\"]([^\'\"]{8,})[\'\"]',
            'title': 'Hardcoded API Key Detected',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-798',
            'description': 'API key hardcoded in source code. Use environment variables or secure vaults instead.',
            'languages': ['python', 'javascript', 'typescript']
        },
        {
            'pattern': r'(secret|SECRET|token|TOKEN|auth_token|AUTH_TOKEN)\s*=\s*[\'\"]([^\'\"]{8,})[\'\"]',
            'title': 'Hardcoded Secret/Token Detected',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-798',
            'description': 'Secret or token hardcoded in source code. Use environment variables or secure vaults instead.',
            'languages': ['python', 'javascript', 'typescript']
        },
        {
            'pattern': r'(DB_PASSWORD|DB_PASS|DATABASE_PASSWORD|db_password|db_pass)\s*=\s*[\'\"]([^\'\"]{3,})[\'\"]',
            'title': 'Hardcoded Database Password',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-798',
            'description': 'Database password hardcoded in source code! Use environment variables immediately.',
            'languages': ['python', 'javascript', 'typescript']
        },
        {
            'pattern': r'(AWS_SECRET|AWS_ACCESS_KEY|AZURE_KEY|GCP_KEY|aws_secret|azure_key|gcp_key)\s*=\s*[\'\"]([^\'\"]{8,})[\'\"]',
            'title': 'Hardcoded Cloud Credentials',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-798',
            'description': 'Cloud service credentials hardcoded in source code! Rotate keys and use IAM roles.',
            'languages': ['python', 'javascript', 'typescript']
        },
        
        # Unsafe Deserialization
        {
            'pattern': r'pickle\.loads\s*\(',
            'title': 'Unsafe Deserialization (pickle.loads)',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-502',
            'description': 'pickle.loads() can execute arbitrary code. Never use with untrusted data.',
            'languages': ['python']
        },
        {
            'pattern': r'pickle\.load\s*\(',
            'title': 'Unsafe Deserialization (pickle.load)',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-502',
            'description': 'pickle.load() can execute arbitrary code. Never use with untrusted data.',
            'languages': ['python']
        },
        # yaml.load without safe loader
        {
            'pattern': r'yaml\.load\s*\([^)]*\)',
            'title': 'Unsafe YAML Deserialization',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-502',
            'description': 'yaml.load() without Loader=SafeLoader can execute arbitrary code.',
            'languages': ['python'],
            'negative_patterns': [r'yaml\.load\s*\([^)]*Loader\s*=\s*SafeLoader']
        },
        # marshal.loads
        {
            'pattern': r'marshal\.loads?\s*\(',
            'title': 'Unsafe Deserialization (marshal)',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-502',
            'description': 'marshal module can execute arbitrary code. Never use with untrusted data.',
            'languages': ['python']
        },
        # shelve (uses pickle internally)
        {
            'pattern': r'shelve\.open\s*\(',
            'title': 'Potential Unsafe Deserialization (shelve)',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-502',
            'description': 'shelve uses pickle internally. Avoid with untrusted data sources.',
            'languages': ['python']
        },
        
        # Weak Random
        {
            'pattern': r'random\.(random|randint)\s*\(',
            'title': 'Weak Random Number Generator',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-338',
            'description': 'random module is not cryptographically secure. Use secrets module for security.',
            'languages': ['python']
        },
        
        # Weak Crypto
        {
            'pattern': r'hashlib\.md5\s*\(',
            'title': 'Weak Cryptographic Hash (MD5)',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-327',
            'description': 'MD5 is cryptographically broken. Use SHA-256 or stronger.',
            'languages': ['python']
        },
        {
            'pattern': r'sha1\s*\(',
            'title': 'Weak Cryptographic Hash (SHA1)',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-327',
            'description': 'SHA1 is deprecated. Use SHA-256 or stronger for security purposes.',
            'languages': ['python']
        },
        
        # ========== JAVASCRIPT/TYPESCRIPT PATTERNS ==========
        {
            'pattern': r'new\s+Function\s*\(',
            'title': 'Code Injection via Function Constructor',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-94',
            'description': 'Function constructor allows arbitrary code execution like eval().',
            'languages': ['javascript', 'typescript']
        },
        # XSS - Express res.send() with template literal
        {
            'pattern': r'res\.send\s*\(\s*`[^`]*\$\{',
            'title': 'Reflected XSS via res.send() with Template Literal',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-79',
            'description': 'User input in res.send() template literal can lead to XSS. Sanitize input or use a templating engine with auto-escaping.',
            'languages': ['javascript', 'typescript']
        },
        # XSS - Express res.send() with string concatenation
        {
            'pattern': r'res\.send\s*\([^)]*\+',
            'title': 'Reflected XSS via res.send() with Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-79',
            'description': 'String concatenation in res.send() can lead to XSS. Sanitize user input.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'\.innerHTML\s*=',
            'title': 'XSS via innerHTML',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-79',
            'description': 'innerHTML can execute script tags. Use textContent or sanitize input.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'\.outerHTML\s*=',
            'title': 'XSS via outerHTML',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-79',
            'description': 'outerHTML can execute scripts. Use safer DOM manipulation methods.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'document\.write\s*\(',
            'title': 'XSS via document.write',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-79',
            'description': 'document.write() can introduce XSS. Use safer DOM methods.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'Math\.random\s*\(',
            'title': 'Weak Random Number Generator',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-338',
            'description': 'Math.random() is not cryptographically secure. Use crypto.getRandomValues().',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'==\s*["\']',
            'title': 'Use Strict Equality (===)',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-1023',
            'description': 'Use === instead of == to avoid type coercion vulnerabilities.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'(SELECT|INSERT|UPDATE|DELETE).*\+\s*\w+',
            'title': 'SQL Injection via String Concatenation',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query uses string concatenation. Use parameterized queries.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'(SELECT|INSERT|UPDATE|DELETE|DROP).*\$\{',
            'title': 'SQL Injection via Template Literal',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-89',
            'description': 'SQL query uses template literals with variables. Use parameterized queries.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'(child_process\.)?(exec|spawn)\s*\(',
            'title': 'Command Injection Risk',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-78',
            'description': 'Command execution detected. Sanitize all user input properly.',
            'languages': ['javascript', 'typescript']
        },
        {
            'pattern': r'(fs\.)?(readFile|readFileSync|writeFile|writeFileSync)\s*\(',
            'title': 'Path Traversal Risk',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-22',
            'description': 'File operations detected. Validate and sanitize file paths to prevent path traversal.',
            'languages': ['javascript', 'typescript']
        },
        
        # ========== A10: SSRF (Server-Side Request Forgery) ==========
        {
            'pattern': r'requests\.(get|post|put|delete|head|patch)\s*\([^"\')]*\+',
            'title': 'SSRF - Dynamic URL in requests',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-918',
            'description': 'HTTP request with dynamic URL. Validate and whitelist allowed URLs to prevent SSRF.',
            'languages': ['python'],
            'owasp': 'A10:2021-SSRF'
        },
        {
            'pattern': r'urllib\.(request\.)?urlopen\s*\([^"\')]*\+',
            'title': 'SSRF - Dynamic URL in urllib',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-918',
            'description': 'urllib with dynamic URL. Validate URLs to prevent SSRF attacks.',
            'languages': ['python'],
            'owasp': 'A10:2021-SSRF'
        },
        {
            'pattern': r'httpx?\.(get|post|request)\s*\([^"\')]*\+',
            'title': 'SSRF - Dynamic URL in http client',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-918',
            'description': 'HTTP client with dynamic URL. Implement URL validation.',
            'languages': ['python'],
            'owasp': 'A10:2021-SSRF'
        },
        {
            'pattern': r'fetch\s*\(\s*[^\'"][^\)]+\)',
            'title': 'SSRF - Dynamic URL in fetch',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-918',
            'description': 'fetch() with dynamic URL variable. Validate URLs to prevent SSRF.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A10:2021-SSRF'
        },
        {
            'pattern': r'axios\.(get|post|put|delete|patch)\s*\([^\'"][^\)]+\)',
            'title': 'SSRF - Dynamic URL in axios',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-918',
            'description': 'axios with dynamic URL. Implement URL whitelist validation.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A10:2021-SSRF'
        },
        
        # ========== A01: Broken Access Control ==========
        {
            'pattern': r'Access-Control-Allow-Origin.*["\']?\s*\*\s*["\']?',
            'title': 'CORS Wildcard - Allows Any Origin',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-942',
            'description': 'CORS allows all origins. Restrict to specific trusted domains.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A01:2021-Broken Access Control'
        },
        {
            'pattern': r'cors\s*\(\s*\{\s*origin\s*:\s*[\'"]?\*[\'"]?',
            'title': 'CORS Misconfiguration',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-942',
            'description': 'CORS middleware with wildcard origin. Use specific origins.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A01:2021-Broken Access Control'
        },
        {
            'pattern': r'\.\./\.\./\.\.',
            'title': 'Path Traversal Pattern',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-22',
            'description': 'Directory traversal pattern detected. Sanitize file paths.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A01:2021-Broken Access Control'
        },
        
        # ========== A03: Enhanced Injection ==========
        {
            'pattern': r'\$where\s*:.*(\+|\$\{)',
            'title': 'NoSQL Injection - MongoDB $where',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-943',
            'description': 'MongoDB $where with dynamic input. Use parameterized queries.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A03:2021-Injection'
        },
        {
            'pattern': r'\.find\s*\(\s*\{[^}]*:\s*\w+\s*\}',
            'title': 'NoSQL Injection Risk',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-943',
            'description': 'MongoDB query with variable input. Validate input types.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A03:2021-Injection'
        },
        {
            'pattern': r'render_template_string\s*\(',
            'title': 'Template Injection (SSTI)',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-1336',
            'description': 'render_template_string can lead to SSTI. Use render_template with separate files.',
            'languages': ['python'],
            'owasp': 'A03:2021-Injection'
        },
        {
            'pattern': r'jinja2\.Template\s*\([^)]*\+',
            'title': 'Template Injection - Jinja2',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-1336',
            'description': 'Dynamic Jinja2 template creation. Use static templates.',
            'languages': ['python'],
            'owasp': 'A03:2021-Injection'
        },
        {
            'pattern': r'(logging\.(info|debug|warning|error)|logger\.\w+)\s*\([^)]*%[^)]*\)',
            'title': 'Log Injection Risk',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-117',
            'description': 'User input may be logged unsafely. Sanitize log inputs.',
            'languages': ['python'],
            'owasp': 'A03:2021-Injection'
        },
        {
            'pattern': r'console\.(log|info|warn|error)\s*\([^)]*\+[^)]*\)',
            'title': 'Log Injection Risk',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-117',
            'description': 'User input concatenated in console log. Sanitize inputs.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A03:2021-Injection'
        },
        
        # ========== A02: Cryptographic Failures ==========
        {
            'pattern': r'DES\s*\(|Blowfish|RC4|RC2',
            'title': 'Weak Encryption Algorithm',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-327',
            'description': 'Weak/deprecated encryption. Use AES-256 or ChaCha20.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A02:2021-Cryptographic Failures'
        },
        {
            'pattern': r'AES.*ECB|ECB.*AES|mode\s*=\s*.*ECB',
            'title': 'Insecure ECB Mode',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-327',
            'description': 'ECB mode is insecure. Use CBC, GCM, or CTR mode.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A02:2021-Cryptographic Failures'
        },
        {
            'pattern': r'verify\s*=\s*False|rejectUnauthorized\s*:\s*false',
            'title': 'SSL Certificate Verification Disabled',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-295',
            'description': 'Certificate verification disabled. Enable proper SSL validation.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A02:2021-Cryptographic Failures'
        },
        {
            'pattern': r'ssl\.PROTOCOL_SSLv[23]|ssl\.PROTOCOL_TLSv1[^23]',
            'title': 'Deprecated SSL/TLS Version',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-326',
            'description': 'Deprecated SSL/TLS version. Use TLS 1.2 or higher.',
            'languages': ['python'],
            'owasp': 'A02:2021-Cryptographic Failures'
        },
        
        # ========== A05: Security Misconfiguration ==========
        {
            'pattern': r'DEBUG\s*=\s*True|debug\s*=\s*true|debug:\s*true',
            'title': 'Debug Mode Enabled',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-489',
            'description': 'Debug mode should be disabled in production.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A05:2021-Security Misconfiguration'
        },
        {
            'pattern': r'app\.run\s*\([^)]*debug\s*=\s*True',
            'title': 'Flask Debug Mode in Production',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-489',
            'description': 'Flask debug mode exposes sensitive info. Disable in production.',
            'languages': ['python'],
            'owasp': 'A05:2021-Security Misconfiguration'
        },
        {
            'pattern': r'SecretKeyCredential.*=.*[\'"]\w+[\'"]|SECRET_KEY\s*=\s*[\'"]\w+[\'"]',
            'title': 'Hardcoded Secret Key',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-798',
            'description': 'Secret key hardcoded. Use environment variables.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A05:2021-Security Misconfiguration'
        },
        
        # ========== A07: Auth Failures ==========
        {
            'pattern': r'jwt\.decode\s*\([^)]*verify\s*=\s*False',
            'title': 'JWT Signature Verification Disabled',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-347',
            'description': 'JWT verification disabled. Always verify JWT signatures.',
            'languages': ['python'],
            'owasp': 'A07:2021-Auth Failures'
        },
        {
            'pattern': r'algorithms\s*=\s*\[[\'"]none[\'"]\]|algorithm.*none',
            'title': 'JWT None Algorithm Attack',
            'severity': SeverityEnum.CRITICAL,
            'cwe_id': 'CWE-347',
            'description': 'JWT allows "none" algorithm. Use RS256 or HS256.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A07:2021-Auth Failures'
        },
        {
            'pattern': r'session\[.*\]\s*=\s*.*password|password.*session',
            'title': 'Password Stored in Session',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-312',
            'description': 'Never store passwords in session. Use secure tokens.',
            'languages': ['python'],
            'owasp': 'A07:2021-Auth Failures'
        },
        
        # ========== A08: Data Integrity Failures ==========
        {
            'pattern': r'yaml\.load\s*\([^)]*\)',
            'title': 'Unsafe YAML Deserialization',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-502',
            'description': 'yaml.load is unsafe. Use yaml.safe_load instead.',
            'languages': ['python'],
            'owasp': 'A08:2021-Data Integrity'
        },
        {
            'pattern': r'dangerouslySetInnerHTML',
            'title': 'XSS via dangerouslySetInnerHTML',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-79',
            'description': 'dangerouslySetInnerHTML can lead to XSS. Sanitize content.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A08:2021-Data Integrity'
        },
        {
            'pattern': r'JSON\.parse\s*\([^)]*\)',
            'title': 'Unsafe JSON Parsing',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-502',
            'description': 'JSON.parse on untrusted data. Validate JSON schema.',
            'languages': ['javascript', 'typescript'],
            'owasp': 'A08:2021-Data Integrity'
        },
        
        # ========== A04: Insecure Design ==========
        {
            'pattern': r'sleep\s*\(\s*\d+\s*\)|time\.sleep|Thread\.sleep',
            'title': 'A04: Timing-based Logic (Potential Race Condition)',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-362',
            'description': 'Sleep in code may indicate timing-based vulnerability or race condition.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A04:2021-Insecure Design'
        },
        {
            'pattern': r'TODO.*security|FIXME.*auth|HACK.*password|XXX.*vuln',
            'title': 'A04: Security TODO/FIXME Comment',
            'severity': SeverityEnum.LOW,
            'cwe_id': 'CWE-1078',
            'description': 'Security-related TODO comment found. Address before production.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A04:2021-Insecure Design'
        },
        {
            'pattern': r'@app\.route.*methods.*POST.*\n(?!.*@(login_required|requires_auth|authenticated))',
            'title': 'A04: POST Route Without Auth Decorator',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-862',
            'description': 'POST endpoint may lack authentication. Verify access control.',
            'languages': ['python'],
            'owasp': 'A04:2021-Insecure Design'
        },
        
        # ========== A06: Vulnerable Components ==========
        # Note: Removed generic import/require patterns as they cause too many false positives
        # Real vulnerable component detection should use npm audit / pip-audit tools
        {
            'pattern': r'pip install|npm install|yarn add',
            'title': 'A06: Package Installation Command',
            'severity': SeverityEnum.INFO,
            'cwe_id': 'CWE-1035',
            'description': 'Package installation detected. Use pip-audit or npm audit to verify.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A06:2021-Vulnerable Components'
        },
        
        # ========== A09: Security Logging Failures ==========
        {
            'pattern': r'(print|console\.log|logging\.\w+|logger\.\w+)\s*\([^)]*password[^)]*\)',
            'title': 'A09: Password in Log Output',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-532',
            'description': 'Password may be logged. Never log sensitive credentials.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A09:2021-Logging Failures'
        },
        {
            'pattern': r'(print|console\.log|logging\.\w+|logger\.\w+)\s*\([^)]*token[^)]*\)',
            'title': 'A09: Token in Log Output',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-532',
            'description': 'Token may be logged. Avoid logging authentication tokens.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A09:2021-Logging Failures'
        },
        {
            'pattern': r'(print|console\.log|logging\.\w+|logger\.\w+)\s*\([^)]*secret[^)]*\)',
            'title': 'A09: Secret in Log Output',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-532',
            'description': 'Secret may be logged. Remove sensitive data from logs.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A09:2021-Logging Failures'
        },
        {
            'pattern': r'(print|console\.log|logging\.\w+|logger\.\w+)\s*\([^)]*api_key[^)]*\)',
            'title': 'A09: API Key in Log Output',
            'severity': SeverityEnum.HIGH,
            'cwe_id': 'CWE-532',
            'description': 'API key may be logged. Never log API credentials.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A09:2021-Logging Failures'
        },
        {
            'pattern': r'except\s*:\s*\n\s*pass|catch\s*\([^)]*\)\s*\{\s*\}',
            'title': 'A09: Empty Exception Handler (Silent Failure)',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-390',
            'description': 'Empty exception handler suppresses errors. Log exceptions for debugging.',
            'languages': ['python', 'javascript', 'typescript'],
            'owasp': 'A09:2021-Logging Failures'
        },
        {
            'pattern': r'except\s*Exception.*pass|except\s*:\s*pass',
            'title': 'A09: Broad Exception with No Logging',
            'severity': SeverityEnum.MEDIUM,
            'cwe_id': 'CWE-754',
            'description': 'Broad exception silently caught. Log errors for security monitoring.',
            'languages': ['python'],
            'owasp': 'A09:2021-Logging Failures'
        },
    ]
    
    # Safe patterns to exclude (reduce false positives)
    SAFE_PATTERNS = {
        'python': [
            r'yaml\.safe_load',  # Safe YAML
            r'execute.*,\s*\([^)]+\)\s*\)',  # Parameterized SQL
            r'execute.*,\s*\[[^\]]+\]\s*\)',  # Parameterized SQL with list
            r'os\.environ\.get',  # Safe env variable access
            r'os\.environ\[',  # Safe env variable access
            r'hashlib\.sha256',  # Safe hash (NOT md5/sha1)
            r'hashlib\.sha512',  # Safe hash
            r'hashlib\.sha384',  # Safe hash
            r'hashlib\.blake2',  # Safe hash
            r'secrets\.',  # Safe random
            r'sanitize\s*\(',  # Sanitized input
            r'escape\s*\(',  # Escaped input
            r'requests\.(get|post|put|delete)\s*\(\s*["\']https?://',  # Static URL requests
            r'urllib\.request\.urlopen\s*\(\s*["\']https?://',  # Static URL urllib
            r'=\s*os\.environ',  # Assignment from env
            r'from\s+\.\w+\s+import',  # Local relative import (A06 safe)
            r'from\s+\.\s+import',  # Local relative import
            r'^\s*\w+\s*=\s*\w+\s*[\+\-\*\/]\s*\w+\s*$',  # Simple arithmetic (A04 safe)
            r'^\s*print\s*\(\s*["\'][^"\']*["\']\s*\)\s*$',  # Simple print statement
        ],
        'javascript': [
            r'textContent\s*=',  # Safe DOM assignment
            r'createElement',  # Safe DOM creation
            r'\?.*=.*\$\d',  # Parameterized query
            r'encodeURIComponent',  # URL encoding
            r'DOMPurify\.sanitize',  # Sanitization
            r'fetch\s*\(\s*["\']https?://',  # Static URL fetch
            r'axios\.(get|post)\s*\(\s*["\']https?://',  # Static URL axios
            r'process\.env\.',  # Safe env variable
            r'require\s*\(\s*[\'"]\.',  # Local require (A06 safe)
            r'from\s+[\'"]\.',  # Local ES import (A06 safe)
        ],
        'typescript': [
            r'textContent\s*=',
            r'createElement',
            r'encodeURIComponent',
            r'fetch\s*\(\s*["\']https?://',  # Static URL fetch
            r'axios\.(get|post)\s*\(\s*["\']https?://',  # Static URL axios
            r'process\.env\.',  # Safe env variable
            r'from\s+[\'"]\.',  # Local ES import (A06 safe)
        ]
    }
    
    def is_safe_code(self, code: str, language: str) -> bool:
        """Check if code contains safe coding patterns - checks entire code block"""
        safe_patterns = self.SAFE_PATTERNS.get(language, [])
        for pattern in safe_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return True
        return False
    
    def is_comment_line(self, line: str, language: str) -> bool:
        """Check if line is a comment and should be skipped"""
        stripped = line.strip()
        
        if not stripped:
            return False
        
        if language == 'python':
            # Python comments start with #
            if stripped.startswith('#'):
                return True
            # Python docstrings (simple check)
            if stripped.startswith('"""') or stripped.startswith("'''"):
                return True
                
        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript single-line comments
            if stripped.startswith('//'):
                return True
            # Block comments
            if stripped.startswith('/*') or stripped.startswith('*'):
                return True
                
        return False
    
    def scan_code(self, code: str, language: str = "python") -> FileScanResult:
        """Scan code string for vulnerabilities"""
        findings: List[VulnerabilityFinding] = []
        lines = code.split('\n')
        
        # Check if entire code block contains safe patterns (for multi-line safe code)
        is_safe_code_block = self.is_safe_code(code, language)
        
        # Filter patterns for this language
        applicable_patterns = [
            p for p in self.PATTERNS 
            if language in p.get('languages', ['python'])
        ]
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments - don't scan comment lines
            if self.is_comment_line(line, language):
                continue
                
            # Skip if entire code block is safe or line contains safe patterns
            if is_safe_code_block or self.is_safe_code(line, language):
                continue
                
            for pattern_info in applicable_patterns:
                if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                    # Check negative patterns (exclusions)
                    negative_patterns = pattern_info.get('negative_patterns', [])
                    is_excluded = False
                    for neg_pattern in negative_patterns:
                        if re.search(neg_pattern, line, re.IGNORECASE):
                            is_excluded = True
                            break
                    
                    if is_excluded:
                        continue
                    
                    finding = VulnerabilityFinding(
                        tool="simple_scanner",
                        rule_id=pattern_info['title'].lower().replace(' ', '_'),
                        severity=pattern_info['severity'],
                        message=pattern_info['description'],
                        start_line=line_num,
                        end_line=line_num,
                        code_snippet=line.strip(),
                        cwe_id=pattern_info['cwe_id']
                    )
                    findings.append(finding)
        
        # Deduplicate findings - keep highest severity per line
        deduplicated = {}
        for finding in findings:
            line_key = finding.start_line
            if line_key not in deduplicated:
                deduplicated[line_key] = finding
            else:
                # Keep the higher severity
                existing = deduplicated[line_key]
                severity_order = {
                    SeverityEnum.CRITICAL: 0,
                    SeverityEnum.HIGH: 1,
                    SeverityEnum.MEDIUM: 2,
                    SeverityEnum.LOW: 3,
                    SeverityEnum.INFO: 4
                }
                if severity_order.get(finding.severity, 99) < severity_order.get(existing.severity, 99):
                    deduplicated[line_key] = finding
        
        return FileScanResult(
            file_path="<code>",
            language=language,
            findings=list(deduplicated.values()),
            scan_duration_ms=0.0
        )
    
    def scan_file(self, file_path: Path) -> FileScanResult:
        """Scan file for vulnerabilities"""
        try:
            # Detect language from file extension
            ext = file_path.suffix.lower()
            language_map = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.jsx': 'javascript',
                '.tsx': 'typescript'
            }
            language = language_map.get(ext, 'python')
            
            code = file_path.read_text(encoding='utf-8')
            result = self.scan_code(code, language=language)
            result.file_path = str(file_path)
            return result
        except Exception as e:
            return FileScanResult(
                file_path=str(file_path),
                language="unknown",
                findings=[],
                scan_duration_ms=0.0
            )
