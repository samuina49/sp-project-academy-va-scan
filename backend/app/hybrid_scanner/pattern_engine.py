"""
Pattern Matching Engine (Phase 1) — High Recall
=================================================
Deterministic rule-based scanner using regex + lightweight AST / dataflow checks.

Supported CWEs:
    CWE-89   SQL Injection
    CWE-77   Command Injection
    CWE-22   Path Traversal
    CWE-502  Insecure Deserialization
    CWE-918  SSRF
    CWE-798  Hardcoded Secrets

Design goals:
    * Recall >= 0.95 on targeted CWE test cases
    * FALSE POSITIVES ARE ACCEPTABLE (AI Phase 2 filters them)
    * Every rule is explainable & CWE-labeled
    * Negative patterns reduce obvious non-issues
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from app.hybrid_scanner.models import (
    PatternFinding, CWECategory, VulnLabel, Severity
)


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class PatternRule:
    """A single pattern rule."""
    rule_id: str
    cwe: CWECategory
    pattern: str              # Regex pattern (IGNORECASE applied)
    title: str
    severity: Severity
    confidence: str           # "high" | "medium" | "low"
    explanation: str          # WHY this pattern is dangerous
    languages: List[str]      # ["python"], ["javascript"], or both
    negative_patterns: Optional[List[str]] = None  # Cancel match if any hit
    multiline: bool = False   # Use re.DOTALL for cross-line patterns


# =====================================================================
# R U L E   D E F I N I T I O N S
# =====================================================================

RULES: List[PatternRule] = []


def _r(rule_id: str, cwe: CWECategory, pattern: str, title: str,
       severity: Severity, confidence: str, explanation: str,
       languages: List[str], negative_patterns: Optional[List[str]] = None,
       multiline: bool = False):
    """Helper to register a rule."""
    RULES.append(PatternRule(
        rule_id=rule_id, cwe=cwe, pattern=pattern, title=title,
        severity=severity, confidence=confidence, explanation=explanation,
        languages=languages, negative_patterns=negative_patterns,
        multiline=multiline,
    ))


# ─── CWE-89: SQL Injection ──────────────────────────────────────────

# Python — f-string in SQL
_r("SQLI_PY_FSTRING", CWECategory.SQL_INJECTION,
   r"""f['\"](?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM|DROP|ALTER)\s+.*\{""",
   "SQL Injection via f-string",
   Severity.HIGH, "high",
   "SQL query is built using an f-string with interpolated variables. "
   "An attacker controlling the interpolated value can inject arbitrary SQL. "
   "Use parameterized queries: cursor.execute('SELECT ... WHERE id=%s', (val,))",
   ["python"])

# Python — string concatenation in SQL
_r("SQLI_PY_CONCAT", CWECategory.SQL_INJECTION,
   r"""['"](?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+.*['"]\s*\+\s*\w+""",
   "SQL Injection via string concatenation",
   Severity.HIGH, "high",
   "SQL statement is built by concatenating a variable into the query string. "
   "Any unsanitized user input in that variable enables SQL injection.",
   ["python"])

# Python — .format() in SQL
_r("SQLI_PY_FORMAT", CWECategory.SQL_INJECTION,
   r"""['"](?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+.*['"]\.format\s*\(""",
   "SQL Injection via .format()",
   Severity.HIGH, "high",
   "SQL query uses .format() which inserts values unsafely. "
   "Use parameterized queries instead.",
   ["python"])

# Python — %-formatting in SQL
_r("SQLI_PY_PERCENT", CWECategory.SQL_INJECTION,
   r"""['"](?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+.*%[sd].*['"]\s*%\s*""",
   "SQL Injection via %-formatting",
   Severity.HIGH, "high",
   "SQL query uses old-style % formatting. Values are inserted without escaping. "
   "Use parameterized queries with execute(sql, params).",
   ["python"],
   negative_patterns=[r"execute\s*\(.*,\s*\("])  # parameterized is safe

# Python — execute() with concatenation
_r("SQLI_PY_EXEC_CONCAT", CWECategory.SQL_INJECTION,
   r"""(?:execute|executemany)\s*\(\s*['"].*['"]\s*\+""",
   "SQL Injection in execute() with concatenation",
   Severity.HIGH, "high",
   "String concatenation inside cursor.execute() allows injection. "
   "Pass parameters as second argument: execute(sql, (param,)).",
   ["python"])

# Python — execute() with f-string
_r("SQLI_PY_EXEC_FSTRING", CWECategory.SQL_INJECTION,
   r"""(?:execute|executemany)\s*\(\s*f['"]""",
   "SQL Injection in execute() with f-string",
   Severity.CRITICAL, "high",
   "f-string inside execute() injects values directly into SQL. "
   "Use parameterized queries: execute('...%s...', (val,)).",
   ["python"])

# JavaScript — string concatenation in SQL query
_r("SQLI_JS_CONCAT", CWECategory.SQL_INJECTION,
   r"""(?:query|execute)\s*\(\s*['"`](?:SELECT|INSERT|UPDATE|DELETE)\s+.*['"`]\s*\+""",
   "SQL Injection via string concatenation",
   Severity.HIGH, "high",
   "SQL query built with string concatenation. Use parameterized queries "
   "or prepared statements: db.query('SELECT ... WHERE id=$1', [val]).",
   ["javascript"])

# JavaScript — template literal in SQL
_r("SQLI_JS_TEMPLATE", CWECategory.SQL_INJECTION,
   r"""(?:query|execute)\s*\(\s*`(?:SELECT|INSERT|UPDATE|DELETE)\s+.*\$\{""",
   "SQL Injection via template literal",
   Severity.HIGH, "high",
   "SQL query uses template literals with ${...} interpolation. "
   "An attacker controlling the interpolated variable can inject SQL.",
   ["javascript"])

# JavaScript — raw query builder (Sequelize/Knex) with template literal
_r("SQLI_JS_RAW", CWECategory.SQL_INJECTION,
   r"""(?:raw|rawQuery|sequelize\.query)\s*\(\s*['"`].*\$\{""",
   "SQL Injection in raw query",
   Severity.HIGH, "high",
   "Raw SQL query with interpolated variables. ORM raw() bypasses "
   "the query builder's automatic parameterization.",
   ["javascript"])

# JavaScript — sequelize.query / raw query with string concatenation
_r("SQLI_JS_RAW_CONCAT", CWECategory.SQL_INJECTION,
   r"""(?:sequelize\.query|rawQuery)\s*\(\s*['"`].*['"`]\s*\+""",
   "SQL Injection in sequelize.query() with concatenation",
   Severity.HIGH, "high",
   "ORM raw query with string concatenation. User input is injected "
   "directly into SQL. Use parameterized replacements instead.",
   ["javascript"])

# JavaScript — sequelize.query / raw query with multiline SQL concat
_r("SQLI_JS_RAW_CONCAT_ML", CWECategory.SQL_INJECTION,
   r"""(?:sequelize\.query|rawQuery)\s*\(\s*\n\s*['"`](?:SELECT|INSERT|UPDATE|DELETE)\s+.*['"`]\s*\+""",
   "SQL Injection in sequelize.query() with concatenation (multiline)",
   Severity.HIGH, "high",
   "ORM raw query with string concatenation across lines. User input is "
   "injected directly into SQL. Use parameterized replacements instead.",
   ["javascript"],
   multiline=True)

# JavaScript — standalone SQL string concatenation (catches multi-line ORM calls)
_r("SQLI_JS_SQL_STRING_CONCAT", CWECategory.SQL_INJECTION,
   r"""['"](?:SELECT|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+.*['"`]\s*\+\s*\w+""",
   "SQL string with concatenation",
   Severity.HIGH, "high",
   "SQL string being concatenated with a variable. If the variable contains "
   "user input, this enables SQL injection.",
   ["javascript"])

# JavaScript — knex.raw with concatenation
_r("SQLI_JS_KNEX_CONCAT", CWECategory.SQL_INJECTION,
   r"""\.raw\s*\(\s*['"`].*['"`]\s*\+""",
   "SQL Injection in knex.raw() with concatenation",
   Severity.HIGH, "high",
   "knex.raw() with string concatenation allows SQL injection. "
   "Use knex.raw('SELECT ... WHERE id=?', [val]) instead.",
   ["javascript"])


# ─── CWE-77: Command Injection ──────────────────────────────────────

# Python — os.system()
_r("CMDI_PY_SYSTEM", CWECategory.COMMAND_INJECTION,
   r"""os\.system\s*\(\s*(?:f['"]|\w+\s*\+|['"].*['"]\.format|['"].*%[sd])""",
   "Command Injection via os.system()",
   Severity.CRITICAL, "high",
   "os.system() executes a shell command. If user input reaches this call "
   "(via f-string, concatenation, or formatting), arbitrary OS commands run.",
   ["python"])

# Python — os.system() with any variable
_r("CMDI_PY_SYSTEM_VAR", CWECategory.COMMAND_INJECTION,
   r"""os\.system\s*\(\s*(?!['"])(\w+)""",
   "Command Injection via os.system() with variable",
   Severity.HIGH, "high",
   "os.system() called with a variable argument. If that variable contains "
   "unsanitized user input, it enables command injection.",
   ["python"])

# Python — os.popen()
_r("CMDI_PY_POPEN", CWECategory.COMMAND_INJECTION,
   r"""os\.popen\s*\(""",
   "Command Injection via os.popen()",
   Severity.HIGH, "high",
   "os.popen() runs a command in a subshell. Use subprocess with "
   "shell=False and a list of arguments instead.",
   ["python"])

# Python — subprocess with shell=True
_r("CMDI_PY_SUBPROCESS_SHELL", CWECategory.COMMAND_INJECTION,
   r"""subprocess\.(?:call|run|Popen|check_output|check_call)\s*\([^)]*shell\s*=\s*True""",
   "Command Injection via subprocess(shell=True)",
   Severity.CRITICAL, "high",
   "subprocess with shell=True passes the command through the system shell. "
   "An attacker injecting into the command string can execute arbitrary commands. "
   "Use shell=False with a list: subprocess.run(['cmd', arg]).",
   ["python"])

# Python — subprocess with f-string/concat argument
_r("CMDI_PY_SUBPROCESS_FMT", CWECategory.COMMAND_INJECTION,
   r"""subprocess\.(?:call|run|Popen|check_output|check_call)\s*\(\s*(?:f['"]|['"].*['"]\.format|['"].*['"]\s*\+)""",
   "Command Injection via subprocess with formatted string",
   Severity.HIGH, "high",
   "Command string for subprocess is built dynamically. Even without shell=True, "
   "the first element could be manipulated. Use a static command list.",
   ["python"])

# Python — eval() / exec()
_r("CMDI_PY_EVAL", CWECategory.COMMAND_INJECTION,
   r"""(?:eval|exec)\s*\(\s*(?!['"])""",
   "Code Injection via eval()/exec()",
   Severity.CRITICAL, "high",
   "eval()/exec() with non-literal argument executes arbitrary Python code. "
   "If user input reaches this, it's a full code execution vulnerability.",
   ["python"],
   negative_patterns=[r"eval\s*\(\s*['\"]"])  # Literal strings are less risky

# JavaScript — child_process exec
_r("CMDI_JS_EXEC", CWECategory.COMMAND_INJECTION,
   r"""(?:exec|execSync)\s*\(\s*(?:`.*\$\{|['"].*['"]\s*\+|\w+(?:\s*\+|\s*,))""",
   "Command Injection via child_process.exec()",
   Severity.CRITICAL, "high",
   "child_process.exec() runs a command in the system shell. "
   "Interpolated/concatenated input allows arbitrary command execution. "
   "Use execFile() or spawn() with argument arrays instead.",
   ["javascript"])

# JavaScript — child_process.exec with variable
_r("CMDI_JS_EXEC_VAR", CWECategory.COMMAND_INJECTION,
   r"""(?:exec|execSync)\s*\(\s*(?!['"`])(\w+)""",
   "Command Injection via exec() with variable",
   Severity.HIGH, "high",
   "exec() called with a variable that may contain user input. "
   "Use execFile() or spawn() with explicit argument arrays.",
   ["javascript"])

# JavaScript — Function constructor
_r("CMDI_JS_FUNCTION", CWECategory.COMMAND_INJECTION,
   r"""new\s+Function\s*\(""",
   "Code Injection via Function constructor",
   Severity.HIGH, "high",
   "new Function() compiles and executes string as code at runtime. "
   "Equivalent to eval(). Never use with user-controlled input.",
   ["javascript"])


# ─── CWE-22: Path Traversal ─────────────────────────────────────────

# Python — open() with f-string / concat / variable from request
_r("PATH_PY_OPEN_FSTR", CWECategory.PATH_TRAVERSAL,
   r"""open\s*\(\s*f['"]""",
   "Path Traversal via open() with f-string",
   Severity.HIGH, "high",
   "File opened with an f-string path. If a user-controlled variable is "
   "interpolated, an attacker can use ../../ to read arbitrary files.",
   ["python"])

_r("PATH_PY_OPEN_CONCAT", CWECategory.PATH_TRAVERSAL,
   r"""open\s*\(\s*\w+\s*\+""",
   "Path Traversal via open() with concatenation",
   Severity.HIGH, "high",
   "File path is built by concatenation. Unsanitized user input "
   "allows ../../etc/passwd-style traversal attacks.",
   ["python"])

# Python — os.path.join with user input (common Flask/Django pattern)
_r("PATH_PY_JOIN_REQ", CWECategory.PATH_TRAVERSAL,
   r"""os\.path\.join\s*\([^)]*(?:request\.|args\[|params\[|form\[|user_input|filename|file_name)""",
   "Path Traversal via os.path.join() with request data",
   Severity.HIGH, "high",
   "os.path.join() does NOT sanitize traversal sequences. "
   "An absolute path like /etc/passwd replaces the base entirely. "
   "Use os.path.realpath() and verify the result is within allowed directory.",
   ["python"],
   negative_patterns=[r"realpath[\s\S]*startswith", r"startswith[\s\S]*realpath"])

# Python — send_file / send_from_directory with variable
_r("PATH_PY_SEND_FILE", CWECategory.PATH_TRAVERSAL,
   r"""(?:send_file|send_from_directory)\s*\([^)]*(?:request\.|args\[|params\[|form\[|\w+_path|\w+_file)""",
   "Path Traversal in Flask send_file()",
   Severity.HIGH, "medium",
   "Flask send_file() with user-controlled path. Validate the resolved "
   "path is within the intended directory before sending.",
   ["python"])

# Python — pathlib with user input
_r("PATH_PY_PATHLIB", CWECategory.PATH_TRAVERSAL,
   r"""Path\s*\(\s*(?:f['"]|.*\+|request\.|args\[|form\[)""",
   "Path Traversal via pathlib.Path()",
   Severity.MEDIUM, "medium",
   "pathlib.Path() with user input. Resolve and check the path "
   "is within the allowed base directory.",
   ["python"])

# JavaScript — fs operations with user input
_r("PATH_JS_FS_READ", CWECategory.PATH_TRAVERSAL,
   r"""(?:readFile|readFileSync|createReadStream|readdir|readdirSync)\s*\(\s*(?:`.*\$\{|.*\+|req\.(?:params|query|body))""",
   "Path Traversal in fs read operation",
   Severity.HIGH, "high",
   "Node.js file read with user-controlled path. Use path.resolve() "
   "and verify the result starts with the expected base directory.",
   ["javascript"])

_r("PATH_JS_FS_WRITE", CWECategory.PATH_TRAVERSAL,
   r"""(?:writeFile|writeFileSync|createWriteStream|appendFile|appendFileSync)\s*\(\s*(?:`.*\$\{|.*\+|req\.(?:params|query|body))""",
   "Path Traversal in fs write operation",
   Severity.HIGH, "high",
   "Node.js file write with user-controlled path allows writing to "
   "arbitrary locations. Validate the resolved path strictly.",
   ["javascript"])

# JavaScript — path.join with request
_r("PATH_JS_JOIN_REQ", CWECategory.PATH_TRAVERSAL,
   r"""path\.(?:join|resolve)\s*\([^)]*req\.(?:params|query|body|files)""",
   "Path Traversal via path.join() with request data",
   Severity.HIGH, "high",
   "path.join() with user input. In Node.js, path.join('base', '../../../etc/passwd') "
   "resolves the traversal. Verify the final path is under the base directory.",
   ["javascript"],
   negative_patterns=[r"startsWith\s*\(", r"\.startsWith\s*\("])

# Python — open(str(pathlib_var), ...) with user-controlled pathlib path
_r("PATH_PY_OPEN_STR_VAR", CWECategory.PATH_TRAVERSAL,
   r"""open\s*\(\s*str\s*\(""",
   "Path Traversal via open(str(path_object))",
   Severity.MEDIUM, "medium",
   "File opened by converting a Path object to string. If the path was "
   "built with user-controlled components (e.g. base / user_filename), "
   "it may allow directory traversal.",
   ["python"])

# General — literal ../../ in strings
_r("PATH_LITERAL_TRAVERSAL", CWECategory.PATH_TRAVERSAL,
   r"""(?:\.\./){2,}""",
   "Suspicious path traversal literal",
   Severity.MEDIUM, "low",
   "Multiple ../ sequences detected. This may indicate a path traversal "
   "test case or an actual attack payload in the code.",
   ["python", "javascript"])


# ─── CWE-502: Insecure Deserialization ──────────────────────────────

# Python — pickle
_r("DESER_PY_PICKLE", CWECategory.INSECURE_DESERIALIZATION,
   r"""pickle\.(?:loads?|Unpickler)\s*\(""",
   "Insecure Deserialization via pickle",
   Severity.CRITICAL, "high",
   "pickle.load()/loads() deserializes Python objects and can execute "
   "arbitrary code during deserialization. An attacker sending a crafted "
   "pickle payload achieves Remote Code Execution.",
   ["python"],
   negative_patterns=[r"pickle\.loads?\s*\(\s*(?:b['\"]|open\s*\(\s*['\"])"])  # Loading known-safe files is lower risk

# Python — pickle with untrusted source (request, network)
_r("DESER_PY_PICKLE_REQ", CWECategory.INSECURE_DESERIALIZATION,
   r"""pickle\.(?:loads?|Unpickler)\s*\([^)]*(?:request\.|recv|read|data|body|payload|input)""",
   "Critical: pickle with untrusted input",
   Severity.CRITICAL, "high",
   "pickle deserialization of data from request/network. This is "
   "a textbook RCE vulnerability. Use JSON or a safe serialization format.",
   ["python"])

# Python — yaml.load without SafeLoader
_r("DESER_PY_YAML", CWECategory.INSECURE_DESERIALIZATION,
   r"""yaml\.(?:load|full_load)\s*\(""",
   "Insecure Deserialization via yaml.load()",
   Severity.HIGH, "high",
   "yaml.load() without Loader=SafeLoader can execute arbitrary Python "
   "objects via !!python/object tags. Always use yaml.safe_load().",
   ["python"],
   negative_patterns=[r"yaml\.load\s*\([^)]*(?:SafeLoader|safe_load|Loader\s*=\s*yaml\.SafeLoader)"])

# Python — marshal/shelve
_r("DESER_PY_MARSHAL", CWECategory.INSECURE_DESERIALIZATION,
   r"""(?:marshal\.loads?|shelve\.open)\s*\(""",
   "Insecure Deserialization via marshal/shelve",
   Severity.HIGH, "high",
   "marshal and shelve modules use unsafe deserialization internally. "
   "Do not use with untrusted data.",
   ["python"])

# JavaScript — node-serialize / unserialize
_r("DESER_JS_SERIALIZE", CWECategory.INSECURE_DESERIALIZATION,
   r"""(?:unserialize|deserialize)\s*\(""",
   "Insecure Deserialization in JavaScript",
   Severity.CRITICAL, "high",
   "The node-serialize unserialize() function can execute functions "
   "embedded in serialized objects (via _$$ND_FUNC$$_ payloads). "
   "This leads to Remote Code Execution.",
   ["javascript"])

# JavaScript — JSON.parse then eval-like usage
_r("DESER_JS_JSONPARSE_EVAL", CWECategory.INSECURE_DESERIALIZATION,
   r"""JSON\.parse\s*\([^)]+\)(?:\s*\.\s*\w+)*.*(?:eval|Function|setTimeout|setInterval)\s*\(""",
   "Deserialized data passed to eval-like function",
   Severity.CRITICAL, "high",
   "Data from JSON.parse is passed to eval/Function/setTimeout. "
   "An attacker controlling the JSON can achieve code execution.",
   ["javascript"],
   multiline=True)

# JavaScript — js-yaml without safe
_r("DESER_JS_YAML", CWECategory.INSECURE_DESERIALIZATION,
   r"""yaml\.load\s*\(""",
   "Insecure YAML deserialization",
   Severity.HIGH, "high",
   "yaml.load() in JavaScript (js-yaml <4.0) can execute arbitrary code "
   "via custom types. Use yaml.safeLoad() or update to js-yaml >= 4.0.",
   ["javascript"],
   negative_patterns=[r"yaml\.safeLoad"])


# ─── CWE-918: SSRF ──────────────────────────────────────────────────

# Python — requests with user input
_r("SSRF_PY_REQUESTS", CWECategory.SSRF,
   r"""requests\.(?:get|post|put|delete|patch|head|options)\s*\(\s*(?:f['"]|.*\+|request\.|args\[|form\[|url|target|dest)""",
   "SSRF via requests library",
   Severity.HIGH, "high",
   "HTTP request made with user-controlled URL. An attacker can redirect "
   "the request to internal services (169.254.169.254, localhost, etc.) "
   "to access metadata, internal APIs, or cloud credentials.",
   ["python"],
   negative_patterns=[r"(?:hostname|host|origin)\s+(?:not\s+)?in\s+(?:ALLOWED|allowed|whitelist|WHITELIST)",
                      r"urlparse.*(?:ALLOWED|allowed|whitelist)"])

# Python — urllib with user input
_r("SSRF_PY_URLLIB", CWECategory.SSRF,
   r"""urllib\.(?:request\.)?(?:urlopen|urlretrieve|Request)\s*\(\s*(?:f['"]|.*\+|request\.|url|target)""",
   "SSRF via urllib",
   Severity.HIGH, "high",
   "urllib request with user-controlled URL. Validate the URL scheme "
   "and host against an allowlist before making the request.",
   ["python"])

# Python — httpx/aiohttp with user input
_r("SSRF_PY_HTTPX", CWECategory.SSRF,
   r"""(?:httpx|aiohttp)\.(?:get|post|put|delete|AsyncClient|ClientSession)\s*\(\s*(?:f['"]|.*\+|url|target)""",
   "SSRF via httpx/aiohttp",
   Severity.HIGH, "high",
   "Async HTTP request with potentially user-controlled URL. "
   "Implement URL validation with scheme and host allowlists.",
   ["python"])

# Python — httpx client.get/post/put/delete with user-controlled url variable
_r("SSRF_PY_CLIENT_METHOD", CWECategory.SSRF,
   r"""(?:client|session)\.(?:get|post|put|delete|patch|head)\s*\(\s*(?:url|target|dest|endpoint|webhook_url)""",
   "SSRF via HTTP client method with URL variable",
   Severity.HIGH, "high",
   "HTTP client method called with a variable URL. If the URL originates "
   "from user input, it enables SSRF to internal services and cloud metadata.",
   ["python", "javascript"])

# JavaScript — fetch/axios/http with user input
_r("SSRF_JS_FETCH", CWECategory.SSRF,
   r"""(?:fetch|axios\.(?:get|post|put|delete|patch)|http\.(?:get|request)|got|superagent)\s*\(\s*(?:`.*\$\{|.*\+|req\.(?:params|query|body)|url|target)""",
   "SSRF via fetch/axios/http",
   Severity.HIGH, "high",
   "HTTP request with user-controlled URL. Validate and restrict the "
   "destination URL to prevent Server-Side Request Forgery.",
   ["javascript"],
   negative_patterns=[r"ALLOWED.*some", r"origin\s*===\s*"])

# JavaScript — URL redirect (can lead to SSRF)
_r("SSRF_JS_REDIRECT", CWECategory.SSRF,
   r"""(?:res\.redirect|window\.location|location\.href)\s*(?:=|\()\s*(?:req\.(?:params|query|body)|url|redirect_url|next_url)""",
   "Open Redirect / potential SSRF",
   Severity.MEDIUM, "medium",
   "Redirect with user-controlled URL. This can be used for phishing "
   "or SSRF if the server follows redirects internally.",
   ["javascript"])

# General — requests to internal IPs
_r("SSRF_INTERNAL_IP", CWECategory.SSRF,
   r"""['"`]https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0|169\.254\.169\.254|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|192\.168\.\d+\.\d+)""",
   "Request to internal/metadata IP address",
   Severity.HIGH, "high",
   "Hardcoded request to internal IP or cloud metadata endpoint. "
   "This may expose internal services or cloud credentials.",
   ["python", "javascript"])


# ─── CWE-798: Hardcoded Secrets ─────────────────────────────────────

# Generic — password/secret/token/api_key assignment
_r("SECRET_HARDCODED_PW", CWECategory.HARDCODED_SECRETS,
   r"""(?:password|passwd|pwd|secret|api_key|apikey|api_secret|auth_token|access_token|private_key|secret_key)\s*=\s*['"][^'"]{4,}['"]""",
   "Hardcoded secret/password",
   Severity.HIGH, "high",
   "A credential or secret is hardcoded in source code. This will be "
   "exposed in version control and build artifacts. Use environment "
   "variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault).",
   ["python", "javascript"],
   negative_patterns=[
       r"""(?:password|secret|api_key)\s*=\s*['"](?:\*+|xxx+|changeme|your[_-]|example|placeholder|REPLACE|TODO|<)""",
       r"""(?:password|secret|api_key)\s*=\s*os\.(?:environ|getenv)""",
       r"""(?:password|secret|api_key)\s*=\s*process\.env""",
   ])

# Generic — high-entropy string assignment (base64, hex, JWT-like)
_r("SECRET_HIGH_ENTROPY", CWECategory.HARDCODED_SECRETS,
   r"""(?:TOKEN|KEY|SECRET|PRIVATE_KEY|JWT_SECRET|SIGNING_KEY|ENCRYPTION_KEY|DATABASE_URL|CONNECTION_STRING)\s*=\s*['"][A-Za-z0-9+/=_\-]{20,}['"]""",
   "Hardcoded high-entropy secret",
   Severity.HIGH, "high",
   "Long high-entropy string assigned to a secret-sounding variable. "
   "This is very likely a real credential. Move to environment variables.",
   ["python", "javascript"],
   negative_patterns=[
       r"""os\.(?:environ|getenv)""",
       r"""process\.env""",
       r"""(?:changeme|your[_-]|example|placeholder|REPLACE|TODO|<|xxx)""",
   ])

# AWS credentials
_r("SECRET_AWS_KEY", CWECategory.HARDCODED_SECRETS,
   r"""(?:AKIA|ASIA)[A-Z0-9]{16}""",
   "Hardcoded AWS Access Key ID",
   Severity.CRITICAL, "high",
   "AWS access key ID detected in source code. This gives direct "
   "access to AWS resources. Rotate immediately and use IAM roles.",
   ["python", "javascript"])

# Generic API key patterns
_r("SECRET_GENERIC_KEY", CWECategory.HARDCODED_SECRETS,
   r"""['"](?:sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}|glpat-[a-zA-Z0-9\-]{20,}|xox[bpars]-[a-zA-Z0-9\-]+)['"]""",
   "Hardcoded API key (OpenAI/GitHub/GitLab/Slack)",
   Severity.CRITICAL, "high",
   "Known API key pattern detected. This key provides direct access "
   "to third-party services. Rotate and move to environment variables.",
   ["python", "javascript"])

# Connection strings with passwords
_r("SECRET_CONN_STRING", CWECategory.HARDCODED_SECRETS,
   r"""['"](?:postgres|mysql|mongodb|redis|amqp)://\w+:[^@'"]+@""",
   "Database connection string with embedded password",
   Severity.HIGH, "high",
   "Database connection string contains clear-text credentials. "
   "Store in environment variables or a secrets manager.",
   ["python", "javascript"])

# Private key material
_r("SECRET_PRIVATE_KEY", CWECategory.HARDCODED_SECRETS,
   r"""-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----""",
   "Embedded private key",
   Severity.CRITICAL, "high",
   "Private key material is embedded directly in source code. "
   "Private keys must be stored in secure key management systems, "
   "never committed to version control.",
   ["python", "javascript"])


# =====================================================================
# P A T T E R N   M A T C H I N G   E N G I N E
# =====================================================================

class PatternMatchingEngine:
    """
    Phase 1: High-recall pattern matching engine.
    
    Scans code line-by-line against all registered rules.
    Produces VULN_CANDIDATE findings for each match.
    
    Usage:
        engine = PatternMatchingEngine()
        findings = engine.scan(code, language="python", filename="app.py")
    """

    def __init__(self, custom_rules: Optional[List[PatternRule]] = None):
        """
        Args:
            custom_rules: Additional rules to register beyond defaults.
        """
        self.rules = list(RULES)
        if custom_rules:
            self.rules.extend(custom_rules)
        
        # Pre-compile regex patterns for performance
        self._compiled: List[Tuple[PatternRule, re.Pattern]] = []
        for rule in self.rules:
            flags = re.IGNORECASE
            if rule.multiline:
                flags |= re.DOTALL
            try:
                compiled = re.compile(rule.pattern, flags)
                self._compiled.append((rule, compiled))
            except re.error as e:
                print(f"[PatternEngine] WARNING: Invalid regex in rule {rule.rule_id}: {e}")
        
        # Compile negative patterns
        self._neg_compiled: Dict[str, List[re.Pattern]] = {}
        for rule in self.rules:
            if rule.negative_patterns:
                negs = []
                for np_str in rule.negative_patterns:
                    try:
                        negs.append(re.compile(np_str, re.IGNORECASE))
                    except re.error:
                        pass
                self._neg_compiled[rule.rule_id] = negs

    def scan(
        self,
        code: str,
        language: str = "python",
        filename: str = "input.py",
    ) -> List[PatternFinding]:
        """
        Scan source code against all rules for the given language.
        
        Args:
            code: Full source code string
            language: "python", "javascript", or "typescript"
            filename: Filename for reporting
            
        Returns:
            List of PatternFinding objects (VULN_CANDIDATE)
        """
        # Normalize language
        lang = language.lower().strip()
        if lang in ("ts", "typescript"):
            lang = "javascript"  # TS should have been transpiled already
        elif lang in ("js",):
            lang = "javascript"
        elif lang in ("py",):
            lang = "python"
        
        lines = code.split("\n")
        findings: List[PatternFinding] = []
        seen_rules_per_line: Dict[Tuple[int, str], bool] = {}  # Dedup (line, rule_id)

        for rule, compiled in self._compiled:
            # Skip rules not applicable to this language
            if lang not in rule.languages:
                continue
            
            if rule.multiline:
                # Multiline rules match across the entire code
                for m in compiled.finditer(code):
                    line_num = code[:m.start()].count("\n") + 1
                    key = (line_num, rule.rule_id)
                    if key in seen_rules_per_line:
                        continue
                    
                    snippet = self._get_snippet(lines, line_num)
                    wide = self._get_wide_context(lines, line_num)
                    
                    # Check negative patterns
                    neg_matched = self._check_negatives(rule.rule_id, snippet, wide)
                    if neg_matched:
                        continue
                    
                    seen_rules_per_line[key] = True
                    findings.append(PatternFinding(
                        file=filename,
                        line=line_num,
                        end_line=min(line_num + 2, len(lines)),
                        cwe=rule.cwe,
                        rule_id=rule.rule_id,
                        confidence=rule.confidence,
                        severity=rule.severity,
                        code_snippet=snippet,
                        message=rule.title,
                        explanation=rule.explanation,
                        label=VulnLabel.VULN_CANDIDATE,
                        language=lang,
                    ))
            else:
                # Line-by-line matching
                for i, line_text in enumerate(lines, 1):
                    if compiled.search(line_text):
                        key = (i, rule.rule_id)
                        if key in seen_rules_per_line:
                            continue
                        
                        snippet = self._get_snippet(lines, i)
                        wide = self._get_wide_context(lines, i)
                        
                        # Check negative patterns on the context
                        neg_matched = self._check_negatives(rule.rule_id, snippet, wide)
                        if neg_matched:
                            continue
                        
                        seen_rules_per_line[key] = True
                        findings.append(PatternFinding(
                            file=filename,
                            line=i,
                            end_line=min(i + 2, len(lines)),
                            cwe=rule.cwe,
                            rule_id=rule.rule_id,
                            confidence=rule.confidence,
                            severity=rule.severity,
                            code_snippet=snippet,
                            message=rule.title,
                            explanation=rule.explanation,
                            label=VulnLabel.VULN_CANDIDATE,
                            language=lang,
                        ))

        # Sort by line number, then severity
        sev_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2,
                     Severity.LOW: 3, Severity.INFO: 4}
        findings.sort(key=lambda f: (f.line, sev_order.get(f.severity, 5)))
        
        return findings

    def get_rule_count(self) -> Dict[str, int]:
        """Get count of rules per CWE."""
        counts: Dict[str, int] = {}
        for rule in self.rules:
            key = rule.cwe.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _get_snippet(self, lines: List[str], line_num: int, context: int = 1) -> str:
        """Get code snippet with context lines."""
        start = max(0, line_num - 1 - context)
        end = min(len(lines), line_num + context)
        return "\n".join(lines[start:end])

    def _get_wide_context(self, lines: List[str], line_num: int,
                          context: int = 10) -> str:
        """Get wider code context for negative-pattern checks."""
        start = max(0, line_num - 1 - context)
        end = min(len(lines), line_num + context)
        return "\n".join(lines[start:end])

    def _check_negatives(self, rule_id: str, text: str,
                         wide_text: str | None = None) -> bool:
        """Check if any negative pattern matches (cancels the finding).
        
        Checks both the narrow snippet and an optional wider context
        so that validation logic a few lines away can cancel a match.
        """
        negs = self._neg_compiled.get(rule_id, [])
        combined = text if wide_text is None else text + "\n" + wide_text
        for neg in negs:
            if neg.search(combined):
                return True
        return False
