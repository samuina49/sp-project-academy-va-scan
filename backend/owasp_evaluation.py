#!/usr/bin/env python3
"""
OWASP Top 10 Evaluation for Hybrid Vulnerability Scanner
==========================================================
Generates realistic vulnerable code samples for all OWASP Top 10 categories,
runs them through the Hybrid Pipeline (Pattern Matching + AI), and produces
poster-ready evaluation results.

HONESTY POLICY:
    - The scanner covers 6 CWEs: SQL Injection (CWE-89), Command Injection (CWE-77),
      Path Traversal (CWE-22), Deserialization (CWE-502), SSRF (CWE-918),
      Hardcoded Secrets (CWE-798).
    - Categories outside this scope are expected to have 0% detection.
    - We report both strengths AND limitations honestly.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Data model for test samples
# ---------------------------------------------------------------------------

@dataclass
class OWASPSample:
    sample_id: str
    owasp_id: str          # e.g. "A03"
    owasp_name: str        # e.g. "Injection"
    sub_type: str          # e.g. "SQL Injection"
    cwe: str               # e.g. "CWE-89"
    language: str          # python | javascript
    code: str
    description: str       # What's wrong
    fix_description: str   # How to fix
    fix_code: str          # Corrected version


@dataclass
class DetectionResult:
    sample: OWASPSample
    detected: bool
    rules_matched: List[str]   # Rule IDs that fired
    engine: str                # "Pattern" | "AI" | "Pattern + AI" | "Not Detected"
    confidence: str            # "high" | "medium" | "low" | "N/A"
    ai_score: float            # 0.0 if AI not used
    ai_available: bool
    verdict: str               # "VULNERABLE" | "LIKELY_VULN" | "NOT_DETECTED"


# =====================================================================
#  OWASP TOP 10 TEST SAMPLES
# =====================================================================

SAMPLES: List[OWASPSample] = []

def _s(sample_id, owasp_id, owasp_name, sub_type, cwe, language, code,
       description, fix_description, fix_code):
    SAMPLES.append(OWASPSample(
        sample_id=sample_id, owasp_id=owasp_id, owasp_name=owasp_name,
        sub_type=sub_type, cwe=cwe, language=language, code=code,
        description=description, fix_description=fix_description,
        fix_code=fix_code,
    ))


# ─────────────────────────────────────────────────────────────────────
# A01: Broken Access Control
# ─────────────────────────────────────────────────────────────────────

_s("A01-PY-01", "A01", "Broken Access Control", "Path Traversal",
   "CWE-22", "python",
   '''
from flask import Flask, request, send_file
import os

app = Flask(__name__)

@app.route("/download")
def download():
    filename = request.args.get("file")
    filepath = os.path.join("/var/uploads", filename)
    return send_file(filepath)
''',
   "User-controlled filename passed directly to os.path.join without validation. "
   "Attacker can use ../../etc/passwd to traverse directories.",
   "Validate path stays within allowed directory using os.path.realpath() and startswith().",
   '''
from flask import Flask, request, send_file, abort
import os

UPLOAD_DIR = os.path.realpath("/var/uploads")

@app.route("/download")
def download():
    filename = request.args.get("file")
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(UPLOAD_DIR):
        abort(403)
    return send_file(filepath)
''')

_s("A01-JS-01", "A01", "Broken Access Control", "Path Traversal",
   "CWE-22", "javascript",
   '''
const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();

app.get('/api/files/:name', (req, res) => {
    const filePath = path.join(__dirname, 'uploads', req.params.name);
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) return res.status(404).send('Not found');
        res.send(data);
    });
});
''',
   "User-controlled filename in path.join allows directory traversal. "
   "Attacker sends ../../../etc/passwd as file name.",
   "Resolve full path and verify it starts with the intended base directory.",
   '''
app.get('/api/files/:name', (req, res) => {
    const basePath = path.resolve(__dirname, 'uploads');
    const filePath = path.resolve(basePath, req.params.name);
    if (!filePath.startsWith(basePath)) {
        return res.status(403).send('Forbidden');
    }
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) return res.status(404).send('Not found');
        res.send(data);
    });
});
''')

_s("A01-PY-02", "A01", "Broken Access Control", "IDOR (Insecure Direct Object Reference)",
   "CWE-639", "python",
   '''
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/user/<int:user_id>/profile")
def get_profile(user_id):
    # No authorization check — any logged-in user can view any profile
    user = db.query("SELECT * FROM users WHERE id = %s", (user_id,))
    return jsonify(user)
''',
   "No access control check — any authenticated user can view any other user's profile "
   "by changing the user_id parameter. This is an Insecure Direct Object Reference.",
   "Verify that the current session user has permission to access the requested resource.",
   '''
@app.route("/api/user/<int:user_id>/profile")
@login_required
def get_profile(user_id):
    if current_user.id != user_id and not current_user.is_admin:
        abort(403)
    user = db.query("SELECT * FROM users WHERE id = %s", (user_id,))
    return jsonify(user)
''')

_s("A01-JS-02", "A01", "Broken Access Control", "Missing Function-Level Access Control",
   "CWE-862", "javascript",
   '''
const express = require('express');
const app = express();

// Admin endpoint — no auth middleware!
app.delete('/api/admin/users/:id', async (req, res) => {
    await User.findByIdAndDelete(req.params.id);
    res.json({ message: 'User deleted' });
});
''',
   "Admin endpoint has no authentication or authorization middleware. "
   "Anyone can delete any user by calling this endpoint.",
   "Add authentication middleware and role-based access control.",
   '''
app.delete('/api/admin/users/:id', authenticate, requireRole('admin'), async (req, res) => {
    await User.findByIdAndDelete(req.params.id);
    res.json({ message: 'User deleted' });
});
''')

_s("A01-PY-03", "A01", "Broken Access Control", "Path Traversal via open()",
   "CWE-22", "python",
   '''
from django.http import FileResponse
from pathlib import Path

def download_report(request):
    report_name = request.GET.get("name", "report.pdf")
    base = Path("/srv/reports")
    file_path = base / report_name
    return FileResponse(open(str(file_path), "rb"))
''',
   "User-controlled filename combined with pathlib division operator. "
   "Attacker can use ../../../etc/shadow to traverse directories.",
   "Validate the resolved path is within the intended directory.",
   '''
def download_report(request):
    report_name = request.GET.get("name", "report.pdf")
    base = Path("/srv/reports").resolve()
    file_path = (base / report_name).resolve()
    if not str(file_path).startswith(str(base)):
        return HttpResponseForbidden()
    return FileResponse(open(str(file_path), "rb"))
''')


# ─────────────────────────────────────────────────────────────────────
# A02: Cryptographic Failures
# ─────────────────────────────────────────────────────────────────────

_s("A02-PY-01", "A02", "Cryptographic Failures", "Hardcoded API Key",
   "CWE-798", "python",
   '''
import requests

API_KEY = "sk-proj-abc123def456ghi789jkl012mno345pqr678stu"
DATABASE_URL = "postgres://admin:SuperSecret123@prod-db.company.com:5432/production"

def call_external_api(data):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    return requests.post("https://api.example.com/process", json=data, headers=headers)
''',
   "API key and database credentials are hardcoded in source code. "
   "They will be exposed in version control and build artifacts.",
   "Use environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault).",
   '''
import os
import requests

API_KEY = os.environ["API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

def call_external_api(data):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    return requests.post("https://api.example.com/process", json=data, headers=headers)
''')

_s("A02-JS-01", "A02", "Cryptographic Failures", "Hardcoded JWT Secret",
   "CWE-798", "javascript",
   '''
const jwt = require('jsonwebtoken');

const JWT_SECRET = "my-super-secret-jwt-signing-key-2024";

function generateToken(user) {
    return jwt.sign({ id: user.id, role: user.role }, JWT_SECRET, { expiresIn: '24h' });
}

function verifyToken(token) {
    return jwt.verify(token, JWT_SECRET);
}
''',
   "JWT signing secret is hardcoded in source code. An attacker with source access "
   "can forge valid tokens for any user, including admin accounts.",
   "Store JWT secret in environment variables and rotate periodically.",
   '''
const jwt = require('jsonwebtoken');

const JWT_SECRET = process.env.JWT_SECRET;

function generateToken(user) {
    return jwt.sign({ id: user.id, role: user.role }, JWT_SECRET, { expiresIn: '24h' });
}
''')

_s("A02-PY-02", "A02", "Cryptographic Failures", "Hardcoded AWS Credentials",
   "CWE-798", "python",
   '''
import boto3

AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

s3 = boto3.client('s3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def upload_file(file_path, bucket):
    s3.upload_file(file_path, bucket, file_path.split("/")[-1])
''',
   "AWS credentials hardcoded directly in source code. "
   "Attacker gains full access to AWS resources.",
   "Use IAM roles, environment variables, or AWS credential provider chain.",
   '''
import boto3

# Uses IAM role or ~/.aws/credentials automatically
s3 = boto3.client('s3')

def upload_file(file_path, bucket):
    s3.upload_file(file_path, bucket, file_path.split("/")[-1])
''')

_s("A02-JS-02", "A02", "Cryptographic Failures", "Weak Hashing (MD5)",
   "CWE-327", "javascript",
   '''
const crypto = require('crypto');

function hashPassword(password) {
    return crypto.createHash('md5').update(password).digest('hex');
}

app.post('/register', (req, res) => {
    const hashedPw = hashPassword(req.body.password);
    db.query('INSERT INTO users (email, password) VALUES (?, ?)',
        [req.body.email, hashedPw]);
    res.json({ message: 'User created' });
});
''',
   "MD5 is cryptographically broken and should never be used for password hashing. "
   "Attackers can crack MD5 hashes using rainbow tables in seconds.",
   "Use bcrypt, scrypt, or argon2 for password hashing.",
   '''
const bcrypt = require('bcrypt');

app.post('/register', async (req, res) => {
    const hashedPw = await bcrypt.hash(req.body.password, 12);
    db.query('INSERT INTO users (email, password) VALUES (?, ?)',
        [req.body.email, hashedPw]);
    res.json({ message: 'User created' });
});
''')

_s("A02-PY-03", "A02", "Cryptographic Failures", "Hardcoded Database Password",
   "CWE-798", "python",
   '''
import psycopg2

DB_PASSWORD = "Pr0duction_S3cret!2024"

def get_connection():
    return psycopg2.connect(
        host="prod-db.internal",
        user="admin",
        password=DB_PASSWORD,
        dbname="production"
    )
''',
   "Database password is hardcoded in source code. This exposes credentials "
   "in version control, CI/CD logs, and build artifacts.",
   "Use environment variables or a secrets manager.",
   '''
import os, psycopg2

def get_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        dbname=os.environ["DB_NAME"]
    )
''')


# ─────────────────────────────────────────────────────────────────────
# A03: Injection
# ─────────────────────────────────────────────────────────────────────

_s("A03-PY-01", "A03", "Injection", "SQL Injection (f-string)",
   "CWE-89", "python",
   '''
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route("/api/search")
def search():
    query = request.args.get("q")
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM products WHERE name LIKE '%{query}%'")
    return {"results": cursor.fetchall()}
''',
   "User input is interpolated directly into SQL using f-string. "
   "Attacker can inject: ' OR '1'='1 to dump entire table, or use UNION SELECT to extract other tables.",
   "Use parameterized queries with placeholders.",
   '''
@app.route("/api/search")
def search():
    query = request.args.get("q")
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM products WHERE name LIKE ?", (f"%{query}%",))
    return {"results": cursor.fetchall()}
''')

_s("A03-PY-02", "A03", "Injection", "SQL Injection (string concatenation)",
   "CWE-89", "python",
   '''
from flask import Flask, request
import psycopg2

@app.route("/api/user")
def get_user():
    user_id = request.args.get("id")
    conn = psycopg2.connect(dbname="myapp")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = " + user_id)
    return {"user": cur.fetchone()}
''',
   "SQL query built with string concatenation. Attacker input like '1 OR 1=1' "
   "bypasses filters and returns all records.",
   "Use parameterized queries.",
   '''
cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
''')

_s("A03-JS-01", "A03", "Injection", "SQL Injection (template literal)",
   "CWE-89", "javascript",
   '''
const express = require('express');
const mysql = require('mysql2');
const app = express();
const db = mysql.createPool({ host: 'localhost', user: 'root', database: 'shop' });

app.get('/api/products', (req, res) => {
    const category = req.query.category;
    db.query(`SELECT * FROM products WHERE category = '${category}'`, (err, rows) => {
        res.json(rows);
    });
});
''',
   "Template literal interpolation in SQL query. Attacker can break out of the string "
   "and inject arbitrary SQL commands.",
   "Use query parameters (prepared statements).",
   '''
db.query('SELECT * FROM products WHERE category = ?', [category], (err, rows) => {
    res.json(rows);
});
''')

_s("A03-JS-02", "A03", "Injection", "SQL Injection (Sequelize raw)",
   "CWE-89", "javascript",
   '''
const { Sequelize } = require('sequelize');
const sequelize = new Sequelize('sqlite::memory:');

async function findUser(username) {
    const results = await sequelize.query(
        "SELECT * FROM users WHERE username = '" + username + "'",
        { type: Sequelize.QueryTypes.SELECT }
    );
    return results;
}
''',
   "ORM raw query with string concatenation. Bypasses Sequelize's built-in "
   "parameterization and allows SQL injection.",
   "Use Sequelize replacements for parameterized queries.",
   '''
const results = await sequelize.query(
    "SELECT * FROM users WHERE username = :username",
    { replacements: { username }, type: Sequelize.QueryTypes.SELECT }
);
''')

_s("A03-PY-03", "A03", "Injection", "Command Injection (os.system)",
   "CWE-77", "python",
   '''
from flask import Flask, request
import os

app = Flask(__name__)

@app.route("/api/ping")
def ping():
    host = request.args.get("host")
    result = os.system(f"ping -c 4 {host}")
    return {"status": "ok", "exit_code": result}
''',
   "User input passed directly to os.system(). Attacker sends 'google.com; cat /etc/passwd' "
   "to execute arbitrary commands on the server.",
   "Use subprocess.run() with a list of arguments (no shell).",
   '''
import subprocess

@app.route("/api/ping")
def ping():
    host = request.args.get("host")
    result = subprocess.run(["ping", "-c", "4", host], capture_output=True, text=True)
    return {"status": "ok", "output": result.stdout}
''')

_s("A03-JS-03", "A03", "Injection", "Command Injection (exec)",
   "CWE-77", "javascript",
   '''
const express = require('express');
const { exec } = require('child_process');
const app = express();

app.get('/api/lookup', (req, res) => {
    const domain = req.query.domain;
    exec('nslookup ' + domain, (error, stdout, stderr) => {
        res.json({ result: stdout });
    });
});
''',
   "User-controlled string concatenated into shell command. "
   "Attacker can inject '; rm -rf /' or '&& cat /etc/paswd'.",
   "Use execFile() with argument arrays — never pass user input to exec().",
   '''
const { execFile } = require('child_process');

app.get('/api/lookup', (req, res) => {
    execFile('nslookup', [req.query.domain], (error, stdout, stderr) => {
        res.json({ result: stdout });
    });
});
''')

_s("A03-PY-04", "A03", "Injection", "Command Injection (subprocess shell=True)",
   "CWE-77", "python",
   '''
import subprocess
from flask import request

@app.route("/api/convert")
def convert_file():
    filename = request.args.get("file")
    subprocess.run(f"convert {filename} output.pdf", shell=True)
    return {"message": "Converted"}
''',
   "shell=True with user input enables command injection through shell metacharacters.",
   "Use subprocess.run() with shell=False and a list of arguments.",
   '''
subprocess.run(["convert", filename, "output.pdf"], shell=False)
''')

_s("A03-JS-04", "A03", "Injection", "Command Injection (eval)",
   "CWE-77", "javascript",
   '''
app.post('/api/calculate', (req, res) => {
    const expression = req.body.expression;
    const result = eval(expression);
    res.json({ result });
});
''',
   "eval() executes arbitrary JavaScript code. Attacker can send "
   "'require(\"child_process\").exec(\"rm -rf /\")' as the expression.",
   "Use a safe math parser library instead of eval().",
   '''
const { evaluate } = require('mathjs');

app.post('/api/calculate', (req, res) => {
    const result = evaluate(req.body.expression);
    res.json({ result });
});
''')


# ─────────────────────────────────────────────────────────────────────
# A04: Insecure Design
# ─────────────────────────────────────────────────────────────────────

_s("A04-PY-01", "A04", "Insecure Design", "No Rate Limiting on Login",
   "CWE-307", "python",
   '''
from flask import Flask, request, jsonify

@app.route("/api/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    user = authenticate(username, password)
    if user:
        return jsonify({"token": generate_token(user)})
    return jsonify({"error": "Invalid credentials"}), 401
''',
   "Login endpoint has no rate limiting, account lockout, or brute-force protection. "
   "Attacker can attempt unlimited password combinations.",
   "Implement rate limiting, account lockout, and CAPTCHA after failed attempts.",
   '''
from flask_limiter import Limiter

limiter = Limiter(app, key_func=get_remote_address)

@app.route("/api/login", methods=["POST"])
@limiter.limit("5 per minute")
def login():
    # ... same logic + account lockout after 5 failures
''')

_s("A04-JS-01", "A04", "Insecure Design", "Predictable Password Reset Token",
   "CWE-330", "javascript",
   '''
app.post('/api/forgot-password', (req, res) => {
    const email = req.body.email;
    const resetToken = Date.now().toString(36);
    saveResetToken(email, resetToken);
    sendEmail(email, `Reset: /reset?token=${resetToken}`);
    res.json({ message: 'Reset email sent' });
});
''',
   "Password reset token is derived from Date.now() which is predictable. "
   "Attacker can guess valid tokens by trying timestamps around the request time.",
   "Use cryptographically secure random tokens.",
   '''
const crypto = require('crypto');
const resetToken = crypto.randomBytes(32).toString('hex');
''')

_s("A04-PY-02", "A04", "Insecure Design", "No Input Validation on Transaction",
   "CWE-20", "python",
   '''
@app.route("/api/transfer", methods=["POST"])
def transfer():
    amount = request.json.get("amount")
    to_account = request.json.get("to_account")
    # No validation on amount — negative values steal money
    db.execute("UPDATE accounts SET balance = balance - %s WHERE id = %s", (amount, current_user.id))
    db.execute("UPDATE accounts SET balance = balance + %s WHERE id = %s", (amount, to_account))
    return jsonify({"success": True})
''',
   "No validation on transaction amount. Negative amounts reverse the transfer direction, "
   "effectively stealing money from the target account.",
   "Validate input: amount must be positive, within limits, and sender has sufficient balance.",
   '''
amount = float(request.json.get("amount"))
if amount <= 0 or amount > 10000:
    return jsonify({"error": "Invalid amount"}), 400
''')


# ─────────────────────────────────────────────────────────────────────
# A05: Security Misconfiguration
# ─────────────────────────────────────────────────────────────────────

_s("A05-PY-01", "A05", "Security Misconfiguration", "Debug Mode in Production",
   "CWE-489", "python",
   '''
from flask import Flask

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "dev-secret"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
''',
   "Flask debug mode exposes interactive debugger in production. "
   "Attacker can execute arbitrary Python code via the Werkzeug debugger.",
   "Disable debug mode in production. Use environment-specific configs.",
   '''
import os
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
''')

_s("A05-JS-01", "A05", "Security Misconfiguration", "CORS Wildcard + Detailed Errors",
   "CWE-942", "javascript",
   '''
const express = require('express');
const cors = require('cors');
const app = express();

app.use(cors({ origin: '*', credentials: true }));

app.use((err, req, res, next) => {
    res.status(500).json({
        error: err.message,
        stack: err.stack,
        query: req.query,
        headers: req.headers
    });
});
''',
   "CORS wildcard allows any website to make authenticated requests. "
   "Detailed error handler exposes stack traces and request data to attackers.",
   "Restrict CORS to trusted origins. Use generic error messages in production.",
   '''
app.use(cors({ origin: ['https://yourdomain.com'], credentials: true }));

app.use((err, req, res, next) => {
    console.error(err);
    res.status(500).json({ error: 'Internal server error' });
});
''')

_s("A05-PY-02", "A05", "Security Misconfiguration", "Hardcoded Secret Key",
   "CWE-798", "python",
   '''
SECRET_KEY = "super-secret-flask-session-key-never-change"
JWT_SECRET = "my-jwt-secret-for-signing-tokens-2024xx"

app.config["SECRET_KEY"] = SECRET_KEY
''',
   "Flask SECRET_KEY and JWT signing key are hardcoded. Session cookies can be forged "
   "and JWT tokens can be created by anyone with source access.",
   "Load secrets from environment variables.",
   '''
import os
app.config["SECRET_KEY"] = os.environ["SECRET_KEY"]
''')


# ─────────────────────────────────────────────────────────────────────
# A06: Vulnerable & Outdated Components
# ─────────────────────────────────────────────────────────────────────

_s("A06-PY-01", "A06", "Vulnerable & Outdated Components", "Known Vulnerable Library Usage",
   "CWE-1035", "python",
   '''
# requirements.txt:
# flask==0.12.2  (CVE-2019-1010083: denial of service)
# pyyaml==3.12   (CVE-2017-18342: arbitrary code execution via yaml.load)
# requests==2.5.0 (CVE-2018-18074: cleartext credentials in redirects)

import yaml
config = yaml.load(open("config.yml"))  # Unsafe yaml.load
''',
   "Project uses outdated libraries with known CVEs. yaml.load() without SafeLoader "
   "allows arbitrary code execution through crafted YAML files.",
   "Update dependencies regularly. Use yaml.safe_load(). Run pip-audit.",
   '''
import yaml
config = yaml.safe_load(open("config.yml"))
# Also: pip install --upgrade flask pyyaml requests
''')

_s("A06-JS-01", "A06", "Vulnerable & Outdated Components", "Outdated NPM Packages",
   "CWE-1035", "javascript",
   '''
// package.json (excerpt):
// "lodash": "4.17.4"     — CVE-2019-10744: Prototype Pollution
// "express": "3.0.0"     — Multiple CVEs
// "serialize-javascript": "1.7.0" — CVE-2019-16769: XSS

const _ = require('lodash');

app.post('/api/config', (req, res) => {
    const defaults = { admin: false, theme: 'light' };
    const config = _.merge(defaults, req.body);
    res.json(config);
});
''',
   "lodash.merge() with user input enables Prototype Pollution. "
   "Attacker sends {\"__proto__\": {\"admin\": true}} to escalate privileges.",
   "Update all dependencies. Use npm audit. Validate user input against a schema.",
   '''
const config = { ...defaults, theme: req.body.theme };
// Also: npm audit fix, update lodash to latest
''')


# ─────────────────────────────────────────────────────────────────────
# A07: Identification & Authentication Failures
# ─────────────────────────────────────────────────────────────────────

_s("A07-PY-01", "A07", "Identification & Authentication Failures", "Hardcoded Admin Credentials",
   "CWE-798", "python",
   '''
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123!@#SecurePassword"

@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    if (request.json["username"] == ADMIN_USERNAME and
        request.json["password"] == ADMIN_PASSWORD):
        return jsonify({"token": create_admin_token()})
    return jsonify({"error": "Invalid"}), 401
''',
   "Admin credentials hardcoded in source code. Anyone with repository access "
   "can authenticate as admin.",
   "Store credentials securely. Use password hashing and proper authentication flow.",
   '''
import bcrypt, os
# Password stored as bcrypt hash in database
# ADMIN_PASSWORD_HASH = bcrypt.hashpw(b"...", bcrypt.gensalt())
''')

_s("A07-JS-01", "A07", "Identification & Authentication Failures", "JWT None Algorithm",
   "CWE-347", "javascript",
   '''
const jwt = require('jsonwebtoken');

function verifyToken(token) {
    // VULNERABLE: algorithms not restricted
    return jwt.verify(token, SECRET_KEY);
}

app.get('/api/dashboard', (req, res) => {
    const decoded = verifyToken(req.headers.authorization?.split(' ')[1]);
    const user = getUserById(decoded.id);
    res.json(user);
});
''',
   "JWT verification without specifying allowed algorithms. Attacker can forge tokens "
   "using 'alg: none' or switch from RS256 to HS256 using the public key.",
   "Always specify allowed algorithms explicitly.",
   '''
function verifyToken(token) {
    return jwt.verify(token, SECRET_KEY, { algorithms: ['HS256'] });
}
''')

_s("A07-PY-02", "A07", "Identification & Authentication Failures", "Plaintext Password Storage",
   "CWE-256", "python",
   '''
import sqlite3

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    conn.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, password)  # Storing plaintext password!
    )
    conn.commit()
''',
   "Passwords stored in plaintext in the database. A database breach exposes "
   "all user credentials directly.",
   "Hash passwords with bcrypt before storing.",
   '''
import bcrypt

def register_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
        (username, hashed))
''')


# ─────────────────────────────────────────────────────────────────────
# A08: Software & Data Integrity Failures
# ─────────────────────────────────────────────────────────────────────

_s("A08-PY-01", "A08", "Software & Data Integrity Failures", "Insecure Deserialization (pickle)",
   "CWE-502", "python",
   '''
import pickle
from flask import Flask, request

app = Flask(__name__)

@app.route("/api/import", methods=["POST"])
def import_data():
    data = pickle.loads(request.data)
    process(data)
    return {"status": "imported"}
''',
   "pickle.loads() on untrusted input allows arbitrary code execution. "
   "Attacker sends crafted pickle payload that executes os.system().",
   "Never unpickle untrusted data. Use JSON or a safe serialization format.",
   '''
import json

@app.route("/api/import", methods=["POST"])
def import_data():
    data = json.loads(request.data)
    process(data)
    return {"status": "imported"}
''')

_s("A08-PY-02", "A08", "Software & Data Integrity Failures", "Insecure Deserialization (yaml.load)",
   "CWE-502", "python",
   '''
import yaml
from flask import request

@app.route("/api/config", methods=["POST"])
def update_config():
    config = yaml.load(request.data)
    apply_config(config)
    return {"status": "updated"}
''',
   "yaml.load() without SafeLoader can execute arbitrary Python code "
   "through YAML tags like !!python/object/apply:os.system.",
   "Use yaml.safe_load() instead of yaml.load().",
   '''
config = yaml.safe_load(request.data)
''')

_s("A08-JS-01", "A08", "Software & Data Integrity Failures", "Insecure Deserialization (node-serialize)",
   "CWE-502", "javascript",
   '''
const serialize = require('node-serialize');

app.post('/api/session', (req, res) => {
    const sessionData = serialize.unserialize(
        Buffer.from(req.cookies.session, 'base64').toString()
    );
    res.json(sessionData);
});
''',
   "node-serialize.unserialize() executes JavaScript functions embedded in the serialized data. "
   "Attacker can achieve Remote Code Execution via crafted session cookie.",
   "Use JSON.parse() for deserialization. Never trust serialized data from clients.",
   '''
const sessionData = JSON.parse(
    Buffer.from(req.cookies.session, 'base64').toString()
);
''')

_s("A08-PY-03", "A08", "Software & Data Integrity Failures", "Insecure Deserialization (shelve)",
   "CWE-502", "python",
   '''
import shelve

def load_user_preferences(user_id):
    filename = f"prefs/{user_id}"
    db = shelve.open(filename)
    prefs = dict(db)
    db.close()
    return prefs
''',
   "shelve uses pickle internally and can execute arbitrary code "
   "when opening files with crafted content.",
   "Use JSON files or a database for storing user preferences.",
   '''
import json

def load_user_preferences(user_id):
    with open(f"prefs/{user_id}.json") as f:
        return json.load(f)
''')


# ─────────────────────────────────────────────────────────────────────
# A09: Security Logging & Monitoring Failures
# ─────────────────────────────────────────────────────────────────────

_s("A09-PY-01", "A09", "Security Logging & Monitoring Failures", "Logging Sensitive Data",
   "CWE-532", "python",
   '''
import logging

logger = logging.getLogger(__name__)

@app.route("/api/login", methods=["POST"])
def login():
    username = request.json.get("username")
    password = request.json.get("password")
    logger.info(f"Login attempt: user={username} password={password}")
    user = authenticate(username, password)
    if not user:
        logger.warning(f"Failed login for {username} with password {password}")
    return jsonify({"success": bool(user)})
''',
   "Passwords logged in plaintext. Log files often have weaker access controls "
   "than databases, exposing credentials to anyone with log access.",
   "Never log sensitive data. Log only events, not values.",
   '''
logger.info(f"Login attempt: user={username}")
if not user:
    logger.warning(f"Failed login for {username} from IP {request.remote_addr}")
''')

_s("A09-JS-01", "A09", "Security Logging & Monitoring Failures", "No Audit Trail",
   "CWE-778", "javascript",
   '''
app.delete('/api/admin/users/:id', authenticate, requireAdmin, async (req, res) => {
    await User.findByIdAndDelete(req.params.id);
    // No logging of who deleted which user
    res.json({ message: 'User deleted' });
});
''',
   "Critical admin actions have no audit trail. If an account is compromised, "
   "there's no way to trace what actions were performed.",
   "Log all security-relevant events with actor, action, target, and timestamp.",
   '''
app.delete('/api/admin/users/:id', authenticate, requireAdmin, async (req, res) => {
    await User.findByIdAndDelete(req.params.id);
    auditLog.info({ actor: req.user.id, action: 'DELETE_USER',
                    target: req.params.id, timestamp: new Date() });
    res.json({ message: 'User deleted' });
});
''')


# ─────────────────────────────────────────────────────────────────────
# A10: Server-Side Request Forgery (SSRF)
# ─────────────────────────────────────────────────────────────────────

_s("A10-PY-01", "A10", "Server-Side Request Forgery", "SSRF via requests",
   "CWE-918", "python",
   '''
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route("/api/preview")
def preview():
    url = request.args.get("url")
    response = requests.get(url)
    return {"content": response.text[:500], "status": response.status_code}
''',
   "Server fetches user-provided URL without validation. Attacker sends "
   "http://169.254.169.254/latest/meta-data/ to access AWS instance metadata.",
   "Validate URL against an allowlist of hosts and schemes.",
   '''
from urllib.parse import urlparse

ALLOWED_HOSTS = {"api.example.com", "cdn.example.com"}

@app.route("/api/preview")
def preview():
    url = request.args.get("url")
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS:
        return {"error": "Host not allowed"}, 403
    response = requests.get(url)
    return {"content": response.text[:500]}
''')

_s("A10-JS-01", "A10", "Server-Side Request Forgery", "SSRF via fetch",
   "CWE-918", "javascript",
   '''
app.get('/api/proxy', async (req, res) => {
    const url = req.query.url;
    const response = await fetch(url);
    const html = await response.text();
    res.send(html);
});
''',
   "Server proxies arbitrary URLs from user input. Attacker can scan internal "
   "network, access cloud metadata, or hit internal admin panels.",
   "Validate and restrict target URLs. Block internal/private IP ranges.",
   '''
const { URL } = require('url');
const ALLOWED = ['https://api.example.com'];

app.get('/api/proxy', async (req, res) => {
    const target = new URL(req.query.url);
    if (!ALLOWED.some(a => target.origin === new URL(a).origin)) {
        return res.status(403).json({ error: 'Host not allowed' });
    }
    const response = await fetch(target.href);
    res.send(await response.text());
});
''')

_s("A10-PY-02", "A10", "Server-Side Request Forgery", "SSRF via httpx AsyncClient",
   "CWE-918", "python",
   '''
import httpx
from fastapi import FastAPI

app = FastAPI()

@app.get("/webhook")
async def trigger_webhook(url: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"event": "test"})
    return {"status_code": resp.status_code}
''',
   "Webhook URL is user-controlled with no validation. Attacker uses this to "
   "make the server send POST requests to internal services.",
   "Validate webhook URLs against an allowlist. Block RFC 1918 ranges.",
   '''
ALLOWED_WEBHOOK_HOSTS = {"hooks.slack.com", "hooks.example.com"}

@app.get("/webhook")
async def trigger_webhook(url: str):
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_WEBHOOK_HOSTS:
        raise HTTPException(403, "Webhook host not allowed")
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"event": "test"})
    return {"status_code": resp.status_code}
''')

_s("A10-PY-03", "A10", "Server-Side Request Forgery", "SSRF to cloud metadata",
   "CWE-918", "python",
   '''
import requests

def fetch_external_resource(target_url):
    """Fetches content from the given URL."""
    response = requests.get(target_url)
    return response.json()

@app.route("/api/fetch")
def fetch():
    url = request.args.get("url")
    return fetch_external_resource(url)
''',
   "User-controlled URL passed to requests.get(). Attacker accesses "
   "cloud metadata at http://169.254.169.254 to steal IAM credentials.",
   "Validate URL hostname and scheme. Block private IP ranges.",
   '''
import ipaddress
from urllib.parse import urlparse

def is_safe_url(url):
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        return ip.is_global
    except ValueError:
        return parsed.hostname in ALLOWED_HOSTS
''')

_s("A10-JS-02", "A10", "Server-Side Request Forgery", "SSRF via axios",
   "CWE-918", "javascript",
   '''
const axios = require('axios');

app.post('/api/import-data', async (req, res) => {
    const url = req.body.data_source_url;
    const response = await axios.get(url);
    await saveImportedData(response.data);
    res.json({ imported: response.data.length });
});
''',
   "Server fetches from user-specified data source URL. "
   "Attacker can exfil data to external servers or access internal APIs.",
   "Validate URL against allowlist before making requests.",
   '''
const allowedHosts = new Set(['data.example.com', 'api.partner.com']);
const parsedUrl = new URL(req.body.data_source_url);
if (!allowedHosts.has(parsedUrl.hostname)) {
    return res.status(403).json({ error: 'Host not allowed' });
}
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A01 (adding 3 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A01-JS-03", "A01", "Broken Access Control", "Path Traversal in Express",
   "CWE-22", "javascript",
   '''
const express = require('express');
const fs = require('fs');
const app = express();

app.get('/files', (req, res) => {
    const filename = req.query.file;
    const content = fs.readFileSync('/var/data/' + filename, 'utf8');
    res.send(content);
});
''',
   "Direct concatenation of user input with filesystem path enables directory traversal.",
   "Validate and sanitize input, use path.resolve() with whitelist check.",
   '''
const path = require('path');
const BASE = path.resolve('/var/data');
const filePath = path.resolve(BASE, req.query.file);
if (!filePath.startsWith(BASE)) return res.status(403).send('Forbidden');
''')

_s("A01-PY-04", "A01", "Broken Access Control", "Path Traversal in Django",
   "CWE-22", "python",
   '''
from django.http import HttpResponse
import os

def serve_file(request):
    filename = request.GET.get('file', '')
    full_path = f"/var/www/uploads/{filename}"
    with open(full_path, 'r') as f:
        return HttpResponse(f.read())
''',
   "User-supplied filename concatenated into path without validation.",
   "Use os.path.abspath() and verify it starts with the safe directory.",
   '''
import os
BASE_DIR = os.path.abspath("/var/www/uploads")
full_path = os.path.abspath(os.path.join(BASE_DIR, filename))
if not full_path.startswith(BASE_DIR):
    return HttpResponseForbidden()
''')

_s("A01-JS-04", "A01", "Broken Access Control", "File Read Without Validation",
   "CWE-22", "javascript",
   '''
const fs = require('fs').promises;

async function getDocument(docName) {
    const path = `./documents/${docName}`;
    return await fs.readFile(path, 'utf8');
}
''',
   "Template literal with user-controlled variable can be exploited with ../",
   "Validate filename against allowed characters and use path.join with basedir check.",
   '''
const path = require('path');
const BASE = path.resolve('./documents');
const fullPath = path.resolve(BASE, docName);
if (!fullPath.startsWith(BASE)) throw new Error('Invalid path');
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A02 (adding 3 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A02-JS-03", "A02", "Cryptographic Failures", "Hardcoded Database Password",
   "CWE-798", "javascript",
   '''
const mongoose = require('mongoose');

const DB_PASSWORD = "Prod_Mongo_P@ssw0rd_2024";
const connectionString = `mongodb://admin:${DB_PASSWORD}@prod.server.com:27017/myapp`;

mongoose.connect(connectionString);
''',
   "MongoDB password hardcoded in source code. Exposed in version control and build logs.",
   "Use environment variables for sensitive credentials.",
   '''
const DB_PASSWORD = process.env.MONGO_PASSWORD;
const connectionString = `mongodb://admin:${DB_PASSWORD}@prod.server.com:27017/myapp`;
''')

_s("A02-PY-04", "A02", "Cryptographic Failures", "Hardcoded Encryption Key",
   "CWE-798", "python",
   '''
from cryptography.fernet import Fernet

ENCRYPTION_KEY = b"aGVsbG93b3JsZGhlbGxvd29ybGRoZWxsbw=="

def encrypt_data(plaintext):
    cipher = Fernet(ENCRYPTION_KEY)
    return cipher.encrypt(plaintext.encode())
''',
   "Encryption key hardcoded as bytes literal. Anyone with source access can decrypt all data.",
   "Generate key at deployment time and store in secure key management system.",
   '''
import os
ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"].encode()
''')

_s("A02-JS-04", "A02", "Cryptographic Failures", "Hardcoded Private Key",
   "CWE-798", "javascript",
   '''
const crypto = require('crypto');

const PRIVATE_KEY = `-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA2dGhPGLmY8jVXh9k3pM...
-----END RSA PRIVATE KEY-----`;

function signToken(data) {
    return crypto.sign('sha256', Buffer.from(data), PRIVATE_KEY);
}
''',
   "RSA private key hardcoded in source. Attacker with code access can forge signatures.",
   "Load private key from secure storage or environment variable.",
   '''
const fs = require('fs');
const PRIVATE_KEY = fs.readFileSync(process.env.PRIVATE_KEY_PATH, 'utf8');
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A03 (adding 2 more → total 10)
# ─────────────────────────────────────────────────────────────────────

_s("A03-PY-05", "A03", "Injection", "SQL Injection in Django ORM raw",
   "CWE-89", "python",
   '''
from django.db import connection

def search_users(keyword):
    with connection.cursor() as cursor:
        query = "SELECT * FROM users WHERE username = '%s'" % keyword
        cursor.execute(query)
        return cursor.fetchall()
''',
   "String formatting in SQL query enables injection attacks.",
   "Use parameterized queries with %s placeholders.",
   '''
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, [keyword])
''')

_s("A03-JS-05", "A03", "Injection", "NoSQL Injection in MongoDB",
   "CWE-89", "javascript",
   '''
const { MongoClient } = require('mongodb');

async function findUser(username) {
    const query = { username: username };
    // If username = {"$gt": ""} it returns all users
    return await db.collection('users').findOne(query);
}
''',
   "Directly passing user input to MongoDB query enables NoSQL injection.",
   "Validate input type and sanitize against operator injection.",
   '''
if (typeof username !== 'string') throw new Error('Invalid input');
const query = { username: String(username) };
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A04 (adding 5 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A04-JS-02", "A04", "Insecure Design", "No CAPTCHA on Registration",
   "CWE-841", "javascript",
   '''
app.post('/api/register', async (req, res) => {
    const { email, password } = req.body;
    await User.create({ email, password });
    res.json({ message: 'Account created' });
});
''',
   "Registration endpoint has no CAPTCHA or rate limiting. Bots can create unlimited accounts.",
   "Implement CAPTCHA (reCAPTCHA) and rate limiting per IP.",
   '''
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 5 });
app.post('/api/register', limiter, verifyCaptcha, async (req, res) => { ... });
''')

_s("A04-PY-03", "A04", "Insecure Design", "Sequential Transaction IDs",
   "CWE-330", "python",
   '''
transaction_counter = 1000

@app.route("/api/payment", methods=["POST"])
def process_payment():
    global transaction_counter
    transaction_counter += 1
    txn_id = f"TXN{transaction_counter}"
    return jsonify({"transaction_id": txn_id})
''',
   "Predictable transaction IDs allow attackers to enumerate and guess other transactions.",
   "Use cryptographically secure random IDs (UUID4 or secure random bytes).",
   '''
import uuid
txn_id = str(uuid.uuid4())
''')

_s("A04-JS-03", "A04", "Insecure Design", "No Email Verification",
   "CWE-20", "javascript",
   '''
app.post('/signup', async (req, res) => {
    const user = new User({ email: req.body.email, verified: false });
    await user.save();
    res.json({ message: 'Signed up' });
});
''',
   "User accounts created without email verification. Attacker can register with others' emails.",
   "Send verification email with token before enabling account.",
   '''
sendVerificationEmail(user.email, user.verificationToken);
''')

_s("A04-PY-04", "A04", "Insecure Design", "No Secondary Authorization for Sensitive Action",
   "CWE-306", "python",
   '''
@app.route("/api/delete-account", methods=["POST"])
@login_required
def delete_account():
    db.delete_user(current_user.id)
    return jsonify({"message": "Account deleted"})
''',
   "Critical action (account deletion) requires no secondary confirmation or password re-entry.",
   "Require password confirmation or 2FA for sensitive operations.",
   '''
password = request.json.get("password")
if not verify_password(current_user, password):
    return jsonify({"error": "Invalid password"}), 401
''')

_s("A04-JS-04", "A04", "Insecure Design", "No Balance Check Before Debit",
   "CWE-20", "javascript",
   '''
app.post('/api/withdraw', async (req, res) => {
    const amount = req.body.amount;
    await Account.updateOne({ _id: req.user.accountId }, {
        $inc: { balance: -amount }
    });
    res.json({ message: 'Withdrawn' });
});
''',
   "No validation that balance is sufficient before debit. Negative balances allowed.",
   "Check balance before transaction and use atomic operations.",
   '''
const account = await Account.findById(req.user.accountId);
if (account.balance < amount) return res.status(400).json({ error: 'Insufficient funds' });
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A05 (adding 5 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A05-PY-03", "A05", "Security Misconfiguration", "Hardcoded Admin Token",
   "CWE-798", "python",
   '''
ADMIN_API_TOKEN = "admin_token_x7f9k2m5p8q1w3"

@app.route("/api/admin/stats")
def admin_stats():
    token = request.headers.get("Authorization")
    if token == ADMIN_API_TOKEN:
        return jsonify(get_system_stats())
''',
   "Admin API token hardcoded. Any developer can access admin endpoints.",
   "Use environment variable and rotate tokens regularly.",
   '''
ADMIN_API_TOKEN = os.environ["ADMIN_TOKEN"]
''')

_s("A05-JS-02", "A05", "Security Misconfiguration", "Exposed Stack Traces",
   "CWE-209", "javascript",
   '''
app.use((err, req, res, next) => {
    res.status(500).json({
        error: err.message,
        stack: err.stack,
        file: err.fileName
    });
});
''',
   "Error handler exposes full stack traces to client. Reveals internal paths and logic.",
   "Log errors server-side, return generic messages to client.",
   '''
logger.error(err);
res.status(500).json({ error: 'Internal server error' });
''')

_s("A05-PY-04", "A05", "Security Misconfiguration", "Hardcoded Session Secret",
   "CWE-798", "python",
   '''
from flask import Flask, session

app = Flask(__name__)
app.secret_key = "my-super-secret-session-key-do-not-share"

@app.route("/login", methods=["POST"])
def login():
    session["user_id"] = 123
    return "Logged in"
''',
   "Flask session secret is hardcoded. Attacker can forge session cookies.",
   "Load secret from environment variable.",
   '''
app.secret_key = os.environ["SESSION_SECRET"]
''')

_s("A05-JS-03", "A05", "Security Misconfiguration", "Default Credentials",
   "CWE-798", "javascript",
   '''
const defaultAdminPassword = "admin123";

async function setupInitialAdmin() {
    if (!await User.findOne({ role: 'admin' })) {
        await User.create({
            username: 'admin',
            password: defaultAdminPassword,
            role: 'admin'
        });
    }
}
''',
   "Default admin account created with predictable password.",
   "Force password change on first login or generate random password.",
   '''
const crypto = require('crypto');
const randomPassword = crypto.randomBytes(16).toString('hex');
sendPasswordEmail(randomPassword);
''')

_s("A05-PY-05", "A05", "Security Misconfiguration", "Hardcoded OAuth Client Secret",
   "CWE-798", "python",
   '''
OAUTH_CLIENT_SECRET = "oauth_secret_abc123xyz789"

def exchange_code_for_token(code):
    response = requests.post("https://oauth.provider.com/token", data={
        "code": code,
        "client_secret": OAUTH_CLIENT_SECRET
    })
    return response.json()
''',
   "OAuth client secret hardcoded. Compromised secret allows token forgery.",
   "Store in environment variable or secrets manager.",
   '''
OAUTH_CLIENT_SECRET = os.environ["OAUTH_CLIENT_SECRET"]
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A06 (adding 6 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A06-PY-02", "A06", "Vulnerable & Outdated Components", "Vulnerable Pillow Library",
   "CWE-1035", "python",
   '''
# requirements.txt:
# Pillow==6.0.0  (CVE-2020-5312: buffer overflow)

from PIL import Image

def process_upload(file_path):
    img = Image.open(file_path)
    img.thumbnail((200, 200))
    img.save("output.jpg")
''',
   "Using Pillow 6.0.0 which has known buffer overflow vulnerability.",
   "Update to latest Pillow version (>=9.0.0).",
   '''
# requirements.txt:
# Pillow>=10.0.0
''')

_s("A06-JS-02", "A06", "Vulnerable & Outdated Components", "Vulnerable jsonwebtoken",
   "CWE-1035", "javascript",
   '''
// package.json: "jsonwebtoken": "8.1.0" (CVE-2022-23529)

const jwt = require('jsonwebtoken');

function verify(token) {
    return jwt.verify(token, publicKey, { algorithms: ['RS256'] });
}
''',
   "jsonwebtoken 8.1.0 has vulnerability allowing secret key exposure.",
   "Update to jsonwebtoken@9.0.0 or later.",
   '''
// package.json: "jsonwebtoken": "^9.0.0"
''')

_s("A06-PY-03", "A06", "Vulnerable & Outdated Components", "Outdated Django",
   "CWE-1035", "python",
   '''
# requirements.txt:
# Django==2.2.0  (Multiple CVEs including SQL injection bypass)

from django.shortcuts import render

def view(request):
    return render(request, "page.html")
''',
   "Django 2.2.0 has multiple known security vulnerabilities.",
   "Update to Django 4.2 LTS or later.",
   '''
# Django>=4.2
''')

_s("A06-JS-03", "A06", "Vulnerable & Outdated Components", "Vulnerable axios",
   "CWE-1035", "javascript",
   '''
// package.json: "axios": "0.21.1" (CVE-2021-3749: ReDoS)

const axios = require('axios');

async function fetchData(url) {
    return await axios.get(url);
}
''',
   "axios 0.21.1 vulnerable to Regular Expression Denial of Service.",
   "Update to axios@1.6.0 or later.",
   '''
// "axios": "^1.6.0"
''')

_s("A06-PY-04", "A06", "Vulnerable & Outdated Components", "Unsafe XML Parser",
   "CWE-611", "python",
   '''
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    root = ET.fromstring(xml_string)
    return root.findall(".//data")
''',
   "Default XML parser vulnerable to XXE (XML External Entity) attacks.",
   "Use defusedxml library to safely parse XML.",
   '''
from defusedxml import ElementTree as ET
root = ET.fromstring(xml_string)
''')

_s("A06-JS-04", "A06", "Vulnerable & Outdated Components", "Vulnerable socket.io",
   "CWE-1035", "javascript",
   '''
// package.json: "socket.io": "2.3.0" (CVE-2020-36048)

const io = require('socket.io')(server);

io.on('connection', (socket) => {
    console.log('Client connected');
});
''',
   "socket.io 2.3.0 has CORS bypass vulnerability.",
   "Update to socket.io@4.0.0 or later.",
   '''
// "socket.io": "^4.6.0"
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A07 (adding 5 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A07-PY-03", "A07", "Identification & Authentication Failures", "Hardcoded TOTP Secret",
   "CWE-798", "python",
   '''
import pyotp

TOTP_SECRET = "JBSWY3DPEHPK3PXP"

def verify_2fa(user_code):
    totp = pyotp.TOTP(TOTP_SECRET)
    return totp.verify(user_code)
''',
   "TOTP secret hardcoded and shared across all users.",
   "Generate unique TOTP secret per user and store in database.",
   '''
totp = pyotp.TOTP(user.totp_secret)
''')

_s("A07-JS-02", "A07", "Identification & Authentication Failures", "Weak Session ID",
   "CWE-330", "javascript",
   '''
let sessionCounter = 1000;

function createSession(userId) {
    sessionCounter++;
    return { id: `sess_${sessionCounter}`, userId };
}
''',
   "Predictable sequential session IDs enable session hijacking.",
   "Use cryptographically secure random session identifiers.",
   '''
const crypto = require('crypto');
const sessionId = crypto.randomBytes(32).toString('hex');
''')

_s("A07-PY-04", "A07", "Identification & Authentication Failures", "No Account Lockout",
   "CWE-307", "python",
   '''
@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    user = User.query.filter_by(username=username).first()
    if user and verify_password(user, password):
        return jsonify({"token": create_token(user)})
    return jsonify({"error": "Invalid credentials"}), 401
''',
   "No failed attempt tracking or account lockout. Brute force attacks possible.",
   "Track failed attempts, implement exponential backoff and CAPTCHA.",
   '''
if user.failed_attempts >= 5:
    if user.locked_until > datetime.now():
        return jsonify({"error": "Account locked"}), 429
''')

_s("A07-JS-03", "A07", "Identification & Authentication Failures", "Hardcoded Cookie Secret",
   "CWE-798", "javascript",
   '''
const session = require('express-session');

app.use(session({
    secret: 'keyboard-cat-session-secret-2024',
    resave: false,
    saveUninitialized: true
}));
''',
   "Session cookie secret is hardcoded. Session cookies can be forged.",
   "Load secret from environment variable.",
   '''
secret: process.env.SESSION_SECRET
''')

_s("A07-PY-05", "A07", "Identification & Authentication Failures", "Hardcoded API Master Key",
   "CWE-798", "python",
   '''
API_MASTER_KEY = "mk_live_a1b2c3d4e5f6g7h8i9j0"

def validate_api_key(provided_key):
    return provided_key == API_MASTER_KEY
''',
   "Master API key hardcoded in source code. Compromised key grants full system access.",
   "Use environment variable and implement key rotation policy.",
   '''
API_MASTER_KEY = os.environ["API_MASTER_KEY"]
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A08 (adding 4 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A08-JS-02", "A08", "Software & Data Integrity Failures", "Deserialization via JSON.parse with __proto__",
   "CWE-502", "javascript",
   '''
app.post('/api/config', (req, res) => {
    const config = JSON.parse(req.body.config);
    Object.assign(appConfig, config);  // Prototype pollution
    res.json({ message: 'Config updated' });
});
''',
   "JSON.parse with Object.assign enables prototype pollution attacks.",
   "Use JSON schema validation and avoid Object.assign with untrusted data.",
   '''
const Ajv = require('ajv');
const ajv = new Ajv();
const valid = ajv.validate(configSchema, config);
if (!valid) return res.status(400).json({ error: 'Invalid config' });
''')

_s("A08-PY-04", "A08", "Software & Data Integrity Failures", "marshal.loads on Untrusted Data",
   "CWE-502", "python",
   '''
import marshal

def load_cache(cache_data):
    obj = marshal.loads(cache_data)
    return obj
''',
   "marshal.loads() can execute arbitrary code like pickle.",
   "Use JSON or MessagePack for serialization.",
   '''
import json
obj = json.loads(cache_data)
''')

_s("A08-JS-03", "A08", "Software & Data Integrity Failures", "vm.runInNewContext Injection",
   "CWE-94", "javascript",
   '''
const vm = require('vm');

function evaluateExpression(expr) {
    return vm.runInNewContext(expr);
}
''',
   "vm.runInNewContext with user input enables code injection.",
   "Never execute user-provided code. Use safe expression parser.",
   '''
const math = require('mathjs');
return math.evaluate(expr);
''')

_s("A08-PY-05", "A08", "Software & Data Integrity Failures", "exec() on User Input",
   "CWE-94", "python",
   '''
def run_formula(formula):
    result = {}
    exec(f"result['value'] = {formula}", {"result": result})
    return result["value"]
''',
   "exec() with user-controlled string enables arbitrary Python code execution.",
   "Use ast.literal_eval() for safe evaluation or expression parser.",
   '''
import ast
return ast.literal_eval(formula)
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A09 (adding 6 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A09-PY-02", "A09", "Security Logging & Monitoring Failures", "No Logging for Failed Auth",
   "CWE-778", "python",
   '''
@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    if not authenticate(username, password):
        return jsonify({"error": "Invalid"}), 401
    # No logging of failed attempt
    return jsonify({"token": create_token(username)})
''',
   "Failed authentication attempts not logged. Cannot detect brute force attacks.",
   "Log all authentication events with timestamp, IP, and outcome.",
   '''
logger.warning(f"Failed login attempt for {username} from {request.remote_addr}")
''')

_s("A09-JS-02", "A09", "Security Logging & Monitoring Failures", "Logging Sensitive Headers",
   "CWE-532", "javascript",
   '''
app.use((req, res, next) => {
    logger.info('Request received', {
        path: req.path,
        headers: req.headers,  // Contains Authorization tokens!
        body: req.body
    });
    next();
});
''',
   "Logging full headers exposes Authorization tokens and sensitive data.",
   "Filter sensitive headers before logging.",
   '''
const safeHeaders = { ...req.headers };
delete safeHeaders.authorization;
delete safeHeaders.cookie;
logger.info('Request', { path: req.path, headers: safeHeaders });
''')

_s("A09-PY-03", "A09", "Security Logging & Monitoring Failures", "No Monitoring for Privilege Escalation",
   "CWE-778", "python",
   '''
@app.route("/api/promote", methods=["POST"])
@admin_required
def promote_user():
    user_id = request.json["user_id"]
    db.update_user(user_id, role="admin")
    # No audit log
    return jsonify({"message": "User promoted"})
''',
   "Privilege escalation not logged or monitored. Security incidents go undetected.",
   "Log all role changes with actor, target, and timestamp.",
   '''
audit_logger.critical(f"User {current_user.id} promoted user {user_id} to admin")
''')

_s("A09-JS-03", "A09", "Security Logging & Monitoring Failures", "PII in Error Logs",
   "CWE-532", "javascript",
   '''
app.post('/api/update-profile', async (req, res) => {
    try {
        await User.updateOne({ _id: req.user.id }, req.body);
        res.json({ message: 'Updated' });
    } catch (error) {
        logger.error('Update failed', { user: req.user, body: req.body, error });
        res.status(500).send('Error');
    }
});
''',
   "Error logs contain full user object and request body with PII.",
   "Log only error message and request ID, not sensitive data.",
   '''
logger.error('Update failed', { userId: req.user.id, error: error.message });
''')

_s("A09-PY-04", "A09", "Security Logging & Monitoring Failures", "No Tamper Detection",
   "CWE-778", "python",
   '''
@app.route("/api/data/<int:record_id>", methods=["PUT"])
def update_record(record_id):
    data = request.json
    db.update(record_id, data)
    return jsonify({"message": "Updated"})
''',
   "Data modifications not logged. No audit trail for tampering detection.",
   "Log before and after values for all data changes.",
   '''
old_value = db.get(record_id)
db.update(record_id, data)
audit_log(user=current_user, action="UPDATE", record=record_id, before=old_value, after=data)
''')

_s("A09-JS-04", "A09", "Security Logging & Monitoring Failures", "Insufficient Security Event Logging",
   "CWE-778", "javascript",
   '''
app.post('/api/admin/delete-all-data', authenticate, requireAdmin, async (req, res) => {
    await Database.dropAllCollections();
    res.json({ message: 'All data deleted' });
});
''',
   "Critical destructive action with no security event logging or alerting.",
   "Log critical actions and trigger immediate alerts.",
   '''
securityLogger.critical({ action: 'DROP_ALL_DATA', actor: req.user.id, timestamp: new Date() });
sendAlert('CRITICAL: All data deleted', req.user);
''')

# ─────────────────────────────────────────────────────────────────────
# EXPANDED SAMPLES - A10 (adding 3 more → total 8)
# ─────────────────────────────────────────────────────────────────────

_s("A10-PY-04", "A10", "Server-Side Request Forgery", "SSRF in urllib",
   "CWE-918", "python",
   '''
from urllib.request import urlopen

def fetch_url(url):
    response = urlopen(url)
    return response.read()
''',
   "urllib.request.urlopen with user-controlled URL enables SSRF.",
   "Validate URL against allowlist and block private IP ranges.",
   '''
from urllib.parse import urlparse
parsed = urlparse(url)
if parsed.hostname not in ALLOWED_HOSTS:
    raise ValueError("Host not allowed")
''')

_s("A10-JS-03", "A10", "Server-Side Request Forgery", "SSRF in http.request",
   "CWE-918", "javascript",
   '''
const http = require('http');

function proxyRequest(targetUrl) {
    return new Promise((resolve) => {
        http.get(targetUrl, (res) => {
            let data = '';
            res.on('data', chunk => data += chunk);
            res.on('end', () => resolve(data));
        });
    });
}
''',
   "http.get() with user URL enables internal network scanning.",
   "Parse URL and validate hostname against allowlist.",
   '''
const { URL } = require('url');
const parsed = new URL(targetUrl);
if (!ALLOWED_HOSTS.includes(parsed.hostname)) throw new Error('Forbidden');
''')

_s("A10-PY-05", "A10", "Server-Side Request Forgery", "SSRF via Webhook URL",
   "CWE-918", "python",
   '''
import requests

@app.route("/api/webhook/test", methods=["POST"])
def test_webhook():
    webhook_url = request.json["url"]
    resp = requests.post(webhook_url, json={"event": "test"})
    return jsonify({"status_code": resp.status_code})
''',
   "Webhook testing endpoint allows arbitrary POST requests to any URL.",
   "Validate webhook URLs against strict allowlist.",
   '''
from urllib.parse import urlparse
parsed = urlparse(webhook_url)
if parsed.hostname not in ALLOWED_WEBHOOK_HOSTS:
    return jsonify({"error": "Invalid webhook host"}), 403
''')


# =====================================================================
#  EVALUATION ENGINE
# =====================================================================

def run_evaluation():
    """Run all samples through the hybrid pipeline and collect results."""
    from app.hybrid_scanner.pipeline import HybridPipeline
    from app.hybrid_scanner.models import Verdict

    print("=" * 80)
    print("OWASP TOP 10 — HYBRID SCANNER EVALUATION")
    print("=" * 80)
    print()

    pipeline = HybridPipeline(ai_enabled=True, threshold=0.7)

    results: List[DetectionResult] = []

    for sample in SAMPLES:
        result = pipeline.scan_code(
            code=sample.code,
            language=sample.language,
            filename=f"{sample.sample_id}.{'py' if sample.language == 'python' else 'js'}",
        )

        detected = result.confirmed_vulns > 0
        rules = []
        engine = "Not Detected"
        confidence = "N/A"
        ai_score = 0.0
        ai_available = result.ai_available
        verdict = "NOT_DETECTED"

        if detected:
            for rf in result.refined_findings:
                if rf.verdict in (Verdict.VULNERABLE, Verdict.LIKELY_VULNERABLE):
                    pf = rf.pattern_finding
                    rules.append(pf.rule_id)
                    ai_score = max(ai_score, rf.ai_score)
                    confidence = pf.confidence

            if ai_score > 0 and any(
                rf.verdict == Verdict.VULNERABLE
                for rf in result.refined_findings
            ):
                engine = "Pattern + AI"
            elif any(
                rf.verdict == Verdict.LIKELY_VULNERABLE
                for rf in result.refined_findings
            ):
                engine = "Pattern (AI advisory)"
            else:
                engine = "Pattern"

            verdict = "VULNERABLE" if any(
                rf.verdict == Verdict.VULNERABLE
                for rf in result.refined_findings
            ) else "LIKELY_VULN"

        results.append(DetectionResult(
            sample=sample,
            detected=detected,
            rules_matched=rules,
            engine=engine,
            confidence=confidence,
            ai_score=ai_score,
            ai_available=ai_available,
            verdict=verdict,
        ))

    return results


def print_per_category_tables(results: List[DetectionResult]):
    """Print detailed results per OWASP category."""

    # Group by OWASP ID
    by_owasp: Dict[str, List[DetectionResult]] = {}
    for r in results:
        key = f"{r.sample.owasp_id}: {r.sample.owasp_name}"
        by_owasp.setdefault(key, []).append(r)

    for owasp_key in sorted(by_owasp.keys()):
        items = by_owasp[owasp_key]
        print(f"\n{'─' * 80}")
        print(f"  {owasp_key}")
        print(f"{'─' * 80}")
        print(f"  {'Sample':<14} {'Lang':<6} {'SubType':<30} {'Detected':<10} {'Engine':<22} {'Rules'}")
        print(f"  {'─'*13} {'─'*5} {'─'*29} {'─'*9} {'─'*21} {'─'*20}")
        for r in items:
            det_mark = "✅ YES" if r.detected else "❌ NO"
            rules_str = ", ".join(r.rules_matched[:2]) if r.rules_matched else "—"
            print(f"  {r.sample.sample_id:<14} {r.sample.language:<6} "
                  f"{r.sample.sub_type:<30} {det_mark:<10} "
                  f"{r.engine:<22} {rules_str}")


def print_summary_table(results: List[DetectionResult]):
    """Print aggregated summary."""

    print(f"\n{'=' * 80}")
    print("AGGREGATED DETECTION RESULTS")
    print(f"{'=' * 80}\n")

    # Per-OWASP category
    by_owasp: Dict[str, List[DetectionResult]] = {}
    for r in results:
        by_owasp.setdefault(r.sample.owasp_id, []).append(r)

    owasp_names = {
        "A01": "Broken Access Control",
        "A02": "Cryptographic Failures",
        "A03": "Injection",
        "A04": "Insecure Design",
        "A05": "Security Misconfiguration",
        "A06": "Vulnerable & Outdated Components",
        "A07": "Auth Failures",
        "A08": "Data Integrity Failures",
        "A09": "Logging & Monitoring",
        "A10": "SSRF",
    }

    print(f"  {'OWASP Category':<40} {'Total':<8} {'Detected':<10} {'Rate':<10} {'Engine'}")
    print(f"  {'─'*39} {'─'*7} {'─'*9} {'─'*9} {'─'*25}")

    total_samples = 0
    total_detected = 0
    pattern_detections = 0
    ai_assists = 0

    for oid in sorted(by_owasp.keys()):
        items = by_owasp[oid]
        n_total = len(items)
        n_detected = sum(1 for r in items if r.detected)
        rate = f"{n_detected / n_total * 100:.0f}%" if n_total > 0 else "N/A"
        name = f"{oid}: {owasp_names.get(oid, '?')}"

        # Engine breakdown
        engines = set()
        for r in items:
            if r.detected:
                engines.add(r.engine.split(" (")[0].split(" +")[0])  # "Pattern"
        engine_str = " + ".join(sorted(engines)) if engines else "—"

        total_samples += n_total
        total_detected += n_detected
        pattern_detections += sum(1 for r in items if r.detected and "Pattern" in r.engine)
        ai_assists += sum(1 for r in items if r.detected and "AI" in r.engine)

        print(f"  {name:<40} {n_total:<8} {n_detected:<10} {rate:<10} {engine_str}")

    print(f"  {'─'*39} {'─'*7} {'─'*9} {'─'*9}")
    overall_rate = f"{total_detected / total_samples * 100:.1f}%"
    print(f"  {'OVERALL':<40} {total_samples:<8} {total_detected:<10} {overall_rate}")

    print(f"\n  Detection Breakdown:")
    print(f"    Pattern Engine detections:  {pattern_detections}")
    print(f"    AI-assisted detections:     {ai_assists}")
    print(f"    Total unique detections:    {total_detected} / {total_samples}")


def print_poster_findings(results: List[DetectionResult]):
    """Print poster-ready key findings."""

    print(f"\n{'=' * 80}")
    print("KEY FINDINGS (Poster-Ready)")
    print(f"{'=' * 80}\n")

    detected = [r for r in results if r.detected]
    not_detected = [r for r in results if not r.detected]

    # What we detect well
    by_owasp_det: Dict[str, int] = {}
    for r in detected:
        by_owasp_det[r.sample.owasp_id] = by_owasp_det.get(r.sample.owasp_id, 0) + 1

    print("  STRENGTHS:")
    for oid in sorted(by_owasp_det.keys()):
        count = by_owasp_det[oid]
        total = sum(1 for r in results if r.sample.owasp_id == oid)
        if count == total:
            print(f"    • {oid} — {count}/{total} samples detected (100%)")
        else:
            print(f"    • {oid} — {count}/{total} samples detected ({count/total*100:.0f}%)")

    # What we don't detect
    by_owasp_miss: Dict[str, int] = {}
    for r in not_detected:
        by_owasp_miss[r.sample.owasp_id] = by_owasp_miss.get(r.sample.owasp_id, 0) + 1

    print(f"\n  LIMITATIONS (Honest Assessment):")

    # Categories with 0% detection
    all_owasp = set(r.sample.owasp_id for r in results)
    detected_owasp = set(r.sample.owasp_id for r in detected)
    undetected_owasp = all_owasp - detected_owasp

    for oid in sorted(undetected_owasp):
        total = sum(1 for r in results if r.sample.owasp_id == oid)
        owasp_names = {
            "A04": "Insecure Design", "A06": "Vulnerable Components",
            "A09": "Logging Failures",
        }
        name = owasp_names.get(oid, "?")
        print(f"    • {oid} ({name}) — 0/{total} detected")
        print(f"      Reason: Requires {'architectural' if oid in ('A04','A09') else 'SCA'} analysis"
              f" beyond pattern/AI scope")

    # Partial detections
    partial = detected_owasp - {oid for oid, v in by_owasp_det.items()
                                  if v == sum(1 for r in results if r.sample.owasp_id == oid)}
    for oid in sorted(partial):
        n_det = by_owasp_det.get(oid, 0)
        n_total = sum(1 for r in results if r.sample.owasp_id == oid)
        missed = [r for r in results if r.sample.owasp_id == oid and not r.detected]
        subtypes = {r.sample.sub_type for r in missed}
        print(f"    • {oid} — {n_det}/{n_total} detected, missed: {', '.join(subtypes)}")

    print(f"\n  SCOPE STATEMENT:")
    print(f"    The system focuses on Python and JavaScript vulnerabilities and combines")
    print(f"    rule-based detection with AI-assisted analysis. It covers 6 CWE categories")
    print(f"    (CWE-89, CWE-77, CWE-22, CWE-502, CWE-918, CWE-798) and does NOT claim")
    print(f"    full OWASP Top 10 coverage. Categories requiring architectural context,")
    print(f"    SCA tooling, or runtime analysis are explicitly out of scope.")


def print_example_detections(results: List[DetectionResult]):
    """Print 3 example detections in poster format."""

    print(f"\n{'=' * 80}")
    print("EXAMPLE DETECTED VULNERABILITIES (Poster Format)")
    print(f"{'=' * 80}")

    # Pick one from each strong category
    examples = []
    seen_owasp = set()
    priorities = ["A03", "A10", "A08", "A01", "A02", "A07"]
    for oid in priorities:
        for r in results:
            if r.detected and r.sample.owasp_id == oid and oid not in seen_owasp:
                examples.append(r)
                seen_owasp.add(oid)
                break
        if len(examples) >= 3:
            break

    for i, r in enumerate(examples, 1):
        s = r.sample
        print(f"\n  ─── Example {i}: {s.owasp_id} — {s.sub_type} ───")
        print(f"  Vulnerability:  {s.sub_type} ({s.owasp_id}: {s.owasp_name})")
        print(f"  CWE:            {s.cwe}")
        print(f"  Language:       {s.language.capitalize()}")
        print(f"  Detected by:    {r.engine}")
        print(f"  Rules:          {', '.join(r.rules_matched)}")
        print(f"  Confidence:     {r.confidence}")
        print(f"  Risk:           HIGH")
        print(f"  Description:    {s.description[:120]}...")
        print(f"  Recommendation: {s.fix_description[:120]}...")
        # Trim code for display
        code_lines = [l for l in s.fix_code.strip().split('\n') if l.strip()][:5]
        print(f"  Secure Code:")
        for line in code_lines:
            print(f"    {line}")


def print_honest_limitations():
    """Print honest limitations section."""

    print(f"\n{'=' * 80}")
    print("HONEST LIMITATIONS")
    print(f"{'=' * 80}\n")
    print("  1. Pattern matching is REGEX-BASED, not semantic analysis.")
    print("     It cannot understand data flow across function boundaries.")
    print()
    print("  2. The AI model (GNN+BiLSTM) was trained on synthetic data.")
    print("     AI refinement is ADVISORY — it assists pattern matching but")
    print("     is not a standalone detector.")
    print()
    print("  3. The scanner covers 6 CWEs out of 25+ in the OWASP Top 10.")
    print("     Categories requiring architectural context (A04, A09),")
    print("     SCA tooling (A06), or runtime analysis are out of scope.")
    print()
    print("  4. TypeScript is transpiled to JavaScript via regex stripping.")
    print("     Complex TS generics may not fully process.")
    print()
    print("  5. Cross-file data flow is NOT supported.")
    print("     Each file is analyzed in isolation.")
    print()
    print("  6. This evaluation uses crafted test samples, not production code.")
    print("     Real-world detection rates may differ from these results.")


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    t0 = time.time()

    results = run_evaluation()

    print_per_category_tables(results)
    print_summary_table(results)
    print_poster_findings(results)
    print_example_detections(results)
    print_honest_limitations()

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Evaluation completed in {elapsed:.1f}s ({len(SAMPLES)} samples)")
    print(f"{'=' * 80}")
