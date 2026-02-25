#!/usr/bin/env python3
"""
Generate expanded OWASP evaluation dataset by creating variations
of existing samples to reach 8+ samples per category.
"""

# This script adds synthetic variations to bring each OWASP category to 8 samples minimum

ADDITIONAL_SAMPLES = '''
# ─────────────────────────────────────────────────────────────────────
# A01: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A01-JS-03", "A01", "Broken Access Control", "Path Traversal in Express",
   "CWE-22", "javascript",
   \'''
const express = require('express');
const fs = require('fs');
const app = express();

app.get('/files', (req, res) => {
    const filename = req.query.file;
    const content = fs.readFileSync('/var/data/' + filename, 'utf8');
    res.send(content);
});
\''',
   "Direct concatenation of user input with filesystem path enables directory traversal.",
   "Validate and sanitize input, use path.resolve() with whitelist check.",
   \'''
const path = require('path');
const BASE = path.resolve('/var/data');
const filePath = path.resolve(BASE, req.query.file);
if (!filePath.startswith(BASE)) return res.status(403).send('Forbidden');
\''')

_s("A01-PY-04", "A01", "Broken Access Control", "Path Traversal in Django",
   "CWE-22", "python",
   \'''
from django.http import HttpResponse
import os

def serve_file(request):
    filename = request.GET.get('file', '')
    full_path = f"/var/www/uploads/{filename}"
    with open(full_path, 'r') as f:
        return HttpResponse(f.read())
\''',
   "User-supplied filename concatenated into path without validation.",
   "Use os.path.abspath() and verify it starts with the safe directory.",
   \'''
import os
BASE_DIR = os.path.abspath("/var/www/uploads")
full_path = os.path.abspath(os.path.join(BASE_DIR, filename))
if not full_path.startswith(BASE_DIR):
    return HttpResponseForbidden()
\''')

_s("A01-JS-04", "A01", "Broken Access Control", "File Read Without Validation",
   "CWE-22", "javascript",
   \'''
const fs = require('fs').promises;

async function getDocument(docName) {
    const path = `./documents/${docName}`;
    return await fs.readFile(path, 'utf8');
}
\''',
   "Template literal with user-controlled variable can be exploited with ../",
   "Validate filename against allowed characters and use path.join with basedir check.",
   \'''
const path = require('path');
const BASE = path.resolve('./documents');
const fullPath = path.resolve(BASE, docName);
if (!fullPath.startsWith(BASE)) throw new Error('Invalid path');
\''')

# ─────────────────────────────────────────────────────────────────────
# A02: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A02-JS-03", "A02", "Cryptographic Failures", "Hardcoded Database Password",
   "CWE-798", "javascript",
   \'''
const mongoose = require('mongoose');

const DB_PASSWORD = "Prod_Mongo_P@ssw0rd_2024";
const connectionString = `mongodb://admin:${DB_PASSWORD}@prod.server.com:27017/myapp`;

mongoose.connect(connectionString);
\''',
   "MongoDB password hardcoded in source code. Exposed in version control and build logs.",
   "Use environment variables for sensitive credentials.",
   \'''
const DB_PASSWORD = process.env.MONGO_PASSWORD;
const connectionString = `mongodb://admin:${DB_PASSWORD}@prod.server.com:27017/myapp`;
\''')

_s("A02-PY-04", "A02", "Cryptographic Failures", "Hardcoded Encryption Key",
   "CWE-798", "python",
   \'''
from cryptography.fernet import Fernet

ENCRYPTION_KEY = b"aGVsbG93b3JsZGhlbGxvd29ybGRoZWxsbw=="

def encrypt_data(plaintext):
    cipher = Fernet(ENCRYPTION_KEY)
    return cipher.encrypt(plaintext.encode())
\''',
   "Encryption key hardcoded as bytes literal. Anyone with source access can decrypt all data.",
   "Generate key at deployment time and store in secure key management system.",
   \'''
import os
ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"].encode()
\''')

_s("A02-JS-04", "A02", "Cryptographic Failures", "Hardcoded Private Key",
   "CWE-798", "javascript",
   \'''
const crypto = require('crypto');

const PRIVATE_KEY = \`-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA2dGhPGLmY8jVXh9k3pM...
-----END RSA PRIVATE KEY-----\`;

function signToken(data) {
    return crypto.sign('sha256', Buffer.from(data), PRIVATE_KEY);
}
\''',
   "RSA private key hardcoded in source. Attacker with code access can forge signatures.",
   "Load private key from secure storage or environment variable.",
   \'''
const fs = require('fs');
const PRIVATE_KEY = fs.readFileSync(process.env.PRIVATE_KEY_PATH, 'utf8');
\''')

# ─────────────────────────────────────────────────────────────────────
# A03: ADDITIONAL SAMPLES (bringing total to 10)
# ─────────────────────────────────────────────────────────────────────

_s("A03-PY-05", "A03", "Injection", "SQL Injection in Django ORM raw",
   "CWE-89", "python",
   \'''
from django.db import connection

def search_users(keyword):
    with connection.cursor() as cursor:
        query = "SELECT * FROM users WHERE username = '%s'" % keyword
        cursor.execute(query)
        return cursor.fetchall()
\''',
   "String formatting in SQL query enables injection attacks.",
   "Use parameterized queries with %s placeholders.",
   \'''
query = "SELECT * FROM users WHERE username = %s"
cursor.execute(query, [keyword])
\''')

_s("A03-JS-05", "A03", "Injection", "NoSQL Injection in MongoDB",
   "CWE-89", "javascript",
   \'''
const { MongoClient } = require('mongodb');

async function findUser(username) {
    const query = { username: username };
    // If username = {"$gt": ""} it returns all users
    return await db.collection('users').findOne(query);
}
\''',
   "Directly passing user input to MongoDB query enables NoSQL injection.",
   "Validate input type and sanitize against operator injection.",
   \'''
if (typeof username !== 'string') throw new Error('Invalid input');
const query = { username: String(username) };
\''')

# ─────────────────────────────────────────────────────────────────────
# A04: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A04-JS-02", "A04", "Insecure Design", "No CAPTCHA on Registration",
   "CWE-841", "javascript",
   \'''
app.post('/api/register', async (req, res) => {
    const { email, password } = req.body;
    await User.create({ email, password });
    res.json({ message: 'Account created' });
});
\''',
   "Registration endpoint has no CAPTCHA or rate limiting. Bots can create unlimited accounts.",
   "Implement CAPTCHA (reCAPTCHA) and rate limiting per IP.",
   \'''
const rateLimit = require('express-rate-limit');
const limiter = rateLimit({ windowMs: 15 * 60 * 1000, max: 5 });
app.post('/api/register', limiter, verifyCaptcha, async (req, res) => { ... });
\''')

_s("A04-PY-03", "A04", "Insecure Design", "Sequential Transaction IDs",
   "CWE-330", "python",
   \'''
transaction_counter = 1000

@app.route("/api/payment", methods=["POST"])
def process_payment():
    global transaction_counter
    transaction_counter += 1
    txn_id = f"TXN{transaction_counter}"
    return jsonify({"transaction_id": txn_id})
\''',
   "Predictable transaction IDs allow attackers to enumerate and guess other transactions.",
   "Use cryptographically secure random IDs (UUID4 or secure random bytes).",
   \'''
import uuid
txn_id = str(uuid.uuid4())
\''')

_s("A04-JS-03", "A04", "Insecure Design", "No Email Verification",
   "CWE-20", "javascript",
   \'''
app.post('/signup', async (req, res) => {
    const user = new User({ email: req.body.email, verified: false });
    await user.save();
    res.json({ message: 'Signed up' });
});
\''',
   "User accounts created without email verification. Attacker can register with others' emails.",
   "Send verification email with token before enabling account.",
   \'''
sendVerificationEmail(user.email, user.verificationToken);
\''')

_s("A04-PY-04", "A04", "Insecure Design", "No Secondary Authorization for Sensitive Action",
   "CWE-306", "python",
   \'''
@app.route("/api/delete-account", methods=["POST"])
@login_required
def delete_account():
    db.delete_user(current_user.id)
    return jsonify({"message": "Account deleted"})
\''',
   "Critical action (account deletion) requires no secondary confirmation or password re-entry.",
   "Require password confirmation or 2FA for sensitive operations.",
   \'''
password = request.json.get("password")
if not verify_password(current_user, password):
    return jsonify({"error": "Invalid password"}), 401
\''')

_s("A04-JS-04", "A04", "Insecure Design", "No Balance Check Before Debit",
   "CWE-20", "javascript",
   \'''
app.post('/api/withdraw', async (req, res) => {
    const amount = req.body.amount;
    await Account.updateOne({ _id: req.user.accountId }, {
        $inc: { balance: -amount }
    });
    res.json({ message: 'Withdrawn' });
});
\''',
   "No validation that balance is sufficient before debit. Negative balances allowed.",
   "Check balance before transaction and use atomic operations.",
   \'''
const account = await Account.findById(req.user.accountId);
if (account.balance < amount) return res.status(400).json({ error: 'Insufficient funds' });
\''')

# ─────────────────────────────────────────────────────────────────────
# A05: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A05-PY-03", "A05", "Security Misconfiguration", "Hardcoded Admin Token",
   "CWE-798", "python",
   \'''
ADMIN_API_TOKEN = "admin_token_x7f9k2m5p8q1w3"

@app.route("/api/admin/stats")
def admin_stats():
    token = request.headers.get("Authorization")
    if token == ADMIN_API_TOKEN:
        return jsonify(get_system_stats())
\''',
   "Admin API token hardcoded. Any developer can access admin endpoints.",
   "Use environment variable and rotate tokens regularly.",
   \'''
ADMIN_API_TOKEN = os.environ["ADMIN_TOKEN"]
\''')

_s("A05-JS-02", "A05", "Security Misconfiguration", "Exposed Stack Traces",
   "CWE-209", "javascript",
   \'''
app.use((err, req, res, next) => {
    res.status(500).json({
        error: err.message,
        stack: err.stack,
        file: err.fileName
    });
});
\''',
   "Error handler exposes full stack traces to client. Reveals internal paths and logic.",
   "Log errors server-side, return generic messages to client.",
   \'''
logger.error(err);
res.status(500).json({ error: 'Internal server error' });
\''')

_s("A05-PY-04", "A05", "Security Misconfiguration", "Hardcoded Session Secret",
   "CWE-798", "python",
   \'''
from flask import Flask, session

app = Flask(__name__)
app.secret_key = "my-super-secret-session-key-do-not-share"

@app.route("/login", methods=["POST"])
def login():
    session["user_id"] = 123
    return "Logged in"
\''',
   "Flask session secret is hardcoded. Attacker can forge session cookies.",
   "Load secret from environment variable.",
   \'''
app.secret_key = os.environ["SESSION_SECRET"]
\''')

_s("A05-JS-03", "A05", "Security Misconfiguration", "Default Credentials",
   "CWE-798", "javascript",
   \'''
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
\''',
   "Default admin account created with predictable password.",
   "Force password change on first login or generate random password.",
   \'''
const crypto = require('crypto');
const randomPassword = crypto.randomBytes(16).toString('hex');
sendPasswordEmail(randomPassword);
\''')

_s("A05-PY-05", "A05", "Security Misconfiguration", "Hardcoded OAuth Client Secret",
   "CWE-798", "python",
   \'''
OAUTH_CLIENT_SECRET = "oauth_secret_abc123xyz789"

def exchange_code_for_token(code):
    response = requests.post("https://oauth.provider.com/token", data={
        "code": code,
        "client_secret": OAUTH_CLIENT_SECRET
    })
    return response.json()
\''',
   "OAuth client secret hardcoded. Compromised secret allows token forgery.",
   "Store in environment variable or secrets manager.",
   \'''
OAUTH_CLIENT_SECRET = os.environ["OAUTH_CLIENT_SECRET"]
\''')

# ─────────────────────────────────────────────────────────────────────
# A06: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A06-PY-02", "A06", "Vulnerable & Outdated Components", "Vulnerable Pillow Library",
   "CWE-1035", "python",
   \'''
# requirements.txt:
# Pillow==6.0.0  (CVE-2020-5312: buffer overflow)

from PIL import Image

def process_upload(file_path):
    img = Image.open(file_path)
    img.thumbnail((200, 200))
    img.save("output.jpg")
\''',
   "Using Pillow 6.0.0 which has known buffer overflow vulnerability.",
   "Update to latest Pillow version (>=9.0.0).",
   \'''
# requirements.txt:
# Pillow>=10.0.0
\''')

_s("A06-JS-02", "A06", "Vulnerable & Outdated Components", "Vulnerable jsonwebtoken",
   "CWE-1035", "javascript",
   \'''
// package.json: "jsonwebtoken": "8.1.0" (CVE-2022-23529)

const jwt = require('jsonwebtoken');

function verify(token) {
    return jwt.verify(token, publicKey, { algorithms: ['RS256'] });
}
\''',
   "jsonwebtoken 8.1.0 has vulnerability allowing secret key exposure.",
   "Update to jsonwebtoken@9.0.0 or later.",
   \'''
// package.json: "jsonwebtoken": "^9.0.0"
\''')

_s("A06-PY-03", "A06", "Vulnerable & Outdated Components", "Outdated Django",
   "CWE-1035", "python",
   \'''
# requirements.txt:
# Django==2.2.0  (Multiple CVEs including SQL injection bypass)

from django.shortcuts import render

def view(request):
    return render(request, "page.html")
\''',
   "Django 2.2.0 has multiple known security vulnerabilities.",
   "Update to Django 4.2 LTS or later.",
   \'''
# Django>=4.2
\''')

_s("A06-JS-03", "A06", "Vulnerable & Outdated Components", "Vulnerable axios",
   "CWE-1035", "javascript",
   \'''
// package.json: "axios": "0.21.1" (CVE-2021-3749: ReDoS)

const axios = require('axios');

async function fetchData(url) {
    return await axios.get(url);
}
\''',
   "axios 0.21.1 vulnerable to Regular Expression Denial of Service.",
   "Update to axios@1.6.0 or later.",
   \'''
// "axios": "^1.6.0"
\''')

_s("A06-PY-04", "A06", "Vulnerable &Outdated Components", "Unsafe XML Parser",
   "CWE-611", "python",
   \'''
import xml.etree.ElementTree as ET

def parse_xml(xml_string):
    root = ET.fromstring(xml_string)
    return root.findall(".//data")
\''',
   "Default XML parser vulnerable to XXE (XML External Entity) attacks.",
   "Use defusedxml library to safely parse XML.",
   \'''
from defusedxml import ElementTree as ET
root = ET.fromstring(xml_string)
\''')

_s("A06-JS-04", "A06", "Vulnerable & Outdated Components", "Vulnerable socket.io",
   "CWE-1035", "javascript",
   \'''
// package.json: "socket.io": "2.3.0" (CVE-2020-36048)

const io = require('socket.io')(server);

io.on('connection', (socket) => {
    console.log('Client connected');
});
\''',
   "socket.io 2.3.0 has CORS bypass vulnerability.",
   "Update to socket.io@4.0.0 or later.",
   \'''
// "socket.io": "^4.6.0"
\''')

# ─────────────────────────────────────────────────────────────────────
# A07: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A07-PY-03", "A07", "Identification & Authentication Failures", "Hardcoded TOTP Secret",
   "CWE-798", "python",
   \'''
import pyotp

TOTP_SECRET = "JBSWY3DPEHPK3PXP"

def verify_2fa(user_code):
    totp = pyotp.TOTP(TOTP_SECRET)
    return totp.verify(user_code)
\''',
   "TOTP secret hardcoded and shared across all users.",
   "Generate unique TOTP secret per user and store in database.",
   \'''
totp = pyotp.TOTP(user.totp_secret)
\''')

_s("A07-JS-02", "A07", "Identification & Authentication Failures", "Weak Session ID",
   "CWE-330", "javascript",
   \'''
let sessionCounter = 1000;

function createSession(userId) {
    sessionCounter++;
    return { id: \`sess_\${sessionCounter}\`, userId };
}
\''',
   "Predictable sequential session IDs enable session hijacking.",
   "Use cryptographically secure random session identifiers.",
   \'''
const crypto = require('crypto');
const sessionId = crypto.randomBytes(32).toString('hex');
\''')

_s("A07-PY-04", "A07", "Identification & Authentication Failures", "No Account Lockout",
   "CWE-307", "python",
   \'''
@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    user = User.query.filter_by(username=username).first()
    if user and verify_password(user, password):
        return jsonify({"token": create_token(user)})
    return jsonify({"error": "Invalid credentials"}), 401
\''',
   "No failed attempt tracking or account lockout. Brute force attacks possible.",
   "Track failed attempts, implement exponential backoff and CAPTCHA.",
   \'''
if user.failed_attempts >= 5:
    if user.locked_until > datetime.now():
        return jsonify({"error": "Account locked"}), 429
\''')

_s("A07-JS-03", "A07", "Identification & Authentication Failures", "Hardcoded Cookie Secret",
   "CWE-798", "javascript",
   \'''
const session = require('express-session');

app.use(session({
    secret: 'keyboard-cat-session-secret-2024',
    resave: false,
    saveUninitialized: true
}));
\''',
   "Session cookie secret is hardcoded. Session cookies can be forged.",
   "Load secret from environment variable.",
   \'''
secret: process.env.SESSION_SECRET
\''')

_s("A07-PY-05", "A07", "Identification & Authentication Failures", "Hardcoded API Master Key",
   "CWE-798", "python",
   \'''
API_MASTER_KEY = "mk_live_a1b2c3d4e5f6g7h8i9j0"

def validate_api_key(provided_key):
    return provided_key == API_MASTER_KEY
\''',
   "Master API key hardcoded in source code. Compromised key grants full system access.",
   "Use environment variable and implement key rotation policy.",
   \'''
API_MASTER_KEY = os.environ["API_MASTER_KEY"]
\''')

# ─────────────────────────────────────────────────────────────────────
# A08: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A08-JS-02", "A08", "Software & Data Integrity Failures", "Deserialization via JSON.parse with __proto__",
   "CWE-502", "javascript",
   \'''
app.post('/api/config', (req, res) => {
    const config = JSON.parse(req.body.config);
    Object.assign(appConfig, config);  // Prototype pollution
    res.json({ message: 'Config updated' });
});
\''',
   "JSON.parse with Object.assign enables prototype pollution attacks.",
   "Use JSON schema validation and avoid Object.assign with untrusted data.",
   \'''
const Ajv = require('ajv');
const ajv = new Ajv();
const valid = ajv.validate(configSchema, config);
if (!valid) return res.status(400).json({ error: 'Invalid config' });
\''')

_s("A08-PY-04", "A08", "Software & Data Integrity Failures", "marshal.loads on Untrusted Data",
   "CWE-502", "python",
   \'''
import marshal

def load_cache(cache_data):
    obj = marshal.loads(cache_data)
    return obj
\''',
   "marshal.loads() can execute arbitrary code like pickle.",
   "Use JSON or MessagePack for serialization.",
   \'''
import json
obj = json.loads(cache_data)
\''')

_s("A08-JS-03", "A08", "Software & Data Integrity Failures", "vm.runInNewContext Injection",
   "CWE-94", "javascript",
   \'''
const vm = require('vm');

function evaluateExpression(expr) {
    return vm.runInNewContext(expr);
}
\''',
   "vm.runInNewContext with user input enables code injection.",
   "Never execute user-provided code. Use safe expression parser.",
   \'''
const math = require('mathjs');
return math.evaluate(expr);
\''')

_s("A08-PY-05", "A08", "Software & Data Integrity Failures", "exec() on User Input",
   "CWE-94", "python",
   \'''
def run_formula(formula):
    result = {}
    exec(f"result['value'] = {formula}", {"result": result})
    return result["value"]
\''',
   "exec() with user-controlled string enables arbitrary Python code execution.",
   "Use ast.literal_eval() for safe evaluation or expression parser.",
   \'''
import ast
return ast.literal_eval(formula)
\''')

# ─────────────────────────────────────────────────────────────────────
# A09: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A09-PY-02", "A09", "Security Logging & Monitoring Failures", "No Logging for Failed Auth",
   "CWE-778", "python",
   \'''
@app.route("/login", methods=["POST"])
def login():
    username = request.json["username"]
    password = request.json["password"]
    if not authenticate(username, password):
        return jsonify({"error": "Invalid"}), 401
    # No logging of failed attempt
    return jsonify({"token": create_token(username)})
\''',
   "Failed authentication attempts not logged. Cannot detect brute force attacks.",
   "Log all authentication events with timestamp, IP, and outcome.",
   \'''
logger.warning(f"Failed login attempt for {username} from {request.remote_addr}")
\''')

_s("A09-JS-02", "A09", "Security Logging & Monitoring Failures", "Logging Sensitive Headers",
   "CWE-532", "javascript",
   \'''
app.use((req, res, next) => {
    logger.info('Request received', {
        path: req.path,
        headers: req.headers,  // Contains Authorization tokens!
        body: req.body
    });
    next();
});
\''',
   "Logging full headers exposes Authorization tokens and sensitive data.",
   "Filter sensitive headers before logging.",
   \'''
const safeHeaders = { ...req.headers };
delete safeHeaders.authorization;
delete safeHeaders.cookie;
logger.info('Request', { path: req.path, headers: safeHeaders });
\''')

_s("A09-PY-03", "A09", "Security Logging & Monitoring Failures", "No Monitoring for Privilege Escalation",
   "CWE-778", "python",
   \'''
@app.route("/api/promote", methods=["POST"])
@admin_required
def promote_user():
    user_id = request.json["user_id"]
    db.update_user(user_id, role="admin")
    # No audit log
    return jsonify({"message": "User promoted"})
\''',
   "Privilege escalation not logged or monitored. Security incidents go undetected.",
   "Log all role changes with actor, target, and timestamp.",
   \'''
audit_logger.critical(f"User {current_user.id} promoted user {user_id} to admin")
\''')

_s("A09-JS-03", "A09", "Security Logging & Monitoring Failures", "PII in Error Logs",
   "CWE-532", "javascript",
   \'''
app.post('/api/update-profile', async (req, res) => {
    try {
        await User.updateOne({ _id: req.user.id }, req.body);
        res.json({ message: 'Updated' });
    } catch (error) {
        logger.error('Update failed', { user: req.user, body: req.body, error });
        res.status(500).send('Error');
    }
});
\''',
   "Error logs contain full user object and request body with PII.",
   "Log only error message and request ID, not sensitive data.",
   \'''
logger.error('Update failed', { userId: req.user.id, error: error.message });
\''')

_s("A09-PY-04", "A09", "Security Logging & Monitoring Failures", "No Tamper Detection",
   "CWE-778", "python",
   \'''
@app.route("/api/data/<int:record_id>", methods=["PUT"])
def update_record(record_id):
    data = request.json
    db.update(record_id, data)
    return jsonify({"message": "Updated"})
\''',
   "Data modifications not logged. No audit trail for tampering detection.",
   "Log before and after values for all data changes.",
   \'''
old_value = db.get(record_id)
db.update(record_id, data)
audit_log(user=current_user, action="UPDATE", record=record_id, before=old_value, after=data)
\''')

_s("A09-JS-04", "A09", "Security Logging & Monitoring Failures", "Insufficient Security Event Logging",
   "CWE-778", "javascript",
   \'''
app.post('/api/admin/delete-all-data', authenticate, requireAdmin, async (req, res) => {
    await Database.dropAllCollections();
    res.json({ message: 'All data deleted' });
});
\''',
   "Critical destructive action with no security event logging or alerting.",
   "Log critical actions and trigger immediate alerts.",
   \'''
securityLogger.critical({ action: 'DROP_ALL_DATA', actor: req.user.id, timestamp: new Date() });
sendAlert('CRITICAL: All data deleted', req.user);
\''')

# ─────────────────────────────────────────────────────────────────────
# A10: ADDITIONAL SAMPLES (bringing total to 8)
# ─────────────────────────────────────────────────────────────────────

_s("A10-PY-04", "A10", "Server-Side Request Forgery", "SSRF in urllib",
   "CWE-918", "python",
   \'''
from urllib.request import urlopen

def fetch_url(url):
    response = urlopen(url)
    return response.read()
\''',
   "urllib.request.urlopen with user-controlled URL enables SSRF.",
   "Validate URL against allowlist and block private IP ranges.",
   \'''
from urllib.parse import urlparse
parsed = urlparse(url)
if parsed.hostname not in ALLOWED_HOSTS:
    raise ValueError("Host not allowed")
\''')

_s("A10-JS-03", "A10", "Server-Side Request Forgery", "SSRF in http.request",
   "CWE-918", "javascript",
   \'''
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
\''',
   "http.get() with user URL enables internal network scanning.",
   "Parse URL and validate hostname against allowlist.",
   \'''
const { URL } = require('url');
const parsed = new URL(targetUrl);
if (!ALLOWED_HOSTS.includes(parsed.hostname)) throw new Error('Forbidden');
\''')

_s("A10-PY-05", "A10", "Server-Side Request Forgery", "SSRF via Webhook URL",
   "CWE-918", "python",
   \'''
import requests

@app.route("/api/webhook/test", methods=["POST"])
def test_webhook():
    webhook_url = request.json["url"]
    resp = requests.post(webhook_url, json={"event": "test"})
    return jsonify({"status_code": resp.status_code})
\''',
   "Webhook testing endpoint allows arbitrary POST requests to any URL.",
   "Validate webhook URLs against strict allowlist.",
   \'''
from urllib.parse import urlparse
parsed = urlparse(webhook_url)
if parsed.hostname not in ALLOWED_WEBHOOK_HOSTS:
    return jsonify({"error": "Invalid webhook host"}), 403
\''')
'''

print("Copy the above code and append to owasp_evaluation.py after the last _s() call and before the evaluation functions.")
print(f"\\nThis adds 42 new samples bringing the total to 82 samples.")
print("\\nDistribution after expansion:")
print("  A01: 8 samples")
print("  A02: 8 samples")
print("  A03: 10 samples")
print("  A04: 8 samples")
print("  A05: 8 samples")
print("  A06: 8 samples") 
print("  A07: 8 samples")
print("  A08: 8 samples")
print("  A09: 8 samples")
print("  A10: 8 samples")
