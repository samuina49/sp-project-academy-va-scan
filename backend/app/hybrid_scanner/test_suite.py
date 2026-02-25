"""
CWE Test Suite for Hybrid Scanner Evaluation
=============================================
Realistic, non-duplicated test cases for each supported CWE.
Each test case is labeled with ground truth (VULNERABLE / SAFE).

Design:
    - REALISTIC code patterns (not synthetic templates)
    - Separate VULNERABLE and SAFE variants
    - Multiple coding styles and frameworks
    - NO duplicate templates between test cases
    - Covers both Python and JavaScript
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TestCase:
    id: str
    cwe: str
    language: str
    is_vulnerable: bool
    description: str
    code: str


# =====================================================================
# CWE-89: SQL Injection
# =====================================================================

CWE89_TESTS: List[TestCase] = [
    # --- VULNERABLE ---
    TestCase("sqli-py-01", "CWE-89", "python", True,
             "Flask route with f-string SQL query",
             '''
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route("/users")
def get_user():
    username = request.args.get("username")
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return {"users": cursor.fetchall()}
'''),

    TestCase("sqli-py-02", "CWE-89", "python", True,
             "Django raw SQL with string concatenation",
             '''
from django.http import JsonResponse
from django.db import connection

def search_products(request):
    term = request.GET.get("q", "")
    with connection.cursor() as cursor:
        sql = "SELECT id, name, price FROM products WHERE name LIKE '%" + term + "%'"
        cursor.execute(sql)
        rows = cursor.fetchall()
    return JsonResponse({"results": rows}, safe=False)
'''),

    TestCase("sqli-py-03", "CWE-89", "python", True,
             "SQLAlchemy text() with format string",
             '''
from sqlalchemy import create_engine, text
from flask import request

engine = create_engine("sqlite:///myapp.db")

def find_order(order_id):
    with engine.connect() as conn:
        query = "SELECT * FROM orders WHERE id = {}".format(order_id)
        result = conn.execute(text(query))
        return result.fetchone()

@app.route("/order/<order_id>")
def order_view(order_id):
    return find_order(order_id)
'''),

    TestCase("sqli-py-04", "CWE-89", "python", True,
             "FastAPI with %-formatting SQL",
             '''
from fastapi import FastAPI, Query
import psycopg2

app = FastAPI()

@app.get("/search")
def search(q: str = Query(...)):
    conn = psycopg2.connect("dbname=mydb user=admin")
    cur = conn.cursor()
    cur.execute("SELECT * FROM articles WHERE title LIKE '%s'" % q)
    return {"articles": cur.fetchall()}
'''),

    TestCase("sqli-py-05", "CWE-89", "python", True,
             "execute() with f-string directly",
             '''
import mysql.connector

def delete_user(user_id):
    db = mysql.connector.connect(host="localhost", database="users")
    cursor = db.cursor()
    cursor.execute(f"DELETE FROM users WHERE id = {user_id}")
    db.commit()
    return True
'''),

    TestCase("sqli-js-01", "CWE-89", "javascript", True,
             "Express + MySQL with template literal",
             '''
const express = require('express');
const mysql = require('mysql2');
const app = express();

const db = mysql.createPool({ host: 'localhost', database: 'shop' });

app.get('/product', (req, res) => {
    const id = req.query.id;
    db.query(`SELECT * FROM products WHERE id = ${id}`, (err, rows) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(rows);
    });
});
'''),

    TestCase("sqli-js-02", "CWE-89", "javascript", True,
             "Sequelize raw query with concatenation",
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

app.post('/login', async (req, res) => {
    const user = await findUser(req.body.username);
    res.json(user);
});
'''),

    TestCase("sqli-js-03", "CWE-89", "javascript", True,
             "Knex.js raw with string interpolation",
             '''
const knex = require('knex')({ client: 'pg' });

app.get('/api/orders', async (req, res) => {
    const status = req.query.status;
    const orders = await knex.raw("SELECT * FROM orders WHERE status = '" + status + "'");
    res.json(orders.rows);
});
'''),

    # --- SAFE ---
    TestCase("sqli-py-safe-01", "CWE-89", "python", False,
             "Parameterized query with tuple",
             '''
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route("/users")
def get_user():
    username = request.args.get("username")
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    return {"users": cursor.fetchall()}
'''),

    TestCase("sqli-py-safe-02", "CWE-89", "python", False,
             "Django ORM query (no raw SQL)",
             '''
from django.http import JsonResponse
from myapp.models import Product

def search_products(request):
    term = request.GET.get("q", "")
    results = Product.objects.filter(name__icontains=term).values()
    return JsonResponse(list(results), safe=False)
'''),

    TestCase("sqli-py-safe-03", "CWE-89", "python", False,
             "SQLAlchemy ORM with filter()",
             '''
from sqlalchemy.orm import Session
from models import User

def get_user(session: Session, user_id: int):
    return session.query(User).filter(User.id == user_id).first()
'''),

    TestCase("sqli-js-safe-01", "CWE-89", "javascript", False,
             "MySQL parameterized query with ?",
             '''
const mysql = require('mysql2');
const db = mysql.createPool({ host: 'localhost', database: 'shop' });

app.get('/product', (req, res) => {
    const id = req.query.id;
    db.query('SELECT * FROM products WHERE id = ?', [id], (err, rows) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(rows);
    });
});
'''),

    TestCase("sqli-js-safe-02", "CWE-89", "javascript", False,
             "Knex query builder (safe)",
             '''
const knex = require('knex')({ client: 'pg' });

app.get('/api/orders', async (req, res) => {
    const status = req.query.status;
    const orders = await knex('orders').where({ status });
    res.json(orders);
});
'''),
]


# =====================================================================
# CWE-77: Command Injection
# =====================================================================

CWE77_TESTS: List[TestCase] = [
    TestCase("cmdi-py-01", "CWE-77", "python", True,
             "os.system with f-string from request",
             '''
from flask import Flask, request
import os

app = Flask(__name__)

@app.route("/ping")
def ping_host():
    host = request.args.get("host", "127.0.0.1")
    os.system(f"ping -c 3 {host}")
    return {"status": "done"}
'''),

    TestCase("cmdi-py-02", "CWE-77", "python", True,
             "subprocess.run with shell=True and user input",
             '''
import subprocess
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/convert")
def convert_file(filename: str = Query(...)):
    cmd = f"ffmpeg -i uploads/{filename} output.mp4"
    subprocess.run(cmd, shell=True, capture_output=True)
    return {"status": "converted"}
'''),

    TestCase("cmdi-py-03", "CWE-77", "python", True,
             "os.popen for DNS lookup",
             '''
import os

def dns_lookup(domain):
    """Perform DNS lookup for the given domain."""
    result = os.popen(f"nslookup {domain}").read()
    return result.strip()

# Called from API handler with user-provided domain
'''),

    TestCase("cmdi-py-04", "CWE-77", "python", True,
             "eval() on user-provided expression",
             '''
from flask import Flask, request

app = Flask(__name__)

@app.route("/calc")
def calculator():
    expression = request.args.get("expr", "1+1")
    result = eval(expression)
    return {"result": result}
'''),

    TestCase("cmdi-py-05", "CWE-77", "python", True,
             "subprocess.check_output with variable command",
             '''
import subprocess

def run_health_check(check_command):
    """Execute a health check command provided by config"""
    output = subprocess.check_output(check_command, shell=True)
    return output.decode()
'''),

    TestCase("cmdi-js-01", "CWE-77", "javascript", True,
             "child_process.exec with template literal",
             '''
const { exec } = require('child_process');
const express = require('express');
const app = express();

app.get('/lookup', (req, res) => {
    const domain = req.query.domain;
    exec(`nslookup ${domain}`, (err, stdout) => {
        res.send(stdout);
    });
});
'''),

    TestCase("cmdi-js-02", "CWE-77", "javascript", True,
             "execSync with concatenation",
             '''
const { execSync } = require('child_process');

function generateThumbnail(imagePath) {
    const output = execSync("convert " + imagePath + " -resize 100x100 thumb.jpg");
    return output;
}

router.post('/upload', (req, res) => {
    generateThumbnail(req.file.path);
    res.json({ ok: true });
});
'''),

    TestCase("cmdi-js-03", "CWE-77", "javascript", True,
             "new Function() with user input",
             '''
app.post('/api/formula', (req, res) => {
    const formula = req.body.formula;
    const compute = new Function('x', 'y', `return ${formula}`);
    const result = compute(10, 20);
    res.json({ result });
});
'''),

    # --- SAFE ---
    TestCase("cmdi-py-safe-01", "CWE-77", "python", False,
             "subprocess with list arguments (no shell)",
             '''
import subprocess

def ping_host(host):
    result = subprocess.run(
        ["ping", "-c", "3", host],
        capture_output=True, text=True
    )
    return result.stdout
'''),

    TestCase("cmdi-py-safe-02", "CWE-77", "python", False,
             "shlex.quote for shell escaping",
             '''
import subprocess
import shlex

def safe_lookup(domain):
    safe_domain = shlex.quote(domain)
    result = subprocess.run(
        f"nslookup {safe_domain}",
        shell=True, capture_output=True, text=True
    )
    return result.stdout
'''),

    TestCase("cmdi-js-safe-01", "CWE-77", "javascript", False,
             "execFile with arguments array",
             '''
const { execFile } = require('child_process');

app.get('/lookup', (req, res) => {
    const domain = req.query.domain;
    execFile('nslookup', [domain], (err, stdout) => {
        res.send(stdout);
    });
});
'''),
]


# =====================================================================
# CWE-22: Path Traversal
# =====================================================================

CWE22_TESTS: List[TestCase] = [
    TestCase("path-py-01", "CWE-22", "python", True,
             "Flask send_file with user-controlled path",
             '''
from flask import Flask, request, send_file
import os

app = Flask(__name__)

@app.route("/download")
def download():
    filename = request.args.get("file")
    filepath = os.path.join("/var/uploads", filename)
    return send_file(filepath)
'''),

    TestCase("path-py-02", "CWE-22", "python", True,
             "open() with f-string path from request",
             '''
from flask import Flask, request

app = Flask(__name__)

@app.route("/read")
def read_file():
    name = request.args.get("name")
    with open(f"/data/reports/{name}") as f:
        content = f.read()
    return {"content": content}
'''),

    TestCase("path-py-03", "CWE-22", "python", True,
             "os.path.join with request data (absolute path bypass)",
             '''
from flask import Flask, request
import os

app = Flask(__name__)
UPLOAD_DIR = "/var/app/uploads"

@app.route("/file/<path:filename>")
def serve_file(filename):
    full_path = os.path.join(UPLOAD_DIR, request.args.get("subdir", ""), filename)
    with open(full_path) as f:
        return f.read()
'''),

    TestCase("path-py-04", "CWE-22", "python", True,
             "Django view reading file from param",
             '''
from django.http import FileResponse
from pathlib import Path

def download_attachment(request, doc_id):
    filename = request.GET.get("name", "report.pdf")
    base = Path("/srv/documents")
    file_path = base / filename
    return FileResponse(open(str(file_path), "rb"))
'''),

    TestCase("path-js-01", "CWE-22", "javascript", True,
             "Express readFile with req.params",
             '''
const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();

app.get('/files/:name', (req, res) => {
    const filePath = path.join(__dirname, 'uploads', req.params.name);
    fs.readFile(filePath, 'utf8', (err, data) => {
        if (err) return res.status(404).send('Not found');
        res.send(data);
    });
});
'''),

    TestCase("path-js-02", "CWE-22", "javascript", True,
             "createReadStream with query parameter",
             '''
const fs = require('fs');
const path = require('path');

app.get('/download', (req, res) => {
    const filename = req.query.file;
    const stream = fs.createReadStream('./public/docs/' + filename);
    stream.pipe(res);
});
'''),

    # --- SAFE ---
    TestCase("path-py-safe-01", "CWE-22", "python", False,
             "Validated path with realpath check",
             '''
from flask import Flask, request, abort
import os

app = Flask(__name__)
UPLOAD_DIR = os.path.realpath("/var/uploads")

@app.route("/download")
def download():
    filename = request.args.get("file")
    filepath = os.path.realpath(os.path.join(UPLOAD_DIR, filename))
    if not filepath.startswith(UPLOAD_DIR):
        abort(403)
    return send_file(filepath)
'''),

    TestCase("path-js-safe-01", "CWE-22", "javascript", False,
             "Path resolved and validated",
             '''
const express = require('express');
const path = require('path');
const fs = require('fs');

const SAFE_DIR = path.resolve(__dirname, 'uploads');

app.get('/files/:name', (req, res) => {
    const requested = path.resolve(SAFE_DIR, req.params.name);
    if (!requested.startsWith(SAFE_DIR)) {
        return res.status(403).send('Forbidden');
    }
    res.sendFile(requested);
});
'''),
]


# =====================================================================
# CWE-502: Insecure Deserialization
# =====================================================================

CWE502_TESTS: List[TestCase] = [
    TestCase("deser-py-01", "CWE-502", "python", True,
             "pickle.loads on request data",
             '''
from flask import Flask, request
import pickle
import base64

app = Flask(__name__)

@app.route("/api/restore", methods=["POST"])
def restore_session():
    data = base64.b64decode(request.form.get("session"))
    session_data = pickle.loads(data)
    return {"user": session_data.get("username")}
'''),

    TestCase("deser-py-02", "CWE-502", "python", True,
             "pickle.load from uploaded file",
             '''
import pickle

def load_model(file_path):
    """Load a trained ML model from disk."""
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

# file_path comes from user upload
'''),

    TestCase("deser-py-03", "CWE-502", "python", True,
             "yaml.load without SafeLoader",
             '''
import yaml

def parse_config(config_text):
    """Parse YAML configuration from user input."""
    config = yaml.load(config_text)
    return config

# Called from API with user-provided YAML
'''),

    TestCase("deser-py-04", "CWE-502", "python", True,
             "shelve.open on user-provided path",
             '''
import shelve

def get_cached_data(cache_name):
    db = shelve.open(f"/tmp/cache/{cache_name}")
    data = dict(db)
    db.close()
    return data
'''),

    TestCase("deser-js-01", "CWE-502", "javascript", True,
             "node-serialize unserialize()",
             '''
const serialize = require('node-serialize');
const express = require('express');
const app = express();

app.post('/api/profile', (req, res) => {
    const cookie = req.cookies.profile;
    const profile = serialize.unserialize(
        Buffer.from(cookie, 'base64').toString()
    );
    res.json(profile);
});
'''),

    TestCase("deser-js-02", "CWE-502", "javascript", True,
             "JSON.parse + eval for computed fields",
             '''
app.post('/api/compute', (req, res) => {
    const payload = JSON.parse(req.body.data);
    const result = eval(payload.expression);
    res.json({ result });
});
'''),

    # --- SAFE ---
    TestCase("deser-py-safe-01", "CWE-502", "python", False,
             "json.loads (safe deserialization)",
             '''
import json
from flask import request

@app.route("/api/data", methods=["POST"])
def receive_data():
    data = json.loads(request.get_data())
    return {"received": len(data)}
'''),

    TestCase("deser-py-safe-02", "CWE-502", "python", False,
             "yaml.safe_load (safe YAML)",
             '''
import yaml

def parse_config(config_text):
    config = yaml.safe_load(config_text)
    return config
'''),

    TestCase("deser-js-safe-01", "CWE-502", "javascript", False,
             "JSON.parse only (no eval)",
             '''
app.post('/api/data', (req, res) => {
    const data = JSON.parse(req.body.payload);
    res.json({ name: data.name, count: data.items.length });
});
'''),
]


# =====================================================================
# CWE-918: SSRF
# =====================================================================

CWE918_TESTS: List[TestCase] = [
    TestCase("ssrf-py-01", "CWE-918", "python", True,
             "requests.get with user-provided URL",
             '''
from flask import Flask, request
import requests

app = Flask(__name__)

@app.route("/proxy")
def proxy():
    url = request.args.get("url")
    response = requests.get(url)
    return response.text
'''),

    TestCase("ssrf-py-02", "CWE-918", "python", True,
             "urllib.request.urlopen with user URL",
             '''
from flask import Flask, request
import urllib.request

app = Flask(__name__)

@app.route("/fetch")
def fetch_url():
    target = request.args.get("target")
    resp = urllib.request.urlopen(target)
    return resp.read().decode()
'''),

    TestCase("ssrf-py-03", "CWE-918", "python", True,
             "httpx with f-string URL construction",
             '''
import httpx
from fastapi import FastAPI

app = FastAPI()

@app.get("/webhook")
async def trigger_webhook(url: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json={"event": "test"})
    return {"status_code": resp.status_code}
'''),

    TestCase("ssrf-py-04", "CWE-918", "python", True,
             "Hardcoded request to cloud metadata",
             '''
import requests

def get_instance_metadata():
    """Get AWS EC2 instance metadata."""
    url = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
    response = requests.get(url)
    return response.json()
'''),

    TestCase("ssrf-js-01", "CWE-918", "javascript", True,
             "fetch() with user-provided URL",
             '''
const express = require('express');
const app = express();

app.get('/api/preview', async (req, res) => {
    const url = req.query.url;
    const response = await fetch(url);
    const html = await response.text();
    res.send(html);
});
'''),

    TestCase("ssrf-js-02", "CWE-918", "javascript", True,
             "axios with req.body URL",
             '''
const axios = require('axios');

app.post('/api/import', async (req, res) => {
    const url = req.body.import_url;
    const { data } = await axios.get(url);
    await processImport(data);
    res.json({ success: true });
});
'''),

    # --- SAFE ---
    TestCase("ssrf-py-safe-01", "CWE-918", "python", False,
             "URL validated against allowlist",
             '''
from flask import Flask, request
import requests
from urllib.parse import urlparse

ALLOWED_HOSTS = {"api.example.com", "cdn.example.com"}

@app.route("/proxy")
def proxy():
    url = request.args.get("url")
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_HOSTS:
        return {"error": "Host not allowed"}, 403
    response = requests.get(url)
    return response.text
'''),

    TestCase("ssrf-js-safe-01", "CWE-918", "javascript", False,
             "URL validated with URL constructor",
             '''
const ALLOWED = ['https://api.example.com', 'https://cdn.example.com'];

app.get('/api/preview', async (req, res) => {
    const url = new URL(req.query.url);
    if (!ALLOWED.some(a => url.origin === new URL(a).origin)) {
        return res.status(403).json({ error: 'Host not allowed' });
    }
    const response = await fetch(url.href);
    res.send(await response.text());
});
'''),
]


# =====================================================================
# CWE-798: Hardcoded Secrets
# =====================================================================

CWE798_TESTS: List[TestCase] = [
    TestCase("secret-py-01", "CWE-798", "python", True,
             "Hardcoded database password",
             '''
import psycopg2

DB_HOST = "prod-db.company.internal"
DB_USER = "admin"
DB_PASSWORD = "SuperSecr3t!2024"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
        dbname="production"
    )
'''),

    TestCase("secret-py-02", "CWE-798", "python", True,
             "Hardcoded API key in config",
             '''
# Application configuration
API_KEY = "openai-key-goes-here-replace-in-production"
DATABASE_URL = "postgres://admin:password123@db.example.com:5432/myapp"

class Config:
    SECRET_KEY = "my-super-secret-key-for-jwt-signing"
    STRIPE_API_KEY = "stripe-live-key-goes-here-replace-in-production"
'''),

    TestCase("secret-py-03", "CWE-798", "python", True,
             "AWS credentials in source",
             '''
import boto3

# AWS Configuration
AWS_ACCESS_KEY_ID = "AKIA-EXAMPLE-REPLACE-WITH-REAL"
AWS_SECRET_ACCESS_KEY = "aws-secret-key-replace-with-real-value"

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)
'''),

    TestCase("secret-py-04", "CWE-798", "python", True,
             "Hardcoded JWT secret",
             '''
import jwt

JWT_SECRET = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9secretkey"

def create_token(user_id):
    return jwt.encode({"user_id": user_id}, JWT_SECRET, algorithm="HS256")
'''),

    TestCase("secret-py-05", "CWE-798", "python", True,
             "Connection string with password",
             '''
from sqlalchemy import create_engine

engine = create_engine("postgres://dbuser:MyP@ssw0rd!@prod-db.internal:5432/main")
'''),

    TestCase("secret-js-01", "CWE-798", "javascript", True,
             "Hardcoded API credentials in Node.js",
             '''
const stripe = require('stripe');

const STRIPE_SECRET_KEY = "sk_live_REPLACE_WITH_REAL_KEY_EXAMPLE";
const api_key = "ghp_REPLACE_WITH_REAL_TOKEN_EXAMPLE";

const client = stripe(STRIPE_SECRET_KEY);
'''),

    TestCase("secret-js-02", "CWE-798", "javascript", True,
             "MongoDB connection with embedded password",
             '''
const mongoose = require('mongoose');

const MONGO_URI = "mongodb://admin:secretpassword@production-mongo.internal:27017/myapp";

mongoose.connect(MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true
});
'''),

    TestCase("secret-js-03", "CWE-798", "javascript", True,
             "Private key embedded in source",
             '''
const jwt = require('jsonwebtoken');

const PRIVATE_KEY = `-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGcY5unfnJhBAkMGJw=
-----END RSA PRIVATE KEY-----`;

function signToken(payload) {
    return jwt.sign(payload, PRIVATE_KEY, { algorithm: 'RS256' });
}
'''),

    # --- SAFE ---
    TestCase("secret-py-safe-01", "CWE-798", "python", False,
             "Environment variable for secrets",
             '''
import os
import psycopg2

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
API_KEY = os.getenv("API_KEY")

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, password=DB_PASSWORD, dbname="app"
    )
'''),

    TestCase("secret-py-safe-02", "CWE-798", "python", False,
             "Placeholder secrets (not real)",
             '''
# Example configuration - replace with environment variables
password = "changeme"
api_key = "your-api-key-here"
secret = "REPLACE_WITH_REAL_SECRET"
'''),

    TestCase("secret-js-safe-01", "CWE-798", "javascript", False,
             "process.env for credentials",
             '''
const mongoose = require('mongoose');

const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost/dev";

mongoose.connect(MONGO_URI, {
    useNewUrlParser: true,
});
'''),

    TestCase("secret-js-safe-02", "CWE-798", "javascript", False,
             "dotenv for secret management",
             '''
require('dotenv').config();

const config = {
    jwtSecret: process.env.JWT_SECRET,
    stripeKey: process.env.STRIPE_KEY,
    dbUrl: process.env.DATABASE_URL,
};

module.exports = config;
'''),
]


# =====================================================================
# ALL TEST CASES
# =====================================================================

ALL_TEST_CASES: Dict[str, List[TestCase]] = {
    "CWE-89": CWE89_TESTS,
    "CWE-77": CWE77_TESTS,
    "CWE-22": CWE22_TESTS,
    "CWE-502": CWE502_TESTS,
    "CWE-918": CWE918_TESTS,
    "CWE-798": CWE798_TESTS,
}


def get_all_tests() -> List[TestCase]:
    """Get all test cases as a flat list."""
    tests = []
    for cwe_tests in ALL_TEST_CASES.values():
        tests.extend(cwe_tests)
    return tests


def get_vulnerable_count(cwe: str) -> int:
    return sum(1 for t in ALL_TEST_CASES.get(cwe, []) if t.is_vulnerable)


def get_safe_count(cwe: str) -> int:
    return sum(1 for t in ALL_TEST_CASES.get(cwe, []) if not t.is_vulnerable)
