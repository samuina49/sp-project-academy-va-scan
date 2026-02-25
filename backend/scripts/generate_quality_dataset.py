"""
High-Quality Vulnerability Dataset Generator
=============================================
Generates a comprehensive Python + JavaScript vulnerability dataset with CLEAR,
learnable vulnerability patterns across multiple CWE categories.

Each sample is a (vulnerable_code, fixed_code) pair, ensuring the labels are
correct and the patterns are discriminative.

Target: 4000+ samples (half vulnerable, half safe) across 10 categories.

Categories:
1. CWE-89:  SQL Injection
2. CWE-79:  Cross-Site Scripting (XSS)
3. CWE-78:  OS Command Injection  
4. CWE-22:  Path Traversal
5. CWE-502: Insecure Deserialization
6. CWE-918: Server-Side Request Forgery (SSRF)
7. CWE-798: Hard-coded Credentials
8. CWE-327: Broken Cryptography
9. CWE-611: XML External Entity (XXE)
10. CWE-94: Code Injection

Author: Senior Project - AI Vulnerability Scanner
"""

import json
import random
import os
from pathlib import Path
from itertools import product

random.seed(42)

# =============================================================================
# TEMPLATE SYSTEM: Each category has multiple template families
# Each family produces (vulnerable_code, safe_code) pairs with variations
# =============================================================================

def make_variations(template: str, var_map: dict, n: int = 10) -> list:
    """Generate n variations of a template by substituting variable placeholders."""
    results = []
    for _ in range(n):
        code = template
        for placeholder, options in var_map.items():
            code = code.replace(placeholder, random.choice(options))
        results.append(code)
    return results


# =============================================
# CWE-89: SQL Injection (Python)
# =============================================
SQL_INJECTION_VULN_PY = [
    # Family 1: f-string interpolation
    '''import sqlite3

def {FUNC}({PARAM}):
    """Search for records by {FIELD}."""
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    query = f"SELECT * FROM {TABLE} WHERE {FIELD} = '{{{PARAM}}}'"
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    return results
''',
    # Family 2: string concatenation
    '''import sqlite3

def {FUNC}({PARAM}):
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    sql = "SELECT * FROM {TABLE} WHERE {FIELD} = '" + {PARAM} + "'"
    cursor.execute(sql)
    return cursor.fetchall()
''',
    # Family 3: format string  
    '''import mysql.connector

def {FUNC}({PARAM}, db_conn):
    cursor = db_conn.cursor()
    query = "SELECT {FIELD} FROM {TABLE} WHERE id = %s" % {PARAM}
    cursor.execute(query)
    return cursor.fetchone()
''',
    # Family 4: multi-field injection
    '''import sqlite3

def {FUNC}({PARAM}, {PARAM2}):
    db = sqlite3.connect("{DB}")
    cursor = db.cursor()
    sql = f"SELECT * FROM {TABLE} WHERE {FIELD} = '{{{PARAM}}}' AND status = '{{{PARAM2}}}'"
    cursor.execute(sql)
    results = cursor.fetchall()
    db.close()
    return results
''',
    # Family 5: INSERT injection
    '''import sqlite3

def {FUNC}({PARAM}, {PARAM2}):
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    query = f"INSERT INTO {TABLE} ({FIELD}, name) VALUES ('{{{PARAM}}}', '{{{PARAM2}}}')"
    cursor.execute(query)
    conn.commit()
    conn.close()
''',
    # Family 6: DELETE injection
    '''import sqlite3

def {FUNC}({PARAM}):
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    sql = "DELETE FROM {TABLE} WHERE {FIELD} = '" + {PARAM} + "'"
    cursor.execute(sql)
    conn.commit()
''',
]

SQL_INJECTION_SAFE_PY = [
    # Family 1: Parameterized query
    '''import sqlite3

def {FUNC}({PARAM}):
    """Search for records by {FIELD} safely."""
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    query = "SELECT * FROM {TABLE} WHERE {FIELD} = ?"
    cursor.execute(query, ({PARAM},))
    results = cursor.fetchall()
    conn.close()
    return results
''',
    # Family 2: Parameterized with tuple
    '''import sqlite3

def {FUNC}({PARAM}):
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    sql = "SELECT * FROM {TABLE} WHERE {FIELD} = ?"
    cursor.execute(sql, ({PARAM},))
    return cursor.fetchall()
''',
    # Family 3: MySQL parameterized
    '''import mysql.connector

def {FUNC}({PARAM}, db_conn):
    cursor = db_conn.cursor()
    query = "SELECT {FIELD} FROM {TABLE} WHERE id = %s"
    cursor.execute(query, ({PARAM},))
    return cursor.fetchone()
''',
    # Family 4: Multi-param safe
    '''import sqlite3

def {FUNC}({PARAM}, {PARAM2}):
    db = sqlite3.connect("{DB}")
    cursor = db.cursor()
    sql = "SELECT * FROM {TABLE} WHERE {FIELD} = ? AND status = ?"
    cursor.execute(sql, ({PARAM}, {PARAM2}))
    results = cursor.fetchall()
    db.close()
    return results
''',
    # Family 5: Safe INSERT
    '''import sqlite3

def {FUNC}({PARAM}, {PARAM2}):
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    query = "INSERT INTO {TABLE} ({FIELD}, name) VALUES (?, ?)"
    cursor.execute(query, ({PARAM}, {PARAM2}))
    conn.commit()
    conn.close()
''',
    # Family 6: Safe DELETE
    '''import sqlite3

def {FUNC}({PARAM}):
    conn = sqlite3.connect("{DB}")
    cursor = conn.cursor()
    sql = "DELETE FROM {TABLE} WHERE {FIELD} = ?"
    cursor.execute(sql, ({PARAM},))
    conn.commit()
''',
]

SQL_VARS = {
    '{FUNC}': ['search_user', 'find_record', 'get_data', 'lookup_entry', 'query_db',
                'fetch_user', 'get_account', 'find_product', 'search_order', 'get_employee'],
    '{PARAM}': ['user_input', 'search_term', 'username', 'user_id', 'query_param',
                'name', 'email', 'record_id', 'product_name', 'account_id'],
    '{PARAM2}': ['status', 'role', 'category', 'department', 'region'],
    '{TABLE}': ['users', 'accounts', 'products', 'orders', 'employees', 'customers'],
    '{FIELD}': ['username', 'email', 'name', 'id', 'account_number', 'product_id'],
    '{DB}': ['app.db', 'database.sqlite', 'production.db', 'main.db', 'data.db'],
}

# =============================================
# CWE-79: XSS (JavaScript)
# =============================================
XSS_VULN_JS = [
    # innerHTML
    '''function {FUNC}({PARAM}) {{
    const element = document.getElementById("{ELEM}");
    element.innerHTML = {PARAM};
}}
''',
    # document.write
    '''function {FUNC}({PARAM}) {{
    document.write("<div class='{CLASS}'>" + {PARAM} + "</div>");
}}
''',
    # jQuery html
    '''function {FUNC}({PARAM}) {{
    $("#{ELEM}").html({PARAM});
}}
''',
    # Dynamic URL with user input
    '''function {FUNC}() {{
    const {PARAM} = new URLSearchParams(window.location.search).get("{QPARAM}");
    document.getElementById("{ELEM}").innerHTML = "<span>" + {PARAM} + "</span>";
}}
''',
    # Template literal injection
    '''function {FUNC}({PARAM}) {{
    const container = document.querySelector(".{CLASS}");
    container.innerHTML = `<div>${{{PARAM}}}</div>`;
}}
''',
    # outerHTML
    '''function {FUNC}({PARAM}) {{
    const el = document.getElementById("{ELEM}");
    el.outerHTML = "<div>" + {PARAM} + "</div>";
}}
''',
]

XSS_SAFE_JS = [
    # textContent
    '''function {FUNC}({PARAM}) {{
    const element = document.getElementById("{ELEM}");
    element.textContent = {PARAM};
}}
''',
    # createElement
    '''function {FUNC}({PARAM}) {{
    const div = document.createElement("div");
    div.classList.add("{CLASS}");
    div.textContent = {PARAM};
    document.body.appendChild(div);
}}
''',
    # jQuery text
    '''function {FUNC}({PARAM}) {{
    $("#{ELEM}").text({PARAM});
}}
''',
    # Safe URL param handling
    '''function {FUNC}() {{
    const {PARAM} = new URLSearchParams(window.location.search).get("{QPARAM}");
    const safeText = document.createTextNode({PARAM} || "");
    document.getElementById("{ELEM}").appendChild(safeText);
}}
''',
    # DOMPurify
    '''function {FUNC}({PARAM}) {{
    const container = document.querySelector(".{CLASS}");
    const sanitized = DOMPurify.sanitize({PARAM});
    container.innerHTML = sanitized;
}}
''',
    # Safe attribute setting  
    '''function {FUNC}({PARAM}) {{
    const el = document.getElementById("{ELEM}");
    el.setAttribute("data-value", encodeURIComponent({PARAM}));
    el.textContent = {PARAM};
}}
''',
]

XSS_VARS = {
    '{FUNC}': ['renderComment', 'displayMessage', 'showResult', 'updateContent',
               'renderUserInput', 'displayName', 'showNotification', 'renderProfile'],
    '{PARAM}': ['userInput', 'content', 'message', 'text', 'data', 'value', 'comment'],
    '{ELEM}': ['output', 'content', 'result', 'message-box', 'display', 'container'],
    '{CLASS}': ['comment', 'message', 'notification', 'alert', 'result-item'],
    '{QPARAM}': ['q', 'search', 'name', 'msg', 'input'],
}

# =============================================
# CWE-78: OS Command Injection (Python)
# =============================================
CMD_INJECTION_VULN_PY = [
    '''import os

def {FUNC}({PARAM}):
    os.system(f"{CMD} {{{PARAM}}}")
''',
    '''import os

def {FUNC}({PARAM}):
    command = "{CMD} " + {PARAM}
    os.system(command)
    return True
''',
    '''import os

def {FUNC}({PARAM}):
    result = os.popen(f"{CMD} {{{PARAM}}}").read()
    return result
''',
    '''import subprocess

def {FUNC}({PARAM}):
    cmd = f"{CMD} {{{PARAM}}}"
    result = subprocess.call(cmd, shell=True)
    return result
''',
    '''import os

def {FUNC}({PARAM}, {PARAM2}):
    os.system(f"{CMD} {{{PARAM}}} {{{PARAM2}}}")
''',
]

CMD_INJECTION_SAFE_PY = [
    '''import subprocess

def {FUNC}({PARAM}):
    subprocess.run(["{CMD}", {PARAM}], check=True)
''',
    '''import subprocess

def {FUNC}({PARAM}):
    result = subprocess.run(
        ["{CMD}", {PARAM}],
        capture_output=True, text=True, check=True
    )
    return result.stdout
''',
    '''import shlex
import subprocess

def {FUNC}({PARAM}):
    safe_param = shlex.quote({PARAM})
    result = subprocess.run(["{CMD}", safe_param], capture_output=True)
    return result
''',
    '''import subprocess

def {FUNC}({PARAM}):
    cmd = ["{CMD}", {PARAM}]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout
''',
    '''import subprocess
import shlex

def {FUNC}({PARAM}, {PARAM2}):
    subprocess.run(["{CMD}", {PARAM}, {PARAM2}], check=True)
''',
]

CMD_VARS = {
    '{FUNC}': ['process_file', 'run_backup', 'compress_data', 'convert_file',
               'deploy_app', 'clean_logs', 'sync_data', 'generate_report'],
    '{PARAM}': ['filename', 'filepath', 'target', 'input_path', 'source',
                'directory', 'archive_name', 'log_file'],
    '{PARAM2}': ['destination', 'output_path', 'backup_dir'],
    '{CMD}': ['cat', 'ls', 'cp', 'mv', 'tar', 'gzip', 'find', 'grep'],
}

# =============================================
# CWE-22: Path Traversal (Python)
# =============================================
PATH_TRAVERSAL_VULN_PY = [
    '''def {FUNC}({PARAM}):
    filepath = os.path.join("{DIR}", {PARAM})
    with open(filepath, "r") as f:
        return f.read()
''',
    '''from flask import request, send_file

@app.route("/{ROUTE}")
def {FUNC}():
    {PARAM} = request.args.get("{QPARAM}")
    return send_file(os.path.join("{DIR}", {PARAM}))
''',
    '''def {FUNC}({PARAM}):
    path = "{DIR}/" + {PARAM}
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None
''',
    '''import os

def {FUNC}({PARAM}):
    full_path = os.path.join("{DIR}", {PARAM})
    with open(full_path, "rb") as f:
        data = f.read()
    return data
''',
]

PATH_TRAVERSAL_SAFE_PY = [
    '''def {FUNC}({PARAM}):
    safe_name = os.path.basename({PARAM})
    filepath = os.path.join("{DIR}", safe_name)
    with open(filepath, "r") as f:
        return f.read()
''',
    '''from flask import request, send_file, abort

@app.route("/{ROUTE}")
def {FUNC}():
    {PARAM} = request.args.get("{QPARAM}")
    safe_path = os.path.realpath(os.path.join("{DIR}", {PARAM}))
    if not safe_path.startswith(os.path.realpath("{DIR}")):
        abort(403)
    return send_file(safe_path)
''',
    '''def {FUNC}({PARAM}):
    safe_name = os.path.basename({PARAM})
    if ".." in safe_name or "/" in safe_name:
        raise ValueError("Invalid filename")
    path = os.path.join("{DIR}", safe_name)
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None
''',
    '''import os

def {FUNC}({PARAM}):
    base_dir = os.path.realpath("{DIR}")
    full_path = os.path.realpath(os.path.join(base_dir, {PARAM}))
    if not full_path.startswith(base_dir):
        raise PermissionError("Access denied")
    with open(full_path, "rb") as f:
        data = f.read()
    return data
''',
]

PATH_VARS = {
    '{FUNC}': ['read_file', 'get_document', 'download_file', 'serve_file',
               'load_template', 'get_attachment', 'read_config', 'get_image'],
    '{PARAM}': ['filename', 'filepath', 'document_name', 'file_path', 'template_name'],
    '{DIR}': ['uploads', 'documents', 'static/files', 'data', 'templates', 'assets'],
    '{ROUTE}': ['download', 'files', 'documents', 'attachments', 'static'],
    '{QPARAM}': ['file', 'name', 'path', 'doc', 'template'],
}

# =============================================
# CWE-502: Insecure Deserialization (Python)
# =============================================
DESER_VULN_PY = [
    '''import pickle

def {FUNC}({PARAM}):
    data = pickle.loads({PARAM})
    return data
''',
    '''import pickle

def {FUNC}({PARAM}):
    with open({PARAM}, "rb") as f:
        obj = pickle.load(f)
    return obj
''',
    '''import yaml

def {FUNC}({PARAM}):
    config = yaml.load({PARAM})
    return config
''',
    '''import marshal

def {FUNC}({PARAM}):
    code = marshal.loads({PARAM})
    return code
''',
    '''import shelve

def {FUNC}({PARAM}):
    db = shelve.open({PARAM})
    data = dict(db)
    db.close()
    return data
''',
]

DESER_SAFE_PY = [
    '''import json

def {FUNC}({PARAM}):
    data = json.loads({PARAM})
    return data
''',
    '''import json

def {FUNC}({PARAM}):
    with open({PARAM}, "r") as f:
        obj = json.load(f)
    return obj
''',
    '''import yaml

def {FUNC}({PARAM}):
    config = yaml.safe_load({PARAM})
    return config
''',
    '''import ast

def {FUNC}({PARAM}):
    data = ast.literal_eval({PARAM})
    return data
''',
    '''import json

def {FUNC}({PARAM}):
    with open({PARAM}, "r") as f:
        data = json.load(f)
    return data
''',
]

DESER_VARS = {
    '{FUNC}': ['load_data', 'parse_config', 'deserialize', 'read_object',
               'load_session', 'restore_state', 'import_data', 'parse_payload'],
    '{PARAM}': ['raw_data', 'payload', 'serialized', 'data_bytes', 'config_path',
                'session_data', 'state_file', 'input_data'],
}

# =============================================
# CWE-918: SSRF (Python)
# =============================================
SSRF_VULN_PY = [
    '''import requests

def {FUNC}({PARAM}):
    response = requests.get({PARAM})
    return response.text
''',
    '''import urllib.request

def {FUNC}({PARAM}):
    response = urllib.request.urlopen({PARAM})
    return response.read().decode()
''',
    '''import requests
from flask import request

@app.route("/{ROUTE}")
def {FUNC}():
    {PARAM} = request.args.get("url")
    resp = requests.get({PARAM}, timeout=10)
    return resp.json()
''',
    '''import requests

def {FUNC}({PARAM}):
    api_url = f"http://{{{PARAM}}}/api/data"
    response = requests.get(api_url)
    return response.json()
''',
]

SSRF_SAFE_PY = [
    '''import requests
from urllib.parse import urlparse

ALLOWED_HOSTS = ["api.example.com", "cdn.example.com"]

def {FUNC}({PARAM}):
    parsed = urlparse({PARAM})
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("URL not in allowlist")
    response = requests.get({PARAM})
    return response.text
''',
    '''import requests
from urllib.parse import urlparse
import ipaddress

def {FUNC}({PARAM}):
    parsed = urlparse({PARAM})
    hostname = parsed.hostname
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback:
            raise ValueError("Internal addresses not allowed")
    except ValueError:
        pass
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid scheme")
    response = requests.get({PARAM}, timeout=5)
    return response.text
''',
    '''import requests
from urllib.parse import urlparse

ALLOWED_DOMAINS = ["api.example.com", "data.example.com"]

@app.route("/{ROUTE}")
def {FUNC}():
    {PARAM} = request.args.get("url")
    parsed = urlparse({PARAM})
    if parsed.hostname not in ALLOWED_DOMAINS:
        return "Forbidden", 403
    resp = requests.get({PARAM}, timeout=10)
    return resp.json()
''',
    '''import requests
from urllib.parse import urlparse

ALLOWED_HOSTS = ["api.trusted.com"]

def {FUNC}({PARAM}):
    parsed = urlparse(f"http://{{{PARAM}}}/api/data")
    if parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("Host not allowed")
    response = requests.get(parsed.geturl())
    return response.json()
''',
]

SSRF_VARS = {
    '{FUNC}': ['fetch_url', 'proxy_request', 'get_remote_data', 'download_resource',
               'check_service', 'get_webhook_data', 'fetch_api', 'load_remote'],
    '{PARAM}': ['url', 'target_url', 'remote_url', 'endpoint', 'api_url', 'host'],
    '{ROUTE}': ['proxy', 'fetch', 'remote', 'webhook', 'callback'],
}

# =============================================
# CWE-798: Hard-coded Credentials (Python)
# =============================================
HARDCODED_VULN_PY = [
    '''import requests

API_KEY = "{SECRET}"

def {FUNC}():
    headers = {{"Authorization": f"Bearer {{API_KEY}}"}}
    response = requests.get("{API_URL}", headers=headers)
    return response.json()
''',
    '''DB_PASSWORD = "{SECRET}"

def {FUNC}():
    import psycopg2
    conn = psycopg2.connect(
        host="localhost",
        database="production",
        user="admin",
        password=DB_PASSWORD
    )
    return conn
''',
    '''SECRET_KEY = "{SECRET}"

def {FUNC}(data):
    import jwt
    token = jwt.encode(data, SECRET_KEY, algorithm="HS256")
    return token
''',
    '''AWS_ACCESS_KEY = "AKIA{SECRET}"
AWS_SECRET_KEY = "{SECRET}"

def {FUNC}():
    import boto3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    return s3
''',
]

HARDCODED_SAFE_PY = [
    '''import os
import requests

def {FUNC}():
    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not set")
    headers = {{"Authorization": f"Bearer {{api_key}}"}}
    response = requests.get("{API_URL}", headers=headers)
    return response.json()
''',
    '''import os

def {FUNC}():
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD")
    )
    return conn
''',
    '''import os

def {FUNC}(data):
    import jwt
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY not configured")
    token = jwt.encode(data, secret_key, algorithm="HS256")
    return token
''',
    '''import os
import boto3

def {FUNC}():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    return s3
''',
]

HARDCODED_VARS = {
    '{FUNC}': ['get_api_data', 'connect_database', 'create_token', 'get_storage_client',
               'authenticate', 'init_service', 'setup_connection', 'get_credentials'],
    '{SECRET}': ['sk_live_abc123def456', 'supersecretpassword123', 'mysecretkey2024',
                 'p@ssw0rd!', 'admin123', 'secret_token_xyz'],
    '{API_URL}': ['https://api.example.com/v1/data', 'https://api.service.com/users',
                  'https://api.platform.io/resources'],
}

# =============================================
# CWE-327: Broken Cryptography (Python)
# =============================================
CRYPTO_VULN_PY = [
    '''import hashlib

def {FUNC}({PARAM}):
    return hashlib.md5({PARAM}.encode()).hexdigest()
''',
    '''import hashlib

def {FUNC}({PARAM}):
    return hashlib.sha1({PARAM}.encode()).hexdigest()
''',
    '''from Crypto.Cipher import DES

def {FUNC}({PARAM}, key):
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt({PARAM})
''',
    '''import base64

def {FUNC}({PARAM}):
    """Encrypt data using base64 encoding."""
    return base64.b64encode({PARAM}.encode()).decode()
''',
    '''import random

def {FUNC}(length=16):
    """Generate a random token."""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(chars) for _ in range(length))
''',
]

CRYPTO_SAFE_PY = [
    '''import hashlib

def {FUNC}({PARAM}):
    salt = hashlib.sha256(os.urandom(32)).hexdigest()
    return hashlib.pbkdf2_hmac("sha256", {PARAM}.encode(), salt.encode(), 100000).hex()
''',
    '''import bcrypt

def {FUNC}({PARAM}):
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw({PARAM}.encode(), salt)
''',
    '''from cryptography.fernet import Fernet

def {FUNC}({PARAM}, key):
    cipher = Fernet(key)
    return cipher.encrypt({PARAM}.encode())
''',
    '''from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

def {FUNC}({PARAM}, key, iv):
    cipher = Cipher(algorithms.AES(key), modes.GCM(iv))
    encryptor = cipher.encryptor()
    return encryptor.update({PARAM}) + encryptor.finalize()
''',
    '''import secrets

def {FUNC}(length=32):
    """Generate a cryptographically secure random token."""
    return secrets.token_hex(length)
''',
]

CRYPTO_VARS = {
    '{FUNC}': ['hash_password', 'encrypt_data', 'generate_token', 'secure_hash',
               'create_digest', 'protect_data', 'encode_secret', 'hash_value'],
    '{PARAM}': ['password', 'plaintext', 'data', 'secret', 'message', 'value'],
}

# =============================================
# CWE-611: XXE (Python)
# =============================================
XXE_VULN_PY = [
    '''from xml.etree.ElementTree import parse

def {FUNC}({PARAM}):
    tree = parse({PARAM})
    root = tree.getroot()
    return root
''',
    '''from lxml import etree

def {FUNC}({PARAM}):
    parser = etree.XMLParser()
    doc = etree.parse({PARAM}, parser)
    return doc.getroot()
''',
    '''import xml.sax

def {FUNC}({PARAM}):
    handler = xml.sax.ContentHandler()
    xml.sax.parseString({PARAM}, handler)
    return handler
''',
]

XXE_SAFE_PY = [
    '''from defusedxml.ElementTree import parse

def {FUNC}({PARAM}):
    tree = parse({PARAM})
    root = tree.getroot()
    return root
''',
    '''from lxml import etree

def {FUNC}({PARAM}):
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    doc = etree.parse({PARAM}, parser)
    return doc.getroot()
''',
    '''import defusedxml.sax

def {FUNC}({PARAM}):
    handler = defusedxml.sax.ContentHandler()
    defusedxml.sax.parseString({PARAM}, handler)
    return handler
''',
]

XXE_VARS = {
    '{FUNC}': ['parse_xml', 'load_config', 'read_xml_data', 'parse_document',
               'import_xml', 'process_xml_file', 'parse_feed'],
    '{PARAM}': ['xml_file', 'xml_path', 'xml_data', 'config_file', 'document_path'],
}

# =============================================
# CWE-94: Code Injection (Python)
# =============================================
CODE_INJECTION_VULN_PY = [
    '''def {FUNC}({PARAM}):
    result = eval({PARAM})
    return result
''',
    '''def {FUNC}({PARAM}):
    exec({PARAM})
''',
    '''def {FUNC}({PARAM}):
    code = compile({PARAM}, "<string>", "exec")
    exec(code)
''',
    '''import importlib

def {FUNC}({PARAM}):
    module = __import__({PARAM})
    return module
''',
]

CODE_INJECTION_SAFE_PY = [
    '''import ast

def {FUNC}({PARAM}):
    tree = ast.parse({PARAM}, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Attribute)):
            raise ValueError("Function calls not allowed")
    result = ast.literal_eval({PARAM})
    return result
''',
    '''import ast

def {FUNC}({PARAM}):
    result = ast.literal_eval({PARAM})
    return result
''',
    '''ALLOWED_OPS = {{"+", "-", "*", "/"}}

def {FUNC}({PARAM}):
    import re
    if not re.match(r"^[\\d\\s+\\-*/().]+$", {PARAM}):
        raise ValueError("Invalid expression")
    return eval({PARAM})
''',
    '''ALLOWED_MODULES = {{"math", "json", "datetime"}}

def {FUNC}({PARAM}):
    if {PARAM} not in ALLOWED_MODULES:
        raise ValueError(f"Module {{{{PARAM}}}} not allowed")
    import importlib
    return importlib.import_module({PARAM})
''',
]

CODE_INJECTION_VARS = {
    '{FUNC}': ['execute_expression', 'run_code', 'evaluate', 'process_input',
               'dynamic_import', 'calc_expression', 'run_script', 'eval_formula'],
    '{PARAM}': ['expression', 'user_code', 'formula', 'code_string',
                'module_name', 'script', 'calc_input', 'dynamic_code'],
}

# =============================================
# XSS Additional (Python/Flask)
# =============================================
XSS_VULN_PY = [
    '''from flask import request, make_response

@app.route("/{ROUTE}")
def {FUNC}():
    name = request.args.get("name", "")
    return f"<h1>Hello {{name}}!</h1>"
''',
    '''from flask import request

@app.route("/{ROUTE}")
def {FUNC}():
    query = request.args.get("q", "")
    return "<div>Search results for: " + query + "</div>"
''',
    '''from flask import request

@app.route("/{ROUTE}")  
def {FUNC}():
    message = request.form.get("message")
    return f"<p class='msg'>{{message}}</p>"
''',
]

XSS_SAFE_PY = [
    '''from flask import request
from markupsafe import escape

@app.route("/{ROUTE}")
def {FUNC}():
    name = request.args.get("name", "")
    return f"<h1>Hello {{escape(name)}}!</h1>"
''',
    '''from flask import request, render_template_string
from markupsafe import escape

@app.route("/{ROUTE}")
def {FUNC}():
    query = escape(request.args.get("q", ""))
    return f"<div>Search results for: {{query}}</div>"
''',
    '''from flask import request, render_template

@app.route("/{ROUTE}")
def {FUNC}():
    message = request.form.get("message")
    return render_template("message.html", message=message)
''',
]

XSS_PY_VARS = {
    '{FUNC}': ['greet', 'search', 'show_message', 'display_result',
               'render_page', 'show_profile', 'display_comment'],
    '{ROUTE}': ['greet', 'search', 'message', 'result', 'profile', 'display'],
}


# =============================================================================
# GENERATOR
# =============================================================================

ALL_CATEGORIES = [
    {
        'name': 'CWE-89: SQL Injection',
        'cwe': 'CWE-89',
        'vuln_templates': SQL_INJECTION_VULN_PY,
        'safe_templates': SQL_INJECTION_SAFE_PY,
        'vars': SQL_VARS,
        'language': 'python',
        'target_per_class': 250,
    },
    {
        'name': 'CWE-79: XSS (JavaScript)',
        'cwe': 'CWE-79',
        'vuln_templates': XSS_VULN_JS,
        'safe_templates': XSS_SAFE_JS,
        'vars': XSS_VARS,
        'language': 'javascript',
        'target_per_class': 250,
    },
    {
        'name': 'CWE-79: XSS (Python/Flask)',
        'cwe': 'CWE-79',
        'vuln_templates': XSS_VULN_PY,
        'safe_templates': XSS_SAFE_PY,
        'vars': XSS_PY_VARS,
        'language': 'python',
        'target_per_class': 150,
    },
    {
        'name': 'CWE-78: Command Injection',
        'cwe': 'CWE-78',
        'vuln_templates': CMD_INJECTION_VULN_PY,
        'safe_templates': CMD_INJECTION_SAFE_PY,
        'vars': CMD_VARS,
        'language': 'python',
        'target_per_class': 250,
    },
    {
        'name': 'CWE-22: Path Traversal',
        'cwe': 'CWE-22',
        'vuln_templates': PATH_TRAVERSAL_VULN_PY,
        'safe_templates': PATH_TRAVERSAL_SAFE_PY,
        'vars': PATH_VARS,
        'language': 'python',
        'target_per_class': 200,
    },
    {
        'name': 'CWE-502: Insecure Deserialization',
        'cwe': 'CWE-502',
        'vuln_templates': DESER_VULN_PY,
        'safe_templates': DESER_SAFE_PY,
        'vars': DESER_VARS,
        'language': 'python',
        'target_per_class': 200,
    },
    {
        'name': 'CWE-918: SSRF',
        'cwe': 'CWE-918',
        'vuln_templates': SSRF_VULN_PY,
        'safe_templates': SSRF_SAFE_PY,
        'vars': SSRF_VARS,
        'language': 'python',
        'target_per_class': 200,
    },
    {
        'name': 'CWE-798: Hard-coded Credentials',
        'cwe': 'CWE-798',
        'vuln_templates': HARDCODED_VULN_PY,
        'safe_templates': HARDCODED_SAFE_PY,
        'vars': HARDCODED_VARS,
        'language': 'python',
        'target_per_class': 200,
    },
    {
        'name': 'CWE-327: Broken Cryptography',
        'cwe': 'CWE-327',
        'vuln_templates': CRYPTO_VULN_PY,
        'safe_templates': CRYPTO_SAFE_PY,
        'vars': CRYPTO_VARS,
        'language': 'python',
        'target_per_class': 200,
    },
    {
        'name': 'CWE-611: XXE',
        'cwe': 'CWE-611',
        'vuln_templates': XXE_VULN_PY,
        'safe_templates': XXE_SAFE_PY,
        'vars': XXE_VARS,
        'language': 'python',
        'target_per_class': 150,
    },
    {
        'name': 'CWE-94: Code Injection',
        'cwe': 'CWE-94',
        'vuln_templates': CODE_INJECTION_VULN_PY,
        'safe_templates': CODE_INJECTION_SAFE_PY,
        'vars': CODE_INJECTION_VARS,
        'language': 'python',
        'target_per_class': 200,
    },
]

def add_realistic_context(code: str, language: str) -> str:
    """Add realistic boilerplate around code snippets."""
    additions = []
    
    if language == 'python':
        # Random imports
        possible_imports = [
            'import os', 'import sys', 'import logging', 'from pathlib import Path',
            'import traceback', 'from typing import Optional, List, Dict',
        ]
        if random.random() > 0.5:
            additions.append(random.choice(possible_imports))
        
        # Random docstring addition
        if random.random() > 0.6:
            additions.append(f'"""Module for handling {random.choice(["data", "requests", "files", "users", "auth"])}."""')
        
        # Logger
        if random.random() > 0.7:
            additions.append('logger = logging.getLogger(__name__)')
    
    elif language == 'javascript':
        if random.random() > 0.5:
            additions.append("'use strict';")
        if random.random() > 0.6:
            additions.append(f"// {random.choice(['Handler', 'Utility', 'Controller', 'Service'])} module")
    
    if additions:
        return '\n'.join(additions) + '\n\n' + code
    return code


def generate_samples(category: dict) -> list:
    """Generate samples for one category."""
    samples = []
    target = category['target_per_class']
    vuln_templates = category['vuln_templates']
    safe_templates = category['safe_templates']
    var_map = category['vars']
    
    for label, templates in [(1, vuln_templates), (0, safe_templates)]:
        generated = 0
        seen_codes = set()
        attempts = 0
        max_attempts = target * 20
        
        while generated < target and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)
            
            # Fill in variables
            code = template
            for placeholder, options in var_map.items():
                code = code.replace(placeholder, random.choice(options))
            
            # Add context
            code = add_realistic_context(code, category['language'])
            
            # Skip duplicates
            code_hash = hash(code.strip())
            if code_hash in seen_codes:
                continue
            seen_codes.add(code_hash)
            
            sample = {
                'code': code.strip(),
                'label': label,
                'language': category['language'],
                'vulnerability_type': category['cwe'] if label == 1 else 'none',
                'source': 'quality_synthetic',
                'metadata': {
                    'category': category['name'],
                    'template_family': templates.index(template),
                }
            }
            samples.append(sample)
            generated += 1
    
    return samples


def main():
    output_dir = Path("data/raw_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("HIGH-QUALITY VULNERABILITY DATASET GENERATOR")
    print("=" * 80)
    
    all_samples = []
    
    for cat in ALL_CATEGORIES:
        samples = generate_samples(cat)
        vuln = sum(1 for s in samples if s['label'] == 1)
        safe = len(samples) - vuln
        print(f"  {cat['name']:40s} | vuln={vuln:4d} | safe={safe:4d} | total={len(samples):4d}")
        all_samples.extend(samples)
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Summary
    total_vuln = sum(1 for s in all_samples if s['label'] == 1)
    total_safe = len(all_samples) - total_vuln
    
    print(f"\n  {'TOTAL':40s} | vuln={total_vuln:4d} | safe={total_safe:4d} | total={len(all_samples):4d}")
    
    # Quick discriminability test
    print(f"\n  Running quick TF-IDF discriminability test...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    codes = [s['code'] for s in all_samples]
    labels = [s['label'] for s in all_samples]
    
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(codes)
    
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0),
        X, labels, cv=5, scoring='roc_auc'
    )
    print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    if cv_scores.mean() >= 0.80:
        print(f"  *** EXCELLENT! Data is highly discriminative ***")
    elif cv_scores.mean() >= 0.65:
        print(f"  *** GOOD: Data is moderately discriminative ***")
    else:
        print(f"  *** WARNING: Data may not be discriminative enough ***")
    
    # Save
    output_file = output_dir / "quality_vulnerability_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\n  Saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
