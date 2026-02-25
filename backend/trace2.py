import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, '.')

from app.scanners.simple_scanner import SimplePatternScanner
from app.scanners.scanner_orchestrator import ScannerOrchestrator

scanner = SimplePatternScanner()
orch = ScannerOrchestrator()

js = "const express = require('express');\nconst app = express();\nconst fs = require('fs');\n\napp.use(express.urlencoded({ extended: true }));\n"
py = "import os\nimport sqlite3\nimport pickle\nfrom flask import Flask, request\n\napp = Flask(__name__)\n"

def test_scanner(code, ext, label):
    with tempfile.NamedTemporaryFile(suffix=ext, mode='w', delete=False, encoding='utf-8') as f:
        f.write(code)
        path = f.name
    r = scanner.scan_file(Path(path))
    print(f"\n[SimplePatternScanner] {label} — lines 1-5:")
    found = [f for f in r.findings if f.start_line <= 5]
    if not found:
        print("  NONE")
    for f in found:
        print(f"  L{f.start_line} [{f.severity}] rule={f.rule_id}  msg={f.message[:60]}")
        print(f"  snippet={repr(f.code_snippet[:80])}")
    os.unlink(path)

def test_orch(code, ext, lang, label):
    with tempfile.NamedTemporaryFile(suffix=ext, mode='w', delete=False, encoding='utf-8') as f:
        f.write(code)
        path = f.name
    r = orch.scan_file(Path(path), language=lang)
    print(f"\n[ScannerOrchestrator] {label} — lines 1-5:")
    found = [f for f in r.findings if f.start_line <= 5]
    if not found:
        print("  NONE")
    for f in found:
        print(f"  L{f.start_line} [{f.severity}] rule={f.rule_id}  msg={f.message[:60]}")
        print(f"  snippet={repr(f.code_snippet[:80])}")
    os.unlink(path)

test_scanner(js, '.js', 'JS require lines')
test_scanner(py, '.py', 'Python import lines')
test_orch(js, '.js', 'javascript', 'JS require lines')
test_orch(py, '.py', 'python',     'Python import lines')
