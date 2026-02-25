#!/usr/bin/env python3
"""
Convenience CLI entry point for the Hybrid Scanner.
Run from the backend directory:
    python hybrid_scan_cli.py scan --file test_samples/vulnerable_app.py
    python hybrid_scan_cli.py scan --code "os.system(user_input)" --language python
    python hybrid_scan_cli.py rules
    python hybrid_scan_cli.py status
"""
import sys
from pathlib import Path

# Ensure the backend root is on the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.hybrid_scanner.cli import main

if __name__ == "__main__":
    main()
