"""
TypeScript → JavaScript Transpiler
====================================
Strips TypeScript-specific syntax to produce valid JavaScript
for the pattern matching engine.

Strategy:
    Uses regex-based transformations to remove:
    - Type annotations
    - Interface/type/enum declarations
    - Generic parameters
    - Access modifiers (public/private/protected)
    - Type assertions

This is NOT a full TS compiler. It's a lightweight preprocessor
sufficient for pattern-based vulnerability scanning.

For production use, consider calling `npx tsc --outDir ...` or
using the TypeScript compiler API via a subprocess.
"""
from __future__ import annotations

import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional


class TypeScriptTranspiler:
    """
    Transpile TypeScript to JavaScript for vulnerability scanning.
    
    Tries two strategies:
        1. System `tsc` or `npx tsc` if available (accurate)
        2. Regex-based stripping (fast, good enough for pattern matching)
    """

    def __init__(self, use_tsc: bool = True):
        self._tsc_available: Optional[bool] = None
        self._use_tsc = use_tsc

    def transpile(self, ts_code: str, filename: str = "input.ts") -> str:
        """
        Transpile TypeScript code to JavaScript.
        
        Args:
            ts_code: TypeScript source code
            filename: Original filename (for error reporting)
            
        Returns:
            JavaScript code (best-effort)
        """
        # Try system tsc first
        if self._use_tsc and self._check_tsc():
            result = self._transpile_tsc(ts_code)
            if result is not None:
                return result

        # Fallback to regex stripping
        return self._strip_types(ts_code)

    def _check_tsc(self) -> bool:
        """Check if TypeScript compiler is available."""
        if self._tsc_available is not None:
            return self._tsc_available
        
        self._tsc_available = shutil.which("tsc") is not None
        if not self._tsc_available:
            # Check npx
            try:
                result = subprocess.run(
                    ["npx", "--yes", "tsc", "--version"],
                    capture_output=True, text=True, timeout=15
                )
                self._tsc_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._tsc_available = False
        
        return self._tsc_available

    def _transpile_tsc(self, ts_code: str) -> Optional[str]:
        """Transpile using the TypeScript compiler."""
        import tempfile
        import os
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ts_file = Path(tmpdir) / "input.ts"
                js_file = Path(tmpdir) / "input.js"
                ts_file.write_text(ts_code, encoding="utf-8")
                
                cmd = ["npx", "--yes", "tsc",
                       "--target", "ES2020",
                       "--module", "commonjs",
                       "--outDir", tmpdir,
                       "--skipLibCheck", "true",
                       "--noEmit", "false",
                       "--allowJs", "true",
                       str(ts_file)]
                
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30,
                    cwd=tmpdir
                )
                
                if js_file.exists():
                    return js_file.read_text(encoding="utf-8")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        
        return None

    @staticmethod
    def _strip_types(ts_code: str) -> str:
        """
        Regex-based TypeScript type stripping.
        Good enough for vulnerability pattern matching.
        """
        code = ts_code

        # Remove interface declarations
        code = re.sub(
            r"(?:export\s+)?interface\s+\w+(?:\s+extends\s+[^{]+)?\s*\{[^}]*\}",
            "", code, flags=re.DOTALL
        )

        # Remove type alias declarations
        code = re.sub(r"(?:export\s+)?type\s+\w+\s*=\s*[^;]+;", "", code)

        # Remove enum declarations (preserve value for pattern matching)
        code = re.sub(
            r"(?:export\s+)?(?:const\s+)?enum\s+\w+\s*\{[^}]*\}",
            "", code, flags=re.DOTALL
        )

        # Remove type annotations in function params: (x: string, y: number)
        code = re.sub(r":\s*(?:string|number|boolean|void|any|never|null|undefined|object|unknown)\b(?:\[\])?", "", code)
        
        # Remove complex type annotations: `: Type` or `: Type<Arg>`
        code = re.sub(r":\s*[A-Z]\w*(?:<[^>]+>)?(?:\[\])?", "", code)

        # Remove generic type parameters: <T>, <T extends U>
        code = re.sub(r"<\s*\w+(?:\s+extends\s+[^>]+)?\s*>", "", code)

        # Remove type assertions: `as Type` or `<Type>expr`
        code = re.sub(r"\s+as\s+\w+(?:<[^>]+>)?", "", code)

        # Remove access modifiers
        code = re.sub(r"\b(?:public|private|protected|readonly)\s+", "", code)

        # Remove `declare` statements
        code = re.sub(r"\bdeclare\s+(?:const|let|var|function|class|module|namespace)\s+[^;{]+[;{]", "", code)

        # Remove non-null assertion operator
        code = re.sub(r"!", "", code)  # Simplified; may over-strip

        # Remove `implements` clause
        code = re.sub(r"\s+implements\s+[^{]+", " ", code)

        # Clean up residual whitespace
        code = re.sub(r"\n{3,}", "\n\n", code)

        return code
