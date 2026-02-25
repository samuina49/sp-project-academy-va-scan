"""
Language Detection Module
=========================
Detects programming language from file extension, shebang, or content heuristics.
Only supports Python, JavaScript, and TypeScript.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# Mapping file extensions to language
_EXT_MAP = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
}

# Shebang patterns
_SHEBANG_PYTHON = re.compile(r"^#!.*(?:python\d?|python3)\b")
_SHEBANG_NODE = re.compile(r"^#!.*\bnode\b")

# Content heuristics (ordered by specificity)
_HEURISTICS = [
    # TypeScript
    (re.compile(r"(?:interface|type|enum)\s+\w+\s*\{"), "typescript"),
    (re.compile(r":\s*(?:string|number|boolean|void|any|never)\b"), "typescript"),
    (re.compile(r"<\w+>\s*\("), "typescript"),
    # Python
    (re.compile(r"^\s*def\s+\w+\s*\(.*\)\s*(?:->.*)?:", re.MULTILINE), "python"),
    (re.compile(r"^\s*import\s+\w+", re.MULTILINE), "python"),
    (re.compile(r"^\s*from\s+\w+\s+import\s+", re.MULTILINE), "python"),
    (re.compile(r"^\s*class\s+\w+.*:", re.MULTILINE), "python"),
    # JavaScript
    (re.compile(r"(?:const|let|var)\s+\w+\s*="), "javascript"),
    (re.compile(r"(?:function|=>)\s*"), "javascript"),
    (re.compile(r"require\s*\("), "javascript"),
    (re.compile(r"module\.exports"), "javascript"),
]


class LanguageDetector:
    """Detect language from filename and/or code content."""

    SUPPORTED = {"python", "javascript", "typescript"}

    @staticmethod
    def detect(
        code: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Detect the programming language.
        
        Priority:
            1. File extension (most reliable)
            2. Shebang (#! line)
            3. Content heuristics
        
        Returns:
            "python", "javascript", "typescript", or None if unsupported.
        """
        # 1) File extension
        if filename:
            ext = Path(filename).suffix.lower()
            lang = _EXT_MAP.get(ext)
            if lang:
                return lang

        # 2) Shebang
        if code:
            first_line = code.split("\n", 1)[0]
            if _SHEBANG_PYTHON.match(first_line):
                return "python"
            if _SHEBANG_NODE.match(first_line):
                return "javascript"

            # 3) Content heuristics
            py_score = 0
            js_score = 0
            ts_score = 0
            for pattern, lang in _HEURISTICS:
                if pattern.search(code):
                    if lang == "python":
                        py_score += 1
                    elif lang == "javascript":
                        js_score += 1
                    elif lang == "typescript":
                        ts_score += 1

            # TS heuristics also count as JS
            if ts_score > 0:
                return "typescript"
            if py_score > js_score:
                return "python"
            if js_score > py_score:
                return "javascript"
            # Ambiguous — fallback
            if py_score > 0:
                return "python"
            if js_score > 0:
                return "javascript"

        return None

    @staticmethod
    def is_supported(language: str) -> bool:
        return language.lower() in LanguageDetector.SUPPORTED

    @staticmethod
    def normalize(language: str) -> str:
        """Normalize language name."""
        lang = language.lower().strip()
        aliases = {
            "py": "python",
            "python3": "python",
            "js": "javascript",
            "jsx": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
        }
        return aliases.get(lang, lang)
