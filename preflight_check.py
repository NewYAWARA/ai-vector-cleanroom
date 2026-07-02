# -*- coding: utf-8 -*-
"""Simple release preflight checks for AI Vector Cleanroom."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BLOCKED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".zip", ".log"}
IGNORED_PARTS = {"__pycache__", ".venv", "venv", "env"}
SENSITIVE_TEXT = [
    "C:" + "\\Users" + "\\Shinichi" + "\\Desk" + "top",
    "ChatGPT" + " Image 2026",
    "給" + "設" + "計" + "師",
    "請" + "先" + "看" + "我",
    "LI" + "NE",
]


def main() -> int:
    problems: list[str] = []

    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in rel.parts):
            continue
        if path.is_file() and path.suffix.lower() in BLOCKED_SUFFIXES:
            if rel.as_posix() not in {"input/.gitkeep", "output/.gitkeep"}:
                problems.append(f"blocked release asset: {rel}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".txt", ".bat", ".cff"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in SENSITIVE_TEXT:
            if needle in text:
                problems.append(f"sensitive text '{needle}' found in {path.relative_to(ROOT)}")

    if problems:
        print("Preflight failed:")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("Preflight OK: no blocked assets or sensitive strings found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
