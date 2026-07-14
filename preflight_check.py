# -*- coding: utf-8 -*-
"""Release preflight checks for AI Vector Cleanroom.

Fails the build if binary assets or private/development-only strings would be
*published*. It checks the set of files git would commit (tracked plus
non-ignored untracked), so gitignored scratch — test fixtures, run outputs,
the portable interpreter, input/output contents — is correctly excluded.
When run outside a git checkout it falls back to a heuristic directory scan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BLOCKED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".zip",
                    ".log", ".pyc", ".exe", ".dll", ".pyd"}
ALLOWED_ASSETS = {"input/.gitkeep", "output/.gitkeep"}

# Heuristic-scan ignores (only used when git is unavailable).
IGNORED_PARTS = {"__pycache__", ".venv", "venv", "env", ".git", "python",
                 "fixtures"}
IGNORED_PREFIX = "_"   # test scratch dirs: _last_run, _highres_*, ...

# Tokens are assembled from fragments so this file never contains the literal
# private string it is guarding against (which would self-trigger the scan).
SENSITIVE_TEXT = [
    "C:" + "\\Users" + "\\Shinichi",       # absolute developer path
    "ChatGPT" + " Image 2026",             # private input filenames
    "嘉" + "義活力",                        # private client logo name
    "logo" + "A-02",                       # private client logo file
    "real_world" + "_validation",          # private validation artifacts
    "給設計師_" + "請先看我",                # private portable delivery note
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".bat", ".cff", ".toml", ".yml", ".yaml"}


def _publishable_files() -> list[Path]:
    """Files git would publish: tracked + untracked-but-not-ignored."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-c", "-o",
             "--exclude-standard"],
            capture_output=True, text=True, check=True)
        files = [ROOT / line for line in out.stdout.splitlines() if line]
        if files:
            return [f for f in files if f.is_file()]
    except Exception:
        pass
    # fallback: heuristic scan
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS or part.startswith(IGNORED_PREFIX)
               for part in rel.parts):
            continue
        result.append(path)
    return result


def main() -> int:
    problems: list[str] = []
    for path in _publishable_files():
        rel = path.relative_to(ROOT)
        posix = rel.as_posix()
        if path.suffix.lower() in BLOCKED_SUFFIXES and posix not in ALLOWED_ASSETS:
            problems.append(f"blocked release asset: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for needle in SENSITIVE_TEXT:
                if needle in text:
                    problems.append(f"sensitive text '{needle}' found in {rel}")

    if problems:
        print("Preflight failed:")
        for item in problems:
            print(f"  - {item}")
        return 1

    print("Preflight OK: no blocked assets or sensitive strings found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
