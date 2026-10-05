# -*- coding: utf-8 -*-
"""Release preflight checks for AI Vector Cleanroom.

Fails the build if binary assets or private/development-only strings would be
*published*. It checks the set of files git would commit (tracked plus
non-ignored untracked), so gitignored scratch — test fixtures, run outputs,
the portable interpreter, input/output contents — is correctly excluded.
An extracted archive is scanned directly. A Git failure in a checkout is a
failure, not permission to skip publication checks.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BLOCKED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".zip",
                    ".log", ".pyc", ".exe", ".dll", ".pyd", ".svg", ".pdf",
                    ".ai", ".eps", ".psd", ".tif", ".tiff", ".7z", ".whl"}

# Direct-scan ignores (only used for an extracted archive without .git).
IGNORED_PARTS = {"__pycache__", ".venv", "venv", "env", ".git", "python",
                 ".pytest_cache", "_last_run"}

# Tokens are assembled from fragments so this file never contains the literal
# private string it is guarding against (which would self-trigger the scan).
SENSITIVE_TEXT = [
    "C:" + "\\Users" + "\\Shinichi",       # absolute developer path
    "ChatGPT" + " Image 2026",             # private input filenames
    "嘉" + "義活力",                        # private client logo name
    "logo" + "A-02",                       # private client logo file
    "real_world" + "_validation",          # private validation artifacts
    "給設計師_" + "請先看我",                # private portable delivery note
    "Chi" + "ayi",                          # romanized private client (嘉義)
    "Ali" + "shan",                         # romanized private test logo (阿里山)
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".bat", ".cmd", ".cff", ".toml",
                 ".yml", ".yaml", ".json", ".js", ".html", ".css", ".cfg"}
ABSOLUTE_USER_PATH_RE = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+", re.IGNORECASE)


def _allowed_asset_hashes() -> dict[str, str]:
    """Only known generated fixtures with their recorded bytes may ship."""
    from release.package_source_beta6 import FIXTURE_FILES

    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    hashes: dict[str, str] = {}
    for record in records:
        relative = record["path"]
        if relative not in FIXTURE_FILES:
            continue
        digest = record["sha256"]
        if relative in hashes or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("duplicate or invalid fixture manifest entry")
        hashes[relative] = digest.lower()
    if set(hashes) != set(FIXTURE_FILES):
        raise ValueError("source manifest must identify every public synthetic fixture")
    return hashes


def _publishable_files() -> list[Path]:
    """Files git would publish: tracked + untracked-but-not-ignored."""
    if (ROOT / ".git").exists():
        # NUL-delimited UTF-8 avoids Git's quoted/non-ASCII filename format and
        # also preserves filenames containing whitespace or line breaks.
        # This invocation trusts this checkout only; no global Git setting changes.
        out = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
             "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            capture_output=True, check=True)
        names = out.stdout.decode("utf-8").split("\0")
        return [ROOT / name for name in dict.fromkeys(names)
                if name and ((ROOT / name).is_file() or (ROOT / name).is_symlink())]
    # An extracted source archive has no Git metadata. Do not skip the fixture
    # directory: every asset must still satisfy the exact fixture allowlist.
    result = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in rel.parts):
            continue
        result.append(path)
    return result


def main() -> int:
    problems: list[str] = []
    try:
        allowed_assets = _allowed_asset_hashes()
        publishable = _publishable_files()
    except (OSError, ValueError, KeyError, TypeError, ImportError,
            subprocess.SubprocessError) as exc:
        print(f"Preflight failed: cannot inspect publication contents ({exc}).")
        return 1
    for path in publishable:
        rel = path.relative_to(ROOT)
        posix = rel.as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            problems.append(f"unsafe release link: {rel}")
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            expected = allowed_assets.get(posix)
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if expected else None
            if expected is None or actual != expected:
                problems.append(f"blocked or modified release asset: {rel}")
        for needle in SENSITIVE_TEXT:
            if needle.casefold() in posix.casefold():
                problems.append(f"sensitive filename found: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".env", ".env.example"}:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                problems.append(f"unreadable UTF-8 public text: {rel} ({exc})")
                continue
            if ABSOLUTE_USER_PATH_RE.search(text):
                problems.append(f"absolute personal user path found in {rel}")
            for needle in SENSITIVE_TEXT:
                if needle.casefold() in text.casefold():
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
