# -*- coding: utf-8 -*-
"""Fail-fast runtime validation for the open-source Windows launchers.

The application deliberately keeps this module stdlib-only.  It must be able
to explain a broken native dependency before importing the workbench or
accepting an image conversion job.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import re
import struct
import sys
from pathlib import Path
from typing import Callable, Mapping

from app_paths import (
    CODE_DIR,
    DataPathError,
    prepare_data_layout,
    resolve_data_dir,
)

TOOL_VERSION = "v0.6.0-alpha"
SUPPORTED_PYTHON = (3, 12)
REQUIRED_POINTER_BITS = 64
DEFAULT_LOCK_PATH = (
    Path(__file__).resolve().parent
    / "requirements"
    / "validated-py312.lock.txt"
)

# Import the modules that actually cross a native-extension boundary.  A
# metadata-only check cannot detect a copied/mismatched NumPy or Pillow DLL.
CORE_IMPORTS = (
    ("numpy", "NumPy"),
    ("PIL.Image", "Pillow"),
    ("vtracer", "vtracer"),
)
FULL_IMPORTS = (
    ("resvg_py", "resvg native SVG renderer"),
    ("svglib.svglib", "svglib"),
    ("reportlab.graphics.renderPM", "ReportLab renderPM"),
    ("rlPyCairo", "rlPyCairo"),
    ("cairo", "pycairo"),
)

_PIN_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;\\]+)"
)


class PreflightError(RuntimeError):
    """An actionable environment error that must prevent application start."""


def _canonical_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _compact_error(exc: BaseException, limit: int = 420) -> str:
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return f"{exc.__class__.__name__}: {text}"


def validate_python_runtime(
    *,
    version_info=None,
    pointer_bits: int | None = None,
) -> dict[str, object]:
    """Require the Python ABI used by the validated Windows dependency lock."""

    info = version_info if version_info is not None else sys.version_info
    version = tuple(int(value) for value in info[:3])
    bits = int(pointer_bits if pointer_bits is not None
               else struct.calcsize("P") * 8)
    if version[:2] != SUPPORTED_PYTHON:
        required = ".".join(str(value) for value in SUPPORTED_PYTHON)
        actual = ".".join(str(value) for value in version)
        raise PreflightError(
            f"需要 64 位 Python {required}.x，目前是 Python {actual}。"
        )
    if bits != REQUIRED_POINTER_BITS:
        raise PreflightError(
            f"需要 {REQUIRED_POINTER_BITS} 位 Python，目前是 {bits} 位。"
        )
    return {"version": version, "pointer_bits": bits}


def parse_locked_requirements(lock_path: Path) -> dict[str, tuple[str, str]]:
    """Read exact ``name==version`` pins from the validated pip lock.

    The validated lock is intentionally stricter than a general requirements
    file: every requirement record must use ``==``.  Optional continuation
    hash lines and pip option lines are accepted, while ranges and unpinned
    packages fail closed.
    """

    path = Path(lock_path)
    if not path.is_file():
        raise PreflightError(f"找不到驗證過的依賴鎖定檔：{path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PreflightError(
            f"無法讀取依賴鎖定檔 {path}：{_compact_error(exc)}"
        ) from exc

    pins: dict[str, tuple[str, str]] = {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if (not line or line.startswith("#") or line.startswith("--")
                or line.startswith("\\")):
            continue
        match = _PIN_RE.match(line)
        if match is None:
            raise PreflightError(
                f"依賴鎖定檔第 {line_number} 行不是精確版本鎖定："
                f"{raw_line.strip()}"
            )
        display_name, version = match.groups()
        remainder = line[match.end():].strip()
        if remainder and not remainder.startswith(("\\", "#")):
            raise PreflightError(
                f"依賴鎖定檔第 {line_number} 行含有未驗證條件："
                f"{raw_line.strip()}"
            )
        canonical = _canonical_distribution_name(display_name)
        previous = pins.get(canonical)
        if previous is not None and previous[1] != version:
            raise PreflightError(
                f"依賴鎖定檔對 {display_name} 指定了衝突版本："
                f"{previous[1]} 與 {version}。"
            )
        pins[canonical] = (display_name, version)

    if not pins:
        raise PreflightError(f"依賴鎖定檔沒有任何精確版本：{path}")
    return pins


def validate_locked_versions(
    pins: Mapping[str, tuple[str, str]],
    *,
    version_getter: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Verify every distribution in the validated lock is installed exactly."""

    get_version = version_getter or importlib.metadata.version
    installed: dict[str, str] = {}
    for canonical in sorted(pins):
        display_name, expected = pins[canonical]
        try:
            actual = str(get_version(display_name))
        except importlib.metadata.PackageNotFoundError as exc:
            raise PreflightError(
                f"尚未安裝必要套件 {display_name}=={expected}。"
            ) from exc
        except Exception as exc:
            raise PreflightError(
                f"無法讀取 {display_name} 的已安裝版本："
                f"{_compact_error(exc)}"
            ) from exc
        if actual != expected:
            raise PreflightError(
                f"套件版本不符：{display_name} 需要 {expected}，"
                f"目前是 {actual}。"
            )
        installed[canonical] = actual
    return installed


def validate_imports(
    *,
    full: bool = False,
    importer: Callable[[str], object] | None = None,
) -> tuple[str, ...]:
    """Import required native modules so ABI/DLL failures surface up front."""

    import_module = importer or importlib.import_module
    requirements = CORE_IMPORTS + (FULL_IMPORTS if full else ())
    loaded: list[str] = []
    for module_name, display_name in requirements:
        try:
            import_module(module_name)
        except Exception as exc:
            raise PreflightError(
                f"無法載入必要套件 {display_name}"
                f"（import {module_name}）：{_compact_error(exc)}"
            ) from exc
        loaded.append(display_name)
    return tuple(loaded)


def run_preflight(
    *,
    full: bool = False,
    strict_versions: bool = False,
    lock_path: Path | None = None,
    importer: Callable[[str], object] | None = None,
    version_getter: Callable[[str], str] | None = None,
    data_dir: Path | None = None,
) -> dict[str, object]:
    """Run checks in fail-fast order and return a printable success summary."""

    runtime = validate_python_runtime()
    try:
        selected_data_dir = (Path(data_dir) if data_dir is not None
                             else resolve_data_dir(code_dir=CODE_DIR))
        data_layout = prepare_data_layout(
            selected_data_dir, code_dir=CODE_DIR)
    except (OSError, DataPathError) as exc:
        raise PreflightError(f"資料目錄不可用：{exc}") from exc
    selected_lock = Path(lock_path) if lock_path is not None else DEFAULT_LOCK_PATH
    installed: dict[str, str] = {}
    if strict_versions:
        pins = parse_locked_requirements(selected_lock)
        installed = validate_locked_versions(
            pins, version_getter=version_getter)
    loaded = validate_imports(full=full, importer=importer)
    return {
        "runtime": runtime,
        "full": bool(full),
        "strict_versions": bool(strict_versions),
        "lock_path": selected_lock if strict_versions else None,
        "locked_distribution_count": len(installed),
        "loaded": loaded,
        "data_layout": data_layout,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Vector Cleanroom 開啟前環境檢查"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="連同 SVG renderer 與預覽依賴一起檢查",
    )
    parser.add_argument(
        "--strict-versions", action="store_true",
        help="要求所有已安裝套件符合驗證過的精確版本",
    )
    parser.add_argument(
        "--lock-file", type=Path, default=DEFAULT_LOCK_PATH,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = build_arg_parser().parse_args(argv)
    try:
        summary = run_preflight(
            full=args.full,
            strict_versions=args.strict_versions,
            lock_path=args.lock_file,
        )
    except PreflightError as exc:
        print(f"[環境檢查失敗] {exc}", file=sys.stderr)
        print(f"[目前直譯器] {sys.executable}", file=sys.stderr)
        print(
            "[修復方式] 請回到專案根目錄執行 setup_windows.bat，"
            "完成後再重新開啟。",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            f"[環境檢查失敗] 預檢程式發生未預期錯誤："
            f"{_compact_error(exc)}",
            file=sys.stderr,
        )
        print(f"[目前直譯器] {sys.executable}", file=sys.stderr)
        print(
            "[修復方式] 請回到專案根目錄執行 setup_windows.bat。",
            file=sys.stderr,
        )
        return 3

    runtime = summary["runtime"]
    version = ".".join(str(value) for value in runtime["version"])
    print(
        f"[環境檢查通過] Python {version} "
        f"{runtime['pointer_bits']} 位：{sys.executable}"
    )
    print(f"[依賴載入通過] {', '.join(summary['loaded'])}")
    data_layout = summary.get("data_layout")
    if isinstance(data_layout, dict) and data_layout.get("root"):
        print(f"[資料目錄通過] {data_layout['root']}")
    if summary["strict_versions"]:
        print(
            f"[版本鎖定通過] {summary['locked_distribution_count']} 個套件："
            f"{summary['lock_path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
