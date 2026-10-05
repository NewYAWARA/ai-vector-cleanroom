# -*- coding: utf-8 -*-
"""Shared source/data path policy for the open-source Beta.6 build.

The source checkout is deliberately treated as read-only application code.
Uploaded images, results and history live in a short, versioned per-user data
root so a deeply extracted ZIP cannot consume the Win32 path budget twice.
This module is stdlib-only because the launcher preflight imports it before
any native dependency.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import secrets
import sys
from typing import Mapping


if getattr(sys, "frozen", False):
    CODE_DIR = Path(sys.executable).resolve().parent
else:
    CODE_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_PARTS = ("AIVC", "designer4")
MAX_WINDOWS_PATH_UNITS = 220
MAX_OUTPUT_BASE_UNITS = 48
MAX_INPUT_STEM_UNITS = 64
LOCK_NAME = ".aivc-writer.lock"


class DataPathError(RuntimeError):
    """The configured data/output location violates the safe path contract."""


class DataDirectoryBusyError(DataPathError):
    """Another process already owns the data/output writer lock."""


def _utf16_units(value: object) -> int:
    """Return Windows' UTF-16 code-unit count, including non-BMP characters."""

    return len(str(value).encode("utf-16-le", errors="surrogatepass")) // 2


def _prefix_within_units(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    used = 0
    result: list[str] = []
    for character in text:
        units = _utf16_units(character)
        if used + units > limit:
            break
        result.append(character)
        used += units
    return "".join(result)


def _bounded_component(text: str, limit: int, fallback: str) -> str:
    original = str(text).strip().rstrip(" .") or fallback
    if _utf16_units(original) <= limit:
        return original
    digest = hashlib.sha256(
        original.encode("utf-8", errors="surrogatepass")
    ).hexdigest()[:12]
    suffix = "_" + digest
    prefix = _prefix_within_units(
        original, limit - _utf16_units(suffix)
    ).rstrip(" ._-")
    if not prefix:
        prefix = _prefix_within_units(fallback, limit - _utf16_units(suffix))
    return prefix + suffix


def bounded_output_base(text: str) -> str:
    """Return a deterministic result basename that fits the path budget."""

    return _bounded_component(text, MAX_OUTPUT_BASE_UNITS, "image")


def bounded_input_filename(name: str) -> str:
    """Bound an uploaded leaf name while preserving its file extension."""

    leaf = Path(str(name)).name.strip().rstrip(" .") or "image.png"
    suffix = Path(leaf).suffix
    stem = leaf[:-len(suffix)] if suffix else leaf
    return _bounded_component(stem, MAX_INPUT_STEM_UNITS, "image") + suffix


def _is_unc_or_device_path(path: Path) -> bool:
    raw = str(path).replace("/", "\\")
    return raw.startswith("\\\\") or raw.startswith("\\?\\")


def _validate_absolute_local_path(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise DataPathError(f"{label} 必須是絕對路徑：{path}")
    if os.name == "nt" and _is_unc_or_device_path(expanded):
        raise DataPathError(
            f"{label} 必須是本機磁碟路徑，不支援 UNC／裝置路徑：{path}")
    resolved = expanded.resolve(strict=False)
    if resolved.parent == resolved:
        raise DataPathError(f"{label} 不可直接使用磁碟根目錄：{resolved}")
    return resolved


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _longest_result_path(output_dir: Path, out_base: str) -> Path:
    # This is the longest current filesystem pattern because the basename is
    # repeated in both the result directory and the paint-role filename.
    return (Path(output_dir) / f"result_{out_base}"
            / f"{out_base}_paint_roles.json")


def ensure_output_path_budget(output_dir: Path, out_base: str) -> int:
    """Fail before conversion if a legacy-safe result path cannot be formed."""

    base = bounded_output_base(out_base)
    candidate = _longest_result_path(
        Path(output_dir).expanduser().resolve(strict=False), base)
    units = _utf16_units(candidate)
    if units > MAX_WINDOWS_PATH_UNITS:
        raise DataPathError(
            "輸出路徑仍過長，最長預估為 "
            f"{units} 個 UTF-16 單位（安全上限 {MAX_WINDOWS_PATH_UNITS}）："
            f"{candidate}。請將 AVC_DATA_DIR 或 --output 設為較短的絕對本機路徑。"
        )
    return units


def resolve_data_dir(
    *,
    env: Mapping[str, str] | None = None,
    code_dir: Path | None = None,
) -> Path:
    """Resolve the complete version-specific data root without creating it."""

    environment = os.environ if env is None else env
    source = Path(code_dir or CODE_DIR).resolve(strict=False)
    if "AVC_DATA_DIR" in environment:
        configured = str(environment.get("AVC_DATA_DIR", "")).strip()
        if not configured:
            raise DataPathError("AVC_DATA_DIR 已設定但內容為空")
        candidate = Path(configured)
    else:
        local_app_data = str(environment.get("LOCALAPPDATA", "")).strip()
        if not local_app_data:
            if os.name == "nt":
                raise DataPathError(
                    "Windows 未提供 LOCALAPPDATA；請將 AVC_DATA_DIR 設為較短的絕對本機路徑"
                )
            local_app_data = str(
                environment.get("XDG_DATA_HOME", "")
            ).strip() or str(Path.home() / ".local" / "share")
        candidate = Path(local_app_data).joinpath(*DEFAULT_DATA_PARTS)

    resolved = _validate_absolute_local_path(candidate, "資料目錄")
    if _inside(resolved, source):
        raise DataPathError(
            f"資料目錄不可位於原始碼目錄內：{resolved}；"
            "請使用 AVC_DATA_DIR 指定另一個較短的本機位置"
        )
    ensure_output_path_budget(resolved / "output", "x" * MAX_OUTPUT_BASE_UNITS)
    return resolved


def prepare_data_layout(
    data_dir: Path | None = None,
    *,
    code_dir: Path | None = None,
) -> dict[str, object]:
    """Create and atomically probe the versioned input/output layout."""

    source = Path(code_dir or CODE_DIR).resolve(strict=False)
    root = (resolve_data_dir(code_dir=source) if data_dir is None
            else _validate_absolute_local_path(Path(data_dir), "資料目錄"))
    if _inside(root, source):
        raise DataPathError(f"資料目錄不可位於原始碼目錄內：{root}")
    max_units = ensure_output_path_budget(
        root / "output", "x" * MAX_OUTPUT_BASE_UNITS)
    input_dir = root / "input"
    output_dir = root / "output"
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{secrets.token_hex(8)}"
        staged = root / f".aivc-write-probe-{token}.tmp"
        committed = root / f".aivc-write-probe-{token}.ok"
        payload = secrets.token_bytes(24)
        try:
            with staged.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staged, committed)
            if committed.read_bytes() != payload:
                raise DataPathError("資料目錄的原子寫入讀回內容不一致")
        finally:
            staged.unlink(missing_ok=True)
            committed.unlink(missing_ok=True)
    except DataPathError:
        raise
    except OSError as exc:
        raise DataPathError(f"無法建立或寫入資料目錄 {root}：{exc}") from exc
    return {
        "root": str(root),
        "input": str(input_dir),
        "output": str(output_dir),
        "max_result_path_units": max_units,
    }


@contextmanager
def writer_lock(lock_root: Path, owner: str):
    """Hold one nonblocking cross-process writer lock for the given root."""

    root = _validate_absolute_local_path(Path(lock_root), "writer 資料夾")
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock_path = root / LOCK_NAME
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
    except OSError as exc:
        raise DataPathError(f"無法建立 writer lock：{root}：{exc}") from exc

    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except (OSError, BlockingIOError) as exc:
            raise DataDirectoryBusyError(
                f"已有另一個工作台或 CLI 正在使用：{root}；"
                "請先關閉它，或改用不同的 AVC_DATA_DIR／--output"
            ) from exc
        yield lock_path
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()
        if acquired:
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                # A racing new owner may already hold the same inode on
                # Windows; that process removes it when releasing its lock.
                pass
