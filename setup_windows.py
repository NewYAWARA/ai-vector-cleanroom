# -*- coding: utf-8 -*-
"""Safe, stdlib-only Windows environment setup for Beta.6.

The tiny batch wrapper stays ASCII-only because ``cmd.exe`` can resume a
UTF-8 batch file in the middle of a multibyte character after ``goto``.  All
localized output and the recoverable external-venv transaction live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Callable, Mapping, Sequence

if os.name == "nt":
    import msvcrt
else:  # Keep source-audit/unit-test imports usable off Windows.
    import fcntl

from environment_preflight import PreflightError, parse_locked_requirements


TOOL_VERSION = "v0.6.0-alpha"
DEFAULT_VENV_KEY = "v0.6.0-alpha"
VENV_MARKER_NAME = ".aivc-venv.json"
VENV_MARKER_SCHEMA = "ai-vector-cleanroom-external-venv/1"
SUPPORTED_PYTHON = (3, 12)
REQUIRED_POINTER_BITS = 64
PROJECT_ROOT = Path(__file__).resolve().parent
LOCK_PATH = PROJECT_ROOT / "requirements" / "validated-py312.lock.txt"

_METADATA_PROBE = r"""
import os
import site
import struct
import sys

if sys.implementation.name != "cpython":
    raise RuntimeError("not CPython")
if sys.version_info[:2] != (3, 12) or struct.calcsize("P") * 8 != 64:
    raise RuntimeError("wrong Python ABI")

for root in site.getsitepackages():
    with os.scandir(root) as entries:
        for entry in entries:
            name = entry.name.casefold()
            if not name.endswith((".dist-info", ".egg-info")):
                continue
            if entry.is_dir(follow_symlinks=False):
                def raise_walk_error(exc):
                    raise exc
                for current, _dirs, files in os.walk(
                        entry.path, onerror=raise_walk_error, followlinks=False):
                    for filename in files:
                        with open(os.path.join(current, filename), "rb") as handle:
                            while handle.read(1024 * 1024):
                                pass
            elif entry.is_file(follow_symlinks=False):
                with open(entry.path, "rb") as handle:
                    while handle.read(1024 * 1024):
                        pass
print("metadata-ok")
"""


class SetupError(RuntimeError):
    """A safe, actionable setup failure."""


@dataclass(frozen=True)
class SetupOutcome:
    action: str
    venv_dir: Path
    backup_dir: Path | None = None


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _compact_error(value: object, limit: int = 520) -> str:
    text = " ".join(str(value).split()) or value.__class__.__name__
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def child_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    result = dict(os.environ if environ is None else environ)
    for name in ("PYTHONHOME", "PYTHONPATH"):
        result.pop(name, None)
    result.update({
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
    })
    return result


def validate_host_runtime(
    *,
    implementation: str | None = None,
    version_info=None,
    pointer_bits: int | None = None,
) -> None:
    active_implementation = implementation or sys.implementation.name
    info = sys.version_info if version_info is None else version_info
    bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    if active_implementation != "cpython":
        raise SetupError("安裝程式需要 CPython，其他 Python 實作不受支援。")
    if tuple(info[:2]) != SUPPORTED_PYTHON or int(bits) != REQUIRED_POINTER_BITS:
        raise SetupError("需要 64 位 CPython 3.12；請先從 python.org 安裝後重試。")


def resolve_venv_dir(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = str(env.get("AVC_VENV_DIR", "")).strip()
    if configured:
        expanded = os.path.expandvars(os.path.expanduser(configured))
        path = Path(expanded)
        if not path.is_absolute():
            raise SetupError("AVC_VENV_DIR 必須是完整的絕對路徑。")
        return Path(os.path.abspath(path))
    local_app_data = str(env.get("LOCALAPPDATA", "")).strip()
    if not local_app_data:
        raise SetupError(
            "Windows 沒有提供 LOCALAPPDATA；請以 AVC_VENV_DIR 指定可寫入的完整路徑。"
        )
    return Path(os.path.abspath(
        Path(local_app_data)
        / "AI-Vector-Cleanroom"
        / "venvs"
        / DEFAULT_VENV_KEY
    ))


def _is_reparse_or_link(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def validate_target_path(venv_dir: Path, *, project_root: Path = PROJECT_ROOT) -> None:
    target = Path(os.path.abspath(venv_dir))
    root = Path(os.path.abspath(project_root))
    if target == Path(target.anchor) or target == root:
        raise SetupError(f"拒絕使用過於寬廣的 venv 路徑：{target}")
    if root in target.parents or target in root.parents:
        raise SetupError(f"venv 必須位於原始碼目錄之外：{target}")
    if target.exists() and _is_reparse_or_link(target):
        raise SetupError(f"拒絕自動處理連結或 reparse venv：{target}")


def validate_lock(lock_path: Path = LOCK_PATH) -> int:
    try:
        pins = parse_locked_requirements(Path(lock_path))
    except PreflightError as exc:
        raise SetupError(str(exc)) from exc
    return len(pins)


def lock_sha256(lock_path: Path = LOCK_PATH) -> str:
    path = Path(lock_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SetupError(
            f"無法讀取驗證鎖定檔 {path}：{_compact_error(exc)}"
        ) from exc
    return hashlib.sha256(data).hexdigest().upper()


def expected_venv_marker(lock_path: Path = LOCK_PATH) -> dict[str, str]:
    return {
        "lock_sha256": lock_sha256(lock_path),
        "runtime_key": DEFAULT_VENV_KEY,
        "schema": VENV_MARKER_SCHEMA,
        "tool_version": TOOL_VERSION,
    }


def _run_command(
    argv: Sequence[str],
    *,
    capture: bool = False,
    environ: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in argv],
        cwd=str(PROJECT_ROOT),
        env=child_environment(environ),
        stdin=subprocess.DEVNULL,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _result_detail(result: subprocess.CompletedProcess[str]) -> str:
    return _compact_error(result.stderr or result.stdout or f"exit {result.returncode}")


def _managed_venv_marker(venv_dir: Path) -> dict[str, str] | None:
    config = venv_dir / "pyvenv.cfg"
    python = venv_dir / "Scripts" / "python.exe"
    marker = venv_dir / VENV_MARKER_NAME
    if (not venv_dir.is_dir() or not config.is_file()
            or not python.is_file() or not marker.is_file()):
        return None
    if any(_is_reparse_or_link(path) for path in (config, python, marker)):
        return None
    try:
        text = config.read_text(encoding="utf-8", errors="strict")
        payload = json.loads(marker.read_text(encoding="utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not any(
        line.strip().casefold().startswith("version = 3.12")
        for line in text.splitlines()
    ):
        return None
    identity = {
        "runtime_key": DEFAULT_VENV_KEY,
        "schema": VENV_MARKER_SCHEMA,
        "tool_version": TOOL_VERSION,
    }
    if (not isinstance(payload, dict)
            or set(payload) != set(identity) | {"lock_sha256"}
            or any(payload.get(key) != value for key, value in identity.items())
            or not isinstance(payload.get("lock_sha256"), str)
            or len(payload["lock_sha256"]) != 64):
        return None
    return {key: str(value) for key, value in payload.items()}


def _write_venv_marker(venv_dir: Path, lock_path: Path) -> None:
    marker = venv_dir / VENV_MARKER_NAME
    temporary = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
    payload = json.dumps(
        expected_venv_marker(lock_path),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii", newline="\n") as handle:
            handle.write(payload)
        os.replace(temporary, marker)
    except OSError as exc:
        raise SetupError(
            f"無法寫入 venv ownership marker：{_compact_error(exc)}"
        ) from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _lock_handle(handle) -> None:
    if os.name == "nt":
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle) -> None:
    if os.name == "nt":
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire_setup_lock(venv_dir: Path):
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = venv_dir.with_name(f"{venv_dir.name}.setup.lock")
    handle = None
    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        _lock_handle(handle)
    except OSError as exc:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        raise SetupError(
            "無法取得 setup lock；可能已有另一個 setup 正在處理此環境，"
            f"請勿連續雙擊：{venv_dir}"
        ) from exc
    return handle, lock_path


def _release_setup_lock(handle) -> None:
    try:
        handle.seek(0)
        _unlock_handle(handle)
    finally:
        handle.close()


def probe_pip_metadata(
    venv_dir: Path,
    *,
    runner: CommandRunner = _run_command,
) -> tuple[bool, str]:
    python = Path(venv_dir) / "Scripts" / "python.exe"
    result = runner([str(python), "-I", "-c", _METADATA_PROBE], capture=True)
    if result.returncode == 0:
        return True, "metadata-ok"
    return False, _result_detail(result)


def _preflight(
    venv_dir: Path,
    *,
    runner: CommandRunner,
    capture: bool,
) -> subprocess.CompletedProcess[str]:
    python = Path(venv_dir) / "Scripts" / "python.exe"
    return runner(
        [
            str(python),
            "-E",
            "-s",
            "-B",
            str(PROJECT_ROOT / "environment_preflight.py"),
            "--full",
            "--strict-versions",
        ],
        capture=capture,
    )


def _install_locked(
    venv_dir: Path,
    lock_path: Path,
    *,
    runner: CommandRunner,
) -> None:
    python = Path(venv_dir) / "Scripts" / "python.exe"
    print("[依賴安裝] 正在依驗證鎖定檔安裝套件，請稍候……")
    result = runner(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            "-r",
            str(lock_path),
        ],
        capture=False,
    )
    if result.returncode != 0:
        raise SetupError(
            "依賴安裝失敗；現有環境未被自動重建，請檢查上方 pip 訊息。"
        )


def _require_preflight(
    venv_dir: Path,
    *,
    runner: CommandRunner,
) -> None:
    result = _preflight(venv_dir, runner=runner, capture=False)
    if result.returncode != 0:
        raise SetupError("安裝完成，但完整環境預檢仍未通過。")


def _provision_at(
    venv_dir: Path,
    lock_path: Path,
    *,
    runner: CommandRunner,
) -> None:
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[環境建置] 正在建立外部 Python 環境：{venv_dir}")
    result = runner(
        [sys.executable, "-I", "-m", "venv", str(venv_dir)],
        capture=False,
    )
    if result.returncode != 0:
        raise SetupError(f"無法建立外部 Python 環境：{venv_dir}")
    _write_venv_marker(venv_dir, lock_path)
    _install_locked(venv_dir, lock_path, runner=runner)
    _require_preflight(venv_dir, runner=runner)


def _unique_sibling(
    target: Path,
    label: str,
    *,
    now: datetime | None = None,
) -> Path:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = target.with_name(f"{target.name}.{label}-{stamp}")
    candidate = base
    counter = 2
    while candidate.exists():
        candidate = target.with_name(f"{base.name}-{counter}")
        counter += 1
    return candidate


def _rebuild_damaged_venv(
    venv_dir: Path,
    lock_path: Path,
    *,
    runner: CommandRunner,
    now: datetime | None = None,
) -> SetupOutcome:
    backup = _unique_sibling(venv_dir, "quarantine", now=now)
    print(f"[環境修復] 既有 venv 的 metadata 無法安全讀取，先保留至：{backup}")
    try:
        venv_dir.rename(backup)
    except OSError as exc:
        raise SetupError(
            f"無法隔離損壞的 venv；未變更原環境：{_compact_error(exc)}"
        ) from exc

    try:
        _provision_at(venv_dir, lock_path, runner=runner)
    except Exception as exc:
        failed_dir: Path | None = None
        recovery_error: BaseException | None = None
        try:
            if venv_dir.exists():
                failed_dir = _unique_sibling(venv_dir, "failed-rebuild", now=now)
                venv_dir.rename(failed_dir)
            backup.rename(venv_dir)
        except OSError as restore_exc:
            recovery_error = restore_exc
        if recovery_error is not None:
            raise SetupError(
                "重建失敗且無法自動還原；請勿刪除任何資料夾。"
                f"舊環境位於 {backup}；錯誤：{_compact_error(recovery_error)}"
            ) from exc
        failed_note = f"；未完成環境保留於 {failed_dir}" if failed_dir else ""
        raise SetupError(
            f"重建失敗，舊環境已原位還原{failed_note}。原因：{_compact_error(exc)}"
        ) from exc

    return SetupOutcome("rebuilt", venv_dir, backup)


def ensure_validated_environment(
    venv_dir: Path,
    lock_path: Path = LOCK_PATH,
    *,
    runner: CommandRunner = _run_command,
    now: datetime | None = None,
) -> SetupOutcome:
    target = Path(venv_dir)
    validate_target_path(target)
    validate_lock(Path(lock_path))
    expected_lock = lock_sha256(Path(lock_path))
    lock_handle, _setup_lock_path = _acquire_setup_lock(target)
    try:
        if not target.exists():
            try:
                _provision_at(target, Path(lock_path), runner=runner)
            except Exception as exc:
                failed_dir: Path | None = None
                if target.exists():
                    try:
                        failed_dir = _unique_sibling(
                            target, "failed-setup", now=now)
                        target.rename(failed_dir)
                    except OSError as preserve_exc:
                        raise SetupError(
                            "初次建立失敗，且無法保留未完成環境："
                            f"{_compact_error(preserve_exc)}"
                        ) from exc
                note = f"；未完成環境保留於 {failed_dir}" if failed_dir else ""
                raise SetupError(
                    f"初次建立外部環境失敗{note}。原因：{_compact_error(exc)}"
                ) from exc
            return SetupOutcome("created", target)

        marker = _managed_venv_marker(target)
        if marker is None:
            raise SetupError(
                "既有路徑沒有有效的 AI Vector Cleanroom ownership marker，"
                f"未安裝、未移動：{target}"
            )

        if marker["lock_sha256"] != expected_lock:
            print("[環境版本] 既有專用 venv 的 lock SHA 已過期，改建乾淨環境。")
            return _rebuild_damaged_venv(
                target, Path(lock_path), runner=runner, now=now)

        healthy, detail = probe_pip_metadata(target, runner=runner)
        if not healthy:
            print(f"[環境損壞] pip metadata probe 失敗：{detail}")
            return _rebuild_damaged_venv(
                target, Path(lock_path), runner=runner, now=now)

        current = _preflight(target, runner=runner, capture=True)
        if current.returncode == 0:
            if current.stdout:
                print(current.stdout.rstrip())
            return SetupOutcome("reused", target)

        print("[環境更新] 既有 venv 可讀，但版本或載入檢查未通過；依鎖定檔補齊。")
        _install_locked(target, Path(lock_path), runner=runner)
        _require_preflight(target, runner=runner)
        return SetupOutcome("updated", target)
    finally:
        _release_setup_lock(lock_handle)


def main() -> int:
    configure_console()
    try:
        validate_host_runtime()
        venv_dir = resolve_venv_dir()
        outcome = ensure_validated_environment(venv_dir)
    except SetupError as exc:
        print(f"[安裝失敗] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            f"[安裝失敗] 未預期錯誤：{exc.__class__.__name__}: {_compact_error(exc)}",
            file=sys.stderr,
        )
        return 3

    print(f"[完成] AI Vector Cleanroom {TOOL_VERSION} 開源版環境已就緒。")
    print(f"[位置] {outcome.venv_dir}")
    if outcome.backup_dir is not None:
        print(f"[舊環境備份] {outcome.backup_dir}")
    print("現在可以執行「工作台.bat」或「清稿.bat」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
