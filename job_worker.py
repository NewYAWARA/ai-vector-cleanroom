# -*- coding: utf-8 -*-
"""One-shot conversion worker used by the local workbench.

The worker intentionally has no authority to publish a result.  It writes to
one private staging directory and records an authenticated completion receipt;
the long-lived workbench process validates that receipt before committing the
staged files to ``output``.
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import math
import os
import platform
import secrets
import struct
import sys
import time
from pathlib import Path

SCHEMA = "ai-vector-cleanroom-job/v1"
ALLOWED_OVERRIDES = {
    "background", "geometry", "strokes", "gradients", "colors",
    "white_threshold", "max_size", "curve_error_percent",
}
_PROCESS_TREE_JOB_HANDLE = None
_FINGERPRINT_ENV_KEYS = (
    "AVC_GPU_MODE",
    "AVC_GPU_MIN_WORK",
    "WGPU_BACKEND_TYPE",
    "AVC_GRADIENT_WORKERS",
    "AVC_GRADIENT_GEOMETRY_WORKERS",
    "AVC_GRADIENT_PROCESS_POOL_SAFE",
    "AVC_GRADIENT_PROCESS_POOL_OWNER_PID",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
)
_FINGERPRINT_DISTRIBUTIONS = (
    "cffi", "charset-normalizer", "cssselect2", "freetype-py", "lxml",
    "numpy", "Pillow", "pycairo", "pycparser", "rendercanvas",
    "reportlab", "rlPyCairo", "svglib", "tinycss2", "vtracer", "webencodings",
    "wgpu",
)
_FINGERPRINT_SOURCE_FILES = (
    "job_worker.py",
    "workbench.py",
    "vector_cleanroom.py",
    "execution_control.py",
    "clean_base.py",
    "compute_backend.py",
    "gradient_reconstruction_stage.py",
    "geometry_error_optimizer.py",
    "curve_refit_stage.py",
)
_FINGERPRINT_LOCK_FILES = (
    "requirements/core-py312.lock.txt",
    "requirements/validated-py312.lock.txt",
)
_CANDIDATE_TRACE_OPTIONS = (
    "background", "strokes", "gradients", "geometry",
)


def _normalise_candidate_summary(value):
    """Return bounded scalar candidate evidence safe for progress files."""
    if not isinstance(value, dict):
        return None
    result = {}
    status = value.get("status")
    if status in {"running", "ok", "failed"}:
        result["status"] = status
    options = value.get("options")
    if isinstance(options, dict):
        safe_options = {
            key: options[key][:80]
            for key in _CANDIDATE_TRACE_OPTIONS
            if isinstance(options.get(key), str)
        }
        if safe_options:
            result["options"] = safe_options
    for key in ("quality_score", "selection_score", "structure_score"):
        number = value.get(key)
        if (isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(float(number))):
            result[key] = float(number)
    visual_status = value.get("visual_gate_status")
    if visual_status in {"accepted", "manual_review", "rejected"}:
        result["visual_gate_status"] = visual_status
    error = value.get("error")
    if result.get("status") == "failed" and isinstance(error, str):
        result["error"] = error[:240]
    return result or None


def _install_windows_process_tree_guard() -> bool:
    """Keep spawned optimiser workers inside a kill-on-close Job Object.

    The workbench supervisor can forcibly terminate this one-shot worker at
    its hard deadline.  A Windows Job Object makes that termination close the
    last owning handle and therefore kills every ProcessPool descendant too.
    If Windows refuses nested job assignment, callers fail closed to serial
    geometry instead of creating children that the supervisor cannot reap.
    """
    global _PROCESS_TREE_JOB_HANDLE
    os.environ["AVC_GRADIENT_PROCESS_POOL_SAFE"] = "0"
    os.environ.pop("AVC_GRADIENT_PROCESS_POOL_OWNER_PID", None)
    if os.name != "nt":
        return False
    if _PROCESS_TREE_JOB_HANDLE is not None:
        os.environ["AVC_GRADIENT_PROCESS_POOL_SAFE"] = "1"
        os.environ["AVC_GRADIENT_PROCESS_POOL_OWNER_PID"] = str(os.getpid())
        return True

    import ctypes
    from ctypes import wintypes

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        return False
    try:
        information = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        information.BasicLimitInformation.LimitFlags = 0x00002000
        if not kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(information),
                ctypes.sizeof(information)):
            return False
        if not kernel32.AssignProcessToJobObject(
                handle, kernel32.GetCurrentProcess()):
            return False
        _PROCESS_TREE_JOB_HANDLE = handle
        handle = None
        os.environ["AVC_GRADIENT_PROCESS_POOL_SAFE"] = "1"
        os.environ["AVC_GRADIENT_PROCESS_POOL_OWNER_PID"] = str(os.getpid())
        return True
    finally:
        if handle:
            kernel32.CloseHandle(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _fingerprinted_files(base: Path, relative_paths) -> list[dict]:
    """Describe the exact local sources/locks used by this worker."""
    rows = []
    for relative in relative_paths:
        path = (base / relative).resolve()
        row = {"path": str(path), "relative_path": str(relative)}
        try:
            stat = path.stat()
            row.update(size=stat.st_size, sha256=_sha256(path))
        except OSError as exc:
            row.update(missing=True, error=f"{type(exc).__name__}: {exc}"[:300])
        rows.append(row)
    return rows


def _installed_distribution_versions() -> dict:
    versions = {}
    for name in _FINGERPRINT_DISTRIBUTIONS:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
        except Exception as exc:
            versions[name] = {
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
    return versions


def _build_runtime_fingerprint(
        manifest: dict, input_path: Path, *, process_tree_guard=None,
        process_tree_guard_error="") -> dict:
    """Capture observed runtime identity without importing the heavy pipeline."""
    base = Path(__file__).resolve().parent
    implementation = getattr(sys, "implementation", None)
    manifest_overrides = manifest.get("overrides")
    if isinstance(manifest_overrides, dict):
        observed_overrides = dict(manifest_overrides)
    else:
        observed_overrides = {
            "invalid_type": type(manifest_overrides).__name__,
        }
    return {
        "schema": "ai-vector-cleanroom-runtime-fingerprint/v1",
        "created_unix": time.time(),
        "job_id": manifest.get("job_id"),
        "input_path": str(input_path),
        "input_sha256": str(manifest.get("input_sha256") or "").upper(),
        "out_base": str(manifest.get("out_base") or ""),
        "overrides": observed_overrides,
        "budget_seconds": manifest.get("budget_seconds"),
        "candidate_cap": manifest.get("candidate_cap"),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": getattr(implementation, "name", ""),
            "cache_tag": getattr(implementation, "cache_tag", None),
            "pointer_bits": struct.calcsize("P") * 8,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "process": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "process_tree_guard": process_tree_guard,
            "process_tree_guard_error": str(process_tree_guard_error or "")[:500],
        },
        "environment": {
            key: os.environ[key] for key in _FINGERPRINT_ENV_KEYS
            if key in os.environ
        },
        "distributions": _installed_distribution_versions(),
        "source_files": _fingerprinted_files(
            base, _FINGERPRINT_SOURCE_FILES),
        "lock_files": _fingerprinted_files(base, _FINGERPRINT_LOCK_FILES),
    }


def _append_progress_trace(stream, payload: dict) -> None:
    """Append one compact event; callers keep ``stream`` open for O(N) I/O."""
    data = (json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")) + "\n").encode("utf-8")
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if not written:
            raise OSError("short write while appending progress trace")
        view = view[written:]


def _file_inventory(root: Path) -> list[dict]:
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RuntimeError("staged result must not contain symbolic links")
        if path.is_file():
            inventory.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return inventory


def _atomic_json(path: Path, payload: dict, *, best_effort=False) -> bool:
    """Write one untorn JSON file, tolerating short Windows read races.

    The supervisor polls ``progress.json`` while the worker replaces it.
    Windows can briefly deny that replace while the reader or antivirus holds
    the old pathname.  Completion receipts remain strict; progress is allowed
    to skip one refresh after bounded retries because it is non-authoritative.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        last_error = None
        for delay in (0.0, 0.005, 0.015, 0.04, 0.10, 0.20):
            if delay:
                time.sleep(delay)
            try:
                os.replace(tmp, path)
                return True
            except OSError as exc:
                if (getattr(exc, "winerror", None) not in (5, 32)
                        and getattr(exc, "errno", None) not in (13,)):
                    raise
                last_error = exc
        if best_effort:
            return False
        raise last_error
    finally:
        tmp.unlink(missing_ok=True)


def _load_manifest(manifest_path: Path) -> tuple[dict, Path]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported worker manifest schema")
    staging_root = Path(str(manifest.get("staging_root", ""))).resolve(
        strict=True)
    if manifest_path.parent != staging_root:
        raise ValueError("manifest is outside its staging directory")
    token = manifest.get("token")
    if (not isinstance(token, str) or len(token) != 32
            or any(ch not in "0123456789abcdef" for ch in token)):
        raise ValueError("invalid worker token")
    job_id = manifest.get("job_id")
    if not isinstance(job_id, int) or job_id < 1:
        raise ValueError("invalid job id")
    return manifest, staging_root


def run_manifest(manifest_path: Path, *, process_tree_guard=None,
                 process_tree_guard_error="") -> int:
    manifest, staging_root = _load_manifest(manifest_path)
    # The backend writes only public hardware/audit state.  Keeping the path
    # inside private staging makes GPU selection or fallback diagnosable even
    # when the supervisor terminates the worker at its hard deadline.
    os.environ["AVC_GPU_AUDIT_PATH"] = str(
        staging_root / "gpu_runtime.json")
    token = manifest["token"]
    job_id = manifest["job_id"]
    candidate_cap = int(manifest.get("candidate_cap", 16))
    budget_seconds = float(manifest.get("budget_seconds", 600.0))
    if (not math.isfinite(budget_seconds) or budget_seconds <= 0.0
            or candidate_cap < 1):
        raise ValueError("invalid execution budget")
    input_path = Path(str(manifest.get("input_path", ""))).resolve()
    expected_input_sha = str(manifest.get("input_sha256", "")).upper()
    # This is deliberately the only runtime-fingerprint write.  It precedes
    # input access and the expensive conversion imports so either kind of
    # stall remains diagnosable after the supervisor kills the worker.
    _atomic_json(
        staging_root / "runtime_fingerprint.json",
        _build_runtime_fingerprint(
            manifest, input_path, process_tree_guard=process_tree_guard,
            process_tree_guard_error=process_tree_guard_error))
    if len(expected_input_sha) != 64:
        raise ValueError("invalid input digest")
    input_path = input_path.resolve(strict=True)
    if _sha256(input_path) != expected_input_sha:
        raise ValueError("input digest changed before conversion")

    # Importing the conversion stack is deliberately deferred until after the
    # small, authenticated manifest has been validated.
    import vector_cleanroom as vc
    from app_paths import bounded_output_base
    from execution_control import ExecutionControl

    out_base = str(manifest.get("out_base", ""))
    if not out_base or bounded_output_base(out_base) != out_base:
        raise ValueError("unsafe output base")
    overrides = manifest.get("overrides") or {}
    if not isinstance(overrides, dict) or set(overrides) - ALLOWED_OVERRIDES:
        raise ValueError("unsupported option override")

    staging_output = staging_root / "output"
    staging_output.mkdir(parents=True, exist_ok=False)
    cancel_path = staging_root / "cancel.request.json"
    control = ExecutionControl(
        budget_seconds=budget_seconds, candidate_cap=candidate_cap,
        cancel_requested=cancel_path.is_file)
    progress_sequence = 0
    trace_stream = (staging_root / "progress_trace.ndjson").open(
        "ab", buffering=0)

    def emit(event):
        """Normalise core progress events into one atomic authenticated file."""
        nonlocal progress_sequence
        if not isinstance(event, dict):
            return

        def event_number(key, fallback, convert):
            value = event.get(key)
            return convert(fallback if value is None else value)

        progress_sequence += 1
        snapshot = control.snapshot()
        stage = str(event.get("stage") or snapshot.get("stage") or "convert")
        payload = {
            "schema": SCHEMA, "token": token, "job_id": job_id,
            "seq": progress_sequence, "stage": stage,
            "detail": str(event.get("detail") or "")[:500],
            "elapsed_seconds": event_number(
                "elapsed_seconds", snapshot["elapsed_seconds"], float),
            "budget_seconds": budget_seconds,
            "candidate_evaluated": event_number(
                "candidate_evaluated", snapshot["candidate_evaluated"], int),
            "candidate_started": event_number(
                "candidate_started", snapshot["candidate_started"], int),
            "candidate_planned": event_number(
                "candidate_planned", snapshot["candidate_planned"], int),
            "candidate_cap": event_number(
                "candidate_cap", snapshot["candidate_cap"], int),
            "updated_unix": time.time(),
        }
        if "stage_index" in event:
            payload["stage_index"] = int(event["stage_index"])
        if "stage_total" in event:
            payload["stage_total"] = int(event["stage_total"])
        if "substage" in event:
            payload["substage"] = str(event["substage"])[:120]
        for key in ("candidate_current", "gradient_candidate_current",
                    "gradient_candidate_total"):
            if key in event:
                payload[key] = int(event[key])
        summary = _normalise_candidate_summary(event.get("candidate_summary"))
        if summary is not None:
            payload["candidate_summary"] = summary
        # The append is intentionally unbuffered but not fsynced.  It survives
        # a process kill without multiplying the existing progress fsync cost;
        # a partial final line is ignored by the supervisor.
        trace_payload = {
            key: value for key, value in payload.items()
            if key not in ("token",)
        }
        _append_progress_trace(trace_stream, trace_payload)
        _atomic_json(
            staging_root / "progress.json", payload, best_effort=True)

    parser = vc.build_arg_parser()
    args = parser.parse_args([])
    args.input = input_path.parent
    args.output = staging_output
    for key, value in overrides.items():
        if key in ("colors", "white_threshold", "max_size"):
            value = int(value)
        elif key == "curve_error_percent":
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("curve error must be finite")
        setattr(args, key, value)
    vc.validate_args(parser, args)

    control.checkpoint("prepare")
    emit({"stage": "prepare", "stage_index": 1, "stage_total": 4,
          "detail": "已驗證輸入與工作參數"})
    emit({"stage": "convert", "stage_index": 2, "stage_total": 4,
          "detail": "正在執行向量清稿與品質閘門"})
    control.checkpoint("convert")
    process_parameters = inspect.signature(vc.process_one).parameters
    optional = {}
    if "progress" in process_parameters:
        optional["progress"] = emit
    if "control" in process_parameters:
        optional["control"] = control
    vc.process_one(input_path, out_base, args, staging_output, **optional)
    control.checkpoint("verify")
    if _sha256(input_path) != expected_input_sha:
        raise RuntimeError("input digest changed during conversion")

    result_dir = staging_output / f"result_{out_base}"
    report_path = result_dir / "report.json"
    zip_path = staging_output / f"result_{out_base}.zip"
    if not report_path.is_file() or not zip_path.is_file():
        raise RuntimeError("conversion returned incomplete staged output")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("input") != input_path.name
            or report.get("output_base") != out_base):
        raise RuntimeError("staged report identity does not match manifest")
    candidates = report.get("candidates") or []
    evaluated = len(candidates) if isinstance(candidates, list) else 0
    planned = max(evaluated, int(control.candidate_planned))
    acceptance = str(report.get("acceptance_status") or "manual_review")

    emit({"stage": "verify", "stage_index": 3, "stage_total": 4,
          "detail": "正在封存並驗證轉檔證據",
          "candidate_evaluated": evaluated,
          "candidate_planned": planned, "candidate_cap": candidate_cap})
    receipt = {
        "schema": SCHEMA,
        "token": token,
        "job_id": job_id,
        "status": "complete",
        "input_name": input_path.name,
        "input_sha256": expected_input_sha,
        "out_base": out_base,
        "acceptance_status": acceptance,
        "report_sha256": _sha256(report_path),
        "zip_sha256": _sha256(zip_path),
        "result_files": _file_inventory(result_dir),
        "candidate_evaluated": evaluated,
        "candidate_planned": planned,
        "candidate_cap": candidate_cap,
        "budget_seconds": budget_seconds,
        "completed_unix": time.time(),
    }
    emit({"stage": "complete", "stage_index": 4, "stage_total": 4,
          "detail": "子程序已完成，等待主程序安全提交",
          "candidate_evaluated": evaluated,
          "candidate_planned": planned, "candidate_cap": candidate_cap})
    trace_stream.close()
    # The receipt is the final authoritative child write.
    _atomic_json(staging_root / "receipt.json", receipt)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: job_worker.py MANIFEST.json", file=sys.stderr)
        return 2
    process_tree_guard = False
    process_tree_guard_error = ""
    try:
        process_tree_guard = _install_windows_process_tree_guard()
    except (OSError, AttributeError, TypeError, ValueError) as exc:
        # Job setup is a performance prerequisite, not a correctness
        # prerequisite.  Refuse to spawn geometry children and continue with
        # the safe serial path if the platform cannot establish tree cleanup.
        os.environ["AVC_GRADIENT_PROCESS_POOL_SAFE"] = "0"
        os.environ.pop("AVC_GRADIENT_PROCESS_POOL_OWNER_PID", None)
        process_tree_guard_error = f"{type(exc).__name__}: {exc}"
        print(
            f"process-tree guard unavailable; using serial geometry: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    manifest_path = Path(argv[0])
    try:
        return run_manifest(
            manifest_path, process_tree_guard=process_tree_guard,
            process_tree_guard_error=process_tree_guard_error)
    except BaseException as exc:
        # SystemExit from argparse validation is converted into a failed job;
        # KeyboardInterrupt likewise must never look like a completed receipt.
        try:
            error_path = manifest_path.resolve().parent / "error.json"
            _atomic_json(error_path, {
                "schema": SCHEMA,
                "status": str(getattr(exc, "terminal_status", "failed")),
                "error_type": type(exc).__name__,
                "detail": str(exc)[:1200],
                "failed_unix": time.time(),
            })
        except Exception:
            pass
        print(f"worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
