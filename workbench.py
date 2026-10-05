# -*- coding: utf-8 -*-
"""
AI Vector Cleanroom — local workbench server.

A zero-dependency (stdlib http.server) local UI:
  - drag & drop images onto the page -> they are converted in a background
    worker and appear in the result list
  - per-image re-run with different options (background / geometry /
    strokes / gradients / colors) without touching the command line
  - one click opens each result's review workbench (zoom, object list,
    hotspots) or downloads the SVG / zip
  - one click builds a randomized visual blind-test page
  - one click builds a timed Stage 2 designer editing test page

Binds to 127.0.0.1 only. Start with:  python workbench.py
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import queue
import random
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from html import escape as html_escape
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as _ThreadingHTTPServer
from pathlib import Path

from app_paths import (
    DataDirectoryBusyError,
    DataPathError,
    bounded_input_filename,
    bounded_output_base,
    writer_lock,
)
import vector_cleanroom as vc


class ThreadingHTTPServer(_ThreadingHTTPServer):
    """One listener per port, including Windows' permissive SO_REUSEADDR."""
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

BASE = vc.BASE
DATA_DIR = vc.DATA_DIR
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
HISTORY_DIR = OUTPUT_DIR / "_history"
HISTORY_KEEP = 8
JOB_ROOT = DATA_DIR / ".jobs"
JOB_SCHEMA = "ai-vector-cleanroom-job/v1"
FAILURE_DIAGNOSTIC_SCHEMA = "ai-vector-cleanroom-failure-diagnostics/v1"
FAILURE_DIAGNOSTIC_STATUSES = {
    "timed_out", "budget_exhausted", "failed",
}
JOB_WORKER = BASE / "job_worker.py"
JOB_CANDIDATE_CAP = 16
JOB_POLL_SECONDS = 0.20
JOB_TERMINATE_GRACE_SECONDS = 2.0


def _job_timeout_seconds():
    """Return a bounded hard wall-clock budget for one conversion."""
    try:
        value = float(os.environ.get("AVC_JOB_TIMEOUT_SECONDS", "1200"))
    except ValueError:
        value = 1200.0
    if not math.isfinite(value):
        value = 1200.0
    return min(1800.0, max(30.0, value))


JOB_TIMEOUT_SECONDS = _job_timeout_seconds()
TERMINAL_JOB_STATUSES = {
    "cancelled", "timed_out", "budget_exhausted", "failed", "done",
    "review", "rejected",
}

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

WB_TOKEN = ""              # set at startup, embedded in the page
# A job is one upload/re-run, not an append-only event.  The distinction is
# important for a one-worker queue: a second upload must stay visibly
# "queued" while the first one is converting, and the browser must keep
# polling until *all* queued work has reached a terminal state.
_jobs = []                 # [{id, name, status, detail, t}]
_jobs_lock = threading.Lock()
_publish_lock = threading.RLock()
_job_runtime = {}          # job id -> non-JSON process/cancel state
_queue = queue.Queue()
_job_sequence = 0


def _new_job(name, detail="已排入等待佇列"):
    """Create a visible queued job and return its stable local identifier."""
    global _job_sequence
    with _jobs_lock:
        _job_sequence += 1
        job_id = _job_sequence
        _jobs.append({
            "id": job_id, "name": name, "status": "queued",
            "detail": detail, "t": time.strftime("%H:%M:%S"),
            "stage": "queued", "stage_index": 0, "stage_total": 4,
            "candidate_evaluated": 0, "candidate_planned": None,
            "candidate_cap": JOB_CANDIDATE_CAP, "elapsed_seconds": 0.0,
            "budget_seconds": JOB_TIMEOUT_SECONDS, "cancellable": True,
            "diagnostic_path": None,
        })
        _job_runtime[job_id] = {
            "cancel_requested": False, "process": None,
            "started_monotonic": None,
        }
        _trim_jobs_locked()
        return job_id


def _trim_jobs_locked():
    """Keep active jobs and the newest terminal history records."""
    terminal_ids = [j["id"] for j in _jobs
                    if j.get("status") in TERMINAL_JOB_STATUSES]
    remove = set(terminal_ids[:-60])
    if remove:
        _jobs[:] = [j for j in _jobs if j.get("id") not in remove]
        for job_id in remove:
            _job_runtime.pop(job_id, None)


def _set_job(job_id, status=None, detail=None, **fields):
    """Update one visible job in place; retain a small completed history."""
    with _jobs_lock:
        for job in reversed(_jobs):
            if job.get("id") == job_id:
                if (job.get("status") in TERMINAL_JOB_STATUSES
                        and status is not None
                        and status != job.get("status")):
                    return
                if status is not None:
                    job["status"] = status
                if detail is not None:
                    job["detail"] = detail
                job.update(fields)
                job["t"] = time.strftime("%H:%M:%S")
                if job.get("status") in TERMINAL_JOB_STATUSES:
                    job["cancellable"] = False
                _trim_jobs_locked()
                return
        # Defensive fallback: a worker should never lose its queue record,
        # but a useful failure message is preferable to silently hiding it.
        _jobs.append({"id": job_id, "name": "unknown",
                      "status": status or "failed", "detail": detail or "",
                      "t": time.strftime("%H:%M:%S"), **fields})
        _trim_jobs_locked()


def _snapshot_jobs():
    """Return JSON-safe jobs with elapsed time sampled from a monotonic clock."""
    now = time.monotonic()
    with _jobs_lock:
        snapshots = []
        for job in _jobs:
            row = dict(job)
            runtime = _job_runtime.get(job.get("id")) or {}
            started = runtime.get("started_monotonic")
            if started is not None and job.get("status") not in TERMINAL_JOB_STATUSES:
                row["elapsed_seconds"] = round(max(0.0, now - started), 1)
            snapshots.append(row)
        return snapshots


def _cancel_job(job_id):
    """Request cancellation without trusting the browser to kill a process."""
    marker = None
    with _jobs_lock:
        job = next((j for j in reversed(_jobs) if j.get("id") == job_id), None)
        if job is None:
            return "missing"
        if job.get("status") in TERMINAL_JOB_STATUSES:
            return "terminal"
        if job.get("status") == "committing":
            return "too_late"
        runtime = _job_runtime.setdefault(job_id, {})
        runtime["cancel_requested"] = True
        if job.get("status") == "queued":
            job.update(status="cancelled", stage="cancelled",
                       detail="已在開始前取消", cancellable=False,
                       t=time.strftime("%H:%M:%S"))
            outcome = "cancelled"
        else:
            staging_root = runtime.get("staging_root")
            if staging_root:
                marker = Path(staging_root) / "cancel.request.json"
            job.update(status="cancelling", detail="正在停止轉檔子程序",
                       cancellable=False, t=time.strftime("%H:%M:%S"))
            outcome = "cancelling"
    if marker is not None:
        try:
            _atomic_json(marker, {"job_id": job_id, "cancel": True,
                                  "requested_unix": time.time()})
        except OSError:
            pass
    return outcome


def _log_job(name, status, detail=""):
    """Compatibility helper for non-queued diagnostic events."""
    job_id = _new_job(name, detail)
    _set_job(job_id, status, detail)
    return job_id


def _enqueue_job(img_path: Path, overrides: dict, requested_base=None):
    """Put work on the single conversion queue and make it immediately visible."""
    job_id = _new_job(img_path.name)
    _queue.put((job_id, img_path, overrides, requested_base))
    return job_id


def _history_files(result_name: str):
    """Return newest-first successful snapshots for one result."""
    if not HISTORY_DIR.exists():
        return []
    prefix = f"{result_name}__"
    files = [p for p in HISTORY_DIR.iterdir()
             if p.is_file() and p.suffix.lower() == ".zip"
             and p.name.startswith(prefix)]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _archive_result(result_dir: Path, *, prune=True):
    """Save the current successful result before a destructive re-run."""
    if not (result_dir.is_dir() and (result_dir / "report.json").is_file()):
        return None
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stamp += f"_{time.time_ns() % 1_000_000_000:09d}"
    dst = HISTORY_DIR / f"{result_dir.name}__{stamp}.zip"
    tmp = dst.with_suffix(".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(result_dir.rglob("*")):
                if f.is_file():
                    z.write(f, arcname=f.relative_to(OUTPUT_DIR))
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    if prune:
        for old in _history_files(result_dir.name)[HISTORY_KEEP:]:
            old.unlink(missing_ok=True)
    return dst


def _restore_result(archive: Path, result_dir: Path, result_zip: Path):
    """Restore a snapshot made by _archive_result after a failed re-run."""
    shutil.rmtree(result_dir, ignore_errors=True)
    result_zip.unlink(missing_ok=True)
    root = OUTPUT_DIR.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (OUTPUT_DIR / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError("history archive contains an unsafe path")
        z.extractall(OUTPUT_DIR)
    shutil.copy2(archive, result_zip)


class _JobCancelled(RuntimeError):
    pass


class _JobTimedOut(RuntimeError):
    pass


class _JobBudgetExceeded(RuntimeError):
    pass


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _file_inventory(root: Path):
    inventory = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise RuntimeError("待提交結果不可包含符號連結")
        if path.is_file():
            inventory.append({
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return inventory


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _cancel_requested(job_id):
    with _jobs_lock:
        return bool((_job_runtime.get(job_id) or {}).get("cancel_requested"))


def _begin_commit(job_id):
    """Close the cancel window and make the publish decision atomically."""
    with _jobs_lock:
        runtime = _job_runtime.get(job_id) or {}
        if runtime.get("cancel_requested"):
            return False
        job = next((j for j in reversed(_jobs) if j.get("id") == job_id), None)
        if job is not None:
            job.update(status="committing", stage="commit",
                       detail="證據已驗證，正在安全提交結果",
                       cancellable=False, t=time.strftime("%H:%M:%S"))
        return True


def _terminate_process(process):
    """Stop the direct worker child and never wait indefinitely."""
    if process.poll() is not None:
        return "already_exited"
    try:
        process.terminate()
        try:
            process.wait(timeout=JOB_TERMINATE_GRACE_SECONDS)
            return "terminate"
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=JOB_TERMINATE_GRACE_SECONDS)
                return "kill"
            except subprocess.TimeoutExpired:
                return "kill_wait_timed_out"
    except OSError as terminate_exc:
        try:
            process.kill()
            try:
                process.wait(timeout=JOB_TERMINATE_GRACE_SECONDS)
                return f"kill_after_{type(terminate_exc).__name__}"
            except subprocess.TimeoutExpired:
                return "kill_wait_timed_out"
        except OSError as kill_exc:
            return (
                f"termination_error:{type(terminate_exc).__name__}/"
                f"{type(kill_exc).__name__}")


def _cleanup_staging(staging_root: Path):
    """Best-effort cleanup; an unremovable tree is hidden outside output."""
    if not staging_root.exists():
        return
    for delay in (0.0, 0.05, 0.15, 0.30):
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(staging_root)
            return
        except OSError:
            pass
    quarantine = staging_root.with_name(
        f".orphan-{staging_root.name}-{secrets.token_hex(4)}")
    try:
        staging_root.replace(quarantine)
    except OSError:
        pass


def _read_progress_trace(path: Path):
    """Read every complete NDJSON event and tolerate a kill-torn tail."""
    events = []
    malformed_lines = 0
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except (UnicodeError, json.JSONDecodeError):
                    malformed_lines += 1
                    continue
                if isinstance(value, dict):
                    events.append(value)
                else:
                    malformed_lines += 1
    except OSError:
        pass
    return events, malformed_lines


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _integer_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summarize_progress_trace(staging_root: Path, runtime: dict):
    """Derive per-candidate and per-stage elapsed time without detail parsing."""
    events, malformed_lines = _read_progress_trace(
        staging_root / "progress_trace.ndjson")
    last_progress = runtime.get("last_progress")
    if isinstance(last_progress, dict):
        known_sequences = {
            _integer_or_none(event.get("seq")) for event in events
        }
        if _integer_or_none(last_progress.get("seq")) not in known_sequences:
            events.append(dict(last_progress))
    events.sort(key=lambda event: (
        _integer_or_none(event.get("seq")) is None,
        _integer_or_none(event.get("seq")) or 0))

    timeline = []
    candidate_starts = {}
    candidate_timings = []
    candidate_summaries = {}
    active_candidate = None
    for event in events:
        elapsed = _finite_number(event.get("elapsed_seconds"))
        if elapsed is None:
            continue
        current = _integer_or_none(event.get("candidate_current"))
        evaluated = _integer_or_none(event.get("candidate_evaluated"))
        started = _integer_or_none(event.get("candidate_started"))
        if current is None and active_candidate is None and started is not None:
            if evaluated is not None and started > evaluated:
                current = started
        if current is not None and current > 0:
            active_candidate = current
            summary = event.get("candidate_summary")
            if isinstance(summary, dict):
                candidate_summaries[current] = dict(summary)
        context = active_candidate
        if (current is not None and evaluated is not None
                and evaluated < current):
            candidate_starts.setdefault(current, elapsed)
        timeline.append({
            "seq": _integer_or_none(event.get("seq")),
            "elapsed_seconds": elapsed,
            "stage": str(event.get("stage") or "unknown")[:120],
            "substage": str(event.get("substage") or "")[:120],
            "candidate_current": context,
            "gradient_candidate_current": _integer_or_none(
                event.get("gradient_candidate_current")),
            "gradient_candidate_total": _integer_or_none(
                event.get("gradient_candidate_total")),
        })
        if (current is not None and evaluated is not None
                and evaluated >= current and current in candidate_starts):
            start = candidate_starts.pop(current)
            candidate_timings.append({
                "candidate": current,
                "start_elapsed_seconds": start,
                "end_elapsed_seconds": elapsed,
                "duration_seconds": max(0.0, elapsed - start),
                "complete": True,
                "candidate_summary": candidate_summaries.get(current),
            })
            if active_candidate == current:
                active_candidate = None

    stage_spans = []
    stage_totals = {}
    gradient_candidate_timings = {}
    for current, following in zip(timeline, timeline[1:]):
        duration = max(
            0.0,
            following["elapsed_seconds"] - current["elapsed_seconds"])
        span = {
            "stage": current["stage"],
            "substage": current["substage"],
            "candidate_current": current["candidate_current"],
            "gradient_candidate_current": current[
                "gradient_candidate_current"],
            "gradient_candidate_total": current[
                "gradient_candidate_total"],
            "start_elapsed_seconds": current["elapsed_seconds"],
            "end_elapsed_seconds": following["elapsed_seconds"],
            "duration_seconds": duration,
            "complete": True,
        }
        stage_spans.append(span)
        total_key = (
            current["candidate_current"], current["stage"],
            current["substage"])
        total = stage_totals.setdefault(total_key, {
            "stage": current["stage"],
            "substage": current["substage"],
            "candidate_current": current["candidate_current"],
            "duration_seconds": 0.0,
            "interval_count": 0,
        })
        total["duration_seconds"] += duration
        total["interval_count"] += 1
        gradient_current = current["gradient_candidate_current"]
        if gradient_current is not None:
            gradient_key = (
                current["candidate_current"], gradient_current)
            gradient = gradient_candidate_timings.setdefault(gradient_key, {
                "candidate_current": current["candidate_current"],
                "gradient_candidate_current": gradient_current,
                "gradient_candidate_total": current[
                    "gradient_candidate_total"],
                "observed_duration_seconds": 0.0,
                "interval_count": 0,
            })
            gradient["observed_duration_seconds"] += duration
            gradient["interval_count"] += 1

    stall = _finite_number(runtime.get("stall_since_last_progress_seconds"))
    stall = 0.0 if stall is None else stall
    active_at_terminal = None
    if timeline:
        last = timeline[-1]
        active_at_terminal = {
            "stage": last["stage"],
            "substage": last["substage"],
            "candidate_current": last["candidate_current"],
            "gradient_candidate_current": last[
                "gradient_candidate_current"],
            "gradient_candidate_total": last["gradient_candidate_total"],
            "last_worker_elapsed_seconds": last["elapsed_seconds"],
            "observed_stall_seconds": stall,
            "duration_seconds_lower_bound": stall,
            "complete": False,
            "duration_kind": "lower_bound",
            "candidate_summary": candidate_summaries.get(
                last["candidate_current"]),
        }
        gradient_current = last["gradient_candidate_current"]
        if gradient_current is not None:
            gradient_key = (last["candidate_current"], gradient_current)
            gradient = gradient_candidate_timings.setdefault(gradient_key, {
                "candidate_current": last["candidate_current"],
                "gradient_candidate_current": gradient_current,
                "gradient_candidate_total": last[
                    "gradient_candidate_total"],
                "observed_duration_seconds": 0.0,
                "interval_count": 0,
            })
            gradient.update(
                active_at_terminal=True,
                observed_stall_seconds=stall,
                duration_seconds_lower_bound=(
                    gradient["observed_duration_seconds"] + stall),
                duration_kind="lower_bound")
        last_elapsed = last["elapsed_seconds"]
        for candidate, start in sorted(candidate_starts.items()):
            candidate_timings.append({
                "candidate": candidate,
                "start_elapsed_seconds": start,
                "last_worker_elapsed_seconds": last_elapsed,
                "observed_stall_seconds": stall,
                "duration_seconds_lower_bound": max(
                    0.0, last_elapsed - start) + stall,
                "complete": False,
                "duration_kind": "lower_bound",
                "candidate_summary": candidate_summaries.get(candidate),
            })

    return {
        "trace_event_count": len(events),
        "timed_event_count": len(timeline),
        "malformed_trace_lines": malformed_lines,
        "candidate_timings": sorted(
            candidate_timings, key=lambda row: row["candidate"]),
        "gradient_candidate_timings": list(
            gradient_candidate_timings.values()),
        "stage_spans": stage_spans,
        "stage_totals": list(stage_totals.values()),
        "active_at_terminal": active_at_terminal,
    }


def _failure_evidence(staging_root: Path):
    evidence = {}
    for name in (
            "manifest.json", "runtime_fingerprint.json",
            "gpu_runtime.json", "progress_trace.ndjson", "progress.json", "error.json",
            "stdout.log", "stderr.log", "receipt.json"):
        path = staging_root / name
        try:
            stat = path.stat()
        except OSError:
            continue
        evidence[name] = {
            "size": stat.st_size,
            "modified_unix": stat.st_mtime,
        }
    partial_output = staging_root / "output"
    if partial_output.exists():
        evidence["output/"] = {"preserved": True}
    return evidence


def _preserve_failed_staging(staging_root: Path, *, job_id: int,
                             status: str, detail: str, img_path: Path,
                             out_base: str, overrides: dict):
    """Write one summary and rename evidence; any error leaves staging intact."""
    with _jobs_lock:
        runtime = dict(_job_runtime.get(job_id) or {})
    manifest = _read_json(staging_root / "manifest.json") or {}
    progress = runtime.get("last_progress")
    if not isinstance(progress, dict):
        progress = _read_json(staging_root / "progress.json") or {}
    progress = {
        key: value for key, value in progress.items() if key != "token"
    }
    failure_root = JOB_ROOT.parent / ".failed_jobs"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = failure_root / (
        f"{stamp}-job-{job_id}-{status}-{secrets.token_hex(4)}")
    summary = {
        "schema": FAILURE_DIAGNOSTIC_SCHEMA,
        "preserved_unix": time.time(),
        "terminal_status": status,
        "detail": str(detail)[:1200],
        "job_id": job_id,
        "input_name": img_path.name,
        "input_path": str(img_path),
        "input_sha256": manifest.get("input_sha256"),
        "out_base": out_base,
        "overrides": dict(overrides),
        "budget_seconds": manifest.get("budget_seconds"),
        "candidate_cap": manifest.get("candidate_cap"),
        "supervisor": {
            "elapsed_seconds": runtime.get("supervisor_elapsed_seconds"),
            "returncode": runtime.get("process_returncode"),
            "termination_mode": runtime.get("termination_mode"),
            "stall_since_last_progress_seconds": runtime.get(
                "stall_since_last_progress_seconds"),
        },
        "last_progress": progress,
        "gpu_runtime": _read_json(staging_root / "gpu_runtime.json"),
        "timings": _summarize_progress_trace(staging_root, runtime),
        "evidence": _failure_evidence(staging_root),
        "source_staging_path": str(staging_root),
        "planned_diagnostic_path": str(destination),
        "actual_diagnostic_path": str(destination),
    }
    try:
        _atomic_json(staging_root / "failure_summary.json", summary)
        failure_root.mkdir(parents=True, exist_ok=True)
        staging_root.replace(destination)
        return destination, None
    except Exception as exc:
        # Fail closed: never call recursive cleanup after a diagnostic failure.
        error = f"{type(exc).__name__}: {exc}"[:500]
        if staging_root.exists():
            try:
                summary["preservation_error"] = error
                summary["actual_diagnostic_path"] = str(staging_root)
                _atomic_json(staging_root / "failure_summary.json", summary)
            except Exception:
                pass
        return staging_root, error


def _validate_receipt(staging_root: Path, token: str, job_id: int,
                      img_path: Path, input_sha: str, out_base: str):
    receipt = _read_json(staging_root / "receipt.json")
    if not receipt:
        raise RuntimeError("子程序未留下完整完成收據")
    required = {
        "schema": JOB_SCHEMA, "token": token, "job_id": job_id,
        "status": "complete", "input_name": img_path.name,
        "input_sha256": input_sha, "out_base": out_base,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("子程序完成收據身分驗證失敗")
    staging_output = staging_root / "output"
    report_path = staging_output / f"result_{out_base}" / "report.json"
    zip_path = staging_output / f"result_{out_base}.zip"
    if not report_path.is_file() or not zip_path.is_file():
        raise RuntimeError("子程序輸出不完整（缺 report.json 或 zip）")
    if (receipt.get("report_sha256") != _sha256(report_path)
            or receipt.get("zip_sha256") != _sha256(zip_path)):
        raise RuntimeError("子程序輸出雜湊與完成收據不符")
    if receipt.get("result_files") != _file_inventory(report_path.parent):
        raise RuntimeError("子程序結果檔案清單與完成收據不符")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("input") != img_path.name
            or report.get("output_base") != out_base):
        raise RuntimeError("子程序報告不屬於本次工作")
    acceptance = str(report.get("acceptance_status") or "manual_review")
    if receipt.get("acceptance_status") != acceptance:
        raise RuntimeError("子程序報告與收據的品質狀態不一致")
    return {"staging_output": staging_output, "report": report,
            "receipt": receipt}


def _publish_progress(job_id, progress, token):
    if (not progress or progress.get("schema") != JOB_SCHEMA
            or progress.get("token") != token
            or progress.get("job_id") != job_id):
        return
    allowed = {
        key: progress[key] for key in (
            "seq", "stage", "stage_index", "stage_total",
            "elapsed_seconds", "budget_seconds", "candidate_evaluated",
            "candidate_started", "candidate_planned", "candidate_cap", "substage",
            "candidate_current", "gradient_candidate_current",
            "gradient_candidate_total") if key in progress
    }
    observed = time.monotonic()
    retained = {
        key: value for key, value in progress.items() if key != "token"
    }
    with _jobs_lock:
        runtime = _job_runtime.setdefault(job_id, {})
        previous = runtime.get("last_progress") or {}
        if previous.get("seq") != retained.get("seq"):
            runtime["last_progress_received_monotonic"] = observed
        runtime["last_progress"] = retained
    detail = str(progress.get("detail") or "")[:300]
    _set_job(job_id, detail=detail, **allowed)


def _execute_job_subprocess(job_id: int, img_path: Path, out_base: str,
                            overrides: dict, staging_root: Path,
                            timeout_seconds: float):
    """Run exactly one conversion in a killable, private child process."""
    input_sha = _sha256(img_path)
    token = secrets.token_hex(16)
    manifest = {
        "schema": JOB_SCHEMA, "token": token, "job_id": job_id,
        "input_path": str(img_path.resolve()), "input_sha256": input_sha,
        "out_base": out_base, "overrides": dict(overrides),
        "staging_root": str(staging_root.resolve()),
        "candidate_cap": JOB_CANDIDATE_CAP,
        "budget_seconds": timeout_seconds,
    }
    manifest_path = staging_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    stdout_path = staging_root / "stdout.log"
    stderr_path = staging_root / "stderr.log"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-B", str(JOB_WORKER), str(manifest_path)],
            cwd=str(BASE), stdin=subprocess.DEVNULL, stdout=stdout,
            stderr=stderr, shell=False, close_fds=True,
            creationflags=creationflags)
        with _jobs_lock:
            runtime = _job_runtime.setdefault(job_id, {})
            runtime.update(process=process, started_monotonic=started,
                           staging_root=staging_root)
        while process.poll() is None:
            _publish_progress(job_id, _read_json(staging_root / "progress.json"),
                              token)
            if _cancel_requested(job_id):
                _terminate_process(process)
                raise _JobCancelled("使用者已取消本次轉檔")
            supervisor_elapsed = time.monotonic() - started
            if supervisor_elapsed >= timeout_seconds:
                _set_job(
                    job_id,
                    elapsed_seconds=round(supervisor_elapsed, 1))
                termination_mode = _terminate_process(process)
                # A worker can commit one last progress snapshot between the
                # preceding poll and termination.  Read it after kill before
                # the staging directory is archived.
                _publish_progress(
                    job_id, _read_json(staging_root / "progress.json"), token)
                with _jobs_lock:
                    runtime = _job_runtime.setdefault(job_id, {})
                    received = runtime.get("last_progress_received_monotonic")
                    runtime.update(
                        supervisor_elapsed_seconds=supervisor_elapsed,
                        process_returncode=process.returncode,
                        termination_mode=termination_mode,
                        terminal_monotonic=started + supervisor_elapsed,
                        stall_since_last_progress_seconds=(
                            max(0.0, started + supervisor_elapsed - received)
                            if received is not None else None),
                    )
                raise _JobTimedOut(
                    f"超過 {int(timeout_seconds)} 秒硬性時間上限，已停止子程序")
            time.sleep(JOB_POLL_SECONDS)
        return_code = process.returncode
    _publish_progress(job_id, _read_json(staging_root / "progress.json"), token)
    finished = time.monotonic()
    with _jobs_lock:
        runtime = _job_runtime.setdefault(job_id, {})
        received = runtime.get("last_progress_received_monotonic")
        runtime.update(
            supervisor_elapsed_seconds=max(0.0, finished - started),
            process_returncode=return_code,
            termination_mode="exited",
            terminal_monotonic=finished,
            stall_since_last_progress_seconds=(
                max(0.0, finished - received)
                if received is not None else None),
        )
    if _cancel_requested(job_id):
        raise _JobCancelled("使用者已取消本次轉檔")
    if return_code != 0:
        error = _read_json(staging_root / "error.json") or {}
        terminal_status = str(error.get("status") or "failed")
        detail = str(error.get("detail") or "")
        if not detail:
            try:
                detail = stderr_path.read_text(
                    encoding="utf-8", errors="replace")[-1200:].strip()
            except OSError:
                detail = ""
        detail = detail or f"轉檔子程序異常結束（code {return_code}）"
        if terminal_status == "cancelled":
            raise _JobCancelled(detail)
        if terminal_status == "timed_out":
            raise _JobTimedOut(detail)
        if terminal_status == "budget_exhausted":
            raise _JobBudgetExceeded(detail)
        raise RuntimeError(detail)
    return _validate_receipt(
        staging_root, token, job_id, img_path, input_sha, out_base)


def _commit_staged_result(staging_output: Path, out_base: str):
    with _publish_lock:
        return _commit_staged_result_locked(staging_output, out_base)


def _commit_staged_result_locked(staging_output: Path, out_base: str):
    """Atomically publish a verified result with rollback on any failure."""
    from handoff_service import has_locked_objects
    if has_locked_objects(OUTPUT_DIR, f"result_{out_base}"):
        raise RuntimeError("這個結果已有已採用並鎖定的物件；請另開版本重跑")
    staged_dir = staging_output / f"result_{out_base}"
    staged_zip = staging_output / f"result_{out_base}.zip"
    if not (staged_dir / "report.json").is_file() or not staged_zip.is_file():
        raise RuntimeError("待提交輸出不完整")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_dir = OUTPUT_DIR / staged_dir.name
    result_zip = OUTPUT_DIR / staged_zip.name
    nonce = secrets.token_hex(8)
    rollback_dir = OUTPUT_DIR / f"._rollback-{out_base}-{nonce}"
    rollback_zip = OUTPUT_DIR / f"._rollback-{out_base}-{nonce}.zip"
    archive = _archive_result(result_dir, prune=False)
    moved_dir = moved_zip = False
    try:
        if result_dir.exists():
            result_dir.replace(rollback_dir)
            moved_dir = True
        if result_zip.exists():
            result_zip.replace(rollback_zip)
            moved_zip = True
        staged_dir.replace(result_dir)
        staged_zip.replace(result_zip)
        if not (result_dir / "report.json").is_file() or not result_zip.is_file():
            raise RuntimeError("提交後驗證失敗")
    except Exception:
        shutil.rmtree(result_dir, ignore_errors=True)
        result_zip.unlink(missing_ok=True)
        if moved_dir and rollback_dir.exists():
            rollback_dir.replace(result_dir)
        if moved_zip and rollback_zip.exists():
            rollback_zip.replace(result_zip)
        if archive:
            archive.unlink(missing_ok=True)
        raise
    else:
        shutil.rmtree(rollback_dir, ignore_errors=True)
        try:
            rollback_zip.unlink(missing_ok=True)
        except OSError:
            pass
        for old in _history_files(result_dir.name)[HISTORY_KEEP:]:
            try:
                old.unlink(missing_ok=True)
            except OSError:
                pass
        return archive


def _run_one(img_path: Path, overrides: dict, requested_base=None, *, job_id=None):
    if requested_base:
        out_base = bounded_output_base(requested_base)
        if out_base != requested_base:
            raise ValueError("重跑結果名稱超過安全路徑預算")
    else:
        plan = vc.plan_output_names(vc.find_inputs(INPUT_DIR))
        out_base = plan.get(img_path, img_path.stem)
    status_detail = (" ".join(f"{k}={v}" for k, v in overrides.items())
                     or "defaults")
    if job_id is None:
        job_id = _new_job(img_path.name)
    if _cancel_requested(job_id):
        _set_job(job_id, "cancelled", "已在開始前取消", stage="cancelled")
        return
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    staging_root = JOB_ROOT / f"job-{job_id}-{secrets.token_hex(8)}"
    staging_root.mkdir(parents=False, exist_ok=False)
    with _jobs_lock:
        _job_runtime.setdefault(job_id, {})["staging_root"] = staging_root
    _set_job(job_id, "running", status_detail, stage="starting",
             stage_index=0, elapsed_seconds=0.0)
    diagnostic_status = None
    diagnostic_detail = None
    try:
        outcome = _execute_job_subprocess(
            job_id, img_path, out_base, overrides, staging_root,
            JOB_TIMEOUT_SECONDS)
        with _publish_lock:
            from handoff_service import has_locked_objects
            if has_locked_objects(OUTPUT_DIR, f"result_{out_base}"):
                raise RuntimeError("這個結果已有已採用並鎖定的物件；請另開版本重跑")
            if not _begin_commit(job_id):
                raise _JobCancelled("使用者已取消本次轉檔")
            previous = _commit_staged_result_locked(
                outcome["staging_output"], out_base)
        report = outcome["report"]
        acceptance = str(report.get("acceptance_status") or "manual_review")
        detail = f"result_{out_base}"
        if previous:
            detail += f" · 上一版已存入歷史：{previous.name}"
        if acceptance == "rejected":
            reasons = list((report.get("visual_gate") or {}).get(
                "reasons") or [])
            designer = report.get("designer_quality") or {}
            reasons += ((designer.get("gradient_object_gate") or {}).get(
                "failure_reasons") or [])
            reasons += ((designer.get("curve_economy_gate") or {}).get(
                "failure_reasons") or [])
            reason = f" · {'；'.join(str(v) for v in reasons[:2])}" if reasons else ""
            _set_job(job_id, "rejected",
                     f"品質或可編輯性未達標，可匯出接手草稿，尚需人工修整 · {detail}{reason}",
                     stage="complete")
        elif acceptance != "accepted":
            _set_job(job_id, "review", f"完成但需要人工檢查 · {detail}",
                     stage="complete")
        else:
            _set_job(job_id, "done", detail, stage="complete")
    except _JobCancelled as exc:
        _set_job(job_id, "cancelled", str(exc), stage="cancelled")
    except _JobTimedOut as exc:
        diagnostic_status = "timed_out"
        diagnostic_detail = str(exc)
        _set_job(job_id, diagnostic_status, diagnostic_detail,
                 stage="timed_out")
    except _JobBudgetExceeded as exc:
        diagnostic_status = "budget_exhausted"
        diagnostic_detail = str(exc)
        _set_job(job_id, diagnostic_status, diagnostic_detail,
                 stage="budget_exhausted")
    except Exception as exc:
        diagnostic_status = "failed"
        diagnostic_detail = f"{str(exc)[:260]} · 正式輸出未變更"
        _set_job(job_id, diagnostic_status, diagnostic_detail, stage="failed")
    finally:
        with _jobs_lock:
            runtime = _job_runtime.get(job_id)
            if runtime is not None:
                runtime["process"] = None
            job = next(
                (item for item in reversed(_jobs)
                 if item.get("id") == job_id), {})
            visible_status = str(job.get("status") or "")
            visible_detail = str(job.get("detail") or "")
            terminal_status = (
                visible_status if visible_status in FAILURE_DIAGNOSTIC_STATUSES
                else diagnostic_status or visible_status)
            terminal_detail = (
                visible_detail if visible_status == terminal_status
                else diagnostic_detail or visible_detail)
        if terminal_status in FAILURE_DIAGNOSTIC_STATUSES:
            diagnostic_path, preservation_error = _preserve_failed_staging(
                staging_root, job_id=job_id, status=terminal_status,
                detail=terminal_detail, img_path=img_path,
                out_base=out_base, overrides=overrides)
            fields = {"diagnostic_path": str(diagnostic_path)}
            if preservation_error:
                fields["diagnostic_preservation_error"] = preservation_error
            _set_job(job_id, **fields)
        else:
            _cleanup_staging(staging_root)


def _worker():
    while True:
        job_id, img_path, overrides, requested_base = _queue.get()
        try:
            _run_one(img_path, overrides, requested_base, job_id=job_id)
        except Exception as e:
            _set_job(job_id, "failed", str(e)[:300])
        finally:
            _queue.task_done()


def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r'[<>:"/\\\\|?*]', "_", name).strip() or "image.png"
    return bounded_input_filename(name)


def _unique_input_path(name: str) -> Path:
    p = INPUT_DIR / name
    stem, suf = p.stem, p.suffix
    i = 2
    while p.exists():
        p = INPUT_DIR / bounded_input_filename(f"{stem}_{i}{suf}")
        i += 1
    return p


def _report_primitive_counts(report):
    """Read current reports and retain compatibility with Beta-era reports."""
    details = report.get("stroke_details", []) or []
    detail_rectangles = sum(
        item.get("primitive") == "rect" for item in details
        if isinstance(item, dict))
    rectangles = int(report.get(
        "native_rectangles", detail_rectangles) or 0)
    if "native_circles" in report:
        circles = int(report.get("native_circles", 0) or 0)
    elif "native_primitives" in report:
        circles = max(0, int(report.get("native_primitives", 0) or 0)
                      - rectangles)
    else:
        # Very early reports sometimes used the shorter ``circles`` key.
        circles = int(report.get("circles", 0) or 0)
    ellipses = int(report.get("native_ellipses", 0) or 0)
    lines = int(report.get("native_lines", 0) or 0)
    polylines = int(report.get("native_polylines", 0) or 0)
    polygons = int(report.get("native_polygons", 0) or 0)
    # Recompute the aggregate from disjoint DOM element types. This makes a
    # partially upgraded report safe and guarantees no component is counted
    # once through native_primitives and again through diagnostic details.
    primitives = circles + rectangles + ellipses + lines + polylines + polygons
    return {
        "native_primitives": primitives,
        "native_circles": circles,
        "native_rectangles": rectangles,
        "native_ellipses": ellipses,
        "native_lines": lines,
        "native_polylines": polylines,
        "native_polygons": polygons,
    }


def _report_options(report):
    """Return requested/effective/fallback options across report versions."""
    legacy = dict(report.get("options", {}) or {})
    requested = dict(report.get("options_requested", legacy) or legacy)
    effective = dict(report.get("options_effective", legacy) or legacy)
    fallback = dict(report.get("auto_fallback", {}) or {})
    requested.setdefault(
        "geometry", report.get("geometry_level", "conservative"))
    effective.setdefault(
        "geometry", report.get("geometry_level", "conservative"))
    # Beta-era reports did not have options_effective. In those files the
    # fallback delta is still the most authoritative description of the
    # delivered settings.
    if "options_effective" not in report:
        effective.update(fallback)
    return requested, effective, fallback


def _mapping(value):
    """Return a JSON object or an empty compatibility value."""
    return value if isinstance(value, dict) else {}


def _count_value(value, default=None):
    """Read either a numeric count or the audit's list-of-operation IDs."""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _report_feature_summary(report):
    """Extract compact editability data without breaking older reports."""
    scene = _mapping(report.get("scene"))
    if not scene:
        # Accept an intermediate Beta.3 report layout used during development.
        enhancements = _mapping(report.get("editability_enhancements"))
        stages = _mapping(enhancements.get("stages"))
        scene = _mapping(stages.get("scene_graph"))

    paint = _mapping(report.get("paint"))
    if not paint:
        paint = _mapping(report.get("paint_roles"))
    resources = _mapping(paint.get("resource_counts"))
    roles = paint.get("roles")
    role_controls = _count_value(resources.get("role_controls"))
    if role_controls is None and isinstance(roles, list):
        role_controls = len(roles)

    operations = _mapping(report.get("designer_operations"))
    operation_summary = _mapping(operations.get("summary"))

    def operation_count(key):
        if key in operation_summary:
            return _count_value(operation_summary.get(key))
        return _count_value(operations.get(key))

    passed = operation_count("passed")
    partial = operation_count("partial")
    failed = operation_count("failed")
    manual_review = operation_count("manual_review")
    automatable = operation_count("automatable")
    total = _count_value(operation_summary.get("total_operations"))
    known_counts = [passed, partial, failed, manual_review]
    if total is None and operations and any(value is not None for value in known_counts):
        total = sum(value or 0 for value in known_counts)

    if not operations:
        operation_status = "not_audited"
    elif failed or manual_review:
        operation_status = "manual_review"
    elif partial:
        operation_status = "partial"
    elif total and passed == total:
        operation_status = "passed"
    else:
        operation_status = str(operations.get("status") or "manual_review")

    return {
        "scene": {
            "status": str(scene.get("status") or "not_audited"),
            "actual_dom_group_count": _count_value(
                scene.get("actual_dom_group_count")),
            "manifest_only_group_count": _count_value(
                scene.get("manifest_only_group_count")),
        },
        "paint": {
            "status": str(paint.get("status") or "not_audited"),
            "role_controls": role_controls,
            "paint_resources_total": _count_value(
                resources.get("paint_resources_total")),
            "manifest_file": str(paint.get("manifest_file") or ""),
        },
        "designer_operations": {
            "status": operation_status,
            "acceptance_scope": str(
                operations.get("acceptance_scope") or
                ("not_audited" if not operations else "legacy_unspecified")),
            "semantic_task_validation": str(
                operations.get("semantic_task_validation") or
                ("not_audited" if not operations else "not_performed")),
            "timed_human_editing_validation": str(
                operations.get("timed_human_editing_validation") or
                ("not_audited" if not operations else "not_performed")),
            "human_acceptance": str(
                operations.get("human_acceptance") or
                ("not_audited" if not operations else "not_tested")),
            "total_operations": total,
            "passed": passed,
            "partial": partial,
            "failed": failed,
            "manual_review": manual_review,
            "automatable": automatable,
        },
    }


def _list_results():
    out = []
    if not OUTPUT_DIR.exists():
        return out
    for rd in sorted(OUTPUT_DIR.iterdir()):
        if not rd.is_dir() or not rd.name.startswith("result_"):
            continue
        rj = rd / "report.json"
        if not rj.exists():
            continue
        try:
            rep = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            continue
        svg = next(iter(rd.glob("*_vector.svg")), None)
        result_zip = OUTPUT_DIR / f"{rd.name}.zip"
        primitive_counts = _report_primitive_counts(rep)
        requested, effective, fallback = _report_options(rep)
        features = _report_feature_summary(rep)
        legacy_status = rep.get("acceptance_status", "accepted")
        visual_status = rep.get("visual_acceptance_status", legacy_status)
        editability_status = rep.get("editability_status", "not_audited")
        designer_quality = _mapping(rep.get("designer_quality"))
        designer_status = rep.get(
            "designer_readiness_status",
            designer_quality.get("designer_readiness_status", "not_audited"))
        gradient_gate = _mapping(rep.get("gradient_object_gate"))
        curve_gate = _mapping(rep.get("curve_economy_gate"))
        automation_readiness = _mapping(rep.get("automation_readiness"))
        redraw_complexity = _mapping(rep.get("redraw_complexity"))
        human_validation = _mapping(rep.get("human_validation"))
        detail_grid = rep.get("detail_grid") or {}
        history = [{
            "url": f"_history/{p.name}",
            "label": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(p.stat().st_mtime)),
        } for p in _history_files(rd.name)]
        designer_anchors = rep.get("designer_anchors_total")
        designer_anchor_source = "canonical_svg_geometry"
        if designer_anchors is None:
            designer_anchors = rep.get("nodes_total", 0)
            designer_anchor_source = "legacy_nodes_total"
        out.append({
            "dir": rd.name,
            "base": rd.name.removeprefix("result_"),
            "input": rep.get("input", ""),
            "source": rep.get("source_match_percent"),
            "foreground": rep.get("foreground_match_percent"),
            "paths": rep.get("paths", 0),
            **primitive_counts,
            "circles": primitive_counts["native_circles"],
            "strokes": rep.get("strokes", 0),
            "gradients": rep.get("gradients", 0),
            "nodes": rep.get("nodes_total", 0),
            "designer_anchors": designer_anchors,
            "designer_anchor_source": designer_anchor_source,
            "hotspots": len(rep.get("hotspots", [])),
            "detail_p10": detail_grid.get("p10_score_percent"),
            "editability_status": editability_status,
            "editability_score": rep.get("editability_score"),
            "automation_readiness_score": automation_readiness.get("score"),
            "automation_readiness_status": automation_readiness.get("status"),
            "redraw_ease_score": redraw_complexity.get("ease_score"),
            "redraw_complexity_level": redraw_complexity.get("level"),
            "human_validation_status": human_validation.get(
                "status", "not_performed" if rep.get("editability_schema") else "not_audited"),
            "visual_acceptance_status": visual_status,
            "designer_readiness_status": designer_status,
            "gradient_object_gate_status": gradient_gate.get(
                "status", "not_audited"),
            "curve_economy_gate_status": curve_gate.get(
                "status", "not_audited"),
            "unique_paints_total": rep.get("unique_paints_total"),
            "options": requested,
            "requested_options": requested,
            "effective_options": effective,
            "auto_fallback": fallback,
            "candidates": rep.get("candidates", []),
            "acceptance_status": legacy_status,
            "manual_review_required": bool(
                rep.get("manual_review_required", False)
                or legacy_status != "accepted"),
            "history": history,
            "svg": f"{rd.name}/{svg.name}" if svg else "",
            "zip": result_zip.name if result_zip.is_file() else "",
            "review": (f"{rd.name}/review.html"
                       if (rd / "review.html").is_file() else ""),
            "handoff": ("/handoff?result=" + urllib.parse.quote(rd.name)
                        if svg and ((rd / "source_original.png").is_file()
                                    or (rd / "source_reference.png").is_file()) else ""),
            "recolor": (f"{rd.name}/色彩調整.html"
                        if (rd / "色彩調整.html").is_file() else ""),
            **features,
            "mtime": rj.stat().st_mtime,
        })
    out.sort(key=lambda r: -r["mtime"])
    return out


def build_blind_test() -> Path:
    """Randomized side-by-side blind test page for the designer."""
    results = _list_results()
    rng = random.Random()
    items = []
    for r in results:
        rd = OUTPUT_DIR / r["dir"]
        src = rd / "source_reference.png"
        prev = next(iter(rd.glob("*_preview.png")), None)
        if not (src.exists() and prev):
            continue
        # a fallback preview is the source image itself: comparing it against
        # the source would be a meaningless pair — skip (review)
        try:
            rep = json.loads((rd / "report.json").read_text(encoding="utf-8"))
            if not rep.get("preview_is_svg_render", True):
                continue
        except Exception:
            continue
        requested, effective, fallback = _report_options(rep)
        flip = rng.random() < 0.5
        left, right = (prev, src) if flip else (src, prev)
        items.append({
            "name": r["dir"].replace("result_", ""),
            "left": vc.data_url(left, max_side=900),
            "right": vc.data_url(right, max_side=900),
            "left_is_tool": flip,
            "options": requested,
            "effective": effective,
            "fallback": fallback,
        })
    rows = []
    for i, it in enumerate(items):
        rows.append(f"""
 <div class="item" data-i="{i}" data-toolside="{'L' if it['left_is_tool'] else 'R'}"
      data-meta="{html_escape(json.dumps({'name': it['name'], 'options': it['options'], 'effective': it['effective'], 'fallback': it['fallback']}, ensure_ascii=False))}">
  <h3>#{i + 1} {html_escape(it['name'])}</h3>
  <div class="pair"><img src="{it['left']}"><img src="{it['right']}"></div>
  <div class="q">哪一張品質較好？
   <label><input type="radio" name="q{i}" value="L">左</label>
   <label><input type="radio" name="q{i}" value="R">右</label>
   <label><input type="radio" name="q{i}" value="same">看不出差別</label>
  </div>
  <div class="q">假設要在這張稿上繼續修改，可以接手嗎？
   <label><input type="radio" name="e{i}" value="yes">可以接手</label>
   <label><input type="radio" name="e{i}" value="partial">部分可用</label>
   <label><input type="radio" name="e{i}" value="no">寧願重畫</label>
  </div>
  <textarea name="c{i}" placeholder="備註（選填）"></textarea>
 </div>""")
    page = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>向量清稿盲測</title><style>
 body{font-family:system-ui,'Microsoft JhengHei';max-width:1100px;margin:20px auto;padding:0 16px;color:#222}
 .item{border:1px solid #ddd;border-radius:10px;padding:14px;margin:18px 0}
 .pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 .pair img{width:100%;border:1px solid #eee;background:
   repeating-conic-gradient(#eee 0 25%,#fff 0 50%) 0 0/24px 24px}
 .q{margin:8px 0} label{margin-right:14px}
 textarea{width:100%;min-height:36px;font:inherit}
 #export{position:sticky;bottom:12px;padding:10px 18px;font-size:15px;
   background:#1a73e8;color:#fff;border:none;border-radius:8px;cursor:pointer}
 .intro{background:#f6f8fa;border-radius:10px;padding:12px 16px;line-height:1.6}
</style></head><body>
<h2>向量清稿盲測（Stage 1：視覺盲評）</h2>
<div class="intro">每一題左右兩張圖，一張是原始圖稿、一張是工具轉出的向量稿（隨機排列，請不要猜）。
請依直覺回答，全部答完後按最下方「匯出結果」，把下載的檔案傳回即可。<br>
<b>這份測試只檢查視覺近似度與主觀接手意願，不能單獨證明省工 80%。</b>
省工驗收仍需 Stage 2：設計師實際編輯 SVG 並計時。</div>
__ROWS__
<button id="export">匯出結果</button>
<script>
document.getElementById('export').onclick=()=>{
 const items=[...document.querySelectorAll('.item')];
 const out=items.map(it=>{
  const i=it.dataset.i, tool=it.dataset.toolside;
  const q=(document.querySelector(`input[name=q${i}]:checked`)||{}).value||'';
  const e=(document.querySelector(`input[name=e${i}]:checked`)||{}).value||'';
  const c=document.querySelector(`textarea[name=c${i}]`).value;
  let verdict='';
  if(q==='same')verdict='tie';
  else if(q)verdict=(q===tool)?'tool_better':'original_better';
  let meta={}; try{meta=JSON.parse(it.dataset.meta||'{}')}catch(_){}
  return {item:+i+1,...meta,quality:verdict,editable:e,comment:c};});
 const payload={tool:'ai-vector-cleanroom',version:'__TOOL_VERSION__',
  generated:new Date().toISOString(),kind:'visual-blind-test-stage1',
  note:'第一階段視覺盲評;省工驗收需第二階段實際編輯計時',results:out};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);
 a.download='盲測結果.json';a.click();};
</script></body></html>"""
    page = page.replace("__ROWS__", "\n".join(rows) if rows else
                        "<p>output 裡還沒有結果，先轉幾張圖。</p>")
    page = page.replace("__TOOL_VERSION__", vc.TOOL_VERSION)
    dst = OUTPUT_DIR / "blind_test.html"
    dst.write_text(page, encoding="utf-8")
    return dst


def build_editing_test() -> Path:
    """Build the offline Stage 2 timed editing handoff page."""
    from editing_test_page import build_editing_test_page
    return build_editing_test_page(
        OUTPUT_DIR, _list_results(), tool_version=vc.TOOL_VERSION)


APP_HTML = """<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>AI Vector Cleanroom 工作台</title><style>
 body{font-family:system-ui,'Microsoft JhengHei';max-width:1150px;margin:18px auto;padding:0 16px;color:#222}
 #drop{border:2px dashed #9ab;border-radius:12px;padding:34px;text-align:center;color:#578;
   background:#f7fafd;transition:.15s;font-size:15px}
 #drop.over{background:#e3f0ff;border-color:#1a73e8}
 table{border-collapse:collapse;width:100%;margin-top:16px;font-size:13px}
 th,td{border-bottom:1px solid #e5e5e5;padding:7px 8px;text-align:left;vertical-align:middle}
 th{background:#f6f8fa} tr:hover td{background:#fbfdff}
 a{color:#1a73e8;text-decoration:none} a:hover{text-decoration:underline}
 .num{text-align:right;font-variant-numeric:tabular-nums}
 button,select,input[type=number]{font:inherit;padding:3px 8px;border:1px solid #bbb;border-radius:6px;background:#fff}
 button{cursor:pointer} button:hover{background:#eef}
 .primary{background:#1a73e8;color:#fff;border-color:#1a73e8}
 #jobs{font-size:12px;color:#555;margin-top:10px;max-height:130px;overflow:auto;
   background:#fafafa;border:1px solid #eee;border-radius:8px;padding:8px 10px}
 .badge{display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;color:#fff}
 .warn{background:#e65100}.bad{background:#b71c1c}.ok{background:#2e7d32}.fallback{background:#6a1b9a}
 details{margin-top:4px} summary{cursor:pointer;color:#666}
 .opts{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:6px}
 .sub{font-size:11px;color:#666;line-height:1.55;margin-top:3px}
 #notice{display:none;margin:10px 0;padding:8px 10px;border-radius:7px;background:#e8f5e9;color:#1b5e20}
 #notice.err{background:#ffebee;color:#b71c1c}
 button:disabled{opacity:.55;cursor:wait}
</style></head><body>
 <h2>AI Vector Cleanroom 工作台 <small>__TOOL_VERSION__</small>
 <span style="float:right"><button id="blind">產生盲測頁</button>
 <button id="editing">Stage 2 實作計時</button></span></h2>
<div id="drop">把 PNG / JPG 拖進來（可多張），放開就排入轉檔佇列<br>
 <small>或點一下選檔；會依序轉檔，排隊中的每張都會列在下方</small><input type="file" id="file" multiple accept="image/*" hidden></div>
<div id="notice"></div>
<p class="sub">先開啟「設計師接手」：採用可用物件、將難修部位交給人工，再匯出接手包。外觀與結構分數只是診斷；是否省時須以實際完稿計時確認。</p>
<div id="jobs"></div>
<table><thead><tr>
 <th>結果</th><th class="num" title="來源相似度，不代表省工率">相似度診斷</th><th class="num">自動檢查</th>
 <th class="num">設計錨點</th><th>結構</th><th class="num">熱區</th><th>動作</th>
</tr></thead><tbody id="rows"></tbody></table>
<script>
const drop=document.getElementById('drop'),file=document.getElementById('file');
drop.onclick=()=>file.click();
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',e=>send(Array.from(e.dataTransfer.files||[])));
file.onchange=()=>{const selected=Array.from(file.files||[]);file.value='';send(selected);};
const TOKEN='__TOKEN__';
const notice=document.getElementById('notice');
function tell(msg,bad=false){notice.textContent=msg;notice.className=bad?'err':'';notice.style.display='block';}
async function api(url,opts={},quiet=false){
 try{
  const res=await fetch(url,opts); let data={};
  try{data=await res.json()}catch(_){data={}}
  if(!res.ok)throw new Error(data.error||('HTTP '+res.status));
  return data;
 }catch(e){if(!quiet)tell('操作失敗：'+e.message,true);throw e;}
}
async function send(files){
 files=Array.from(files||[]);
 if(!files.length)return;
 let queued=0,failed=[];
 for(const f of files){try{
  await api('/api/upload?name='+encodeURIComponent(f.name),{method:'POST',body:f,headers:{'X-WB-Token':TOKEN}},true);
  queued++;
 }catch(e){failed.push(f.name+'：'+e.message);}}
 if(failed.length){
  const ok=queued?'已排入 '+queued+' 張；':'';
  tell(ok+'有 '+failed.length+' 個檔案未排入：'+failed.join(' ； '),true);
 }else if(queued)tell('已排入 '+queued+' 張圖。工作台會依序轉檔，完成後自動更新。');
 if(queued)poll(true);
}
document.getElementById('blind').onclick=async()=>{
 try{const r=await api('/api/blindtest',{method:'POST',headers:{'X-WB-Token':TOKEN}});
  window.open(r.url,'_blank');}catch(_){}};
document.getElementById('editing').onclick=async()=>{
 try{const r=await api('/api/editingtest',{method:'POST',headers:{'X-WB-Token':TOKEN}});
  window.open(r.url,'_blank');}catch(_){}};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function outputUrl(rel){return '/output/'+String(rel||'').split('/').map(encodeURIComponent).join('/');}
function optionSet(values,current){return values.map(v=>'<option value="'+esc(v)+'"'+(String(v)===String(current)?' selected':'')+'>'+esc(v)+'</option>').join('');}
function optText(opts){const names={background:'背景',geometry:'幾何',strokes:'筆畫',gradients:'漸層',colors:'色數',curve_error_percent:'曲線誤差率%'};
 return Object.entries(opts||{}).filter(([k])=>names[k]).map(([k,v])=>names[k]+'='+v).join(' · ')||'預設';}
function featureText(r){
 const bits=[],scene=r.scene||{},paint=r.paint||{},ops=r.designer_operations||{};
 if(scene.actual_dom_group_count!=null){
  let text='可選群組 '+scene.actual_dom_group_count;
  if((scene.manifest_only_group_count||0)>0)text+='（另 '+scene.manifest_only_group_count+' 個僅建議）';
  bits.push(text);
 }
 if(paint.role_controls!=null){
  let text='換色角色 '+paint.role_controls;
  if(paint.paint_resources_total!=null)text+='／資源 '+paint.paint_resources_total;
  bits.push(text);
 }
 if(ops.total_operations!=null){
  let text='通用結構把手 '+(ops.passed||0)+'/'+ops.total_operations;
  if(ops.human_acceptance==='not_tested'||ops.semantic_task_validation==='not_performed')text+='／真人未驗';
  const pending=[];
  if(ops.partial)pending.push('部分 '+ops.partial);
  if(ops.failed)pending.push('未過 '+ops.failed);
  if(ops.manual_review)pending.push('人工 '+ops.manual_review);
  if(pending.length)text+='（'+pending.join('、')+'）';
  bits.push(text);
 }
 if(r.automation_readiness_score!=null)bits.push('自動化準備 '+Number(r.automation_readiness_score).toFixed(1)+'/100');
 if(r.redraw_ease_score!=null)bits.push('描點收尾 '+Number(r.redraw_ease_score).toFixed(1)+'/100');
 return bits.length?'<div class="sub">結構：'+bits.map(esc).join(' · ')+'</div>':'';
}
async function refresh(){
 let rs; try{rs=await api('/api/list',{},true)}catch(e){tell('無法讀取結果：'+e.message,true);return;}
 const tb=document.getElementById('rows');tb.innerHTML='';
  for(const r of rs){
   const tr=document.createElement('tr');
   const fg=r.foreground==null?'n/a':r.foreground.toFixed(1)+'%';
   const visualBadge=r.visual_acceptance_status==='accepted'
    ?' <span class="badge ok">外觀通過</span>'
    :(r.visual_acceptance_status==='rejected'
      ?' <span class="badge bad">外觀未達標</span>'
      :' <span class="badge warn">外觀需查</span>');
   const editScore=r.editability_score==null?'':(' '+Number(r.editability_score).toFixed(1)+'/100');
   const editBadge=r.editability_status==='accepted'
    ?' <span class="badge ok">基礎結構通過</span>'
    :(r.editability_status==='not_audited'
      ?' <span class="badge warn">基礎結構未評估</span>'
      :' <span class="badge warn">基礎結構需查'+editScore+'</span>');
   const designerBadge=r.designer_readiness_status==='designer_ready'
    ?' <span class="badge ok">漸層／曲線通過</span>'
    :(r.designer_readiness_status==='manual_rework_required'
      ?' <span class="badge bad">漸層／曲線未達標</span>'
      :' <span class="badge warn">漸層／曲線需查</span>');
   const detail=r.detail_p10==null?'':('<div class="sub">局部細節 P10：'+Number(r.detail_p10).toFixed(1)+'%</div>');
   const src=r.source==null?'n/a':r.source.toFixed(1)+'%';
  const fb=Object.keys(r.auto_fallback||{}).length;
  const fbb=fb?' <span class="badge fallback">自動回退</span>':'';
  const cand=(r.candidates||[]).map((c,i)=>{
   const score=c.scores&&c.scores.foreground!=null?Number(c.scores.foreground).toFixed(1)+'%':'n/a';
    return '#'+(i+1)+' '+esc(optText(c.options))+' → 外觀 '+score+'（不含可編輯性）';
  }).join('<br>');
   const hist=(r.history||[]).map((h,i)=>'<a href="'+outputUrl(h.url)+'" download>'+(i===0?'上一版':'更早')+' '+esc(h.label)+'</a>').join('<br>');
   const recolor=r.recolor?' · <a href="'+outputUrl(r.recolor)+'" target="_blank">換色</a>':'';
   const zip=r.zip?' · <a href="'+outputUrl(r.zip)+'" download>下載完整 ZIP</a>':'';
   const eo=r.effective_options||{};
  tr.innerHTML=
   '<td><b>'+esc(r.dir.replace('result_',''))+'</b>'+fbb+'<br><small>'+esc(r.input)+'</small><div class="sub">實際：'+esc(optText(eo))+'</div></td>'+
   '<td class="num">'+src+'</td><td class="num">'+fg+visualBadge+editBadge+designerBadge+detail+'</td>'+ 
   '<td class="num">'+r.designer_anchors+
    (r.designer_anchor_source==='legacy_nodes_total'?'<div class="sub">舊版節點估計</div>':'')+'</td>'+
   '<td>'+r.paths+' 路徑 / '+r.native_primitives+' 原生元件 ('+
    r.native_circles+' 圓、'+r.native_rectangles+' 矩形、'+
    r.native_ellipses+' 橢圓、'+r.native_lines+' 線段、'+
    r.native_polylines+' 折線、'+r.native_polygons+' 多邊形) / '+
     r.strokes+' 筆畫 / '+r.gradients+' 漸層'+
     (r.unique_paints_total==null?'':' / '+r.unique_paints_total+' 種實際色彩／漸層資源')+
     featureText(r)+
     '<details><summary>候選 '+(r.candidates||[]).length+' 個</summary><div class="sub">'+(cand||'無候選資料')+'</div></details></td>'+
   '<td class="num">'+r.hotspots+'</td>'+
    '<td>'+(r.handoff?'<a class="primary" style="padding:5px 8px;border-radius:5px;display:inline-block" href="'+esc(r.handoff)+'" target="_blank">設計師接手</a><br>':'')+
    (r.review?'<a href="'+outputUrl(r.review)+'" target="_blank">開啟校稿</a> · ':'')+
    '<a href="'+outputUrl(r.svg)+'" download>下載 SVG</a>'+zip+recolor+
   ((r.history||[]).length?'<details><summary>歷史版本 '+r.history.length+'</summary><div class="sub">'+hist+'</div></details>':'')+
   '<details><summary>此圖重跑</summary><div class="opts">'+
    '背景 <select class="o" data-k="background">'+optionSet(['auto','keep','transparent'],eo.background||'auto')+'</select>'+ 
    '幾何 <select class="o" data-k="geometry">'+optionSet(['conservative','normal','off'],eo.geometry||'conservative')+'</select>'+ 
    '筆畫 <select class="o" data-k="strokes">'+optionSet(['on','off'],eo.strokes||'on')+'</select>'+ 
    '漸層 <select class="o" data-k="gradients">'+optionSet(['on','off'],eo.gradients||'on')+'</select>'+ 
    '色數 <input type="number" class="o" data-k="colors" value="'+esc(eo.colors??0)+'" min="0" max="64" title="0=自動，或 2–64" style="width:56px">'+
    '曲線誤差% <input type="number" class="o" data-k="curve_error_percent" value="'+esc(eo.curve_error_percent??0.25)+'" min="0.05" max="2" step="0.05" title="先限制幾何誤差，再最小化錨點" style="width:64px">'+
    '<button class="primary rerun" data-input="'+esc(r.input)+'" data-base="'+esc(r.base)+'">另開版本重跑</button>'+ 
   '</div></details></td>';
  tb.appendChild(tr);
 }
 document.querySelectorAll('.rerun').forEach(b=>b.onclick=async()=>{
  const opts={};
  b.parentElement.querySelectorAll('.o').forEach(el=>opts[el.dataset.k]=el.value);
  const q=new URLSearchParams({name:b.dataset.input,base:b.dataset.base,branch:'1',...opts});
  b.disabled=true;
  try{await api('/api/rerun?'+q,{method:'POST',headers:{'X-WB-Token':TOKEN}});
   tell('已排入獨立新版本；現有物件採用決策和接手包保持可用。');poll(true);
  }catch(_){}finally{b.disabled=false;}});
}
let polling=null;
async function poll(force){
 let js;try{js=await api('/api/jobs',{},true)}catch(_){return;}
 const el=document.getElementById('jobs');
 const statusLabel={queued:'排隊中',running:'轉檔中',committing:'安全提交中',cancelling:'正在停止',
                    cancelled:'已取消',timed_out:'已逾時',budget_exhausted:'候選預算用盡',done:'完成',
                    review:'完成但需檢查',rejected:'未達標',failed:'失敗'};
 el.innerHTML=js.slice(-12).reverse().map(j=>
  '['+j.t+'] '+esc(j.name)+' — '+(statusLabel[j.status]||esc(j.status))+
  (j.elapsed_seconds?' · 已用 '+Number(j.elapsed_seconds).toFixed(1)+' 秒'+
    (j.budget_seconds?'（硬上限 '+Math.round(Number(j.budget_seconds))+' 秒，非 ETA）':''):'')+
  (j.candidate_evaluated?' · 候選 '+j.candidate_evaluated+
    (j.candidate_planned!=null?'/'+j.candidate_planned:''):'')+
  (j.detail?' · '+esc(j.detail):'')+
  (j.cancellable?' <button class="cancel-job" data-job="'+j.id+'">取消</button>':''))
  .join('<br>')||'（尚無工作）';
 el.querySelectorAll('.cancel-job').forEach(b=>b.onclick=async()=>{
  b.disabled=true;
  try{await api('/api/cancel?job_id='+encodeURIComponent(b.dataset.job),
    {method:'POST',headers:{'X-WB-Token':TOKEN}});poll(true);}catch(_){}
 });
 // Jobs are updated in place.  Looking only at the last event used to stop
 // refresh as soon as a later file was merely waiting in the queue.
 const busy=js.some(j=>j.status==='queued'||j.status==='running'||j.status==='committing'||j.status==='cancelling');
 // A completed earlier item must appear immediately even when later files
 // are still queued.  Previously this refreshed only after the whole batch,
 // making a real successful result look missing.
 refresh();
 if(busy){ if(!polling)polling=setInterval(poll,1500);}
 else if(polling){clearInterval(polling);polling=null;}
}
refresh();poll();
</script></body></html>""".replace("__TOOL_VERSION__", vc.TOOL_VERSION)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle_one_request(self):
        try:
            return super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            # A browser can close a persistent connection between requests or
            # while receiving a response. Other server errors must propagate.
            self.close_connection = True

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/":
            return self._send(200, APP_HTML.replace("__TOKEN__", WB_TOKEN))
        if path == "/api/list":
            with _publish_lock:
                payload = json.dumps(_list_results())
            return self._send(200, payload, "application/json")
        if path == "/api/jobs":
            return self._send(200, json.dumps(_snapshot_jobs()),
                              "application/json")
        if path == "/handoff":
            from handoff_service import build_page, HandoffConflict
            query = urllib.parse.parse_qs(parsed.query)
            try:
                with _publish_lock:
                    body = build_page(OUTPUT_DIR, query.get("result", [""])[0], WB_TOKEN)
                return self._send(200, body)
            except HandoffConflict as exc:
                return self._send(409, html_escape(str(exc)), "text/plain; charset=utf-8")
            except (ValueError, OSError) as exc:
                return self._send(400, str(exc), "text/plain; charset=utf-8")
        if path.startswith("/output/"):
            with _publish_lock:
                rel = urllib.parse.unquote(path[len("/output/"):])
                f = (OUTPUT_DIR / rel).resolve()
                if OUTPUT_DIR.resolve() not in f.parents or not f.is_file():
                    return self._send(404, "not found", "text/plain")
                ctype = {"svg": "image/svg+xml", "png": "image/png",
                         "html": "text/html; charset=utf-8",
                         "json": "application/json",
                         "zip": "application/zip",
                         "txt": "text/plain; charset=utf-8"}.get(
                    f.suffix.lstrip(".").lower(), "application/octet-stream")
                body = f.read_bytes()
            return self._send(200, body, ctype)
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if WB_TOKEN and self.headers.get("X-WB-Token") != WB_TOKEN:
            # Consume a small rejected request before closing, otherwise
            # Windows can reset the socket while the client is sending JSON.
            try:
                rejected_length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                rejected_length = 0
            if 0 < rejected_length <= 2 * 1024 * 1024:
                self.rfile.read(rejected_length)
            else:
                self.close_connection = True
            return self._send(403, json.dumps({"error": "bad token"}),
                              "application/json")
        if parsed.path in ("/api/handoff/save", "/api/handoff/export", "/api/handoff/refine", "/api/handoff/prepare"):
            from handoff_service import save_decisions, export_package, HandoffConflict
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2 * 1024 * 1024:
                    raise ValueError("決策資料大小不正確")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("決策必須是 JSON 物件")
                if parsed.path.endswith("/prepare"):
                    from auto_prepare import prepare_result
                    result = prepare_result(OUTPUT_DIR, payload, lock=_publish_lock)
                elif parsed.path.endswith("/refine"):
                    from local_refine import refine_result
                    # Rendering can take seconds. The refinement writes a new
                    # result; only source snapshots and publication take the lock.
                    result = refine_result(OUTPUT_DIR, payload, lock=_publish_lock)
                else:
                    with _publish_lock:
                        result = (save_decisions(OUTPUT_DIR, payload)
                                  if parsed.path.endswith("/save")
                                  else export_package(OUTPUT_DIR, payload))
                return self._send(200, json.dumps(result, ensure_ascii=False), "application/json")
            except HandoffConflict as exc:
                return self._send(409, json.dumps({"error": str(exc)}, ensure_ascii=False), "application/json")
            except (ValueError, OSError, RuntimeError) as exc:
                return self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False), "application/json")
        if parsed.path == "/api/cancel":
            try:
                job_id = int(qs.get("job_id", [""])[0])
            except ValueError:
                return self._send(400, json.dumps(
                    {"error": "job_id must be an integer"}),
                    "application/json")
            outcome = _cancel_job(job_id)
            if outcome == "missing":
                return self._send(404, json.dumps(
                    {"error": "job not found"}), "application/json")
            if outcome == "terminal":
                return self._send(409, json.dumps(
                    {"error": "job is already finished"}),
                    "application/json")
            if outcome == "too_late":
                return self._send(409, json.dumps(
                    {"error": "job is already committing its verified result"}),
                    "application/json")
            return self._send(200, json.dumps(
                {"job_id": job_id, "status": outcome}),
                "application/json")
        if parsed.path == "/api/upload":
            name = _safe_name(qs.get("name", ["image.png"])[0])
            if Path(name).suffix.lower() not in vc.EXTS:
                return self._send(400, json.dumps(
                    {"error": "unsupported file type"}), "application/json")
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 80 * 1024 * 1024:
                return self._send(400, json.dumps(
                    {"error": "bad size"}), "application/json")
            data = self.rfile.read(length)
            try:
                import io
                from PIL import Image as _Im
                _Im.open(io.BytesIO(data)).verify()
            except Exception:
                return self._send(400, json.dumps(
                    {"error": "file is not a decodable image"}),
                    "application/json")
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            dst = _unique_input_path(name)
            dst.write_bytes(data)
            job_id = _enqueue_job(dst, {}, None)
            return self._send(200, json.dumps({"queued": dst.name,
                                               "job_id": job_id}),
                              "application/json")
        if parsed.path == "/api/rerun":
            name = _safe_name(qs.get("name", [""])[0])
            src = INPUT_DIR / name
            if not src.exists():
                return self._send(404, json.dumps(
                    {"error": f"input/{name} not found"}), "application/json")
            base = _safe_name(qs.get("base", [""])[0])
            target = OUTPUT_DIR / f"result_{base}"
            report_path = target / "report.json"
            with _publish_lock:
                if not base or not report_path.is_file():
                    return self._send(404, json.dumps(
                        {"error": "the selected successful result no longer exists"}),
                        "application/json")
                try:
                    current_report = json.loads(
                        report_path.read_text(encoding="utf-8"))
                except Exception:
                    return self._send(400, json.dumps(
                        {"error": "the selected result report is unreadable"}),
                        "application/json")
                if current_report.get("input") != name:
                    return self._send(400, json.dumps(
                        {"error": "the selected result does not belong to this input"}),
                        "application/json")
                from handoff_service import has_locked_objects
                branch = qs.get("branch", [""])[0] == "1"
                try:
                    if not branch and has_locked_objects(OUTPUT_DIR, target.name):
                        return self._send(409, json.dumps({"error": "已採用物件已鎖定，請另開版本重跑"}, ensure_ascii=False), "application/json")
                except (ValueError, OSError):
                    return self._send(409, json.dumps({"error": "現有接手決策需先確認；請另開版本重跑"}, ensure_ascii=False), "application/json")
            overrides = {k: qs[k][0] for k in
                         ("background", "geometry", "strokes", "gradients",
                          "colors", "curve_error_percent") if k in qs}
            choices = {
                "background": {"auto", "keep", "transparent"},
                "geometry": {"conservative", "normal", "off"},
                "strokes": {"on", "off"},
                "gradients": {"on", "off"},
            }
            for key, allowed in choices.items():
                if key in overrides and overrides[key] not in allowed:
                    return self._send(400, json.dumps(
                        {"error": f"invalid {key} option"}),
                        "application/json")
            try:
                colors = int(overrides.get("colors", "0"))
                if colors != 0 and not 2 <= colors <= 64:
                    raise ValueError
            except ValueError:
                return self._send(400, json.dumps(
                    {"error": "colors must be 0 or between 2 and 64"}),
                    "application/json")
            try:
                curve_error = float(overrides.get("curve_error_percent", "0.25"))
                if not math.isfinite(curve_error) or not 0.05 <= curve_error <= 2.0:
                    raise ValueError
            except ValueError:
                return self._send(400, json.dumps(
                    {"error": "curve_error_percent must be between 0.05 and 2.0"}),
                    "application/json")
            if branch:
                base = bounded_output_base(base[:28] + "_r" + secrets.token_hex(4))
            job_id = _enqueue_job(src, overrides, base)
            return self._send(200, json.dumps({"queued": name,
                                               "job_id": job_id}),
                              "application/json")
        if parsed.path == "/api/blindtest":
            with _publish_lock:
                dst = build_blind_test()
            return self._send(200, json.dumps(
                {"url": f"/output/{dst.name}"}), "application/json")
        if parsed.path == "/api/editingtest":
            with _publish_lock:
                dst = build_editing_test()
            return self._send(200, json.dumps(
                {"url": f"/output/{dst.name}"}), "application/json")
        return self._send(404, "not found", "text/plain")


def _serve_workbench():
    global WB_TOKEN
    import secrets as _sec
    WB_TOKEN = _sec.token_hex(12)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    port = 8765
    for _ in range(20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    else:
        sys.exit("no free port found")
    url = f"http://127.0.0.1:{port}/"
    print("=" * 52)
    print("  AI Vector Cleanroom workbench")
    print(f"  {url}")
    print(f"  Data: {DATA_DIR}")
    print("  (Ctrl+C to stop)")
    print("=" * 52)
    if os.environ.get("AVC_NO_BROWSER") != "1":
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        with _jobs_lock:
            processes = []
            for job in _jobs:
                if job.get("status") in TERMINAL_JOB_STATUSES:
                    continue
                if job.get("status") == "committing":
                    # Publishing is a short non-cancellable transaction. Wait
                    # for it below instead of labelling a committed result as
                    # cancelled.
                    continue
                runtime = _job_runtime.setdefault(job["id"], {})
                runtime["cancel_requested"] = True
                process = runtime.get("process")
                if process is not None:
                    processes.append(process)
                job.update(status="cancelled", stage="cancelled",
                           detail="工作台已關閉", cancellable=False,
                           t=time.strftime("%H:%M:%S"))
        for process in processes:
            try:
                _terminate_process(process)
            except (OSError, subprocess.SubprocessError):
                pass
        # A commit owns this lock; acquiring it guarantees the output pair is
        # no longer inside its rollback window before the interpreter exits.
        with _publish_lock:
            pass
        worker_thread.join(timeout=JOB_TERMINATE_GRACE_SECONDS + 3.0)


def main():
    try:
        with writer_lock(DATA_DIR, "workbench"):
            _serve_workbench()
    except (DataDirectoryBusyError, DataPathError) as exc:
        print(f"[資料目錄錯誤] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
