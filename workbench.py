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
import queue
import random
import re
import shutil
import socket
import sys
import threading
import time
import urllib.parse
import zipfile
from html import escape as html_escape
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import vector_cleanroom as vc

BASE = vc.BASE
INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output"
HISTORY_DIR = OUTPUT_DIR / "_history"
HISTORY_KEEP = 8

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
_queue = queue.Queue()
_job_sequence = 0


def _new_job(name, detail="已排入等待佇列"):
    """Create a visible queued job and return its stable local identifier."""
    global _job_sequence
    with _jobs_lock:
        _job_sequence += 1
        job_id = _job_sequence
        _jobs.append({"id": job_id, "name": name, "status": "queued",
                      "detail": detail, "t": time.strftime("%H:%M:%S")})
        del _jobs[:-60]
        return job_id


def _set_job(job_id, status, detail=""):
    """Update one visible job in place; retain a small completed history."""
    with _jobs_lock:
        for job in reversed(_jobs):
            if job.get("id") == job_id:
                job.update(status=status, detail=detail,
                           t=time.strftime("%H:%M:%S"))
                return
        # Defensive fallback: a worker should never lose its queue record,
        # but a useful failure message is preferable to silently hiding it.
        _jobs.append({"id": job_id, "name": "unknown", "status": status,
                      "detail": detail, "t": time.strftime("%H:%M:%S")})
        del _jobs[:-60]


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


def _archive_result(result_dir: Path):
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


def _run_one(img_path: Path, overrides: dict, requested_base=None, *, job_id=None):
    ap = vc.build_arg_parser()
    args = ap.parse_args([])
    args.input = INPUT_DIR
    args.output = OUTPUT_DIR
    for k, v in overrides.items():
        if hasattr(args, k) and v not in (None, ""):
            setattr(args, k, int(v) if k in ("colors", "white_threshold",
                                             "max_size") else v)
    if requested_base:
        out_base = requested_base
    else:
        plan = vc.plan_output_names(vc.find_inputs(INPUT_DIR))
        out_base = plan.get(img_path, img_path.stem)
    result_dir = OUTPUT_DIR / f"result_{out_base}"
    result_zip = OUTPUT_DIR / f"result_{out_base}.zip"
    status_detail = (" ".join(f"{k}={v}" for k, v in overrides.items())
                     or "defaults")
    if job_id is None:
        job_id = _new_job(img_path.name)
    _set_job(job_id, "running", status_detail)
    try:
        previous = _archive_result(result_dir)
    except Exception as e:
        _set_job(job_id, "failed",
                 f"無法儲存上一個成功版，已取消重跑：{str(e)[:220]}")
        return
    try:
        vc.process_one(img_path, out_base, args, OUTPUT_DIR)
        if not ((result_dir / "report.json").is_file() and result_zip.is_file()):
            raise RuntimeError("輸出不完整（缺 report.json 或 zip）")
        report = json.loads(
            (result_dir / "report.json").read_text(encoding="utf-8"))
        acceptance = str(report.get("acceptance_status") or "manual_review")
        detail = f"result_{out_base}"
        if previous:
            detail += f" · 上一版已存入歷史：{previous.name}"
        if acceptance == "rejected":
            reasons = ((report.get("visual_gate") or {}).get("reasons") or [])
            reason = f" · {'；'.join(str(v) for v in reasons[:2])}" if reasons else ""
            _set_job(job_id, "rejected",
                     f"未達標，已保留診斷檔但請勿交付 · {detail}{reason}")
        elif acceptance != "accepted":
            _set_job(job_id, "review", f"完成但需要人工檢查 · {detail}")
        else:
            _set_job(job_id, "done", detail)
    except Exception as e:
        shutil.rmtree(result_dir, ignore_errors=True)
        result_zip.unlink(missing_ok=True)
        if previous:
            try:
                _restore_result(previous, result_dir, result_zip)
                detail = f"{str(e)[:220]} · 已復原上一個成功版"
            except Exception as restore_error:
                detail = (f"{str(e)[:160]} · 復原失敗："
                          f"{str(restore_error)[:120]}")
        else:
            detail = f"{str(e)[:250]} · 無可復原的舊版"
        _set_job(job_id, "failed", detail)


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
    return name


def _unique_input_path(name: str) -> Path:
    p = INPUT_DIR / name
    stem, suf = p.stem, p.suffix
    i = 2
    while p.exists():
        p = INPUT_DIR / f"{stem}_{i}{suf}"
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
        primitive_counts = _report_primitive_counts(rep)
        requested, effective, fallback = _report_options(rep)
        features = _report_feature_summary(rep)
        legacy_status = rep.get("acceptance_status", "accepted")
        visual_status = rep.get("visual_acceptance_status", legacy_status)
        editability_status = rep.get("editability_status", "not_audited")
        automation_readiness = _mapping(rep.get("automation_readiness"))
        redraw_complexity = _mapping(rep.get("redraw_complexity"))
        human_validation = _mapping(rep.get("human_validation"))
        detail_grid = rep.get("detail_grid") or {}
        history = [{
            "url": f"_history/{p.name}",
            "label": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(p.stat().st_mtime)),
        } for p in _history_files(rd.name)]
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
            "review": f"{rd.name}/review.html",
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
<div id="jobs"></div>
<table><thead><tr>
 <th>結果</th><th class="num">source</th><th class="num">外觀／可編輯性</th>
 <th class="num">節點</th><th>結構</th><th class="num">熱區</th><th>動作</th>
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
function optionSet(values,current){return values.map(v=>'<option value="'+esc(v)+'"'+(String(v)===String(current)?' selected':'')+'>'+esc(v)+'</option>').join('');}
function optText(opts){const names={background:'背景',geometry:'幾何',strokes:'筆畫',gradients:'漸層',colors:'色數'};
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
    ?' <span class="badge ok">可編輯性通過</span>'
    :(r.editability_status==='not_audited'
      ?' <span class="badge warn">可編輯性未評估</span>'
      :' <span class="badge warn">可編輯性需查'+editScore+'</span>');
   const detail=r.detail_p10==null?'':('<div class="sub">局部細節 P10：'+Number(r.detail_p10).toFixed(1)+'%</div>');
   const src=r.source==null?'n/a':r.source.toFixed(1)+'%';
  const fb=Object.keys(r.auto_fallback||{}).length;
  const fbb=fb?' <span class="badge fallback">自動回退</span>':'';
  const cand=(r.candidates||[]).map((c,i)=>{
   const score=c.scores&&c.scores.foreground!=null?Number(c.scores.foreground).toFixed(1)+'%':'n/a';
    return '#'+(i+1)+' '+esc(optText(c.options))+' → 外觀 '+score+'（不含可編輯性）';
  }).join('<br>');
   const hist=(r.history||[]).map((h,i)=>'<a href="/output/'+encodeURI(h.url)+'" download>'+(i===0?'上一版':'更早')+' '+esc(h.label)+'</a>').join('<br>');
   const recolor=r.recolor?' · <a href="/output/'+encodeURI(r.recolor)+'" target="_blank">換色</a>':'';
   const eo=r.effective_options||{};
  tr.innerHTML=
   '<td><b>'+esc(r.dir.replace('result_',''))+'</b>'+fbb+'<br><small>'+esc(r.input)+'</small><div class="sub">實際：'+esc(optText(eo))+'</div></td>'+
   '<td class="num">'+src+'</td><td class="num">'+fg+visualBadge+editBadge+detail+'</td>'+ 
   '<td class="num">'+r.nodes+'</td>'+
   '<td>'+r.paths+' 路徑 / '+r.native_primitives+' 原生元件 ('+
    r.native_circles+' 圓、'+r.native_rectangles+' 矩形、'+
    r.native_ellipses+' 橢圓、'+r.native_lines+' 線段、'+
    r.native_polylines+' 折線、'+r.native_polygons+' 多邊形) / '+
     r.strokes+' 筆畫 / '+r.gradients+' 漸層'+
     (r.unique_paints_total==null?'':' / '+r.unique_paints_total+' 種實際色彩／漸層資源')+
     featureText(r)+
     '<details><summary>候選 '+(r.candidates||[]).length+' 個</summary><div class="sub">'+(cand||'無候選資料')+'</div></details></td>'+
   '<td class="num">'+r.hotspots+'</td>'+
    '<td><a href="/output/'+encodeURI(r.review)+'" target="_blank">校稿</a> · '+
    '<a href="/output/'+encodeURI(r.svg)+'" target="_blank">SVG</a>'+recolor+
   ((r.history||[]).length?'<details><summary>歷史版本 '+r.history.length+'</summary><div class="sub">'+hist+'</div></details>':'')+
   '<details><summary>此圖重跑</summary><div class="opts">'+
    '背景 <select class="o" data-k="background">'+optionSet(['auto','keep','transparent'],eo.background||'auto')+'</select>'+ 
    '幾何 <select class="o" data-k="geometry">'+optionSet(['conservative','normal','off'],eo.geometry||'conservative')+'</select>'+ 
    '筆畫 <select class="o" data-k="strokes">'+optionSet(['on','off'],eo.strokes||'on')+'</select>'+ 
    '漸層 <select class="o" data-k="gradients">'+optionSet(['on','off'],eo.gradients||'on')+'</select>'+ 
    '色數 <input type="number" class="o" data-k="colors" value="'+esc(eo.colors??0)+'" min="0" max="64" title="0=自動，或 2–64" style="width:56px">'+
    '<button class="primary rerun" data-input="'+esc(r.input)+'" data-base="'+esc(r.base)+'">此圖重跑</button>'+ 
   '</div></details></td>';
  tb.appendChild(tr);
 }
 document.querySelectorAll('.rerun').forEach(b=>b.onclick=async()=>{
  const opts={};
  b.parentElement.querySelectorAll('.o').forEach(el=>opts[el.dataset.k]=el.value);
  const q=new URLSearchParams({name:b.dataset.input,base:b.dataset.base,...opts});
  b.disabled=true;
  try{await api('/api/rerun?'+q,{method:'POST',headers:{'X-WB-Token':TOKEN}});
   tell('已排入此圖重跑。工作台會先保存上一個成功版，失敗時自動復原。');poll(true);
  }catch(_){}finally{b.disabled=false;}});
}
let polling=null;
async function poll(force){
 let js;try{js=await api('/api/jobs',{},true)}catch(_){return;}
 const el=document.getElementById('jobs');
 const statusLabel={queued:'排隊中',running:'轉檔中',done:'完成',review:'完成但需檢查',
                    rejected:'未達標',failed:'失敗'};
 el.innerHTML=js.slice(-12).reverse().map(j=>
  '['+j.t+'] '+esc(j.name)+' — '+(statusLabel[j.status]||esc(j.status))+
  (j.detail?' · '+esc(j.detail):'')).join('<br>')||'（尚無工作）';
 // Jobs are updated in place.  Looking only at the last event used to stop
 // refresh as soon as a later file was merely waiting in the queue.
 const busy=js.some(j=>j.status==='queued'||j.status==='running');
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
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            return self._send(200, APP_HTML.replace("__TOKEN__", WB_TOKEN))
        if path == "/api/list":
            return self._send(200, json.dumps(_list_results()),
                              "application/json")
        if path == "/api/jobs":
            with _jobs_lock:
                return self._send(200, json.dumps(list(_jobs)),
                                  "application/json")
        if path.startswith("/output/"):
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
            return self._send(200, f.read_bytes(), ctype)
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if WB_TOKEN and self.headers.get("X-WB-Token") != WB_TOKEN:
            return self._send(403, json.dumps({"error": "bad token"}),
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
            INPUT_DIR.mkdir(exist_ok=True)
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
            overrides = {k: qs[k][0] for k in
                         ("background", "geometry", "strokes", "gradients",
                          "colors") if k in qs}
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
            job_id = _enqueue_job(src, overrides, base)
            return self._send(200, json.dumps({"queued": name,
                                               "job_id": job_id}),
                              "application/json")
        if parsed.path == "/api/blindtest":
            dst = build_blind_test()
            return self._send(200, json.dumps(
                {"url": f"/output/{dst.name}"}), "application/json")
        if parsed.path == "/api/editingtest":
            dst = build_editing_test()
            return self._send(200, json.dumps(
                {"url": f"/output/{dst.name}"}), "application/json")
        return self._send(404, "not found", "text/plain")


def main():
    global WB_TOKEN
    import secrets as _sec
    WB_TOKEN = _sec.token_hex(12)
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
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
    print("  (Ctrl+C to stop)")
    print("=" * 52)
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
