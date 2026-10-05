"""Persistent designer decisions and immutable exports for the local workbench.

The original conversion is never edited by this module. A decision is bound to
the exact SVG bytes, and saving decisions does not certify designer acceptance.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import threading
import time
import urllib.parse
import zipfile

from designer_handoff import (build_handoff_manifest, export_handoff,
                              normalized_svg_text, validate_decisions)

STATE_NAME = "handoff_state.json"
_state_lock = threading.RLock()


class HandoffConflict(ValueError):
    pass


def result_context(output_root: Path, result: str):
    root = Path(output_root).resolve()
    if (not isinstance(result, str) or not result.startswith("result_")
            or Path(result).name != result or any(c in result for c in "/\\:")):
        raise ValueError("結果名稱無效")
    folder = (root / result).resolve()
    if folder.parent != root or not folder.is_dir():
        raise ValueError("找不到這個結果")
    report_path = folder / "report.json"
    svgs = sorted(folder.glob("*_vector.svg"))
    source = folder / "source_original.png"
    original_available = source.is_file()
    if not original_available:
        source = folder / "source_reference.png"
    if len(svgs) != 1 or not report_path.is_file() or not source.is_file():
        raise ValueError("結果缺少唯一 SVG、原圖或報告，無法建立接手稿")
    for path in (svgs[0], report_path, source):
        if path.resolve().parent != folder:
            raise ValueError("結果檔案不可連結到資料夾外")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["_handoff_reference_kind"] = "original" if original_available else "processed_reference"
    report["_handoff_source_png"] = str(source)
    return folder, svgs[0], source, report


def _atomic_json(path: Path, value: dict):
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_state(folder: Path):
    path = folder / STATE_NAME
    if not path.is_file():
        if path.is_symlink() or path.exists():
            raise HandoffConflict("決策檔案無法讀取；請保留舊記錄並另開版本")
        return None
    if path.resolve().parent != folder.resolve():
        raise ValueError("決策檔案不可連結到資料夾外")
    raw = path.read_bytes()
    try:
        state = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise HandoffConflict("已保存的決策檔案損壞；未覆寫舊記錄") from exc
    if not isinstance(state, dict):
        raise HandoffConflict("已保存的決策格式不正確；未覆寫舊記錄")
    if "revision" not in state:
        # Read-only compatibility for pre-CAS records: two old tabs still get
        # a stable comparison token without silently rewriting the saved file.
        state["revision"] = "legacy-" + hashlib.sha256(raw).hexdigest()
    if not isinstance(state["revision"], str) or not state["revision"]:
        raise HandoffConflict("已保存的決策版本不正確；未覆寫舊記錄")
    return state


def validate_payload_revision(folder: Path, payload: dict):
    """Compare-and-swap precondition; callers must hold their publication lock.

    New results require an explicit null revision. Existing results require
    the revision from the last page/save/export response, including legacy
    records. The returned record is a snapshot, not an acceptance certificate.
    """
    if not isinstance(payload, dict) or "revision" not in payload:
        raise HandoffConflict("缺少決策版本；請重新開啟接手頁，現有判斷未被覆寫")
    revision = payload["revision"]
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise HandoffConflict("決策版本不正確；請重新開啟接手頁")
    state = _read_state(folder)
    current = state["revision"] if state is not None else None
    if revision != current:
        raise HandoffConflict("另一個頁面已更新接手判斷；未覆寫較新的決策，請重新開啟接手頁")
    return state


def load_decisions(folder: Path, manifest: dict):
    state = _read_state(folder)
    if state is None:
        return None
    if state.get("svg_sha256") != manifest["svg_sha256"]:
        raise HandoffConflict("SVG 已變更；舊決策已保留，請建立新的結果版本再校稿")
    validate_decisions(manifest, state.get("decisions"), state.get("svg_sha256"))
    return state["decisions"]


def build_page(output_root: Path, result: str, token: str):
    from handoff_page import build_handoff_page
    folder, svg, source, report = result_context(output_root, result)
    manifest = build_handoff_manifest(svg, report, source_png=source)
    manifest["reference_kind"] = report["_handoff_reference_kind"]
    manifest["auto_prepare"] = report.get("auto_prepare")
    decisions = load_decisions(folder, manifest)
    state = _read_state(folder)
    manifest["saved_revision"] = state["revision"] if state is not None else None
    reference = "data:image/png;base64," + base64.b64encode(source.read_bytes()).decode("ascii")
    return build_handoff_page(manifest, normalized_svg_text(svg), reference,
                              result_dir=result, token=token,
                              saved_decisions=decisions)


def _next_state(folder, svg, report, payload):
    manifest = build_handoff_manifest(svg, report, source_png=report.get('_handoff_source_png'))
    # Preserve an existing stale record even if a caller sends the new digest.
    load_decisions(folder, manifest)
    validate_payload_revision(folder, payload)
    if payload.get("svg_sha256") != manifest["svg_sha256"]:
        raise HandoffConflict("這張 SVG 已更新，請重新開啟接手頁；未覆寫你的舊決策")
    decisions = validate_decisions(manifest, payload.get("decisions"), payload.get("svg_sha256"))
    return {"schema": "aivc.handoff-state/v1", "svg_sha256": manifest["svg_sha256"],
             "decisions": decisions, "saved_unix": time.time(),
             "revision": secrets.token_hex(16),
             "human_timed_validation": False}


def save_decisions(output_root: Path, payload: dict):
    if not isinstance(payload, dict):
        raise ValueError("決策必須是 JSON 物件")
    with _state_lock:
        folder, svg, _source, report = result_context(output_root, payload.get("result"))
        state = _next_state(folder, svg, report, payload)
        _atomic_json(folder / STATE_NAME, state)
        return state


def has_locked_objects(output_root: Path, result: str):
    """Fail closed on corrupt/stale state so a rerun cannot destroy decisions."""
    folder = Path(output_root) / result
    if not (folder / STATE_NAME).exists():
        return False
    folder, svg, _source, report = result_context(output_root, result)
    manifest = build_handoff_manifest(svg, report, source_png=report.get('_handoff_source_png'))
    decisions = load_decisions(folder, manifest) or {}
    return any(value == "keep" for value in decisions.values())


def _url(output_root: Path, path: Path):
    relative = path.resolve().relative_to(Path(output_root).resolve()).as_posix()
    return "/output/" + urllib.parse.quote(relative, safe="/")


def export_package(output_root: Path, payload: dict):
    if not isinstance(payload, dict):
        raise ValueError("決策必須是 JSON 物件")
    with _state_lock:
        return _export_package_locked(output_root, payload)


def _export_package_locked(output_root: Path, payload: dict):
    folder, svg, source, report = result_context(output_root, payload.get("result"))
    # Do not advance the client's revision when rendering/packaging fails.
    # Both the immutable package and the decision write must succeed first.
    state = _next_state(folder, svg, report, payload)
    destination = folder / ("handoff_" + secrets.token_hex(6))
    if destination.exists():
        raise RuntimeError("接手包名稱衝突，請再試一次")
    result = export_handoff(svg, source, state["decisions"], destination,
                            expected_sha256=state["svg_sha256"], report=report)
    instruction = destination / "OPEN_IN_ILLUSTRATOR.txt"
    reference_label = "原圖參考" if report["_handoff_reference_kind"] == "original" else "清理後參考圖（非未處理原圖）"
    instruction.write_text(
        "Illustrator 接手包\n\n"
        "直接接著修：先開啟 working.svg，完整候選向量可直接編輯，不必先逐一標記採用。\n"
        f"working.svg 的{reference_label}預設隱藏，需要比對時再顯示並鎖定；交人工區域也仍保留原候選。\n"
        f"想照原圖重描：開啟 draft.svg，內含對位的{reference_label}及人工標記。\n"
        "accepted.svg 只含你已採用的物件；有待確認或待重畫物件時，它是不完整的向量底稿。\n"
        f"請在 Illustrator 圖層面板檢查並鎖定{reference_label}，依 handoff.json 的清單完成剩餘部位。\n"
        "候選保留既有向量的堆疊與曲線；群組是目前可辨認的編輯單位，未承諾恢復原作者語意。\n"
        "瀏覽器預覽不等於 Illustrator 匯入驗收；請確認漸層、透明度、孔洞與堆疊。\n"
        "working.svg 和 draft.svg 都含嵌入點陣參考，並非純向量完稿；最後移除參考／標記、完成設計檢查，再另存 .ai。\n",
        encoding="utf-8")
    archive = destination.with_suffix(".zip")
    # The unique path means a failed archive can never replace a good export.
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as package:
        for path in sorted(destination.iterdir()):
            if path.is_file():
                package.write(path, path.name)
    _atomic_json(folder / STATE_NAME, state)
    files = [{"name": "完整 Illustrator 接手包.zip", "url": _url(output_root, archive)}]
    files.extend({"name": Path(path).name, "url": _url(output_root, Path(path))}
                 for path in result["files"].values())
    return {"files": files, "summary": result["summary"],
            "svg_sha256": state["svg_sha256"], "revision": state["revision"]}
