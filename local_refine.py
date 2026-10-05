"""Bounded local curve simplification; immutable output and fail-closed gates.

This does not redraw an image or certify final designer acceptance. Only a
selected solid, closed fill can change. All other SVG elements stay identical
to the normalized parent tree, including order and presentation attributes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET

from designer_handoff import (DRAWABLE, _geometry, _local, _style,
                              build_handoff_manifest, normalized_svg_text,
                              validate_decisions)
from handoff_service import (HandoffConflict, result_context, save_decisions,
                             validate_payload_revision)

MAX_UNITS = 4
MAX_PATHS = 4
MAX_PATH_ANCHORS = 512
MAX_TOTAL_ANCHORS = 1536
MAX_SVG_BYTES = 5 * 1024 * 1024
TIMEOUT_SECONDS = 60
GEOMETRY = {"d", "cx", "cy", "r", "rx", "ry", "transform"}


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _write_svg(path, root):
    Path(path).write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))


def _by_id(root):
    return {element.get("id"): element for element in root.iter() if element.get("id")}


def _selection(root, manifest, decisions, object_ids, history):
    if (not isinstance(object_ids, list) or not 1 <= len(object_ids) <= MAX_UNITS
            or any(not isinstance(item, str) for item in object_ids)
            or len(set(object_ids)) != len(object_ids)):
        raise ValueError("一次請選擇 1–4 個不同物件")
    units = {item["id"]: item for item in manifest["objects"]}
    if any(item not in units for item in object_ids):
        raise ValueError("選取物件不存在")
    if any(decisions[item] == "keep" for item in object_ids):
        raise ValueError("已採用並鎖定的物件不可精修；請先明確改為待確認")
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise ValueError("精修歷史無效，不能安全判斷累積漂移")
    previous = set()
    for item in history:
        members = item.get("member_ids")
        if not isinstance(members, list) or any(not isinstance(member, str) for member in members):
            raise ValueError("精修歷史缺少有效物件身分")
        previous.update(members)
    members = [member for identifier in object_ids for member in units[identifier]["member_ids"]]
    if len(members) > MAX_PATHS:
        raise ValueError("一次最多精修 4 條封閉單色路徑，請縮小選取範圍")
    if previous.intersection(members):
        raise ValueError("這個物件已精修過；請回到原始版本選擇另一誤差，避免累積漂移")
    parents = {child: parent for parent in root.iter() for child in parent}
    elements = _by_id(root)
    total = 0
    for identifier in members:
        element = elements[identifier]
        if _local(element.tag) != "path" or len(element):
            raise ValueError("目前僅支援封閉單色填色 path；原生形狀、文字與其他物件請人工處理")
        if len(element.get("d", "")) > 65536:
            raise ValueError("單一路徑過長，請改為人工處理")
        # Cheap closure preflight before spawning; the engine performs the
        # full grammar/geometry parse in the timed child process.
        parts = re.split(r"[Mm]", element.get("d", ""))[1:]
        if not parts or any(not re.search(r"[Zz]\s*$", part) for part in parts):
            raise ValueError("目前僅支援封閉填色輪廓")
        chain, current = [], element
        while current is not None:
            chain.append(current)
            current = parents.get(current)
        fill = "black"
        for ancestor in reversed(chain):
            attrs = {**ancestor.attrib, **_style(ancestor)}
            if any(key in attrs for key in ("transform", "mask", "clip-path", "filter")):
                raise ValueError("有 transform、mask、clip 或 filter 的物件暫不支援局部精修")
            if any("gradient-object" in key or "url(" in str(value).lower()
                   for key, value in attrs.items()):
                raise ValueError("漸層歸屬或資源依賴物件不可用此模式精修")
            if str(attrs.get("stroke", "none")).strip().lower() != "none":
                raise ValueError("帶筆畫的物件不可用填色曲線精修")
            # Unknown CSS layout/compositing cannot be proven equivalent here.
            if any(key not in {"fill", "fill-rule", "fill-opacity", "stroke", "stroke-width",
                               "opacity", "display", "visibility", "color"}
                   for key in _style(ancestor)):
                raise ValueError("物件含不支援的樣式上下文")
            fill = attrs.get("fill", fill)
        if str(fill).strip().lower() in {"none", "transparent", "inherit", "currentcolor"}:
            raise ValueError("需要明確的單色填色")
        anchors, box = _geometry(element)
        if anchors < 5 or anchors > MAX_PATH_ANCHORS or box is None:
            raise ValueError("路徑節點需介於 5–512 且有明確邊界；簡單形狀不需再減點")
        total += anchors
    if total > MAX_TOTAL_ANCHORS:
        raise ValueError("選取節點總數超過 1536，請縮小範圍")
    return members


def _isolate(root, identifier, *, silhouette=False, box=None):
    """Clone only the target and its ancestor presentation chain."""
    target = _by_id(root)[identifier]
    parents = {child: parent for parent in root.iter() for child in parent}
    chain, current = [], target
    while current is not root:
        chain.append(current)
        current = parents[current]
    result = ET.Element(root.tag, dict(root.attrib))
    parent = result
    for original in reversed(chain):
        child = ET.Element(original.tag, dict(original.attrib))
        parent.append(child)
        parent = child
    if silhouette:
        # A square, target-sized viewport stops a small object disappearing
        # under a large whole-document render score. Preserve its fill rule.
        x, y, width, height = box
        side = max(width, height)
        if side <= 0:
            raise ValueError("選取輪廓無有效面積")
        side *= 1.1
        result.set("viewBox", f"{x + width / 2 - side / 2} {y + height / 2 - side / 2} {side} {side}")
        result.set("width", "768")
        result.set("height", "768")
        for item in result.iter():
            style = _style(item)
            if "fill-rule" in style:
                item.set("fill-rule", style["fill-rule"])
            item.attrib.pop("style", None)
            for key in ("fill", "stroke", "opacity", "fill-opacity", "display", "visibility"):
                item.attrib.pop(key, None)
        parent.set("fill", "#000000")
        parent.set("stroke", "none")
    return result


def _target_gate(before, after, identifier, folder):
    import numpy as np
    from PIL import Image
    import vector_cleanroom as vc
    from curve_refit import _mask_topology
    box = _geometry(_by_id(before)[identifier])[1]
    svg_paths, png_paths = [], []
    for name, tree in (("a", before), ("b", after)):
        path = folder / f"target-{name}.svg"
        png = folder / f"target-{name}.png"
        _write_svg(path, _isolate(tree, identifier, silhouette=True, box=box))
        if not vc.render_svg_png(path, png, size=768):
            raise ValueError("局部輪廓 renderer 不可用，未產生精修結果")
        svg_paths.append(path)
        png_paths.append(png)
    guard = vc.validate_svg_stage_renders(*svg_paths, "local_refine_target", render_size=768)
    if guard.get("external_render_check") != "completed" or guard.get("accepted") is not True:
        raise ValueError("局部輪廓相似度未過關，未產生精修結果")
    topology = []
    for png in png_paths:
        with Image.open(png) as image:
            mask = np.asarray(image.convert("L")) < 128
        if not mask.any():
            raise ValueError("局部輪廓無可驗證像素，未產生精修結果")
        topology.append(_mask_topology(mask))
    if topology[0] != topology[1]:
        raise ValueError("局部連通元件或孔洞改變，已拒絕精修")
    return {"id": identifier, "render": guard, "components_and_holes": list(topology[0]),
            "target_silhouette_topology": "passed", "render_size": 768}


def _compute(request):
    """Run in the bounded worker. The caller publishes only validated bytes."""
    from curve_refit_stage import propose_svg_curve_refit
    import vector_cleanroom as vc
    folder = Path(request["work"])
    before_path = folder / "before.svg"
    root = ET.parse(before_path).getroot()
    after = copy.deepcopy(root)
    changed, proposals = [], []
    for identifier in request["member_ids"]:
        isolated = folder / "selected.svg"
        candidate = folder / "candidate.svg"
        _write_svg(isolated, _isolate(root, identifier))
        evidence = propose_svg_curve_refit(
            isolated, candidate, error_budget_percent=request["error_budget_percent"],
            minimum_nodes=5, maximum_segments=MAX_PATH_ANCHORS, sample_step=2.0)
        proposed = _by_id(ET.parse(candidate).getroot())[identifier]
        original = _by_id(after)[identifier]
        before_count = _geometry(original)[0]
        after_count = _geometry(proposed)[0]
        if evidence.get("status") != "proposed" or not 0 < after_count < before_count:
            continue
        if _local(proposed.tag) not in {"path", "circle", "ellipse"}:
            raise ValueError("精修提案含未支援形狀")
        # Copy geometry only. The proposal cannot change presentation,
        # resources, metadata, identity, siblings or stacking order.
        original.tag = proposed.tag
        for key in GEOMETRY:
            original.attrib.pop(key, None)
            if key in proposed.attrib:
                original.set(key, proposed.get(key))
        changed.append(identifier)
        proposals.append({"id": identifier, "anchors_before": before_count,
                          "anchors_after": after_count, "proposal": evidence})
    if not changed:
        raise ValueError("所選輪廓在誤差範圍內沒有可安全減少的節點；未產生新版本")
    after_path = folder / "after.svg"
    _write_svg(after_path, after)
    guard = vc.validate_svg_stage_renders(before_path, after_path, "local_refine")
    if guard.get("external_render_check") != "completed" or guard.get("accepted") is not True:
        raise ValueError("完整 renderer 檢查未完成或未通過，未產生精修結果")
    from alpha_topology import compare_composed_alpha
    from designer_handoff import _view_box
    from svg_renderer import render_svg_reference
    view = _view_box(root)
    width = max(1, round(2048 * min(1.0, view[2] / view[3])))
    before_alpha, after_alpha = folder / "before-alpha.png", folder / "after-alpha.png"
    try:
        provenance = {
            'before': render_svg_reference(before_path, before_alpha, width=width, background=None),
            'after': render_svg_reference(after_path, after_alpha, width=width, background=None)}
        composed = compare_composed_alpha(before_alpha, after_alpha)
    except (ValueError, OSError, RuntimeError) as error:
        raise ValueError("簡化後可能使分開的物件黏合、改變孔洞或透明邊緣，或透明圖驗證未完成；已保留原稿") from error
    guard['composed_alpha'] = {**composed, 'render_width': width, 'renderer_provenance': provenance}
    local = [_target_gate(root, after, identifier, folder) for identifier in changed]
    return {"changed_member_ids": changed, "proposals": proposals,
            "whole_svg_render": guard, "target_gates": local,
            "anchors_removed": sum(item["anchors_before"] - item["anchors_after"] for item in proposals)}


def _run_worker(request):
    try:
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request), text=True, encoding="utf-8",
            capture_output=True, timeout=TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise ValueError("局部精修超過 60 秒；原稿與決策保留，請縮小選取範圍") from exc
    try:
        response = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise ValueError("局部精修程序未回傳完整驗證結果") from exc
    if completed.returncode or "error" in response:
        raise ValueError(response.get("error", "局部精修程序失敗"))
    return response


def _derived_report(report, before_manifest, after_manifest, evidence, budget, parent_name):
    result = copy.deepcopy(report)
    # Retain provenance, not stale authoritative score/evidence fields.
    for key in list(result):
        if ("match_percent" in key or key.startswith("foreground_") or key in {
                "visual_gate", "editability_details", "editability_acceptance_gate",
                "designer_quality", "designer_operations", "gradient_object_gate",
                "curve_economy_gate", "automation_readiness", "redraw_complexity",
                "workflow_friction", "named_operation_evidence", "candidate_selection_policy",
                "scores", "editability_score", "final_structure", "engine_structure_before_postprocess",
                "editability_enhancements", "component_repair", "detail_grid", "transparent_light_fidelity",
                "gradient_details", "gradient_geometry_consistency"}):
            result[key] = None
    result.update({"acceptance_status": "manual_review", "manual_review_required": True,
                   "visual_acceptance_status": "manual_review", "editability_status": "not_audited",
                   "designer_readiness_status": "manual_review_required", "candidates": [], "hotspots": [],
                   "paths": sum(item["path_count"] for item in after_manifest["objects"]),
                   "nodes_total": sum(item["anchor_count"] for item in after_manifest["objects"]),
                   "designer_anchors_total": sum(item["anchor_count"] for item in after_manifest["objects"]),
                   "human_validation": {"status": "not_performed", "timed_editing_test_performed": False},
                   "local_refine_validation": {"scope": "parent_to_candidate_curve_simplification_only",
                                               "source_image_quality": "not_revalidated",
                                               "full_image_scores": "invalidated", **evidence}})
    result["local_refine_history"] = list(report.get("local_refine_history", [])) + [{
        "parent_result": parent_name, "parent_svg_sha256": before_manifest["svg_sha256"],
        "derived_svg_sha256": after_manifest["svg_sha256"],
        "member_ids": evidence["changed_member_ids"], "error_budget_percent": budget}]
    result["warnings"] = list(report.get("warnings") or []) + [
        "局部曲線簡化版本：原圖整體品質分數已失效，尚未重新驗收；請人工校稿。"]
    return result


def refine_result(output_root, payload, *, lock):
    root = Path(output_root).resolve()
    if not isinstance(payload, dict):
        raise ValueError("精修參數必須是物件")
    budget = payload.get("error_budget_percent")
    if isinstance(budget, bool) or budget not in (0.1, 0.25, 0.5):
        raise ValueError("局部誤差僅支援 0.1%、0.25% 或 0.5%")
    folder, svg, source, report = result_context(root, payload.get("result"))
    reference = folder / "source_reference.png"
    snapshot_paths = {svg, source, folder / "report.json"}
    if reference.exists():
        if reference.resolve().parent != folder or not reference.is_file():
            raise ValueError("參考圖不可連結到結果資料夾外")
        snapshot_paths.add(reference)
    originals = {path: path.read_bytes() for path in snapshot_paths}
    report = json.loads(originals[folder / "report.json"])
    report["_handoff_reference_kind"] = "original" if source.name == "source_original.png" else "processed_reference"
    state_path = folder / "handoff_state.json"
    with lock:
        validate_payload_revision(folder, payload)
        old_state = state_path.read_bytes() if state_path.is_file() else None
    if len(originals[svg]) > MAX_SVG_BYTES:
        raise ValueError("此 SVG 過大，暫不支援互動精修")
    # Private staging has no result_ prefix and is never listed as a result.
    with tempfile.TemporaryDirectory(prefix=".local-refine-", dir=root) as temporary:
        work = Path(temporary)
        snapshot = work / "parent.svg"
        snapshot.write_bytes(originals[svg])
        manifest = build_handoff_manifest(snapshot, report)
        _x, _y, width, height = manifest["view_box"]
        if max(width / height, height / width) > 8:
            raise ValueError("畫布長寬比超過 8:1，暫不支援互動 renderer 精修")
        if payload.get("svg_sha256") != manifest["svg_sha256"]:
            raise HandoffConflict("原稿已變更，請重新開啟接手頁")
        decisions = validate_decisions(manifest, payload.get("decisions"), payload.get("svg_sha256"))
        normalized = normalized_svg_text(snapshot)
        members = _selection(ET.fromstring(normalized), manifest, decisions,
                             payload.get("object_ids"), report.get("local_refine_history", []))
        (work / "before.svg").write_text(normalized, encoding="utf-8")
        evidence = _run_worker({"work": str(work), "member_ids": members,
                                "error_budget_percent": budget})
        after_path = work / "after.svg"
        if not after_path.is_file() or not evidence.get("changed_member_ids") or evidence.get("anchors_removed", 0) <= 0:
            raise ValueError("精修未減少節點，未產生新版本")
        whole = evidence.get("whole_svg_render") or {}
        if whole.get("external_render_check") != "completed" or whole.get("accepted") is not True:
            raise ValueError("精修缺少完成的 renderer 驗證")
        alpha = whole.get("composed_alpha") or {}
        if alpha.get("external_render_check") != "completed" or alpha.get("accepted") is not True:
            raise ValueError("精修缺少完整透明度與連通區驗證；已保留原稿")
        targets = evidence.get("target_gates") or []
        if ({item.get("id") for item in targets} != set(evidence["changed_member_ids"])
                or any(item.get("target_silhouette_topology") != "passed"
                       or item.get("render", {}).get("external_render_check") != "completed"
                       or item.get("render", {}).get("accepted") is not True for item in targets)):
            raise ValueError("精修缺少完整局部拓撲驗證")
        after_manifest = build_handoff_manifest(after_path)
        total_before = sum(item["anchor_count"] for item in manifest["objects"])
        total_after = sum(item["anchor_count"] for item in after_manifest["objects"])
        if not total_before > total_after or total_before - total_after != evidence["anchors_removed"]:
            raise ValueError("精修節點減量與實際 SVG 不符")
        # Verify the worker's modification boundary independently before publishing.
        before_tree, after_tree = ET.fromstring(normalized), ET.parse(after_path).getroot()
        changed = set(evidence["changed_member_ids"])
        if not changed.issubset(set(members)):
            raise ValueError("精修變更超出選取範圍")
        for tree in (before_tree, after_tree):
            for identifier in changed:
                item = _by_id(tree)[identifier]
                item.tag = "refined-geometry"
                for key in GEOMETRY:
                    item.attrib.pop(key, None)
        if ET.tostring(before_tree) != ET.tostring(after_tree):
            raise ValueError("未選取物件、樣式或堆疊遭變更，已拒絕精修")
        # Inert acceptance metadata describes the old SVG and must not travel
        # as an accepted certificate on changed geometry. Paint metadata stays.
        final_tree = ET.parse(after_path).getroot()
        metadata = next((item for item in final_tree.iter()
                         if _local(item.tag) == "metadata" and
                         item.get("id") == "ai-vector-cleanroom-metadata"), None)
        if metadata is not None:
            metadata.clear()
            metadata.set("id", "ai-vector-cleanroom-metadata")
            metadata.text = json.dumps({"acceptance_status": "manual_review",
                                        "manual_review_required": True,
                                        "source_image_quality": "not_revalidated",
                                        "parent_svg_sha256": manifest["svg_sha256"]})
        for identifier in changed:
            item = _by_id(final_tree)[identifier]
            for key in list(item.attrib):
                if (key.startswith("data-avc-curve-") or key.startswith("data-avc-anchors-")
                        or key in {"data-avc-designer-anchors", "data-avc-error-budget-percent",
                                   "data-avc-p95-error-percent", "data-avc-max-error-percent",
                                   "data-avc-source-edge"}):
                    item.attrib.pop(key)
            item.set("data-local-refine-status", "manual-review")
            item.set("data-local-refine-error-budget-percent", str(budget))
        _write_svg(after_path, final_tree)
        after_manifest = build_handoff_manifest(after_path)
        suffix = secrets.token_hex(6)
        base = "local_refine_" + suffix
        new_name = "result_" + base
        staged = work / new_name
        staged.mkdir()
        derived_svg = staged / (base + "_vector.svg")
        derived_svg.write_bytes(after_path.read_bytes())
        if reference in originals:
            (staged / "source_reference.png").write_bytes(originals[reference])
        if source.name == "source_original.png":
            (staged / "source_original.png").write_bytes(originals[source])
        new_report = _derived_report(report, manifest, after_manifest, evidence, budget, folder.name)
        new_report["output_base"] = base
        drawable_ids = {member for unit in after_manifest["objects"] for member in unit["member_ids"]}
        drawables = [item for item in final_tree.iter() if item.get("id") in drawable_ids]
        for key, tag in (("native_circles", "circle"), ("native_rectangles", "rect"),
                         ("native_ellipses", "ellipse"), ("native_lines", "line"),
                         ("native_polylines", "polyline"), ("native_polygons", "polygon")):
            new_report[key] = sum(_local(item.tag) == tag for item in drawables)
        new_report["native_primitives"] = sum(new_report[key] for key in (
            "native_circles", "native_rectangles", "native_ellipses", "native_lines",
            "native_polylines", "native_polygons"))
        new_report["editability_reasons"] = ["局部曲線精修後尚未重新執行整張可編輯性驗收"]
        (staged / "report.json").write_text(json.dumps(new_report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        new_decisions = dict(decisions)
        for unit in manifest["objects"]:
            if changed.intersection(unit["member_ids"]):
                new_decisions[unit["id"]] = "review"
        save_decisions(work, {"result": new_name, "svg_sha256": after_manifest["svg_sha256"],
                              "decisions": new_decisions, "revision": None})
        # After the initial revision snapshot, only this short recheck and
        # atomic directory publish hold the UI lock. Compute never does.
        with lock:
            for path, expected in originals.items():
                if not path.is_file() or path.read_bytes() != expected:
                    raise HandoffConflict("原稿、原图或報告已變更，未發布過期精修結果")
            current_state = state_path.read_bytes() if state_path.is_file() else None
            if current_state != old_state:
                raise HandoffConflict("接手決策已在精修期間更新；未發布過期版本")
            destination = root / new_name
            if destination.exists():
                raise ValueError("結果名稱衝突，請再試一次")
            staged.rename(destination)
    return {"url": "/handoff?result=" + urllib.parse.quote(new_name),
            "summary": {"changed_objects": sum(bool(changed.intersection(unit["member_ids"])) for unit in manifest["objects"]),
                        "changed_paths": len(changed), "anchors_removed": evidence["anchors_removed"],
                        "status": "manual_review", "source_image_quality": "not_revalidated"}}


if __name__ == "__main__":
    import contextlib
    import io
    try:
        request = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(io.StringIO()):
            response = _compute(request)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        response = {"error": str(error)}
        if error.__cause__ is not None:
            response['debug_reason'] = str(error.__cause__)
        print(json.dumps(response, ensure_ascii=False))
        sys.exit(1)
