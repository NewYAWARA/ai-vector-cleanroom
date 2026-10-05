"""Prepare all eligible handoff units with bounded, reversible transactions.

The input SVG is preserved as an immutable version. Curve simplification is
compared with its parent geometry; source primitive reconstruction is separately
compared with the original PNG. Individual failures are reported and skipped.
Native SVG paint, defs, unselected units and stacking order are preserved;
gradients require a real reference renderer, never flat substitutes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET

from designer_handoff import (_geometry, _local, _style, build_handoff_manifest,
                              normalized_svg_text, validate_decisions)
from handoff_service import (HandoffConflict, result_context, save_decisions,
                             validate_payload_revision)
from local_refine import GEOMETRY, _by_id, _derived_report, _isolate, _write_svg

TOTAL_SECONDS = 180
PATH_SECONDS = 12
MAX_ATTEMPTS = 256
MAX_PATH_ANCHORS = 4096
MAX_TOTAL_ANCHORS = 48000
MAX_SVG_BYTES = 10 * 1024 * 1024
TARGET_SIZE = 512
GLOBAL_SIZE = 1200


def _render_reference(svg_path, png_path, width, *, background=None):
    """Adapter intentionally isolated so tests and renderer deployment agree."""
    from svg_renderer import render_svg_reference
    result = render_svg_reference(Path(svg_path), Path(png_path), width=int(width), background=background)
    if result is False or not Path(png_path).is_file():
        raise ValueError("reference_renderer_unavailable")
    return result


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _history_members(report):
    history = report.get("local_refine_history", [])
    if not isinstance(history, list):
        raise ValueError("精修歷史無效；不能判斷是否累積漂移")
    previous = set()
    for row in history:
        if (not isinstance(row, dict) or not isinstance(row.get("member_ids"), list)
                or any(not isinstance(value, str) for value in row["member_ids"])):
            raise ValueError("精修歷史缺少有效物件身分")
        previous.update(row["member_ids"])
    return previous


def _path_support(root, identifier, previous):
    if identifier in previous:
        return "already_refined_original_baseline_required", False
    element = _by_id(root)[identifier]
    if _local(element.tag) != "path" or len(element):
        return "already_native_or_not_a_path", False
    data = element.get("d", "")
    if len(data) > 512 * 1024:
        return "path_text_budget_exceeded", False
    parts = re.split(r"[Mm]", data)[1:]
    if not parts or any(not re.search(r"[Zz]\s*$", part) for part in parts):
        return "open_or_unsupported_path", False
    anchors, box = _geometry(element)
    if anchors < 5:
        return "already_low_anchor", False
    if anchors > MAX_PATH_ANCHORS or box is None or not all(math.isfinite(value) for value in box):
        return "path_anchor_or_geometry_budget_exceeded", False
    parents = {child: parent for parent in root.iter() for child in parent}
    chain, current = [], element
    while current is not None:
        chain.append(current)
        current = parents.get(current)
    fill = "black"
    for ancestor in reversed(chain):
        attrs = {**ancestor.attrib, **_style(ancestor)}
        if any(key in attrs for key in ("transform", "mask", "clip-path", "filter")):
            return "unsupported_transform_clip_mask_or_filter", False
        if str(attrs.get("stroke", "none")).strip().lower() != "none":
            return "stroke_geometry_requires_different_engine", False
        if any(key not in {"fill", "fill-rule", "fill-opacity", "stroke", "stroke-width",
                           "opacity", "display", "visibility", "color"} for key in _style(ancestor)):
            return "unsupported_style_context", False
        if any("url(" in str(value).lower() and key not in {"fill", "style"}
               for key, value in attrs.items()):
            return "unsupported_paint_dependency", False
        fill = str(attrs.get("fill", fill)).strip()
    if fill.lower() in {"none", "transparent", "inherit", "currentcolor"}:
        return "unresolved_or_unfilled_paint", False
    match = re.fullmatch(r"url\(\s*['\"]?#([^\s)'\"]+)['\"]?\s*\)", fill, re.I)
    if match:
        resource = _by_id(root).get(match.group(1))
        if resource is None or _local(resource.tag) not in {"linearGradient", "radialGradient"}:
            return "unsupported_fill_resource", False
        # Isolation retains root-level defs with their inherited root paint.
        # A nested resource can inherit extra presentation context that would
        # be lost by hoisting it, so do not claim a local colour validation.
        seen = set()
        while resource is not None:
            if resource in seen:
                return "unsupported_cyclic_gradient_dependency", False
            seen.add(resource)
            resource_parent = parents.get(resource)
            if (resource_parent is None or _local(resource_parent.tag) != "defs"
                    or parents.get(resource_parent) is not root):
                return "unsupported_nested_gradient_context", False
            hrefs = [value for key, value in resource.attrib.items() if _local(key) == "href"]
            if len(hrefs) > 1 or len(seen) > 32:
                return "unsupported_gradient_dependency", False
            if not hrefs:
                break
            resource = _by_id(root).get(hrefs[0][1:]) if hrefs[0].startswith("#") else None
            if resource is None or _local(resource.tag) not in {"linearGradient", "radialGradient"}:
                return "unsupported_gradient_dependency", False
        return None, True
    if "url" in fill.lower():
        return "unsupported_fill_resource", False
    return None, False


def _target_document(root, identifier, box, silhouette=False):
    result = _isolate(root, identifier, silhouette=silhouette, box=box)
    x, y, width, height = box
    side = max(width, height) * 1.1
    if not side > 0:
        raise ValueError("degenerate_target_bounds")
    result.set("viewBox", f"{x + width / 2 - side / 2} {y + height / 2 - side / 2} {side} {side}")
    result.set("width", str(TARGET_SIZE))
    result.set("height", str(TARGET_SIZE))
    if not silhouette:
        # Retain every resource and its coordinate system unchanged. The real
        # SVG renderer recomputes objectBoundingBox gradients from each shape.
        for resource in reversed(list(root)):
            if _local(resource.tag) == "defs":
                result.insert(0, copy.deepcopy(resource))
    return result


def _rgb(path, size=None):
    from PIL import Image
    import numpy as np
    with Image.open(path) as source:
        image = source.convert("RGBA")
        if size is not None and image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        return np.asarray(background.convert("RGB"), dtype=np.int16)


def _target_gate(before, after, identifier, folder):
    import numpy as np
    from curve_refit import _mask_topology
    from annulus_detector import compare_rendered_pngs
    box = _geometry(_by_id(before)[identifier])[1]
    rendered = {}
    provenance = {}
    for kind, silhouette in (("paint", False), ("mask", True)):
        for name, root in (("before", before), ("after", after)):
            svg = folder / f"target-{kind}-{name}.svg"
            png = svg.with_suffix(".png")
            _write_svg(svg, _target_document(root, identifier, box, silhouette))
            provenance[f"{kind}_{name}"] = _render_reference(svg, png, TARGET_SIZE)
            rendered[kind, name] = png
    masks = [_rgb(rendered["mask", name])[:, :, 0] < 128 for name in ("before", "after")]
    if not masks[0].any() or not masks[1].any():
        raise ValueError("target_renderer_has_no_ink")
    topology = [_mask_topology(mask) for mask in masks]
    if topology[0] != topology[1]:
        raise ValueError("target_components_or_holes_changed")
    geometry = compare_rendered_pngs(rendered["mask", "before"], rendered["mask", "after"], tolerance_px=1)
    if geometry.get("accepted") is not True:
        raise ValueError("target_bidirectional_ink_guard_failed")
    a, b = [_rgb(rendered["paint", name]) for name in ("before", "after")]
    union = masks[0] | masks[1]
    core = masks[0] & masks[1]
    # Ignore one-pixel antialias edges only for the interior-paint test;
    # silhouette overlap and full-union colour error remain separately gated.
    for delta in (-1, 1):
        core &= np.roll(masks[0] & masks[1], delta, axis=0)
        core &= np.roll(masks[0] & masks[1], delta, axis=1)
    error = np.abs(a - b).max(axis=2)
    mean = float(error[union].mean())
    core_mean = float(error[core].mean()) if core.any() else None
    core_p99 = float(np.percentile(error[core], 99)) if core.any() else None
    if core_mean is None or mean > 3.0 or core_mean > 1.5 or core_p99 > 8.0:
        raise ValueError("target_actual_paint_guard_failed")
    return {"id": identifier, "target_silhouette_topology": "passed",
            "components_and_holes": list(topology[0]), "render_size": TARGET_SIZE,
            "renderer_provenance": provenance,
            "render": {"external_render_check": "completed", "accepted": True, **geometry},
            "actual_paint": {"accepted": True, "mean_max_channel_error": mean,
                             "interior_mean_error": core_mean, "interior_p99_error": core_p99,
                             "object_bounding_box_recomputed_by_svg_renderer": True}}


def _composed_alpha_gate(before_png, after_png):
    """Opt in without changing the legacy pipeline alpha-topology API."""
    from alpha_topology import compare_composed_alpha
    return compare_composed_alpha(before_png, after_png)


def _global_gate(original, candidate, source, folder, target_box):
    import numpy as np
    from designer_handoff import _view_box
    view = _view_box(ET.parse(original).getroot())
    width = max(1, round(GLOBAL_SIZE * min(1.0, view[2] / view[3])))
    before_png = folder / "parent-reference.png"
    provenance_file = folder / "parent-reference-provenance.json"
    after_png = folder / "candidate-reference.png"
    if not before_png.exists():
        provenance = _render_reference(original, before_png, width)
        provenance_file.write_text(json.dumps(provenance), encoding="utf-8")
    parent_provenance = json.loads(provenance_file.read_text(encoding="utf-8")) if provenance_file.is_file() else None
    candidate_provenance = _render_reference(candidate, after_png, width)
    alpha = _composed_alpha_gate(before_png, after_png)
    a, b = _rgb(before_png), _rgb(after_png)
    if a.shape != b.shape:
        raise ValueError("whole_render_dimensions_changed")
    ink = ((255 - a).max(axis=2) > 8) | ((255 - b).max(axis=2) > 8)
    error = np.abs(a - b).max(axis=2)
    mean = float(error[ink].mean()) if ink.any() else 0.0
    changed_fraction = float((error[ink] > 16).mean()) if ink.any() else 0.0
    if mean > 1.0 or changed_fraction > 0.02:
        raise ValueError("whole_actual_paint_guard_failed")
    raw = _rgb(source, (a.shape[1], a.shape[0]))
    before_source = float(np.abs(a - raw).mean())
    after_source = float(np.abs(b - raw).mean())
    if after_source > before_source + 0.15:
        raise ValueError("original_reference_regression")
    # A small object cannot hide a source regression in the blank canvas.
    # Compare its visible full-composition crop, not the isolated silhouette,
    # because other layers may cover parts of the object in the source.
    x, y, w, h = target_box
    sx, sy = a.shape[1] / view[2], a.shape[0] / view[3]
    left = max(0, int(math.floor((x - view[0]) * sx)) - 2)
    top = max(0, int(math.floor((y - view[1]) * sy)) - 2)
    right = min(a.shape[1], int(math.ceil((x + w - view[0]) * sx)) + 2)
    bottom = min(a.shape[0], int(math.ceil((y + h - view[1]) * sy)) + 2)
    if right <= left or bottom <= top:
        raise ValueError("target_outside_visible_canvas")
    region = (slice(top, bottom), slice(left, right))
    local_before = float(np.abs(a[region] - raw[region]).mean())
    local_after = float(np.abs(b[region] - raw[region]).mean())
    if local_after > local_before + 0.25:
        raise ValueError("local_original_reference_regression")
    return {"external_render_check": "completed", "accepted": True,
            "reference_renderer": "native_svg", "paint_resources_rendered": True,
            "renderer_provenance": {"parent": parent_provenance, "candidate": candidate_provenance},
            "composed_alpha": alpha,
            "parent_ink_mean_max_channel_error": mean, "parent_changed_ink_fraction": changed_fraction,
            "raw_source_comparison": {"baseline_mean_absolute_rgb_error": before_source,
                                      "candidate_mean_absolute_rgb_error": after_source,
                                      "maximum_regression": 0.15, "accepted": True,
                                      "target_crop_baseline_error": local_before,
                                      "target_crop_candidate_error": local_after,
                                      "target_crop_maximum_regression": 0.25,
                                      "source_png_sha256": _sha(Path(source).read_bytes()),
                                      "alignment": "entire_source_resized_to_svg_canvas"},
            "scope": "limited_change_relative_to_parent_not_final_source_acceptance"}


def _worker_compute(request):
    from curve_refit_stage import propose_svg_curve_refit
    folder = Path(request["work"])
    current = ET.parse(request["current_svg"]).getroot()
    identifier = request["member_id"]
    input_svg = folder / "geometry-input.svg"
    proposal_svg = folder / "geometry-proposal.svg"
    box = _geometry(_by_id(current)[identifier])[1]
    source_rejection = None
    if request.get("allow_source_primitive"):
        try:
            from source_primitive import propose_source_ellipse
            source_candidate, source_gate = propose_source_ellipse(current,
                Path(request["source"]), identifier, request["error_budget_percent"], folder,
                lambda svg, png, width: _render_reference(svg, png, width, background="#ffffff"))
            before_count = _geometry(_by_id(current)[identifier])[0]
            after_count = _geometry(_by_id(source_candidate)[identifier])[0]
            if not 0 < after_count < before_count:
                raise ValueError("source_primitive_no_anchor_reduction")
            _write_svg(folder / "candidate.svg", source_candidate)
            return {"status": "improved", "proposal_kind": "source_primitive",
                    "member_id": identifier, "anchors_before": before_count, "anchors_after": after_count,
                    "paths_before": 1, "paths_after": 0, "source_primitive_gate": source_gate,
                    "gradient_fill_preserved": False}
        except ValueError as error:
            source_rejection = str(error)
    geometry = _target_document(current, identifier, box, silhouette=True)
    for element in geometry.iter():
        for key in list(element.attrib):
            if "gradient" in key.lower():
                element.attrib.pop(key)
    _write_svg(input_svg, geometry)
    proposal = propose_svg_curve_refit(input_svg, proposal_svg,
        error_budget_percent=request["error_budget_percent"], minimum_nodes=5,
        maximum_segments=MAX_PATH_ANCHORS, sample_step=2.0)
    proposed = _by_id(ET.parse(proposal_svg).getroot())[identifier]
    before_count, _ = _geometry(_by_id(current)[identifier])
    after_count, _ = _geometry(proposed)
    if proposal.get("status") != "proposed" or not 0 < after_count < before_count:
        return {"status": "skipped", "reason": "no_safe_reduction", "source_primitive_rejection": source_rejection}
    if _local(proposed.tag) not in {"path", "circle", "ellipse"}:
        raise ValueError("unsupported_proposal_shape")
    after = copy.deepcopy(current)
    element = _by_id(after)[identifier]
    element.tag = proposed.tag
    for key in GEOMETRY:
        element.attrib.pop(key, None)
        if key in proposed.attrib:
            element.set(key, proposed.get(key))
    candidate = folder / "candidate.svg"
    _write_svg(candidate, after)
    target = _target_gate(current, after, identifier, folder)
    whole = _global_gate(Path(request["original_svg"]), candidate,
                         Path(request["source"]), Path(request["shared_work"]), box)
    return {"status": "improved", "proposal_kind": "curve_simplification", "member_id": identifier,
            "anchors_before": before_count, "anchors_after": after_count,
            "paths_before": 1, "paths_after": int(_local(element.tag) == "path"),
            "target_gate": target, "whole_svg_render": whole, "proposal": proposal,
            "source_primitive_rejection": source_rejection,
            "gradient_fill_preserved": bool(request["gradient"])}


def _invoke_worker(request, timeout):
    try:
        completed = subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(request), text=True, encoding="utf-8", capture_output=True,
            timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return {"status": "skipped", "reason": "per_path_time_budget_exhausted"}
    try:
        result = json.loads(completed.stdout)
    except ValueError:
        return {"status": "skipped", "reason": "worker_failed_without_valid_result"}
    if completed.returncode or "error" in result:
        return {"status": "skipped", "reason": str(result.get("error", "worker_failed"))[:240]}
    return result


def _verify_boundary(before, after, allowed):
    a, b = copy.deepcopy(before), copy.deepcopy(after)
    for tree in (a, b):
        elements = _by_id(tree)
        for identifier in allowed:
            if identifier not in elements:
                raise ValueError("selected_identity_missing")
            element = elements[identifier]
            element.tag = "refined-geometry"
            for key in GEOMETRY:
                element.attrib.pop(key, None)
    if ET.tostring(a) != ET.tostring(b):
        raise ValueError("unselected_geometry_paint_resources_or_order_changed")


def _invalidate_metadata(tree, changed, parent_digest, budget):
    for element in tree.iter():
        if _local(element.tag) == "metadata" and element.get("id") == "ai-vector-cleanroom-metadata":
            element.clear()
            element.set("id", "ai-vector-cleanroom-metadata")
            element.text = json.dumps({"acceptance_status": "manual_review",
                "parent_svg_sha256": parent_digest, "source_image_quality": "not_finally_accepted"})
        if element.get("id") in changed:
            for key in list(element.attrib):
                if key.startswith("data-avc-curve-") or key.startswith("data-avc-anchors-") or key in {
                        "data-avc-designer-anchors", "data-avc-error-budget-percent",
                        "data-avc-p95-error-percent", "data-avc-max-error-percent",
                        "data-avc-source-edge"}:
                    element.attrib.pop(key)
            element.set("data-local-refine-status", "manual-review")
            element.set("data-local-refine-error-budget-percent", str(budget))


def _build_recolor_artifacts(folder, svg, report):
    """Build convenience files from the exact derivative without annotating it."""
    from paint_roles import build_paint_role_manifest
    from recolor_page import make_recolor_html
    manifest_path = folder / (svg.stem.removesuffix("_vector") + "_paint_roles.json")
    recolor_path = folder / "色彩調整.html"
    original = svg.read_bytes()
    try:
        paint = build_paint_role_manifest(svg)
        if paint.get("unsupported_paints"):
            raise ValueError("derived_svg_contains_paints_not_supported_by_offline_recolor")
        if not paint.get("roles"):
            raise ValueError("derived_svg_has_no_editable_color_roles")
        paint["source"]["sha256_scope"] = "exact_delivered_derived_svg"
        make_recolor_html(recolor_path, svg, paint,
                          download_filename=svg.stem + "_換色.svg",
                          tool_version=report.get("tool_version", ""))
        if svg.read_bytes() != original:
            raise ValueError("recolor_generation_changed_vector_bytes")
        manifest_path.write_text(json.dumps(paint, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        report["paint_role_manifest"] = manifest_path.name
        report["recolor_page"] = recolor_path.name
        report["recolor_artifacts"] = {"status": "available_not_designer_validated",
            "svg_sha256": _sha(original), "geometry_and_ids_unchanged": True,
            "role_controls": len(paint["roles"])}
    except Exception as error:
        # This optional editor cannot turn a valid vector transaction into a
        # failure, and unsupported paints must not produce a partial editor
        # labelled as complete global recolouring.
        recolor_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        if svg.read_bytes() != original:
            svg.write_bytes(original)
        report["recolor_artifacts"] = {"status": "unavailable",
            "reason": str(error)[:240], "svg_sha256": _sha(original)}


def prepare_result(output_root, payload, *, lock):
    started = time.monotonic()
    root = Path(output_root).resolve()
    if not isinstance(payload, dict):
        raise ValueError("自動準備參數必須是物件")
    budget = payload.get("error_budget_percent", 0.25)
    if isinstance(budget, bool) or budget not in (0.1, 0.25, 0.5):
        raise ValueError("誤差僅支援 0.1%、0.25% 或 0.5%")
    folder, svg, source, report = result_context(root, payload.get("result"))
    paths = {svg, source, folder / "report.json"}
    reference = folder / "source_reference.png"
    if reference.exists():
        if reference.resolve().parent != folder or not reference.is_file():
            raise ValueError("參考圖不可指向資料夾外")
        paths.add(reference)
    originals = {path: path.read_bytes() for path in paths}
    report = json.loads(originals[folder / "report.json"])
    if len(originals[svg]) > MAX_SVG_BYTES:
        raise ValueError("SVG 超過自動準備的 10 MiB 上限")
    state_path = folder / "handoff_state.json"
    with lock:
        validate_payload_revision(folder, payload)
        old_state = state_path.read_bytes() if state_path.is_file() else None
    previous = _history_members(report)
    with tempfile.TemporaryDirectory(prefix=".auto-prepare-", dir=root) as temporary:
        work = Path(temporary)
        original = work / "original.svg"
        original.write_bytes(originals[svg])
        manifest = build_handoff_manifest(original, report)
        if payload.get("svg_sha256") != manifest["svg_sha256"]:
            raise HandoffConflict("SVG 已變更，請重新開啟接手頁")
        decisions = validate_decisions(manifest, payload.get("decisions"), payload.get("svg_sha256"))
        _, _, width, height = manifest["view_box"]
        if max(width / height, height / width) > 8:
            raise ValueError("畫布長寬比超過 8:1，需先拆分版面")
        normalized = normalized_svg_text(original)
        original.write_text(normalized, encoding="utf-8")
        current_path = work / "current.svg"
        current_path.write_text(normalized, encoding="utf-8")
        source_snapshot = work / "source.png"
        source_snapshot.write_bytes(originals[source])
        original_tree = ET.fromstring(normalized)
        current_tree = copy.deepcopy(original_tree)
        units, attempts, changed, target_gates, proposals = [], [], [], [], []
        source_gates = []
        whole = None
        for unit in manifest["objects"]:
            record = {"id": unit["id"], "status": "skipped", "reason": "no_eligible_paths",
                      "anchors_before": unit["anchor_count"], "anchors_after": unit["anchor_count"],
                      "paths_before": unit["path_count"], "paths_after": unit["path_count"], "members": []}
            units.append(record)
            for member in unit["member_ids"]:
                reason, gradient = ("keep_locked", False) if decisions[unit["id"]] == "keep" else _path_support(original_tree, member, previous)
                anchors = _geometry(_by_id(original_tree)[member])[0]
                row = {"id": member, "status": "skipped", "reason": reason,
                       "anchors_before": anchors, "anchors_after": anchors, "gradient": gradient}
                record["members"].append(row)
                if reason is None:
                    attempts.append((anchors, record, row))
            if decisions[unit["id"]] == "keep":
                record["reason"] = "keep_locked"
        attempts.sort(key=lambda entry: (-entry[0], entry[2]["id"]))
        processed, input_anchors = 0, 0
        for anchors, unit, row in attempts:
            remaining = TOTAL_SECONDS - (time.monotonic() - started)
            if remaining < 2:
                row["reason"] = "total_time_budget_exhausted"
                continue
            if processed >= MAX_ATTEMPTS or input_anchors + anchors > MAX_TOTAL_ANCHORS:
                row["reason"] = "candidate_or_total_anchor_budget_exhausted"
                continue
            processed += 1
            input_anchors += anchors
            transaction = work / f"path-{processed:03d}"
            transaction.mkdir()
            result = _invoke_worker({"work": str(transaction), "shared_work": str(work),
                "current_svg": str(current_path), "original_svg": str(original),
                "source": str(source_snapshot), "member_id": row["id"], "gradient": row["gradient"],
                "allow_source_primitive": (source.name == "source_original.png" and not row["gradient"]
                    and len(manifest["objects"]) == 1 and len(manifest["objects"][0]["member_ids"]) == 1),
                "error_budget_percent": budget}, min(PATH_SECONDS, remaining))
            if result.get("source_primitive_rejection"):
                row["source_primitive_rejection"] = result["source_primitive_rejection"]
            if result.get("status") != "improved":
                row["reason"] = result.get("reason", "no_safe_reduction")
                continue
            try:
                candidate_path = transaction / "candidate.svg"
                candidate = ET.parse(candidate_path).getroot()
                _verify_boundary(current_tree, candidate, [row["id"]])
                before_count = _geometry(_by_id(current_tree)[row["id"]])[0]
                after_count = _geometry(_by_id(candidate)[row["id"]])[0]
                target = result.get("target_gate") or {}
                gate = result.get("whole_svg_render") or {}
                source_gate = result.get("source_primitive_gate") or {}
                source_reconstructed = result.get("proposal_kind") == "source_primitive"
                if source_reconstructed:
                    if (not 0 < after_count < before_count or len(manifest["objects"]) != 1
                            or source.name != "source_original.png" or row["gradient"]
                            or source_gate.get("accepted") is not True
                            or source_gate.get("external_render_check") != "completed"
                            or source_gate.get("proposal_kind") != "source_primitive"
                            or source_gate.get("source_components_and_holes") != [1, 0]
                            or source_gate.get("render_components_and_holes") != [1, 0]
                            or source_gate.get("source_contour", {}).get("salient_corner_count") != 0
                            or source_gate.get("original_png_sha256") != _sha(originals[source])):
                        raise ValueError("source_primitive_evidence_incomplete")
                elif (not 0 < after_count < before_count or target.get("id") != row["id"]
                        or target.get("target_silhouette_topology") != "passed"
                        or target.get("actual_paint", {}).get("accepted") is not True
                        or target.get("render", {}).get("accepted") is not True
                        or gate.get("external_render_check") != "completed" or gate.get("accepted") is not True
                        or gate.get("composed_alpha", {}).get("external_render_check") != "completed"
                        or gate.get("composed_alpha", {}).get("accepted") is not True):
                    raise ValueError("candidate_evidence_incomplete")
            except (ValueError, OSError, ET.ParseError) as error:
                row["reason"] = str(error)
                continue
            current_tree = candidate
            current_path.write_bytes(candidate_path.read_bytes())
            changed.append(row["id"])
            if not source_reconstructed:
                target_gates.append(target)
            if source_reconstructed:
                source_gates.append(source_gate)
            proposals.append(result)
            whole = gate
            row.update({"status": "improved", "reason": "source_primitive_reconstructed" if source_reconstructed else "geometry_reduced_paint_and_topology_preserved",
                        "proposal_kind": "source_primitive" if source_reconstructed else "curve_simplification",
                        "anchors_after": after_count})
            unit["status"] = "improved"
            unit["proposal_kind"] = row["proposal_kind"]
            unit["reason"] = "improved_eligible_parts_other_parts_preserved"
            unit["anchors_after"] -= before_count - after_count
            unit["paths_after"] += int(_local(_by_id(candidate)[row["id"]].tag) == "path") - 1
        for unit in units:
            if unit["status"] == "skipped" and unit["reason"] != "keep_locked":
                reasons = list(dict.fromkeys(row["reason"] or "not_attempted" for row in unit["members"]))
                unit["reason"] = "; ".join(reasons)
        summary = {"status": "manual_review" if changed else "unchanged", "units_total": len(units),
                   "units_improved": sum(unit["status"] == "improved" for unit in units),
                   "units_skipped": sum(unit["status"] == "skipped" for unit in units),
                   "paths_improved": len(changed), "attempted_paths": processed,
                   "source_reconstructed_paths": len(source_gates),
                   "curve_simplified_paths": len(changed) - len(source_gates),
                   "anchors_before": sum(unit["anchors_before"] for unit in units),
                   "anchors_after": sum(unit["anchors_after"] for unit in units),
                   "paths_before": sum(unit["paths_before"] for unit in units),
                   "paths_after": sum(unit["paths_after"] for unit in units),
                   "machine_elapsed_seconds": round(time.monotonic() - started, 3),
                   "human_time_saving_validated": False}
        summary["anchors_removed"] = summary["anchors_before"] - summary["anchors_after"]
        if not changed:
            return {"url": None, "summary": summary, "units": units}
        _verify_boundary(original_tree, current_tree, changed)
        _invalidate_metadata(current_tree, changed, manifest["svg_sha256"], budget)
        _write_svg(current_path, current_tree)
        after_manifest = build_handoff_manifest(current_path)
        evidence = {"changed_member_ids": changed, "anchors_removed": summary["anchors_removed"],
                    "whole_svg_render": whole, "target_gates": target_gates, "proposals": proposals,
                    "source_primitive_gates": source_gates,
                    "whole_original_source_render": source_gates[-1] if source_gates else None,
                    "units": units, "budgets": {"total_seconds": TOTAL_SECONDS, "path_seconds": PATH_SECONDS,
                    "maximum_attempts": MAX_ATTEMPTS, "path_anchor_maximum": MAX_PATH_ANCHORS},
                    "fixed_original_parent_baseline": not bool(source_gates),
                    "source_primitive_is_distinct_from_parent_equivalence": bool(source_gates),
                    "paint_and_defs_preserved": True, "old_gradient_geometry_certificates": "invalidated"}
        new_report = _derived_report(report, manifest, after_manifest, evidence, budget, folder.name)
        if source_gates:
            new_report["local_refine_validation"]["scope"] = "source_primitive_reconstruction_original_image_fit"
            new_report["local_refine_validation"]["source_image_quality"] = "narrow_solid_ellipse_fit_validated_not_final_acceptance"
            new_report["warnings"] = list(report.get("warnings") or []) + [
                "來源幾何重建版本：單色橢圓更貼近原始 PNG，但不代表還原設計意圖；仍請人工校稿。"]
        from svg_postprocess import measure_svg_structure
        structure = measure_svg_structure(current_path)
        # Keep the project's legacy node count separate from editable anchors;
        # a native ellipse counts as one legacy node and four editable anchors.
        new_report.update({key: structure[key] for key in (
            "paths", "nodes_total", "designer_anchors_total", "groups", "gradients")})
        new_report["final_structure"] = structure
        new_report["paint_role_manifest"] = None
        new_report["recolor_page"] = None
        new_report["render_ink_pixels"] = None
        new_report["preview_is_svg_render"] = None
        new_report["auto_prepare"] = {"summary": summary, "units": units, "evidence": evidence}
        new_report["_handoff_reference_kind"] = "original" if source.name == "source_original.png" else "processed_reference"
        base = "auto_prepare_" + secrets.token_hex(6)
        new_name = "result_" + base
        staged = work / new_name
        staged.mkdir()
        (staged / (base + "_vector.svg")).write_bytes(current_path.read_bytes())
        for path in originals:
            if path.name in {"source_original.png", "source_reference.png"}:
                (staged / path.name).write_bytes(originals[path])
        new_report["output_base"] = base
        native_keys = {"native_circles": "circle", "native_rectangles": "rect", "native_ellipses": "ellipse",
                       "native_lines": "line", "native_polylines": "polyline", "native_polygons": "polygon"}
        for key, tag in native_keys.items():
            new_report[key] = structure[key]
        new_report["native_primitives"] = sum(new_report[key] for key in native_keys)
        _build_recolor_artifacts(staged, staged / (base + "_vector.svg"), new_report)
        (staged / "report.json").write_text(json.dumps(new_report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        new_decisions = {unit["id"]: ("review" if unit["status"] == "improved" else decisions[unit["id"]]) for unit in units}
        save_decisions(work, {"result": new_name, "svg_sha256": after_manifest["svg_sha256"],
                             "decisions": new_decisions, "revision": None})
        with lock:
            validate_payload_revision(folder, payload)
            for path, expected in originals.items():
                if not path.is_file() or path.read_bytes() != expected:
                    raise HandoffConflict("原稿或參考圖已更新；未發布過期自動準備結果")
            if (state_path.read_bytes() if state_path.is_file() else None) != old_state:
                raise HandoffConflict("接手判斷已更新；未發布過期結果")
            destination = root / new_name
            if destination.exists():
                raise ValueError("結果名稱衝突")
            staged.rename(destination)
    return {"url": "/handoff?result=" + urllib.parse.quote(new_name), "summary": summary, "units": units}


if __name__ == "__main__":
    import contextlib
    import io
    try:
        request = json.loads(sys.stdin.read())
        with contextlib.redirect_stdout(io.StringIO()):
            result = _worker_compute(request)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        sys.exit(1)
