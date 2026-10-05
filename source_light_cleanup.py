"""Source-backed proposals for thin pale trace artifacts, never an export gate.

Processed transparency is only corroboration. Opaque white paper cannot reveal
whether an enclosed white area was a white object or negative space: such areas
may receive a source-verified paper paint, but their geometry/alpha is retained.
The caller must run its whole-scene source transaction before committing each
proposal. No image, SVG or caller-owned array is changed by this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

SCHEMA = "ai-vector-cleanroom.source-light-cleanup/v1"
DRAWABLES = {"path", "rect", "circle", "ellipse", "line", "polygon", "polyline", "image", "use", "text"}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _local(node):
    return node.tag.rsplit("}", 1)[-1]


def _element_bytes(node):
    clone = copy.deepcopy(node)
    clone.tail = None
    return ET.tostring(clone, encoding="utf8")


def _context(root, node):
    parents = {child: parent for parent in root.iter() for child in parent}
    ancestors, fill, current = [], None, node
    while current is not None:
        if any(k in current.attrib for k in ("transform", "style", "mask", "clip-path", "filter", "mix-blend-mode")):
            raise ValueError("unsupported_presentation_context")
        if current.get("display") not in (None, "inline") or current.get("visibility") not in (None, "visible"):
            raise ValueError("hidden_presentation_context")
        if any(current.get(k) not in (None, "1", "1.0", "100%") for k in ("opacity", "fill-opacity")):
            raise ValueError("translucent_presentation_context")
        if current.get("stroke", "none") != "none":
            raise ValueError("stroke_presentation_context")
        if fill is None and current.get("fill"):
            fill = current.get("fill")
        if current is not node:
            ancestors.append([current.tag, sorted(current.attrib.items())])
        current = parents.get(current)
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill or ""):
        raise ValueError("solid_rgb_paint_required")
    return fill.lower(), _sha(json.dumps(ancestors, ensure_ascii=True).encode())


def _identify(root, proposal):
    identifier = proposal.get("drawable_id")
    matches = ([n for n in root.iter() if n.get("id") == identifier] if identifier else
               [n for n in root.iter() if _local(n) == "path" and _sha(_element_bytes(n)) == proposal["element_sha256"]])
    if len(matches) != 1:
        raise ValueError("light_candidate_identity_changed_or_ambiguous")
    node = matches[0]
    _, context = _context(root, node)
    if _sha(_element_bytes(node)) != proposal["element_sha256"] or context != proposal["parent_context_sha256"]:
        raise ValueError("light_candidate_geometry_or_context_changed")
    return node


def apply_light_fill_candidate(svg_text, proposal):
    """Apply one authenticated unit to current SVG; caller retains rollback SVG.

    Other independent removals do not stale a candidate: identity binds this
    exact element and its ancestor attributes, not sibling numeric indices.
    """
    if proposal.get("schema") != SCHEMA or proposal.get("operation") not in {"remove_drawable", "recolor_paper"}:
        raise ValueError("invalid_light_cleanup_proposal")
    root = ET.fromstring(svg_text)
    node = _identify(root, proposal)
    if proposal["operation"] == "remove_drawable":
        next(parent for parent in root.iter() if node in list(parent)).remove(node)
    else:
        paint = proposal.get("replacement_fill", "")
        if not re.fullmatch(r"#[0-9a-f]{6}", paint):
            raise ValueError("invalid_replacement_paper_paint")
        node.set("fill", paint)
    return ET.tostring(root, encoding="unicode")


def _rgba_composite(rgba):
    rgba = np.asarray(rgba, dtype=float)
    alpha = rgba[:, :, 3:4] / 255
    return rgba[:, :, :3] * alpha + 255 * (1 - alpha)


def _paper_evidence(original):
    from stroke_engine import connected_components
    border = np.concatenate((original[0], original[-1], original[:, 0], original[:, -1]))
    paper = np.rint(np.median(border[:, :3].astype(float), axis=0)).astype(np.uint8)
    border_ok = (border[:, 3] == 255) & (np.abs(border[:, :3].astype(float) - paper).max(1) <= 4)
    if paper.min() < 245 or int(paper.max()) - int(paper.min()) > 3 or border_ok.mean() < .9:
        raise ValueError("no_reliable_opaque_neutral_paper")
    rgb = original[:, :, :3].astype(np.int16)
    distance = np.abs(rgb - paper).max(2)
    chroma = rgb.max(2) - rgb.min(2)
    # Two tiers distinguish clean paper samples from antialiased paper edges.
    strong = (original[:, :, 3] == 255) & (distance <= 9) & (chroma <= 6)
    compatible = (original[:, :, 3] == 255) & (distance <= 20) & (chroma <= 12)
    labels, _ = connected_components(compatible)
    external_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    exterior = compatible & np.isin(labels, external_labels)
    return paper, strong, compatible, compatible & ~exterior


def propose_light_fill_cleanup(svg_text, source_original_path, processed_rgba, *, maximum_candidates=32, maximum_seconds=20):
    """Propose reversible path removals or paper recolors, without publication.

    Original source resolution is authoritative. Processed RGBA is aligned by
    nearest-neighbour alpha only; no processed RGB is treated as original paper.
    Caps bound scene size and candidates. One native render may finish after the
    deadline; no subsequent unit starts. A successful proposal still needs the
    caller's complete source/scene guard and remains manual review.
    """
    from clean_base import _parse_subpaths
    from svg_renderer import render_svg_reference, _validated_svg
    started = time.monotonic()
    if isinstance(maximum_candidates, bool) or not isinstance(maximum_candidates, int) or not 0 <= maximum_candidates <= 32:
        raise ValueError("maximum_candidates must be an integer in 0..32")
    if isinstance(maximum_seconds, bool) or not math.isfinite(float(maximum_seconds)) or not 0 <= float(maximum_seconds) <= 20:
        raise ValueError("maximum_seconds must be finite in 0..20")
    source_original_path = Path(source_original_path)
    payload = svg_text.encode("utf8")
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("light_cleanup_svg_size_budget")
    _validated_svg(payload)
    root = ET.fromstring(svg_text)
    nodes = list(root.iter())
    if len(nodes) > 5000:
        raise ValueError("light_cleanup_dom_size_budget")
    report = {"schema": SCHEMA, "status": "no_change", "proposals": [], "retained": [],
              "input_svg_sha256": _sha(payload), "original_png_sha256": _sha(source_original_path.read_bytes()),
              "downstream_source_scene_guard_required": True, "human_acceptance": "not_performed",
              "paper_semantics": "opaque_source_does_not_resolve_white_object_vs_negative_space",
              "maximum_candidates": maximum_candidates, "search_budget_seconds": maximum_seconds}
    def finish(reason=None):
        report["status"] = "proposals_available" if report["proposals"] else "no_change"
        if reason:
            report["reason"] = reason
        report["machine_elapsed_seconds"] = time.monotonic() - started
        return report
    ids = [n.get("id") for n in nodes if n.get("id")]
    if len(ids) != len(set(ids)):
        return finish("duplicate_svg_identity")
    if any(_local(n) in {"style", "animate", "animateTransform", "set", "switch"} for n in nodes):
        return finish("unsupported_global_presentation")
    with Image.open(source_original_path) as image:
        if max(image.size) > 2048 or image.width * image.height > 4 * 1024 * 1024:
            return finish("original_resolution_budget")
        original = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    processed = np.asarray(processed_rgba)
    if processed.dtype != np.uint8 or processed.ndim != 3 or processed.shape[2] != 4 or min(processed.shape[:2]) < 1:
        raise ValueError("processed_rgba8_required")
    report["processed_rgba_sha256"] = _sha(str(processed.shape).encode() + processed.tobytes())
    height, width = original.shape[:2]
    if abs((processed.shape[1] / processed.shape[0]) / (width / height) - 1) > 1e-6:
        raise ValueError("processed_original_aspect_mismatch")
    clear = np.asarray(Image.fromarray(processed[:, :, 3]).resize((width, height), Image.Resampling.NEAREST)) <= 8
    report["source_size"] = [width, height]
    report["processed_alignment"] = "native" if processed.shape[:2] == original.shape[:2] else "same_canvas_nearest_alpha"
    try:
        paper, strong_paper, compatible_paper, enclosed_paper = _paper_evidence(original)
    except ValueError as error:
        return finish(str(error))
    paper_paint = "#{:02x}{:02x}{:02x}".format(*paper)
    report["estimated_paper_rgb"] = paper.tolist()
    deadline = started + float(maximum_seconds)
    candidates = []
    for ordinal, node in enumerate(nodes):
        if _local(node) != "path":
            continue
        identity = node.get("id") or "anonymous-path-" + str(ordinal)
        try:
            paint, context = _context(root, node)
            rgb = np.array([int(paint[i:i + 2], 16) for i in (1, 3, 5)])
            if rgb.min() < 200:
                continue
            if np.max(np.abs(rgb.astype(float) - paper)) <= 2:
                raise ValueError("white_or_paper_paint_preserved")
            if any(k.startswith("data-avc-gradient") for k in node.attrib):
                raise ValueError("owned_gradient_preserved")
            if len(node.get("d", "")) > 50000:
                raise ValueError("path_geometry_budget")
            paths = _parse_subpaths(node.get("d", ""))
            if len(paths) != 1 or not paths[0].get("closed"):
                raise ValueError("compound_or_open_path_preserved")
            identifier = node.get("id")
            if identifier and any(re.search(r"(?:#|url\(['\"]?#)" + re.escape(identifier) + r"(?:[)'\"]|$)", str(value))
                                  for other in nodes if other is not node for value in other.attrib.values()):
                raise ValueError("referenced_drawable_preserved")
            record = {"schema": SCHEMA, "drawable_id": identifier, "label": identity,
                      "element_sha256": _sha(_element_bytes(node)), "parent_context_sha256": context,
                      "original_element_xml": _element_bytes(node).decode("utf8"), "original_fill": paint,
                      "source_original_png_sha256": report["original_png_sha256"],
                      "processed_rgba_sha256": report["processed_rgba_sha256"]}
            _identify(root, record)
            candidates.append((node, rgb, record))
        except ValueError as error:
            report["retained"].append({"drawable_id": node.get("id"), "label": identity, "reason": str(error)})
    report["candidate_shortlist_count"] = len(candidates)
    examined = 0
    with tempfile.TemporaryDirectory(prefix="avc-source-light-") as temp:
        folder = Path(temp)
        def render(text, label):
            path = folder / (label + ".svg")
            path.write_text(text, encoding="utf8")
            provenance = render_svg_reference(path, path.with_suffix(".png"), width=width, background=None)
            with Image.open(path.with_suffix(".png")) as image:
                array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if array.shape != original.shape:
                raise ValueError("original_svg_canvas_alignment_failed")
            return array, provenance
        baseline = baseline_provenance = None
        for node, paint_rgb, record in candidates:
            if examined >= maximum_candidates or time.monotonic() >= deadline:
                report["retained"].append({"drawable_id": record["drawable_id"], "label": record["label"], "reason": "candidate_or_time_budget_exhausted"})
                continue
            examined += 1
            diagnostics = {}
            try:
                isolated = copy.deepcopy(root)
                target = _identify(isolated, record)
                for parent in isolated.iter():
                    for child in list(parent):
                        if _local(child) in DRAWABLES and child is not target:
                            parent.remove(child)
                raster, _ = render(ET.tostring(isolated, encoding="unicode"), "target")
                support = raster[:, :, 3] >= 128
                count = int(support.sum())
                if count < 4 or count > width * height * .025:
                    raise ValueError("not_a_small_light_band")
                inner = np.asarray(Image.fromarray(support.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 0
                thickness = 2 * count / max(1, int((support & ~inner).sum()))
                if thickness > max(3., 7 * max(width, height) / 1024):
                    raise ValueError("not_a_thin_light_band")
                source_rgb = original[:, :, :3].astype(np.int16)
                paint_supported = (np.max(np.abs(source_rgb - paint_rgb), axis=2) <= 12) & ~compatible_paper
                diagnostics = {"core_pixels": count, "approximate_thickness_pixels": thickness,
                    "original_strong_paper_share": float(strong_paper[support].mean()),
                    "original_paper_compatible_share": float(compatible_paper[support].mean()),
                    "processed_transparent_share": float(clear[support].mean()),
                    "joint_paper_and_processed_clear_share": float((strong_paper & clear)[support].mean()),
                    "source_supported_nonpaper_paint_share": float(paint_supported[support].mean()),
                    "enclosed_paper_core_pixels": int((enclosed_paper & support).sum()),
                    "original_other_ink_core_pixels": int((~compatible_paper & support).sum())}
                if not np.all(original[:, :, 3][support] == 255):
                    raise ValueError("original_alpha_ambiguity_preserved")
                if diagnostics["source_supported_nonpaper_paint_share"] >= .1:
                    raise ValueError("original_supports_real_pastel_content")
                if (diagnostics["original_paper_compatible_share"] < .5 or diagnostics["original_strong_paper_share"] < .45
                        or diagnostics["processed_transparent_share"] < .5 or diagnostics["joint_paper_and_processed_clear_share"] < .4):
                    raise ValueError("original_paper_and_processed_alpha_evidence_insufficient")
                operation = ("remove_drawable" if diagnostics["enclosed_paper_core_pixels"] == 0
                             and diagnostics["original_paper_compatible_share"] >= .9
                             and diagnostics["joint_paper_and_processed_clear_share"] >= .8 else "recolor_paper")
                proposal = {**record, "operation": operation, "replacement_fill": paper_paint if operation == "recolor_paper" else None,
                    "status": "manual_review", "reason": ("white_object_vs_negative_space_unresolved" if operation == "recolor_paper"
                                                           else "source_paper_and_processed_alpha_support_thin_artifact"),
                    "source_classification": diagnostics, "downstream_source_scene_guard_required": True}
                if baseline is None:
                    baseline, baseline_provenance = render(svg_text, "before")
                after_text = apply_light_fill_candidate(svg_text, proposal)
                after, after_provenance = render(after_text, "after")
                if operation == "recolor_paper" and not np.array_equal(baseline[:, :, 3], after[:, :, 3]):
                    raise ValueError("paper_recolor_alpha_changed")
                changed = np.any(baseline != after, axis=2)
                if not changed.any():
                    raise ValueError("no_visible_change")
                if np.any(changed & (raster[:, :, 3] == 0)):
                    raise ValueError("change_outside_target_support")
                yy, xx = np.nonzero(changed)
                box = [max(0, int(xx.min()) - 2), max(0, int(yy.min()) - 2), min(width, int(xx.max()) + 3), min(height, int(yy.max()) + 3)]
                roi = (slice(box[1], box[3]), slice(box[0], box[2]))
                old_error = np.abs(_rgba_composite(baseline) - _rgba_composite(original)).mean(2)
                new_error = np.abs(_rgba_composite(after) - _rgba_composite(original)).mean(2)
                old_roi, new_roi = float(old_error[roi].mean()), float(new_error[roi].mean())
                actual_paper = changed & compatible_paper
                if not actual_paper.any() or new_roi >= old_roi - 1e-9 or new_error[actual_paper].mean() >= old_error[actual_paper].mean() - 1e-9:
                    raise ValueError("original_roi_or_paper_error_not_strictly_improved")
                proposal.update(roi_xyxy=box, source_local_guard={"accepted": True, "roi_mae_before": old_roi,
                    "roi_mae_after": new_roi, "global_mae_before": float(old_error.mean()), "global_mae_after": float(new_error.mean()),
                    "changed_pixels": int(changed.sum()), "source_paper_changed_pixels": int(actual_paper.sum()),
                    "source_nonpaper_worsened_pixels": int((changed & ~compatible_paper & (new_error > old_error + 1e-9)).sum()),
                    "recolor_alpha_identical": bool(np.array_equal(baseline[:, :, 3], after[:, :, 3])),
                    "whole_scene_acceptance": "pending_caller_source_scene_guard"},
                    renderer_before=baseline_provenance, renderer_after=after_provenance)
                report["proposals"].append(proposal)
            except (ValueError, OSError, RuntimeError) as error:
                report["retained"].append({"drawable_id": record["drawable_id"], "label": record["label"],
                                           "reason": str(error), "source_classification": diagnostics})
    report["examined_candidates"] = examined
    return finish()
