"""Recover a source gradient while retaining existing SVG path geometry.

One verified source field may legitimately paint several existing paths. This
route neither merges them nor claims a geometry-economy certificate. Every
pixel of final alpha must remain identical; source error must not increase in
the complete scene or any selected path, including its edge halo.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

SCHEMA = "ai-vector-cleanroom.gradient-paint-only/v1"
SOLVER = "gradient_paint_only.exact_existing_geometry"
SOURCE_POLICY = "original_every_path_mean_and_p95_processed_path_mean_whole_field_mean_and_p95"
PARTIAL_POLICY = "original_each_path_and_outside_field_mean_p95_processed_mean_changed_halo_mean_p95"
PARTIAL_SCOPE = "same_paths_same_order_same_alpha_partial_source_paint"
NS = "{http://www.w3.org/2000/svg}"
DRAW = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line", "text", "image", "use"}


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _local(node):
    return node.tag.rsplit("}", 1)[-1]


def _context(root, node):
    from source_gradient_primitive import _presentation_context
    parents = {child: parent for parent in root.iter() for child in parent}
    current, fill = node, None
    while current is not None:
        if any(key in current.attrib for key in ("transform", "filter", "mask", "clip-path", "style")):
            raise ValueError("paint_only_unsupported_effect_context")
        if current.get("display") not in (None, "inline") or current.get("visibility") not in (None, "visible"):
            raise ValueError("paint_only_hidden_context")
        if current.get("stroke", "none") != "none":
            raise ValueError("paint_only_stroke_context")
        if fill is None and current.get("fill"):
            fill = current.get("fill")
        current = parents.get(current)
    return fill, _presentation_context(root, node)


def _isolate(root, identifier):
    result = copy.deepcopy(root)
    for parent in result.iter():
        for child in list(parent):
            if _local(child) in DRAW and child.get("id") != identifier:
                parent.remove(child)
    return result


def _halo(mask):
    return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3))) > 0


def _composite(rgba):
    alpha = rgba[:, :, 3, None].astype(float)/255
    return rgba[:, :, :3]*alpha + 255*(1-alpha)


def _native_field_mask(mask, width, height):
    """Sample working pixel cells through the renderer's uniform meet map.

    A true mask cell covers its half-open working pixel square. Each native
    pixel centre is mapped back to that square; letterbox padding is outside
    the field. This transfers a discrete ownership ROI, never source colours.
    """
    from source_edge_reconstruction import _native_mapping
    mask = np.asarray(mask, dtype=bool)
    working_h, working_w = mask.shape
    scale, offset = _native_mapping((working_w, working_h), (width, height))
    xs = np.floor((np.arange(width, dtype=float)+.5-offset[0])/scale).astype(np.int64)
    ys = np.floor((np.arange(height, dtype=float)+.5-offset[1])/scale).astype(np.int64)
    valid_x = (xs >= 0) & (xs < working_w)
    valid_y = (ys >= 0) & (ys < working_h)
    result = mask[np.clip(ys, 0, working_h-1)[:, None], np.clip(xs, 0, working_w-1)[None, :]]
    result[~valid_y, :] = False
    result[:, ~valid_x] = False
    return result


def apply_paint_only_gradient(before_svg, gradient_svg, region, source_path, processed_rgba,
                              *, maximum_paths=4, budget_percent=.25):
    """Return (SVG, source-backed geometry record, proof) or reject.

    ``region`` is an independently validated source-space gradient proposal;
    ``gradient_svg`` provides its unchanged native paint resource. Candidate
    paths are selected from the ordinary-trace counterfactual using ownership
    support, never a fixture's coordinates or colour name.
    """
    from clean_base import _parse_subpaths, _sub_bbox_accumulate
    from source_gradient_primitive import gradient_paint_sha256
    from svg_postprocess import _designer_path_anchors
    from svg_renderer import render_svg_reference, _validated_svg
    _validated_svg(before_svg.encode())
    _validated_svg(gradient_svg.encode())
    root, proposal_root = ET.fromstring(before_svg), ET.fromstring(gradient_svg)
    if any(_local(n) in {"style", "animate", "animateTransform", "set", "switch"} for n in root.iter()):
        raise ValueError("paint_only_unsupported_global_context")
    mask = np.asarray(region["mask"], dtype=bool)
    processed = np.asarray(processed_rgba, dtype=np.uint8)
    if processed.shape != (*mask.shape, 4) or not mask.any():
        raise ValueError("paint_only_source_alignment_required")
    height, width = mask.shape
    view = [float(x) for x in re.split(r"[\s,]+", root.get("viewBox", "").strip())]
    if view != [0., 0., float(width), float(height)]:
        raise ValueError("paint_only_requires_source_pixel_viewbox")
    with Image.open(source_path) as image:
        original = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if max(original.shape[:2]) > 2048 or abs((original.shape[1]/original.shape[0])/(width/height)-1) > 1e-6:
        raise ValueError("paint_only_original_resolution_or_aspect_budget")
    definition = [node for node in proposal_root.iter() if node.get("id") == region["id"]]
    if len(definition) != 1 or _local(definition[0]) not in {"linearGradient", "radialGradient"}:
        raise ValueError("paint_only_native_paint_missing")
    yy, xx = np.nonzero(mask)
    bbox = (float(xx.min())-4, float(yy.min())-4, float(xx.max())+5, float(yy.max())+5)
    existing_ids = [node.get("id") for node in root.iter() if node.get("id")]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("paint_only_duplicate_identity")
    candidates = []
    for ordinal, node in enumerate(root.iter()):
        if _local(node) != "path" or node.get("data-avc-gradient-object"):
            continue
        try:
            fill, _ = _context(root, node)
        except ValueError:
            continue
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill or ""):
            continue
        paths = _parse_subpaths(node.get("d", ""))
        if not paths or any(not path.get("closed") for path in paths):
            continue
        xs, ys = [], []
        for path in paths:
            _sub_bbox_accumulate(path, xs, ys)
        if not xs or not (bbox[0] <= min(xs) <= max(xs) <= bbox[2] and bbox[1] <= min(ys) <= max(ys) <= bbox[3]):
            continue
        identifier = node.get("id") or ("avc-paint-only-path-" + _hash((str(ordinal)+node.get("d", "")).encode())[:18])
        node.set("id", identifier)
        candidates.append(node)
    if not candidates or len(candidates) > 16:
        raise ValueError("paint_only_path_shortlist_ambiguous")
    with tempfile.TemporaryDirectory(prefix="aivc-paint-only-") as temp:
        folder = Path(temp)
        def render(tree, name, render_width):
            svg = folder/(name+".svg")
            ET.ElementTree(tree).write(svg, encoding="utf8")
            provenance = render_svg_reference(svg, svg.with_suffix(".png"), render_width, background=None)
            with Image.open(svg.with_suffix(".png")) as image:
                return np.asarray(image.convert("RGBA"), dtype=np.uint8), provenance
        expanded = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5))) > 0
        selected, support_union = [], np.zeros_like(mask)
        for index, node in enumerate(candidates):
            rgba, _ = render(_isolate(root, node.get("id")), "member-"+str(index), width)
            support = rgba[:, :, 3] > 0
            core = rgba[:, :, 3] >= 128
            if (not core.any() or not np.all(expanded[support])
                    or np.count_nonzero(core & mask)/int(core.sum()) < .9):
                continue
            selected.append(node)
            support_union |= core
        if not 1 <= len(selected) <= maximum_paths or np.count_nonzero(support_union & mask)/int(mask.sum()) < .9:
            raise ValueError("paint_only_complete_existing_path_ownership_not_verified")
        candidate = copy.deepcopy(root)
        identity_token = _hash(str(region["candidate_id"]).encode())[:16]
        gradient_id = "avc-paint-only-gradient-"+identity_token
        if gradient_id in existing_ids:
            raise ValueError("paint_only_gradient_identity_collision")
        definition = copy.deepcopy(definition[0])
        definition.set("id", gradient_id)
        defs = next((node for node in candidate if _local(node) == "defs"), None)
        if defs is None:
            defs = ET.SubElement(candidate, NS+"defs")
        defs.append(definition)
        records = []
        for node in selected:
            identifier = node.get("id")
            target = next(n for n in candidate.iter() if n.get("id") == identifier)
            target.set("fill", "url(#"+gradient_id+")")
            target.set("data-avc-gradient-object", str(region["proposal_id"]))
            anchors = _designer_path_anchors(target.get("d", ""))
            target.set("data-avc-designer-anchors", str(anchors))
            target.set("data-avc-error-budget-percent", str(budget_percent))
            target.set("data-avc-p95-error-percent", "0")
            target.set("data-avc-max-error-percent", "0")
            target.set("data-avc-geometry-evidence-scope", "unchanged_existing_svg_not_original_contour_error")
            target.set("data-avc-paint-only-reconstruction", SCHEMA)
            _, context = _context(candidate, target)
            records.append({"id": identifier, "path_sha256": _hash(target.get("d", "").encode()),
                            "anchors": anchors, "presentation_context": context})
        measurements = []
        for label, source in (("unmodified_input", original), ("processed_reference", processed)):
            h, w = source.shape[:2]
            before, bp = render(root, label+"-before", w)
            after, ap = render(candidate, label+"-after", w)
            if before.shape != source.shape or after.shape != source.shape or not np.array_equal(before[:, :, 3], after[:, :, 3]):
                raise ValueError(label+"_paint_only_alpha_or_canvas_changed")
            errors = {name: np.abs(_composite(arr)-_composite(source)).mean(2)
                      for name, arr in (("before", before), ("after", after))}
            changed = np.any(before != after, axis=2)
            scopes = [("changed_area", _halo(changed))]
            for index, record in enumerate(records):
                isolated, _ = render(_isolate(root, record["id"]), label+"-support-"+str(index), w)
                scopes.append((record["id"], _halo(isolated[:, :, 3] > 0)))
            local = []
            for identifier, support in scopes:
                if not support.any():
                    raise ValueError("paint_only_empty_source_support")
                values = {name: {"rgb_mae": float(error[support].mean()), "rgb_p95": float(np.percentile(error[support], 95))}
                          for name, error in errors.items()}
                # A quantized palette band is not a semantic source boundary.
                # Keep the original image's per-path tail gate, while judging
                # processed-reference tails over the complete replaced field.
                required = ("rgb_mae", "rgb_p95") if label == "unmodified_input" or identifier == "changed_area" else ("rgb_mae",)
                if any(values["after"][key] > values["before"][key]+1e-9 for key in required):
                    raise ValueError(label+"_paint_only_local_source_fidelity_worse:"+identifier+":"+str(values))
                local.append({"scope": identifier, "pixels": int(support.sum()), "measurements": values,
                              "required_nonregression_metrics": list(required),
                              "p95_change": values["after"]["rgb_p95"]-values["before"]["rgb_p95"]})
            global_errors = {name: float(error.mean()) for name, error in errors.items()}
            if global_errors["after"] >= global_errors["before"]:
                raise ValueError(label+"_paint_only_no_original_improvement")
            measurements.append({"source_reference_kind": label, "source_size": [w, h],
                                 "source_rgba_sha256": _hash(source.tobytes()), "alpha_unchanged": True,
                                 "before_alpha_sha256": _hash(before[:, :, 3].tobytes()),
                                 "after_alpha_sha256": _hash(after[:, :, 3].tobytes()),
                                 "global_rgb_mae": global_errors, "local": local,
                                 "renderer_before": bp, "renderer_after": ap})
    proof = {"schema": SCHEMA, "status": "source_paint_verified_existing_geometry",
             "candidate_id": region["candidate_id"], "gradient_object_id": region["proposal_id"],
             "gradient_id": gradient_id, "paint_sha256": gradient_paint_sha256(candidate, gradient_id),
             "ownership_mask_sha256": _hash(mask.tobytes()), "drawable_records": records,
             "measurements": measurements, "scope": "same_paths_same_order_same_alpha_new_source_verified_paint",
             "source_validation_policy": SOURCE_POLICY,
             "geometry_optimized": False, "human_acceptance": "not_performed"}
    anchors = sum(row["anchors"] for row in records)
    geometry = {"solver": SOLVER, "anchor_count": anchors, "designer_anchor_count": anchors,
                "segment_count": anchors, "source_paint_only": proof,
                "topology": {"topology_preserved": True, "scope": "exact_existing_svg_alpha_at_two_source_resolutions"},
                "error_budget": {"passed": True, "requested_max_percent": budget_percent,
                                 "actual_p95_error_percent": 0., "actual_max_error_percent": 0.,
                                 "metric": "exact_existing_svg_path_geometry_not_original_contour_error"}}
    return ET.tostring(candidate, encoding="unicode"), geometry, proof


def apply_partial_paint_only_gradient(before_svg, gradient_svg, region, source_path,
                                      processed_rgba, *, maximum_paths=4,
                                      maximum_trials=16, budget_percent=.25,
                                      native_reference_rgba=None):
    """Try a bounded set of exact existing paths, retaining only source wins.

    The field mask limits discovery, never authorizes recolouring its exterior.
    Every selected path and its out-of-field halo are measured independently.
    This is partial paint assistance requiring review, not a completed object.
    """
    from clean_base import _parse_subpaths, _sub_bbox_accumulate
    from source_gradient_primitive import gradient_paint_sha256
    from source_scene_guard import _render_native_payload, validate_source_scene_arrays
    from svg_postprocess import _designer_path_anchors
    from svg_renderer import _validated_svg
    _validated_svg(before_svg.encode())
    _validated_svg(gradient_svg.encode())
    if not 1 <= maximum_paths <= 4 or not 1 <= maximum_trials <= 16:
        raise ValueError("partial_paint_search_budget_invalid")
    root, proposal = ET.fromstring(before_svg), ET.fromstring(gradient_svg)
    if any(_local(n) in {"style", "animate", "animateTransform", "set", "switch"} for n in root.iter()):
        raise ValueError("paint_only_unsupported_global_context")
    mask = np.asarray(region["mask"], dtype=bool)
    processed = np.asarray(processed_rgba, dtype=np.uint8)
    if mask.ndim != 2 or processed.shape != (*mask.shape, 4) or not mask.any():
        raise ValueError("paint_only_source_alignment_required")
    h, w = mask.shape
    view = [float(x) for x in re.split(r"[\s,]+", root.get("viewBox", "").strip())]
    if view != [0., 0., float(w), float(h)]:
        raise ValueError("paint_only_requires_source_pixel_viewbox")
    with Image.open(source_path) as image:
        if max(image.size) > 2048:
            raise ValueError("paint_only_original_resolution_budget")
        original = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    native_reference = (np.asarray(native_reference_rgba, dtype=np.uint8)
                        if native_reference_rgba is not None else
                        processed if processed.shape == original.shape else None)
    if native_reference is not None and native_reference.shape != original.shape:
        raise ValueError("partial_paint_native_reference_dimensions_differ")
    resources = [node for node in proposal.iter() if node.get("id") == region["id"]]
    if len(resources) != 1 or _local(resources[0]) not in {"linearGradient", "radialGradient"}:
        raise ValueError("paint_only_native_paint_missing")
    ids = [n.get("id") for n in root.iter() if n.get("id")]
    if len(ids) != len(set(ids)):
        raise ValueError("paint_only_duplicate_identity")
    parents = {child: parent for parent in root.iter() for child in parent}
    yy, xx = np.nonzero(mask)
    bbox = (float(xx.min())-4, float(yy.min())-4, float(xx.max())+5, float(yy.max())+5)
    candidates = []
    for ordinal, node in enumerate(root.iter()):
        if _local(node) != "path":
            continue
        ancestor, owned = node, False
        while ancestor is not None:
            owned |= bool(ancestor.get("data-avc-gradient-object"))
            ancestor = parents.get(ancestor)
        if owned:
            continue
        try:
            fill, _ = _context(root, node)
        except ValueError:
            continue
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill or ""):
            continue
        paths = _parse_subpaths(node.get("d", ""))
        if not paths or any(not part.get("closed") for part in paths):
            continue
        xs, ys = [], []
        for part in paths:
            _sub_bbox_accumulate(part, xs, ys)
        if not xs or max(xs) < bbox[0] or min(xs) > bbox[2] or max(ys) < bbox[1] or min(ys) > bbox[3]:
            continue
        candidates.append((ordinal, node))
    # Avoid a geometry-wide expensive scan: too many intersecting paths is an
    # unsupported fragmented scene, not permission to cherry-pick an index.
    if not candidates or len(candidates) > 64:
        raise ValueError("partial_paint_path_shortlist_ambiguous")
    def render(tree, source):
        return _render_native_payload(ET.tostring(tree), source.shape[1], source.shape[0])
    shortlist = []
    for ordinal, node in candidates:
        identifier = node.get("id") or "avc-partial-paint-path-"+_hash((str(ordinal)+node.get("d", "")).encode())[:18]
        node.set("id", identifier)
        alpha = render(_isolate(root, identifier), processed)[:, :, 3]
        core = alpha >= 128
        overlap = int((core & mask).sum())
        if overlap and overlap / max(1, int(core.sum())) >= .5:
            shortlist.append((-overlap, ordinal, identifier))
    shortlist.sort()
    if not shortlist:
        raise ValueError("partial_paint_no_supported_search_paths")
    original_root = copy.deepcopy(root)
    sources = {"unmodified_input": original, "processed_reference": processed}
    initial = {name: render(root, source) for name, source in sources.items()}
    scenes = dict(initial)
    fields = {name: _native_field_mask(mask, source.shape[1], source.shape[0])
              for name, source in sources.items()}
    support_cache = {}
    gradient_id = "avc-partial-paint-gradient-"+_hash(str(region["candidate_id"]).encode())[:16]
    if gradient_id in ids:
        raise ValueError("paint_only_gradient_identity_collision")
    definition = copy.deepcopy(resources[0]); definition.set("id", gradient_id)
    selected, trials, scene_evidence = [], [], []
    def scene_checks(previous, after):
        evidence = []
        for name, source in sources.items():
            changed = np.any(previous[name] != after[name], axis=2)
            yy, xx = np.nonzero(changed)
            if not len(xx):
                raise ValueError("partial_paint_empty_scene_change")
            roi = [max(0, int(xx.min())-1), max(0, int(yy.min())-1),
                   min(source.shape[1], int(xx.max())+2), min(source.shape[0], int(yy.max())+2)]
            reference = native_reference if name == "unmodified_input" else processed
            result = validate_source_scene_arrays(previous[name], after[name], source,
                processed_reference_rgba=reference, roi_xyxy=roi)
            if result.get("accepted") is not True:
                raise ValueError(name+"_partial_paint_source_scene_rejected:"+",".join(result.get("reasons", [])))
            evidence.append({"source_reference_kind": name, "before_rgba_sha256": _hash(previous[name].tobytes()),
                "after_rgba_sha256": _hash(after[name].tobytes()), "source_rgba_sha256": _hash(source.tobytes()),
                "processed_reference_rgba_sha256": None if reference is None else _hash(reference.tobytes()),
                "accepted": True, "boundary_mode": "strict", "roi_xyxy": roi, "result": result})
        return evidence
    def measure(previous, after, identifiers):
        measurements = []
        for name, source in sources.items():
            a, b = previous[name], after[name]
            if not np.array_equal(a[:, :, 3], b[:, :, 3]):
                raise ValueError(name+"_paint_only_alpha_changed")
            errors = {key: np.abs(_composite(arr)-_composite(source)).mean(2)
                      for key, arr in (("before", a), ("after", b))}
            scopes = [("changed_area", _halo(np.any(a != b, axis=2)))]
            for identifier in identifiers:
                key = (name, identifier)
                if key not in support_cache:
                    support_cache[key] = _halo(render(_isolate(original_root, identifier), source)[:, :, 3] > 0)
                support = support_cache[key]
                scopes.extend(((identifier, support), (identifier+":outside_field", support & ~fields[name])))
            local = []
            for identifier, support in scopes:
                count = int(support.sum())
                if not count and not identifier.endswith(":outside_field"):
                    raise ValueError("paint_only_empty_source_support")
                values = {key: {"rgb_mae": float(e[support].mean()) if count else 0.,
                                "rgb_p95": float(np.percentile(e[support], 95)) if count else 0.}
                          for key, e in errors.items()}
                required = ("rgb_mae", "rgb_p95") if name == "unmodified_input" or identifier == "changed_area" else ("rgb_mae",)
                if any(values["after"][key] > values["before"][key]+1e-9 for key in required):
                    raise ValueError(name+"_partial_paint_local_source_worse:"+identifier)
                local.append({"scope": identifier, "pixels": count, "measurements": values,
                              "required_nonregression_metrics": list(required)})
            global_errors = {key: float(value.mean()) for key, value in errors.items()}
            if global_errors["after"] >= global_errors["before"]:
                raise ValueError(name+"_paint_only_no_original_improvement")
            provenance = {"renderer": "resvg", "paint_model": "native_svg", "background": None,
                          "canvas": [source.shape[1], source.shape[0]], "viewport": "explicit_native_default_meet"}
            measurements.append({"source_reference_kind": name, "source_size": [source.shape[1], source.shape[0]],
                "field_mask_sha256": _hash(fields[name].tobytes()),
                "field_mask_sampling": "inverse_default_meet_native_pixel_centres_half_open_working_cells",
                "source_rgba_sha256": _hash(source.tobytes()), "alpha_unchanged": True,
                "before_alpha_sha256": _hash(a[:, :, 3].tobytes()), "after_alpha_sha256": _hash(b[:, :, 3].tobytes()),
                "global_rgb_mae": global_errors, "local": local, "renderer_before": provenance, "renderer_after": dict(provenance)})
        return measurements
    for _overlap, _ordinal, identifier in shortlist[:maximum_trials]:
        if len(selected) >= maximum_paths:
            break
        candidate = copy.deepcopy(root)
        defs = next((n for n in candidate if _local(n) == "defs"), None)
        if defs is None:
            defs = ET.SubElement(candidate, NS+"defs")
        if not any(n.get("id") == gradient_id for n in defs):
            defs.append(copy.deepcopy(definition))
        next(n for n in candidate.iter() if n.get("id") == identifier).set("fill", "url(#"+gradient_id+")")
        after = {name: render(candidate, source) for name, source in sources.items()}
        try:
            measure(scenes, after, [identifier])
            # Re-evaluate all previously accepted paths against the original
            # scene: overlapping later paints must not undo an earlier win.
            measure(initial, after, selected+[identifier])
            evidence = scene_checks(initial, after)
        except ValueError as exc:
            trials.append({"path_id": identifier, "accepted": False, "reason": str(exc)})
            continue
        selected.append(identifier); root, scenes = candidate, after
        scene_evidence = evidence
        trials.append({"path_id": identifier, "accepted": True})
    if not selected:
        raise ValueError("partial_paint_no_path_passed_original_source_checks")
    selected_set = set(selected)
    selected = [n.get("id") for n in root.iter() if n.get("id") in selected_set]
    measurements = measure(initial, scenes, selected)
    records = []
    for node in root.iter():
        if node.get("id") not in selected_set:
            continue
        node.set("data-avc-gradient-object", str(region["proposal_id"]))
        anchors = _designer_path_anchors(node.get("d", ""))
        node.set("data-avc-designer-anchors", str(anchors))
        node.set("data-avc-paint-only-reconstruction", SCHEMA)
        node.set("data-avc-partial-paint-review", "required")
        node.set("data-avc-geometry-evidence-scope", "unchanged_existing_svg_not_original_contour_error")
        node.set("data-avc-error-budget-percent", str(budget_percent))
        node.set("data-avc-p95-error-percent", "0")
        node.set("data-avc-max-error-percent", "0")
        _, context = _context(root, node)
        records.append({"id": node.get("id"), "path_sha256": _hash(node.get("d", "").encode()),
                        "anchors": anchors, "presentation_context": context})
    def geometry_order(tree):
        return [(n.tag, tuple(sorted((k, v) for k, v in n.attrib.items()
                                    if k != "fill" and not k.startswith("data-avc-"))))
                for n in tree.iter() if _local(n) in DRAW]
    if geometry_order(root) != geometry_order(original_root):
        raise ValueError("partial_paint_geometry_or_order_changed")
    proof = {"schema": SCHEMA, "status": "source_paint_verified_existing_geometry", "candidate_id": region["candidate_id"],
        "gradient_object_id": region["proposal_id"], "gradient_id": gradient_id,
        "paint_sha256": gradient_paint_sha256(root, gradient_id), "ownership_mask_sha256": _hash(mask.tobytes()),
        "drawable_records": records, "measurements": measurements, "scope": PARTIAL_SCOPE,
        "source_validation_policy": PARTIAL_POLICY, "geometry_optimized": False, "partial_selection": True,
        "manual_review_required": True, "complete_field_reconstructed": False, "geometry_order_unchanged": True,
        "source_scene_checks": scene_evidence, "search_trials": trials, "human_acceptance": "not_performed"}
    anchors = sum(row["anchors"] for row in records)
    geometry = {"solver": SOLVER, "anchor_count": anchors, "designer_anchor_count": anchors, "segment_count": anchors,
        "source_paint_only": proof, "topology": {"topology_preserved": True, "scope": "exact_existing_svg_alpha_at_two_source_resolutions"},
        "error_budget": {"passed": True, "requested_max_percent": budget_percent,
                         "actual_p95_error_percent": 0., "actual_max_error_percent": 0.,
                         "metric": "exact_existing_svg_path_geometry_not_original_contour_error"}}
    return ET.tostring(root, encoding="unicode"), geometry, proof


def apply_pending_paint_alternatives(svg_text, alternatives, source_path, processed_rgba,
                                     *, budget_percent=.25, maximum_fields=1,
                                     native_reference_rgba=None):
    """Recover partial paint from an explicitly uncommitted stage queue.

    At most eight alternatives and one accepted field are considered. Returned
    details have independent native-render evidence, never selected geometry.
    """
    from gradient_reconstruction_stage import decode_mask_rle
    if isinstance(maximum_fields, bool) or maximum_fields != 1:
        raise ValueError("partial_paint_one_field_transaction_limit")
    current, details, decisions, proofs, accepted_masks = svg_text, [], [], [], []
    h, w = np.asarray(processed_rgba).shape[:2]
    for option in list(alternatives)[:8]:
        if len(details) >= maximum_fields:
            break
        candidate_id = option.get("candidate_id")
        try:
            heldout = option.get("heldout_evidence") or {}
            if (option.get("schema") != "ai-vector-cleanroom.paint-ready-alternative/v1"
                    or option.get("status") != "pending_native_source_and_existing_path_validation"
                    or option.get("geometry_certified") is not False
                    or heldout.get("validation", {}).get("passed") is not True
                    or heldout.get("independent_revalidation", {}).get("passed") is not True):
                raise ValueError("partial_paint_pending_evidence_incomplete")
            mask = decode_mask_rle(option["mask"])
            if any(int((mask & prior).sum()) / max(1, min(int(mask.sum()), int(prior.sum()))) >= .8 for prior in accepted_masks):
                raise ValueError("partial_paint_overlaps_already_assisted_field")
            model, stops = option["model"], option["stops"]
            if not 2 <= len(stops) <= 5:
                raise ValueError("partial_paint_native_stop_budget")
            resource = ET.Element(NS+"svg", {"viewBox": f"0 0 {w} {h}", "width": str(w), "height": str(h)})
            defs = ET.SubElement(resource, NS+"defs")
            if model.get("type") == "linear":
                attrs = {key: str(model[key]) for key in ("x1", "y1", "x2", "y2")}
                paint = ET.SubElement(defs, NS+"linearGradient", {"id": "pending-paint", "gradientUnits": "userSpaceOnUse", **attrs})
            elif model.get("type") == "radial":
                cx, cy = model["center"]
                paint = ET.SubElement(defs, NS+"radialGradient", {"id": "pending-paint", "gradientUnits": "userSpaceOnUse",
                    "cx": "0", "cy": "0", "r": "1", "gradientTransform":
                    f'translate({cx} {cy}) rotate({model.get("rotation_degrees", 0)}) scale({model["radius_x"]} {model["radius_y"]})'})
            else:
                raise ValueError("partial_paint_native_model_unsupported")
            for stop in stops:
                colour = stop.get("color") or "#{:02x}{:02x}{:02x}".format(*(int(v) for v in stop["rgb"]))
                ET.SubElement(paint, NS+"stop", {"offset": str(stop["offset"]), "stop-color": colour})
            region = {"mask": mask, "id": "pending-paint", "candidate_id": candidate_id,
                      "proposal_id": "partial-paint-object-"+_hash(str(candidate_id).encode())[:16]}
            result, geometry, proof = apply_partial_paint_only_gradient(
                current, ET.tostring(resource, encoding="unicode"), region, source_path, processed_rgba,
                budget_percent=budget_percent, native_reference_rgba=native_reference_rgba)
            if not paint_only_certificate_valid(geometry):
                raise ValueError("partial_paint_certificate_invalid")
            detail = {"id": proof["gradient_id"], "candidate_id": candidate_id,
                "candidate_family": option.get("candidate_family"), "model": dict(model),
                "type": model["type"], "stops": copy.deepcopy(stops), "viewbox": [w, h], "opacity": 1.,
                "manual_review_required": True,
                "validation": {"engine": "source_space_heldout_gradient_object", "paint": copy.deepcopy(heldout),
                    "geometry": geometry, "selection": {"colour_used_for_geometry": False,
                        "geometry_source": "unchanged_existing_svg_geometry", "partial_paint_only": True}}}
            if model["type"] == "linear":
                detail.update({key: float(model[key]) for key in ("x1", "y1", "x2", "y2")})
            current = result; details.append(detail); proofs.append(proof); accepted_masks.append(mask)
            decisions.append({"candidate_id": candidate_id, "status": "partial_paint_selected",
                              "path_count": len(proof["drawable_records"]), "manual_review_required": True,
                              "reasons": ["native_original_and_processed_source_improved_on_unchanged_existing_paths"]})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            decisions.append({"candidate_id": candidate_id, "status": "partial_paint_rejected", "reasons": [str(exc)[:240]]})
    return current, details, decisions, proofs


def paint_only_certificate_valid(geometry):
    try:
        proof = geometry["source_paint_only"]
        partial = proof.get("partial_selection") is True
        if (geometry.get("solver") != SOLVER or proof.get("schema") != SCHEMA
                or proof.get("status") != "source_paint_verified_existing_geometry"
                or proof.get("geometry_optimized") is not False
                or proof.get("source_validation_policy") != (PARTIAL_POLICY if partial else SOURCE_POLICY)
                or proof.get("scope") != (PARTIAL_SCOPE if partial else "same_paths_same_order_same_alpha_new_source_verified_paint")):
            return False
        if partial and (proof.get("manual_review_required") is not True
                        or proof.get("complete_field_reconstructed") is not False
                        or proof.get("geometry_order_unchanged") is not True):
            return False
        if partial:
            checks = proof.get("source_scene_checks")
            if (not isinstance(checks, list) or len(checks) != 2
                    or {row.get("source_reference_kind") for row in checks} != {"unmodified_input", "processed_reference"}):
                return False
            for check in checks:
                result = check.get("result") or {}
                if (check.get("accepted") is not True or check.get("boundary_mode") != "strict"
                        or result.get("accepted") is not True or result.get("reasons") != []
                        or result.get("boundary_mode") != "strict"
                        or result.get("metrics", {}).get("global_color_nonregression") is not True
                        or result.get("metrics", {}).get("roi_color_nonregression") is not True
                        or any(not re.fullmatch(r"[0-9a-f]{64}", check.get(key, "")) for key in
                               ("before_rgba_sha256", "after_rgba_sha256", "source_rgba_sha256"))):
                    return False
        records = proof["drawable_records"]
        ids = [record["id"] for record in records]
        if not 1 <= len(ids) <= 4 or len(set(ids)) != len(ids) or any(not value for value in ids):
            return False
        if any(not isinstance(row["anchors"], int) or row["anchors"] < 1 for row in records):
            return False
        if geometry["anchor_count"] != sum(row["anchors"] for row in records) or geometry["designer_anchor_count"] != geometry["anchor_count"]:
            return False
        rows = proof["measurements"]
        if {row["source_reference_kind"] for row in rows} != {"unmodified_input", "processed_reference"} or len(rows) != 2:
            return False
        finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for row in rows:
            if row["alpha_unchanged"] is not True or row["before_alpha_sha256"] != row["after_alpha_sha256"]:
                return False
            expected_scopes = {*ids, "changed_area"}
            if partial:
                expected_scopes.update(identifier+":outside_field" for identifier in ids)
            if {record["scope"] for record in row["local"]} != expected_scopes:
                return False
            for local in row["local"]:
                m = local["measurements"]
                if any(not finite(m[name][key]) or m[name][key] < 0 for name in ("before", "after") for key in ("rgb_mae", "rgb_p95")):
                    return False
                required = ("rgb_mae", "rgb_p95") if row["source_reference_kind"] == "unmodified_input" or local["scope"] == "changed_area" else ("rgb_mae",)
                if local.get("required_nonregression_metrics") != list(required):
                    return False
                if any(m["after"][key] > m["before"][key]+1e-9 for key in required):
                    return False
            m = row["global_rgb_mae"]
            if not all(finite(m[k]) for k in ("before", "after")) or not 0 <= m["after"] < m["before"]:
                return False
            for key in ("renderer_before", "renderer_after"):
                renderer = row[key]
                if renderer.get("renderer") != "resvg" or renderer.get("paint_model") != "native_svg" or renderer.get("background") is not None:
                    return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def final_paint_only_matches(root, geometry, source_path):
    from source_gradient_primitive import gradient_paint_sha256
    from svg_postprocess import _designer_path_anchors
    if not paint_only_certificate_valid(geometry):
        return False
    try:
        proof = geometry["source_paint_only"]
        ids = [row["id"] for row in proof["drawable_records"]]
        actual = [n for n in root.iter() if n.get("data-avc-gradient-object") == proof["gradient_object_id"]]
        if [n.get("id") for n in actual] != ids or gradient_paint_sha256(root, proof["gradient_id"]) != proof["paint_sha256"]:
            return False
        for node, record in zip(actual, proof["drawable_records"]):
            fill, context = _context(root, node)
            if (_local(node) != "path" or context != record["presentation_context"]
                    or fill != "url(#"+proof["gradient_id"]+")"
                    or _hash(node.get("d", "").encode()) != record["path_sha256"]
                    or _designer_path_anchors(node.get("d", "")) != record["anchors"]):
                return False
        with Image.open(source_path) as image:
            raw = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        record = next(row for row in proof["measurements"] if row["source_reference_kind"] == "unmodified_input")
        return record["source_size"] == [raw.shape[1], raw.shape[0]] and record["source_rgba_sha256"] == _hash(raw.tobytes())
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return False
