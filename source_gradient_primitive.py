"""Source-proven circle/rounded-rectangle reconstruction for one gradient.

This is a new source reconstruction, not simplification of the binary traced
boundary. It never changes the paint, claims author intent, or grants a binary
geometry exemption. All unsupported/ambiguous scenes fail closed.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SCHEMA = "ai-vector-cleanroom.source-gradient-circle/v1"
RECT_SCHEMA = "ai-vector-cleanroom.source-gradient-roundrect/v1"
RECT_SOLVER = "source_gradient_primitive.source_roundrect"
SOLVER = "source_gradient_primitive.source_circle"
NS = "{http://www.w3.org/2000/svg}"
DRAWABLES = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon", "text", "image", "use"}


def _local(node):
    return node.tag.rsplit("}", 1)[-1]


def gradient_paint_sha256(root, gradient_id):
    """Ignore inert annotations; bind actual attributes, stop order and paint."""
    nodes = [node for node in root.iter() if node.get("id") == gradient_id]
    if len(nodes) != 1 or _local(nodes[0]) not in {"linearGradient", "radialGradient"}:
        raise ValueError("source_gradient_paint_identity_ambiguous")
    def record(node):
        return [_local(node), sorted((key, value) for key, value in node.attrib.items()
                                    if not key.startswith("data-")),
                [record(child) for child in node]]
    return hashlib.sha256(json.dumps(record(nodes[0]), separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _context(root, *, allow_residuals=False):
    if any(_local(node) in {"style", "animate", "animateTransform", "set", "switch"} for node in root.iter()):
        raise ValueError("source_gradient_unsupported_styles_or_animation")
    shapes = [node for node in root.iter() if _local(node) in DRAWABLES]
    targets = [item for item in shapes if item.get("data-avc-gradient-object")]
    if len(targets) != 1 or _local(targets[0]) not in {"path", "circle", "rect"}:
        raise ValueError("source_gradient_requires_one_isolated_drawable")
    node = targets[0]
    extras = [item for item in shapes if item is not node]
    if extras and (not allow_residuals or len(extras) > 8
                   or any(_local(item) != "path" for item in extras)):
        raise ValueError("source_gradient_requires_one_isolated_drawable")
    # Stable local IDs also bind anonymous residuals to exact source geometry.
    used = [item.get("id") for item in root.iter() if item.get("id")]
    if len(used) != len(set(used)):
        raise ValueError("source_gradient_duplicate_identity")
    for index, extra in enumerate(extras):
        if not extra.get("id"):
            identity = "source-aa-residual-" + hashlib.sha256(ET.tostring(extra)).hexdigest()[:20]
            if identity in used:
                raise ValueError("source_gradient_residual_identity_ambiguous")
            extra.set("id", identity)
            used.append(identity)
        if any(key in extra.attrib for key in ("transform", "filter", "clip-path", "mask", "style")):
            raise ValueError("source_gradient_unsupported_residual_context")
    if not node.get("id") or not node.get("data-avc-gradient-object"):
        raise ValueError("source_gradient_stable_identity_missing")
    if _local(node) == "path":
        from clean_base import _parse_subpaths
        loops = _parse_subpaths(node.get("d", ""))
        if len(loops) != 1 or not loops[0].get("closed"):
            raise ValueError("source_gradient_compound_or_open_boundary")
    parents = {child: parent for parent in root.iter() for child in parent}
    current, paint = node, None
    while current is not None:
        if current.get("display") not in (None, "inline") or current.get("visibility") not in (None, "visible"):
            raise ValueError("source_gradient_hidden_presentation")
        if any(key in current.attrib for key in ("transform", "filter", "clip-path", "mask", "style")):
            raise ValueError("source_gradient_unsupported_effect_context")
        for key in ("opacity", "fill-opacity", "stroke-opacity"):
            if current.get(key) not in (None, "1", "1.0", "100%"):
                raise ValueError("source_gradient_transparency_ambiguity")
        if current.get("stroke", "none") != "none":
            raise ValueError("source_gradient_stroke_or_occlusion")
        if paint is None and current.get("fill"):
            paint = current.get("fill")
        current = parents.get(current)
    match = re.fullmatch(r"url\(#([^ )]+)\)", paint or "")
    if not match:
        raise ValueError("source_gradient_requires_native_gradient_fill")
    gradient_id = match[1]
    definitions = [item for item in root.iter() if item.get("id") == gradient_id]
    if len(definitions) != 1:
        raise ValueError("source_gradient_paint_identity_ambiguous")
    gradient = definitions[0]
    if (_local(gradient) not in {"linearGradient", "radialGradient"}
            or gradient.get("gradientUnits") != "userSpaceOnUse"
            or not 2 <= len(list(gradient)) <= 5
            or any(_local(stop) != "stop" or stop.get("stop-opacity", "1") not in ("1", "1.0")
                   for stop in gradient)
            or any(key.rsplit("}", 1)[-1] == "href" for key in gradient.attrib)):
        raise ValueError("source_gradient_unsupported_paint")
    for extra in extras:
        current, fill = extra, None
        while current is not None:
            if current.get("display") not in (None, "inline") or current.get("visibility") not in (None, "visible"):
                raise ValueError("source_gradient_hidden_presentation")
            if any(key in current.attrib for key in ("transform", "filter", "clip-path", "mask", "style")):
                raise ValueError("source_gradient_unsupported_residual_context")
            if current.get("stroke", "none") != "none" or any(
                    current.get(key) not in (None, "1", "1.0", "100%")
                    for key in ("opacity", "fill-opacity", "stroke-opacity")):
                raise ValueError("source_gradient_unsupported_residual_paint")
            if fill is None and current.get("fill"):
                fill = current.get("fill")
            current = parents.get(current)
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", fill or ""):
            raise ValueError("source_gradient_residual_requires_plain_solid_fill")
    return node, gradient_id, extras


def _presentation_context(root, target):
    parents = {child: parent for parent in root.iter() for child in parent}
    current, inherited = target, {}
    keys = ("fill", "fill-rule", "clip-rule", "color", "color-interpolation", "color-rendering",
            "shape-rendering", "display", "visibility", "vector-effect", "paint-order",
            "mix-blend-mode", "isolation", "overflow")
    while current is not None:
        for key in keys:
            if key not in inherited and current.get(key) is not None:
                inherited[key] = current.get(key)
        current = parents.get(current)
    return {"canvas": {key: root.get(key) for key in ("width", "height", "viewBox", "preserveAspectRatio")},
            "inherited_presentation": inherited}


def _circle(points, budget):
    from curve_refit import _fit_circle
    from geometry_error_optimizer import _nearest_distances, _salient_corner_indices
    fitted = _fit_circle(points)
    if fitted is None:
        raise ValueError("source_gradient_circle_fit_failed")
    centre, radius, _ = fitted
    if not np.isfinite(centre).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError("source_gradient_circle_parameters_invalid")
    points = points[np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))]
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    theta = np.linspace(0, math.tau, 2049)
    curve = centre + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    errors = np.concatenate((_nearest_distances(points, curve),
                             _nearest_distances(curve[:-1], np.vstack((points, points[0])))))
    p95 = float(np.percentile(errors, 95)) * 100 / scale
    maximum = float(errors.max()) * 100 / scale
    tail = float(np.mean(errors > scale * budget / 100))
    corners = _salient_corner_indices(points, closed=True, scale=scale)
    if p95 > budget or maximum > 3 * budget or tail > 0.05 or corners:
        raise ValueError("source_gradient_original_circle_contour_budget_exceeded")
    primitive = {"element": "circle", "cx": float(centre[0]), "cy": float(centre[1]), "r": float(radius)}
    return primitive, {"bidirectional_p95_percent": p95, "bidirectional_maximum_percent": maximum,
                       "over_budget_share": tail, "salient_corner_count": len(corners),
                       "source_contour_samples": len(points), "circle_contour_samples": 2049,
                       "error_budget_percent": budget, "source_bbox_diagonal": scale}


def _remove_drawables_except(root, ids):
    for parent in root.iter():
        for node in list(parent):
            if _local(node) in DRAWABLES and node.get("id") not in ids:
                parent.remove(node)
    for parent in reversed(list(root.iter())):
        for node in list(parent):
            if _local(node) == "g" and not len(node):
                parent.remove(node)


def _dilate(mask):
    height, width = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[y:y+height, x:x+width]
                                 for y in range(3) for x in range(3)])


def _roundrect(points, budget, pixel_size, radius_override=None):
    """Infer four straight sides and one common corner radius from the source.

    The fit is only a proposal. Independent bidirectional contour distances,
    corner checks and actual source-paint validation decide whether it is usable.
    """
    from geometry_error_optimizer import _nearest_distances, _salient_corner_indices
    left = float(np.median(points[points[:, 0] <= np.quantile(points[:, 0], .1), 0]))
    right = float(np.median(points[points[:, 0] >= np.quantile(points[:, 0], .9), 0]))
    top = float(np.median(points[points[:, 1] <= np.quantile(points[:, 1], .1), 1]))
    bottom = float(np.median(points[points[:, 1] >= np.quantile(points[:, 1], .9), 1]))
    width, height = right - left, bottom - top
    epsilon = max(pixel_size * .02, min(width, height) * .001)
    supports = [int(np.count_nonzero(np.abs(points[:, axis] - value) < epsilon))
                for axis, value in ((0, left), (0, right), (1, top), (1, bottom))]
    if min(supports) < 4:
        raise ValueError("source_gradient_roundrect_four_straight_sides_missing")
    u = np.minimum(points[:, 0] - left, right - points[:, 0])
    v = np.minimum(points[:, 1] - top, bottom - points[:, 1])
    corner = ((u > epsilon) & (v > epsilon)
              & (u < min(width, height) * .4) & (v < min(width, height) * .4))
    mid = np.array([(left+right)/2, (top+bottom)/2])
    quadrants = ((points[:, 0] > mid[0]).astype(int)
                 + 2 * (points[:, 1] > mid[1]).astype(int))
    corner_supports = [int(np.count_nonzero(corner & (quadrants == q))) for q in range(4)]
    if min(corner_supports) < 3:
        raise ValueError("source_gradient_roundrect_four_round_corners_missing")
    radii = u[corner] + v[corner] + np.sqrt(2*u[corner]*v[corner])
    radius = float(np.median(radii)) if radius_override is None else float(radius_override)
    if not pixel_size <= radius < min(width, height)/2 - pixel_size:
        raise ValueError("source_gradient_roundrect_radius_ambiguous")
    contour = []
    for cx, cy, start in ((right-radius, top+radius, -math.pi/2),
                           (right-radius, bottom-radius, 0),
                           (left+radius, bottom-radius, math.pi/2),
                           (left+radius, top+radius, math.pi)):
        theta = np.linspace(start, start + math.pi/2, 257)
        contour.extend(np.column_stack((cx + radius*np.cos(theta), cy + radius*np.sin(theta))))
    curve = np.asarray(contour)
    curve = np.vstack((curve, curve[0]))
    points = points[np.argsort(np.arctan2(points[:, 1]-mid[1], points[:, 0]-mid[0]))]
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    errors = np.concatenate((_nearest_distances(points, curve),
                             _nearest_distances(curve[:-1], np.vstack((points, points[0])))))
    p95, maximum = float(np.percentile(errors, 95))*100/scale, float(errors.max())*100/scale
    tail = float(np.mean(errors > scale*budget/100))
    corners = _salient_corner_indices(points, closed=True, scale=scale)
    if p95 > budget or maximum > 3*budget or tail > .05 or corners:
        raise ValueError("source_gradient_original_roundrect_contour_budget_exceeded")
    return ({"element": "rect", "x": left, "y": top, "width": width, "height": height,
             "rx": radius, "ry": radius},
            {"bidirectional_p95_percent": p95, "bidirectional_maximum_percent": maximum,
             "over_budget_share": tail, "salient_corner_count": len(corners),
             "source_contour_samples": len(points), "roundrect_contour_samples": len(curve),
             "flat_edge_samples": supports, "corner_samples": corner_supports,
             "error_budget_percent": budget, "source_bbox_diagonal": scale})


def propose_isolated_gradient_circle(svg_text, source_path, *, budget_percent=0.25):
    return _propose(svg_text, source_path, budget_percent=budget_percent, shape="circle")


def propose_isolated_gradient_roundrect(svg_text, source_path, *, budget_percent=0.25):
    return _propose(svg_text, source_path, budget_percent=budget_percent, shape="roundrect")


def propose_isolated_gradient_primitive(svg_text, source_path, *, budget_percent=0.25):
    """Try the two explicit source reconstructions, without relaxing either."""
    failures = []
    for proposer in (propose_isolated_gradient_circle, propose_isolated_gradient_roundrect):
        try:
            return proposer(svg_text, source_path, budget_percent=budget_percent)
        except ValueError as exc:
            failures.append(str(exc))
    raise ValueError("; ".join(failures))


def _propose(svg_text, source_path, *, budget_percent, shape):
    """Return (SVG text, geometry evidence, source certificate), or reject.

    The supplied file is the conversion's unmodified input PNG, not a flattened
    palette/reference. Source and native-render canvas coordinates are checked.
    """
    from curve_refit import _mask_topology
    from svg_renderer import render_svg_reference, _validated_svg
    budget = float(budget_percent)
    if not math.isfinite(budget) or not .05 <= budget <= 2:
        raise ValueError("source_gradient_invalid_budget")
    _validated_svg(svg_text.encode("utf-8"))
    root = ET.fromstring(svg_text)
    target, gradient_id, extras = _context(root, allow_residuals=shape == "roundrect")
    schema = SCHEMA if shape == "circle" else RECT_SCHEMA
    solver = SOLVER if shape == "circle" else RECT_SOLVER
    identifier, object_id = target.get("id"), target.get("data-avc-gradient-object")
    view = np.array([float(value) for value in re.split(r"[\s,]+", root.get("viewBox", "").strip())])
    if len(view) != 4 or not np.isfinite(view).all() or min(view[2:]) <= 0:
        raise ValueError("source_gradient_invalid_viewbox")
    source_bytes = Path(source_path).read_bytes()
    with Image.open(source_path) as original:
        if min(original.size) < 32 or max(original.size) > 2048:
            raise ValueError("source_gradient_source_resolution_budget")
        raw = np.asarray(original.convert("RGBA"), dtype=np.float64)
    h, w = raw.shape[:2]
    if np.any(raw[:, :, 3] != 255) or abs((w / h) / (view[2] / view[3]) - 1) > 1e-6:
        raise ValueError("source_gradient_alpha_or_alignment_ambiguity")
    rgb = raw[:, :, :3]
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    if np.abs(border - 255).max() > 1:
        raise ValueError("source_gradient_requires_plain_white_background")
    plane = copy.deepcopy(root)
    _remove_drawables_except(plane, {identifier})
    node = next(item for item in plane.iter() if item.get("id") == identifier)
    node.tag = NS + "rect"
    for key in ("d", "cx", "cy", "r", "rx", "ry"):
        node.attrib.pop(key, None)
    node.attrib.update(x=str(view[0]), y=str(view[1]), width=str(view[2]), height=str(view[3]))
    with tempfile.TemporaryDirectory(prefix="aivc-source-gradient-") as temp:
        folder = Path(temp)
        provenance = {}
        def render(tree, name):
            svg = folder / (name + ".svg")
            ET.ElementTree(tree).write(svg, encoding="utf-8", xml_declaration=True)
            provenance[name] = render_svg_reference(svg, svg.with_suffix(".png"), w, background=None)
            with Image.open(svg.with_suffix(".png")) as image:
                arr = np.asarray(image.convert("RGBA"), dtype=np.float64)
            if arr.shape != raw.shape:
                raise ValueError("source_gradient_render_alignment_failed")
            return arr
        paint_rgba = render(plane, "paint-plane")
        if np.any(paint_rgba[:, :, 3] != 255):
            raise ValueError("source_gradient_nonopaque_paint")
        direction = 255 - paint_rgba[:, :, :3]
        denominator = np.sum(direction ** 2, axis=2)
        if denominator.min() <= 32 ** 2:
            raise ValueError("source_gradient_near_white_coverage_unstable")
        coverage = np.clip(np.sum((255 - rgb) * direction, axis=2) / denominator, 0, 1)
        residual = np.abs(255 - coverage[:, :, None] * direction - rgb).max(axis=2)
        if np.percentile(residual, 99) > 2 or residual.max() > 10:
            raise ValueError("source_gradient_paint_does_not_explain_source")
        mask = coverage >= .5
        if _mask_topology(mask) != (1, 0):
            raise ValueError("source_gradient_multiple_components_or_holes")
        ys, xs = np.nonzero(mask)
        if min(np.ptp(xs), np.ptp(ys)) < 16:
            raise ValueError("source_gradient_source_shape_too_small")
        core = mask.copy()
        for d in (-1, 1):
            core &= np.roll(mask, d, axis=0) & np.roll(mask, d, axis=1)
        if not core.any() or np.any(coverage[core] < .97):
            raise ValueError("source_gradient_interior_opacity_ambiguity")
        points = []
        for axis in (0, 1):
            a = coverage[:-1, :] if axis == 0 else coverage[:, :-1]
            b = coverage[1:, :] if axis == 0 else coverage[:, 1:]
            yy, xx = np.where((a > .5) != (b > .5))
            t = (.5 - a[yy, xx]) / (b[yy, xx] - a[yy, xx])
            points.extend(zip(xx + .5 + (t if axis == 1 else 0), yy + .5 + (t if axis == 0 else 0)))
        if not 32 <= len(points) <= 8192:
            raise ValueError("source_gradient_contour_sample_budget")
        points = np.asarray(points) / [w, h] * view[2:] + view[:2]
        primitive, contour = (_circle(points, budget) if shape == "circle"
                              else _roundrect(points, budget, min(view[2:] / [w, h])))
        candidate = copy.deepcopy(root)
        _remove_drawables_except(candidate, {identifier})
        native = next(item for item in candidate.iter() if item.get("id") == identifier)
        native.tag = NS + primitive["element"]
        for key in ("d", "cx", "cy", "r", "rx", "ry", "x", "y", "width", "height"):
            native.attrib.pop(key, None)
        native.attrib.update({key: format(value, ".12g") for key, value in primitive.items() if key != "element"})
        native.set("data-avc-designer-anchors", "4" if shape == "circle" else "8")
        native.set("data-avc-error-budget-percent", format(budget, ".9g"))
        native.set("data-avc-p95-error-percent", format(contour["bidirectional_p95_percent"], ".9g"))
        native.set("data-avc-max-error-percent", format(contour["bidirectional_maximum_percent"], ".9g"))
        native.set("data-avc-source-reconstruction", schema)
        radius_refinement = None
        if shape == "roundrect":
            # A rasterizer's 50% interpolation is a geometric proposal, not
            # necessarily its best subpixel native radius. A fixed, small
            # source-pixel neighbourhood can remove sampling bias. Every
            # proposal independently retains the unchanged contour budget;
            # all final source and per-residual checks still apply below.
            inferred_radius = primitive["rx"]
            pixel_size = float(min(view[2:] / [w, h]))
            trials = []
            for offset in (-.15, -.10, -.05, 0, .05, .10, .15):
                try:
                    trial_primitive, trial_contour = _roundrect(
                        points, budget, pixel_size, inferred_radius + offset*pixel_size)
                except ValueError:
                    continue
                trial = copy.deepcopy(candidate)
                trial_native = next(item for item in trial.iter() if item.get("id") == identifier)
                trial_native.set("rx", format(trial_primitive["rx"], ".12g"))
                trial_native.set("ry", format(trial_primitive["ry"], ".12g"))
                rgba = render(trial, "radius-proposal-" + str(len(trials)))
                composed = rgba[:, :, :3]*rgba[:, :, 3, None]/255 + 255-rgba[:, :, 3, None]
                trials.append((float(np.abs(composed-rgb).mean()), abs(offset),
                               trial_primitive, trial_contour))
            if not trials:
                raise ValueError("source_gradient_roundrect_no_verified_radius")
            _, _, primitive, contour = min(trials, key=lambda row: row[:2])
            native.set("rx", format(primitive["rx"], ".12g"))
            native.set("ry", format(primitive["ry"], ".12g"))
            native.set("data-avc-p95-error-percent", format(contour["bidirectional_p95_percent"], ".9g"))
            native.set("data-avc-max-error-percent", format(contour["bidirectional_maximum_percent"], ".9g"))
            radius_refinement = {"inferred_radius": inferred_radius, "selected_radius": primitive["rx"],
                                 "source_pixel_size": pixel_size, "maximum_source_pixel_displacement": .15,
                                 "verified_proposals": len(trials), "selection": "minimum_original_rgb_mae"}
        before, after = render(root, "before"), render(candidate, "after")
        if _mask_topology(after[:, :, 3] >= 128) != (1, 0):
            raise ValueError("source_gradient_render_topology_failed")
        union = mask | (before[:, :, 3] >= 128) | (after[:, :, 3] >= 128)
        edge = (coverage > .01) & (coverage < .99)
        if not edge.any():
            raise ValueError("source_gradient_no_subpixel_source_evidence")
        metrics = {}
        pixel_errors = {}
        for name, rgba in (("before", before), ("after", after)):
            image = rgba[:, :, :3] * (rgba[:, :, 3, None] / 255) + 255 * (1 - rgba[:, :, 3, None] / 255)
            error = np.abs(image - rgb)
            pixel_errors[name] = error
            metrics[name] = {"global_rgb_mae": float(error.mean()),
                             "object_rgb_mae": float(error[union].mean()),
                             "edge_rgb_mae": float(error[edge].mean())}
        if (metrics["after"]["global_rgb_mae"] > metrics["before"]["global_rgb_mae"] - max(.01, metrics["before"]["global_rgb_mae"] * .05)
                or metrics["after"]["object_rgb_mae"] >= metrics["before"]["object_rgb_mae"]
                or metrics["after"]["edge_rgb_mae"] >= metrics["before"]["edge_rgb_mae"]):
            raise ValueError("source_gradient_actual_paint_source_fidelity_not_improved")
        removed = []
        edge_band = _dilate((coverage > .01) & (coverage < .99))
        for extra in extras:
            isolated = copy.deepcopy(root)
            _remove_drawables_except(isolated, {extra.get("id")})
            fragment = render(isolated, "residual-" + str(len(removed)))
            support = fragment[:, :, 3] > 0
            count = int(support.sum())
            if not 1 <= count <= 8 or not np.all(edge_band[support]):
                raise ValueError("source_gradient_residual_not_tiny_boundary_aa")
            if np.any(coverage[support] >= .5) or np.any(coverage[support] <= .01):
                raise ValueError("source_gradient_residual_source_intent_ambiguous")
            halo = _dilate(support)
            local_metrics = {name: {"patch_rgb_mae": float(error[support].mean()),
                                    "halo_rgb_mae": float(error[halo].mean())}
                             for name, error in pixel_errors.items()}
            if any(local_metrics["after"][key] > local_metrics["before"][key] + 1e-9
                   for key in ("patch_rgb_mae", "halo_rgb_mae")):
                raise ValueError("source_gradient_residual_local_source_fidelity_worse")
            pixel_regression = float((pixel_errors["after"].mean(axis=2)
                                      - pixel_errors["before"].mean(axis=2))[support].max())
            if pixel_regression > 1e-9:
                raise ValueError("source_gradient_residual_support_pixel_worse")
            yy, xx = np.nonzero(support)
            removed.append({"id": extra.get("id"), "element": "path",
                            "element_sha256": hashlib.sha256(ET.tostring(extra)).hexdigest(),
                            "support_pixels": count, "source_pixel_bbox": [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)],
                            "coverage_range": [float(coverage[support].min()), float(coverage[support].max())],
                            "maximum_support_pixel_rgb_mae_regression": pixel_regression,
                            "measurements": local_metrics, "renderer_provenance": provenance["residual-" + str(len(removed))]})
        if sum(row["support_pixels"] for row in removed) > 32:
            raise ValueError("source_gradient_residual_recovery_budget_exceeded")
    paint_hash = gradient_paint_sha256(root, gradient_id)
    if paint_hash != gradient_paint_sha256(candidate, gradient_id):
        raise ValueError("source_gradient_paint_changed")
    certificate = {"schema": schema, "status": "source_reconstruction_verified",
                   "source_reference_kind": "unmodified_input", "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                   "source_sha256_scope": "unmodified_input_file_bytes",
                   "source_rgba_sha256": hashlib.sha256(raw.astype(np.uint8).tobytes()).hexdigest(),
                   "source_rgba_sha256_scope": "decoded_row_major_rgba8_pixels",
                   "source_size": [w, h], "view_box": view.tolist(), "drawable_id": identifier,
                   "gradient_object_id": object_id, "gradient_id": gradient_id,
                   "paint_sha256": paint_hash, "source_contour": contour,
                   "source_components_and_holes": [1, 0], "render_components_and_holes": [1, 0],
                   "paint_residual_p99": float(np.percentile(residual, 99)), "paint_residual_max": float(residual.max()),
                   "measurements": metrics, "renderer_provenance": provenance,
                   "scope": "original_input_coverage50_to_" + shape + "_not_binary_mask_equivalence",
                   "primitive_kind": shape, "removed_drawables": removed,
                   "before_drawable_count": 1 + len(extras), "after_drawable_count": 1,
                   "final_drawable_ids": [identifier],
                   "presentation_context": _presentation_context(candidate, native),
                   "radius_refinement": radius_refinement,
                   "human_acceptance": "not_performed"}
    if shape == "circle":
        cx, cy, r = (primitive[key] for key in ("cx", "cy", "r"))
        path = f"M{cx-r:.9f} {cy:.9f} A{r:.9f} {r:.9f} 0 1 1 {cx+r:.9f} {cy:.9f} A{r:.9f} {r:.9f} 0 1 1 {cx-r:.9f} {cy:.9f} Z"
    else:
        from native_geometry_contract import roundrect_path
        path = roundrect_path(primitive)
    segments, handles = (2, 4) if shape == "circle" else (8, 8)
    geometry = {"solver": solver, "primitive_first": True, "native_primitives": [primitive],
                "native_whole_object_path": path, "anchor_count": segments, "designer_anchor_count": handles, "segment_count": segments,
                "anchors_before": int(target.get("data-avc-designer-anchors", "4")),
                "topology": {"components": 1, "holes": 0, "expected_loops": 1, "actual_loops": 1,
                             "topology_preserved": True, "scope": "original_source_coverage50"},
                "selection_evidence": {"selected_candidate_id": "source_" + shape, "identity_rollback_selected": False},
                "error_budget": {"metric": "original_source_coverage50_bidirectional_over_source_bbox_diagonal_percent",
                                 "requested_max_percent": budget, "actual_p95_error_percent": contour["bidirectional_p95_percent"],
                                 "actual_max_error_percent": contour["bidirectional_maximum_percent"], "passed": True},
                "source_reconstruction": certificate}
    return ET.tostring(candidate, encoding="unicode"), geometry, certificate


def source_circle_certificate_valid(geometry):
    return (isinstance(geometry, dict) and geometry.get("solver") == SOLVER
            and source_primitive_certificate_valid(geometry))


def source_primitive_certificate_valid(geometry):
    """Validate a source-specific certificate independently of curve economy."""
    from native_geometry_contract import whole_object_native_primitive
    if not isinstance(geometry, dict) or geometry.get("solver") not in {SOLVER, RECT_SOLVER}:
        return False
    cert = geometry.get("source_reconstruction") or {}
    kind = "circle" if geometry["solver"] == SOLVER else "roundrect"
    schema = SCHEMA if kind == "circle" else RECT_SCHEMA
    segments, handles = (2, 4) if kind == "circle" else (8, 8)
    contour = cert.get("source_contour") or {}
    budget = geometry.get("error_budget") or {}
    try:
        finite = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        p95, maximum, limit, share = [contour[key] for key in (
            "bidirectional_p95_percent", "bidirectional_maximum_percent", "error_budget_percent", "over_budget_share")]
        if not all(finite(value) for value in (p95, maximum, limit, share)):
            return False
        measurements = cert["measurements"]
        before, after = measurements["before"], measurements["after"]
        if not all(finite(row[key]) and row[key] >= 0 for row in (before, after)
                   for key in ("global_rgb_mae", "object_rgb_mae", "edge_rgb_mae")):
            return False
        provenance = cert.get("renderer_provenance") or {}
        if not all(isinstance(provenance.get(name), dict)
                   and provenance[name].get("renderer") == "resvg"
                   and provenance[name].get("paint_model") == "native_svg"
                   and provenance[name].get("background") is None
                   and re.fullmatch(r"[0-9a-f]{64}", str(provenance[name].get("source_svg_sha256", "")))
                   and re.fullmatch(r"[0-9a-f]{64}", str(provenance[name].get("png_sha256", "")))
                   for name in ("paint-plane", "before", "after")):
            return False
        removed = cert.get("removed_drawables")
        if not isinstance(removed, list) or len(removed) > (0 if kind == "circle" else 8):
            return False
        if cert.get("before_drawable_count") != 1+len(removed) or cert.get("after_drawable_count") != 1:
            return False
        if cert.get("final_drawable_ids") != [cert.get("drawable_id")]:
            return False
        ids = [row.get("id") for row in removed]
        if len(set(ids)) != len(ids) or cert.get("drawable_id") in ids or any(not value for value in ids):
            return False
        if sum(row.get("support_pixels", 999) for row in removed) > 32:
            return False
        for row in removed:
            count, low_high, bbox = row.get("support_pixels"), row.get("coverage_range"), row.get("source_pixel_bbox")
            if (row.get("element") != "path" or not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 8
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("element_sha256", "")))
                    or len(low_high) != 2 or not all(finite(value) for value in low_high)
                    or not .01 < low_high[0] <= low_high[1] < .5
                    or len(bbox) != 4 or any(not isinstance(value, int) or isinstance(value, bool) for value in bbox)
                    or not 0 <= bbox[0] < bbox[2] <= cert["source_size"][0]
                    or not 0 <= bbox[1] < bbox[3] <= cert["source_size"][1]):
                return False
            local = row["measurements"]
            if not all(finite(local[name][key]) and local[name][key] >= 0
                       for name in ("before", "after") for key in ("patch_rgb_mae", "halo_rgb_mae")):
                return False
            if any(local["after"][key] > local["before"][key] + 1e-9
                   for key in ("patch_rgb_mae", "halo_rgb_mae")):
                return False
            if (not finite(row.get("maximum_support_pixel_rgb_mae_regression"))
                    or row["maximum_support_pixel_rgb_mae_regression"] > 1e-9):
                return False
            record = row.get("renderer_provenance") or {}
            if (record.get("renderer") != "resvg" or record.get("paint_model") != "native_svg"
                    or record.get("background") is not None
                    or any(not re.fullmatch(r"[0-9a-f]{64}", str(record.get(key, "")))
                           for key in ("source_svg_sha256", "png_sha256"))):
                return False
        if kind == "roundrect" and (len(contour.get("flat_edge_samples", [])) != 4
                or min(contour["flat_edge_samples"]) < 4
                or len(contour.get("corner_samples", [])) != 4 or min(contour["corner_samples"]) < 3):
            return False
        if kind == "roundrect":
            refinement = cert.get("radius_refinement") or {}
            if (not all(finite(refinement.get(key)) for key in (
                    "inferred_radius", "selected_radius", "source_pixel_size"))
                    or refinement["source_pixel_size"] <= 0
                    or refinement.get("maximum_source_pixel_displacement") != .15
                    or not 1 <= refinement.get("verified_proposals", 0) <= 7
                    or abs(refinement["selected_radius"]-refinement["inferred_radius"])
                        > .15*refinement["source_pixel_size"] + 1e-9
                    or refinement["selected_radius"] != geometry["native_primitives"][0]["rx"]):
                return False
        return bool(cert.get("schema") == schema and cert.get("status") == "source_reconstruction_verified"
                    and cert.get("source_reference_kind") == "unmodified_input"
                    and cert.get("scope") == "original_input_coverage50_to_" + kind + "_not_binary_mask_equivalence"
                    and cert.get("primitive_kind") == kind
                    and all(re.fullmatch(r"[0-9a-f]{64}", str(cert.get(key, ""))) for key in ("source_sha256", "source_rgba_sha256", "paint_sha256"))
                    and cert.get("source_sha256_scope") == "unmodified_input_file_bytes"
                    and cert.get("source_rgba_sha256_scope") == "decoded_row_major_rgba8_pixels"
                    and all(cert.get(key) for key in ("drawable_id", "gradient_object_id", "gradient_id"))
                    and cert.get("source_components_and_holes") == [1, 0]
                    and cert.get("render_components_and_holes") == [1, 0]
                    and geometry.get("anchor_count") == segments and geometry.get("designer_anchor_count") == handles
                    and geometry.get("segment_count") == segments
                    and finite(cert.get("paint_residual_p99")) and 0 <= cert["paint_residual_p99"] <= 2
                    and finite(cert.get("paint_residual_max")) and 0 <= cert["paint_residual_max"] <= 10
                    and .05 <= limit <= 2 and 0 <= p95 <= limit and 0 <= maximum <= 3 * limit and 0 <= share <= .05
                    and contour.get("salient_corner_count") == 0
                    and budget.get("passed") is True and budget.get("requested_max_percent") == limit
                    and budget.get("actual_p95_error_percent") == p95 and budget.get("actual_max_error_percent") == maximum
                    and after["global_rgb_mae"] <= before["global_rgb_mae"] - max(.01, before["global_rgb_mae"] * .05)
                    and after["object_rgb_mae"] < before["object_rgb_mae"] and after["edge_rgb_mae"] < before["edge_rgb_mae"]
                    and whole_object_native_primitive(geometry) is not None)
    except (KeyError, TypeError, ValueError, OverflowError, IndexError):
        return False


def final_source_primitive_matches(root, geometry, source_path=None):
    """Bind source proof to the complete final drawable set and delivered pixels.

    Source PNG compression/metadata may differ; decoded RGBA bytes may not.
    Final native geometry is additionally checked by the caller's native proof.
    """
    if not source_primitive_certificate_valid(geometry):
        return False
    cert = geometry["source_reconstruction"]
    try:
        target, gradient_id, extras = _context(root)
        if (extras or target.get("id") != cert["drawable_id"]
                or target.get("data-avc-gradient-object") != cert["gradient_object_id"]
                or gradient_id != cert["gradient_id"]
                or gradient_paint_sha256(root, gradient_id) != cert["paint_sha256"]):
            return False
        view = [float(value) for value in re.split(r"[\s,]+", root.get("viewBox", "").strip())]
        if view != cert["view_box"] or _presentation_context(root, target) != cert.get("presentation_context"):
            return False
        from native_geometry_contract import native_geometry_matches, whole_object_native_primitive
        actual = {"element": _local(target), **target.attrib}
        if not native_geometry_matches(whole_object_native_primitive(geometry), actual):
            return False
        if source_path is not None:
            with Image.open(source_path) as image:
                if list(image.size) != cert["source_size"]:
                    return False
                raw = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if np.any(raw[:, :, 3] != 255) or hashlib.sha256(raw.tobytes()).hexdigest() != cert["source_rgba_sha256"]:
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return False
