"""Propose a solid paint only when native source evidence beats a gradient.

The working-resolution gradient fitter can faithfully fit resampling ringing.
This bounded second opinion uses the original pixels, a stable interior, and
separate interior/edge holdouts. It never changes geometry, stacking, or alpha.
Proposals are NOT approved scene changes: the caller must apply its complete
source-scene, geometry and paint guards and update final report provenance.
"""
from __future__ import annotations

import copy
import hashlib
import io
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

from gradient_object_engine import _sample_and_split, _srgb_to_oklab
from source_scene_guard import MAX_SCENE_PIXELS, MAX_SCENE_SIDE
from svg_renderer import _validated_svg, native_canvas_aspect_compatible

SCHEMA = "aivc.source-flat-paint-proposal/v1"
_GRAPHICS = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "use", "image", "text"}
_RESOURCE_CONTAINERS = {"defs", "clipPath", "mask", "pattern", "marker", "symbol"}
_REF = re.compile(r"^url\(\s*#([^\s)]+)\s*\)$")


def _tag(node):
    return node.tag.rsplit("}", 1)[-1]


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _rgba_render(root, width, height):
    import resvg_py
    from svg_renderer import svg_with_native_viewport
    svg = svg_with_native_viewport(ET.tostring(root, encoding='unicode'), width, height)
    png = resvg_py.svg_to_bytes(svg_string=svg,
                                width=width, height=height, background=None,
                                skip_system_fonts=True, log_information=False,
                                shape_rendering="geometric_precision")
    with Image.open(io.BytesIO(png)) as image:
        if image.mode != "RGBA" or image.size != (width, height):
            raise ValueError("native_rgba_renderer_required")
        return np.asarray(image).copy()


def _isolate(root, drawable_id):
    result = copy.deepcopy(root)

    def visit(node, in_resource=False):
        in_resource = in_resource or _tag(node) in _RESOURCE_CONTAINERS
        for child in list(node):
            if not in_resource and _tag(child) in _GRAPHICS and child.get("id") != drawable_id:
                node.remove(child)
            else:
                visit(child, in_resource)
    visit(result)
    return result


def _composite(rgba):
    alpha = rgba[:, :, 3:4].astype(np.float64) / 255
    return rgba[:, :, :3].astype(np.float64) * alpha + 255 * (1 - alpha)


def _erode(mask, radius):
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('source_erode_requires_nonnegative_integer_radius')
    if radius == 0:
        return np.asarray(mask, dtype=bool).copy()
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(
        ImageFilter.MinFilter(2 * radius + 1))) > 0


def _samples(mask, seed=912):
    ys, xs = np.nonzero(mask)
    order, training, heldout = _sample_and_split(ys, xs, 8192, .2, seed)
    return ys[order], xs[order], training, heldout


def _errors(actual, predicted):
    values = np.linalg.norm(_srgb_to_oklab(actual) - _srgb_to_oklab(predicted), axis=1) * 100
    return {"mean": float(values.mean()), "p90": float(np.percentile(values, 90)),
            "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def _holdout(mask, original, baseline, candidate, *, seed=912):
    ys, xs, training, heldout = _samples(mask, seed)
    y, x = ys[heldout], xs[heldout]
    before, after = _errors(original[y, x], baseline[y, x]), _errors(original[y, x], candidate[y, x])
    return {"pixels": int(mask.sum()), "training_samples": int(len(training)),
            "heldout_samples": int(len(heldout)), "gradient_error": before, "solid_error": after,
            "mean_improvement": before["mean"] - after["mean"],
            "mean_p90_p99_nonregression": all(after[key] <= before[key] + 1e-6 for key in ("mean", "p90", "p99"))}


def _replace(node, color, gradient_id):
    node.set("fill", color)
    # The path remains geometrically identical, but a gradient ownership/economy
    # certificate must not accidentally continue to certify a different paint.
    for key in list(node.attrib):
        if key.startswith("data-avc-"):
            del node.attrib[key]
    node.set("data-avc-source-flat-paint", "native-core-and-edge-heldout-v1")
    node.set("data-avc-source-flat-from", gradient_id)


def propose_source_flat_paint(svg_text, source_rgba, *, max_candidates=16):
    """Return independent, uncommitted per-path paint alternatives.

    Every proposal is based on the input SVG fingerprint. To apply several,
    rebase each replacement element after checking its original element hash;
    never overwrite a previously committed scene with another proposal's full
    ``svg_text``. Unused/shared gradient definitions are intentionally retained.
    """
    result = {"schema": SCHEMA, "status": "skipped", "proposals": [], "decisions": [],
              "requires_whole_scene_source_guard": True,
              "preserves": ["geometry", "stacking", "rendered_alpha", "shared_gradient_definitions"]}
    try:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 16:
            raise ValueError("source_flat_candidate_budget_must_be_1_to_16")
        original = np.asarray(source_rgba)
        if original.ndim != 3 or original.shape[2] != 4 or original.dtype != np.uint8:
            raise ValueError("source_flat_requires_native_uint8_rgba")
        height, width = original.shape[:2]
        if not (0 < width <= MAX_SCENE_SIDE and 0 < height <= MAX_SCENE_SIDE and width * height <= MAX_SCENE_PIXELS):
            raise ValueError("source_flat_native_pixel_budget_exceeded")
        payload = svg_text.encode("utf-8")
        _, svg_width, svg_height = _validated_svg(payload)
        if not native_canvas_aspect_compatible(svg_width, svg_height, width, height):
            raise ValueError("source_flat_svg_aspect_ratio_differs")
        root = ET.fromstring(payload)
        result["before_svg_sha256"] = _sha(payload)
        result["source_rgba_sha256"] = _sha(original.tobytes())
        result["native_size"] = [width, height]
        ids = [n.get("id") for n in root.iter() if n.get("id")]
        if len(ids) != len(set(ids)):
            raise ValueError("source_flat_duplicate_svg_id")
        gradients = {n.get("id") for n in root.iter() if _tag(n) in {"linearGradient", "radialGradient"}}
        parent = {child: node for node in root.iter() for child in node}
        candidates = []
        for node in root.iter():
            match = _REF.fullmatch(node.get("fill", ""))
            if _tag(node) != "path" or not node.get("id") or not match or match[1] not in gradients:
                continue
            ancestors, at = [], node
            while at is not None:
                ancestors.append(at)
                at = parent.get(at)
            if any(_tag(n) in _RESOURCE_CONTAINERS for n in ancestors):
                continue
            if any(n.get("data-avc-gradient-object") for n in ancestors[1:]):
                result["decisions"].append({"drawable_id": node.get("id"), "status": "retained",
                                            "reason": "inherited_gradient_owner_requires_manual_projection"})
                continue
            # CSS and a stroke can make an isolated fill's source ownership
            # ambiguous. Preserve these cases instead of guessing precedence.
            if any(n.get("style") or n.get("class") or n.get("stroke", "none") != "none" for n in ancestors):
                result["decisions"].append({"drawable_id": node.get("id"), "status": "retained",
                                            "reason": "css_or_stroke_ownership_not_supported"})
                continue
            candidates.append((node, match[1]))
        if any(_tag(n) == "style" for n in root.iter()):
            raise ValueError("source_flat_stylesheet_ownership_not_supported")
        result["candidate_count"] = len(candidates)
        result["candidate_budget"] = max_candidates
        if not candidates:
            return result
        full_scene = _rgba_render(root, width, height)
        source_rgb = _composite(original)
        for node, gradient_id in candidates[:max_candidates]:
            drawable_id = node.get("id")
            decision = {"drawable_id": drawable_id, "gradient_id": gradient_id, "status": "retained"}
            result["decisions"].append(decision)
            isolated = _isolate(root, drawable_id)
            baseline = _rgba_render(isolated, width, height)
            alpha = baseline[:, :, 3]
            # Restrict evidence to visible, opaque ownership. Hidden pixels or
            # another layer's source colour cannot certify this path's paint.
            visible = np.max(np.abs(full_scene.astype(np.int16) - baseline.astype(np.int16)), axis=2) <= 1
            support = (alpha >= 240) & (original[:, :, 3] >= 240)
            core = _erode(support, 4) & visible
            edge = support & ~_erode(support, 4) & visible
            # Absolute independent evidence matters for a long thin object:
            # four-pixel erosion can leave hundreds of stable core samples
            # while their area is less than one fifth of the whole strip.
            if int(core.sum()) < 64 or int(edge.sum()) < 64:
                decision["reason"] = "insufficient_native_core_or_edge_preserve_gradient"
                continue
            ys, xs, training, heldout = _samples(core)
            colours = original[ys, xs, :3]
            solid = np.rint(np.median(colours[training], axis=0)).astype(np.uint8)
            delta = np.max(np.abs(colours.astype(np.int16) - solid.astype(np.int16)), axis=1)
            exact_share = float(np.mean(delta == 0))
            p99 = float(np.percentile(delta, 99))
            source_core = {"inset_native_pixels": 4, "pixels": int(core.sum()),
                           "minimum_visible_core_pixels": 64,
                           "support_fraction_diagnostic_only": float(core.sum() / max(1, support.sum())),
                           "median_rgb": solid.tolist(), "exact_median_share": exact_share,
                           "p99_channel_distance_from_median": p99,
                           "rgb_standard_deviation": np.std(colours, axis=0).tolist()}
            decision["source_core"] = source_core
            if exact_share < .98 or p99 > 1:
                decision["reason"] = "native_core_has_colour_variation_preserve_gradient"
                continue
            color = "#" + "".join(f"{int(value):02x}" for value in solid)
            flat_isolated = copy.deepcopy(isolated)
            flat_node = next(n for n in flat_isolated.iter() if n.get("id") == drawable_id)
            _replace(flat_node, color, gradient_id)
            candidate = _rgba_render(flat_isolated, width, height)
            if not np.array_equal(alpha, candidate[:, :, 3]):
                decision["reason"] = "solid_would_change_gradient_alpha_preserve_gradient"
                continue
            before_rgb, after_rgb = _composite(baseline), _composite(candidate)
            core_cost = _holdout(core, source_rgb, before_rgb, after_rgb)
            edge_cost = _holdout(edge, source_rgb, before_rgb, after_rgb)
            evidence = {"source_core": source_core, "core_holdout": core_cost, "edge_holdout": edge_cost,
                        "metric": "oklab_delta_e_100", "native_resolution": True,
                        "strategy": "native_coordinate_hash_20_percent_core_and_edge_separate",
                        "validation_seed": 912, "solid_colour_fit": "native_core_training_median",
                        "paint_selection_policy": "constant_native_core_nonregression_and_strict_edge_improvement",
                        "rendered_alpha_identical": True, "geometry_and_stack_unchanged": True,
                        "whole_scene_acceptance": "pending_caller_source_scene_guard"}
            decision["evidence"] = evidence
            if (not core_cost["mean_p90_p99_nonregression"] or not edge_cost["mean_p90_p99_nonregression"]
                    or edge_cost["mean_improvement"] <= .05):
                decision["reason"] = "native_core_and_edge_do_not_both_favour_solid"
                continue
            proposal_root = copy.deepcopy(root)
            replacement = next(n for n in proposal_root.iter() if n.get("id") == drawable_id)
            _replace(replacement, color, gradient_id)
            y, x = np.nonzero(alpha > 0)
            proposal = {"drawable_id": drawable_id, "gradient_id": gradient_id,
                        "operation": "replace_gradient_with_source_solid", "replacement_fill": color,
                        "before_svg_sha256": result["before_svg_sha256"],
                        "element_sha256": _sha(ET.tostring(node, encoding="utf-8")),
                        "original_element_xml": ET.tostring(node, encoding="unicode"),
                        "replacement_element_xml": ET.tostring(replacement, encoding="unicode"),
                        "svg_text": ET.tostring(proposal_root, encoding="unicode"),
                        "roi_xyxy": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                        "evidence": evidence, "status": "proposed_pending_scene_guard"}
            result["proposals"].append(proposal)
            decision.update(status="proposed", reason="native_source_core_and_edge_favour_solid")
        for node, gradient_id in candidates[max_candidates:]:
            result["decisions"].append({"drawable_id": node.get("id"), "gradient_id": gradient_id,
                                        "status": "retained", "reason": "candidate_budget_exhausted"})
        result["status"] = "proposed" if result["proposals"] else "skipped"
        return result
    except (ValueError, TypeError, OSError, ImportError, RuntimeError, MemoryError) as exc:
        result["status"] = "unverified_fail_closed"
        result["reason"] = str(exc) or type(exc).__name__
        result["proposals"] = []
        return result
