"""Conservative phase-one proposals for completely missing components.

This module never opens or modifies an SVG.  It reconstructs the exact
strong-ink source labels used by the topology diagnostic and, only for small
isolated opaque single-colour components, returns an append-on-top SVG
fragment for a later transactional validator to consider.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from quality_diagnostics import source_ink_roi, structural_core_threshold
from stroke_engine import connected_components
from trace_engine import binary_mask_to_compound_path


SCHEMA = "ai-vector-cleanroom.component-repair-proposal/v1"
TRANSACTION_SCHEMA = "ai-vector-cleanroom.component-repair-transaction/v1"
SVG_NS = "http://www.w3.org/2000/svg"


def _load_rgba(value, name):
    if isinstance(value, Image.Image):
        return value.convert("RGBA").copy()
    if isinstance(value, (str, Path)):
        try:
            with Image.open(value) as opened:
                return opened.convert("RGBA")
        except Exception as exc:
            raise ValueError(f"{name} could not be opened: {exc}") from exc
    raise TypeError(f"{name} must be a Pillow image or filesystem path")


def _composite_white(image):
    base = Image.new("RGB", image.size, (255, 255, 255))
    base.paste(image, (0, 0), image)
    return np.asarray(base, dtype=np.int16)


def _finite_number(name, value, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _positive_int(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least 1")
    return result


def _dilate(mask, radius):
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(result)
        height, width = result.shape
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy:dy + height, dx:dx + width]
        result = grown
    return result


def _bbox(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return None
    return [int(xx.min()), int(yy.min()),
            int(xx.max() - xx.min() + 1),
            int(yy.max() - yy.min() + 1)]


def _normalise_viewbox(viewbox, width, height):
    try:
        values = list(viewbox)
    except TypeError:
        return None
    if len(values) == 2:
        values = [0.0, 0.0, values[0], values[1]]
    if len(values) != 4:
        return None
    normalised = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        normalised.append(value)
    x, y, vb_width, vb_height = normalised
    if (abs(x) > 1e-9 or abs(y) > 1e-9
            or abs(vb_width - width) > 1e-9
            or abs(vb_height - height) > 1e-9):
        return None
    return normalised


def _image_sha256(image):
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _skip_result(audit, reason):
    audit["status"] = "skipped"
    audit["skipped_reason"] = reason
    counts = Counter(
        record.get("reason") for record in audit["records"]
        if record.get("status") == "skipped" and record.get("reason"))
    audit["skipped_reason_counts"] = dict(sorted(counts.items()))
    return {
        "status": "skipped",
        "svg_fragment": "",
        "repair_count": 0,
        "path_count": 0,
        "node_count": 0,
        "bbox": None,
        "repairs": [],
        "audit": audit,
    }


def propose_missing_component_repairs(
        source_reference, render, flat, failed_examples, *, viewbox,
        max_components=4, max_component_area=4096, max_total_area=8192,
        max_component_nodes=512, max_total_nodes=1024,
        edge_margin=3, moat_radius=3, dominant_color_min=0.90,
        opaque_alpha_min=250, max_failed_score=5.0,
        simplify=0.35, min_area=1.0, smooth=0.0, curve=0.0):
    """Return a deterministic append-only repair proposal and full audit.

    Only topology examples that describe a completely missing component are
    considered.  Ambiguous, fragmented, overlapping, translucent, multicolour
    or coordinate-scaled cases are reported as skipped.  The returned SVG
    fragment is inert data; this function performs no filesystem writes and
    never edits a live SVG.
    """

    max_components = _positive_int("max_components", max_components)
    max_component_area = _positive_int(
        "max_component_area", max_component_area)
    max_total_area = _positive_int("max_total_area", max_total_area)
    max_component_nodes = _positive_int(
        "max_component_nodes", max_component_nodes)
    max_total_nodes = _positive_int("max_total_nodes", max_total_nodes)
    edge_margin = _positive_int("edge_margin", edge_margin)
    moat_radius = _positive_int("moat_radius", moat_radius)
    opaque_alpha_min = _positive_int("opaque_alpha_min", opaque_alpha_min)
    if opaque_alpha_min > 255:
        raise ValueError("opaque_alpha_min must be at most 255")
    dominant_color_min = _finite_number(
        "dominant_color_min", dominant_color_min, minimum=0.0, maximum=1.0)
    max_failed_score = _finite_number(
        "max_failed_score", max_failed_score, minimum=0.0, maximum=100.0)
    simplify = _finite_number("simplify", simplify, minimum=0.0)
    min_area = _finite_number("min_area", min_area, minimum=0.0)
    smooth = _finite_number("smooth", smooth, minimum=0.0)
    curve = _finite_number("curve", curve, minimum=0.0, maximum=1.0)
    if isinstance(failed_examples, (str, bytes)):
        raise TypeError("failed_examples must be a sequence of dictionaries")
    try:
        examples = list(failed_examples)
    except TypeError as exc:
        raise TypeError(
            "failed_examples must be a sequence of dictionaries") from exc

    source_image = _load_rgba(source_reference, "source_reference")
    render_image = _load_rgba(render, "render")
    flat_image = _load_rgba(flat, "flat")
    source_original_size = source_image.size
    width, height = render_image.size
    audit = {
        "schema": SCHEMA,
        "status": "evaluating",
        "policy": "missing_isolated_opaque_single_colour_append_only",
        "source_original_size": list(source_original_size),
        "measurement_size": [width, height],
        "render_size": list(render_image.size),
        "flat_size": list(flat_image.size),
        "viewbox": list(viewbox) if isinstance(viewbox, (list, tuple)) else None,
        "input_sha256": {
            "source_reference": _image_sha256(source_image),
            "render": _image_sha256(render_image),
            "flat": _image_sha256(flat_image),
        },
        "limits": {
            "max_components": max_components,
            "max_component_area": max_component_area,
            "max_total_area": max_total_area,
            "max_component_nodes": max_component_nodes,
            "max_total_nodes": max_total_nodes,
            "edge_margin": edge_margin,
            "moat_radius": moat_radius,
            "dominant_color_min": dominant_color_min,
            "opaque_alpha_min": opaque_alpha_min,
            "max_failed_score": max_failed_score,
        },
        "trace_parameters": {
            "simplify": simplify,
            "min_area": min_area,
            "smooth": smooth,
            "curve": curve,
        },
        "failed_examples_received": len(examples),
        "records": [],
    }

    if render_image.size != flat_image.size:
        return _skip_result(audit, "render_flat_size_mismatch")
    if source_image.size != render_image.size:
        source_ratio = source_image.width / max(1.0, float(source_image.height))
        render_ratio = render_image.width / max(1.0, float(render_image.height))
        relative_aspect_error = abs(source_ratio - render_ratio) / max(
            1e-9, source_ratio)
        audit["source_render_aspect_error"] = round(
            relative_aspect_error, 9)
        if relative_aspect_error > 0.002:
            return _skip_result(audit, "source_render_aspect_mismatch")
        # This is the exact normalisation used by
        # quality_diagnostics.compute_quality_diagnostics before it assigns
        # source_component labels.  Repeating it here preserves label/area/
        # bbox identity while allowing high-resolution source references to
        # repair a 2048px validation/render candidate safely.
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        source_image = source_image.resize(render_image.size, resampling)
        audit["source_normalisation"] = {
            "status": "resampled",
            "method": "Pillow.LANCZOS",
            "size": [width, height],
        }
    else:
        audit["source_normalisation"] = {
            "status": "not_needed",
            "method": None,
            "size": [width, height],
        }
    audit["input_sha256"]["source_measurement"] = _image_sha256(source_image)
    normalised_viewbox = _normalise_viewbox(viewbox, width, height)
    if normalised_viewbox is None:
        return _skip_result(audit, "viewbox_not_one_to_one")
    audit["viewbox"] = normalised_viewbox

    roi = source_ink_roi(source_image)
    source_mask = np.asarray(roi["mask"], dtype=bool)
    source_strength = np.asarray(roi["strength"], dtype=np.float32)
    threshold = float(roi["ink_threshold"])
    core_threshold = structural_core_threshold(threshold)
    core_source = source_mask & (source_strength >= core_threshold)
    source_labels, source_count = connected_components(core_source)
    source_ink_pixels = int(source_mask.sum())
    core_source_ink_pixels = int(core_source.sum())
    topology_min_area = max(
        16, min(128, int(round(core_source_ink_pixels * 1e-4))))

    render_rgb = _composite_white(render_image)
    background = np.asarray(roi["background_rgb"], dtype=np.float32)
    render_strength = np.abs(
        render_rgb.astype(np.float32) - background).max(axis=2)
    core_render = render_strength >= core_threshold
    flat_rgba = np.asarray(flat_image, dtype=np.uint8)

    audit.update({
        "source_component_count": int(source_count),
        "source_ink_pixels": source_ink_pixels,
        "core_source_ink_pixels": core_source_ink_pixels,
        "topology_minimum_component_area": topology_min_area,
        "ink_threshold": round(threshold, 6),
        "core_threshold": round(core_threshold, 6),
        "background_rgb": [round(float(value), 6) for value in background],
    })

    preliminary = []
    seen_labels = set()
    for index, example in enumerate(examples):
        record = {"input_index": index, "status": "skipped"}
        audit["records"].append(record)
        if not isinstance(example, dict):
            record["reason"] = "invalid_example"
            continue
        label = example.get("source_component")
        if isinstance(label, (bool, np.bool_)) or not isinstance(
                label, (int, np.integer)) or int(label) <= 0:
            record["reason"] = "invalid_source_component"
            continue
        label = int(label)
        record["source_component"] = label
        if label in seen_labels:
            record["reason"] = "duplicate_source_component"
            continue
        seen_labels.add(label)
        try:
            score = _finite_number(
                "score_percent", example.get("score_percent"),
                minimum=0.0, maximum=100.0)
            coverage = _finite_number(
                "coverage_percent", example.get("coverage_percent"),
                minimum=0.0, maximum=100.0)
        except (TypeError, ValueError):
            record["reason"] = "invalid_failure_scores"
            continue
        fragment_count = example.get("fragment_count")
        if isinstance(fragment_count, (bool, np.bool_)) or not isinstance(
                fragment_count, (int, np.integer)) or int(fragment_count) < 0:
            record["reason"] = "invalid_fragment_count"
            continue
        record.update({
            "reported_score_percent": score,
            "reported_coverage_percent": coverage,
            "reported_fragment_count": int(fragment_count),
        })
        if score > max_failed_score:
            record["reason"] = "score_above_missing_limit"
            continue
        if coverage > max_failed_score:
            record["reason"] = "coverage_above_missing_limit"
            continue
        if int(fragment_count) != 0:
            record["reason"] = "fragmented_component"
            continue
        component = source_labels == label
        actual_area = int(component.sum())
        actual_bbox = _bbox(component)
        record.update({"area_px": actual_area, "bbox_px": actual_bbox})
        if actual_area == 0:
            record["reason"] = "source_component_not_found"
            continue
        if actual_area < topology_min_area:
            record["reason"] = "below_topology_area"
            continue
        reported_area = example.get("area_px")
        reported_bbox = example.get("bbox_px")
        if reported_area != actual_area:
            record["reason"] = "reported_area_mismatch"
            continue
        if reported_bbox != actual_bbox:
            record["reason"] = "reported_bbox_mismatch"
            continue
        if actual_area > max_component_area:
            record["reason"] = "component_area_limit"
            continue
        record["status"] = "eligible"
        record.pop("reason", None)
        preliminary.append((label, component, record))

    if len(preliminary) > max_components:
        for _label, _component, record in preliminary:
            record["status"] = "skipped"
            record["reason"] = "component_count_limit"
        return _skip_result(audit, "component_count_limit")
    total_area = sum(record["area_px"] for _, _, record in preliminary)
    audit["eligible_total_area"] = int(total_area)
    if total_area > max_total_area:
        for _label, _component, record in preliminary:
            record["status"] = "skipped"
            record["reason"] = "total_area_limit"
        return _skip_result(audit, "total_area_limit")

    proposals = []
    for label, component, record in preliminary:
        x, y, box_width, box_height = record["bbox_px"]
        if (x < edge_margin or y < edge_margin
                or x + box_width > width - edge_margin
                or y + box_height > height - edge_margin):
            record["status"] = "skipped"
            record["reason"] = "component_touches_edge_guard"
            continue
        neighbourhood = _dilate(component, moat_radius)
        source_moat_pixels = int(np.count_nonzero(
            neighbourhood & core_source & ~component))
        render_moat_pixels = int(np.count_nonzero(
            neighbourhood & core_render))
        record.update({
            "source_moat_pixels": source_moat_pixels,
            "render_moat_pixels": render_moat_pixels,
        })
        if source_moat_pixels:
            record["status"] = "skipped"
            record["reason"] = "source_moat_not_clear"
            continue
        if render_moat_pixels:
            record["status"] = "skipped"
            record["reason"] = "render_moat_not_clear"
            continue

        component_rgba = flat_rgba[component]
        alpha_min = int(component_rgba[:, 3].min())
        record["flat_alpha_min"] = alpha_min
        if alpha_min < opaque_alpha_min:
            record["status"] = "skipped"
            record["reason"] = "flat_component_not_opaque"
            continue
        colours, counts = np.unique(
            component_rgba[:, :3], axis=0, return_counts=True)
        dominant_index = int(np.argmax(counts))
        dominant = tuple(int(value) for value in colours[dominant_index])
        dominant_share = float(counts[dominant_index] / component_rgba.shape[0])
        dominant_contrast = float(np.abs(
            np.asarray(dominant, dtype=np.float32) - background).max())
        fill = "#{:02x}{:02x}{:02x}".format(*dominant)
        record.update({
            "flat_dominant_fill": fill,
            "flat_dominant_share": round(dominant_share, 6),
            "flat_dominant_contrast": round(dominant_contrast, 6),
        })
        if dominant_share < dominant_color_min:
            record["status"] = "skipped"
            record["reason"] = "flat_component_not_single_colour"
            continue
        if dominant_contrast < core_threshold:
            record["status"] = "skipped"
            record["reason"] = "flat_component_not_high_contrast"
            continue

        traced = binary_mask_to_compound_path(
            component, simplify=simplify, min_area=min_area,
            smooth=smooth, curve=curve)
        if not traced["path"] or traced["loop_count"] < 1:
            record["status"] = "skipped"
            record["reason"] = "local_trace_empty"
            continue
        if traced["node_count"] > max_component_nodes:
            record["status"] = "skipped"
            record["reason"] = "component_node_limit"
            continue
        path_id = f"component-repair-{label}"
        mask_digest = hashlib.sha256(
            np.packbits(component.reshape(-1), bitorder="little").tobytes()
        ).hexdigest()
        repair = {
            "source_component": label,
            "id": path_id,
            "fill": fill,
            "fill_rule": traced["fill_rule"],
            "path": traced["path"],
            "node_count": traced["node_count"],
            "loop_count": traced["loop_count"],
            "bbox": traced["bbox"],
            "area_px": record["area_px"],
            "mask_sha256": mask_digest,
        }
        record.update({
            "status": "proposed",
            "path_id": path_id,
            "path_bbox": traced["bbox"],
            "path_nodes": traced["node_count"],
            "path_loops": traced["loop_count"],
            "mask_sha256": mask_digest,
        })
        proposals.append(repair)

    proposals.sort(key=lambda item: item["source_component"])
    if not proposals:
        return _skip_result(audit, "no_safe_components")
    total_nodes = sum(item["node_count"] for item in proposals)
    if total_nodes > max_total_nodes:
        proposed_labels = {
            item["source_component"] for item in proposals
        }
        for record in audit["records"]:
            if (record.get("source_component") in proposed_labels
                    and record.get("status") == "proposed"):
                record["status"] = "skipped"
                record["reason"] = "total_node_limit"
        return _skip_result(audit, "total_node_limit")

    lines = [
        '<g id="component-repairs" '
        'data-avc-stage="isolated-missing-component-repair" '
        'fill-rule="evenodd">'
    ]
    for repair in proposals:
        lines.append(
            f'  <path id="{repair["id"]}" '
            f'data-source-component="{repair["source_component"]}" '
            f'fill="{repair["fill"]}" d="{repair["path"]}"/>')
    lines.append("</g>")
    fragment = "\n".join(lines)
    boxes = [repair["bbox"] for repair in proposals]
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    y1 = max(box[1] + box[3] for box in boxes)
    proposal_bbox = [x0, y0, x1 - x0, y1 - y0]
    audit.update({
        "status": "proposed",
        "skipped_reason": None,
        "repair_count": len(proposals),
        "proposal_bbox": proposal_bbox,
        "proposal_sha256": hashlib.sha256(
            fragment.encode("utf-8")).hexdigest(),
    })
    counts = Counter(
        record.get("reason") for record in audit["records"]
        if record.get("status") == "skipped" and record.get("reason"))
    audit["skipped_reason_counts"] = dict(sorted(counts.items()))
    return {
        "status": "proposed",
        "svg_fragment": fragment,
        "repair_count": len(proposals),
        "path_count": len(proposals),
        "node_count": total_nodes,
        "bbox": proposal_bbox,
        "repairs": proposals,
        "audit": audit,
    }


def append_repair_fragment(svg_bytes, svg_fragment):
    """Return proposal bytes with one validated repair group appended.

    This function is deliberately pure: it neither accepts a path nor writes
    the live SVG.  The caller must still render and validate the returned bytes
    before committing them atomically.
    """

    if not isinstance(svg_bytes, (bytes, bytearray)):
        raise TypeError("svg_bytes must be bytes")
    if not isinstance(svg_fragment, str) or not svg_fragment.strip():
        raise TypeError("svg_fragment must be a non-empty string")
    raw = bytes(svg_bytes)
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    try:
        text = raw[len(bom):].decode("utf-8")
        root = ET.fromstring(text)
    except Exception as exc:
        raise ValueError(f"base SVG is not valid UTF-8 XML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError("base document root is not svg")
    if any(element.get("id") == "component-repairs" for element in root.iter()):
        raise ValueError("base SVG already contains component-repairs")

    try:
        wrapper = ET.fromstring(
            f'<svg xmlns="{SVG_NS}">{svg_fragment}</svg>')
    except Exception as exc:
        raise ValueError(f"repair fragment is not valid XML: {exc}") from exc
    children = list(wrapper)
    if (len(children) != 1 or children[0].tag.rsplit("}", 1)[-1] != "g"
            or children[0].get("id") != "component-repairs"):
        raise ValueError("repair fragment must contain one component-repairs group")
    allowed_group_attributes = {"id", "data-avc-stage", "fill-rule"}
    allowed_path_attributes = {"id", "data-source-component", "fill", "d"}
    path_ids = set()
    for element in children[0].iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "g":
            if set(element.attrib) - allowed_group_attributes:
                raise ValueError("repair group contains unsupported attributes")
            continue
        if tag != "path" or set(element.attrib) - allowed_path_attributes:
            raise ValueError("repair fragment may contain only safe path elements")
        path_id = element.get("id", "")
        fill = element.get("fill", "")
        path_data = element.get("d", "")
        if (not path_id or path_id in path_ids
                or not re.fullmatch(r"#[0-9a-fA-F]{6}", fill)
                or not path_data.strip()):
            raise ValueError("repair path has invalid id, fill or geometry")
        path_ids.add(path_id)

    matches = list(re.finditer(r"</(?:[A-Za-z_][\w.-]*:)?svg\s*>", text,
                               flags=re.IGNORECASE))
    if len(matches) != 1 or text[matches[0].end():].strip():
        raise ValueError("base SVG must have one final closing svg tag")
    match = matches[0]
    newline = "\r\n" if "\r\n" in text else "\n"
    prefix = text[:match.start()]
    if not prefix.endswith(("\n", "\r")):
        prefix += newline
    proposal_text = prefix + svg_fragment.strip() + newline + text[match.start():]
    return bom + proposal_text.encode("utf-8")


def _score_value(scores, path):
    value = scores
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _failure_labels(scores):
    topology = ((scores.get("detail_grid") or {}).get(
        "component_topology") or {}) if isinstance(scores, dict) else {}
    examples = topology.get("failed_examples")
    if not isinstance(examples, list):
        return None
    labels = set()
    for example in examples:
        if isinstance(example, dict):
            label = example.get("source_component")
            if isinstance(label, (int, np.integer)) and not isinstance(
                    label, (bool, np.bool_)):
                labels.add(int(label))
    return labels


def validate_repair_transaction(
        proposal, before_scores, after_scores, before_gate, after_gate,
        before_render, after_render, *, bbox_padding=2):
    """Validate a rendered repair proposal and return an auditable verdict."""

    bbox_padding = _positive_int("bbox_padding", bbox_padding)
    audit = {
        "schema": TRANSACTION_SCHEMA,
        "status": "rejected",
        "policy": "render_locality_and_non_regression_then_atomic_commit",
        "bbox_padding": bbox_padding,
        "reasons": [],
    }
    if not isinstance(proposal, dict) or proposal.get("status") != "proposed":
        audit["reasons"].append("proposal_not_ready")
        return audit
    repairs = proposal.get("repairs")
    if not isinstance(repairs, list) or not repairs:
        audit["reasons"].append("proposal_has_no_repairs")
        return audit

    before_image = _load_rgba(before_render, "before_render")
    after_image = _load_rgba(after_render, "after_render")
    if before_image.size != after_image.size:
        audit["reasons"].append("render_size_mismatch")
        return audit
    width, height = before_image.size
    before_array = np.asarray(before_image, dtype=np.int16)
    after_array = np.asarray(after_image, dtype=np.int16)
    changed = np.any(before_array != after_array, axis=2)
    allowed = np.zeros((height, width), dtype=bool)
    target_labels = set()
    for repair in repairs:
        if not isinstance(repair, dict):
            audit["reasons"].append("invalid_repair_record")
            continue
        label = repair.get("source_component")
        box = repair.get("bbox")
        if (not isinstance(label, (int, np.integer))
                or isinstance(label, (bool, np.bool_))
                or not isinstance(box, (list, tuple)) or len(box) != 4):
            audit["reasons"].append("invalid_repair_locator")
            continue
        try:
            x, y, box_width, box_height = (float(value) for value in box)
        except (TypeError, ValueError):
            audit["reasons"].append("invalid_repair_bbox")
            continue
        if (not all(math.isfinite(value) for value in
                    (x, y, box_width, box_height))
                or box_width <= 0.0 or box_height <= 0.0):
            audit["reasons"].append("invalid_repair_bbox")
            continue
        x0 = max(0, int(math.floor(x)) - bbox_padding)
        y0 = max(0, int(math.floor(y)) - bbox_padding)
        x1 = min(width, int(math.ceil(x + box_width)) + bbox_padding)
        y1 = min(height, int(math.ceil(y + box_height)) + bbox_padding)
        allowed[y0:y1, x0:x1] = True
        target_labels.add(int(label))
    if audit["reasons"]:
        return audit
    changed_pixels = int(changed.sum())
    outside_changed = int((changed & ~allowed).sum())
    audit.update({
        "render_size": [width, height],
        "changed_pixels": changed_pixels,
        "outside_allowed_pixels": outside_changed,
        "allowed_pixels": int(allowed.sum()),
        "target_components": sorted(target_labels),
        "render_sha256": {
            "before": _image_sha256(before_image),
            "after": _image_sha256(after_image),
        },
    })
    if changed_pixels == 0:
        audit["reasons"].append("proposal_render_unchanged")
    if outside_changed:
        audit["reasons"].append("render_changed_outside_repair_bbox")

    before_failures = _failure_labels(before_scores)
    after_failures = _failure_labels(after_scores)
    if before_failures is None or after_failures is None:
        audit["reasons"].append("topology_failure_evidence_missing")
    else:
        unresolved = target_labels & after_failures
        new_failures = after_failures - before_failures
        audit.update({
            "failed_components_before": sorted(before_failures),
            "failed_components_after": sorted(after_failures),
            "unresolved_target_components": sorted(unresolved),
            "new_failed_components": sorted(new_failures),
        })
        if not target_labels.issubset(before_failures):
            audit["reasons"].append("target_not_in_before_failures")
        if unresolved:
            audit["reasons"].append("target_component_not_repaired")
        if new_failures:
            audit["reasons"].append("repair_created_new_topology_failure")

    gate_rank = {"rejected": 0, "manual_review": 1, "accepted": 2}
    before_status = before_gate.get("status") if isinstance(before_gate, dict) else None
    after_status = after_gate.get("status") if isinstance(after_gate, dict) else None
    audit["visual_gate"] = {"before": before_status, "after": after_status}
    if before_status not in gate_rank or after_status not in gate_rank:
        audit["reasons"].append("visual_gate_evidence_missing")
    elif gate_rank[after_status] < gate_rank[before_status]:
        audit["reasons"].append("visual_gate_regressed")

    metric_specs = [
        ("foreground", ("foreground",), 0.05),
        ("color_fidelity", ("foreground_color_fidelity",), 0.25),
        ("detail_p10", ("detail_grid", "p10_score_percent"), 0.05),
        ("detail_mean", ("detail_grid", "mean_score_percent"), 0.05),
        ("topology_p10", ("detail_grid", "component_topology",
                          "p10_score_percent"), 0.01),
        ("topology_worst", ("detail_grid", "component_topology",
                            "worst_score_percent"), 0.01),
    ]
    metrics = {}
    for name, path, tolerance in metric_specs:
        before = _score_value(before_scores, path)
        after = _score_value(after_scores, path)
        metrics[name] = {"before": before, "after": after,
                         "allowed_drop": tolerance}
        if before is None or after is None:
            audit["reasons"].append(f"{name}_evidence_missing")
        elif after < before - tolerance:
            audit["reasons"].append(f"{name}_regressed")
    before_light = _score_value(
        before_scores, ("transparent_light_fidelity", "coverage_percent"))
    after_light = _score_value(
        after_scores, ("transparent_light_fidelity", "coverage_percent"))
    before_light_applicable = bool((before_scores.get(
        "transparent_light_fidelity") or {}).get("applicable"))
    if before_light_applicable:
        metrics["light_object_coverage"] = {
            "before": before_light, "after": after_light,
            "allowed_drop": 0.1,
        }
        if before_light is None or after_light is None:
            audit["reasons"].append("light_object_coverage_evidence_missing")
        elif after_light < before_light - 0.1:
            audit["reasons"].append("light_object_coverage_regressed")
    audit["metrics"] = metrics
    if not audit["reasons"]:
        audit["status"] = "accepted"
    return audit


__all__ = [
    "append_repair_fragment",
    "propose_missing_component_repairs",
    "validate_repair_transaction",
]
