# -*- coding: utf-8 -*-
"""Transactional proposal stage for reducing traced SVG path anchors.

This module does not decide whether a proposal is safe to deliver.  It writes
an independently renderable candidate SVG and returns deterministic evidence;
the caller must compare that candidate against the pre-stage SVG and source
image before committing it.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

from clean_base import _parse_subpaths
from geometry_error_optimizer import optimize_compound_contours


_URL_FILL = re.compile(r"^\s*url\(", re.IGNORECASE)
_GRADIENT_OBJECT_SKIP_REASON = (
    "gradient_object_requires_source_ownership_revalidation"
)
_RETAINED_IDENTITY_ECONOMY_REASON = (
    "source_path_baseline_is_no_more_complex_than_best_error_eligible_"
    "evaluated_candidate"
)
_REQUIRED_ECONOMY_OBJECTIVE = frozenset({
    "preserve_topology_hard_constraint",
    "p95_bidirectional_geometric_error_percent_within_budget_hard_constraint",
    "maximum_error_within_three_times_budget_hard_tail_constraint",
    "over_budget_share_at_most_5_percent_hard_tail_constraint",
    "salient_corner_error_within_two_times_budget_hard_constraint",
    "minimize_designer_anchor_count",
    "minimize_anchor_count",
    "minimize_fragment_count",
    "minimize_segment_count",
})
_MAX_ARC_SAMPLES_PER_SEGMENT = 1024
_MAX_ARC_SUBPATH_SAMPLES = 131072


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _cubic(p0, values, t):
    u = 1.0 - t
    c1 = (values[1], values[2])
    c2 = (values[3], values[4])
    p3 = (values[5], values[6])
    return (
        u ** 3 * p0[0] + 3.0 * u * u * t * c1[0]
        + 3.0 * u * t * t * c2[0] + t ** 3 * p3[0],
        u ** 3 * p0[1] + 3.0 * u * u * t * c1[1]
        + 3.0 * u * t * t * c2[1] + t ** 3 * p3[1],
    )


def _quadratic(p0, values, t):
    u = 1.0 - t
    c = (values[1], values[2])
    p2 = (values[3], values[4])
    return (
        u * u * p0[0] + 2.0 * u * t * c[0] + t * t * p2[0],
        u * u * p0[1] + 2.0 * u * t * c[1] + t * t * p2[1],
    )


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _svg_arc_center_parameters(start, segment):
    """Convert one absolute SVG ``A`` segment to center parameters.

    This follows the SVG endpoint-to-center conversion, including absolute
    radii, x-axis rotation, large-arc/sweep flags, and radii correction.  A
    malformed or degenerate arc returns ``None`` so callers can fail closed.
    """
    if len(segment) != 8 or segment[0] != "A":
        return None
    try:
        x1, y1 = (float(start[0]), float(start[1]))
        rx, ry, rotation, large_arc, sweep, x2, y2 = (
            float(value) for value in segment[1:])
    except (TypeError, ValueError, OverflowError):
        return None
    values = (x1, y1, rx, ry, rotation, large_arc, sweep, x2, y2)
    if not all(math.isfinite(value) for value in values):
        return None
    if large_arc not in (0.0, 1.0) or sweep not in (0.0, 1.0):
        return None
    rx, ry = abs(rx), abs(ry)
    coordinate_scale = max(1.0, abs(x1), abs(y1), abs(x2), abs(y2))
    if (rx <= 0.0 or ry <= 0.0
            or _distance((x1, y1), (x2, y2))
            <= 1.0e-12 * coordinate_scale):
        return None

    try:
        phi = math.radians(math.fmod(rotation, 360.0))
        cos_phi, sin_phi = math.cos(phi), math.sin(phi)
        dx, dy = (x1 - x2) * 0.5, (y1 - y2) * 0.5
        x1_prime = cos_phi * dx + sin_phi * dy
        y1_prime = -sin_phi * dx + cos_phi * dy
        radius_ratio = ((x1_prime / rx) ** 2
                        + (y1_prime / ry) ** 2)
        if not math.isfinite(radius_ratio) or radius_ratio <= 0.0:
            return None
        if radius_ratio > 1.0:
            correction = math.sqrt(radius_ratio)
            rx *= correction
            ry *= correction
            radius_ratio = ((x1_prime / rx) ** 2
                            + (y1_prime / ry) ** 2)
        if (not all(math.isfinite(value) for value in (rx, ry, radius_ratio))
                or radius_ratio <= 0.0):
            return None

        center_scale = math.sqrt(max(
            0.0, (1.0 - min(1.0, radius_ratio)) / radius_ratio))
        if large_arc == sweep:
            center_scale = -center_scale
        cx_prime = center_scale * rx * y1_prime / ry
        cy_prime = -center_scale * ry * x1_prime / rx
        cx = (cos_phi * cx_prime - sin_phi * cy_prime
              + (x1 + x2) * 0.5)
        cy = (sin_phi * cx_prime + cos_phi * cy_prime
              + (y1 + y2) * 0.5)

        ux = (x1_prime - cx_prime) / rx
        uy = (y1_prime - cy_prime) / ry
        vx = (-x1_prime - cx_prime) / rx
        vy = (-y1_prime - cy_prime) / ry
        theta = math.atan2(uy, ux)
        delta = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        if sweep == 0.0 and delta > 0.0:
            delta -= 2.0 * math.pi
        elif sweep == 1.0 and delta < 0.0:
            delta += 2.0 * math.pi
    except (ArithmeticError, OverflowError, ValueError):
        return None

    parameters = (cx, cy, rx, ry, phi, theta, delta, x2, y2)
    if (not all(math.isfinite(value) for value in parameters)
            or abs(delta) <= 1.0e-12
            or abs(delta) > 2.0 * math.pi + 1.0e-9):
        return None
    return parameters


def _sample_svg_arc(start, segment, sample_step, existing_point_count):
    parameters = _svg_arc_center_parameters(start, segment)
    if parameters is None:
        return None
    cx, cy, rx, ry, phi, theta, delta, x2, y2 = parameters
    conservative_length = abs(delta) * max(rx, ry)
    if not math.isfinite(conservative_length):
        return None
    count = max(2, int(math.ceil(conservative_length / sample_step)))
    if (count > _MAX_ARC_SAMPLES_PER_SEGMENT
            or existing_point_count + count > _MAX_ARC_SUBPATH_SAMPLES):
        return None
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    sampled = []
    for index in range(1, count + 1):
        angle = theta + delta * index / float(count)
        ellipse_x = rx * math.cos(angle)
        ellipse_y = ry * math.sin(angle)
        point = (
            cx + cos_phi * ellipse_x - sin_phi * ellipse_y,
            cy + sin_phi * ellipse_x + cos_phi * ellipse_y,
        )
        if not all(math.isfinite(value) for value in point):
            return None
        sampled.append(point)
    # Avoid accumulated trigonometric drift at the authenticated endpoint.
    sampled[-1] = (x2, y2)
    return sampled


def _sample_subpath(subpath, sample_step):
    """Return dense source points, or ``None`` for unsafe geometry."""
    current = tuple(float(value) for value in subpath["start"])
    points = [current]
    for segment in subpath["segs"]:
        kind = segment[0]
        if kind == "L":
            endpoint = (float(segment[1]), float(segment[2]))
            length = _distance(current, endpoint)
            count = max(1, min(16, int(math.ceil(length / sample_step))))
            for index in range(1, count + 1):
                t = index / float(count)
                points.append((
                    current[0] + (endpoint[0] - current[0]) * t,
                    current[1] + (endpoint[1] - current[1]) * t,
                ))
            current = endpoint
        elif kind == "C":
            control1 = (float(segment[1]), float(segment[2]))
            control2 = (float(segment[3]), float(segment[4]))
            endpoint = (float(segment[5]), float(segment[6]))
            length = (_distance(current, control1)
                      + _distance(control1, control2)
                      + _distance(control2, endpoint))
            count = max(2, min(20, int(math.ceil(length / sample_step))))
            for index in range(1, count + 1):
                points.append(_cubic(current, segment, index / float(count)))
            current = endpoint
        elif kind == "Q":
            control = (float(segment[1]), float(segment[2]))
            endpoint = (float(segment[3]), float(segment[4]))
            length = _distance(current, control) + _distance(control, endpoint)
            count = max(2, min(20, int(math.ceil(length / sample_step))))
            for index in range(1, count + 1):
                points.append(_quadratic(current, segment, index / float(count)))
            current = endpoint
        elif kind == "A":
            sampled = _sample_svg_arc(
                current, segment, sample_step, len(points))
            if sampled is None:
                return None
            points.extend(sampled)
            current = (float(segment[6]), float(segment[7]))
        else:
            return None
    if subpath.get("closed", False) and len(points) > 1:
        if _distance(points[0], points[-1]) <= 1e-7:
            points.pop()
    return points


def _source_anchor_count(subpath):
    count = len(subpath["segs"]) + 1
    if subpath.get("closed") and subpath["segs"]:
        last = subpath["segs"][-1]
        endpoint = ((last[1], last[2]) if last[0] == "L"
                    else (last[-2], last[-1]))
        if math.hypot(endpoint[0] - subpath["start"][0],
                      endpoint[1] - subpath["start"][1]) <= 1.0e-7:
            count -= 1
    return int(count)


def _arc_sampling_policy_allows(subpaths):
    """Allow arcs only as complete, isolated loops in mixed compounds.

    An arc-only shape is already economical and must not be approximated by
    the generic optimizer.  Arc sampling is useful only when it lets a noisy
    sibling loop be simplified while an exact arc loop participates in the
    compound topology contract.  Keeping arc and non-arc commands in separate
    subpaths also avoids inventing semantics for partially analytic loops.
    """
    arc_loops = []
    non_arc_loops = []
    for subpath in subpaths:
        segments = subpath.get("segs") or []
        kinds = {segment[0] for segment in segments if segment}
        if "A" not in kinds:
            non_arc_loops.append(subpath)
            continue
        if kinds != {"A"} or len(segments) < 2 or not subpath.get("closed"):
            return False
        endpoint = (float(segments[-1][6]), float(segments[-1][7]))
        start = tuple(float(value) for value in subpath["start"])
        if _distance(start, endpoint) > 1.0e-7:
            return False
        arc_loops.append(subpath)
    if not arc_loops:
        return True
    return any(_source_anchor_count(item) >= 4 for item in non_arc_loops)


def _effective_fill(element, parents):
    node = element
    while node is not None:
        fill = node.get("fill")
        if fill is not None:
            return fill.strip()
        style = node.get("style") or ""
        for item in style.split(";"):
            key, sep, value = item.partition(":")
            if sep and key.strip().lower() == "fill":
                return value.strip()
        node = parents.get(node)
    return "#000000"


def _has_gradient_object_ownership(element, parents):
    """Return true when this drawable belongs to a reconstructed gradient.

    Gradient geometry was accepted against its original source ownership mask.
    This global refit stage does not receive that mask, so changing even a
    geometrically plausible path here would make the earlier source-space
    evidence stale.  Treat direct and inherited ownership metadata alike.
    """
    node = element
    while node is not None:
        if node.get("data-avc-gradient-object") is not None:
            return True
        node = parents.get(node)
    return False


def _number(value):
    value = float(value)
    if abs(value) < 0.0000005:
        value = 0.0
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _finite_float(value):
    """Return one finite float, or ``None`` for incomplete evidence."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _path_data_sha256(path_data):
    return hashlib.sha256(path_data.encode("utf-8")).hexdigest()


def _ensure_stable_refit_identifier(
        element, index, used_identifiers, *, source_svg_sha256,
        source_path_data):
    """Give an optimiser-evaluated id-less path a deterministic SVG ID.

    Economy evidence must join to exactly one final drawable.  The historic
    ``path-{index}`` report-only fallback could not prove that join because it
    was never written to the SVG.  A render-neutral ID is therefore assigned
    only after the optimiser has actually evaluated the path.  Existing IDs
    are preserved verbatim; generated IDs are collision-safe across the whole
    document and are still covered by the transaction's renderer/source
    guards in case an unusual stylesheet makes an ID render-affecting.
    """
    original_identifier = element.get("id")
    path_digest = _path_data_sha256(source_path_data)
    if original_identifier:
        identifier = original_identifier
        original_state = "existing_preserved"
        assignment_applied = False
    else:
        base = f"avc-refit-path-{index}-{path_digest[:12]}"
        identifier = base
        suffix = 2
        while identifier in used_identifiers:
            identifier = f"{base}-{suffix}"
            suffix += 1
        element.set("id", identifier)
        used_identifiers.add(identifier)
        original_state = "missing_assigned"
        assignment_applied = True
    return {
        "source_svg_sha256": source_svg_sha256,
        "global_path_ordinal_1_based": index,
        "source_element": "path",
        "source_path_data_sha256": path_digest,
        "source_path_data_digest_scope": "utf8_svg_path_d_attribute",
        "original_id_state": original_state,
        "original_id": original_identifier if original_identifier else None,
        "assigned_id": identifier,
        "assignment_applied": assignment_applied,
    }


def _existing_identifier_inventory(root):
    """Return unique existing IDs or fail closed on an ambiguous document."""
    identifiers = [
        identifier
        for item in root.iter()
        for identifier in (item.get("id"),)
        if identifier
    ]
    seen = set()
    duplicates = set()
    for identifier in identifiers:
        if identifier in seen:
            duplicates.add(identifier)
        seen.add(identifier)
    duplicates = sorted(duplicates)
    if duplicates:
        preview = ", ".join(duplicates[:3])
        raise RuntimeError(
            "curve-refit source contains duplicate existing SVG IDs: "
            f"{preview}")
    return set(identifiers)


def _committed_geometry_evidence(element):
    """Bind one committed detail to the geometry actually written to SVG."""
    final_element = _local(element.tag)
    identifier = element.get("id") or ""
    base = {
        "final_element": final_element,
        "final_drawable_id": identifier,
    }
    if final_element == "path":
        path_data = element.get("d") or ""
        return {
            **base,
            "path_data_sha256": _path_data_sha256(path_data),
            "path_data_digest_scope": "utf8_svg_path_d_attribute",
        }

    parameter_names = {
        "circle": ("cx", "cy", "r"),
        "ellipse": ("cx", "cy", "rx", "ry", "transform"),
    }.get(final_element, ())
    parameters = {
        name: element.get(name)
        for name in parameter_names if element.get(name) is not None
    }
    canonical = json.dumps({
        "element": final_element,
        "id": identifier,
        "parameters": parameters,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        **base,
        "native_parameters": parameters,
        "geometry_sha256": hashlib.sha256(
            canonical.encode("utf-8")).hexdigest(),
        "geometry_digest_scope": (
            "canonical_json_element_id_and_svg_geometry_attributes"
        ),
    }


def _identity_retention_evidence(
        element, path_data, result, *, before, after, designer_after,
        error_budget_percent, contour_count, fill):
    """Validate and compact a source-identity optimizer selection.

    This is deliberately stricter than trusting ``data-avc-*`` attributes or
    a single optimizer boolean.  The original SVG path is an implicit exact,
    zero-error baseline candidate.  It is certifiable only when it is no more
    complex than the optimizer's best error-eligible candidate, all four
    geometry constraints are finite and satisfied, the economy objective is
    complete, and the final path can be joined by a stable SVG id plus the
    exact UTF-8 digest of its unchanged ``d`` attribute.
    """
    failures = []
    if not isinstance(result, Mapping):
        return None, ["optimizer_result_not_mapping"]

    identifier = (element.get("id") or "").strip()
    if not identifier:
        failures.append("stable_svg_id_missing")
    optimizer_status = result.get("status")
    identity_selected = result.get("identity_rollback_selected")
    safe_refit_selected = result.get("safe_refit_selected")
    selection_outcome = result.get("selection_outcome")
    valid_selection_states = {
        ("identity_rollback_no_safe_reduction", True, False,
         "source_identity_rollback"),
        ("selected_within_geometry_budget", False, True,
         "certified_geometry_refit"),
    }
    if (optimizer_status, identity_selected, safe_refit_selected,
            selection_outcome) not in valid_selection_states:
        failures.append("optimizer_selection_state_inconsistent")
    if result.get("optimization_basis") != "geometry_only":
        failures.append("optimization_basis_not_geometry_only")
    if result.get("uses_colour_or_pixel_similarity") is not False:
        failures.append("colour_or_pixel_similarity_participated")

    try:
        optimizer_input_anchors = int(result.get("anchors_before"))
    except (TypeError, ValueError):
        optimizer_input_anchors = None
    try:
        selected_segments = int(result.get("segment_count_after"))
    except (TypeError, ValueError):
        selected_segments = None
    if optimizer_input_anchors is None or optimizer_input_anchors < 1:
        failures.append("optimizer_input_anchor_count_missing_or_invalid")
    if selected_segments is None or selected_segments < 1:
        failures.append("selected_segment_count_missing_or_invalid")
    source_economy = (before, before, before)
    selected_economy = None if selected_segments is None else (
        designer_after, after, selected_segments)
    if selected_economy is not None and selected_economy < source_economy:
        failures.append("optimizer_found_more_economical_candidate")

    try:
        loops = int(result.get("loop_count"))
    except (TypeError, ValueError):
        loops = None
    if loops != contour_count or not contour_count:
        failures.append("loop_count_mismatch")
    selected_candidate_id = result.get("selected_candidate_id")
    if not isinstance(selected_candidate_id, str) \
            or not selected_candidate_id.strip():
        failures.append("selected_candidate_id_missing")
    selected_source = result.get("selected_source")
    if not isinstance(selected_source, str) or not selected_source.strip():
        failures.append("selected_source_missing")
    try:
        candidate_count = int(result.get("candidate_count"))
        eligible_candidate_count = int(result.get("eligible_candidate_count"))
    except (TypeError, ValueError):
        candidate_count = None
        eligible_candidate_count = None
    if candidate_count is None or candidate_count < 1:
        failures.append("candidate_count_missing_or_invalid")
    if (eligible_candidate_count is None or eligible_candidate_count < 1
            or (candidate_count is not None
                and eligible_candidate_count > candidate_count)):
        failures.append("eligible_candidate_count_missing_or_invalid")

    objective = result.get("lexicographic_objective")
    if not isinstance(objective, list):
        objective = []
        failures.append("lexicographic_objective_missing")
    else:
        objective_names = {
            item for item in objective if isinstance(item, str)}
        if len(objective_names) != len(objective):
            failures.append("lexicographic_objective_contains_invalid_items")
        missing_objectives = sorted(
            _REQUIRED_ECONOMY_OBJECTIVE.difference(objective_names))
        if missing_objectives:
            failures.append("lexicographic_objective_incomplete")

    reported_budget = _finite_float(result.get("error_budget_percent"))
    p95 = _finite_float(result.get("actual_p95_error_percent"))
    maximum = _finite_float(result.get("actual_max_error_percent"))
    over_budget_share = _finite_float(result.get("over_budget_share"))
    salient = _finite_float(result.get("salient_corner_max_percent"))
    if reported_budget is None or not math.isclose(
            reported_budget, error_budget_percent,
            rel_tol=0.0, abs_tol=1.0e-9):
        failures.append("error_budget_mismatch")
    if p95 is None:
        failures.append("p95_error_missing_or_nonfinite")
    elif p95 > error_budget_percent + 1.0e-12:
        failures.append("p95_error_exceeds_budget")
    if maximum is None:
        failures.append("maximum_error_missing_or_nonfinite")
    elif maximum > 3.0 * error_budget_percent + 1.0e-12:
        failures.append("maximum_error_exceeds_tail_budget")
    if over_budget_share is None:
        failures.append("over_budget_share_missing_or_nonfinite")
    elif not 0.0 <= over_budget_share <= 0.05 + 1.0e-12:
        failures.append("over_budget_share_exceeds_limit")
    if salient is None:
        failures.append("salient_corner_error_missing_or_nonfinite")
    elif salient > 2.0 * error_budget_percent + 1.0e-12:
        failures.append("salient_corner_error_exceeds_budget")

    if failures:
        return None, failures

    primitive_evidence = result.get("primitive_complexity") or {}
    if not isinstance(primitive_evidence, Mapping):
        primitive_evidence = {}
    return {
        "id": identifier,
        "fill": fill,
        "outcome": "retained_identity_minimum",
        "economy_certified": True,
        "economy_reason": _RETAINED_IDENTITY_ECONOMY_REASON,
        "retention_basis": "source_svg_path_implicit_zero_error_candidate",
        "geometry_unchanged": True,
        "path_data_sha256": _path_data_sha256(path_data),
        "path_data_digest_scope": "utf8_svg_path_d_attribute",
        "source_baseline_geometry_error": {
            "actual_p95_error_percent": 0.0,
            "actual_max_error_percent": 0.0,
            "over_budget_share": 0.0,
            "salient_corner_max_percent": 0.0,
            "passed": True,
        },
        "economy_comparison_order": [
            "designer_anchor_count",
            "anchor_count",
            "fragment_count",
            "segment_count",
        ],
        "source_baseline_economy": {
            "designer_anchor_count": before,
            "anchor_count": before,
            "fragment_count": 1,
            "segment_count": before,
        },
        "optimizer_selected_economy": {
            "designer_anchor_count": designer_after,
            "anchor_count": after,
            "fragment_count": 1,
            "segment_count": selected_segments,
        },
        "anchors_before": before,
        "anchors_after": before,
        "designer_anchors_before": before,
        "designer_anchors_after": before,
        "optimizer_input_anchor_count": optimizer_input_anchors,
        "optimizer_selected_anchor_count": after,
        "optimizer_selected_designer_anchor_count": designer_after,
        "optimizer_selected_segment_count": selected_segments,
        "error_budget_percent": error_budget_percent,
        "actual_p95_error_percent": p95,
        "actual_max_error_percent": maximum,
        "over_budget_share": over_budget_share,
        "salient_corner_max_percent": salient,
        "selected_candidate_id": selected_candidate_id,
        "selected_source": selected_source,
        "identity_rollback_selected": bool(identity_selected),
        "lexicographic_objective": list(objective),
        "loops": loops,
        "candidate_count": candidate_count,
        "eligible_candidate_count": eligible_candidate_count,
        "primitive": {
            "category": primitive_evidence.get("category"),
            "native": primitive_evidence.get("native_primitives", []),
            "emitted_element": None,
        },
    }, []


def _svg_tag_like(element, local_name):
    if element.tag.startswith("{"):
        namespace = element.tag.split("}", 1)[0] + "}"
        return namespace + local_name
    return local_name


def _native_primitive(result):
    """Return the sole contour's native circle/ellipse evidence, if any."""
    fit = result.get("fit") or {}
    contours = fit.get("contours") or []
    if len(contours) != 1:
        return None, None
    contour = contours[0]
    primitive = contour.get("primitive")
    native = contour.get("native_primitive")
    if primitive not in {"circle", "ellipse"} or not isinstance(native, dict):
        return None, None
    return primitive, native


def _replace_with_native_primitive(element, primitive, native):
    """Mutate one path into a true SVG circle/ellipse, preserving metadata."""
    element.attrib.pop("d", None)
    element.attrib.pop("transform", None)
    if primitive == "circle":
        element.tag = _svg_tag_like(element, "circle")
        element.set("cx", _number(native["cx"]))
        element.set("cy", _number(native["cy"]))
        element.set("r", _number(native["r"]))
        return
    element.tag = _svg_tag_like(element, "ellipse")
    cx, cy = float(native["cx"]), float(native["cy"])
    element.set("cx", _number(cx))
    element.set("cy", _number(cy))
    element.set("rx", _number(native["rx"]))
    element.set("ry", _number(native["ry"]))
    rotation = float(native.get("rotation_degrees", 0.0))
    if abs(rotation) > 0.000001:
        element.set(
            "transform",
            f"rotate({_number(rotation)} {_number(cx)} {_number(cy)})")


def build_svg_curve_refit_path_frontier(
        source_svg, target_id, *, error_budget_percent=0.25,
        sample_step=2.0, maximum_segments=4096):
    """Build a geometry-only conservative frontier for one authenticated path."""
    error_budget_percent = float(error_budget_percent)
    sample_step = float(sample_step)
    if (not isinstance(target_id, str) or not target_id
            or not math.isfinite(error_budget_percent)
            or error_budget_percent <= 0
            or not math.isfinite(sample_step) or sample_step <= 0
            or int(maximum_segments) < 1):
        raise ValueError("invalid single-path refinement frontier request")
    candidate_tolerances = tuple(sorted({
        error_budget_percent * factor
        for factor in (0.10, 0.25, 0.50, 1.0, 1.5, 2.25, 3.0, 4.0)
    }))
    root = ET.fromstring(Path(source_svg).read_bytes())
    parents = {child: parent for parent in root.iter() for child in parent}
    matches = [item for item in root.iter() if item.get("id") == target_id]
    if len(matches) != 1 or _local(matches[0].tag) != "path":
        raise RuntimeError("single-path frontier target is missing or ambiguous")
    element = matches[0]
    path_data = element.get("d") or ""
    subpaths = _parse_subpaths(path_data)
    before = sum(_source_anchor_count(item) for item in subpaths)
    fill = _effective_fill(element, parents)
    arc_sampling_policy = _arc_sampling_policy_allows(subpaths)
    sampled_subpaths = ([
        _sample_subpath(item, sample_step) for item in subpaths]
        if arc_sampling_policy else [])
    eligibility = {
        "not_gradient_owned": not _has_gradient_object_ownership(
            element, parents),
        "supported_geometry": bool(
            sampled_subpaths
            and all(points is not None and len(points) >= 3
                    for points in sampled_subpaths)),
        "arc_sampling_policy": arc_sampling_policy,
        "closed": bool(
            subpaths and all(item.get("closed") for item in subpaths)),
        "untransformed": not bool(element.get("transform")),
        "filled": fill.lower() != "none",
        "source_has_reducible_anchor_budget": before >= 4,
    }
    if not all(eligibility.values()):
        return {
            "schema": "ai-vector-cleanroom.curve-refit-path-frontier/v1",
            "status": "ineligible",
            "target_id": target_id,
            "source_path_data_sha256": hashlib.sha256(
                path_data.encode("utf-8")).hexdigest(),
            "source_anchor_count": before,
            "eligibility": eligibility,
            "candidates": [],
        }
    contours = sampled_subpaths
    result = optimize_compound_contours(
        contours,
        include_independently_valid_loop_candidates=True,
        error_budget_percent=error_budget_percent,
        tolerance_percents=candidate_tolerances,
        sampling_step_percent=0.08,
        max_samples_per_segment=1024,
        allow_primitives=True,
        max_segments=int(maximum_segments),
        include_refinement_frontier=True,
    )
    frontier = result.get("refinement_frontier") or {}
    rows = frontier.get("candidates") or []
    base_anchors = int(result.get("anchors_after", before))
    base_designer = int(result.get("designer_anchor_count", base_anchors))

    def finite(name, row):
        value = row.get(name)
        return (float(value) if isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(float(value))
                else float("inf"))

    def ordering(row):
        return (
            int(row.get("designer_anchor_count", 0)),
            int(row.get("anchors_after", 0)),
            int(row.get("segment_count", 0)),
            finite("actual_max_error_percent", row),
            finite("actual_p95_error_percent", row),
            finite("over_budget_share", row),
            finite("salient_corner_max_percent", row),
            str(row.get("candidate_id") or ""),
        )

    candidates = []
    for row in rows:
        if (not isinstance(row, Mapping)
                or row.get("candidate_kind") != "single_loop_refinement"
                or row.get("eligible") is not True):
            continue
        anchors = row.get("anchors_after")
        designer = row.get("designer_anchor_count")
        if (not isinstance(anchors, int) or isinstance(anchors, bool)
                or not isinstance(designer, int) or isinstance(designer, bool)
                or anchors <= base_anchors or designer <= base_designer
                or anchors >= before or designer >= before
                or not isinstance(row.get("path"), str) or not row["path"]):
            continue
        candidate = dict(row)
        candidate["path_data_sha256"] = hashlib.sha256(
            candidate["path"].encode("utf-8")).hexdigest()
        candidates.append(candidate)
    candidates.sort(key=ordering)
    return {
        "schema": "ai-vector-cleanroom.curve-refit-path-frontier/v1",
        "status": "candidates_available" if candidates else "no_candidate",
        "target_id": target_id,
        "optimization_basis": "geometry_only",
        "uses_colour_or_pixel_similarity": False,
        "source_path_data_sha256": hashlib.sha256(
            path_data.encode("utf-8")).hexdigest(),
        "source_anchor_count": before,
        "source_loop_anchor_counts": [
            _source_anchor_count(item) for item in subpaths],
        "loop_count": len(subpaths),
        "fill": fill,
        "eligibility": eligibility,
        "error_budget_percent": error_budget_percent,
        "candidate_tolerance_percents": [
            round(value, 9) for value in candidate_tolerances],
        "measurement_step_percent": 0.08,
        "measurement_max_samples_per_segment": 1024,
        "base_candidate": {
            "selected_candidate_id": result.get("selected_candidate_id"),
            "path_data_sha256": hashlib.sha256(
                (result.get("path") or "").encode("utf-8")).hexdigest(),
            "anchors_after": base_anchors,
            "designer_anchor_count": base_designer,
            "actual_p95_error_percent": result.get(
                "actual_p95_error_percent"),
            "actual_max_error_percent": result.get(
                "actual_max_error_percent"),
            "safe_refit_selected": result.get("safe_refit_selected"),
        },
        "lexicographic_objective": list(
            result.get("lexicographic_objective") or []),
        "selection_basis": (
            "designer_anchors_then_anchors_then_segments_then_geometry_then_id"),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def apply_svg_curve_refit_path_frontier_candidate(
        svg_bytes, *, target_id, frontier, candidate):
    """Apply one certified frontier row to one path and emit proposal detail."""
    if (not isinstance(frontier, Mapping)
            or frontier.get("schema") !=
            "ai-vector-cleanroom.curve-refit-path-frontier/v1"
            or frontier.get("status") != "candidates_available"
            or frontier.get("target_id") != target_id
            or frontier.get("uses_colour_or_pixel_similarity") is not False
            or not isinstance(candidate, Mapping)
            or candidate.get("candidate_kind") != "single_loop_refinement"
            or candidate.get("eligible") is not True):
        raise RuntimeError("single-path frontier candidate evidence is invalid")
    candidate_id = str(candidate.get("candidate_id") or "")
    authoritative = [item for item in frontier.get("candidates", [])
                     if isinstance(item, Mapping)
                     and item.get("candidate_id") == candidate_id]
    if len(authoritative) != 1 or dict(authoritative[0]) != dict(candidate):
        raise RuntimeError("single-path frontier candidate is not authoritative")
    root = ET.fromstring(bytes(svg_bytes))
    parents = {child: parent for parent in root.iter() for child in parent}
    matches = [item for item in root.iter() if item.get("id") == target_id]
    if len(matches) != 1 or _local(matches[0].tag) != "path":
        raise RuntimeError("single-path frontier apply target is ambiguous")
    element = matches[0]
    source_path = element.get("d") or ""
    source_sha = hashlib.sha256(source_path.encode("utf-8")).hexdigest()
    if source_sha != frontier.get("source_path_data_sha256"):
        raise RuntimeError("single-path frontier source geometry changed")
    replacement = candidate.get("path") or ""
    replacement_sha = hashlib.sha256(
        replacement.encode("utf-8")).hexdigest()
    if candidate.get("path_data_sha256") != replacement_sha:
        raise RuntimeError("single-path frontier candidate digest mismatch")
    anchors_before = int(frontier.get("source_anchor_count"))
    anchors_after = int(candidate.get("anchors_after"))
    designer_after = int(candidate.get("designer_anchor_count"))
    if not (0 < anchors_after < anchors_before
            and 0 < designer_after < anchors_before):
        raise RuntimeError("single-path frontier candidate is not a reduction")
    element.set("d", replacement)
    element.set("data-avc-curve-refit", "geometry-budgeted")
    element.set("data-avc-anchors-before", str(anchors_before))
    element.set("data-avc-anchors-after", str(anchors_after))
    element.set("data-avc-designer-anchors", str(designer_after))
    element.set("data-avc-error-budget-percent", _number(
        frontier.get("error_budget_percent")))
    element.set("data-avc-p95-error-percent", _number(
        candidate.get("actual_p95_error_percent")))
    element.set("data-avc-max-error-percent", _number(
        candidate.get("actual_max_error_percent")))
    committed = _committed_geometry_evidence(element)
    if committed.get("path_data_sha256") != replacement_sha:
        raise RuntimeError("single-path frontier committed digest mismatch")
    detail = {
        "id": target_id,
        "fill": _effective_fill(element, parents),
        "outcome": "committed_refit",
        "economy_certified": True,
        **committed,
        "anchors_before": anchors_before,
        "anchors_after": anchors_after,
        "designer_anchors_before": anchors_before,
        "designer_anchors_after": designer_after,
        "reduction_ratio": round(
            (anchors_before - anchors_after) / float(anchors_before), 6),
        "designer_reduction_ratio": round(
            (anchors_before - designer_after) / float(anchors_before), 6),
        "error_budget_percent": frontier.get("error_budget_percent"),
        "candidate_tolerance_percents": list(
            frontier.get("candidate_tolerance_percents") or []),
        "measurement_step_percent": frontier.get(
            "measurement_step_percent"),
        "measurement_max_samples_per_segment": frontier.get(
            "measurement_max_samples_per_segment"),
        "actual_p95_error_percent": candidate.get(
            "actual_p95_error_percent"),
        "actual_max_error_percent": candidate.get(
            "actual_max_error_percent"),
        "over_budget_share": candidate.get("over_budget_share"),
        "salient_corner_max_percent": candidate.get(
            "salient_corner_max_percent"),
        "primitive": {
            "category": (candidate.get("primitive_complexity") or {}).get(
                "category"),
            "native": (candidate.get("primitive_complexity") or {}).get(
                "native_primitives", []),
            "emitted_element": None,
        },
        "selected_candidate_id": candidate_id,
        "lexicographic_objective": list(
            frontier.get("lexicographic_objective") or []),
        "loops": frontier.get("loop_count"),
        "refinement_frontier": {
            "schema": frontier.get("schema"),
            "selection_basis": frontier.get("selection_basis"),
            "changed_loop_index": candidate.get("changed_loop_index"),
            "replacement_candidate_id": candidate.get(
                "replacement_candidate_id"),
            "base_candidate": dict(frontier.get("base_candidate") or {}),
        },
    }
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace(
        "inkscape", "http://www.inkscape.org/namespaces/inkscape")
    output = io.BytesIO()
    ET.ElementTree(root).write(
        output, encoding="utf-8", xml_declaration=True)
    return {
        "bytes": output.getvalue(),
        "detail": detail,
        "integrity": {
            "target_id_unique": True,
            "source_path_data_sha256": source_sha,
            "candidate_path_data_sha256": committed.get(
                "path_data_sha256"),
            "only_target_path_requested": True,
        },
    }


def propose_svg_curve_refit(
    source_svg,
    candidate_svg,
    *,
    tolerance=0.72,
    line_tolerance=None,
    error_budget_percent=0.25,
    sample_step=2.2,
    minimum_nodes=10,
    minimum_reduction_ratio=0.0,
    maximum_segments=4096,
):
    """Write a low-anchor candidate and return JSON-safe proposal evidence.

    Filled traced paths are eligible.  Open paths, real strokes, paths using
    transforms, and unsafe or degenerate geometry are deliberately left
    untouched.  Complete arc-only loops may be sampled through the standard
    endpoint-to-center conversion only when a separate noisy non-arc sibling
    loop makes the closed compound reducible; final source/render validation
    remains the caller's transaction boundary.
    """
    source_svg = Path(source_svg)
    candidate_svg = Path(candidate_svg)
    tolerance = float(tolerance)
    error_budget_percent = float(error_budget_percent)
    sample_step = float(sample_step)
    minimum_nodes = int(minimum_nodes)
    minimum_reduction_ratio = float(minimum_reduction_ratio)
    maximum_segments = int(maximum_segments)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if line_tolerance is not None:
        line_tolerance = float(line_tolerance)
        if not math.isfinite(line_tolerance) or line_tolerance <= 0:
            raise ValueError("line_tolerance must be positive and finite")
    if (not math.isfinite(error_budget_percent)
            or error_budget_percent <= 0):
        raise ValueError("error_budget_percent must be positive and finite")
    if not math.isfinite(sample_step) or sample_step <= 0:
        raise ValueError("sample_step must be positive and finite")
    if minimum_nodes < 3:
        raise ValueError("minimum_nodes must be at least 3")
    if not 0.0 <= minimum_reduction_ratio < 1.0:
        raise ValueError("minimum_reduction_ratio must be in [0, 1)")
    if maximum_segments < 1:
        raise ValueError("maximum_segments must be at least 1")
    candidate_tolerance_percents = tuple(sorted({
        error_budget_percent * factor
        for factor in (0.10, 0.25, 0.50, 1.0, 1.5, 2.25, 3.0, 4.0)
    }))

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")
    source_bytes = source_svg.read_bytes()
    source_svg_sha256 = hashlib.sha256(source_bytes).hexdigest()
    root = ET.fromstring(source_bytes)
    tree = ET.ElementTree(root)
    parents = {child: parent for parent in root.iter() for child in parent}
    used_identifiers = _existing_identifier_inventory(root)
    source_path_count = sum(
        1 for item in root.iter() if _local(item.tag) == "path")
    totals_before = 0
    totals_after = 0
    designer_totals_before = 0
    designer_totals_after = 0
    eligible = 0
    accepted = 0
    optimizer_evaluated = 0
    skipped = {}
    details = []
    evaluated_but_retained = []
    uncertified_evaluations = []
    stable_id_records = []

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    def record_uncertified(element, index, *, stage_reason,
                           integrity_failures, before=None, after=None,
                           designer_after=None, result=None):
        result = result if isinstance(result, Mapping) else {}
        uncertified_evaluations.append({
            "id": element.get("id") or f"path-{index}",
            "outcome": "evaluated_not_economy_certified",
            "economy_certified": False,
            "stage_reason": stage_reason,
            "integrity_failures": list(integrity_failures),
            "anchors_before": before,
            "anchors_after": after,
            "designer_anchors_before": before,
            "designer_anchors_after": designer_after,
            "selected_candidate_id": result.get("selected_candidate_id"),
            "identity_rollback_selected": result.get(
                "identity_rollback_selected"),
            "lexicographic_objective": result.get(
                "lexicographic_objective", []),
            "loops": result.get("loop_count"),
        })

    for index, element in enumerate(
            (item for item in root.iter() if _local(item.tag) == "path"), 1):
        path_data = element.get("d") or ""
        try:
            subpaths = _parse_subpaths(path_data)
        except Exception:
            skip("parse_error")
            continue
        before = sum(_source_anchor_count(subpath) for subpath in subpaths)
        totals_before += before
        totals_after += before
        designer_totals_before += before
        designer_totals_after += before
        if _has_gradient_object_ownership(element, parents):
            skip(_GRADIENT_OBJECT_SKIP_REASON)
            continue
        if not _arc_sampling_policy_allows(subpaths):
            skip("unsupported_or_degenerate_geometry")
            continue
        if before < minimum_nodes:
            skip("below_minimum_nodes")
            continue
        if any(not subpath.get("closed", False) for subpath in subpaths):
            skip("open_path")
            continue
        if element.get("transform"):
            skip("transformed_path")
            continue
        fill = _effective_fill(element, parents)
        if fill.lower() == "none":
            skip("stroke_or_unfilled_path")
            continue
        contours = []
        supported = True
        for subpath in subpaths:
            points = _sample_subpath(subpath, sample_step)
            if points is None:
                supported = False
                break
            if len(points) < 3:
                supported = False
                break
            contours.append(points)
        if not supported:
            skip("unsupported_or_degenerate_geometry")
            continue
        eligible += 1
        try:
            result = optimize_compound_contours(
                contours,
                include_independently_valid_loop_candidates=True,
                error_budget_percent=error_budget_percent,
                tolerance_percents=candidate_tolerance_percents,
                sampling_step_percent=0.08,
                max_samples_per_segment=1024,
                allow_primitives=True,
                max_segments=maximum_segments,
            )
        except Exception:
            skip("optimizer_error")
            continue
        optimizer_evaluated += 1
        stable_id_records.append(_ensure_stable_refit_identifier(
            element, index, used_identifiers,
            source_svg_sha256=source_svg_sha256,
            source_path_data=path_data))
        if not isinstance(result, Mapping):
            skip("optimizer_evidence_invalid")
            record_uncertified(
                element, index, stage_reason="optimizer_evidence_invalid",
                integrity_failures=["optimizer_result_not_mapping"],
                before=before, result=result)
            continue
        try:
            after = int(result["anchors_after"])
            designer_after = int(result.get("designer_anchor_count", after))
        except (KeyError, TypeError, ValueError, OverflowError):
            skip("optimizer_evidence_invalid")
            record_uncertified(
                element, index, stage_reason="optimizer_evidence_invalid",
                integrity_failures=[
                    "anchor_or_designer_anchor_count_missing_or_invalid"],
                before=before, result=result)
            continue
        reduction = (before - after) / float(max(1, before))
        designer_reduction = (
            (before - designer_after) / float(max(1, before)))
        primitive, native = _native_primitive(result)
        improves_complexity = bool(
            primitive is not None or after < before or designer_after < before)
        insufficient_reason = None
        if designer_after > before:
            insufficient_reason = "designer_anchor_count_exceeds_source"
        elif not improves_complexity:
            insufficient_reason = "no_strict_complexity_reduction"
        elif max(reduction, designer_reduction) < minimum_reduction_ratio:
            insufficient_reason = "reduction_below_stage_commit_threshold"
        if insufficient_reason is not None:
            skip("insufficient_reduction")
            retained, failures = _identity_retention_evidence(
                element, path_data, result,
                before=before, after=after,
                designer_after=designer_after,
                error_budget_percent=error_budget_percent,
                contour_count=len(contours), fill=fill)
            if retained is not None:
                retained["stage_reason"] = insufficient_reason
                evaluated_but_retained.append(retained)
            else:
                record_uncertified(
                    element, index, stage_reason=insufficient_reason,
                    integrity_failures=failures, before=before, after=after,
                    designer_after=designer_after, result=result)
            continue
        replacement = result.get("path") or ""
        if not replacement:
            skip("empty_replacement")
            record_uncertified(
                element, index, stage_reason="empty_replacement",
                integrity_failures=["replacement_path_missing"],
                before=before, after=after,
                designer_after=designer_after, result=result)
            continue
        if primitive is not None:
            _replace_with_native_primitive(element, primitive, native)
        else:
            element.set("d", replacement)
        element.set("data-avc-curve-refit", "geometry-budgeted")
        element.set("data-avc-anchors-before", str(before))
        element.set("data-avc-anchors-after", str(after))
        element.set("data-avc-designer-anchors", str(designer_after))
        element.set("data-avc-error-budget-percent",
                    _number(error_budget_percent))
        element.set("data-avc-p95-error-percent",
                    _number(result["actual_p95_error_percent"]))
        element.set("data-avc-max-error-percent",
                    _number(result["actual_max_error_percent"]))
        totals_after += after - before
        designer_totals_after += designer_after - before
        accepted += 1
        primitive_evidence = result.get("primitive_complexity") or {}
        details.append({
            "id": element.get("id") or f"path-{index}",
            "fill": fill,
            "outcome": "committed_refit",
            "economy_certified": True,
            **_committed_geometry_evidence(element),
            "anchors_before": before,
            "anchors_after": after,
            "designer_anchors_before": before,
            "designer_anchors_after": designer_after,
            "reduction_ratio": round(reduction, 6),
            "designer_reduction_ratio": round(designer_reduction, 6),
            "error_budget_percent": error_budget_percent,
            "candidate_tolerance_percents": [
                round(value, 9) for value in candidate_tolerance_percents
            ],
            "measurement_step_percent": 0.08,
            "measurement_max_samples_per_segment": 1024,
            "actual_p95_error_percent": result.get(
                "actual_p95_error_percent"),
            "actual_max_error_percent": result.get(
                "actual_max_error_percent"),
            "over_budget_share": result.get("over_budget_share"),
            "salient_corner_max_percent": result.get(
                "salient_corner_max_percent"),
            "primitive": {
                "category": primitive_evidence.get("category"),
                "native": primitive_evidence.get("native_primitives", []),
                "emitted_element": primitive,
            },
            "selected_candidate_id": result.get("selected_candidate_id"),
            "lexicographic_objective": result.get(
                "lexicographic_objective", []),
            "loops": result.get("loop_count", len(contours)),
        })

    candidate_ids = [
        identifier
        for item in root.iter()
        for identifier in (item.get("id"),)
        if identifier
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(
            "curve-refit candidate contains duplicate assigned SVG IDs")
    candidate_output = io.BytesIO()
    tree.write(candidate_output, encoding="utf-8", xml_declaration=True)
    candidate_bytes = candidate_output.getvalue()
    candidate_svg.parent.mkdir(parents=True, exist_ok=True)
    candidate_svg.write_bytes(candidate_bytes)
    removed = totals_before - totals_after
    designer_removed = designer_totals_before - designer_totals_after
    committed_ids = [str(item.get("id")) for item in details
                     if item.get("id")]
    retained_ids = [str(item.get("id")) for item in evaluated_but_retained
                    if item.get("id")]
    uncertified_ids = [str(item.get("id")) for item in uncertified_evaluations
                       if item.get("id")]
    accounted_evaluations = (
        len(details) + len(evaluated_but_retained)
        + len(uncertified_evaluations))
    evidence_ids = committed_ids + retained_ids + uncertified_ids
    return {
        "schema": "ai-vector-cleanroom.curve-refit-proposal/v3",
        "status": "proposed" if accepted else "no_change",
        "source": source_svg.name,
        "candidate": candidate_svg.name,
        "path_count_refit": accepted,
        "eligible_path_count": eligible,
        "optimizer_evaluated_path_count": optimizer_evaluated,
        "retained_identity_path_count": len(evaluated_but_retained),
        "anchors_before": totals_before,
        "anchors_after": totals_after,
        "anchors_removed": removed,
        "anchor_reduction_ratio": round(
            removed / float(max(1, totals_before)), 6),
        "designer_anchors_before": designer_totals_before,
        "designer_anchors_after": designer_totals_after,
        "designer_anchors_removed": designer_removed,
        "designer_anchor_reduction_ratio": round(
            designer_removed / float(max(1, designer_totals_before)), 6),
        "optimization_basis": "geometry_only",
        "uses_colour_or_pixel_similarity": False,
        "error_budget_percent": error_budget_percent,
        "error_contract": {
            "primary": "p95_error_percent <= error_budget_percent",
            "tail_max": "max_error_percent <= 3 * error_budget_percent",
            "tail_share": "share(error > budget) <= 0.05",
            "salient_corners": (
                "corner_error_percent <= 2 * error_budget_percent"
            ),
        },
        "lexicographic_objective": [
            "preserve_topology_hard_constraint",
            "robust_geometric_error_contract_hard_constraint",
            "prefer_native_then_low_sided_analytic_primitives",
            "minimize_designer_anchor_count",
            "minimize_anchor_count",
            "minimize_fragment_and_segment_count",
        ],
        "protected_gradient_objects": {
            "policy": (
                "fail_closed_without_original_ownership_mask_revalidation"
            ),
            "ownership_mask_revalidation_performed": False,
            "skip_reason": _GRADIENT_OBJECT_SKIP_REASON,
            "skipped_path_count": skipped.get(
                _GRADIENT_OBJECT_SKIP_REASON, 0),
        },
        "skipped": dict(sorted(skipped.items())),
        "details": details,
        "evaluated_but_retained": evaluated_but_retained,
        "uncertified_evaluations": uncertified_evaluations,
        "stable_id_normalization": {
            "schema": (
                "ai-vector-cleanroom.curve-refit-stable-id-normalization/v1"
            ),
            "source_svg_sha256": source_svg_sha256,
            "source_svg_digest_scope": "exact_source_svg_bytes",
            "source_path_count": source_path_count,
            "optimizer_evaluated_path_count": optimizer_evaluated,
            "record_count": len(stable_id_records),
            "assigned_id_count": sum(
                1 for item in stable_id_records
                if item["assignment_applied"]),
            "existing_id_preserved_count": sum(
                1 for item in stable_id_records
                if not item["assignment_applied"]),
            "all_optimizer_evaluations_authenticated": (
                len(stable_id_records) == optimizer_evaluated),
            "assigned_ids_unique": len(stable_id_records) == len({
                item["assigned_id"] for item in stable_id_records}),
            "candidate_all_svg_ids_unique": True,
            "candidate_svg_sha256": hashlib.sha256(
                candidate_bytes).hexdigest(),
            "records": stable_id_records,
        },
        "evaluation_evidence_integrity": {
            "schema": (
                "ai-vector-cleanroom.curve-refit-evaluation-evidence/v1"
            ),
            "optimizer_evaluated_path_count": optimizer_evaluated,
            "committed_detail_count": len(details),
            "retained_identity_detail_count": len(
                evaluated_but_retained),
            "uncertified_evaluation_count": len(
                uncertified_evaluations),
            "accounted_evaluation_count": accounted_evaluations,
            "all_optimizer_results_accounted": (
                optimizer_evaluated == accounted_evaluations),
            "evidence_ids_unique": len(evidence_ids) == len(
                set(evidence_ids)),
            "committed_and_retained_ids_disjoint": not (
                set(committed_ids) & set(retained_ids)),
            "retained_identity_contract": {
                "outcome": "retained_identity_minimum",
                "economy_reason": _RETAINED_IDENTITY_ECONOMY_REASON,
                "requires_stable_svg_id": True,
                "requires_exact_path_data_sha256_match": True,
                "requires_complete_geometry_error_contract": True,
                "requires_complete_lexicographic_objective": True,
            },
        },
        "parameters": {
            "tolerance": tolerance,
            "line_tolerance": line_tolerance,
            "legacy_tolerance_role": (
                "API compatibility only; it does not rank candidates"
            ),
            "error_budget_percent": error_budget_percent,
            "sample_step": sample_step,
            "minimum_nodes": minimum_nodes,
            "minimum_reduction_ratio": minimum_reduction_ratio,
            "maximum_segments": maximum_segments,
        },
        "transaction_note": (
            "This is a proposal only; commit requires whole-image render and "
            "source fidelity validation."
        ),
        "optimality_note": (
            "The selected result is the deterministic lexicographic minimum "
            "among evaluated candidates, not a claim of a globally analytic "
            "minimum over all possible SVG curves."
        ),
    }


__all__ = ["propose_svg_curve_refit"]
