# -*- coding: utf-8 -*-
"""Read-only layered editability audit for generated SVG files.

The audit keeps separate whether common edits have dependable SVG handles
(automation readiness), how costly freeform outline reshaping would be
(redraw complexity), and non-outline navigation/selection friction.  These
generic structural estimates are not a count of the user's original human
tasks and do not establish an "80% time saved" claim.

Only the Python standard library is used.  The SVG is never rewritten.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET


AUDIT_SCHEMA = "ai-vector-cleanroom.editability/v2"
_DRAWABLE_TAGS = {
    "path", "circle", "rect", "ellipse", "line", "polyline", "polygon",
    "text", "use",
}
_NATIVE_TAGS = {"circle", "rect", "ellipse", "line", "polyline", "polygon"}
_PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_COMMAND_ARITY = {
    "A": 7, "C": 6, "H": 1, "L": 2, "M": 2, "Q": 4,
    "S": 4, "T": 2, "V": 1, "Z": 0,
}
_COMMAND_TOKEN = re.compile(r"^[A-Za-z]$")
_URL_PAINT = re.compile(r"^url\s*\(", re.IGNORECASE)
_URL_REFERENCE = re.compile(
    r"^url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)$", re.IGNORECASE)
_TRAILING_NUMBER = re.compile(r"(?:[-_ ]?\d+)+$")
_GENERIC_LAYER_WORDS = {
    "color", "colour", "fill", "gradient", "layer", "paint", "path",
    "paths", "shape", "shapes", "stroke", "strokes",
}
_COLOR_WORDS = {
    "aqua", "aquamarine", "beige", "black", "blue", "brown", "coral",
    "crimson", "cyan", "dark", "fuchsia", "gold", "gray", "green",
    "grey", "indigo", "ivory", "lavender", "light", "lime", "magenta",
    "maroon", "navy", "olive", "orange", "orchid", "pink", "plum",
    "purple", "red", "salmon", "silver", "tan", "teal", "turquoise",
    "violet", "white", "yellow",
}


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _local_attribute(element: ET.Element, wanted: str) -> str:
    for name, value in element.attrib.items():
        if _local_name(name) == wanted:
            return value
    return ""


def _style_map(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in raw.split(";"):
        if ":" not in declaration:
            continue
        name, value = declaration.split(":", 1)
        name = name.strip().lower()
        if name:
            result[name] = value.strip()
    return result


def _presentation(element: ET.Element,
                  inherited: Mapping[str, str]) -> dict[str, str]:
    """Resolve the small subset of inherited presentation data we inspect."""
    result = dict(inherited)
    for name in ("fill", "stroke", "stroke-width"):
        if name in element.attrib:
            result[name] = element.attrib[name].strip()
    # Inline style has higher priority than presentation attributes.
    for name, value in _style_map(element.attrib.get("style", "")).items():
        if name in ("fill", "stroke", "stroke-width"):
            result[name] = value
    return result


def _solid_paint(value: str) -> str | None:
    paint = re.sub(r"\s+", "", (value or "").strip().lower())
    if (not paint or paint in {
            "none", "transparent", "currentcolor", "context-fill",
            "context-stroke", "inherit", "initial", "unset",
    } or _URL_PAINT.match(paint)):
        return None
    return paint


def _has_visible_stroke(presentation: Mapping[str, str]) -> bool:
    if _solid_paint(presentation.get("stroke", "")) is None:
        return False
    width = presentation.get("stroke-width", "").strip()
    if not width:
        return True
    match = re.match(r"^[-+]?(?:\d+(?:\.\d*)?|\.\d+)", width)
    return not match or float(match.group(0)) != 0.0


def _path_metrics_detailed(path_data: str) -> tuple[int, int, int, int]:
    """Return commands, subpaths, anchors and explicit Bezier controls.

    Repeated parameter sets are counted as repeated commands, even when SVG
    omits the command letter.  That makes ``L 1 1 2 2`` count as two line
    commands and gives a better editing-burden estimate than letter counting.
    """
    tokens = _PATH_TOKEN_RE.findall(path_data or "")
    index = 0
    current = ""
    commands = 0
    subpaths = 0
    anchors = 0
    control_points = 0

    while index < len(tokens):
        token = tokens[index]
        if _COMMAND_TOKEN.match(token):
            current = token
            index += 1
            if current.upper() == "Z":
                commands += 1
                current = ""
                continue
        elif not current:
            index += 1
            continue

        upper = current.upper()
        arity = _COMMAND_ARITY.get(upper)
        if arity is None:
            current = ""
            continue

        first_parameter_set = True
        consumed = False
        while index + arity <= len(tokens):
            parameter_set = tokens[index:index + arity]
            if any(_COMMAND_TOKEN.match(item) for item in parameter_set):
                break
            commands += 1
            anchors += 1
            control_points += {"C": 2, "S": 1, "Q": 1}.get(upper, 0)
            if upper == "M" and first_parameter_set:
                subpaths += 1
            index += arity
            consumed = True
            first_parameter_set = False
            # Extra moveto coordinate pairs are implicit lineto commands.
            if upper == "M":
                upper = "L"
                current = "L" if current.isupper() else "l"
                arity = _COMMAND_ARITY["L"]
            if index >= len(tokens) or _COMMAND_TOKEN.match(tokens[index]):
                break
        if not consumed:
            # Invalid/truncated path data: make progress without inventing a
            # command.  Generated SVG should not normally take this branch.
            if index < len(tokens) and not _COMMAND_TOKEN.match(tokens[index]):
                index += 1
            else:
                current = ""

    return commands, subpaths, anchors, control_points


def _path_metrics(path_data: str) -> tuple[int, int, int]:
    """Compatibility view used by structural counting callers."""
    commands, subpaths, anchors, _controls = _path_metrics_detailed(path_data)
    return commands, subpaths, anchors


def _looks_like_color_layer(group: ET.Element) -> bool:
    name = (_local_attribute(group, "label") or group.attrib.get("id", ""))
    normalized = _TRAILING_NUMBER.sub("", name.strip().lower())
    words = [word for word in re.split(r"[^a-z]+", normalized) if word]
    if words and all(word in (_GENERIC_LAYER_WORDS | _COLOR_WORDS)
                     for word in words):
        return True
    if re.match(r"^(?:#|rgb\s*\(|hsl\s*\()", normalized):
        return True
    # An unnamed Inkscape layer carrying a layer-wide paint is structural
    # colour separation, not a semantic object group.
    group_mode = _local_attribute(group, "groupmode").lower()
    has_layer_paint = any(name in group.attrib for name in ("fill", "stroke"))
    return not words and group_mode == "layer" and has_layer_paint


def _looks_like_semantic_group(group: ET.Element) -> bool:
    """Recognise selectable object groups, excluding paint-stack scaffolding."""
    if group.attrib.get("data-group-mode", "").lower() == "actual-dom":
        return True
    if _looks_like_color_layer(group):
        return False
    label = _local_attribute(group, "label").strip()
    name = label or group.attrib.get("id", "").strip()
    normalized = _TRAILING_NUMBER.sub("", name.lower())
    words = [word for word in re.split(r"[^a-z]+", normalized) if word]
    if not words:
        return False
    # These are container mechanics, not designer-facing semantic objects.
    structural_words = _GENERIC_LAYER_WORDS | {
        "graph", "root", "scene", "stack", "vector",
    }
    return not all(word in (structural_words | _COLOR_WORDS) for word in words)


def _number_from(source: Mapping[str, object], *names: str) -> int | None:
    for name in names:
        value = source.get(name)
        if isinstance(value, bool) or value is None:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= 0:
            return number
    return None


def _count_from(source: Mapping[str, object], name: str) -> int | None:
    """Read either a non-negative count or the length of a result list."""
    value = source.get(name)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return _number_from(source, name)


def _named_operation_evidence(
        supplied: Mapping[str, object]) -> dict[str, object]:
    """Expose a separate operation audit without turning it into human proof."""
    raw = supplied.get("designer_operations")
    if not isinstance(raw, Mapping):
        return {
            "status": "not_supplied",
            "audit_schema": None,
            "structural_checks_passed": None,
            "structural_checks_total": None,
            "all_structural_checks_passed": None,
            "scope_note": (
                "Named SVG-operation evidence was not supplied to this audit. "
                "No human-task result may be inferred."
            ),
        }

    summary_value = raw.get("summary")
    summary = summary_value if isinstance(summary_value, Mapping) else raw
    passed = _count_from(summary, "passed")
    total = _count_from(summary, "total_operations")
    all_passed = (
        passed == total if passed is not None and total is not None and total > 0
        else None
    )
    schema = raw.get("schema")
    return {
        "status": "reported_by_separate_structural_audit",
        "audit_schema": str(schema) if schema else None,
        "structural_checks_passed": passed,
        "structural_checks_total": total,
        "all_structural_checks_passed": all_passed,
        "scope_note": (
            "These counts cover encoded native-resource and SVG-DOM checks only. "
            "They are not the user's original human task set, timed editing, "
            "semantic approval, or final designer acceptance."
        ),
    }


def _embedded_metadata_object(root: ET.Element,
                              element_id: str) -> dict[str, object]:
    for element in root.iter():
        if (_local_name(element.tag) != "metadata"
                or element.attrib.get("id") != element_id
                or not (element.text or "").strip()):
            continue
        try:
            value = json.loads(element.text or "")
        except (json.JSONDecodeError, TypeError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _load_report_or_stats(value: object, svg_path: Path) -> dict[str, object]:
    if value is None:
        sibling = svg_path.with_name("report.json")
        if not sibling.is_file():
            return {}
        value = sibling
    if isinstance(value, (str, Path)):
        loaded = json.loads(Path(value).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("report JSON must contain an object")
        return loaded
    if isinstance(value, Mapping):
        return dict(value)

    # Support the engine's stats dataclass without importing the engine.
    aliases = {
        "paths": ("paths", "n_paths"),
        "native_primitives": ("native_primitives", "n_native"),
        "strokes": ("strokes", "n_strokes"),
        "gradients": ("gradients", "n_gradients"),
        "nodes_total": ("nodes_total", "nodes", "n_nodes"),
        "groups": ("groups", "colors"),
    }
    result: dict[str, object] = {}
    for destination, names in aliases.items():
        for name in names:
            if hasattr(value, name):
                result[destination] = getattr(value, name)
                break
    return result


def _designer_certified_gradient_paths(
        supplied: Mapping[str, object], svg_sha256: str,
        root: ET.Element,
        path_elements: list[tuple[ET.Element, dict[str, str]]],
) -> dict[str, object]:
    """Authenticate compound paths that are one source-topology colour field.

    A generic many-subpath warning assumes unrelated shapes were coupled into
    one path.  It is inapplicable only when the designer audit, for these exact
    SVG bytes, already proved that the path is one ownership-closed gradient
    object whose source topology and final geometry are authoritative.
    """
    result: dict[str, object] = {
        "available": False,
        "authoritative": False,
        "certified_path_ids": [],
        "failure_reasons": [],
        "invalid_path_ids": [],
        "scope": (
            "exact_final_svg_source_topology_gradient_compound_paths"),
    }
    quality = supplied.get("designer_quality")
    if not isinstance(quality, Mapping):
        return result
    result["available"] = True
    failures: list[str] = []
    source = quality.get("source")
    if (not isinstance(source, Mapping)
            or str(source.get("sha256") or "").lower()
            != svg_sha256.lower()):
        failures.append("designer_quality_svg_sha256_mismatch")
    gradient_gate = quality.get("gradient_object_gate")
    gradient_gate = gradient_gate if isinstance(gradient_gate, Mapping) else {}
    source_evidence = gradient_gate.get("source_space_field_evidence")
    source_evidence = (source_evidence
                       if isinstance(source_evidence, Mapping) else {})
    if source_evidence.get("authoritative") is not True:
        failures.append("source_topology_gradient_evidence_not_authoritative")
    objects = source_evidence.get("objects")
    if not isinstance(objects, list):
        failures.append("source_topology_gradient_objects_missing")
        objects = []

    curve_gate = quality.get("curve_economy_gate")
    curve_gate = curve_gate if isinstance(curve_gate, Mapping) else {}
    optimizer = curve_gate.get("optimizer_economy_evidence")
    optimizer = optimizer if isinstance(optimizer, Mapping) else {}
    gradient_optimizer = optimizer.get("gradient")
    gradient_optimizer = (gradient_optimizer
                          if isinstance(gradient_optimizer, Mapping) else {})
    if gradient_optimizer.get("authoritative") is not True:
        failures.append("gradient_optimizer_evidence_not_authoritative")
    certified_optimizer_ids = {
        str(value) for value in gradient_optimizer.get(
            "certified_path_ids", [])
        if isinstance(value, str) and value
    }

    parents = {child: parent for parent in root.iter() for child in parent}

    def inherited(element: ET.Element, name: str) -> str | None:
        node: ET.Element | None = element
        while node is not None:
            value = node.attrib.get(name)
            if value is not None:
                return value.strip()
            node = parents.get(node)
        return None

    paths_by_id: dict[str, list[tuple[ET.Element, dict[str, str]]]] = {}
    for element, presentation in path_elements:
        identifier = element.attrib.get("id", "").strip()
        if identifier:
            paths_by_id.setdefault(identifier, []).append(
                (element, presentation))

    certified: list[str] = []
    invalid: list[str] = []
    seen_candidates: set[str] = set()
    for index, raw in enumerate(objects):
        item = raw if isinstance(raw, Mapping) else {}
        if item.get("final_element") != "path":
            continue
        identifier = str(item.get("final_drawable_id") or "")
        candidate = str(item.get("candidate_id") or "")
        item_failures: list[str] = []
        if not candidate or candidate in seen_candidates:
            item_failures.append("candidate_identity_missing_or_duplicate")
        seen_candidates.add(candidate)
        if (item.get("passed") is not True
                or item.get("economy_certificate_passed") is not True):
            item_failures.append("gradient_object_certificate_not_passed")
        if not identifier or identifier not in certified_optimizer_ids:
            item_failures.append("gradient_path_not_optimizer_certified")
        matches = paths_by_id.get(identifier, [])
        if len(matches) != 1:
            item_failures.append("final_gradient_path_id_not_unique")
        else:
            element, presentation = matches[0]
            fill_match = _URL_REFERENCE.fullmatch(
                str(presentation.get("fill") or "").strip())
            if (fill_match is None
                    or fill_match.group(1) != str(item.get("gradient_id") or "")):
                item_failures.append("gradient_resource_identity_mismatch")
            if inherited(element, "data-avc-gradient-object") != str(
                    item.get("gradient_object_id") or ""):
                item_failures.append("gradient_object_identity_mismatch")
        if item_failures:
            label = identifier or f"gradient-path-{index + 1}"
            invalid.append(label)
            failures.extend(f"{label}:{reason}" for reason in item_failures)
        else:
            certified.append(identifier)

    authoritative = not failures
    result.update({
        "authoritative": authoritative,
        "certified_path_ids": sorted(set(certified)) if authoritative else [],
        "failure_reasons": failures[:60],
        "invalid_path_ids": invalid[:30],
    })
    return result


def _designer_certified_curve_paths(
        supplied: Mapping[str, object], svg_sha256: str,
        root: ET.Element,
        path_elements: list[tuple[ET.Element, dict[str, str]]],
) -> dict[str, object]:
    """Authenticate exact final-SVG paths covered by the curve optimiser.

    This deliberately consumes the fail-closed certificate emitted by
    ``designer_quality`` instead of reinterpreting a curve-refit proposal.  A
    path is excluded from generic redraw heuristics only when that certificate
    is authoritative for these exact SVG bytes, internally coherent, and tied
    to one unique final path ID.  Raw SVG complexity remains reported
    separately regardless of certification.
    """
    result: dict[str, object] = {
        "available": False,
        "authoritative": False,
        "certified_refit_path_ids": [],
        "certified_retained_identity_path_ids": [],
        "certified_path_ids": [],
        "failure_reasons": [],
        "invalid_path_ids": [],
        "scope": (
            "exact_final_svg_transaction_backed_curve_optimizer_paths"),
    }
    quality = supplied.get("designer_quality")
    if not isinstance(quality, Mapping):
        return result
    curve_gate = quality.get("curve_economy_gate")
    curve_gate = curve_gate if isinstance(curve_gate, Mapping) else {}
    optimizer = curve_gate.get("optimizer_economy_evidence")
    optimizer = optimizer if isinstance(optimizer, Mapping) else {}
    curve = optimizer.get("curve_refit")
    if not isinstance(curve, Mapping):
        return result
    result["available"] = True
    failures: list[str] = []
    invalid: list[str] = []

    source = quality.get("source")
    if (not isinstance(source, Mapping)
            or str(source.get("sha256") or "").lower()
            != svg_sha256.lower()):
        failures.append("designer_quality_svg_sha256_mismatch")
    if curve.get("available") is not True:
        failures.append("curve_optimizer_certificate_not_available")
    if curve.get("authoritative") is not True:
        failures.append("curve_optimizer_evidence_not_authoritative")
    if curve.get("scope") != (
            "transaction_backed_geometry_only_optimizer_economy_certificate"):
        failures.append("curve_optimizer_certificate_scope_invalid")
    if curve.get("transaction_status") != "committed":
        failures.append("curve_optimizer_transaction_not_committed")
    if curve.get("proposal_schema") != (
            "ai-vector-cleanroom.curve-refit-proposal/v3"):
        failures.append("curve_optimizer_proposal_schema_invalid")
    if curve.get("proposal_status") != "proposed":
        failures.append("curve_optimizer_proposal_status_invalid")
    if curve.get("failure_reasons") != []:
        failures.append("curve_optimizer_certificate_has_failures")
    if curve.get("invalid_detail_ids") != []:
        failures.append("curve_optimizer_certificate_has_invalid_details")

    def exact_id_list(name: str) -> list[str]:
        value = curve.get(name)
        if (not isinstance(value, list)
                or any(not isinstance(item, str) or not item for item in value)
                or len(value) != len(set(value))):
            failures.append(f"curve_optimizer_{name}_invalid")
            return []
        return list(value)

    refit_ids = exact_id_list("certified_refit_path_ids")
    retained_ids = exact_id_list("certified_retained_identity_path_ids")
    certified_ids = exact_id_list("certified_path_ids")
    if set(refit_ids) & set(retained_ids):
        failures.append("curve_optimizer_certificate_classes_overlap")
    expected_ids = set(refit_ids) | set(retained_ids)
    if set(certified_ids) != expected_ids or len(certified_ids) != len(
            expected_ids):
        failures.append("curve_optimizer_certified_path_union_mismatch")
    try:
        certified_count = int(curve.get("certified_path_count"))
    except (TypeError, ValueError, OverflowError):
        certified_count = -1
    if certified_count != len(expected_ids):
        failures.append("curve_optimizer_certified_path_count_mismatch")

    uncertified = curve.get("uncertified_evaluation_ids")
    if (not isinstance(uncertified, list)
            or any(not isinstance(item, str) or not item for item in uncertified)
            or len(uncertified) != len(set(uncertified))
            or expected_ids.intersection(uncertified)):
        failures.append("curve_optimizer_uncertified_id_evidence_invalid")

    parents = {child: parent for parent in root.iter() for child in parent}

    def inherited(element: ET.Element, name: str) -> str | None:
        node: ET.Element | None = element
        while node is not None:
            value = node.attrib.get(name)
            if value is not None:
                return value.strip()
            node = parents.get(node)
        return None

    paths_by_id: dict[str, list[ET.Element]] = {}
    for element, _presentation in path_elements:
        identifier = element.attrib.get("id", "").strip()
        if identifier:
            paths_by_id.setdefault(identifier, []).append(element)

    for identifier in sorted(expected_ids):
        item_failures: list[str] = []
        matches = paths_by_id.get(identifier, [])
        if len(matches) != 1:
            item_failures.append("final_curve_path_id_not_unique")
        elif identifier in refit_ids:
            if inherited(matches[0], "data-avc-curve-refit") != (
                    "geometry-budgeted"):
                item_failures.append("committed_curve_refit_claim_missing")
        else:
            if inherited(matches[0], "data-avc-curve-refit"):
                item_failures.append("retained_identity_has_refit_claim")
            if inherited(matches[0], "data-avc-gradient-object"):
                item_failures.append("retained_identity_has_gradient_claim")
        if item_failures:
            invalid.append(identifier)
            failures.extend(
                f"{identifier}:{reason}" for reason in item_failures)

    authoritative = not failures
    result.update({
        "authoritative": authoritative,
        "certified_refit_path_ids": (
            sorted(refit_ids) if authoritative else []),
        "certified_retained_identity_path_ids": (
            sorted(retained_ids) if authoritative else []),
        "certified_path_ids": (
            sorted(expected_ids) if authoritative else []),
        "failure_reasons": failures[:80],
        "invalid_path_ids": invalid[:30],
    })
    return result


def _scaled_penalty(value: float, free: float, severe: float,
                    maximum: float) -> float:
    if value <= free:
        return 0.0
    if severe <= free:
        return maximum
    fraction = min(1.0, (value - free) / (severe - free))
    return maximum * fraction


def _single_drawable_selection(root, drawables, unique_ids):
    """Prove a direct selection target without inventing a semantic group."""
    evidence = {"available": False, "drawable_id": None,
                "basis": "single_identifiable_visible_drawable",
                "semantic_group_created": False}
    if len(drawables) != 1 or any(_local_name(e.tag) == "style" for e in root.iter()):
        return evidence
    element, presentation = drawables[0]
    identifier = element.get("id", "").strip()
    tag = _local_name(element.tag)
    if not identifier or identifier not in unique_ids or tag not in _NATIVE_TAGS | {"path"}:
        return evidence
    parents = {child: parent for parent in root.iter() for child in parent}
    node = element
    fill_visible = presentation.get("fill", "#000000").lower() not in {"none", "transparent"}
    stroke_visible = _has_visible_stroke(presentation)
    while node is not None:
        style = _style_map(node.get("style", ""))
        if (style.get("display", node.get("display", "")).lower() == "none"
                or style.get("visibility", node.get("visibility", "")).lower() in {"hidden", "collapse"}
                or node.get("class")):
            return evidence
        if any(style.get(key, node.get(key, "none")).lower() != "none"
               for key in ("clip-path", "mask", "filter")):
            return evidence
        try:
            from svg_bounds import parse_transform
            transform = parse_transform(style.get("transform", node.get("transform", "")))
            if abs(transform[0] * transform[3] - transform[1] * transform[2]) <= 1e-12:
                return evidence
            opacity = float(style.get("opacity", node.get("opacity", "1")))
            if not math.isfinite(opacity) or opacity <= 0:
                return evidence
            fill_visible &= float(style.get("fill-opacity", node.get("fill-opacity", "1"))) > 0
            stroke_visible &= float(style.get("stroke-opacity", node.get("stroke-opacity", "1"))) > 0
        except ValueError:
            return evidence
        node = parents.get(node)
    if not (fill_visible or stroke_visible) or (tag == "line" and not stroke_visible):
        return evidence
    try:
        def number(name):
            value = float(element.get(name, "0"))
            if not math.isfinite(value):
                raise ValueError("nonfinite native geometry")
            return value
        if tag == "circle":
            extent = number("r") > 0
        elif tag == "ellipse":
            extent = number("rx") > 0 and number("ry") > 0
        elif tag == "rect":
            extent = number("width") > 0 and number("height") > 0
        elif tag == "line":
            extent = number("x1") != number("x2") or number("y1") != number("y2")
        elif tag == "path":
            data = element.get("d", "")
            commands, subpaths, anchors = _path_metrics(data)
            values = [float(t) for t in _PATH_TOKEN_RE.findall(data) if not _COMMAND_TOKEN.match(t)]
            extent = (subpaths == 1 and commands >= 2 and anchors >= 2 and values
                      and all(math.isfinite(v) for v in values) and max(values) > min(values)
                      and (stroke_visible or "z" in data.lower()))
        else:
            values = [float(t) for t in re.split(r"[\s,]+", element.get("points", "").strip()) if t]
            extent = (len(values) >= (6 if tag == "polygon" else 4)
                      and len(values) % 2 == 0 and all(math.isfinite(v) for v in values)
                      and (len(set(values[::2])) > 1 or len(set(values[1::2])) > 1))
    except (ValueError, OverflowError):
        return evidence
    if extent:
        evidence.update(available=True, drawable_id=identifier, drawable_type=tag)
    return evidence


def audit_editability(svg_path: str | Path,
                      report_or_stats: object = None) -> dict[str, object]:
    """Audit one generated SVG and return a JSON-serializable result.

    ``report_or_stats`` may be a report mapping, a report JSON path, the
    engine's stats object, or ``None``.  With ``None``, a sibling report.json
    is used when present.  Direct SVG measurements remain separately visible
    so a stale report cannot conceal structural complexity.
    """
    path = Path(svg_path)
    payload = path.read_bytes()
    svg_sha256 = hashlib.sha256(payload).hexdigest()
    root = ET.fromstring(payload)
    supplied = _load_report_or_stats(report_or_stats, path)
    named_operation_evidence = _named_operation_evidence(supplied)
    scene_graph_metadata = _embedded_metadata_object(
        root, "scene-graph-metadata")

    groups: list[ET.Element] = []
    drawables: list[tuple[ET.Element, dict[str, str]]] = []
    drawable_paint_roles: dict[ET.Element, dict[str, str]] = {}
    gradient_elements: list[ET.Element] = []

    def walk(element: ET.Element, inherited: Mapping[str, str],
             in_defs: bool = False,
             inherited_paint_roles: Mapping[str, str] | None = None) -> None:
        tag = _local_name(element.tag)
        now_in_defs = in_defs or tag == "defs"
        presentation = _presentation(element, inherited)
        paint_roles = dict(inherited_paint_roles or {})
        inline_style = _style_map(element.attrib.get("style", ""))
        for property_name in ("fill", "stroke"):
            if property_name in inline_style:
                declared_value = inline_style[property_name].strip().lower()
                declared_here = True
            elif property_name in element.attrib:
                declared_value = element.attrib[property_name].strip().lower()
                declared_here = True
            else:
                declared_value = ""
                declared_here = False
            if declared_here and declared_value not in {"inherit", "unset"}:
                role_id = _local_attribute(
                    element, "data-paint-role-" + property_name).strip()
                if role_id:
                    paint_roles[property_name] = role_id
                else:
                    # A local paint declaration severs the parent's control.
                    # Do not credit an ancestor role for a child override.
                    paint_roles.pop(property_name, None)
        if tag in {"linearGradient", "radialGradient"}:
            gradient_elements.append(element)
        if not now_in_defs:
            if tag == "g":
                groups.append(element)
            if tag in _DRAWABLE_TAGS:
                drawables.append((element, presentation))
                drawable_paint_roles[element] = paint_roles
        for child in element:
            walk(child, presentation, now_in_defs, paint_roles)

    walk(root, {"fill": "#000000", "stroke": "none", "stroke-width": "1"})

    path_elements = [item for item in drawables if _local_name(item[0].tag) == "path"]
    certified_gradient_paths = _designer_certified_gradient_paths(
        supplied, svg_sha256, root, path_elements)
    certified_gradient_path_ids = set(
        certified_gradient_paths.get("certified_path_ids", []))
    certified_curve_paths = _designer_certified_curve_paths(
        supplied, svg_sha256, root, path_elements)
    certified_curve_path_ids = set(
        certified_curve_paths.get("certified_path_ids", []))
    certified_optimizer_path_ids = (
        certified_gradient_path_ids | certified_curve_path_ids)
    command_counts: list[int] = []
    subpath_counts: list[int] = []
    control_point_counts: list[int] = []
    path_metric_records: list[dict[str, object]] = []
    estimated_path_anchors = 0
    for element, _presentation_data in path_elements:
        commands, subpaths, anchors, controls = _path_metrics_detailed(
            element.attrib.get("d", ""))
        command_counts.append(commands)
        subpath_counts.append(subpaths)
        control_point_counts.append(controls)
        estimated_path_anchors += anchors
        path_metric_records.append({
            "id": element.attrib.get("id", "").strip() or None,
            "commands": commands,
            "subpaths": subpaths,
            "anchors": anchors,
            "controls": controls,
        })

    native_breakdown = {
        tag: sum(1 for element, _ in drawables if _local_name(element.tag) == tag)
        for tag in sorted(_NATIVE_TAGS)
    }
    svg_native_count = sum(native_breakdown.values())
    svg_stroke_count = sum(
        1 for _element, presentation in drawables
        if _has_visible_stroke(presentation)
    )
    gradient_ids = {
        element.attrib["id"] for element in gradient_elements
        if element.attrib.get("id")
    }
    gradient_resource_count = len(gradient_ids) + sum(
        1 for element in gradient_elements if not element.attrib.get("id")
    )

    solid_paints: set[str] = set()
    for _element, presentation in drawables:
        for property_name in ("fill", "stroke"):
            paint = _solid_paint(presentation.get(property_name, ""))
            if paint is not None:
                solid_paints.add(paint)

    all_ids = [element.get("id", "").strip() for element in root.iter()
               if element.get("id", "").strip()]
    id_counts = Counter(all_ids)
    unique_ids = {identifier for identifier, count in id_counts.items() if count == 1}
    object_ids = [
        element.attrib["id"] for element, _ in drawables
        if element.attrib.get("id", "").strip() in unique_ids
    ]
    color_layer_count = sum(_looks_like_color_layer(group) for group in groups)
    semantic_group_count = len(groups) - color_layer_count
    selectable_semantic_groups = [
        group for group in groups if _looks_like_semantic_group(group)
    ]
    actual_dom_groups = [
        group for group in groups
        if group.attrib.get("data-group-mode", "").lower() == "actual-dom"
    ]
    manifest_only_group_count = _number_from(
        scene_graph_metadata, "manifest_only_group_count")
    drawable_elements = {element for element, _ in drawables}
    semantically_grouped_drawables = {
        element
        for group in selectable_semantic_groups
        for element in group.iter()
        if element in drawable_elements
    }
    paint_role_ids: set[str] = set()
    for element in root.iter():
        for attribute, value in element.attrib.items():
            if _local_name(attribute).startswith("data-paint-role-") and value:
                paint_role_ids.add(value)
    paint_role_drawables = {
        element for element, roles in drawable_paint_roles.items() if roles
    }
    only_color_layers = (
        bool(groups) and color_layer_count == len(groups)
        and semantic_group_count == 0
    )

    reported_paths = _number_from(supplied, "paths", "n_paths")
    reported_native = _number_from(supplied, "native_primitives", "n_native")
    if reported_native is None:
        # These six fields are disjoint.  ``native_polygonal_shapes`` is a
        # convenience aggregate and must not be added on top of polygon and
        # polyline counts.
        reported_native_parts = [
            _number_from(supplied, name) for name in (
                "native_circles", "native_rectangles", "native_ellipses",
                "native_lines", "native_polylines", "native_polygons",
            )
        ]
        if any(value is not None for value in reported_native_parts):
            reported_native = sum(value or 0 for value in reported_native_parts)
    reported_strokes = _number_from(supplied, "strokes", "n_strokes")
    reported_gradients = _number_from(supplied, "gradients", "n_gradients")
    reported_nodes = _number_from(supplied, "nodes_total", "nodes", "n_nodes")

    path_count = len(path_elements)
    native_count = reported_native if reported_native is not None else svg_native_count
    stroke_count = reported_strokes if reported_strokes is not None else svg_stroke_count
    gradient_count = (
        reported_gradients if reported_gradients is not None
        else gradient_resource_count
    )
    svg_estimated_nodes = estimated_path_anchors + svg_native_count
    # A stale or differently-scoped report must never hide complexity that is
    # directly visible in the delivered SVG.  The larger defensible count is
    # used for the gate while both source values remain in the evidence.
    node_count = max(reported_nodes or 0, svg_estimated_nodes)
    if reported_nodes is None:
        node_count_source = "svg_estimate"
    elif reported_nodes >= svg_estimated_nodes:
        node_count_source = "report_or_stats"
    else:
        node_count_source = "conservative_svg_estimate_over_report"

    ordered_commands = sorted(command_counts)
    if ordered_commands:
        median_commands = float(statistics.median(ordered_commands))
        p95_index = max(0, math.ceil(0.95 * len(ordered_commands)) - 1)
        p95_commands = ordered_commands[p95_index]
        max_commands = ordered_commands[-1]
        total_commands = sum(ordered_commands)
        max_command_share = max_commands / total_commands if total_commands else 0.0
    else:
        median_commands = 0.0
        p95_commands = 0
        max_commands = 0
        total_commands = 0
        max_command_share = 0.0

    total_subpaths = sum(subpath_counts)
    total_control_points = sum(control_point_counts)
    max_control_points = max(control_point_counts, default=0)
    multi_subpath_paths = sum(count > 1 for count in subpath_counts)
    max_subpaths = max(subpath_counts, default=0)
    generic_path_metric_records = [
        item for item in path_metric_records
        if item.get("id") not in certified_optimizer_path_ids
    ]
    certified_path_metric_records = [
        item for item in path_metric_records
        if item.get("id") in certified_optimizer_path_ids
    ]
    generic_path_count = len(generic_path_metric_records)
    generic_path_anchors = sum(
        int(item["anchors"]) for item in generic_path_metric_records)
    certified_path_anchors = sum(
        int(item["anchors"]) for item in certified_path_metric_records)
    certified_total_commands = sum(
        int(item["commands"]) for item in certified_path_metric_records)
    certified_total_subpaths = sum(
        int(item["subpaths"]) for item in certified_path_metric_records)
    certified_total_control_points = sum(
        int(item["controls"]) for item in certified_path_metric_records)
    certified_max_commands = max(
        (int(item["commands"]) for item in certified_path_metric_records),
        default=0)
    certified_max_subpaths = max(
        (int(item["subpaths"]) for item in certified_path_metric_records),
        default=0)
    # Preserve any conservative report-only node surplus in the uncertified
    # scope.  Exact SVG path anchors can be subtracted safely; opaque report
    # counts cannot be attributed to certified paths and therefore remain a
    # generic burden instead of being silently forgiven.
    report_only_node_surplus = max(
        0, (reported_nodes or 0) - svg_estimated_nodes)
    generic_node_count = (
        generic_path_anchors + report_only_node_surplus
        if certified_optimizer_path_ids else node_count)
    generic_command_counts = sorted(
        int(item["commands"]) for item in generic_path_metric_records)
    generic_subpath_counts = [
        int(item["subpaths"]) for item in generic_path_metric_records]
    generic_control_point_counts = [
        int(item["controls"]) for item in generic_path_metric_records]
    generic_total_commands = sum(generic_command_counts)
    generic_total_control_points = sum(generic_control_point_counts)
    generic_max_control_points = max(generic_control_point_counts, default=0)
    if generic_command_counts:
        generic_median_commands = float(statistics.median(
            generic_command_counts))
        generic_p95_index = max(
            0, math.ceil(0.95 * len(generic_command_counts)) - 1)
        generic_p95_commands = generic_command_counts[generic_p95_index]
        generic_max_commands = generic_command_counts[-1]
        generic_max_command_share = (
            generic_max_commands / generic_total_commands
            if generic_total_commands else 0.0)
    else:
        generic_median_commands = 0.0
        generic_p95_commands = 0
        generic_max_commands = 0
        generic_max_command_share = 0.0
    generic_coupling_max_subpaths = max(
        generic_subpath_counts,
        default=0)
    certified_compound_records = [
        item for item in path_metric_records
        if item.get("id") in certified_gradient_path_ids
        and int(item["subpaths"]) > 20
    ]
    drawable_count = len(drawables)
    object_id_count = len(object_ids)

    object_id_coverage = object_id_count / drawable_count if drawable_count else 0.0
    semantic_group_coverage = (
        len(semantically_grouped_drawables) / drawable_count
        if drawable_count else 0.0
    )
    direct_selection = _single_drawable_selection(root, drawables, unique_ids)
    selection_coverage = (1.0 if direct_selection["available"]
                          else semantic_group_coverage)
    needs_semantic_grouping = only_color_layers and not direct_selection["available"]
    paint_role_coverage = (
        len(paint_role_drawables) / drawable_count if drawable_count else 0.0
    )

    # Automation readiness rewards dependable handles. It intentionally does
    # not cancel outline complexity: a logo can be excellent for recolouring,
    # hiding decorations and changing a native ring while still being costly
    # to redraw point by point.
    automation_components: dict[str, float] = {
        "object_identity": 30.0 * min(1.0, object_id_coverage / 0.80)
        if drawable_count else 0.0,
        "semantic_selection": 25.0 * min(1.0, selection_coverage / 0.40)
        if drawable_count else 0.0,
        "paint_roles": (
            20.0 * min(1.0, paint_role_coverage / 0.80)
            if paint_role_ids else (8.0 if solid_paints or gradient_elements else 0.0)
        ),
        "native_edit_handles": (
            (5.0 if svg_native_count else 0.0)
            + (5.0 if svg_stroke_count else 0.0)
            + (5.0 if gradient_resource_count else 0.0)
        ),
    }
    isolation_score = 10.0
    if generic_max_commands >= 500:
        isolation_score -= 4.0
    if generic_coupling_max_subpaths > 20:
        isolation_score -= 4.0
    if generic_max_command_share > 0.45:
        isolation_score -= 2.0
    automation_components["object_isolation"] = max(0.0, isolation_score)
    automation_score = round(min(100.0, sum(automation_components.values())), 1)
    automation_status = (
        "ready_for_common_operations" if automation_score >= 80.0
        else "partially_ready" if automation_score >= 55.0
        else "limited"
    )

    # The delivered SVG's raw structure is always retained below.  Only paths
    # bound to an authoritative exact-SVG optimiser certificate are removed
    # from this generic redraw scope.  Correlated warnings are still combined
    # by maximum within each family; no thresholds are relaxed.
    def build_risk_indicators(
            *, scope_path_count: int, scope_node_count: int,
            scope_control_points: int, scope_median_commands: float,
            scope_p95_commands: int, scope_max_commands: int,
            scope_max_control_points: int, scope_max_subpaths: int,
            scope_max_command_share: float) -> dict[str, float]:
        indicators: dict[str, float] = {}

        def add_scaled(name: str, value: float, free: float, severe: float,
                       maximum: float) -> None:
            penalty = _scaled_penalty(value, free, severe, maximum)
            if penalty:
                indicators[name] = penalty

        add_scaled("many_paths", scope_path_count, 40, 300, 22)
        add_scaled("many_nodes", scope_node_count, 500, 5000, 24)
        add_scaled("many_bezier_control_points", scope_control_points,
                   500, 5000, 18)
        add_scaled("excessive_group_navigation", len(groups), 60, 180, 6)
        add_scaled("high_median_path_commands", scope_median_commands,
                   40, 150, 8)
        add_scaled("high_p95_path_commands", scope_p95_commands, 120, 450, 10)
        add_scaled("single_very_complex_path", scope_max_commands,
                   250, 800, 12)
        add_scaled("single_path_many_bezier_controls",
                   scope_max_control_points, 100, 1200, 10)
        add_scaled("many_subpaths_in_one_path", scope_max_subpaths,
                   20, 100, 6)
        add_scaled("path_command_concentration", scope_max_command_share,
                   0.45, 0.85, 6)
        if drawable_count >= 20 and object_id_count == 0:
            indicators["no_object_ids"] = 8.0
        elif drawable_count >= 50 and object_id_coverage < 0.10:
            indicators["very_low_object_id_coverage"] = 5.0
        if needs_semantic_grouping:
            indicators["color_layers_without_semantic_groups"] = 10.0
        return indicators

    raw_risk_indicators = build_risk_indicators(
        scope_path_count=path_count,
        scope_node_count=node_count,
        scope_control_points=total_control_points,
        scope_median_commands=median_commands,
        scope_p95_commands=p95_commands,
        scope_max_commands=max_commands,
        scope_max_control_points=max_control_points,
        scope_max_subpaths=max_subpaths,
        scope_max_command_share=max_command_share,
    )
    risk_indicators = build_risk_indicators(
        scope_path_count=generic_path_count,
        scope_node_count=generic_node_count,
        scope_control_points=generic_total_control_points,
        scope_median_commands=generic_median_commands,
        scope_p95_commands=generic_p95_commands,
        scope_max_commands=generic_max_commands,
        scope_max_control_points=generic_max_control_points,
        scope_max_subpaths=generic_coupling_max_subpaths,
        scope_max_command_share=generic_max_command_share,
    )

    def family_max(*names: str) -> float:
        return max((risk_indicators.get(name, 0.0) for name in names), default=0.0)

    all_penalty_families = {
        "geometry_volume": family_max(
            "many_paths", "many_nodes", "many_bezier_control_points"),
        "local_reshape": family_max(
            "high_median_path_commands", "high_p95_path_commands",
            "single_very_complex_path", "many_subpaths_in_one_path",
            "path_command_concentration", "single_path_many_bezier_controls",
        ),
        "navigation": family_max("excessive_group_navigation"),
        "selection_identity": family_max(
            "no_object_ids", "very_low_object_id_coverage"),
        "semantic_structure": family_max(
            "color_layers_without_semantic_groups"),
    }
    all_penalty_families = {
        name: value for name, value in all_penalty_families.items() if value
    }
    outline_penalty_families = {
        name: value for name, value in all_penalty_families.items()
        if name in {"geometry_volume", "local_reshape"}
    }
    workflow_penalty_families = {
        name: value for name, value in all_penalty_families.items()
        if name in {"navigation", "selection_identity", "semantic_structure"}
    }
    redraw_burden = round(sum(outline_penalty_families.values()), 1)
    workflow_burden = round(sum(workflow_penalty_families.values()), 1)
    score = round(max(0.0, 100.0 - redraw_burden), 1)
    workflow_ease = round(max(0.0, 100.0 - workflow_burden), 1)
    # Preserve the earlier conservative gate without mislabelling its mixed
    # score as outline redraw ease.  The two component axes remain independent
    # and the combined value exists only as an acceptance guardrail.
    combined_structural_ease = round(
        max(0.0, 100.0 - redraw_burden - workflow_burden), 1)
    def build_review_triggers(*, scope_path_count: int,
                              scope_node_count: int,
                              scope_max_commands: int,
                              scope_max_subpaths: int) -> list[str]:
        triggers: list[str] = []
        if scope_path_count >= 200:
            triggers.append("path_count_at_least_200")
        if scope_node_count >= 4000:
            triggers.append("node_count_at_least_4000")
        if scope_max_commands >= 500:
            triggers.append("one_path_at_least_500_commands")
        if scope_max_subpaths >= 50:
            triggers.append("one_path_at_least_50_subpaths")
        if only_color_layers and len(groups) >= 20:
            triggers.append(
                "twenty_plus_color_layers_without_semantic_groups")
        if drawable_count >= 100 and object_id_count == 0:
            triggers.append("one_hundred_plus_objects_without_ids")
        return triggers

    raw_review_triggers = build_review_triggers(
        scope_path_count=path_count,
        scope_node_count=node_count,
        scope_max_commands=max_commands,
        scope_max_subpaths=max_subpaths,
    )
    review_triggers = build_review_triggers(
        scope_path_count=generic_path_count,
        scope_node_count=generic_node_count,
        scope_max_commands=generic_max_commands,
        scope_max_subpaths=generic_coupling_max_subpaths,
    )
    outline_trigger_names = {
        "path_count_at_least_200",
        "node_count_at_least_4000",
        "one_path_at_least_500_commands",
        "one_path_at_least_50_subpaths",
    }
    outline_review_triggers = [
        item for item in review_triggers if item in outline_trigger_names
    ]
    raw_outline_review_triggers = [
        item for item in raw_review_triggers if item in outline_trigger_names
    ]
    status = (
        "accepted"
        if (score >= 75.0 and combined_structural_ease >= 75.0
            and automation_score >= 55.0 and not review_triggers)
        else "manual_review"
    )
    if score >= 85.0 and not outline_review_triggers:
        redraw_level = "low"
    elif score >= 70.0 and generic_max_commands < 500:
        redraw_level = "moderate"
    elif score >= 50.0:
        redraw_level = "high"
    else:
        redraw_level = "very_high"
    if workflow_burden == 0.0:
        workflow_level = "low"
    elif workflow_burden <= 6.0:
        workflow_level = "moderate"
    elif workflow_burden <= 15.0:
        workflow_level = "high"
    else:
        workflow_level = "very_high"

    reasons: list[str] = []
    if generic_path_count > 40 or generic_node_count > 500:
        reasons.append(
            "Uncertified bulk reshaping spans "
            f"{generic_path_count} paths and {generic_node_count} nodes; "
            "these correlated volume signals are penalized once."
        )
    if generic_max_commands > 250:
        reasons.append(
            "The largest uncertified path has "
            f"{generic_max_commands} commands and is costly to reshape."
        )
    if len(groups) > 60:
        reasons.append(f"{len(groups)} groups/layers make stack navigation heavier.")
    if object_id_count == 0 and drawable_count >= 20:
        reasons.append(
            f"None of the {drawable_count} drawable objects has an object ID."
        )
    elif drawable_count and object_id_count / drawable_count < 0.10:
        reasons.append(
            f"Only {object_id_count} of {drawable_count} drawable objects has an ID."
        )
    if needs_semantic_grouping:
        reasons.append(
            "Groups separate paint layers only; no semantic object grouping was detected."
        )
    if generic_coupling_max_subpaths > 20:
        reasons.append(
            f"One uncertified path contains {generic_coupling_max_subpaths} "
            "subpaths, coupling many shapes together."
        )
    if certified_compound_records:
        reasons.append(
            f"{len(certified_compound_records)} source-topology-certified "
            "gradient compound path(s) retain their closed loops as one "
            "ownership object instead of fragmenting the colour field."
        )
    if certified_curve_path_ids:
        reasons.append(
            f"{len(certified_curve_path_ids)} exact-SVG curve-optimizer path(s) "
            "are disclosed in raw totals but excluded from generic cleanup "
            "heuristics under their geometry-error contracts."
        )
    if automation_status == "ready_for_common_operations" and status != "accepted":
        reasons.append(
            "Common structured operations are ready, but this does not make the "
            "most complex outlines inexpensive to reshape."
        )
    if not reasons:
        reasons.append("No encoded structural editability threshold was exceeded.")

    details: dict[str, object] = {
        "path_count": path_count,
        "reported_path_count": reported_paths,
        "native_primitive_count": native_count,
        "svg_native_primitive_count": svg_native_count,
        "native_primitive_breakdown": native_breakdown,
        "stroke_count": stroke_count,
        "svg_stroked_object_count": svg_stroke_count,
        "gradient_count": gradient_count,
        "gradient_resource_count": gradient_resource_count,
        "node_count": node_count,
        "svg_estimated_node_count": svg_estimated_nodes,
        "raw_scope_metrics": {
            "scope": "all_final_svg_paths_and_conservative_report_counts",
            "path_count": path_count,
            "node_count": node_count,
            "svg_path_anchor_count": estimated_path_anchors,
            "command_count": total_commands,
            "control_point_count": total_control_points,
            "subpath_count": total_subpaths,
            "max_commands_in_one_path": max_commands,
            "max_subpaths_in_one_path": max_subpaths,
            "review_triggers": raw_review_triggers,
            "outline_review_triggers": raw_outline_review_triggers,
            "scope_note": (
                "Disclosure only: authoritative optimiser certificates never "
                "remove paths, nodes, commands or subpaths from these raw "
                "delivered-SVG totals."),
        },
        "certified_optimizer_scope": {
            "scope": "exact_final_svg_authoritative_optimizer_certificates",
            "path_count": len(certified_path_metric_records),
            "path_ids": sorted(str(item) for item in
                               certified_optimizer_path_ids),
            "curve_path_count": len(certified_curve_path_ids),
            "gradient_path_count": len(certified_gradient_path_ids),
            "curve_gradient_overlap_count": len(
                certified_curve_path_ids & certified_gradient_path_ids),
            "path_anchor_count": certified_path_anchors,
            "command_count": certified_total_commands,
            "control_point_count": certified_total_control_points,
            "subpath_count": certified_total_subpaths,
            "max_commands_in_one_path": certified_max_commands,
            "max_subpaths_in_one_path": certified_max_subpaths,
            "curve_evidence": certified_curve_paths,
            "gradient_evidence": certified_gradient_paths,
            "scope_note": (
                "These exact paths remain fully disclosed above. They are "
                "excluded only from generic redraw heuristics after fail-closed "
                "SHA, certificate, transaction and unique-final-ID checks."),
        },
        "generic_uncertified_redraw_scope": {
            "scope": "final_svg_paths_without_authoritative_optimizer_certificate",
            "path_count": generic_path_count,
            "node_count": generic_node_count,
            "svg_path_anchor_count": generic_path_anchors,
            "report_only_node_surplus": report_only_node_surplus,
            "unattributed_conservative_node_count": max(
                0, generic_node_count - generic_path_anchors),
            "command_count": generic_total_commands,
            "control_point_count": generic_total_control_points,
            "subpath_count": sum(generic_subpath_counts),
            "median_commands_per_path": generic_median_commands,
            "p95_commands_per_path": generic_p95_commands,
            "max_commands_in_one_path": generic_max_commands,
            "max_control_points_in_one_path": generic_max_control_points,
            "max_subpaths_in_one_path": generic_coupling_max_subpaths,
            "max_path_command_share": round(
                generic_max_command_share, 6),
            "review_triggers": review_triggers,
            "outline_review_triggers": outline_review_triggers,
            "scope_note": (
                "Only this uncertified redraw scope feeds outline penalties "
                "and formal structural acceptance; thresholds are unchanged."),
        },
        "group_count": len(groups),
        "color_layer_group_count": color_layer_count,
        "semantic_group_count": semantic_group_count,
        "selectable_semantic_group_count": len(selectable_semantic_groups),
        "actual_dom_group_count": len(actual_dom_groups),
        "semantically_grouped_drawable_count": len(
            semantically_grouped_drawables),
        "semantic_group_coverage": round(semantic_group_coverage, 6),
        "direct_single_drawable_selection": direct_selection,
        "effective_selection_coverage": round(selection_coverage, 6),
        "selection_coverage_basis": (
            "single_identifiable_visible_drawable" if direct_selection["available"]
            else "semantic_group_coverage"),
        "duplicate_document_ids": sorted(identifier for identifier, count in id_counts.items() if count > 1),
        "unique_solid_paint_count": len(solid_paints),
        "unique_solid_paints": sorted(solid_paints),
        "paint_role_count": len(paint_role_ids),
        "paint_role_ids": sorted(paint_role_ids),
        "paint_role_annotated_drawable_count": len(paint_role_drawables),
        "paint_role_annotation_coverage": round(paint_role_coverage, 6),
        "total_subpaths": total_subpaths,
        "multi_subpath_path_count": multi_subpath_paths,
        "max_subpaths_per_path": max_subpaths,
        "generic_coupling_max_subpaths": generic_coupling_max_subpaths,
        "source_topology_certified_compound_path_ids": sorted(
            str(item["id"]) for item in certified_compound_records),
        "source_topology_gradient_evidence": certified_gradient_paths,
        "curve_optimizer_certificate_evidence": certified_curve_paths,
        "total_path_commands": total_commands,
        "explicit_bezier_control_point_count": total_control_points,
        "max_explicit_bezier_control_points_per_path": max_control_points,
        "outline_handle_count_estimate": (
            estimated_path_anchors + total_control_points),
        "path_command_count_median": median_commands,
        "path_command_count_p95": p95_commands,
        "path_command_count_max": max_commands,
        "max_path_command_share": round(max_command_share, 6),
        "max_path_command_share_percent": round(max_command_share * 100.0, 2),
        "drawable_object_count": drawable_count,
        "object_id_count": object_id_count,
        "object_id_coverage": round(object_id_coverage, 6),
        "has_object_ids": bool(object_id_count),
        "all_drawable_objects_have_ids": (
            bool(drawable_count) and object_id_count == drawable_count
        ),
        "only_color_layers_without_semantic_groups": only_color_layers,
        "count_sources": {
            "paths": "svg",
            "native_primitives": "report_or_stats" if reported_native is not None else "svg",
            "strokes": "report_or_stats" if reported_strokes is not None else "svg",
            "gradients": "report_or_stats" if reported_gradients is not None else "svg",
            "nodes": node_count_source,
        },
        "risk_penalties": {
            name: round(value, 2)
            for name, value in sorted(risk_indicators.items())
        },
        "raw_risk_penalties": {
            name: round(value, 2)
            for name, value in sorted(raw_risk_indicators.items())
        },
        "applied_penalty_families": {
            name: round(value, 2)
            for name, value in sorted(all_penalty_families.items())
        },
        "applied_outline_penalty_families": {
            name: round(value, 2)
            for name, value in sorted(outline_penalty_families.items())
        },
        "workflow_friction_penalty_families": {
            name: round(value, 2)
            for name, value in sorted(workflow_penalty_families.items())
        },
        "penalty_combination": (
            "Raw structure remains disclosed; outline indicators use only "
            "paths without an authoritative exact-SVG optimiser certificate. "
            "Correlated indicators use the maximum within each family, then "
            "outline families and workflow-friction families are scored on "
            "separate axes. Their sum is retained only for the conservative "
            "acceptance guardrail. No visual-style or brush-texture discount "
            "is applied."
        ),
        "visual_style_discount_applied": False,
        "review_triggers": review_triggers,
        "generic_review_triggers": review_triggers,
        "raw_review_triggers": raw_review_triggers,
        "outline_review_triggers": outline_review_triggers,
        "raw_outline_review_triggers": raw_outline_review_triggers,
        "automation_readiness": {
            "score": automation_score,
            "status": automation_status,
            "evidence_class": "generic_structural_heuristic",
            "score_is_operation_pass_count": False,
            "components": {
                name: round(value, 2)
                for name, value in sorted(automation_components.items())
            },
            "scope_note": (
                "Measures dependable IDs, direct single-drawable or semantic-group selection, paint-role "
                "targets and native SVG handles. This score is not a task count. "
                "Named-operation evidence is audited separately."
            ),
        },
        "redraw_complexity": {
            "ease_score": score,
            "burden_score": redraw_burden,
            "level": redraw_level,
            "penalty_families": {
                name: round(value, 2)
                for name, value in sorted(outline_penalty_families.items())
            },
            "scope_note": (
                "Measures freeform point-level reshaping and cleanup burden in "
                "the generic uncertified path scope. Raw final-SVG complexity "
                "remains separately disclosed. Intentional brush edges remain "
                "real redraw complexity even when common automated operations "
                "pass."
            ),
        },
        "workflow_friction": {
            "ease_score": workflow_ease,
            "burden_score": workflow_burden,
            "level": workflow_level,
            "penalty_families": {
                name: round(value, 2)
                for name, value in sorted(workflow_penalty_families.items())
            },
            "scope_note": (
                "Measures group navigation, stable object identity and semantic "
                "structure friction. It is deliberately excluded from the "
                "freeform outline-cleanup score."
            ),
        },
        "scope_note": (
            "Layered structural heuristic only: automation readiness and redraw "
            "complexity answer different questions. This result does not prove an "
            "80% designer time-saving claim; timed human editing is required."
        ),
    }
    human_validation = {
        "status": "not_performed",
        "timed_editing_test_performed": False,
        "designer_acceptance": None,
        "original_human_tasks_passed": None,
        "original_human_tasks_total": None,
        "scope_note": (
            "No timed designer session or user-authored human task checklist is "
            "performed by this structural audit. Structural check counts must "
            "not be relabelled as original human tasks passed."
        ),
    }
    acceptance_gate = {
        "scope": "structural_editability_only",
        "status": status,
        "passed": status == "accepted",
        "requirements": {
            "redraw_ease_minimum": 75.0,
            "combined_structural_ease_minimum": 75.0,
            "automation_readiness_minimum": 55.0,
            "review_trigger_count_maximum": 0,
        },
        "observed": {
            "redraw_ease": score,
            "workflow_friction_ease": workflow_ease,
            "combined_structural_ease": combined_structural_ease,
            "automation_readiness": automation_score,
            "review_trigger_count": len(review_triggers),
            "generic_review_trigger_count": len(review_triggers),
            "raw_review_trigger_count": len(raw_review_triggers),
        },
        "scope_note": (
            "Passing this guardrail means only that no encoded blocker fired "
            "in the generic uncertified redraw scope or workflow structure. "
            "Raw totals remain disclosed. It is not visual acceptance, a "
            "human task pass, timed labour evidence, or final designer approval."
        ),
    }
    result: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "editability_details": details,
        "audit_model": "layered-v2",
        "status": status,
        "status_scope": "structural_editability_gate",
        "score": score,
        "score_axis": "redraw_ease",
        "automation_readiness": details["automation_readiness"],
        "redraw_complexity": details["redraw_complexity"],
        "workflow_friction": details["workflow_friction"],
        "acceptance_gate": acceptance_gate,
        "named_operation_evidence": named_operation_evidence,
        "human_validation": human_validation,
        "reasons": reasons,
        "interpretation": (
            "Automation readiness and redraw complexity are independent. A file "
            "may pass all named common operations while retaining expensive brush "
            "or compound outlines that require manual review. Separate structural "
            "operation counts are never human-task results."
        ),
    }
    # Fail here during development if a non-serializable value slips in.
    json.dumps(result, ensure_ascii=False)
    return result


# Readable alias for callers that prefer "analyze" terminology.
analyze_editability = audit_editability


__all__ = ["AUDIT_SCHEMA", "analyze_editability", "audit_editability"]
