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


def _scaled_penalty(value: float, free: float, severe: float,
                    maximum: float) -> float:
    if value <= free:
        return 0.0
    if severe <= free:
        return maximum
    fraction = min(1.0, (value - free) / (severe - free))
    return maximum * fraction


def audit_editability(svg_path: str | Path,
                      report_or_stats: object = None) -> dict[str, object]:
    """Audit one generated SVG and return a JSON-serializable result.

    ``report_or_stats`` may be a report mapping, a report JSON path, the
    engine's stats object, or ``None``.  With ``None``, a sibling report.json
    is used when present.  Direct SVG measurements remain separately visible
    so a stale report cannot conceal structural complexity.
    """
    path = Path(svg_path)
    root = ET.parse(path).getroot()
    supplied = _load_report_or_stats(report_or_stats, path)
    named_operation_evidence = _named_operation_evidence(supplied)
    scene_graph_metadata = _embedded_metadata_object(
        root, "scene-graph-metadata")

    groups: list[ET.Element] = []
    drawables: list[tuple[ET.Element, dict[str, str]]] = []
    gradient_elements: list[ET.Element] = []

    def walk(element: ET.Element, inherited: Mapping[str, str],
             in_defs: bool = False) -> None:
        tag = _local_name(element.tag)
        now_in_defs = in_defs or tag == "defs"
        presentation = _presentation(element, inherited)
        if tag in {"linearGradient", "radialGradient"}:
            gradient_elements.append(element)
        if not now_in_defs:
            if tag == "g":
                groups.append(element)
            if tag in _DRAWABLE_TAGS:
                drawables.append((element, presentation))
        for child in element:
            walk(child, presentation, now_in_defs)

    walk(root, {"fill": "#000000", "stroke": "none", "stroke-width": "1"})

    path_elements = [item for item in drawables if _local_name(item[0].tag) == "path"]
    command_counts: list[int] = []
    subpath_counts: list[int] = []
    control_point_counts: list[int] = []
    estimated_path_anchors = 0
    for element, _presentation_data in path_elements:
        commands, subpaths, anchors, controls = _path_metrics_detailed(
            element.attrib.get("d", ""))
        command_counts.append(commands)
        subpath_counts.append(subpaths)
        control_point_counts.append(controls)
        estimated_path_anchors += anchors

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

    object_ids = [
        element.attrib["id"] for element, _ in drawables
        if element.attrib.get("id", "").strip()
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
    paint_role_drawables: set[ET.Element] = set()
    for element in root.iter():
        annotated = False
        for attribute, value in element.attrib.items():
            if _local_name(attribute).startswith("data-paint-role-") and value:
                paint_role_ids.add(value)
                annotated = True
        if annotated and element in drawable_elements:
            paint_role_drawables.add(element)
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
    drawable_count = len(drawables)
    object_id_count = len(object_ids)

    object_id_coverage = object_id_count / drawable_count if drawable_count else 0.0
    semantic_group_coverage = (
        len(semantically_grouped_drawables) / drawable_count
        if drawable_count else 0.0
    )
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
        "semantic_selection": 25.0 * min(1.0, semantic_group_coverage / 0.40)
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
    if max_commands >= 500:
        isolation_score -= 4.0
    if max_subpaths > 20:
        isolation_score -= 4.0
    if max_command_share > 0.45:
        isolation_score -= 2.0
    automation_components["object_isolation"] = max(0.0, isolation_score)
    automation_score = round(min(100.0, sum(automation_components.values())), 1)
    automation_status = (
        "ready_for_common_operations" if automation_score >= 80.0
        else "partially_ready" if automation_score >= 55.0
        else "limited"
    )

    # Raw indicators remain visible, but correlated warnings are combined by
    # maximum within a family. Paths and nodes describe the same global volume;
    # maximum commands/subpaths/concentration describe the same worst object.
    # Summing all of them made deliberate brush texture count three times.
    risk_indicators: dict[str, float] = {}

    def add_scaled(name: str, value: float, free: float, severe: float,
                   maximum: float) -> None:
        penalty = _scaled_penalty(value, free, severe, maximum)
        if penalty:
            risk_indicators[name] = penalty

    add_scaled("many_paths", path_count, 40, 300, 22)
    add_scaled("many_nodes", node_count, 500, 5000, 24)
    add_scaled("many_bezier_control_points", total_control_points,
               500, 5000, 18)
    add_scaled("excessive_group_navigation", len(groups), 60, 180, 6)
    add_scaled("high_median_path_commands", median_commands, 40, 150, 8)
    add_scaled("high_p95_path_commands", p95_commands, 120, 450, 10)
    add_scaled("single_very_complex_path", max_commands, 250, 800, 12)
    add_scaled("single_path_many_bezier_controls", max_control_points,
               100, 1200, 10)
    add_scaled("many_subpaths_in_one_path", max_subpaths, 20, 100, 6)
    add_scaled("path_command_concentration", max_command_share, 0.45, 0.85, 6)
    if drawable_count >= 20 and object_id_count == 0:
        risk_indicators["no_object_ids"] = 8.0
    elif drawable_count >= 50 and object_id_coverage < 0.10:
        risk_indicators["very_low_object_id_coverage"] = 5.0
    if only_color_layers:
        risk_indicators["color_layers_without_semantic_groups"] = 10.0

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
    review_triggers: list[str] = []
    if path_count >= 200:
        review_triggers.append("path_count_at_least_200")
    if node_count >= 4000:
        review_triggers.append("node_count_at_least_4000")
    if max_commands >= 500:
        review_triggers.append("one_path_at_least_500_commands")
    if max_subpaths >= 50:
        review_triggers.append("one_path_at_least_50_subpaths")
    if only_color_layers and len(groups) >= 20:
        review_triggers.append("twenty_plus_color_layers_without_semantic_groups")
    if drawable_count >= 100 and object_id_count == 0:
        review_triggers.append("one_hundred_plus_objects_without_ids")
    outline_trigger_names = {
        "path_count_at_least_200",
        "node_count_at_least_4000",
        "one_path_at_least_500_commands",
        "one_path_at_least_50_subpaths",
    }
    outline_review_triggers = [
        item for item in review_triggers if item in outline_trigger_names
    ]
    status = (
        "accepted"
        if (score >= 75.0 and combined_structural_ease >= 75.0
            and automation_score >= 55.0 and not review_triggers)
        else "manual_review"
    )
    if score >= 85.0 and not outline_review_triggers:
        redraw_level = "low"
    elif score >= 70.0 and max_commands < 500:
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
    if path_count > 40 or node_count > 500:
        reasons.append(
            f"Bulk reshaping spans {path_count} paths and {node_count} nodes; "
            "these correlated volume signals are penalized once."
        )
    if max_commands > 250:
        reasons.append(
            f"The largest path has {max_commands} commands and is costly to reshape."
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
    if only_color_layers:
        reasons.append(
            "Groups separate paint layers only; no semantic object grouping was detected."
        )
    if max_subpaths > 20:
        reasons.append(
            f"One path contains {max_subpaths} subpaths, coupling many shapes together."
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
        "group_count": len(groups),
        "color_layer_group_count": color_layer_count,
        "semantic_group_count": semantic_group_count,
        "selectable_semantic_group_count": len(selectable_semantic_groups),
        "actual_dom_group_count": len(actual_dom_groups),
        "semantically_grouped_drawable_count": len(
            semantically_grouped_drawables),
        "semantic_group_coverage": round(semantic_group_coverage, 6),
        "unique_solid_paint_count": len(solid_paints),
        "unique_solid_paints": sorted(solid_paints),
        "paint_role_count": len(paint_role_ids),
        "paint_role_ids": sorted(paint_role_ids),
        "paint_role_annotated_drawable_count": len(paint_role_drawables),
        "paint_role_annotation_coverage": round(paint_role_coverage, 6),
        "total_subpaths": total_subpaths,
        "multi_subpath_path_count": multi_subpath_paths,
        "max_subpaths_per_path": max_subpaths,
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
            "Correlated indicators use the maximum within each family, then "
            "outline families and workflow-friction families are scored on "
            "separate axes. Their sum is retained only for the conservative "
            "acceptance guardrail. No visual-style or brush-texture discount "
            "is applied."
        ),
        "visual_style_discount_applied": False,
        "review_triggers": review_triggers,
        "outline_review_triggers": outline_review_triggers,
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
                "Measures dependable IDs, semantic selection groups, paint-role "
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
                "Measures freeform point-level reshaping and cleanup burden. "
                "Intentional brush edges remain real redraw complexity even when "
                "common automated operations pass."
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
        },
        "scope_note": (
            "Passing this guardrail means only that no encoded structural "
            "editability blocker fired. It is not visual acceptance, a human "
            "task pass, timed labour evidence, or final designer approval."
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
