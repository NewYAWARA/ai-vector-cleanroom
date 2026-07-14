from __future__ import annotations

"""Conservative structural audit for five common designer operations.

The audit deliberately distinguishes SVG capabilities from semantic quality.
It never treats a sidecar-only scene-graph proposal as a selectable object and
it does not infer that an operation saves designer time.  Results contain only
JSON-serialisable values so they can be embedded in ``report.json``.
"""

from collections import Counter
from collections.abc import Mapping
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET


AUDIT_SCHEMA = "ai-vector-cleanroom.designer-operations/v1"
SVG_NS = "http://www.w3.org/2000/svg"
DRAWABLES = {
    "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
}
GRADIENTS = {"linearGradient", "radialGradient"}
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_URL_REF = re.compile(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", re.I)
_HEX = re.compile(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", re.I)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _style_map(raw: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in (raw or "").split(";"):
        if ":" not in item:
            continue
        name, value = item.split(":", 1)
        name = name.strip().lower()
        if name:
            result[name] = value.strip()
    return result


def _property(element: ET.Element, parents: Mapping[ET.Element, ET.Element],
              name: str, default: str = "") -> str:
    node: ET.Element | None = element
    while node is not None:
        styled = _style_map(node.get("style"))
        if name in styled:
            return styled[name]
        if name in node.attrib:
            return node.attrib[name]
        node = parents.get(node)
    return default


def _number(value: Any) -> float | None:
    match = _NUMBER.search(str(value or ""))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _length(value: Any, percent_scale: float) -> float | None:
    parsed = _number(value)
    if parsed is None:
        return None
    if str(value).strip().endswith("%"):
        return parsed * percent_scale / 100.0
    return parsed


def _viewbox(root: ET.Element) -> list[float]:
    values = [float(value) for value in _NUMBER.findall(root.get("viewBox", ""))]
    if len(values) == 4 and values[2] > 0 and values[3] > 0:
        return values
    width = _number(root.get("width"))
    height = _number(root.get("height"))
    if width and height and width > 0 and height > 0:
        return [0.0, 0.0, width, height]
    return [0.0, 0.0, 1.0, 1.0]


def _normalise_hex(value: str) -> str | None:
    value = value.strip().lower()
    match = _HEX.fullmatch(value)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(channel * 2 for channel in digits)
    return "#" + digits


def _load_json(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    text_value = str(value)
    stripped = text_value.lstrip()
    if stripped.startswith("{"):
        loaded = json.loads(text_value)
    else:
        loaded = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("JSON report must be an object")
    return loaded


def _embedded_json(root: ET.Element, element_id: str) -> dict[str, Any] | None:
    for element in root.iter():
        if element.get("id") != element_id or not (element.text or "").strip():
            continue
        try:
            loaded = json.loads(element.text or "")
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def _paint_manifest(value: Mapping[str, Any] | str | Path | None,
                    svg_path: Path, root: ET.Element) -> tuple[dict[str, Any] | None, str]:
    if value is not None:
        return _load_json(value), "argument"
    embedded = _embedded_json(root, "ai-vector-cleanroom-paint-roles")
    if embedded is not None:
        return embedded, "embedded-metadata"
    candidates = [
        svg_path.with_suffix(".paint_roles.json"),
        svg_path.parent / f"{svg_path.stem}.paint_roles.json",
        svg_path.parent / "paint_roles.json",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return _load_json(candidate), str(candidate)
    return None, "not-found"


def _scene_report(value: Mapping[str, Any] | str | Path | None,
                  root: ET.Element) -> tuple[dict[str, Any] | None, str]:
    if value is None:
        embedded = _embedded_json(root, "scene-graph-metadata")
        return (embedded, "embedded-metadata") if embedded else (None, "not-found")
    loaded = _load_json(value)
    if loaded is None:
        return None, "not-found"
    # Accept an integration report that nests the postprocessor's own report.
    for key in ("scene_graph_report", "scene_graph", "report"):
        nested = loaded.get(key)
        if isinstance(nested, Mapping) and any(
                name in nested for name in ("actual_dom_groups", "groups",
                                             "manifest_only_groups")):
            return dict(nested), f"argument.{key}"
    return loaded, "argument"


def _property_name(value: Any) -> str:
    token = str(value).strip().lower().replace("_", "-")
    if ":" in token:
        token = token.rsplit(":", 1)[-1]
    return token


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple, set)) else []


def _explicit_paints(root: ET.Element) -> set[str]:
    result: set[str] = set()
    for element in root.iter():
        style = _style_map(element.get("style"))
        for name in ("fill", "stroke", "stop-color"):
            values = []
            if name in element.attrib:
                values.append(element.attrib[name])
            if name in style:
                values.append(style[name])
            for value in values:
                normalised = _normalise_hex(value)
                if normalised:
                    result.add(normalised)
    return result


def _gradient_inventory(root: ET.Element) -> dict[str, Any]:
    gradients: dict[str, ET.Element] = {}
    duplicate_ids: set[str] = set()
    anonymous: list[ET.Element] = []
    for element in root.iter():
        if _local(element.tag) not in GRADIENTS:
            continue
        identifier = (element.get("id") or "").strip()
        if not identifier:
            anonymous.append(element)
            continue
        if identifier in gradients:
            duplicate_ids.add(identifier)
        if identifier:
            gradients[identifier] = element

    references: dict[str, int] = {identifier: 0 for identifier in gradients}
    for element in root.iter():
        for value in element.attrib.values():
            for identifier in _URL_REF.findall(value):
                if identifier in references:
                    references[identifier] += 1

    def resolved_stops(identifier: str, stack: set[str] | None = None) -> int:
        stack = set(stack or ())
        if identifier in stack or identifier not in gradients:
            return 0
        stack.add(identifier)
        gradient = gradients[identifier]
        own = sum(_local(item.tag) == "stop" for item in gradient)
        if own:
            return own
        href = gradient.get("href") or gradient.get(
            "{http://www.w3.org/1999/xlink}href", "")
        return resolved_stops(href[1:], stack) if href.startswith("#") else 0

    resources = []
    for identifier, element in sorted(gradients.items()):
        stop_count = resolved_stops(identifier)
        resources.append({
            "id": identifier,
            "type": _local(element.tag),
            "reference_count": references[identifier],
            "resolved_stop_count": stop_count,
            "independently_editable": bool(
                identifier not in duplicate_ids
                and references[identifier] > 0
                and stop_count >= 2
            ),
        })
    for index, element in enumerate(anonymous, start=1):
        resources.append({
            "id": None,
            "anonymous_index": index,
            "type": _local(element.tag),
            "reference_count": 0,
            "resolved_stop_count": sum(_local(item.tag) == "stop" for item in element),
            "independently_editable": False,
        })
    return {
        "resources": resources,
        "gradient_ids": sorted(gradients),
        "anonymous_count": len(anonymous),
        "duplicate_ids": sorted(duplicate_ids),
        "referenced_count": sum(item["reference_count"] > 0 for item in resources),
        "editable_count": sum(item["independently_editable"] for item in resources),
    }


def _audit_global_colour(manifest: dict[str, Any] | None, manifest_source: str,
                         root: ET.Element, gradients: dict[str, Any]) -> dict[str, Any]:
    operation_id = "global-colour-role"
    if not manifest:
        annotations = {
            _property_name(name.removeprefix("data-paint-role-"))
            for element in root.iter() for name in element.attrib
            if name.startswith("data-paint-role-")
        }
        status = "partial" if annotations else "failed"
        return {
            "id": operation_id,
            "label": "One-control global colour-role change",
            "status": status,
            "automatable": False,
            "evidence": {
                "manifest_source": manifest_source,
                "annotation_channels": sorted(annotations),
                "complete_role_ids": [],
            },
            "scope_note": (
                "Paint-role annotations without a matching role manifest do not prove "
                "that one action can rewrite fills, strokes and gradient stops."
            ),
        }

    roles = manifest.get("roles", [])
    if not isinstance(roles, list):
        roles = []
    schema = str(manifest.get("schema", ""))
    schema_valid = schema.startswith("ai-vector-cleanroom.paint-roles/")
    explicit = _explicit_paints(root)
    available_gradients = set(gradients["gradient_ids"])
    role_details = []
    complete = []
    covered: set[str] = set()
    for role in roles:
        if not isinstance(role, Mapping):
            continue
        channels = {_property_name(item) for item in _sequence(role.get("properties"))}
        member_colours: set[str] = set()
        gradient_ids = {str(item) for item in _sequence(role.get("gradient_ids")) if item}
        for member in _sequence(role.get("members")):
            if not isinstance(member, Mapping):
                continue
            colour = _normalise_hex(str(member.get("hex", "")))
            if colour:
                member_colours.add(colour)
            member_channels = member.get("channels", {})
            if isinstance(member_channels, Mapping):
                channels.update(_property_name(item) for item in member_channels)
            gradient_ids.update(str(item) for item in
                                _sequence(member.get("gradient_ids")) if item)
        channels &= {"fill", "stroke", "stop-color"}
        covered.update(channels)
        identifier = str(role.get("id", ""))
        control = role.get("control")
        control_count = _integer(role.get("control_count", 0))
        matching_paints = member_colours & explicit
        missing_paints = member_colours - explicit
        paints_match = bool(member_colours and not missing_paints)
        gradient_match = bool(gradient_ids & available_gradients)
        control_valid = bool(isinstance(control, Mapping)
                             and control.get("default_hex")
                             and control.get("transform"))
        is_complete = bool(
            identifier and channels == {"fill", "stroke", "stop-color"}
            and schema_valid and control_count == 1 and control_valid
            and paints_match and gradient_match
        )
        if is_complete:
            complete.append(identifier)
        role_details.append({
            "id": identifier,
            "channels": sorted(channels),
            "control_count": control_count,
            "control_valid": control_valid,
            "paint_inventory_matches_svg": paints_match,
            "matching_paint_count": len(matching_paints),
            "missing_paints": sorted(missing_paints),
            "gradient_ids_in_svg": sorted(gradient_ids & available_gradients),
        })

    unsupported = manifest.get("unsupported_paints", [])
    unsupported_count = len(unsupported) if isinstance(unsupported, list) else 1
    if complete and unsupported_count == 0:
        status, automatable = "passed", True
    elif roles and covered == {"fill", "stroke", "stop-color"}:
        status, automatable = "partial", False
    else:
        status, automatable = "failed", False
    return {
        "id": operation_id,
        "label": "One-control global colour-role change",
        "status": status,
        "automatable": automatable,
        "evidence": {
            "manifest_source": manifest_source,
            "manifest_schema": manifest.get("schema"),
            "manifest_schema_valid": schema_valid,
            "role_count": len(role_details),
            "covered_channels": sorted(covered),
            "complete_role_ids": complete,
            "unsupported_paint_count": unsupported_count,
            "roles": role_details,
        },
        "scope_note": (
            "A pass requires the same one-control role to map ordinary fill and "
            "stroke declarations plus native gradient stop colours. It proves a "
            "portable rewrite target, not that the chosen colour family is semantic."
        ),
    }


def _audit_gradients(gradients: dict[str, Any]) -> dict[str, Any]:
    resources = gradients["resources"]
    present = len(resources)
    referenced = gradients["referenced_count"]
    editable = gradients["editable_count"]
    if present == 0:
        status, automatable = "manual_review", False
    elif referenced > 0 and editable == referenced:
        status, automatable = "passed", True
    elif editable > 0 or referenced > 0:
        status, automatable = "partial", bool(editable)
    else:
        status, automatable = "failed", False
    return {
        "id": "native-gradient-resource",
        "label": "Independently editable gradient resources",
        "status": status,
        "automatable": automatable,
        "evidence": {
            "native_gradient_count": present,
            "referenced_gradient_count": referenced,
            "editable_gradient_count": editable,
            "duplicate_ids": gradients["duplicate_ids"],
            "anonymous_gradient_count": gradients["anonymous_count"],
            "resources": resources,
        },
        "scope_note": (
            "No gradient is not automatically a failure because flat artwork may not "
            "need one. A pass requires referenced native gradient definitions with "
            "stable IDs and at least two editable stops."
        ),
    }


def _bbox_union(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _element_bbox(element: ET.Element) -> list[float] | None:
    kind = _local(element.tag)
    if kind == "circle":
        cx, cy, radius = (_number(element.get(name)) for name in ("cx", "cy", "r"))
        if None not in (cx, cy, radius):
            return [cx - radius, cy - radius, cx + radius, cy + radius]
    if kind == "ellipse":
        cx, cy, rx, ry = (_number(element.get(name))
                          for name in ("cx", "cy", "rx", "ry"))
        if None not in (cx, cy, rx, ry):
            return [cx - rx, cy - ry, cx + rx, cy + ry]
    if kind == "rect":
        x = _number(element.get("x")) or 0.0
        y = _number(element.get("y")) or 0.0
        width, height = (_number(element.get(name)) for name in ("width", "height"))
        if width is not None and height is not None:
            return [x, y, x + width, y + height]
    if kind == "line":
        x1, y1, x2, y2 = (_number(element.get(name))
                          for name in ("x1", "y1", "x2", "y2"))
        if None not in (x1, y1, x2, y2):
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    value = element.get("points") if kind in {"polygon", "polyline"} else (
        element.get("d") if kind == "path" else None)
    if value:
        numbers = [float(item) for item in _NUMBER.findall(value)]
        # This is deliberately only a fallback for spatial classification.  A
        # scene-graph report's audited bbox is preferred whenever available.
        if len(numbers) >= 4:
            pairs = list(zip(numbers[0::2], numbers[1::2]))
            return [min(x for x, _ in pairs), min(y for _, y in pairs),
                    max(x for x, _ in pairs), max(y for _, y in pairs)]
    return None


def _group_bbox(group: ET.Element) -> list[float] | None:
    # Raw child coordinates cannot be classified against the viewBox when a
    # transform is present.  An audited scene-report bbox may still be used by
    # the caller, but this fallback refuses to guess.
    if any(item.get("transform") for item in group.iter()):
        return None
    return _bbox_union([
        box for item in group.iter()
        if _local(item.tag) in DRAWABLES
        for box in [_element_bbox(item)] if box is not None
    ])


def _group_paints(group: ET.Element,
                  parents: Mapping[ET.Element, ET.Element]) -> set[str]:
    paints: set[str] = set()
    for element in group.iter():
        kind = _local(element.tag)
        if kind not in DRAWABLES:
            continue
        fill_default = "none" if kind in {"line", "polyline"} else "#000000"
        for name, default in (("fill", fill_default), ("stroke", "none")):
            value = _property(element, parents, name, default).strip().lower()
            opacity = _number(_property(element, parents, f"{name}-opacity", "1"))
            if value and value != "none" and opacity != 0:
                paints.add(f"{name}:{value}")
    return paints


def _normalise_report_groups(report: dict[str, Any] | None
                             ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    actual: dict[str, dict[str, Any]] = {}
    manifest_only: list[dict[str, Any]] = []
    if not report:
        return actual, manifest_only
    actual_items = report.get("actual_dom_groups", report.get("groups", []))
    if isinstance(actual_items, list):
        for item in actual_items:
            if not isinstance(item, Mapping):
                continue
            mode = str(item.get("mode", "actual-dom")).lower()
            identifier = str(item.get("id", ""))
            if identifier and mode == "actual-dom":
                actual[identifier] = dict(item)
    candidates = report.get("manifest_only_groups", report.get("skipped_unsafe_groups", []))
    if isinstance(candidates, list):
        manifest_only = [dict(item) for item in candidates if isinstance(item, Mapping)]
    return actual, manifest_only


def _reason_set(group: ET.Element, report_item: Mapping[str, Any]) -> set[str]:
    reasons = set()
    value = group.get("data-group-reasons", "")
    reasons.update(item.strip() for item in re.split(r"[,;]", value) if item.strip())
    raw = report_item.get("reasons", [])
    if isinstance(raw, str):
        raw = re.split(r"[,;]", raw)
    if isinstance(raw, list):
        reasons.update(str(item).strip() for item in raw if str(item).strip())
    role = (group.get("data-group-role") or group.get("data-object-role") or "").lower()
    if "half" in role or "dot" in role:
        reasons.add("declared-halftone-role")
    if "parallel" in role or "speed-line" in role or "speedline" in role:
        reasons.add("declared-parallel-line-role")
    return reasons


def _groups(root: ET.Element, report: dict[str, Any] | None,
            viewbox: list[float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents = {child: parent for parent in root.iter() for child in parent}
    report_actual, manifest_only = _normalise_report_groups(report)
    identifiers = [item.get("id") for item in root.iter() if item.get("id")]
    id_counts = Counter(identifiers)
    unique_ids = {item for item, count in id_counts.items() if count == 1}
    result = []
    canvas_area = viewbox[2] * viewbox[3]
    for group in root.iter():
        if _local(group.tag) != "g":
            continue
        identifier = group.get("id", "")
        report_item = report_actual.get(identifier, {})
        declared = group.get("data-group-mode", "").lower() == "actual-dom"
        if not declared and not report_item:
            continue
        drawables = [item for item in group.iter()
                     if _local(item.tag) in DRAWABLES]
        if not drawables:
            continue
        box = report_item.get("bbox")
        if not (isinstance(box, list) and len(box) == 4
                and all(isinstance(item, (int, float)) for item in box)):
            box = _group_bbox(group)
        paints = _group_paints(group, parents)
        paint_count = max(len(paints), _integer(report_item.get("paint_count", 0)))
        reasons = _reason_set(group, report_item)
        confidence = _number(report_item.get("confidence"))
        if box:
            width = max(0.0, float(box[2]) - float(box[0]))
            height = max(0.0, float(box[3]) - float(box[1]))
            area_fraction = width * height / max(canvas_area, 1e-9)
            width_fraction = width / max(viewbox[2], 1e-9)
            height_fraction = height / max(viewbox[3], 1e-9)
            centre_fraction = ((float(box[0]) + float(box[2])) / 2.0 - viewbox[0]) / viewbox[2]
            region = "left" if centre_fraction < 0.40 else (
                "right" if centre_fraction > 0.60 else "centre")
            local = bool(area_fraction <= 0.15 and width_fraction <= 0.60
                         and height_fraction <= 0.60)
        else:
            area_fraction = width_fraction = height_fraction = None
            region, local = "unknown", False
        result.append({
            "id": identifier,
            "actual_dom": True,
            "unique_id": identifier in unique_ids,
            "drawable_count": len(drawables),
            "paint_count": paint_count,
            "paints": sorted(paints),
            "bbox": [round(float(item), 4) for item in box] if box else None,
            "bbox_area_fraction": round(area_fraction, 6) if area_fraction is not None else None,
            "region": region,
            "local": local,
            "reasons": sorted(reasons),
            "confidence": round(confidence, 4) if confidence is not None else None,
            "hideable": bool(identifier in unique_ids),
            "movable": bool(identifier in unique_ids),
        })
    return result, manifest_only


def _item_reasons(item: Mapping[str, Any]) -> set[str]:
    raw = item.get("reasons", [])
    if isinstance(raw, str):
        raw = re.split(r"[,;]", raw)
    return {str(value).strip() for value in raw if str(value).strip()} if isinstance(
        raw, (list, tuple, set)
    ) else set()


def _audit_local_groups(groups: list[dict[str, Any]],
                        manifest_only: list[dict[str, Any]]) -> dict[str, Any]:
    semantic_reasons = {
        "cross-paint-overlay", "cross-paint-overlap", "matching-bounds",
        "fragment-proximity",
    }
    decoration_reasons = {"repeated-dot-proximity", "parallel-stroke-proximity"}
    candidates = [
        group for group in groups
        if group["local"] and group["paint_count"] >= 2
        and group["unique_id"]
        and semantic_reasons.intersection(group["reasons"])
        and not decoration_reasons.intersection(group["reasons"])
        and (group["bbox_area_fraction"] or 0.0) >= 0.0005
        and (group["confidence"] is None or group["confidence"] >= 0.74)
    ]
    regions = {group["region"] for group in candidates}
    if {"left", "right"} <= regions:
        status, automatable = "passed", True
    elif candidates:
        status, automatable = "partial", True
    else:
        manifest_semantic = [
            group for group in manifest_only
            if semantic_reasons.intersection(_item_reasons(group))
        ]
        status = "failed" if manifest_semantic or groups else "manual_review"
        automatable = False
    return {
        "id": "local-multicolour-dom-groups",
        "label": "Selectable local multicolour groups on both sides",
        "status": status,
        "automatable": automatable,
        "evidence": {
            "qualifying_actual_dom_group_count": len(candidates),
            "covered_regions": sorted(regions),
            "qualifying_groups": candidates,
            "manifest_only_group_count": len(manifest_only),
            "manifest_only_groups_do_not_count": True,
        },
        "scope_note": (
            "A pass needs distinct left and right local multicolour groups in the "
            "actual SVG DOM. Repeated-dot and parallel-line decoration groups cannot "
            "stand in for semantic logo components; report-only proposals never pass."
        ),
    }


def _audit_outer_ring(root: ET.Element, viewbox: list[float]) -> dict[str, Any]:
    parents = {child: parent for parent in root.iter() for child in parent}
    scale = min(viewbox[2], viewbox[3])
    threshold = 0.18 * scale
    candidates = []
    large_non_ring = []
    for element in root.iter():
        if _local(element.tag) != "circle":
            continue
        radius = _length(element.get("r"), scale)
        if radius is None or radius < threshold:
            continue
        stroke = _property(element, parents, "stroke", "none").strip()
        stroke_width = _length(_property(element, parents, "stroke-width", "1"), scale)
        fill = _property(element, parents, "fill", "#000000").strip().lower()
        fill_opacity = _number(_property(element, parents, "fill-opacity", "1"))
        stroke_opacity = _number(_property(element, parents, "stroke-opacity", "1"))
        opacity = _number(_property(element, parents, "opacity", "1"))
        display = _property(element, parents, "display", "inline").strip().lower()
        visibility = _property(element, parents, "visibility", "visible").strip().lower()
        visible = bool(opacity != 0 and display != "none"
                       and visibility not in {"hidden", "collapse"})
        ring_like = bool(visible and stroke.lower() not in {"none", "transparent"}
                         and stroke_width
                         and stroke_width > 0 and stroke_opacity != 0
                         and (fill in {"none", "transparent"} or fill_opacity == 0))
        item = {
            "id": element.get("id"),
            "radius": round(radius, 4),
            "radius_canvas_fraction": round(radius / max(scale, 1e-9), 6),
            "stroke": stroke,
            "stroke_width": round(stroke_width, 4) if stroke_width is not None else None,
            "fill": fill,
            "visible": visible,
            "has_dasharray": bool(_property(element, parents, "stroke-dasharray", "")),
            "merged_source_count": len([
                item for item in element.get("data-merged-from", "").split(",") if item
            ]),
            "detector": element.get("data-detector"),
            "radius_editable": _number(element.get("r")) is not None,
            "width_editable": stroke_width is not None,
        }
        (candidates if ring_like else large_non_ring).append(item)
    candidates.sort(key=lambda item: item["radius"], reverse=True)
    if candidates and candidates[0]["radius_editable"] and candidates[0]["width_editable"]:
        status, automatable = "passed", True
    elif candidates or large_non_ring:
        status, automatable = "partial", False
    else:
        status, automatable = "failed", False
    return {
        "id": "native-main-outer-ring",
        "label": "Large native circle outer ring with editable radius and width",
        "status": status,
        "automatable": automatable,
        "evidence": {
            "minimum_large_radius": round(threshold, 4),
            "native_ring_candidates": candidates,
            "large_non_ring_circles": large_non_ring,
        },
        "scope_note": (
            "The radius threshold scales with the SVG viewBox, which prevents small "
            "halftone dots from being mistaken for the main ring. A path that merely "
            "looks circular does not pass this native-circle operation."
        ),
    }


def _decoration_kind(reasons: set[str]) -> set[str]:
    kinds = set()
    if "repeated-dot-proximity" in reasons or "declared-halftone-role" in reasons:
        kinds.add("halftone")
    if ("parallel-stroke-proximity" in reasons
            or "declared-parallel-line-role" in reasons):
        kinds.add("parallel-lines")
    return kinds


def _audit_decorations(groups: list[dict[str, Any]],
                       manifest_only: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {"halftone": [], "parallel-lines": []}
    for group in groups:
        if not group["unique_id"]:
            continue
        for kind in _decoration_kind(set(group["reasons"])):
            by_kind[kind].append(group)
    manifest_kinds: set[str] = set()
    for group in manifest_only:
        manifest_kinds.update(_decoration_kind(_item_reasons(group)))
    actual_kinds = {kind for kind, values in by_kind.items() if values}
    if actual_kinds == {"halftone", "parallel-lines"}:
        status, automatable = "passed", True
    elif actual_kinds:
        status, automatable = "partial", True
    elif manifest_kinds:
        status, automatable = "failed", False
    else:
        status, automatable = "manual_review", False
    return {
        "id": "hideable-decoration-groups",
        "label": "Hideable halftone and parallel-line decoration groups",
        "status": status,
        "automatable": automatable,
        "evidence": {
            "actual_dom_kinds": sorted(actual_kinds),
            "manifest_only_kinds": sorted(manifest_kinds),
            "actual_dom_groups": {
                kind: [group["id"] for group in values]
                for kind, values in by_kind.items()
            },
            "manifest_only_groups_do_not_count": True,
        },
        "scope_note": (
            "A pass requires real, uniquely identified DOM groups for both repeated-dot "
            "halftones and parallel stroke decorations. One class alone is partial; "
            "manifest-only candidates are not hideable in a vector editor."
        ),
    }


def audit_designer_operations(
    svg_path: str | Path,
    paint_manifest: Mapping[str, Any] | str | Path | None = None,
    scene_graph_report: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Audit five edit operations without claiming semantic or time-saved proof."""

    path = Path(svg_path)
    root = ET.parse(path).getroot()
    if _local(root.tag) != "svg":
        raise ValueError("document root is not SVG")
    viewbox = _viewbox(root)
    manifest, manifest_source = _paint_manifest(paint_manifest, path, root)
    scene_report, scene_source = _scene_report(scene_graph_report, root)
    gradients = _gradient_inventory(root)
    groups, manifest_only = _groups(root, scene_report, viewbox)
    operations = [
        _audit_global_colour(manifest, manifest_source, root, gradients),
        _audit_gradients(gradients),
        _audit_local_groups(groups, manifest_only),
        _audit_outer_ring(root, viewbox),
        _audit_decorations(groups, manifest_only),
    ]
    statuses = ("passed", "partial", "failed", "manual_review")
    buckets = {
        status: [item["id"] for item in operations if item["status"] == status]
        for status in statuses
    }
    automatable = [item["id"] for item in operations if item["automatable"]]
    result = {
        "schema": AUDIT_SCHEMA,
        "acceptance_scope": "generic_machine_detectable_structural_handles",
        "semantic_task_validation": "not_performed",
        "timed_human_editing_validation": "not_performed",
        "human_acceptance": "not_tested",
        "svg": {
            "filename": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "viewBox": [round(item, 6) for item in viewbox],
            "drawable_count": sum(_local(item.tag) in DRAWABLES for item in root.iter()),
            "actual_dom_group_count": len(groups),
            "scene_graph_report_source": scene_source,
        },
        "summary": {
            "total_operations": len(operations),
            "passed": len(buckets["passed"]),
            "partial": len(buckets["partial"]),
            "failed": len(buckets["failed"]),
            "manual_review": len(buckets["manual_review"]),
            "automatable": len(automatable),
        },
        "passed": buckets["passed"],
        "partial": buckets["partial"],
        "failed": buckets["failed"],
        "manual_review": buckets["manual_review"],
        "automatable": automatable,
        "operations": operations,
        "scope_note": (
            "This is a conservative structural SVG audit. A pass means the native "
            "resource or actual DOM group can support the named edit; it does not prove "
            "semantic correctness, editor-specific interaction quality, an 80% labour "
            "saving, or final designer acceptance. Partial/manual_review must not be "
            "reported as passed."
        ),
    }
    # Fail immediately during development if a future field is not portable.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Audit five SVG designer operations")
    parser.add_argument("svg")
    parser.add_argument("--paint-manifest")
    parser.add_argument("--scene-graph-report")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = audit_designer_operations(
        arguments.svg,
        paint_manifest=arguments.paint_manifest,
        scene_graph_report=arguments.scene_graph_report,
    )
    body = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(body, encoding="utf-8")
    else:
        print(body, end="")
    return 0


__all__ = ["AUDIT_SCHEMA", "audit_designer_operations"]


if __name__ == "__main__":
    raise SystemExit(_cli())
