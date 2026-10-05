from __future__ import annotations

"""Designer-facing quality gates for editable SVG output.

Raster similarity and designer readiness answer different questions.  This
module deliberately keeps them separate: a visually accepted trace may still
require manual rework when colour ramps remain split into solid bands or when
smooth contours contain excessive anchors.

The audit is conservative and read-only.  It does not claim semantic object
correctness; inferred colour-band clusters are reported as candidates and are
strengthened, when available, by proposal/scene metadata.  Every public value
is JSON serialisable so the result can be embedded directly in ``report.json``.
"""

from collections import Counter
from collections.abc import Mapping
import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET


AUDIT_SCHEMA = "ai-vector-cleanroom.designer-quality/v1"
SVG_NS = "http://www.w3.org/2000/svg"
DRAWABLES = {
    "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
}
GRADIENTS = {"linearGradient", "radialGradient"}
PRIMITIVES = {"circle", "ellipse", "rect", "line"}

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_TOKEN = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_URL_REF = re.compile(r"url\(\s*['\"]?#([^)'\"\s]+)['\"]?\s*\)", re.I)
_HEX = re.compile(r"^#([0-9a-f]{3}|[0-9a-f]{6})$", re.I)
_RGB = re.compile(
    r"^rgba?\(\s*([\d.]+)%?\s*[, ]\s*([\d.]+)%?\s*[, ]\s*([\d.]+)%?",
    re.I,
)

_GEOMETRY_EVIDENCE_ALIASES = {
    "error_budget_percent": ("data-avc-error-budget-percent",),
    "p95_error_percent": (
        "data-avc-p95-error-percent", "data-avc-error-p95-percent",
        "data-avc-error-p95",
    ),
    "max_error_percent": (
        "data-avc-max-error-percent", "data-avc-error-max-percent",
        "data-avc-error-max",
    ),
    "designer_anchors": ("data-avc-designer-anchors",),
}
_GEOMETRY_TRANSFORM_CLAIMS = (
    "data-avc-curve-refit", "data-avc-gradient-object",
    "data-avc-anchors-before", "data-avc-anchors-after",
)


DEFAULT_THRESHOLDS: dict[str, float] = {
    # All geometric thresholds scale with the SVG viewBox diagonal.
    "short_segment_diagonal_fraction": 0.0030,
    "near_collinear_diagonal_fraction": 0.0008,
    "cluster_gap_diagonal_fraction": 0.0060,
    "short_segment_ratio_warning": 0.15,
    "short_segment_ratio_failure": 0.30,
    "near_collinear_ratio_warning": 0.55,
    "near_collinear_ratio_failure": 0.75,
    "anchors_per_100_warning": 5.0,
    "anchors_per_100_failure": 8.0,
    "high_node_path_warning": 40.0,
    "high_node_path_failure": 80.0,
    "gradient_usage_ratio_warning": 0.15,
    "solid_fragment_warning": 5.0,
    "solid_fragment_failure": 8.0,
    "solid_paint_warning": 3.0,
    "solid_paint_failure": 4.0,
    # Reusing one native gradient for a handful of related shapes remains
    # editable.  Large fan-outs are usually evidence that one painted object
    # was still emitted as trace fragments.
    "gradient_fill_object_warning": 3.0,
    "gradient_fill_object_failure": 8.0,
}


_ECONOMY_OBJECTIVE_REQUIRED = {
    "preserve_topology_hard_constraint",
    "p95_bidirectional_geometric_error_percent_within_budget_hard_constraint",
    "maximum_error_within_three_times_budget_hard_tail_constraint",
    "over_budget_share_at_most_5_percent_hard_tail_constraint",
    "salient_corner_error_within_two_times_budget_hard_constraint",
    "minimize_designer_anchor_count",
    "minimize_anchor_count",
    "minimize_fragment_count",
    "minimize_segment_count",
}


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
        style = _style_map(node.get("style"))
        if name in style:
            return style[name]
        if name in node.attrib:
            return node.attrib[name]
        node = parents.get(node)
    return default


def _number(value: Any) -> float | None:
    match = _NUMBER.search("" if value is None else str(value))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return None


def _first_property(element: ET.Element,
                    parents: Mapping[ET.Element, ET.Element],
                    names: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Return the first present property and its canonical source name."""
    for name in names:
        value = _property(element, parents, name, "").strip()
        if value:
            return value, name
    return None, None


def _geometry_evidence(element: ET.Element,
                       parents: Mapping[ET.Element, ET.Element]) -> dict[str, Any]:
    """Read the error-bounded geometry contract without trusting it as shape.

    Beta.5 emitted ``p95-error`` / ``max-error`` while some experimental files
    used ``error-p95`` / ``error-max``.  Accept both spellings so old reports
    remain auditable.  A native SVG circle has four designer anchors by policy;
    metadata may describe provenance but cannot change that semantic count.
    """
    raw: dict[str, str | None] = {}
    sources: dict[str, str | None] = {}
    values: dict[str, float | None] = {}
    for key, names in _GEOMETRY_EVIDENCE_ALIASES.items():
        raw[key], sources[key] = _first_property(element, parents, names)
        values[key] = _number(raw[key])

    # Identity artwork has no approximation to prove.  The error contract is
    # applicable only when the pipeline claims a geometry-changing operation,
    # emits any part of the contract, or supplies a native-gradient fill whose
    # legacy geometry has not otherwise been certified.
    has_contract_attribute = any(value is not None for value in raw.values())
    has_transform_claim = any(
        _property(element, parents, name, "").strip()
        for name in _GEOMETRY_TRANSFORM_CLAIMS
    )
    has_gradient_fill = bool(_URL_REF.search(
        _property(element, parents, "fill", "")))
    applicable = bool(
        has_contract_attribute or has_transform_claim or has_gradient_fill)

    missing = [key for key, value in raw.items() if value is None]
    invalid: list[str] = []
    for key in ("error_budget_percent", "p95_error_percent",
                "max_error_percent"):
        value = values[key]
        if raw[key] is not None and (value is None or value < 0
                                     or (key == "error_budget_percent"
                                         and value <= 0)):
            invalid.append(key)
    designer_value = values["designer_anchors"]
    if raw["designer_anchors"] is not None and (
            designer_value is None or designer_value < 1
            or abs(designer_value - round(designer_value)) > 1e-9):
        invalid.append("designer_anchors")

    kind = _local(element.tag)
    claimed_designer = (None if designer_value is None else
                        int(round(designer_value)))
    designer_source = sources["designer_anchors"]
    if kind == "circle":
        # A native circle is controlled by centre + radius and maps to four
        # designer anchors when expanded to Beziers.  Never count a metadata
        # typo as 1, 8, or the number of sampled raster points.
        values["designer_anchors"] = 4.0
        designer_source = "native_circle_policy"
        if "designer_anchors" in missing:
            missing.remove("designer_anchors")
        if "designer_anchors" in invalid:
            invalid.remove("designer_anchors")

    budget = values["error_budget_percent"]
    p95 = values["p95_error_percent"]
    maximum = values["max_error_percent"]
    p95_violation = bool(
        budget is not None and budget > 0 and p95 is not None and p95 >= 0
        and p95 > budget + 1e-12)
    max_violation = bool(
        budget is not None and budget > 0 and maximum is not None
        and maximum >= 0 and maximum > 3.0 * budget + 1e-12)
    return {
        "applicable": applicable,
        "error_budget_percent": _rounded(budget),
        "p95_error_percent": _rounded(p95),
        "max_error_percent": _rounded(maximum),
        "designer_anchors": (None if values["designer_anchors"] is None else
                             int(round(values["designer_anchors"]))),
        "designer_anchors_source": designer_source,
        "claimed_designer_anchors": claimed_designer,
        "attribute_sources": sources,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "complete": not missing and not invalid,
        "p95_budget_violation": p95_violation,
        "max_tail_violation": max_violation,
        "native_circle_claim_normalized": bool(
            kind == "circle" and claimed_designer is not None
            and claimed_designer != 4),
    }


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rounded(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _ratio(numerator: float, denominator: float) -> float | None:
    return _rounded(numerator / denominator) if denominator > 0 else None


def _viewbox(root: ET.Element) -> list[float]:
    values = [float(item) for item in _NUMBER.findall(root.get("viewBox", ""))]
    if len(values) == 4 and values[2] > 0 and values[3] > 0:
        return values
    width = _number(root.get("width"))
    height = _number(root.get("height"))
    if width and height and width > 0 and height > 0:
        return [0.0, 0.0, width, height]
    return [0.0, 0.0, 1.0, 1.0]


def _load_json(value: Mapping[str, Any] | str | Path | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    raw = str(value)
    if raw.lstrip().startswith("{"):
        loaded = json.loads(raw)
    else:
        loaded = json.loads(Path(value).read_text(encoding="utf-8-sig"))
    if not isinstance(loaded, dict):
        raise ValueError("audit metadata must be a JSON object")
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


def _normalise_colour(value: str) -> tuple[str, tuple[int, int, int]] | None:
    token = value.strip().lower()
    match = _HEX.fullmatch(token)
    if match:
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(channel * 2 for channel in digits)
        rgb = tuple(int(digits[index:index + 2], 16) for index in (0, 2, 4))
        return "#" + digits, rgb  # type: ignore[return-value]
    match = _RGB.match(token)
    if not match:
        return None
    percent = "%" in token[:match.end()]
    values = [float(item) for item in match.groups()]
    if percent:
        values = [item * 2.55 for item in values]
    rgb = tuple(max(0, min(255, round(item))) for item in values)
    return "#" + "".join(f"{item:02x}" for item in rgb), rgb  # type: ignore[return-value]


def _bbox_union(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    ]


def _bbox_gap(first: list[float], second: list[float]) -> float:
    dx = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    dy = max(0.0, max(first[1], second[1]) - min(first[3], second[3]))
    return math.hypot(dx, dy)


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _point(value_x: float, value_y: float, relative: bool,
           current: tuple[float, float]) -> tuple[float, float]:
    return ((current[0] + value_x, current[1] + value_y)
            if relative else (value_x, value_y))


def _path_segments(path_data: str) -> tuple[list[dict[str, Any]], bool]:
    """Expand common SVG path commands into absolute geometric segments."""
    tokens = _PATH_TOKEN.findall(path_data or "")
    segments: list[dict[str, Any]] = []
    index = 0
    command: str | None = None
    current = (0.0, 0.0)
    subpath = current
    last_cubic_control: tuple[float, float] | None = None
    last_quad_control: tuple[float, float] | None = None
    valid = True

    def available(count: int) -> bool:
        return index + count <= len(tokens) and not any(
            len(tokens[position]) == 1 and tokens[position].isalpha()
            for position in range(index, index + count)
        )

    while index < len(tokens):
        token = tokens[index]
        if len(token) == 1 and token.isalpha():
            command = token
            index += 1
            if command in "Zz":
                segments.append({"type": "Z", "start": current, "end": subpath,
                                 "controls": []})
                current = subpath
                last_cubic_control = last_quad_control = None
                command = None
            continue
        if command is None:
            valid = False
            break
        relative = command.islower()
        kind = command.upper()
        counts = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
                  "Q": 4, "T": 2, "A": 7}
        count = counts.get(kind)
        if count is None or not available(count):
            valid = False
            break
        values = [float(tokens[index + offset]) for offset in range(count)]
        index += count
        start = current
        controls: list[tuple[float, float]] = []
        arc: dict[str, Any] | None = None
        segment_kind = kind
        if kind == "M":
            current = _point(values[0], values[1], relative, current)
            subpath = current
            segments.append({"type": "M", "start": current, "end": current,
                             "controls": []})
            command = "l" if relative else "L"
            last_cubic_control = last_quad_control = None
            continue
        if kind == "L":
            current = _point(values[0], values[1], relative, current)
        elif kind == "H":
            current = ((current[0] + values[0]) if relative else values[0],
                       current[1])
            segment_kind = "L"
        elif kind == "V":
            current = (current[0],
                       (current[1] + values[0]) if relative else values[0])
            segment_kind = "L"
        elif kind == "C":
            controls = [_point(values[0], values[1], relative, start),
                        _point(values[2], values[3], relative, start)]
            current = _point(values[4], values[5], relative, start)
            last_cubic_control = controls[-1]
            last_quad_control = None
        elif kind == "S":
            reflected = (start if last_cubic_control is None else
                         (2 * start[0] - last_cubic_control[0],
                          2 * start[1] - last_cubic_control[1]))
            controls = [reflected, _point(values[0], values[1], relative, start)]
            current = _point(values[2], values[3], relative, start)
            last_cubic_control = controls[-1]
            last_quad_control = None
            segment_kind = "C"
        elif kind == "Q":
            controls = [_point(values[0], values[1], relative, start)]
            current = _point(values[2], values[3], relative, start)
            last_quad_control = controls[-1]
            last_cubic_control = None
        elif kind == "T":
            reflected = (start if last_quad_control is None else
                         (2 * start[0] - last_quad_control[0],
                          2 * start[1] - last_quad_control[1]))
            controls = [reflected]
            current = _point(values[0], values[1], relative, start)
            last_quad_control = reflected
            last_cubic_control = None
            segment_kind = "Q"
        elif kind == "A":
            current = _point(values[5], values[6], relative, start)
            arc = {
                "rx": abs(values[0]),
                "ry": abs(values[1]),
                "rotation_degrees": values[2],
                "large_arc": bool(round(values[3])),
                "sweep": bool(round(values[4])),
            }
            last_cubic_control = last_quad_control = None
        if segment_kind not in {"C", "Q"}:
            last_cubic_control = last_quad_control = None
        segment = {"type": segment_kind, "start": start, "end": current,
                   "controls": controls}
        if arc is not None:
            segment["arc"] = arc
        segments.append(segment)
    return segments, valid


def _path_loops(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measure each ``M`` subpath independently.

    A compound path may contain dozens of small, intentionally separate loops.
    Summing all of them and calling that one over-anchored curve is a category
    error, so the hard economy limits are evaluated on these loop records.
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for segment in segments:
        if segment["type"] == "M":
            if current:
                groups.append(current)
            current = [segment]
        else:
            if not current:
                current = [{"type": "M", "start": segment["start"],
                            "end": segment["start"], "controls": []}]
            current.append(segment)
    if current:
        groups.append(current)

    loops: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        drawable = [segment for segment in group
                    if segment["type"] not in {"M", "Z"}]
        length_segments = [segment for segment in group
                           if segment["type"] != "M"]
        length = sum(_segment_length(segment) for segment in length_segments)
        # ``M`` names the first anchor; it is not an additional designer node.
        # Most traced closed paths explicitly end their last curve at that same
        # point before ``Z``.  Count unique anchors, while retaining the extra
        # start anchor for an open path or an implicit closing line.
        nodes = 1 + len(drawable) if group else 0
        closed = any(segment["type"] == "Z" for segment in group)
        if closed and drawable:
            first = group[0]["end"]
            last = drawable[-1]["end"]
            if _distance(first, last) <= 1.0e-7:
                nodes -= 1
        density = _rounded(100.0 * nodes / length) if length > 0 else None
        loops.append({
            "index": index,
            "node_count": nodes,
            "command_count": len(group),
            "draw_segment_count": len(drawable),
            "approximated_length": length,
            "anchors_per_100_user_units": density,
            "closed": closed,
            "bbox": _path_bbox(group),
            "endpoints": [segment["end"] for segment in drawable],
        })
    return loops


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _line_deviation(point: tuple[float, float], start: tuple[float, float],
                    end: tuple[float, float]) -> float:
    denominator = _distance(start, end)
    if denominator <= 1e-12:
        return _distance(point, start)
    return abs((end[0] - start[0]) * (start[1] - point[1])
               - (start[0] - point[0]) * (end[1] - start[1])) / denominator


def _curve_point(segment: dict[str, Any], t: float) -> tuple[float, float]:
    start = segment["start"]
    end = segment["end"]
    controls = segment["controls"]
    one = 1.0 - t
    if segment["type"] == "C":
        first, second = controls
        return (
            one ** 3 * start[0] + 3 * one ** 2 * t * first[0]
            + 3 * one * t ** 2 * second[0] + t ** 3 * end[0],
            one ** 3 * start[1] + 3 * one ** 2 * t * first[1]
            + 3 * one * t ** 2 * second[1] + t ** 3 * end[1],
        )
    if segment["type"] == "Q":
        control = controls[0]
        return (one ** 2 * start[0] + 2 * one * t * control[0] + t ** 2 * end[0],
                one ** 2 * start[1] + 2 * one * t * control[1] + t ** 2 * end[1])
    if segment["type"] == "A":
        parameters = _arc_center_parameters(segment)
        if parameters is not None:
            cx, cy, rx, ry, phi, theta, delta = parameters
            angle = theta + float(t) * delta
            cosine, sine = math.cos(angle), math.sin(angle)
            return (cx + math.cos(phi) * rx * cosine
                    - math.sin(phi) * ry * sine,
                    cy + math.sin(phi) * rx * cosine
                    + math.cos(phi) * ry * sine)
    return (start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1]))


def _arc_center_parameters(segment: dict[str, Any]):
    """Convert SVG endpoint arc parameters to a centre representation."""
    arc = segment.get("arc") or {}
    rx, ry = float(arc.get("rx") or 0.0), float(arc.get("ry") or 0.0)
    start, end = segment["start"], segment["end"]
    if rx <= 1.0e-12 or ry <= 1.0e-12 or _distance(start, end) <= 1.0e-12:
        return None
    phi = math.radians(float(arc.get("rotation_degrees") or 0.0) % 360.0)
    cosine, sine = math.cos(phi), math.sin(phi)
    dx, dy = (start[0] - end[0]) * 0.5, (start[1] - end[1]) * 0.5
    x1p = cosine * dx + sine * dy
    y1p = -sine * dx + cosine * dy
    scale = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if scale > 1.0:
        factor = math.sqrt(scale)
        rx *= factor
        ry *= factor
    numerator = max(
        0.0,
        rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p,
    )
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coefficient = (math.sqrt(numerator / denominator)
                   if denominator > 1.0e-18 else 0.0)
    if bool(arc.get("large_arc")) == bool(arc.get("sweep")):
        coefficient = -coefficient
    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx
    cx = cosine * cxp - sine * cyp + (start[0] + end[0]) * 0.5
    cy = sine * cxp + cosine * cyp + (start[1] + end[1]) * 0.5

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta = math.atan2(uy, ux)
    delta = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
    if not bool(arc.get("sweep")) and delta > 0.0:
        delta -= 2.0 * math.pi
    elif bool(arc.get("sweep")) and delta < 0.0:
        delta += 2.0 * math.pi
    return cx, cy, rx, ry, phi, theta, delta


def _segment_length(segment: dict[str, Any]) -> float:
    if segment["type"] not in {"C", "Q", "A"}:
        return _distance(segment["start"], segment["end"])
    previous = segment["start"]
    total = 0.0
    steps = 24 if segment["type"] == "A" else 8
    for step in range(1, steps + 1):
        point = _curve_point(segment, step / float(steps))
        total += _distance(previous, point)
        previous = point
    return total


def _short_counterturn_diagnostic(element: ET.Element,
                                  parents: Mapping[ET.Element, ET.Element],
                                  segments: list[dict[str, Any]],
                                  diagonal: float) -> dict[str, Any]:
    """Read-only sampled shape hint, independent of distance certificates.

    A small cubic can stay inside its distance budget while doubling back.
    One-way round turns are intentionally excluded. This is not a proof of
    unintended geometry, a source comparison, or a command-count statistic.
    """
    from svg_bounds import IDENTITY, multiply, parse_transform
    limit = diagonal * .006
    result: dict[str, Any] = {
        "status": "completed", "short_cubic_count": 0,
        "high_turn_short_cubic_count": 0, "counterturn_segment_count": 0,
        "requires_review": False, "bbox": None, "segments": [],
        "segment_record_limit": 12, "segment_records_truncated": False,
        "coordinate_scope": "canvas_user_units_after_svg_affine_transforms",
    }
    chain = []
    node = element
    while node is not None:
        chain.append(node)
        node = parents.get(node)
    matrix = IDENTITY
    try:
        for node in reversed(chain):
            if "transform" in _style_map(node.get("style")):
                raise ValueError("css_transform_requires_layout")
            if _local(node.tag) == "svg" and parents.get(node) is not None:
                raise ValueError("nested_svg_viewport_requires_layout")
            matrix = multiply(matrix, parse_transform(node.get("transform", "")))
        if not all(math.isfinite(value) for value in matrix):
            raise ValueError("nonfinite_transform")
    except ValueError as exc:
        result.update(status="unassessed", reason=str(exc))
        return result
    a, b, c, d, e, f = matrix

    def point(value):
        x, y = value
        return (a*x+c*y+e, b*x+d*y+f)

    boxes = []
    for index, segment in enumerate(segments, 1):
        if segment["type"] != "C":
            continue
        transformed = {
            "type": "C", "start": point(segment["start"]),
            "end": point(segment["end"]),
            "controls": [point(value) for value in segment["controls"]],
        }
        coordinates = [transformed["start"], transformed["end"], *transformed["controls"]]
        if not all(math.isfinite(value) for position in coordinates for value in position):
            result.update(status="partially_assessed", reason="nonfinite_path_geometry")
            continue
        chord = _distance(transformed["start"], transformed["end"])
        if chord > limit:
            continue
        result["short_cubic_count"] += 1
        samples = [_curve_point(transformed, step/23.0) for step in range(24)]
        directions = [math.atan2(right[1]-left[1], right[0]-left[0])
                      for left, right in zip(samples, samples[1:])
                      if _distance(left, right) > 1e-9]
        turns = [math.degrees((right-left+math.pi) % (2*math.pi)-math.pi)
                 for left, right in zip(directions, directions[1:])]
        positive = sum(max(0.0, angle) for angle in turns)
        negative = sum(max(0.0, -angle) for angle in turns)
        total = positive + negative
        if total < 120:
            continue
        result["high_turn_short_cubic_count"] += 1
        counterturn = min(positive, negative)
        if counterturn < 45:
            continue
        result["counterturn_segment_count"] += 1
        box = _path_bbox([transformed])
        if box is not None:
            boxes.append(box)
        if len(result["segments"]) < result["segment_record_limit"]:
            result["segments"].append({
                "parsed_segment_index_1_based": index,
                "endpoint_chord": _rounded(chord),
                "sampled_total_turn_degrees": _rounded(total, 3),
                "sampled_counterturn_degrees": _rounded(counterturn, 3),
                "bbox": [_rounded(value, 4) for value in box] if box else None,
            })
    result["requires_review"] = result["counterturn_segment_count"] >= 4
    result["segment_records_truncated"] = result["counterturn_segment_count"] > len(result["segments"])
    box = _bbox_union(boxes) if boxes else None
    result["bbox"] = [_rounded(value, 4) for value in box] if box else None
    return result


def _path_bbox(segments: list[dict[str, Any]]) -> list[float] | None:
    points: list[tuple[float, float]] = []
    for segment in segments:
        points.extend([segment["start"], segment["end"]])
        if segment["type"] in {"C", "Q"}:
            points.extend(segment["controls"])
        elif segment["type"] == "A":
            points.extend(_curve_point(segment, step / 32.0)
                          for step in range(1, 32))
    if not points:
        return None
    return [min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points)]


def _element_bbox(element: ET.Element,
                  path_segments: list[dict[str, Any]] | None = None) -> list[float] | None:
    kind = _local(element.tag)
    if kind == "path":
        return _path_bbox(path_segments or [])
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
        width = _number(element.get("width"))
        height = _number(element.get("height"))
        if width is not None and height is not None:
            return [x, y, x + width, y + height]
    if kind == "line":
        values = [_number(element.get(name)) for name in ("x1", "y1", "x2", "y2")]
        if None not in values:
            x1, y1, x2, y2 = values
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    if kind in {"polygon", "polyline"}:
        values = [float(item) for item in _NUMBER.findall(element.get("points", ""))]
        points = list(zip(values[0::2], values[1::2]))
        if points:
            return [min(x for x, _ in points), min(y for _, y in points),
                    max(x for x, _ in points), max(y for _, y in points)]
    return None


def _gradient_inventory(root: ET.Element,
                        parents: Mapping[ET.Element, ET.Element]) -> dict[str, Any]:
    resources: dict[str, ET.Element] = {}
    duplicates: set[str] = set()
    anonymous = 0
    for element in root.iter():
        if _local(element.tag) not in GRADIENTS:
            continue
        identifier = (element.get("id") or "").strip()
        if not identifier:
            anonymous += 1
        elif identifier in resources:
            duplicates.add(identifier)
        else:
            resources[identifier] = element

    references: Counter[str] = Counter()
    usage_elements: Counter[str] = Counter()
    fill_usage_elements: Counter[str] = Counter()
    stroke_usage_elements: Counter[str] = Counter()
    invalid_references: Counter[str] = Counter()
    for element in root.iter():
        if _local(element.tag) not in DRAWABLES:
            continue
        for name in ("fill", "stroke"):
            value = _property(element, parents, name, "")
            for identifier in _URL_REF.findall(value):
                references[identifier] += 1
                usage_elements[identifier] += 1
                if name == "fill":
                    fill_usage_elements[identifier] += 1
                else:
                    stroke_usage_elements[identifier] += 1
                if identifier not in resources:
                    invalid_references[identifier] += 1

    def stop_count(identifier: str, stack: set[str] | None = None) -> int:
        stack = set(stack or ())
        if identifier in stack or identifier not in resources:
            return 0
        stack.add(identifier)
        element = resources[identifier]
        own = sum(_local(child.tag) == "stop" for child in element)
        if own:
            return own
        href = element.get("href") or element.get(
            "{http://www.w3.org/1999/xlink}href", "")
        return stop_count(href[1:], stack) if href.startswith("#") else 0

    items = []
    for identifier, element in sorted(resources.items()):
        stops = stop_count(identifier)
        items.append({
            "id": identifier,
            "type": _local(element.tag),
            "usage_count": usage_elements[identifier],
            "fill_usage_count": fill_usage_elements[identifier],
            "stroke_usage_count": stroke_usage_elements[identifier],
            "resolved_stop_count": stops,
            "valid_editable_resource": bool(
                identifier not in duplicates and usage_elements[identifier] > 0
                and 2 <= stops <= 5
            ),
        })
    return {
        "resource_count": len(resources) + anonymous,
        "linear_resource_count": sum(_local(item.tag) == "linearGradient"
                                     for item in resources.values()),
        "radial_resource_count": sum(_local(item.tag) == "radialGradient"
                                     for item in resources.values()),
        "usage_count": sum(usage_elements.values()),
        "fill_usage_count": sum(fill_usage_elements.values()),
        "stroke_usage_count": sum(stroke_usage_elements.values()),
        "used_resource_count": sum(item["usage_count"] > 0 for item in items),
        "valid_used_resource_count": sum(item["valid_editable_resource"] for item in items),
        "anonymous_resource_count": anonymous,
        "duplicate_ids": sorted(duplicates),
        "invalid_references": dict(sorted(invalid_references.items())),
        "resources": items,
    }


def _colour_family_close(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_rgb = first["rgb"]
    second_rgb = second["rgb"]
    euclidean = math.sqrt(sum((a - b) ** 2 for a, b in zip(first_rgb, second_rgb)))
    if euclidean > 105:
        return False
    first_h, first_s, first_v = first["hsv"]
    second_h, second_s, second_v = second["hsv"]
    hue_distance = min(abs(first_h - second_h), 1.0 - abs(first_h - second_h))
    return hue_distance <= 0.13 and abs(first_v - second_v) <= 0.42 \
        and min(first_s, second_s) >= 0.10


def _solid_fragment_clusters(records: list[dict[str, Any]], diagonal: float,
                             viewbox_area: float,
                             thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    eligible = [record for record in records if record.get("chromatic")
                and record.get("bbox") is not None]
    parent = list(range(len(eligible)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    gap_limit = max(1.0, diagonal * thresholds["cluster_gap_diagonal_fraction"])
    for first in range(len(eligible)):
        for second in range(first):
            if not _colour_family_close(eligible[first], eligible[second]):
                continue
            if _bbox_gap(eligible[first]["bbox"], eligible[second]["bbox"]) <= gap_limit:
                union(first, second)

    components: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(eligible):
        components.setdefault(find(index), []).append(record)
    clusters = []
    for members in components.values():
        paints = sorted({member["fill"] for member in members})
        if len(members) < 3 or len(paints) < 2:
            continue
        box = _bbox_union([member["bbox"] for member in members])
        assert box is not None
        area_fraction = _bbox_area(box) / viewbox_area if viewbox_area > 0 else 0.0
        clusters.append({
            "member_count": len(members),
            "distinct_solid_paint_count": len(paints),
            "solid_paints": paints[:20],
            "bbox": [_rounded(item, 3) for item in box],
            "bbox_area_fraction": _rounded(area_fraction),
            "member_ids": [member["id"] for member in members[:40]],
            "truncated_member_ids": len(members) > 40,
            "severity": (
                "failure" if len(members) >= thresholds["solid_fragment_failure"]
                and len(paints) >= thresholds["solid_paint_failure"]
                else "warning"
            ),
            "inference_scope": "spatial_colour_family_candidate_not_semantic_proof",
        })
    return sorted(clusters, key=lambda item: (-item["member_count"],
                                               -item["distinct_solid_paint_count"]))


def _proposal_groups(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "available": False,
            "drawable_count": None,
            "candidate_group_count": 0,
            "actual_dom_group_count": 0,
            "unresolved_group_count": 0,
            "candidate_member_coverage_rate": None,
            "selectable_member_coverage_rate": None,
            "unresolved_member_coverage_rate": None,
            "band_fragmentation_candidates": [],
        }
    nested = report
    for key in ("scene_graph_report", "scene_graph", "gradient_object_gate",
                "gradient_object_report"):
        value = nested.get(key)
        if isinstance(value, Mapping):
            nested = dict(value)
            break
    actual_raw = nested.get("actual_dom_groups", nested.get("groups", []))
    unresolved_raw = nested.get(
        "manifest_only_groups", nested.get("skipped_unsafe_groups", []))
    actual = [dict(item) for item in actual_raw if isinstance(item, Mapping)] \
        if isinstance(actual_raw, list) else []
    unresolved = [dict(item) for item in unresolved_raw if isinstance(item, Mapping)] \
        if isinstance(unresolved_raw, list) else []
    all_groups = actual + unresolved
    drawable_count = _integer(nested.get("drawable_count"), 0)

    def member_ids(groups: list[dict[str, Any]]) -> set[str]:
        return {str(identifier) for item in groups
                for identifier in item.get("node_ids", []) if identifier}

    actual_ids = member_ids(actual)
    unresolved_ids = member_ids(unresolved)
    all_ids = actual_ids | unresolved_ids
    if not actual_ids and drawable_count > 0:
        grouped = _integer(nested.get("grouped_drawables"), 0)
        actual_count = min(grouped, drawable_count)
    else:
        actual_count = len(actual_ids)
    candidates = []
    for item in all_groups:
        member_count = _integer(item.get("member_count"), len(item.get("node_ids", [])))
        paint_count = _integer(item.get("paint_count"), len(item.get("paints", [])))
        reasons = [str(reason) for reason in item.get("reasons", [])]
        # Scene grouping reasons such as proximity/overlap describe selectable
        # object structure, not a continuous colour field.  Only an explicit
        # gradient-band classification may corroborate band fragmentation.
        band_reason = any(reason in {
            "band-fragmentation", "gradient-band-proximity",
        } for reason in reasons)
        if member_count >= 3 and paint_count >= 2 and band_reason:
            candidates.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "mode": item.get("mode", "actual-dom"),
                "member_count": member_count,
                "paint_count": paint_count,
                "bbox": item.get("bbox"),
                "reasons": reasons,
                "unresolved": item in unresolved,
                "not_applied_reason": item.get("not_applied_reason"),
            })
    denominator = float(drawable_count)
    return {
        "available": True,
        "drawable_count": drawable_count or None,
        "candidate_group_count": _integer(nested.get("candidate_groups"),
                                          len(all_groups)),
        "actual_dom_group_count": len(actual),
        "unresolved_group_count": len(unresolved),
        "candidate_member_coverage_rate": _ratio(len(all_ids), denominator),
        "selectable_member_coverage_rate": _ratio(actual_count, denominator),
        "unresolved_member_coverage_rate": _ratio(len(unresolved_ids), denominator),
        "band_fragmentation_candidates": sorted(
            candidates, key=lambda item: (-item["member_count"], -item["paint_count"])
        ),
    }


def _source_space_gradient_field_evidence(
        report: dict[str, Any] | None, gradient_drawable_count: int) -> dict[str, Any]:
    """Validate authoritative source-space field ownership evidence.

    Global colour-family clustering is intentionally only a candidate detector:
    a logo may contain many unrelated green words, leaves, outlines and flat
    decorations.  It can be superseded only by the reconstruction stage's
    stronger source-space proof: complete held-out paint validation, material
    hard-edge rejection, independent salted confirmation, topology/error
    evidence and disjoint selected ownership masks.
    """
    base = {
        "available": False,
        "authoritative": False,
        "failure_reasons": [],
        "stage_status": None,
        "objects_selected": None,
        "gradient_drawable_count": int(gradient_drawable_count),
        "selected_candidate_ids": [],
        "detail_candidate_ids": [],
        "objects": [],
        "partial_paint_fields": [],
        "scope": (
            "source_space_complete_field_ownership_not_global_colour_frequency"
        ),
    }
    if not isinstance(report, Mapping):
        return base
    stage = report.get("gradient_reconstruction_report")
    details = report.get("gradient_details")
    if stage is None and details is None:
        return base
    base["available"] = True
    failures: list[str] = []
    if not isinstance(stage, Mapping):
        failures.append("gradient_reconstruction_report_missing")
        stage = {}
    if not isinstance(details, list):
        failures.append("gradient_details_missing")
        details = []

    def partial_paint(detail):
        if not isinstance(detail, Mapping):
            return False
        validation = detail.get('validation')
        geometry = validation.get('geometry') if isinstance(validation, Mapping) else None
        proof = geometry.get('source_paint_only') if isinstance(geometry, Mapping) else None
        return isinstance(proof, Mapping) and proof.get('partial_selection') is True

    partial_details = [item for item in details if partial_paint(item)]
    complete_details = [item for item in details if not partial_paint(item)]
    base['partial_paint_fields'] = [item.get('candidate_id') for item in partial_details]

    summary = stage.get("summary") if isinstance(stage.get("summary"), Mapping) else {}
    objective = (stage.get("objective")
                 if isinstance(stage.get("objective"), Mapping) else {})
    decisions = stage.get("decisions") if isinstance(stage.get("decisions"), list) else []
    base["stage_status"] = stage.get("status")
    objects_selected = _integer(summary.get("objects_selected"), -1)
    base["objects_selected"] = objects_selected
    required_constraints = {
        "paint_model_beats_solid_on_deterministic_heldout",
        "no_material_internal_hard_edge",
        "topology_preserved",
        "p95_geometry_error_percent_within_budget",
        "maximum_geometry_error_within_three_times_budget_tail",
        "selected_masks_are_pairwise_disjoint",
    }
    constraints = set(objective.get("hard_constraints", [])) \
        if isinstance(objective.get("hard_constraints"), list) else set()
    if stage.get("schema") != "ai-vector-cleanroom.gradient-reconstruction-stage/v1":
        failures.append("gradient_reconstruction_schema_invalid")
    if stage.get("status") != "proposed" and not (
            partial_details and objects_selected == 0 and stage.get('status') == 'skipped'):
        failures.append("gradient_reconstruction_not_proposed")
    if not required_constraints.issubset(constraints):
        failures.append("gradient_reconstruction_hard_constraints_incomplete")
    from gradient_paint_only import paint_only_certificate_valid
    expected_drawables = 0
    for item in details:
        validation_item = item.get("validation") if isinstance(item, Mapping) else None
        geometry_item = validation_item.get("geometry") if isinstance(validation_item, Mapping) else None
        if isinstance(geometry_item, Mapping) and paint_only_certificate_valid(geometry_item):
            expected_drawables += len(geometry_item["source_paint_only"]["drawable_records"])
        else:
            expected_drawables += 1
    if expected_drawables != int(gradient_drawable_count) \
            or objects_selected != len(complete_details):
        failures.append("selected_object_count_mismatch")
    if partial_details and _integer(summary.get('partial_paint_fields'), -1) != len(partial_details):
        failures.append('partial_paint_field_count_mismatch')
    if _integer(summary.get("overlap_pixels_between_selected"), -1) != 0:
        failures.append("selected_ownership_masks_overlap")
    if _integer(summary.get("geometry_shortlist_deferred"), -1) != 0:
        failures.append("geometry_shortlist_not_exhausted")

    selected_ids = [str(item.get("candidate_id")) for item in decisions
                    if isinstance(item, Mapping)
                    and item.get("status") == "selected"
                    and item.get("candidate_id")]
    detail_ids = [str(item.get("candidate_id")) for item in complete_details
                  if isinstance(item, Mapping) and item.get("candidate_id")]
    base["selected_candidate_ids"] = sorted(selected_ids)
    base["detail_candidate_ids"] = sorted(detail_ids)
    if (len(selected_ids) != objects_selected
            or len(set(selected_ids)) != len(selected_ids)
            or sorted(selected_ids) != sorted(detail_ids)):
        failures.append("selected_candidate_identity_mismatch")

    object_evidence = []
    for index, raw_detail in enumerate(details):
        item_failures = []
        if not isinstance(raw_detail, Mapping):
            item_failures.append("detail_not_mapping")
            raw_detail = {}
        candidate_id = raw_detail.get("candidate_id")
        stops = raw_detail.get("stops")
        validation = (raw_detail.get("validation")
                      if isinstance(raw_detail.get("validation"), Mapping)
                      else {})
        paint = (validation.get("paint")
                 if isinstance(validation.get("paint"), Mapping) else {})
        paint_validation = (paint.get("validation")
                            if isinstance(paint.get("validation"), Mapping)
                            else {})
        edges = (paint_validation.get("internal_edges")
                 if isinstance(paint_validation.get("internal_edges"), Mapping)
                 else {})
        independent = (paint.get("independent_revalidation")
                       if isinstance(paint.get("independent_revalidation"), Mapping)
                       else {})
        geometry = (validation.get("geometry")
                    if isinstance(validation.get("geometry"), Mapping) else {})
        topology = (geometry.get("topology")
                    if isinstance(geometry.get("topology"), Mapping) else {})
        error_budget = (geometry.get("error_budget")
                        if isinstance(geometry.get("error_budget"), Mapping)
                        else {})
        selection_evidence = (
            geometry.get("selection_evidence")
            if isinstance(geometry.get("selection_evidence"), Mapping) else {})
        final_consistency = (
            geometry.get("final_svg_consistency")
            if isinstance(geometry.get("final_svg_consistency"), Mapping) else {})
        selection = (validation.get("selection")
                     if isinstance(validation.get("selection"), Mapping) else {})
        from source_gradient_primitive import (
            SOLVER as SOURCE_CIRCLE_SOLVER, RECT_SOLVER, source_primitive_certificate_valid)
        source_circle = geometry.get("solver") in {SOURCE_CIRCLE_SOLVER, RECT_SOLVER}
        source_circle_valid = source_primitive_certificate_valid(geometry) if source_circle else False
        from gradient_contour_spans import source_contour_spans_certificate_valid
        source_spans = "source_contour_spans" in geometry
        source_spans_valid = source_contour_spans_certificate_valid(geometry) if source_spans else False
        if source_spans and not source_spans_valid:
            item_failures.append("source_contour_span_certificate_invalid")
        from source_edge_reconstruction import source_edge_certificate_valid
        source_edge = "source_edge_reconstruction" in geometry
        source_edge_valid = (source_edge_certificate_valid(
            geometry, require_scene_commit=True) if source_edge else False)
        if source_edge and (not source_edge_valid
                            or final_consistency.get("source_edge_reconstruction_verified") is not True):
            item_failures.append("source_edge_reconstruction_certificate_invalid")
        paint_only = "source_paint_only" in geometry
        paint_only_valid = paint_only_certificate_valid(geometry) if paint_only else False
        is_partial_paint = partial_paint(raw_detail)
        if is_partial_paint and (geometry['source_paint_only'].get('manual_review_required') is not True
                                 or geometry['source_paint_only'].get('geometry_optimized') is not False):
            item_failures.append('partial_paint_must_retain_manual_review_without_geometry_claim')
        if paint_only and (not paint_only_valid
                           or selection.get("geometry_source") != "unchanged_existing_svg_geometry"
                           or final_consistency.get("status") != "verified_unchanged"
                           or final_consistency.get("source_original_rgba_verified") is not True
                           or final_consistency.get("final_drawable_ids") != [r["id"] for r in geometry.get("source_paint_only", {}).get("drawable_records", [])]):
            item_failures.append("source_paint_only_certificate_invalid")
        if not candidate_id:
            item_failures.append("candidate_id_missing")
        if not isinstance(stops, list) or not 2 <= len(stops) <= 5:
            item_failures.append("native_stop_count_outside_2_to_5")
        if validation.get("engine") != "source_space_heldout_gradient_object":
            item_failures.append("source_space_engine_missing")
        if paint_validation.get("passed") is not True:
            item_failures.append("heldout_validation_not_passed")
        if (edges.get("material_internal_hard_edge") is not False
                or edges.get("material_hard_label_boundary") is not False):
            item_failures.append("material_hard_edge_not_rejected")
        if independent.get("passed") is not True:
            item_failures.append("independent_salted_revalidation_not_passed")
        if topology.get("topology_preserved") is not True:
            item_failures.append("topology_not_preserved")
        if error_budget.get("passed") is not True:
            item_failures.append("geometry_error_contract_not_passed")
        if source_circle and (not source_circle_valid
                              or selection.get("colour_used_for_geometry") is not True
                              or selection.get("geometry_source") != "unmodified_input_coverage50_with_verified_gradient_paint"):
            item_failures.append("source_gradient_primitive_certificate_invalid")
        elif source_edge and (not source_edge_valid
                              or selection.get("colour_used_for_geometry") is not True
                              or selection.get("geometry_source") != "source_coverage_and_guarded_ownership_interface_reconstruction"):
            item_failures.append("source_edge_geometry_selection_scope_invalid")
        elif not source_circle and not source_edge and selection.get("colour_used_for_geometry") is not False:
            item_failures.append("geometry_selection_scope_invalid")

        # The source-space paint proof and the curve-economy certificate are
        # intentionally separate.  Older fixtures may carry only the former;
        # in that case the gradient remains paint-authoritative but is not
        # exempted from generic curve heuristics.  When the stronger optimiser
        # evidence is present it must be complete and tied to the final SVG.
        economy_available = not paint_only_valid and any(
            key in geometry for key in (
                "solver", "selection_evidence", "lexicographic_objective",
                "final_svg_consistency"))
        economy_failures: list[str] = []
        objective_values = geometry.get("lexicographic_objective")
        objective_set = (set(objective_values)
                         if isinstance(objective_values, list) else set())
        anchor_count = _integer(geometry.get("anchor_count"), -1)
        designer_count = _integer(geometry.get("designer_anchor_count"), -1)
        segment_count = _integer(geometry.get("segment_count"), -1)
        if economy_available:
            if not (source_circle_valid or source_spans_valid or source_edge_valid) and geometry.get("solver") != (
                    "geometry_error_optimizer.optimize_compound_contours"):
                economy_failures.append("geometry_optimizer_identity_missing")
            if not (source_circle_valid or source_spans_valid or source_edge_valid) and not _ECONOMY_OBJECTIVE_REQUIRED.issubset(objective_set):
                economy_failures.append("geometry_economy_objective_incomplete")
            if anchor_count < 1 or designer_count < 1 or segment_count < 1:
                economy_failures.append("geometry_economy_counts_invalid")
            if selection_evidence.get("identity_rollback_selected") is not False:
                economy_failures.append("geometry_identity_rollback_selected")
            if not selection_evidence.get("selected_candidate_id"):
                economy_failures.append("geometry_selected_candidate_missing")
            if final_consistency.get("status") != "verified_unchanged":
                economy_failures.append("final_svg_geometry_not_verified")
            if not final_consistency.get("final_drawable_id") \
                    or not final_consistency.get("gradient_object_id"):
                economy_failures.append("final_svg_identity_missing")
            if final_consistency.get("final_element") not in DRAWABLES:
                economy_failures.append("final_svg_element_invalid")
            from native_geometry_contract import (
                native_geometry_matches, whole_object_native_primitive)

            final_count = _integer(final_consistency.get("final_anchor_count"), -1)
            whole_native = whole_object_native_primitive(geometry)
            verified_native = bool(
                final_consistency.get("final_element") in {"circle", "ellipse", "rect"}
                and final_consistency.get("anchor_count_semantics") == (
                    "native_svg_element_vs_designer_handles")
                and whole_native is not None
                and final_consistency.get("final_element") == whole_native["element"]
                and native_geometry_matches(
                    whole_native, final_consistency.get("native_geometry"))
                and final_count == 1)
            if (_integer(final_consistency.get("reconstruction_anchor_count"), -1)
                    != anchor_count or (final_count != anchor_count and not verified_native)):
                economy_failures.append("final_svg_anchor_count_mismatch")
            if (final_consistency.get("final_element") in {"circle", "ellipse", "rect"}
                    and not verified_native):
                economy_failures.append("final_svg_native_whole_object_proof_invalid")
            if final_consistency.get("curve_refit_applied") is not False:
                economy_failures.append("gradient_curve_refit_was_applied")
            if not str(geometry.get("evidence_scope") or "").startswith(
                    "gradient_reconstruction_against_original_source"):
                economy_failures.append("source_ownership_geometry_scope_invalid")
        if item_failures:
            failures.append(f"gradient_detail_{index + 1}_invalid")
        object_evidence.append({
            "candidate_id": candidate_id,
            "gradient_id": raw_detail.get("id"),
            "stop_count": len(stops) if isinstance(stops, list) else None,
            "passed": not item_failures,
            "partial_paint_requires_manual_review": is_partial_paint,
            "failure_reasons": item_failures,
            "economy_certificate_available": economy_available,
            "economy_certificate_passed": bool(
                economy_available and not economy_failures),
            "economy_failure_reasons": economy_failures,
            "final_drawable_id": final_consistency.get("final_drawable_id"),
            "final_drawable_ids": ([row['id'] for row in geometry['source_paint_only']['drawable_records']]
                                   if paint_only_valid else []),
            "gradient_object_id": final_consistency.get("gradient_object_id"),
            "final_element": final_consistency.get("final_element"),
            "anchor_count": anchor_count if anchor_count >= 0 else None,
            "designer_anchor_count": (
                designer_count if designer_count >= 0 else None),
            "segment_count": segment_count if segment_count >= 0 else None,
        })

    base["objects"] = object_evidence
    base["failure_reasons"] = failures
    base["authoritative"] = bool(base["available"] and not failures and not partial_details)
    return base


def _visual_status(root: ET.Element,
                   report: dict[str, Any] | None) -> dict[str, Any]:
    raw = report or _embedded_json(root, "ai-vector-cleanroom-metadata") or {}
    for key in ("quality", "acceptance", "result"):
        if isinstance(raw.get(key), Mapping):
            raw = dict(raw[key])
            break
    status = raw.get("visual_acceptance_status")
    visual_gate = raw.get("visual_gate")
    if status is None and isinstance(visual_gate, Mapping):
        status = visual_gate.get("status")
    if status is None:
        status = raw.get("acceptance_status", "not_reported")
    status = str(status)
    metrics = visual_gate.get("metrics", {}) if isinstance(visual_gate, Mapping) else {}
    return {
        "status": status,
        "accepted": status.lower() in {"accepted", "passed", "pass"},
        "metrics": dict(metrics) if isinstance(metrics, Mapping) else {},
        "scope": "raster_render_similarity_not_designer_readiness",
    }


def _gradient_object_gate(root: ET.Element,
                          parents: Mapping[ET.Element, ET.Element],
                          records: list[dict[str, Any]], diagonal: float,
                          viewbox_area: float,
                          proposal_report: dict[str, Any] | None,
                          proposal_source: str,
                          thresholds: Mapping[str, float]) -> dict[str, Any]:
    gradients = _gradient_inventory(root, parents)
    solid_records = [record for record in records if record["fill_kind"] == "solid"]
    chromatic = [record for record in solid_records if record.get("chromatic")]
    gradient_records = [record for record in records if record["fill_kind"] == "gradient"]
    clusters = _solid_fragment_clusters(records, diagonal, viewbox_area, thresholds)
    proposals = _proposal_groups(proposal_report)
    proposal_candidates = proposals["band_fragmentation_candidates"]
    source_fields = _source_space_gradient_field_evidence(
        proposal_report, len(gradient_records))
    authoritative_fields = bool(source_fields["authoritative"])
    distinct_chromatic = sorted({record["fill"] for record in chromatic})
    gradient_ratio = _ratio(len(gradient_records), len(gradient_records) + len(chromatic))
    clustered_ids = {identifier for cluster in clusters
                     for identifier in cluster["member_ids"]}
    band_ratio = _ratio(len(clustered_ids), len(gradient_records) + len(chromatic))

    used_bad_stop_resources = [
        item for item in gradients["resources"]
        if item["usage_count"] > 0
        and not 2 <= item["resolved_stop_count"] <= 5
    ]
    fill_groups: Counter[str] = Counter()
    for record in gradient_records:
        references = record.get("gradient_references") or []
        fallback = references[0] if references else record["id"]
        key = record.get("gradient_object_id") or f"resource:{fallback}"
        fill_groups[str(key)] += 1
    excessive_fill_groups = sorted(
        ({"id": key, "drawable_count": count}
         for key, count in fill_groups.items()
         if count > thresholds["gradient_fill_object_failure"]),
        key=lambda item: (-item["drawable_count"], item["id"]),
    )
    review_fill_groups = sorted(
        ({"id": key, "drawable_count": count}
         for key, count in fill_groups.items()
         if thresholds["gradient_fill_object_warning"] < count
         <= thresholds["gradient_fill_object_failure"]),
        key=lambda item: (-item["drawable_count"], item["id"]),
    )
    evidence_records = [(record["id"], record["geometry_evidence"])
                        for record in gradient_records]
    missing_evidence = [identifier for identifier, evidence in evidence_records
                        if evidence["missing_fields"]]
    invalid_evidence = [identifier for identifier, evidence in evidence_records
                        if evidence["invalid_fields"]]
    p95_violations = [identifier for identifier, evidence in evidence_records
                      if evidence["p95_budget_violation"]]
    max_violations = [identifier for identifier, evidence in evidence_records
                      if evidence["max_tail_violation"]]
    normalized_circles = [identifier for identifier, evidence in evidence_records
                          if evidence["native_circle_claim_normalized"]]

    failures: list[str] = []
    warnings: list[str] = []
    informational: list[str] = []
    if source_fields["available"] and not authoritative_fields and (
            source_fields['failure_reasons'] or not source_fields['partial_paint_fields']):
        failures.append("source_space_gradient_field_evidence_invalid")
    if source_fields['partial_paint_fields']:
        warnings.append('partial_source_paint_requires_manual_review')
    if gradients["invalid_references"]:
        failures.append("invalid_gradient_resource_references")
    if used_bad_stop_resources:
        failures.append("used_gradient_stop_count_outside_2_to_5")
    if excessive_fill_groups:
        failures.append("gradient_fill_split_into_excessive_objects")
    if p95_violations:
        failures.append("gradient_object_p95_exceeds_error_budget")
    if max_violations:
        failures.append("gradient_object_max_error_exceeds_tail_budget")
    severe_clusters = [cluster for cluster in clusters if cluster["severity"] == "failure"]
    if severe_clusters and not authoritative_fields:
        failures.append("spatial_colour_band_fragmentation")
    elif severe_clusters:
        informational.append(
            "global_colour_clusters_superseded_by_source_space_field_evidence")
    severe_proposals = [item for item in proposal_candidates
                        if item["member_count"] >= thresholds["solid_fragment_failure"]
                        and item["paint_count"] >= thresholds["solid_paint_failure"]]
    if severe_proposals:
        failures.append("proposal_metadata_confirms_multiband_fragmentation")
    low_global_gradient_usage = bool(
        len(chromatic) >= 20 and len(distinct_chromatic) >= 6
        and (gradient_ratio or 0.0) < thresholds["gradient_usage_ratio_warning"])
    if low_global_gradient_usage and not authoritative_fields:
        failures.append("low_gradient_usage_among_many_chromatic_fragments")
    elif low_global_gradient_usage:
        informational.append(
            "global_gradient_usage_superseded_by_source_space_field_evidence")
    if clusters and not severe_clusters and not authoritative_fields:
        warnings.append("moderate_spatial_colour_band_fragmentation")
    elif clusters and authoritative_fields:
        informational.append(
            "spatial_colour_family_candidates_outside_authoritative_field_scope")
    if any(item["unresolved"] for item in proposal_candidates):
        warnings.append("unresolved_object_group_proposals")
    if review_fill_groups:
        warnings.append("gradient_fill_uses_many_editable_objects")
    if missing_evidence:
        warnings.append("gradient_object_error_evidence_missing")
    if invalid_evidence:
        warnings.append("gradient_object_error_evidence_invalid")
    if normalized_circles:
        warnings.append("native_circle_designer_anchor_claim_normalized_to_four")
    if gradients["resource_count"] and (
            gradients["valid_used_resource_count"] < gradients["used_resource_count"]):
        warnings.append("not_all_used_gradients_are_independently_editable")

    status = "failed" if failures else ("manual_review" if warnings else "passed")
    applicable = bool(gradients["resource_count"] or clusters or proposal_candidates)
    return {
        "id": "gradient_object_gate",
        "status": status,
        "passed": status == "passed",
        "applicable": applicable,
        "failure_reasons": failures,
        "warning_reasons": warnings,
        "informational_reasons": sorted(set(informational)),
        "metrics": {
            "drawable_count": len(records),
            "solid_fill_drawable_count": len(solid_records),
            "chromatic_solid_fill_drawable_count": len(chromatic),
            "distinct_chromatic_solid_paint_count": len(distinct_chromatic),
            "gradient_fill_drawable_count": len(gradient_records),
            "gradient_usage_among_chromatic_fills": gradient_ratio,
            "suspected_band_cluster_count": len(clusters),
            "severe_band_cluster_count": len(severe_clusters),
            "suspected_band_fragment_drawable_count": len(clustered_ids),
            "band_fragmentation_ratio": band_ratio,
            "proposal_band_candidate_count": len(proposal_candidates),
            "severe_proposal_band_candidate_count": len(severe_proposals),
            "gradient_fill_object_group_count": len(fill_groups),
            "max_drawables_per_gradient_fill": max(fill_groups.values(), default=0),
            "gradient_evidence_complete_count": sum(
                evidence["complete"] for _, evidence in evidence_records),
            "gradient_evidence_missing_count": len(missing_evidence),
            "gradient_evidence_invalid_count": len(invalid_evidence),
            "gradient_p95_budget_violation_count": len(p95_violations),
            "gradient_max_tail_violation_count": len(max_violations),
        },
        "gradient_resources": gradients,
        "gradient_fill_object_groups": sorted(
            ({"id": key, "drawable_count": count}
             for key, count in fill_groups.items()),
            key=lambda item: (-item["drawable_count"], item["id"]),
        )[:30],
        "gradient_geometry_evidence": {
            "contract": {
                "stops": "2 <= native_gradient_stops <= 5",
                "p95": "p95_error_percent <= error_budget_percent",
                "tail": "max_error_percent <= 3 * error_budget_percent",
            },
            "missing_object_ids": missing_evidence[:30],
            "invalid_object_ids": invalid_evidence[:30],
            "p95_violation_object_ids": p95_violations[:30],
            "max_violation_object_ids": max_violations[:30],
            "native_circle_normalized_object_ids": normalized_circles[:30],
        },
        "solid_fragment_clusters": clusters[:30],
        "proposal_coverage": {"source": proposal_source, **proposals},
        "source_space_field_evidence": source_fields,
        "scope_note": (
            "A pass requires a native linear/radial gradient with 2-5 stops, a small "
            "editable object count, and no source-space evidence of colour-band "
            "fragmentation. Global colour-family clustering is non-semantic candidate "
            "evidence only; complete held-out ownership, hard-edge, topology, geometry "
            "and independent-revalidation evidence may supersede it. Missing evidence "
            "is never fabricated from raster similarity."
        ),
    }


def _near_circle_candidate(path: dict[str, Any], diagonal: float) -> bool:
    box = path.get("bbox")
    if (not box or path.get("loop_count", 1) != 1
            or not path.get("closed") or path["node_count"] < 4):
        return False
    width, height = box[2] - box[0], box[3] - box[1]
    if min(width, height) <= diagonal * 0.01 or not 0.86 <= width / height <= 1.14:
        return False
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    points = path.get("endpoints", [])
    radii = [_distance(point, center) for point in points]
    if len(radii) < 4 or sum(radii) <= 0:
        return False
    mean = sum(radii) / len(radii)
    variance = sum((radius - mean) ** 2 for radius in radii) / len(radii)
    radial_cv = math.sqrt(variance) / mean
    radial_max = max(abs(radius - mean) for radius in radii) / mean
    # Endpoint polygon circularity rejects leaf/teardrop silhouettes that have
    # a square-ish bbox and modest radial variance but are not circles.  This is
    # only a primitive hint; the actual conversion still needs the independent
    # normalised geometry-error proof.
    polygon_area = abs(sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))) * 0.5)
    perimeter = float(path.get("approximated_length") or 0.0)
    circularity = (4.0 * math.pi * polygon_area / (perimeter * perimeter)
                   if perimeter > 1.0e-9 else 0.0)
    return (radial_cv <= 0.10 and radial_max <= 0.22
            and 0.70 <= circularity <= 1.30)


def _numeric_match(left: Any, right: Any, tolerance: float = 1.0e-6) -> bool:
    first = _number(left)
    second = _number(right)
    if first is None or second is None:
        return False
    return abs(first - second) <= max(
        tolerance, tolerance * max(abs(first), abs(second), 1.0))


def _is_finite_zero(value: Any, tolerance: float = 1.0e-12) -> bool:
    """Accept numeric zero without relying on `_number`, which treats 0 as empty."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(parsed) and abs(parsed) <= tolerance


def _final_element_geometry_identity(element: ET.Element) -> dict[str, Any]:
    """Rebuild the exact final-SVG geometry identity used by curve evidence."""
    kind = _local(element.tag)
    identifier = str(element.get("id") or "")
    result: dict[str, Any] = {
        "final_element": kind,
        "final_drawable_id": identifier,
    }
    if kind == "path":
        path_data = element.get("d") or ""
        result.update({
            "path_data_sha256": hashlib.sha256(
                path_data.encode("utf-8")).hexdigest(),
            "path_data_digest_scope": "utf8_svg_path_d_attribute",
        })
        return result

    parameter_names = {
        "circle": ("cx", "cy", "r"),
        "ellipse": ("cx", "cy", "rx", "ry", "transform"),
    }.get(kind, ())
    parameters = {
        name: element.get(name)
        for name in parameter_names if element.get(name) is not None
    }
    canonical = json.dumps({
        "element": kind,
        "id": identifier,
        "parameters": parameters,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result.update({
        "native_parameters": parameters,
        "geometry_sha256": hashlib.sha256(
            canonical.encode("utf-8")).hexdigest(),
        "geometry_digest_scope": (
            "canonical_json_element_id_and_svg_geometry_attributes"),
        "unexpected_native_transform": bool(
            kind == "circle" and element.get("transform") is not None),
    })
    return result


def _optimizer_detail_contract_failures(detail: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    objective = detail.get("lexicographic_objective")
    objective_set = set(objective) if isinstance(objective, list) else set()
    if not _ECONOMY_OBJECTIVE_REQUIRED.issubset(objective_set):
        failures.append("lexicographic_objective_incomplete")
    if not detail.get("selected_candidate_id"):
        failures.append("selected_candidate_id_missing")
    budget = _number(detail.get("error_budget_percent"))
    p95 = _number(detail.get("actual_p95_error_percent"))
    maximum = _number(detail.get("actual_max_error_percent"))
    over_share = _number(detail.get("over_budget_share"))
    salient = _number(detail.get("salient_corner_max_percent"))
    if budget is None or budget <= 0:
        failures.append("error_budget_invalid")
    else:
        if p95 is None or p95 < 0 or p95 > budget + 1.0e-12:
            failures.append("p95_error_contract_failed")
        if maximum is None or maximum < 0 or maximum > 3.0 * budget + 1.0e-12:
            failures.append("maximum_error_contract_failed")
        if over_share is None or not 0.0 <= over_share <= 0.05 + 1.0e-12:
            failures.append("over_budget_share_contract_failed")
        if salient is None or salient < 0 or salient > 2.0 * budget + 1.0e-12:
            failures.append("salient_corner_contract_failed")
    return failures


def _gradient_economy_certificates(
        gradient_gate: Mapping[str, Any],
        path_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Tie source-space gradient optimiser evidence to exact final paths."""
    result = {
        "available": False,
        "authoritative": False,
        "certified_path_ids": [],
        "certified_path_count": 0,
        "failure_reasons": [],
        "invalid_object_ids": [],
        "scope": "source_ownership_geometry_frontier_tied_to_final_svg_path",
    }
    source = gradient_gate.get("source_space_field_evidence")
    if not isinstance(source, Mapping) or not source.get("authoritative"):
        return result
    objects = source.get("objects")
    if not isinstance(objects, list):
        return result
    economy_objects = [item for item in objects if isinstance(item, Mapping)
                       and item.get("economy_certificate_available")]
    if not economy_objects:
        return result
    result["available"] = True
    by_id = {str(path["id"]): path for path in path_records}
    certified: list[str] = []
    invalid: list[str] = []
    failure_reasons: list[str] = []
    for index, item in enumerate(economy_objects):
        identifier = str(item.get("final_drawable_id") or
                         f"gradient-object-{index + 1}")
        item_failures: list[str] = []
        if item.get("economy_certificate_passed") is not True:
            item_failures.append("optimizer_economy_evidence_invalid")
        if item.get("final_element") == "path":
            path = by_id.get(identifier)
            if path is None:
                item_failures.append("final_gradient_path_missing")
            else:
                if path.get("gradient_object_id") != item.get(
                        "gradient_object_id"):
                    item_failures.append("gradient_object_identity_mismatch")
                references = set(_URL_REF.findall(str(path.get("fill") or "")))
                if str(item.get("gradient_id")) not in references:
                    item_failures.append("gradient_resource_identity_mismatch")
                if path.get("node_count") != item.get("anchor_count"):
                    item_failures.append("gradient_anchor_count_mismatch")
                evidence = path.get("geometry_evidence") or {}
                if (not evidence.get("complete")
                        or evidence.get("p95_budget_violation")
                        or evidence.get("max_tail_violation")):
                    item_failures.append("gradient_svg_error_evidence_invalid")
                if evidence.get("designer_anchors") != item.get(
                        "designer_anchor_count"):
                    item_failures.append("gradient_designer_anchor_mismatch")
        if item_failures:
            invalid.append(identifier)
            failure_reasons.extend(
                f"{identifier}:{reason}" for reason in item_failures)
        elif item.get("final_element") == "path":
            certified.append(identifier)
    if invalid:
        failure_reasons.insert(0, "gradient_economy_certificate_invalid")
    result.update({
        "authoritative": not failure_reasons,
        "certified_path_ids": sorted(certified) if not failure_reasons else [],
        "certified_path_count": len(certified) if not failure_reasons else 0,
        "failure_reasons": failure_reasons[:60],
        "invalid_object_ids": invalid[:30],
    })
    return result


def _curve_optimizer_economy_certificates(
        report: Mapping[str, Any] | None,
        path_records: list[dict[str, Any]],
        element_geometry: Mapping[str, Mapping[str, Any]],
        element_id_counts: Mapping[str, int]) -> dict[str, Any]:
    """Authenticate optimiser economy results against the final SVG paths."""
    result = {
        "available": False,
        "authoritative": False,
        "transaction_status": None,
        "proposal_schema": None,
        "proposal_status": None,
        "certified_refit_path_ids": [],
        "certified_retained_identity_path_ids": [],
        "certified_path_ids": [],
        "certified_path_count": 0,
        "uncertified_evaluation_ids": [],
        "failure_reasons": [],
        "invalid_detail_ids": [],
        "scope": (
            "transaction_backed_geometry_only_optimizer_economy_certificate"),
    }
    if not isinstance(report, Mapping):
        return result
    enhancements = report.get("editability_enhancements")
    if not isinstance(enhancements, Mapping):
        return result
    stages = enhancements.get("stages")
    curve = stages.get("curve_refit") if isinstance(stages, Mapping) else None
    if not isinstance(curve, Mapping):
        return result
    result["available"] = True
    failures: list[str] = []
    if enhancements.get("schema") != (
            "ai-vector-cleanroom.editability-enhancements/v1"):
        failures.append("editability_enhancement_schema_invalid")
    if curve.get("schema") != "ai-vector-cleanroom.curve-refit-transaction/v1":
        failures.append("curve_refit_transaction_schema_invalid")
    status = curve.get("status")
    result["transaction_status"] = status
    proposal = curve.get("proposal")
    if not isinstance(proposal, Mapping):
        failures.append("curve_refit_proposal_missing")
        proposal = {}
    result["proposal_schema"] = proposal.get("schema")
    result["proposal_status"] = proposal.get("status")
    if proposal.get("schema") != "ai-vector-cleanroom.curve-refit-proposal/v3":
        failures.append("curve_refit_proposal_schema_invalid")
    if proposal.get("optimization_basis") != "geometry_only" \
            or proposal.get("uses_colour_or_pixel_similarity") is not False:
        failures.append("curve_refit_selection_scope_invalid")

    stable_ids = proposal.get("stable_id_normalization")
    identity_transaction = curve.get("identity_normalization")
    identity_view = (identity_transaction
                     if isinstance(identity_transaction, Mapping) else {})
    stable_records = stable_ids.get("records") if isinstance(
        stable_ids, Mapping) else None
    identity_records = identity_transaction.get("records") if isinstance(
        identity_transaction, Mapping) else None
    stable_id_contract_valid = bool(
        isinstance(stable_ids, Mapping)
        and stable_ids.get("schema") ==
        "ai-vector-cleanroom.curve-refit-stable-id-normalization/v1"
        and stable_ids.get("source_svg_sha256") ==
        curve.get("before_svg_sha256")
        and stable_ids.get("source_svg_digest_scope") ==
        "exact_source_svg_bytes"
        and stable_ids.get("all_optimizer_evaluations_authenticated") is True
        and stable_ids.get("assigned_ids_unique") is True
        and stable_ids.get("candidate_all_svg_ids_unique") is True
        and isinstance(stable_records, list)
        and stable_ids.get("record_count") == len(stable_records)
        and stable_ids.get("optimizer_evaluated_path_count") ==
        proposal.get("optimizer_evaluated_path_count")
        and isinstance(identity_transaction, Mapping)
        and identity_transaction.get("schema") ==
        "ai-vector-cleanroom.curve-refit-stable-id-transaction/v1"
        and identity_transaction.get("source_svg_sha256") ==
        curve.get("before_svg_sha256")
        and identity_transaction.get("record_count") == len(stable_records)
        and identity_transaction.get("optimizer_evaluated_path_count") ==
        proposal.get("optimizer_evaluated_path_count")
        and identity_transaction.get("assigned_ids_unique") is True
        and identity_transaction.get("source_existing_ids_unique") is True
        and identity_transaction.get("normalized_all_svg_ids_unique") is True
        and isinstance(identity_records, list)
        and identity_records == stable_records)
    if stable_id_contract_valid:
        record_ids = []
        record_ordinals = []
        assignment_count = 0
        for item in stable_records:
            if not isinstance(item, Mapping):
                stable_id_contract_valid = False
                break
            identifier = item.get("assigned_id")
            ordinal = item.get("global_path_ordinal_1_based")
            source_digest = item.get("source_path_data_sha256")
            state = item.get("original_id_state")
            assigned = item.get("assignment_applied")
            record_valid = bool(
                item.get("source_svg_sha256") == curve.get(
                    "before_svg_sha256")
                and item.get("source_element") == "path"
                and item.get("source_path_data_digest_scope") ==
                "utf8_svg_path_d_attribute"
                and isinstance(source_digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", source_digest)
                and isinstance(identifier, str) and identifier
                and isinstance(ordinal, int) and not isinstance(ordinal, bool)
                and ordinal >= 1
                and ((state == "existing_preserved"
                      and item.get("original_id") == identifier
                      and assigned is False)
                     or (state == "missing_assigned"
                         and item.get("original_id") is None
                         and assigned is True)))
            if not record_valid:
                stable_id_contract_valid = False
                break
            record_ids.append(identifier)
            record_ordinals.append(ordinal)
            assignment_count += int(assigned is True)
        stable_id_contract_valid = bool(
            stable_id_contract_valid
            and len(record_ids) == len(set(record_ids))
            and len(record_ordinals) == len(set(record_ordinals))
            and stable_ids.get("assigned_id_count") == assignment_count
            and identity_transaction.get("assigned_id_count") ==
            assignment_count
            and stable_ids.get("existing_id_preserved_count") ==
            len(stable_records) - assignment_count
            and identity_transaction.get("existing_id_preserved_count") ==
            len(stable_records) - assignment_count)
    else:
        record_ids = []
        assignment_count = -1
    candidate_identity_guard = (
        identity_transaction.get("candidate_identity_guard")
        if isinstance(identity_transaction, Mapping) else None)
    if (not isinstance(candidate_identity_guard, Mapping)
            or candidate_identity_guard.get("status") != "verified"
            or candidate_identity_guard.get("candidate_all_svg_ids_unique")
            is not True
            or candidate_identity_guard.get("evaluated_path_count") !=
            len(stable_records or [])):
        stable_id_contract_valid = False
    if not stable_id_contract_valid:
        failures.append("curve_refit_stable_id_contract_invalid")

    gradient_guard = curve.get("gradient_geometry_guard")
    if (not isinstance(gradient_guard, Mapping)
            or gradient_guard.get("status") != "verified_unchanged"
            or gradient_guard.get("ownership_mask_revalidation_performed")
            is not False):
        failures.append("curve_refit_gradient_guard_invalid")
    if status == "committed":
        commit_scope = curve.get("commit_scope")
        identity_only = bool(
            proposal.get("status") == "no_change"
            and commit_scope == "stable_id_normalization_only"
            and assignment_count > 0
            and proposal.get("path_count_refit") == 0
            and proposal.get("details") == [])
        geometry_commit = bool(
            proposal.get("status") == "proposed"
            and commit_scope in {
                "geometry_refit",
                "geometry_refit_with_stable_id_normalization",
            }
            and ((commit_scope == "geometry_refit"
                  and assignment_count == 0)
                 or (commit_scope ==
                     "geometry_refit_with_stable_id_normalization"
                     and assignment_count > 0)))
        if not (identity_only or geometry_commit):
            failures.append("committed_curve_refit_proposal_status_invalid")
        baseline_validation = (
            identity_view.get("baseline_validation"))
        if assignment_count > 0 and (
                not isinstance(baseline_validation, Mapping)
                or baseline_validation.get("status") != "verified"
                or baseline_validation.get("accepted") is not True
                or not isinstance(baseline_validation.get(
                    "renderer_topology_guard"), Mapping)
                or baseline_validation["renderer_topology_guard"].get(
                    "accepted") is not True
                or not isinstance(baseline_validation.get("source_guard"),
                                  Mapping)
                or baseline_validation["source_guard"].get("accepted")
                is not True
                or not isinstance(baseline_validation.get(
                    "gradient_geometry_guard"), Mapping)
                or baseline_validation["gradient_geometry_guard"].get(
                    "accepted") is not True):
            failures.append("curve_refit_identity_baseline_guard_invalid")
        render_guard = curve.get("render_guard")
        source_guard = curve.get("source_guard")
        topology_guard = curve.get("renderer_topology_guard")
        if (not isinstance(render_guard, Mapping)
                or render_guard.get("external_render_check") != "completed"):
            failures.append("curve_refit_render_guard_invalid")
        if not isinstance(source_guard, Mapping) \
                or source_guard.get("accepted") is not True:
            failures.append("curve_refit_source_guard_invalid")
        required_topology_metrics = (
            "ink_recall_percent", "ink_precision_percent", "ink_f1_percent")
        topology_comparisons = (
            topology_guard.get("comparisons")
            if isinstance(topology_guard, Mapping) else None)
        topology_evidence_valid = bool(
            isinstance(render_guard, Mapping)
            and isinstance(topology_guard, Mapping)
            and topology_guard.get("accepted") is True
            and topology_guard.get("policy") == (
                "bidirectional_ink_topology_guard")
            and topology_guard.get("colour_similarity_excluded") is True
            and isinstance(topology_comparisons, Mapping)
        )
        if topology_evidence_valid:
            for key in required_topology_metrics:
                comparison = topology_comparisons.get(key)
                render_value = render_guard.get(key)
                compared_value = (
                    comparison.get("value")
                    if isinstance(comparison, Mapping) else None)
                minimum = (
                    comparison.get("minimum")
                    if isinstance(comparison, Mapping) else None)
                try:
                    render_number = float(render_value)
                    compared_number = float(compared_value)
                    minimum_number = float(minimum)
                except (TypeError, ValueError):
                    topology_evidence_valid = False
                    break
                if (not math.isfinite(render_number)
                        or not math.isfinite(compared_number)
                        or not math.isfinite(minimum_number)
                        or render_number < 99.0
                        or compared_number < 99.0
                        or not _numeric_match(render_number, compared_number)
                        or not _numeric_match(minimum_number, 99.0)
                        or comparison.get("accepted") is not True):
                    topology_evidence_valid = False
                    break
        if not topology_evidence_valid:
            failures.append("curve_refit_topology_guard_invalid")
        for key in ("before_svg_sha256", "after_svg_sha256"):
            value = curve.get(key)
            if not isinstance(value, str) or len(value) != 64:
                failures.append(f"curve_refit_{key}_invalid")
        precommit_records = curve.get("precommit_final_digest_records")
        postcommit_records = curve.get("postcommit_final_digest_records")
        proposal_precommit = proposal.get(
            "transaction_precommit_final_digest_records")
        proposal_postcommit = proposal.get(
            "transaction_postcommit_final_digest_records")
        transaction_digest_valid = bool(
            isinstance(precommit_records, list)
            and isinstance(postcommit_records, list)
            and precommit_records == postcommit_records
            and proposal_precommit == precommit_records
            and proposal_postcommit == postcommit_records
            and len(precommit_records) == len(stable_records or [])
            and len({item.get("id") for item in precommit_records
                     if isinstance(item, Mapping)}) == len(record_ids)
            and {item.get("id") for item in precommit_records
                 if isinstance(item, Mapping)} == set(record_ids)
            and identity_view.get("status") == "committed"
            and identity_view.get("commit_scope") == commit_scope
            and identity_view.get("postcommit_svg_sha256") ==
            curve.get("after_svg_sha256")
            and identity_view.get(
                "precommit_final_digest_records") == precommit_records
            and identity_view.get(
                "postcommit_final_digest_records") == postcommit_records)
        if not transaction_digest_valid:
            failures.append("curve_refit_transaction_final_digest_invalid")
    elif status == "not_needed":
        if (proposal.get("status") != "no_change"
                or curve.get("reason") != "no_safe_reductions"
                or curve.get("commit_scope") != "none"
                or assignment_count != 0
                or identity_view.get("status") != "authenticated"):
            failures.append("no_change_curve_refit_transaction_invalid")
    elif status != "rolled_back":
        failures.append("curve_refit_transaction_not_committed_or_not_needed")

    details = proposal.get("details")
    retained = proposal.get("evaluated_but_retained")
    uncertified = proposal.get("uncertified_evaluations")
    integrity = proposal.get("evaluation_evidence_integrity")
    if not isinstance(details, list):
        failures.append("curve_refit_details_missing")
        details = []
    if not isinstance(retained, list):
        failures.append("retained_identity_evidence_missing")
        retained = []
    if not isinstance(uncertified, list):
        failures.append("uncertified_evaluation_evidence_missing")
        uncertified = []
    if not isinstance(integrity, Mapping):
        failures.append("curve_refit_evidence_integrity_missing")
        integrity = {}

    frontier_refinement = proposal.get("transaction_frontier_refinement")
    if frontier_refinement is not None:
        frontier_valid = isinstance(frontier_refinement, Mapping)
        target_id = (
            frontier_refinement.get("target_id")
            if frontier_valid else None)
        target_details = [
            item for item in details
            if isinstance(item, Mapping) and item.get("id") == target_id
        ]
        target_detail = target_details[0] if len(target_details) == 1 else {}
        detail_frontier = (
            target_detail.get("refinement_frontier")
            if isinstance(target_detail, Mapping) else None)
        expected_selection = (
            "designer_anchors_then_anchors_then_segments_then_geometry_then_id")
        frontier_valid = bool(
            frontier_valid
            and frontier_refinement.get("schema") ==
            "ai-vector-cleanroom.curve-refit-path-frontier/v1"
            and isinstance(target_id, str) and target_id
            and len(target_details) == 1
            and frontier_refinement.get("selected_candidate_id") ==
            target_detail.get("selected_candidate_id")
            and frontier_refinement.get("selection_basis") ==
            expected_selection
            and frontier_refinement.get("optimization_basis") ==
            "geometry_only"
            and frontier_refinement.get("uses_colour_or_pixel_similarity")
            is False
            and isinstance(detail_frontier, Mapping)
            and detail_frontier.get("schema") ==
            "ai-vector-cleanroom.curve-refit-path-frontier/v1"
            and detail_frontier.get("selection_basis") == expected_selection)
        if not frontier_valid:
            failures.append("curve_refit_frontier_refinement_invalid")

    optimizer_evaluated = _integer(
        proposal.get("optimizer_evaluated_path_count"), -1)
    accounted = len(details) + len(retained) + len(uncertified)
    eligible = _integer(proposal.get("eligible_path_count"), -1)
    if (_integer(proposal.get("path_count_refit"), -1) != len(details)
            or _integer(proposal.get("retained_identity_path_count"), -1)
            != len(retained)
            or optimizer_evaluated != accounted
            or eligible < optimizer_evaluated):
        failures.append("curve_refit_evaluated_count_mismatch")
    expected_integrity = {
        "schema": "ai-vector-cleanroom.curve-refit-evaluation-evidence/v1",
        "optimizer_evaluated_path_count": optimizer_evaluated,
        "committed_detail_count": len(details),
        "retained_identity_detail_count": len(retained),
        "uncertified_evaluation_count": len(uncertified),
        "accounted_evaluation_count": accounted,
        "all_optimizer_results_accounted": True,
        "evidence_ids_unique": True,
        "committed_and_retained_ids_disjoint": True,
    }
    if any(integrity.get(key) != value
           for key, value in expected_integrity.items()):
        failures.append("curve_refit_evidence_integrity_invalid")

    by_id = {str(path["id"]): path for path in path_records}
    committed_ids: list[str] = []
    retained_ids: list[str] = []
    invalid_ids: list[str] = []
    seen: set[str] = set()

    def record_invalid(identifier: str, reasons: list[str]) -> None:
        invalid_ids.append(identifier)
        failures.extend(f"{identifier}:{reason}" for reason in reasons)

    for index, raw in enumerate(details):
        detail = raw if isinstance(raw, Mapping) else {}
        identifier = str(detail.get("id") or f"refit-detail-{index + 1}")
        item_failures = _optimizer_detail_contract_failures(detail)
        if identifier in seen:
            item_failures.append("duplicate_optimizer_detail_id")
        seen.add(identifier)
        if detail.get("outcome") != "committed_refit" \
                or detail.get("economy_certified") is not True:
            item_failures.append("committed_economy_outcome_invalid")
        before = _integer(detail.get("anchors_before"), -1)
        after = _integer(detail.get("anchors_after"), -1)
        designer_before = _integer(detail.get("designer_anchors_before"), -1)
        designer_after = _integer(detail.get("designer_anchors_after"), -1)
        primitive = detail.get("primitive") if isinstance(
            detail.get("primitive"), Mapping) else {}
        emitted = primitive.get("emitted_element")
        if (before < 1 or after < 1 or designer_before != before
                or designer_after < 1 or designer_after > designer_before
                or (after >= before and designer_after >= designer_before
                    and emitted not in PRIMITIVES)):
            item_failures.append("committed_economy_counts_invalid")
        path = by_id.get(identifier)
        final_geometry = element_geometry.get(identifier)
        if element_id_counts.get(identifier, 0) != 1:
            item_failures.append("committed_final_drawable_id_not_unique")
        if detail.get("final_drawable_id") != identifier:
            item_failures.append("committed_final_drawable_id_mismatch")
        if emitted in {"circle", "ellipse"}:
            if (not isinstance(final_geometry, Mapping)
                    or final_geometry.get("final_element") != emitted
                    or detail.get("final_element") != emitted):
                item_failures.append("committed_native_element_mismatch")
            else:
                required_parameters = (
                    {"cx", "cy", "r"} if emitted == "circle"
                    else {"cx", "cy", "rx", "ry"})
                native_parameters = detail.get("native_parameters")
                if (not isinstance(native_parameters, Mapping)
                        or not required_parameters.issubset(
                            native_parameters.keys())
                        or dict(native_parameters) != final_geometry.get(
                            "native_parameters")):
                    item_failures.append(
                        "committed_native_parameters_mismatch")
                if final_geometry.get("unexpected_native_transform"):
                    item_failures.append(
                        "committed_circle_transform_outside_digest_scope")
                if (detail.get("geometry_digest_scope") !=
                        "canonical_json_element_id_and_svg_geometry_attributes"
                        or detail.get("geometry_sha256") !=
                        final_geometry.get("geometry_sha256")):
                    item_failures.append("committed_native_digest_mismatch")
            if path is not None:
                item_failures.append("committed_native_path_id_collision")
        elif emitted is not None:
            item_failures.append("committed_native_element_unsupported")
        elif path is None:
            item_failures.append("committed_refit_path_missing")
        else:
            if path.get("curve_refit_claim") != "geometry-budgeted":
                item_failures.append("curve_refit_claim_missing")
            if path.get("node_count") != after:
                item_failures.append("committed_anchor_count_mismatch")
            if path.get("anchors_before_claim") != before \
                    or path.get("anchors_after_claim") != after:
                item_failures.append("committed_anchor_claim_mismatch")
            if detail.get("final_element") != "path":
                item_failures.append("committed_final_element_mismatch")
            detail_digest = detail.get("path_data_sha256")
            if (detail.get("path_data_digest_scope") !=
                    "utf8_svg_path_d_attribute"
                    or not isinstance(detail_digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", detail_digest)
                    or path.get("path_data_sha256") != detail_digest):
                item_failures.append("committed_path_digest_mismatch")
            evidence = path.get("geometry_evidence") or {}
            if (not evidence.get("complete")
                    or evidence.get("p95_budget_violation")
                    or evidence.get("max_tail_violation")):
                item_failures.append("committed_svg_error_evidence_invalid")
            if evidence.get("designer_anchors") != designer_after:
                item_failures.append("committed_designer_anchor_mismatch")
            for evidence_key, detail_key in (
                    ("error_budget_percent", "error_budget_percent"),
                    ("p95_error_percent", "actual_p95_error_percent"),
                    ("max_error_percent", "actual_max_error_percent")):
                if not _numeric_match(evidence.get(evidence_key),
                                      detail.get(detail_key), 1.0e-5):
                    item_failures.append(f"committed_{evidence_key}_mismatch")
        if item_failures:
            record_invalid(identifier, item_failures)
        elif path is not None:
            committed_ids.append(identifier)

    comparison_order = [
        "designer_anchor_count", "anchor_count", "fragment_count",
        "segment_count",
    ]
    expected_reason = (
        "source_path_baseline_is_no_more_complex_than_best_error_eligible_"
        "evaluated_candidate")
    for index, raw in enumerate(retained):
        detail = raw if isinstance(raw, Mapping) else {}
        identifier = str(detail.get("id") or f"retained-detail-{index + 1}")
        item_failures = _optimizer_detail_contract_failures(detail)
        if identifier in seen:
            item_failures.append("duplicate_optimizer_detail_id")
        seen.add(identifier)
        path = by_id.get(identifier)
        before = _integer(detail.get("anchors_before"), -1)
        after = _integer(detail.get("anchors_after"), -1)
        designer_before = _integer(detail.get("designer_anchors_before"), -1)
        designer_after = _integer(detail.get("designer_anchors_after"), -1)
        baseline = detail.get("source_baseline_economy")
        selected = detail.get("optimizer_selected_economy")
        baseline_error = detail.get("source_baseline_geometry_error")
        if (detail.get("outcome") != "retained_identity_minimum"
                or detail.get("economy_certified") is not True
                or detail.get("economy_reason") != expected_reason
                or detail.get("retention_basis") != (
                    "source_svg_path_implicit_zero_error_candidate")
                or detail.get("geometry_unchanged") is not True
                or detail.get("path_data_digest_scope") != (
                    "utf8_svg_path_d_attribute")
                or detail.get("economy_comparison_order") != comparison_order):
            item_failures.append("retained_identity_contract_invalid")
        if (not isinstance(baseline_error, Mapping)
                or baseline_error.get("passed") is not True
                or any(not _is_finite_zero(baseline_error.get(key)) for key in (
                    "actual_p95_error_percent", "actual_max_error_percent",
                    "over_budget_share", "salient_corner_max_percent"))):
            item_failures.append("source_baseline_error_contract_invalid")
        if path is None:
            item_failures.append("retained_identity_path_missing")
        else:
            if (before < 1 or before != after or before != designer_before
                    or before != designer_after or path.get("node_count") != before
                    or path.get("loop_count") != _integer(
                        detail.get("loops"), -1)):
                item_failures.append("retained_identity_count_mismatch")
            if path.get("path_data_sha256") != detail.get("path_data_sha256"):
                item_failures.append("retained_identity_path_digest_mismatch")
            if path.get("curve_refit_claim") or path.get("gradient_object_id"):
                item_failures.append("retained_identity_scope_invalid")
        if not isinstance(baseline, Mapping) or not isinstance(selected, Mapping):
            item_failures.append("retained_economy_comparison_missing")
        else:
            source_tuple = tuple(_integer(baseline.get(key), -1)
                                 for key in comparison_order)
            selected_tuple = tuple(_integer(selected.get(key), -1)
                                   for key in comparison_order)
            if any(value < 1 for value in source_tuple + selected_tuple) \
                    or selected_tuple < source_tuple:
                item_failures.append("optimizer_found_more_economical_candidate")
            if source_tuple != (before, before, 1, before):
                item_failures.append("source_baseline_economy_mismatch")
            if selected_tuple != (
                    _integer(detail.get(
                        "optimizer_selected_designer_anchor_count"), -1),
                    _integer(detail.get("optimizer_selected_anchor_count"), -1),
                    1,
                    _integer(detail.get("optimizer_selected_segment_count"), -1)):
                item_failures.append("optimizer_selected_economy_mismatch")
        candidate_count = _integer(detail.get("candidate_count"), -1)
        eligible_count = _integer(detail.get("eligible_candidate_count"), -1)
        if candidate_count < 1 or eligible_count < 1 \
                or eligible_count > candidate_count \
                or _integer(detail.get("optimizer_input_anchor_count"), -1) < 1:
            item_failures.append("retained_identity_frontier_count_invalid")
        if not isinstance(detail.get("selected_source"), str) \
                or not detail.get("selected_source"):
            item_failures.append("retained_selected_source_missing")
        if item_failures:
            record_invalid(identifier, item_failures)
        else:
            retained_ids.append(identifier)

    uncertified_ids = [str(item.get("id")) for item in uncertified
                       if isinstance(item, Mapping) and item.get("id")]
    if status == "rolled_back":
        # A rejected proposal provides no economy certificate. Authenticate its
        # documented source path data/IDs against the final paths. Later paint
        # metadata changes mean this is not a proof of full-scene restoration.
        subset = curve.get("subset_fallback")
        rollback_records = {
            str(item.get("id")): item for item in uncertified
            if isinstance(item, Mapping) and item.get("id")}
        rollback_ids = proposal.get("uncertified_transaction_rollback_ids")
        baseline_validation = identity_view.get("baseline_validation")
        rejected_validation = (subset.get("full_candidate_validation")
                               if isinstance(subset, Mapping) else None)
        before_digest = curve.get("before_svg_sha256")
        rollback_valid = bool(
            stable_id_contract_valid
            and isinstance(before_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", before_digest)
            and curve.get("after_svg_sha256") == before_digest
            and curve.get("live_svg_unchanged") is True
            and curve.get("reason") == "renderer_or_source_guard_rejected"
            and curve.get("commit_scope") is None
            and curve.get("precommit_final_digest_records") is None
            and curve.get("postcommit_final_digest_records") is None
            and identity_view.get("status") == "authenticated"
            and isinstance(baseline_validation, Mapping)
            and baseline_validation.get("accepted") is True
            and ((assignment_count == 0
                  and baseline_validation.get("status") == "not_needed"
                  and baseline_validation.get("reason") ==
                      "no_missing_ids_required_assignment")
                 or (baseline_validation.get("status") == "verified"
                     and all(isinstance(baseline_validation.get(key), Mapping)
                             and baseline_validation[key].get("accepted") is True
                             for key in ("renderer_topology_guard", "source_guard",
                                         "gradient_geometry_guard"))))
            and proposal.get("status") == "no_change"
            and details == [] and retained == []
            and optimizer_evaluated > 0
            and len(uncertified_ids) == len(set(uncertified_ids))
            and set(uncertified_ids) == set(record_ids)
            and stable_ids.get("source_path_count") == len(path_records)
            and identity_view.get("source_path_count") == len(path_records)
            and isinstance(subset, Mapping)
            and subset.get("schema") ==
                "ai-vector-cleanroom.curve-refit-subset-fallback/v1"
            and subset.get("status") == "all_refits_rolled_back"
            and isinstance(rejected_validation, Mapping)
            and any(isinstance(rejected_validation.get(key), Mapping)
                    and rejected_validation[key].get("accepted") is False
                    for key in ("renderer_topology_guard", "source_guard"))
            and proposal.get("transaction_subset_fallback") == subset
            and isinstance(rollback_ids, list)
            and all(isinstance(item, str) for item in rollback_ids)
            and len(rollback_ids) == len(set(rollback_ids))
            and set(rollback_ids) == set(record_ids)
            and integrity.get("transaction_rollback_count") == len(record_ids))
        if rollback_valid:
            for record in stable_records:
                ordinal = record["global_path_ordinal_1_based"]
                item = rollback_records[record["assigned_id"]]
                if not 1 <= ordinal <= len(path_records):
                    rollback_valid = False
                    break
                actual = path_records[ordinal - 1]
                if not (
                    actual.get("svg_id") == record.get("original_id")
                    and actual.get("path_data_sha256") ==
                        record["source_path_data_sha256"]
                    and item.get("outcome") ==
                        "transaction_guard_rollback_to_source_identity"
                    and item.get("economy_certified") is False
                    and item.get("rollback_basis") ==
                        "exact_source_svg_element_restoration"
                    and item.get("stage_reason") ==
                        "renderer_or_source_guard_rejected_candidate"
                    and item.get("integrity_failures") ==
                        ["geometry_candidate_not_committed_after_transaction_guard"]
                    and item.get("anchors_before") == item.get("anchors_after")
                        == actual["node_count"]
                    and item.get("designer_anchors_before") ==
                        item.get("designer_anchors_after") == actual["node_count"]):
                    rollback_valid = False
                    break
        if not rollback_valid:
            failures.append("curve_refit_rollback_source_identity_invalid")
    if invalid_ids:
        failures.insert(0, "curve_optimizer_detail_evidence_invalid")
    certified = sorted(set(committed_ids + retained_ids))
    retained_source_paths = status == "rolled_back" and not failures
    authoritative = not failures and not retained_source_paths
    result.update({
        "available": not retained_source_paths,
        "authoritative": authoritative,
        "source_path_data_restored": retained_source_paths,
        "rollback_identity_scope": (
            "restored_source_path_data_and_original_ids_not_full_presentation"
            if retained_source_paths else None),
        "presentation_not_verified": retained_source_paths,
        "evidence_status": (
            "retained_source_paths_economy_unavailable" if retained_source_paths else
            "authoritative" if authoritative else "invalid"),
        "certified_refit_path_ids": (
            sorted(committed_ids) if authoritative else []),
        "certified_retained_identity_path_ids": (
            sorted(retained_ids) if authoritative else []),
        "certified_path_ids": certified if authoritative else [],
        "certified_path_count": len(certified) if authoritative else 0,
        "uncertified_evaluation_ids": sorted(set(uncertified_ids))[:60],
        "failure_reasons": failures[:100],
        "invalid_detail_ids": invalid_ids[:30],
    })
    return result


def _summarise_curve_scope(path_records: list[dict[str, Any]],
                           thresholds: Mapping[str, float]) -> dict[str, Any]:
    total_nodes = sum(path["node_count"] for path in path_records)
    total_commands = sum(path["command_count"] for path in path_records)
    total_cubic = sum(path["cubic_count"] for path in path_records)
    total_draw = sum(path["draw_segment_count"] for path in path_records)
    short = sum(path["short_segment_count"] for path in path_records)
    collinear = sum(path["near_collinear_cubic_count"] for path in path_records)
    length = sum(path["approximated_length"] for path in path_records)
    loops = [{"path_id": path["id"], **loop}
             for path in path_records for loop in path.get("loops", [])]
    densities = [loop["anchors_per_100_user_units"] for loop in loops
                 if loop["anchors_per_100_user_units"] is not None]
    dense_failures = [
        loop for loop in loops
        if loop["node_count"] >= 40
        and (loop["anchors_per_100_user_units"] or 0.0)
        >= thresholds["anchors_per_100_failure"]
    ]
    dense_warnings = [
        loop for loop in loops
        if loop["node_count"] >= 20
        and (loop["anchors_per_100_user_units"] or 0.0)
        >= thresholds["anchors_per_100_warning"]
        and loop not in dense_failures
    ]
    return {
        "path_count": len(path_records),
        "node_count": total_nodes,
        "command_count": total_commands,
        "cubic_segment_count": total_cubic,
        "draw_segment_count": total_draw,
        "short_segment_count": short,
        "short_segment_ratio": _ratio(short, total_draw),
        "near_collinear_cubic_count": collinear,
        "near_collinear_cubic_ratio": _ratio(collinear, total_cubic),
        "approximated_path_length": _rounded(length, 3),
        "anchors_per_100_user_units": (
            _rounded(100.0 * total_nodes / length) if length > 0 else None),
        "max_nodes_in_single_path": max(
            (path["node_count"] for path in path_records), default=0),
        "paths_over_40_nodes": sum(
            path["node_count"] > 40 for path in path_records),
        "paths_over_80_nodes": sum(
            path["node_count"] > 80 for path in path_records),
        "loop_count": len(loops),
        "max_nodes_in_single_loop": max(
            (loop["node_count"] for loop in loops), default=0),
        "loops_over_40_nodes": sum(loop["node_count"] > 40 for loop in loops),
        "loops_over_80_nodes": sum(loop["node_count"] > 80 for loop in loops),
        "max_loop_anchors_per_100_user_units": _rounded(
            max(densities, default=None)),
        "dense_loop_failure_count": len(dense_failures),
        "dense_loop_warning_count": len(dense_warnings),
        "dense_failure_loops": dense_failures,
        "dense_warning_loops": dense_warnings,
        "loop_records": loops,
    }


def _curve_economy_gate(root: ET.Element,
                        parents: Mapping[ET.Element, ET.Element],
                        path_records: list[dict[str, Any]],
                        diagonal: float,
                        thresholds: Mapping[str, float],
                        proposal_report: Mapping[str, Any] | None = None,
                        gradient_gate: Mapping[str, Any] | None = None,
                        svg_sha256: str = "",
                        element_geometry: Mapping[
                            str, Mapping[str, Any]] | None = None,
                        element_id_counts: Mapping[str, int] | None = None,
                        ) -> dict[str, Any]:
    short_limit = max(1.0, diagonal * thresholds["short_segment_diagonal_fraction"])
    collinear_limit = max(0.5, diagonal * thresholds[
        "near_collinear_diagonal_fraction"])
    gradient_certificate = _gradient_economy_certificates(
        gradient_gate or {}, path_records)
    curve_certificate = _curve_optimizer_economy_certificates(
        proposal_report, path_records, element_geometry or {},
        element_id_counts or {})
    certified_ids = set()
    if gradient_certificate.get("authoritative"):
        certified_ids.update(gradient_certificate.get("certified_path_ids", []))
    if curve_certificate.get("authoritative"):
        certified_ids.update(curve_certificate.get("certified_path_ids", []))
    heuristic_records = [path for path in path_records
                         if path["id"] not in certified_ids]
    global_metrics = _summarise_curve_scope(path_records, thresholds)
    heuristic_metrics = _summarise_curve_scope(heuristic_records, thresholds)
    total_nodes = global_metrics["node_count"]
    total_commands = global_metrics["command_count"]
    total_cubic = global_metrics["cubic_segment_count"]
    total_draw_segments = global_metrics["draw_segment_count"]
    short_segments = global_metrics["short_segment_count"]
    collinear = global_metrics["near_collinear_cubic_count"]
    short_ratio = global_metrics["short_segment_ratio"]
    collinear_ratio = global_metrics["near_collinear_cubic_ratio"]
    length = global_metrics["approximated_path_length"] or 0.0
    anchors_per_100 = global_metrics["anchors_per_100_user_units"]
    max_nodes = global_metrics["max_nodes_in_single_path"]
    paths_over_40 = global_metrics["paths_over_40_nodes"]
    paths_over_80 = global_metrics["paths_over_80_nodes"]
    loop_records = global_metrics["loop_records"]
    max_loop_nodes = global_metrics["max_nodes_in_single_loop"]
    loops_over_40 = global_metrics["loops_over_40_nodes"]
    loops_over_80 = global_metrics["loops_over_80_nodes"]
    max_loop_density = global_metrics[
        "max_loop_anchors_per_100_user_units"]
    dense_failure_loops = global_metrics["dense_failure_loops"]
    dense_warning_loops = global_metrics["dense_warning_loops"]
    invalid_paths = sum(not path["parse_valid"] for path in path_records)
    counterturn_paths = [path for path in path_records
                        if path.get("short_counterturn", {}).get("counterturn_segment_count", 0)]
    counterturn_review_paths = [path for path in counterturn_paths
                               if path["short_counterturn"]["requires_review"]]
    counterturn_count = sum(path["short_counterturn"]["counterturn_segment_count"]
                           for path in counterturn_paths)
    counterturn_incomplete_paths = [path for path in path_records
                                   if path["short_counterturn"]["status"] != "completed"]

    evidence_records: list[tuple[str, dict[str, Any]]] = [
        (path["id"], path["geometry_evidence"])
        for path in path_records if path["geometry_evidence"]["applicable"]
    ]
    native_evidence_index = 0
    for element in root.iter():
        kind = _local(element.tag)
        if kind not in PRIMITIVES:
            continue
        evidence = _geometry_evidence(element, parents)
        if not evidence["applicable"]:
            continue
        native_evidence_index += 1
        identifier = element.get("id") or f"{kind}-evidence-{native_evidence_index}"
        evidence_records.append((identifier, evidence))
    missing_evidence = [identifier for identifier, evidence in evidence_records
                        if evidence["missing_fields"]]
    invalid_evidence = [identifier for identifier, evidence in evidence_records
                        if evidence["invalid_fields"]]
    p95_violations = [identifier for identifier, evidence in evidence_records
                      if evidence["p95_budget_violation"]]
    max_violations = [identifier for identifier, evidence in evidence_records
                      if evidence["max_tail_violation"]]
    normalized_circles = [identifier for identifier, evidence in evidence_records
                          if evidence["native_circle_claim_normalized"]]

    failures: list[str] = []
    warnings: list[str] = []
    informational: list[str] = []
    if gradient_certificate.get("available") \
            and not gradient_certificate.get("authoritative"):
        failures.append("gradient_optimizer_economy_certificate_invalid")
    if curve_certificate.get("available") \
            and not curve_certificate.get("authoritative"):
        failures.append("curve_optimizer_economy_certificate_invalid")
    if curve_certificate.get("source_path_data_restored"):
        warnings.append("curve_refit_retained_source_paths_economy_unavailable")
    if certified_ids:
        informational.append(
            "optimizer_certified_paths_use_geometry_contract_not_generic_"
            "trace_heuristics")
    if invalid_paths:
        warnings.append("some_path_data_could_not_be_fully_parsed")
    if counterturn_review_paths:
        warnings.append("repeated_short_counterturns_require_review")
    if p95_violations:
        failures.append("geometry_p95_exceeds_error_budget")
    if max_violations:
        failures.append("geometry_max_error_exceeds_tail_budget")
    if missing_evidence:
        warnings.append("geometry_error_budget_evidence_missing")
    if invalid_evidence:
        warnings.append("geometry_error_budget_evidence_invalid")
    if normalized_circles:
        warnings.append("native_circle_designer_anchor_claim_normalized_to_four")
    heuristic_draw = heuristic_metrics["draw_segment_count"]
    heuristic_short_ratio = heuristic_metrics["short_segment_ratio"]
    heuristic_cubic = heuristic_metrics["cubic_segment_count"]
    heuristic_collinear_ratio = heuristic_metrics[
        "near_collinear_cubic_ratio"]
    heuristic_dense_failures = heuristic_metrics["dense_failure_loops"]
    heuristic_dense_warnings = heuristic_metrics["dense_warning_loops"]
    heuristic_max_loop_nodes = heuristic_metrics["max_nodes_in_single_loop"]
    if heuristic_draw >= 20 and (heuristic_short_ratio or 0.0) >= thresholds[
            "short_segment_ratio_failure"]:
        failures.append("excessive_short_curve_segments")
    elif heuristic_draw >= 10 and (heuristic_short_ratio or 0.0) >= thresholds[
            "short_segment_ratio_warning"]:
        warnings.append("elevated_short_curve_segment_ratio")
    if heuristic_cubic >= 20 and (heuristic_collinear_ratio or 0.0) >= thresholds[
            "near_collinear_ratio_failure"]:
        failures.append("excessive_near_collinear_cubic_segments")
    elif heuristic_cubic >= 10 and (heuristic_collinear_ratio or 0.0) >= thresholds[
            "near_collinear_ratio_warning"]:
        warnings.append("elevated_near_collinear_cubic_ratio")
    if heuristic_dense_failures:
        failures.append("anchor_density_exceeds_designer_budget")
    elif heuristic_dense_warnings:
        warnings.append("anchor_density_requires_review")
    if heuristic_max_loop_nodes >= thresholds["high_node_path_failure"]:
        failures.append("single_path_anchor_count_is_excessive")
    elif heuristic_max_loop_nodes >= thresholds["high_node_path_warning"]:
        warnings.append("high_anchor_single_path")

    native = Counter(_local(element.tag) for element in root.iter()
                     if _local(element.tag) in PRIMITIVES)
    circle_candidates = [path for path in heuristic_records
                         if _near_circle_candidate(path, diagonal)]
    # A native circle elsewhere in the document does not excuse a different
    # round object that is still represented by a noisy trace.  When a circle
    # is geometrically admissible it must not retain the raster stair-step as
    # extra anchors.  Keep a narrow warning band for uncertain 5-8-node hints;
    # fail only clearly over-traced candidates.  The conversion itself still
    # requires an independent geometry error-budget proof.
    overanchored_circle_candidates = [
        path for path in circle_candidates if path["max_nodes_in_loop"] > 8]
    mildly_overanchored_circle_candidates = [
        path for path in circle_candidates
        if 4 < path["max_nodes_in_loop"] <= 8]
    if overanchored_circle_candidates:
        failures.append("circle_like_paths_have_excessive_anchors")
    elif mildly_overanchored_circle_candidates:
        warnings.append("circle_like_paths_should_be_native_primitives")
    elif circle_candidates and not (native["circle"] or native["ellipse"]):
        warnings.append("near_circle_paths_not_recovered_as_native_primitives")
    status = "failed" if failures else ("manual_review" if warnings else "passed")
    top_paths = sorted(path_records, key=lambda item: (
        -item["node_count"], -item["short_segment_count"]))[:20]
    top_heuristic_paths = sorted(heuristic_records, key=lambda item: (
        -item["node_count"], -item["short_segment_count"]))[:20]
    public_heuristic_metrics = {
        key: value for key, value in heuristic_metrics.items()
        if key not in {"dense_failure_loops", "dense_warning_loops",
                       "loop_records"}
    }
    return {
        "id": "curve_economy_gate",
        "status": status,
        "passed": status == "passed",
        "failure_reasons": failures,
        "warning_reasons": warnings,
        "informational_reasons": informational,
        "thresholds": {
            "short_segment_user_units": _rounded(short_limit),
            "near_collinear_user_units": _rounded(collinear_limit),
            "short_segment_ratio_warning": thresholds["short_segment_ratio_warning"],
            "short_segment_ratio_failure": thresholds["short_segment_ratio_failure"],
            "near_collinear_ratio_warning": thresholds["near_collinear_ratio_warning"],
            "near_collinear_ratio_failure": thresholds["near_collinear_ratio_failure"],
            "anchors_per_100_warning": thresholds["anchors_per_100_warning"],
            "anchors_per_100_failure": thresholds["anchors_per_100_failure"],
            "density_minimum_loop_nodes_warning": 20,
            "density_minimum_loop_nodes_failure": 40,
        },
        "metrics": {
            "path_count": len(path_records),
            "node_count": total_nodes,
            "command_count": total_commands,
            "cubic_segment_count": total_cubic,
            "draw_segment_count": total_draw_segments,
            "short_segment_count": short_segments,
            "short_segment_ratio": short_ratio,
            "near_collinear_cubic_count": collinear,
            "near_collinear_cubic_ratio": collinear_ratio,
            "approximated_path_length": _rounded(length, 3),
            "anchors_per_100_user_units": anchors_per_100,
            "max_nodes_in_single_path": max_nodes,
            "paths_over_40_nodes": paths_over_40,
            "paths_over_80_nodes": paths_over_80,
            "loop_count": len(loop_records),
            "max_nodes_in_single_loop": max_loop_nodes,
            "loops_over_40_nodes": loops_over_40,
            "loops_over_80_nodes": loops_over_80,
            "max_loop_anchors_per_100_user_units": _rounded(max_loop_density),
            "dense_loop_failure_count": len(dense_failure_loops),
            "dense_loop_warning_count": len(dense_warning_loops),
            "native_circle_designer_anchor_count": native["circle"] * 4,
            "unparsed_path_count": invalid_paths,
            "short_counterturn_segment_count": counterturn_count,
            "short_counterturn_path_count": len(counterturn_paths),
            "short_counterturn_review_path_count": len(counterturn_review_paths),
        },
        "short_counterturn_diagnostic": {
            "status": ("manual_review" if counterturn_review_paths else
                       "coverage_incomplete" if counterturn_incomplete_paths else
                       "no_repeated_counterturn_observed"),
            "scope": "all_svg_paths_including_optimizer_certified_paths",
            "semantic_defect_claim": False,
            "thresholds": {
                "endpoint_chord_diagonal_fraction": .006,
                "endpoint_chord_user_units": _rounded(diagonal*.006),
                "sample_points_per_short_cubic": 24,
                "minimum_total_turn_degrees": 120,
                "minimum_counterturn_degrees": 45,
                "minimum_qualifying_segments_per_path": 4,
            },
            "path_count": len(path_records),
            "assessed_path_count": sum(path["short_counterturn"]["status"] == "completed"
                                       for path in path_records),
            "unassessed_or_partial_path_count": len(counterturn_incomplete_paths),
            "short_cubic_sampled_count": sum(path["short_counterturn"]["short_cubic_count"]
                                              for path in path_records),
            "high_turn_short_cubic_count": sum(path["short_counterturn"]["high_turn_short_cubic_count"]
                                               for path in path_records),
            "counterturn_segment_count": counterturn_count,
            "affected_path_count": len(counterturn_paths),
            "review_path_count": len(counterturn_review_paths),
            "path_record_limit": 50,
            "path_records_truncated": len(counterturn_paths) > 50,
            "paths": [{
                "id": path["id"], "source_layer": path["source_layer"],
                "optimizer_certified": path["id"] in certified_ids,
                "short_counterturn_count": path["short_counterturn"]["counterturn_segment_count"],
                **path["short_counterturn"],
            } for path in sorted(counterturn_paths, key=lambda item:
                                 -item["short_counterturn"]["counterturn_segment_count"])[:50]],
            "coverage_exceptions": [{"id": path["id"],
                                     "reason": path["short_counterturn"].get("reason")}
                                    for path in path_records
                                    if path["short_counterturn"]["status"] != "completed"][:50],
            "scope_note": (
                "Sampled directional reversal is a manual inspection hint, not an "
                "automatic defect or source-fidelity verdict. Smooth one-way turns "
                "do not qualify. Geometry-distance certificates do not waive this "
                "independent observation; public geometry counts remain unchanged."),
        },
        "heuristic_scope_metrics": {
            **public_heuristic_metrics,
            "certified_path_count_excluded": len(certified_ids),
            "uncertified_path_ids": [
                path["id"] for path in heuristic_records[:60]],
            "scope": (
                "only_paths_without_authoritative_optimizer_economy_"
                "certificate"),
        },
        "optimizer_economy_evidence": {
            "gradient": gradient_certificate,
            "curve_refit": curve_certificate,
            "certified_path_count": len(certified_ids),
            "certified_path_ids": sorted(certified_ids)[:200],
            "generic_heuristic_path_count": len(heuristic_records),
            "contract": (
                "transaction-backed lexicographic minimum under topology, "
                "p95/tail/share/salient error hard constraints supersedes "
                "generic trace-density heuristics for that exact path only"),
        },
        "geometry_error_budget_evidence": {
            "contract": {
                "p95": "p95_error_percent <= error_budget_percent",
                "tail": "max_error_percent <= 3 * error_budget_percent",
                "anchor_objective": (
                    "minimize_designer_anchors_subject_to_the_error_contract"
                ),
                "visual_similarity_can_override": False,
            },
            "applicable_object_count": len(evidence_records),
            "complete_object_count": sum(
                evidence["complete"] for _, evidence in evidence_records),
            "missing_object_ids": missing_evidence[:30],
            "invalid_object_ids": invalid_evidence[:30],
            "p95_violation_object_ids": p95_violations[:30],
            "max_violation_object_ids": max_violations[:30],
            "native_circle_normalized_object_ids": normalized_circles[:30],
        },
        "primitive_recovery": {
            "native_counts": dict(sorted(native.items())),
            "native_designer_anchor_policy": {"circle": 4},
            "native_circle_designer_anchor_count": native["circle"] * 4,
            "near_circle_path_candidate_count": len(circle_candidates),
            "near_circle_path_candidates": [path["id"] for path in circle_candidates[:20]],
            "overanchored_circle_path_count": len(
                overanchored_circle_candidates),
            "overanchored_circle_paths": [
                path["id"] for path in overanchored_circle_candidates[:20]],
            "mildly_overanchored_circle_path_count": len(
                mildly_overanchored_circle_candidates),
            "scope_note": "Near-circle candidates are geometric hints, not semantic proof.",
        },
        "highest_anchor_paths": [{
            "id": path["id"],
            "source_layer": path["source_layer"],
            "fill": path["fill"],
            "node_count": path["node_count"],
            "loop_count": path["loop_count"],
            "max_nodes_in_single_loop": path["max_nodes_in_loop"],
            "max_loop_anchors_per_100_user_units": path[
                "max_loop_anchors_per_100_user_units"],
            "command_count": path["command_count"],
            "cubic_segment_count": path["cubic_count"],
            "short_segment_count": path["short_segment_count"],
            "near_collinear_cubic_count": path["near_collinear_cubic_count"],
            "approximated_length": _rounded(path["approximated_length"], 3),
            "bbox": [_rounded(item, 3) for item in path["bbox"]]
                    if path["bbox"] else None,
        } for path in top_paths],
        "highest_anchor_uncertified_paths": [{
            "id": path["id"],
            "source_layer": path["source_layer"],
            "fill": path["fill"],
            "node_count": path["node_count"],
            "loop_count": path["loop_count"],
            "max_nodes_in_single_loop": path["max_nodes_in_loop"],
            "short_segment_count": path["short_segment_count"],
            "near_collinear_cubic_count": path[
                "near_collinear_cubic_count"],
        } for path in top_heuristic_paths],
        "scope_note": (
            "Generic thresholds operate only on paths that lack a transaction-backed "
            "optimizer economy certificate. Certified paths still must pass topology, "
            "p95/tail/share/salient geometry constraints and exact final-SVG identity; "
            "raster or colour similarity cannot override those contracts."
        ),
    }


def audit_designer_quality(
    svg_path: str | Path,
    proposal_metadata: Mapping[str, Any] | str | Path | None = None,
    visual_report: Mapping[str, Any] | str | Path | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Audit gradient-object integrity and curve economy of one SVG."""
    path = Path(svg_path)
    payload = path.read_bytes()
    svg_sha256 = hashlib.sha256(payload).hexdigest()
    root = ET.fromstring(payload)
    parents = {child: parent for parent in root.iter() for child in parent}
    identified_elements = [element for element in root.iter()
                           if element.get("id")]
    element_id_counts = Counter(
        str(element.get("id")) for element in identified_elements)
    element_geometry = {
        str(element.get("id")): _final_element_geometry_identity(element)
        for element in identified_elements
    }
    viewbox = _viewbox(root)
    diagonal = math.hypot(viewbox[2], viewbox[3])
    viewbox_area = viewbox[2] * viewbox[3]
    effective_thresholds = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        effective_thresholds.update({key: float(value) for key, value in thresholds.items()
                                     if key in effective_thresholds})

    path_records: list[dict[str, Any]] = []
    path_by_element: dict[ET.Element, dict[str, Any]] = {}
    for index, element in enumerate(
            item for item in root.iter() if _local(item.tag) == "path"):
        path_data = element.get("d", "")
        segments, valid = _path_segments(path_data)
        draw_segments = [segment for segment in segments
                         if segment["type"] not in {"M", "Z"}]
        cubics = [segment for segment in draw_segments if segment["type"] == "C"]
        short_limit = max(1.0, diagonal * effective_thresholds[
            "short_segment_diagonal_fraction"])
        collinear_limit = max(0.5, diagonal * effective_thresholds[
            "near_collinear_diagonal_fraction"])
        near_collinear = sum(
            max(_line_deviation(control, segment["start"], segment["end"])
                for control in segment["controls"]) <= collinear_limit
            for segment in cubics if segment["controls"]
        )
        moves = [segment for segment in segments if segment["type"] == "M"]
        loops = _path_loops(segments)
        loop_densities = [loop["anchors_per_100_user_units"] for loop in loops
                          if loop["anchors_per_100_user_units"] is not None]
        record = {
            "id": element.get("id") or f"path-{index + 1}",
            "svg_id": element.get("id") or None,
            "source_layer": element.get("data-source-layer"),
            "fill": _property(element, parents, "fill", "#000000"),
            "node_count": sum(loop["node_count"] for loop in loops),
            "loop_count": len(loops),
            "loops": loops,
            "max_nodes_in_loop": max(
                (loop["node_count"] for loop in loops), default=0),
            "max_loop_anchors_per_100_user_units": _rounded(
                max(loop_densities, default=None)),
            "command_count": len(segments),
            "cubic_count": len(cubics),
            "draw_segment_count": len(draw_segments),
            "short_segment_count": sum(
                _segment_length(segment) <= short_limit
                for segment in draw_segments),
            "near_collinear_cubic_count": near_collinear,
            "approximated_length": sum(_segment_length(segment)
                                       for segment in draw_segments),
            "bbox": _path_bbox(segments),
            "closed": any(segment["type"] == "Z" for segment in segments),
            "endpoints": [segment["end"] for segment in draw_segments],
            "parse_valid": valid,
            "short_counterturn": _short_counterturn_diagnostic(element, parents, segments, diagonal),
            "geometry_evidence": _geometry_evidence(element, parents),
            "path_data_sha256": hashlib.sha256(
                path_data.encode("utf-8")).hexdigest(),
            "curve_refit_claim": _property(
                element, parents, "data-avc-curve-refit", "").strip() or None,
            "anchors_before_claim": (
                None if _number(_property(
                    element, parents, "data-avc-anchors-before", "")) is None
                else int(round(_number(_property(
                    element, parents, "data-avc-anchors-before", ""))))),
            "anchors_after_claim": (
                None if _number(_property(
                    element, parents, "data-avc-anchors-after", "")) is None
                else int(round(_number(_property(
                    element, parents, "data-avc-anchors-after", ""))))),
            "gradient_object_id": _property(
                element, parents, "data-avc-gradient-object", "").strip()
                or None,
        }
        path_records.append(record)
        if not valid:
            record["short_counterturn"].update(status="partially_assessed",
                                             reason="path_data_not_fully_parsed")
        path_by_element[element] = record

    drawable_records: list[dict[str, Any]] = []
    transformed = 0
    for index, element in enumerate(
            item for item in root.iter() if _local(item.tag) in DRAWABLES):
        kind = _local(element.tag)
        hidden = _property(element, parents, "display", "").strip().lower() == "none"
        opacity = _number(_property(element, parents, "opacity", "1"))
        fill_opacity = _number(_property(element, parents, "fill-opacity", "1"))
        default_fill = "none" if kind in {"line", "polyline"} else "#000000"
        fill = _property(element, parents, "fill", default_fill).strip()
        if hidden or opacity == 0 or fill_opacity == 0:
            fill_kind = "hidden"
        elif _URL_REF.search(fill):
            fill_kind = "gradient"
        elif fill.lower() == "none":
            fill_kind = "none"
        else:
            fill_kind = "solid"
        normalised = _normalise_colour(fill) if fill_kind == "solid" else None
        rgb = normalised[1] if normalised else None
        hsv = colorsys.rgb_to_hsv(*(channel / 255.0 for channel in rgb)) if rgb else None
        chromatic = bool(hsv and hsv[1] >= 0.10 and hsv[2] <= 0.985)
        path_record = path_by_element.get(element)
        bbox = _element_bbox(element, None if path_record is None else
                             _path_segments(element.get("d", ""))[0])
        node = element
        has_transform = False
        while node is not None:
            if node.get("transform"):
                has_transform = True
                break
            node = parents.get(node)
        if has_transform:
            transformed += 1
            bbox = None
        drawable_records.append({
            "id": element.get("id") or f"{kind}-{index + 1}",
            "tag": kind,
            "fill": normalised[0] if normalised else fill,
            "fill_kind": fill_kind,
            "rgb": rgb,
            "hsv": hsv,
            "chromatic": chromatic,
            "bbox": bbox,
            "source_layer": element.get("data-source-layer"),
            "gradient_references": _URL_REF.findall(fill),
            "gradient_object_id": _property(
                element, parents, "data-avc-gradient-object", "").strip() or None,
            "geometry_evidence": _geometry_evidence(element, parents),
        })

    proposal = _load_json(proposal_metadata)
    proposal_source = "argument" if proposal is not None else "not-found"
    if proposal is None:
        for identifier in (
                "gradient-object-proposals", "gradient-object-metadata",
                "scene-graph-metadata"):
            proposal = _embedded_json(root, identifier)
            if proposal is not None:
                proposal_source = f"embedded:{identifier}"
                break
    visual = _visual_status(root, _load_json(visual_report))
    gradient_gate = _gradient_object_gate(
        root, parents, drawable_records, diagonal, viewbox_area,
        proposal, proposal_source, effective_thresholds,
    )
    curve_gate = _curve_economy_gate(
        root, parents, path_records, diagonal, effective_thresholds,
        proposal_report=proposal,
        gradient_gate=gradient_gate,
        svg_sha256=svg_sha256,
        element_geometry=element_geometry,
        element_id_counts=element_id_counts,
    )
    stroke_evidence = (proposal or {}).get("stroke_reconstruction_report")
    stroke_gate = {"status": "not_assessed", "scope": "native_source_stroke_caps",
                   "reasons": []}
    if isinstance(stroke_evidence, Mapping):
        value = stroke_evidence.get("complex_strokes_without_native_cap_proof")
        valid_count = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        stroke_gate.update(status="manual_review" if not valid_count or value else "passed",
                           complex_strokes_without_native_cap_proof=value,
                           evidence=stroke_evidence)
        if not valid_count:
            stroke_gate["reasons"].append("native_stroke_cap_evidence_incomplete")
        elif value:
            stroke_gate["reasons"].append("complex_stroke_caps_need_source_review")
    topology_gate = _source_topology_gate(proposal or {}, svg_sha256)
    gate_statuses = [gradient_gate["status"], curve_gate["status"], stroke_gate["status"], topology_gate["status"]]
    if "failed" in gate_statuses:
        readiness = "manual_rework_required"
    elif "manual_review" in gate_statuses:
        readiness = "manual_review_required"
    else:
        readiness = "designer_ready"
    divergence = bool(visual["accepted"] and readiness != "designer_ready")
    result = {
        "schema": AUDIT_SCHEMA,
        "source": {
            "filename": path.name,
            "sha256": svg_sha256,
            "bytes": len(payload),
            "viewbox": [_rounded(item, 3) for item in viewbox],
        },
        "raster_visual_gate": visual,
        "gradient_object_gate": gradient_gate,
        "curve_economy_gate": curve_gate,
        "stroke_source_gate": stroke_gate,
        "source_topology_gate": topology_gate,
        "designer_readiness_status": readiness,
        "designer_ready": readiness == "designer_ready",
        "raster_visual_designer_status_divergence": divergence,
        "status_explanation": (
            "Raster visual acceptance does not override designer-quality failures."
            if divergence else
            "Designer readiness is determined independently from raster similarity."
        ),
        "document_inventory": {
            "drawable_count": len(drawable_records),
            "path_count": len(path_records),
            "transformed_drawable_count_excluded_from_spatial_clustering": transformed,
        },
        "human_designer_validation": "not_performed_by_this_machine_audit",
        "scope_note": (
            "This gate measures machine-detectable gradient-object fragmentation and "
            "curve economy. Final semantic correctness remains a designer decision."
        ),
    }
    # Fail early during development if a future edit leaks NaN or an XML object.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def _source_topology_gate(proposal, svg_sha256):
    """Unavailable or stale source checks cannot turn into a readiness pass."""
    report = proposal.get("source_topology_report")
    repair = (proposal.get("editability_enhancements") or {}).get("stages", {}).get(
        "source_reconstruction") or {}
    required = repair.get("status") == "committed"
    result = {"status": "not_assessed", "scope": "stable_native_source_structure",
              "reasons": [], "required_after_source_reconstruction": required}
    if report is None and not required:
        return result
    result["status"] = "manual_review"
    if not isinstance(report, Mapping):
        result["reasons"] = ["source_topology_evidence_missing_or_invalid"]
        return result
    result["evidence"] = report
    status = report.get("status")
    if status == "not_applicable" and not required:
        # This audit covers reliable opaque paper only. Native-alpha sources
        # retain their separate alpha checks, without claiming a paper pass.
        result.update(status="not_applicable", reasons=list(report.get("reasons") or []))
        return result
    if status != "completed":
        result["reasons"] = ["source_topology_audit_not_completed"]
        result["reasons"].extend(str(reason) for reason in report.get("reasons", [])
                                 if isinstance(reason, str))
        if report.get("reason"):
            result["reasons"].append(str(report["reason"]))
        return result
    if (report.get("source_hashes") or {}).get("svg") != svg_sha256:
        result["reasons"] = ["source_topology_svg_evidence_stale"]
        return result
    if (report.get("schema") != "aivc.source-topology-audit/v1"
            or report.get("native_resolution") is not True
            or report.get("inputs_unchanged") is not True
            or not isinstance(report.get("manual_review"), bool)
            or not isinstance(report.get("stable_defects"), list)):
        result["reasons"] = ["source_topology_evidence_incomplete"]
        return result
    needs_review = report["manual_review"] or bool(report["stable_defects"])
    result.update(status="manual_review" if needs_review else "passed",
                  reasons=list(report.get("reasons") or []))
    if needs_review and not result["reasons"]:
        result["reasons"] = ["source_topology_local_defects_need_review"]
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description="Audit SVG designer readiness")
    parser.add_argument("svg", type=Path)
    parser.add_argument("--proposal-metadata", type=Path)
    parser.add_argument("--visual-report", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = audit_designer_quality(
        arguments.svg, arguments.proposal_metadata, arguments.visual_report)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


__all__ = ["AUDIT_SCHEMA", "DEFAULT_THRESHOLDS", "audit_designer_quality"]


if __name__ == "__main__":
    raise SystemExit(_main())
