"""Conservative SVG scene-graph post-processing.

Bitmap tracers commonly emit one group per paint layer.  That preserves the
picture but makes a multi-colour object awkward to select: its highlight,
shadow and base fill live in unrelated groups.  This module builds spatial
object groups without relying on labels, language, or image-specific
coordinates.

The important constraint is paint order.  Moving two SVG elements into one
``<g>`` can change their order relative to intervening elements.  A candidate
group is therefore applied only when every order inversion is between
non-overlapping conservative bounding boxes.  The final order is checked a
second time; any invariant or optional render validation failure returns the
original SVG byte-for-byte.

The module is deliberately standalone.  It is suitable for a later pipeline
integration, but importing it does not change vector_cleanroom output.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Callable, Iterable
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)

DRAWABLE_TAGS = {
    "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
}

# Attributes inherited by ordinary SVG paint groups.  They are materialised
# on leaves before the old layer tree is flattened, so the picture remains the
# same even though selection structure changes.
PRESENTATION_ATTRS = {
    "fill", "fill-rule", "fill-opacity", "stroke", "stroke-width",
    "stroke-opacity", "stroke-linecap", "stroke-linejoin",
    "stroke-miterlimit", "stroke-dasharray", "stroke-dashoffset",
    "clip-rule", "color", "color-interpolation", "color-rendering",
    "marker-start", "marker-mid", "marker-end", "vector-effect",
    "shape-rendering", "paint-order", "pointer-events", "visibility",
    "display",
}

# Transform/clip/filter semantics cannot safely be flattened by copying a few
# attributes.  Encountering any of them causes a full, explicit rollback.
UNSAFE_GROUP_ATTRS = {
    "transform", "style", "class", "clip-path", "mask", "filter",
    "mix-blend-mode", "isolation", "opacity",
}

NON_SCENE_ROOT_TAGS = {"defs", "metadata", "title", "desc", "namedview"}

PATH_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
SVG_LENGTH_RE = re.compile(
    r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:px)?\s*",
    re.IGNORECASE,
)
PATH_ARITY = {
    "M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
    "Q": 4, "T": 2, "A": 7,
}

UNSAFE_DRAWABLE_ATTRS = {
    "transform", "style", "class", "mask", "filter",
    "marker-start", "marker-mid", "marker-end",
    "mix-blend-mode", "isolation",
}

_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SELF_ROTATED_CIRCLE_RE = re.compile(
    rf"\s*rotate\(\s*({_FLOAT_PATTERN})[ ,]+({_FLOAT_PATTERN})"
    rf"[ ,]+({_FLOAT_PATTERN})\s*\)\s*",
    re.IGNORECASE,
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _qname(local: str) -> str:
    return f"{{{SVG_NS}}}{local}"


def _safe_self_rotated_circle(element: ET.Element) -> bool:
    """A circle rotated about its own centre keeps the same trusted bbox.

    Annulus reconstruction uses this exact transform only to phase a dashed
    stroke. Other transforms remain fail-closed because their geometry or
    stroke width would require a complete matrix evaluation.
    """

    if _local(element.tag) != "circle":
        return False
    match = _SELF_ROTATED_CIRCLE_RE.fullmatch(element.get("transform", ""))
    if not match:
        return False
    cx = _numbers(element.get("cx", ""))
    cy = _numbers(element.get("cy", ""))
    if not cx or not cy:
        return False
    rotate_cx = float(match.group(2))
    rotate_cy = float(match.group(3))
    tolerance = 1e-5 * max(1.0, abs(cx[0]), abs(cy[0]))
    return (abs(rotate_cx - cx[0]) <= tolerance
            and abs(rotate_cy - cy[0]) <= tolerance)


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def union(self, other: "BBox") -> "BBox":
        return BBox(min(self.x0, other.x0), min(self.y0, other.y0),
                    max(self.x1, other.x1), max(self.y1, other.y1))

    def expanded(self, amount: float) -> "BBox":
        return BBox(self.x0 - amount, self.y0 - amount,
                    self.x1 + amount, self.y1 + amount)

    def intersection_area(self, other: "BBox") -> float:
        width = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        height = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        return width * height

    def overlaps(self, other: "BBox", pad: float = 0.0) -> bool:
        a = self.expanded(pad) if pad else self
        b = other.expanded(pad) if pad else other
        return not (a.x1 < b.x0 or b.x1 < a.x0
                    or a.y1 < b.y0 or b.y1 < a.y0)

    def gap(self, other: "BBox") -> float:
        dx = max(self.x0 - other.x1, other.x0 - self.x1, 0.0)
        dy = max(self.y0 - other.y1, other.y0 - self.y1, 0.0)
        return math.hypot(dx, dy)

    def as_list(self) -> list[float]:
        return [round(self.x0, 3), round(self.y0, 3),
                round(self.x1, 3), round(self.y1, 3)]


@dataclass
class SceneNode:
    element: ET.Element
    index: int
    node_id: str
    source_layer: str
    paint: str
    bbox: BBox
    tag: str
    area_fraction: float
    is_scaffold: bool
    primitive_like: bool
    role: str
    orientation: float
    paint_family: str
    paint_hue: float | None


@dataclass(frozen=True)
class Edge:
    left: int
    right: int
    confidence: float
    reason: str


@dataclass
class ObjectCandidate:
    members: list[int]
    confidence: float
    reasons: list[str]
    bbox: BBox
    anchor: int | None = None
    group_id: str = ""
    kind: str = "object-cluster"
    label: str = "Object cluster"


@dataclass
class SceneGraphResult:
    svg_text: str
    status: str
    report: dict[str, object] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.status == "applied"


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.members = {index: {index} for index in range(size)}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> int:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return left
        if len(self.members[left]) < len(self.members[right]):
            left, right = right, left
        self.parent[right] = left
        self.members[left].update(self.members.pop(right))
        return left


def _numbers(value: str) -> list[float]:
    return [float(item) for item in re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value or "")]


def _path_bbox(path_data: str) -> BBox | None:
    """Return a conservative path bbox including Bezier control points.

    Generated cleanroom paths use absolute M/L/C commands, but supporting all
    ordinary SVG path commands costs little and makes the postprocessor useful
    for later engines too.  Arc bounds deliberately overestimate rather than
    risk a false non-overlap during paint-order validation.
    """
    tokens = PATH_TOKEN_RE.findall(path_data or "")
    if not tokens:
        return None
    xs: list[float] = []
    ys: list[float] = []
    current = (0.0, 0.0)
    sub_start = current
    previous_command = ""
    last_cubic: tuple[float, float] | None = None
    last_quad: tuple[float, float] | None = None
    command = ""
    index = 0

    def point(x: float, y: float) -> None:
        xs.append(x)
        ys.append(y)

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command.upper() == "Z":
                current = sub_start
                point(*current)
                last_cubic = last_quad = None
                previous_command = command
                continue
        if not command or command.upper() not in PATH_ARITY:
            return None
        upper = command.upper()
        relative = command.islower()
        arity = PATH_ARITY[upper]
        first_set = True
        consumed = False
        while index + arity <= len(tokens) and not tokens[index].isalpha():
            try:
                values = [float(value) for value in tokens[index:index + arity]]
            except ValueError:
                return None
            index += arity
            consumed = True
            ox, oy = current
            if upper == "M":
                x, y = values
                if relative:
                    x += ox; y += oy
                current = (x, y)
                if first_set:
                    sub_start = current
                point(x, y)
            elif upper == "L":
                x, y = values
                if relative:
                    x += ox; y += oy
                current = (x, y); point(x, y)
            elif upper == "H":
                x = values[0] + (ox if relative else 0.0)
                current = (x, oy); point(*current)
            elif upper == "V":
                y = values[0] + (oy if relative else 0.0)
                current = (ox, y); point(*current)
            elif upper == "C":
                c1x, c1y, c2x, c2y, x, y = values
                if relative:
                    c1x += ox; c1y += oy; c2x += ox; c2y += oy
                    x += ox; y += oy
                point(c1x, c1y); point(c2x, c2y); point(x, y)
                current = (x, y); last_cubic = (c2x, c2y); last_quad = None
            elif upper == "S":
                c2x, c2y, x, y = values
                if relative:
                    c2x += ox; c2y += oy; x += ox; y += oy
                if previous_command.upper() in {"C", "S"} and last_cubic:
                    point(2 * ox - last_cubic[0], 2 * oy - last_cubic[1])
                point(c2x, c2y); point(x, y)
                current = (x, y); last_cubic = (c2x, c2y); last_quad = None
            elif upper == "Q":
                qx, qy, x, y = values
                if relative:
                    qx += ox; qy += oy; x += ox; y += oy
                point(qx, qy); point(x, y)
                current = (x, y); last_quad = (qx, qy); last_cubic = None
            elif upper == "T":
                x, y = values
                if relative:
                    x += ox; y += oy
                if previous_command.upper() in {"Q", "T"} and last_quad:
                    point(2 * ox - last_quad[0], 2 * oy - last_quad[1])
                point(x, y); current = (x, y); last_cubic = None
            elif upper == "A":
                rx, ry, rotation, _large, _sweep, x, y = values
                if relative:
                    x += ox; y += oy
                # A rotated ellipse can project its larger radius onto either
                # axis.  Use the maximum radius for both dimensions; this is
                # deliberately loose, but never underestimates solely because
                # rx/ry were swapped by rotation.
                rx = abs(rx); ry = abs(ry)
                if rx and ry:
                    radians = math.radians(rotation % 360.0)
                    dx = (ox - x) / 2.0
                    dy = (oy - y) / 2.0
                    transformed_x = math.cos(radians) * dx + math.sin(radians) * dy
                    transformed_y = -math.sin(radians) * dx + math.cos(radians) * dy
                    correction = math.sqrt(max(
                        1.0,
                        transformed_x ** 2 / rx ** 2
                        + transformed_y ** 2 / ry ** 2,
                    ))
                else:
                    correction = 1.0
                radius = 2 * max(rx * correction, ry * correction)
                point(x - radius, y - radius)
                point(x + radius, y + radius)
                point(ox - radius, oy - radius)
                point(ox + radius, oy + radius)
                current = (x, y); last_cubic = last_quad = None
            if upper not in {"C", "S", "Q", "T"}:
                last_cubic = last_quad = None
            previous_command = command
            first_set = False
            # Extra moveto pairs become lineto pairs.
            if upper == "M":
                command = "l" if relative else "L"
                upper = "L"
                arity = 2
        if not consumed and index < len(tokens) and not tokens[index].isalpha():
            return None
    if not xs:
        return None
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _element_bbox(element: ET.Element) -> BBox | None:
    tag = _local(element.tag)
    try:
        if tag == "path":
            bbox = _path_bbox(element.get("d", ""))
        elif tag == "circle":
            cx = float(element.get("cx", "0")); cy = float(element.get("cy", "0"))
            radius = abs(float(element.get("r", "0")))
            bbox = BBox(cx - radius, cy - radius, cx + radius, cy + radius)
        elif tag == "ellipse":
            cx = float(element.get("cx", "0")); cy = float(element.get("cy", "0"))
            rx = abs(float(element.get("rx", "0"))); ry = abs(float(element.get("ry", "0")))
            bbox = BBox(cx - rx, cy - ry, cx + rx, cy + ry)
        elif tag == "rect":
            x = float(element.get("x", "0")); y = float(element.get("y", "0"))
            width = float(element.get("width", "0")); height = float(element.get("height", "0"))
            bbox = BBox(x, y, x + width, y + height)
        elif tag == "line":
            x1 = float(element.get("x1", "0")); y1 = float(element.get("y1", "0"))
            x2 = float(element.get("x2", "0")); y2 = float(element.get("y2", "0"))
            bbox = BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        elif tag in {"polyline", "polygon"}:
            values = _numbers(element.get("points", ""))
            if len(values) < 2:
                return None
            xs = values[0::2]; ys = values[1::2]
            bbox = BBox(min(xs), min(ys), max(xs), max(ys))
        else:
            return None
    except (TypeError, ValueError):
        return None
    if bbox is None:
        return None
    stroke = element.get("stroke", "none").strip().lower()
    if stroke not in {"", "none", "transparent"}:
        width_match = SVG_LENGTH_RE.fullmatch(element.get("stroke-width", "1"))
        if width_match is None:
            return None
        width = abs(float(width_match.group(1)))
        join = element.get("stroke-linejoin", "miter").strip().lower()
        if join == "miter":
            limits = _numbers(element.get("stroke-miterlimit", "4"))
            miter_limit = max(1.0, limits[0]) if limits else 4.0
        else:
            miter_limit = 1.0
        pad = width * miter_limit / 2.0
        bbox = bbox.expanded(pad)
    return bbox


def _viewbox(root: ET.Element) -> BBox:
    values = _numbers(root.get("viewBox", ""))
    if len(values) == 4 and values[2] > 0 and values[3] > 0:
        return BBox(values[0], values[1], values[0] + values[2], values[1] + values[3])
    width = _numbers(root.get("width", ""))
    height = _numbers(root.get("height", ""))
    if width and height and width[0] > 0 and height[0] > 0:
        return BBox(0.0, 0.0, width[0], height[0])
    raise ValueError("SVG has no usable viewBox or width/height")


def _effective_paint(element: ET.Element) -> str:
    fill = element.get("fill", "black").strip().lower()
    stroke = element.get("stroke", "none").strip().lower()
    if fill not in {"", "none", "transparent"}:
        return "fill:" + fill
    if stroke not in {"", "none", "transparent"}:
        return "stroke:" + stroke
    return "none"


def _paint_profile(paint: str) -> tuple[str, float | None]:
    value = paint.split(":", 1)[-1].strip().lower()
    match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", value)
    if not match:
        return ("unknown" if value.startswith("url(") else "other", None)
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(char * 2 for char in digits)
    red, green, blue = (int(digits[index:index + 2], 16)
                        for index in (0, 2, 4))
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum
    if maximum <= 45 or delta <= 20 or delta / max(maximum, 1) <= 0.14:
        return "neutral", None
    if delta == 0:
        return "neutral", None
    if maximum == red:
        hue = (60.0 * ((green - blue) / delta)) % 360.0
    elif maximum == green:
        hue = 60.0 * ((blue - red) / delta + 2.0)
    else:
        hue = 60.0 * ((red - green) / delta + 4.0)
    return "chromatic", hue


def _paint_compatible(left: SceneNode, right: SceneNode,
                      *, allow_unknown: bool = False) -> bool:
    if left.paint == right.paint:
        return True
    if left.paint_family == right.paint_family == "neutral":
        return True
    if left.paint_family == right.paint_family == "chromatic":
        assert left.paint_hue is not None and right.paint_hue is not None
        difference = abs(left.paint_hue - right.paint_hue)
        return min(difference, 360.0 - difference) <= 55.0
    if allow_unknown and "unknown" in {left.paint_family, right.paint_family}:
        return True
    return False


def _canonical_element(element: ET.Element, source_layer: str) -> str:
    ignored = {"id", f"{{{INKSCAPE_NS}}}label", "data-source-layer",
               "data-object-id", "data-group-confidence"}
    attrs = sorted((key, value) for key, value in element.attrib.items()
                   if key not in ignored)
    return json.dumps([_local(element.tag), source_layer, attrs],
                      ensure_ascii=False, separators=(",", ":"))


def _stable_node_ids(elements: list[tuple[ET.Element, str]]) -> list[str]:
    bases: list[str] = []
    for element, source_layer in elements:
        existing = element.get("id")
        if existing:
            bases.append(existing)
        else:
            digest = hashlib.sha1(
                _canonical_element(element, source_layer).encode("utf-8")
            ).hexdigest()[:12]
            bases.append("sg-node-" + digest)
    totals = Counter(bases)
    seen: Counter[str] = Counter()
    result = []
    for base in bases:
        seen[base] += 1
        suffix = f"-{seen[base]}" if totals[base] > 1 else ""
        result.append(base + suffix)
    if len(result) != len(set(result)):
        raise ValueError("could not generate unique stable scene-node IDs")
    return result


def _flatten_drawables(root: ET.Element) -> tuple[list[tuple[ET.Element, str]], list[ET.Element]]:
    """Copy drawables in paint order and return root children they replace."""
    flattened: list[tuple[ET.Element, str]] = []
    scene_children: list[ET.Element] = []

    def visit(element: ET.Element, inherited: dict[str, str], layer: str) -> None:
        local = _local(element.tag)
        if local == "g":
            for key in element.attrib:
                bare = _local(key)
                if bare in UNSAFE_GROUP_ATTRS:
                    raise ValueError(f"unsafe inherited group attribute: {bare}")
                is_metadata = (
                    bare == "id" or bare == "role"
                    or bare.startswith("data-") or bare.startswith("aria-")
                    or (key.startswith(f"{{{INKSCAPE_NS}}}")
                        and bare in {"label", "groupmode"})
                )
                if bare not in PRESENTATION_ATTRS and not is_metadata:
                    raise ValueError(
                        f"unsupported inherited group attribute: {bare}")
            merged = dict(inherited)
            for key, value in element.attrib.items():
                bare = _local(key)
                if bare in PRESENTATION_ATTRS:
                    merged[bare] = value
            next_layer = element.get("id") or layer
            for child in element:
                visit(child, merged, next_layer)
            return
        if local not in DRAWABLE_TAGS:
            # Definitions and metadata are handled at root level.  Unknown
            # content inside a paint group could affect appearance, so fail
            # closed rather than silently dropping it.
            raise ValueError(f"unsupported content inside scene group: {local}")
        copied = deepcopy(element)
        for key, value in inherited.items():
            copied.attrib.setdefault(key, value)
        flattened.append((copied, layer or "root"))

    for child in list(root):
        local = _local(child.tag)
        if local == "g" and any(_local(item.tag) in DRAWABLE_TAGS for item in child.iter()):
            scene_children.append(child)
            visit(child, {}, child.get("id") or "layer")
        elif local in DRAWABLE_TAGS:
            scene_children.append(child)
            visit(child, {}, "root")
        elif local not in NON_SCENE_ROOT_TAGS:
            # Leaving a visual root child in place while consolidating the
            # surrounding paint groups would silently change its z-order.
            # Embedded CSS could likewise depend on the old group hierarchy.
            raise ValueError(f"unsupported root scene content: {local}")
    return flattened, scene_children


def _primitive_like(node: SceneNode) -> bool:
    if node.tag in {"circle", "ellipse", "line", "rect"}:
        return True
    if node.tag in {"polygon", "polyline"}:
        return True
    if node.tag == "path":
        commands = len(re.findall(r"[A-Za-z]", node.element.get("d", "")))
        return commands <= 12
    return False


def _make_nodes(flattened: list[tuple[ET.Element, str]], canvas: BBox,
                max_group_bbox_fraction: float) -> list[SceneNode]:
    node_ids = _stable_node_ids(flattened)
    nodes: list[SceneNode] = []
    canvas_area = max(canvas.area, 1.0)
    canvas_diagonal = max(canvas.diagonal, 1.0)
    for index, ((element, source_layer), node_id) in enumerate(zip(flattened, node_ids)):
        unsafe = sorted(
            _local(key) for key in element.attrib
            if (_local(key) in UNSAFE_DRAWABLE_ATTRS
                and not (_local(key) == "transform"
                         and _safe_self_rotated_circle(element)))
        )
        if unsafe:
            raise ValueError(
                "unsafe drawable attribute: " + ", ".join(unsafe))
        bbox = _element_bbox(element)
        if bbox is None or bbox.area <= 0:
            raise ValueError(f"unusable bbox for drawable {index + 1}")
        element.set("id", node_id)
        element.set("data-source-layer", source_layer)
        fraction = bbox.area / canvas_area
        paint = _effective_paint(element)
        paint_family, paint_hue = _paint_profile(paint)
        stroke_only = paint.startswith("stroke:")
        # Large fills and canvas-spanning arcs are visual scaffolds.  They stay
        # in paint order but cannot bridge every local object into one group.
        scaffold = (
            fraction >= max(0.20, max_group_bbox_fraction * 2.5)
            or fraction >= max_group_bbox_fraction
            or (stroke_only and bbox.diagonal / canvas_diagonal >= 0.42)
        )
        node = SceneNode(
            element=element, index=index, node_id=node_id,
            source_layer=source_layer, paint=paint, bbox=bbox,
            tag=_local(element.tag), area_fraction=fraction,
            is_scaffold=scaffold, primitive_like=False,
            role="fragment", orientation=math.atan2(bbox.height, bbox.width),
            paint_family=paint_family, paint_hue=paint_hue,
        )
        node.primitive_like = _primitive_like(node)
        aspect = (max(bbox.width, bbox.height)
                  / max(min(bbox.width, bbox.height), 1e-6))
        if stroke_only:
            node.role = "stroke"
        elif node.tag in {"circle", "ellipse"}:
            node.role = "dot"
        elif (node.primitive_like and aspect <= 2.4
              and bbox.diagonal <= 0.035 * canvas.diagonal):
            node.role = "dot"
        nodes.append(node)
    return nodes


def _edge_for(left: SceneNode, right: SceneNode,
              canvas: BBox) -> Edge | None:
    if left.is_scaffold or right.is_scaffold:
        return None
    small = min(left.bbox.area, right.bbox.area)
    large = max(left.bbox.area, right.bbox.area)
    if small <= 0:
        return None
    size_ratio = large / small
    intersection = left.bbox.intersection_area(right.bbox)
    min_coverage = intersection / small
    union_area = left.bbox.area + right.bbox.area - intersection
    iou = intersection / union_area if union_area else 0.0
    different_paint = left.paint != right.paint
    paint_compatible = _paint_compatible(left, right, allow_unknown=True)

    if iou >= 0.68 and size_ratio <= 6.0 and paint_compatible:
        return Edge(left.index, right.index, 0.97, "matching-bounds")
    roles_compatible = (left.role == right.role
                        or {left.role, right.role} <= {"fragment", "stroke"})
    contained_accent = (min_coverage >= 0.88 and size_ratio >= 2.8)
    if (different_paint and roles_compatible and (paint_compatible or contained_accent)
            and min_coverage >= 0.72 and size_ratio <= 6.0):
        score = min(0.96, 0.86 + 0.10 * min_coverage)
        return Edge(left.index, right.index, score, "cross-paint-overlay")
    if (different_paint and roles_compatible and paint_compatible
            and min_coverage >= 0.42 and size_ratio <= 3.5):
        return Edge(left.index, right.index, 0.80 + 0.08 * min_coverage,
                    "cross-paint-overlap")

    gap = left.bbox.gap(right.bbox)
    scale = math.sqrt(small)
    threshold = min(34.0, max(6.0, 2.6 * scale))
    canvas_limit = 0.025 * min(canvas.width, canvas.height)
    threshold = min(threshold, max(8.0, canvas_limit))
    if gap > threshold:
        return None

    comparable = size_ratio <= 5.0
    if left.role == right.role == "dot" and comparable:
        score = 0.74 + 0.10 * (1.0 - gap / max(threshold, 1.0))
        return Edge(left.index, right.index, score, "repeated-dot-proximity")
    if (left.role == right.role == "stroke" and comparable
            and _paint_compatible(left, right)):
        angle_delta = abs(left.orientation - right.orientation)
        angle_delta = min(angle_delta, math.pi - angle_delta)
        if angle_delta <= math.radians(18):
            score = 0.74 + 0.08 * (1.0 - gap / max(threshold, 1.0))
            return Edge(left.index, right.index, score,
                        "parallel-stroke-proximity")
    fragment_scale_ok = min(left.area_fraction, right.area_fraction) >= 0.00025
    if (left.role == right.role == "fragment" and different_paint
            and size_ratio <= 3.2 and fragment_scale_ok
            and _paint_compatible(left, right)):
        score = 0.76 + 0.10 * (1.0 - gap / max(threshold, 1.0))
        return Edge(left.index, right.index, score, "fragment-proximity")
    return None


def _component_bbox(members: Iterable[int], nodes: list[SceneNode]) -> BBox:
    iterator = iter(members)
    bbox = nodes[next(iterator)].bbox
    for member in iterator:
        bbox = bbox.union(nodes[member].bbox)
    return bbox


def _build_candidates(nodes: list[SceneNode], canvas: BBox,
                      confidence_threshold: float,
                      max_group_bbox_fraction: float,
                      max_members: int) -> list[ObjectCandidate]:
    edges: list[Edge] = []
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            edge = _edge_for(nodes[left], nodes[right], canvas)
            if edge and edge.confidence >= confidence_threshold:
                edges.append(edge)
    edges.sort(key=lambda item: (-item.confidence, item.left, item.right))

    union = _UnionFind(len(nodes))
    accepted_edges: list[Edge] = []
    for edge in edges:
        left_root = union.find(edge.left)
        right_root = union.find(edge.right)
        if left_root == right_root:
            accepted_edges.append(edge)
            continue
        members = union.members[left_root] | union.members[right_root]
        if len(members) > max_members:
            continue
        bbox = _component_bbox(sorted(members), nodes)
        if bbox.area / max(canvas.area, 1.0) > max_group_bbox_fraction:
            continue
        # Stop weak chains from spanning a mostly empty region.  Sparse dot
        # clusters are allowed down to 1%; ordinary shapes need 3% bbox mass.
        mass = sum(nodes[index].bbox.area for index in members) / max(bbox.area, 1.0)
        primitives = sum(nodes[index].primitive_like for index in members)
        density_floor = 0.01 if primitives >= max(4, int(0.7 * len(members))) else 0.03
        if mass < density_floor:
            continue
        union.union(left_root, right_root)
        accepted_edges.append(edge)

    by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(nodes)):
        by_root[union.find(index)].append(index)
    candidates: list[ObjectCandidate] = []
    for members in by_root.values():
        if len(members) < 2:
            continue
        paints = {nodes[index].paint for index in members}
        primitive_count = sum(nodes[index].primitive_like for index in members)
        if len(paints) < 2 and (len(members) < 4 or primitive_count < 4):
            continue
        member_set = set(members)
        component_edges = [edge for edge in accepted_edges
                           if edge.left in member_set and edge.right in member_set]
        if not component_edges:
            continue
        confidences = [edge.confidence for edge in component_edges]
        # Median is robust to one weak proximity bridge, while the lower-tail
        # term keeps chain-heavy clusters from claiming certainty.
        confidence = 0.65 * median(confidences) + 0.35 * min(confidences)
        reasons = sorted({edge.reason for edge in component_edges})
        bbox = _component_bbox(members, nodes)
        candidates.append(ObjectCandidate(
            members=sorted(members), confidence=confidence,
            reasons=reasons, bbox=bbox,
        ))
    return sorted(candidates, key=lambda item: (item.members[0], item.members))


def _order_conflict(member: int, target: int, outside: int,
                    nodes: list[SceneNode], pad: float) -> bool:
    if member == target:
        return False
    low, high = sorted((member, target))
    if not (low < outside <= high):
        return False
    # Moving to target flips order only when outside lies between the two.
    return nodes[member].bbox.overlaps(nodes[outside].bbox, pad=pad)


def _choose_anchor(candidate: ObjectCandidate, nodes: list[SceneNode],
                   pad: float) -> int | None:
    members = set(candidate.members)
    outside = [index for index in range(len(nodes))
               if index not in members]
    choices = []
    for target in candidate.members:
        conflicts = sum(
            _order_conflict(member, target, other, nodes, pad)
            for member in candidate.members for other in outside
        )
        movement = sum(abs(member - target) for member in candidate.members)
        choices.append((conflicts, movement, target))
    conflicts, _movement, target = min(choices)
    return target if conflicts == 0 else None


def _stable_group_id(members: Iterable[int], nodes: list[SceneNode]) -> str:
    payload = "\0".join(sorted(nodes[index].node_id for index in members))
    return "object-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _describe_candidate(candidate: ObjectCandidate, nodes: list[SceneNode],
                        canvas: BBox) -> tuple[str, str]:
    """Return a useful spatial label without claiming semantic OCR."""

    reasons = set(candidate.reasons)
    roles = {nodes[index].role for index in candidate.members}
    if "repeated-dot-proximity" in reasons:
        kind, noun = "dot-cluster", "dot cluster"
    elif "parallel-stroke-proximity" in reasons and roles <= {"stroke"}:
        kind, noun = "stroke-cluster", "stroke cluster"
    elif reasons.intersection({"cross-paint-overlay", "matching-bounds",
                               "cross-paint-overlap"}):
        kind, noun = "layered-object", "layered object"
    else:
        kind, noun = "object-cluster", "object cluster"
    center_x, center_y = candidate.bbox.center
    x_ratio = (center_x - canvas.x0) / max(canvas.width, 1.0)
    y_ratio = (center_y - canvas.y0) / max(canvas.height, 1.0)
    horizontal = ("left" if x_ratio < 0.34
                  else "right" if x_ratio > 0.66 else "center")
    vertical = ("top" if y_ratio < 0.34
                else "bottom" if y_ratio > 0.66 else "middle")
    if horizontal == "center" and vertical == "middle":
        position = "center"
    elif horizontal == "center":
        position = vertical
    elif vertical == "middle":
        position = horizontal
    else:
        position = f"{vertical}-{horizontal}"
    label = f"{position.title()} {noun} ({len(candidate.members)} parts)"
    return kind, label


def _candidate_manifest(candidate: ObjectCandidate, nodes: list[SceneNode],
                        mode: str, reason: str = "") -> dict[str, object]:
    paints = sorted({nodes[item].paint for item in candidate.members})
    item: dict[str, object] = {
        "id": candidate.group_id,
        "label": candidate.label,
        "kind": candidate.kind,
        "mode": mode,
        "confidence": round(candidate.confidence, 4),
        "reasons": candidate.reasons,
        "member_count": len(candidate.members),
        "paint_count": len(paints),
        "paints": paints,
        "source_layers": sorted({nodes[index].source_layer
                                 for index in candidate.members}),
        "bbox": candidate.bbox.as_list(),
        "node_ids": [nodes[index].node_id for index in candidate.members],
    }
    if reason:
        item["not_applied_reason"] = reason
    return item


def _new_order(candidates: list[ObjectCandidate], node_count: int) -> list[int]:
    by_anchor = {candidate.anchor: candidate for candidate in candidates}
    grouped = {member for candidate in candidates for member in candidate.members}
    result: list[int] = []
    for index in range(node_count):
        if index in by_anchor:
            result.extend(by_anchor[index].members)
        elif index not in grouped:
            result.append(index)
    return result


def _validate_order(original: list[int], candidate: list[int],
                    nodes: list[SceneNode], pad: float) -> tuple[bool, str]:
    if sorted(original) != sorted(candidate):
        return False, "candidate order lost or duplicated drawables"
    old_position = {node: index for index, node in enumerate(original)}
    new_position = {node: index for index, node in enumerate(candidate)}
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            old = old_position[left] < old_position[right]
            new = new_position[left] < new_position[right]
            if old != new and nodes[left].bbox.overlaps(nodes[right].bbox, pad=pad):
                return False, (
                    f"paint-order inversion between overlapping {nodes[left].node_id} "
                    f"and {nodes[right].node_id}"
                )
    return True, ""


def _drawable_signature(element: ET.Element) -> tuple[str, tuple[tuple[str, str], ...]]:
    ignored = {"id", "data-source-layer", "data-object-id"}
    attrs = tuple(sorted((key, value) for key, value in element.attrib.items()
                         if key not in ignored))
    return _local(element.tag), attrs


def _serialize(root: ET.Element) -> str:
    body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n"


def build_scene_graph(
    svg_text: str,
    *,
    confidence_threshold: float = 0.74,
    max_group_bbox_fraction: float = 0.075,
    max_members: int = 96,
    overlap_safety_pad: float = 0.35,
    validator: Callable[[str, str], bool] | None = None,
) -> SceneGraphResult:
    """Build conservative, selectable object groups.

    ``validator`` can render/compare the two SVG strings in a future pipeline.
    It must return true to accept the candidate.  On every failure path the
    returned ``svg_text`` is exactly the original input.
    """
    base_report: dict[str, object] = {
        "scene_graph_version": "beta3-0.1",
        "confidence_threshold": confidence_threshold,
        "max_group_bbox_fraction": max_group_bbox_fraction,
        "rollback_safe": True,
    }
    try:
        root = ET.fromstring(svg_text)
        if _local(root.tag) != "svg":
            raise ValueError("document root is not SVG")
        if root.get("data-scene-graph-version"):
            return SceneGraphResult(svg_text, "already_processed", {
                **base_report, "reason": "scene graph metadata already present",
            })
        canvas = _viewbox(root)
        flattened, scene_children = _flatten_drawables(root)
        if not flattened:
            return SceneGraphResult(svg_text, "no_change", {
                **base_report, "reason": "no drawable scene content",
            })
        nodes = _make_nodes(flattened, canvas, max_group_bbox_fraction)
        before_signatures = Counter(_drawable_signature(node.element) for node in nodes)
        candidates = _build_candidates(
            nodes, canvas, confidence_threshold,
            max_group_bbox_fraction, max_members,
        )
        skipped_unsafe = []
        applied: list[ObjectCandidate] = []
        reserved: set[int] = set()
        for candidate in candidates:
            candidate.group_id = _stable_group_id(candidate.members, nodes)
            candidate.kind, candidate.label = _describe_candidate(
                candidate, nodes, canvas)
            if reserved.intersection(candidate.members):
                continue
            anchor = _choose_anchor(candidate, nodes, overlap_safety_pad)
            if anchor is None:
                skipped_unsafe.append(_candidate_manifest(
                    candidate, nodes, "manifest-only",
                    "overlapping paint-order inversion",
                ))
                continue
            candidate.anchor = anchor
            applied.append(candidate)
            reserved.update(candidate.members)
        if not applied:
            return SceneGraphResult(svg_text, "no_change", {
                **base_report,
                "drawable_count": len(nodes),
                "candidate_groups": len(candidates),
                "actual_dom_group_count": 0,
                "manifest_only_group_count": len(skipped_unsafe),
                "actual_dom_groups": [],
                "manifest_only_groups": skipped_unsafe,
                "skipped_unsafe_groups": skipped_unsafe,
                "reason": "no candidate could be grouped without paint-order risk",
            })

        order = _new_order(applied, len(nodes))
        safe, reason = _validate_order(
            list(range(len(nodes))), order, nodes, overlap_safety_pad)
        if not safe:
            raise ValueError(reason)

        scene_root = ET.Element(_qname("g"), {
            "id": "scene-graph-root",
            "data-scene-graph-version": "beta3-0.1",
        })
        by_anchor = {candidate.anchor: candidate for candidate in applied}
        grouped = {member for candidate in applied for member in candidate.members}
        group_reports = []
        for index in range(len(nodes)):
            candidate = by_anchor.get(index)
            if candidate is not None:
                group = ET.SubElement(scene_root, _qname("g"), {
                    "id": candidate.group_id,
                    f"{{{INKSCAPE_NS}}}label": candidate.label,
                    "data-group-kind": candidate.kind,
                    "data-group-mode": "actual-dom",
                    "data-group-confidence": f"{candidate.confidence:.3f}",
                    "data-group-reasons": ",".join(candidate.reasons),
                    "data-member-count": str(len(candidate.members)),
                })
                for member in candidate.members:
                    nodes[member].element.set("data-object-id", candidate.group_id)
                    group.append(nodes[member].element)
                group_reports.append(_candidate_manifest(
                    candidate, nodes, "actual-dom"))
            elif index not in grouped:
                scene_root.append(nodes[index].element)

        root_children = list(root)
        positions = [root_children.index(child) for child in scene_children]
        insert_at = min(positions)
        for child in scene_children:
            root.remove(child)
        root.insert(insert_at, scene_root)
        root.set("data-scene-graph-version", "beta3-0.1")

        report = {
            **base_report,
            "drawable_count": len(nodes),
            "object_group_count": len(applied),
            "actual_dom_group_count": len(applied),
            "manifest_only_group_count": len(skipped_unsafe),
            "grouped_drawables": len(grouped),
            "ungrouped_drawables": len(nodes) - len(grouped),
            "scaffold_drawables": sum(node.is_scaffold for node in nodes),
            "candidate_groups": len(candidates),
            "skipped_unsafe_groups": skipped_unsafe,
            "groups": group_reports,
            "actual_dom_groups": group_reports,
            "manifest_only_groups": skipped_unsafe,
            "paint_order_validation": "passed",
            "scope_note": (
                "actual-dom groups are selectable SVG groups; manifest-only groups "
                "are annotations that were not moved because paint order could change. "
                "Neither mode proves semantic correctness or designer time savings."
            ),
        }
        metadata = ET.Element(_qname("metadata"), {"id": "scene-graph-metadata"})
        metadata.text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        root.insert(insert_at, metadata)

        after = [item for item in scene_root.iter()
                 if _local(item.tag) in DRAWABLE_TAGS]
        after_signatures = Counter(_drawable_signature(item) for item in after)
        if before_signatures != after_signatures:
            raise ValueError("drawable geometry/style invariant changed")
        candidate_text = _serialize(root)
        if validator is not None and not validator(svg_text, candidate_text):
            return SceneGraphResult(svg_text, "rolled_back", {
                **base_report, "reason": "external render validator rejected candidate",
                "attempted_object_groups": len(applied),
            })
        return SceneGraphResult(candidate_text, "applied", report)
    except Exception as exc:
        return SceneGraphResult(svg_text, "rolled_back", {
            **base_report,
            "reason": str(exc)[:300],
        })


def process_svg_file(
    source: str | Path,
    destination: str | Path,
    **kwargs: object,
) -> SceneGraphResult:
    """Postprocess a file, writing only a validated applied result.

    A rolled-back/no-change result writes the original SVG, so callers may use
    a temporary destination and atomically replace their deliverable later.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    source_bytes = source_path.read_bytes()
    original = source_bytes.decode("utf-8-sig")
    result = build_scene_graph(original, **kwargs)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if result.status == "applied":
        destination_path.write_bytes(result.svg_text.encode("utf-8"))
    else:
        # Preserve BOM and original newline bytes on every fallback path.
        destination_path.write_bytes(source_bytes)
    return result


__all__ = [
    "BBox", "SceneGraphResult", "build_scene_graph", "process_svg_file",
]
