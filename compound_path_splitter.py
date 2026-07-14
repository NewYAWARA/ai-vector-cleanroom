"""Conservatively split independent families in compound SVG paths.

The splitter is intentionally standalone: importing it does not change the
cleanroom pipeline.  Its only public operation accepts SVG text and returns a
transactional result.  A successful edit is made directly against the source
tags, rather than by serialising an XML tree, so path coordinates, attribute
spelling, whitespace, namespaces, and document order are retained.

Safety model
------------

Each explicit ``M``/``m`` starts one subpath.  Every ordinary SVG path command
is parsed, and a conservative bounding box is built from endpoints and Bezier
control points (or the complete corrected ellipse for an arc).  Subpaths whose
boxes overlap or touch are kept in one family.  This deliberately groups a
root with all of its holes/islands and also refuses to separate any ambiguous
overlap.  Only families for which every pair of member boxes is disjoint are
emitted as separate paths.  The rule is conservative, but the disjointness it
does claim is independent of curve flattening or raster resolution.

Eligible fill-only compound paths also receive one exact simplification:
an explicitly commanded absolute cubic is changed to a line only when exact
rational arithmetic proves that both controls occur monotonically on the
closed segment between its endpoints.  The locus and fill boundary are then
identical; no flattening tolerance or raster-resolution assumption is used.

Transforms, references to a candidate path, live CSS/script/animation,
visible strokes, object-bounding-box paint servers, malformed/unsupported
geometry, and failed invariants cause a byte-exact rollback.  An optional
validator receives ``(original_svg, candidate_svg)`` and can also veto the
entire transaction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
import re
from typing import Callable, Iterable
import xml.etree.ElementTree as ET


Validator = Callable[[str, str], bool]

_NUMBER = r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?"
_TOKEN_RE = re.compile(rf"(?P<command>[AaCcHhLlMmQqSsTtVvZz])|(?P<number>{_NUMBER})")
_SEPARATOR_RE = re.compile(r"[\x20\t\r\n\f,]*\Z")
_ATTR_NAME_CHARS = r"A-Za-z0-9_.:-"
_D_ATTR_RE = re.compile(
    rf"(?<![{_ATTR_NAME_CHARS}])d\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)
_ID_ATTR_RE = re.compile(
    rf"(?<![{_ATTR_NAME_CHARS}])id\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)
_COMMAND_ARITY = {
    "M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
    "Q": 4, "T": 2, "A": 7,
}
_NON_RENDER_CONTAINERS = {
    "defs", "clipPath", "mask", "marker", "pattern", "symbol",
    "linearGradient", "radialGradient", "filter",
}
_UNSAFE_ATTRIBUTES = {
    "transform", "clip-path", "mask", "filter", "marker-start",
    "marker-mid", "marker-end", "mix-blend-mode", "isolation",
    "pathLength",
}
_UNSAFE_STYLE_PROPERTIES = {
    "transform", "clip-path", "mask", "filter", "marker-start",
    "marker-mid", "marker-end", "mix-blend-mode", "isolation",
}
_ACTIVE_TAGS = {
    "script", "style", "animate", "animateMotion", "animateTransform",
    "set", "discard",
}
_REFERENCE_ATTRIBUTES = {
    "href", "begin", "end", "aria-labelledby", "aria-describedby",
    "for",
}


class _UnsafeSplit(ValueError):
    """A fail-closed condition that rolls the whole transaction back."""


@dataclass(frozen=True)
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def overlaps(self, other: "BBox", epsilon: float = 0.0) -> bool:
        return not (
            self.x1 + epsilon < other.x0
            or other.x1 + epsilon < self.x0
            or self.y1 + epsilon < other.y0
            or other.y1 + epsilon < self.y0
        )

    def union(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x0, other.x0), min(self.y0, other.y0),
            max(self.x1, other.x1), max(self.y1, other.y1),
        )

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _Subpath:
    index: int
    start: int
    end: int
    core: str
    bbox: BBox
    drawn: bool


@dataclass(frozen=True)
class _RawPathTag:
    start: int
    end: int
    text: str
    self_closing: bool


@dataclass
class _SplitPlan:
    path_index: int
    raw_tag: _RawPathTag
    subpaths: list[_Subpath]
    families: list[list[int]]
    family_data: list[str]
    family_ids: list[str]
    family_boxes: list[BBox]
    replacement: str
    linear_cubics_simplified: int = 0
    path_data_bytes_saved: int = 0


@dataclass
class CompoundPathSplitResult:
    svg_text: str
    status: str
    report: dict[str, object] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.status == "applied"


# Short public alias for callers that prefer the API name from the proposal.
Result = CompoundPathSplitResult


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left != right:
            self.parent[right] = left


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _find_tag_end(text: str, start: int) -> int:
    quote = ""
    index = start
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                quote = ""
        elif char in {'"', "'"}:
            quote = char
        elif char == ">":
            return index + 1
        index += 1
    raise _UnsafeSplit("unterminated XML tag")


def _scan_path_tags(text: str) -> list[_RawPathTag]:
    """Find source path start tags while ignoring comments/CDATA/PIs."""
    found: list[_RawPathTag] = []
    index = 0
    while True:
        start = text.find("<", index)
        if start < 0:
            break
        if text.startswith("<!--", start):
            end = text.find("-->", start + 4)
            if end < 0:
                raise _UnsafeSplit("unterminated XML comment")
            index = end + 3
            continue
        if text.startswith("<![CDATA[", start):
            end = text.find("]]>", start + 9)
            if end < 0:
                raise _UnsafeSplit("unterminated CDATA section")
            index = end + 3
            continue
        if text.startswith("<?", start):
            end = text.find("?>", start + 2)
            if end < 0:
                raise _UnsafeSplit("unterminated processing instruction")
            index = end + 2
            continue
        if text.startswith("<!", start):
            # Entity-bearing doctypes make a raw d-value mapping ambiguous.
            raise _UnsafeSplit("DOCTYPE/declaration is unsupported")
        end = _find_tag_end(text, start + 1)
        raw = text[start:end]
        match = re.match(r"<\s*(/?)\s*([^\s/>]+)", raw)
        if match and not match.group(1) and _local(match.group(2)) == "path":
            found.append(_RawPathTag(
                start=start,
                end=end,
                text=raw,
                self_closing=bool(re.search(r"/\s*>\Z", raw)),
            ))
        index = end
    return found


def _lex_path(data: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    for match in _TOKEN_RE.finditer(data):
        gap = data[position:match.start()]
        if not _SEPARATOR_RE.fullmatch(gap) or gap.count(",") > 1:
            raise _UnsafeSplit("unsupported or malformed path-data token")
        kind = "command" if match.group("command") else "number"
        tokens.append(_Token(kind, match.group(0), match.start(), match.end()))
        position = match.end()
    tail = data[position:]
    if not _SEPARATOR_RE.fullmatch(tail) or tail.count(",") > 1:
        raise _UnsafeSplit("unsupported or malformed path-data tail")
    if not tokens:
        raise _UnsafeSplit("empty path data")
    return tokens


class _BBoxBuilder:
    def __init__(self) -> None:
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.drawn = False

    def point(self, x: float, y: float) -> None:
        if not math.isfinite(x) or not math.isfinite(y):
            raise _UnsafeSplit("non-finite path geometry")
        self.xs.append(x)
        self.ys.append(y)

    def box(self, box: BBox) -> None:
        self.point(box.x0, box.y0)
        self.point(box.x1, box.y1)

    def finish(self) -> BBox:
        if not self.xs:
            raise _UnsafeSplit("subpath has no geometry")
        return BBox(min(self.xs), min(self.ys), max(self.xs), max(self.ys))


def _arc_box(
    start: tuple[float, float], end: tuple[float, float],
    rx_value: float, ry_value: float, rotation: float,
    large: float, sweep: float,
) -> BBox:
    """Conservative full-ellipse box after SVG endpoint correction."""
    x1, y1 = start
    x2, y2 = end
    rx = abs(rx_value)
    ry = abs(ry_value)
    if rx == 0.0 or ry == 0.0 or start == end:
        return BBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if large not in {0.0, 1.0} or sweep not in {0.0, 1.0}:
        raise _UnsafeSplit("arc flags must be 0 or 1")
    phi = math.radians(rotation % 360.0)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    dx = (x1 - x2) / 2.0
    dy = (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    correction = x1p * x1p / (rx * rx) + y1p * y1p / (ry * ry)
    if correction > 1.0:
        scale = math.sqrt(correction)
        rx *= scale
        ry *= scale
    numerator = (
        rx * rx * ry * ry
        - rx * rx * y1p * y1p
        - ry * ry * x1p * x1p
    )
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    sign = -1.0 if large == sweep else 1.0
    coefficient = 0.0 if denominator == 0.0 else sign * math.sqrt(
        max(0.0, numerator / denominator)
    )
    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0
    x_extent = math.hypot(rx * cos_phi, ry * sin_phi)
    y_extent = math.hypot(rx * sin_phi, ry * cos_phi)
    values = (cx, cy, x_extent, y_extent)
    if not all(math.isfinite(value) for value in values):
        raise _UnsafeSplit("non-finite arc geometry")
    return BBox(cx - x_extent, cy - y_extent,
                cx + x_extent, cy + y_extent)


def _subpath_boxes(data: str) -> list[_Subpath]:
    tokens = _lex_path(data)
    move_positions = [
        index for index, token in enumerate(tokens)
        if token.kind == "command" and token.text.upper() == "M"
    ]
    if not move_positions or move_positions[0] != 0:
        raise _UnsafeSplit("path data must start with moveto")

    # Parse the complete path once.  State intentionally crosses explicit
    # relative moveto boundaries, as required by the SVG grammar.
    builders: list[_BBoxBuilder] = []
    subpath_token_ranges: list[tuple[int, int]] = []
    current = (0.0, 0.0)
    sub_start = current
    previous_command = ""
    last_cubic: tuple[float, float] | None = None
    last_quad: tuple[float, float] | None = None
    token_index = 0
    active_builder: _BBoxBuilder | None = None

    while token_index < len(tokens):
        command_token_index = token_index
        token = tokens[token_index]
        if token.kind != "command":
            raise _UnsafeSplit("implicit command without a preceding command")
        command = token.text
        upper = command.upper()
        relative = command.islower()
        token_index += 1
        if upper == "M":
            if active_builder is not None:
                subpath_token_ranges[-1] = (
                    subpath_token_ranges[-1][0], command_token_index,
                )
            active_builder = _BBoxBuilder()
            builders.append(active_builder)
            subpath_token_ranges.append((command_token_index, len(tokens)))
        elif active_builder is None:
            raise _UnsafeSplit("path data must start with moveto")

        if upper == "Z":
            assert active_builder is not None
            active_builder.point(*sub_start)
            active_builder.drawn = True
            current = sub_start
            previous_command = command
            last_cubic = last_quad = None
            # No numbers may implicitly follow closepath.
            if token_index < len(tokens) and tokens[token_index].kind != "command":
                raise _UnsafeSplit("parameters cannot follow closepath")
            continue
        if upper not in _COMMAND_ARITY:
            raise _UnsafeSplit(f"unsupported path command {command!r}")

        arity = _COMMAND_ARITY[upper]
        values_tokens: list[_Token] = []
        while token_index < len(tokens) and tokens[token_index].kind == "number":
            values_tokens.append(tokens[token_index])
            token_index += 1
        if not values_tokens or len(values_tokens) % arity:
            raise _UnsafeSplit(f"wrong parameter count for {command!r}")
        try:
            values_all = [float(item.text) for item in values_tokens]
        except ValueError as exc:
            raise _UnsafeSplit("invalid numeric path parameter") from exc
        if not all(math.isfinite(value) for value in values_all):
            raise _UnsafeSplit("non-finite path parameter")

        first_set = True
        for offset in range(0, len(values_all), arity):
            values = values_all[offset:offset + arity]
            parameter_tokens = values_tokens[offset:offset + arity]
            effective_upper = upper
            if upper == "M" and not first_set:
                effective_upper = "L"
            ox, oy = current
            assert active_builder is not None

            def absolute_pair(x: float, y: float) -> tuple[float, float]:
                return (x + ox, y + oy) if relative else (x, y)

            if effective_upper == "M":
                current = absolute_pair(values[0], values[1])
                sub_start = current
                active_builder.point(*current)
            elif effective_upper == "L":
                current = absolute_pair(values[0], values[1])
                active_builder.point(ox, oy)
                active_builder.point(*current)
                active_builder.drawn = True
            elif effective_upper == "H":
                x = values[0] + (ox if relative else 0.0)
                current = (x, oy)
                active_builder.point(ox, oy)
                active_builder.point(*current)
                active_builder.drawn = True
            elif effective_upper == "V":
                y = values[0] + (oy if relative else 0.0)
                current = (ox, y)
                active_builder.point(ox, oy)
                active_builder.point(*current)
                active_builder.drawn = True
            elif effective_upper == "C":
                c1 = absolute_pair(values[0], values[1])
                c2 = absolute_pair(values[2], values[3])
                end = absolute_pair(values[4], values[5])
                for point in ((ox, oy), c1, c2, end):
                    active_builder.point(*point)
                current = end
                last_cubic = c2
                last_quad = None
                active_builder.drawn = True
            elif effective_upper == "S":
                if previous_command.upper() in {"C", "S"} and last_cubic:
                    c1 = (2.0 * ox - last_cubic[0],
                          2.0 * oy - last_cubic[1])
                else:
                    c1 = (ox, oy)
                c2 = absolute_pair(values[0], values[1])
                end = absolute_pair(values[2], values[3])
                for point in ((ox, oy), c1, c2, end):
                    active_builder.point(*point)
                current = end
                last_cubic = c2
                last_quad = None
                active_builder.drawn = True
            elif effective_upper == "Q":
                control = absolute_pair(values[0], values[1])
                end = absolute_pair(values[2], values[3])
                for point in ((ox, oy), control, end):
                    active_builder.point(*point)
                current = end
                last_quad = control
                last_cubic = None
                active_builder.drawn = True
            elif effective_upper == "T":
                if previous_command.upper() in {"Q", "T"} and last_quad:
                    control = (2.0 * ox - last_quad[0],
                               2.0 * oy - last_quad[1])
                else:
                    control = (ox, oy)
                end = absolute_pair(values[0], values[1])
                for point in ((ox, oy), control, end):
                    active_builder.point(*point)
                current = end
                last_quad = control
                last_cubic = None
                active_builder.drawn = True
            elif effective_upper == "A":
                if (
                    parameter_tokens[3].text not in {"0", "1"}
                    or parameter_tokens[4].text not in {"0", "1"}
                ):
                    raise _UnsafeSplit("arc flags must be literal 0 or 1")
                end = absolute_pair(values[5], values[6])
                active_builder.box(_arc_box(
                    (ox, oy), end, values[0], values[1], values[2],
                    values[3], values[4],
                ))
                current = end
                last_cubic = last_quad = None
                active_builder.drawn = True
            if effective_upper not in {"C", "S", "Q", "T"}:
                last_cubic = last_quad = None
            previous_command = (
                ("l" if relative else "L")
                if upper == "M" and not first_set else command
            )
            first_set = False

    if active_builder is not None:
        subpath_token_ranges[-1] = (subpath_token_ranges[-1][0], len(tokens))
    if len(builders) != len(subpath_token_ranges):
        raise _UnsafeSplit("subpath parser invariant failed")

    subpaths: list[_Subpath] = []
    for index, (builder, token_range) in enumerate(
        zip(builders, subpath_token_ranges)
    ):
        first_token = tokens[token_range[0]]
        last_token = tokens[token_range[1] - 1]
        start = first_token.start
        end = last_token.end
        subpaths.append(_Subpath(
            index=index,
            start=start,
            end=end,
            core=data[start:end],
            bbox=builder.finish(),
            drawn=builder.drawn,
        ))
    return subpaths


def _style_declarations(value: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for item in value.split(";"):
        if not item.strip():
            continue
        if ":" not in item:
            raise _UnsafeSplit("malformed inline style")
        name, setting = item.split(":", 1)
        key = name.strip().lower()
        if not re.fullmatch(r"[-a-zA-Z]+", key):
            raise _UnsafeSplit("unsupported inline style property")
        declarations[key] = setting.strip()
    return declarations


def _ancestor_chain(
    element: ET.Element, parent_map: dict[ET.Element, ET.Element]
) -> list[ET.Element]:
    chain = [element]
    while element in parent_map:
        element = parent_map[element]
        chain.append(element)
    return chain


def _effective_property(
    element: ET.Element, parent_map: dict[ET.Element, ET.Element],
    name: str, default: str,
) -> str:
    for node in _ancestor_chain(element, parent_map):
        style = node.get("style")
        if style:
            declarations = _style_declarations(style)
            if name in declarations:
                return declarations[name]
        if name in node.attrib:
            return node.attrib[name]
    return default


def _check_candidate_context(
    element: ET.Element, parent_map: dict[ET.Element, ET.Element]
) -> None:
    for node in _ancestor_chain(element, parent_map):
        for attribute in node.attrib:
            local = _local(attribute)
            if local in _UNSAFE_ATTRIBUTES or local.lower().startswith("on"):
                raise _UnsafeSplit(f"unsafe attribute {local!r}")
        style = node.get("style")
        if style:
            declarations = _style_declarations(style)
            unsafe = sorted(set(declarations) & _UNSAFE_STYLE_PROPERTIES)
            if unsafe:
                raise _UnsafeSplit(f"unsafe style property {unsafe[0]!r}")


def _in_non_render_container(
    element: ET.Element, parent_map: dict[ET.Element, ET.Element]
) -> bool:
    return any(
        _local(node.tag) in _NON_RENDER_CONTAINERS
        for node in _ancestor_chain(element, parent_map)[1:]
    )


def _is_safe_paint_server(value: str, id_map: dict[str, ET.Element]) -> bool:
    match = re.fullmatch(r"url\(\s*#([^\s)]+)\s*\)", value.strip())
    if not match:
        return not re.search(r"(?:url|var)\s*\(", value, re.IGNORECASE)
    target = id_map.get(match.group(1))
    if target is None or _local(target.tag) not in {"linearGradient", "radialGradient"}:
        return False
    if target.get("gradientUnits", "objectBoundingBox") != "userSpaceOnUse":
        return False
    if any(_local(key) in {"gradientTransform", "href"}
           for key in target.attrib):
        return False
    return True


def _path_is_referenced(root: ET.Element, path: ET.Element) -> bool:
    node_id = path.get("id")
    if not node_id:
        return False
    hash_reference = re.compile(
        rf"#\s*{re.escape(node_id)}(?=$|[\s)\]}}>'\";,])"
    )
    word_reference = re.compile(rf"(?:^|\s){re.escape(node_id)}(?:\s|$)")
    for node in root.iter():
        for attribute, value in node.attrib.items():
            if node is path and _local(attribute) == "id":
                continue
            if hash_reference.search(value):
                return True
            if _local(attribute) in _REFERENCE_ATTRIBUTES and (
                word_reference.search(value) or value.startswith(node_id + ".")
            ):
                return True
    return False


def _families(subpaths: list[_Subpath]) -> tuple[list[list[int]], float]:
    if not subpaths:
        raise _UnsafeSplit("compound path has no subpaths")
    coordinates = [value for item in subpaths for value in item.bbox.as_list()]
    scale = max(1.0, max(abs(value) for value in coordinates))
    epsilon = max(1e-12, scale * 1e-12)
    union_find = _UnionFind(len(subpaths))
    for left in range(len(subpaths)):
        for right in range(left + 1, len(subpaths)):
            if subpaths[left].bbox.overlaps(subpaths[right].bbox, epsilon):
                union_find.union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(subpaths)):
        grouped.setdefault(union_find.find(index), []).append(index)
    families = sorted(grouped.values(), key=lambda members: min(members))

    # Geometry invariant: no conservative member box may overlap a member of
    # another emitted family.  This restates the proof at the output boundary.
    for family_index, family in enumerate(families):
        for other in families[family_index + 1:]:
            if any(subpaths[a].bbox.overlaps(subpaths[b].bbox, epsilon)
                   for a in family for b in other):
                raise _UnsafeSplit("cross-family geometry invariant failed")
    return families, epsilon


def _family_data(data: str, subpaths: list[_Subpath], members: list[int]) -> str:
    pieces: list[str] = []
    for position, member in enumerate(members):
        item = subpaths[member]
        if position:
            previous_member = members[position - 1]
            if member == previous_member + 1:
                separator = data[subpaths[previous_member].end:item.start]
                if not separator or "," in separator:
                    separator = " "
            else:
                # The geometry core stays byte-exact; only the separator for
                # formerly non-adjacent source subpaths is canonicalised.
                separator = " "
            pieces.append(separator)
        elif member == 0:
            pieces.append(data[:item.start])
        pieces.append(item.core)
    if members and members[-1] == len(subpaths) - 1:
        pieces.append(data[subpaths[-1].end:])
    return "".join(pieces)


def _simplify_exact_linear_cubics(data: str) -> tuple[str, int, int]:
    """Replace provably straight absolute cubics with equivalent lines.

    This deliberately recognises only the narrow grammar emitted by the fill
    tracer: every command is explicit and uppercase, with one parameter set,
    and the command set is limited to ``M``, ``C``, ``A`` and ``Z``.  Anything
    outside that grammar is returned byte-for-byte unchanged.  ``Fraction``
    arithmetic makes the collinearity and ordering proof exact for the source
    decimal tokens; the token-size limits keep adversarial numeric input from
    causing unbounded integer construction.
    """

    try:
        tokens = _lex_path(data)
    except _UnsafeSplit:
        return data, 0, 0

    def exact_number(token: _Token) -> Fraction | None:
        text = token.text
        # All ordinary SVG coordinates are far below these limits.  Refusing
        # unusually long/exponent-heavy values is safer than approximating.
        if len(text) > 64:
            return None
        exponent = re.search(r"[eE]([-+]?\d+)\Z", text)
        if exponent and abs(int(exponent.group(1))) > 308:
            return None
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError, OverflowError):
            return None

    zero = Fraction(0)
    current = (zero, zero)
    subpath_start = current
    replacements: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        command_token = tokens[index]
        if command_token.kind != "command":
            return data, 0, 0
        command = command_token.text
        if command not in {"M", "C", "A", "Z"}:
            return data, 0, 0
        index += 1
        if command == "Z":
            if index < len(tokens) and tokens[index].kind != "command":
                return data, 0, 0
            current = subpath_start
            continue

        arity = _COMMAND_ARITY[command]
        values = tokens[index:index + arity]
        if (
            len(values) != arity
            or any(token.kind != "number" for token in values)
            or (
                index + arity < len(tokens)
                and tokens[index + arity].kind != "command"
            )
        ):
            return data, 0, 0
        exact_values = [exact_number(token) for token in values]
        if any(value is None for value in exact_values):
            return data, 0, 0
        numbers = [value for value in exact_values if value is not None]
        index += arity

        if command == "M":
            current = (numbers[0], numbers[1])
            subpath_start = current
            continue
        if command == "A":
            current = (numbers[5], numbers[6])
            continue

        start_x, start_y = current
        control_1 = (numbers[0], numbers[1])
        control_2 = (numbers[2], numbers[3])
        end = (numbers[4], numbers[5])
        delta_x = end[0] - start_x
        delta_y = end[1] - start_y
        squared_length = delta_x * delta_x + delta_y * delta_y
        cross_1 = (
            delta_x * (control_1[1] - start_y)
            - delta_y * (control_1[0] - start_x)
        )
        cross_2 = (
            delta_x * (control_2[1] - start_y)
            - delta_y * (control_2[0] - start_x)
        )
        projection_1 = (
            delta_x * (control_1[0] - start_x)
            + delta_y * (control_1[1] - start_y)
        )
        projection_2 = (
            delta_x * (control_2[0] - start_x)
            + delta_y * (control_2[1] - start_y)
        )
        if (
            squared_length > 0
            and cross_1 == 0
            and cross_2 == 0
            and 0 <= projection_1 <= projection_2 <= squared_length
        ):
            endpoint_text = data[values[4].start:values[5].end]
            replacements.append((
                command_token.start,
                values[-1].end,
                "L" + endpoint_text,
            ))
        current = end

    if not replacements:
        return data, 0, 0
    simplified = data
    for start, end, replacement in reversed(replacements):
        simplified = simplified[:start] + replacement + simplified[end:]
    return simplified, len(replacements), len(data) - len(simplified)


def _replace_match_value(text: str, match: re.Match[str], value: str) -> str:
    return text[:match.start("value")] + value + text[match.end("value"):]


def _set_id(raw_tag: str, node_id: str) -> str:
    match = _ID_ATTR_RE.search(raw_tag)
    if match:
        return _replace_match_value(raw_tag, match, node_id)
    closing = re.search(
        r"(?P<before>\s*)/(?P<after>\s*)>\Z", raw_tag
    )
    if not closing:
        raise _UnsafeSplit("only self-closing path tags can be split")
    return (
        raw_tag[:closing.start()]
        + f' id="{node_id}"'
        + closing.group("before")
        + "/"
        + closing.group("after")
        + ">"
    )


def _set_d(raw_tag: str, data: str) -> str:
    match = _D_ATTR_RE.search(raw_tag)
    if not match:
        raise _UnsafeSplit("path tag has no raw d attribute")
    return _replace_match_value(raw_tag, match, data)


def _sibling_separator(svg_text: str, raw_tag: _RawPathTag) -> str:
    line_start = svg_text.rfind("\n", 0, raw_tag.start)
    prefix = svg_text[line_start + 1:raw_tag.start]
    if prefix and not prefix.isspace():
        return ""
    if line_start >= 0:
        return "\n" + prefix
    return ""


def _reserve_id(preferred: str, used_ids: set[str], seed: str) -> str:
    candidate = preferred
    attempt = 0
    while candidate in used_ids:
        attempt += 1
        suffix = hashlib.sha256(f"{seed}:{attempt}".encode()).hexdigest()[:8]
        candidate = f"{preferred}-{suffix}"
    used_ids.add(candidate)
    return candidate


def _family_union_box(subpaths: list[_Subpath], members: Iterable[int]) -> BBox:
    iterator = iter(members)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise _UnsafeSplit("empty containment family") from exc
    box = subpaths[first].bbox
    for member in iterator:
        box = box.union(subpaths[member].bbox)
    return box


def _validate_output(
    original_root: ET.Element, candidate_text: str,
    original_path_count: int, added_paths: int,
    original_subpath_count: int,
) -> None:
    try:
        candidate_root = ET.fromstring(candidate_text)
    except ET.ParseError as exc:
        raise _UnsafeSplit("candidate XML is not well formed") from exc
    output_paths = [
        node for node in candidate_root.iter() if _local(node.tag) == "path"
    ]
    if len(output_paths) != original_path_count + added_paths:
        raise _UnsafeSplit("path-count invariant failed")
    output_ids = [
        node.get("id") for node in candidate_root.iter() if node.get("id")
    ]
    if len(output_ids) != len(set(output_ids)):
        raise _UnsafeSplit("output contains duplicate IDs")
    output_subpaths = 0
    for node in output_paths:
        data = node.get("d")
        if data:
            output_subpaths += len(_subpath_boxes(data))
    if output_subpaths != original_subpath_count:
        raise _UnsafeSplit("subpath-count invariant failed")
    # Ensure the document root itself did not change identity through editing.
    if _local(original_root.tag) != _local(candidate_root.tag):
        raise _UnsafeSplit("root-element invariant failed")


def process_compound_paths(
    svg_text: str,
    validator: Validator | None = None,
) -> CompoundPathSplitResult:
    """Split/simplify provably safe compound paths transactionally.

    ``validator``, when supplied, is called only after all internal invariants
    pass.  Returning a false value or raising an exception yields the original
    ``svg_text`` object unchanged with status ``"rolled_back"``.
    """
    base_report: dict[str, object] = {
        "version": "compound-split-0.2",
        "proof": (
            "disjoint conservative command/control-point bounding boxes; "
            "exact rational monotone-collinear cubic identity"
        ),
        "id_policy": "source-order and geometry digest; deterministic",
    }
    try:
        if not isinstance(svg_text, str):
            raise _UnsafeSplit("svg_text must be a string")
        if re.search(r"<\?xml-stylesheet\b", svg_text, re.IGNORECASE):
            raise _UnsafeSplit("external stylesheet processing is unsupported")
        try:
            root = ET.fromstring(svg_text)
        except ET.ParseError as exc:
            raise _UnsafeSplit("source SVG is not well formed") from exc
        if _local(root.tag) != "svg":
            raise _UnsafeSplit("document root is not svg")

        all_nodes = list(root.iter())
        paths = [node for node in all_nodes if _local(node.tag) == "path"]
        raw_tags = _scan_path_tags(svg_text)
        if len(raw_tags) != len(paths):
            raise _UnsafeSplit("raw/XML path mapping invariant failed")
        parent_map = {child: parent for parent in all_nodes for child in parent}
        id_values = [node.get("id") for node in all_nodes if node.get("id")]
        if len(id_values) != len(set(id_values)):
            raise _UnsafeSplit("source contains duplicate IDs")
        used_ids = set(id_values)
        id_map = {node.get("id"): node for node in all_nodes if node.get("id")}

        active_tags = sorted({_local(node.tag) for node in all_nodes} & _ACTIVE_TAGS)
        all_subpath_count = 0
        compound_path_count = 0
        eligible_path_count = 0
        overlap_locked = 0
        plans: list[_SplitPlan] = []

        for path_index, (element, raw_tag) in enumerate(zip(paths, raw_tags)):
            data = element.get("d")
            if not data:
                continue
            # Counting is strict as well: malformed path data is only fatal if
            # it is a multi-subpath fill candidate, not an unrelated one-path
            # stroke emitted elsewhere in the SVG.
            explicit_moves = len(re.findall(r"[Mm]", data))
            if explicit_moves < 2:
                all_subpath_count += max(1, explicit_moves)
                continue
            compound_path_count += 1
            if _in_non_render_container(element, parent_map):
                all_subpath_count += explicit_moves
                continue
            fill = _effective_property(element, parent_map, "fill", "black").strip()
            display = _effective_property(
                element, parent_map, "display", "inline"
            ).strip().lower()
            visibility = _effective_property(
                element, parent_map, "visibility", "visible"
            ).strip().lower()
            if (
                fill.lower() in {"", "none", "transparent"}
                or display == "none"
                or visibility in {"hidden", "collapse"}
            ):
                all_subpath_count += explicit_moves
                continue
            eligible_path_count += 1
            if active_tags:
                raise _UnsafeSplit(
                    f"active CSS/script/animation is unsupported ({active_tags[0]})"
                )
            if not raw_tag.self_closing:
                raise _UnsafeSplit("only self-closing path tags can be split")
            d_match = _D_ATTR_RE.search(raw_tag.text)
            if (
                not d_match
                or "&" in d_match.group("value")
                or d_match.group("value") != data
            ):
                raise _UnsafeSplit("raw d attribute cannot be mapped byte-exactly")
            _check_candidate_context(element, parent_map)
            stroke = _effective_property(
                element, parent_map, "stroke", "none"
            ).strip()
            if stroke.lower() not in {"", "none", "transparent"}:
                raise _UnsafeSplit("visible-stroke compound paths are unsupported")
            fill_rule = _effective_property(
                element, parent_map, "fill-rule", "nonzero"
            ).strip().lower()
            if fill_rule not in {"nonzero", "evenodd"}:
                raise _UnsafeSplit("unsupported fill-rule")
            if not _is_safe_paint_server(fill, id_map):
                raise _UnsafeSplit("unsafe or object-bounding-box fill paint server")
            if _path_is_referenced(root, element):
                raise _UnsafeSplit("candidate path is referenced")

            subpaths = _subpath_boxes(data)
            all_subpath_count += len(subpaths)
            if len(subpaths) != explicit_moves:
                raise _UnsafeSplit("explicit-moveto/subpath invariant failed")
            families, _epsilon = _families(subpaths)
            if len(families) < 2:
                overlap_locked += 1

            raw_family_data = [
                _family_data(data, subpaths, members) for members in families
            ]
            simplified = [
                _simplify_exact_linear_cubics(item)
                for item in raw_family_data
            ]
            family_data = [item[0] for item in simplified]
            linear_cubics_simplified = sum(item[1] for item in simplified)
            path_data_bytes_saved = sum(item[2] for item in simplified)
            if len(families) < 2 and not linear_cubics_simplified:
                continue
            if sum(len(members) for members in families) != len(subpaths):
                raise _UnsafeSplit("family partition invariant failed")
            members_seen = Counter(
                member for family in families for member in family
            )
            if members_seen != Counter(range(len(subpaths))):
                raise _UnsafeSplit("family membership invariant failed")
            for new_data, members in zip(family_data, families):
                reparsed = _subpath_boxes(new_data)
                expected_cores = [
                    _simplify_exact_linear_cubics(subpaths[index].core)[0]
                    for index in members
                ]
                if [item.core for item in reparsed] != expected_cores:
                    raise _UnsafeSplit("path-data preservation invariant failed")

            source_digest = hashlib.sha256(
                f"{path_index}\0{data}".encode("utf-8")
            ).hexdigest()
            original_id = element.get("id")
            stable_base = f"compound-path-{path_index + 1}-{source_digest[:10]}"
            family_ids: list[str] = []
            for family_index, (members, new_data) in enumerate(
                zip(families, family_data)
            ):
                family_digest = hashlib.sha256(
                    new_data.encode("utf-8")
                ).hexdigest()[:8]
                if family_index == 0 and original_id:
                    family_ids.append(original_id)
                    continue
                if family_index == 0:
                    preferred = stable_base
                else:
                    preferred = f"{stable_base}--part-{family_index + 1}-{family_digest}"
                family_ids.append(_reserve_id(
                    preferred, used_ids,
                    f"{source_digest}:{family_index}:{members}",
                ))

            clones = [
                _set_id(_set_d(raw_tag.text, new_data), node_id)
                for new_data, node_id in zip(family_data, family_ids)
            ]
            replacement = _sibling_separator(svg_text, raw_tag).join(clones)
            plans.append(_SplitPlan(
                path_index=path_index,
                raw_tag=raw_tag,
                subpaths=subpaths,
                families=families,
                family_data=family_data,
                family_ids=family_ids,
                family_boxes=[
                    _family_union_box(subpaths, members) for members in families
                ],
                replacement=replacement,
                linear_cubics_simplified=linear_cubics_simplified,
                path_data_bytes_saved=path_data_bytes_saved,
            ))

        # Complete the honest input subpath count for all one-path nodes and
        # skipped defs/strokes.  Strict parsing here would make unrelated
        # vendor extensions block an otherwise safe transaction.
        if not plans:
            return CompoundPathSplitResult(svg_text, "no_change", {
                **base_report,
                "input_paths": len(paths),
                "output_paths": len(paths),
                "input_subpaths": all_subpath_count,
                "output_subpaths": all_subpath_count,
                "compound_paths": compound_path_count,
                "eligible_compound_paths": eligible_path_count,
                "source_paths_split": 0,
                "split_paths": 0,
                "new_paths_added": 0,
                "selectable_path_delta": 0,
                "source_paths_simplified": 0,
                "linear_cubics_simplified": 0,
                "path_data_bytes_saved": 0,
                "overlap_locked_paths": overlap_locked,
            })

        candidate_text = svg_text
        for plan in sorted(plans, key=lambda item: item.raw_tag.start, reverse=True):
            candidate_text = (
                candidate_text[:plan.raw_tag.start]
                + plan.replacement
                + candidate_text[plan.raw_tag.end:]
            )
        new_paths_added = sum(len(plan.families) - 1 for plan in plans)
        # Count every source path strictly for the postcondition.  A malformed
        # one-path node is an invariant failure once we are about to commit.
        strict_input_subpaths = sum(
            len(_subpath_boxes(path.get("d")))
            for path in paths if path.get("d")
        )
        _validate_output(
            root, candidate_text, len(paths), new_paths_added,
            strict_input_subpaths,
        )

        split_plans = [plan for plan in plans if len(plan.families) > 1]
        simplified_plans = [
            plan for plan in plans if plan.linear_cubics_simplified
        ]

        report: dict[str, object] = {
            **base_report,
            "input_paths": len(paths),
            "output_paths": len(paths) + new_paths_added,
            "input_subpaths": strict_input_subpaths,
            "output_subpaths": strict_input_subpaths,
            "compound_paths": compound_path_count,
            "eligible_compound_paths": eligible_path_count,
            "source_paths_split": len(split_plans),
            "split_paths": sum(len(plan.families) for plan in split_plans),
            "new_paths_added": new_paths_added,
            "selectable_path_delta": new_paths_added,
            "subpaths_redistributed": sum(
                len(plan.subpaths) for plan in split_plans
            ),
            "source_paths_simplified": len(simplified_plans),
            "linear_cubics_simplified": sum(
                plan.linear_cubics_simplified for plan in simplified_plans
            ),
            "path_data_bytes_saved": sum(
                plan.path_data_bytes_saved for plan in simplified_plans
            ),
            "overlap_locked_paths": overlap_locked,
            "paths": [
                {
                    "source_path_index": plan.path_index,
                    "source_subpaths": len(plan.subpaths),
                    "family_count": len(plan.families),
                    "families": [
                        {
                            "id": node_id,
                            "subpath_indices": members,
                            "subpath_count": len(members),
                            "bbox": box.as_list(),
                        }
                        for node_id, members, box in zip(
                            plan.family_ids, plan.families, plan.family_boxes
                        )
                    ],
                }
                for plan in split_plans
            ],
            "simplified_paths": [
                {
                    "source_path_index": plan.path_index,
                    "id": plan.family_ids[0],
                    "source_subpaths": len(plan.subpaths),
                    "linear_cubics_simplified": (
                        plan.linear_cubics_simplified
                    ),
                    "path_data_bytes_saved": plan.path_data_bytes_saved,
                }
                for plan in simplified_plans
            ],
        }
        if validator is not None:
            def rejected_report(reason: str) -> dict[str, object]:
                rolled_back = {**report, "reason": reason}
                if simplified_plans:
                    rolled_back["attempted_simplification"] = {
                        "source_paths_simplified": report[
                            "source_paths_simplified"
                        ],
                        "linear_cubics_simplified": report[
                            "linear_cubics_simplified"
                        ],
                        "path_data_bytes_saved": report[
                            "path_data_bytes_saved"
                        ],
                        "simplified_paths": report["simplified_paths"],
                    }
                    # These top-level fields describe committed output.  The
                    # complete rejected proposal remains available above.
                    rolled_back.update({
                        "source_paths_simplified": 0,
                        "linear_cubics_simplified": 0,
                        "path_data_bytes_saved": 0,
                        "simplified_paths": [],
                    })
                return rolled_back

            try:
                accepted = bool(validator(svg_text, candidate_text))
            except Exception as exc:  # validators are an external trust boundary
                return CompoundPathSplitResult(svg_text, "rolled_back", {
                    **rejected_report(
                        f"external validator raised {type(exc).__name__}"
                    ),
                })
            if not accepted:
                return CompoundPathSplitResult(
                    svg_text,
                    "rolled_back",
                    rejected_report("external validator rejected candidate"),
                )
        return CompoundPathSplitResult(candidate_text, "applied", report)
    except (_UnsafeSplit, OverflowError, ZeroDivisionError) as exc:
        return CompoundPathSplitResult(svg_text, "rolled_back", {
            **base_report,
            "reason": str(exc) or type(exc).__name__,
        })


__all__ = [
    "BBox", "CompoundPathSplitResult", "Result", "process_compound_paths",
]
