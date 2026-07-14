"""High-confidence coalescing of circular stroke fragments.

The tracing engine often reconstructs a logo ring as two or more editable
``<path>`` strokes because foreground artwork crosses the ring.  This module
detects fragments that are demonstrably parts of the same mathematical circle
and proposes one native ``<circle>`` with ``stroke-dasharray``.  It is kept
independent from the production pipeline so a proposal can be audited before
integration.

Safety is deliberately conservative:

* stroke paint, opacity, caps and joins must agree;
* widths, fitted centres and radii must agree within tight scale-aware limits;
* every fragment must be a monotonic circular arc with low radial residual;
* the union must cover a useful portion of the circle without duplicate arcs;
* a bidirectional one-pixel raster gate must pass before ``safe_to_replace`` is
  true.

No filename, logo colour, element id or canvas coordinate is special-cased.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import html
import math
from pathlib import Path
import re
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)
DETECTOR_VERSION = "annulus-detector-v1"
_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_TOKEN_RE = re.compile(
    r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
          "Q": 4, "T": 2, "A": 7}


@dataclass(frozen=True)
class AnnulusConfig:
    """Scale-aware thresholds for a release-safe native-circle proposal."""

    min_radius_fraction: float = 0.12
    max_radius_fraction: float = 0.55
    min_arc_degrees: float = 35.0
    max_arc_degrees: float = 300.0
    min_monotonicity: float = 0.94
    max_radial_p95_px: float = 2.6
    max_radial_p95_fraction: float = 0.008
    center_tolerance_px: float = 5.0
    center_tolerance_fraction: float = 0.012
    radius_tolerance_px: float = 3.0
    radius_tolerance_fraction: float = 0.010
    width_tolerance_px: float = 1.0
    width_tolerance_fraction: float = 0.12
    min_cluster_degrees: float = 180.0
    max_overlap_fraction: float = 0.12
    sample_step_px: float = 3.0
    angular_bins: int = 1440
    raster_max_side: int = 1600
    raster_tolerance_px: int = 1
    min_raster_recall: float = 0.985
    min_raster_precision: float = 0.985


@dataclass
class StrokeArc:
    element_id: str
    paint: str
    width: float
    opacity: float
    linecap: str
    linejoin: str
    points: np.ndarray
    cx: float
    cy: float
    radius: float
    residual_rms: float
    residual_p95: float
    span_degrees: float
    monotonicity: float


@dataclass(frozen=True)
class AnnulusCandidate:
    source_ids: tuple[str, ...]
    paint: str
    stroke_width: float
    opacity: float
    linecap: str
    linejoin: str
    cx: float
    cy: float
    radius: float
    coverage_degrees: float
    overlap_fraction: float
    residual_rms: float
    residual_p95: float
    center_spread: float
    radius_spread: float
    rotation_degrees: float
    dasharray: tuple[float, ...]
    raster_recall: float
    raster_precision: float
    raster_f1: float
    safe_to_replace: bool
    reasons: tuple[str, ...]

    @property
    def native_id(self) -> str:
        compact = "-".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", x).strip("-")
                           for x in self.source_ids)
        return f"annulus-{compact}"[:180]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["source_ids"] = list(self.source_ids)
        data["dasharray"] = list(self.dasharray)
        data["reasons"] = list(self.reasons)
        data["native_id"] = self.native_id
        data["representation"] = "native_circle_with_dasharray"
        return data

    def svg_element(self) -> str:
        """Return the audited native-circle replacement as an SVG fragment."""

        if not self.safe_to_replace:
            raise ValueError("candidate did not pass the geometry/raster gates")
        attrs = {
            "id": self.native_id,
            "cx": _fmt(self.cx),
            "cy": _fmt(self.cy),
            "r": _fmt(self.radius),
            "fill": "none",
            "stroke": self.paint,
            "stroke-width": _fmt(self.stroke_width),
            "stroke-linecap": self.linecap,
            "stroke-linejoin": self.linejoin,
            "data-merged-from": ",".join(self.source_ids),
            "data-detector": DETECTOR_VERSION,
        }
        if self.opacity < 0.9995:
            attrs["stroke-opacity"] = _fmt(self.opacity)
        if self.dasharray:
            attrs["stroke-dasharray"] = " ".join(_fmt(x) for x in self.dasharray)
            attrs["transform"] = (
                f"rotate({_fmt(self.rotation_degrees)} {_fmt(self.cx)} {_fmt(self.cy)})"
            )
        body = " ".join(f'{key}="{html.escape(value, quote=True)}"'
                        for key, value in attrs.items())
        return f"<circle {body}/>"


def _fmt(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _parse_number(value: str | None, default: float | None = None) -> float | None:
    match = _NUM_RE.search(value or "")
    return float(match.group(0)) if match else default


def _style_map(element: ET.Element) -> dict[str, str]:
    out = {key: value for key, value in element.attrib.items()
           if not key.startswith("{")}
    for part in element.attrib.get("style", "").split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _effective_style(element: ET.Element,
                     parents: dict[ET.Element, ET.Element]) -> dict[str, str]:
    chain = []
    node = element
    while node is not None:
        chain.append(node)
        node = parents.get(node)
    out: dict[str, str] = {}
    opacity = 1.0
    stroke_opacity = 1.0
    for node in reversed(chain):
        style = _style_map(node)
        for key in ("fill", "stroke", "stroke-width", "stroke-linecap",
                    "stroke-linejoin", "display", "visibility"):
            if key in style:
                out[key] = style[key]
        if "opacity" in style:
            opacity *= float(style["opacity"])
        if "stroke-opacity" in style:
            stroke_opacity *= float(style["stroke-opacity"])
    out["effective-opacity"] = str(max(0.0, min(1.0, opacity * stroke_opacity)))
    return out


def _parse_subpaths(d: str) -> list[dict]:
    tokens = _TOKEN_RE.findall(d or "")
    i = 0
    cmd = None
    cur = (0.0, 0.0)
    last_c2 = None
    last_q = None
    sub = None
    subs: list[dict] = []

    def finish(closed: bool = False) -> None:
        nonlocal sub
        if sub and sub["segs"]:
            sub["closed"] = closed
            subs.append(sub)
        sub = None

    while i < len(tokens):
        if tokens[i][0].isalpha():
            cmd = tokens[i]
            i += 1
            if cmd.upper() == "Z":
                if sub:
                    cur = sub["start"]
                finish(True)
                last_c2 = last_q = None
                cmd = None
                continue
        if cmd is None:
            raise ValueError("SVG path data starts without a command")
        up = cmd.upper()
        if up not in _ARITY:
            raise ValueError(f"unsupported SVG path command {cmd!r}")
        n = _ARITY[up]
        first = True
        while i + n <= len(tokens) and not tokens[i][0].isalpha():
            values = [float(tokens[i + k]) for k in range(n)]
            i += n
            rel = cmd.islower()
            if up == "M":
                x, y = values
                if rel:
                    x += cur[0]; y += cur[1]
                if first:
                    finish(False)
                    sub = {"start": (x, y), "segs": []}
                else:
                    sub["segs"].append(("L", x, y))
                cur = (x, y)
                last_c2 = last_q = None
            elif up == "L":
                x, y = values
                if rel:
                    x += cur[0]; y += cur[1]
                sub["segs"].append(("L", x, y)); cur = (x, y)
                last_c2 = last_q = None
            elif up == "H":
                x = values[0] + (cur[0] if rel else 0.0)
                sub["segs"].append(("L", x, cur[1])); cur = (x, cur[1])
                last_c2 = last_q = None
            elif up == "V":
                y = values[0] + (cur[1] if rel else 0.0)
                sub["segs"].append(("L", cur[0], y)); cur = (cur[0], y)
                last_c2 = last_q = None
            elif up == "C":
                c1x, c1y, c2x, c2y, x, y = values
                if rel:
                    c1x += cur[0]; c1y += cur[1]
                    c2x += cur[0]; c2y += cur[1]
                    x += cur[0]; y += cur[1]
                sub["segs"].append(("C", c1x, c1y, c2x, c2y, x, y))
                cur = (x, y); last_c2 = (c2x, c2y); last_q = None
            elif up == "S":
                c2x, c2y, x, y = values
                if rel:
                    c2x += cur[0]; c2y += cur[1]; x += cur[0]; y += cur[1]
                c1 = ((2 * cur[0] - last_c2[0], 2 * cur[1] - last_c2[1])
                      if last_c2 else cur)
                sub["segs"].append(("C", c1[0], c1[1], c2x, c2y, x, y))
                cur = (x, y); last_c2 = (c2x, c2y); last_q = None
            elif up == "Q":
                qx, qy, x, y = values
                if rel:
                    qx += cur[0]; qy += cur[1]; x += cur[0]; y += cur[1]
                sub["segs"].append(("Q", qx, qy, x, y))
                cur = (x, y); last_q = (qx, qy); last_c2 = None
            elif up == "T":
                x, y = values
                if rel:
                    x += cur[0]; y += cur[1]
                q = ((2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1])
                     if last_q else cur)
                sub["segs"].append(("Q", q[0], q[1], x, y))
                cur = (x, y); last_q = q; last_c2 = None
            elif up == "A":
                rx, ry, rot, large, sweep, x, y = values
                if rel:
                    x += cur[0]; y += cur[1]
                sub["segs"].append(("A", rx, ry, rot, large, sweep, x, y))
                cur = (x, y); last_c2 = last_q = None
            first = False
            if up == "M":
                cmd = "l" if cmd.islower() else "L"
                up = "L"; n = 2
        if i < len(tokens) and tokens[i][0].isalpha():
            continue
        if i >= len(tokens):
            break
    finish(False)
    return subs


def _arc_points(start: tuple[float, float], seg: Sequence[float],
                step: float) -> list[tuple[float, float]]:
    rx, ry, phi_deg, large, sweep, x2, y2 = map(float, seg[1:])
    x1, y1 = start
    rx, ry = abs(rx), abs(ry)
    if rx <= 1e-9 or ry <= 1e-9 or math.hypot(x2 - x1, y2 - y1) <= 1e-9:
        return [(x2, y2)]
    phi = math.radians(phi_deg % 360.0)
    cp, sp = math.cos(phi), math.sin(phi)
    dx = (x1 - x2) / 2.0; dy = (y1 - y2) / 2.0
    xp = cp * dx + sp * dy
    yp = -sp * dx + cp * dy
    lam = (xp * xp) / (rx * rx) + (yp * yp) / (ry * ry)
    if lam > 1.0:
        scale = math.sqrt(lam); rx *= scale; ry *= scale
    num = max(0.0, rx * rx * ry * ry - rx * rx * yp * yp - ry * ry * xp * xp)
    den = max(1e-12, rx * rx * yp * yp + ry * ry * xp * xp)
    coef = (-1.0 if bool(large) == bool(sweep) else 1.0) * math.sqrt(num / den)
    cxp = coef * (rx * yp / ry)
    cyp = coef * (-ry * xp / rx)
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        cross = ux * vy - uy * vx
        return math.atan2(cross, dot)

    ux, uy = (xp - cxp) / rx, (yp - cyp) / ry
    vx, vy = (-xp - cxp) / rx, (-yp - cyp) / ry
    theta = angle(1.0, 0.0, ux, uy)
    delta = angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    length = max(rx, ry) * abs(delta)
    count = max(2, min(512, int(math.ceil(length / max(0.5, step)))))
    out = []
    for index in range(1, count + 1):
        a = theta + delta * index / count
        ca, sa = math.cos(a), math.sin(a)
        out.append((cx + cp * rx * ca - sp * ry * sa,
                    cy + sp * rx * ca + cp * ry * sa))
    return out


def _sample_subpath(sub: dict, step: float) -> np.ndarray:
    points = [sub["start"]]
    previous = sub["start"]
    for seg in sub["segs"]:
        kind = seg[0]
        if kind == "L":
            end = (seg[1], seg[2])
            count = max(1, int(math.ceil(math.dist(previous, end) / step)))
            points.extend((previous[0] + (end[0] - previous[0]) * i / count,
                           previous[1] + (end[1] - previous[1]) * i / count)
                          for i in range(1, count + 1))
        elif kind == "C":
            c1 = (seg[1], seg[2]); c2 = (seg[3], seg[4]); end = (seg[5], seg[6])
            length = math.dist(previous, c1) + math.dist(c1, c2) + math.dist(c2, end)
            count = max(3, min(512, int(math.ceil(length / step))))
            for i in range(1, count + 1):
                t = i / count; u = 1.0 - t
                points.append((u ** 3 * previous[0] + 3 * u * u * t * c1[0]
                               + 3 * u * t * t * c2[0] + t ** 3 * end[0],
                               u ** 3 * previous[1] + 3 * u * u * t * c1[1]
                               + 3 * u * t * t * c2[1] + t ** 3 * end[1]))
        elif kind == "Q":
            control = (seg[1], seg[2]); end = (seg[3], seg[4])
            length = math.dist(previous, control) + math.dist(control, end)
            count = max(2, min(512, int(math.ceil(length / step))))
            for i in range(1, count + 1):
                t = i / count; u = 1.0 - t
                points.append((u * u * previous[0] + 2 * u * t * control[0]
                               + t * t * end[0],
                               u * u * previous[1] + 2 * u * t * control[1]
                               + t * t * end[1]))
        elif kind == "A":
            points.extend(_arc_points(previous, seg, step))
            end = (seg[6], seg[7])
        else:
            continue
        if kind == "L":
            end = (seg[1], seg[2])
        previous = end
    return np.asarray(points, dtype=np.float64)


def _least_squares_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    if len(points) < 5:
        return None
    matrix = np.c_[2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))]
    target = np.square(points).sum(axis=1)
    try:
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = float(solution[0]), float(solution[1])
    radius2 = float(solution[2]) + cx * cx + cy * cy
    if not math.isfinite(radius2) or radius2 <= 0:
        return None
    return cx, cy, math.sqrt(radius2)


def _robust_circle(points: np.ndarray) -> tuple[float, float, float, np.ndarray] | None:
    selected = np.ones(len(points), dtype=bool)
    fit = _least_squares_circle(points)
    for _ in range(4):
        if fit is None or selected.sum() < 5:
            return None
        cx, cy, radius = fit
        signed = np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius
        centre = float(np.median(signed[selected]))
        mad = float(np.median(np.abs(signed[selected] - centre)))
        tolerance = max(1.5, 4.5 * 1.4826 * mad)
        new_selected = np.abs(signed - centre) <= tolerance
        if new_selected.sum() < max(5, int(0.65 * len(points))):
            break
        selected = new_selected
        fit = _least_squares_circle(points[selected])
    if fit is None:
        return None
    return *fit, selected


def _arc_from_element(element: ET.Element, style: dict[str, str],
                      viewbox: tuple[float, float], config: AnnulusConfig) -> StrokeArc | None:
    element_id = element.attrib.get("id", "").strip()
    if not element_id or any("transform" in _style_map(node)
                             for node in (element,)):
        return None
    paint = style.get("stroke", "none").strip().lower()
    fill = style.get("fill", "none").strip().lower()
    width = _parse_number(style.get("stroke-width"), 1.0)
    if paint in {"", "none", "transparent"} or fill not in {"", "none", "transparent"}:
        return None
    if width is None or width <= 0:
        return None
    try:
        subs = _parse_subpaths(element.attrib.get("d", ""))
    except (ValueError, IndexError):
        return None
    if len(subs) != 1 or subs[0].get("closed"):
        return None
    points = _sample_subpath(subs[0], config.sample_step_px)
    fit = _robust_circle(points)
    if fit is None:
        return None
    cx, cy, radius, selected = fit
    scale = min(viewbox)
    if not (config.min_radius_fraction * scale <= radius
            <= config.max_radius_fraction * scale):
        return None
    residual = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)
    p95 = float(np.percentile(residual[selected], 95))
    limit = max(config.max_radial_p95_px,
                config.max_radial_p95_fraction * radius)
    angles = np.unwrap(np.arctan2(points[:, 1] - cy, points[:, 0] - cx))
    differences = np.diff(angles)
    total = float(np.abs(differences).sum())
    net = float(abs(angles[-1] - angles[0]))
    monotonicity = net / total if total > 1e-12 else 0.0
    span = math.degrees(net)
    if (p95 > limit or monotonicity < config.min_monotonicity
            or span < config.min_arc_degrees or span > config.max_arc_degrees):
        return None
    return StrokeArc(
        element_id=element_id, paint=paint, width=float(width),
        opacity=float(style.get("effective-opacity", "1")),
        linecap=style.get("stroke-linecap", "butt"),
        linejoin=style.get("stroke-linejoin", "miter"),
        points=points, cx=cx, cy=cy, radius=radius,
        residual_rms=float(math.sqrt(np.mean(np.square(residual[selected])))),
        residual_p95=p95, span_degrees=span, monotonicity=monotonicity,
    )


def _compatible(a: StrokeArc, b: StrokeArc, config: AnnulusConfig) -> bool:
    radius = (a.radius + b.radius) / 2.0
    width = (a.width + b.width) / 2.0
    return (
        a.paint == b.paint
        and abs(a.opacity - b.opacity) <= 0.01
        and a.linecap == b.linecap
        and a.linejoin == b.linejoin
        and math.hypot(a.cx - b.cx, a.cy - b.cy)
            <= max(config.center_tolerance_px,
                   config.center_tolerance_fraction * radius)
        and abs(a.radius - b.radius)
            <= max(config.radius_tolerance_px,
                   config.radius_tolerance_fraction * radius)
        and abs(a.width - b.width)
            <= max(config.width_tolerance_px,
                   config.width_tolerance_fraction * width)
    )


def _components(arcs: Sequence[StrokeArc], config: AnnulusConfig) -> list[list[StrokeArc]]:
    remaining = set(range(len(arcs)))
    components = []
    while remaining:
        stack = [remaining.pop()]
        component = []
        while stack:
            index = stack.pop()
            component.append(arcs[index])
            neighbours = [other for other in remaining
                          if _compatible(arcs[index], arcs[other], config)]
            for other in neighbours:
                remaining.remove(other)
                stack.append(other)
        if len(component) >= 2:
            components.append(component)
    return components


def _coverage_mask(arcs: Sequence[StrokeArc], cx: float, cy: float,
                   bins: int) -> tuple[np.ndarray, float]:
    masks = []
    for arc in arcs:
        angles = np.unwrap(np.arctan2(arc.points[:, 1] - cy,
                                      arc.points[:, 0] - cx))
        lo, hi = sorted((float(angles[0]), float(angles[-1])))
        samples = max(2, int(math.ceil((hi - lo) / (2 * math.pi) * bins)) + 1)
        physical = np.linspace(lo, hi, samples)
        # SVG circles traverse from the 3-o'clock point toward negative
        # mathematical angles.  Store occupancy in that path coordinate.
        indices = np.floor((np.mod(-physical, 2 * math.pi)
                            / (2 * math.pi)) * bins).astype(int) % bins
        mask = np.zeros(bins, dtype=bool)
        mask[indices] = True
        # Fill discrete holes introduced by rounding a continuous interval.
        mask |= np.roll(mask, 1) & np.roll(mask, -1)
        masks.append(mask)
    union = np.logical_or.reduce(masks)
    overlap = ((sum(int(mask.sum()) for mask in masks) - int(union.sum()))
               / max(1, int(union.sum())))
    return union, float(max(0.0, overlap))


def _dash_pattern(coverage: np.ndarray, radius: float) -> tuple[float, tuple[float, ...]]:
    """Encode circular occupancy without dashoffset.

    The circle is rotated so its path starts in the largest gap.  A leading
    zero-length dash then makes the first real run a gap.  Actual user-unit
    lengths are used because the bundled preview renderer ignores pathLength.
    """

    bins = len(coverage)
    if coverage.all():
        return 0.0, ()
    gaps = ~coverage
    doubled = np.r_[gaps, gaps]
    best_start = best_len = 0
    index = 0
    while index < bins:
        if not gaps[index]:
            index += 1
            continue
        end = index
        while end < index + bins and doubled[end]:
            end += 1
        length = min(bins, end - index)
        if length > best_len:
            best_start, best_len = index, length
        index = end
    origin = (best_start + best_len // 2) % bins
    rotated = np.roll(coverage, -origin)
    # Starting at the middle of a gap guarantees both ends are gap runs.
    runs = []
    state = bool(rotated[0])
    count = 1
    for value in rotated[1:]:
        value = bool(value)
        if value == state:
            count += 1
        else:
            runs.append((state, count)); state = value; count = 1
    runs.append((state, count))
    if runs[0][0] or runs[-1][0]:
        raise AssertionError("dash origin was not placed inside a gap")
    circumference = 2.0 * math.pi * radius
    values = [0.0]
    values.extend(count / bins * circumference for _state, count in runs)
    # Keep an even-length list so repeat parity is stable at the closed seam.
    if len(values) % 2:
        values.append(0.0)
    # Path-coordinate origin maps to physical angle -2*pi*origin/bins.
    rotation = -360.0 * origin / bins
    return rotation, tuple(float(value) for value in values)


def _draw_polyline(draw: ImageDraw.ImageDraw, points: np.ndarray,
                   scale: float, width: int, round_caps: bool) -> None:
    coords = [(float(x * scale), float(y * scale)) for x, y in points]
    draw.line(coords, fill=1, width=width, joint="curve")
    if round_caps:
        radius = width / 2.0
        for x, y in (coords[0], coords[-1]):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)


def _dilate(mask: np.ndarray, pixels: int) -> np.ndarray:
    out = mask.copy()
    source = mask
    for distance in range(1, pixels + 1):
        for dy in range(-distance, distance + 1):
            for dx in range(-distance, distance + 1):
                if max(abs(dx), abs(dy)) != distance:
                    continue
                shifted = np.zeros_like(source)
                sy0, sy1 = max(0, -dy), min(source.shape[0], source.shape[0] - dy)
                sx0, sx1 = max(0, -dx), min(source.shape[1], source.shape[1] - dx)
                shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = source[sy0:sy1, sx0:sx1]
                out |= shifted
    return out


def _raster_gate(arcs: Sequence[StrokeArc], cx: float, cy: float,
                 radius: float, width: float, coverage: np.ndarray,
                 viewbox: tuple[float, float], config: AnnulusConfig
                 ) -> tuple[float, float, float]:
    scale = min(1.0, config.raster_max_side / max(viewbox))
    canvas = (max(1, int(math.ceil(viewbox[0] * scale))),
              max(1, int(math.ceil(viewbox[1] * scale))))
    source_image = Image.new("1", canvas)
    proposal_image = Image.new("1", canvas)
    source_draw = ImageDraw.Draw(source_image)
    proposal_draw = ImageDraw.Draw(proposal_image)
    pixel_width = max(1, int(round(width * scale)))
    round_caps = arcs[0].linecap == "round"
    for arc in arcs:
        _draw_polyline(source_draw, arc.points, scale, pixel_width, round_caps)

    bins = len(coverage)
    # Convert each continuous occupied run back to a precise circular arc.
    doubled = np.r_[coverage, coverage]
    visited = np.zeros(bins, dtype=bool)
    for start in range(bins):
        if not coverage[start] or visited[start]:
            continue
        end = start
        while end < start + bins and doubled[end] and not visited[end % bins]:
            visited[end % bins] = True
            end += 1
        count = end - start
        if count <= 0:
            continue
        # Stored path coordinate u corresponds to physical angle -2*pi*u.
        u0 = start / bins; u1 = end / bins
        a0 = -2.0 * math.pi * u0; a1 = -2.0 * math.pi * u1
        samples = max(3, int(math.ceil(radius * abs(a1 - a0) * scale / 2.0)))
        angles = np.linspace(a0, a1, samples)
        points = np.c_[cx + radius * np.cos(angles),
                       cy + radius * np.sin(angles)]
        _draw_polyline(proposal_draw, points, scale, pixel_width, round_caps)

    source = np.asarray(source_image, dtype=bool)
    proposal = np.asarray(proposal_image, dtype=bool)
    tolerance = max(1, int(round(config.raster_tolerance_px * scale)))
    proposal_near = _dilate(proposal, tolerance)
    source_near = _dilate(source, tolerance)
    recall = float(proposal_near[source].mean()) if source.any() else 0.0
    precision = float(source_near[proposal].mean()) if proposal.any() else 0.0
    f1 = (2.0 * recall * precision / (recall + precision)
          if recall + precision else 0.0)
    return recall, precision, f1


def _candidate(component: Sequence[StrokeArc], viewbox: tuple[float, float],
               config: AnnulusConfig) -> AnnulusCandidate | None:
    points = np.vstack([arc.points for arc in component])
    fit = _robust_circle(points)
    if fit is None:
        return None
    cx, cy, radius, selected = fit
    residual = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - radius)
    rms = float(math.sqrt(np.mean(np.square(residual[selected]))))
    p95 = float(np.percentile(residual[selected], 95))
    limit = max(config.max_radial_p95_px,
                config.max_radial_p95_fraction * radius)
    if p95 > limit:
        return None
    # Recheck every source arc against the shared circle; pairwise similarity
    # alone is not enough for a safe merge of three or more fragments.
    for arc in component:
        local = np.abs(np.hypot(arc.points[:, 0] - cx,
                                arc.points[:, 1] - cy) - radius)
        if float(np.percentile(local, 95)) > limit:
            return None
    coverage, overlap = _coverage_mask(component, cx, cy, config.angular_bins)
    coverage_degrees = float(coverage.mean() * 360.0)
    if coverage_degrees < config.min_cluster_degrees or overlap > config.max_overlap_fraction:
        return None
    width = float(np.average([arc.width for arc in component],
                             weights=[len(arc.points) for arc in component]))
    recall, precision, f1 = _raster_gate(
        component, cx, cy, radius, width, coverage, viewbox, config)
    safe = (recall >= config.min_raster_recall
            and precision >= config.min_raster_precision)
    reasons = []
    if safe:
        reasons.append("shared-circle geometry and bidirectional 1px raster gate passed")
    else:
        if recall < config.min_raster_recall:
            reasons.append(f"raster recall {recall:.4f} below {config.min_raster_recall:.4f}")
        if precision < config.min_raster_precision:
            reasons.append(
                f"raster precision {precision:.4f} below {config.min_raster_precision:.4f}")
    rotation, dasharray = _dash_pattern(coverage, radius)
    return AnnulusCandidate(
        source_ids=tuple(sorted(arc.element_id for arc in component)),
        paint=component[0].paint, stroke_width=width,
        opacity=component[0].opacity, linecap=component[0].linecap,
        linejoin=component[0].linejoin, cx=cx, cy=cy, radius=radius,
        coverage_degrees=coverage_degrees, overlap_fraction=overlap,
        residual_rms=rms, residual_p95=p95,
        center_spread=max(math.hypot(arc.cx - cx, arc.cy - cy)
                          for arc in component),
        radius_spread=max(abs(arc.radius - radius) for arc in component),
        rotation_degrees=rotation, dasharray=dasharray,
        raster_recall=recall, raster_precision=precision, raster_f1=f1,
        safe_to_replace=safe, reasons=tuple(reasons),
    )


def detect_svg_annuli(svg_path: str | Path,
                      config: AnnulusConfig | None = None
                      ) -> list[AnnulusCandidate]:
    """Detect safe co-circular stroke clusters in an already-cleaned SVG."""

    config = config or AnnulusConfig()
    root = ET.parse(svg_path).getroot()
    viewbox_numbers = [float(value) for value in
                       re.findall(_NUM_RE, root.attrib.get("viewBox", ""))]
    if len(viewbox_numbers) == 4:
        viewbox = (viewbox_numbers[2], viewbox_numbers[3])
    else:
        width = _parse_number(root.attrib.get("width"))
        height = _parse_number(root.attrib.get("height"))
        if not width or not height:
            return []
        viewbox = (float(width), float(height))
    parents = {child: parent for parent in root.iter() for child in parent}
    arcs = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        # The cleanroom bakes transforms before this stage.  A transformed
        # candidate is skipped until a future implementation can prove the
        # complete transform matrix was applied to stroke width as well.
        node = element
        transformed = False
        while node is not None:
            if _style_map(node).get("transform", "").strip():
                transformed = True
                break
            node = parents.get(node)
        if transformed:
            continue
        style = _effective_style(element, parents)
        arc = _arc_from_element(element, style, viewbox, config)
        if arc is not None:
            arcs.append(arc)
    candidates = []
    for component in _components(arcs, config):
        proposed = _candidate(component, viewbox, config)
        if proposed is not None:
            candidates.append(proposed)
    return sorted(candidates,
                  key=lambda item: (not item.safe_to_replace,
                                    -item.coverage_degrees,
                                    -item.raster_f1,
                                    item.source_ids))


def apply_candidate(svg_path: str | Path, candidate: AnnulusCandidate,
                    output_path: str | Path) -> Path:
    """Write a proposal SVG, replacing only a candidate that passed all gates."""

    if not candidate.safe_to_replace:
        raise ValueError("refusing to apply an unsafe annulus candidate")
    source = Path(svg_path)
    output = Path(output_path)
    tree = ET.parse(source)
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    by_id = {element.attrib.get("id"): element for element in root.iter()
             if element.attrib.get("id")}
    elements = []
    for element_id in candidate.source_ids:
        if element_id not in by_id:
            raise ValueError(f"source element {element_id!r} is missing")
        elements.append(by_id[element_id])
    parent_set = {parents.get(element) for element in elements}
    if len(parent_set) != 1 or None in parent_set:
        raise ValueError("source strokes do not share one SVG parent")
    parent = parent_set.pop()
    children = list(parent)
    insert_at = min(children.index(element) for element in elements)
    for element in elements:
        parent.remove(element)
    replacement = ET.fromstring(candidate.svg_element())
    # Match the document's SVG namespace rather than emitting xmlns="".
    replacement.tag = f"{{{SVG_NS}}}circle"
    parent.insert(insert_at, replacement)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def compare_rendered_pngs(before_png: str | Path, after_png: str | Path,
                          tolerance_px: int = 1) -> dict:
    """Independent final-render gate for a caller's preferred SVG renderer."""

    before = Image.open(before_png).convert("RGBA")
    after = Image.open(after_png).convert("RGBA")
    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.LANCZOS)

    def composite(image: Image.Image) -> np.ndarray:
        rgba = np.asarray(image, dtype=np.float32)
        alpha = rgba[..., 3:4] / 255.0
        return rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)

    a = composite(before)
    b = composite(after)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    background = np.median(border, axis=0)
    before_ink = np.max(np.abs(a - background), axis=2) >= 6.0
    after_ink = np.max(np.abs(b - background), axis=2) >= 6.0
    before_near = _dilate(before_ink, max(0, tolerance_px))
    after_near = _dilate(after_ink, max(0, tolerance_px))
    recall = float(after_near[before_ink].mean()) if before_ink.any() else 1.0
    precision = float(before_near[after_ink].mean()) if after_ink.any() else 1.0
    union = before_ink | after_ink
    color_error = np.max(np.abs(a - b), axis=2)
    color_similarity = (float(np.clip(1.0 - color_error[union] / 128.0,
                                      0.0, 1.0).mean())
                        if union.any() else 1.0)
    f1 = 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0
    score = 100.0 * (0.65 * f1 + 0.35 * color_similarity)
    return {
        "score_percent": score,
        "ink_recall_percent": recall * 100.0,
        "ink_precision_percent": precision * 100.0,
        "ink_f1_percent": f1 * 100.0,
        "color_similarity_percent": color_similarity * 100.0,
        "changed_canvas_fraction": float((color_error >= 2.0).mean()),
        "tolerance_px": int(tolerance_px),
        "accepted": bool(score >= 99.0 and recall >= 0.99 and precision >= 0.99),
    }


__all__ = [
    "AnnulusCandidate", "AnnulusConfig", "DETECTOR_VERSION", "apply_candidate",
    "compare_rendered_pngs", "detect_svg_annuli",
]
