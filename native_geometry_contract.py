"""Prove that an entire compound path can become one native SVG ellipse.

``native_primitives`` describes individual contours, including holes.  A lone
entry in that list does not mean that the whole object is a primitive.
"""
from __future__ import annotations

import math
import re


_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def roundrect_path(primitive):
    """Canonical eight-segment rounded-rectangle numeric proof."""
    x, y, w, h, rx, ry = [float(primitive[key]) for key in ("x", "y", "width", "height", "rx", "ry")]
    if not all(math.isfinite(value) for value in (x, y, w, h, rx, ry)) or not (
            w > 0 and h > 0 and 0 < rx < w/2 and 0 < ry < h/2):
        raise ValueError("invalid rounded rectangle")
    return (f"M{x+rx:.9f} {y:.9f} L{x+w-rx:.9f} {y:.9f} "
            f"A{rx:.9f} {ry:.9f} 0 0 1 {x+w:.9f} {y+ry:.9f} L{x+w:.9f} {y+h-ry:.9f} "
            f"A{rx:.9f} {ry:.9f} 0 0 1 {x+w-rx:.9f} {y+h:.9f} L{x+rx:.9f} {y+h:.9f} "
            f"A{rx:.9f} {ry:.9f} 0 0 1 {x:.9f} {y+h-ry:.9f} L{x:.9f} {y+ry:.9f} "
            f"A{rx:.9f} {ry:.9f} 0 0 1 {x+rx:.9f} {y:.9f} Z")


def whole_object_native_primitive(geometry, path=None):
    """Return a native primitive only after structural AND numeric proof.

    The fitter's native representation is exactly two diametric SVG arcs.
    Restricting promotion to that representation is intentional: an unfamiliar
    representation remains an equally editable path instead of risking loss.
    """
    if not isinstance(geometry, dict) or geometry.get("primitive_first") is not True:
        return None
    topology = geometry.get("topology") or {}
    if (topology.get("topology_preserved") is not True
            or topology.get("components") != 1 or topology.get("holes") != 0
            or any(topology.get(key, 1) != 1
                   for key in ("expected_loops", "actual_loops"))):
        return None
    native = geometry.get("native_primitives") or []
    if len(native) != 1 or not isinstance(native[0], dict):
        return None
    primitive = native[0]
    element = primitive.get("element")
    if element not in {"circle", "ellipse", "rect"}:
        return None
    if path is None:
        path = geometry.get("native_whole_object_path") or geometry.get("path")
    if not isinstance(path, str) or _TOKEN.sub("", path).strip(" ,\t\r\n"):
        return None
    tokens = _TOKEN.findall(path)
    if element == "rect":
        try:
            expected = _TOKEN.findall(roundrect_path(primitive))
            if len(tokens) != len(expected):
                return None
            if all(a == b if b.isalpha() else abs(float(a)-float(b)) <= .000002
                   for a, b in zip(tokens, expected)):
                return dict(primitive)
        except (KeyError, TypeError, ValueError, OverflowError):
            pass
        return None
    if len(tokens) != 20 or [tokens[i] for i in (0, 3, 11, 19)] != ["M", "A", "A", "Z"]:
        return None
    try:
        cx, cy = float(primitive["cx"]), float(primitive["cy"])
        rx = float(primitive["r"] if element == "circle" else primitive["rx"])
        ry = float(primitive["r"] if element == "circle" else primitive["ry"])
        rotation = float(primitive.get("rotation_degrees", 0.0))
        start = tuple(map(float, tokens[1:3]))
        arcs = [tuple(map(float, tokens[4:11])), tuple(map(float, tokens[12:19]))]
    except (ValueError, TypeError, KeyError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (cx, cy, rx, ry, rotation, *start, *arcs[0], *arcs[1])):
        return None
    if rx <= 0 or ry <= 0:
        return None
    # curve_refit serialises coordinates to four decimals. This allowance is
    # serialization precision, not a shape/error-budget exemption.
    epsilon = 0.00015
    close = lambda a, b: abs(a - b) <= epsilon
    if not all(close(arc[0], rx) and close(arc[1], ry)
               and arc[3] in (0, 1) and arc[4] in (0, 1)
               for arc in arcs):
        return None
    if not close(arcs[0][2], arcs[1][2]) or arcs[0][4] != arcs[1][4]:
        return None
    if element == "ellipse" and not close((arcs[0][2] - rotation + 180) % 360 - 180, 0):
        return None
    middle, end = arcs[0][-2:], arcs[1][-2:]
    if (not all(close(a, b) for a, b in zip(start, end))
            or not close((start[0] + middle[0]) / 2, cx)
            or not close((start[1] + middle[1]) / 2, cy)):
        return None
    phi = math.radians(arcs[0][2])
    dx, dy = start[0] - cx, start[1] - cy
    x = dx * math.cos(phi) + dy * math.sin(phi)
    y = -dx * math.sin(phi) + dy * math.cos(phi)
    radius = math.hypot(x / rx, y / ry)
    if abs(radius - 1) * max(rx, ry) > epsilon * 4:
        return None
    return dict(primitive)


def native_geometry_matches(primitive, actual, *, precision=0.000002):
    """Compare serialized final geometry to the independently proven native."""
    if not isinstance(primitive, dict) or not isinstance(actual, dict):
        return False
    if primitive.get("element") != actual.get("element"):
        return False
    keys = {"circle": ("cx", "cy", "r"),
            "ellipse": ("cx", "cy", "rx", "ry", "rotation_degrees"),
            "rect": ("x", "y", "width", "height", "rx", "ry")}.get(primitive.get("element"))
    if not keys or any(key not in actual for key in keys if key != "rotation_degrees"):
        return False
    try:
        return all(math.isfinite(float(actual.get(key, 0)))
                   and abs(float(actual.get(key, 0)) - float(primitive.get(key, 0))) <= precision
                   for key in keys)
    except (TypeError, ValueError, OverflowError):
        return False
