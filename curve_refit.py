# -*- coding: utf-8 -*-
"""Deterministic, error-bounded low-anchor contour refitting.

The main tracer intentionally favours pixel fidelity.  This module is a
side-effect-free proposal primitive for the later, stricter step: turn an
explicit contour (or boolean ownership mask) into the simplest SVG geometry
that stays inside a caller supplied error budget.

Only NumPy, Pillow and existing project modules are used.  The public results
are JSON-safe and include enough evidence for a transactional caller to render
and accept or roll back the proposal.

Public entry points
-------------------
``fit_curve``
    Fit one open or closed subpixel contour.
``fit_compound_contours``
    Fit several closed contours into one even-odd compound path.
``fit_mask``
    Extract subpixel contours from a boolean mask, protect its component/hole
    topology, then call ``fit_compound_contours``.

The fitter considers, in order, a straight line, a circle/rotated ellipse, a
circular arc, and recursively fitted cubic Beziers.  Cubics use the Schneider
least-squares construction with chord-length parameterisation and bounded
reparameterisation.  Adjacent fitted segments are then offered to a second
merge pass; a join is removed only when the combined source samples still fit
inside the same error budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np


_EPS = 1.0e-12


def _finite_real(name, value, *, minimum=0.0, maximum=None, strict=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strict and result <= minimum:
        raise ValueError(f"{name} must be greater than {minimum:g}")
    if not strict and result < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return result


def _positive_int(name, value, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _fmt(value):
    value = float(value)
    if abs(value) < 0.0000005:
        value = 0.0
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _point_text(point):
    return f"{_fmt(point[0])} {_fmt(point[1])}"


def _json_point(point):
    return [round(float(point[0]), 6), round(float(point[1]), 6)]


def _unit(vector, fallback=None):
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length > _EPS:
        return vector / length
    if fallback is not None:
        return _unit(fallback)
    return np.array([1.0, 0.0], dtype=np.float64)


def _as_points(points, *, closed):
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("points must be an Nx2 numeric array") from exc
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(array) < (3 if closed else 2):
        raise ValueError(
            f"{'closed' if closed else 'open'} contours require at least "
            f"{3 if closed else 2} points")
    if not np.isfinite(array).all():
        raise ValueError("points must contain only finite coordinates")

    # Drop a repeated closing point and consecutive duplicates.  Do not mutate
    # the caller's array and do not apply shape-changing smoothing here.
    result = np.array(array, dtype=np.float64, copy=True)
    if closed and len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= 1e-9:
        result = result[:-1]
    keep = np.ones(len(result), dtype=bool)
    if len(result) > 1:
        keep[1:] = np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-9
    result = result[keep]
    if closed and len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= 1e-9:
        result = result[:-1]
    if len(result) < (3 if closed else 2):
        raise ValueError("contour collapses after duplicate points are removed")
    return result


def _polyline_length(points, *, closed=False):
    if len(points) < 2:
        return 0.0
    delta = np.diff(points, axis=0)
    total = float(np.linalg.norm(delta, axis=1).sum())
    if closed:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return total


def _line_errors(points, p0, p1):
    points = np.asarray(points, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    vector = p1 - p0
    denominator = float(np.dot(vector, vector))
    if denominator <= _EPS:
        return np.linalg.norm(points - p0, axis=1)
    t = np.clip(((points - p0) @ vector) / denominator, 0.0, 1.0)
    projected = p0 + t[:, None] * vector
    return np.linalg.norm(points - projected, axis=1)


def _bezier_eval(control, u):
    u = np.asarray(u, dtype=np.float64)
    one = 1.0 - u
    return (
        (one ** 3)[:, None] * control[0]
        + (3.0 * one * one * u)[:, None] * control[1]
        + (3.0 * one * u * u)[:, None] * control[2]
        + (u ** 3)[:, None] * control[3]
    )


def _bezier_point(control, u):
    one = 1.0 - float(u)
    u = float(u)
    return (
        one ** 3 * control[0]
        + 3.0 * one * one * u * control[1]
        + 3.0 * one * u * u * control[2]
        + u ** 3 * control[3]
    )


def _bezier_derivative(control, u):
    one = 1.0 - float(u)
    u = float(u)
    return (
        3.0 * one * one * (control[1] - control[0])
        + 6.0 * one * u * (control[2] - control[1])
        + 3.0 * u * u * (control[3] - control[2])
    )


def _bezier_second_derivative(control, u):
    one = 1.0 - float(u)
    u = float(u)
    return (
        6.0 * one * (control[2] - 2.0 * control[1] + control[0])
        + 6.0 * u * (control[3] - 2.0 * control[2] + control[1])
    )


def _chord_parameters(points):
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    if cumulative[-1] <= _EPS:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / cumulative[-1]


def _generate_bezier(points, parameters, tangent_start, tangent_end):
    """Return the Schneider least-squares cubic for fixed end tangents.

    ``tangent_end`` points backwards from the final anchor toward the curve,
    matching the Graphics Gems convention.
    """
    p0 = points[0]
    p3 = points[-1]
    u = np.asarray(parameters, dtype=np.float64)
    one = 1.0 - u
    b0 = one ** 3
    b1 = 3.0 * u * one * one
    b2 = 3.0 * u * u * one
    b3 = u ** 3

    a0 = b1[:, None] * tangent_start
    a1 = b2[:, None] * tangent_end
    base = (b0 + b1)[:, None] * p0 + (b2 + b3)[:, None] * p3
    residual = points - base

    c00 = float(np.sum(a0 * a0))
    c01 = float(np.sum(a0 * a1))
    c11 = float(np.sum(a1 * a1))
    x0 = float(np.sum(a0 * residual))
    x1 = float(np.sum(a1 * residual))
    determinant = c00 * c11 - c01 * c01
    if abs(determinant) > _EPS:
        alpha_start = (x0 * c11 - x1 * c01) / determinant
        alpha_end = (c00 * x1 - c01 * x0) / determinant
    else:
        alpha_start = alpha_end = 0.0

    chord = float(np.linalg.norm(p3 - p0))
    minimum = max(1.0e-6, chord * 1.0e-6)
    maximum = max(minimum, chord * 4.0)
    if (not math.isfinite(alpha_start) or not math.isfinite(alpha_end)
            or alpha_start < minimum or alpha_end < minimum
            or alpha_start > maximum or alpha_end > maximum):
        alpha_start = alpha_end = chord / 3.0

    return np.vstack((
        p0,
        p0 + alpha_start * tangent_start,
        p3 + alpha_end * tangent_end,
        p3,
    ))


def _reparameterize(points, parameters, control):
    updated = np.asarray(parameters, dtype=np.float64).copy()
    values = updated
    one = 1.0 - values
    curve_points = (
        (one ** 3)[:, None] * control[0]
        + (3.0 * one * one * values)[:, None] * control[1]
        + (3.0 * one * values * values)[:, None] * control[2]
        + (values ** 3)[:, None] * control[3]
    )
    first = (
        (3.0 * one * one)[:, None] * (control[1] - control[0])
        + (6.0 * one * values)[:, None] * (control[2] - control[1])
        + (3.0 * values * values)[:, None] * (control[3] - control[2])
    )
    second = (
        (6.0 * one)[:, None]
        * (control[2] - 2.0 * control[1] + control[0])
        + (6.0 * values)[:, None]
        * (control[3] - 2.0 * control[2] + control[1])
    )
    difference = curve_points - np.asarray(points, dtype=np.float64)
    denominator = (
        first[:, 0] * first[:, 0] + first[:, 1] * first[:, 1]
        + difference[:, 0] * second[:, 0]
        + difference[:, 1] * second[:, 1]
    )
    numerator = (
        difference[:, 0] * first[:, 0]
        + difference[:, 1] * first[:, 1]
    )
    usable = np.abs(denominator) > _EPS
    updated[usable] = np.clip(
        values[usable] - numerator[usable] / denominator[usable],
        0.0, 1.0)
    updated[0] = 0.0
    updated[-1] = 1.0
    # A non-monotone mapping can make an apparently small algebraic residual
    # walk backwards along the curve.  Reject it instead of sorting samples.
    if np.any(np.diff(updated) <= 1.0e-9):
        return np.asarray(parameters, dtype=np.float64)
    return updated


@dataclass
class _Segment:
    kind: str
    p0: np.ndarray
    p1: np.ndarray
    source: np.ndarray
    errors: np.ndarray
    c1: np.ndarray | None = None
    c2: np.ndarray | None = None
    arc: dict | None = None


def _line_segment(points):
    p0 = np.asarray(points[0], dtype=np.float64)
    p1 = np.asarray(points[-1], dtype=np.float64)
    return _Segment(
        "line", p0, p1, np.asarray(points, dtype=np.float64),
        _line_errors(points, p0, p1))


def _cubic_segment(points, control, parameters):
    fitted = _bezier_eval(control, parameters)
    errors = np.linalg.norm(fitted - points, axis=1)
    return _Segment(
        "cubic", control[0], control[3], np.asarray(points, dtype=np.float64),
        errors, c1=control[1], c2=control[2])


def _exact_float64_key(values):
    """Collision-free value key for one process-local fit context."""
    array = np.ascontiguousarray(values, dtype=np.float64)
    return tuple(int(value) for value in array.shape), array.tobytes()


def _one_segment_candidate(points, tolerance, line_tolerance,
                           tangent_start=None, tangent_end=None,
                           candidate_cache=None):
    if len(points) <= 2:
        return _line_segment(points), 1

    if candidate_cache is None:
        line = _line_segment(points)
    else:
        point_key = _exact_float64_key(points)
        line_cache = candidate_cache.setdefault("lines", {})
        line = line_cache.get(point_key)
        if line is None:
            line = _line_segment(points)
            line_cache[point_key] = line
    if float(line.errors.max(initial=0.0)) <= line_tolerance:
        return line, int(np.argmax(line.errors))

    tangent_start = _unit(
        points[1] - points[0] if tangent_start is None else tangent_start)
    tangent_end = _unit(
        points[-2] - points[-1] if tangent_end is None else tangent_end)
    if candidate_cache is None:
        parameters = _chord_parameters(points)
        best = None
        best_index = max(1, len(points) // 2)
        for _ in range(5):
            control = _generate_bezier(
                points, parameters, tangent_start, tangent_end)
            segment = _cubic_segment(points, control, parameters)
            index = int(np.argmax(segment.errors))
            if best is None or segment.errors[index] < best.errors[best_index]:
                best, best_index = segment, index
            if float(segment.errors[index]) <= tolerance:
                return segment, index
            parameters = _reparameterize(points, parameters, control)
        return None, max(1, min(len(points) - 2, best_index))

    profile_key = (
        point_key,
        _exact_float64_key(tangent_start),
        _exact_float64_key(tangent_end),
    )
    profiles = candidate_cache.setdefault("profiles", {})
    profile = profiles.get(profile_key)
    if profile is None:
        profile = {
            "iterations": [],
            "parameters": _chord_parameters(points),
            "pending_control": None,
            "best": None,
            "best_index": max(1, len(points) // 2),
        }
        profiles[profile_key] = profile

    # Query cached iterations in original order.  This preserves the exact
    # "first cubic inside tolerance" rule for both looser and stricter calls.
    for segment, index in profile["iterations"]:
        if float(segment.errors[index]) <= tolerance:
            return segment, index

    while len(profile["iterations"]) < 5:
        pending = profile["pending_control"]
        if pending is not None:
            profile["parameters"] = _reparameterize(
                points, profile["parameters"], pending)
            profile["pending_control"] = None
        parameters = profile["parameters"]
        control = _generate_bezier(
            points, parameters, tangent_start, tangent_end)
        segment = _cubic_segment(points, control, parameters)
        index = int(np.argmax(segment.errors))
        profile["iterations"].append((segment, index))
        best = profile["best"]
        best_index = profile["best_index"]
        if best is None or segment.errors[index] < best.errors[best_index]:
            profile["best"] = segment
            profile["best_index"] = index
        if float(segment.errors[index]) <= tolerance:
            # The uncached path returns before reparameterising.  Defer that
            # work until a future stricter query actually needs another pass.
            profile["pending_control"] = control
            return segment, index
        if len(profile["iterations"]) < 5:
            profile["parameters"] = _reparameterize(
                points, parameters, control)

    return None, max(1, min(len(points) - 2, profile["best_index"]))


def _fit_recursive(points, tolerance, line_tolerance, tangent_start,
                   tangent_end, depth, max_depth, budget,
                   candidate_cache=None):
    if budget[0] <= 0:
        raise RuntimeError("curve fitting exceeded max_segments")
    candidate, split = _one_segment_candidate(
        points, tolerance, line_tolerance, tangent_start, tangent_end,
        candidate_cache=candidate_cache)
    if candidate is not None:
        budget[0] -= 1
        return [candidate]

    if depth >= max_depth or len(points) <= 3:
        result = []
        for start, end in zip(points[:-1], points[1:]):
            if budget[0] <= 0:
                raise RuntimeError("curve fitting exceeded max_segments")
            result.append(_line_segment(np.vstack((start, end))))
            budget[0] -= 1
        return result

    split = max(1, min(len(points) - 2, int(split)))
    if 0 < split < len(points) - 1:
        centre = _unit(points[split - 1] - points[split + 1],
                       fallback=points[split - 1] - points[split])
    else:
        centre = _unit(points[split - 1] - points[split])
    left = _fit_recursive(
        points[:split + 1], tolerance, line_tolerance,
        tangent_start, centre, depth + 1, max_depth, budget,
        candidate_cache=candidate_cache)
    right = _fit_recursive(
        points[split:], tolerance, line_tolerance,
        -centre, tangent_end, depth + 1, max_depth, budget,
        candidate_cache=candidate_cache)
    return left + right


def _segment_start_tangent(segment):
    if segment.kind == "cubic":
        return _unit(segment.c1 - segment.p0, fallback=segment.p1 - segment.p0)
    return _unit(segment.p1 - segment.p0)


def _segment_end_tangent(segment):
    """Backward-pointing tangent at the final anchor."""
    if segment.kind == "cubic":
        return _unit(segment.c2 - segment.p1, fallback=segment.p0 - segment.p1)
    return _unit(segment.p0 - segment.p1)


def _merge_segments(segments, tolerance, line_tolerance,
                    candidate_cache=None):
    segments = list(segments)
    attempts = 0
    accepted = 0
    changed = True
    while changed and len(segments) > 1:
        changed = False
        index = 0
        merged = []
        while index < len(segments):
            if index + 1 >= len(segments):
                merged.append(segments[index])
                break
            left, right = segments[index], segments[index + 1]
            attempts += 1
            source = np.vstack((left.source[:-1], right.source))
            candidate, _ = _one_segment_candidate(
                source, tolerance, line_tolerance,
                _segment_start_tangent(left), _segment_end_tangent(right),
                candidate_cache=candidate_cache)
            if candidate is not None:
                merged.append(candidate)
                accepted += 1
                changed = True
                index += 2
            else:
                merged.append(left)
                index += 1
        segments = merged
    return segments, attempts, accepted


def _point_at_distance(points, index, distance, direction, closed):
    count = len(points)
    current = index
    travelled = 0.0
    guard = 0
    while guard < count:
        nxt = current + direction
        if closed:
            nxt %= count
        elif nxt < 0 or nxt >= count:
            return points[current]
        step = float(np.linalg.norm(points[nxt] - points[current]))
        if travelled + step >= distance and step > _EPS:
            fraction = (distance - travelled) / step
            return points[current] + fraction * (points[nxt] - points[current])
        travelled += step
        current = nxt
        guard += 1
    return points[current]


def _turn_score(points, index, window, closed):
    centre = points[index]
    scores = []
    for scale in (window, window * 2.0):
        before = _point_at_distance(points, index, scale, -1, closed)
        after = _point_at_distance(points, index, scale, 1, closed)
        incoming = _unit(centre - before)
        outgoing = _unit(after - centre)
        cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        scores.append(math.degrees(math.acos(cosine)))
    return min(scores)


def _arc_positions(points, closed):
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(delta)))
    total = float(cumulative[-1])
    if closed:
        total += float(np.linalg.norm(points[0] - points[-1]))
    return cumulative, total


def _detect_corners(points, *, closed, angle_degrees, window):
    count = len(points)
    first = 0 if closed else 1
    last = count if closed else count - 1
    scored = []
    for index in range(first, last):
        score = _turn_score(points, index, window, closed)
        if score >= angle_degrees:
            scored.append((score, index))
    if not scored:
        return [], [0.0] * count

    cumulative, total = _arc_positions(points, closed)
    minimum_spacing = max(1.0, window * 1.5)
    selected = []
    for score, index in sorted(scored, reverse=True):
        position = float(cumulative[index])
        too_close = False
        for other in selected:
            gap = abs(position - float(cumulative[other]))
            if closed:
                gap = min(gap, max(0.0, total - gap))
            if gap < minimum_spacing:
                too_close = True
                break
        if not too_close:
            selected.append(index)
    all_scores = [0.0] * count
    for score, index in scored:
        all_scores[index] = score
    return sorted(selected), all_scores


def _fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points))))
    target = x * x + y * y
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, constant = (float(value) for value in solution)
    radius2 = constant + cx * cx + cy * cy
    if not math.isfinite(radius2) or radius2 <= _EPS:
        return None
    radius = math.sqrt(radius2)
    centre = np.array([cx, cy], dtype=np.float64)
    errors = np.abs(np.linalg.norm(points - centre, axis=1) - radius)
    return centre, radius, errors


def _fit_ellipse(points):
    mean = points.mean(axis=0)
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    if scale <= _EPS:
        return None
    normal = (points - mean) / scale
    x, y = normal[:, 0], normal[:, 1]
    design = np.column_stack((x * x, x * y, y * y, x, y))
    try:
        coefficients, *_ = np.linalg.lstsq(
            design, np.ones(len(points)), rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c, d, e = coefficients
    matrix = np.array([[a, b / 2.0], [b / 2.0, c]], dtype=np.float64)
    linear = np.array([d, e], dtype=np.float64)
    try:
        centre_n = -0.5 * np.linalg.solve(matrix, linear)
    except np.linalg.LinAlgError:
        return None
    level = 1.0 + float(centre_n @ matrix @ centre_n)
    if not math.isfinite(level) or level <= _EPS:
        return None
    quadratic = matrix / level
    try:
        values, vectors = np.linalg.eigh(quadratic)
    except np.linalg.LinAlgError:
        return None
    if np.any(values <= _EPS) or not np.isfinite(values).all():
        return None
    radii_n = 1.0 / np.sqrt(values)
    order = np.argsort(radii_n)[::-1]
    radii = radii_n[order] * scale
    axes = vectors[:, order]
    centre = mean + centre_n * scale

    local = (points - centre) @ axes
    theta = np.arctan2(local[:, 1] / radii[1],
                       local[:, 0] / radii[0])
    candidate_local = np.column_stack((
        radii[0] * np.cos(theta), radii[1] * np.sin(theta)))
    candidates = centre + candidate_local @ axes.T
    errors = np.linalg.norm(points - candidates, axis=1)
    return centre, radii, axes, errors


def _ellipse_primitive(points, tolerance):
    circle = _fit_circle(points)
    if circle is not None:
        centre, radius, errors = circle
        if float(errors.max(initial=0.0)) <= tolerance:
            start = centre + np.array([radius, 0.0])
            opposite = centre - np.array([radius, 0.0])
            arc = {
                "rx": radius, "ry": radius, "rotation": 0.0,
                "large_arc": 1, "sweep": 1,
                "cx": float(centre[0]), "cy": float(centre[1]),
            }
            segments = [
                _Segment("arc", start, opposite, points[:len(points) // 2 + 1],
                         errors[:len(points) // 2 + 1], arc=arc.copy()),
                _Segment("arc", opposite, start, points[len(points) // 2:],
                         errors[len(points) // 2:], arc=arc.copy()),
            ]
            return {
                "name": "circle",
                "segments": segments,
                "errors": errors,
                "native": {
                    "element": "circle",
                    "cx": round(float(centre[0]), 6),
                    "cy": round(float(centre[1]), 6),
                    "r": round(float(radius), 6),
                },
            }

    ellipse = _fit_ellipse(points)
    if ellipse is None:
        return None
    centre, radii, axes, errors = ellipse
    if float(errors.max(initial=0.0)) > tolerance:
        return None
    axis = axes[:, 0]
    start = centre + axis * radii[0]
    opposite = centre - axis * radii[0]
    rotation = math.degrees(math.atan2(axis[1], axis[0]))
    arc = {
        "rx": float(radii[0]), "ry": float(radii[1]),
        "rotation": float(rotation), "large_arc": 1, "sweep": 1,
        "cx": float(centre[0]), "cy": float(centre[1]),
    }
    midpoint = len(points) // 2
    segments = [
        _Segment("arc", start, opposite, points[:midpoint + 1],
                 errors[:midpoint + 1], arc=arc.copy()),
        _Segment("arc", opposite, start, points[midpoint:],
                 errors[midpoint:], arc=arc.copy()),
    ]
    return {
        "name": "ellipse",
        "segments": segments,
        "errors": errors,
        "native": {
            "element": "ellipse",
            "cx": round(float(centre[0]), 6),
            "cy": round(float(centre[1]), 6),
            "rx": round(float(radii[0]), 6),
            "ry": round(float(radii[1]), 6),
            "rotation_degrees": round(float(rotation), 6),
        },
    }


def _arc_primitive(points, tolerance):
    fitted = _fit_circle(points)
    if fitted is None:
        return None
    centre, radius, errors = fitted
    if float(errors.max(initial=0.0)) > tolerance:
        return None
    angles = np.unwrap(np.arctan2(
        points[:, 1] - centre[1], points[:, 0] - centre[0]))
    span = float(angles[-1] - angles[0])
    absolute_span = abs(span)
    if absolute_span < math.radians(8.0) or absolute_span >= math.radians(355.0):
        return None
    differences = np.diff(angles)
    direction = 1.0 if span >= 0.0 else -1.0
    if len(differences):
        reverse_share = float(np.mean(differences * direction < -0.002))
        if reverse_share > 0.08:
            return None
    start = centre + radius * np.array([
        math.cos(angles[0]), math.sin(angles[0])])
    end = centre + radius * np.array([
        math.cos(angles[-1]), math.sin(angles[-1])])
    arc = {
        "rx": radius, "ry": radius, "rotation": 0.0,
        "large_arc": int(absolute_span > math.pi),
        "sweep": int(span > 0.0),
        "cx": float(centre[0]), "cy": float(centre[1]),
    }
    return {
        "name": "circular_arc",
        "segments": [_Segment(
            "arc", start, end, points, errors, arc=arc)],
        "errors": errors,
        "native": {
            "element": "arc",
            "cx": round(float(centre[0]), 6),
            "cy": round(float(centre[1]), 6),
            "r": round(float(radius), 6),
            "start_degrees": round(math.degrees(float(angles[0])), 6),
            "end_degrees": round(math.degrees(float(angles[-1])), 6),
        },
    }


def _split_closed(points, indices):
    count = len(points)
    result = []
    for start, end in zip(indices, indices[1:] + indices[:1]):
        if end > start:
            chain = points[start:end + 1]
        else:
            chain = np.vstack((points[start:], points[:end + 1]))
        if len(chain) >= 2:
            result.append(chain)
    return result


def _fit_chains(points, *, closed, tolerance, line_tolerance,
                corner_angle, corner_window, max_depth, max_segments,
                candidate_cache=None, merge_artificial_seams=False):
    corners, scores = _detect_corners(
        points, closed=closed, angle_degrees=corner_angle,
        window=corner_window)
    if closed:
        protected = list(corners)
        if len(protected) < 2:
            first = protected[0] if protected else int(np.argmin(scores))
            opposite = (first + len(points) // 2) % len(points)
            protected = sorted(set((first, opposite)))
        chains = _split_closed(points, protected)
    else:
        protected = [index for index in corners if 0 < index < len(points) - 1]
        breaks = [0] + protected + [len(points) - 1]
        chains = [points[left:right + 1]
                  for left, right in zip(breaks[:-1], breaks[1:])
                  if right > left]

    budget = [max_segments]
    all_segments = []
    merge_attempts = 0
    merges_accepted = 0
    for chain in chains:
        tangent_start = _unit(chain[1] - chain[0])
        tangent_end = _unit(chain[-2] - chain[-1])
        segments = _fit_recursive(
            chain, tolerance, line_tolerance, tangent_start, tangent_end,
            0, max_depth, budget, candidate_cache=candidate_cache)
        segments, attempts, accepted = _merge_segments(
            segments, tolerance, line_tolerance,
            candidate_cache=candidate_cache)
        all_segments.extend(segments)
        merge_attempts += attempts
        merges_accepted += accepted
    if closed and merge_artificial_seams:
        # Two starting seams are needed to initialise a closed fit; unlike
        # detected corners, they are not features of the source geometry.
        # Offer merges ONLY across these artificial boundaries. The optimizer
        # retains the unmerged candidate and independently checks both.
        artificial = [points[index] for index in protected if index not in corners]
        for boundary in artificial:
            if len(all_segments) < 3:
                break
            junction = next((index for index, segment in enumerate(all_segments)
                             if np.linalg.norm(segment.p1 - boundary) <= 1.0e-9), None)
            if junction is None:
                continue
            left, right = all_segments[junction], all_segments[(junction + 1) % len(all_segments)]
            source = np.vstack((left.source[:-1], right.source))
            merge_attempts += 1
            candidate, _ = _one_segment_candidate(
                source, tolerance, line_tolerance,
                _segment_start_tangent(left), _segment_end_tangent(right),
                candidate_cache=candidate_cache)
            if candidate is not None:
                if junction == len(all_segments) - 1:
                    all_segments = [candidate] + all_segments[1:-1]
                else:
                    all_segments[junction:junction + 2] = [candidate]
                merges_accepted += 1
    return all_segments, corners, merge_attempts, merges_accepted


def _segment_dict(segment):
    result = {
        "type": segment.kind,
        "start": _json_point(segment.p0),
        "end": _json_point(segment.p1),
        "source_points": int(len(segment.source)),
        "max_error": round(float(segment.errors.max(initial=0.0)), 6),
    }
    if segment.kind == "cubic":
        result["control1"] = _json_point(segment.c1)
        result["control2"] = _json_point(segment.c2)
    elif segment.kind == "arc":
        result.update({
            "rx": round(float(segment.arc["rx"]), 6),
            "ry": round(float(segment.arc["ry"]), 6),
            "rotation": round(float(segment.arc["rotation"]), 6),
            "large_arc": int(segment.arc["large_arc"]),
            "sweep": int(segment.arc["sweep"]),
            "center": [round(float(segment.arc["cx"]), 6),
                       round(float(segment.arc["cy"]), 6)],
        })
    return result


def _segments_path(segments, *, closed):
    if not segments:
        return ""
    parts = [f"M{_point_text(segments[0].p0)}"]
    for segment in segments:
        if segment.kind == "line":
            parts.append(f"L{_point_text(segment.p1)}")
        elif segment.kind == "cubic":
            parts.append(
                f"C{_point_text(segment.c1)} {_point_text(segment.c2)} "
                f"{_point_text(segment.p1)}")
        elif segment.kind == "arc":
            arc = segment.arc
            parts.append(
                f"A{_fmt(arc['rx'])} {_fmt(arc['ry'])} "
                f"{_fmt(arc['rotation'])} {int(arc['large_arc'])} "
                f"{int(arc['sweep'])} {_point_text(segment.p1)}")
        else:  # pragma: no cover - internal invariant
            raise AssertionError(f"unsupported segment type: {segment.kind}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _segments_bbox(segments, error=0.0):
    if not segments:
        return None
    points = []
    for segment in segments:
        points.extend((segment.p0, segment.p1))
        if segment.kind == "cubic":
            # A cubic lies in the convex hull of its four controls.
            points.extend((segment.c1, segment.c2))
        elif segment.kind == "arc":
            # The whole supporting ellipse/circle is a conservative bound for
            # a partial arc and avoids fragile angle-extremum special cases.
            rx = float(segment.arc["rx"])
            ry = float(segment.arc["ry"])
            rotation = math.radians(float(segment.arc["rotation"]))
            extent_x = math.sqrt(
                (rx * math.cos(rotation)) ** 2
                + (ry * math.sin(rotation)) ** 2)
            extent_y = math.sqrt(
                (rx * math.sin(rotation)) ** 2
                + (ry * math.cos(rotation)) ** 2)
            centre = np.array([
                float(segment.arc["cx"]), float(segment.arc["cy"]),
            ])
            points.extend((
                centre + np.array([extent_x, extent_y]),
                centre - np.array([extent_x, extent_y]),
            ))
        if len(segment.source):
            points.extend(segment.source)
    array = np.asarray(points, dtype=np.float64)
    pad = max(0.0, float(error))
    low = array.min(axis=0) - pad
    high = array.max(axis=0) + pad
    return [
        round(float(low[0]), 6), round(float(low[1]), 6),
        round(float(high[0] - low[0]), 6),
        round(float(high[1] - low[1]), 6),
    ]


def _error_summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        return {"max": 0.0, "p95": 0.0, "mean": 0.0, "rms": 0.0}
    return {
        "max": round(float(errors.max()), 6),
        "p95": round(float(np.quantile(errors, 0.95)), 6),
        "mean": round(float(errors.mean()), 6),
        "rms": round(float(math.sqrt(np.mean(errors * errors))), 6),
    }


def fit_curve(points, *, closed=False, tolerance=0.75,
              line_tolerance=None, primitive_tolerance=None,
              corner_angle=48.0, corner_window=3.0,
              allow_primitives=True, max_depth=32, max_segments=4096,
              _fit_context=None, merge_artificial_seams=False):
    """Fit one subpixel contour into low-anchor SVG path geometry.

    ``tolerance`` is measured in the same coordinate system as ``points``.
    A closed circle/ellipse is represented by two SVG arc commands so it can
    safely participate in an even-odd compound path; ``native_primitive`` also
    tells an integration layer when a standalone ``<circle>``/``<ellipse>``
    would be equivalent.
    """
    if not isinstance(closed, (bool, np.bool_)):
        raise TypeError("closed must be boolean")
    if not isinstance(allow_primitives, (bool, np.bool_)):
        raise TypeError("allow_primitives must be boolean")
    if not isinstance(merge_artificial_seams, (bool, np.bool_)):
        raise TypeError("merge_artificial_seams must be boolean")
    if _fit_context is not None and not isinstance(_fit_context, dict):
        raise TypeError("_fit_context must be a dictionary or None")
    tolerance = _finite_real("tolerance", tolerance, minimum=0.0, strict=True)
    if line_tolerance is None:
        line_tolerance = tolerance
    else:
        line_tolerance = _finite_real(
            "line_tolerance", line_tolerance, minimum=0.0, strict=True)
    if primitive_tolerance is None:
        primitive_tolerance = tolerance
    else:
        primitive_tolerance = _finite_real(
            "primitive_tolerance", primitive_tolerance,
            minimum=0.0, strict=True)
    corner_angle = _finite_real(
        "corner_angle", corner_angle, minimum=1.0, maximum=179.0)
    corner_window = _finite_real(
        "corner_window", corner_window, minimum=0.0, strict=True)
    max_depth = _positive_int("max_depth", max_depth)
    max_segments = _positive_int("max_segments", max_segments)
    points = _as_points(points, closed=bool(closed))
    input_count = int(len(points))

    primitive = None
    if allow_primitives:
        if closed:
            primitive = _ellipse_primitive(points, primitive_tolerance)
        else:
            line = _line_segment(points)
            if float(line.errors.max(initial=0.0)) <= line_tolerance:
                primitive = {
                    "name": "line", "segments": [line],
                    "errors": line.errors,
                    "native": {"element": "line"},
                }
            else:
                primitive = _arc_primitive(points, primitive_tolerance)

    if primitive is not None:
        segments = primitive["segments"]
        errors = primitive["errors"]
        corners = []
        merge_attempts = merges_accepted = 0
        primitive_name = primitive["name"]
        native = primitive["native"]
    else:
        segments, corners, merge_attempts, merges_accepted = _fit_chains(
            points, closed=bool(closed), tolerance=tolerance,
            line_tolerance=line_tolerance, corner_angle=corner_angle,
            corner_window=corner_window, max_depth=max_depth,
            max_segments=max_segments, candidate_cache=_fit_context,
            merge_artificial_seams=bool(merge_artificial_seams))
        errors = np.concatenate([
            segment.errors for segment in segments if len(segment.errors)
        ]) if segments else np.zeros(0, dtype=np.float64)
        primitive_name = None
        native = None

    path = _segments_path(segments, closed=bool(closed))
    segment_count = len(segments)
    anchor_count = segment_count if closed else segment_count + 1
    removed = max(0, input_count - anchor_count)
    error_summary = _error_summary(errors)
    type_counts = {
        name: sum(segment.kind == name for segment in segments)
        for name in ("line", "cubic", "arc")
    }
    return {
        "path": path,
        "closed": bool(closed),
        "primitive": primitive_name,
        "native_primitive": native,
        "segments": [_segment_dict(segment) for segment in segments],
        "segment_count": int(segment_count),
        "anchor_count": int(anchor_count),
        "input_point_count": input_count,
        "corner_count": int(len(corners)),
        "corner_indices": [int(index) for index in corners],
        "segment_type_counts": type_counts,
        "error": error_summary,
        "bbox": _segments_bbox(segments, error_summary["max"]),
        "economy": {
            "anchors_before": input_count,
            "anchors_after": int(anchor_count),
            "anchors_removed": int(removed),
            "reduction_ratio": round(
                removed / float(input_count), 6) if input_count else 0.0,
            "merge_attempts": int(merge_attempts),
            "merges_accepted": int(merges_accepted),
            "locally_merge_minimal": bool(merge_attempts >= 0),
        },
        "parameters": {
            "tolerance": tolerance,
            "line_tolerance": line_tolerance,
            "primitive_tolerance": primitive_tolerance,
            "corner_angle": corner_angle,
            "corner_window": corner_window,
            "allow_primitives": bool(allow_primitives),
            "merge_artificial_seams": bool(merge_artificial_seams),
            "max_depth": max_depth,
            "max_segments": max_segments,
        },
    }


def fit_compound_contours(contours: Iterable[Sequence], *, tolerance=0.75,
                          line_tolerance=None, primitive_tolerance=None,
                          corner_angle=48.0, corner_window=3.0,
                          allow_primitives=True, max_depth=32,
                          max_segments=4096, _fit_context=None,
                          merge_artificial_seams=False):
    """Fit closed contours into one even-odd compound SVG path."""
    try:
        contour_list = list(contours)
    except TypeError as exc:
        raise TypeError("contours must be an iterable of Nx2 point arrays") from exc
    results = []
    if _fit_context is not None and not isinstance(_fit_context, dict):
        raise TypeError("_fit_context must be a dictionary or None")
    remaining = _positive_int("max_segments", max_segments)
    for contour in contour_list:
        result = fit_curve(
            contour, closed=True, tolerance=tolerance,
            line_tolerance=line_tolerance,
            primitive_tolerance=primitive_tolerance,
            corner_angle=corner_angle, corner_window=corner_window,
            allow_primitives=allow_primitives, max_depth=max_depth,
            max_segments=remaining, _fit_context=_fit_context,
            merge_artificial_seams=merge_artificial_seams)
        remaining -= result["segment_count"]
        if remaining < 0:
            raise RuntimeError("compound fitting exceeded max_segments")
        results.append(result)

    errors = []
    bboxes = []
    for result in results:
        errors.extend(
            segment["max_error"] for segment in result["segments"])
        if result["bbox"] is not None:
            bboxes.append(result["bbox"])
    if bboxes:
        x0 = min(box[0] for box in bboxes)
        y0 = min(box[1] for box in bboxes)
        x1 = max(box[0] + box[2] for box in bboxes)
        y1 = max(box[1] + box[3] for box in bboxes)
        bbox = [round(x0, 6), round(y0, 6),
                round(x1 - x0, 6), round(y1 - y0, 6)]
    else:
        bbox = None
    return {
        "path": " ".join(result["path"] for result in results if result["path"]),
        "fill_rule": "evenodd",
        "loop_count": int(len(results)),
        "contours": results,
        "segment_count": int(sum(result["segment_count"] for result in results)),
        "anchor_count": int(sum(result["anchor_count"] for result in results)),
        "input_point_count": int(sum(
            result["input_point_count"] for result in results)),
        "error": {
            "max": round(max(errors), 6) if errors else 0.0,
            "p95_segment_max": round(float(np.quantile(errors, 0.95)), 6)
            if errors else 0.0,
        },
        "bbox": bbox,
        "economy": {
            "anchors_before": int(sum(
                result["economy"]["anchors_before"] for result in results)),
            "anchors_after": int(sum(
                result["economy"]["anchors_after"] for result in results)),
            "anchors_removed": int(sum(
                result["economy"]["anchors_removed"] for result in results)),
            "reduction_ratio": round(
                (sum(result["economy"]["anchors_removed"] for result in results)
                 / float(max(1, sum(result["economy"]["anchors_before"]
                                    for result in results)))), 6),
            "merges_accepted": int(sum(
                result["economy"]["merges_accepted"] for result in results)),
        },
    }


# Geometry-error optimisation probes this capability before passing its
# private, single-call cache.  Injected/third-party fitters keep their former
# signature and are never given the private keyword.
fit_curve._supports_exact_fit_context = True
fit_compound_contours._supports_exact_fit_context = True
fit_curve._supports_artificial_seam_candidates = True
fit_compound_contours._supports_artificial_seam_candidates = True


def _mask_topology(mask):
    from stroke_engine import connected_components

    _labels, components = connected_components(mask)
    padded_background = ~np.pad(mask, 1, mode="constant", constant_values=False)
    labels, background_components = connected_components(padded_background)
    exterior = int(labels[0, 0]) if labels.size else 0
    holes = 0
    for label in range(1, background_components + 1):
        if label != exterior:
            holes += 1
    return int(components), int(holes)


def fit_mask(mask, *, tolerance=0.75, smooth=0.55, min_area=1.0,
             line_tolerance=None, primitive_tolerance=None,
             corner_angle=48.0, corner_window=3.0,
             allow_primitives=True, max_depth=32, max_segments=4096):
    """Extract and refit all boundaries of an explicit boolean mask.

    At unit ``min_area`` the input component/hole count is a hard contract.  A
    smoothing pass that changes the number of contour loops is discarded and
    the exact pixel-edge contour is used instead.  Every returned subpath is
    closed and the compound path uses ``evenodd``, so hole orientation cannot
    invert the fill.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a numpy.ndarray with dtype bool")
    if mask.dtype != np.bool_:
        raise TypeError("mask dtype must be bool")
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    if mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError("mask dimensions must be non-zero")
    smooth = _finite_real("smooth", smooth, minimum=0.0)
    min_area = _finite_real("min_area", min_area, minimum=0.0, strict=True)
    working = np.ascontiguousarray(mask, dtype=np.bool_)
    components, holes = _mask_topology(working)
    expected_loops = components + holes

    if not working.any():
        return {
            "path": "", "fill_rule": "evenodd", "loop_count": 0,
            "contours": [], "segment_count": 0, "anchor_count": 0,
            "input_point_count": 0, "error": {"max": 0.0,
                                                "p95_segment_max": 0.0},
            "bbox": None,
            "economy": {"anchors_before": 0, "anchors_after": 0,
                         "anchors_removed": 0, "reduction_ratio": 0.0,
                         "merges_accepted": 0},
            "mask_pixels": 0,
            "mask_size": [int(working.shape[1]), int(working.shape[0])],
            "topology": {"components": 0, "holes": 0,
                         "expected_loops": 0, "topology_preserved": True,
                         "smoothing_fallback": False},
        }

    from trace_engine import _mask_to_smooth_loops

    loops = _mask_to_smooth_loops(
        working, simplify=0.0, min_area=min_area, smooth=smooth)
    smoothing_fallback = False
    if min_area <= 1.0 and len(loops) != expected_loops and smooth > 0.0:
        loops = _mask_to_smooth_loops(
            working, simplify=0.0, min_area=min_area, smooth=0.0)
        smoothing_fallback = True
    topology_preserved = len(loops) == expected_loops
    if min_area <= 1.0 and not topology_preserved:
        raise RuntimeError(
            "mask contour extraction changed component/hole topology")

    result = fit_compound_contours(
        loops, tolerance=tolerance, line_tolerance=line_tolerance,
        primitive_tolerance=primitive_tolerance,
        corner_angle=corner_angle, corner_window=corner_window,
        allow_primitives=allow_primitives, max_depth=max_depth,
        max_segments=max_segments)
    result.update({
        "mask_pixels": int(np.count_nonzero(working)),
        "mask_size": [int(working.shape[1]), int(working.shape[0])],
        "topology": {
            "components": components,
            "holes": holes,
            "expected_loops": expected_loops,
            "topology_preserved": bool(topology_preserved),
            "smoothing_fallback": bool(smoothing_fallback),
        },
        "parameters": {
            "tolerance": float(tolerance),
            "smooth": smooth,
            "min_area": min_area,
            "line_tolerance": (None if line_tolerance is None
                               else float(line_tolerance)),
            "primitive_tolerance": (None if primitive_tolerance is None
                                    else float(primitive_tolerance)),
            "corner_angle": float(corner_angle),
            "corner_window": float(corner_window),
            "allow_primitives": bool(allow_primitives),
            "max_depth": int(max_depth),
            "max_segments": int(max_segments),
        },
    })
    return result


# Explicit aliases make the intended integration wording discoverable without
# creating separate behaviours or compatibility branches.
refit_curve = fit_curve
refit_compound_contours = fit_compound_contours
refit_mask = fit_mask


__all__ = [
    "fit_curve", "fit_compound_contours", "fit_mask",
    "refit_curve", "refit_compound_contours", "refit_mask",
]
