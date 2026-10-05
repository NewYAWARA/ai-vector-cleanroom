# -*- coding: utf-8 -*-
"""Geometry-first, error-budgeted anchor minimisation.

This module implements the designer-facing objective that a traced contour is
*not* a request to reproduce every pixel stair-step.  The source contour is a
geometric observation.  Within an explicit, object-scale-normalised geometric
error budget, the winning SVG first has the fewest designer-facing anchors,
then the fewest serialised anchors.

The objective is deliberately lexicographic after the hard gates:

1. preserve loop/closure topology and stay inside the maximum bidirectional
   geometric error budget;
2. minimise designer-facing anchors, then serialised anchors;
3. minimise fragments and SVG segments;
4. prefer simpler primitive categories only when all four economy measures
   above are equal;
5. use measured maximum and p95 error only as deterministic tie-breaks.

Primitive classification is evidence and a late tie-break.  It must never make
a dense source polyline beat a materially smaller safe refit.

Colour and raster/pixel similarity are never inputs to the optimiser.  They
belong to a later transactional safety gate, not to geometry selection.

``curve_refit`` supplies candidate curves for a fixed, deterministic set of
tolerances.  This module independently flattens each candidate and measures an
approximate symmetric Hausdorff distance (plus p95) in both directions.  The
distance is divided by the source object's bounding-box diagonal and reported
as a percentage, so the same budget has the same meaning at different scales.
The original polyline is always included as a zero-error rollback candidate.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Sequence

import numpy as np

from curve_refit import fit_compound_contours, fit_curve


_EPS = 1.0e-12

# Percent of source bbox diagonal.  The grid is independent of the caller's
# budget.  Consequently the eligible set for a looser budget is a superset of
# the eligible set for a tighter one, so relaxing the budget cannot increase
# the selected anchor count (given the same grid and fitter parameters).
DEFAULT_TOLERANCE_PERCENTS = (
    0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.10, 0.16,
    0.25, 0.40, 0.63, 1.0, 1.6, 2.5, 4.0, 6.3, 10.0,
)


def _finite_positive(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _round(value):
    return round(float(value), 9)


def _format_number(value):
    value = float(value)
    if abs(value) < 0.0000005:
        value = 0.0
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _point_text(point):
    return f"{_format_number(point[0])} {_format_number(point[1])}"


def _normalise_contour(points, *, closed):
    try:
        result = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("each contour must be an Nx2 numeric array") from exc
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("each contour must have shape (N, 2)")
    if not np.isfinite(result).all():
        raise ValueError("contours must contain only finite coordinates")
    minimum = 3 if closed else 2
    if len(result) < minimum:
        raise ValueError(
            f"{'closed' if closed else 'open'} contours need at least "
            f"{minimum} points")
    result = np.array(result, dtype=np.float64, copy=True)
    if closed and np.linalg.norm(result[0] - result[-1]) <= 1.0e-9:
        result = result[:-1]
    keep = np.ones(len(result), dtype=bool)
    if len(result) > 1:
        keep[1:] = np.linalg.norm(np.diff(result, axis=0), axis=1) > 1.0e-9
    result = result[keep]
    if closed and len(result) > 1 \
            and np.linalg.norm(result[0] - result[-1]) <= 1.0e-9:
        result = result[:-1]
    if len(result) < minimum:
        raise ValueError("contour collapses after duplicate points are removed")
    return result


def _normalise_tolerance_grid(tolerance_percents):
    if tolerance_percents is None:
        return DEFAULT_TOLERANCE_PERCENTS
    try:
        values = list(tolerance_percents)
    except TypeError as exc:
        raise TypeError("tolerance_percents must be an iterable") from exc
    if not values:
        raise ValueError("tolerance_percents must not be empty")
    normalised = sorted({_finite_positive("tolerance percent", item)
                         for item in values})
    return tuple(normalised)


def _source_scale(contours):
    combined = np.vstack(contours)
    low = combined.min(axis=0)
    high = combined.max(axis=0)
    diagonal = float(np.linalg.norm(high - low))
    if diagonal <= _EPS:
        raise ValueError("source bbox diagonal must be greater than zero")
    return diagonal, [
        _round(low[0]), _round(low[1]),
        _round(high[0] - low[0]), _round(high[1] - low[1]),
    ]


def _identity_path(contours, closed_flags):
    parts = []
    for points, closed in zip(contours, closed_flags):
        commands = [f"M{_point_text(points[0])}"]
        commands.extend(f"L{_point_text(point)}" for point in points[1:])
        if closed:
            commands.append("Z")
        parts.append(" ".join(commands))
    return " ".join(parts)


def _arc_delta(segment):
    centre = np.asarray(segment["center"], dtype=np.float64)
    start = np.asarray(segment["start"], dtype=np.float64) - centre
    end = np.asarray(segment["end"], dtype=np.float64) - centre
    rx = max(_EPS, abs(float(segment["rx"])))
    ry = max(_EPS, abs(float(segment["ry"])))
    phi = math.radians(float(segment.get("rotation", 0.0)))
    cosine, sine = math.cos(phi), math.sin(phi)

    def angle(vector):
        x = cosine * vector[0] + sine * vector[1]
        y = -sine * vector[0] + cosine * vector[1]
        return math.atan2(y / ry, x / rx)

    start_angle = angle(start)
    end_angle = angle(end)
    delta = end_angle - start_angle
    sweep = bool(segment.get("sweep", 0))
    if sweep:
        while delta < 0.0:
            delta += 2.0 * math.pi
    else:
        while delta > 0.0:
            delta -= 2.0 * math.pi
    large = bool(segment.get("large_arc", 0))
    if large and abs(delta) < math.pi - 1.0e-9:
        delta += (2.0 * math.pi if sweep else -2.0 * math.pi)
    elif not large and abs(delta) > math.pi + 1.0e-9:
        delta += (-2.0 * math.pi if sweep else 2.0 * math.pi)
    return start_angle, delta


def _sample_segment(segment, step, max_samples_per_segment):
    kind = segment["type"]
    start = np.asarray(segment["start"], dtype=np.float64)
    end = np.asarray(segment["end"], dtype=np.float64)
    if kind == "line":
        length_bound = float(np.linalg.norm(end - start))
        minimum = 1
    elif kind == "cubic":
        c1 = np.asarray(segment["control1"], dtype=np.float64)
        c2 = np.asarray(segment["control2"], dtype=np.float64)
        length_bound = float(
            np.linalg.norm(c1 - start)
            + np.linalg.norm(c2 - c1)
            + np.linalg.norm(end - c2))
        minimum = 8
    elif kind == "arc":
        _start_angle, delta = _arc_delta(segment)
        length_bound = abs(delta) * max(
            abs(float(segment["rx"])), abs(float(segment["ry"])))
        minimum = 8
    else:
        raise ValueError(f"unsupported fitted segment type: {kind!r}")
    count = max(minimum, int(math.ceil(length_bound / step)))
    count = min(int(max_samples_per_segment), count)
    parameter = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    if kind == "line":
        return start[None, :] + parameter[:, None] * (end - start)[None, :]
    if kind == "cubic":
        one_minus = 1.0 - parameter
        return (
            (one_minus ** 3)[:, None] * start
            + (3.0 * one_minus * one_minus * parameter)[:, None] * c1
            + (3.0 * one_minus * parameter * parameter)[:, None] * c2
            + (parameter ** 3)[:, None] * end
        )
    centre = np.asarray(segment["center"], dtype=np.float64)
    rx, ry = float(segment["rx"]), float(segment["ry"])
    phi = math.radians(float(segment.get("rotation", 0.0)))
    cosine, sine = math.cos(phi), math.sin(phi)
    start_angle, delta = _arc_delta(segment)
    theta = start_angle + parameter * delta
    local_x = rx * np.cos(theta)
    local_y = ry * np.sin(theta)
    return np.column_stack((
        centre[0] + cosine * local_x - sine * local_y,
        centre[1] + sine * local_x + cosine * local_y,
    ))


def _flatten_fit(fit_result, *, step, closed, max_samples_per_segment):
    samples = []
    for index, segment in enumerate(fit_result.get("segments", [])):
        segment_points = _sample_segment(
            segment, step, max_samples_per_segment)
        if index:
            segment_points = segment_points[1:]
        samples.append(segment_points)
    if not samples:
        raise ValueError("fit has no segments")
    result = np.vstack(samples)
    if closed and np.linalg.norm(result[0] - result[-1]) > 1.0e-8:
        result = np.vstack((result, result[0]))
    return result


def _source_polyline(points, closed):
    if closed:
        return np.vstack((points, points[0]))
    return points


class _SegmentBVHNode:
    """One deterministic AABB node over original polyline segments."""

    __slots__ = ("bounds", "indices", "left", "right")

    def __init__(self, bounds, *, indices=None, left=None, right=None):
        self.bounds = bounds
        self.indices = indices
        self.left = left
        self.right = right


def _build_segment_bvh(segment_low, segment_high, centres, indices, leaf_size):
    bounds = np.asarray((
        float(segment_low[indices, 0].min()),
        float(segment_low[indices, 1].min()),
        float(segment_high[indices, 0].max()),
        float(segment_high[indices, 1].max()),
    ), dtype=np.float64)
    if len(indices) <= leaf_size:
        return _SegmentBVHNode(
            bounds, indices=np.sort(indices, kind="stable"))
    local_centres = centres[indices]
    spread = np.ptp(local_centres, axis=0)
    axis = int(np.argmax(spread))
    order = np.argsort(local_centres[:, axis], kind="stable")
    ordered = indices[order]
    middle = len(ordered) // 2
    left = _build_segment_bvh(
        segment_low, segment_high, centres, ordered[:middle], leaf_size)
    right = _build_segment_bvh(
        segment_low, segment_high, centres, ordered[middle:], leaf_size)
    return _SegmentBVHNode(bounds, left=left, right=right)


def _prepare_polyline_segments(polyline, *, leaf_size):
    polyline = np.asarray(polyline, dtype=np.float64)
    starts = polyline[:-1]
    ends = polyline[1:]
    if not len(starts):
        raise ValueError("distance polyline must contain a segment")
    delta = ends - starts
    denominator = np.sum(delta * delta, axis=1)
    # Expand every leaf box outwards by one representable value.  AABB
    # pruning must remain conservative even at a floating-point boundary.
    segment_low = np.nextafter(np.minimum(starts, ends), -np.inf)
    segment_high = np.nextafter(np.maximum(starts, ends), np.inf)
    centres = starts + 0.5 * delta
    indices = np.arange(len(starts), dtype=np.intp)
    tree = _build_segment_bvh(
        segment_low, segment_high, centres, indices, int(leaf_size))
    return starts, delta, denominator, tree


def _bbox_distance_squared(points, bounds):
    lower = bounds[:2]
    upper = bounds[2:]
    outside = np.maximum(
        np.maximum(lower[None, :] - points, 0.0),
        points - upper[None, :])
    return np.sum(outside * outside, axis=1)


def _bvh_nearest_squared(points, prepared):
    starts, delta, denominator, tree = prepared
    best = np.full(len(points), np.inf, dtype=np.float64)
    machine_epsilon = np.finfo(np.float64).eps

    def visit(node, active):
        if not len(active):
            return
        point_block = points[active]
        lower_bound = _bbox_distance_squared(point_block, node.bounds)
        current = best[active]
        # Outward-rounded boxes are already conservative.  The small extra
        # slack prevents a final subtract/square rounding from excluding a
        # segment whose mathematical lower bound equals the current best.
        slack = 64.0 * machine_epsilon * np.maximum(1.0, current)
        active = active[lower_bound <= current + slack]
        if not len(active):
            return
        if node.indices is not None:
            index = node.indices
            a = starts[index]
            segment_delta = delta[index]
            segment_denominator = denominator[index]
            point_block = points[active]
            offset = point_block[:, None, :] - a[None, :, :]
            numerator = np.sum(
                offset * segment_delta[None, :, :], axis=2)
            parameter = np.divide(
                numerator, segment_denominator[None, :],
                out=np.zeros_like(numerator),
                where=segment_denominator[None, :] > _EPS)
            parameter = np.clip(parameter, 0.0, 1.0)
            projected = (
                a[None, :, :]
                + parameter[:, :, None] * segment_delta[None, :, :])
            distances2 = np.sum(
                (point_block[:, None, :] - projected) ** 2, axis=2)
            best[active] = np.minimum(
                best[active], distances2.min(axis=1))
            return

        left_bound = _bbox_distance_squared(
            points[active], node.left.bounds)
        right_bound = _bbox_distance_squared(
            points[active], node.right.bounds)
        first, second = (
            (node.left, node.right)
            if float(left_bound.min()) <= float(right_bound.min())
            else (node.right, node.left))
        visit(first, active)
        visit(second, active)

    visit(tree, np.arange(len(points), dtype=np.intp))
    return best


def _nearest_distances(points, polyline, *, point_chunk=512,
                       segment_chunk=64):
    """Exact point-to-polyline distance with conservative AABB pruning."""
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if isinstance(point_chunk, (bool, np.bool_)) or int(point_chunk) < 1:
        raise ValueError("point_chunk must be a positive integer")
    if isinstance(segment_chunk, (bool, np.bool_)) or int(segment_chunk) < 1:
        raise ValueError("segment_chunk must be a positive integer")
    prepared = _prepare_polyline_segments(
        polyline, leaf_size=int(segment_chunk))
    output = np.empty(len(points), dtype=np.float64)
    for point_start in range(0, len(points), point_chunk):
        point_block = points[point_start:point_start + point_chunk]
        best = _bvh_nearest_squared(point_block, prepared)
        output[point_start:point_start + len(point_block)] = np.sqrt(best)
    return output


def _quantile95(values):
    return float(np.quantile(values, 0.95)) if len(values) else 0.0


def _measure_loop(source, fit_result, *, closed, scale, sampling_step_percent,
                  max_samples_per_segment):
    step = max(scale * sampling_step_percent / 100.0, scale * 1.0e-9)
    fitted = _flatten_fit(
        fit_result, step=step, closed=closed,
        max_samples_per_segment=max_samples_per_segment)
    source_polyline = _source_polyline(source, closed)
    source_to_fit = _nearest_distances(source, fitted)
    fit_to_source = _nearest_distances(fitted, source_polyline)
    combined = np.concatenate((source_to_fit, fit_to_source))
    return {
        "source_to_fit": source_to_fit,
        "fit_to_source": fit_to_source,
        "combined": combined,
        "fit_sample_count": int(len(fitted)),
        "polyline": fitted,
    }


def _salient_corner_indices(points, *, closed, scale,
                            window_percent=1.5, minimum_turn_degrees=45.0):
    """Find object-scale corners while ignoring one-pixel staircase turns."""
    count = len(points)
    window = max(scale * window_percent / 100.0, scale * 1.0e-9)

    def reference_distant_index(start, direction):
        current = start
        travelled = 0.0
        for _ in range(count - 1):
            next_index = current + direction
            if closed:
                next_index %= count
            elif next_index < 0 or next_index >= count:
                return current
            travelled += float(np.linalg.norm(points[next_index] - points[current]))
            current = next_index
            if travelled >= window:
                break
        return current

    # The old walk is quadratic for densely sampled exact contact curves.
    # Search the same accumulated polyline lengths instead. Around floating
    # boundaries, retain the old per-start accumulation order as a tie-breaker.
    lengths = np.asarray([float(np.linalg.norm(points[(i + 1) % count] - points[i]))
                          for i in range(count if closed else max(0, count - 1))])
    if not len(lengths):
        return []
    uniform = bool(np.all(lengths == lengths[0]))
    if uniform:
        travelled, uniform_steps = 0.0, 0
        while uniform_steps < count - 1 and travelled < window:
            travelled += float(lengths[0])
            uniform_steps += 1
    cumulative = np.r_[0.0, np.cumsum(np.tile(lengths, 3) if closed else lengths)]
    offset = count if closed else 0
    starts = offset + np.arange(count)
    forward = np.searchsorted(cumulative, cumulative[starts] + window, side='left')
    backward = np.searchsorted(cumulative, cumulative[starts] - window, side='right') - 1
    forward = np.minimum(forward, starts + count - 1 if closed else count - 1)
    backward = np.maximum(backward, starts - count + 1 if closed else 0)
    slack = np.finfo(float).eps * max(1.0, float(cumulative[-1])) * (4 * count + 8)

    def distant_index(start, direction):
        if uniform:
            index = start + direction * uniform_steps
            return index % count if closed else min(count - 1, max(0, index))
        origin = int(starts[start])
        index = int(forward[start] if direction > 0 else backward[start])
        distance = direction * (float(cumulative[index]) - float(cumulative[origin]))
        inner = index - direction
        inner_distance = (direction * (float(cumulative[inner]) - float(cumulative[origin]))
                          if 0 <= inner < len(cumulative) else -1.0)
        if abs(distance - window) <= slack or abs(inner_distance - window) <= slack:
            return reference_distant_index(start, direction)
        return index % count if closed else index

    result = []
    first = 0 if closed else 1
    last = count if closed else count - 1
    threshold = math.radians(minimum_turn_degrees)
    for index in range(first, last):
        previous = distant_index(index, -1)
        following = distant_index(index, 1)
        if previous == index or following == index:
            continue
        incoming = points[index] - points[previous]
        outgoing = points[following] - points[index]
        denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
        if denominator <= _EPS:
            continue
        cosine = float(np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0))
        if math.acos(cosine) >= threshold:
            result.append(index)
    return result


def measure_fit_error(source_contours, fit_result, *, closed=None,
                      normalization_scale=None, sampling_step_percent=0.02,
                      max_samples_per_segment=4096,
                      error_budget_percent=None,
                      corner_window_percent=1.5,
                      corner_minimum_turn_degrees=45.0,
                      _relationship_cache=None):
    """Measure approximate bidirectional Hausdorff and p95 fit error.

    ``source_contours`` may be one Nx2 contour for a single ``fit_curve``
    result, or an iterable of closed contours for a
    ``fit_compound_contours`` result.  The returned values are JSON-safe.
    """
    sampling_step_percent = _finite_positive(
        "sampling_step_percent", sampling_step_percent)
    if error_budget_percent is not None:
        error_budget_percent = _finite_positive(
            "error_budget_percent", error_budget_percent)
    corner_window_percent = _finite_positive(
        "corner_window_percent", corner_window_percent)
    corner_minimum_turn_degrees = _finite_positive(
        "corner_minimum_turn_degrees", corner_minimum_turn_degrees)
    if corner_minimum_turn_degrees >= 180.0:
        raise ValueError("corner_minimum_turn_degrees must be less than 180")
    if isinstance(max_samples_per_segment, (bool, np.bool_)) \
            or not isinstance(max_samples_per_segment, (int, np.integer)):
        raise TypeError("max_samples_per_segment must be an integer")
    max_samples_per_segment = int(max_samples_per_segment)
    if max_samples_per_segment < 16:
        raise ValueError("max_samples_per_segment must be at least 16")
    if not isinstance(fit_result, dict):
        raise TypeError("fit_result must be a curve_refit result dictionary")

    is_compound = "contours" in fit_result and "loop_count" in fit_result
    if is_compound:
        try:
            raw_contours = list(source_contours)
        except TypeError as exc:
            raise TypeError("compound source_contours must be iterable") from exc
        contours = [_normalise_contour(item, closed=True)
                    for item in raw_contours]
        fits = list(fit_result.get("contours", []))
        closed_flags = [True] * len(contours)
        if len(fits) != len(contours):
            raise ValueError("source and fitted compound loop counts differ")
    else:
        if closed is None:
            closed = bool(fit_result.get("closed", False))
        if not isinstance(closed, (bool, np.bool_)):
            raise TypeError("closed must be boolean")
        contours = [_normalise_contour(source_contours, closed=bool(closed))]
        fits = [fit_result]
        closed_flags = [bool(closed)]

    source_scale, bbox = _source_scale(contours)
    if normalization_scale is None:
        scale = source_scale
    else:
        scale = _finite_positive("normalization_scale", normalization_scale)
    loop_records = []
    all_source_to_fit = []
    all_fit_to_source = []
    fit_sample_count = 0
    all_corner_distances = []
    fitted_polylines = []
    for index, (source, fitted, is_closed) in enumerate(
            zip(contours, fits, closed_flags)):
        measured = _measure_loop(
            source, fitted, closed=is_closed, scale=scale,
            sampling_step_percent=sampling_step_percent,
            max_samples_per_segment=max_samples_per_segment)
        source_to_fit = measured["source_to_fit"]
        fit_to_source = measured["fit_to_source"]
        fitted_polylines.append(measured["polyline"])
        all_source_to_fit.append(source_to_fit)
        all_fit_to_source.append(fit_to_source)
        fit_sample_count += measured["fit_sample_count"]
        corner_indices = _salient_corner_indices(
            source, closed=is_closed, scale=scale,
            window_percent=corner_window_percent,
            minimum_turn_degrees=corner_minimum_turn_degrees)
        corner_distances = source_to_fit[corner_indices] if corner_indices \
            else np.zeros(0, dtype=np.float64)
        all_corner_distances.append(corner_distances)
        loop_max = max(
            float(source_to_fit.max(initial=0.0)),
            float(fit_to_source.max(initial=0.0)))
        loop_p95 = max(_quantile95(source_to_fit),
                       _quantile95(fit_to_source))
        loop_scale, _ = _source_scale([source])
        if error_budget_percent is None:
            loop_over_budget_share = None
            loop_over_budget_sample_count = None
        else:
            threshold = scale * error_budget_percent / 100.0
            loop_combined = np.concatenate((source_to_fit, fit_to_source))
            loop_over_budget_share = _round(
                float(np.mean(loop_combined > threshold))
                if len(loop_combined) else 0.0)
            loop_over_budget_sample_count = int(
                np.sum(loop_combined > threshold))
        loop_records.append({
            "loop_index": int(index),
            "closed": bool(is_closed),
            "source_point_count": int(len(source)),
            "fit_sample_count": measured["fit_sample_count"],
            "max_absolute": _round(loop_max),
            "p95_absolute": _round(loop_p95),
            "max_percent": _round(100.0 * loop_max / scale),
            "p95_percent": _round(100.0 * loop_p95 / scale),
            "salient_corner_count": int(len(corner_indices)),
            "salient_corner_max_percent": _round(
                100.0 * float(corner_distances.max(initial=0.0)) / scale),
            "over_budget_share": loop_over_budget_share,
            "over_budget_sample_count": loop_over_budget_sample_count,
            # A large outer boundary must not grant a tiny hole a large
            # absolute distortion budget. Keep both object and local units.
            "local_normalization_scale": _round(loop_scale),
            "local_max_percent": _round(100.0 * loop_max / loop_scale),
            "local_p95_percent": _round(100.0 * loop_p95 / loop_scale),
            "local_salient_corner_max_percent": _round(
                100.0 * float(corner_distances.max(initial=0.0)) / loop_scale),
            "local_over_budget_share": (None if error_budget_percent is None else _round(
                float(np.mean(measured["combined"] > loop_scale * error_budget_percent / 100.0)))),
        })
    source_to_fit = np.concatenate(all_source_to_fit)
    fit_to_source = np.concatenate(all_fit_to_source)
    combined = np.concatenate((source_to_fit, fit_to_source))
    max_source = float(source_to_fit.max(initial=0.0))
    max_fit = float(fit_to_source.max(initial=0.0))
    max_error = max(max_source, max_fit)
    # Use the worse directional p95 rather than allowing many samples in one
    # direction to conceal a poor tail in the other direction.
    p95_source = _quantile95(source_to_fit)
    p95_fit = _quantile95(fit_to_source)
    p95_error = max(p95_source, p95_fit)
    corner_distances = np.concatenate(all_corner_distances) \
        if all_corner_distances else np.zeros(0, dtype=np.float64)
    result = {
        "method": "approximate_bidirectional_hausdorff_polyline_sampling",
        "approximate": True,
        "normalization_basis": "source_bbox_diagonal",
        "normalization_scale": _round(scale),
        "source_bbox": bbox,
        "max_absolute": _round(max_error),
        "p95_absolute": _round(p95_error),
        "max_percent": _round(100.0 * max_error / scale),
        "p95_percent": _round(100.0 * p95_error / scale),
        "salient_corner_count": int(len(corner_distances)),
        "salient_corner_max_absolute": _round(
            float(corner_distances.max(initial=0.0))),
        "salient_corner_max_percent": _round(
            100.0 * float(corner_distances.max(initial=0.0)) / scale),
        "mean_absolute": _round(float(combined.mean()) if len(combined) else 0.0),
        "source_to_fit": {
            "max_absolute": _round(max_source),
            "p95_absolute": _round(p95_source),
            "sample_count": int(len(source_to_fit)),
        },
        "fit_to_source": {
            "max_absolute": _round(max_fit),
            "p95_absolute": _round(p95_fit),
            "sample_count": int(len(fit_to_source)),
        },
        "sampling": {
            "step_percent_of_scale": _round(sampling_step_percent),
            "requested_step_absolute": _round(
                scale * sampling_step_percent / 100.0),
            "max_samples_per_segment": int(max_samples_per_segment),
            "fit_sample_count": int(fit_sample_count),
            "corner_window_percent_of_scale": _round(corner_window_percent),
            "corner_minimum_turn_degrees": _round(
                corner_minimum_turn_degrees),
        },
        "loops": loop_records,
    }
    if error_budget_percent is not None:
        threshold = scale * error_budget_percent / 100.0
        result["error_budget_percent"] = _round(error_budget_percent)
        result["over_budget_share"] = _round(
            float(np.mean(combined > threshold)) if len(combined) else 0.0)
        result["over_budget_sample_count"] = int(np.sum(combined > threshold))
    if is_compound and len(contours) > 1:
        from contour_relationships import loop_relationship_signature, relationship_evidence

        cache = _relationship_cache if isinstance(_relationship_cache, dict) else {}
        source_key = ("source_loop_relationships", tuple(
            (points.shape, points.tobytes()) for points in contours))
        if source_key not in cache:
            cache[source_key] = loop_relationship_signature(contours)
        result["compound_relationships"] = relationship_evidence(
            cache[source_key], loop_relationship_signature(fitted_polylines))
    return result


def _topology_evidence(fit_result, *, closed_flags, compound):
    if compound:
        fitted = list(fit_result.get("contours", []))
        loop_count_ok = (int(fit_result.get("loop_count", -1))
                         == len(closed_flags) == len(fitted))
        closure_ok = loop_count_ok and all(
            bool(item.get("closed", False)) for item in fitted)
        fill_rule_ok = fit_result.get("fill_rule") == "evenodd"
    else:
        loop_count_ok = True
        closure_ok = bool(fit_result.get("closed", False)) == closed_flags[0]
        fill_rule_ok = True
    segment_ok = int(fit_result.get("segment_count", 0)) > 0
    preserved = bool(loop_count_ok and closure_ok and fill_rule_ok and segment_ok)
    return {
        "preserved": preserved,
        "loop_count_preserved": bool(loop_count_ok),
        "closure_preserved": bool(closure_ok),
        "evenodd_preserved": bool(fill_rule_ok),
        "nonempty_geometry": bool(segment_ok),
    }


def _add_relationship_topology(topology, error):
    relationships = error.get("compound_relationships")
    if isinstance(relationships, dict):
        topology["compound_relationships"] = relationships
        topology["preserved"] = bool(topology["preserved"] and relationships.get("preserved") is True)


def _identity_fit(contours, closed_flags, *, compound):
    path = _identity_path(contours, closed_flags)
    per_contour = []
    for points, closed in zip(contours, closed_flags):
        segment_count = len(points) if closed else len(points) - 1
        segments = []
        for index in range(segment_count):
            following = (index + 1) % len(points)
            segments.append({
                "type": "line",
                "start": [_round(points[index, 0]), _round(points[index, 1])],
                "end": [_round(points[following, 0]),
                        _round(points[following, 1])],
                "source_points": 2,
                "max_error": 0.0,
            })
        per_contour.append({
            "path": _identity_path([points], [closed]),
            "closed": bool(closed),
            "segments": segments,
            "segment_count": int(segment_count),
            "anchor_count": int(len(points)),
            "input_point_count": int(len(points)),
            # This flag is deliberately present on every contour, not only on
            # the compound wrapper.  Per-loop selection and mixed-compound
            # evidence must not mistake a dense exact rollback for analytic
            # line/arc geometry merely because it serialises as L commands.
            "source_identity": True,
        })
    anchors = sum(len(points) for points in contours)
    segments = sum(item["segment_count"] for item in per_contour)
    if compound:
        return {
            "path": path,
            "fill_rule": "evenodd",
            "loop_count": int(len(contours)),
            "contours": per_contour,
            "segment_count": int(segments),
            "anchor_count": int(anchors),
            "input_point_count": int(anchors),
            "source_identity": True,
        }
    result = per_contour[0]
    result["source_identity"] = True
    return result


def _candidate_error_row(error):
    return {
        "actual_max_error_percent": error["max_percent"],
        "actual_p95_error_percent": error["p95_percent"],
        "actual_max_error_absolute": error["max_absolute"],
        "actual_p95_error_absolute": error["p95_absolute"],
        "source_to_fit_max_absolute": error["source_to_fit"]["max_absolute"],
        "fit_to_source_max_absolute": error["fit_to_source"]["max_absolute"],
        "over_budget_share": error.get("over_budget_share"),
        "salient_corner_count": error.get("salient_corner_count", 0),
        "salient_corner_max_percent": error.get(
            "salient_corner_max_percent", 0.0),
    }


def _primitive_complexity(fit_result):
    """Return designer-facing primitive evidence and an ordering rank.

    A native primitive is structurally simpler than a collection of Beziers,
    even if a particular SVG serialisation gives the latter a superficially
    similar command count.  Low-sided all-line geometry is the rectangle /
    polygon tier.  A general all-line polyline is not analytic line/arc
    geometry: rank 2 requires at least one actual arc segment.  Exact source
    rollbacks are explicitly classified as such.  The classification never
    bypasses the geometric, topology, or salient-corner hard constraints and
    is only a tie-break after all economy measures are equal.
    """
    if "contours" in fit_result and "loop_count" in fit_result:
        contours = list(fit_result.get("contours", []))
    else:
        contours = [fit_result]
    primitives = [item.get("primitive") for item in contours]
    segments = [segment for item in contours
                for segment in item.get("segments", [])]
    types = [segment.get("type") for segment in segments]
    contains_source_identity = bool(fit_result.get("source_identity")) or any(
        bool(item.get("source_identity")) for item in contours)
    native_names = {"circle", "ellipse", "line", "circular_arc"}
    if contains_source_identity:
        rank = 3
        category = (
            "source_polyline_rollback"
            if all(bool(item.get("source_identity")) for item in contours)
            else "mixed_geometry_with_source_polyline_rollback"
        )
    elif contours and all(name in native_names for name in primitives):
        rank = 0
        category = "native_primitive"
    elif segments and all(name == "line" for name in types) \
            and len(segments) <= 4 * max(1, len(contours)):
        rank = 1
        category = "low_sided_line_primitive"
    elif segments and all(name in {"line", "arc"} for name in types) \
            and any(name == "arc" for name in types):
        rank = 2
        category = "analytic_line_arc_geometry"
    elif segments and all(name == "line" for name in types):
        rank = 3
        category = "general_polyline_geometry"
    else:
        rank = 3
        category = "bezier_geometry"
    designer_anchors = 0
    for item, primitive in zip(contours, primitives):
        if primitive in {"circle", "ellipse"}:
            designer_anchors += 4
        elif primitive in {"line", "circular_arc"}:
            designer_anchors += 2
        else:
            designer_anchors += int(item.get("anchor_count", 0))
    return {
        "rank": int(rank),
        "category": category,
        "native_primitives": [name for name in primitives if name],
        "designer_anchor_count": int(designer_anchors),
    }


def _error_contract_evidence(error, budget):
    """Apply the one authoritative geometry contract to measured evidence."""
    epsilon = max(1.0e-9, float(budget) * 1.0e-9)
    p95_ok = float(error.get("p95_percent", float("inf"))) <= budget + epsilon
    max_tail_ok = (
        float(error.get("max_percent", float("inf")))
        <= 3.0 * budget + epsilon)
    outlier_share_ok = (
        float(error.get("over_budget_share", 1.0)) <= 0.05 + 1.0e-9)
    corner_ok = (
        float(error.get("salient_corner_max_percent", float("inf")))
        <= 2.0 * budget + epsilon)
    local_contracts = [
        _error_contract_evidence(_loop_error_evidence(loop), budget)
        for loop in error.get("loops", [])
        if "local_normalization_scale" in loop]
    local_ok = all(row["within_budget"] for row in local_contracts)
    relationships_ok = (error.get("compound_relationships") or {}).get("preserved", True) is True
    return {
        "within_budget": bool(
            p95_ok and max_tail_ok and outlier_share_ok and corner_ok and local_ok and relationships_ok),
        "p95_within_budget": bool(p95_ok),
        "max_within_three_times_budget": bool(max_tail_ok),
        "over_budget_share_at_most_5_percent": bool(outlier_share_ok),
        "salient_corner_max_within_two_times_budget": bool(corner_ok),
        "each_loop_local_scale_within_budget": bool(local_ok),
        "compound_loop_relationships_preserved": bool(relationships_ok),
    }


def _loop_error_evidence(loop_error):
    """Adapt one loop record to the global error-contract field names."""
    return {
        "max_percent": float(loop_error.get("local_max_percent",
                                            loop_error.get("max_percent", float("inf")))),
        "p95_percent": float(loop_error.get("local_p95_percent",
                                            loop_error.get("p95_percent", float("inf")))),
        "over_budget_share": float(
            loop_error.get("local_over_budget_share", loop_error.get("over_budget_share", 1.0))),
        "salient_corner_max_percent": float(
            loop_error.get("local_salient_corner_max_percent",
                           loop_error.get("salient_corner_max_percent", float("inf")))),
    }


def _combine_compound_loop_fits(loop_fits):
    """Combine independently fitted closed loops without changing topology."""
    loop_fits = list(loop_fits)
    paths = [item.get("path", "") for item in loop_fits]
    if not loop_fits or any(not path.strip() for path in paths):
        raise ValueError("each mixed compound loop must have a nonempty path")
    if any(not bool(item.get("closed", False)) for item in loop_fits):
        raise ValueError("each mixed compound loop must remain closed")
    segment_count = int(sum(
        int(item.get("segment_count", 0)) for item in loop_fits))
    anchor_count = int(sum(
        int(item.get("anchor_count", 0)) for item in loop_fits))
    input_point_count = int(sum(
        int(item.get("input_point_count", 0)) for item in loop_fits))
    if segment_count <= 0 or anchor_count <= 0:
        raise ValueError("mixed compound fit must contain geometry")
    bboxes = [item.get("bbox") for item in loop_fits
              if item.get("bbox") is not None]
    if bboxes:
        x0 = min(float(box[0]) for box in bboxes)
        y0 = min(float(box[1]) for box in bboxes)
        x1 = max(float(box[0]) + float(box[2]) for box in bboxes)
        y1 = max(float(box[1]) + float(box[3]) for box in bboxes)
        bbox = [_round(x0), _round(y0), _round(x1 - x0), _round(y1 - y0)]
    else:
        bbox = None
    removed = max(0, input_point_count - anchor_count)
    return {
        "path": " ".join(paths),
        "fill_rule": "evenodd",
        "loop_count": int(len(loop_fits)),
        "contours": loop_fits,
        "segment_count": segment_count,
        "anchor_count": anchor_count,
        "input_point_count": input_point_count,
        "bbox": bbox,
        "economy": {
            "anchors_before": input_point_count,
            "anchors_after": anchor_count,
            "anchors_removed": removed,
            "reduction_ratio": _round(
                removed / float(input_point_count)
                if input_point_count else 0.0),
            "merges_accepted": int(sum(
                int(item.get("economy", {}).get("merges_accepted", 0))
                for item in loop_fits)),
        },
        "mixed_loop_tolerances": True,
    }


def _segment_type_counts(fit_result):
    """Return compact deterministic segment counts without copying geometry."""
    counts = {}
    for segment in fit_result.get("segments", []):
        name = str(segment.get("type") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return {name: int(counts[name]) for name in sorted(counts)}


def _compact_loop_frontier_entry(option, *, source_point_count):
    """Strip a per-loop option to diagnostic evidence (never path/fit data)."""
    primitive = option.get("primitive_complexity") or {}
    fit_result = option.get("fit") or {}
    is_identity = bool(
        option.get("source") == "source_identity_per_loop_rollback"
        or fit_result.get("source_identity"))
    tolerance_percent = option.get(
        "tolerance_percent_of_bbox_diagonal")
    primitive_rank = primitive.get("rank")
    primitive_category = primitive.get("category")
    rejection_reasons = list(option.get("rejection_reasons", []))
    return {
        "candidate_id": option.get("candidate_id"),
        "source": option.get("source"),
        "source_point_count": int(source_point_count),
        "tolerance_percent": tolerance_percent,
        "tolerance_percent_of_bbox_diagonal": tolerance_percent,
        "eligible": bool(option.get("eligible", False)),
        "reasons": rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "primitive": {
            "rank": primitive_rank,
            "category": primitive_category,
        },
        "primitive_rank": primitive_rank,
        "primitive_category": primitive_category,
        "designer_anchor_count": option.get("designer_anchor_count"),
        "anchors_after": option.get("anchors_after"),
        "segment_count": option.get("segment_count"),
        "segment_type_counts": _segment_type_counts(fit_result),
        "actual_p95_error_percent": option.get(
            "actual_p95_error_percent"),
        "actual_max_error_percent": option.get(
            "actual_max_error_percent"),
        "over_budget_share": option.get("over_budget_share"),
        "salient_corner_max_percent": option.get(
            "salient_corner_max_percent"),
        "identity": is_identity,
        "identity_rollback": is_identity,
    }


def _selected_mixed_identity_summary(selected_row):
    """Summarise exact loops retained by the selected mixed candidate."""
    selections = list(selected_row.get("per_loop_selection", []))
    identity_loops = [
        item for item in selections if item.get("identity_rollback")]
    largest = None
    if identity_loops:
        item = max(
            identity_loops,
            key=lambda row: (
                int(row.get("source_point_count", 0)),
                -int(row.get("loop_index", 0)),
            ))
        largest = {
            "loop_index": int(item["loop_index"]),
            "source_point_count": int(item["source_point_count"]),
            "anchors_after": int(item["anchors_after"]),
            "designer_anchor_count": int(item["designer_anchor_count"]),
            "primitive_rank": item.get("primitive_rank"),
            "primitive_category": item.get("primitive_category"),
        }
    return {
        "is_mixed_loop_candidate": bool(selections),
        "identity_loop_count": int(len(identity_loops)),
        "identity_source_point_count": int(sum(
            int(item.get("source_point_count", 0))
            for item in identity_loops)),
        "identity_anchor_count": int(sum(
            int(item.get("anchors_after", 0)) for item in identity_loops)),
        "identity_designer_anchor_count": int(sum(
            int(item.get("designer_anchor_count", 0))
            for item in identity_loops)),
        "largest_identity_loop": largest,
    }


def _selection_key(row):
    """Deterministic economy-first ordering after all hard gates pass."""
    return (
        int(row["designer_anchor_count"]),
        int(row["anchors_after"]),
        int(row["fragment_count"]),
        int(row["segment_count"]),
        # Primitive class is evidence, but can only decide between candidates
        # with exactly the same designer/serialised economy.
        int(row["primitive_complexity"]["rank"]),
        float(row["actual_max_error_percent"]),
        float(row["actual_p95_error_percent"]),
        str(row["candidate_id"]),
    )


def _refinement_loop_selection(option, *, loop_index, source_point_count):
    """Return render-independent evidence for one selected compound loop."""
    primitive = dict(option.get("primitive_complexity") or {})
    fit_result = option.get("fit") or {}
    identity = bool(
        option.get("source") == "source_identity_per_loop_rollback"
        or fit_result.get("source_identity"))
    tolerance_percent = option.get(
        "tolerance_percent_of_bbox_diagonal")
    return {
        "loop_index": int(loop_index),
        "source_point_count": int(source_point_count),
        "selected_candidate_id": option.get("candidate_id"),
        "selected_source": option.get("source"),
        "source": option.get("source"),
        "selected_tolerance_percent_of_bbox_diagonal": tolerance_percent,
        "tolerance_percent": tolerance_percent,
        "tolerance_absolute": option.get("tolerance_absolute"),
        "anchors_after": int(option.get("anchors_after", 0)),
        "designer_anchor_count": int(
            option.get("designer_anchor_count", 0)),
        "segment_count": int(option.get("segment_count", 0)),
        "primitive_rank": primitive.get("rank"),
        "primitive_category": primitive.get("category"),
        "primitive_complexity": primitive,
        "selected_fit_input_point_count": int(
            fit_result.get("input_point_count", source_point_count)),
        "selected_fit_source_identity": identity,
        "source_identity": identity,
        "actual_p95_error_percent": option.get(
            "actual_p95_error_percent"),
        "actual_max_error_percent": option.get(
            "actual_max_error_percent"),
        "over_budget_share": option.get("over_budget_share"),
        "salient_corner_max_percent": option.get(
            "salient_corner_max_percent"),
        "error_contract": dict(option.get("error_contract") or {}),
        "identity_rollback": identity,
        "identity_rollback_point_count": (
            int(source_point_count) if identity else 0),
    }


def _failed_refinement_contract():
    return {
        "p95_within_budget": False,
        "max_within_three_times_budget": False,
        "over_budget_share_at_most_5_percent": False,
        "salient_corner_max_within_two_times_budget": False,
    }


def _refinement_sort_key(row):
    """The existing economy key, tolerant of failed global remeasurement."""
    primitive = row.get("primitive_complexity") or {}

    def measured(name):
        value = row.get(name)
        return float(value) if value is not None else float("inf")

    return (
        int(row.get("designer_anchor_count", 0)),
        int(row.get("anchors_after", 0)),
        int(row.get("fragment_count", 0)),
        int(row.get("segment_count", 0)),
        int(primitive.get("rank", 3)),
        measured("actual_max_error_percent"),
        measured("actual_p95_error_percent"),
        str(row.get("candidate_id", "")),
    )


def _is_strictly_more_conservative_loop_option(option, base_option):
    """Require a real per-loop anchor refinement at no wider tolerance."""
    if not option.get("eligible"):
        return False
    if (base_option.get("source") == "source_identity_per_loop_rollback"
            or (base_option.get("fit") or {}).get("source_identity")):
        return False
    option_fit = option.get("fit") or {}
    base_fit = base_option.get("fit") or {}
    if option_fit.get("path", "") == base_fit.get("path", ""):
        return False
    option_designer = int(option.get("designer_anchor_count", 0))
    base_designer = int(base_option.get("designer_anchor_count", 0))
    option_anchors = int(option.get("anchors_after", 0))
    base_anchors = int(base_option.get("anchors_after", 0))
    strictly_more_anchors = bool(
        option_designer > base_designer
        or (option_designer == base_designer
            and option_anchors > base_anchors))
    if not strictly_more_anchors:
        return False
    for name in (
            "actual_p95_error_percent", "actual_max_error_percent",
            "over_budget_share", "salient_corner_max_percent"):
        option_value = option.get(name)
        base_value = base_option.get(name)
        if (not isinstance(option_value, Real)
                or isinstance(option_value, bool)
                or not math.isfinite(float(option_value))
                or not isinstance(base_value, Real)
                or isinstance(base_value, bool)
                or not math.isfinite(float(base_value))
                or float(option_value) > float(base_value) + _EPS):
            return False
    if option.get("source") == "source_identity_per_loop_rollback":
        return True
    option_tolerance = option.get(
        "tolerance_percent_of_bbox_diagonal")
    base_tolerance = base_option.get(
        "tolerance_percent_of_bbox_diagonal")
    return bool(
        option_tolerance is not None and base_tolerance is not None
        and float(option_tolerance) <= float(base_tolerance) + _EPS)


def _measure_refinement_candidate(
        candidate_id, candidate_kind, selected_options, *,
        changed_loop_index, replacement_option, contours, closed_flags,
        budget, scale, sampling_step_percent, max_samples_per_segment,
        metric_cache, base_anchor_count, base_designer_anchor_count,
        exact_error=None):
    """Combine and globally certify one renderable compound refinement."""
    selections = [
        _refinement_loop_selection(
            option, loop_index=index,
            source_point_count=len(contours[index]))
        for index, option in enumerate(selected_options)
    ]
    anchors = int(sum(item["anchors_after"] for item in selections))
    designer_anchors = int(sum(
        item["designer_anchor_count"] for item in selections))
    segments = int(sum(item["segment_count"] for item in selections))
    fallback_rank = max(
        (int(item["primitive_rank"] or 3) for item in selections),
        default=3)
    row = {
        "candidate_id": candidate_id,
        "candidate_kind": candidate_kind,
        "source": "compound_per_loop_refinement_frontier",
        "changed_loop_index": changed_loop_index,
        "replacement_candidate_id": (
            replacement_option.get("candidate_id")
            if replacement_option is not None else None),
        "replacement_tolerance_percent": (
            replacement_option.get("tolerance_percent_of_bbox_diagonal")
            if replacement_option is not None else None),
        "replacement_tolerance_absolute": (
            replacement_option.get("tolerance_absolute")
            if replacement_option is not None else None),
        "path": None,
        "fit_succeeded": False,
        "global_remeasurement_after_loop_merge": True,
        "global_remeasurement_succeeded": False,
        "topology": {
            "preserved": False,
            "loop_count_preserved": False,
            "closure_preserved": False,
            "evenodd_preserved": False,
            "nonempty_geometry": False,
        },
        "within_error_budget": False,
        "eligible": False,
        "anchors_after": anchors,
        "designer_anchor_count": designer_anchors,
        "anchor_cost": {
            "anchors_added_from_base": int(anchors - base_anchor_count),
            "designer_anchors_added_from_base": int(
                designer_anchors - base_designer_anchor_count),
        },
        "segment_count": segments,
        "fragment_count": int(len(contours)),
        "primitive_complexity": {
            "rank": fallback_rank,
            "category": "uncombined_compound_refinement",
            "native_primitives": [],
            "designer_anchor_count": designer_anchors,
        },
        "actual_max_error_percent": None,
        "actual_p95_error_percent": None,
        "over_budget_share": None,
        "salient_corner_max_percent": None,
        "error_contract": _failed_refinement_contract(),
        "per_loop_selection": selections,
        "rejection_reasons": [],
    }
    try:
        combined = _combine_compound_loop_fits(
            [option["fit"] for option in selected_options])
    except (RuntimeError, TypeError, ValueError,
            np.linalg.LinAlgError) as exc:
        row.update({
            "rejection_reasons": ["compound_loop_combine_failed"],
            "failure_stage": "combine",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        })
        return row

    row["path"] = combined.get("path")
    row["fit_succeeded"] = True
    topology = _topology_evidence(
        combined, closed_flags=closed_flags, compound=True)
    primitive = _primitive_complexity(combined)
    row.update({
        "topology": topology,
        "anchors_after": int(combined["anchor_count"]),
        "designer_anchor_count": int(primitive["designer_anchor_count"]),
        "segment_count": int(combined["segment_count"]),
        "primitive_complexity": primitive,
    })
    row["anchor_cost"] = {
        "anchors_added_from_base": int(
            row["anchors_after"] - base_anchor_count),
        "designer_anchors_added_from_base": int(
            row["designer_anchor_count"] - base_designer_anchor_count),
    }
    try:
        path_key = combined.get("path", "")
        if exact_error is not None:
            error = exact_error
        else:
            if path_key not in metric_cache:
                metric_cache[path_key] = measure_fit_error(
                    contours, combined,
                    normalization_scale=scale,
                    sampling_step_percent=sampling_step_percent,
                    max_samples_per_segment=max_samples_per_segment,
                    error_budget_percent=budget, _relationship_cache=metric_cache)
            error = metric_cache[path_key]
        _add_relationship_topology(topology, error)
        contract = _error_contract_evidence(error, budget)
        within_budget = contract.pop("within_budget")
        row.update({
            "global_remeasurement_succeeded": True,
            "within_error_budget": bool(within_budget),
            "eligible": bool(topology["preserved"] and within_budget),
            "error_contract": contract,
            **_candidate_error_row(error),
        })
        reasons = []
        if not topology["preserved"]:
            reasons.append("topology_not_preserved")
        if not within_budget:
            reasons.append("geometric_error_budget_exceeded")
        row["rejection_reasons"] = reasons
    except (KeyError, RuntimeError, TypeError, ValueError,
            np.linalg.LinAlgError) as exc:
        reasons = []
        if not topology["preserved"]:
            reasons.append("topology_not_preserved")
        reasons.append("global_remeasurement_failed")
        row.update({
            "rejection_reasons": reasons,
            "failure_stage": "global_remeasurement",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        })
    return row


def _build_compound_refinement_frontier(
        contours, closed_flags, *, base_loop_options, loop_fit_options,
        identity_loop_options, identity, zero_error, budget, scale,
        sampling_step_percent, max_samples_per_segment, metric_cache,
        selected_candidate_id, selected_path):
    """Build one opt-in, one-step-more-conservative candidate per loop."""
    base_anchors = int(sum(
        option["anchors_after"] for option in base_loop_options))
    base_designer = int(sum(
        option["designer_anchor_count"] for option in base_loop_options))
    candidates = [_measure_refinement_candidate(
        "compound_refinement_base", "base_mixed", base_loop_options,
        changed_loop_index=None, replacement_option=None,
        contours=contours, closed_flags=closed_flags, budget=budget,
        scale=scale, sampling_step_percent=sampling_step_percent,
        max_samples_per_segment=max_samples_per_segment,
        metric_cache=metric_cache, base_anchor_count=base_anchors,
        base_designer_anchor_count=base_designer)]

    for loop_index, base_option in enumerate(base_loop_options):
        ordered = sorted(
            [option for option in loop_fit_options[loop_index].values()
             if option.get("eligible")] + [identity_loop_options[loop_index]],
            key=_selection_key)
        replacement = next(
            (option for option in ordered
             if _is_strictly_more_conservative_loop_option(
                 option, base_option)),
            None)
        if replacement is None:
            continue
        refined = list(base_loop_options)
        refined[loop_index] = replacement
        candidates.append(_measure_refinement_candidate(
            f"compound_refinement_loop_{loop_index:02d}_next",
            "single_loop_refinement", refined,
            changed_loop_index=int(loop_index),
            replacement_option=replacement, contours=contours,
            closed_flags=closed_flags, budget=budget, scale=scale,
            sampling_step_percent=sampling_step_percent,
            max_samples_per_segment=max_samples_per_segment,
            metric_cache=metric_cache, base_anchor_count=base_anchors,
            base_designer_anchor_count=base_designer))

    identity_candidate = _measure_refinement_candidate(
        "compound_refinement_source_identity", "source_identity_control",
        identity_loop_options, changed_loop_index=None,
        replacement_option=None, contours=contours,
        closed_flags=closed_flags, budget=budget, scale=scale,
        sampling_step_percent=sampling_step_percent,
        max_samples_per_segment=max_samples_per_segment,
        metric_cache=metric_cache, base_anchor_count=base_anchors,
        base_designer_anchor_count=base_designer, exact_error=zero_error)
    # The independently combined exact loops must serialize identically to the
    # authoritative source identity path.
    identity_path_matches = bool(
        identity_candidate.get("fit_succeeded")
        and identity_candidate.get("path") == identity.get("path"))
    identity_candidate["exact_source_identity"] = identity_path_matches
    if identity_path_matches:
        identity_candidate["path"] = identity["path"]
    else:
        identity_candidate["eligible"] = False
        identity_candidate["within_error_budget"] = False
        identity_candidate["rejection_reasons"] = list(dict.fromkeys(
            list(identity_candidate.get("rejection_reasons") or [])
            + ["source_identity_serialization_mismatch"]))
        identity_candidate["failure_stage"] = "source_identity_control"
    candidates.append(identity_candidate)
    candidates.sort(key=_refinement_sort_key)
    base_candidate = next(
        row for row in candidates
        if row["candidate_id"] == "compound_refinement_base")
    return {
        "schema_version": 1,
        "generation_basis": (
            "geometry_only_next_more_conservative_option_per_loop"),
        "uses_colour_or_pixel_similarity": False,
        "sort_basis": "existing_economy_key_then_candidate_id",
        "selected_candidate_id": selected_candidate_id,
        "selected_path": selected_path,
        "base_candidate_id": base_candidate["candidate_id"],
        "base_path": base_candidate.get("path"),
        "base_per_loop_selection": base_candidate["per_loop_selection"],
        "source_identity_candidate_id": identity_candidate["candidate_id"],
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
    }


def _optimise(contours, closed_flags, *, compound, error_budget_percent,
              tolerance_percents, sampling_step_percent,
              max_samples_per_segment, fitter_options,
              include_refinement_frontier=False,
              include_independently_valid_loop_candidates=False):
    budget = _finite_positive("error_budget_percent", error_budget_percent)
    full_grid = _normalise_tolerance_grid(tolerance_percents)
    # curve_refit's internal one-way tolerance only has to reach the robust
    # contract's 3x maximum-tail cap.  Four times the p95 budget leaves margin
    # for its different estimator without paying for obviously unusable 6.3%
    # and 10% fits on every path.  Because the cap grows monotonically with the
    # budget, looser runs still contain every candidate from tighter runs.
    if tolerance_percents is None:
        cap = max(0.10, 4.0 * budget)
        grid = tuple(value for value in full_grid if value <= cap + 1.0e-12)
        if not grid:
            grid = (full_grid[0],)
    else:
        grid = full_grid
    scale, bbox = _source_scale(contours)
    anchors_before = int(sum(len(points) for points in contours))
    segments_before = int(sum(
        len(points) if closed else len(points) - 1
        for points, closed in zip(contours, closed_flags)))
    fit_cache = {}
    metric_cache = {}
    fit_call_options = dict(fitter_options)
    active_fitter = fit_compound_contours if compound else fit_curve
    if getattr(active_fitter, "_supports_exact_fit_context", False) is True:
        # One optimiser call owns this cache.  It never crosses paths, masks,
        # images or processes, and it carries no tolerance-dependent answer.
        fit_call_options["_fit_context"] = {}
    rows = []
    candidate_fits = {}
    # Every all-loops-at-one-tolerance fit also gives us one independently
    # measurable fit for each loop.  Retain that frontier so mixed-scale
    # compounds can keep a tiny hole exact while simplifying a large outline.
    # No extra fitter calls and no colour evidence are involved.
    loop_fit_options = [dict() for _ in contours] if compound else []
    largest_loop_index = (
        max(range(len(contours)), key=lambda item: (len(contours[item]), -item))
        if compound else None)
    largest_loop_frontier_entries = []

    supports_seams = bool(
        any(closed_flags)
        and getattr(active_fitter, "_supports_artificial_seam_candidates", False) is True
        and "merge_artificial_seams" not in fitter_options)
    jobs = [(index, tolerance, variant)
            for index, tolerance in enumerate(grid)
            for variant in ((False, True) if supports_seams else (False,))]
    for index, tolerance_percent, seam_variant in jobs:
        candidate_id = f"curve_refit_{index:02d}" + ("_seam" if seam_variant else "")
        tolerance_absolute = scale * tolerance_percent / 100.0
        if seam_variant:
            baseline = fit_cache.get((_round(tolerance_absolute), False)) or {}
            baseline_loops = baseline.get("contours", []) if compound else [baseline]
            if not any(loop.get("closed") and not loop.get("primitive")
                       and loop.get("corner_count", 2) < 2
                       and loop.get("segment_count", 0) >= 3 for loop in baseline_loops):
                continue
        largest_frontier_recorded = False
        row = {
            "candidate_id": candidate_id,
            "source": "curve_refit",
            "tolerance_percent_of_bbox_diagonal": _round(tolerance_percent),
            "tolerance_absolute": _round(tolerance_absolute),
            "artificial_seam_merge_candidate": bool(seam_variant),
        }
        try:
            cache_key = (_round(tolerance_absolute), seam_variant)
            candidate_options = dict(fit_call_options)
            if seam_variant:
                candidate_options["merge_artificial_seams"] = True
            if cache_key not in fit_cache:
                if compound:
                    fit_cache[cache_key] = fit_compound_contours(
                        contours, tolerance=tolerance_absolute,
                        line_tolerance=tolerance_absolute,
                        primitive_tolerance=tolerance_absolute,
                        **candidate_options)
                else:
                    fit_cache[cache_key] = fit_curve(
                        contours[0], closed=closed_flags[0],
                        tolerance=tolerance_absolute,
                        line_tolerance=tolerance_absolute,
                        primitive_tolerance=tolerance_absolute,
                        **candidate_options)
            fitted = fit_cache[cache_key]
            topology = _topology_evidence(
                fitted, closed_flags=closed_flags, compound=compound)
            loop_structure_preserved = bool(topology["preserved"])
            path_key = fitted.get("path", "")
            if path_key not in metric_cache:
                metric_cache[path_key] = measure_fit_error(
                    contours if compound else contours[0], fitted,
                    closed=(None if compound else closed_flags[0]),
                    normalization_scale=scale,
                    sampling_step_percent=sampling_step_percent,
                    max_samples_per_segment=max_samples_per_segment,
                    error_budget_percent=budget, _relationship_cache=metric_cache)
            error = metric_cache[path_key]
            _add_relationship_topology(topology, error)
            contract = _error_contract_evidence(error, budget)
            within_budget = contract.pop("within_budget")
            primitive = _primitive_complexity(fitted)
            row.update({
                "fit_succeeded": True,
                "topology": topology,
                "within_error_budget": within_budget,
                "eligible": bool(topology["preserved"] and within_budget),
                "anchors_after": int(fitted["anchor_count"]),
                "segment_count": int(fitted["segment_count"]),
                "fragment_count": int(len(contours)),
                "primitive_complexity": primitive,
                "designer_anchor_count": primitive["designer_anchor_count"],
                "error_contract": contract,
                **_candidate_error_row(error),
            })
            reasons = []
            if not topology["preserved"]:
                reasons.append("topology_not_preserved")
            if not within_budget:
                reasons.append("geometric_error_budget_exceeded")
            row["rejection_reasons"] = reasons
            candidate_fits[candidate_id] = (fitted, error)
            # A different loop's unsafe primitive can change nesting in the
            # all-at-one-tolerance candidate. Do not discard an independently
            # valid outer fit with it: mixed candidates restore unsafe loops
            # to exact identity and recheck every pair relationship below.
            if compound and loop_structure_preserved and (
                    topology["preserved"] or include_independently_valid_loop_candidates):
                for loop_index, (loop_fit, loop_error) in enumerate(zip(
                        fitted.get("contours", []), error.get("loops", []))):
                    loop_contract = _error_contract_evidence(
                        _loop_error_evidence(loop_error), budget)
                    loop_within_budget = loop_contract.pop("within_budget")
                    loop_primitive = _primitive_complexity(loop_fit)
                    loop_reasons = (
                        [] if loop_within_budget
                        else ["geometric_error_budget_exceeded"])
                    loop_option = {
                        "candidate_id": (
                            f"loop_{loop_index:02d}_{candidate_id}"),
                        "source": "curve_refit_per_loop_tolerance",
                        "loop_index": int(loop_index),
                        "tolerance_percent_of_bbox_diagonal": _round(
                            tolerance_percent),
                        "tolerance_absolute": _round(tolerance_absolute),
                        "fit": loop_fit,
                        "eligible": bool(loop_within_budget),
                        "within_error_budget": bool(loop_within_budget),
                        "anchors_after": int(loop_fit["anchor_count"]),
                        "segment_count": int(loop_fit["segment_count"]),
                        "fragment_count": 1,
                        "primitive_complexity": loop_primitive,
                        "designer_anchor_count": loop_primitive[
                            "designer_anchor_count"],
                        "actual_max_error_percent": _round(
                            loop_error["max_percent"]),
                        "actual_p95_error_percent": _round(
                            loop_error["p95_percent"]),
                        "over_budget_share": _round(
                            loop_error.get("over_budget_share", 1.0)),
                        "salient_corner_max_percent": _round(
                            loop_error.get(
                                "salient_corner_max_percent", float("inf"))),
                        "error_contract": loop_contract,
                        "rejection_reasons": loop_reasons,
                    }
                    if loop_index == largest_loop_index:
                        largest_loop_frontier_entries.append(
                            _compact_loop_frontier_entry(
                                loop_option,
                                source_point_count=len(contours[loop_index])))
                        largest_frontier_recorded = True
                    loop_path = loop_fit.get("path", "")
                    previous = loop_fit_options[loop_index].get(loop_path)
                    if previous is None or _selection_key(loop_option) < \
                            _selection_key(previous):
                        loop_fit_options[loop_index][loop_path] = loop_option
            if compound and not largest_frontier_recorded:
                frontier_reasons = list(reasons)
                if not frontier_reasons:
                    frontier_reasons.append("loop_fit_unavailable")
                unavailable = {
                    "candidate_id": (
                        f"loop_{largest_loop_index:02d}_{candidate_id}"),
                    "source": "curve_refit_per_loop_tolerance",
                    "tolerance_percent_of_bbox_diagonal": _round(
                        tolerance_percent),
                    "eligible": False,
                    "rejection_reasons": frontier_reasons,
                }
                largest_loop_frontier_entries.append(
                    _compact_loop_frontier_entry(
                        unavailable,
                        source_point_count=len(
                            contours[largest_loop_index])))
                largest_frontier_recorded = True
        except (RuntimeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            row.update({
                "fit_succeeded": False,
                "eligible": False,
                "within_error_budget": False,
                "rejection_reasons": ["curve_refit_failed"],
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
            })
            if compound and not largest_frontier_recorded:
                unavailable = {
                    "candidate_id": (
                        f"loop_{largest_loop_index:02d}_{candidate_id}"),
                    "source": "curve_refit_per_loop_tolerance",
                    "tolerance_percent_of_bbox_diagonal": _round(
                        tolerance_percent),
                    "eligible": False,
                    "rejection_reasons": ["curve_refit_failed"],
                }
                largest_loop_frontier_entries.append(
                    _compact_loop_frontier_entry(
                        unavailable,
                        source_point_count=len(
                            contours[largest_loop_index])))
        rows.append(row)

    identity_id = "source_identity"
    identity = _identity_fit(contours, closed_flags, compound=compound)
    zero_error = {
        "method": "exact_source_identity",
        "approximate": False,
        "normalization_basis": "source_bbox_diagonal",
        "normalization_scale": _round(scale),
        "source_bbox": bbox,
        "max_absolute": 0.0,
        "p95_absolute": 0.0,
        "max_percent": 0.0,
        "p95_percent": 0.0,
        "over_budget_share": 0.0,
        "over_budget_sample_count": 0,
        "salient_corner_count": 0,
        "salient_corner_max_absolute": 0.0,
        "salient_corner_max_percent": 0.0,
        "source_to_fit": {"max_absolute": 0.0, "p95_absolute": 0.0,
                          "sample_count": anchors_before},
        "fit_to_source": {"max_absolute": 0.0, "p95_absolute": 0.0,
                          "sample_count": anchors_before},
        "sampling": None,
        "loops": [],
    }

    if compound:
        per_loop_selection = []
        mixed_loop_fits = []
        selected_loop_options = []
        identity_loop_options = []
        for loop_index, (points, identity_loop) in enumerate(zip(
                contours, identity.get("contours", []))):
            identity_primitive = _primitive_complexity(identity_loop)
            identity_option = {
                "candidate_id": f"loop_{loop_index:02d}_source_identity",
                "source": "source_identity_per_loop_rollback",
                "loop_index": int(loop_index),
                "tolerance_percent_of_bbox_diagonal": 0.0,
                "tolerance_absolute": 0.0,
                "fit": identity_loop,
                "eligible": True,
                "within_error_budget": True,
                "anchors_after": int(len(points)),
                "segment_count": int(len(points)),
                "fragment_count": 1,
                "primitive_complexity": identity_primitive,
                "designer_anchor_count": int(len(points)),
                "actual_max_error_percent": 0.0,
                "actual_p95_error_percent": 0.0,
                "over_budget_share": 0.0,
                "salient_corner_max_percent": 0.0,
                "error_contract": {
                    "p95_within_budget": True,
                    "max_within_three_times_budget": True,
                    "over_budget_share_at_most_5_percent": True,
                    "salient_corner_max_within_two_times_budget": True,
                },
                "rejection_reasons": [],
            }
            identity_loop_options.append(identity_option)
            if loop_index == largest_loop_index:
                largest_loop_frontier_entries.append(
                    _compact_loop_frontier_entry(
                        identity_option,
                        source_point_count=len(points)))
            eligible_loop_options = [
                item for item in loop_fit_options[loop_index].values()
                if item.get("eligible")]
            chosen = min(
                eligible_loop_options + [identity_option],
                key=_selection_key)
            selected_loop_options.append(chosen)
            mixed_loop_fits.append(chosen["fit"])
            per_loop_selection.append({
                "loop_index": int(loop_index),
                "source_point_count": int(len(points)),
                "selected_candidate_id": chosen["candidate_id"],
                "selected_source": chosen["source"],
                "selected_tolerance_percent_of_bbox_diagonal": chosen[
                    "tolerance_percent_of_bbox_diagonal"],
                "anchors_after": int(chosen["anchors_after"]),
                "designer_anchor_count": int(
                    chosen["designer_anchor_count"]),
                "primitive_rank": int(
                    chosen["primitive_complexity"]["rank"]),
                "primitive_category": chosen[
                    "primitive_complexity"]["category"],
                "selected_fit_input_point_count": int(
                    chosen["fit"].get("input_point_count", len(points))),
                "selected_fit_source_identity": bool(
                    chosen["fit"].get("source_identity", False)),
                "actual_p95_error_percent": chosen[
                    "actual_p95_error_percent"],
                "actual_max_error_percent": chosen[
                    "actual_max_error_percent"],
                "over_budget_share": chosen["over_budget_share"],
                "salient_corner_max_percent": chosen[
                    "salient_corner_max_percent"],
                "identity_rollback": bool(
                    chosen["source"] == "source_identity_per_loop_rollback"),
                "identity_rollback_point_count": int(
                    len(points)
                    if chosen["source"]
                    == "source_identity_per_loop_rollback"
                    else 0),
            })

        if any(not item["identity_rollback"]
               for item in per_loop_selection):
            mixed_id = "curve_refit_mixed_loops"
            try:
                mixed_fit = _combine_compound_loop_fits(mixed_loop_fits)
                mixed_topology = _topology_evidence(
                    mixed_fit, closed_flags=closed_flags, compound=True)
                mixed_path_key = mixed_fit.get("path", "")
                if mixed_path_key not in metric_cache:
                    metric_cache[mixed_path_key] = measure_fit_error(
                        contours, mixed_fit,
                        normalization_scale=scale,
                        sampling_step_percent=sampling_step_percent,
                        max_samples_per_segment=max_samples_per_segment,
                        error_budget_percent=budget, _relationship_cache=metric_cache)
                mixed_error = metric_cache[mixed_path_key]
                _add_relationship_topology(mixed_topology, mixed_error)
                mixed_contract = _error_contract_evidence(
                    mixed_error, budget)
                mixed_within_budget = mixed_contract.pop("within_budget")
                mixed_primitive = _primitive_complexity(mixed_fit)
                mixed_designer_reduction = bool(
                    mixed_primitive["designer_anchor_count"]
                    < anchors_before)
                mixed_row = {
                    "candidate_id": mixed_id,
                    "source": "curve_refit_per_loop_mixed_tolerance",
                    "tolerance_percent_of_bbox_diagonal": None,
                    "tolerance_absolute": None,
                    "fit_succeeded": True,
                    "topology": mixed_topology,
                    "within_error_budget": bool(mixed_within_budget),
                    "eligible": bool(
                        mixed_topology["preserved"]
                        and mixed_within_budget
                        and mixed_designer_reduction),
                    "anchors_after": int(mixed_fit["anchor_count"]),
                    "segment_count": int(mixed_fit["segment_count"]),
                    "fragment_count": int(len(contours)),
                    "primitive_complexity": mixed_primitive,
                    "designer_anchor_count": mixed_primitive[
                        "designer_anchor_count"],
                    "designer_anchor_reduction_required": True,
                    "designer_anchor_reduction_achieved": (
                        mixed_designer_reduction),
                    "error_contract": mixed_contract,
                    "per_loop_selection": per_loop_selection,
                    "global_remeasurement_after_loop_merge": True,
                    **_candidate_error_row(mixed_error),
                }
                mixed_reasons = []
                if not mixed_topology["preserved"]:
                    mixed_reasons.append("topology_not_preserved")
                if not mixed_within_budget:
                    mixed_reasons.append("geometric_error_budget_exceeded")
                if not mixed_designer_reduction:
                    mixed_reasons.append("no_designer_anchor_reduction")
                mixed_row["rejection_reasons"] = mixed_reasons
                rows.append(mixed_row)
                candidate_fits[mixed_id] = (mixed_fit, mixed_error)
            except (RuntimeError, TypeError, ValueError,
                    np.linalg.LinAlgError) as exc:
                rows.append({
                    "candidate_id": mixed_id,
                    "source": "curve_refit_per_loop_mixed_tolerance",
                    "fit_succeeded": False,
                    "eligible": False,
                    "within_error_budget": False,
                    "per_loop_selection": per_loop_selection,
                    "global_remeasurement_after_loop_merge": True,
                    "rejection_reasons": ["mixed_loop_refit_failed"],
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc),
                })

    rows.append({
        "candidate_id": identity_id,
        "source": "source_identity_rollback",
        "tolerance_percent_of_bbox_diagonal": 0.0,
        "tolerance_absolute": 0.0,
        "fit_succeeded": True,
        "topology": {
            "preserved": True,
            "loop_count_preserved": True,
            "closure_preserved": True,
            "evenodd_preserved": True,
            "nonempty_geometry": True,
        },
        "within_error_budget": True,
        "eligible": True,
        "anchors_after": anchors_before,
        "segment_count": segments_before,
        "fragment_count": int(len(contours)),
        "primitive_complexity": {
            # Identity is a general path, not analytic line/arc geometry.  Its
            # rank can only break a tie after all economy measures are equal.
            "rank": 3,
            "category": "source_polyline_rollback",
            "native_primitives": [],
            "designer_anchor_count": anchors_before,
        },
        "designer_anchor_count": anchors_before,
        "error_contract": {
            "p95_within_budget": True,
            "max_within_three_times_budget": True,
            "over_budget_share_at_most_5_percent": True,
            "salient_corner_max_within_two_times_budget": True,
        },
        "actual_max_error_percent": 0.0,
        "actual_p95_error_percent": 0.0,
        "actual_max_error_absolute": 0.0,
        "actual_p95_error_absolute": 0.0,
        "source_to_fit_max_absolute": 0.0,
        "fit_to_source_max_absolute": 0.0,
        "over_budget_share": 0.0,
        "salient_corner_count": 0,
        "salient_corner_max_percent": 0.0,
        "rejection_reasons": [],
    })
    candidate_fits[identity_id] = (identity, zero_error)

    eligible = [row for row in rows if row.get("eligible")]
    # Hard topology/error eligibility is applied before this key.  Economy is
    # authoritative; primitive category can only break an exact economy tie.
    # Fragment count is included for the future case where a proposal may
    # consolidate loops; here topology preservation normally keeps it equal.
    selected_row = min(eligible, key=_selection_key)
    selected_fit, selected_error = candidate_fits[selected_row["candidate_id"]]
    anchors_after = int(selected_row["anchors_after"])
    segments_after = int(selected_row["segment_count"])
    removed = anchors_before - anchors_after
    identity_rollback_selected = bool(
        selected_row["candidate_id"] == identity_id)
    largest_loop_frontier = (
        largest_loop_frontier_entries if compound else [])
    mixed_identity_summary = _selected_mixed_identity_summary(selected_row)
    result = {
        "schema_version": 1,
        "status": (
            "identity_rollback_no_safe_reduction"
            if identity_rollback_selected
            else "selected_within_geometry_budget"),
        "identity_rollback_selected": identity_rollback_selected,
        "safe_refit_selected": bool(not identity_rollback_selected),
        "selection_outcome": (
            "source_identity_rollback"
            if identity_rollback_selected
            else "certified_geometry_refit"),
        "optimization_basis": "geometry_only",
        "uses_colour_or_pixel_similarity": False,
        "lexicographic_objective": [
            "preserve_topology_hard_constraint",
            "p95_bidirectional_geometric_error_percent_within_budget_hard_constraint",
            "maximum_error_within_three_times_budget_hard_tail_constraint",
            "over_budget_share_at_most_5_percent_hard_tail_constraint",
            "salient_corner_error_within_two_times_budget_hard_constraint",
            "minimize_designer_anchor_count",
            "minimize_anchor_count",
            "minimize_fragment_count",
            "minimize_segment_count",
            "prefer_simpler_primitive_category_when_economy_equal",
            "minimize_actual_max_error_percent_tiebreak",
            "minimize_actual_p95_error_percent_tiebreak",
        ],
        "error_budget_percent": _round(budget),
        "error_contract": {
            "primary": "p95_error_percent <= error_budget_percent",
            "tail_max": "max_error_percent <= 3 * error_budget_percent",
            "tail_share": "share(error > budget) <= 0.05",
            "salient_corners": (
                "corner_error_percent <= 2 * error_budget_percent"
            ),
            "each_loop": "the same p95/tail/share/corner contract in that loop's own bbox scale",
            "compound_relationships": "sampled polyline nesting/crossing/contact relationships preserved",
        },
        "normalization": {
            "basis": "source_bbox_diagonal",
            "scale": _round(scale),
            "source_bbox": bbox,
        },
        "closed": bool(closed_flags[0]) if not compound else True,
        "compound": bool(compound),
        "loop_count": int(len(contours)),
        "fill_rule": "evenodd" if compound else None,
        "path": selected_fit["path"],
        "fit": selected_fit,
        "selected_candidate_id": selected_row["candidate_id"],
        "selected_source": selected_row["source"],
        "actual_error": selected_error,
        "actual_max_error_percent": selected_row["actual_max_error_percent"],
        "actual_p95_error_percent": selected_row["actual_p95_error_percent"],
        "over_budget_share": selected_row["over_budget_share"],
        "salient_corner_max_percent": selected_row[
            "salient_corner_max_percent"],
        "primitive_complexity": selected_row["primitive_complexity"],
        "designer_anchor_count": selected_row["designer_anchor_count"],
        "anchors_before": anchors_before,
        "anchors_after": anchors_after,
        "anchors_removed": int(removed),
        "anchor_reduction_ratio": _round(
            removed / float(anchors_before) if anchors_before else 0.0),
        "segment_count_before": segments_before,
        "segment_count_after": segments_after,
        "candidate_count": int(len(rows)),
        "eligible_candidate_count": int(len(eligible)),
        "tolerance_grid_percent": [_round(value) for value in grid],
        "candidates": rows,
        "largest_loop_frontier": largest_loop_frontier,
        "selected_mixed_identity_summary": mixed_identity_summary,
        "evidence_note": (
            "Candidate ranking uses topology and normalised geometric error; "
            "no colour or pixel-fidelity value participates in optimisation. "
            "An identity rollback is explicitly marked and is not evidence "
            "of a successful geometry refit."
        ),
    }
    if compound and include_refinement_frontier:
        result["refinement_frontier"] = _build_compound_refinement_frontier(
            contours, closed_flags,
            base_loop_options=selected_loop_options,
            loop_fit_options=loop_fit_options,
            identity_loop_options=identity_loop_options,
            identity=identity, zero_error=zero_error,
            budget=budget, scale=scale,
            sampling_step_percent=sampling_step_percent,
            max_samples_per_segment=max_samples_per_segment,
            metric_cache=metric_cache,
            selected_candidate_id=selected_row["candidate_id"],
            selected_path=selected_fit["path"])
    return result


def optimize_curve(points, *, closed=False, error_budget_percent=0.25,
                   tolerance_percents=None, sampling_step_percent=0.02,
                   max_samples_per_segment=4096, **fitter_options):
    """Select the economy-minimal open/closed fit inside an error budget."""
    if not isinstance(closed, (bool, np.bool_)):
        raise TypeError("closed must be boolean")
    contour = _normalise_contour(points, closed=bool(closed))
    return _optimise(
        [contour], [bool(closed)], compound=False,
        error_budget_percent=error_budget_percent,
        tolerance_percents=tolerance_percents,
        sampling_step_percent=sampling_step_percent,
        max_samples_per_segment=max_samples_per_segment,
        fitter_options=fitter_options)


def optimize_compound_contours(contours: Iterable[Sequence], *,
                               error_budget_percent=0.25,
                               tolerance_percents=None,
                               sampling_step_percent=0.02,
                               max_samples_per_segment=4096,
                               include_refinement_frontier=False,
                               include_independently_valid_loop_candidates=False,
                               **fitter_options):
    """Select the economy-minimal even-odd fit while preserving every loop.

    Independent loop candidates from rejected whole fits are opt-in: callers
    must compare the complete unchanged scene and resulting SVG before commit.
    Geometry validity alone does not protect contact with other paint objects.
    """
    if not isinstance(include_refinement_frontier, (bool, np.bool_)):
        raise TypeError("include_refinement_frontier must be boolean")
    if not isinstance(include_independently_valid_loop_candidates, (bool, np.bool_)):
        raise TypeError("include_independently_valid_loop_candidates must be boolean")
    try:
        contour_list = list(contours)
    except TypeError as exc:
        raise TypeError("contours must be an iterable of closed contours") from exc
    if not contour_list:
        raise ValueError("contours must not be empty")
    normalised = [_normalise_contour(item, closed=True)
                  for item in contour_list]
    return _optimise(
        normalised, [True] * len(normalised), compound=True,
        error_budget_percent=error_budget_percent,
        tolerance_percents=tolerance_percents,
        sampling_step_percent=sampling_step_percent,
        max_samples_per_segment=max_samples_per_segment,
        fitter_options=fitter_options,
        include_refinement_frontier=bool(include_refinement_frontier),
        include_independently_valid_loop_candidates=bool(include_independently_valid_loop_candidates))


# American spellings for integration code that already uses ``optimize``.
optimise_curve = optimize_curve
optimise_compound_contours = optimize_compound_contours


__all__ = [
    "DEFAULT_TOLERANCE_PERCENTS",
    "measure_fit_error",
    "optimize_curve",
    "optimize_compound_contours",
    "optimise_curve",
    "optimise_compound_contours",
]
