"""Conservative compound-loop relationship checks in vector coordinates.

Checks use the same adaptively sampled polylines as the geometry error gate;
they are numerical evidence, not a proof about unsampled Bezier extrema.
"""
from __future__ import annotations

import numpy as np


def _closed(points):
    points = np.asarray(points, dtype=np.float64)
    if np.linalg.norm(points[0] - points[-1]) > 1e-10:
        points = np.vstack((points, points[0]))
    return points


def _inside(point, polygon):
    a, b = polygon[:-1], polygon[1:]
    crossed = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    a, b = a[crossed], b[crossed]
    if not len(a):
        return False
    intersections = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(intersections > point[0]) % 2)


def _intersection(first, second, epsilon):
    if len(first) > len(second):
        first, second = second, first
    starts, ends = second[:-1], second[1:]
    low, high = np.minimum(starts, ends), np.maximum(starts, ends)
    # Most nested-loop pairs have overlapping *loop* bounds but no nearby
    # boundary segments. Prune against the entire smaller loop once before
    # considering individual edges (exact broad phase, no geometric change).
    first_low, first_high = first.min(axis=0) - epsilon, first.max(axis=0) + epsilon
    relevant = ((high[:, 0] >= first_low[0]) & (high[:, 1] >= first_low[1])
                & (low[:, 0] <= first_high[0]) & (low[:, 1] <= first_high[1]))
    if not relevant.any():
        return None
    starts, ends, low, high = starts[relevant], ends[relevant], low[relevant], high[relevant]
    touched = False
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for p, q in zip(first[:-1], first[1:]):
        edge_low, edge_high = np.minimum(p, q) - epsilon, np.maximum(p, q) + epsilon
        selected = ((high[:, 0] >= edge_low[0]) & (high[:, 1] >= edge_low[1])
                    & (low[:, 0] <= edge_high[0]) & (low[:, 1] <= edge_high[1]))
        if not selected.any():
            continue
        a, b = starts[selected], ends[selected]
        one, two = cross(q - p, a - p), cross(q - p, b - p)
        three, four = cross(b - a, p - a), cross(b - a, q - a)
        if np.any((one * two < -epsilon ** 2) & (three * four < -epsilon ** 2)):
            return "crossing"
        if np.any((one * two <= epsilon ** 2) & (three * four <= epsilon ** 2)):
            touched = True
    return "touching" if touched else None


def loop_relationship_signature(contours):
    polygons = [_closed(points) for points in contours]
    bounds = [(points.min(axis=0), points.max(axis=0)) for points in polygons]
    scale = max(1., *(float(np.linalg.norm(hi - lo)) for lo, hi in bounds))
    epsilon = scale * 1e-10
    degenerate = [index for index, p in enumerate(polygons)
                  if abs(float(np.sum(p[:-1, 0] * p[1:, 1] - p[1:, 0] * p[:-1, 1]))) <= epsilon ** 2]
    relations = []
    for first in range(len(polygons)):
        for second in range(first + 1, len(polygons)):
            low_a, high_a = bounds[first]
            low_b, high_b = bounds[second]
            if np.any(high_a < low_b - epsilon) or np.any(high_b < low_a - epsilon):
                relation = "disjoint"
            else:
                relation = _intersection(polygons[first], polygons[second], epsilon)
                if relation is None:
                    a_in_b = _inside(polygons[first][0], polygons[second])
                    b_in_a = _inside(polygons[second][0], polygons[first])
                    relation = ("first_inside_second" if a_in_b else
                                "second_inside_first" if b_in_a else "disjoint")
            relations.append([first, second, relation])
    return {"loop_count": len(polygons), "degenerate_loop_indices": degenerate,
            "pair_relations": relations}


def relationship_evidence(source, candidate):
    return {"method": "sampled_polyline_pair_relationships",
            "approximate": True,
            "preserved": source == candidate,
            "source": source, "candidate": candidate,
            "self_intersection_proof": False}
