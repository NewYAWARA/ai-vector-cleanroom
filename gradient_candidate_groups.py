# -*- coding: utf-8 -*-
"""Deterministic source-space candidates for gradient-object fitting.

The original gradient detector joined every palette component connected by a
plausibly smooth boundary and fitted the resulting transitive closure once.
That is brittle: one weak bridge can join two independently useful ramps, and
one failed fit then discards all of their good sub-groups.

This module deliberately does *not* decide whether a gradient model is good.
It emits overlapping, de-duplicated alternatives (pairs, monotonic chains,
weak-bridge communities and broad chromatic source regions) for the model
fitter to validate transactionally.  Pixel work is linear or sort-based; no
component-count-squared image allocation is used.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np

KIND_PRIORITY = {
    "smooth_field": 0,
    "community": 1,
    "monotonic_chain": 2,
    "pair": 3,
    "source_chromatic": 4,
}


def _connected_components_4(mask):
    """Run-based strict four-connected labelling for vector ownership masks.

    The shared historical helper joins diagonally touching runs despite its
    older four-connectivity comment.  That behaviour is useful to some legacy
    stroke heuristics but is wrong for filled vector objects: marching squares
    emits two loops for such a contact.  Keep the corrected policy local so no
    unrelated Beta.5 behaviour changes underneath this stage.
    """
    source = np.asarray(mask, dtype=np.bool_)
    height, width = source.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = []

    def find(value):
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != root:
            parent[value], value = root, parent[value]
        return root

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
        return min(left, right)

    previous_runs = []
    for y in range(height):
        indices = np.flatnonzero(source[y])
        if indices.size == 0:
            previous_runs = []
            continue
        splits = np.flatnonzero(np.diff(indices) > 1)
        starts = np.concatenate(([indices[0]], indices[splits + 1]))
        ends = np.concatenate((indices[splits], [indices[-1]]))
        runs = []
        # Both run lists are ordered, disjoint intervals.  Advance past old
        # runs that can no longer overlap, then inspect only the remaining
        # interval window.  Overlaps are still visited left-to-right, preserving
        # the exact first-label and union call order of the former nested scan.
        previous_cursor = 0
        for start, end in zip(starts, ends):
            while (previous_cursor < len(previous_runs)
                   and previous_runs[previous_cursor][1] < start):
                previous_cursor += 1
            label = -1
            old_index = previous_cursor
            while (old_index < len(previous_runs)
                   and previous_runs[old_index][0] <= end):
                old_start, old_end, old_label = previous_runs[old_index]
                # Strict vertical overlap: a one-pixel diagonal is separate.
                if old_start <= end and old_end >= start:
                    label = (union(label, old_label) if label != -1
                             else find(old_label))
                old_index += 1
            if label == -1:
                label = len(parent)
                parent.append(label)
            labels[y, start:end + 1] = label + 1
            runs.append((start, end, label))
        previous_runs = runs
    if not parent:
        return labels, 0
    roots = np.asarray([find(index) for index in range(len(parent))],
                       dtype=np.int32)
    unique = np.unique(roots)
    remap = np.zeros(len(parent) + 1, dtype=np.int32)
    remap[1:] = np.searchsorted(unique, roots) + 1
    return remap[labels], int(len(unique))


def _as_inputs(den, lab_all, visible, palette):
    rgb = np.asarray(den)
    labels = np.asarray(lab_all)
    vis = np.asarray(visible, dtype=bool)
    pal = np.asarray(palette, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("den must have shape (height, width, 3)")
    if labels.shape != rgb.shape[:2] or vis.shape != rgb.shape[:2]:
        raise ValueError("lab_all and visible must match den's first two axes")
    if pal.ndim != 2 or pal.shape[1] != 3 or len(pal) == 0:
        raise ValueError("palette must have shape (colors, 3)")
    return (rgb.astype(np.float32, copy=False),
            labels.astype(np.int32, copy=False), vis, pal)


def _palette_components(labels, visible, palette_count, min_area):
    """Return a global component map and compact 1-indexed metadata arrays."""
    height, width = labels.shape
    comp_map = np.zeros((height, width), dtype=np.int32)
    comp_label = [0]
    comp_area = [0]
    comp_bbox = [(0, 0, 0, 0)]
    comp_centroid = [(0.0, 0.0)]

    next_id = 0
    for palette_id in range(int(palette_count)):
        mask = visible & (labels == palette_id)
        if not mask.any():
            continue
        local, count = _connected_components_4(mask)
        if count <= 0:
            continue
        ys, xs = np.nonzero(local)
        local_ids = local[ys, xs]
        counts = np.bincount(local_ids, minlength=count + 1)
        sum_x = np.bincount(local_ids, weights=xs, minlength=count + 1)
        sum_y = np.bincount(local_ids, weights=ys, minlength=count + 1)
        min_x = np.full(count + 1, width, dtype=np.int32)
        min_y = np.full(count + 1, height, dtype=np.int32)
        max_x = np.full(count + 1, -1, dtype=np.int32)
        max_y = np.full(count + 1, -1, dtype=np.int32)
        np.minimum.at(min_x, local_ids, xs)
        np.minimum.at(min_y, local_ids, ys)
        np.maximum.at(max_x, local_ids, xs)
        np.maximum.at(max_y, local_ids, ys)
        local_to_global = np.zeros(count + 1, dtype=np.int32)
        for local_id in range(1, count + 1):
            area = int(counts[local_id])
            if area < int(min_area):
                # Tiny islands remain unassigned.  They are unsuitable as a
                # gradient band and must not create accidental graph bridges.
                continue
            next_id += 1
            local_to_global[local_id] = next_id
            comp_label.append(int(palette_id))
            comp_area.append(area)
            comp_bbox.append((int(min_x[local_id]), int(min_y[local_id]),
                              int(max_x[local_id]) + 1,
                              int(max_y[local_id]) + 1))
            comp_centroid.append((float(sum_x[local_id] / area),
                                  float(sum_y[local_id] / area)))
        comp_map[ys, xs] = local_to_global[local_ids]

    return {
        "map": comp_map,
        "label": np.asarray(comp_label, dtype=np.int32),
        "area": np.asarray(comp_area, dtype=np.int64),
        "bbox": comp_bbox,
        "centroid": np.asarray(comp_centroid, dtype=np.float64),
        "count": int(next_id),
    }


def _collect_boundary_samples(comp_map, rgb, smooth_delta):
    """Collect adjacent component pairs without an NC*NC dense table."""
    height, width = comp_map.shape
    chunks = []

    left = comp_map[:, :-1]
    right = comp_map[:, 1:]
    valid = (left > 0) & (right > 0) & (left != right)
    if valid.any():
        a = left[valid].astype(np.int64)
        b = right[valid].astype(np.int64)
        delta = np.abs(rgb[:, :-1] - rgb[:, 1:]).max(axis=2)[valid]
        chunks.append((a, b, delta.astype(np.float32, copy=False)))

    top = comp_map[:-1, :]
    bottom = comp_map[1:, :]
    valid = (top > 0) & (bottom > 0) & (top != bottom)
    if valid.any():
        a = top[valid].astype(np.int64)
        b = bottom[valid].astype(np.int64)
        delta = np.abs(rgb[:-1, :] - rgb[1:, :]).max(axis=2)[valid]
        chunks.append((a, b, delta.astype(np.float32, copy=False)))

    if not chunks:
        return []
    aa = np.concatenate([chunk[0] for chunk in chunks])
    bb = np.concatenate([chunk[1] for chunk in chunks])
    dd = np.concatenate([chunk[2] for chunk in chunks])
    lo = np.minimum(aa, bb)
    hi = np.maximum(aa, bb)
    base = int(comp_map.max()) + 1
    keys = lo * base + hi
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    dd = dd[order]
    unique, starts = np.unique(keys, return_index=True)

    result = []
    for index, key in enumerate(unique):
        start = int(starts[index])
        end = int(starts[index + 1]) if index + 1 < len(starts) else len(keys)
        values = dd[start:end]
        a, b = divmod(int(key), base)
        result.append({
            "a": int(a),
            "b": int(b),
            "shared_boundary": int(len(values)),
            "source_delta_mean": float(values.mean()),
            "source_delta_p90": float(np.quantile(values, 0.90)),
            "smooth_fraction": float((values <= float(smooth_delta)).mean()),
        })
    return result


def _eligible_edges(boundaries, components, palette, *,
                    min_shared_boundary, smooth_delta,
                    min_smooth_fraction, min_palette_distance,
                    max_palette_distance):
    """Decorate and filter component edges using their source boundary."""
    comp_label = components["label"]
    comp_area = components["area"]
    edges = []
    for raw in boundaries:
        a, b = raw["a"], raw["b"]
        n = raw["shared_boundary"]
        if n < int(min_shared_boundary):
            continue
        ca, cb = int(comp_label[a]), int(comp_label[b])
        palette_distance = float(np.linalg.norm(palette[ca] - palette[cb]))
        if palette_distance < float(min_palette_distance):
            continue
        if palette_distance > float(max_palette_distance):
            continue
        mean_delta = float(raw["source_delta_mean"])
        p90_delta = float(raw["source_delta_p90"])
        smooth_fraction = float(raw["smooth_fraction"])
        if (smooth_fraction < float(min_smooth_fraction)
                or p90_delta > float(smooth_delta) * 1.8):
            continue

        balance = math.sqrt(float(min(comp_area[a], comp_area[b])) /
                            max(1.0, float(max(comp_area[a], comp_area[b]))))
        boundary_term = min(1.0, n / max(32.0, min_shared_boundary * 4.0))
        span_term = min(1.0, palette_distance / 48.0)
        score = (0.40 * smooth_fraction + 0.22 * boundary_term
                 + 0.20 * span_term + 0.18 * balance)
        edge = dict(raw)
        edge.update({
            "palette_distance": palette_distance,
            "smooth_fraction": smooth_fraction,
            "balance": balance,
            "strength": float(n * smooth_fraction),
            "score": float(score),
        })
        edges.append(edge)

    edges.sort(key=lambda edge: (-edge["score"],
                                 -edge["shared_boundary"],
                                 edge["a"], edge["b"]))
    return edges


def _color_geometry_evidence(component_ids, components, palette):
    """Summarize whether component colours form one spatial colour ramp."""
    ids = np.asarray(tuple(sorted(component_ids)), dtype=np.int32)
    areas = components["area"][ids].astype(np.float64)
    labels = components["label"][ids]
    colors = palette[labels].astype(np.float64)
    centers = components["centroid"][ids].astype(np.float64)
    weights = areas / max(1.0, float(areas.sum()))

    color_mean = np.sum(colors * weights[:, None], axis=0)
    color_centered = colors - color_mean
    color_cov = (color_centered * weights[:, None]).T @ color_centered
    eigvals, eigvecs = np.linalg.eigh(color_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(0.0, eigvals[order])
    color_axis = eigvecs[:, order[0]] if len(order) else np.array([1., 0., 0.])
    denom = max(1e-9, float(eigvals[0]))
    rank1_ratio = float((eigvals[1] + eigvals[2]) / denom)
    color_scalar = color_centered @ color_axis

    center_mean = np.sum(centers * weights[:, None], axis=0)
    center_centered = centers - center_mean
    spatial_cov = (center_centered * weights[:, None]).T @ center_centered
    seigvals, seigvecs = np.linalg.eigh(spatial_cov)
    sorder = np.argsort(seigvals)[::-1]
    spatial_axis = seigvecs[:, sorder[0]] if len(sorder) else np.array([1., 0.])
    spatial_scalar = center_centered @ spatial_axis
    if np.std(color_scalar) < 1e-6 or np.std(spatial_scalar) < 1e-6:
        correlation = 0.0
    else:
        correlation = abs(float(np.corrcoef(color_scalar, spatial_scalar)[0, 1]))

    pair_span = np.abs(colors[:, None, :] - colors[None, :, :]).max(axis=2)
    return {
        "palette_span": float(pair_span.max()) if pair_span.size else 0.0,
        "color_rank1_ratio": rank1_ratio,
        "spatial_color_correlation": correlation,
    }


def _group_edge_evidence(component_ids, edge_lookup, components, palette):
    ids = tuple(sorted(component_ids))
    id_set = set(ids)
    internal = [edge for (a, b), edge in edge_lookup.items()
                if a in id_set and b in id_set]
    external = [edge for (a, b), edge in edge_lookup.items()
                if (a in id_set) != (b in id_set)]
    color = _color_geometry_evidence(ids, components, palette)
    if internal:
        shared = sum(edge["shared_boundary"] for edge in internal)
        smooth = sum(edge["smooth_fraction"] * edge["shared_boundary"]
                     for edge in internal) / max(1, shared)
        weakest = min(edge["smooth_fraction"] for edge in internal)
        base_score = sum(edge["score"] * edge["shared_boundary"]
                         for edge in internal) / max(1, shared)
    else:
        shared = 0
        smooth = weakest = base_score = 0.0
    score = (0.72 * base_score
             + 0.16 * max(0.0, 1.0 - min(1.0, color["color_rank1_ratio"]))
             + 0.12 * color["spatial_color_correlation"])
    return {
        **color,
        "internal_edges": int(len(internal)),
        "shared_boundary": int(shared),
        "external_smooth_edges": int(len(external)),
        "external_smooth_shared_boundary": int(sum(
            edge["shared_boundary"] for edge in external)),
        "ownership_closed_over_smooth_graph": not external,
        "smooth_fraction": float(smooth),
        "weakest_smooth_fraction": float(weakest),
        "proposal_score": float(score),
    }


def _is_coherent(evidence, count):
    if count < 2 or evidence["palette_span"] < 10.0:
        return False
    if count == 2:
        return True
    # Repeated ramps joined end-to-start remain close to a colour line but
    # lose their monotonic spatial relationship.  Requiring both properties
    # prevents that transitive closure from being proposed as one object.
    return (evidence["color_rank1_ratio"] <= 0.34
            and evidence["spatial_color_correlation"] >= 0.52)


def _path_reversal_count(component_ids, edges, components, palette):
    """Return monotonic colour reversals for a simple graph path, else None."""
    ids = tuple(sorted(component_ids))
    id_set = set(ids)
    graph = defaultdict(list)
    for edge in edges:
        if edge["a"] in id_set and edge["b"] in id_set:
            graph[edge["a"]].append(edge["b"])
            graph[edge["b"]].append(edge["a"])
    endpoints = sorted(node for node in ids if len(graph[node]) == 1)
    if len(endpoints) != 2 or any(len(graph[node]) > 2 for node in ids):
        return None
    ordered = [endpoints[0]]
    previous = 0
    current = endpoints[0]
    while len(ordered) < len(ids):
        remaining = [node for node in graph[current] if node != previous]
        if not remaining:
            return None
        previous, current = current, remaining[0]
        ordered.append(current)
    colors = palette[components["label"][np.asarray(ordered, dtype=np.int32)]]
    centered = colors - colors.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    scalar = colors @ (vh[0] if len(vh) else np.array([1.0, 0.0, 0.0]))
    signs = np.sign(np.diff(scalar))
    signs = signs[signs != 0]
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _smooth_field_specs(edges, components, palette, edge_lookup):
    """Offer complete closures of the eligible smooth-boundary graph.

    Pairs and short chains are useful evidence seeds but are often only two
    quantisation bands from a larger painted surface.  A closure has no smooth
    edge leading to an omitted palette component, so it is the first candidate
    that can honestly claim full ownership of that continuous field.  Raw
    transitive closure is still not trusted blindly: a simple path whose colour
    progression reverses is split/rejected by the existing alternatives, and
    every surviving closure must pass the independent source-space model gate.
    """
    count = int(components["count"])
    if count <= 1 or not edges:
        return []
    parent = np.arange(count + 1, dtype=np.int32)

    def find(value):
        value = int(value)
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    for edge in edges:
        left, right = find(edge["a"]), find(edge["b"])
        if left != right:
            parent[max(left, right)] = min(left, right)
    groups = defaultdict(list)
    for component_id in range(1, count + 1):
        groups[find(component_id)].append(component_id)

    specs = []
    for ids in groups.values():
        if len(ids) < 3:
            continue
        evidence = _group_edge_evidence(ids, edge_lookup, components, palette)
        reversals = _path_reversal_count(ids, edges, components, palette)
        evidence["monotonic_reversals"] = (
            int(reversals) if reversals is not None else None)
        evidence["ownership_mode"] = "complete_smooth_graph_closure"
        if not _is_coherent(evidence, len(ids)):
            continue
        if reversals not in (None, 0):
            continue
        specs.append(("smooth_field", tuple(sorted(ids)), evidence))
    return specs


def _community_specs(edges, components, palette, edge_lookup,
                     *, mutual_support=0.38):
    """Build weak-bridge-cut graph communities, never raw transitive closure."""
    count = components["count"]
    if count <= 1 or not edges:
        return []
    strongest = np.zeros(count + 1, dtype=np.float64)
    for edge in edges:
        strongest[edge["a"]] = max(strongest[edge["a"]], edge["strength"])
        strongest[edge["b"]] = max(strongest[edge["b"]], edge["strength"])

    parent = np.arange(count + 1, dtype=np.int32)

    def find(value):
        value = int(value)
        while int(parent[value]) != value:
            parent[value] = parent[int(parent[value])]
            value = int(parent[value])
        return value

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for edge in edges:
        a, b = edge["a"], edge["b"]
        ratio_a = edge["strength"] / max(1e-9, strongest[a])
        ratio_b = edge["strength"] / max(1e-9, strongest[b])
        edge["mutual_support"] = float(min(ratio_a, ratio_b))
        if min(ratio_a, ratio_b) >= float(mutual_support):
            union(a, b)

    groups = defaultdict(list)
    for component_id in range(1, count + 1):
        groups[find(component_id)].append(component_id)

    specs = []
    for ids in groups.values():
        if len(ids) < 3:
            continue
        evidence = _group_edge_evidence(ids, edge_lookup, components, palette)
        id_set = set(ids)
        path_graph = defaultdict(list)
        for edge in edges:
            if edge["a"] in id_set and edge["b"] in id_set:
                path_graph[edge["a"]].append(edge["b"])
                path_graph[edge["b"]].append(edge["a"])
        monotonic = True
        endpoints = sorted(node for node in ids if len(path_graph[node]) == 1)
        if (len(endpoints) == 2
                and all(len(path_graph[node]) <= 2 for node in ids)):
            ordered = [endpoints[0]]
            previous = 0
            current = endpoints[0]
            while len(ordered) < len(ids):
                nxt = [node for node in path_graph[current]
                       if node != previous]
                if not nxt:
                    break
                previous, current = current, nxt[0]
                ordered.append(current)
            colors = palette[components["label"][
                np.asarray(ordered, dtype=np.int32)]]
            centered = colors - colors.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            scalar = colors @ vh[0]
            signs = np.sign(np.diff(scalar))
            signs = signs[signs != 0]
            reversals = int(np.count_nonzero(signs[1:] != signs[:-1]))
            evidence["monotonic_reversals"] = reversals
            monotonic = reversals == 0
        else:
            evidence["monotonic_reversals"] = 0
        if _is_coherent(evidence, len(ids)) and monotonic:
            specs.append(("community", tuple(sorted(ids)), evidence))
    return specs


def _path_chain_specs(edges, components, palette, edge_lookup):
    """Split simple graph paths whenever their palette ramp reverses."""
    graph = defaultdict(list)
    for edge in edges:
        graph[edge["a"]].append(edge["b"])
        graph[edge["b"]].append(edge["a"])
    for node in graph:
        graph[node].sort()

    visited_edges = set()
    specs = []
    starts = sorted(node for node, neighbors in graph.items()
                    if len(neighbors) != 2)
    # Pure cycles have no endpoint.  They do not define a trustworthy ramp;
    # community proposals can still cover a coherent one.
    for start in starts:
        for first in graph[start]:
            key = tuple(sorted((start, first)))
            if key in visited_edges:
                continue
            path = [start, first]
            visited_edges.add(key)
            previous, current = start, first
            while len(graph[current]) == 2:
                next_nodes = [node for node in graph[current] if node != previous]
                if not next_nodes:
                    break
                nxt = next_nodes[0]
                edge_key = tuple(sorted((current, nxt)))
                if edge_key in visited_edges:
                    break
                path.append(nxt)
                visited_edges.add(edge_key)
                previous, current = current, nxt

            if len(path) < 3:
                continue
            colors = palette[components["label"][np.asarray(path, dtype=np.int32)]]
            centered = colors - colors.mean(axis=0)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0] if len(vh) else np.array([1.0, 0.0, 0.0])
            scalar = colors @ axis
            deltas = np.diff(scalar)
            signs = np.sign(deltas)
            # Numerical zero is a genuine break: eligible edges already have
            # a meaningful palette distance, so a zero principal projection
            # is not evidence for the same one-dimensional ramp.
            run_start = 0
            active_sign = 0.0
            for edge_index, sign in enumerate(signs):
                if sign == 0:
                    if edge_index + 1 - run_start >= 3:
                        ids = tuple(path[run_start:edge_index + 1])
                        evidence = _group_edge_evidence(ids, edge_lookup,
                                                        components, palette)
                        if _is_coherent(evidence, len(ids)):
                            specs.append(("monotonic_chain",
                                          tuple(sorted(ids)), evidence))
                    run_start = edge_index + 1
                    active_sign = 0.0
                    continue
                if active_sign == 0.0:
                    active_sign = sign
                elif sign != active_sign:
                    if edge_index + 1 - run_start >= 3:
                        ids = tuple(path[run_start:edge_index + 1])
                        evidence = _group_edge_evidence(ids, edge_lookup,
                                                        components, palette)
                        if _is_coherent(evidence, len(ids)):
                            specs.append(("monotonic_chain",
                                          tuple(sorted(ids)), evidence))
                    # The reversing edge is a bridge, not a member of either
                    # monotonic object.  Start the next run at its far end.
                    run_start = edge_index + 1
                    active_sign = 0.0
            if len(path) - run_start >= 3:
                ids = tuple(path[run_start:])
                evidence = _group_edge_evidence(ids, edge_lookup,
                                                components, palette)
                if _is_coherent(evidence, len(ids)):
                    specs.append(("monotonic_chain", tuple(sorted(ids)),
                                  evidence))
    return specs


def _bbox_for_mask(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _mask_digest(mask, bbox):
    x0, y0, x1, y1 = bbox
    packed = np.packbits(mask[y0:y1, x0:x1], bitorder="little")
    return hashlib.sha1(packed.tobytes()).hexdigest()


def _source_chromatic_specs(rgb, visible, comp_map, components, palette,
                            *, chroma_threshold, min_area):
    """Offer broad connected colour fields independently of palette bands."""
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = visible & (chroma >= float(chroma_threshold))
    labels, count = _connected_components_4(mask)
    specs = []
    for region_id in range(1, count + 1):
        region = labels == region_id
        area = int(region.sum())
        if area < int(min_area):
            continue
        ids = tuple(int(value) for value in np.unique(comp_map[region])
                    if int(value) > 0)
        if len(ids) < 2:
            continue
        evidence = _color_geometry_evidence(ids, components, palette)
        values = chroma[region]
        # Broad masks are intentionally lower-priority than boundary-backed
        # proposals.  The model fitter may split or reject them.
        evidence.update({
            "source_chroma_mean": float(values.mean()),
            "source_chroma_p10": float(np.quantile(values, 0.10)),
            "proposal_score": float(0.38
                                    + 0.12 * min(1.0, values.mean() / 80.0)
                                    + 0.10 * min(1.0, evidence["palette_span"] / 80.0)),
            "internal_edges": 0,
            "shared_boundary": 0,
            "smooth_fraction": 0.0,
            "weakest_smooth_fraction": 0.0,
        })
        specs.append(("source_chromatic", ids, evidence, region))
    return specs


def _overlap_metadata(candidates, component_areas, *,
                      minimum_fraction=0.05, limit=8):
    for candidate in candidates:
        candidate["parent_ids"] = []
        candidate["overlap_info"] = []

    for i, left in enumerate(candidates):
        lx0, ly0, lx1, ly1 = left["bbox"]
        for j in range(i + 1, len(candidates)):
            right = candidates[j]
            rx0, ry0, rx1, ry1 = right["bbox"]
            x0, y0 = max(lx0, rx0), max(ly0, ry0)
            x1, y1 = min(lx1, rx1), min(ly1, ry1)
            if x0 >= x1 or y0 >= y1:
                continue
            if left["_component_exact"] and right["_component_exact"]:
                shared_components = (set(left["component_ids"])
                                     & set(right["component_ids"]))
                intersection = int(sum(component_areas[value]
                                       for value in shared_components))
            else:
                # Broad source candidates are few and mutually disjoint; only
                # their intersections with component-union alternatives need
                # a cropped pixel operation.
                intersection = int(np.logical_and(
                    left["mask"][y0:y1, x0:x1],
                    right["mask"][y0:y1, x0:x1]).sum())
            if intersection <= 0:
                continue
            smaller = min(left["area"], right["area"])
            smaller_fraction = intersection / max(1, smaller)
            if smaller_fraction < float(minimum_fraction):
                continue
            left["overlap_info"].append({
                "candidate_id": right["candidate_id"],
                "intersection": intersection,
                "self_fraction": round(intersection / max(1, left["area"]), 5),
                "smaller_fraction": round(smaller_fraction, 5),
            })
            right["overlap_info"].append({
                "candidate_id": left["candidate_id"],
                "intersection": intersection,
                "self_fraction": round(intersection / max(1, right["area"]), 5),
                "smaller_fraction": round(smaller_fraction, 5),
            })
            if intersection == left["area"] and right["area"] > left["area"]:
                left["parent_ids"].append(right["candidate_id"])
            if intersection == right["area"] and left["area"] > right["area"]:
                right["parent_ids"].append(left["candidate_id"])

    area_by_id = {item["candidate_id"]: item["area"] for item in candidates}
    for candidate in candidates:
        candidate["parent_ids"].sort(key=lambda value: (area_by_id[value], value))
        candidate["overlap_info"].sort(
            key=lambda item: (-item["smaller_fraction"], item["candidate_id"]))
        del candidate["overlap_info"][int(limit):]
        del candidate["_component_exact"]


def propose_gradient_candidates(den, lab_all, visible=None, palette=None, *,
                                vis_fill=None, max_candidates=48,
                                min_component_area=24,
                                min_candidate_area=96,
                                min_shared_boundary=8,
                                smooth_delta=30.0,
                                min_smooth_fraction=0.55,
                                min_palette_distance=8.0,
                                max_palette_distance=170.0,
                                mutual_support=0.38,
                                chroma_threshold=14.0,
                                min_chromatic_area=320):
    """Return sorted, de-duplicated gradient-object mask proposals.

    Parameters are trace-space values.  ``visible`` and ``vis_fill`` are
    aliases; ``vis_fill`` wins when both are supplied.  Each returned mapping
    contains ``mask``, ``component_ids``, ``bbox`` (exclusive right/bottom),
    ``area``, ``kind``, ``score``, ``evidence``, ``parent_ids`` and compact
    ``overlap_info``.  Candidate ids and ordering are deterministic.

    This function proposes alternatives only.  A caller must still compare a
    fitted solid/linear/radial model against the exact flat baseline and roll
    back any candidate which worsens its held-out pixels or rendered image.
    """
    if vis_fill is not None:
        visible = vis_fill
    if visible is None:
        raise ValueError("visible or vis_fill is required")
    if palette is None:
        raise ValueError("palette is required")
    rgb, labels, visible, palette = _as_inputs(
        den, lab_all, visible, palette)
    components = _palette_components(labels, visible, len(palette),
                                     min_component_area)
    if components["count"] < 1:
        return []

    boundaries = _collect_boundary_samples(components["map"], rgb,
                                           smooth_delta)
    edges = _eligible_edges(
        boundaries, components, palette,
        min_shared_boundary=min_shared_boundary,
        smooth_delta=smooth_delta,
        min_smooth_fraction=min_smooth_fraction,
        min_palette_distance=min_palette_distance,
        max_palette_distance=max_palette_distance)
    edge_lookup = {(edge["a"], edge["b"]): edge for edge in edges}

    component_specs = []
    for edge in edges:
        ids = (edge["a"], edge["b"])
        evidence = _group_edge_evidence(ids, edge_lookup, components, palette)
        component_specs.append(("pair", tuple(sorted(ids)), evidence))
    component_specs.extend(_smooth_field_specs(
        edges, components, palette, edge_lookup))
    component_specs.extend(_community_specs(
        edges, components, palette, edge_lookup,
        mutual_support=mutual_support))
    component_specs.extend(_path_chain_specs(
        edges, components, palette, edge_lookup))

    # De-duplicate equal component unions.  Prefer coherent larger-object
    # evidence over the pair/chain aliases which happen to cover the same mask.
    best_by_components = {}
    for kind, ids, evidence in component_specs:
        ids = tuple(sorted(set(int(value) for value in ids)))
        if len(ids) < 2:
            continue
        area = int(components["area"][np.asarray(ids, dtype=np.int32)].sum())
        if area < int(min_candidate_area):
            continue
        key = ids
        choice = (kind, ids, evidence)
        old = best_by_components.get(key)
        if old is None:
            best_by_components[key] = choice
            continue
        old_rank = (KIND_PRIORITY.get(old[0], 99),
                    -float(old[2].get("proposal_score", 0.0)))
        new_rank = (KIND_PRIORITY.get(kind, 99),
                    -float(evidence.get("proposal_score", 0.0)))
        if new_rank < old_rank:
            best_by_components[key] = choice

    raw_candidates = []
    comp_map = components["map"]
    if components["count"] == 1:
        # Palette quantisation can collapse a real gentle ramp to one colour.
        # Source interior variation proposes a paint fit; it does not approve
        # a gradient. The independent heldout model/scene gates still decide.
        mask = comp_map == 1
        core = mask.copy()
        for _ in range(3):
            padded = np.pad(core, 1, constant_values=False)
            core = np.logical_and.reduce([
                padded[dy:dy + core.shape[0], dx:dx + core.shape[1]]
                for dy in range(3) for dx in range(3)])
        values = rgb[core]
        if len(values) >= max(96, int(min_candidate_area)):
            span = float(np.max(np.quantile(values, .90, axis=0)
                                - np.quantile(values, .10, axis=0)))
            if span >= 8.0:
                raw_candidates.append({
                    'kind': 'source_single_component', 'component_ids': (1,),
                    'mask': mask, 'bbox': _bbox_for_mask(mask), 'area': int(mask.sum()),
                    'score': .38, '_component_exact': True,
                    'evidence': {'source_interior_p90_p10_rgb_span': span,
                                 'source_interior_pixels': len(values),
                                 'palette_band_count': 1,
                                 'proposal_only_requires_heldout_paint_fit': True}})
    for kind, ids, evidence in best_by_components.values():
        mask = np.isin(comp_map, np.asarray(ids, dtype=np.int32))
        bbox = _bbox_for_mask(mask)
        raw_candidates.append({
            "kind": kind,
            "component_ids": ids,
            "mask": mask,
            "bbox": bbox,
            "area": int(mask.sum()),
            "score": float(evidence.get("proposal_score", 0.0)),
            "evidence": evidence,
            "_component_exact": True,
        })

    for kind, ids, evidence, mask in _source_chromatic_specs(
            rgb, visible, comp_map, components, palette,
            chroma_threshold=chroma_threshold,
            min_area=max(min_chromatic_area, min_candidate_area)):
        bbox = _bbox_for_mask(mask)
        raw_candidates.append({
            "kind": kind,
            "component_ids": ids,
            "mask": mask,
            "bbox": bbox,
            "area": int(mask.sum()),
            "score": float(evidence.get("proposal_score", 0.0)),
            "evidence": evidence,
            "_component_exact": False,
        })

    # Exact cropped-mask digest removes aliases across proposal families while
    # keeping memory proportional to each candidate's own bounding box.
    unique = {}
    for candidate in raw_candidates:
        digest = _mask_digest(candidate["mask"], candidate["bbox"])
        key = (candidate["bbox"], candidate["area"], digest)
        old = unique.get(key)
        rank = (-candidate["score"],
                KIND_PRIORITY.get(candidate["kind"], 99),
                candidate["component_ids"])
        if old is None:
            unique[key] = candidate
        else:
            old_rank = (-old["score"], KIND_PRIORITY.get(old["kind"], 99),
                        old["component_ids"])
            if rank < old_rank:
                unique[key] = candidate

    candidates = list(unique.values())
    sort_key = lambda item: (
        -round(item["score"], 12), -item["area"],
        KIND_PRIORITY.get(item["kind"], 99),
        item["component_ids"], item["bbox"])
    candidates.sort(key=sort_key)

    # A detailed logo may contain dozens of excellent pair edges.  Keeping
    # only the globally highest scores would starve the broad-object and chain
    # families before the model fitter ever sees them.  Reserve deterministic
    # family quotas, then fill unused capacity from the global ordering.
    capacity = max(0, int(max_candidates))
    if capacity == 0:
        return []
    pair_quota = max(1, int(math.floor(capacity * 0.50)))
    structural_quota = max(1, int(math.floor(capacity * 0.30)))
    broad_quota = max(1, capacity - pair_quota - structural_quota)
    selected = []
    selected_keys = set()

    def take(items, count, *, key=sort_key):
        taken = 0
        for item in sorted(items, key=key):
            identity = (item["bbox"], item["area"],
                        _mask_digest(item["mask"], item["bbox"]))
            if identity in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(identity)
            taken += 1
            if taken >= count:
                break

    pairs = [item for item in candidates if item["kind"] == "pair"]
    structural = [item for item in candidates
                  if item["kind"] in {
                      "smooth_field", "community", "monotonic_chain"}]
    broad = [item for item in candidates
             if item["kind"] == "source_chromatic"]
    # Broad candidates are source-object hypotheses; area is more useful than
    # the deliberately modest proposal score for choosing which ones survive.
    take(pairs, pair_quota)
    take(structural, structural_quota)
    take(broad, broad_quota,
         key=lambda item: (-item["area"], -item["score"],
                           item["component_ids"], item["bbox"]))
    for item in candidates:
        if len(selected) >= capacity:
            break
        identity = (item["bbox"], item["area"],
                    _mask_digest(item["mask"], item["bbox"]))
        if identity not in selected_keys:
            selected.append(item)
            selected_keys.add(identity)
    candidates = sorted(selected[:capacity], key=sort_key)
    for index, candidate in enumerate(candidates, 1):
        candidate["candidate_id"] = f"gradient-candidate-{index:04d}"
        candidate["score"] = round(float(candidate["score"]), 6)
        candidate["evidence"] = {
            key: (round(float(value), 6)
                  if isinstance(value, (float, np.floating)) else value)
            for key, value in candidate["evidence"].items()
        }
    _overlap_metadata(candidates, components["area"])
    # Shared in-memory context lets the reconstruction stage expand a
    # paint-valid seed over exact palette components without rebuilding the
    # graph or serialising a full component map into public reports.
    component_context = {
        "schema": "ai-vector-cleanroom.gradient-component-context/v1",
        "component_map": components["map"],
        "component_areas": components["area"],
        "component_labels": components["label"],
        "component_bboxes": tuple(components["bbox"]),
        "component_centroids": components["centroid"],
        "palette": palette,
        "eligible_edges": tuple(dict(edge) for edge in edges),
        "mutual_support_base": float(mutual_support),
    }
    for candidate in candidates:
        candidate["_component_context"] = component_context
    return candidates


__all__ = ["propose_gradient_candidates"]
