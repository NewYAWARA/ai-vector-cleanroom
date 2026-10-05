# -*- coding: utf-8 -*-
"""Fail-closed orchestration for native gradient-object reconstruction.

This module deliberately keeps *paint* and *geometry* as separate problems:

* :mod:`gradient_candidate_groups` proposes source-space ownership masks;
* :mod:`gradient_object_engine` decides whether a solid, linear, or radial
  paint model is supported by held-out colour evidence;
* a geometry optimiser then minimises anchors subject to a caller supplied
  maximum geometric-error percentage.  Colour error never asks the contour
  fitter to chase a pixel edge.

The stage does not mutate an SVG.  It returns a JSON-safe proposal containing
an SVG path and a compact, reversible mask RLE.  A production integration can
therefore render and transactionally accept or roll back the proposal without
re-running candidate discovery.  Overlapping alternatives are selected once,
deterministically, so a pair proposal cannot be painted on top of the broader
object that superseded it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing
import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from typing import Any, Callable, Optional

import numpy as np

from curve_refit import fit_mask
from gradient_candidate_groups import propose_gradient_candidates
from gradient_object_engine import fit_gradient_object_proposal

try:
    from geometry_error_optimizer import optimize_compound_contours
except ImportError:  # Portable fallback for an older unpacked engine.
    optimize_compound_contours = None


SCHEMA = "ai-vector-cleanroom.gradient-reconstruction-stage/v1"
MASK_SCHEMA = "ai-vector-cleanroom.mask-row-runs/v1"
GEOMETRY_FIT_CACHE_SCHEMA = (
    "ai-vector-cleanroom.gradient-geometry-fit-cache/v1")

_FAMILY_PRIORITY = {
    "source_chromatic": 0,
    "model_guided_field": 1,
    "smooth_field": 2,
    "community": 3,
    "monotonic_chain": 4,
    "pair": 5,
}


def new_geometry_fit_cache() -> dict[str, Any]:
    """Return a per-image cache for exact ownership-mask geometry fits."""
    return {
        "schema": GEOMETRY_FIT_CACHE_SCHEMA,
        "entries": {},
        "audit": {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "errors": 0,
        },
    }


def _normalise_geometry_fit_cache(cache):
    if cache is None:
        return None
    if not isinstance(cache, dict):
        raise TypeError("geometry_fit_cache must be a dict or None")
    if not cache:
        cache.update(new_geometry_fit_cache())
    if cache.get("schema") != GEOMETRY_FIT_CACHE_SCHEMA:
        raise ValueError("unsupported gradient geometry-fit cache schema")
    entries = cache.get("entries")
    audit = cache.get("audit")
    if not isinstance(entries, dict) or not isinstance(audit, dict):
        raise ValueError("malformed gradient geometry-fit cache")
    for name in ("requests", "hits", "misses", "stores", "errors"):
        audit[name] = int(audit.get(name, 0) or 0)
    return cache


def geometry_fit_cache_audit(cache) -> dict[str, Any]:
    state = _normalise_geometry_fit_cache(cache)
    if state is None:
        return {
            "schema": GEOMETRY_FIT_CACHE_SCHEMA,
            "scope": "disabled",
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "errors": 0,
            "entry_count": 0,
            "keys_sha256": [],
        }
    return {
        "schema": GEOMETRY_FIT_CACHE_SCHEMA,
        "scope": "single_process_one",
        **{name: int(state["audit"].get(name, 0) or 0)
           for name in ("requests", "hits", "misses", "stores", "errors")},
        "entry_count": len(state["entries"]),
        "keys_sha256": sorted(str(key) for key in state["entries"]),
    }


def _geometry_fit_cache_key(mask_digest, bbox, *, error_budget_percent,
                            smooth, max_segments):
    contract = {
        "schema": GEOMETRY_FIT_CACHE_SCHEMA,
        "stage_schema": SCHEMA,
        "mask_sha256": str(mask_digest),
        "bbox_xyxy": [int(value) for value in bbox],
        "error_budget_percent": float(error_budget_percent),
        "smooth": float(smooth),
        "max_segments": int(max_segments),
        # Cache lifetime is process-local.  Callable identities prevent reuse
        # across a monkeypatch/code reload even if all numeric options match.
        "fit_geometry_identity": id(_fit_geometry),
        "optimizer_identity": id(optimize_compound_contours),
        "fallback_identity": id(fit_mask),
    }
    encoded = json.dumps(
        contract, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest().upper()


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy values and tuples to strict JSON values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are not JSON-safe")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1,
            int(ys.max()) + 1)


def _mask_digest(mask: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"{mask.shape[0]}x{mask.shape[1]}:".encode("ascii"))
    digest.update(np.packbits(mask.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()


def encode_mask_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean mask as deterministic row runs inside its bbox."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    array = np.ascontiguousarray(array, dtype=np.bool_)
    height, width = array.shape
    x0, y0, x1, y1 = _bbox(array)
    runs: list[list[int]] = []
    if x1 > x0 and y1 > y0:
        cropped = array[y0:y1, x0:x1]
        for row_index, row in enumerate(cropped):
            padded = np.pad(row.astype(np.int8), (1, 1))
            changes = np.diff(padded)
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            for start, end in zip(starts, ends):
                runs.append([int(row_index), int(start), int(end - start)])
    return {
        "schema": MASK_SCHEMA,
        "image_size": [int(width), int(height)],
        "bbox_xyxy": [x0, y0, x1, y1],
        "pixel_count": int(array.sum()),
        "sha256": _mask_digest(array),
        "runs": runs,
    }


def decode_mask_rle(encoded: dict[str, Any]) -> np.ndarray:
    """Decode :func:`encode_mask_rle` output and verify its evidence."""
    if not isinstance(encoded, dict) or encoded.get("schema") != MASK_SCHEMA:
        raise ValueError("unsupported mask encoding")
    size = encoded.get("image_size")
    bbox = encoded.get("bbox_xyxy")
    if (not isinstance(size, list) or len(size) != 2
            or not isinstance(bbox, list) or len(bbox) != 4):
        raise ValueError("invalid mask dimensions")
    width, height = (int(size[0]), int(size[1]))
    x0, y0, x1, y1 = (int(value) for value in bbox)
    if (width < 0 or height < 0 or x0 < 0 or y0 < 0
            or x1 < x0 or y1 < y0 or x1 > width or y1 > height):
        raise ValueError("mask bbox is outside the image")
    mask = np.zeros((height, width), dtype=np.bool_)
    for raw in encoded.get("runs", []):
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("invalid row run")
        row, start, length = (int(value) for value in raw)
        yy, xx0, xx1 = y0 + row, x0 + start, x0 + start + length
        if (row < 0 or yy >= y1 or start < 0 or length <= 0 or xx1 > x1):
            raise ValueError("row run is outside the mask bbox")
        if mask[yy, xx0:xx1].any():
            raise ValueError("overlapping mask runs")
        mask[yy, xx0:xx1] = True
    if int(mask.sum()) != int(encoded.get("pixel_count", -1)):
        raise ValueError("mask pixel count does not match evidence")
    if _mask_digest(mask) != encoded.get("sha256"):
        raise ValueError("mask digest does not match evidence")
    return mask


def _native_primitive_evidence(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    direct = geometry.get("native_primitive")
    if isinstance(direct, dict):
        return [_json_safe(direct)]
    result = []
    fit = geometry.get("fit") if isinstance(geometry.get("fit"), dict) else geometry
    for contour in fit.get("contours", []) if isinstance(fit, dict) else []:
        native = contour.get("native_primitive") if isinstance(contour, dict) else None
        if isinstance(native, dict):
            result.append(_json_safe(native))
    return result


def _geometry_value(result: dict[str, Any], name: str, fallback: Any = None) -> Any:
    if name in result:
        return result[name]
    fit = result.get("fit")
    if isinstance(fit, dict) and name in fit:
        return fit[name]
    return fallback


def _component_count_4(mask: np.ndarray) -> int:
    """Count strict four-connected components without changing shared helpers.

    The contour tracer uses marching-squares case 5/10 pairings that keep
    diagonal foreground pixels in separate loops.  ``stroke_engine``'s
    historical helper is documented as four-connected but deliberately joins
    diagonals.  Using that helper for the expected foreground topology made a
    valid contour extraction look as if it had invented loops, rejecting many
    real gradient objects before the error-budget optimiser could assess them.
    """
    source = np.asarray(mask, dtype=np.bool_)
    if source.ndim != 2 or not source.any():
        return 0
    seen = np.zeros_like(source, dtype=np.bool_)
    height, width = source.shape
    count = 0
    for seed_y, seed_x in zip(*np.nonzero(source & ~seen)):
        seed_y, seed_x = int(seed_y), int(seed_x)
        if seen[seed_y, seed_x]:
            continue
        count += 1
        seen[seed_y, seed_x] = True
        stack = [(seed_y, seed_x)]
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x),
                           (y, x - 1), (y, x + 1)):
                if (0 <= yy < height and 0 <= xx < width
                        and source[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
    return count


def _mask_topology(mask: np.ndarray) -> tuple[int, int]:
    from stroke_engine import connected_components

    # Foreground is four-connected to match marching-squares diagonal cases;
    # background remains eight-connected so a diagonal background contact is
    # not falsely counted as a vector hole.  This complementary convention is
    # the standard unambiguous topology for a binary pixel grid.
    components = _component_count_4(mask)
    padded_background = ~np.pad(
        mask, 1, mode="constant", constant_values=False)
    labels, background_components = connected_components(padded_background)
    exterior = int(labels[0, 0]) if labels.size else 0
    holes = sum(
        label != exterior for label in range(1, background_components + 1))
    return int(components), int(holes)


def _contour_extraction_evidence(loops, *, x_offset=0, y_offset=0):
    """Return compact, source-coordinate evidence for extracted mask loops.

    The tracing engine orients foreground-component boundaries and hole
    boundaries in opposite directions.  Retaining the signed area therefore
    makes a smoothing fallback auditable without embedding every contour point
    in the public report.
    """
    evidence = []
    for loop_index, loop in enumerate(loops):
        points = np.asarray(loop, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points):
            continue
        shifted = np.roll(points, -1, axis=0)
        signed_area = 0.5 * float(np.sum(
            points[:, 0] * shifted[:, 1]
            - shifted[:, 0] * points[:, 1]))
        x0 = float(points[:, 0].min()) + float(x_offset)
        y0 = float(points[:, 1].min()) + float(y_offset)
        x1 = float(points[:, 0].max()) + float(x_offset)
        y1 = float(points[:, 1].max()) + float(y_offset)
        evidence.append({
            "loop_index": int(loop_index),
            "role": (
                "component_boundary" if signed_area > 0.0
                else "hole_boundary" if signed_area < 0.0
                else "degenerate_boundary"),
            "source_point_count": int(len(points)),
            "signed_area": round(signed_area, 6),
            "abs_area": round(abs(signed_area), 6),
            "bbox_xyxy": [round(value, 6)
                          for value in (x0, y0, x1, y1)],
        })
    return evidence


def _optimise_mask_geometry(mask: np.ndarray, *, error_budget_percent: float,
                            smooth: float, max_segments: int) -> dict[str, Any]:
    """Adapt the contour optimiser to an explicit ownership mask."""
    if optimize_compound_contours is None:
        raise RuntimeError("geometry_error_optimizer is unavailable")
    from trace_engine import _mask_to_smooth_loops

    source_mask = np.asarray(mask, dtype=np.bool_)
    ys, xs = np.nonzero(source_mask)
    if not len(xs):
        raise ValueError("ownership mask is empty")
    # Marching squares used to scan the entire trace canvas once per candidate
    # (and again after a smoothing fallback).  A zero halo larger than the
    # Gaussian support is mathematically the same field around this object, so
    # crop first and translate the extracted points back to source coordinates.
    halo = max(2, int(math.ceil(4.0 * max(0.0, float(smooth)))))
    x0 = max(0, int(xs.min()) - halo)
    y0 = max(0, int(ys.min()) - halo)
    x1 = min(source_mask.shape[1], int(xs.max()) + 1 + halo)
    y1 = min(source_mask.shape[0], int(ys.max()) + 1 + halo)
    local_mask = np.ascontiguousarray(source_mask[y0:y1, x0:x1])

    components, holes = _mask_topology(local_mask)
    expected_loops = components + holes
    smoothed_loops = _mask_to_smooth_loops(
        local_mask, simplify=0.0, min_area=1.0, smooth=float(smooth))
    if smooth > 0.0:
        unsmoothed_loops = _mask_to_smooth_loops(
            local_mask, simplify=0.0, min_area=1.0, smooth=0.0)
    else:
        unsmoothed_loops = smoothed_loops
    smoothing_fallback = bool(
        smooth > 0.0 and len(smoothed_loops) != expected_loops)
    loops = unsmoothed_loops if smoothing_fallback else smoothed_loops
    if len(loops) != expected_loops:
        raise RuntimeError("mask contour extraction changed topology")
    extraction_evidence = {
        "requested_smooth": round(float(smooth), 6),
        "expected_loop_count": int(expected_loops),
        "smoothed_loop_count_before_fallback": int(len(smoothed_loops)),
        "unsmoothed_loop_count": int(len(unsmoothed_loops)),
        "fallback_used": bool(smoothing_fallback),
        "unsmoothed_loops": _contour_extraction_evidence(
            unsmoothed_loops, x_offset=x0, y_offset=y0),
        "smoothed_loops": _contour_extraction_evidence(
            smoothed_loops, x_offset=x0, y_offset=y0),
    }
    if x0 or y0:
        loops = [[(float(x) + x0, float(y) + y0) for x, y in loop]
                 for loop in loops]
    result = optimize_compound_contours(
        loops,
        error_budget_percent=float(error_budget_percent),
        max_segments=int(max_segments),
    )
    selected_id = result.get("selected_candidate_id")
    selected_row = next((
        row for row in result.get("candidates", [])
        if row.get("candidate_id") == selected_id), {})
    optimiser_topology = selected_row.get("topology", {})
    per_loop_selection = selected_row.get("per_loop_selection") or []
    identity_loops = [
        item for item in per_loop_selection
        if bool(item.get("identity_rollback", False))]
    largest_identity = max(
        identity_loops,
        key=lambda item: (int(item.get("source_point_count", 0)),
                          -int(item.get("loop_index", 0))),
        default=None)
    source_loop_evidence = extraction_evidence[
        "unsmoothed_loops" if smoothing_fallback else "smoothed_loops"]
    largest_source = max(
        source_loop_evidence,
        key=lambda item: (int(item.get("source_point_count", 0)),
                          -int(item.get("loop_index", 0))),
        default=None)
    result["selection_evidence"] = {
        "selected_candidate_id": selected_id,
        "selected_source": result.get("selected_source"),
        "identity_rollback_selected": bool(
            result.get("identity_rollback_selected", False)),
        "mixed_loop_tolerances": bool(
            (result.get("fit") or {}).get("mixed_loop_tolerances", False)),
        "identity_loop_count": int(len(identity_loops)),
        "identity_anchor_count": int(sum(
            int(item.get("anchors_after", 0)) for item in identity_loops)),
        "identity_designer_anchor_count": int(sum(
            int(item.get("designer_anchor_count", 0))
            for item in identity_loops)),
        "largest_identity_loop": (
            None if largest_identity is None else {
                "loop_index": int(largest_identity.get("loop_index", -1)),
                "source_point_count": int(
                    largest_identity.get("source_point_count", 0)),
                "anchors_after": int(
                    largest_identity.get("anchors_after", 0)),
                "designer_anchor_count": int(
                    largest_identity.get("designer_anchor_count", 0)),
            }),
        "largest_source_loop": (
            None if largest_source is None else {
                "loop_index": int(largest_source["loop_index"]),
                "source_point_count": int(
                    largest_source["source_point_count"]),
            }),
        "largest_loop_frontier": _json_safe(
            result.get("largest_loop_frontier") or []),
        "designer_anchor_reduction": int(
            int(result.get("anchors_before", 0))
            - int(result.get("designer_anchor_count", 0))),
        "designer_anchor_reduction_ratio": round(
            (int(result.get("anchors_before", 0))
             - int(result.get("designer_anchor_count", 0)))
            / float(max(1, int(result.get("anchors_before", 0)))), 9),
    }
    result["topology"] = {
        "components": components,
        "holes": holes,
        "connectivity_policy": {
            "foreground": 4,
            "background": 8,
            "reason": "matches marching-squares diagonal case topology",
        },
        "expected_loops": expected_loops,
        "actual_loops": int(result.get("loop_count", len(loops))),
        "topology_preserved": bool(
            optimiser_topology.get("preserved", False)
            and int(result.get("loop_count", len(loops))) == expected_loops),
        "smoothing_fallback": smoothing_fallback,
        "extraction_evidence": extraction_evidence,
        "contour_crop": {
            "bbox_xyxy": [x0, y0, x1, y1],
            "halo_px": halo,
            "source_canvas": [int(source_mask.shape[1]),
                              int(source_mask.shape[0])],
            "cells_scanned": int(max(0, x1 - x0) * max(0, y1 - y0)),
        },
        "optimizer": _json_safe(optimiser_topology),
    }
    return result


def _fit_geometry(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    error_budget_percent: float,
    smooth: float,
    max_segments: int,
    geometry_optimizer: Optional[Callable[..., dict[str, Any]]],
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Fit one ownership mask under a hard percentage error budget."""
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    reference_length = math.hypot(width, height)
    tolerance_px = reference_length * float(error_budget_percent) / 100.0
    tolerance_px = max(1e-6, tolerance_px)
    reasons: list[str] = []
    try:
        bundled_optimizer = bool(
            geometry_optimizer is None
            and optimize_compound_contours is not None)
        if bundled_optimizer:
            raw = _optimise_mask_geometry(
                mask,
                error_budget_percent=float(error_budget_percent),
                smooth=float(smooth),
                max_segments=int(max_segments),
            )
            solver = "geometry_error_optimizer.optimize_compound_contours"
            minimum_claim = "verified_lexicographic_candidate_minimum"
        elif geometry_optimizer is None:
            raw = fit_mask(
                mask,
                tolerance=tolerance_px,
                line_tolerance=tolerance_px,
                primitive_tolerance=tolerance_px,
                smooth=float(smooth),
                min_area=1.0,
                max_segments=int(max_segments),
            )
            solver = "curve_refit_local_merge_fallback"
            minimum_claim = "locally_merge_minimal_not_global"
        else:
            raw = geometry_optimizer(
                mask,
                error_budget_percent=float(error_budget_percent),
                tolerance_px=float(tolerance_px),
                reference_length_px=float(reference_length),
                smooth=float(smooth),
                min_area=1.0,
                max_segments=int(max_segments),
            )
            solver = getattr(geometry_optimizer, "__name__", "injected_optimizer")
            minimum_claim = "optimizer_reported_lexicographic_minimum"
        if not isinstance(raw, dict):
            raise TypeError("geometry optimizer must return a mapping")
        # The bundled optimiser may safely preserve the original polyline when
        # no anchor reduction satisfies the error contract.  That is a valid
        # rollback for a generic geometry transaction, but it is not a useful
        # native-gradient reconstruction: hundreds of raster-edge anchors
        # would be delivered under a misleading zero-error identity result.
        # Injected/legacy optimisers keep their historical contract.
        if bundled_optimizer and (
                bool(raw.get("identity_rollback_selected", False))
                or raw.get("status") == "identity_rollback_no_safe_reduction"):
            reasons.append(
                "geometry_identity_rollback_no_safe_anchor_reduction")
        if bundled_optimizer:
            selected_id = raw.get("selected_candidate_id")
            selected_row = next((
                row for row in raw.get("candidates", [])
                if row.get("candidate_id") == selected_id), {})
            per_loop = selected_row.get("per_loop_selection") or []
            largest_loop = max(
                per_loop,
                key=lambda item: (int(item.get("source_point_count", 0)),
                                  -int(item.get("loop_index", 0))),
                default=None)
            # Simplifying a few tiny holes must not disguise a dense identity
            # rollback of the object's dominant outer contour.  Eight or fewer
            # anchors is already the explicit low-sided analytic tier; a larger
            # identity loop remains raster-bound geometry and fails closed.
            if (largest_loop is not None
                    and bool(largest_loop.get("identity_rollback", False))
                    and int(largest_loop.get("source_point_count", 0)) > 8):
                reasons.append("geometry_dominant_source_polyline_rollback")

        path = _geometry_value(raw, "path", "")
        fill_rule = _geometry_value(raw, "fill_rule", "evenodd")
        topology = _geometry_value(raw, "topology")
        if not isinstance(path, str) or not path.strip():
            reasons.append("geometry_path_missing")
        if not isinstance(topology, dict):
            reasons.append("geometry_topology_evidence_missing")
            topology = {}
        elif not bool(topology.get("topology_preserved", False)):
            reasons.append("geometry_topology_not_preserved")

        anchors = _geometry_value(raw, "anchor_count")
        if anchors is None:
            anchors = _geometry_value(raw, "anchors_after")
        segments = _geometry_value(raw, "segment_count")
        if segments is None:
            segments = _geometry_value(raw, "segment_count_after")
        before = _geometry_value(raw, "input_point_count")
        if before is None:
            before = _geometry_value(raw, "anchors_before", anchors)
        if anchors is None or int(anchors) < 1:
            reasons.append("geometry_anchor_count_missing")
            anchors = 0
        if segments is None or int(segments) < 1:
            reasons.append("geometry_segment_count_missing")
            segments = 0

        actual_percent = raw.get("actual_max_error_percent")
        actual_p95_percent = raw.get("actual_p95_error_percent")
        actual_px = None
        error = _geometry_value(raw, "error", {})
        if actual_percent is not None:
            actual_percent = float(actual_percent)
            actual_px = actual_percent * reference_length / 100.0
        elif isinstance(error, dict) and error.get("max") is not None:
            actual_px = float(error["max"])
            actual_percent = 100.0 * actual_px / max(1e-9, reference_length)
        else:
            reasons.append("geometry_error_evidence_missing")
            actual_percent = float("inf")
        optimiser_contract = (
            raw.get("status") == "selected_within_geometry_budget"
            and isinstance(raw.get("error_contract"), dict))
        if optimiser_contract:
            selected_id = raw.get("selected_candidate_id")
            selected_row = next((row for row in raw.get("candidates", [])
                                 if row.get("candidate_id") == selected_id), {})
            if not bool(selected_row.get("eligible", False)):
                reasons.append("geometry_optimizer_selected_ineligible_candidate")
            if actual_p95_percent is None:
                reasons.append("geometry_p95_error_evidence_missing")
            elif float(actual_p95_percent) > float(error_budget_percent) + 1e-7:
                reasons.append("geometry_p95_error_budget_exceeded")
            if (not math.isfinite(float(actual_percent))
                    or float(actual_percent) > 3.0 * float(error_budget_percent) + 1e-7):
                reasons.append("geometry_tail_max_budget_exceeded")
            if float(raw.get("over_budget_share", 1.0)) > 0.05 + 1e-9:
                reasons.append("geometry_over_budget_share_exceeded")
            if float(raw.get("salient_corner_max_percent", float("inf"))) > (
                    2.0 * float(error_budget_percent) + 1e-7):
                reasons.append("geometry_salient_corner_budget_exceeded")
        elif (not math.isfinite(float(actual_percent))
              or float(actual_percent) > float(error_budget_percent) + 1e-7):
            reasons.append("geometry_error_budget_exceeded")

        # Legacy/injected optimisers sometimes certify only a hard maximum.
        # A verified maximum is also a conservative upper bound on P95, so
        # record that derivation explicitly instead of leaving a false
        # evidence hole in the emitted designer-quality contract.
        p95_evidence_source = "optimizer_reported"
        if (actual_p95_percent is None and not optimiser_contract
                and math.isfinite(float(actual_percent))):
            actual_p95_percent = float(actual_percent)
            p95_evidence_source = "conservative_upper_bound_from_maximum"

        native = _native_primitive_evidence(raw)
        primitive_complexity = raw.get("primitive_complexity", {})
        direct_native = isinstance(raw.get("native_primitive"), dict)
        raw_fit = raw.get("fit") if isinstance(raw.get("fit"), dict) else {}
        fit_contours = raw_fit.get("contours")
        all_contours_native = bool(
            isinstance(fit_contours, list) and fit_contours
            and all(isinstance(item, dict)
                    and isinstance(item.get("native_primitive"), dict)
                    for item in fit_contours))
        primitive_first = (
            direct_native or all_contours_native or (
                isinstance(primitive_complexity, dict)
                and primitive_complexity.get("category") in {
                    "native_primitive", "low_sided_line_primitive"}))
        designer_anchors = int(raw.get("designer_anchor_count", anchors))
        geometry = {
            "path": path,
            "fill_rule": str(fill_rule),
            "solver": solver,
            "primitive_first": primitive_first,
            "native_primitives": native,
            "primitive_complexity": _json_safe(primitive_complexity),
            "topology": _json_safe(topology),
            "selection_evidence": _json_safe(
                raw.get("selection_evidence") or {}),
            "anchor_count": int(anchors),
            "designer_anchor_count": designer_anchors,
            "segment_count": int(segments),
            "anchors_before": int(before or 0),
            "economy": _json_safe(_geometry_value(raw, "economy", {})),
            "error_budget": {
                "metric": "bidirectional_geometry_over_source_bbox_diagonal_percent",
                "reference_length_px": round(reference_length, 6),
                "requested_max_percent": round(float(error_budget_percent), 6),
                "tolerance_px": round(tolerance_px, 6),
                "actual_max_error_px": (None if actual_px is None
                                        else round(float(actual_px), 6)),
                "actual_max_error_percent": (None if not math.isfinite(
                    float(actual_percent)) else round(float(actual_percent), 6)),
                "actual_p95_error_percent": (None if actual_p95_percent is None
                                             else round(float(actual_p95_percent), 6)),
                "p95_evidence_source": p95_evidence_source,
                "contract": _json_safe(raw.get("error_contract", {
                    "primary": "max_error_percent <= error_budget_percent",
                })),
                "passed": not any(reason.startswith("geometry_") for reason in reasons),
            },
            "lexicographic_objective": _json_safe(raw.get(
                "lexicographic_objective", {
                    "hard_constraints": [
                        "topology_preserved",
                        "actual_max_error_percent<=error_budget_percent",
                    ],
                    "minimise_in_order": [
                        "anchor_count", "segment_count", "object_fragments",
                    ],
                    "colour_used_for_geometry": False,
                    "minimum_claim": minimum_claim,
                })),
            "fit": _json_safe(raw),
        }
        from native_geometry_contract import whole_object_native_primitive

        geometry["native_whole_object_path"] = (
            path if whole_object_native_primitive(geometry, path) else None)
        if reasons:
            return None, list(dict.fromkeys(reasons))
        return geometry, []
    except Exception as exc:
        return None, [f"geometry_optimizer_error:{type(exc).__name__}:{exc}"[:240]]


def _fit_default_geometry_process_job(job):
    """Fit one default geometry job in an isolated spawn worker.

    The payload and result contain only pickle-safe deterministic values.  In
    particular, no paint/colour score is passed into the geometry optimiser,
    and the parent still consumes results in the original shortlist order.
    """
    (digest, mask, bbox, error_budget_percent, smooth, max_segments) = job
    return digest, _fit_geometry(
        mask,
        bbox,
        error_budget_percent=float(error_budget_percent),
        smooth=float(smooth),
        max_segments=int(max_segments),
        geometry_optimizer=None,
    )


def _normalise_inputs(den: Any, lab_all: Any, visible: Any,
                      palette: Any, alpha: Any) -> tuple[np.ndarray, ...]:
    rgb = np.asarray(den)
    labels = np.asarray(lab_all)
    vis = np.asarray(visible, dtype=np.bool_)
    pal = np.asarray(palette)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("den must have shape HxWx3 or HxWx4")
    if labels.shape != rgb.shape[:2] or vis.shape != rgb.shape[:2]:
        raise ValueError("lab_all and visible must match den")
    if pal.ndim != 2 or pal.shape[1] != 3 or len(pal) < 1:
        raise ValueError("palette must have shape Nx3")
    alpha_array = None
    if alpha is not None:
        alpha_array = np.asarray(alpha)
        if alpha_array.shape != rgb.shape[:2]:
            raise ValueError("alpha must match den")
    return rgb, labels, vis, pal, alpha_array


def _paint_gain(proposal: dict[str, Any]) -> float:
    error = proposal.get("error") if isinstance(proposal, dict) else None
    if not isinstance(error, dict):
        return 0.0
    try:
        return max(0.0, float(error.get("heldout_mean_improvement", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _paint_fit_diagnostic(fit: Any) -> dict[str, Any]:
    """Keep compact rejection evidence instead of reducing it to one reason."""
    if not isinstance(fit, dict):
        return {"status": "invalid_result"}
    attempts = []
    for candidate in fit.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        attempts.append({
            "type": candidate.get("type"),
            "status": candidate.get("status"),
            "model": candidate.get("model"),
            "stops": candidate.get("stops", []),
            "train_error": candidate.get("train_error"),
            "heldout_error": candidate.get("heldout_error"),
            "comparison_to_solid": candidate.get("comparison_to_solid"),
            "dominant_improvement_tail_exception": candidate.get(
                "dominant_improvement_tail_exception"),
            "reasons": candidate.get("reasons", []),
        })
    validation = fit.get("validation")
    return _json_safe({
        "status": fit.get("status"),
        "reasons": fit.get("reasons", []),
        "mask": fit.get("mask"),
        "solid_baseline": ((fit.get("error") or {}).get("solid_baseline")
                           if isinstance(fit.get("error"), dict) else None),
        "internal_edges": (validation.get("internal_edges")
                           if isinstance(validation, dict) else None),
        "model_attempts": attempts,
    })


def _complete_field_fit_policy(
        options: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return bounded paint options reserved for a complete structural field.

    A complete multi-band object can improve almost every source pixel while a
    small textured tail remains closer to one of the old flat palette bands.
    The general fitter defaults must remain strict for arbitrary fragments, so
    this policy is applied only after graph ownership is closed.  Caller
    values may make the policy stricter, but never wider than the audited caps.
    Absolute mean/P90/P99, solid-gain and material-edge gates are untouched.
    """
    result = dict(options)
    caps = {
        "maximum_p99_sample_degradation": 8.0,
        "maximum_degraded_share": 0.12,
        "maximum_materially_degraded_share": 0.07,
    }
    applied = {}
    for name, cap in caps.items():
        raw = result.get(name, cap)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = cap
        value = min(cap, value) if math.isfinite(value) else cap
        result[name] = value
        applied[name] = round(value, 6)
    return result, {
        "policy": "bounded_complete_structural_field_tail_policy",
        "scope": "ownership_closed_structural_field_only",
        "options": applied,
        "unchanged_hard_gates": [
            "absolute_mean_p90_p99_error",
            "material_improvement_over_solid",
            "material_internal_hard_edge",
            "material_hard_label_boundary",
            "independently_salted_same_options_confirmation",
        ],
        "outer_holdout_used_for_candidate_generation": False,
    }


def _stage_srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Small local OKLab conversion for graph-expansion diagnostics."""
    srgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    red, green, blue = np.moveaxis(linear, -1, 0)
    ll = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    mm = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    ss = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    ll, mm, ss = np.cbrt(ll), np.cbrt(mm), np.cbrt(ss)
    return np.stack((
        0.2104542553 * ll + 0.7936177850 * mm - 0.0040720468 * ss,
        1.9779984951 * ll - 2.4285922050 * mm + 0.4505937099 * ss,
        0.0259040371 * ll + 0.7827717662 * mm - 0.8086757660 * ss,
    ), axis=-1)


def _stage_delta_e(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return 100.0 * np.linalg.norm(
        _stage_srgb_to_oklab(actual) - _stage_srgb_to_oklab(predicted),
        axis=-1,
    )


def _gradient_stop_arrays(
        stops: Any) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(stops, (list, tuple)) or not 2 <= len(stops) <= 5:
        return None, None
    offsets = []
    colours = []
    try:
        for stop in stops:
            if not isinstance(stop, dict):
                return None, None
            offsets.append(float(stop["offset"]))
            colour = np.asarray(stop.get("rgb"), dtype=np.float64)
            if colour.shape != (3,) or not np.isfinite(colour).all():
                return None, None
            colours.append(colour)
    except (KeyError, TypeError, ValueError):
        return None, None
    offsets_array = np.asarray(offsets, dtype=np.float64)
    colours_array = np.asarray(colours, dtype=np.float64)
    if (not np.isfinite(offsets_array).all()
            or abs(float(offsets_array[0])) > 1e-6
            or abs(float(offsets_array[-1]) - 1.0) > 1e-6
            or np.any(np.diff(offsets_array) <= 1e-6)):
        return None, None
    return offsets_array, colours_array


def _render_discovery_model(
        fit: dict[str, Any], xs: np.ndarray, ys: np.ndarray,
        *, extrapolate: bool = True) -> Optional[np.ndarray]:
    """Render a fitted native gradient at source coordinates.

    Endpoint extrapolation is used only to propose graph neighbours beyond a
    seed's current bbox.  It is never delivered: the expanded mask is fitted
    again by the production fitter and independently revalidated before it can
    reach geometry.
    """
    model = fit.get("model") if isinstance(fit, dict) else None
    if not isinstance(model, dict):
        return None
    offsets, colours = _gradient_stop_arrays(fit.get("stops"))
    if offsets is None or colours is None:
        return None
    try:
        x_values = np.asarray(xs, dtype=np.float64)
        y_values = np.asarray(ys, dtype=np.float64)
        if x_values.shape != y_values.shape:
            return None
        if model.get("type") == "linear":
            x1, y1 = float(model["x1"]), float(model["y1"])
            x2, y2 = float(model["x2"]), float(model["y2"])
            vx, vy = x2 - x1, y2 - y1
            denominator = vx * vx + vy * vy
            if not math.isfinite(denominator) or denominator <= 1e-12:
                return None
            t = ((x_values - x1) * vx + (y_values - y1) * vy) / denominator
        elif model.get("type") == "radial":
            center = np.asarray(model.get("center"), dtype=np.float64)
            if center.shape != (2,) or not np.isfinite(center).all():
                return None
            radius_x = float(model["radius_x"])
            radius_y = float(model["radius_y"])
            angle = math.radians(float(model.get("rotation_degrees", 0.0)))
            if (not all(math.isfinite(value) for value in
                        (radius_x, radius_y, angle))
                    or radius_x <= 1e-9 or radius_y <= 1e-9):
                return None
            dx, dy = x_values - center[0], y_values - center[1]
            cosine, sine = math.cos(angle), math.sin(angle)
            local_x = cosine * dx + sine * dy
            local_y = -sine * dx + cosine * dy
            t = np.sqrt((local_x / radius_x) ** 2
                        + (local_y / radius_y) ** 2)
        else:
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(t).all():
        return None
    if not extrapolate:
        t = np.clip(t, 0.0, 1.0)
    segment = np.searchsorted(offsets, t, side="right") - 1
    segment = np.clip(segment, 0, len(offsets) - 2)
    left = offsets[segment]
    right = offsets[segment + 1]
    fraction = (t - left) / np.maximum(1e-12, right - left)
    prediction = (colours[segment] * (1.0 - fraction[:, None])
                  + colours[segment + 1] * fraction[:, None])
    return np.clip(prediction, 0.0, 255.0)


def _robust_error_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "p90": round(float(np.quantile(values, 0.90)), 4),
        "p99": round(float(np.quantile(values, 0.99)), 4),
    }


def _material_safe_expansion_edges(
        edges: list[dict[str, Any]], *, min_smooth_fraction: float,
        smooth_delta: float, min_palette_distance: float,
        max_palette_distance: float) -> tuple[bool, list[str]]:
    reasons = []
    if not edges:
        return False, ["not_adjacent_to_current_component_set"]
    required = (
        "shared_boundary", "smooth_fraction", "source_delta_mean",
        "source_delta_p90", "palette_distance",
    )
    for edge in edges:
        if not isinstance(edge, dict) or any(name not in edge
                                             for name in required):
            reasons.append("edge_missing_material_boundary_evidence")
            continue
        try:
            shared = int(edge["shared_boundary"])
            smooth = float(edge["smooth_fraction"])
            mean_delta = float(edge["source_delta_mean"])
            p90_delta = float(edge["source_delta_p90"])
            palette_delta = float(edge["palette_distance"])
        except (TypeError, ValueError):
            reasons.append("edge_invalid_material_boundary_evidence")
            continue
        if (shared < 1 or not all(math.isfinite(value) for value in
                                  (smooth, mean_delta, p90_delta,
                                   palette_delta))):
            reasons.append("edge_invalid_material_boundary_evidence")
        if smooth < float(min_smooth_fraction) - 1e-12:
            reasons.append("material_hard_edge_smooth_fraction")
        if p90_delta > float(smooth_delta) * 1.8 + 1e-12:
            reasons.append("material_hard_edge_source_delta_p90")
        if not (float(min_palette_distance) - 1e-12 <= palette_delta
                <= float(max_palette_distance) + 1e-12):
            reasons.append("material_hard_edge_palette_distance")
    return not reasons, list(dict.fromkeys(reasons))


def _component_model_compatibility(
        rgb: np.ndarray, component_mask: np.ndarray,
        discovery_fit: dict[str, Any], *, alpha: Optional[np.ndarray],
        model_options: dict[str, Any], edge_smooth_fraction: float,
        min_smooth_fraction: float) -> dict[str, Any]:
    sample_mask = np.asarray(component_mask, dtype=np.bool_).copy()
    if alpha is not None:
        threshold = float(model_options.get("alpha_threshold", 12.0))
        sample_mask &= np.asarray(alpha) >= threshold
    flat = np.flatnonzero(sample_mask)
    if not len(flat):
        return {"accepted": False, "reasons": ["component_has_no_opaque_pixels"]}
    if len(flat) > 8192:
        positions = np.linspace(0, len(flat) - 1, 8192, dtype=np.int64)
        flat = flat[positions]
    ys, xs = np.unravel_index(flat, sample_mask.shape)
    source = np.asarray(rgb[ys, xs, :3], dtype=np.float64)
    predicted = _render_discovery_model(
        discovery_fit, xs.astype(np.float64), ys.astype(np.float64),
        extrapolate=True)
    if predicted is None:
        return {"accepted": False,
                "reasons": ["discovery_gradient_model_not_renderable"]}
    model_error = _stage_delta_e(source, predicted)
    solid_colour = np.median(source, axis=0)
    solid_render = np.broadcast_to(solid_colour, source.shape)
    solid_error = _stage_delta_e(source, solid_render)
    model_stats = _robust_error_stats(model_error)
    solid_stats = _robust_error_stats(solid_error)
    representative_error = float(_stage_delta_e(
        np.median(source, axis=0)[None, :],
        np.median(predicted, axis=0)[None, :],
    )[0])
    degradation = model_error - solid_error
    materially_degraded_share = float(np.mean(degradation > 2.0))
    mean_limit = min(5.5, float(model_options.get(
        "maximum_mean_error", 5.5)))
    p90_limit = min(8.5, float(model_options.get(
        "maximum_p90_error", 8.5)))
    p99_limit = min(13.0, float(model_options.get(
        "maximum_p99_error", 13.0)))
    absolute_fit = bool(
        model_stats["mean"] <= mean_limit
        and model_stats["p90"] <= p90_limit
        and model_stats["p99"] <= p99_limit)
    mean_gain = solid_stats["mean"] - model_stats["mean"]
    material_improvement = bool(
        mean_gain >= max(0.75, 0.12 * solid_stats["mean"])
        and model_stats["p90"] <= solid_stats["p90"] + 0.5
        and materially_degraded_share <= 0.20)
    continuous_band_equivalence = bool(
        solid_stats["mean"] <= 3.0
        and representative_error <= 3.25
        and model_stats["mean"] <= min(mean_limit, 3.5)
        and model_stats["p90"] <= min(p90_limit, 6.5)
        and edge_smooth_fraction >= max(float(min_smooth_fraction), 0.70))
    reasons = []
    if not absolute_fit:
        reasons.append("robust_delta_e_threshold_failed")
    if not (material_improvement or continuous_band_equivalence):
        reasons.append("not_better_than_component_solid_or_continuous_band")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "samples": int(len(flat)),
        "gradient_error": model_stats,
        "component_solid_error": solid_stats,
        "mean_improvement_over_component_solid": round(mean_gain, 4),
        "representative_delta_e": round(representative_error, 4),
        "materially_degraded_share": round(materially_degraded_share, 6),
        "acceptance_path": (
            "material_improvement_over_component_solid"
            if material_improvement else
            ("equivalent_continuous_colour_band"
             if continuous_band_equivalence else None)),
        "outer_holdout_used_for_expansion_choice": False,
    }


def _model_guided_graph_expansion(
        item: dict[str, Any], rgb: np.ndarray, labels: np.ndarray,
        alpha: Optional[np.ndarray], *, model_fitter: Callable[..., Any],
        model_options: dict[str, Any], candidate_options: dict[str, Any],
        apply_complete_field_policy: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Conservatively grow a paint-valid seed over its eligible graph.

    Neighbours are chosen only with the discovery model and source pixels.
    The expanded union is then sent through the full production fitter; its
    outer holdout is an accept-or-skip gate, never a neighbour selector.  The
    caller still performs the independently salted second validation.
    """
    before_ids = tuple(sorted(set(int(value) for value in
                                  item.get("component_ids", []))))
    audit: dict[str, Any] = {
        "strategy": "deterministic_model_guided_eligible_graph_expansion",
        "status": "not_attempted",
        "before_component_ids": list(before_ids),
        "before_area": int(item.get("area", 0)),
        "rounds": [],
        "included_components": [],
        "rejected_components": [],
        "candidate_generation_uses_outer_holdout": False,
        "delivery_requires_full_refit_and_independent_salted_revalidation": True,
    }
    if item.get("candidate_family") not in {"community", "monotonic_chain"}:
        audit["reason"] = "candidate_family_not_expandable"
        return item, audit
    context = item.get("_component_context")
    if not isinstance(context, dict):
        audit["reason"] = "component_context_missing"
        return item, audit
    component_map = np.asarray(context.get("component_map"))
    if (component_map.shape != item["_mask"].shape
            or component_map.ndim != 2):
        audit["reason"] = "component_map_invalid"
        return item, audit
    exact_seed = np.isin(component_map, np.asarray(before_ids, dtype=np.int32))
    if not before_ids or not np.array_equal(exact_seed, item["_mask"]):
        audit["reason"] = "seed_mask_not_exact_component_union"
        return item, audit
    raw_edges = context.get("eligible_edges")
    if not isinstance(raw_edges, (list, tuple)):
        audit["reason"] = "eligible_graph_missing"
        return item, audit
    edges = []
    try:
        for raw in raw_edges:
            if not isinstance(raw, dict):
                raise TypeError
            edge = dict(raw)
            a, b = int(edge["a"]), int(edge["b"])
            if a <= 0 or b <= 0 or a == b:
                raise ValueError
            edge["a"], edge["b"] = min(a, b), max(a, b)
            edges.append(edge)
    except (KeyError, TypeError, ValueError):
        audit["reason"] = "eligible_graph_invalid"
        return item, audit
    edges.sort(key=lambda edge: (
        int(edge["a"]), int(edge["b"]),
        -int(edge.get("shared_boundary", 0))))
    min_smooth = float(candidate_options.get("min_smooth_fraction", 0.55))
    smooth_delta = float(candidate_options.get("smooth_delta", 30.0))
    min_palette = float(candidate_options.get("min_palette_distance", 8.0))
    max_palette = float(candidate_options.get("max_palette_distance", 170.0))
    owned = set(before_ids)
    resolved = set(owned)
    discovery_fit = item.get("_paint")
    round_number = 0
    while True:
        frontier: dict[int, list[dict[str, Any]]] = {}
        for edge in edges:
            a, b = int(edge["a"]), int(edge["b"])
            if (a in owned) == (b in owned):
                continue
            neighbour = b if a in owned else a
            if neighbour in resolved:
                continue
            frontier.setdefault(neighbour, []).append(edge)
        if not frontier:
            break
        round_number += 1
        round_audit = {"round": round_number, "frontier": [], "included": []}
        additions = []
        for component_id in sorted(frontier):
            connecting = frontier[component_id]
            safe, edge_reasons = _material_safe_expansion_edges(
                connecting, min_smooth_fraction=min_smooth,
                smooth_delta=smooth_delta,
                min_palette_distance=min_palette,
                max_palette_distance=max_palette)
            record: dict[str, Any] = {
                "component_id": component_id,
                "adjacent_owned_component_ids": sorted({
                    int(edge["a"]) if int(edge["a"]) in owned
                    else int(edge["b"]) for edge in connecting}),
                "edge_count": len(connecting),
                "minimum_edge_smooth_fraction": round(min(
                    float(edge.get("smooth_fraction", 0.0))
                    for edge in connecting), 6),
            }
            if not safe:
                record.update({"accepted": False,
                               "reasons": edge_reasons})
            else:
                compatibility = _component_model_compatibility(
                    rgb, component_map == component_id, discovery_fit,
                    alpha=alpha, model_options=model_options,
                    edge_smooth_fraction=float(record[
                        "minimum_edge_smooth_fraction"]),
                    min_smooth_fraction=min_smooth)
                record.update(compatibility)
            round_audit["frontier"].append(record)
            resolved.add(component_id)
            if record.get("accepted"):
                additions.append(component_id)
            else:
                audit["rejected_components"].append({
                    "component_id": component_id,
                    "round": round_number,
                    "reasons": list(record.get("reasons", [])),
                })
        if not additions:
            audit["rounds"].append(round_audit)
            break
        additions.sort()
        owned.update(additions)
        round_audit["included"] = additions
        audit["included_components"].extend(additions)
        audit["rounds"].append(round_audit)
    after_ids = tuple(sorted(owned))
    audit["after_component_ids"] = list(after_ids)
    if after_ids == before_ids:
        audit["status"] = "no_compatible_neighbour"
        audit["after_area"] = audit["before_area"]
        return item, audit
    expanded_mask = np.ascontiguousarray(np.isin(
        component_map, np.asarray(after_ids, dtype=np.int32)))
    audit["after_area"] = int(expanded_mask.sum())
    audit["area_added"] = int(audit["after_area"] - audit["before_area"])
    fit_options = dict(item.get("_model_fit_options", model_options))
    policy_evidence = None
    if apply_complete_field_policy:
        fit_options, policy_evidence = _complete_field_fit_policy(fit_options)
    try:
        expanded_fit = model_fitter(
            rgb[:, :, :3], expanded_mask, alpha=alpha,
            label_map=labels, **fit_options)
    except Exception as exc:
        expanded_fit = {
            "status": "error",
            "reasons": [
                f"expanded_field_refit_error:{type(exc).__name__}:{exc}"[:240]
            ],
        }
    if (not isinstance(expanded_fit, dict)
            or expanded_fit.get("status") != "proposed"):
        audit["status"] = "expanded_refit_rejected"
        audit["refit"] = _paint_fit_diagnostic(expanded_fit)
        return item, audit
    expanded = dict(item)
    expanded["_mask"] = expanded_mask
    expanded["_mask_digest"] = _mask_digest(expanded_mask)
    expanded["bbox"] = _bbox(expanded_mask)
    expanded["area"] = int(expanded_mask.sum())
    expanded["component_ids"] = list(after_ids)
    expanded["candidate_family"] = "model_guided_field"
    expanded["_paint"] = expanded_fit
    expanded["_model_fit_options"] = fit_options
    expanded["_precertified_partition"] = None
    evidence = dict(item.get("candidate_evidence") or {})
    audit["status"] = "expanded_refit_proposed_pending_independent_revalidation"
    audit["refit"] = _paint_fit_diagnostic(expanded_fit)
    evidence.update({
        "original_candidate_family": item["candidate_family"],
        "ownership_mode": "model_guided_graph_expansion",
        "ownership_closed_over_raw_smooth_graph": False,
        "ownership_closed_after_model_guided_expansion": True,
        "raw_external_smooth_edges": int(evidence.get(
            "external_smooth_edges") or 0),
        "unresolved_external_smooth_edges": 0,
        "model_guided_expansion": audit,
    })
    if policy_evidence is not None:
        evidence["complete_field_paint_policy"] = policy_evidence
    expanded["candidate_evidence"] = evidence
    expanded["_model_guided_expansion"] = audit
    expanded["_expansion_seed_snapshot"] = {
        "candidate_family": item["candidate_family"],
        "component_ids": list(before_ids),
        "area": int(item["area"]),
        "bbox_xyxy": list(item["bbox"]),
    }
    return expanded, audit


def _certified_weak_bridge_partition(
        item: dict[str, Any]) -> dict[str, Any] | None:
    """Certify a paint-valid community separated only by weak graph bridges.

    A failed broad model alone never proves an object boundary.  This helper is
    deliberately narrower: the candidate must be an actual graph community,
    internally connected, and every omitted eligible edge must fall below the
    same mutual-support threshold that constructed the community.  Paint and a
    second independent holdout still have to pass later.
    """
    if item.get("candidate_family") != "community":
        return None
    ids = {int(value) for value in item.get("component_ids", [])}
    context = item.get("_component_context")
    if len(ids) < 3 or not isinstance(context, dict):
        return None
    edges = context.get("eligible_edges")
    if not isinstance(edges, (list, tuple)):
        return None
    threshold = float(context.get("mutual_support_base", 0.38))
    internal = []
    external = []
    for raw in edges:
        if not isinstance(raw, dict):
            return None
        a, b = int(raw.get("a", 0)), int(raw.get("b", 0))
        if a in ids and b in ids:
            internal.append(raw)
        elif (a in ids) != (b in ids):
            external.append(raw)
    reported = int((item.get("candidate_evidence") or {}).get(
        "external_smooth_edges") or 0)
    if not external or reported != len(external):
        return None
    if any(float(edge.get("mutual_support", 1.0)) >= threshold - 1e-12
           for edge in external):
        return None

    graph = {component_id: set() for component_id in ids}
    for edge in internal:
        a, b = int(edge["a"]), int(edge["b"])
        graph[a].add(b)
        graph[b].add(a)
    reached = set()
    pending = [min(ids)]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(sorted(graph[current] - reached, reverse=True))
    if reached != ids:
        return None

    cuts = [{
        "a": int(edge["a"]),
        "b": int(edge["b"]),
        "shared_boundary": int(edge.get("shared_boundary", 0)),
        "smooth_fraction": round(float(edge.get("smooth_fraction", 0.0)), 6),
        "mutual_support": round(float(edge.get("mutual_support", 0.0)), 6),
        "cut_reason": "certified_weak_bridge_below_community_threshold",
    } for edge in sorted(external, key=lambda value: (
        int(value.get("a", 0)), int(value.get("b", 0))))]
    return {
        "ownership_mode": "paint_valid_weak_bridge_graph_partition",
        "ownership_closed_over_raw_smooth_graph": False,
        "ownership_closed_after_certified_cuts": True,
        "raw_external_smooth_edges": len(external),
        "unresolved_external_smooth_edges": 0,
        "community_mutual_support_threshold": round(threshold, 6),
        "certified_cut_edges": cuts,
    }


def _paint_models_stable(first: dict[str, Any], second: Any
                         ) -> tuple[bool, dict[str, Any]]:
    if not isinstance(second, dict) or second.get("status") != "proposed":
        return False, {
            "passed": False,
            "reason": "independent_holdout_did_not_propose",
            "confirmation": _paint_fit_diagnostic(second),
        }
    first_model = first.get("model") or {}
    second_model = second.get("model") or {}

    def finite_vector(value: Any, shape: tuple[int, ...]) -> Optional[np.ndarray]:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if array.shape != shape or not np.isfinite(array).all():
            return None
        return array

    first_type = first_model.get("type")
    second_type = second_model.get("type")
    reasons = []
    if first_type not in {"linear", "radial"}:
        reasons.append("discovery_model_type_missing_or_invalid")
    if second_type not in {"linear", "radial"}:
        reasons.append("confirmation_model_type_missing_or_invalid")
    if first_type != second_type:
        reasons.append("paint_model_type_changed")
    direction_agreement = None
    if first_type == second_type == "linear":
        left = finite_vector(first_model.get("direction", []), (2,))
        right = finite_vector(second_model.get("direction", []), (2,))
        if left is None or right is None:
            reasons.append("linear_direction_missing_or_invalid")
        else:
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if not math.isfinite(denom) or denom <= 1e-12:
                reasons.append("linear_direction_missing_or_invalid")
            else:
                direction_agreement = abs(float(left @ right)) / denom
            if (direction_agreement is not None
                    and direction_agreement < 0.97):
                reasons.append("linear_direction_unstable")
    if first_type == second_type == "radial":
        first_polarity = first_model.get("polarity")
        second_polarity = second_model.get("polarity")
        valid_polarities = {"centre_to_edge", "edge_to_centre"}
        if (first_polarity not in valid_polarities
                or second_polarity not in valid_polarities):
            reasons.append("radial_polarity_missing_or_invalid")
        elif first_polarity != second_polarity:
            reasons.append("radial_polarity_changed")
        for prefix, model in (("discovery", first_model),
                              ("confirmation", second_model)):
            center = finite_vector(model.get("center", []), (2,))
            radii = finite_vector(
                (model.get("radius_x"), model.get("radius_y")), (2,))
            rotation = model.get("rotation_degrees")
            if (center is None or radii is None or (radii <= 0.0).any()
                    or isinstance(rotation, bool)
                    or not isinstance(rotation, (int, float))
                    or not math.isfinite(float(rotation))):
                reasons.append(f"{prefix}_radial_geometry_missing_or_invalid")
    first_stops = first_model.get("stop_count")
    second_stops = second_model.get("stop_count")
    if (not isinstance(first_stops, int) or isinstance(first_stops, bool)
            or not 2 <= first_stops <= 5
            or not isinstance(second_stops, int)
            or isinstance(second_stops, bool)
            or not 2 <= second_stops <= 5):
        reasons.append("stop_count_missing_or_invalid")
    elif abs(first_stops - second_stops) > 1:
        reasons.append("stop_count_unstable")
    return not reasons, {
        "passed": not reasons,
        "reasons": reasons,
        "model_type": first_type,
        "confirmation_model_type": second_type,
        "linear_direction_abs_cosine": (None if direction_agreement is None
                                         else round(direction_agreement, 6)),
        "discovery_stop_count": first_stops,
        "confirmation_stop_count": second_stops,
        "confirmation": _paint_fit_diagnostic(second),
    }


def _intersection_pixels(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Count intersection pixels using only the overlapping bbox crop."""
    lx0, ly0, lx1, ly1 = left["bbox"]
    rx0, ry0, rx1, ry1 = right["bbox"]
    x0, y0 = max(lx0, rx0), max(ly0, ry0)
    x1, y1 = min(lx1, rx1), min(ly1, ry1)
    if x0 >= x1 or y0 >= y1:
        return 0
    return int(np.logical_and(
        left["_mask"][y0:y1, x0:x1],
        right["_mask"][y0:y1, x0:x1],
    ).sum())


def _overlap_matrix(items: list[dict[str, Any]]) -> list[list[int]]:
    count = len(items)
    result = [[0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i + 1, count):
            value = _intersection_pixels(items[i], items[j])
            if value:
                result[i][j] = result[j][i] = value
    return result


def _geometry_shortlist(
    paint_eligible: list[dict[str, Any]],
    *,
    limit: int,
    max_objects: int,
) -> tuple[list[int], list[list[int]], dict[str, list[str]]]:
    """Reserve a bounded, overlap-aware and family-balanced geometry frontier.

    Geometry is expensive, so paint validation is completed for every source
    candidate first.  This shortlist deliberately keeps several different
    kinds of evidence: one large candidate per family, disjoint large-area
    ownership seeds, one high paint-gain candidate per family, and overlapping
    alternatives for the coverage seeds.  Remaining capacity alternates global
    area and paint-gain ranks.  No geometry result participates in this step.
    """
    count = len(paint_eligible)
    if count == 0:
        return [], [], {}
    capacity = min(max(1, int(limit)), count)
    intersections = _overlap_matrix(paint_eligible)
    selected: list[int] = []
    selected_set: set[int] = set()
    reasons: dict[str, list[str]] = {}

    def identity(index: int) -> str:
        return paint_eligible[index]["candidate_id"]

    def add(index: int, reason: str) -> bool:
        candidate_id = identity(index)
        if index in selected_set:
            reasons.setdefault(candidate_id, [])
            if reason not in reasons[candidate_id]:
                reasons[candidate_id].append(reason)
            return False
        if len(selected) >= capacity:
            return False
        selected.append(index)
        selected_set.add(index)
        reasons.setdefault(candidate_id, [])
        if reason not in reasons[candidate_id]:
            reasons[candidate_id].append(reason)
        return True

    area_order = sorted(range(count), key=lambda index: (
        -paint_eligible[index]["area"],
        _FAMILY_PRIORITY.get(
            paint_eligible[index]["candidate_family"], 99),
        -_paint_gain(paint_eligible[index]["_paint"]),
        identity(index),
    ))
    gain_order = sorted(range(count), key=lambda index: (
        -_paint_gain(paint_eligible[index]["_paint"]),
        -paint_eligible[index]["area"],
        _FAMILY_PRIORITY.get(
            paint_eligible[index]["candidate_family"], 99),
        identity(index),
    ))
    families = sorted(
        {item["candidate_family"] for item in paint_eligible},
        key=lambda family: (_FAMILY_PRIORITY.get(family, 99), family),
    )

    # Large disjoint ownership seeds preserve multiple adjacent objects.  They
    # are tracked separately from other reservations so overlap alternatives
    # can be kept later without making the seed set itself overlap.
    coverage_seeds: list[int] = []
    coverage_target = min(
        max(1, int(max_objects)),
        capacity,
        max(2 if capacity >= 2 else 1,
            int(math.ceil(capacity * 0.40))),
    )
    for index in area_order:
        if all(intersections[index][other] == 0 for other in coverage_seeds):
            if add(index, "disjoint_area_coverage_seed") or index in selected_set:
                coverage_seeds.append(index)
            if len(coverage_seeds) >= coverage_target:
                break

    # Family coverage prevents numerous high-scoring pair edges from starving
    # broad/community/chain hypotheses before geometry is even measured.  It
    # follows the disjoint seed reserve so a very tight caller cap still keeps
    # separate objects instead of spending every slot on overlapping aliases.
    for family in families:
        indices = [index for index in area_order
                   if paint_eligible[index]["candidate_family"] == family]
        if indices:
            add(indices[0], f"family_area_reserve:{family}")

    # A strong small ramp can have lower area than a broad candidate while
    # still carrying substantially better held-out evidence.  Preserve one
    # such option per family when it differs from the family-area representative.
    for family in families:
        indices = [index for index in gain_order
                   if paint_eligible[index]["candidate_family"] == family]
        if indices:
            add(indices[0], f"family_paint_gain_reserve:{family}")

    # Keep one competing ownership interpretation around each major seed.  It
    # will only survive the final set packing if its measured geometry makes
    # the whole non-overlapping solution better.
    for seed in coverage_seeds:
        alternatives = [
            index for index in range(count)
            if index != seed and intersections[seed][index] > 0
        ]
        alternatives.sort(key=lambda index: (
            -paint_eligible[index]["area"],
            -_paint_gain(paint_eligible[index]["_paint"]),
            _FAMILY_PRIORITY.get(
                paint_eligible[index]["candidate_family"], 99),
            identity(index),
        ))
        for alternative in alternatives:
            if alternative not in selected_set:
                add(alternative, f"overlap_alternative_for:{identity(seed)}")
                break

    # Alternate the global area and gain order so remaining capacity includes
    # more than one scale of candidate without making colour a geometry goal.
    cursor = 0
    while len(selected) < capacity and cursor < max(len(area_order), len(gain_order)):
        if cursor < len(area_order):
            add(area_order[cursor], "global_area_frontier")
        if cursor < len(gain_order):
            add(gain_order[cursor], "global_paint_gain_frontier")
        cursor += 1
    for index in area_order:
        if len(selected) >= capacity:
            break
        add(index, "deterministic_capacity_fill")

    return selected, intersections, reasons


def _beam_select(viable: list[dict[str, Any]], *, max_objects: int,
                 beam_width: int) -> tuple[list[int], list[list[int]]]:
    """Approximate maximum safe coverage with deterministic set packing.

    Colour evidence is an eligibility gate, not a contour objective.  Among
    non-overlapping eligible sets we maximise owned area, then minimise total
    anchors, segments and object fragments.  Paint gain is only a late tie
    breaker.  The bounded beam keeps this deterministic for the normal <=48
    candidate workload without exponential image-sized state.
    """
    count = len(viable)
    intersections = _overlap_matrix(viable)
    conflict_bits = [0] * count
    for i in range(count):
        for j in range(i + 1, count):
            value = intersections[i][j]
            if value:
                conflict_bits[i] |= 1 << j
                conflict_bits[j] |= 1 << i

    any_non_pair = any(item["candidate_family"] != "pair" for item in viable)
    pair_limit = int(max_objects) if not any_non_pair else max(0, int(max_objects) - 1)

    # (selected_bits, selected_indices, coverage, anchors, segments,
    #  object_count, pair_count, nonpair_count, weighted_paint_gain)
    states = [(0, (), 0, 0, 0, 0, 0, 0, 0.0)]

    def state_key(state: tuple[Any, ...]) -> tuple[Any, ...]:
        return (-state[2], state[3], state[4], state[5], state[6],
                -state[7], -round(state[8], 9), state[1])

    for index, item in enumerate(viable):
        next_states = list(states)
        bit = 1 << index
        for state in states:
            selected_bits, selected = state[0], state[1]
            if state[5] >= int(max_objects):
                continue
            if selected_bits & conflict_bits[index]:
                continue
            is_pair = item["candidate_family"] == "pair"
            if is_pair and state[6] >= pair_limit:
                continue
            next_states.append((
                selected_bits | bit,
                selected + (index,),
                state[2] + int(item["area"]),
                state[3] + int(item["geometry"]["anchor_count"]),
                state[4] + int(item["geometry"]["segment_count"]),
                state[5] + 1,
                state[6] + int(is_pair),
                state[7] + int(not is_pair),
                state[8] + float(item["area"]) * _paint_gain(item["_paint"]),
            ))
        # De-duplicate exact selected sets and keep a broad enough frontier to
        # preserve alternatives that skip a large overlapping early mask.
        unique = {state[0]: state for state in next_states}
        states = sorted(unique.values(), key=state_key)[:max(8, int(beam_width))]

    best = min(states, key=state_key)
    return list(best[1]), intersections


def propose_gradient_reconstruction(
    den: Any,
    lab_all: Any,
    visible: Any = None,
    palette: Any = None,
    *,
    vis_fill: Any = None,
    alpha: Any = None,
    original_source_rgba: Any = None,
    geometry_error_percent: float = 0.35,
    geometry_smooth: float = 0.55,
    max_candidates: int = 48,
    max_geometry_candidates: int = 16,
    max_objects: int = 12,
    max_segments_per_object: int = 4096,
    selection_beam_width: int = 256,
    candidate_options: Optional[dict[str, Any]] = None,
    model_options: Optional[dict[str, Any]] = None,
    geometry_optimizer: Optional[Callable[..., dict[str, Any]]] = None,
    geometry_fit_cache: Optional[dict[str, Any]] = None,
    candidate_provider: Callable[..., list[dict[str, Any]]] = propose_gradient_candidates,
    model_fitter: Callable[..., dict[str, Any]] = fit_gradient_object_proposal,
    progress: Optional[Callable[[dict[str, Any]], None]] = None,
    control: Any = None,
) -> dict[str, Any]:
    """Return deterministic, non-overlapping native gradient proposals.

    ``geometry_error_percent`` is the primary bidirectional P95 deviation
    budget relative to the source bbox diagonal.  Max/tail-share/salient-corner
    guards prevent the P95 allowance from hiding local breakage.  The geometry
    objective is lexicographic: preserve topology and pass those error guards,
    prefer a robust primitive, then minimise designer anchors, ordinary
    anchors, fragments and segments.  Source colours only choose the fill.

    An injected ``geometry_optimizer`` receives the ownership mask plus the
    explicit percentage and pixel budgets.  By default the bundled global
    tolerance-grid optimiser is used; :func:`curve_refit.fit_mask` remains the
    conservative fallback when that optional module is absent.

    ``geometry_fit_cache`` may be shared only by candidate builds for the
    same input image.  Entries are keyed by the exact full-canvas mask digest,
    bbox, geometry parameters and process-local solver identities; paint and
    colour values never enter or alter a cached geometry result.
    """
    def _checkpoint(substage: str, detail: str = "", *, current=None,
                    total=None) -> None:
        if control is not None:
            control.checkpoint(f"gradient_reconstruction:{substage}")
        if progress is not None:
            event = {
                "stage": "gradient_reconstruction",
                "substage": str(substage),
                "detail": str(detail),
            }
            if current is not None:
                event["gradient_candidate_current"] = int(current)
            if total is not None:
                event["gradient_candidate_total"] = int(total)
            if control is not None and hasattr(control, "snapshot"):
                event.update(control.snapshot())
                event["stage"] = "gradient_reconstruction"
                event["substage"] = str(substage)
            progress(event)

    if vis_fill is not None:
        visible = vis_fill
    if visible is None or palette is None:
        raise ValueError("visible/vis_fill and palette are required")
    geometry_error_percent = float(geometry_error_percent)
    if (not math.isfinite(geometry_error_percent)
            or not 0.01 <= geometry_error_percent <= 5.0):
        raise ValueError("geometry_error_percent must be between 0.01 and 5")
    if int(max_candidates) < 1 or int(max_objects) < 1:
        raise ValueError("max_candidates and max_objects must be positive")
    if int(max_geometry_candidates) < 1:
        raise ValueError("max_geometry_candidates must be positive")
    if int(selection_beam_width) < 8:
        raise ValueError("selection_beam_width must be at least 8")

    rgb, labels, vis, pal, alpha_array = _normalise_inputs(
        den, lab_all, visible, palette, alpha)
    candidate_kwargs = dict(candidate_options or {})
    candidate_kwargs["max_candidates"] = int(max_candidates)
    model_kwargs = dict(model_options or {})
    _checkpoint("candidate_discovery", "正在建立有界的漸層所有權候選")
    raw_candidates = candidate_provider(
        rgb[:, :, :3], labels, vis, pal, **candidate_kwargs)
    if not isinstance(raw_candidates, list):
        raise TypeError("candidate_provider must return a list")
    _checkpoint(
        "candidate_discovery_complete",
        f"已建立 {len(raw_candidates)} 個漸層候選",
        current=0, total=len(raw_candidates))

    prepared = []
    sampling_label_digests = {}
    for index, candidate in enumerate(raw_candidates, 1):
        if index == 1 or index % 4 == 0 or index == len(raw_candidates):
            _checkpoint(
                "candidate_preparation",
                f"正在整理漸層候選 {index}/{len(raw_candidates)}",
                current=index, total=len(raw_candidates))
        if not isinstance(candidate, dict):
            continue
        mask = np.asarray(candidate.get("mask"), dtype=np.bool_)
        if mask.shape != rgb.shape[:2] or not mask.any():
            continue
        candidate_id = str(candidate.get(
            "candidate_id", f"gradient-candidate-{index:04d}"))
        family = str(candidate.get("kind", "unknown"))
        sampling_labels = np.asarray(candidate.get("_sampling_labels", labels))
        sampling_palette = np.asarray(candidate.get("_sampling_palette", pal))
        if ("_sampling_labels" in candidate and (sampling_labels.shape != labels.shape
                or sampling_labels.dtype.kind not in "iu"
                or sampling_palette.ndim != 2 or sampling_palette.shape[1] != 3
                or len(sampling_palette) < 1
                or sampling_labels.min() < 0
                or sampling_labels.max() >= len(sampling_palette))):
            raise ValueError("candidate sampling labels and palette must be paired")
        identity = id(sampling_labels)
        if identity not in sampling_label_digests:
            sampling_label_digests[identity] = hashlib.sha256(
                np.ascontiguousarray(sampling_labels).tobytes()).hexdigest()
        prepared.append({
            "candidate_id": candidate_id,
            "candidate_family": family,
            "_mask": np.ascontiguousarray(mask),
            "bbox": _bbox(mask),
            "area": int(mask.sum()),
            "score": float(candidate.get("score", 0.0)),
            "component_ids": [int(value) for value in candidate.get(
                "component_ids", [])],
            "candidate_evidence": _json_safe(candidate.get("evidence", {})),
            "candidate_parent_ids": [str(value) for value in candidate.get(
                "parent_ids", [])],
            "candidate_overlap_info": _json_safe(candidate.get(
                "overlap_info", [])),
            "_component_context": candidate.get("_component_context"),
            "_sampling_labels": sampling_labels,
            "_sampling_labels_digest": sampling_label_digests[identity],
            "_mask_digest": _mask_digest(mask),
        })
    prepared.sort(key=lambda item: (
        item["candidate_id"], _FAMILY_PRIORITY.get(item["candidate_family"], 99),
        item["bbox"], item["_mask_digest"]))

    paint_eligible: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    direction_search_min_area = max(512, int(round(int(vis.sum()) * 0.01)))
    direction_search_enabled = 0
    for item in prepared:
        fit_kwargs = dict(model_kwargs)
        # Direction search is intentionally selective.  Large, connected
        # field candidates can hide a diagonal colour flow that a single
        # regression axis misses; tiny pairs do not justify the additional
        # fits and are more vulnerable to unstable directions.
        precertified_partition = _certified_weak_bridge_partition(item)
        external_smooth_edges = int(
            (item.get("candidate_evidence") or {}).get(
                "external_smooth_edges") or 0)
        potentially_ownership_ready = bool(
            external_smooth_edges == 0 or precertified_partition is not None)
        expandable_seed = bool(
            external_smooth_edges > 0
            and item["candidate_family"] in {"community", "monotonic_chain"})
        complete_field_policy = None
        if (
            model_fitter is fit_gradient_object_proposal
            and potentially_ownership_ready
            and item["candidate_family"] in {
                "smooth_field", "community", "monotonic_chain"}
        ):
            fit_kwargs, complete_field_policy = _complete_field_fit_policy(
                fit_kwargs)
            evidence = dict(item.get("candidate_evidence") or {})
            evidence["complete_field_paint_policy"] = complete_field_policy
            item["candidate_evidence"] = evidence
        if (
            model_fitter is fit_gradient_object_proposal
            and item["candidate_family"] in {
                "smooth_field", "community", "monotonic_chain"}
            and (potentially_ownership_ready or expandable_seed)
            and int(item["area"]) >= direction_search_min_area
        ):
            fit_kwargs.setdefault("linear_direction_candidates", 5)
            direction_search_enabled += 1
        item["_precertified_partition"] = precertified_partition
        item["_model_fit_options"] = fit_kwargs
        item["_paint_cache_key"] = (
            item["_mask_digest"],
            item["_sampling_labels_digest"],
            json.dumps(
                _json_safe(fit_kwargs), ensure_ascii=True, sort_keys=True,
                separators=(",", ":")),
        )

    fits: list[Any]
    if model_fitter is fit_gradient_object_proposal and prepared:
        # Several discovery families can yield the exact same ownership mask.
        # The default paint fitter receives no candidate metadata, so an exact
        # mask+options key is a proof that the calculation and result are
        # identical.  Fit each unique key once, with a small bounded thread
        # pool; NumPy releases the GIL in the expensive colour operations.
        unique_jobs = {}
        for item in prepared:
            unique_jobs.setdefault(
                item["_paint_cache_key"],
                (item["_mask"], item["_model_fit_options"], item["_sampling_labels"]),
            )
        try:
            configured_workers = int(os.environ.get(
                "AVC_GRADIENT_WORKERS", "4"))
        except ValueError:
            configured_workers = 4
        worker_count = min(
            len(unique_jobs), max(1, min(4, configured_workers)),
            max(1, int(os.cpu_count() or 1)),
        )
        ordered_jobs = list(unique_jobs.items())
        _checkpoint(
            "paint_model_fit",
            f"正在驗證 {len(unique_jobs)} 個唯一漸層色彩模型"
            f"（{worker_count} 個有界工作緒）",
            current=0, total=len(unique_jobs))

        def _fit_default(job):
            key, (mask, options, fit_labels) = job
            return key, model_fitter(
                rgb[:, :, :3], mask, alpha=alpha_array,
                label_map=fit_labels, **options)

        if worker_count > 1:
            with ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix="aivc-gradient-paint") as executor:
                futures = [executor.submit(_fit_default, job)
                           for job in ordered_jobs]
                fitted_pairs = []
                for fit_index, future in enumerate(
                        as_completed(futures), 1):
                    fitted_pairs.append(future.result())
                    _checkpoint(
                        "paint_model_fit_progress",
                        f"已驗證漸層色彩模型 {fit_index}/{len(futures)}",
                        current=fit_index, total=len(futures))
        else:
            fitted_pairs = []
            for fit_index, job in enumerate(ordered_jobs, 1):
                fitted_pairs.append(_fit_default(job))
                _checkpoint(
                    "paint_model_fit_progress",
                    f"已驗證漸層色彩模型 {fit_index}/{len(ordered_jobs)}",
                    current=fit_index, total=len(ordered_jobs))
        fit_cache = dict(fitted_pairs)
        fits = [copy.deepcopy(fit_cache[item["_paint_cache_key"]])
                for item in prepared]
    else:
        # Injected fitters are an extension/test seam and may intentionally
        # keep call-order state.  Preserve the legacy one-call-per-candidate
        # sequence exactly for that contract.
        fits = []
        for paint_index, item in enumerate(prepared, 1):
            _checkpoint(
                "paint_model_fit",
                f"正在驗證漸層色彩模型 {paint_index}/{len(prepared)}",
                current=paint_index, total=len(prepared))
            fits.append(model_fitter(
                rgb[:, :, :3], item["_mask"], alpha=alpha_array,
                label_map=item["_sampling_labels"], **item["_model_fit_options"]))

    for paint_index, (item, fit) in enumerate(zip(prepared, fits), 1):
        _checkpoint(
            "paint_model_fit_complete",
            f"已取得漸層色彩模型 {paint_index}/{len(prepared)}",
            current=paint_index, total=len(prepared))
        if not isinstance(fit, dict) or fit.get("status") != "proposed":
            decisions.append({
                "candidate_id": item["candidate_id"],
                "candidate_family": item["candidate_family"],
                "status": "paint_model_rejected",
                "area": item["area"],
                "bbox_xyxy": list(item["bbox"]),
                "component_ids": item["component_ids"],
                "candidate_evidence": item["candidate_evidence"],
                "reasons": _json_safe(fit.get("reasons", [
                    "paint_model_not_proposed"]) if isinstance(fit, dict)
                    else ["paint_model_invalid_result"]),
                "paint_status": (fit.get("status") if isinstance(fit, dict)
                                  else "error"),
                "paint_diagnostic": _paint_fit_diagnostic(fit),
            })
            continue
        item["_paint"] = fit
        paint_eligible.append(item)

    # Community/chain discovery candidates are often only the strongest core
    # of a quantised colour field.  Grow those seeds over adjacent eligible
    # components using source-space colour evidence, then refit the complete
    # union.  The original fitter holdout is not consulted while deciding
    # which neighbours to add; it only accepts or skips the generated union.
    model_guided_expansion_attempts = 0
    model_guided_expansions_proposed = 0
    model_guided_expansion_components_added = 0
    model_guided_expansion_refit_rejected = 0
    model_guided_expansion_no_compatible = 0
    expanded_paint_eligible: list[dict[str, Any]] = []
    for expansion_index, item in enumerate(paint_eligible, 1):
        _checkpoint(
            "ownership_expansion",
            f"正在驗證漸層所有權 {expansion_index}/{len(paint_eligible)}",
            current=expansion_index, total=len(paint_eligible))
        evidence = item.get("candidate_evidence") or {}
        external_edges = int(evidence.get("external_smooth_edges") or 0)
        if (external_edges <= 0 or item["candidate_family"] not in {
                "community", "monotonic_chain"}):
            expanded_paint_eligible.append(item)
            continue
        model_guided_expansion_attempts += 1
        expanded, expansion = _model_guided_graph_expansion(
            item, rgb, item["_sampling_labels"], alpha_array,
            model_fitter=model_fitter,
            model_options=dict(item.get("_model_fit_options", model_kwargs)),
            candidate_options=candidate_kwargs,
            apply_complete_field_policy=(
                model_fitter is fit_gradient_object_proposal),
        )
        status = str(expansion.get("status"))
        if expanded is item:
            expanded = dict(item)
            expanded_evidence = dict(item.get("candidate_evidence") or {})
            expanded_evidence["model_guided_expansion"] = expansion
            expanded["candidate_evidence"] = expanded_evidence
            if status == "expanded_refit_rejected":
                expanded["_expansion_refit_failed"] = True
                expanded["_precertified_partition"] = None
        if status.startswith("expanded_refit_proposed"):
            model_guided_expansions_proposed += 1
            model_guided_expansion_components_added += len(
                expansion.get("included_components", []))
            decision_status = "model_guided_ownership_expanded"
            decision_reasons = [
                "source_space_model_compatible_adjacent_components_added",
                "expanded_union_formal_refit_passed",
                "pending_independent_salted_revalidation",
            ]
        elif status == "expanded_refit_rejected":
            model_guided_expansion_refit_rejected += 1
            decision_status = "model_guided_expansion_refit_rejected"
            decision_reasons = [
                "expanded_union_formal_refit_failed",
                "original_seed_retained_for_defer",
                "no_geometry_mutation_authorised",
            ]
        else:
            model_guided_expansion_no_compatible += 1
            # Keep the public decision vocabulary backward-compatible: this
            # candidate is still an incomplete ownership seed and therefore
            # deferred.  The more specific expansion outcome remains in the
            # reasons and model_guided_expansion audit below.
            decision_status = "ownership_seed_deferred"
            decision_reasons = [
                "model_guided_expansion_no_compatible_neighbour",
                "no_adjacent_component_passed_material_and_model_compatibility",
                "original_seed_retained",
            ]
        decisions.append({
            "candidate_id": item["candidate_id"],
            "candidate_family": item["candidate_family"],
            "status": decision_status,
            "area": int(item["area"]),
            "component_ids": list(item["component_ids"]),
            "reasons": decision_reasons,
            "model_guided_expansion": expansion,
        })
        expanded_paint_eligible.append(expanded)
    paint_eligible = expanded_paint_eligible

    # A pair/chain/community with a smooth eligible edge leading to an omitted
    # component is an evidence seed, not a complete paint object.  Emitting it
    # would recreate the exact failure designers reported: one native gradient
    # over two bands while neighbouring bands remain separate solid shapes.
    ownership_ready: list[dict[str, Any]] = []
    ownership_seed_deferred = 0
    certified_partition_count = 0
    tiny_model_guided_rejected = 0
    for item in paint_eligible:
        evidence = item.get("candidate_evidence") or {}
        external_edges = int(evidence.get("external_smooth_edges") or 0)
        containment_parents = item.get("candidate_parent_ids") or []
        has_graph_evidence = "external_smooth_edges" in evidence
        original_family = item["candidate_family"]
        partition = item.get("_precertified_partition")
        if partition is None and not item.get("_expansion_refit_failed"):
            partition = _certified_weak_bridge_partition(item)
        if partition is not None:
            item = dict(item)
            evidence = dict(evidence)
            evidence.update(partition)
            evidence["original_candidate_family"] = original_family
            item["candidate_evidence"] = evidence
            item["candidate_family"] = "model_guided_field"
            certified_partition_count += 1
            # A graph partition made of several sub-minimum crumbs is not a
            # useful editable object even when its paint fit is numerically
            # valid.  Require roughly one fitter-sized evidence block per
            # owned component; this removes tiny noisy gradient fragments
            # without imposing a resolution-dependent percentage threshold.
            fit_min_pixels = max(16, int(model_kwargs.get("min_pixels", 128)))
            minimum_partition_area = max(
                256, fit_min_pixels * max(1, len(item["component_ids"])))
            if int(item["area"]) < minimum_partition_area:
                tiny_model_guided_rejected += 1
                decisions.append({
                    "candidate_id": item["candidate_id"],
                    "candidate_family": item["candidate_family"],
                    "status": "tiny_model_guided_partition_rejected",
                    "area": item["area"],
                    "bbox_xyxy": list(item["bbox"]),
                    "component_ids": item["component_ids"],
                    "candidate_evidence": item["candidate_evidence"],
                    "minimum_partition_area": minimum_partition_area,
                    "reasons": [
                        "paint_validation_passed",
                        "owned_partition_too_small_for_useful_editable_object",
                        "no_geometry_mutation_authorised",
                    ],
                    "paint_diagnostic": _paint_fit_diagnostic(item["_paint"]),
                })
                continue
        containment_fallback = bool(
            not has_graph_evidence
            and original_family in {
                "pair", "community", "monotonic_chain"}
            and containment_parents)
        incomplete = bool(
            original_family != "source_chromatic"
            and partition is None
            and not evidence.get(
                "ownership_closed_after_model_guided_expansion", False)
            and (external_edges > 0 or containment_fallback))
        if not incomplete:
            ownership_ready.append(item)
            continue
        ownership_seed_deferred += 1
        decisions.append({
            "candidate_id": item["candidate_id"],
            "candidate_family": item["candidate_family"],
            "status": "ownership_seed_deferred",
            "area": item["area"],
            "bbox_xyxy": list(item["bbox"]),
            "component_ids": item["component_ids"],
            "candidate_evidence": item["candidate_evidence"],
            "reasons": [
                "paint_validation_passed",
                "candidate_is_incomplete_smooth_field_seed",
                "gradient_must_own_complete_continuous_object",
            ],
            "paint_status": "proposed",
            "external_smooth_edges": external_edges,
            "external_smooth_shared_boundary": int(evidence.get(
                "external_smooth_shared_boundary") or 0),
            "containment_parent_ids": containment_parents,
            "paint_diagnostic": _paint_fit_diagnostic(item["_paint"]),
        })

    # Discovery may inspect many candidate masks, so every real-provider object
    # that could reach geometry is fitted again with an independently salted
    # coordinate holdout.  This prevents adaptive candidate search from
    # turning the original heldout set into training data by repetition.
    geometry_ready: list[dict[str, Any]] = []
    paint_revalidation_attempts = 0
    paint_revalidation_passed = 0
    paint_revalidation_rejected = 0
    model_guided_expansion_revalidation_rejected = 0
    for confirmation_index, item in enumerate(ownership_ready, 1):
        _checkpoint(
            "paint_revalidation",
            f"正在做獨立色彩複驗 {confirmation_index}/{len(ownership_ready)}",
            current=confirmation_index, total=len(ownership_ready))
        if not isinstance(item.get("_component_context"), dict):
            item = dict(item)
            item["_paint_revalidation"] = {
                "passed": True,
                "status": "not_required_for_contextless_custom_provider",
            }
            geometry_ready.append(item)
            continue
        confirmation_kwargs = dict(item.get(
            "_model_fit_options", model_kwargs))
        base_seed = int(confirmation_kwargs.get("validation_seed", 0))
        confirmation_kwargs["validation_seed"] = base_seed ^ 0x5A17
        paint_revalidation_attempts += 1
        try:
            confirmation = model_fitter(
                rgb[:, :, :3], item["_mask"], alpha=alpha_array,
                label_map=item["_sampling_labels"], **confirmation_kwargs)
        except Exception as exc:
            confirmation = {
                "status": "error",
                "reasons": [
                    f"independent_revalidation_error:{type(exc).__name__}:{exc}"[:240]
                ],
            }
        stable, stability = _paint_models_stable(item["_paint"], confirmation)
        if not stable:
            paint_revalidation_rejected += 1
            decisions.append({
                "candidate_id": item["candidate_id"],
                "candidate_family": item["candidate_family"],
                "status": "independent_paint_revalidation_rejected",
                "area": item["area"],
                "bbox_xyxy": list(item["bbox"]),
                "component_ids": item["component_ids"],
                "candidate_evidence": item["candidate_evidence"],
                "reasons": [
                    "independent_holdout_or_model_stability_failed",
                    "no_geometry_mutation_authorised",
                ],
                "paint_revalidation": stability,
            })
            seed = item.get("_expansion_seed_snapshot")
            if isinstance(seed, dict):
                model_guided_expansion_revalidation_rejected += 1
                decisions.append({
                    "candidate_id": item["candidate_id"],
                    "candidate_family": seed.get(
                        "candidate_family", "community"),
                    "status": (
                        "ownership_seed_deferred_after_expansion_"
                        "revalidation_failure"),
                    "area": int(seed.get("area", 0)),
                    "bbox_xyxy": seed.get("bbox_xyxy", []),
                    "component_ids": seed.get("component_ids", []),
                    "reasons": [
                        "expanded_field_independent_revalidation_failed",
                        "rolled_back_to_original_seed",
                        "original_seed_not_deliverable_as_complete_object",
                        "no_geometry_mutation_authorised",
                    ],
                })
            continue
        item = dict(item)
        item["_paint_confirmation"] = confirmation
        item["_paint_revalidation"] = stability
        paint_revalidation_passed += 1
        geometry_ready.append(item)

    shortlist_indices, _shortlist_intersections, shortlist_reasons = (
        _geometry_shortlist(
            geometry_ready,
            limit=int(max_geometry_candidates),
            max_objects=int(max_objects),
        ))
    shortlist_set = set(shortlist_indices)
    for index, item in enumerate(geometry_ready):
        if index in shortlist_set:
            continue
        decisions.append({
            "candidate_id": item["candidate_id"],
            "candidate_family": item["candidate_family"],
            "status": "geometry_shortlist_deferred",
            "area": item["area"],
            "reasons": [
                "paint_validation_passed",
                "geometry_budget_reserved_for_balanced_shortlist",
                "candidate_not_rejected_and_may_be_revisited",
            ],
            "paint_status": "proposed",
            "paint_gain": round(_paint_gain(item["_paint"]), 6),
        })

    viable: list[dict[str, Any]] = []
    geometry_cache: dict[str, tuple[Optional[dict[str, Any]], list[str]]] = {}
    geometry_jobs = {}
    for index in shortlist_indices:
        item = geometry_ready[index]
        geometry_jobs.setdefault(
            item["_mask_digest"], (item["_mask"], item["bbox"]))

    def _run_serial_geometry_job(job):
        digest, (mask, bbox) = job
        return digest, _fit_geometry(
            mask, bbox,
            error_budget_percent=geometry_error_percent,
            smooth=float(geometry_smooth),
            max_segments=int(max_segments_per_object),
            geometry_optimizer=geometry_optimizer,
        )

    ordered_geometry_jobs = list(geometry_jobs.items())
    shared_geometry_state = (
        _normalise_geometry_fit_cache(geometry_fit_cache)
        if geometry_optimizer is None else None)
    shared_geometry_hits = 0
    shared_key_by_digest = {}
    pending_geometry_jobs = []
    for digest, (mask, bbox) in ordered_geometry_jobs:
        if shared_geometry_state is not None:
            shared_key = _geometry_fit_cache_key(
                digest,
                bbox,
                error_budget_percent=float(geometry_error_percent),
                smooth=float(geometry_smooth),
                max_segments=int(max_segments_per_object),
            )
            shared_key_by_digest[digest] = shared_key
            shared_audit = shared_geometry_state["audit"]
            shared_audit["requests"] += 1
            cached = shared_geometry_state["entries"].get(shared_key)
            if cached is not None:
                shared_audit["hits"] += 1
                shared_geometry_hits += 1
                geometry_cache[digest] = copy.deepcopy(cached)
                continue
            shared_audit["misses"] += 1
        pending_geometry_jobs.append((digest, (mask, bbox)))

    def _record_geometry_outcome(digest, outcome):
        geometry_cache[digest] = outcome
        if shared_geometry_state is None:
            return
        shared_key = shared_key_by_digest[digest]
        shared_geometry_state["entries"][shared_key] = copy.deepcopy(outcome)
        shared_geometry_state["audit"]["stores"] += 1

    process_pool_safe = (
        os.environ.get("AVC_GRADIENT_PROCESS_POOL_SAFE") == "1"
        and os.environ.get("AVC_GRADIENT_PROCESS_POOL_OWNER_PID")
        == str(os.getpid()))
    if (geometry_optimizer is None and len(pending_geometry_jobs) > 1
            and process_pool_safe):
        try:
            configured_geometry_workers = int(os.environ.get(
                "AVC_GRADIENT_GEOMETRY_WORKERS", "4"))
        except ValueError:
            configured_geometry_workers = 4
        geometry_worker_count = min(
            len(pending_geometry_jobs),
            max(1, min(4, configured_geometry_workers)),
            max(1, int(os.cpu_count() or 1)),
        )
    elif not pending_geometry_jobs:
        geometry_worker_count = 0
    else:
        # Injected optimizers are an extension/test seam and may intentionally
        # keep call-order state.  Preserve their serial legacy contract.
        geometry_worker_count = 1
    _checkpoint(
        "geometry_fit",
        f"正在驗證 {len(pending_geometry_jobs)} 個新漸層幾何"
        f"（跨候選命中 {shared_geometry_hits}；"
        f"{geometry_worker_count} 個有界程序）",
        current=0, total=len(pending_geometry_jobs))
    if geometry_worker_count > 1:
        process_jobs = [
            (
                digest,
                mask,
                bbox,
                float(geometry_error_percent),
                float(geometry_smooth),
                int(max_segments_per_object),
            )
            for digest, (mask, bbox) in pending_geometry_jobs
        ]
        # Always use spawn.  The stage previously completed bounded paint
        # threads, so forking that interpreter on POSIX would be unsafe; spawn
        # also matches the Windows workbench delivery path exactly.
        with ProcessPoolExecutor(
                max_workers=geometry_worker_count,
                mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(
                _fit_default_geometry_process_job, job)
                for job in process_jobs]
            for completed_count, future in enumerate(
                    as_completed(futures), 1):
                digest, outcome = future.result()
                _record_geometry_outcome(digest, outcome)
                _checkpoint(
                    "geometry_fit_progress",
                    f"已驗證漸層幾何 "
                    f"{completed_count}/{len(futures)}",
                    current=completed_count, total=len(futures))
    else:
        for completed_count, job in enumerate(pending_geometry_jobs, 1):
            digest, outcome = _run_serial_geometry_job(job)
            _record_geometry_outcome(digest, outcome)
            _checkpoint(
                "geometry_fit_progress",
                f"已驗證漸層幾何 "
                f"{completed_count}/{len(ordered_geometry_jobs)}",
                current=completed_count, total=len(pending_geometry_jobs))

    geometry_optimizer_calls = len(pending_geometry_jobs)
    geometry_cache_hits = 0
    consumed_geometry_digests = set()
    paint_ready_alternatives = []
    for geometry_index, index in enumerate(shortlist_indices, 1):
        _checkpoint(
            "geometry_fit_complete",
            f"正在整理漸層幾何 {geometry_index}/{len(shortlist_indices)}",
            current=geometry_index, total=len(shortlist_indices))
        item = geometry_ready[index]
        digest = item["_mask_digest"]
        cache_hit = digest in consumed_geometry_digests
        if cache_hit:
            geometry_cache_hits += 1
        consumed_geometry_digests.add(digest)
        geometry, reasons = geometry_cache[digest]
        if geometry is None:
            # Keep independently revalidated paint as an uncommitted option.
            # A failed contour never becomes a selected object or certificate.
            paint = item['_paint']
            paint_ready_alternatives.append(_json_safe({
                'schema': 'ai-vector-cleanroom.paint-ready-alternative/v1',
                'candidate_id': item['candidate_id'],
                'candidate_family': item['candidate_family'],
                'component_ids': item['component_ids'],
                'area': item['area'], 'bbox_xyxy': list(item['bbox']),
                'mask': encode_mask_rle(item['_mask']),
                'model': paint.get('model'), 'stops': paint.get('stops', []),
                'heldout_evidence': {
                    'scope': 'processed_reference_only',
                    'error': paint.get('error'), 'validation': paint.get('validation'),
                    'independent_revalidation': item.get('_paint_revalidation')},
                'reasons': reasons,
                'status': 'pending_native_source_and_existing_path_validation',
                'geometry_certified': False, 'original_source_verified': False,
            }))
            decisions.append({
                "candidate_id": item["candidate_id"],
                "candidate_family": item["candidate_family"],
                "status": "geometry_rejected",
                "area": item["area"],
                "reasons": reasons,
                "paint_status": "proposed",
            })
            continue
        item["geometry"] = geometry
        item["geometry_shortlist_reasons"] = shortlist_reasons.get(
            item["candidate_id"], [])
        item["geometry_cache_hit"] = cache_hit
        viable.append(item)

    # Largest ownership hypotheses are considered first; beam selection still
    # preserves skip alternatives and applies the full set objective.
    viable.sort(key=lambda item: (
        -item["area"], item["geometry"]["anchor_count"],
        item["geometry"]["segment_count"],
        _FAMILY_PRIORITY.get(item["candidate_family"], 99),
        item["candidate_id"], _mask_digest(item["_mask"])))
    _checkpoint("selection", "正在選擇互不重疊的合格漸層物件")
    selected_indices, intersections = _beam_select(
        viable, max_objects=int(max_objects), beam_width=int(selection_beam_width))
    selected_set = set(selected_indices)

    proposals = []
    rejected_by_selected: dict[int, list[str]] = {
        index: [] for index in selected_indices}
    for index, item in enumerate(viable):
        if index in selected_set:
            continue
        overlaps = []
        for chosen in selected_indices:
            intersection = intersections[index][chosen]
            if intersection:
                overlaps.append({
                    "selected_candidate_id": viable[chosen]["candidate_id"],
                    "intersection_pixels": int(intersection),
                    "self_fraction": round(intersection / item["area"], 6),
                    "selected_fraction": round(
                        intersection / viable[chosen]["area"], 6),
                })
                rejected_by_selected[chosen].append(item["candidate_id"])
        if overlaps:
            status = "overlap_rejected"
            reasons = ["overlaps_selected_gradient_object"]
        elif item["candidate_family"] == "pair" and any(
                value["candidate_family"] != "pair" for value in viable):
            status = "pair_quota_rejected"
            reasons = ["pair_family_cannot_consume_all_object_slots"]
        else:
            status = "selection_capacity_rejected"
            reasons = ["lower_lexicographic_set_utility"]
        decisions.append({
            "candidate_id": item["candidate_id"],
            "candidate_family": item["candidate_family"],
            "status": status,
            "area": item["area"],
            "reasons": reasons,
            "overlaps": overlaps,
            "geometry": {
                "anchor_count": item["geometry"]["anchor_count"],
                "segment_count": item["geometry"]["segment_count"],
                "actual_max_error_percent": item["geometry"][
                    "error_budget"]["actual_max_error_percent"],
            },
        })

    for chosen in selected_indices:
        item = viable[chosen]
        paint = item["_paint"]
        geometry = item["geometry"]
        proposal = {
            "proposal_id": f"gradient-object-{len(proposals) + 1:04d}",
            "candidate_id": item["candidate_id"],
            "candidate_family": item["candidate_family"],
            "component_ids": item["component_ids"],
            "area": item["area"],
            "bbox_xyxy": list(item["bbox"]),
            "mask": encode_mask_rle(item["_mask"]),
            "path": geometry["path"],
            "fill_rule": geometry["fill_rule"],
            "model": _json_safe(paint.get("model")),
            "stops": _json_safe(paint.get("stops", [])),
            "confidence": float(paint.get("confidence", 0.0)),
            "heldout_evidence": _json_safe({
                "error": paint.get("error"),
                "validation": paint.get("validation"),
                "reasons": paint.get("reasons", []),
                "independent_revalidation": item.get(
                    "_paint_revalidation"),
            }),
            "candidate_evidence": item["candidate_evidence"],
            "geometry": geometry,
            "topology": geometry["topology"],
            "economy": {
                "anchors_before": geometry["anchors_before"],
                "anchors_after": geometry["anchor_count"],
                "designer_anchor_count": geometry["designer_anchor_count"],
                "segment_count": geometry["segment_count"],
                "native_primitive_count": len(geometry["native_primitives"]),
                "primitive_first": geometry["primitive_first"],
            },
            "selection": {
                "objective": [
                    "hard:paint_two_holdouts_and_topology_and_geometry_error",
                    "maximise:safe_owned_area",
                    "minimise:anchor_count",
                    "minimise:segment_count",
                    "minimise:object_fragments",
                    "tie_break:paint_improvement_then_candidate_id",
                ],
                "colour_used_for_geometry": False,
                "overlap_rejections": sorted(rejected_by_selected[chosen]),
                "geometry_shortlist_reasons": item[
                    "geometry_shortlist_reasons"],
                "geometry_mask_cache_hit": item["geometry_cache_hit"],
            },
        }
        proposals.append(_json_safe(proposal))
        decisions.append({
            "candidate_id": item["candidate_id"],
            "candidate_family": item["candidate_family"],
            "status": "selected",
            "proposal_id": proposal["proposal_id"],
            "area": item["area"],
            "reasons": ["passed_paint_and_independent_revalidation_and_geometry_hard_gates",
                        "selected_by_nonoverlap_lexicographic_objective"],
        })

    # These source-supported alternatives never replace baseline ownership at
    # discovery time. The SVG assembler must compare the complete native scene
    # against its actual baseline before committing any added component.
    enclosed_source_pending = 0
    if original_source_rgba is not None and proposals:
        from gradient_source_components import propose_enclosed_source_components
        masks = [decode_mask_rle(proposal["mask"]) for proposal in proposals]
        fit_options_by_id = {item["candidate_id"]: item.get("_model_fit_options", model_kwargs)
                             for item in viable}
        for index, proposal in enumerate(proposals):
            other_owners = np.zeros_like(masks[index])
            for other_index, ownership in enumerate(masks):
                if other_index != index:
                    other_owners |= ownership
            try:
                alternative = propose_enclosed_source_components(
                    proposal, rgb[:, :, :3], original_source_rgba, vis,
                    other_owners, alpha=alpha_array, labels=labels,
                    model_fit_options=fit_options_by_id.get(proposal["candidate_id"], model_kwargs),
                    geometry_error_percent=geometry_error_percent,
                    geometry_smooth=float(geometry_smooth),
                    max_segments=int(max_segments_per_object), model_fitter=model_fitter)
                if alternative is not None:
                    proposal["enclosed_source_component_candidate"] = alternative
                    enclosed_source_pending += 1
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                decisions.append({"candidate_id": proposal["candidate_id"],
                    "status": "enclosed_source_component_proposal_rejected",
                    "reasons": [str(exc)[:240]], "baseline_unchanged": True})

    decisions.sort(key=lambda item: (item["candidate_id"], item["status"]))
    pending_paints, pending_keys = [], set()
    for option in sorted(paint_ready_alternatives, key=lambda entry: (-entry['area'], entry['candidate_id'])):
        key = json.dumps({field: option[field] for field in ('mask', 'model', 'stops')}, sort_keys=True)
        if key in pending_keys:
            continue
        pending_keys.add(key)
        pending_paints.append(option)
        if len(pending_paints) == 8:
            break
    covered = int(sum(proposal["area"] for proposal in proposals))
    result = {
        "schema": SCHEMA,
        "status": "proposed" if proposals else "skipped",
        "proposals": proposals,
        "paint_ready_alternatives": pending_paints,
        "decisions": _json_safe(decisions),
        "summary": {
            "paint_ready_alternatives_pending_native_validation": len(pending_paints),
            "candidates_generated": len(prepared),
            "paint_eligible": len(paint_eligible),
            "direction_search_area_threshold": direction_search_min_area,
            "direction_search_candidates": direction_search_enabled,
            "model_guided_expansion_attempts": (
                model_guided_expansion_attempts),
            "model_guided_expansions_proposed": (
                model_guided_expansions_proposed),
            "model_guided_expansion_components_added": (
                model_guided_expansion_components_added),
            "model_guided_expansion_refit_rejected": (
                model_guided_expansion_refit_rejected),
            "model_guided_expansion_no_compatible_neighbour": (
                model_guided_expansion_no_compatible),
            "model_guided_expansion_revalidation_rejected": (
                model_guided_expansion_revalidation_rejected),
            "ownership_complete_paint_eligible": len(ownership_ready),
            "ownership_seed_deferred": ownership_seed_deferred,
            "weak_bridge_partitions_certified": certified_partition_count,
            "tiny_model_guided_partitions_rejected": tiny_model_guided_rejected,
            "independent_paint_revalidation_attempts": paint_revalidation_attempts,
            "independent_paint_revalidation_passed": paint_revalidation_passed,
            "independent_paint_revalidation_rejected": paint_revalidation_rejected,
            "geometry_shortlisted": len(shortlist_indices),
            "geometry_shortlist_deferred": (
                len(geometry_ready) - len(shortlist_indices)),
            "unique_geometry_masks_evaluated": len(ordered_geometry_jobs),
            "geometry_optimizer_calls": geometry_optimizer_calls,
            "shared_geometry_fit_cache_hits": shared_geometry_hits,
            "geometry_mask_cache_hits": geometry_cache_hits,
            "paint_and_geometry_eligible": len(viable),
            "objects_selected": len(proposals),
            "covered_pixels": covered,
            "pair_objects_selected": sum(
                proposal["candidate_family"] == "pair" for proposal in proposals),
            "total_anchors": sum(
                proposal["economy"]["anchors_after"] for proposal in proposals),
            "total_segments": sum(
                proposal["economy"]["segment_count"] for proposal in proposals),
            "overlap_pixels_between_selected": 0,
            "enclosed_source_component_pending_native_transactions": enclosed_source_pending,
        },
        "geometry_shortlist": {
            "policy": [
                "paint_validate_all_candidates_first",
                "expand_community_and_chain_seeds_on_material_safe_graph_edges",
                "select_expansion_neighbours_without_outer_holdout",
                "refit_expanded_union_before_independent_confirmation",
                "bound_tail_policy_to_ownership_complete_structural_fields",
                "defer_incomplete_smooth_field_seeds_before_geometry",
                "certify_only_graph_communities_cut_by_weak_bridges",
                "require_independently_salted_paint_revalidation",
                "reserve_one_large_candidate_per_family",
                "reserve_disjoint_large_area_coverage_seeds",
                "reserve_one_high_paint_gain_candidate_per_family",
                "reserve_overlap_alternatives_for_coverage_seeds",
                "alternate_global_area_and_paint_gain_frontiers",
                "cache_identical_masks_by_sha256",
            ],
            "limit": int(max_geometry_candidates),
            "candidate_ids": [
                geometry_ready[index]["candidate_id"]
                for index in shortlist_indices
            ],
            "reasons_by_candidate": _json_safe({
                geometry_ready[index]["candidate_id"]: shortlist_reasons.get(
                    geometry_ready[index]["candidate_id"], [])
                for index in shortlist_indices
            }),
            "optimizer_calls": geometry_optimizer_calls,
            "shared_fit_cache_hits": shared_geometry_hits,
            "mask_cache_hits": geometry_cache_hits,
            "scope_note": (
                "Only shortlisted paint-valid candidates received geometry "
                "optimisation. Incomplete pair/chain/community seeds cannot be "
                "delivered until they own the complete smooth field; capacity-"
                "deferred complete candidates may be revisited with a larger "
                "budget."
            ),
        },
        "objective": {
            "hard_constraints": [
                "paint_model_beats_solid_on_deterministic_heldout",
                "no_material_internal_hard_edge",
                "topology_preserved",
                "p95_geometry_error_percent_within_budget",
                "maximum_geometry_error_within_three_times_budget_tail",
                "selected_masks_are_pairwise_disjoint",
            ],
            "geometry_minimise_in_order": [
                "designer_anchor_count", "anchor_count",
                "object_fragments", "segment_count",
                "simpler_primitive_category_when_economy_equal",
            ],
            "set_select_in_order": [
                "safe_owned_area_max", "anchor_count_min",
                "segment_count_min", "object_fragments_min",
            ],
            "set_selection_scope": "geometry_shortlist_only",
            "colour_used_for_geometry": False,
            "pixel_boundary_is_guardrail_not_curve_target": True,
        },
        "parameters": {
            "geometry_error_percent": round(geometry_error_percent, 6),
            "geometry_smooth": float(geometry_smooth),
            "max_candidates": int(max_candidates),
            "max_geometry_candidates": int(max_geometry_candidates),
            "max_objects": int(max_objects),
            "selection_beam_width": int(selection_beam_width),
            "geometry_optimizer": (
                ("geometry_error_optimizer.optimize_compound_contours"
                 if optimize_compound_contours is not None
                 else "curve_refit.fit_mask")
                if geometry_optimizer is None
                else getattr(geometry_optimizer, "__name__",
                             "injected_optimizer")
            ),
        },
    }
    result = _json_safe(result)
    # A strict serialization pass is part of the public contract.
    json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return result


# Discoverable aliases for integration code.
build_gradient_reconstruction_stage = propose_gradient_reconstruction
reconstruct_gradient_objects = propose_gradient_reconstruction


__all__ = [
    "GEOMETRY_FIT_CACHE_SCHEMA",
    "MASK_SCHEMA",
    "SCHEMA",
    "build_gradient_reconstruction_stage",
    "decode_mask_rle",
    "encode_mask_rle",
    "geometry_fit_cache_audit",
    "new_geometry_fit_cache",
    "propose_gradient_reconstruction",
    "reconstruct_gradient_objects",
]
