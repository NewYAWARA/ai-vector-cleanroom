"""An additional colour-discovery hypothesis, never a drawing/ownership mask.

Small slender source components can dominate quantisation with their fringe
colours.  Omitting their samples is useful even when they cannot safely become
editable strokes.  Every real fill pixel is still passed to discovery and to
the renderer.  This module neither approves strokes nor approves gradients.
"""
from __future__ import annotations

import hashlib
import numpy as np

SCHEMA = "ai-vector-cleanroom.palette-sampling/v1"
MAX_PIXELS = 4_194_304


def _digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def independent_sampling_mask(rgb, visible, counter_mask=None):
    """Return an estimation-only mask and JSON-safe evidence.

    Bounds are work-resolution budgets: component area scales with canvas area
    relative to 1024 squared (64..8192 pixels); slenderness is dimensionless.
    At most 15% of artwork samples may be omitted.  A line-only drawing, tiny
    sample pool, oversized canvas or no eligible component keeps the baseline.
    White is the existing preprocessing neutral, not a claim of paper meaning.
    """
    image = np.asarray(rgb)
    vis = np.asarray(visible, dtype=bool)
    if image.shape != vis.shape + (3,) or vis.ndim != 2:
        raise ValueError("RGB and visibility shapes must match")
    counter = (np.zeros_like(vis) if counter_mask is None
               else np.asarray(counter_mask, dtype=bool))
    if counter.shape != vis.shape:
        raise ValueError("counter sample shape must match")
    baseline = vis | counter
    audit = {"schema": SCHEMA, "status": "baseline", "reason": "not_needed",
             "sampling_only": True, "renderable_mask_changed": False,
             "source_visible_pixels": int(vis.sum()), "excluded_pixels": 0,
             "counter_samples_preserved": True, "components": []}
    if vis.size > MAX_PIXELS:
        audit["reason"] = "canvas_exceeds_sampling_budget"
        return baseline, audit
    if int(vis.sum()) < 512:
        audit["reason"] = "insufficient_artwork_samples"
        return baseline, audit
    # Source-connected, colour-independent components: a multicolour line is
    # one component.  Never infer a stroke from individual palette fragments.
    from stroke_engine import connected_components, _dilate_one
    contrast = np.asarray(image, dtype=np.float32) - 255.0
    ink = vis & (np.einsum("ijk,ijk->ij", contrast, contrast) > 24.0 ** 2)
    runs = ink.copy()
    runs[:, 1:] &= ~ink[:, :-1]
    run_counts = runs.sum(axis=1)
    # The existing run-based labeler compares neighbouring rows' runs.  Bound
    # fragmented/noisy input before entering it, not after it has done work.
    if int(run_counts.sum()) > 16384 or int(run_counts.max(initial=0)) > 512:
        audit["reason"] = "source_fragmentation_exceeds_sampling_budget"
        return baseline, audit
    labels, count = connected_components(ink)
    max_area = max(64, min(8192, int(round(vis.size / 512))))
    audit["parameters"] = {
        "neutral_rgb": [255, 255, 255], "contrast_euclidean": 24,
        "component_area_min": 32, "component_area_max": max_area,
        "bbox_diagonal_squared_over_area_min": 20,
        "max_excluded_fraction": .15, "minimum_remaining_samples": 512,
        "connectivity": 8, "maximum_canvas_pixels": MAX_PIXELS}
    audit["parameters"].update(maximum_runs=16384, maximum_runs_per_row=512)
    if count > 16384:
        audit["reason"] = "component_count_exceeds_sampling_budget"
        return baseline, audit
    area = np.bincount(labels.ravel(), minlength=count + 1)
    yy, xx = np.nonzero(ink)
    ids = labels[yy, xx]
    xmin = np.full(count + 1, vis.shape[1], dtype=np.int32)
    xmax = np.zeros(count + 1, dtype=np.int32)
    ymin = np.full(count + 1, vis.shape[0], dtype=np.int32)
    ymax = np.zeros(count + 1, dtype=np.int32)
    np.minimum.at(xmin, ids, xx); np.maximum.at(xmax, ids, xx)
    np.minimum.at(ymin, ids, yy); np.maximum.at(ymax, ids, yy)
    chosen = []
    for index in np.flatnonzero((area >= 32) & (area <= max_area)):
        if index == 0:
            continue
        width = int(xmax[index] - xmin[index] + 1)
        height = int(ymax[index] - ymin[index] + 1)
        slenderness = (width * width + height * height) / int(area[index])
        if slenderness >= 20:
            chosen.append(int(index))
            if len(audit["components"]) < 64:
                audit["components"].append({"area": int(area[index]),
                    "bbox_xyxy": [int(xmin[index]), int(ymin[index]),
                                  int(xmax[index] + 1), int(ymax[index] + 1)],
                    "slenderness": round(slenderness, 6)})
    audit["eligible_component_count"] = len(chosen)
    if not chosen:
        return baseline, audit
    excluded = _dilate_one(np.isin(labels, chosen)) & vis & ~counter
    omitted = int(excluded.sum())
    remaining = int(vis.sum()) - omitted
    if omitted > .15 * int(vis.sum()) or remaining < 512:
        audit.update(reason="insufficient_independent_fill_samples",
                     proposed_excluded_pixels=omitted)
        return baseline, audit
    samples = (vis & ~excluded) | counter
    audit.update(status="proposed", reason="independent_slender_sample_hypothesis",
                 excluded_pixels=omitted, remaining_artwork_samples=remaining,
                 sampling_mask_sha256=_digest(samples))
    return samples, audit


def discover_with_sampling_hypothesis(rgb, labels, visible, palette, *,
                                     hypothesis, max_candidates=48,
                                     discovery=None, **kwargs):
    """Merge paired discovery contexts under the original total candidate cap.

    Two thirds of slots are reserved for baseline discovery (including its
    family quotas); at most one third introduce the new hypothesis.  Equal
    masks prefer baseline evidence.  Spare slots return to the baseline.
    Full visibility and source RGB are passed unchanged to both discoveries.
    """
    if discovery is None:
        from gradient_candidate_groups import propose_gradient_candidates
        discovery = propose_gradient_candidates
    limit = max(0, int(max_candidates))
    if not limit:
        return []
    alternate_palette = np.asarray(hypothesis["palette"])
    alternate_labels = np.asarray(hypothesis["lab_all"])
    if alternate_labels.shape != np.asarray(labels).shape:
        raise ValueError("sampling labels must match full source canvas")
    if (alternate_palette.ndim != 2 or alternate_palette.shape[1] != 3
            or alternate_labels.dtype.kind not in "iu"
            or alternate_labels.min() < 0
            or alternate_labels.max() >= len(alternate_palette)):
        raise ValueError("invalid paired sampling palette and labels")
    if limit < 6:
        return discovery(rgb, labels, visible, palette,
                         max_candidates=limit, **kwargs)
    alt_limit = limit // 3
    baseline = discovery(rgb, labels, visible, palette,
                         max_candidates=limit, **kwargs)
    # Ask the provider to preserve its pair/structural/broad family quotas,
    # rather than slicing the first N globally high-ranked pairs.
    reserved = discovery(rgb, labels, visible, palette,
                         max_candidates=limit - alt_limit, **kwargs)
    alternate_pool = discovery(rgb, alternate_labels, visible, alternate_palette,
                               max_candidates=limit, **kwargs)
    # This hypothesis addresses coherent fill fields.  A score-only slice is
    # dominated by small pair edges and would throw away the broad fields it
    # was created to discover.  Reserve half its slots for large structural
    # unions, one quarter for source fields, then use provider score order.
    alternate = []
    alt_seen = set()
    def reserve(items, count):
        used = 0
        for item in items:
            key = _digest(np.asarray(item["mask"], dtype=bool))
            if key in alt_seen:
                continue
            alt_seen.add(key); alternate.append(item); used += 1
            if used >= count:
                break
    area_order = lambda item: (-int(item.get("area", np.asarray(item["mask"]).sum())),
                              -float(item.get("score", 0)), str(item["candidate_id"]))
    reserve(sorted((v for v in alternate_pool if v.get("kind") in {
        "smooth_field", "community", "monotonic_chain"}), key=area_order), max(1, alt_limit // 2))
    reserve(sorted((v for v in alternate_pool if v.get("kind") in {
        "source_chromatic", "source_single_component"}), key=area_order), max(1, alt_limit // 4))
    reserve(alternate_pool, max(1, alt_limit - len(alternate)))
    alternate = alternate[:alt_limit]
    out, seen = [], set()
    label_hash = _digest(alternate_labels)
    palette_hash = _digest(alternate_palette)

    def take(items, origin, cap):
        prefix = "baseline" if origin == "baseline" else "sampling"
        digests = {str(v["candidate_id"]): _digest(np.asarray(v["mask"], dtype=bool))
                   for v in items}
        renamed = {key: prefix + ":" + value[:24] for key, value in digests.items()}
        for candidate in items:
            if len(out) >= cap:
                break
            key = digests[str(candidate["candidate_id"])]
            if key in seen:
                continue
            seen.add(key)
            item = dict(candidate)
            # IDs remain stable across merged budgets; metadata references
            # must stay within their own discovery graph.
            item["candidate_id"] = renamed[str(candidate["candidate_id"])]
            item["parent_ids"] = [renamed.get(str(v), prefix + ":missing:" + str(v))
                                  for v in candidate.get("parent_ids", [])]
            item["overlap_info"] = [dict(v) for v in candidate.get("overlap_info", [])]
            for overlap in item["overlap_info"]:
                for field in ("candidate_id", "other_candidate_id"):
                    if field in overlap:
                        overlap[field] = renamed.get(str(overlap[field]), prefix + ":missing:" + str(overlap[field]))
            evidence = dict(item.get("evidence") or {})
            evidence["palette_sampling"] = {
                "schema": SCHEMA, "origin": origin, "sampling_only": True,
                "renderable_mask_changed": False,
                "candidate_limit": limit, "baseline_reserved": limit - alt_limit,
                "sampling_reserved": alt_limit}
            if origin != "baseline":
                item["_sampling_labels"] = alternate_labels
                item["_sampling_palette"] = alternate_palette
                evidence["palette_sampling"].update(
                    labels_sha256=label_hash, palette_sha256=palette_hash)
            item["evidence"] = evidence
            out.append(item)

    take(reserved, "baseline", limit - alt_limit)
    take(alternate, "independent_slender_samples", min(limit, len(out) + alt_limit))
    take(baseline, "baseline", limit)
    return out
