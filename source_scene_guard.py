"""Source-supported validation for an explicitly opted-in reconstruction stage.

This is NOT a replacement for the ordinary before/after SVG topology guard.
It permits repairs of old trace errors only where the original pixels support
them. It cannot establish semantic editability or continuous vector topology.
All spatial evidence is at the original image dimensions; global MAE never
compensates for a small new hole, closed channel, or lost white object.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image

ALPHA_THRESHOLDS = (32, 128, 224)
MAX_SCENE_SIDE = 8192
MAX_SCENE_PIXELS = 8 * 1024 * 1024
MAX_RUNS = 500_000
MAX_REPORTED_REGIONS = 100
MAX_HOLES = 10_000


def _base():
    return {"schema": "aivc.source-scene-guard/v1", "accepted": False,
            "status": "unverified_fail_closed", "reasons": [],
            "scope": "source_reconstruction_only_not_generic_guard_relaxation",
            "native_resolution": True, "multi_alpha": [],
            "localized_defects": [], "source_supported_hole_repairs": []}


def _array(value):
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 4 or array.dtype != np.uint8:
        raise ValueError("source_scene_requires_uint8_rgba")
    height, width = array.shape[:2]
    if not (0 < width <= MAX_SCENE_SIDE and 0 < height <= MAX_SCENE_SIDE
            and width * height <= MAX_SCENE_PIXELS):
        raise ValueError("source_scene_native_pixel_budget_exceeded")
    return array


def _read_rgba(path):
    with Image.open(path) as image:
        width, height = image.size
        if not (0 < width <= MAX_SCENE_SIDE and 0 < height <= MAX_SCENE_SIDE
                and width * height <= MAX_SCENE_PIXELS):
            raise ValueError("source_scene_native_pixel_budget_exceeded")
        return np.asarray(image.convert("RGBA"))


def _near(mask):
    height, width = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[y:y + height, x:x + width]
                                 for y in range(3) for x in range(3)])


def _components(mask):
    """Bounded scanline union-find with *four*-connected background regions.

    Do not reuse stroke_engine.connected_components: despite its old docstring
    that routine joins diagonals. A diagonal contact is not an open channel.
    """
    labels = np.zeros(mask.shape, np.int32)
    parents = [0]

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    previous = []
    for y, row in enumerate(mask):
        indices = np.flatnonzero(row)
        if not indices.size:
            previous = []
            continue
        splits = np.flatnonzero(np.diff(indices) > 1)
        starts = np.r_[indices[0], indices[splits + 1]]
        ends = np.r_[indices[splits], indices[-1]]
        current, cursor = [], 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            while cursor < len(previous) and previous[cursor][1] < start:
                cursor += 1
            overlaps, scan = [], cursor
            while scan < len(previous) and previous[scan][0] <= end:
                overlaps.append(find(previous[scan][2]))
                scan += 1
            if overlaps:
                label = min(overlaps)
                for other in overlaps:
                    parents[other] = label
            else:
                label = len(parents)
                if label > MAX_RUNS:
                    raise ValueError("source_scene_component_budget_exceeded")
                parents.append(label)
            labels[y, start:end + 1] = label
            current.append((start, end, label))
        previous = current
    roots = np.array([find(index) for index in range(len(parents))], np.int32)
    _, dense = np.unique(roots, return_inverse=True)
    labels = dense.astype(np.int32)[labels]
    return labels, int(dense.max())


def _holes(alpha, threshold):
    labels, count = _components(alpha < threshold)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    kept = sizes > 0  # One-pixel holes are evidence too; no area cutoff.
    kept[np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = False
    kept[0] = False
    if int(kept.sum()) > MAX_HOLES:
        raise ValueError("source_scene_hole_budget_exceeded")
    return labels, kept


def _composite(rgba, background=255):
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    return rgba[:, :, :3].astype(np.float32) * alpha + np.asarray(background) * (1 - alpha)


def _source_masks(source, reference):
    alpha = source[:, :, 3]
    if np.any(alpha < 254):
        foreground, background = alpha >= 240, alpha <= 8
        return foreground, background, {"kind": "original_alpha", "confident": True,
                                       "paper_inference": False}, None
    # Opaque white pixels are not intrinsically empty: both an independently
    # processed alpha reference and a consistent light paper border are needed.
    border = np.concatenate((source[0, :, :3], source[-1, :, :3],
                             source[:, 0, :3], source[:, -1, :3])).astype(np.int16)
    paper = np.median(border, axis=0)
    border_fraction = float(np.mean(np.max(np.abs(border - paper), axis=1) <= 5))
    light_fraction = float(np.mean(np.min(border, axis=1) >= 235))
    reference_border_fraction = None
    if reference is not None:
        a = reference[:, :, 3]
        reference_border_fraction = float(np.mean(np.r_[a[0], a[-1], a[:, 0], a[:, -1]] <= 8))
    confident = (reference is not None and float(paper.min()) >= 235
                 and border_fraction >= .98 and light_fraction >= .98
                 and reference_border_fraction >= .98)
    evidence = {"kind": "opaque_paper_with_processed_reference" if confident else "opaque_ambiguous",
                "confident": bool(confident), "paper_inference": bool(confident),
                "paper_rgb": paper.tolist(), "consistent_border_fraction": border_fraction,
                "light_border_fraction": light_fraction,
                "processed_clear_border_fraction": reference_border_fraction,
                "processed_reference_present": reference is not None}
    if not confident:
        return np.zeros(alpha.shape, bool), np.zeros(alpha.shape, bool), evidence, paper
    distance = np.max(np.abs(source[:, :, :3].astype(np.float32) - paper), axis=2)
    # Clearly colored source pixels override an accidental reference erasure.
    # Retained near-white reference pixels protect white objects, not paper.
    foreground = ((distance >= 48)
                  | ((reference[:, :, 3] >= 240) & ((distance >= 24) | (distance <= 8))))
    background = (distance <= 8) & (reference[:, :, 3] <= 8)
    return foreground, background, evidence, paper


def _regions(mask):
    labels, count = _components(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ids = sorted(range(1, count + 1), key=lambda key: (-int(sizes[key]), key))
    return labels, ids, sizes


def _region_record(mask, source, before, after, reference=None):
    ys, xs = np.nonzero(mask)
    record = {"pixels": int(xs.size),
              "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
    # Distributed concrete probes, including both ends, not a mean-color claim.
    probes = []
    for index in np.unique(np.linspace(0, xs.size - 1, min(5, xs.size), dtype=int)):
        x, y = int(xs[index]), int(ys[index])
        point = {"xy": [x, y], "source_rgba": source[y, x].tolist(),
                 "before_rgba": before[y, x].tolist(), "after_rgba": after[y, x].tolist()}
        if reference is not None:
            point["processed_reference_rgba"] = reference[y, x].tolist()
        probes.append(point)
    record["probes"] = probes
    return record


def _add_defects(result, kind, mask, arrays, threshold=None):
    if not np.any(mask):
        return
    if kind not in result["reasons"]:
        result["reasons"].append(kind)
    labels, ids, sizes = _regions(mask)
    result.setdefault("defect_totals", []).append({"kind": kind, "alpha_threshold": threshold,
                                                   "pixels": int(mask.sum()), "regions": len(ids)})
    for label in ids[:max(0, MAX_REPORTED_REGIONS - len(result["localized_defects"]))]:
        record = _region_record(labels == label, *arrays)
        record.update(kind=kind, alpha_threshold=threshold)
        result["localized_defects"].append(record)
    if len(ids) > MAX_REPORTED_REGIONS:
        result["localized_defects_truncated"] = True


def _hole_correspondence(result, before_alpha, after_alpha, threshold,
                         foreground, background, source, before, after, reference,
                         certified_aa, source_exterior_background):
    a, keep_a = _holes(before_alpha, threshold)
    b, keep_b = _holes(after_alpha, threshold)
    ids_a, ids_b = np.flatnonzero(keep_a), np.flatnonzero(keep_b)
    # Direct region overlap, not equality of hole counts or nearest-centroid
    # guesses. Coverage validation separately checks growth, shrinkage and cuts.
    both = keep_a[a] & keep_b[b]
    radix = len(keep_b)
    keys = np.unique(a[both].astype(np.int64) * radix + b[both])
    pairs = [[int(key // radix), int(key % radix)] for key in keys]
    matched_a, matched_b = {p[0] for p in pairs}, {p[1] for p in pairs}
    row = {"alpha_threshold": threshold, "holes_before": int(len(ids_a)),
           "holes_after": int(len(ids_b)), "connectivity": 4, "minimum_region_pixels": 1,
           "correspondence": "direct_spatial_overlap_with_local_coverage_checks",
           "pairs": pairs, "created": [], "erased": [],
           "splits": sum(sum(p[0] == int(i) for p in pairs) > 1 for i in ids_a),
           "merges": sum(sum(p[1] == int(i) for p in pairs) > 1 for i in ids_b)}
    arrays = (source, before, after, reference)
    for label in ids_b:
        if int(label) in matched_b:
            continue
        mask = b == label
        record = _region_record(mask, *arrays)
        record["label"] = int(label)
        row["created"].append(record)
        if np.all(background[mask]):
            if np.any(source_exterior_background[mask]):
                # Source paper connected to the exterior is a visible open
                # channel. Sealing its mouth can create a hole whose pixels
                # still all match source white; that is not restoration of an
                # originally closed source hole. This topology condition is
                # independent of AA/boundary allowances and whole-image MAE.
                closing = (_near(mask) & source_exterior_background
                           & (before_alpha < threshold) & (after_alpha >= threshold))
                _add_defects(result,"closed_source_open_channel",
                             closing if np.any(closing) else mask,arrays,threshold)
                record["kind"]="sealed_exterior_connected_source_paper"
            else:
                result["source_supported_hole_repairs"].append({**record, "alpha_threshold": threshold,
                                                               "kind": "restored_source_empty_region"})
        elif np.all(certified_aa[mask]):
            # A threshold island on a source antialias boundary is not a hard
            # void when every pixel agrees with the original within 8/255.
            # Improvement alone is insufficient (the D2E crack improves too).
            record["kind"] = "source_matched_antialias_boundary"
        else:
            # In particular, closing an OLD open crack can create a NEW hole:
            # every pixel may improve, but it is still an unsupported hole.
            kind = "created_hole_on_source_ink" if np.any(foreground[mask]) else "created_hole_source_ambiguous"
            _add_defects(result, kind, mask, arrays, threshold)
    for label in ids_a:
        if int(label) in matched_a:
            continue
        mask = a == label
        painted = mask & (after_alpha >= threshold)
        record = _region_record(mask, *arrays)
        record["label"] = int(label)
        row["erased"].append(record)
        if not np.any(painted):
            # Opening a hole to the exterior is assessed by the newly opened
            # channel pixels, not incorrectly labelled a filled true hole.
            record["kind"] = "opened_to_exterior_checked_by_coverage"
        elif np.all(foreground[painted]):
            result["source_supported_hole_repairs"].append({**record, "alpha_threshold": threshold,
                                                           "kind": "filled_source_supported_false_hole"})
        else:
            kind = "filled_source_true_hole" if np.any(background[painted]) else "erased_hole_source_ambiguous"
            _add_defects(result, kind, painted, arrays, threshold)
    result["multi_alpha"].append(row)


def _feature_source_costs(regions, before_rgb, after_rgb, source_rgb):
    """Necessary source-cost gates on each bound feature's exact local support."""
    h,w=source_rgb.shape[:2]
    if not isinstance(regions,list) or len(regions)>MAX_HOLES:
        raise ValueError('source_scene_feature_support_budget_or_type')
    failed=np.zeros((h,w),bool);rows=[];total_area=0
    for region in regions:
        if not isinstance(region,dict):raise ValueError('source_scene_feature_support_invalid')
        box=region.get('bbox_xyxy')
        if (not isinstance(box,list) or len(box)!=4
                or any(isinstance(v,bool) or not isinstance(v,int) for v in box)):
            raise ValueError('source_scene_feature_support_invalid')
        x0,y0,x1,y1=box
        if not(0<=x0<x1<=w and 0<=y0<y1<=h):
            raise ValueError('source_scene_feature_support_outside_canvas')
        size=(x1-x0)*(y1-y0);total_area+=size
        if total_area>MAX_SCENE_PIXELS*4:
            raise ValueError('source_scene_feature_support_budget_or_type')
        payload=region.get('support_mask_packbits_hex')
        if not isinstance(payload,str) or len(payload)!=2*((size+7)//8):
            raise ValueError('source_scene_feature_support_encoding_invalid')
        packed=bytes.fromhex(payload)
        bits=np.unpackbits(np.frombuffer(packed,dtype=np.uint8))
        if np.any(bits[size:]):raise ValueError('source_scene_feature_support_padding_invalid')
        mask=bits[:size].reshape(y1-y0,x1-x0).astype(bool)
        if not mask.any() or int(mask.sum())!=region.get('support_pixels'):
            raise ValueError('source_scene_feature_support_pixel_count_invalid')
        section=np.s_[y0:y1,x0:x1]
        old=np.mean(np.abs(before_rgb[section]-source_rgb[section]),axis=2)
        new=np.mean(np.abs(after_rgb[section]-source_rgb[section]),axis=2)
        cost0=float(old[mask].mean());cost1=float(new[mask].mean());passed=cost1<=cost0+1e-6
        rows.append({'kind':region.get('kind'),'bbox_xyxy':box,'support_pixels':int(mask.sum()),
                     'before':cost0,'after':cost1,'nonregression':passed})
        if not passed:failed[section] |= mask & (new>old+1e-6)
    return rows,failed


def validate_source_scene_arrays(before_rgba, after_rgba, source_rgba, *,
                                 processed_reference_rgba=None,
                                 alpha_thresholds=ALPHA_THRESHOLDS, roi_xyxy=None,
                                 _boundary_allowance=None, _boundary_evidence=None):
    """Evaluate RGBA render arrays; errors return an unverified rejection.

    Original and rendered dimensions must agree. The processed reference is
    evidence for paper/retained white objects, never a replacement original.
    Pixel-sized defects are retained even on otherwise improved large images.
    """
    result = _base()
    try:
        before, after, source = map(_array, (before_rgba, after_rgba, source_rgba))
        reference = None if processed_reference_rgba is None else _array(processed_reference_rgba)
        if before.shape != source.shape or after.shape != source.shape or (reference is not None and reference.shape != source.shape):
            raise ValueError("source_scene_dimensions_differ_no_resampling")
        thresholds = tuple(alpha_thresholds)
        if (not thresholds or any(isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= 254 for t in thresholds)
                or len(set(thresholds)) != len(thresholds)):
            raise ValueError("invalid_source_scene_alpha_thresholds")
        height, width = source.shape[:2]
        boundary_allow = np.zeros((height, width), bool)
        boundary_bound=False
        if _boundary_evidence is not None:
            result["source_boundary_evidence"] = _boundary_evidence
            if _boundary_evidence.get("verified") is True and _boundary_allowance is not None:
                from source_edge_reconstruction import rgba_sha256
                claimed = np.asarray(_boundary_allowance)
                binding = _boundary_evidence.get("render_binding", {})
                if (claimed.shape == boundary_allow.shape and claimed.dtype == np.bool_
                        and _boundary_evidence.get("allowance_sha256") == hashlib.sha256(claimed.tobytes()).hexdigest()
                        and binding.get("before") == rgba_sha256(before)
                        and binding.get("after") == rgba_sha256(after)
                        and _boundary_evidence.get("source_rgba_sha256") == rgba_sha256(source)
                        and reference is not None
                        and _boundary_evidence.get("processed_rgba_sha256") == rgba_sha256(reference)):
                    boundary_allow = claimed
                    boundary_bound=True
                else:
                    raise ValueError("source_boundary_array_binding_mismatch")
        result["boundary_mode"] = "verified_native_free_paper_envelope" if np.any(boundary_allow) else "strict"
        result["width"], result["height"] = width, height
        foreground, background, evidence, paper = _source_masks(source, reference)
        source_background_labels, _ = _components(background)
        exterior_ids=np.unique(np.r_[source_background_labels[0],source_background_labels[-1],
                                     source_background_labels[:,0],source_background_labels[:,-1]])
        exterior_ids=exterior_ids[exterior_ids!=0]
        source_exterior_background=np.isin(source_background_labels,exterior_ids) & background
        unknown = ~(foreground | background)
        evidence.update(foreground_pixels=int(foreground.sum()), background_pixels=int(background.sum()),
                        ambiguous_pixels=int(unknown.sum()))
        result["source_support"] = evidence
        source_rgb, before_rgb, after_rgb = map(_composite, (source, before, after))
        error_before = np.max(np.abs(before_rgb - source_rgb), axis=2)
        error_after = np.max(np.abs(after_rgb - source_rgb), axis=2)
        result["metrics"] = {"white_composite_rgb_mae_before": float(np.mean(np.abs(before_rgb - source_rgb))),
                             "white_composite_rgb_mae_after": float(np.mean(np.abs(after_rgb - source_rgb))),
                             "global_mae_is_not_acceptance": True}
        epsilon = 1e-6
        cost = result["metrics"]
        cost["nonregression_epsilon"] = epsilon
        cost["global_color_nonregression"] = (cost["white_composite_rgb_mae_after"]
                                              <= cost["white_composite_rgb_mae_before"] + epsilon)
        roi_mask = None
        if roi_xyxy is not None:
            if (not isinstance(roi_xyxy, (list, tuple)) or len(roi_xyxy) != 4
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in roi_xyxy)):
                raise ValueError("source_scene_roi_requires_native_integer_xyxy")
            x0, y0, x1, y1 = roi_xyxy
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("source_scene_roi_outside_native_canvas")
            roi = np.s_[y0:y1, x0:x1]
            roi_before = float(np.mean(np.abs(before_rgb[roi] - source_rgb[roi])))
            roi_after = float(np.mean(np.abs(after_rgb[roi] - source_rgb[roi])))
            cost.update(roi_xyxy=list(roi_xyxy), roi_rgb_mae_before=roi_before,
                        roi_rgb_mae_after=roi_after, roi_color_nonregression=roi_after <= roi_before + epsilon)
            roi_mask = np.zeros((height, width), bool)
            roi_mask[roi] = True
        # The boundary's own target ROI is mandatory even when a caller asks
        # for an additional, smaller reporting ROI. A transaction chain keeps
        # every independently established local baseline as a necessary gate.
        additional_rois = []
        if np.any(boundary_allow):
            additional_rois = _boundary_evidence.get("required_rois", [])
            if not additional_rois and _boundary_evidence.get("roi_xyxy"):
                additional_rois = [_boundary_evidence["roi_xyxy"]]
        required_costs = []
        failed_roi_mask = np.zeros((height,width),bool)
        if cost.get("roi_color_nonregression") is False:
            failed_roi_mask |= roi_mask
        for box in additional_rois:
            if (not isinstance(box,(list,tuple)) or len(box)!=4
                    or any(isinstance(v,bool) or not isinstance(v,int) for v in box)):
                raise ValueError("source_scene_roi_requires_native_integer_xyxy")
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=width and 0<=y0<y1<=height):
                raise ValueError("source_scene_roi_outside_native_canvas")
            section=np.s_[y0:y1,x0:x1]
            old_cost=float(np.mean(np.abs(before_rgb[section]-source_rgb[section])))
            new_cost=float(np.mean(np.abs(after_rgb[section]-source_rgb[section])))
            passed=new_cost<=old_cost+epsilon
            required_costs.append({"roi_xyxy":list(box),"before":old_cost,"after":new_cost,"nonregression":passed})
            if not passed:
                failed_roi_mask[section]=True
        if required_costs:
            cost["required_roi_costs"]=required_costs
            cost["all_required_roi_nonregression"]=all(r["nonregression"] for r in required_costs)
        feature_costs,failed_feature_mask=_feature_source_costs(
            _boundary_evidence.get('source_feature_regions',[]) if boundary_bound else [],
            before_rgb,after_rgb,source_rgb)
        if feature_costs:
            cost['required_source_feature_costs']=feature_costs
            cost['all_source_feature_nonregression']=all(r['nonregression'] for r in feature_costs)
        old_alpha, new_alpha = before[:, :, 3], after[:, :, 3]
        # A limited AA allowance requires actual source agreement. It cannot
        # waive any created-hole check. Unknown opaque areas outside a proven
        # processed boundary get no allowance, even if white RGB looks similar.
        if evidence["kind"] == "original_alpha":
            source_alpha = source[:, :, 3].astype(np.int16)
            aa_allow = ((np.abs(new_alpha.astype(np.int16) - source_alpha)
                         <= np.abs(old_alpha.astype(np.int16) - source_alpha))
                        & (error_after <= error_before + 1)
                        & (source_alpha > 8) & (source_alpha < 240))
        elif evidence["confident"]:
            reference_alpha = reference[:, :, 3]
            boundary = _near(reference_alpha >= 240) & _near(reference_alpha <= 8)
            distance = np.max(np.abs(source[:, :, :3].astype(np.float32) - paper), axis=2)
            # Necessary alpha lower bound for compositing any RGB on paper.
            lower_bound = distance / np.maximum(paper.max(), 255 - paper.min()) * 255
            aa_allow = (boundary & (error_after <= error_before + 1)
                        & (new_alpha.astype(np.float32) + 2 >= lower_bound)
                        & ~(foreground & (distance <= 8)))
        else:
            aa_allow = np.zeros(old_alpha.shape, bool)
        result["antialias_policy"] = "boundary_source_agreement_only_created_hole_error_at_most_8"
        arrays = (source, before, after, reference)
        # Closed holes first so they cannot be omitted from capped diagnostics
        # after many low-priority boundary reports.
        for threshold in thresholds:
            hole_foreground, hole_background = foreground, background
            if evidence["kind"] == "original_alpha":
                hole_foreground = source[:, :, 3] >= threshold
                hole_background = source[:, :, 3] < threshold
            _hole_correspondence(result, old_alpha, new_alpha, threshold, hole_foreground, hole_background,
                                 source, before, after, reference, aa_allow & (error_after <= 8),
                                 source_exterior_background)
        for threshold in thresholds:
            emptied = (old_alpha >= threshold) & (new_alpha < threshold)
            painted = (old_alpha < threshold) & (new_alpha >= threshold)
            _add_defects(result, "new_gap_on_source_ink", emptied & foreground & ~aa_allow & ~boundary_allow, arrays, threshold)
            _add_defects(result, "new_paint_on_source_empty", painted & background & ~aa_allow & ~boundary_allow, arrays, threshold)
            _add_defects(result, "coverage_change_source_ambiguous", (emptied | painted) & unknown & ~aa_allow & ~boundary_allow, arrays, threshold)
        # A white opaque fill can hide ink without changing alpha at all.
        paper_rgb = 255 if paper is None else paper
        src_ink = np.max(np.abs(source_rgb - paper_rgb), axis=2) >= 48
        old_distance = np.max(np.abs(before_rgb - paper_rgb), axis=2)
        new_distance = np.max(np.abs(after_rgb - paper_rgb), axis=2)
        _add_defects(result, "new_white_exposure_on_source_ink",
                     foreground & src_ink & (new_distance <= 16) & (old_distance > 16) & ~boundary_allow, arrays)
        _add_defects(result, "new_color_on_source_empty",
                     background & (new_distance >= 16) & (new_distance > old_distance + 8) & ~boundary_allow, arrays)
        _add_defects(result, "new_local_color_error",
                     (foreground | background) & (error_after >= 48) & (error_after > error_before + 16) & ~boundary_allow, arrays)
        # Unknown opaque regions cannot silently change paint merely because
        # neither alpha nor a global mean notices a tiny new defect.
        _add_defects(result, "paint_change_source_ambiguous",
                     unknown & ~aa_allow & ~boundary_allow & (np.max(np.abs(after_rgb - before_rgb), axis=2) > 8), arrays)
        # Necessary conditions in addition to, never instead of, every local
        # topology/paint check above. A small uniform colour error must not slip
        # under the severe-defect thresholds. A supplied local baseline cannot
        # be compensated by improvements elsewhere on the canvas either.
        if not cost["global_color_nonregression"] or np.any(failed_roi_mask):
            worse = (np.mean(np.abs(after_rgb - source_rgb), axis=2)
                     > np.mean(np.abs(before_rgb - source_rgb), axis=2) + epsilon)
            if cost["global_color_nonregression"]:
                worse &= failed_roi_mask
            _add_defects(result, "source_color_error_increased", worse, arrays)
        _add_defects(result,'source_feature_color_error_increased',failed_feature_mask,arrays)
        result["accepted"] = not result["reasons"]
        result["status"] = "accepted" if result["accepted"] else "rejected_source_contradiction"
        return result
    except (ValueError, MemoryError) as exc:
        result["reasons"].append(str(exc) or type(exc).__name__)
        result["accepted"] = False
        result["status"] = "unverified_fail_closed"
        return result


def _render_native(path, width, height):
    payload = Path(path).read_bytes()
    return _render_native_payload(payload,width,height),hashlib.sha256(payload).hexdigest()


def _render_native_payload(payload,width,height):
    from svg_renderer import _validated_svg, native_canvas_aspect_compatible, svg_with_native_viewport
    svg, svg_width, svg_height = _validated_svg(payload)
    if not native_canvas_aspect_compatible(svg_width, svg_height, width, height):
        raise ValueError("source_scene_svg_aspect_ratio_differs")
    svg = svg_with_native_viewport(svg, width, height)
    import resvg_py
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height,
                                background=None, skip_system_fonts=True,
                                log_information=False, shape_rendering="geometric_precision")
    with Image.open(io.BytesIO(png)) as image:
        if image.mode != "RGBA" or image.size != (width, height):
            raise ValueError("source_scene_native_rgba_render_required")
        rgba = np.asarray(image).copy()
    return rgba


def validate_source_scene(before_svg, after_svg, source_png, *,
                          processed_reference_png=None,
                          alpha_thresholds=ALPHA_THRESHOLDS, roi_xyxy=None,
                          source_edge_geometry=None):
    """Render at native source size and return serializable local evidence.

    The 8192-side / 8M-pixel budget is checked before decoding/rendering. Failure
    or unavailable rendering rejects the candidate; there is no resized fallback.
    This API authorizes only source-supported scene changes. The caller still
    applies SVG syntax, geometry, paint and component guards independently.
    """
    result = _base()
    try:
        source = _read_rgba(source_png)
        reference = None if processed_reference_png is None else _read_rgba(processed_reference_png)
        if reference is not None and reference.shape != source.shape:
            raise ValueError("source_scene_dimensions_differ_no_resampling")
        height, width = source.shape[:2]
        before, before_hash = _render_native(before_svg, width, height)
        after, after_hash = _render_native(after_svg, width, height)
        allowance, boundary_evidence = None, None
        if source_edge_geometry is not None:
            from source_boundary_evidence import build_source_boundary_evidence
            from source_edge_reconstruction import rgba_sha256
            allowance, boundary_evidence = build_source_boundary_evidence(
                Path(before_svg).read_bytes().decode("utf8"), Path(after_svg).read_bytes().decode("utf8"),
                source, reference, source_edge_geometry)
            if boundary_evidence.get("verified") is True:
                boundary_evidence["render_binding"] = {"before":rgba_sha256(before),"after":rgba_sha256(after)}
                if roi_xyxy is None:
                    roi_xyxy = boundary_evidence["roi_xyxy"]
        result = validate_source_scene_arrays(before, after, source,
                                              processed_reference_rgba=reference,
                                              alpha_thresholds=alpha_thresholds, roi_xyxy=roi_xyxy,
                                              _boundary_allowance=allowance,_boundary_evidence=boundary_evidence)
        result["provenance"] = {"before_svg_sha256": before_hash, "after_svg_sha256": after_hash,
                                "source_sha256": hashlib.sha256(Path(source_png).read_bytes()).hexdigest(),
                                "processed_reference_sha256": None if processed_reference_png is None else
                                hashlib.sha256(Path(processed_reference_png).read_bytes()).hexdigest(),
                                "renderer": "resvg", "canvas": [width, height]}
    except (OSError, ValueError, ImportError, RuntimeError, MemoryError) as exc:
        result["reasons"].append(str(exc) or type(exc).__name__)
        result["accepted"] = False
        result["status"] = "unverified_fail_closed"
    return result


def validate_source_scene_chain(before_svg, after_svg, source_png, *,
                                processed_reference_png=None, transactions,
                                alpha_thresholds=ALPHA_THRESHOLDS):
    """Replay complete bound transactions, then validate the aggregate scene.

    Each transaction contains before_svg_text, after_svg_text, optional
    source_edge_geometry and roi_xyxy. No saved accepted flag is trusted. A
    paint-only transaction uses the strict scene guard. Only successfully
    revalidated boundary masks are unioned; final topology and global/local
    source costs are checked again. Reports contain hashes, never SVG payloads.
    """
    result=_base()
    summaries=[]
    try:
        from source_boundary_evidence import build_source_boundary_evidence
        from source_edge_reconstruction import rgba_sha256
        if not isinstance(transactions,(list,tuple)) or len(transactions)>64:
            raise ValueError("source_scene_chain_transaction_budget_or_type")
        source=_read_rgba(source_png)
        reference=None if processed_reference_png is None else _read_rgba(processed_reference_png)
        if reference is not None and reference.shape!=source.shape:
            raise ValueError("source_scene_dimensions_differ_no_resampling")
        h,w=source.shape[:2]
        original_text=Path(before_svg).read_bytes().decode("utf8")
        final_text=Path(after_svg).read_bytes().decode("utf8")
        current_text=original_text
        original=_render_native_payload(original_text.encode("utf8"),w,h)
        current=original
        union=np.zeros((h,w),bool)
        required_rois=[]
        required_features={}
        for index,transaction in enumerate(transactions):
            old_text,new_text=transaction["before_svg_text"],transaction["after_svg_text"]
            if not isinstance(old_text,str) or not isinstance(new_text,str) or old_text!=current_text:
                raise ValueError("source_scene_chain_broken_before_binding")
            new=_render_native_payload(new_text.encode("utf8"),w,h)
            geometry=transaction.get("source_edge_geometry")
            allowance,evidence=None,None
            roi=transaction.get("roi_xyxy")
            if geometry is not None:
                allowance,evidence=build_source_boundary_evidence(old_text,new_text,source,reference,geometry)
                if evidence.get("verified") is True:
                    evidence["render_binding"]={"before":rgba_sha256(current),"after":rgba_sha256(new)}
                    if roi is None:
                        roi=evidence["roi_xyxy"]
            checked=validate_source_scene_arrays(current,new,source,processed_reference_rgba=reference,
                                                  alpha_thresholds=alpha_thresholds,roi_xyxy=roi,
                                                  _boundary_allowance=allowance,_boundary_evidence=evidence)
            summary={"index":index,"before_svg_sha256":hashlib.sha256(old_text.encode("utf8")).hexdigest(),
                     "after_svg_sha256":hashlib.sha256(new_text.encode("utf8")).hexdigest(),
                     "accepted":checked["accepted"],"boundary_mode":checked.get("boundary_mode","strict"),
                     "reasons":checked["reasons"],"metrics":checked.get("metrics",{}),
                     "boundary_evidence":evidence}
            summaries.append(summary)
            if not checked["accepted"]:
                checked["transaction_chain"]={"verified":False,"count":len(transactions),
                    "replayed_count":len(summaries),"failed_index":index,"steps":summaries}
                return checked
            if evidence is not None and evidence.get("verified") is True:
                union |= allowance
                if evidence["roi_xyxy"] not in required_rois:
                    required_rois.append(evidence["roi_xyxy"])
                for feature in evidence.get('source_feature_regions',[]):
                    key=repr((feature.get('kind'),feature['bbox_xyxy'],feature['support_mask_packbits_hex']))
                    required_features[key]=feature
            if roi is not None and list(roi) not in required_rois:
                required_rois.append(list(roi))
            current_text,current=new_text,new
        if current_text!=final_text:
            raise ValueError("source_scene_chain_final_binding_mismatch")
        final_evidence=None
        if np.any(union) or required_features:
            final_evidence={"schema":"aivc.source-boundary-chain/v1","verified":True,
                "scope":"replayed_transaction_union_native_free_paper_only",
                "allowance_sha256":hashlib.sha256(union.tobytes()).hexdigest(),
                "allowance_pixels":int(union.sum()),"source_rgba_sha256":rgba_sha256(source),
                "processed_rgba_sha256":rgba_sha256(reference),"required_rois":required_rois,
                "source_feature_regions":list(required_features.values()),
                "render_binding":{"before":rgba_sha256(original),"after":rgba_sha256(current)},
                "contacts_authorized":False,"holes_authorized":False}
        result=validate_source_scene_arrays(original,current,source,processed_reference_rgba=reference,
                                             alpha_thresholds=alpha_thresholds,
                                             _boundary_allowance=union if final_evidence else None,
                                             _boundary_evidence=final_evidence)
        # Even strict-only chains retain every local final baseline.
        if not np.any(union):
            for roi in required_rois:
                local=validate_source_scene_arrays(original,current,source,processed_reference_rgba=reference,
                    alpha_thresholds=alpha_thresholds,roi_xyxy=roi)
                if not local["accepted"]:
                    result=local
                    break
        result["transaction_chain"]={"verified":result["accepted"],"count":len(transactions),
            "replayed_count":len(summaries),"steps":summaries,
            "before_svg_sha256":hashlib.sha256(original_text.encode("utf8")).hexdigest(),
            "after_svg_sha256":hashlib.sha256(final_text.encode("utf8")).hexdigest(),
            "union_allowance_pixels":int(union.sum()),"final_topology_and_cost_rechecked":True}
        result["provenance"]={"source_sha256":hashlib.sha256(Path(source_png).read_bytes()).hexdigest(),
            "processed_reference_sha256":None if processed_reference_png is None else
            hashlib.sha256(Path(processed_reference_png).read_bytes()).hexdigest(),
            "renderer":"resvg","canvas":[w,h]}
    except (OSError,ValueError,TypeError,KeyError,ImportError,RuntimeError,MemoryError) as exc:
        result["reasons"].append(str(exc) or type(exc).__name__)
        result["accepted"]=False
        result["status"]="unverified_fail_closed"
        result["transaction_chain"]={"verified":False,"count":len(transactions) if isinstance(transactions,(list,tuple)) else None,
                                     "replayed_count":len(summaries),"steps":summaries}
    return result
