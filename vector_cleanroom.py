# -*- coding: utf-8 -*-
"""
AI Vector Cleanroom

Batch-process images in the input folder and create editable SVG vector drafts:
  result_<name>/
      <name>_vector.svg              editable vector paths, grouped by color/layer
      <name>_preview.png             preview rendered from the SVG
      source_reference.png           cleaned source reference
      review.html                    overlay review page
      色彩調整.html                  offline role-based recolour page
      <name>_paint_roles.json        portable paint-role manifest
      report.json                    machine-readable run report
      OUTPUT_README.txt              output notes
  result_<name>.zip                  zipped output folder

If two inputs share the same stem (e.g. same.png and same.jpg), the extension
is appended to keep their outputs separate.

This tool does not recover original vector artwork from a bitmap. It creates a
clean, editable vector approximation that must still be reviewed by a human.
"""

from __future__ import annotations

import argparse
import base64
import html
import itertools
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

from app_paths import (
    CODE_DIR,
    DataDirectoryBusyError,
    DataPathError,
    bounded_output_base,
    ensure_output_path_budget,
    resolve_data_dir,
    writer_lock,
)
from execution_control import ConversionInterrupted

# ``BASE`` remains the code/resource directory for compatibility with helper
# modules.  User material must never inherit the source checkout's path depth.
BASE = CODE_DIR
DATA_DIR = resolve_data_dir(code_dir=CODE_DIR)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TOOL_VERSION = "v0.6.0-alpha"
MATERIAL_FALLBACK_GAIN = 1.0
RECONSTRUCTION_KEYS = ("strokes", "gradients", "geometry")
_CANDIDATE_GAIN = {
    "foreground": 1.0,
    "color_fidelity": 1.0,
    "detail_p10": 1.0,
    "detail_mean": 1.0,
    "topology_p10": 2.0,
    "light_object_coverage": 2.0,
}
_CANDIDATE_REGRESSION_BUDGET = {
    "foreground": 0.5,
    "color_fidelity": 3.0,
    # A topology-preserving candidate may move a few low-scoring grid cells
    # while keeping glyphs/arcs whole.  Three points is the measured Ali-tea
    # tradeoff; foreground, colour, mean-detail and topology guards still cap
    # every other regression.
    "detail_p10": 3.0,
    "detail_mean": 1.0,
    "topology_p10": 2.0,
    "light_object_coverage": 2.0,
}

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_inputs(input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTS)


def _apply_validation_hole_mask(image, hole_mask):
    """Clear only stroke-proven negative space from a review/metric image.

    The mask is produced by ``build_clean_base`` after it has identified a
    native circle/rectangle stroke and applied the conservative broad-pocket
    classifier.  Do not infer extra holes from light colour here: real white
    lettering and highlights must remain opaque.  A nearest-neighbour resize
    carries the trace-resolution decision to the native source reference.
    """
    import numpy as np
    from PIL import Image

    result = image.convert("RGBA").copy()
    if hole_mask is None:
        return result
    mask = np.asarray(hole_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return result
    if (mask.shape[1], mask.shape[0]) != result.size:
        resampling = getattr(Image, "Resampling", Image)
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
        mask_image = mask_image.resize(result.size, resampling.NEAREST)
        mask = np.asarray(mask_image, dtype=np.uint8) >= 128
    rgba = np.asarray(result).copy()
    rgba[mask, 3] = 0
    return Image.fromarray(rgba, "RGBA")


def _candidate_metric_vector(item):
    """Return the comparable visual evidence carried by one candidate.

    Foreground coverage alone cannot distinguish a faithful glyph from a
    centre-line reconstruction that occupies roughly the same pixels.  The
    local grid and colour fields already exist in every real candidate; make
    them first-class selection evidence instead of report-only diagnostics.
    """
    scores = item[4] if len(item) > 4 and isinstance(item[4], dict) else {}
    detail = scores.get("detail_grid") or {}
    topology = detail.get("component_topology") or {}
    transparency = scores.get("transparent_light_fidelity") or {}

    def number(value):
        return float(value) if isinstance(value, (int, float)) else None

    topology_p10 = (number(topology.get("p10_score_percent"))
                    if int(topology.get("eligible_components") or 0) >= 8
                    else None)
    light_object_coverage = (
        number(transparency.get("coverage_percent"))
        if bool(transparency.get("applicable"))
        and int(transparency.get("source_pixels") or 0) >= 64
        else None)
    return {
        "foreground": number(item[1]),
        "color_fidelity": number(scores.get("foreground_color_fidelity")),
        "detail_p10": number(detail.get("p10_score_percent")),
        "detail_mean": number(detail.get("mean_score_percent")),
        "topology_p10": topology_p10,
        "light_object_coverage": light_object_coverage,
    }


def _candidate_safely_dominates(candidate, other):
    """True when candidate buys a material gain without hiding a regression."""
    a = _candidate_metric_vector(candidate)
    b = _candidate_metric_vector(other)
    material_gain = False
    comparable = False
    for key, required_gain in _CANDIDATE_GAIN.items():
        av, bv = a.get(key), b.get(key)
        if av is None or bv is None:
            continue
        comparable = True
        delta = av - bv
        if delta < -_CANDIDATE_REGRESSION_BUDGET[key]:
            return False
        if delta >= required_gain:
            material_gain = True
    return comparable and material_gain


def _evaluate_visual_gate(scores):
    """Return an auditable accepted/manual-review/rejected visual verdict.

    A single whole-logo percentage cannot protect small text, thin lines and
    colour.  Nor can a white review canvas expose a white object that became a
    transparent hole.  Rejection therefore also consumes the contrasting-
    background light-object check when applicable.
    """
    scores = scores or {}
    detail = scores.get("detail_grid") or {}
    topology = detail.get("component_topology") or {}
    transparency = scores.get("transparent_light_fidelity") or {}

    def number(value):
        return float(value) if isinstance(value, (int, float)) else None

    local_applicable = int(detail.get("eligible_cells") or 0) >= 8
    topology_applicable = int(topology.get("eligible_components") or 0) >= 8
    transparency_applicable = (
        bool(transparency.get("applicable"))
        and int(transparency.get("source_pixels") or 0) >= 64)
    metrics = {
        "foreground": number(scores.get("foreground")),
        "color_fidelity": number(scores.get("foreground_color_fidelity")),
        "detail_p10": (number(detail.get("p10_score_percent"))
                       if local_applicable else None),
        "detail_mean": (number(detail.get("mean_score_percent"))
                        if local_applicable else None),
        "topology_p10": (number(topology.get("p10_score_percent"))
                         if topology_applicable else None),
        "light_object_coverage": (
            number(transparency.get("coverage_percent"))
            if transparency_applicable else None),
    }
    accept_at = {
        "foreground": 88.0,
        "color_fidelity": 85.0,
        "detail_p10": 80.0,
        "detail_mean": 88.0,
        "topology_p10": 90.0,
        "light_object_coverage": 95.0,
    }
    reject_below = {
        "foreground": 60.0,
        "color_fidelity": 70.0,
        "detail_p10": 55.0,
        "detail_mean": 75.0,
        "topology_p10": 70.0,
        "light_object_coverage": 85.0,
    }
    soft_below = {
        "foreground": 85.0,
        "color_fidelity": 85.0,
        "detail_p10": 75.0,
        "detail_mean": 88.0,
        "topology_p10": 85.0,
        "light_object_coverage": 95.0,
    }
    catastrophic = [
        key for key, threshold in reject_below.items()
        if metrics[key] is not None and metrics[key] < threshold
    ]
    soft = [
        key for key, threshold in soft_below.items()
        if metrics[key] is not None and metrics[key] < threshold
    ]
    required_metrics = ["foreground", "color_fidelity"]
    if local_applicable:
        required_metrics.extend(["detail_p10", "detail_mean"])
    if topology_applicable:
        required_metrics.append("topology_p10")
    if transparency_applicable:
        required_metrics.append("light_object_coverage")
    acceptance_breaches = [
        key for key in required_metrics
        if metrics[key] is None or metrics[key] < accept_at[key]
    ]
    compound_local_failure = bool(
        local_applicable
        and metrics["detail_p10"] is not None
        and metrics["detail_mean"] is not None
        and metrics["detail_p10"] < accept_at["detail_p10"]
        and metrics["detail_mean"] < accept_at["detail_mean"]
    )
    if catastrophic or len(soft) >= 2 or compound_local_failure:
        status = "rejected"
    elif acceptance_breaches:
        status = "manual_review"
    else:
        status = "accepted"

    label = {
        "foreground": "整體前景",
        "color_fidelity": "顏色",
        "detail_p10": "局部低分區",
        "detail_mean": "局部平均",
        "topology_p10": "元件連續性",
        "light_object_coverage": "透明底白／淺色物件",
    }
    reasons = []
    if catastrophic:
        reasons.append("嚴重失守：" + "、".join(label[k] for k in catastrophic))
    if len(soft) >= 2:
        reasons.append("多項失守：" + "、".join(label[k] for k in soft))
    if compound_local_failure and len(soft) < 2:
        reasons.append("局部低分區與局部平均同時未達驗收門檻")
    if status == "manual_review":
        reasons.append("未達自動驗收：" + "、".join(
            label[k] for k in acceptance_breaches))
    return {
        "status": status,
        "metrics": metrics,
        "applicability": {
            "local_detail": local_applicable,
            "component_topology": topology_applicable,
            "transparent_light_objects": transparency_applicable,
            "eligible_cells": int(detail.get("eligible_cells") or 0),
            "eligible_components": int(topology.get("eligible_components") or 0),
            "transparent_light_source_pixels": int(
                transparency.get("source_pixels") or 0),
        },
        "acceptance_thresholds": accept_at,
        "catastrophic_rejection_thresholds": reject_below,
        "multi_metric_rejection_thresholds": soft_below,
        "acceptance_breaches": acceptance_breaches,
        "catastrophic_breaches": catastrophic,
        "soft_breaches": soft,
        "compound_local_failure": compound_local_failure,
        "reasons": reasons,
        "policy": (
            "reject_one_catastrophic_two_independent_soft_or_compound_local_failure"
        ),
    }


def _candidate_editing_structure(item):
    """Measured handles, not enabled feature switches or a human time score."""
    public = item[5] if len(item) > 5 and isinstance(item[5], dict) else {}
    structure = public.get("structure") or {}
    keys = ("nodes", "paths", "native_primitives", "strokes", "gradients")
    if any(not isinstance(structure.get(key), (int, float))
           or not math.isfinite(structure[key]) or structure[key] < 0 for key in keys):
        return None
    return {key: structure[key] for key in keys}


def _candidate_editing_dominates(candidate, other):
    """Prefer simpler equivalent geometry without trading away real handles.

    This conservative partial order is only an edit-burden proxy. It cannot
    certify semantic grouping or designer time saved. Missing evidence never
    earns a simplicity advantage, and a lower node count cannot excuse loss
    of local detail, colour, strokes or gradient resources.
    """
    a, b = _candidate_editing_structure(candidate), _candidate_editing_structure(other)
    if a is None or b is None or a["nodes"] <= 0 or b["nodes"] <= 0:
        return False
    av, bv = _candidate_metric_vector(candidate), _candidate_metric_vector(other)
    if av["foreground"] is None or bv["foreground"] is None:
        return False
    for key in av:
        if (av[key] is None) != (bv[key] is None):
            return False
        if av[key] is not None and abs(av[key] - bv[key]) > 0.25:
            return False
    if any(a[key] < b[key] for key in ("native_primitives", "strokes", "gradients")):
        return False
    a_objects = a["paths"] + a["native_primitives"]
    b_objects = b["paths"] + b["native_primitives"]
    if a_objects > b_objects or a_objects <= 0:
        return False
    return b["nodes"] - a["nodes"] >= max(4, b["nodes"] * 0.20)


def _candidate_structure_concerns(stats, svg_payload):
    """A dense object must not disappear in an average of many simple ones."""
    import xml.etree.ElementTree as ET
    from designer_quality import DEFAULT_THRESHOLDS, _path_segments
    concerns = []
    elements = max(1, stats.n_paths + stats.n_strokes + stats.n_native + stats.n_gradients)
    if stats.n_nodes > 24 * elements:
        concerns.append("high_average_anchor_burden")
    if any(detail.get("closed") and not detail.get("primitive") for detail in stats.stroke_info):
        concerns.append("free_closed_stroke")
    audit = (getattr(stats, "palette_audit", {}) or {}).get("stroke_reconstruction") or {}
    if audit.get("complex_strokes_without_native_cap_proof", 0):
        concerns.append("unverified_native_stroke")
    maximum = 0
    for element in ET.fromstring(svg_payload).iter():
        if element.tag.rsplit("}", 1)[-1] != "path":
            continue
        segments, valid = _path_segments(element.get("d", ""))
        if not valid:
            concerns.append("unparsed_path_geometry")
            break
        maximum = max(maximum, len(segments))
    if maximum > DEFAULT_THRESHOLDS["high_node_path_failure"]:
        concerns.append("high_anchor_individual_path")
    return concerns


def _select_viable_candidate(viable, requested_options,
                             material_gain=MATERIAL_FALLBACK_GAIN):
    """Select by visual gates, safe dominance, then measured editing economy.

    Equivalent candidates can prefer substantially fewer anchors when actual
    native/stroke/gradient handles are retained. Remaining incomparable ties
    preserve requested features. None of these proxies certify human savings.
    """
    if not viable:
        raise ValueError("no viable candidates")
    # Exclude rejected candidates whenever there is a reviewable alternative.
    # An acceptance threshold is not an exchange rate: crossing the colour
    # threshold must not buy arbitrarily damaged thin lines or local detail.
    # Keep reviewable candidates in the comparison until we have inspected
    # their actual trade-offs, rather than discarding them by label alone.
    status_rank = {"rejected": 0, "manual_review": 1, "accepted": 2}
    visual_status = {
        id(item): _evaluate_visual_gate(item[4])["status"] for item in viable
    }
    best_visual_status_rank = max(
        status_rank.get(visual_status[id(item)], 0) for item in viable)
    visual_tier = [
        item for item in viable
        if status_rank.get(visual_status[id(item)], 0)
        >= min(1, best_visual_status_rank)
    ]

    # Remove candidates that are measurably worse on the multi-axis visual
    # evidence.  This is deliberately asymmetric: a tiny foreground gain no
    # longer excuses broken local detail, while a local-detail win may spend
    # only tightly bounded foreground/colour regressions.  The old feature-
    # retention tie-break remains for genuinely equivalent candidates.
    survivors = [
        item for item in viable
        if not any(
            other is not item and _candidate_safely_dominates(other, item)
            for other in visual_tier
        )
    ]
    survivors = [item for item in survivors if item in visual_tier]
    survivors = survivors or list(visual_tier)
    # Prefer accepted output only if it respects every measured regression
    # budget against the other survivors. Otherwise keep the incomparable
    # options and their honest manual-review status. This is especially
    # important for a few thin strokes, where global colour can improve as
    # coverage deteriorates.
    def within_budgets(candidate, other):
        a, b = _candidate_metric_vector(candidate), _candidate_metric_vector(other)
        return all(a[key] is None or b[key] is None
                   or a[key] >= b[key] - budget
                   for key, budget in _CANDIDATE_REGRESSION_BUDGET.items())

    safe_accepted = [item for item in survivors
                     if visual_status[id(item)] == "accepted"
                     and all(within_budgets(item, other) for other in survivors)]
    gate_tradeoffs_retained = bool(
        not safe_accepted and any(visual_status[id(item)] == "accepted"
                                 for item in survivors)
        and any(visual_status[id(item)] == "manual_review" for item in survivors))
    if safe_accepted:
        survivors = safe_accepted
    # Every survivor represents a real Pareto trade-off.  Re-applying the old
    # scalar foreground window here would undo the safety pruning (for
    # example, a 1.2 foreground gain could still hide a 2.5-point local-detail
    # loss).  Feature retention is therefore allowed only among these
    # non-dominated candidates.
    economy_survivors = [item for item in survivors if not any(
        other is not item and _candidate_editing_dominates(other, item)
        for other in survivors)]
    tied = economy_survivors or survivors

    def retention(item):
        options = item[2]
        return sum(
            1 for key in RECONSTRUCTION_KEYS
            if requested_options.get(key) not in (None, "off")
            and options.get(key) == requested_options.get(key)
        )

    selected = max(tied, key=lambda item: (retention(item), item[0], item[1]))
    visual_status_counts = {
        status: sum(1 for item in viable
                    if visual_status[id(item)] == status)
        for status in ("accepted", "manual_review", "rejected")
    }
    return selected, {
        "material_visual_gain_required": material_gain,
        "best_visual_quality": max(item[1] for item in viable),
        "selected_visual_quality": selected[1],
        "selected_requested_features_retained": retention(selected),
        "requested_features_total": sum(
            1 for key in RECONSTRUCTION_KEYS
            if requested_options.get(key) not in (None, "off")),
        "best_visual_status": max(visual_status.values(), key=status_rank.get),
        "selected_visual_status": visual_status[id(selected)],
        "visual_status_counts": visual_status_counts,
        "visual_status_survivor_count": len(visual_tier),
        "incomparable_gate_tradeoffs_retained": gate_tradeoffs_retained,
        "policy": "visual_gate_then_safe_dominance_then_measured_editing_economy",
        "editing_economy": {
            "candidate_count_before": len(survivors),
            "candidate_count_after": len(tied),
            "selected_structure": _candidate_editing_structure(selected),
            "visual_equivalence_max_delta_points": 0.25,
            "minimum_node_reduction_fraction": 0.20,
            "preserves_actual_native_stroke_gradient_counts": True,
            "human_time_saving_validated": False,
        },
        "dominance_budgets": {
            "material_gain": dict(_CANDIDATE_GAIN),
            "maximum_regression": dict(_CANDIDATE_REGRESSION_BUDGET),
        },
        "survivor_count": len(survivors),
        "candidate_count": len(viable),
        "selected_metric_vector": _candidate_metric_vector(selected),
    }


def plan_output_names(paths):
    """Map each input path to a globally unique output base name.

    Candidates are tried in order — stem, stem_ext, stem_ext_2, stem_ext_3 …
    — against a global used-set (case-insensitive, since Windows filesystems
    are). This also survives adversarial sets like
    same.png / same.jpg / same_png.bmp / same_jpg.webp.
    """
    by_stem = {}
    for p in paths:
        by_stem.setdefault(p.stem, []).append(p)

    plan = {}
    used = set()
    for p in paths:
        stem = p.stem
        ext = p.suffix.lstrip(".").lower()
        # stems shared by several inputs always carry the extension
        first = stem if len(by_stem[stem]) == 1 else f"{stem}_{ext}"
        candidates = [bounded_output_base(first),
                      bounded_output_base(f"{stem}_{ext}")]
        base = next((c for c in candidates if c.lower() not in used), None)
        if base is None:
            i = 2
            while bounded_output_base(
                    f"{stem}_{ext}_{i}").lower() in used:
                i += 1
            base = bounded_output_base(f"{stem}_{ext}_{i}")
        used.add(base.lower())
        plan[p] = base
    return plan


def _paint_gradients(png_path: Path, gradient_info):
    """Paint native gradients strictly inside their placeholder fill masks.

    ``svglib``/ReportLab cannot reliably rasterize native gradients.  The
    self-check renderer therefore substitutes one isolated key colour per
    gradient, rasterizes the existing SVG geometry, and evaluates the paint
    model only on pixels owned by that key.  Both the legacy flat two-stop
    linear record and the JSON-safe ``gradient_object_engine`` record are
    accepted; the latter keeps geometry under ``record["model"]`` and may be
    an arbitrary-angle linear or a rotated elliptical radial gradient.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(png_path).convert("RGB")
    arr = np.asarray(im).astype(np.int16)
    h, w, _ = arr.shape
    out = arr.copy()
    parsed_keys = [
        np.array([int(g["key"][1:3], 16), int(g["key"][3:5], 16),
                  int(g["key"][5:7], 16)], dtype=np.int16)
        for g in gradient_info
    ]
    # Assign every rendered pixel to at most one placeholder before growing
    # antialiased fringes.  This prevents sequential gradient passes from
    # repainting one another even when an old file contains poorly-spaced keys.
    owner = np.full((h, w), -1, dtype=np.int16)
    best_distance = np.full((h, w), 32767, dtype=np.int16)
    for gi, key in enumerate(parsed_keys):
        distance = np.abs(arr - key).max(axis=2)
        better = distance < best_distance
        owner[better] = gi
        best_distance[better] = distance[better]

    for gi, (g, key) in enumerate(zip(gradient_info, parsed_keys)):
        viewbox = g.get("viewbox", [w, h])
        vb_w = float(viewbox[0]) if len(viewbox) >= 1 else float(w)
        vb_h = float(viewbox[1]) if len(viewbox) >= 2 else float(h)
        fx = w / vb_w if vb_w else 1.0
        fy = h / vb_h if vb_h else 1.0
        # Thin traced slivers may be entirely antialiased and never contain an
        # exact placeholder pixel.  The allocator keeps real palette colours
        # at least 48 levels away, so a 47-level seed safely recovers those
        # disconnected slivers without mistaking an ordinary solid fill for a
        # gradient key.
        isolation = max(2, int(g.get("key_distance", 48)))
        seed_limit = min(47, isolation - 1)
        mask = (owner == gi) & (best_distance <= seed_limit)
        # Absorb the whole connected antialiased component, not merely three
        # spatial hops.  Long sub-pixel slivers can otherwise retain a purple
        # placeholder tail even though their first pixels were recognised.
        # The key's measured palette isolation caps the flood strictly before
        # any genuine solid colour can join it.
        distance = np.abs(arr - key).max(axis=2)
        near_limit = min(96, isolation - 1)
        near = (owner == gi) & (distance <= near_limit)
        if isolation > 96:
            # With this much measured clearance, every near-key pixel is an
            # internal placeholder contribution.  This also recovers a tiny
            # disconnected sliver whose raster contains no strong seed at all.
            mask = near
        else:
            if not mask.any():
                continue
            from stroke_engine import connected_components
            labels, _ = connected_components(near)
            keep = np.unique(labels[mask])
            keep = keep[keep != 0]
            if len(keep):
                mask = np.isin(labels, keep)
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        # Evaluate in SVG user space, not raster space.  This keeps arbitrary
        # angles and ellipse rotation correct even if the preview dimensions
        # are not an exact integer multiple of the viewBox.
        user_x = xs.astype(np.float64) / fx
        user_y = ys.astype(np.float64) / fy
        model = g.get("model") if isinstance(g.get("model"), dict) else g
        model_type = str(model.get("type", "")).lower()
        svg_type = str(model.get("svg_type", "")).lower()
        if not model_type:
            model_type = "radial" if svg_type == "radialgradient" else "linear"

        try:
            if model_type == "radial" or svg_type == "radialgradient":
                center = model.get("center")
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    cx, cy = float(center[0]), float(center[1])
                else:
                    cx, cy = float(model["cx"]), float(model["cy"])
                radius_x = float(model.get("radius_x", model.get("r", 0.0)))
                radius_y = float(model.get("radius_y", model.get("r", 0.0)))
                if radius_x <= 0.0 or radius_y <= 0.0:
                    continue
                angle = np.deg2rad(float(model.get("rotation_degrees", 0.0)))
                cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
                rel_x, rel_y = user_x - cx, user_y - cy
                rotated_x = cos_a * rel_x + sin_a * rel_y
                rotated_y = -sin_a * rel_x + cos_a * rel_y
                tt = np.sqrt(
                    (rotated_x / radius_x) ** 2
                    + (rotated_y / radius_y) ** 2)
            else:
                x1, y1 = float(model["x1"]), float(model["y1"])
                x2, y2 = float(model["x2"]), float(model["y2"])
                dx, dy = x2 - x1, y2 - y1
                length_squared = dx * dx + dy * dy
                if length_squared <= 0.0:
                    continue
                tt = ((user_x - x1) * dx + (user_y - y1) * dy) / length_squared
        except (KeyError, TypeError, ValueError, OverflowError):
            # Preview painting is fail-closed: malformed evidence leaves the
            # isolated placeholder visible, causing the visual gate to reject
            # rather than inventing a gradient.
            continue
        tt = np.clip(tt, 0.0, 1.0)

        parsed_stops = []
        try:
            for stop in g["stops"]:
                offset = float(stop["offset"])
                colour = stop.get("color")
                if isinstance(colour, str) and len(colour) == 7 and colour[0] == "#":
                    rgb = [int(colour[index:index + 2], 16)
                           for index in (1, 3, 5)]
                else:
                    rgb = [float(value) for value in stop["rgb"]]
                    if len(rgb) != 3:
                        raise ValueError("gradient stop RGB must have three channels")
                parsed_stops.append((offset, rgb))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if len(parsed_stops) < 2:
            continue
        parsed_stops.sort(key=lambda item: item[0])
        offs = np.asarray([item[0] for item in parsed_stops], dtype=np.float64)
        cols = np.asarray([item[1] for item in parsed_stops], dtype=np.float64)
        if (not np.isfinite(offs).all() or not np.isfinite(cols).all()
                or np.any(np.diff(offs) <= 0.0)):
            continue
        offs = np.clip(offs, 0.0, 1.0)
        opacity = float(g.get("opacity", 1.0))
        if opacity < 0.999:
            cols = cols * opacity + 255.0 * (1.0 - opacity)
        for ch in range(3):
            out[ys, xs, ch] = np.interp(tt, offs, cols[:, ch]).astype(np.int16)
    Image.fromarray(out.astype(np.uint8), "RGB").save(png_path)


def _flatten_svg_opacity(text, bg=0xffffff):
    """Preblend simple SVG paint opacity for svglib/reportlab.

    ReportLab versions bundled by the portable app ignore fill-opacity and
    stroke-opacity.  The self-check renders on an opaque background, so an
    equivalent opaque paint is obtained by blending the hex color toward that
    background before svglib sees it.
    """
    import re as _re
    br = (int(bg) >> 16) & 255
    bgc = (br, (int(bg) >> 8) & 255, int(bg) & 255)

    def _tag(match):
        tag = match.group(0)
        overall_m = _re.search(r'(?<!-)\sopacity="([0-9.]+)"', tag)
        overall = float(overall_m.group(1)) if overall_m else 1.0
        changed = False
        for paint in ("fill", "stroke"):
            color_m = _re.search(
                rf'\b{paint}="(#[0-9a-fA-F]{{6}})"', tag)
            op_m = _re.search(rf'\s{paint}-opacity="([0-9.]+)"', tag)
            if not color_m or (not op_m and overall >= 0.999):
                continue
            opacity = overall * (float(op_m.group(1)) if op_m else 1.0)
            opacity = max(0.0, min(1.0, opacity))
            hx = color_m.group(1)
            src = tuple(int(hx[i:i + 2], 16) for i in (1, 3, 5))
            mixed = tuple(round(src[i] * opacity + bgc[i] * (1.0 - opacity))
                          for i in range(3))
            repl = "#{:02x}{:02x}{:02x}".format(*mixed)
            tag = tag[:color_m.start(1)] + repl + tag[color_m.end(1):]
            if op_m:
                tag = _re.sub(rf'\s{paint}-opacity="[0-9.]+"', "", tag, count=1)
            changed = True
        if changed and overall_m:
            tag = _re.sub(r'(?<!-)\sopacity="[0-9.]+"', "", tag, count=1)
        return tag

    return _re.sub(r'<(?:g|path|circle|rect|ellipse|polygon|polyline)\b[^>]*>',
                   _tag, text)


def render_svg_png(svg_path: Path, png_path: Path, size=2000, bg=0xffffff,
                   gradient_info=None):
    """Render actual SVG paint with the offline native resvg renderer.

    gradient_info remains a compatible argument for older stage callers, but
    colors, ownership, transforms and opacity come from the SVG itself.
    A missing renderer never falls back to a misleading flat-color preview.
    """
    try:
        from svg_renderer import render_svg_reference
        background = None if bg is None else (f"#{bg:06x}" if isinstance(bg, int) else bg)
        render_svg_reference(svg_path, png_path, width=int(size), background=background)
        return Path(png_path).is_file()
    except (ImportError, ValueError, OSError, TypeError, RuntimeError):
        return False


def validate_svg_stage_renders(before_svg: Path, after_svg: Path, stage: str,
                               gradient_info=None, *, render_cache=None,
                               render_size=None, tolerance_px=None):
    """Renderer-backed transaction guard for SVG post-processing.

    Exact stages must be pixel-identical.  Annulus regularisation is allowed
    a sub-pixel boundary change only when the independent bidirectional ink
    comparison remains above its 99% gate and transparency topology holds.
    A failed or unavailable renderer cannot approve a geometry mutation.
    """

    import tempfile

    validation_width = max(1, int(render_size or SELF_CHECK_MAX_SIDE))
    # Renderer scratch names used to inherit the SVG/stage path.  A valid SVG
    # in a deep Windows project could therefore cross the legacy MAX_PATH
    # boundary only while being validated and make the renderer appear
    # unavailable.  Render short byte-for-byte SVG copies to bounded PNG names;
    # cache identity still comes from the authoritative source bytes below.
    render_temp = tempfile.TemporaryDirectory(prefix="avc-render-")
    render_root = Path(render_temp.name)
    before_render_svg = render_root / "before.svg"
    after_render_svg = render_root / "after.svg"
    before_png = render_root / "before.png"
    after_png = render_root / "after.png"

    def _cache_key(svg_bytes):
        if render_cache is None:
            return None
        import hashlib
        gradient_payload = json.dumps(
            gradient_info or [], ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), default=str).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"ai-vector-cleanroom-stage-render-v2\0")
        digest.update(str(validation_width).encode("ascii"))
        digest.update(b"\0transparent-native-resvg\0")
        digest.update(gradient_payload)
        digest.update(b"\0")
        # Path.write_text uses CRLF on Windows while the committed candidate
        # is written from UTF-8 bytes with LF.  XML normalises literal line
        # endings before parsing, so canonicalise them in the cache identity;
        # otherwise the same document is needlessly rendered twice.
        canonical_bytes = svg_bytes.replace(b"\r\n", b"\n").replace(
            b"\r", b"\n")
        digest.update(canonical_bytes)
        return digest.hexdigest()

    def _render(source_svg_path, render_svg_path, png_path):
        svg_bytes = Path(source_svg_path).read_bytes()
        key = _cache_key(svg_bytes)
        if key is not None:
            cached = render_cache.get(key)
            if isinstance(cached, bytes) and cached:
                png_path.write_bytes(cached)
                return True, True
        render_svg_path.write_bytes(svg_bytes)
        rendered = render_svg_png(
            render_svg_path, png_path, size=validation_width,
            gradient_info=gradient_info, bg=None)
        if rendered and key is not None and png_path.is_file():
            render_cache[key] = png_path.read_bytes()
        return rendered, False

    try:
        rendered_before, before_cache_hit = _render(
            before_svg, before_render_svg, before_png)
        rendered_after, after_cache_hit = _render(
            after_svg, after_render_svg, after_png)
        if not (rendered_before and rendered_after):
            return {
                "accepted": False,
                "external_render_check": "unavailable",
                "validation_level": "unverified_fail_closed",
                "reason": "SVG render verification unavailable or failed",
                "validation_render_width_px": validation_width,
                "render_cache_hits": {
                    "before": before_cache_hit,
                    "after": after_cache_hit,
                },
            }
        from annulus_detector import compare_rendered_pngs
        exact = stage.endswith("_exact")
        if exact:
            comparison_tolerance = 0
        elif tolerance_px is None:
            comparison_tolerance = 1
        else:
            comparison_tolerance = max(0, int(tolerance_px))
        metrics = compare_rendered_pngs(
            before_png, after_png, tolerance_px=comparison_tolerance)
        if exact:
            import hashlib
            from PIL import Image
            with Image.open(before_png) as image:
                before_size = image.size
                before_pixels = image.convert("RGBA").tobytes()
            with Image.open(after_png) as image:
                after_size = image.size
                after_pixels = image.convert("RGBA").tobytes()
            before_hash = hashlib.sha256(before_pixels).hexdigest()
            after_hash = hashlib.sha256(after_pixels).hexdigest()
            metrics["exact_before_pixel_sha256"] = before_hash
            metrics["exact_after_pixel_sha256"] = after_hash
            metrics["exact_before_render_size_px"] = list(before_size)
            metrics["exact_after_render_size_px"] = list(after_size)
            metrics["exact_pixel_array_equal"] = (
                before_size == after_size and before_hash == after_hash)
            metrics["accepted"] = metrics["exact_pixel_array_equal"]
            metrics["required_equivalence"] = "pixel_array_exact_at_validation_resolution"
        else:
            metrics["required_equivalence"] = "bidirectional_1px_99_percent"
            from alpha_topology import compare_alpha_topology
            alpha_guard = compare_alpha_topology(before_png, after_png)
            metrics["alpha_topology"] = alpha_guard
            metrics["accepted"] = bool(metrics["accepted"] and alpha_guard["accepted"])
            if stage.startswith("curve_refit"):
                from alpha_topology import compare_composed_alpha
                try:
                    composed = compare_composed_alpha(
                        before_png, after_png, check_coverage=False)
                except ValueError as error:
                    composed = {"external_render_check": "completed",
                                "accepted": False, "reason": str(error)}
                metrics["composed_alpha"] = composed
                metrics["accepted"] = bool(metrics["accepted"] and composed["accepted"])
        metrics["validation_background"] = "transparent"
        metrics["external_render_check"] = "completed"
        metrics["validation_level"] = "renderer_and_internal_invariants"
        metrics["validation_render_width_px"] = validation_width
        metrics["render_cache_hits"] = {
            "before": before_cache_hit,
            "after": after_cache_hit,
        }
        return metrics
    finally:
        before_render_svg.unlink(missing_ok=True)
        after_render_svg.unlink(missing_ok=True)
        before_png.unlink(missing_ok=True)
        after_png.unlink(missing_ok=True)
        render_temp.cleanup()


SELF_CHECK_MAX_SIDE = 2048


def _validation_render_width(viewbox, *, max_longest=SELF_CHECK_MAX_SIDE,
                             min_longest=512):
    """Return renderer width for a bounded, aspect-preserving guard image."""

    try:
        if not viewbox or len(viewbox) < 2:
            raise ValueError("missing viewBox")
        source_width = float(viewbox[0])
        source_height = float(viewbox[1])
        if (not math.isfinite(source_width) or not math.isfinite(source_height)
                or source_width <= 0 or source_height <= 0):
            raise ValueError("invalid viewBox")
    except (TypeError, ValueError, OverflowError):
        return max(1, int(max_longest))
    source_longest = max(source_width, source_height)
    validation_longest = max(
        float(min_longest), min(float(max_longest), source_longest))
    return max(
        1, int(round(source_width * validation_longest / source_longest)))


def _match_percent(render_png: Path, reference_png: Path,
                   foreground_only=False, max_side=SELF_CHECK_MAX_SIDE,
                   return_details=False):
    """Compare a rendered candidate with a source reference.

    The foreground score combines ink-mask precision/recall, a one-pixel
    spatial tolerance, and color fidelity.  There is deliberately no broad
    binary RGB pass threshold: a #dddddd line disappearing into white, or a
    semi-transparent mark being flattened/dropped, must score low.
    """
    import numpy as np
    from PIL import Image
    with Image.open(reference_png) as image:
        reference_has_alpha = ("A" in image.getbands()
                               or "transparency" in image.info)
        alpha_origin = image.info.get(
            "avc_reference_alpha_origin", "native_or_unspecified")
        if alpha_origin not in {"native", "opaque_canvas_derived"}:
            alpha_origin = "native_or_unspecified"
        ref = image.convert("RGBA")
    with Image.open(render_png) as image:
        render_has_alpha = ("A" in image.getbands()
                            or "transparency" in image.info)
        ren = image.convert("RGBA")
    # Alpha is part of the design when the prepared reference is transparent.
    # An opaque reference may intentionally have its canvas removed, so that
    # choice remains the responsibility of the background preparation guards.
    compare_alpha = ref.getchannel("A").getextrema()[0] < 255
    tw, th = ren.size
    if ref.size[0] * ref.size[1] < tw * th:
        tw, th = ref.size
    longest = max(tw, th)
    if longest > max_side:
        k = max_side / longest
        tw, th = max(1, int(tw * k)), max(1, int(th * k))
    # Downsampling a one-pixel source line distributes its ink over adjacent
    # rows, while an SVG renderer may put the same total ink into one row.
    # Remember that phase-sensitive case so colour fidelity can compare local
    # ink mass below; coverage still uses the ordinary bidirectional masks.
    reference_was_resized = ref.size != (tw, th)
    if ren.size != (tw, th):
        ren = ren.resize((tw, th))
    if ref.size != (tw, th):
        ref = ref.resize((tw, th))
    def _on_white(image):
        base = Image.new("RGB", (tw, th), (255, 255, 255))
        base.paste(image, (0, 0), image)
        return np.asarray(base, dtype=np.int16)

    a, b = _on_white(ren), _on_white(ref)
    alpha = np.asarray(ref.getchannel("A"), dtype=np.int16)
    render_alpha = np.asarray(ren.getchannel("A"), dtype=np.int16)
    alpha_evidence = {
        "reference_has_alpha_channel": reference_has_alpha,
        "render_has_alpha_channel": render_has_alpha,
        "alpha_comparison_applied": compare_alpha,
        "reference_alpha_origin": alpha_origin,
        "alpha_policy": (
            "opaque_reference_rgb_background_policy" if not compare_alpha
            else "symmetric_support_rgb_ink_and_strict_light_background_alpha"
            if alpha_origin == "opaque_canvas_derived"
            else "symmetric_support_and_strict_alpha_magnitude"),
    }

    border_rgb = np.concatenate([b[0, :], b[-1, :], b[:, 0], b[:, -1]])
    border_alpha = np.concatenate(
        [alpha[0, :], alpha[-1, :], alpha[:, 0], alpha[:, -1]])
    bgc = np.median(border_rgb, axis=0)
    bga = float(np.median(border_alpha))
    border_rgb_noise = np.abs(border_rgb - bgc).max(1)
    border_alpha_noise = np.abs(border_alpha.astype(np.float32) - bga)
    ink_threshold = float(max(6.0, min(24.0,
                          np.percentile(border_rgb_noise, 95) + 3.0)))
    alpha_threshold = float(max(6.0, min(24.0,
                            np.percentile(border_alpha_noise, 95) + 3.0)))
    src_rgb_strength = np.abs(b - bgc).max(2)
    src_alpha_strength = np.abs(alpha.astype(np.float32) - bga)
    # Background removal produces a binary occupancy mask, not a recovered
    # physical opacity field. Opaque-source antialias grey may legitimately
    # become the same white-composited RGB using a translucent dark stroke.
    # Keep alpha support symmetric everywhere; still compare alpha magnitude
    # for light paint/background, and everywhere for an original RGBA source.
    strict_alpha = (src_rgb_strength < ink_threshold
                    if alpha_origin == "opaque_canvas_derived"
                    else np.ones_like(alpha, dtype=bool))
    if not foreground_only:
        # Diagnostic whole-canvas score. Candidate selection never relies on
        # this diluted number when a foreground score exists.
        err = np.abs(a - b).max(2).astype(np.float32)
        if compare_alpha:
            err = np.maximum(err, np.where(
                strict_alpha, np.abs(render_alpha - alpha), 0))
        return float(np.clip(1.0 - err / 128.0, 0.0, 1.0).mean() * 100)
    src_ink = src_rgb_strength >= ink_threshold
    ren_ink = np.abs(a - bgc).max(2) >= ink_threshold
    if compare_alpha:
        # Both masks must use the same source-background alpha baseline.
        # Otherwise an opaque white object on transparency exists only in the
        # source mask, and a faithful SVG loses foreground recall on white.
        src_ink |= src_alpha_strength >= alpha_threshold
        ren_ink |= (np.abs(render_alpha.astype(np.float32) - bga)
                    >= alpha_threshold)
    if not src_ink.any():
        details = {"score": None, "recall": None, "precision": None,
                   "coverage_f1": None, "color_fidelity": None,
                   "source_ink_pixels": 0,
                   "render_ink_pixels": int(ren_ink.sum()),
                   "ink_threshold": ink_threshold, **alpha_evidence}
        return details if return_details else None

    def _shift(arr, dy, dx, fill):
        """Shift without wrapping opposite edges into false neighbours."""
        out = np.full(arr.shape, fill, dtype=arr.dtype)
        sy0, sy1 = max(0, -dy), min(th, th - dy)
        sx0, sx1 = max(0, -dx), min(tw, tw - dx)
        dy0, dy1 = sy0 + dy, sy1 + dy
        dx0, dx1 = sx0 + dx, sx1 + dx
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
        return out

    best_src = np.full((th, tw), 256.0, dtype=np.float32)
    best_ren = np.full((th, tw), 256.0, dtype=np.float32)
    if reference_was_resized:
        # Signed, per-channel deviation from the measured background.  int16
        # is sufficient for a 3x3 sum (9 * 255) and materially cheaper than
        # another pair of full-size float images.
        bgc_i = np.rint(bgc).astype(np.int16)
        src_mass = np.zeros_like(b, dtype=np.int16)
        ren_mass = np.zeros_like(a, dtype=np.int16)
        if compare_alpha:
            src_alpha_mass = np.zeros((th, tw), dtype=np.float32)
            ren_alpha_mass = np.zeros((th, tw), dtype=np.float32)
        src_support = np.zeros((th, tw), dtype=np.uint8)
        ren_support = np.zeros((th, tw), dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted_a = _shift(a, dy, dx, 255)
            shifted_am = _shift(ren_ink, dy, dx, False)
            src_err = np.abs(shifted_a - b).max(2).astype(np.float32)
            if compare_alpha:
                shifted_ra = _shift(render_alpha, dy, dx, bga)
                src_err = np.maximum(src_err, np.where(
                    strict_alpha, np.abs(shifted_ra - alpha), 0))
            best_src = np.minimum(
                best_src, np.where(shifted_am, src_err, 256.0))

            shifted_b = _shift(b, dy, dx, 255)
            shifted_bm = _shift(src_ink, dy, dx, False)
            ren_err = np.abs(a - shifted_b).max(2).astype(np.float32)
            if compare_alpha:
                shifted_sa = _shift(alpha, dy, dx, bga)
                shifted_strict = _shift(strict_alpha, dy, dx, True)
                ren_err = np.maximum(ren_err, np.where(
                    shifted_strict, np.abs(render_alpha - shifted_sa), 0))
            best_ren = np.minimum(
                best_ren, np.where(shifted_bm, ren_err, 256.0))

            if reference_was_resized:
                src_support += shifted_bm
                ren_support += shifted_am
                # Sum only classified ink. Background and out-of-canvas
                # padding therefore contribute exactly zero even when the
                # detected background is not pure white.
                for channel in range(3):
                    src_mass[..., channel] += np.where(
                        shifted_bm,
                        bgc_i[channel] - shifted_b[..., channel], 0)
                    ren_mass[..., channel] += np.where(
                        shifted_am,
                        bgc_i[channel] - shifted_a[..., channel], 0)
                if compare_alpha:
                    src_alpha_mass += np.where(
                        shifted_bm & shifted_strict, shifted_sa - bga, 0)
                    ren_alpha_mass += np.where(
                        shifted_am & shifted_strict, shifted_ra - bga, 0)

    src_cov = best_src < 256.0
    ren_cov = best_ren < 256.0
    recall = float(src_cov[src_ink].mean())
    precision = float(ren_cov[ren_ink].mean()) if ren_ink.any() else 0.0
    coverage_f1 = (2 * recall * precision / (recall + precision)
                   if recall + precision else 0.0)

    # Geometry with a seriously wrong color receives at most the coverage
    # portion; ordinary palette rounding still retains most color credit.
    if reference_was_resized:
        # Compare average signed RGB ink inside the same 3x3 support.  Using
        # the larger support count makes two antialiased source rows and one
        # darker rendered row equivalent when their total ink is equivalent.
        # Channel direction is retained: a red replacement for a black line,
        # or an opaque replacement for a translucent line, still differs.
        support = np.maximum(src_support, ren_support).astype(np.float32)
        support = np.maximum(support, 1.0)
        src_local = src_mass.astype(np.float32) / support[..., None]
        ren_local = ren_mass.astype(np.float32) / support[..., None]
        local_err = np.abs(src_local - ren_local).max(2)
        if compare_alpha:
            local_err = np.maximum(
                local_err, np.abs(src_alpha_mass - ren_alpha_mass) / support)
        src_sim = np.clip(1.0 - local_err / 128.0, 0.0, 1.0)
        ren_sim = src_sim
    else:
        src_sim = np.clip(1.0 - best_src / 128.0, 0.0, 1.0)
        ren_sim = np.clip(1.0 - best_ren / 128.0, 0.0, 1.0)
    recall_q = float(src_sim[src_ink].mean())
    precision_q = float(ren_sim[ren_ink].mean()) if ren_ink.any() else 0.0
    # Acceptance uses source-directed colour fidelity.  A correct one-pixel
    # vector line is commonly rendered as one dark core plus two grey
    # antialias rows; charging those display-only grey pixels as colour errors
    # made a faithful stroke score in the low 80s.  Extra rendered ink is
    # still penalised by bidirectional coverage precision, while missing,
    # wrong-colour, or wrong-opacity source ink loses the full colour term.
    quality_f1 = recall_q
    color_samples = []
    if (src_ink & src_cov).any():
        color_samples.append(float(src_sim[src_ink & src_cov].mean()))
    if (ren_ink & ren_cov).any():
        color_samples.append(float(ren_sim[ren_ink & ren_cov].mean()))
    color_fidelity = (float(sum(color_samples) / len(color_samples))
                      if color_samples else 0.0)
    # A geometrically identical but completely wrong colour tops out at 65%,
    # below automatic acceptance; a correct antialiased 1 px stroke can still
    # reach 100% because its dark core matches the source within one pixel.
    score = 100.0 * (0.65 * coverage_f1 + 0.35 * quality_f1)
    details = {
        "score": score,
        "recall": recall * 100.0,
        "precision": precision * 100.0,
        "coverage_f1": coverage_f1 * 100.0,
        "color_fidelity": color_fidelity * 100.0,
        "source_ink_pixels": int(src_ink.sum()),
        "render_ink_pixels": int(ren_ink.sum()),
        "ink_threshold": ink_threshold,
        **alpha_evidence,
    }
    return details if return_details else score


def _source_has_transparent_light_objects(source_png: Path,
                                          minimum_pixels=64):
    """Cheaply decide whether the contrasting-background render is needed."""
    from PIL import Image
    import numpy as np

    try:
        with Image.open(source_png) as source_image:
            source = np.asarray(source_image.convert("RGBA"), dtype=np.uint8)
        alpha = source[:, :, 3]
        if not bool((alpha < 250).any()):
            return False
        light = (
            (alpha >= 128)
            & (source[:, :, :3].mean(axis=2) >= 210.0)
            & (source[:, :, :3].min(axis=2) >= 185)
        )
        padded = np.pad(light, 1, mode="constant", constant_values=False)
        core = np.ones_like(light, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                core &= padded[dy:dy + light.shape[0],
                               dx:dx + light.shape[1]]
        return int(core.sum()) >= int(minimum_pixels)
    except Exception:
        return False


def _transparent_light_fidelity(render_png: Path, source_png: Path,
                                background=(91, 75, 138)):
    """Measure whether internal white/light objects stay opaque on colour.

    A one-pixel white matte around otherwise coloured artwork is an
    antialiasing boundary, not a white design object.  Light objects are
    therefore measured on a one-pixel-eroded core.  Images without a stable
    light core stay outside this specialised gate; their thin details remain
    covered by the bidirectional foreground, detail-grid and topology gates.
    """
    from PIL import Image
    import numpy as np

    with Image.open(render_png) as rendered_image:
        rendered = np.asarray(rendered_image.convert("RGB"), dtype=np.float32)
    with Image.open(source_png) as source_image:
        source_rgba = source_image.convert("RGBA")
        if source_rgba.size != (rendered.shape[1], rendered.shape[0]):
            source_rgba = source_rgba.resize(
                (rendered.shape[1], rendered.shape[0]), Image.Resampling.LANCZOS)
        source = np.asarray(source_rgba, dtype=np.float32)
    alpha = source[:, :, 3] / 255.0
    has_transparency = bool((alpha < 0.98).any())
    light = (
        (alpha >= 0.5)
        & (source[:, :, :3].mean(axis=2) >= 210.0)
        & (source[:, :, :3].min(axis=2) >= 185.0)
    )
    source_pixels = int(light.sum())
    core = np.ones_like(light, dtype=bool)
    padded = np.pad(light, 1, mode="constant", constant_values=False)
    for dy in range(3):
        for dx in range(3):
            core &= padded[dy:dy + light.shape[0],
                           dx:dx + light.shape[1]]
    core_pixels = int(core.sum())
    measurement = core
    measurement_pixels = core_pixels
    if not has_transparency or core_pixels < 64:
        return {
            "applicable": False,
            "source_pixels": source_pixels,
            "core_pixels": core_pixels,
            "measurement_pixels": 0,
            "measurement_mask": None,
            "spatial_tolerance_px": None,
            "coverage_percent": None,
            "non_background_coverage_percent": None,
            "mean_color_error": None,
            "p90_color_error": None,
            "match_tolerance_rgb": 48,
            "error_metric": "max_channel_rgb",
            "background_rgb": list(background),
            "inapplicable_reason": (
                "source_is_opaque" if not has_transparency
                else "fewer_than_64_stable_light_core_pixels"),
        }
    bg = np.asarray(background, dtype=np.float32)
    expected = (source[:, :, :3] * alpha[:, :, None]
                + bg.reshape(1, 1, 3) * (1.0 - alpha[:, :, None]))
    error = np.abs(rendered - expected).max(axis=2)
    distance_from_background = np.abs(
        rendered - bg.reshape(1, 1, 3)).max(axis=2)
    return {
        "applicable": True,
        "source_pixels": source_pixels,
        "core_pixels": core_pixels,
        "measurement_pixels": measurement_pixels,
        "measurement_mask": "one_pixel_eroded_light_core",
        "spatial_tolerance_px": 0,
        "coverage_percent": round(
            float((error[measurement] <= 48.0).mean() * 100.0), 3),
        "non_background_coverage_percent": round(float(
            (distance_from_background[measurement] > 32.0).mean() * 100.0), 3),
        "mean_color_error": round(float(error[measurement].mean()), 3),
        "p90_color_error": round(
            float(np.percentile(error[measurement], 90)), 3),
        "match_tolerance_rgb": 48,
        "error_metric": "max_channel_rgb",
        "background_rgb": [int(value) for value in background],
        "policy": "eroded_core_expected_light_colour_match",
    }


def self_check(svg_path: Path, flat_png: Path, source_png: Path,
               gradient_info=None, keep_render: Path = None, viewbox=None):
    """Render the SVG back and compare it against the references.

    Returns {"flat": float|None, "source": float|None, "foreground": float|None}.
      flat       — fidelity to the flattened (palette-reduced) tracing input.
      source     — whole-canvas similarity to the cleaned source image.
      foreground — similarity measured on source-ink ROI after adaptive
                   background estimation; catches small foreground details
                   that whole-canvas scores miss.
    Rendering is capped to SELF_CHECK_MAX_SIDE on the longest side.
    """
    out = {"flat": None, "source": None, "foreground": None,
           "foreground_recall": None, "foreground_precision": None,
           "foreground_coverage_f1": None,
           "foreground_color_fidelity": None,
           "source_ink_pixels": None, "render_ink_pixels": None,
           "ink_threshold": None, "detail_grid": None, "hotspots": [],
           "transparent_light_fidelity": {
               "applicable": False, "source_pixels": 0,
               "core_pixels": 0, "measurement_pixels": 0,
               "measurement_mask": None, "spatial_tolerance_px": None,
               "coverage_percent": None, "mean_color_error": None,
               "p90_color_error": None,
               "non_background_coverage_percent": None,
               "match_tolerance_rgb": 48,
               "error_metric": "max_channel_rgb",
               "background_rgb": [91, 75, 138],
               "inapplicable_reason": "not_evaluated",
           }}
    try:
        tmp = svg_path.parent / "_selfcheck.png"
        alpha_tmp = svg_path.parent / "_selfcheck_alpha.png"
        # Render at the comparison resolution when possible.  Rendering every
        # small logo at 2048 px and then shrinking it back blurred an exact
        # one-pixel stroke into grey antialias rows before scoring.
        from PIL import Image
        with Image.open(source_png) as _source:
            sw, sh = _source.size
        scale = min(1.0, SELF_CHECK_MAX_SIDE / max(sw, sh))
        render_width = max(1, int(round(sw * scale)))
        if not render_svg_png(svg_path, alpha_tmp, size=render_width,
                              bg=None, gradient_info=gradient_info):
            return out
        # Keep the native alpha for symmetric source/candidate scoring, then
        # provide the existing diagnostics and repair tools their explicit
        # white composite rather than silently discarding transparent RGB.
        with Image.open(alpha_tmp) as rendered:
            rgba = rendered.convert("RGBA")
            white = Image.new("RGB", rgba.size, (255, 255, 255))
            white.paste(rgba, (0, 0), rgba)
            white.save(tmp)
        try:
            out["flat"] = _match_percent(alpha_tmp, flat_png)
        except Exception:
            pass
        try:
            out["source"] = _match_percent(alpha_tmp, source_png)
            fg = _match_percent(alpha_tmp, source_png, foreground_only=True,
                                return_details=True)
            out["foreground"] = fg["score"]
            out["foreground_recall"] = fg["recall"]
            out["foreground_precision"] = fg["precision"]
            out["foreground_coverage_f1"] = fg["coverage_f1"]
            out["foreground_color_fidelity"] = fg["color_fidelity"]
            out["source_ink_pixels"] = fg["source_ink_pixels"]
            out["render_ink_pixels"] = fg["render_ink_pixels"]
            out["ink_threshold"] = fg["ink_threshold"]
            out["alpha_comparison"] = {
                key: fg[key] for key in (
                    "reference_has_alpha_channel", "render_has_alpha_channel",
                    "alpha_comparison_applied", "reference_alpha_origin",
                    "alpha_policy")}
        except Exception:
            pass
        try:
            from quality_diagnostics import compute_quality_diagnostics
            diag = compute_quality_diagnostics(
                tmp, source_png, viewbox=viewbox, cell=48, max_spots=40)
            # Per-cell data is useful while calculating percentiles but far
            # too bulky to duplicate into every candidate record.
            out["detail_grid"] = {
                key: value for key, value in diag["detail_grid"].items()
                if key != "cells"
            }
            out["hotspots"] = diag["hotspots"]
        except Exception:
            pass
        if _source_has_transparent_light_objects(source_png):
            try:
                contrast_tmp = svg_path.parent / "_selfcheck_contrast.png"
                if render_svg_png(
                        svg_path, contrast_tmp, size=render_width, bg=0x5b4b8a,
                        gradient_info=gradient_info):
                    out["transparent_light_fidelity"] = (
                        _transparent_light_fidelity(contrast_tmp, source_png))
                contrast_tmp.unlink(missing_ok=True)
            except Exception:
                pass
        if keep_render is not None:
            try:
                tmp.replace(keep_render)
            except Exception:
                tmp.unlink(missing_ok=True)
        else:
            tmp.unlink(missing_ok=True)
    except Exception:
        pass
    finally:
        if "alpha_tmp" in locals():
            alpha_tmp.unlink(missing_ok=True)
    return out


def _attempt_isolated_component_repair(
        svg_path: Path, flat_png: Path, source_png: Path, stats,
        before_scores: dict, before_render: Path):
    """Propose, render and transactionally commit safe missing components.

    The live candidate is never touched until a separate proposal SVG has
    passed the same visual gate, per-metric non-regression checks, complete
    failed-component evidence and an exact outside-bbox render guard.
    """

    import hashlib

    from component_repair import (
        append_repair_fragment,
        propose_missing_component_repairs,
        validate_repair_transaction,
    )
    from svg_postprocess import atomic_replace_bytes

    topology = ((before_scores.get("detail_grid") or {}).get(
        "component_topology") or {})
    failed_examples = topology.get("failed_examples")
    base_audit = {
        "schema": "ai-vector-cleanroom.component-repair/v1",
        "status": "not_needed",
        "policy": "safe_proposal_render_validate_atomic_commit",
        "failed_examples_received": (
            len(failed_examples) if isinstance(failed_examples, list) else 0),
        "repair_count": 0,
        "path_count": 0,
        "node_count": 0,
        "repairs": [],
        "proposal": None,
        "transaction": None,
    }
    if not isinstance(failed_examples, list):
        base_audit.update({
            "status": "skipped",
            "reason": "complete_failed_examples_unavailable",
        })
        return before_scores, base_audit
    if not failed_examples:
        return before_scores, base_audit
    missing_like = []
    for example in failed_examples:
        if not isinstance(example, dict):
            continue
        try:
            score = float(example.get("score_percent"))
            coverage = float(example.get("coverage_percent"))
            fragments = int(example.get("fragment_count"))
        except (TypeError, ValueError):
            continue
        if (math.isfinite(score) and math.isfinite(coverage)
                and score <= 5.0 and coverage <= 5.0 and fragments == 0):
            missing_like.append(example)
    base_audit["completely_missing_examples"] = len(missing_like)
    if not missing_like:
        base_audit.update({
            "status": "skipped",
            "reason": "no_completely_missing_components",
        })
        return before_scores, base_audit
    if not before_render.is_file():
        base_audit.update({
            "status": "skipped",
            "reason": "before_render_unavailable",
        })
        return before_scores, base_audit

    original_bytes = svg_path.read_bytes()
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    proposal_path = svg_path.with_name("_component_repair_proposal.svg")
    after_render = svg_path.with_name("_component_repair_after.png")
    try:
        proposal = propose_missing_component_repairs(
            source_png, before_render, flat_png, missing_like,
            viewbox=stats.viewbox)
        base_audit["proposal"] = proposal.get("audit")
        base_audit["before_svg_sha256"] = original_sha
        if proposal.get("status") != "proposed":
            base_audit.update({
                "status": "skipped",
                "reason": (proposal.get("audit") or {}).get(
                    "skipped_reason", "no_safe_components"),
            })
            return before_scores, base_audit

        proposal_bytes = append_repair_fragment(
            original_bytes, proposal["svg_fragment"])
        proposal_sha = hashlib.sha256(proposal_bytes).hexdigest()
        atomic_replace_bytes(proposal_path, proposal_bytes)
        after_scores = self_check(
            proposal_path, flat_png, source_png,
            gradient_info=stats.gradient_info,
            keep_render=after_render, viewbox=stats.viewbox)
        before_gate = _evaluate_visual_gate(before_scores)
        after_gate = _evaluate_visual_gate(after_scores)
        transaction = validate_repair_transaction(
            proposal, before_scores, after_scores, before_gate, after_gate,
            before_render, after_render, source_reference=source_png)
        base_audit["transaction"] = transaction
        public_repairs = [
            {key: value for key, value in repair.items() if key != "path"}
            for repair in proposal.get("repairs", [])
        ]
        base_audit.update({
            "proposal_svg_sha256": proposal_sha,
            "repair_count": int(proposal.get("repair_count") or 0),
            "path_count": int(proposal.get("path_count") or 0),
            "node_count": int(proposal.get("node_count") or 0),
            "repairs": public_repairs,
        })
        if transaction.get("status") != "accepted":
            base_audit.update({
                "status": "rolled_back",
                "reason": "transaction_guard_rejected",
                "after_svg_sha256": original_sha,
                "live_svg_unchanged": svg_path.read_bytes() == original_bytes,
            })
            return before_scores, base_audit

        atomic_replace_bytes(svg_path, proposal_bytes)
        committed_bytes = svg_path.read_bytes()
        committed_sha = hashlib.sha256(committed_bytes).hexdigest()
        if committed_bytes != proposal_bytes:
            raise OSError("atomic component repair commit did not preserve bytes")
        stats.n_paths += int(proposal.get("path_count") or 0)
        stats.n_nodes += int(proposal.get("node_count") or 0)
        stats.geometry_notes.append(
            f"{proposal.get('repair_count', 0)} gap-separated missing component(s) "
            "restored by renderer-validated local trace")
        base_audit.update({
            "status": "committed",
            "reason": None,
            "after_svg_sha256": committed_sha,
            "live_svg_unchanged": False,
        })
        return after_scores, base_audit
    except Exception as exc:
        # The proposal path is separate and the only live write occurs after
        # validation.  If that final atomic replace itself fails, its helper
        # guarantees the previous target remains intact.
        current_bytes = svg_path.read_bytes() if svg_path.is_file() else b""
        base_audit.update({
            "status": "error",
            "reason": "component_repair_exception",
            "error": repr(exc)[:240],
            "after_svg_sha256": (
                hashlib.sha256(current_bytes).hexdigest()
                if current_bytes else None),
            "live_svg_unchanged": current_bytes == original_bytes,
        })
        return before_scores, base_audit
    finally:
        proposal_path.unlink(missing_ok=True)
        after_render.unlink(missing_ok=True)


_GRADIENT_DRAWABLE_TAGS = {
    "path", "circle", "ellipse", "rect", "line", "polyline", "polygon",
}
_GRADIENT_FILL_REFERENCE = re.compile(
    r"^\s*url\(\s*#([^\s)]+)\s*\)\s*$", re.IGNORECASE)
_GRADIENT_GEOMETRY_ATTRIBUTES = {
    "path": ("d",),
    "circle": ("cx", "cy", "r"),
    "ellipse": ("cx", "cy", "rx", "ry"),
    "rect": ("x", "y", "width", "height", "rx", "ry"),
    "line": ("x1", "y1", "x2", "y2"),
    "polyline": ("points",),
    "polygon": ("points",),
}


def _bind_delivered_source_audits(before_bytes, after_bytes, quality, enhancements):
    """Bind final-file reports after proving only our inert metadata changed.

    Preserve the actual render/audit input hashes. This is a narrow delivery
    annotation transition, not permission to reuse evidence for changed art.
    """
    import hashlib
    from xml.etree import ElementTree as ET

    def drawing(payload):
        root = ET.fromstring(payload)
        known = [node for node in root.iter()
                 if node.get('id') == 'ai-vector-cleanroom-metadata']
        if len(known) > 1:
            raise RuntimeError('ambiguous_delivery_metadata')
        for node in known:
            if node not in list(root) or node.tag.rsplit('}', 1)[-1] != 'metadata':
                raise RuntimeError('delivery_metadata_is_not_root_metadata')
            root.remove(node)
        for node in root.iter():
            if node.text is not None and not node.text.strip():
                node.text = None
            if node.tail is not None and not node.tail.strip():
                node.tail = None
        return ET.tostring(root, encoding='utf8')

    before_drawing, after_drawing = drawing(before_bytes), drawing(after_bytes)
    if before_drawing != after_drawing:
        raise RuntimeError('artwork_changed_after_final_source_audit')
    old_sha = hashlib.sha256(before_bytes).hexdigest()
    new_sha = hashlib.sha256(after_bytes).hexdigest()
    binding = {'schema': 'aivc.inert-metadata-delivery-binding/v1',
               'audit_input_svg_sha256': old_sha, 'delivered_svg_sha256': new_sha,
               'drawing_sha256': hashlib.sha256(before_drawing).hexdigest(),
               'scope': 'only_root_ai_vector_cleanroom_metadata_and_xml_indentation_changed'}
    source_info = quality.get('source')
    if isinstance(source_info, dict):
        if source_info.get('sha256') != old_sha:
            raise RuntimeError('designer_audit_does_not_match_pre_metadata_svg')
        source_info.update(audit_input_sha256=old_sha, sha256=new_sha,
                           bytes=len(after_bytes), sha256_scope='delivered_svg_after_inert_metadata')
        quality['delivery_binding'] = dict(binding)
    records = [((enhancements.get('stages') or {}).get('source_topology_audit')),
               ((quality.get('source_topology_gate') or {}).get('evidence'))]
    seen = set()
    for audit in records:
        if not isinstance(audit, dict) or id(audit) in seen:
            continue
        seen.add(id(audit))
        hashes = audit.get('source_hashes')
        if not isinstance(hashes, dict) or not hashes.get('svg'):
            continue
        if hashes['svg'] != old_sha:
            raise RuntimeError('topology_audit_does_not_match_pre_metadata_svg')
        hashes.update(audit_input_svg=old_sha, svg=new_sha)
        audit['delivery_binding'] = dict(binding)
    return binding


def _gradient_geometry_snapshot(svg_path: Path):
    """Describe only geometry that belongs to reconstructed gradient objects.

    The snapshot deliberately excludes paint-role and other inert metadata so
    later editor annotations cannot create a false mismatch.  Conversely, it
    includes every attribute that can move, clip, mask, or change the winding
    of the owned drawable, plus an independently counted final anchor total.
    """
    import hashlib
    from xml.etree import ElementTree as ET

    from clean_base import _parse_subpaths

    root = ET.parse(svg_path).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}

    def inherited(element, attribute, *, include_style=False):
        node = element
        while node is not None:
            value = node.get(attribute)
            if value is not None:
                return value.strip()
            if include_style:
                for item in (node.get("style") or "").split(";"):
                    key, separator, value = item.partition(":")
                    if separator and key.strip().lower() == attribute:
                        return value.strip()
            node = parents.get(node)
        return None

    def path_anchor_count(path_data):
        total = 0
        for subpath in _parse_subpaths(path_data):
            count = len(subpath["segs"]) + 1
            if subpath.get("closed") and subpath["segs"]:
                last = subpath["segs"][-1]
                endpoint = ((last[1], last[2]) if last[0] == "L"
                            else (last[-2], last[-1]))
                if math.hypot(endpoint[0] - subpath["start"][0],
                              endpoint[1] - subpath["start"][1]) <= 1.0e-7:
                    count -= 1
            total += count
        return total

    def point_count(points):
        values = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            points or "")
        return len(values) // 2

    def anchor_count(element, local_name):
        if local_name == "path":
            return path_anchor_count(element.get("d") or "")
        if local_name in {"circle", "ellipse", "rect"}:
            return 1
        if local_name == "line":
            return 2
        if local_name in {"polyline", "polygon"}:
            return point_count(element.get("points"))
        return None

    snapshot = []
    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name not in _GRADIENT_DRAWABLE_TAGS:
            continue
        owner = inherited(element, "data-avc-gradient-object")
        if owner is None:
            continue
        fill = inherited(element, "fill", include_style=True) or ""
        fill_match = _GRADIENT_FILL_REFERENCE.fullmatch(fill)
        geometry = {"element": local_name}
        for attribute in _GRADIENT_GEOMETRY_ATTRIBUTES[local_name]:
            if element.get(attribute) is not None:
                geometry[attribute] = element.get(attribute)
        for attribute in ("transform", "fill-rule", "clip-path", "mask"):
            value = inherited(element, attribute)
            if value is not None:
                geometry[attribute] = value
        geometry_bytes = json.dumps(
            geometry, ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        snapshot.append({
            "gradient_object_id": owner,
            "gradient_id": fill_match.group(1) if fill_match else None,
            "drawable_id": element.get("id"),
            "element": local_name,
            "geometry_sha256": hashlib.sha256(geometry_bytes).hexdigest(),
            "anchor_count": anchor_count(element, local_name),
            "designer_anchor_count": element.get(
                "data-avc-designer-anchors"),
            "error_budget_percent": element.get(
                "data-avc-error-budget-percent"),
            "p95_error_percent": element.get(
                "data-avc-p95-error-percent"),
            "max_error_percent": element.get(
                "data-avc-max-error-percent"),
            "curve_refit": inherited(element, "data-avc-curve-refit"),
        })
    return sorted(snapshot, key=lambda item: (
        str(item.get("gradient_id") or ""),
        str(item.get("gradient_object_id") or ""),
        str(item.get("drawable_id") or ""),
        item["element"],
        item["geometry_sha256"],
    ))


def _gradient_geometry_digest(snapshot):
    import hashlib

    payload = json.dumps(
        snapshot, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _final_gradient_report_details(svg_path: Path, gradient_info,
                                   curve_refit_report):
    """Join source-mask geometry evidence to the unchanged final SVG.

    Geometry error values in ``gradient_info`` were measured when the native
    gradient object was reconstructed from its original ownership mask.  They
    are valid for delivery only while the final drawable is byte-semantically
    the same geometry.  Any mismatch is a release error, never a report-only
    warning, because remeasuring against another SVG cannot recreate that
    source ownership contract.
    """
    import copy
    from xml.etree import ElementTree as ET
    from native_geometry_contract import (
        native_geometry_matches, whole_object_native_primitive)

    details = copy.deepcopy(list(gradient_info or ()))
    svg_elements = {node.get("id"): node for node in ET.parse(svg_path).iter()
                    if node.get("id")}
    final_snapshot = _gradient_geometry_snapshot(svg_path)
    final_digest = _gradient_geometry_digest(final_snapshot)
    curve_guard = ((curve_refit_report or {}).get(
        "gradient_geometry_guard") or {})
    before_digest = curve_guard.get("before_geometry_sha256")

    if details and not before_digest:
        raise RuntimeError(
            "final gradient geometry has no pre-curve-refit ownership snapshot")
    if before_digest and final_digest != before_digest:
        raise RuntimeError(
            "final gradient geometry differs from the ownership-validated "
            "pre-curve-refit SVG")
    if any(item.get("curve_refit") for item in final_snapshot):
        raise RuntimeError(
            "a gradient object was curve-refit without source ownership-mask "
            "revalidation")

    by_gradient_id = {}
    for item in final_snapshot:
        by_gradient_id.setdefault(item.get("gradient_id"), []).append(item)
    expected_ids = [str(detail.get("id") or "") for detail in details]
    if set(by_gradient_id) != set(expected_ids):
        raise RuntimeError(
            "final SVG gradient drawables do not match reconstruction evidence")

    def exact_integer(value, label):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"final gradient drawable is missing {label}") from exc

    def rounded_number_matches(value, expected, label):
        try:
            actual = float(value)
            expected = float(expected)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"final gradient drawable is missing {label}") from exc
        # clean_base serialises these audit attributes to two decimals.
        if abs(actual - expected) > 0.0050001:
            raise RuntimeError(
                f"final gradient {label} contradicts reconstruction evidence")

    final_details = []
    for detail in details:
        gradient_id = str(detail.get("id") or "")
        matches = by_gradient_id.get(gradient_id, [])
        validation = detail.get("validation")
        geometry = (validation.get("geometry")
                    if isinstance(validation, dict) else None)
        if isinstance(geometry, dict) and "source_paint_only" in geometry:
            from gradient_paint_only import final_paint_only_matches
            if not final_paint_only_matches(ET.parse(svg_path).getroot(), geometry,
                                            Path(svg_path).parent / "source_original.png"):
                raise RuntimeError("final gradient paint-only source or geometry certificate is invalid")
            proof = geometry["source_paint_only"]
            records = proof["drawable_records"]
            if (gradient_id != proof["gradient_id"]
                    or sorted(d.get("drawable_id") for d in matches) != sorted(r["id"] for r in records)):
                raise RuntimeError("final gradient paint-only drawable ownership differs")
            by_id = {d["drawable_id"]: d for d in matches}
            matches = [by_id[r["id"]] for r in records]
            for drawable, record in zip(matches, records):
                if (drawable.get("element") != "path"
                        or exact_integer(drawable.get("designer_anchor_count"), "designer anchors") != record["anchors"]):
                    raise RuntimeError("final gradient paint-only designer metadata differs")
                for field, value in (("error_budget_percent", geometry["error_budget"]["requested_max_percent"]),
                                     ("p95_error_percent", 0), ("max_error_percent", 0)):
                    rounded_number_matches(drawable.get(field), value, field)
            final_consistency = {
                "status": "verified_unchanged", "final_element": "path",
                "final_drawable_id": records[0]["id"],
                "final_drawable_ids": [r["id"] for r in records],
                "final_drawable_count": len(records),
                "gradient_object_id": proof["gradient_object_id"],
                "geometry_sha256": final_digest,
                "reconstruction_anchor_count": geometry["anchor_count"],
                "final_anchor_count": geometry["anchor_count"],
                "anchor_count_semantics": "existing_paths_designer_handles_no_geometry_reduction",
                "curve_refit_applied": False, "source_original_rgba_verified": True}
            geometry["final_svg_consistency"] = final_consistency
            geometry["evidence_scope"] = "unchanged_existing_svg_geometry_source_verified_paint_only"
            detail["geometry_evidence"] = {
                "measurement_scope": "exact_existing_geometry_not_original_contour_error",
                **final_consistency}
            final_details.append(detail)
            continue
        if len(matches) != 1:
            raise RuntimeError(
                f"gradient {gradient_id!r} must own exactly one final drawable")
        drawable = matches[0]
        validation = detail.get("validation")
        geometry = (validation.get("geometry")
                    if isinstance(validation, dict) else None)
        if not isinstance(geometry, dict):
            raise RuntimeError(
                f"gradient {gradient_id!r} has no ownership-mask geometry evidence")
        expected_anchor_count = exact_integer(
            geometry.get("anchor_count"), "reconstruction anchor count")
        whole_native = whole_object_native_primitive(geometry)
        native_semantic_normalization = bool(
            drawable["element"] in {"circle", "ellipse", "rect"}
            and whole_native is not None
            and drawable["element"] == whole_native["element"])
        final_native_geometry = None
        if drawable["element"] in {"circle", "ellipse", "rect"}:
            if not native_semantic_normalization:
                raise RuntimeError(
                    f"final native gradient {gradient_id!r} has no whole-object proof")
            element = svg_elements.get(drawable.get("drawable_id"))
            if element is None:
                raise RuntimeError("final native gradient has no unique drawable")
            final_native_geometry = {"element": drawable["element"]}
            for key in ("cx", "cy", "r", "rx", "ry", "x", "y", "width", "height"):
                if element.get(key) is not None:
                    final_native_geometry[key] = element.get(key)
            transform = element.get("transform") or ""
            if transform:
                match = re.fullmatch(
                    r"rotate\(\s*([-+\d.eE]+)[ ,]+([-+\d.eE]+)[ ,]+([-+\d.eE]+)\s*\)",
                    transform)
                if (not match or abs(float(match[2]) - whole_native["cx"]) > 0.000002
                        or abs(float(match[3]) - whole_native["cy"]) > 0.000002):
                    raise RuntimeError("final native gradient transform contradicts proof")
                final_native_geometry["rotation_degrees"] = match[1]
            if not native_geometry_matches(whole_native, final_native_geometry):
                raise RuntimeError("final native gradient geometry contradicts whole-object proof")
        final_anchor_count = drawable.get("anchor_count")
        if (native_semantic_normalization and final_anchor_count != 1):
            raise RuntimeError(
                f"final native gradient {gradient_id!r} is not one SVG element")
        if (not native_semantic_normalization
                and final_anchor_count != expected_anchor_count):
            raise RuntimeError(
                f"final gradient {gradient_id!r} anchor count contradicts "
                "ownership-mask geometry evidence")
        expected_designer_anchors = exact_integer(
            geometry.get("designer_anchor_count"),
            "reconstruction designer-anchor count")
        final_designer_anchors = exact_integer(
            drawable.get("designer_anchor_count"),
            "designer-anchor metadata")
        allowed_designer_anchors = {expected_designer_anchors}
        if native_semantic_normalization:
            # clean_base exposes native circle/ellipse handles as four
            # designer anchors even though the SVG DOM contains one element.
            allowed_designer_anchors.add(8 if drawable["element"] == "rect" else 4)
        if final_designer_anchors not in allowed_designer_anchors:
            raise RuntimeError(
                f"final gradient {gradient_id!r} designer anchors contradict "
                "ownership-mask geometry evidence")
        error_budget = geometry.get("error_budget") or {}
        rounded_number_matches(
            drawable.get("error_budget_percent"),
            error_budget.get("requested_max_percent"), "error budget")
        rounded_number_matches(
            drawable.get("p95_error_percent"),
            error_budget.get("actual_p95_error_percent"), "p95 error")
        rounded_number_matches(
            drawable.get("max_error_percent"),
            error_budget.get("actual_max_error_percent"), "maximum error")
        final_consistency = {
            "status": "verified_unchanged",
            "final_element": drawable["element"],
            "final_drawable_id": drawable.get("drawable_id"),
            "gradient_object_id": drawable.get("gradient_object_id"),
            "geometry_sha256": drawable["geometry_sha256"],
            "reconstruction_anchor_count": expected_anchor_count,
            "final_anchor_count": final_anchor_count,
            "anchor_count_semantics": (
                "native_svg_element_vs_designer_handles"
                if native_semantic_normalization else "exact_path_anchors"
            ),
            "curve_refit_applied": False,
        }
        if final_native_geometry is not None:
            final_consistency["native_geometry"] = final_native_geometry
        source_circle = geometry.get("source_reconstruction")
        if source_circle is not None:
            from source_gradient_primitive import (
                gradient_paint_sha256, final_source_primitive_matches)
            source_snapshot = Path(svg_path).parent / "source_original.png"
            if not source_snapshot.is_file():
                raise RuntimeError("final source-primitive certificate has no unmodified source snapshot")
            if (not final_source_primitive_matches(
                    ET.parse(svg_path).getroot(), geometry,
                    source_snapshot)
                    or source_circle.get("drawable_id") != drawable.get("drawable_id")
                    or source_circle.get("gradient_object_id") != drawable.get("gradient_object_id")
                    or source_circle.get("gradient_id") != gradient_id
                    or source_circle.get("paint_sha256") != gradient_paint_sha256(
                        ET.parse(svg_path).getroot(), gradient_id)):
                raise RuntimeError("final gradient source-primitive certificate is invalid or paint changed")
            final_consistency["source_original_rgba_verified"] = True
        source_edge = geometry.get("source_edge_reconstruction")
        if source_edge is not None:
            from source_edge_reconstruction import final_source_edge_matches
            from PIL import Image
            import numpy as np
            source_file = Path(svg_path).parent / "source_original.png"
            reference_file = Path(svg_path).parent / "source_reference.png"
            if not final_source_edge_matches(
                    ET.parse(svg_path).getroot(), geometry,
                    np.asarray(Image.open(source_file).convert("RGBA")),
                    np.asarray(Image.open(reference_file).convert("RGBA"))):
                raise RuntimeError("final source-edge reconstruction certificate is invalid")
            final_consistency["source_original_rgba_verified"] = True
            final_consistency["source_edge_reconstruction_verified"] = True
        ownership_completion = geometry.get("source_ownership_completion")
        if ownership_completion is not None:
            from gradient_source_components import final_source_component_matches
            if not final_source_component_matches(
                    ET.parse(svg_path).getroot(), geometry,
                    Path(svg_path).parent / "source_original.png"):
                raise RuntimeError("final source component ownership certificate is invalid")
            final_consistency["source_original_rgba_verified"] = True
            final_consistency["source_ownership_completion_verified"] = True
        if geometry.get("source_contour_spans") is not None:
            from gradient_contour_spans import final_source_contour_spans_matches
            if not final_source_contour_spans_matches(
                    ET.parse(svg_path).getroot(), geometry,
                    Path(svg_path).parent / "source_original.png"):
                raise RuntimeError("final gradient contour-span certificate is invalid")
            final_consistency["source_contour_spans_verified"] = True
        measurement_scope = (
            "unmodified_input_coverage50_" + source_circle["primitive_kind"] + "_with_same_native_gradient_paint"
            if source_circle is not None else
            "original_source_ownership_mask_at_gradient_reconstruction")
        geometry["evidence_scope"] = (
            "gradient_reconstruction_against_original_source_coverage50_" + source_circle["primitive_kind"]
            if source_circle is not None else
            "gradient_reconstruction_against_original_source_ownership_mask")
        if ownership_completion is not None:
            measurement_scope = "original_source_supported_enclosed_component_ownership_reconstruction"
            geometry["evidence_scope"] = "gradient_reconstruction_against_original_source_supported_completed_ownership"
        if source_edge is not None:
            measurement_scope = "source_coverage_and_guarded_ownership_interface_reconstruction"
            geometry["evidence_scope"] = "gradient_reconstruction_against_original_source_coverage_and_guarded_ownership_interfaces"
        geometry["final_svg_consistency"] = final_consistency
        detail["geometry_evidence"] = {
            "measurement_scope": measurement_scope,
            "final_svg_scope": (
                "verified_same_geometry_not_post_curve_refit_remeasurement"
            ),
            "global_curve_refit_policy": (
                "fail_closed_skip_without_source_ownership_mask"
            ),
            **final_consistency,
        }
        final_details.append(detail)

    summary = {
        "schema": "ai-vector-cleanroom.gradient-geometry-consistency/v1",
        "status": "verified_unchanged" if details else "not_applicable",
        "gradient_count": len(details),
        "gradient_drawable_count": len(final_snapshot),
        "geometry_sha256": final_digest,
        "measurement_scope": (
            "original_source_ownership_mask_at_gradient_reconstruction"
        ),
        "final_svg_scope": (
            "verified_same_geometry_not_post_curve_refit_remeasurement"
        ),
        "global_curve_refit_policy": (
            "fail_closed_skip_without_source_ownership_mask"
        ),
    }
    return final_details, summary


def _verify_designer_gradient_evidence_join(final_gradient_details,
                                            designer_quality):
    """Fail closed if the designer gate audited stale gradient evidence."""
    details = list(final_gradient_details or ())
    if not details:
        return
    gate = ((designer_quality or {}).get("gradient_object_gate") or {})
    source = gate.get("source_space_field_evidence") or {}
    objects = source.get("objects")
    if not isinstance(objects, list):
        raise RuntimeError(
            "designer audit has no final gradient evidence objects")
    by_candidate = {}
    for item in objects:
        if not isinstance(item, dict) or not item.get("candidate_id"):
            continue
        candidate_id = str(item["candidate_id"])
        if candidate_id in by_candidate:
            raise RuntimeError(
                "designer audit has duplicate gradient candidate evidence")
        by_candidate[candidate_id] = item
    if len(by_candidate) != len(details):
        raise RuntimeError(
            "designer audit did not consume every final gradient detail")

    for detail in details:
        candidate_id = str(detail.get("candidate_id") or "")
        actual = by_candidate.get(candidate_id)
        validation = detail.get("validation") or {}
        geometry = validation.get("geometry") or {}
        final = geometry.get("final_svg_consistency") or {}
        expected = {
            "gradient_id": detail.get("id"),
            "final_drawable_id": final.get("final_drawable_id"),
            "gradient_object_id": final.get("gradient_object_id"),
            "final_element": final.get("final_element"),
            "anchor_count": geometry.get("anchor_count"),
            "designer_anchor_count": geometry.get("designer_anchor_count"),
            "segment_count": geometry.get("segment_count"),
        }
        if (actual is None
                or final.get("status") != "verified_unchanged"
                or any(actual.get(key) != value
                       for key, value in expected.items())):
            raise RuntimeError(
                "designer audit did not consume finalized gradient geometry "
                f"for {candidate_id or '<missing-candidate>'}")
        from gradient_paint_only import paint_only_certificate_valid
        economy_expected = not paint_only_certificate_valid(geometry) and any(
            key in geometry for key in (
                "solver", "selection_evidence", "lexicographic_objective",
                "final_svg_consistency"))
        if bool(actual.get("economy_certificate_available")) \
                != economy_expected:
            raise RuntimeError(
                "designer audit gradient economy scope contradicts final "
                f"evidence for {candidate_id}")


def _curve_refit_hybrid_bytes(original_bytes, candidate_bytes, rollback_ids,
                              detail_by_id):
    """Restore selected source elements inside an otherwise refitted SVG.

    IDs are the transaction join key.  A subset candidate is refused unless
    every requested ID occurs exactly once in both documents and the full
    candidate element still matches the geometry digest emitted by the fitter.
    This fail-closed join prevents a partial rollback from restoring the wrong
    drawable after a primitive conversion or future scene-graph change.
    """
    import copy
    import io
    import xml.etree.ElementTree as ET

    from curve_refit_stage import _committed_geometry_evidence

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace(
        "inkscape", "http://www.inkscape.org/namespaces/inkscape")
    original_root = ET.fromstring(original_bytes)
    candidate_root = ET.fromstring(candidate_bytes)

    def index(root):
        parents = {child: parent for parent in root.iter() for child in parent}
        by_id = {}
        for element in root.iter():
            identifier = element.get("id")
            if identifier:
                by_id.setdefault(identifier, []).append(element)
        return parents, by_id

    _original_parents, original_by_id = index(original_root)
    candidate_parents, candidate_by_id = index(candidate_root)
    for identifier in rollback_ids:
        source_matches = original_by_id.get(identifier, [])
        candidate_matches = candidate_by_id.get(identifier, [])
        if len(source_matches) != 1 or len(candidate_matches) != 1:
            raise RuntimeError(
                f"curve-refit subset ID is missing or ambiguous: {identifier}")
        detail = detail_by_id.get(identifier)
        if not isinstance(detail, dict):
            raise RuntimeError(
                f"curve-refit subset detail is missing: {identifier}")
        candidate_element = candidate_matches[0]
        actual = _committed_geometry_evidence(candidate_element)
        digest_fields = (
            ("final_element", "final_drawable_id", "path_data_sha256",
             "path_data_digest_scope")
            if actual.get("final_element") == "path" else
            ("final_element", "final_drawable_id", "native_parameters",
             "geometry_sha256", "geometry_digest_scope")
        )
        if any(actual.get(key) != detail.get(key) for key in digest_fields):
            raise RuntimeError(
                f"curve-refit subset candidate digest mismatch: {identifier}")
        parent = candidate_parents.get(candidate_element)
        if parent is None:
            raise RuntimeError(
                f"curve-refit subset candidate has no parent: {identifier}")
        child_index = list(parent).index(candidate_element)
        restored = copy.deepcopy(source_matches[0])
        # Formatting whitespace belongs to the candidate tree, not geometry.
        restored.tail = candidate_element.tail
        parent[child_index] = restored

    output = io.BytesIO()
    ET.ElementTree(candidate_root).write(
        output, encoding="utf-8", xml_declaration=True)
    return output.getvalue()


def _curve_refit_final_digest_records(svg_bytes, details):
    """Authenticate final committed detail IDs against exact SVG bytes."""
    import xml.etree.ElementTree as ET

    from curve_refit_stage import _committed_geometry_evidence

    root = ET.fromstring(svg_bytes)
    by_id = {}
    for element in root.iter():
        identifier = element.get("id")
        if identifier:
            by_id.setdefault(identifier, []).append(element)
    records = []
    for detail in details:
        identifier = str(detail.get("id") or "")
        matches = by_id.get(identifier, [])
        if not identifier or len(matches) != 1:
            raise RuntimeError(
                f"final curve-refit ID is missing or ambiguous: {identifier}")
        actual = _committed_geometry_evidence(matches[0])
        digest_fields = (
            ("final_element", "final_drawable_id", "path_data_sha256",
             "path_data_digest_scope")
            if actual.get("final_element") == "path" else
            ("final_element", "final_drawable_id", "native_parameters",
             "geometry_sha256", "geometry_digest_scope")
        )
        if any(actual.get(key) != detail.get(key) for key in digest_fields):
            raise RuntimeError(
                f"final curve-refit evidence digest mismatch: {identifier}")
        records.append({"id": identifier, **actual})
    return records


def _curve_refit_identity_normalized_bytes(original_bytes, proposal):
    """Authenticate stage ID evidence and rebuild a frozen-source baseline.

    The proposal SVG is never trusted to define source identity.  Every
    optimizer-evaluated path is re-located in the exact frozen source by its
    global 1-based path ordinal, full UTF-8 ``d`` digest and original ID state.
    Only then are deterministic missing IDs assigned.  Existing IDs are never
    rewritten, and duplicate IDs in either source or normalized output fail
    closed.
    """
    import hashlib
    import io
    import xml.etree.ElementTree as ET

    if (not isinstance(proposal, dict)
            or proposal.get("schema") !=
            "ai-vector-cleanroom.curve-refit-proposal/v3"):
        raise RuntimeError(
            "curve-refit identity normalization requires proposal/v3 evidence")
    evidence = proposal.get("stable_id_normalization")
    if (not isinstance(evidence, dict)
            or evidence.get("schema") !=
            "ai-vector-cleanroom.curve-refit-stable-id-normalization/v1"):
        raise RuntimeError("curve-refit stable-ID normalization evidence missing")
    original_bytes = bytes(original_bytes)
    source_sha = hashlib.sha256(original_bytes).hexdigest()
    if (evidence.get("source_svg_sha256") != source_sha
            or evidence.get("source_svg_digest_scope") !=
            "exact_source_svg_bytes"):
        raise RuntimeError("curve-refit stable-ID source bytes changed")

    root = ET.fromstring(original_bytes)
    paths = [item for item in root.iter()
             if item.tag.rsplit("}", 1)[-1] == "path"]
    if evidence.get("source_path_count") != len(paths):
        raise RuntimeError("curve-refit stable-ID source path count mismatch")
    existing_ids = [item.get("id") for item in root.iter() if item.get("id")]
    if len(existing_ids) != len(set(existing_ids)):
        raise RuntimeError("curve-refit source contains duplicate existing IDs")

    records = evidence.get("records")
    optimizer_count = proposal.get("optimizer_evaluated_path_count")
    if (not isinstance(records, list)
            or isinstance(optimizer_count, bool)
            or not isinstance(optimizer_count, int)
            or evidence.get("optimizer_evaluated_path_count") != optimizer_count
            or evidence.get("record_count") != len(records)
            or len(records) != optimizer_count
            or evidence.get("all_optimizer_evaluations_authenticated") is not True
            or evidence.get("assigned_ids_unique") is not True
            or evidence.get("candidate_all_svg_ids_unique") is not True):
        raise RuntimeError("curve-refit stable-ID evidence accounting mismatch")

    evidence_rows = []
    seen_ordinals = set()
    assigned_ids = set()
    deterministic_used_identifiers = set(existing_ids)
    previous_ordinal = 0
    assignment_count = 0
    for raw in records:
        if not isinstance(raw, dict):
            raise RuntimeError("curve-refit stable-ID record is not a mapping")
        ordinal = raw.get("global_path_ordinal_1_based")
        assigned_id = raw.get("assigned_id")
        if (isinstance(ordinal, bool) or not isinstance(ordinal, int)
                or not 1 <= ordinal <= len(paths)
                or ordinal <= previous_ordinal
                or ordinal in seen_ordinals
                or not isinstance(assigned_id, str) or not assigned_id
                or assigned_id in assigned_ids
                or raw.get("source_svg_sha256") != source_sha
                or raw.get("source_element") != "path"
                or raw.get("source_path_data_digest_scope") !=
                "utf8_svg_path_d_attribute"):
            raise RuntimeError("curve-refit stable-ID locator is invalid")
        seen_ordinals.add(ordinal)
        previous_ordinal = ordinal
        assigned_ids.add(assigned_id)
        element = paths[ordinal - 1]
        source_path_data = element.get("d") or ""
        source_path_sha = hashlib.sha256(
            source_path_data.encode("utf-8")).hexdigest()
        if raw.get("source_path_data_sha256") != source_path_sha:
            raise RuntimeError("curve-refit stable-ID source path digest mismatch")
        original_id = element.get("id")
        state = raw.get("original_id_state")
        assignment_applied = raw.get("assignment_applied")
        if state == "existing_preserved":
            if (not original_id or raw.get("original_id") != original_id
                    or assigned_id != original_id
                    or assignment_applied is not False):
                raise RuntimeError(
                    "curve-refit existing stable ID was not preserved")
        elif state == "missing_assigned":
            if (original_id or raw.get("original_id") is not None
                    or assignment_applied is not True
                    or assigned_id in existing_ids):
                raise RuntimeError(
                    "curve-refit generated stable ID evidence is invalid")
            base = f"avc-refit-path-{ordinal}-{source_path_sha[:12]}"
            expected_identifier = base
            suffix = 2
            while expected_identifier in deterministic_used_identifiers:
                expected_identifier = f"{base}-{suffix}"
                suffix += 1
            if assigned_id != expected_identifier:
                raise RuntimeError(
                    "curve-refit generated stable ID is not deterministic")
            element.set("id", assigned_id)
            assignment_count += 1
        else:
            raise RuntimeError("curve-refit original ID state is invalid")
        deterministic_used_identifiers.add(assigned_id)
        evidence_rows.append(dict(raw))

    accounted_rows = []
    for key in ("details", "evaluated_but_retained",
                "uncertified_evaluations"):
        rows = proposal.get(key)
        if not isinstance(rows, list):
            raise RuntimeError("curve-refit proposal evidence lists are incomplete")
        accounted_rows.extend(rows)
    accounted_ids = [
        str(item.get("id") or "") for item in accounted_rows
        if isinstance(item, dict)
    ]
    if (len(accounted_ids) != len(accounted_rows) or not all(accounted_ids)
            or len(accounted_ids) != len(set(accounted_ids))
            or set(accounted_ids) != assigned_ids):
        raise RuntimeError(
            "curve-refit stable-ID records do not match evaluated evidence")
    if evidence.get("assigned_id_count") != assignment_count:
        raise RuntimeError("curve-refit assigned stable-ID count mismatch")
    if evidence.get("existing_id_preserved_count") != (
            len(records) - assignment_count):
        raise RuntimeError("curve-refit preserved stable-ID count mismatch")

    normalized_ids = [item.get("id") for item in root.iter() if item.get("id")]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise RuntimeError("curve-refit normalized baseline IDs are ambiguous")
    if assignment_count:
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        ET.register_namespace(
            "inkscape", "http://www.inkscape.org/namespaces/inkscape")
        output = io.BytesIO()
        ET.ElementTree(root).write(
            output, encoding="utf-8", xml_declaration=True)
        normalized_bytes = output.getvalue()
    else:
        # Preserve exact bytes when normalization has nothing to assign.
        normalized_bytes = original_bytes
    normalized_sha = hashlib.sha256(normalized_bytes).hexdigest()
    return normalized_bytes, {
        "schema": (
            "ai-vector-cleanroom.curve-refit-stable-id-transaction/v1"),
        "status": "authenticated",
        "policy": (
            "frozen_source_bytes_plus_global_path_ordinal_plus_full_d_digest_"
            "plus_original_id_state"),
        "source_svg_sha256": source_sha,
        "normalized_baseline_svg_sha256": normalized_sha,
        "source_path_count": len(paths),
        "optimizer_evaluated_path_count": optimizer_count,
        "record_count": len(evidence_rows),
        "assigned_id_count": assignment_count,
        "existing_id_preserved_count": len(records) - assignment_count,
        "assigned_ids_unique": True,
        "source_existing_ids_unique": True,
        "normalized_all_svg_ids_unique": True,
        "records": evidence_rows,
    }


def _curve_refit_expected_final_digest_details(proposal):
    """Build exact ID-plus-geometry expectations for every evaluated path."""
    if not isinstance(proposal, dict):
        raise RuntimeError("curve-refit final digest proposal is invalid")
    normalization = proposal.get("stable_id_normalization")
    records = normalization.get("records") if isinstance(
        normalization, dict) else None
    if not isinstance(records, list):
        raise RuntimeError("curve-refit final digest stable-ID records missing")
    record_by_id = {
        str(item.get("assigned_id") or ""): item
        for item in records if isinstance(item, dict)
    }
    if (len(record_by_id) != len(records) or "" in record_by_id):
        raise RuntimeError("curve-refit final digest stable IDs are invalid")
    expected = []
    seen = set()
    for raw in proposal.get("details") or []:
        if not isinstance(raw, dict):
            raise RuntimeError("curve-refit committed detail is invalid")
        identifier = str(raw.get("id") or "")
        if not identifier or identifier in seen or identifier not in record_by_id:
            raise RuntimeError("curve-refit committed final ID is invalid")
        seen.add(identifier)
        expected.append(raw)
    for key in ("evaluated_but_retained", "uncertified_evaluations"):
        rows = proposal.get(key)
        if not isinstance(rows, list):
            raise RuntimeError("curve-refit noncommitted evidence is incomplete")
        for raw in rows:
            if not isinstance(raw, dict):
                raise RuntimeError("curve-refit noncommitted detail is invalid")
            identifier = str(raw.get("id") or "")
            record = record_by_id.get(identifier)
            if not identifier or identifier in seen or not isinstance(record, dict):
                raise RuntimeError("curve-refit noncommitted final ID is invalid")
            seen.add(identifier)
            expected.append({
                "id": identifier,
                "final_element": "path",
                "final_drawable_id": identifier,
                "path_data_sha256": record.get("source_path_data_sha256"),
                "path_data_digest_scope": "utf8_svg_path_d_attribute",
            })
    if seen != set(record_by_id):
        raise RuntimeError("curve-refit final digest accounting is incomplete")
    return expected


def _curve_refit_candidate_identity_guard(candidate_bytes, proposal):
    """Reject candidate ID or geometry tampering before renderer validation."""
    import hashlib
    import xml.etree.ElementTree as ET

    candidate_bytes = bytes(candidate_bytes)
    root = ET.fromstring(candidate_bytes)
    identifiers = [item.get("id") for item in root.iter() if item.get("id")]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("curve-refit candidate contains duplicate SVG IDs")
    normalization = proposal.get("stable_id_normalization")
    if not isinstance(normalization, dict):
        raise RuntimeError("curve-refit candidate stable-ID evidence missing")
    if normalization.get("candidate_svg_sha256") != hashlib.sha256(
            candidate_bytes).hexdigest():
        raise RuntimeError("curve-refit candidate bytes digest mismatch")
    expected = _curve_refit_expected_final_digest_details(proposal)
    records = _curve_refit_final_digest_records(candidate_bytes, expected)
    return {
        "status": "verified",
        "candidate_svg_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_all_svg_ids_unique": True,
        "evaluated_path_count": len(expected),
        "final_digest_records": records,
    }


def _reconcile_curve_refit_subset_proposal(
        proposal, rollback_ids, final_svg_bytes, final_candidate_name,
        subset_evidence):
    """Make proposal counts and certificates describe the exact final SVG."""
    import copy

    if (not isinstance(proposal, dict)
            or proposal.get("schema") !=
            "ai-vector-cleanroom.curve-refit-proposal/v3"):
        raise RuntimeError("curve-refit subset requires proposal/v3 evidence")
    details = proposal.get("details")
    retained = proposal.get("evaluated_but_retained")
    uncertified = proposal.get("uncertified_evaluations")
    if (not isinstance(details, list) or not isinstance(retained, list)
            or not isinstance(uncertified, list)):
        raise RuntimeError("curve-refit subset evidence lists are incomplete")
    detail_ids = [str(item.get("id") or "") for item in details
                  if isinstance(item, dict)]
    if (len(detail_ids) != len(details) or not all(detail_ids)
            or len(detail_ids) != len(set(detail_ids))):
        raise RuntimeError("curve-refit subset detail IDs are invalid")
    rollback_ids = list(rollback_ids)
    rollback_set = set(rollback_ids)
    if (len(rollback_ids) != len(rollback_set)
            or not rollback_set.issubset(detail_ids)):
        raise RuntimeError("curve-refit rollback IDs are invalid")

    updated = copy.deepcopy(proposal)
    committed = [item for item in updated["details"]
                 if str(item.get("id")) not in rollback_set]
    rolled_details = {str(item.get("id")): item
                      for item in updated["details"]
                      if str(item.get("id")) in rollback_set}
    rollback_records = []
    for identifier in rollback_ids:
        detail = rolled_details[identifier]
        before = int(detail["anchors_before"])
        designer_before = int(detail.get("designer_anchors_before", before))
        rollback_records.append({
            "id": identifier,
            "outcome": "transaction_guard_rollback_to_source_identity",
            "economy_certified": False,
            "stage_reason": "renderer_or_source_guard_rejected_candidate",
            "integrity_failures": [
                "geometry_candidate_not_committed_after_transaction_guard"],
            "rollback_basis": "exact_source_svg_element_restoration",
            "anchors_before": before,
            "anchors_after": before,
            "designer_anchors_before": designer_before,
            "designer_anchors_after": designer_before,
            "proposed_anchors_after": int(detail["anchors_after"]),
            "proposed_designer_anchors_after": int(
                detail.get("designer_anchors_after", detail["anchors_after"])),
            "proposed_final_element": detail.get("final_element"),
            "proposed_path_data_sha256": detail.get("path_data_sha256"),
            "proposed_geometry_sha256": detail.get("geometry_sha256"),
            "error_budget_percent": detail.get("error_budget_percent"),
            "actual_p95_error_percent": detail.get(
                "actual_p95_error_percent"),
            "actual_max_error_percent": detail.get(
                "actual_max_error_percent"),
        })

    full_after = int(updated["anchors_after"])
    full_designer_after = int(updated["designer_anchors_after"])
    restored_anchors = sum(
        int(item["anchors_before"]) - int(item["anchors_after"])
        for item in rolled_details.values())
    restored_designer_anchors = sum(
        int(item.get("designer_anchors_before", item["anchors_before"]))
        - int(item.get("designer_anchors_after", item["anchors_after"]))
        for item in rolled_details.values())
    anchors_before = int(updated["anchors_before"])
    designer_before = int(updated["designer_anchors_before"])
    anchors_after = full_after + restored_anchors
    designer_after = full_designer_after + restored_designer_anchors

    combined_uncertified = list(updated["uncertified_evaluations"])
    combined_uncertified.extend(rollback_records)
    committed_ids = [str(item.get("id")) for item in committed]
    retained_ids = [str(item.get("id")) for item in updated["evaluated_but_retained"]]
    uncertified_ids = [str(item.get("id")) for item in combined_uncertified]
    evidence_ids = committed_ids + retained_ids + uncertified_ids
    optimizer_evaluated = int(updated["optimizer_evaluated_path_count"])
    accounted = len(committed) + len(retained_ids) + len(combined_uncertified)
    if optimizer_evaluated != accounted:
        raise RuntimeError("curve-refit subset accounted evaluation mismatch")
    final_digests = _curve_refit_final_digest_records(
        final_svg_bytes, committed)

    updated.update({
        "status": "proposed" if committed else "no_change",
        "candidate": final_candidate_name,
        "path_count_refit": len(committed),
        "anchors_after": anchors_after,
        "anchors_removed": anchors_before - anchors_after,
        "anchor_reduction_ratio": round(
            (anchors_before - anchors_after) / float(max(1, anchors_before)),
            6),
        "designer_anchors_after": designer_after,
        "designer_anchors_removed": designer_before - designer_after,
        "designer_anchor_reduction_ratio": round(
            (designer_before - designer_after)
            / float(max(1, designer_before)), 6),
        "details": committed,
        "uncertified_evaluations": combined_uncertified,
        "final_committed_path_ids": committed_ids,
        "final_committed_geometry_digests": final_digests,
        "uncertified_transaction_rollback_ids": rollback_ids,
        "transaction_selection_basis": (
            "geometry_only_refits_filtered_by_renderer_and_source_missing_ink_"
            "transaction_guards"),
        "transaction_subset_fallback": copy.deepcopy(subset_evidence),
    })
    integrity = copy.deepcopy(updated.get("evaluation_evidence_integrity") or {})
    integrity.update({
        "schema": "ai-vector-cleanroom.curve-refit-evaluation-evidence/v1",
        "optimizer_evaluated_path_count": optimizer_evaluated,
        "committed_detail_count": len(committed),
        "retained_identity_detail_count": len(retained_ids),
        "uncertified_evaluation_count": len(combined_uncertified),
        "accounted_evaluation_count": accounted,
        "all_optimizer_results_accounted": optimizer_evaluated == accounted,
        "evidence_ids_unique": len(evidence_ids) == len(set(evidence_ids)),
        "committed_and_retained_ids_disjoint": not (
            set(committed_ids) & set(retained_ids)),
        "committed_retained_and_uncertified_ids_disjoint": not (
            set(committed_ids) & set(retained_ids)
            or set(committed_ids) & set(uncertified_ids)
            or set(retained_ids) & set(uncertified_ids)),
        "transaction_rollback_count": len(rollback_ids),
    })
    if (not integrity["all_optimizer_results_accounted"]
            or not integrity["evidence_ids_unique"]
            or not integrity[
                "committed_retained_and_uncertified_ids_disjoint"]):
        raise RuntimeError("curve-refit subset evidence integrity failed")
    updated["evaluation_evidence_integrity"] = integrity
    return updated


def _replace_curve_refit_detail_with_frontier(
        proposal, target_id, replacement_detail, frontier_evidence):
    """Replace one geometry detail without changing evaluated path scope."""
    import copy

    if (not isinstance(proposal, dict)
            or proposal.get("schema") !=
            "ai-vector-cleanroom.curve-refit-proposal/v3"
            or not isinstance(replacement_detail, dict)
            or replacement_detail.get("id") != target_id
            or replacement_detail.get("economy_certified") is not True
            or not isinstance(frontier_evidence, dict)
            or frontier_evidence.get("schema") !=
            "ai-vector-cleanroom.curve-refit-path-frontier/v1"
            or frontier_evidence.get("target_id") != target_id
            or frontier_evidence.get("uses_colour_or_pixel_similarity")
            is not False):
        raise RuntimeError("curve-refit frontier proposal evidence is invalid")
    details = proposal.get("details")
    retained = proposal.get("evaluated_but_retained")
    uncertified = proposal.get("uncertified_evaluations")
    if not all(isinstance(items, list)
               for items in (details, retained, uncertified)):
        raise RuntimeError("curve-refit frontier proposal lists are incomplete")
    target_rows = [item for item in details
                   if isinstance(item, dict) and item.get("id") == target_id]
    other_ids = [str(item.get("id") or "")
                 for item in retained + uncertified if isinstance(item, dict)]
    if len(target_rows) != 1 or target_id in other_ids:
        raise RuntimeError("curve-refit frontier target accounting is invalid")
    previous = target_rows[0]
    before = int(previous.get("anchors_before"))
    designer_before = int(previous.get("designer_anchors_before", before))
    if (int(replacement_detail.get("anchors_before")) != before
            or int(replacement_detail.get(
                "designer_anchors_before", before)) != designer_before
            or not 0 < int(replacement_detail.get("anchors_after")) < before
            or not 0 < int(replacement_detail.get(
                "designer_anchors_after")) < designer_before):
        raise RuntimeError("curve-refit frontier detail economy is invalid")
    updated = copy.deepcopy(proposal)
    updated["details"] = [
        copy.deepcopy(replacement_detail)
        if item.get("id") == target_id else item
        for item in updated["details"]
    ]
    anchor_delta = (int(replacement_detail["anchors_after"])
                    - int(previous["anchors_after"]))
    designer_delta = (int(replacement_detail["designer_anchors_after"])
                      - int(previous.get(
                          "designer_anchors_after", previous["anchors_after"])))
    updated["anchors_after"] = int(updated["anchors_after"]) + anchor_delta
    updated["designer_anchors_after"] = (
        int(updated["designer_anchors_after"]) + designer_delta)
    updated["anchors_removed"] = (
        int(updated["anchors_before"]) - int(updated["anchors_after"]))
    updated["designer_anchors_removed"] = (
        int(updated["designer_anchors_before"])
        - int(updated["designer_anchors_after"]))
    updated["anchor_reduction_ratio"] = round(
        updated["anchors_removed"] / float(max(1, updated["anchors_before"])),
        6)
    updated["designer_anchor_reduction_ratio"] = round(
        updated["designer_anchors_removed"]
        / float(max(1, updated["designer_anchors_before"])), 6)
    updated["transaction_frontier_refinement"] = {
        "schema": frontier_evidence.get("schema"),
        "target_id": target_id,
        "selected_candidate_id": replacement_detail.get(
            "selected_candidate_id"),
        "selection_basis": frontier_evidence.get("selection_basis"),
        "optimization_basis": frontier_evidence.get("optimization_basis"),
        "uses_colour_or_pixel_similarity": False,
    }
    return updated


def _attempt_curve_refit_transaction(svg_path: Path, flat_png: Path,
                                     source_png: Path, stats,
                                     error_budget_percent=0.25):
    """Propose a lower-anchor SVG and commit only after renderer/source guards.

    The proposal fitter has a geometric error bound, but geometry alone is not
    enough: antialiasing, narrow counters and compound-path topology can turn a
    locally plausible curve into a visible regression.  The live SVG therefore
    remains untouched until a separate candidate passes both SVG-to-SVG and
    source-image comparisons.
    """
    import hashlib

    from curve_refit_stage import propose_svg_curve_refit
    from svg_postprocess import atomic_replace_bytes

    original = svg_path.read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    candidate = svg_path.with_name("_curve_refit_proposal.svg")
    hybrid = svg_path.with_name("_curve_refit_hybrid.svg")
    identity_baseline = svg_path.with_name(
        "_curve_refit_identity_baseline.svg")
    commit_started = False
    commit_write_completed = False
    report = {
        "schema": "ai-vector-cleanroom.curve-refit-transaction/v1",
        "status": "not_attempted",
        "before_svg_sha256": original_sha,
        "proposal": None,
        "render_guard": None,
        "source_guard": None,
        "renderer_topology_guard": None,
        "identity_normalization": None,
        "precommit_final_digest_records": None,
        "postcommit_final_digest_records": None,
        "commit_scope": None,
        "gradient_geometry_guard": {
            "status": "not_checked",
            "policy": (
                "gradient_objects_must_remain_unchanged_without_original_"
                "ownership_mask_revalidation"
            ),
            "ownership_mask_revalidation_performed": False,
        },
    }

    error_budget_percent = float(error_budget_percent)

    def metric_guard(before, after):
        # Curve selection has already passed the geometry contract.  Pixel
        # measurements are deliberately only a missing-ink/topology guard:
        # colour similarity and literal source-edge matching must never make
        # the fitter preserve raster stair-steps as anchors.
        limits = {
            "foreground_recall": 0.25,
            "foreground_precision": 0.25,
            "foreground_coverage_f1": 0.25,
        }
        comparisons = {}
        accepted = True
        for key, allowance in limits.items():
            old, new = before.get(key), after.get(key)
            if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
                comparisons[key] = {
                    "before": old, "after": new, "accepted": False,
                    "reason": "metric_unavailable",
                }
                accepted = False
                continue
            delta = float(new) - float(old)
            passed = delta >= -allowance
            comparisons[key] = {
                "before": round(float(old), 6),
                "after": round(float(new), 6),
                "delta": round(delta, 6),
                "maximum_allowed_regression": allowance,
                "accepted": passed,
            }
            accepted &= passed
        before_ink = before.get("render_ink_pixels")
        after_ink = after.get("render_ink_pixels")
        if (isinstance(before_ink, (int, float))
                and isinstance(after_ink, (int, float)) and before_ink > 0):
            ratio = float(after_ink) / float(before_ink)
            ink_passed = 0.985 <= ratio <= 1.015
            comparisons["render_ink_area_ratio"] = {
                "before": int(before_ink),
                "after": int(after_ink),
                "ratio": round(ratio, 6),
                "accepted_range": [0.985, 1.015],
                "accepted": ink_passed,
            }
            accepted &= ink_passed
        return {
            "accepted": bool(accepted),
            "policy": "missing_ink_and_topology_guard_only",
            "excluded_from_decision": [
                "whole_canvas_source_similarity",
                "foreground_colour_fidelity",
                "local_pixel_colour_detail",
            ],
            "comparisons": comparisons,
        }

    def renderer_topology_guard(render_guard):
        required = ("ink_recall_percent", "ink_precision_percent",
                    "ink_f1_percent")
        comparisons = {}
        alpha_guard = render_guard.get("alpha_topology") or {}
        composed_alpha = render_guard.get("composed_alpha") or {}
        accepted = (render_guard.get("external_render_check") == "completed"
                    and alpha_guard.get("accepted") is True
                    and composed_alpha.get("external_render_check") == "completed"
                    and composed_alpha.get("accepted") is True)
        for key in required:
            value = render_guard.get(key)
            passed = isinstance(value, (int, float)) and float(value) >= 99.0
            comparisons[key] = {
                "value": (round(float(value), 6)
                          if isinstance(value, (int, float)) else value),
                "minimum": 99.0,
                "accepted": passed,
            }
            accepted &= passed
        return {
            "accepted": bool(accepted),
            "policy": "bidirectional_ink_topology_guard",
            "comparisons": comparisons,
            "alpha_topology": alpha_guard,
            "composed_alpha": composed_alpha,
            "colour_similarity_excluded": True,
        }

    try:
        gradient_before = _gradient_geometry_snapshot(svg_path)
        gradient_before_digest = _gradient_geometry_digest(gradient_before)
        report["gradient_geometry_guard"].update({
            "before_object_count": len(gradient_before),
            "before_geometry_sha256": gradient_before_digest,
            "before_objects": gradient_before,
        })
        proposal = propose_svg_curve_refit(
            svg_path, candidate,
            error_budget_percent=error_budget_percent,
            # Retained only for backwards-compatible stage diagnostics.  The
            # geometry optimiser selects by normalised error, never by this
            # fixed pixel value.
            tolerance=0.50,
            sample_step=2.0,
            # Four-plus-anchor shapes remain eligible because a proven circle
            # should become one native object even when its designer-anchor
            # count is already four.  Every safe one-anchor reduction matters;
            # there is no percentage floor above the geometry contract.
            minimum_nodes=4,
            minimum_reduction_ratio=0.0,
        )
        report["proposal"] = proposal
        normalized_baseline_bytes, identity_evidence = (
            _curve_refit_identity_normalized_bytes(original, proposal))
        identity_baseline.write_bytes(normalized_baseline_bytes)
        normalized_readback = identity_baseline.read_bytes()
        if normalized_readback != normalized_baseline_bytes:
            raise OSError(
                "curve-refit identity baseline bytes changed after write")
        report["identity_normalization"] = identity_evidence
        full_candidate_bytes = candidate.read_bytes()
        candidate_identity_guard = _curve_refit_candidate_identity_guard(
            full_candidate_bytes, proposal)
        identity_evidence["candidate_identity_guard"] = (
            candidate_identity_guard)
        gradient_after = _gradient_geometry_snapshot(candidate)
        gradient_after_digest = _gradient_geometry_digest(gradient_after)
        skipped_gradient_paths = (
            ((proposal.get("protected_gradient_objects") or {}).get(
                "skipped_path_count"))
            if isinstance(proposal, dict) else None)
        report["gradient_geometry_guard"].update({
            "after_object_count": len(gradient_after),
            "after_geometry_sha256": gradient_after_digest,
            "skipped_gradient_path_count": int(
                skipped_gradient_paths or 0),
        })
        if gradient_after != gradient_before:
            report["gradient_geometry_guard"].update({
                "status": "rejected",
                "reason": "gradient_geometry_changed_without_ownership_mask",
            })
            report.update({
                "status": "rolled_back",
                "reason": "gradient_geometry_ownership_revalidation_missing",
                "after_svg_sha256": original_sha,
                "live_svg_unchanged": svg_path.read_bytes() == original,
            })
            return report
        report["gradient_geometry_guard"].update({
            "status": "verified_unchanged",
            "reason": None,
        })
        # At 512 px a subpixel gap may disappear from the checker while a
        # normal working-size render reveals a real merge of white objects.
        validation_width = _validation_render_width(stats.viewbox, min_longest=1200)
        # Match the primary p95 geometry budget at the independent guard's
        # render scale.  The optimiser separately enforces a 3x maximum tail.
        topology_tolerance_px = max(
            1, int(math.ceil(validation_width
                             * error_budget_percent / 100.0)))
        full_candidate_sha = hashlib.sha256(full_candidate_bytes).hexdigest()
        if full_candidate_sha != candidate_identity_guard.get(
                "candidate_svg_sha256"):
            raise RuntimeError(
                "curve-refit candidate identity digest changed")
        assignment_count = int(identity_evidence.get("assigned_id_count", 0))
        before_scores = self_check(
            svg_path, flat_png, source_png,
            gradient_info=stats.gradient_info, viewbox=stats.viewbox)
        baseline_validation = {
            "status": "not_needed",
            "reason": "no_missing_ids_required_assignment",
            "render_guard": None,
            "renderer_topology_guard": None,
            "source_guard": None,
            "gradient_geometry_guard": None,
            "accepted": True,
        }
        if assignment_count:
            baseline_gradient = _gradient_geometry_snapshot(identity_baseline)
            baseline_gradient_digest = _gradient_geometry_digest(
                baseline_gradient)
            baseline_gradient_accepted = baseline_gradient == gradient_before
            baseline_render_guard = validate_svg_stage_renders(
                svg_path, identity_baseline,
                "curve_refit_identity_normalization",
                gradient_info=stats.gradient_info,
                render_size=validation_width,
                tolerance_px=topology_tolerance_px,
            )
            baseline_topology_guard = renderer_topology_guard(
                baseline_render_guard)
            baseline_scores = self_check(
                identity_baseline, flat_png, source_png,
                gradient_info=stats.gradient_info, viewbox=stats.viewbox)
            baseline_source_guard = metric_guard(before_scores, baseline_scores)
            baseline_accepted = bool(
                baseline_gradient_accepted
                and baseline_topology_guard.get("accepted")
                and baseline_source_guard.get("accepted"))
            baseline_validation = {
                "status": ("verified" if baseline_accepted else "rejected"),
                "reason": (None if baseline_accepted else
                           "identity_normalization_guard_rejected"),
                "render_guard": baseline_render_guard,
                "renderer_topology_guard": baseline_topology_guard,
                "source_guard": baseline_source_guard,
                "gradient_geometry_guard": {
                    "status": ("verified_unchanged"
                               if baseline_gradient_accepted else "rejected"),
                    "before_geometry_sha256": gradient_before_digest,
                    "after_geometry_sha256": baseline_gradient_digest,
                    "accepted": baseline_gradient_accepted,
                },
                "accepted": baseline_accepted,
            }
        identity_evidence["baseline_validation"] = baseline_validation
        if proposal.get("status") != "proposed":
            if proposal.get("status") != "no_change":
                raise RuntimeError("curve-refit proposal status is invalid")
            if not assignment_count:
                report.update({
                    "status": "not_needed",
                    "reason": "no_safe_reductions",
                    "commit_scope": "none",
                })
                return report
            if (full_candidate_bytes != normalized_baseline_bytes
                    or not baseline_validation.get("accepted")):
                report.update({
                    "status": "rolled_back",
                    "reason": "stable_id_normalization_guard_rejected",
                    "after_svg_sha256": original_sha,
                    "live_svg_unchanged": svg_path.read_bytes() == original,
                    "commit_scope": "none",
                })
                return report
            report["render_guard"] = baseline_validation["render_guard"]
            report["renderer_topology_guard"] = baseline_validation[
                "renderer_topology_guard"]
            report["source_guard"] = baseline_validation["source_guard"]
            report["gradient_geometry_guard"].update({
                "after_object_count": len(gradient_before),
                "after_geometry_sha256": gradient_before_digest,
                "status": "verified_unchanged",
                "reason": None,
            })
            report["commit_scope"] = "stable_id_normalization_only"
            expected_final = _curve_refit_expected_final_digest_details(
                proposal)
            precommit_records = _curve_refit_final_digest_records(
                normalized_baseline_bytes, expected_final)
            report["precommit_final_digest_records"] = precommit_records
            proposal["transaction_precommit_final_digest_records"] = (
                precommit_records)
            if svg_path.read_bytes() != original:
                raise RuntimeError(
                    "live SVG changed before stable-ID normalization commit")
            commit_started = True
            atomic_replace_bytes(svg_path, normalized_baseline_bytes)
            commit_write_completed = True
            committed = svg_path.read_bytes()
            if committed != normalized_baseline_bytes:
                raise OSError(
                    "atomic stable-ID normalization did not preserve bytes")
            postcommit_records = _curve_refit_final_digest_records(
                committed, expected_final)
            if postcommit_records != precommit_records:
                raise RuntimeError(
                    "stable-ID normalization postcommit digest mismatch")
            report["postcommit_final_digest_records"] = postcommit_records
            proposal["transaction_postcommit_final_digest_records"] = (
                postcommit_records)
            identity_evidence.update({
                "status": "committed",
                "commit_scope": "stable_id_normalization_only",
                "postcommit_svg_sha256": hashlib.sha256(
                    committed).hexdigest(),
                "precommit_final_digest_records": precommit_records,
                "postcommit_final_digest_records": postcommit_records,
            })
            stats.geometry_notes.append(
                f"{assignment_count} optimizer-evaluated paths received "
                "authenticated stable SVG IDs without geometry changes")
            report.update({
                "status": "committed",
                "reason": None,
                "after_svg_sha256": hashlib.sha256(committed).hexdigest(),
                "live_svg_unchanged": False,
            })
            return report

        precommit_candidate_records = _curve_refit_final_digest_records(
            full_candidate_bytes,
            _curve_refit_expected_final_digest_details(proposal))
        report["full_candidate_precommit_final_digest_records"] = (
            precommit_candidate_records)
        render_guard = validate_svg_stage_renders(
            svg_path, candidate, "curve_refit",
            gradient_info=stats.gradient_info,
            render_size=validation_width,
            tolerance_px=topology_tolerance_px,
        )
        report["render_guard"] = render_guard
        report["renderer_topology_guard"] = renderer_topology_guard(render_guard)
        after_scores = self_check(
            candidate, flat_png, source_png,
            gradient_info=stats.gradient_info, viewbox=stats.viewbox)
        source_guard = metric_guard(before_scores, after_scores)
        report["source_guard"] = source_guard
        if (not report["renderer_topology_guard"].get("accepted")
                or not source_guard.get("accepted")):
            details = proposal.get("details") if isinstance(proposal, dict) else None
            subset_available = bool(
                proposal.get("schema") ==
                "ai-vector-cleanroom.curve-refit-proposal/v3"
                and isinstance(details, list) and details)
            full_validation = {
                "candidate_svg_sha256": full_candidate_sha,
                "render_guard": render_guard,
                "renderer_topology_guard": report[
                    "renderer_topology_guard"],
                "source_guard": source_guard,
            }
            selected = None
            fallback = None
            ranking = []
            if subset_available:
                try:
                    detail_ids = [str(item.get("id") or "")
                                  for item in details
                                  if isinstance(item, dict)]
                    if (len(detail_ids) != len(details) or not all(detail_ids)
                            or len(detail_ids) != len(set(detail_ids))):
                        raise RuntimeError(
                            "curve-refit subset detail IDs are invalid")

                    def rollback_rank(item):
                        before = int(item.get("anchors_before", 0))
                        after = int(item.get("anchors_after", before))
                        designer_before = int(item.get(
                            "designer_anchors_before", before))
                        designer_after = int(item.get(
                            "designer_anchors_after", after))
                        return (
                            designer_before - designer_after,
                            before - after,
                            int(item.get("loops", 0) or 0),
                            str(item.get("id") or ""),
                        )

                    ranking = sorted(details, key=rollback_rank)
                    ranking_ids = [str(item["id"]) for item in ranking]
                    detail_by_id = {str(item["id"]): item for item in details}
                    fallback = {
                        "schema": (
                            "ai-vector-cleanroom.curve-refit-subset-fallback/v1"),
                        "status": "searching",
                        "selection_policy": (
                            "cumulative_exact_source_element_rollback_ordered_by_"
                            "minimum_designer_anchor_restoration_cost_then_"
                            "minimum_anchor_restoration_cost_then_loop_count_"
                            "then_id"),
                        "search_policy": (
                            "deterministic_exponential_bracket_then_bisection"),
                        "search_assumption": (
                            "cumulative exact-source restoration is expected to "
                            "move missing-ink guards toward the source baseline; "
                            "tested non-monotonic results fail closed and the "
                            "selected exact bytes are independently revalidated"),
                        "full_candidate_path_count": len(details),
                        "full_candidate_svg_sha256": full_candidate_sha,
                        "full_candidate_validation": full_validation,
                        "rollback_order_ids": ranking_ids,
                        "rollback_costs": [{
                            "id": str(item["id"]),
                            "designer_anchors_restored": (
                                int(item.get("designer_anchors_before",
                                             item["anchors_before"]))
                                - int(item.get("designer_anchors_after",
                                               item["anchors_after"]))),
                            "anchors_restored": (
                                int(item["anchors_before"])
                                - int(item["anchors_after"])),
                            "loops": int(item.get("loops", 0) or 0),
                        } for item in ranking],
                        "attempts": [],
                    }
                    evaluations = {}

                    def candidate_economy(rollback_ids):
                        rollback_set = set(rollback_ids)
                        restored_anchors = sum(
                            int(detail_by_id[identifier]["anchors_before"])
                            - int(detail_by_id[identifier]["anchors_after"])
                            for identifier in rollback_ids)
                        restored_designer = sum(
                            int(detail_by_id[identifier].get(
                                "designer_anchors_before",
                                detail_by_id[identifier]["anchors_before"]))
                            - int(detail_by_id[identifier].get(
                                "designer_anchors_after",
                                detail_by_id[identifier]["anchors_after"]))
                            for identifier in rollback_ids)
                        anchors_before = int(proposal["anchors_before"])
                        designer_before = int(
                            proposal["designer_anchors_before"])
                        anchors_after = (
                            int(proposal["anchors_after"])
                            + restored_anchors)
                        designer_after = (
                            int(proposal["designer_anchors_after"])
                            + restored_designer)
                        return {
                            "rollback_count": len(rollback_ids),
                            "rollback_ids": list(rollback_ids),
                            "committed_refit_ids": [
                                identifier for identifier in detail_ids
                                if identifier not in rollback_set],
                            "anchors_before": anchors_before,
                            "anchors_after": anchors_after,
                            "anchors_removed": anchors_before - anchors_after,
                            "designer_anchors_before": designer_before,
                            "designer_anchors_after": designer_after,
                            "designer_anchors_removed": (
                                designer_before - designer_after),
                        }

                    def evaluate_candidate_bytes(
                            hybrid_bytes, *, phase, rollback_ids,
                            economy, integrity_guard):
                        hybrid_bytes = bytes(hybrid_bytes)
                        hybrid.write_bytes(hybrid_bytes)
                        hybrid_sha = hashlib.sha256(hybrid_bytes).hexdigest()
                        hybrid_gradient = _gradient_geometry_snapshot(hybrid)
                        hybrid_gradient_digest = _gradient_geometry_digest(
                            hybrid_gradient)
                        gradient_accepted = hybrid_gradient == gradient_before
                        attempt = {
                            "phase": phase,
                            "rollback_count": len(rollback_ids),
                            "rollback_ids": list(rollback_ids),
                            "candidate_economy": economy,
                            "integrity_guard": dict(integrity_guard),
                            "candidate_svg_sha256": hybrid_sha,
                            "gradient_geometry_guard": {
                                "status": ("verified_unchanged"
                                           if gradient_accepted else "rejected"),
                                "before_geometry_sha256": (
                                    gradient_before_digest),
                                "after_geometry_sha256": hybrid_gradient_digest,
                                "accepted": gradient_accepted,
                            },
                            "render_guard": None,
                            "renderer_topology_guard": None,
                            "source_guard": None,
                            "accepted": False,
                        }
                        result = {
                            "evidence": attempt,
                            "bytes": hybrid_bytes,
                            "gradient": hybrid_gradient,
                            "gradient_digest": hybrid_gradient_digest,
                        }
                        if gradient_accepted:
                            candidate_render_guard = validate_svg_stage_renders(
                                svg_path, hybrid, "curve_refit_subset",
                                gradient_info=stats.gradient_info,
                                render_size=validation_width,
                                tolerance_px=topology_tolerance_px,
                            )
                            candidate_topology_guard = renderer_topology_guard(
                                candidate_render_guard)
                            candidate_scores = self_check(
                                hybrid, flat_png, source_png,
                                gradient_info=stats.gradient_info,
                                viewbox=stats.viewbox)
                            candidate_source_guard = metric_guard(
                                before_scores, candidate_scores)
                            candidate_accepted = bool(
                                candidate_topology_guard.get("accepted")
                                and candidate_source_guard.get("accepted"))
                            attempt.update({
                                "render_guard": candidate_render_guard,
                                "renderer_topology_guard": (
                                    candidate_topology_guard),
                                "source_guard": candidate_source_guard,
                                "accepted": candidate_accepted,
                            })
                        return result

                    def evaluate_rollback_ids(rollback_ids, *, phase):
                        rollback_ids = list(rollback_ids)
                        if (not rollback_ids
                                or len(rollback_ids) != len(set(rollback_ids))
                                or not set(rollback_ids).issubset(detail_by_id)):
                            raise RuntimeError(
                                "curve-refit rollback probe IDs are invalid")
                        hybrid_bytes = _curve_refit_hybrid_bytes(
                            normalized_baseline_bytes, full_candidate_bytes,
                            rollback_ids,
                            detail_by_id)
                        return evaluate_candidate_bytes(
                            hybrid_bytes, phase=phase,
                            rollback_ids=rollback_ids,
                            economy=candidate_economy(rollback_ids),
                            integrity_guard={
                                "requested_ids_unique": True,
                                "requested_ids_known": True,
                                "exact_source_element_join_authenticated": True,
                            })

                    def evaluate_rollback_count(
                            count, *, use_cache=True, phase="search"):
                        if use_cache and count in evaluations:
                            return evaluations[count]
                        result = evaluate_rollback_ids(
                            ranking_ids[:count], phase=phase)
                        attempt = result["evidence"]
                        if use_cache:
                            evaluations[count] = result
                            fallback["attempts"].append(attempt)
                        else:
                            fallback["final_revalidation"] = attempt
                        return result

                    max_partial = len(ranking) - 1
                    failed_lower = 0
                    passing_upper = None
                    probe = 1
                    while max_partial > 0:
                        probe = min(probe, max_partial)
                        evaluation = evaluate_rollback_count(probe)
                        if evaluation["evidence"]["accepted"]:
                            passing_upper = probe
                            break
                        failed_lower = probe
                        if probe >= max_partial:
                            break
                        probe = min(max_partial, probe * 2)
                    if passing_upper is not None:
                        while passing_upper - failed_lower > 1:
                            middle = (passing_upper + failed_lower) // 2
                            evaluation = evaluate_rollback_count(middle)
                            if evaluation["evidence"]["accepted"]:
                                passing_upper = middle
                            else:
                                failed_lower = middle
                        ordered_evaluations = [
                            evaluations[count]["evidence"]
                            for count in sorted(evaluations)
                        ]
                        seen_passing = False
                        monotonicity_violation = False
                        for evidence in ordered_evaluations:
                            if evidence.get("accepted"):
                                seen_passing = True
                            elif seen_passing:
                                monotonicity_violation = True
                                break
                        fallback["tested_monotonicity"] = {
                            "accepted": not monotonicity_violation,
                            "evaluated_rollback_counts": [
                                item["rollback_count"]
                                for item in ordered_evaluations],
                            "accepted_rollback_counts": [
                                item["rollback_count"]
                                for item in ordered_evaluations
                                if item.get("accepted")],
                        }
                        if not monotonicity_violation:
                            cached = evaluations[passing_upper]
                            final_validation = evaluate_rollback_count(
                                passing_upper, use_cache=False,
                                phase="final_revalidation")
                            exact_bytes_stable = (
                                final_validation["evidence"][
                                    "candidate_svg_sha256"]
                                == cached["evidence"][
                                    "candidate_svg_sha256"])
                            fallback["final_revalidation"][
                                "exact_candidate_bytes_stable"] = (
                                    exact_bytes_stable)
                            if (final_validation["evidence"]["accepted"]
                                    and exact_bytes_stable):
                                selected = final_validation
                                selected["rollback_ids"] = (
                                    ranking_ids[:passing_upper])
                                prefix_ids = list(selected["rollback_ids"])
                                prefix_economy = candidate_economy(prefix_ids)
                                fallback.update({
                                    "status": "partial_candidate_selected",
                                    "selected_candidate_strategy": (
                                        "minimal_passing_prefix"),
                                    "selection_reason": (
                                        "minimal_guard_passing_prefix_passed_"
                                        "exact_final_revalidation"),
                                    "selected_rollback_count": passing_upper,
                                    "selected_rollback_ids": (
                                        selected["rollback_ids"]),
                                    "selected_candidate_svg_sha256": (
                                        selected["evidence"][
                                            "candidate_svg_sha256"]),
                                    "prefix_candidate": {
                                        "status": "passed",
                                        "rollback_count": passing_upper,
                                        "rollback_ids": prefix_ids,
                                        "candidate_economy": prefix_economy,
                                        "candidate_svg_sha256": (
                                            selected["evidence"][
                                                "candidate_svg_sha256"]),
                                        "final_revalidation": dict(
                                            fallback["final_revalidation"]),
                                    },
                                })
                                boundary_id = ranking_ids[
                                    passing_upper - 1]
                                boundary_probe = {
                                    "schema": (
                                        "ai-vector-cleanroom.curve-refit-"
                                        "boundary-single-path-probe/v1"),
                                    "strategy": (
                                        "rollback_only_the_boundary_rank_of_"
                                        "the_minimal_passing_prefix"),
                                    "minimal_passing_prefix_count": (
                                        passing_upper),
                                    "failed_predecessor_count": (
                                        passing_upper - 1),
                                    "boundary_rank_1_based": passing_upper,
                                    "boundary_id": boundary_id,
                                    "prefix_candidate_economy": (
                                        prefix_economy),
                                    "boundary_candidate_economy": (
                                        candidate_economy([boundary_id])),
                                    "status": "not_run",
                                    "probe": None,
                                    "final_revalidation": None,
                                    "selection_reason": None,
                                }
                                fallback["selection_refinement_policy"] = (
                                    "after_a_minimal_passing_prefix_and_"
                                    "failed_immediate_predecessor_test_the_"
                                    "single_boundary_path_without_weakening_"
                                    "any_guard")
                                fallback["boundary_single_path_probe"] = (
                                    boundary_probe)
                                if passing_upper == 1:
                                    boundary_probe.update({
                                        "status": "not_run_equivalent",
                                        "selection_reason": (
                                            "boundary_only_candidate_equals_"
                                            "the_one_path_passing_prefix"),
                                    })
                                else:
                                    predecessor = evaluations.get(
                                        passing_upper - 1)
                                    predecessor_evidence = (
                                        predecessor.get("evidence")
                                        if isinstance(predecessor, dict)
                                        else None)
                                    predecessor_failed = bool(
                                        isinstance(predecessor_evidence, dict)
                                        and predecessor_evidence.get(
                                            "accepted") is False)
                                    boundary_probe[
                                        "failed_predecessor_verified"] = (
                                            predecessor_failed)
                                    boundary_probe[
                                        "failed_predecessor"] = (
                                            predecessor_evidence)
                                    if not predecessor_failed:
                                        boundary_probe.update({
                                            "status": "not_run_fail_closed",
                                            "selection_reason": (
                                                "immediate_predecessor_failure_"
                                                "was_not_authenticated"),
                                        })
                                    else:
                                        try:
                                            probe_result = (
                                                evaluate_rollback_ids(
                                                    [boundary_id],
                                                    phase=(
                                                        "boundary_single_path_"
                                                        "probe")))
                                            boundary_probe["probe"] = (
                                                probe_result["evidence"])
                                            if probe_result["evidence"].get(
                                                    "accepted"):
                                                boundary_final = (
                                                    evaluate_rollback_ids(
                                                        [boundary_id],
                                                        phase=(
                                                            "boundary_single_"
                                                            "path_final_"
                                                            "revalidation")))
                                                boundary_final_evidence = (
                                                    boundary_final[
                                                        "evidence"])
                                                boundary_exact_stable = (
                                                    boundary_final_evidence[
                                                        "candidate_svg_sha256"]
                                                    == probe_result[
                                                        "evidence"][
                                                        "candidate_svg_sha256"])
                                                boundary_final_evidence[
                                                    "exact_candidate_bytes_"
                                                    "stable"] = (
                                                        boundary_exact_stable)
                                                boundary_probe[
                                                    "final_revalidation"] = (
                                                        boundary_final_evidence)
                                                prefix_pair = (
                                                    prefix_economy[
                                                        "designer_anchors_after"],
                                                    prefix_economy[
                                                        "anchors_after"])
                                                boundary_economy = (
                                                    boundary_probe[
                                                        "boundary_candidate_"
                                                        "economy"])
                                                boundary_pair = (
                                                    boundary_economy[
                                                        "designer_anchors_after"],
                                                    boundary_economy[
                                                        "anchors_after"])
                                                economy_improved = (
                                                    boundary_pair < prefix_pair)
                                                boundary_probe[
                                                    "economy_improves_prefix"] = (
                                                        economy_improved)
                                                if (boundary_final_evidence.get(
                                                        "accepted")
                                                        and boundary_exact_stable
                                                        and economy_improved):
                                                    selected = boundary_final
                                                    selected[
                                                        "rollback_ids"] = [
                                                            boundary_id]
                                                    fallback[
                                                        "final_revalidation"] = (
                                                            boundary_final_evidence)
                                                    fallback.update({
                                                        "selected_candidate_"
                                                        "strategy": (
                                                            "boundary_single_"
                                                            "path_probe"),
                                                        "selection_reason": (
                                                            "boundary_only_"
                                                            "candidate_passed_"
                                                            "all_guards_and_"
                                                            "exact_final_"
                                                            "revalidation_with_"
                                                            "fewer_designer_"
                                                            "anchors_than_"
                                                            "the_prefix"),
                                                        "selected_rollback_count": 1,
                                                        "selected_rollback_ids": [
                                                            boundary_id],
                                                        "selected_candidate_"
                                                        "svg_sha256": (
                                                            boundary_final_evidence[
                                                                "candidate_svg_"
                                                                "sha256"]),
                                                    })
                                                    boundary_probe.update({
                                                        "status": "selected",
                                                        "selection_reason": (
                                                            fallback[
                                                                "selection_reason"]),
                                                    })
                                                else:
                                                    boundary_probe.update({
                                                        "status": (
                                                            "rejected_final_"
                                                            "revalidation"),
                                                        "selection_reason": (
                                                            "boundary_probe_"
                                                            "did_not_preserve_"
                                                            "accepted_stable_"
                                                            "bytes_or_improve_"
                                                            "prefix_economy"),
                                                    })
                                            else:
                                                boundary_probe.update({
                                                    "status": (
                                                        "rejected_by_guards"),
                                                    "selection_reason": (
                                                        "boundary_only_"
                                                        "candidate_failed_the_"
                                                        "same_transaction_"
                                                        "guards"),
                                                })
                                        except Exception as boundary_exc:
                                            boundary_probe.update({
                                                "status": (
                                                    "unavailable_fail_closed"),
                                                "error": repr(
                                                    boundary_exc)[:300],
                                                "selection_reason": (
                                                    "boundary_probe_integrity_"
                                                    "or_validation_failed;_"
                                                    "retained_verified_prefix"),
                                            })
                            else:
                                fallback.update({
                                    "status": "all_refits_rolled_back",
                                    "reason": (
                                        "selected_candidate_final_"
                                        "revalidation_failed"),
                                    "selected_rollback_count": len(ranking),
                                    "selected_rollback_ids": ranking_ids,
                                    "selected_candidate_svg_sha256": (
                                        original_sha),
                                })
                        else:
                            fallback.update({
                                "status": "all_refits_rolled_back",
                                "reason": (
                                    "non_monotonic_subset_guard_results"),
                                "selected_rollback_count": len(ranking),
                                "selected_rollback_ids": ranking_ids,
                                "selected_candidate_svg_sha256": original_sha,
                            })
                    else:
                        fallback.update({
                            "status": "all_refits_rolled_back",
                            "reason": (
                                "no_guard_passing_partial_candidate"
                                if max_partial > 0 else
                                "single_refit_has_no_nonempty_partial_subset"),
                            "selected_rollback_count": len(ranking),
                            "selected_rollback_ids": ranking_ids,
                            "selected_candidate_svg_sha256": original_sha,
                        })
                except Exception as subset_exc:
                    fallback = {
                        "schema": (
                            "ai-vector-cleanroom.curve-refit-subset-fallback/v1"),
                        "status": "unavailable",
                        "reason": "subset_evidence_or_join_invalid",
                        "error": repr(subset_exc)[:300],
                        "full_candidate_svg_sha256": full_candidate_sha,
                        "full_candidate_validation": full_validation,
                        "attempts": [],
                    }

            # Failure along one rollback order is not proof that every refit
            # is unsafe. A single high-saving path can remain in every prefix
            # probe and poison all of them. Test bounded blocks against the
            # frozen original, accumulating only globally verified changes.
            # This changes the search, never the geometry/render/source gates.
            if (selected is None and isinstance(fallback, dict)
                    and fallback.get("reason") == "no_guard_passing_partial_candidate"
                    and len(ranking) > 1):
                import time
                search_started = time.monotonic()
                search = {"policy": "bounded_divide_and_keep_verified_blocks",
                          "maximum_probes": 24, "search_budget_seconds": 45,
                          "one_probe_and_final_revalidation_may_overrun_budget": True,
                          "attempts": [], "status": "searching",
                          "optimality_proven": False}
                fallback["bounded_subset_search"] = search
                try:
                    by_gain = list(reversed(ranking_ids))
                    middle = len(by_gain) // 2
                    pending = [by_gain[:middle], by_gain[middle:]]
                    kept = set()
                    last_verified = None
                    while (pending and len(search["attempts"]) < 24
                           and time.monotonic() - search_started < 45):
                        block = pending.pop(0)
                        trial_kept = kept | set(block)
                        restore = [identifier for identifier in ranking_ids
                                   if identifier not in trial_kept]
                        if not restore:
                            # Exact full proposal already failed above.
                            passed = False
                        else:
                            result = evaluate_rollback_ids(
                                restore, phase="bounded_block_search")
                            search["attempts"].append(result["evidence"])
                            passed = result["evidence"].get("accepted") is True
                            if passed:
                                kept = trial_kept
                                last_verified = result
                        if not passed and len(block) > 1:
                            middle = len(block) // 2
                            pending[0:0] = [block[:middle], block[middle:]]
                    search["remaining_block_count"] = len(pending)
                    if kept and last_verified is not None:
                        restore = [identifier for identifier in ranking_ids
                                   if identifier not in kept]
                        final = evaluate_rollback_ids(
                            restore, phase="bounded_block_final_revalidation")
                        stable = (final["evidence"]["candidate_svg_sha256"]
                                  == last_verified["evidence"]["candidate_svg_sha256"])
                        final["evidence"]["exact_candidate_bytes_stable"] = stable
                        search["final_revalidation"] = final["evidence"]
                        if final["evidence"].get("accepted") and stable:
                            selected = final
                            selected["rollback_ids"] = restore
                            fallback.update({
                                "status": "partial_candidate_selected", "reason": None,
                                "selected_candidate_strategy": "bounded_verified_blocks",
                                "selection_reason": "independent_blocks_passed_all_unchanged_guards_and_exact_final_revalidation",
                                "selected_rollback_count": len(restore),
                                "selected_rollback_ids": restore,
                                "selected_candidate_svg_sha256": final["evidence"]["candidate_svg_sha256"],
                                "final_revalidation": final["evidence"],
                            })
                            search["status"] = "verified_subset_selected"
                        else:
                            search["status"] = "rejected_final_revalidation"
                    else:
                        search["status"] = "no_verified_subset_within_budget"
                except Exception as search_exc:
                    search.update({"status": "unavailable_fail_closed",
                                   "error": repr(search_exc)[:300]})
                search["machine_elapsed_seconds"] = round(time.monotonic() - search_started, 3)

            if (selected is not None and isinstance(fallback, dict)
                    and fallback.get("selected_candidate_strategy") ==
                    "boundary_single_path_probe"
                    and fallback.get("selected_rollback_count") == 1):
                boundary_ids = list(fallback.get("selected_rollback_ids") or [])
                frontier_transaction = {
                    "schema": (
                        "ai-vector-cleanroom.curve-refit-path-frontier-"
                        "transaction/v1"),
                    "status": "not_attempted",
                    "selection_policy": (
                        "geometry_only_per_loop_conservative_candidates_"
                        "ordered_by_designer_anchors_then_anchors_then_"
                        "segments_then_geometry_then_id;_pixels_only_guard_"
                        "the_transaction"),
                    "attempts": [],
                }
                fallback["per_loop_refinement_frontier"] = frontier_transaction
                try:
                    if len(boundary_ids) != 1:
                        raise RuntimeError(
                            "boundary refinement requires exactly one ID")
                    boundary_id = boundary_ids[0]
                    original_detail = detail_by_id.get(boundary_id)
                    if not isinstance(original_detail, dict):
                        raise RuntimeError(
                            "boundary refinement detail is missing")
                    source_anchors = int(original_detail.get(
                        "anchors_before", 0))
                    # Match the existing designer high-node failure threshold.
                    # This only controls whether an expensive diagnostic
                    # frontier is attempted; it never relaxes acceptance.
                    if source_anchors < 80:
                        frontier_transaction.update({
                            "status": "not_needed",
                            "reason": "source_path_below_high_node_threshold",
                        })
                    else:
                        from curve_refit_stage import (
                            apply_svg_curve_refit_path_frontier_candidate,
                            build_svg_curve_refit_path_frontier,
                        )
                        path_frontier = build_svg_curve_refit_path_frontier(
                            identity_baseline, boundary_id,
                            error_budget_percent=error_budget_percent,
                            sample_step=2.0,
                            maximum_segments=4096)
                        frontier_transaction["frontier"] = {
                            key: path_frontier.get(key) for key in (
                                "schema", "status", "target_id",
                                "optimization_basis",
                                "uses_colour_or_pixel_similarity",
                                "source_path_data_sha256",
                                "source_anchor_count",
                                "source_loop_anchor_counts", "loop_count",
                                "error_budget_percent", "selection_basis",
                                "base_candidate", "candidate_count")
                        }
                        if path_frontier.get("status") != "candidates_available":
                            frontier_transaction.update({
                                "status": "not_available",
                                "reason": path_frontier.get("status"),
                            })
                        else:
                            base_boundary_bytes = selected["bytes"]
                            frontier_selected = None
                            frontier_selected_proposal = None
                            for frontier_row in path_frontier.get(
                                    "candidates", []):
                                applied = (
                                    apply_svg_curve_refit_path_frontier_candidate(
                                        base_boundary_bytes,
                                        target_id=boundary_id,
                                        frontier=path_frontier,
                                        candidate=frontier_row))
                                replacement_detail = applied["detail"]
                                updated_proposal = (
                                    _replace_curve_refit_detail_with_frontier(
                                        proposal, boundary_id,
                                        replacement_detail, path_frontier))
                                economy = {
                                    "rollback_count": 0,
                                    "rollback_ids": [],
                                    "committed_refit_ids": list(detail_ids),
                                    "anchors_before": int(
                                        updated_proposal["anchors_before"]),
                                    "anchors_after": int(
                                        updated_proposal["anchors_after"]),
                                    "anchors_removed": int(
                                        updated_proposal["anchors_removed"]),
                                    "designer_anchors_before": int(
                                        updated_proposal[
                                            "designer_anchors_before"]),
                                    "designer_anchors_after": int(
                                        updated_proposal[
                                            "designer_anchors_after"]),
                                    "designer_anchors_removed": int(
                                        updated_proposal[
                                            "designer_anchors_removed"]),
                                }
                                probe = evaluate_candidate_bytes(
                                    applied["bytes"],
                                    phase=(
                                        "boundary_per_loop_refinement_probe"),
                                    rollback_ids=[], economy=economy,
                                    integrity_guard={
                                        **dict(applied.get("integrity") or {}),
                                        "frontier_candidate_authoritative": True,
                                        "geometry_only_ranking": True,
                                    })
                                attempt = {
                                    "candidate_id": frontier_row.get(
                                        "candidate_id"),
                                    "changed_loop_index": frontier_row.get(
                                        "changed_loop_index"),
                                    "designer_anchor_count": frontier_row.get(
                                        "designer_anchor_count"),
                                    "anchors_after": frontier_row.get(
                                        "anchors_after"),
                                    "segment_count": frontier_row.get(
                                        "segment_count"),
                                    "actual_max_error_percent": (
                                        frontier_row.get(
                                            "actual_max_error_percent")),
                                    "actual_p95_error_percent": (
                                        frontier_row.get(
                                            "actual_p95_error_percent")),
                                    "path_data_sha256": hashlib.sha256(
                                        frontier_row["path"].encode(
                                            "utf-8")).hexdigest(),
                                    "probe": probe["evidence"],
                                    "final_revalidation": None,
                                }
                                frontier_transaction["attempts"].append(attempt)
                                if not probe["evidence"].get("accepted"):
                                    continue
                                final_applied = (
                                    apply_svg_curve_refit_path_frontier_candidate(
                                        base_boundary_bytes,
                                        target_id=boundary_id,
                                        frontier=path_frontier,
                                        candidate=frontier_row))
                                final = evaluate_candidate_bytes(
                                    final_applied["bytes"],
                                    phase=(
                                        "boundary_per_loop_refinement_"
                                        "final_revalidation"),
                                    rollback_ids=[], economy=economy,
                                    integrity_guard={
                                        **dict(final_applied.get(
                                            "integrity") or {}),
                                        "frontier_candidate_authoritative": True,
                                        "geometry_only_ranking": True,
                                    })
                                exact_stable = bool(
                                    final_applied["bytes"] == applied["bytes"]
                                    and final["evidence"][
                                        "candidate_svg_sha256"] ==
                                    probe["evidence"][
                                        "candidate_svg_sha256"])
                                final["evidence"][
                                    "exact_candidate_bytes_stable"] = exact_stable
                                attempt["final_revalidation"] = final["evidence"]
                                if (final["evidence"].get("accepted")
                                        and exact_stable):
                                    frontier_selected = final
                                    frontier_selected["rollback_ids"] = []
                                    frontier_selected_proposal = updated_proposal
                                    frontier_transaction.update({
                                        "status": "selected",
                                        "selected_candidate_id": (
                                            frontier_row.get("candidate_id")),
                                        "selected_candidate_svg_sha256": (
                                            final["evidence"][
                                                "candidate_svg_sha256"]),
                                        "selected_designer_anchor_count": (
                                            frontier_row.get(
                                                "designer_anchor_count")),
                                        "selected_anchor_count": (
                                            frontier_row.get("anchors_after")),
                                    })
                                    break
                            if frontier_selected is None:
                                frontier_transaction.update({
                                    "status": "no_guard_passing_candidate",
                                    "reason": (
                                        "retained_verified_boundary_identity"),
                                })
                            else:
                                selected = frontier_selected
                                proposal = frontier_selected_proposal
                                fallback.update({
                                    "selected_candidate_strategy": (
                                        "per_loop_conservative_refinement_"
                                        "frontier"),
                                    "selection_reason": (
                                        "geometry_ranked_single_loop_"
                                        "refinement_passed_unchanged_"
                                        "renderer_source_gradient_guards_and_"
                                        "exact_final_revalidation"),
                                    "selected_rollback_count": 0,
                                    "selected_rollback_ids": [],
                                    "selected_candidate_svg_sha256": (
                                        selected["evidence"][
                                            "candidate_svg_sha256"]),
                                    "final_revalidation": (
                                        selected["evidence"]),
                                })
                except Exception as frontier_exc:
                    frontier_transaction.update({
                        "status": "unavailable_fail_closed",
                        "error": repr(frontier_exc)[:300],
                        "reason": "retained_verified_boundary_identity",
                    })

            if selected is not None:
                selected_bytes = selected["bytes"]
                hybrid.write_bytes(selected_bytes)
                if hashlib.sha256(hybrid.read_bytes()).hexdigest() != (
                        selected["evidence"]["candidate_svg_sha256"]):
                    raise OSError(
                        "curve-refit hybrid bytes changed after selection")
                proposal = _reconcile_curve_refit_subset_proposal(
                    proposal, selected["rollback_ids"], selected_bytes,
                    hybrid.name, fallback)
                report["proposal"] = proposal
                report["subset_fallback"] = fallback
                report["render_guard"] = selected["evidence"]["render_guard"]
                report["renderer_topology_guard"] = selected[
                    "evidence"]["renderer_topology_guard"]
                report["source_guard"] = selected["evidence"]["source_guard"]
                report["gradient_geometry_guard"].update({
                    "after_object_count": len(selected["gradient"]),
                    "after_geometry_sha256": selected["gradient_digest"],
                    "status": "verified_unchanged",
                    "reason": None,
                })
                full_candidate_bytes = selected_bytes
            else:
                if fallback is not None:
                    report["subset_fallback"] = fallback
                if (fallback is not None
                        and fallback.get("status") ==
                        "all_refits_rolled_back"):
                    proposal = _reconcile_curve_refit_subset_proposal(
                        proposal, [str(item["id"]) for item in ranking],
                        normalized_baseline_bytes, identity_baseline.name,
                        fallback)
                    report["proposal"] = proposal
                report.update({
                    "status": "rolled_back",
                    "reason": "renderer_or_source_guard_rejected",
                    "after_svg_sha256": original_sha,
                    "live_svg_unchanged": svg_path.read_bytes() == original,
                })
                return report

        candidate_bytes = full_candidate_bytes
        expected_final = _curve_refit_expected_final_digest_details(proposal)
        precommit_records = _curve_refit_final_digest_records(
            candidate_bytes, expected_final)
        report["precommit_final_digest_records"] = precommit_records
        proposal["transaction_precommit_final_digest_records"] = (
            precommit_records)
        if svg_path.read_bytes() != original:
            raise RuntimeError("live SVG changed before curve-refit commit")
        commit_started = True
        atomic_replace_bytes(svg_path, candidate_bytes)
        commit_write_completed = True
        committed = svg_path.read_bytes()
        if committed != candidate_bytes:
            raise OSError("atomic curve-refit commit did not preserve bytes")
        postcommit_records = _curve_refit_final_digest_records(
            committed, expected_final)
        if postcommit_records != precommit_records:
            raise RuntimeError("curve-refit postcommit digest records changed")
        report["postcommit_final_digest_records"] = postcommit_records
        proposal["transaction_postcommit_final_digest_records"] = (
            postcommit_records)
        after_sha = hashlib.sha256(committed).hexdigest()
        assignment_count = int(
            (report.get("identity_normalization") or {}).get(
                "assigned_id_count", 0))
        report["commit_scope"] = (
            "geometry_refit_with_stable_id_normalization"
            if assignment_count else "geometry_refit")
        if isinstance(report.get("identity_normalization"), dict):
            report["identity_normalization"].update({
                "status": "committed",
                "commit_scope": report["commit_scope"],
                "postcommit_svg_sha256": after_sha,
                "precommit_final_digest_records": precommit_records,
                "postcommit_final_digest_records": postcommit_records,
            })
        stats.geometry_notes.append(
            f"{proposal.get('anchors_removed', 0)} redundant path anchors removed "
            f"inside a {error_budget_percent:.3g}% geometry budget; native "
            "primitives and minimum designer anchors selected first")
        report.update({
            "status": "committed",
            "reason": None,
            "after_svg_sha256": after_sha,
            "live_svg_unchanged": False,
        })
        return report
    except Exception as exc:
        current = svg_path.read_bytes() if svg_path.is_file() else b""
        rollback_error = None
        rollback_verified = current == original
        rollback_required = bool(
            commit_started and (commit_write_completed or not rollback_verified))
        if rollback_required and not rollback_verified:
            try:
                atomic_replace_bytes(svg_path, original)
                rollback_verified = svg_path.read_bytes() == original
                if not rollback_verified:
                    raise OSError(
                        "curve-refit rollback readback did not match original")
                current = original
            except Exception as restore_exc:
                rollback_error = repr(restore_exc)[:300]
                current = svg_path.read_bytes() if svg_path.is_file() else b""
        status = "rolled_back" if rollback_required and rollback_verified else "error"
        reason = (
            "postcommit_verification_failed_and_original_restored"
            if status == "rolled_back" else "curve_refit_exception")
        if isinstance(report.get("identity_normalization"), dict):
            report["identity_normalization"].update({
                "status": status,
                "commit_scope": "none",
                "rollback_to_frozen_original_verified": rollback_verified,
            })
        report.update({
            "status": status,
            "reason": reason,
            "commit_scope": "none",
            "error": repr(exc)[:300],
            "rollback_error": rollback_error,
            "rollback_to_frozen_original_verified": rollback_verified,
            "after_svg_sha256": (
                hashlib.sha256(current).hexdigest() if current else None),
            "live_svg_unchanged": current == original,
        })
        return report
    finally:
        candidate.unlink(missing_ok=True)
        hybrid.unlink(missing_ok=True)
        identity_baseline.unlink(missing_ok=True)


REVIEW_PREVIEW_MAX_SIDE = 1600


def data_url(path: Path, max_side=REVIEW_PREVIEW_MAX_SIDE):
    """Embed the image as a data URL, downscaled to keep review.html small."""
    import io
    from PIL import Image
    im = Image.open(path)
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b}"


def compute_hotspots(render_png: Path, source_png: Path, viewbox,
                     cell=48, max_spots=40):
    """Compatibility wrapper for source-ink local-detail diagnostics."""
    try:
        from quality_diagnostics import compute_hotspots as _compute
        return _compute(render_png, source_png, viewbox,
                        cell=cell, max_spots=max_spots)
    except Exception:
        return []


def make_review_html(out: Path, name: str, original_png: Path, svg_text: str,
                     size, hotspots=None, scores=None, structure=None,
                     acceptance_status="accepted",
                     manual_review_required=False,
                     visual_acceptance_status="accepted",
                     editability_status="accepted", editability_score=None,
                     automation_readiness_score=None,
                     human_validation_status="not_performed",
                     designer_readiness_status="designer_ready",
                     gradient_object_gate_status="passed",
                     curve_economy_gate_status="passed",
                     detail_grid=None, recolor_filename=None):
    """Review workbench: zoomable overlay (100%%-1600%%), object list with
    layer toggles and click-to-highlight, and clickable problem hotspots."""
    import json as _json
    w, h = size
    svg_inline = svg_text.split("?>", 1)[-1].strip()
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_inline)
    vb = [float(m.group(1)), float(m.group(2))] if m else [w, h]
    sc = scores or {}

    def _s(k):
        v = sc.get(k)
        return f"{v:.1f}%" if isinstance(v, (int, float)) else "n/a"

    st = structure or {}
    native_primitives = st.get(
        "native_primitives", st.get("circles", 0))
    native_circles = st.get(
        "native_circles", st.get("circles", native_primitives))
    native_rectangles = st.get("native_rectangles", 0)
    native_ellipses = st.get("native_ellipses", 0)
    native_lines = st.get("native_lines", 0)
    native_polylines = st.get("native_polylines", 0)
    native_polygons = st.get("native_polygons", 0)
    native_parts = [
        (native_circles, "circles"),
        (native_rectangles, "rectangles"),
        (native_ellipses, "ellipses"),
        (native_lines, "lines"),
        (native_polylines, "polylines"),
        (native_polygons, "polygons"),
    ]
    native_detail = ", ".join(
        f"{int(count or 0)} {label}" for count, label in native_parts
        if int(count or 0)
    ) or "0 objects"
    st_line = (
        f"{st.get('paths', 0)} paths · {native_primitives} native primitives "
        f"({native_detail}) · "
        f"{st.get('strokes', 0)} strokes · "
        f"{st.get('gradients', 0)} gradients · "
        f"{st.get('designer_anchors_total', st.get('nodes', 0))} designer anchors")
    rejected = (acceptance_status == "rejected"
                or visual_acceptance_status == "rejected")
    manual_review = (manual_review_required
                     or acceptance_status != "accepted")
    gate_class = ("rejected" if rejected
                  else "manual" if manual_review else "accepted")
    if rejected:
        if designer_readiness_status == "manual_rework_required":
            gate_text = (
                "設計師品質未達標：部分結構仍需人工修整。"
                "可匯出接手草稿，請先查看下方的具體原因")
        else:
            gate_text = ("品質或可編輯性檢查未達標。可匯出接手草稿；"
                         "請查看熱區，判斷保留修整或局部重畫")
    elif manual_review:
        visual_text = ("通過" if visual_acceptance_status == "accepted"
                       else "需檢查")
        edit_text = ("通過" if editability_status == "accepted"
                     else "需檢查")
        designer_text = ("通過" if designer_readiness_status == "designer_ready"
                         else "需檢查")
        gate_text = (
            f"需人工確認：外觀 {visual_text}；可編輯性 {edit_text}；"
            f"漸層／曲線品質 {designer_text}。請勿只看總分直接交付")
    else:
        gate_text = "accepted：外觀與可編輯性均通過自動品質閘門"
    detail = detail_grid or {}
    p10 = detail.get("p10_score_percent")
    detail_text = (f" · 局部細節 p10 {p10:.1f}%"
                   if isinstance(p10, (int, float)) else "")
    edit_score_text = (f" · 描點收尾 {editability_score:.1f}/100"
                       if isinstance(editability_score, (int, float)) else "")
    automation_text = (
        f" · 自動化準備 {automation_readiness_score:.1f}/100"
        if isinstance(automation_readiness_score, (int, float)) else "")
    human_text = (" · 真人實作未驗"
                  if human_validation_status != "performed" else "")
    designer_text = (
        f" · 設計師品質 {designer_readiness_status}"
        f"（漸層 {gradient_object_gate_status}／曲線 {curve_economy_gate_status}）")
    page = REVIEW_TEMPLATE
    page = page.replace("__TITLE__", html.escape(name))
    page = page.replace(
        "__SCORES__",
        f"source {_s('source')} · foreground {_s('foreground')} · "
        f"flat {_s('flat')}{detail_text}{edit_score_text}"
        f"{automation_text}{designer_text}{human_text}")
    page = page.replace("__STRUCT__", st_line)
    page = page.replace("__TOOL_VERSION__", html.escape(TOOL_VERSION))
    page = page.replace("__GATE_CLASS__", gate_class)
    page = page.replace("__GATE_TEXT__", html.escape(gate_text))
    recolor_link = (
        f'<a class="actionlink" href="{html.escape(recolor_filename, quote=True)}" '
        f'target="_blank">全域換色</a>' if recolor_filename else "")
    page = page.replace("__RECOLOR_LINK__", recolor_link)
    page = page.replace("__IMGURL__", data_url(original_png))
    page = page.replace("__VIEWBOX__", _json.dumps(vb))
    page = page.replace("__HOTSPOTS__", _json.dumps(hotspots or []))
    page = page.replace("__SVGBODY__", svg_inline)
    p2 = out / "review.html"
    p2.write_text(page, encoding="utf-8")
    return p2


REVIEW_TEMPLATE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ Review</title>
<style>
 *{box-sizing:border-box} html,body{margin:0;height:100%}
 body{display:grid;grid-template-columns:320px 1fr;font-family:system-ui,'Microsoft JhengHei',sans-serif;font-size:13px;color:#222}
 #side{overflow:auto;border-right:1px solid #ccc;background:#fafafa;padding:12px}
 #side h3{margin:0 0 4px;font-size:15px;word-break:break-all}
 .meta{color:#666;margin:2px 0 10px;line-height:1.5}
 .row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:6px 0}
 button{font:inherit;padding:3px 9px;border:1px solid #bbb;background:#fff;border-radius:5px;cursor:pointer}
 button:hover{background:#eef} button.on{background:#1a73e8;color:#fff;border-color:#1a73e8}
 .actionlink{display:inline-block;padding:4px 10px;border-radius:5px;background:#1769d3;color:#fff;text-decoration:none;font-weight:700}
 input[type=range]{width:110px}
 h4{margin:14px 0 6px;font-size:13px;border-top:1px solid #ddd;padding-top:10px}
 #spots li{cursor:pointer;padding:2px 4px;border-radius:4px;margin:1px 0}
 #spots li:hover,#spots li.sel{background:#ffe9c7}
 .sev{display:inline-block;width:38px;font-weight:600}
 .sev.hi{color:#c62828}.sev.mid{color:#e65100}.sev.lo{color:#b58900}
 #tree .layer{margin:3px 0}
 #tree .lh{display:flex;align-items:center;gap:6px;font-weight:600;cursor:pointer;padding:2px 4px;border-radius:4px}
 #tree .lh:hover{background:#e8f0fe}
 #tree .sw{width:12px;height:12px;border-radius:3px;border:1px solid #999;display:inline-block}
 #tree ul{list-style:none;margin:0 0 0 22px;padding:0;display:none}
 #tree .open>ul{display:block}
 #tree li{cursor:pointer;padding:1px 4px;border-radius:4px;color:#444}
 #tree li:hover,#tree li.sel{background:#e8f0fe}
 #stagewrap{position:relative;overflow:hidden;background:#e8e8e8;cursor:grab;touch-action:none}
 #stagewrap.checker{background-image:linear-gradient(45deg,#ddd 25%,transparent 25%),linear-gradient(-45deg,#ddd 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ddd 75%),linear-gradient(-45deg,transparent 75%,#ddd 75%);background-size:24px 24px;background-position:0 0,0 12px,12px -12px,-12px 0;background-color:#fff}
 #stagewrap.white{background:#fff}#stagewrap.black{background:#111}
 #stagewrap.grabbing{cursor:grabbing}
 #stage{position:absolute;transform-origin:0 0}
 #stage>img{position:absolute;left:0;top:0;width:100%;height:100%;opacity:.5;pointer-events:none}
 #stage>svg{position:absolute;left:0;top:0;width:100%;height:100%;display:block}
 #zl{min-width:52px;text-align:center;font-weight:600}
 .note{color:#888;font-size:12px;line-height:1.5;margin-top:10px}
 .gate{margin:8px 0 10px;padding:7px 9px;border-radius:6px;font-weight:700}
 .gate.accepted{background:#e8f5e9;color:#1b5e20;border:1px solid #a5d6a7}
 .gate.manual{background:#fff3e0;color:#b71c1c;border:2px solid #e65100}
 .gate.rejected{background:#ffebee;color:#8e0000;border:3px solid #b71c1c}
 @keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
 .hlrect{animation:blink .5s 3}
</style></head><body>
<div id="side">
 <h3>__TITLE__</h3>
 <div class="meta">AI Vector Cleanroom __TOOL_VERSION__<br>__SCORES__<br>__STRUCT__</div>
 <div class="gate __GATE_CLASS__">__GATE_TEXT__</div>
 <div class="row"><button id="fit">符合視窗</button><span id="zl">100%</span>__RECOLOR_LINK__</div>
 <div class="row" id="zooms"></div>
 <div class="row">原圖 <input id="o" type="range" min="0" max="100" value="50">
      向量 <input id="v" type="range" min="0" max="100" value="100"></div>
 <div class="row">背景 <button data-bg="checker" class="on">棋盤</button><button data-bg="white">白</button><button data-bg="black">黑</button></div>
 <h4>問題熱區 Hotspots (<span id="nspots"></span>) <label style="font-weight:400"><input type="checkbox" id="showspots" checked> 顯示標記</label></h4>
 <ol id="spots"></ol>
 <h4>物件清單 Objects</h4>
 <div id="tree"></div>
 <div class="note">滾輪縮放（游標為中心）、拖曳平移。點物件清單可定位並閃爍該物件；勾選方塊可隱藏整層。熱區=與原圖差異最大的區塊，點擊自動放大檢視。</div>
</div>
<div id="stagewrap" class="checker"><div id="stage">
 <img id="orig" src="__IMGURL__" alt="source">
 __SVGBODY__
</div></div>
<script>
const VB=__VIEWBOX__, HOTSPOTS=__HOTSPOTS__;
const wrap=document.getElementById('stagewrap'), stage=document.getElementById('stage');
stage.style.width=VB[0]+'px'; stage.style.height=VB[1]+'px';
const svg=stage.querySelector('svg');
const NS='http://www.w3.org/2000/svg';
const hl=document.createElementNS(NS,'g'); hl.setAttribute('id','_hl'); svg.appendChild(hl);
let zoom=1,px=0,py=0;
function apply(){stage.style.transform=`translate(${px}px,${py}px) scale(${zoom})`;
 document.getElementById('zl').textContent=Math.round(zoom*100)+'%';}
function fit(){const r=wrap.getBoundingClientRect();
 zoom=Math.min(r.width/VB[0],r.height/VB[1])*0.95;
 px=(r.width-VB[0]*zoom)/2; py=(r.height-VB[1]*zoom)/2; apply();}
function setZoom(z,cx,cy){const r=wrap.getBoundingClientRect();
 cx=cx??r.width/2; cy=cy??r.height/2;
 const gx=(cx-px)/zoom, gy=(cy-py)/zoom;
 zoom=Math.max(0.05,Math.min(16,z));
 px=cx-gx*zoom; py=cy-gy*zoom; apply();}
const zr=document.getElementById('zooms');
[1,2,4,8,16].forEach(z=>{const b=document.createElement('button');
 b.textContent=(z*100)+'%'; b.onclick=()=>setZoom(z); zr.appendChild(b);});
document.getElementById('fit').onclick=fit;
wrap.addEventListener('wheel',e=>{e.preventDefault();
 const r=wrap.getBoundingClientRect();
 setZoom(zoom*(e.deltaY<0?1.2:1/1.2),e.clientX-r.left,e.clientY-r.top);},{passive:false});
let drag=null;
wrap.addEventListener('pointerdown',e=>{drag={x:e.clientX-px,y:e.clientY-py};
 wrap.classList.add('grabbing');wrap.setPointerCapture(e.pointerId);});
wrap.addEventListener('pointermove',e=>{if(!drag)return;
 px=e.clientX-drag.x; py=e.clientY-drag.y; apply();});
wrap.addEventListener('pointerup',()=>{drag=null;wrap.classList.remove('grabbing');});
document.getElementById('o').oninput=e=>document.getElementById('orig').style.opacity=e.target.value/100;
document.getElementById('v').oninput=e=>{svg.style.opacity=e.target.value/100;};
document.querySelectorAll('[data-bg]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('[data-bg]').forEach(x=>x.classList.remove('on'));
 b.classList.add('on'); wrap.className=b.dataset.bg;});
// ---------- object tree ----------
const tree=document.getElementById('tree');
let selRow=null;
function flash(el){ [...hl.querySelectorAll('.hlrect')].forEach(r=>r.remove());
 try{const b=el.getBBox();
  const r=document.createElementNS(NS,'rect');
  r.setAttribute('x',b.x-2);r.setAttribute('y',b.y-2);
  r.setAttribute('width',b.width+4);r.setAttribute('height',b.height+4);
  r.setAttribute('fill','none');r.setAttribute('stroke','#e91e63');
  r.setAttribute('stroke-width','2');r.setAttribute('vector-effect','non-scaling-stroke');
  r.classList.add('hlrect'); hl.appendChild(r);}catch(e){}}
const INK='http://www.inkscape.org/namespaces/inkscape';
 function addDrawable(el,ul,i){
  const li=document.createElement('li');let d='';
  const tag=el.localName;
  if(tag==='path'){const n=(el.getAttribute('d')||'').match(/[A-Za-z]/g);
   d='path · '+(n?n.length:0)+' nodes';
   if(el.getAttribute('stroke-width'))d='stroke · w'+el.getAttribute('stroke-width');}
  else if(tag==='circle')d='circle · r'+Math.round(+el.getAttribute('r'));
  else d=tag;
 const oid=el.id?(' · '+el.id):'';li.textContent='#'+(i+1)+' '+d+oid;
 li.onclick=()=>{if(selRow)selRow.classList.remove('sel');selRow=li;li.classList.add('sel');flash(el);};
 ul.appendChild(li);
}
function addGroup(g,parent,depth=0){
 if(g.id==='_hl')return;
 const div=document.createElement('div');div.className='layer'+(depth===0?' open':'');
 const head=document.createElement('div');head.className='lh';
 const cb=document.createElement('input');cb.type='checkbox';cb.checked=true;
 cb.onclick=e=>{e.stopPropagation();g.style.display=cb.checked?'':'none';};
 const sw=document.createElement('span');sw.className='sw';
 const sample=g.querySelector('[fill],[stroke]');const f=g.getAttribute('fill')||(sample&&sample.getAttribute('fill'));
 sw.style.background=(f&&f.startsWith('url'))?'linear-gradient(45deg,#ff0,#0c0)':(f&&f!=='none'?f:'#777');
 const lab=document.createElement('span');
 const label=g.getAttributeNS(INK,'label')||g.getAttribute('inkscape:label')||g.id||'group';
 const count=g.querySelectorAll('path,circle,rect,ellipse,line,polyline,polygon').length;
 lab.textContent=label+' ('+count+')';head.append(cb,sw,lab);
 head.onclick=()=>div.classList.toggle('open');
 const ul=document.createElement('ul');let item=0;
 [...g.children].forEach(el=>{if(el.localName==='g')addGroup(el,ul,depth+1);
   else if(['path','circle','rect','ellipse','line','polyline','polygon'].includes(el.localName))addDrawable(el,ul,item++);});
 div.append(head,ul);parent.appendChild(div);
}
[...svg.children].forEach(g=>{if(g.localName==='g')addGroup(g,tree,0);});
// ---------- hotspots ----------
const spotsOl=document.getElementById('spots');
document.getElementById('nspots').textContent=HOTSPOTS.length;
const markers=[];
HOTSPOTS.forEach((s,i)=>{
 const r=document.createElementNS(NS,'rect');
 r.setAttribute('x',s.x);r.setAttribute('y',s.y);
 r.setAttribute('width',s.w);r.setAttribute('height',s.h);
 r.setAttribute('fill','none');
 r.setAttribute('stroke',s.severity>=0.5?'#e53935':s.severity>=0.35?'#fb8c00':'#c9a800');
 r.setAttribute('stroke-width','1.6');r.setAttribute('vector-effect','non-scaling-stroke');
 hl.appendChild(r); markers.push(r);
 const li=document.createElement('li');
 const cls=s.severity>=0.5?'hi':s.severity>=0.35?'mid':'lo';
 const badge=document.createElement('span');badge.className='sev '+cls;
 badge.textContent=s.structural_defect?'檢查':Math.round(s.severity*100)+'%';
 li.append(badge,document.createTextNode(' '+(s.label||('區塊 '+(i+1)))));
 li.onclick=()=>{[...spotsOl.children].forEach(x=>x.classList.remove('sel'));
  li.classList.add('sel');
  const r2=wrap.getBoundingClientRect();
  const target=Math.min(8,Math.max(2,0.35*Math.min(r2.width/s.w,r2.height/s.h)));
  zoom=target;
  px=r2.width/2-(s.x+s.w/2)*zoom; py=r2.height/2-(s.y+s.h/2)*zoom; apply();};
 spotsOl.appendChild(li);});
document.getElementById('showspots').onchange=e=>markers.forEach(m=>m.style.display=e.target.checked?'':'none');
fit(); window.addEventListener('resize',fit);
</script>
</body></html>"""


def _native_primitive_counts(stats):
    """Return truthful native primitive counts without changing engine stats.

    ``n_native`` is the engine's legacy native-circle counter. Rebuilt
    rectangles are identified by ``stroke_info`` and added to the aggregate;
    keeping this translation here avoids changing candidate ranking in a
    release-polish pass.
    """
    details = getattr(stats, "stroke_info", ()) or ()
    rectangles = sum(item.get("primitive") == "rect" for item in details)
    circles = max(0, int(getattr(stats, "n_native", 0)))
    return {
        "native_primitives": circles + rectangles,
        "native_circles": circles,
        "native_rectangles": rectangles,
    }


def _final_stroke_details(svg_path: Path, stats):
    """Return details for stroke-N elements that still exist in the final DOM.

    Post-processing can merge several rebuilt strokes into a native annulus.
    Reporting the tracer's original list after that transaction would make the
    stroke count and details disagree.  IDs are deliberately stable, so the
    final SVG can be joined back to the original diagnostic record safely.
    """
    from xml.etree import ElementTree as ET

    source = list(getattr(stats, "stroke_info", ()) or ())
    found = []
    annuli = []
    for element in ET.parse(svg_path).getroot().iter():
        element_id = element.get("id", "")
        match = re.fullmatch(r"stroke-(\d+)", element_id)
        if not match:
            if element_id.startswith("annulus-"):
                try:
                    width = float(element.get("stroke-width", "0"))
                    opacity = float(element.get("stroke-opacity", "1"))
                except ValueError:
                    width, opacity = 0.0, 1.0
                annuli.append({
                    "id": element_id,
                    "element": element.tag.rsplit("}", 1)[-1],
                    "color": element.get("stroke", ""),
                    "width": width,
                    "closed": True,
                    "nodes": 1,
                    "opacity": opacity,
                    "primitive": "circle",
                    "representation": "native_circle_with_dasharray",
                    "merged_from": [item for item in
                                    element.get("data-merged-from", "").split(",")
                                    if item],
                })
            continue
        source_index = int(match.group(1)) - 1
        detail = (dict(source[source_index])
                  if 0 <= source_index < len(source) else {})
        detail["id"] = element.get("id")
        detail["element"] = element.tag.rsplit("}", 1)[-1]
        found.append((source_index, detail))
    return ([item for _index, item in sorted(found, key=lambda pair: pair[0])]
            + sorted(annuli, key=lambda item: item["id"]))


def _rolled_back_stage_report(stage_name, stage_report):
    """Describe a stage whose proposal was not committed to the final SVG."""
    rolled = dict(stage_report)
    rolled["attempted_status"] = rolled.get("status")
    rolled["status"] = "rolled_back_final_source_guard"
    if stage_name == "annulus":
        rolled["applied_candidates"] = 0
        rolled["committed_candidates"] = []
    elif stage_name == "exact_native_shapes":
        rolled.update({
            "committed": False,
            "committed_candidate_count": 0,
            "committed_line_count": 0,
            "committed_polyline_count": 0,
        })
    elif stage_name == "compound_paths":
        rolled.update({
            "output_paths": rolled.get("input_paths", 0),
            "output_subpaths": rolled.get("input_subpaths", 0),
            "source_paths_split": 0,
            "split_paths": 0,
            "new_paths_added": 0,
            "selectable_path_delta": 0,
            "subpaths_redistributed": 0,
            "source_paths_simplified": 0,
            "linear_cubics_simplified": 0,
            "path_data_bytes_saved": 0,
            "simplified_paths": [],
            "paths": [],
        })
    elif stage_name == "scene_graph":
        rolled.update({
            "object_group_count": 0,
            "actual_dom_group_count": 0,
            "manifest_only_group_count": 0,
            "grouped_drawables": 0,
            "ungrouped_drawables": rolled.get("drawable_count", 0),
            "actual_dom_groups": [],
            "manifest_only_groups": [],
            "groups": [],
            "skipped_unsafe_groups": [],
        })
    return rolled


def _paint_resource_summary(stats):
    """Separate stack layers from the unique paints designers can recolor."""
    layers = []
    solids = {}
    gradients = list(getattr(stats, "gradient_info", ()) or ())

    # A gradient paint can occur in more than one non-contiguous stack run.
    # ``stats.palette`` records every run, while ``gradient_info`` records the
    # unique SVG resources.  Resolve a run through the real middle stop used
    # by clean_base for its presentation colour; never let an extra gradient
    # run fall through and masquerade as a solid paint.
    gradients_by_palette_hex = {}
    for gradient in gradients:
        stops = list(gradient.get("stops", ()) or ())
        if not stops:
            continue
        try:
            middle = min(
                stops,
                key=lambda stop: abs(float(stop.get("offset", 0.0)) - 0.5),
            )
        except (AttributeError, TypeError, ValueError):
            continue
        hx = str(middle.get("color", "")).lower()
        if re.fullmatch(r"#[0-9a-f]{6}", hx):
            gradients_by_palette_hex.setdefault(hx, []).append(gradient)
    assigned_gradient_ids = set()

    for name, value in (getattr(stats, "palette", ()) or ()):
        if str(name).lower().startswith("gradient"):
            hx = str(value).lower()
            candidates = gradients_by_palette_hex.get(hx, ())
            gradient = next(
                (item for item in candidates
                 if item.get("id", "") not in assigned_gradient_ids),
                candidates[0] if candidates else None,
            )
            if gradient is None:
                gradient = next(
                    (item for item in gradients
                     if item.get("id", "") not in assigned_gradient_ids),
                    None,
                )
            gradient_id = gradient.get("id", "") if gradient else ""
            if gradient_id:
                assigned_gradient_ids.add(gradient_id)
            gradient_type = str(
                (gradient or {}).get("type")
                or ((gradient or {}).get("model") or {}).get("type")
                or "linear")
            resource_type = (
                "radialGradient" if gradient_type == "radial"
                else "linearGradient")
            layers.append({"name": name, "type": resource_type,
                           "gradient_id": gradient_id})
            continue
        hx = str(value).lower()
        layers.append({"name": name, "type": "solid", "hex": hx})
        solids.setdefault(hx, name)

    # Stroke paints are not necessarily present in the recalculated fill
    # palette. Include their canonical colors in the actual paint-resource
    # count so the report describes what an SVG editor will expose.
    for detail in (getattr(stats, "stroke_info", ()) or ()):
        hx = str(detail.get("color", "")).lower()
        if re.fullmatch(r"#[0-9a-f]{6}", hx):
            solids.setdefault(hx, f"stroke-{len(solids) + 1}")

    solid_resources = [
        {"name": name, "type": "solid", "hex": hx}
        for hx, name in solids.items()
    ]
    gradient_resources = []
    for gradient in gradients:
        gradient_type = str(
            gradient.get("type")
            or (gradient.get("model") or {}).get("type") or "linear")
        gradient_resources.append({
            "name": gradient.get("id", "gradient"),
            "type": ("radialGradient" if gradient_type == "radial"
                     else "linearGradient"),
            "id": gradient.get("id", ""),
            "model": gradient.get("model") or {},
            "stops": list(gradient.get("stops", ())),
        })
    return {
        "layers": layers,
        "palette": solid_resources,
        "paint_resources": solid_resources + gradient_resources,
        "solid_paints": len(solid_resources),
        "gradient_paints": len(gradient_resources),
        "unique_paints_total": len(solid_resources) + len(gradient_resources),
    }


def _options_summary(options):
    labels = {
        "background": "背景", "strokes": "筆畫", "gradients": "漸層",
        "geometry": "幾何", "colors": "色數", "white_threshold": "白色閾值",
        "max_size": "處理上限", "curve_error_percent": "曲線誤差率%",
    }
    options = options or {}
    ordered = [key for key in labels if key in options]
    ordered.extend(key for key in options if key not in labels)
    return "；".join(
        f"{labels.get(key, key)}={options[key]}" for key in ordered) or "（無）"


def make_output_readme(out: Path, name: str, palette, geometry_notes=None,
                       preview_is_fallback=False, scores=None,
                       acceptance_status="accepted", requested_options=None,
                       effective_options=None, auto_fallback=None,
                       visual_acceptance_status="accepted",
                       editability_status="accepted", editability_score=None,
                       automation_readiness_score=None,
                       human_validation_status="not_performed",
                       editability_reasons=None, detail_grid=None,
                       enhancement_report=None, paint_role_report=None,
                       paint_manifest_name=None, recolor_filename=None,
                       designer_operations=None, final_structure=None):
    pal_lines = "\n".join(f"    {nm}: {hx}" for nm, hx in palette)
    geo_block = ""
    if geometry_notes:
        geo_lines = "\n".join(f"  - {g}" for g in geometry_notes)
        geo_block = f"""
Geometry regularization applied
-------------------------------
{geo_lines}
  Review the SVG visually; geometry detection is heuristic and may need manual correction.
"""
    if preview_is_fallback:
        preview_line = (f"  {name}_preview.png     WARNING: NOT rendered from the SVG.\n"
                        "                         SVG rendering packages were missing, so this\n"
                        "                         is the cleaned source image. Open review.html\n"
                        "                         or the SVG itself to inspect the real result.")
    else:
        preview_line = f"  {name}_preview.png     Preview rendered from the SVG."
    score_block = ""
    if scores and any(scores.get(k) is not None for k in ("flat", "source", "foreground")):
        def _s(k):
            return f"{scores[k]:.1f}%" if scores.get(k) is not None else "n/a"
        grid = detail_grid or {}
        p10 = grid.get("p10_score_percent")
        p10_line = (f"\n  local detail p10 {p10:.1f}%   weakest 10% of source-ink grid cells"
                    if isinstance(p10, (int, float)) else "")
        edit_line = (f"\n  editability      {editability_score:.1f}/100   structural heuristic; not a time-saving claim"
                     if isinstance(editability_score, (int, float)) else "")
        automation_line = (
            f"\n  automation ready {automation_readiness_score:.1f}/100   generic SVG handles; not human task acceptance"
            if isinstance(automation_readiness_score, (int, float)) else "")
        human_line = ("\n  human validation NOT PERFORMED   run Stage 2 timed editing before any labour-saving claim"
                      if human_validation_status != "performed" else "")
        score_block = f"""
Self-check scores
-----------------
  flat match       {_s('flat')}   fidelity to the palette-flattened tracing input
  source match     {_s('source')}   whole-canvas similarity to the cleaned source
  foreground match {_s('foreground')}   similarity on the adaptive source-ink ROI;
                            catches small foreground details that whole-canvas
                            scores would miss{p10_line}{edit_line}{automation_line}{human_line}
"""
    foreground = (f"{scores['foreground']:.1f}%"
                  if scores and isinstance(scores.get("foreground"), (int, float))
                  else "n/a")
    if acceptance_status == "rejected":
        acceptance_line = "rejected：品質或可編輯性未達標，可作接手草稿，尚需人工修整"
        acceptance_warning = (
            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "品質或可編輯性檢查未達標；請看下方各項結果與具體原因。\n"
            "可由設計師接手頁匯出 working.svg，保留可用部分並修整或補畫。\n"
            "這是可編輯草稿，尚未完成設計驗收。\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif acceptance_status == "manual_review":
        acceptance_line = "manual_review"
        visual_line = ("accepted" if visual_acceptance_status == "accepted"
                       else "manual_review")
        edit_line = ("accepted" if editability_status == "accepted"
                     else "manual_review")
        acceptance_warning = (
            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            f"需人工確認：外觀={visual_line}；可編輯性={edit_line}。\n"
            "高視覺分不代表物件已妥善分組或節點容易修改。\n"
            "請先開啟 review.html 疊圖與物件清單，再決定是否接手修整。\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        acceptance_line = "accepted：外觀與可編輯性均通過自動品質閘門"
        acceptance_warning = ""
    fallback_line = (_options_summary(auto_fallback)
                     if auto_fallback else "未啟用")
    edit_reason_block = ""
    if editability_status != "accepted" and editability_reasons:
        edit_reason_block = "\n可編輯性警示：\n" + "\n".join(
            f"  - {reason}" for reason in list(editability_reasons)[:6]) + "\n"

    stages = (enhancement_report or {}).get("stages", {})
    annulus = stages.get("annulus", {})
    exact_native = stages.get("exact_native_shapes", {})
    compound = stages.get("compound_paths", {})
    scene = stages.get("scene_graph", {})
    role_counts = (paint_role_report or {}).get("resource_counts", {})
    ops = (designer_operations or {}).get("summary", {})
    enhancement_block = f"""
Fidelity, topology and editability enhancements
---------------------------------------------
  - Native annulus replacements: {int(annulus.get('applied_candidates', 0) or 0)}
  - Pixel-exact path-to-native conversions: {int(exact_native.get('committed_candidate_count', 0) or 0)} ({int(exact_native.get('committed_line_count', 0) or 0)} lines, {int(exact_native.get('committed_polyline_count', 0) or 0)} polylines)
  - Additional independently selectable paths: +{int(compound.get('selectable_path_delta', 0) or 0)}
  - Exact collinear Beziers simplified to lines: {int(compound.get('linear_cubics_simplified', 0) or 0)} ({int(compound.get('path_data_bytes_saved', 0) or 0)} path-data bytes removed)
  - Actual SVG object groups: {int(scene.get('actual_dom_group_count', 0) or 0)}
  - Unsafe groups kept as manifest-only notes: {int(scene.get('manifest_only_group_count', 0) or 0)}
  - Global paint-role controls: {int(role_counts.get('role_controls', 0) or 0)}
  - Generic machine-detectable structural handles passed: {int(ops.get('passed', 0) or 0)}/{int(ops.get('total_operations', 5) or 5)} (not human task acceptance)

These are conservative structural improvements. They do not recover the
original font/layer file and do not prove an 80% time saving; timed designer
editing remains the final acceptance test.
"""
    optional_files = ""
    if paint_manifest_name:
        optional_files += (f"\n  {paint_manifest_name:<28} Paint-role manifest for repeatable global recolouring.")
    if recolor_filename:
        optional_files += (f"\n  {recolor_filename:<28} Offline colour-role editor; exports an explicit-paint SVG.")
    structure = final_structure or {}
    final_structure_line = (
        f"  Final DOM: {int(structure.get('paths', 0) or 0)} paths, "
        f"{int(structure.get('native_primitives', 0) or 0)} native primitives, "
        f"{int(structure.get('groups', 0) or 0)} groups, "
        f"{int(structure.get('designer_anchors_total', structure.get('nodes', structure.get('nodes_total', 0))) or 0)} designer anchors.\n"
        if structure else "")
    summary = f"""AI 向量清稿工具｜本次輸出摘要
==================================
工具版本：{TOOL_VERSION}
驗收狀態：{acceptance_line}
{acceptance_warning}
整體墨水／前景符合度：{foreground}
外觀閘門：{visual_acceptance_status}
可編輯性閘門：{editability_status}{f'（{editability_score:.1f}/100）' if isinstance(editability_score, (int, float)) else ''}
請求設定：{_options_summary(requested_options)}
實際設定：{_options_summary(effective_options)}
自動回退：{fallback_line}
{edit_reason_block}

"""
    note = summary + f"""{name} - Vector Cleanroom Output
=================================

This folder contains an editable vector approximation generated from a bitmap
source image. It is intended as a clean starting point for further review,
editing, and production cleanup.

Files
-----
  {name}_vector.svg      Editable SVG vector paths. Open with Illustrator,
                         Inkscape, Affinity Designer, Figma, etc.
{preview_line}
  source_reference.png   Source reference after background cleanup.
  review.html            Browser-based overlay page for visual comparison.
  report.json            Machine-readable run report.
  OUTPUT_README.txt      This file.{optional_files}

What the tool did
-----------------
  - Converted the image into SVG paths; no bitmap is embedded in the SVG.
  - Reduced noisy antialiasing while retaining the solid and gradient paint
    resources listed below.
  - Preserved visible stack order so shapes that sit on top remain on top.
  - Recorded original paint/layer attribution. Scene reconstruction may
    flatten wrapper groups, then adds only spatial object groups that pass
    ordering invariants and exact validation-resolution raster checks.
  - Assigned stable object IDs and kept unsafe group proposals as report-only
    manifests instead of changing SVG stacking order.
{final_structure_line}{geo_block}{score_block}{enhancement_block}
Limitations
-----------
  - Bitmap images do not contain original vector curves, font data, or layer
    structure, so this is an approximation rather than lossless recovery.
  - Text is converted to outline paths, not editable font text.
  - Highly detailed photos, soft shadows, and complex gradients are not the
    target use case; flat logos and graphic marks work best.

Actual paint resources
----------------------
{pal_lines}
"""
    p = out / "OUTPUT_README.txt"
    p.write_text(note, encoding="utf-8")
    return p


def process_one(img_path: Path, out_base: str, args, output_dir: Path,
                progress=None, control=None):
    from PIL import Image

    from trace_engine import _prepare_image
    from clean_base import (
        build_clean_base,
        gradient_stage_cache_audit,
        new_gradient_stage_cache,
        new_pre_gradient_state_cache,
        pre_gradient_state_cache_audit,
    )
    from svg_postprocess import atomic_replace_bytes

    def _emit(stage, detail="", **extra):
        if control is not None:
            control.checkpoint(stage)
        if progress is None:
            return
        event = {"stage": str(stage), "detail": str(detail)}
        if control is not None and hasattr(control, "snapshot"):
            event.update(control.snapshot())
            event["stage"] = str(stage)
        event.update(extra)
        progress(event)

    _emit("prepare_source", "正在準備來源影像")
    out_base = bounded_output_base(out_base)
    ensure_output_path_budget(output_dir, out_base)
    warnings = []
    # Share expensive source-space analysis only among this image's serial
    # candidate builds.  clean_base hashes every effective stage input.
    gradient_stage_cache = new_gradient_stage_cache()
    pre_gradient_state_cache = new_pre_gradient_state_cache()
    print(f"\n[ {img_path.name} ]")
    deliver = output_dir / f"result_{out_base}"
    zip_path = output_dir / f"result_{out_base}.zip"
    # Remove BOTH previous outputs up front: a stale zip surviving a failed
    # re-run would masquerade as a fresh successful result.
    if deliver.exists():
        shutil.rmtree(deliver)
    zip_path.unlink(missing_ok=True)
    deliver.mkdir(parents=True)

    requested_options = {
        "strokes": args.strokes,
        "gradients": args.gradients,
        "geometry": args.geometry,
        "background": args.background,
        "colors": args.colors,
        "white_threshold": args.white_threshold,
        "max_size": args.max_size,
        "curve_error_percent": args.curve_error_percent,
    }

    # 1) Source reference for review.  Candidate validation must use the same
    # background mode as that candidate: scoring an auto-cleaned logo against
    # the deliberately retained AI paper glow mistakes removable background
    # texture for lost vector detail.  Low-contrast enclosed marks still remain
    # in the auto-cleaned reference and have their ordinary source-ink score.
    # Establish provenance from actual original pixels. Never inherit a source
    # file's self-declared metadata: an input may contain arbitrary PNG tags.
    with Image.open(img_path) as original_image:
        original_rgba = original_image.convert("RGBA")
        reference_alpha_origin = (
            "native" if original_rgba.getchannel("A").getextrema()[0] < 255
            else "opaque_canvas_derived")
        original_rgba.save(deliver / "source_original.png")

    def _save_reference(image, path):
        from PIL.PngImagePlugin import PngInfo
        metadata = PngInfo()
        metadata.add_text("avc_reference_alpha_origin", reference_alpha_origin)
        metadata.add_text("avc_reference_alpha_provenance", "original_pixel_alpha_extrema")
        image.save(path, pnginfo=metadata)

    clean_img, _sz, removed = _prepare_image(
        img_path, max_size=0, background=args.background,
        white_threshold=args.white_threshold, alpha_threshold=12)
    ref_png = deliver / "source_reference.png"
    _save_reference(clean_img, ref_png)
    msg = " (outer light/checker background removed)" if removed else ""
    print(f"  Source reference OK{msg}")
    metric_refs = {}

    def _metric_reference(options, hole_mask=None):
        mode = options["background"]
        threshold = int(options["white_threshold"])
        base_key = ("base", mode, threshold)
        if base_key not in metric_refs:
            safe_mode = re.sub(r"[^a-z0-9_-]+", "_", mode.lower())
            path = deliver / f"_metric_reference_{safe_mode}_{threshold}.png"
            metric_img, _metric_sz, _metric_removed = _prepare_image(
                img_path, max_size=0, background=mode,
                white_threshold=threshold, alpha_threshold=2)
            _save_reference(metric_img, path)
            metric_refs[base_key] = path
        base_path = metric_refs[base_key]

        # The circle/rectangle hole guard runs after stroke extraction, so the
        # raw prepared reference cannot know about its decision.  Reuse that
        # exact mask for candidate scoring instead of reclassifying light
        # pixels here.  This keeps white letters/emblems opaque while removing
        # only the broad canvas-coloured pocket the SVG itself omitted.
        if hole_mask is None:
            return base_path
        import hashlib
        import numpy as np
        from PIL import Image
        mask = np.asarray(hole_mask, dtype=bool)
        if mask.ndim != 2 or not mask.any():
            return base_path
        packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
        digest = hashlib.sha256(
            f"{mask.shape[0]}x{mask.shape[1]}:".encode("ascii") + packed
        ).hexdigest()[:16]
        hole_key = ("holes", mode, threshold, digest)
        if hole_key not in metric_refs:
            safe_mode = re.sub(r"[^a-z0-9_-]+", "_", mode.lower())
            path = deliver / (
                f"_metric_reference_{safe_mode}_{threshold}_holes_{digest}.png")
            with Image.open(base_path) as metric_img:
                canonical = _apply_validation_hole_mask(metric_img, mask)
            _save_reference(canonical, path)
            metric_refs[hole_key] = path
        return metric_refs[hole_key]

    metric_ref = _metric_reference(requested_options)
    _emit("prepare_source_complete", "來源參考已建立")

    # 2) Clean vector result. Low-scoring conversions trigger candidate
    # comparison (strokes/gradients/geometry off) and automatic fallback to
    # whichever variant reproduces the source best; a result that stays
    # below the confidence floor is REJECTED, not shipped (review P0-2).
    svg_path = deliver / f"{out_base}_vector.svg"
    flat_chk = deliver / "_flat_check.png"

    def _build(options):
        st = build_clean_base(img_path, svg_path,
                              forced_colors=options["colors"],
                              white_threshold=options["white_threshold"],
                              background=options["background"],
                              max_size=options["max_size"],
                              geometry=options["geometry"],
                              strokes=options["strokes"],
                              gradients=options["gradients"],
                              curve_error_percent=options[
                                  "curve_error_percent"],
                              flat_out=flat_chk,
                              gradient_stage_cache=gradient_stage_cache,
                              pre_gradient_state_cache=(
                                  pre_gradient_state_cache),
                              progress=progress, control=control)
        hole_getter = getattr(st, "_validation_hole_mask", None)
        hole_mask = hole_getter() if callable(hole_getter) else None
        candidate_metric_ref = _metric_reference(options, hole_mask)
        component_before_render = deliver / "_component_repair_before.png"
        sc = self_check(svg_path, flat_chk, candidate_metric_ref,
                        gradient_info=st.gradient_info,
                        keep_render=component_before_render,
                        viewbox=st.viewbox)
        try:
            sc, st.component_repair = _attempt_isolated_component_repair(
                svg_path, flat_chk, candidate_metric_ref, st, sc,
                component_before_render)
        finally:
            component_before_render.unlink(missing_ok=True)
        return st, sc, candidate_metric_ref

    def _eff(sc):
        # Never fall back to whole-canvas similarity: that is precisely how a
        # small soft-alpha mark used to disappear behind a 98% white canvas.
        return sc.get("foreground")

    def _structure(st):
        primitive_counts = _native_primitive_counts(st)
        elements = st.n_paths + st.n_strokes + st.n_native + st.n_gradients
        score = (100.0 - 12.0 * math.log10(1.0 + st.n_nodes)
                 - min(12.0, 0.4 * elements)
                 + min(8.0, 2.0 * st.n_native))
        return {
            "paths": st.n_paths,
            **primitive_counts,
            "strokes": st.n_strokes,
            "gradients": st.n_gradients,
            "nodes": st.n_nodes,
            "score": max(0.0, min(100.0, score)),
        }

    # Every combination of disabling an enabled reconstruction stage is a
    # real candidate. This catches interactions (e.g. strokes+geometry) that
    # single-feature fallbacks miss.
    disable_keys = []
    if args.strokes == "on":
        disable_keys.append("strokes")
    if args.gradients == "on":
        disable_keys.append("gradients")
    if args.geometry != "off":
        disable_keys.append("geometry")
    variants = []
    for count in range(len(disable_keys) + 1):
        for subset in itertools.combinations(disable_keys, count):
            ov = {}
            for key in subset:
                ov[key] = "off"
            variants.append(ov)

    candidate_plan_upper = len(variants) * (
        2 if args.background != "keep" else 1)
    if control is not None and hasattr(control, "set_candidate_plan"):
        control.set_candidate_plan(candidate_plan_upper)
    _emit(
        "candidate_search",
        f"準備評估最多 {candidate_plan_upper} 個必要候選",
        candidate_planned=candidate_plan_upper)

    candidates = []
    internal = []
    attempted = set()

    def _attempt(background_mode, ov):
        opts = dict(requested_options)
        opts.update(ov)
        opts["background"] = background_mode
        signature = tuple(sorted(opts.items()))
        if signature in attempted:
            return
        attempted.add(signature)
        if control is not None:
            control.before_candidate(signature)
        candidate_number = len(candidates) + 1
        _emit(
            "candidate_search",
            f"正在評估候選 {candidate_number}/{candidate_plan_upper}",
            candidate_current=candidate_number,
            candidate_planned=candidate_plan_upper,
            candidate_summary={
                "status": "running",
                "options": {
                    key: opts.get(key) for key in (
                        "background", "strokes", "gradients", "geometry")
                },
            })
        public = {"status": "failed", "options": dict(opts)}
        try:
            st, sc, candidate_metric_ref = _build(opts)
            structure = _structure(st)
            quality = _eff(sc)
            rank = ((0.90 * quality + 0.10 * structure["score"])
                    if quality is not None else -1.0)
            public_scores = {key: value for key, value in sc.items()
                             if key != "hotspots"}
            public.update({"status": "ok", "scores": public_scores,
                           "quality_score": quality,
                           "structure": structure,
                           "component_repair": st.component_repair,
                           "selection_score": rank,
                           "selected": False})
            # Keep the small candidate artifacts in memory. Rebuilding the
            # winner used to repeat the most expensive tracing work after the
            # full matrix had already completed; restoring these exact bytes
            # is both faster and guarantees the delivered SVG is the one that
            # was actually scored.
            svg_snapshot = svg_path.read_bytes()
            flat_snapshot = flat_chk.read_bytes()
            internal.append((rank, quality, opts, st, sc, public,
                             svg_snapshot, flat_snapshot,
                             candidate_metric_ref))
        except ConversionInterrupted:
            raise
        except Exception as exc:
            public["error"] = str(exc)[:240]
        candidates.append(public)
        if control is not None:
            control.after_candidate()
        candidate_summary = {
            "status": public["status"],
            "options": {
                key: public["options"].get(key) for key in (
                    "background", "strokes", "gradients", "geometry")
            },
        }
        if public["status"] == "ok":
            candidate_summary.update({
                "quality_score": public.get("quality_score"),
                "selection_score": public.get("selection_score"),
                "structure_score": public.get("structure", {}).get("score"),
                "visual_gate_status": _evaluate_visual_gate(
                    sc).get("status"),
            })
        elif public.get("error"):
            candidate_summary["error"] = public["error"]
        _emit(
            "candidate_search",
            f"候選 {candidate_number} 已完成",
            candidate_current=candidate_number,
            candidate_planned=candidate_plan_upper,
            candidate_summary=candidate_summary)

    # Staged evaluation keeps visually accepted, structurally low-risk logos at
    # one render.  A structurally risky base still evaluates the independent
    # stage disables: an aggregate visual score can hide an overlap-specific
    # reconstruction fault, and the candidate report must remain auditable.
    _attempt(args.background, {})
    base_item = next((item for item in internal
                      if item[2] == requested_options), None)
    structure_risk = False
    base_quality = None
    base_visual_status = None
    base_structure_concerns = []
    if base_item is not None:
        base_quality = base_item[1]
        base_stats = base_item[3]
        base_visual_status = _evaluate_visual_gate(base_item[4])["status"]
        base_structure_concerns = _candidate_structure_concerns(base_stats, base_item[6])
        structure_risk = bool(base_structure_concerns)
    expand_primary = (base_item is None or base_quality is None
                      or base_quality < 88.0
                      or base_visual_status != "accepted"
                      or structure_risk)
    matrix_strategy = "base_only"
    if expand_primary:
        # First evaluate each reconstruction stage independently.  High-score
        # logos with a merely complex structure do not need every 2-/3-way
        # disable combination when no individual disable buys a material
        # visual gain; requested editing features would win that visual tie
        # anyway. Low-quality/failed bases still receive the exhaustive matrix.
        matrix_strategy = "single_disables"
        single_variants = [ov for ov in variants if len(ov) == 1]
        for ov in single_variants:
            _attempt(args.background, ov)
        # Catastrophic/failed bases still deserve exhaustive rescue.  Above
        # that floor, only combine stages whose single-disable candidate has a
        # measurable visual or structural effect.  Ali-tea used to spend half
        # its six-minute run repeating geometry-on/off renders that differed by
        # 0.002 points; the consequential strokes+gradients pair is retained.
        if base_item is None or base_quality is None or base_quality < 80.0:
            matrix_strategy = "full_disable_matrix"
            for ov in variants:
                if len(ov) >= 2:
                    _attempt(args.background, ov)
        else:
            impactful = []
            base_structure = base_item[5].get("structure", {})
            base_structure_score = float(base_structure.get("score") or 0.0)
            for key in disable_keys:
                wanted = dict(requested_options)
                wanted[key] = "off"
                single = next((item for item in internal
                               if item[2] == wanted), None)
                if single is None or single[1] is None:
                    continue
                structural_gain = (
                    float(single[5].get("structure", {}).get("score") or 0.0)
                    - base_structure_score)
                if (_candidate_safely_dominates(single, base_item)
                        or single[1] >= base_quality + 0.5
                        or structural_gain >= 3.0):
                    impactful.append(key)
            if len(impactful) >= 2:
                matrix_strategy = "impactful_disable_combinations"
                for count in range(2, len(impactful) + 1):
                    for subset in itertools.combinations(impactful, count):
                        _attempt(args.background,
                                 {key: "off" for key in subset})
            else:
                matrix_strategy = "single_disables_pruned_inert_combinations"

    primary_quality = max(
        (item[1] for item in internal if item[1] is not None), default=-1.0)
    # A build-time failure (including an empty light-on-light result) and a
    # sub-80 validation both warrant retrying the complete matrix with the
    # background preserved.
    if args.background != "keep" and primary_quality < 80.0:
        for ov in variants:
            _attempt("keep", ov)

    if not internal:
        errors = "; ".join(c.get("error", "unknown failure")
                           for c in candidates[:4])
        raise RuntimeError(f"all vector candidates failed: {errors}")
    viable = [item for item in internal if item[1] is not None]
    if not viable:
        raise RuntimeError("all vector candidates lacked a foreground quality score")
    selected_item, selection_policy = _select_viable_candidate(
        viable, requested_options)
    _emit("select_candidate", "已選出合格候選，準備後處理")
    selection_policy["matrix_strategy"] = matrix_strategy
    selection_policy["evaluated_candidates"] = len(candidates)
    selection_policy["base_structure_risk"] = bool(structure_risk)
    selection_policy["base_structure_concerns"] = base_structure_concerns
    selection_policy["gradient_stage_cache"] = (
        gradient_stage_cache_audit(gradient_stage_cache))
    selection_policy["pre_gradient_state_cache"] = (
        pre_gradient_state_cache_audit(pre_gradient_state_cache))
    try:
        from compute_backend import accelerator_audit
        selection_policy["compute_backend"] = accelerator_audit()
    except Exception as exc:
        # Reporting an accelerator audit is never allowed to affect the
        # already-scored vector candidate or its CPU fallback.
        selection_policy["compute_backend"] = {
            "schema": "ai-vector-cleanroom.compute-backend.v1",
            "provider": "cpu",
            "status": "audit_unavailable",
            "last_error": type(exc).__name__,
        }
    (_rank, _quality, effective_options, _st, _sc, chosen_public,
     selected_svg, selected_flat, selected_metric_ref) = selected_item
    best_visual_quality = selection_policy["best_visual_quality"]
    for item in viable:
        item[5]["visual_gap_from_best"] = round(
            best_visual_quality - item[1], 6)
        item[5]["requested_features_retained"] = sum(
            1 for key in RECONSTRUCTION_KEYS
            if requested_options.get(key) not in (None, "off")
            and item[2].get(key) == requested_options.get(key))
    chosen_public["selected"] = True
    chosen = {key: value for key, value in effective_options.items()
              if requested_options.get(key) != value}

    # Restore the exact candidate that was scored; do not rebuild it a second
    # time. The later self-check still renders these restored bytes afresh for
    # the final report and hotspot image.
    atomic_replace_bytes(svg_path, selected_svg)
    flat_chk.write_bytes(selected_flat)
    stats, scores = _st, _sc
    metric_ref = selected_metric_ref
    e_final = _eff(scores)
    initial_public = next((c for c in candidates
                           if c["options"] == requested_options), None)
    e0 = initial_public.get("quality_score") if initial_public else None
    if chosen:
        before = f"{e0:.1f}%" if e0 is not None else "failed"
        after = f"{e_final:.1f}%" if e_final is not None else "unscored"
        msg = f"auto-fallback applied: {chosen} ({before} -> {after})"
        warnings.append(msg)
        print(f"    AUTO-FALLBACK: {chosen} ({before} -> {after})")

    # Source review must reflect the background mode actually delivered.
    clean_img, _sz, removed = _prepare_image(
        img_path, max_size=0, background=effective_options["background"],
        white_threshold=effective_options["white_threshold"], alpha_threshold=12)
    selected_hole_getter = getattr(stats, "_validation_hole_mask", None)
    selected_hole_mask = (
        selected_hole_getter() if callable(selected_hole_getter) else None)
    clean_img = _apply_validation_hole_mask(clean_img, selected_hole_mask)
    _save_reference(clean_img, ref_png)
    if removed:
        warnings.append("auto background removal was applied; if a light "
                        "design element touching the border disappeared, "
                        "re-run with --background keep")

    e_final = _eff(scores)
    if e_final is None:
        raise RuntimeError("conversion could not be validated on source ink")
    if e_final < 60.0:
        raise RuntimeError(
            f"low-confidence conversion ({e_final:.1f}% foreground match "
            f"after trying {len(candidates)} candidate(s)); manual "
            "vectorization recommended for this image")

    # 2a) Transactional editability enhancements.  Annulus smoothing may
    # differ at sub-pixel boundaries; compound splitting and scene grouping
    # must render pixel-exactly.  Every stage has an independent rollback.
    from svg_postprocess import (attach_paint_roles, enhance_svg_structure,
                                 measure_svg_structure)
    _emit("editability_enhancements", "正在做可編輯結構的交易式增強")
    original_svg_bytes = svg_path.read_bytes()
    baseline_scores = scores
    # Post-processing stages are sequential: the accepted "after" document
    # of one stage is normally the "before" document of the next.  Cache
    # renderer output by SVG content so that safety checks do not rasterise
    # that identical document again.  Validate at the source's native scale
    # (with a 512px minimum and a 2048px longest-side cap).  render_svg_png's
    # size argument is an output *width*, so derive it from both viewBox axes
    # to keep portrait artwork inside the same resource bound.
    stage_render_cache = {}
    validation_render_width = _validation_render_width(stats.viewbox)

    def render_validator(before, after, stage):
        return validate_svg_stage_renders(
            before, after, stage, gradient_info=stats.gradient_info,
            render_cache=stage_render_cache,
            render_size=validation_render_width)
    enhancement_report = enhance_svg_structure(
        svg_path, validator=render_validator)

    def _detail_p10(score_block):
        value = (score_block.get("detail_grid") or {}).get(
            "p10_score_percent")
        return float(value) if isinstance(value, (int, float)) else None

    enhanced_scores = self_check(
        svg_path, flat_chk, metric_ref,
        gradient_info=stats.gradient_info, viewbox=stats.viewbox)
    before_fg = _eff(baseline_scores)
    after_fg = _eff(enhanced_scores)
    before_p10 = _detail_p10(baseline_scores)
    after_p10 = _detail_p10(enhanced_scores)
    degradation_triggers = []
    if after_fg is None:
        degradation_triggers.append("foreground_unscored")
    elif before_fg is not None and after_fg < before_fg - 0.25:
        degradation_triggers.append("foreground_drop_gt_0.25")
    if before_p10 is not None and after_p10 is None:
        degradation_triggers.append("detail_p10_unscored")
    elif (before_p10 is not None and after_p10 is not None
          and after_p10 < before_p10 - 1.0):
        degradation_triggers.append("detail_p10_drop_gt_1.0")
    degraded = bool(degradation_triggers)
    if degraded:
        # Preserve the exact/grouping wins and remove only the approximate
        # annulus stage before considering a full rollback.
        atomic_replace_bytes(svg_path, original_svg_bytes)
        retry = enhance_svg_structure(
            svg_path, validator=render_validator, enable_annulus=False)
        retry_scores = self_check(
            svg_path, flat_chk, metric_ref,
            gradient_info=stats.gradient_info, viewbox=stats.viewbox)
        retry_fg = _eff(retry_scores)
        retry_p10 = _detail_p10(retry_scores)
        retry_triggers = []
        if retry_fg is None:
            retry_triggers.append("retry_foreground_unscored")
        elif before_fg is not None and retry_fg < before_fg - 0.05:
            retry_triggers.append("retry_foreground_drop_gt_0.05")
        if before_p10 is not None and retry_p10 is None:
            retry_triggers.append("retry_detail_p10_unscored")
        elif (before_p10 is not None and retry_p10 is not None
              and retry_p10 < before_p10 - 0.1):
            retry_triggers.append("retry_detail_p10_drop_gt_0.1")
        retry_degraded = bool(retry_triggers)
        if retry_degraded:
            atomic_replace_bytes(svg_path, original_svg_bytes)
            scores = baseline_scores
            attempted_with_annulus = enhancement_report
            rolled_stages = {
                stage_name: _rolled_back_stage_report(stage_name, stage_report)
                for stage_name, stage_report in retry.get("stages", {}).items()
            }
            final_guard = {
                "status": "rolled_back_all",
                "foreground_before": before_fg,
                "foreground_attempted": after_fg,
                "foreground_retry": retry_fg,
                "detail_p10_before": before_p10,
                "detail_p10_attempted": after_p10,
                "detail_p10_retry": retry_p10,
                "triggered_by": degradation_triggers + retry_triggers,
                "maximum_allowed_foreground_drop": 0.25,
                "maximum_allowed_detail_p10_drop": 1.0,
                "retry_maximum_allowed_foreground_drop": 0.05,
                "retry_maximum_allowed_detail_p10_drop": 0.1,
                "reason": "post-processing could not reproduce the validated source score",
            }
            original_structure = measure_svg_structure(svg_path)
            enhancement_report = {
                "schema": "ai-vector-cleanroom.editability-enhancements/v1",
                "stages": rolled_stages,
                "structure_before": original_structure,
                "structure_after": original_structure,
                "attempts": {
                    "with_annulus": attempted_with_annulus,
                    "exact_only": retry,
                },
                "final_source_guard": final_guard,
                "scope_note": (
                    "All structural proposals were rolled back by the final "
                    "source-quality guard; the delivered SVG is the selected tracer output."
                ),
            }
            warnings.append(
                "editability post-processing rolled back: final source guard rejected it")
        else:
            scores = retry_scores
            retry["final_source_guard"] = {
                "status": "annulus_rolled_back",
                "foreground_before": before_fg,
                "foreground_with_annulus": after_fg,
                "foreground_final": retry_fg,
                "detail_p10_before": before_p10,
                "detail_p10_with_annulus": after_p10,
                "detail_p10_final": retry_p10,
                "triggered_by": degradation_triggers,
                "maximum_allowed_foreground_drop": 0.25,
                "maximum_allowed_detail_p10_drop": 1.0,
            }
            retry["attempts"] = {"with_annulus": enhancement_report}
            enhancement_report = retry
            warnings.append(
                "native annulus proposal rolled back by the final source-quality guard")
    else:
        scores = enhanced_scores
        enhancement_report["final_source_guard"] = {
            "status": "accepted",
            "foreground_before": before_fg,
            "foreground_after": after_fg,
            "detail_p10_before": before_p10,
            "detail_p10_after": after_p10,
            "maximum_allowed_foreground_drop": 0.25,
            "maximum_allowed_detail_p10_drop": 1.0,
            "triggered_by": [],
        }
    e_final = _eff(scores)

    # A trace's pixel staircase and false holes are observations, not the
    # ground truth. Source reconstruction has its own per-region native guard;
    # ordinary later curve simplification retains the prior topology policy.
    _emit("source_reconstruction", "正在依原圖修復輪廓與不確定淺色區")
    from source_repair_stage import attempt_source_repairs, project_gradient_report
    source_repair_report, updated_gradient_info = attempt_source_repairs(
        svg_path, deliver / "source_original.png", ref_png, stats.gradient_info,
        error_budget_percent=args.curve_error_percent)
    enhancement_report.setdefault("stages", {})[
        "source_reconstruction"] = source_repair_report
    if source_repair_report.get("status") == "committed":
        stats.gradient_info = updated_gradient_info
        if source_repair_report.get("replaced_gradient_details"):
            stats.n_gradients = len(updated_gradient_info)
            stats.palette_audit['gradient_reconstruction'] = project_gradient_report(
                stats.palette_audit['gradient_reconstruction'],
                updated_gradient_info, source_repair_report)
            stats.geometry_notes = [
                "initial trace (before source repair): " + note
                if " continuous source colour field(s) rebuilt " in note else note
                for note in stats.geometry_notes]
            stats.geometry_notes.append(
                f"{len(source_repair_report['replaced_gradient_details'])} provisional "
                "gradient(s) replaced with native-source-supported solid fills")
        print(f"  Source reconstruction committed: "
              f"{len(source_repair_report.get('committed', []))} source-verified repairs")
    elif source_repair_report.get("status") == "error":
        warnings.append("source reconstruction unavailable; previous validated SVG preserved")

    # Error-bounded refitting targets the designer's chief curve complaint:
    # trace-like micro cubics and redundant anchors.  It runs before paint-role
    # metadata is attached so the manifest identifies the exact committed SVG.
    _emit("curve_refit", "正在做誤差有界的曲線節點精簡")
    curve_refit_report = _attempt_curve_refit_transaction(
        svg_path, flat_chk, metric_ref, stats,
        error_budget_percent=args.curve_error_percent)
    enhancement_report.setdefault("stages", {})[
        "curve_refit"] = curve_refit_report
    if curve_refit_report.get("status") == "committed":
        proposal = curve_refit_report.get("proposal") or {}
        print(
            f"  Curve refit committed: {proposal.get('path_count_refit', 0)} "
            f"paths / {proposal.get('anchors_removed', 0)} redundant anchors removed")
    elif curve_refit_report.get("status") == "rolled_back":
        print("  Curve refit rolled back: renderer/source guard rejected it")
    elif curve_refit_report.get("status") == "error":
        warnings.append(
            "curve-refit proposal unavailable; original validated paths preserved")

    # Portable paint-role resources keep explicit SVG paints authoritative.
    _emit("paint_roles", "正在建立可攜式換色角色")
    paint_manifest_path = deliver / f"{out_base}_paint_roles.json"
    paint_manifest = None
    paint_role_report = {"status": "unavailable"}
    try:
        paint_manifest, paint_role_report = attach_paint_roles(
            svg_path, paint_manifest_path, validator=render_validator)
    except Exception as exc:
        paint_manifest_path.unlink(missing_ok=True)
        paint_role_report = {
            "status": "rolled_back_error",
            "reason": f"{type(exc).__name__}: {exc}"[:300],
        }
        warnings.append("paint-role controls unavailable; ordinary SVG paints remain intact")

    final_structure = measure_svg_structure(svg_path)
    scene_report = (enhancement_report.get("stages", {})
                    .get("scene_graph", {}))
    # Join the source-ownership geometry proof to the exact final SVG before
    # the designer audit.  The report and the gate must consume the same
    # post-curve-refit evidence object; auditing ``stats.gradient_info`` here
    # would omit final_svg_consistency until after readiness was decided.
    final_gradient_details, gradient_geometry_consistency = (
        _final_gradient_report_details(
            svg_path, stats.gradient_info, curve_refit_report))
    _emit("designer_audits", "正在稽核設計操作與結構證據")
    try:
        from designer_ops_audit import audit_designer_operations
        designer_operations = audit_designer_operations(
            svg_path, paint_manifest=paint_manifest,
            scene_graph_report=scene_report)
    except Exception as exc:
        designer_operations = {
            "status": "manual_review",
            "acceptance_scope": "generic_machine_detectable_structural_handles",
            "semantic_task_validation": "not_performed",
            "timed_human_editing_validation": "not_performed",
            "human_acceptance": "not_tested",
            "passed": 0,
            "partial": 0,
            "failed": 0,
            "manual_review": 5,
            "automatable": 0,
            "reason": f"designer operation audit failed: {exc!r}"[:300],
            "scope_note": "Timed human editing remains required.",
        }

    applied_annuli = (enhancement_report.get("stages", {})
                      .get("annulus", {}).get("applied_candidates", 0))
    exact_native_count = (enhancement_report.get("stages", {})
                          .get("exact_native_shapes", {})
                          .get("committed_candidate_count", 0))
    compound_delta = (enhancement_report.get("stages", {})
                      .get("compound_paths", {}).get("selectable_path_delta", 0))
    exact_cubic_count = (enhancement_report.get("stages", {})
                         .get("compound_paths", {})
                         .get("linear_cubics_simplified", 0))
    scene_actual = scene_report.get("actual_dom_group_count", 0)
    role_controls = ((paint_manifest or {}).get("resource_counts", {})
                     .get("role_controls", 0))
    print(f"  Editability enhancements: {applied_annuli} native annulus / "
          f"{exact_native_count} exact native lines or polylines / "
          f"{exact_cubic_count} exact cubic-to-line simplifications / "
          f"+{compound_delta} selectable compound parts / "
          f"{scene_actual} actual object groups / {role_controls} paint controls")

    # The verdict must use the exact SVG after every paint-role/structure
    # mutation.  Previously an early score set the status, then this later
    # self-check silently replaced the report metrics without recomputing the
    # gate; the workbench could therefore say Done for a visibly broken file.
    chk_render = deliver / "_render_check.png"
    scores = self_check(svg_path, flat_chk, metric_ref,
                        gradient_info=stats.gradient_info,
                        keep_render=chk_render, viewbox=stats.viewbox)
    e_final = _eff(scores)
    if e_final is None:
        raise RuntimeError("final delivered SVG could not be validated on source ink")
    if e_final < 60.0:
        raise RuntimeError(
            f"final delivered SVG failed catastrophically ({e_final:.1f}% "
            "foreground match); manual vectorization recommended")
    visual_gate = _evaluate_visual_gate(scores)
    _emit("final_quality_gates", "正在執行外觀、拓撲、漸層與曲線最終閘門")
    visual_acceptance_status = visual_gate["status"]
    visual_review_required = visual_acceptance_status != "accepted"
    selected_detail_grid = scores.get("detail_grid") or {}
    detail_p10 = selected_detail_grid.get("p10_score_percent")
    if visual_acceptance_status == "rejected":
        reason = "; ".join(visual_gate["reasons"])
        warnings.append(
            "VISUAL REVIEW REQUIRED: compare this editable draft with the source "
            f"and repair flagged areas before using it as finished artwork ({reason})")
        print(f"    VISUAL NOT ACCEPTED: {reason}")
    elif visual_acceptance_status == "manual_review":
        reason = "; ".join(visual_gate["reasons"])
        warnings.append(f"VISUAL REVIEW REQUIRED: {reason}")
        print(f"    VISUAL REVIEW REQUIRED: {reason}")
    if (isinstance(detail_p10, (int, float)) and detail_p10 < 80.0):
        warnings.append(
            f"LOCAL DETAIL REVIEW REQUIRED: the weakest 10% of ink cells "
            f"scored {detail_p10:.1f}% (below 80%); small lines or dots may "
            "be missing even though the whole-logo score is higher")
        print(f"    LOCAL DETAIL REVIEW REQUIRED: grid p10 {detail_p10:.1f}%.")
    if (final_structure["paths"]
            + final_structure["native_primitives"]
            + final_structure["strokes"] == 0):
        # Belt-and-braces: the engine also raises on empty output. An "empty
        # success" must never reach the user as a valid result.
        raise RuntimeError(
            "no vector elements were produced (the visible foreground may be "
            "too small, or was removed as background; try --background keep)")

    # Designer readiness is deliberately independent from raster similarity.
    # A blocky multi-band mountain can score highly as pixels and still be the
    # wrong editable object model; likewise a visually faithful contour can be
    # unusable when it contains hundreds of micro anchors.
    try:
        from designer_quality import audit_designer_quality
        from source_topology_audit import audit_source_topology
        source_topology_report = audit_source_topology(
            svg_path, deliver / "source_original.png", ref_png)
        enhancement_report.setdefault("stages", {})["source_topology_audit"] = source_topology_report
        proposal_metadata = {
            "scene_graph_report": scene_report,
            "editability_enhancements": enhancement_report,
            "stroke_reconstruction_report": (getattr(stats, "palette_audit", {}) or {}).get(
                "stroke_reconstruction"),
            "source_topology_report": source_topology_report,
        }
        if stats.gradient_info:
            palette_audit = getattr(stats, "palette_audit", {}) or {}
            proposal_metadata.update({
                "gradient_reconstruction_report": (
                    palette_audit.get("gradient_reconstruction") or {}),
                "gradient_details": final_gradient_details,
            })
        designer_quality = audit_designer_quality(
            svg_path,
            proposal_metadata=proposal_metadata,
            visual_report={
                "visual_acceptance_status": visual_acceptance_status,
                "visual_gate": visual_gate,
            },
        )
    except Exception as exc:
        designer_quality = {
            "schema": "ai-vector-cleanroom.designer-quality/v1",
            "designer_readiness_status": "manual_review_required",
            "designer_ready": False,
            "raster_visual_designer_status_divergence": (
                visual_acceptance_status == "accepted"),
            "status_explanation": (
                f"designer-quality audit failed: {exc!r}"[:260]),
            "gradient_object_gate": {
                "status": "not_audited", "passed": False,
                "failure_reasons": [],
                "warning_reasons": ["audit_unavailable"],
            },
            "curve_economy_gate": {
                "status": "not_audited", "passed": False,
                "failure_reasons": [],
                "warning_reasons": ["audit_unavailable"],
            },
            "human_designer_validation": "not_performed_by_this_machine_audit",
        }
    _verify_designer_gradient_evidence_join(
        final_gradient_details, designer_quality)

    # The generic many-subpath heuristic is evaluated after the exact
    # designer certificate.  A source-topology-certified gradient compound
    # path may need many closed loops to preserve holes in one ownership
    # object; unrelated compound paths remain subject to the same trigger.
    try:
        from editability_audit import audit_editability
        editability = audit_editability(svg_path, {
            **final_structure,
            "designer_operations": designer_operations,
            "designer_quality": designer_quality,
        })
    except Exception as exc:
        editability = {
            "status": "manual_review", "score": None,
            "schema": "ai-vector-cleanroom.editability/v2",
            "audit_model": "layered-v2-audit-error",
            "reasons": [f"editability audit failed: {exc!r}"[:220]],
            "automation_readiness": {"status": "not_audited", "score": None},
            "redraw_complexity": {"status": "not_audited", "ease_score": None},
            "workflow_friction": {"status": "not_audited", "ease_score": None},
            "acceptance_gate": {"status": "manual_review", "passed": False},
            "named_operation_evidence": {"status": "not_audited"},
            "human_validation": {
                "status": "not_performed",
                "timed_editing_test_performed": False,
                "designer_acceptance": None,
            },
            "editability_details": {
                "scope_note": "Structural audit unavailable; timed human "
                              "editing tests are still required."},
        }
    editability_status = editability.get("status", "manual_review")
    if editability_status != "accepted":
        edit_score = editability.get("score")
        score_text = (f"{edit_score:.1f}" if isinstance(edit_score, (int, float))
                      else "unavailable")
        warnings.append(
            f"EDITABILITY REVIEW REQUIRED: structural score {score_text}; "
            "this SVG may look correct but still need grouping/node cleanup")
        print(f"    EDITABILITY REVIEW REQUIRED: structural score {score_text}.")

    designer_readiness_status = designer_quality.get(
        "designer_readiness_status", "manual_review_required")
    if designer_readiness_status != "designer_ready":
        gradient_reasons = (
            designer_quality.get("gradient_object_gate", {}).get(
                "failure_reasons", []))
        curve_reasons = (
            designer_quality.get("curve_economy_gate", {}).get(
                "failure_reasons", []))
        reason_text = "; ".join((gradient_reasons + curve_reasons)[:6])
        warnings.append(
            "DESIGNER READINESS NOT PASSED: raster similarity does not make "
            "fragmented gradients or over-anchored curves deliverable"
            + (f" ({reason_text})" if reason_text else ""))
        print(f"    DESIGNER READINESS: {designer_readiness_status}"
              + (f" — {reason_text}" if reason_text else ""))

    if (visual_acceptance_status == "rejected"
            or editability_status == "rejected"
            or designer_readiness_status == "manual_rework_required"):
        acceptance_status = "rejected"
    elif (visual_review_required or editability_status != "accepted"
          or designer_readiness_status != "designer_ready"):
        acceptance_status = "manual_review"
    else:
        acceptance_status = "accepted"
    manual_review_required = acceptance_status != "accepted"

    # Keep the SVG self-describing even when it is copied away from its ZIP.
    # The JSON is XML-escaped text, readable by editors and scripts without
    # introducing proprietary namespaces or non-vector payloads.
    svg_text = svg_path.read_text(encoding="utf-8")
    audited_svg_bytes = svg_path.read_bytes()
    metadata_payload = {
        "tool": "AI Vector Cleanroom",
        "version": TOOL_VERSION,
        "tool_version": TOOL_VERSION,
        "options_requested": requested_options,
        "options_effective": effective_options,
        "visual_acceptance_status": visual_acceptance_status,
        "visual_gate": visual_gate,
        "editability_status": editability_status,
        "designer_readiness_status": designer_readiness_status,
        "gradient_object_gate_status": designer_quality.get(
            "gradient_object_gate", {}).get("status", "not_audited"),
        "curve_economy_gate_status": designer_quality.get(
            "curve_economy_gate", {}).get("status", "not_audited"),
        "acceptance_status": acceptance_status,
        "editability_enhancements": {
            key: value.get("status")
            for key, value in enhancement_report.get("stages", {}).items()
        },
        "component_repair_status": (stats.component_repair or {}).get(
            "status", "not_audited"),
        "paint_role_controls": role_controls,
        "designer_operations_passed": designer_operations.get(
            "summary", {}).get("passed", 0),
    }
    metadata = (f'<metadata id="ai-vector-cleanroom-metadata">'
                f'{html.escape(json.dumps(metadata_payload, ensure_ascii=False, separators=(",", ":")), quote=False)}'
                '</metadata>')
    if 'id="ai-vector-cleanroom-metadata"' in svg_text:
        svg_text = re.sub(
            r'<metadata\b[^>]*\bid="ai-vector-cleanroom-metadata"[^>]*>.*?</metadata>',
            metadata, svg_text, count=1, flags=re.DOTALL)
    else:
        svg_text = re.sub(r'(<svg\b[^>]*>)', r'\1\n  ' + metadata,
                          svg_text, count=1)
    atomic_replace_bytes(svg_path, svg_text.encode("utf-8"))
    _bind_delivered_source_audits(audited_svg_bytes, svg_path.read_bytes(),
                                 designer_quality, enhancement_report)
    # The structural audit runs before the self-describing metadata is added.
    # Metadata cannot change any audited edit operation, but the report's
    # primary SHA must still identify the exact file handed to the designer.
    svg_audit_info = designer_operations.get("svg")
    if isinstance(svg_audit_info, dict):
        import hashlib
        audit_input_sha = svg_audit_info.get("sha256")
        svg_audit_info["audit_input_sha256"] = audit_input_sha
        svg_audit_info["sha256"] = hashlib.sha256(svg_path.read_bytes()).hexdigest()
        svg_audit_info["sha256_scope"] = "delivered_svg_after_inert_metadata"

    recolor_filename = None
    if paint_manifest is not None:
        try:
            from recolor_page import make_recolor_html
            recolor_filename = "色彩調整.html"
            make_recolor_html(
                deliver / recolor_filename, svg_path, paint_manifest,
                download_filename=f"{out_base}_換色.svg",
                tool_version=TOOL_VERSION)
        except Exception as exc:
            (deliver / "色彩調整.html").unlink(missing_ok=True)
            recolor_filename = None
            warnings.append(
                f"offline recolor page unavailable: {type(exc).__name__}: {exc}"[:240])
    paint_summary = _paint_resource_summary(stats)
    paint_labels = []
    readme_palette = []
    for resource in paint_summary["paint_resources"]:
        if resource["type"] == "solid":
            paint_labels.append(resource["hex"])
            readme_palette.append((resource["name"], resource["hex"]))
        else:
            stops = resource.get("stops") or []
            colors = [stop.get("color") for stop in stops if stop.get("color")]
            span = f"{colors[0]}->{colors[-1]}" if colors else "linearGradient"
            paint_labels.append(f"{resource.get('id') or 'gradient'}({span})")
            readme_palette.append((resource.get("id") or "gradient", span))
    vector_label = ("Vector OK" if acceptance_status == "accepted"
                    else "Vector generated — REVIEW REQUIRED"
                    if acceptance_status == "manual_review"
                    else "Editable draft generated — manual repair required before finished artwork")
    print(f"  {vector_label}: {final_structure['groups']} final SVG groups; "
          f"{paint_summary['unique_paints_total']} unique paint resources -> "
          + ", ".join(paint_labels))
    primitive_counts = {
        "native_primitives": final_structure["native_primitives"],
        "native_circles": final_structure["native_circles"],
        "native_rectangles": final_structure["native_rectangles"],
        "native_ellipses": final_structure["native_ellipses"],
        "native_lines": final_structure["native_lines"],
        "native_polylines": final_structure["native_polylines"],
        "native_polygons": final_structure["native_polygons"],
    }
    final_stroke_details = _final_stroke_details(svg_path, stats)
    print(f"  Structure: {final_structure['paths']} paths / "
          f"{primitive_counts['native_primitives']} native primitives "
          f"({primitive_counts['native_circles']} circles, "
          f"{primitive_counts['native_rectangles']} rectangles, "
          f"{primitive_counts['native_ellipses']} ellipses, "
          f"{primitive_counts['native_lines']} lines, "
          f"{primitive_counts['native_polylines']} polylines, "
          f"{primitive_counts['native_polygons']} polygons) / "
          f"{final_structure['strokes']} rebuilt strokes / "
          f"{final_structure['gradients']} gradients / "
          f"{final_structure['designer_anchors_total']} designer anchors "
          f"({final_structure['nodes']} legacy nodes)")
    for g in stats.geometry_notes:
        print(f"    - {g}")

    # 2b) Report the final render check already used by the visual gate above.
    if any(scores.get(k) is not None for k in ("flat", "source", "foreground")):
        def _s(k):
            return f"{scores[k]:.1f}%" if scores.get(k) is not None else "n/a"
        print(f"  Self-check: flat {_s('flat')} / source {_s('source')} / "
              f"foreground {_s('foreground')}")
        if scores["source"] is not None and scores["source"] < 90:
            warnings.append("source match below 90%: the image likely has "
                            "gradients, shadows, or fine detail outside this "
                            "tool's target use case; review the overlay page")
            print("    WARNING: low source match, review the overlay page.")
        if scores["foreground"] is not None and scores["foreground"] < 80:
            warnings.append("foreground match below 80%: visible design "
                            "elements may be missing or heavily altered in "
                            "the SVG; open review.html and verify")
            print("    WARNING: low foreground match, small design elements "
                  "may be missing.")
    flat_chk.unlink(missing_ok=True)

    # 2c) review hotspots: source-ink cells only. JPEG alpha is opaque across
    # the white canvas, so alpha-based foreground diluted missing dots/lines.
    from quality_diagnostics import source_structure_hotspots
    topology = enhancement_report.get('stages', {}).get('source_topology_audit', {})
    hotspots = source_structure_hotspots(topology, [0, 0, stats.width, stats.height]) + (scores.get("hotspots") or [])
    detail_grid = scores.get("detail_grid")
    chk_render.unlink(missing_ok=True)
    for _metric_path in metric_refs.values():
        _metric_path.unlink(missing_ok=True)

    # 3) Preview. Never silently pass off the source as an SVG render.
    _emit("preview", "正在產生 SVG 實際渲染預覽")
    preview = deliver / f"{out_base}_preview.png"
    preview_is_fallback = not render_svg_png(
        svg_path, preview, gradient_info=stats.gradient_info)
    if preview_is_fallback:
        bg = Image.new("RGB", clean_img.size, (255, 255, 255))
        bg.paste(clean_img, (0, 0), clean_img)
        bg.save(preview)
        warnings.append("SVG preview unavailable (svglib/reportlab missing); "
                        f"{preview.name} is the cleaned source image, not an "
                        "SVG render")
        print("  Preview: SVG render unavailable — wrote cleaned source image "
              "instead (install requirements-preview.txt for real previews)")
    else:
        print("  Preview OK")

    # 4) Review page, JSON report, output notes.
    _emit("report", "正在建立審查頁與可驗證報告")
    make_review_html(deliver, out_base, ref_png, svg_path.read_text(encoding="utf-8"),
                     (stats.width, stats.height), hotspots=hotspots,
                     scores=scores, structure=final_structure,
                     acceptance_status=acceptance_status,
                     manual_review_required=manual_review_required,
                     visual_acceptance_status=visual_acceptance_status,
                     editability_status=editability_status,
                     editability_score=editability.get("score"),
                     automation_readiness_score=(
                         editability.get("automation_readiness", {}).get("score")),
                     human_validation_status=(
                         editability.get("human_validation", {}).get(
                             "status", "not_performed")),
                     designer_readiness_status=designer_readiness_status,
                     gradient_object_gate_status=designer_quality.get(
                         "gradient_object_gate", {}).get(
                             "status", "not_audited"),
                     curve_economy_gate_status=designer_quality.get(
                         "curve_economy_gate", {}).get(
                             "status", "not_audited"),
                     detail_grid=detail_grid,
                     recolor_filename=recolor_filename)
    report = {
        "tool_version": TOOL_VERSION,
        "input": img_path.name,
        "output_base": out_base,
        "size": [stats.width, stats.height],
        "palette": paint_summary["palette"],
        "palette_detection": getattr(stats, "palette_audit", {}),
        "layers": paint_summary["layers"],
        "paint_resources": paint_summary["paint_resources"],
        "solid_paints": paint_summary["solid_paints"],
        "gradient_paints": paint_summary["gradient_paints"],
        "unique_paints_total": paint_summary["unique_paints_total"],
        "groups": final_structure["groups"],
        "paths": final_structure["paths"],
        **primitive_counts,
        "strokes": final_structure["strokes"],
        "stroke_details": final_stroke_details,
        "gradients": final_structure["gradients"],
        "gradient_details": [
            {key: value for key, value in detail.items() if key != "key"}
            for detail in final_gradient_details
        ],
        "gradient_geometry_consistency": gradient_geometry_consistency,
        "nodes_total": final_structure["nodes"],
        "designer_anchors_total": final_structure["designer_anchors_total"],
        "final_structure": final_structure,
        "engine_structure_before_postprocess": {
            "paint_layers": stats.colors,
            "paths": stats.n_paths,
            **_native_primitive_counts(stats),
            "strokes": stats.n_strokes,
            "gradients": stats.n_gradients,
            "nodes_total": stats.n_nodes,
        },
        "component_repair": stats.component_repair,
        "editability_enhancements": enhancement_report,
        "paint_roles": paint_role_report,
        "paint_role_manifest": (paint_manifest_path.name
                                if paint_manifest is not None else None),
        "recolor_page": recolor_filename,
        "designer_operations": designer_operations,
        "background_removed": stats.removed_background,
        "geometry_level": effective_options["geometry"],
        "geometry_notes": stats.geometry_notes,
        "flat_match_percent": scores["flat"],
        "source_match_percent": scores["source"],
        "foreground_match_percent": scores["foreground"],
        "foreground_recall_percent": scores["foreground_recall"],
        "foreground_precision_percent": scores["foreground_precision"],
        "foreground_coverage_f1_percent": scores["foreground_coverage_f1"],
        "foreground_color_fidelity_percent": scores["foreground_color_fidelity"],
        "source_ink_pixels": scores["source_ink_pixels"],
        "render_ink_pixels": scores["render_ink_pixels"],
        "ink_threshold": scores["ink_threshold"],
        "foreground_alpha_comparison": scores.get("alpha_comparison"),
        "self_check_max_side": SELF_CHECK_MAX_SIDE,
        "review_preview_max_side": REVIEW_PREVIEW_MAX_SIDE,
        "preview_is_svg_render": not preview_is_fallback,
        "detail_grid": detail_grid,
        "transparent_light_fidelity": scores.get(
            "transparent_light_fidelity"),
        "hotspots": hotspots,
        "candidates": candidates,
        "candidate_selection_policy": selection_policy,
        "auto_fallback": chosen,
        "visual_acceptance_status": visual_acceptance_status,
        "visual_gate": visual_gate,
        "editability_status": editability_status,
        "editability_score": editability.get("score"),
        "editability_reasons": editability.get("reasons", []),
        "editability_details": editability.get("editability_details", {}),
        "editability_schema": editability.get("schema"),
        "editability_audit_model": editability.get("audit_model"),
        "automation_readiness": editability.get("automation_readiness", {}),
        "redraw_complexity": editability.get("redraw_complexity", {}),
        "workflow_friction": editability.get("workflow_friction", {}),
        "editability_acceptance_gate": editability.get("acceptance_gate", {}),
        "named_operation_evidence": editability.get(
            "named_operation_evidence", {}),
        "human_validation": editability.get("human_validation", {}),
        "designer_quality": designer_quality,
        "designer_readiness_status": designer_readiness_status,
        "gradient_object_gate": designer_quality.get(
            "gradient_object_gate", {}),
        "curve_economy_gate": designer_quality.get(
            "curve_economy_gate", {}),
        "acceptance_status": acceptance_status,
        "execution": (
            control.snapshot() if control is not None
            and hasattr(control, "snapshot") else {
                "schema": "aivc-execution-control-v1",
                "status": "not_supervised",
            }),
        "manual_review_required": manual_review_required,
        "warnings": warnings,
        "options": dict(effective_options),
        "options_requested": dict(requested_options),
        "options_effective": dict(effective_options),
    }
    (deliver / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_output_readme(deliver, out_base, readme_palette, stats.geometry_notes,
                       preview_is_fallback=preview_is_fallback, scores=scores,
                       acceptance_status=acceptance_status,
                       requested_options=requested_options,
                       effective_options=effective_options,
                       auto_fallback=chosen,
                       visual_acceptance_status=visual_acceptance_status,
                       editability_status=editability_status,
                       editability_score=editability.get("score"),
                       automation_readiness_score=(
                           editability.get("automation_readiness", {}).get("score")),
                       human_validation_status=(
                           editability.get("human_validation", {}).get(
                               "status", "not_performed")),
                       editability_reasons=editability.get("reasons", []),
                       detail_grid=detail_grid,
                       enhancement_report=enhancement_report,
                       paint_role_report=paint_role_report,
                       paint_manifest_name=(paint_manifest_path.name
                                            if paint_manifest is not None else None),
                       recolor_filename=recolor_filename,
                       designer_operations=designer_operations,
                       final_structure=final_structure)
    print("  Review page + report + output notes OK")

    # 5) Zip the result folder.
    _emit("package", "正在封裝完整結果；完成前不會提交到工作台")
    zip_path = output_dir / f"result_{out_base}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(deliver.iterdir()):
            z.write(f, arcname=f"{deliver.name}/{f.name}")
    mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  Result package: {zip_path.name} ({mb:.2f} MB)")
    _emit("package_complete", "結果與證據封裝完成")
    return zip_path


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="AI Vector Cleanroom — bitmap logo/icon to editable SVG draft")
    ap.add_argument("--input", type=Path, default=DATA_DIR / "input",
                    help=f"input folder (default: {DATA_DIR / 'input'})")
    ap.add_argument("--output", type=Path, default=DATA_DIR / "output",
                    help=f"output folder (default: {DATA_DIR / 'output'})")
    ap.add_argument("--colors", type=int, default=0,
                    help="force a fixed palette size; default 0 = auto-detect")
    ap.add_argument("--white-threshold", type=int, default=220,
                    help="threshold for light/checker background cleanup; default 220")
    ap.add_argument("--background", choices=["auto", "keep", "transparent"],
                    default="auto",
                    help="background handling: auto = heuristic removal of light "
                         "border-connected background; keep = never remove; "
                         "transparent = force removal (default: auto)")
    ap.add_argument("--max-size", type=int, default=2048,
                    help="downscale the longest side before tracing; 0 disables "
                         "(default: 2048)")
    ap.add_argument("--strokes", choices=["on", "off"], default="on",
                    help="rebuild uniform-width line work as real strokes "
                         "with stroke-width (default: on)")
    ap.add_argument("--gradients", choices=["on", "off"], default="on",
                    help="rebuild banded color ramps as linear gradient "
                         "fills (default: on)")
    ap.add_argument("--geometry", choices=["conservative", "normal", "off"],
                    default="conservative",
                    help="geometry regularization level (default: conservative; "
                         "normal additionally straightens ring/band edges into "
                         "mathematical arcs)")
    ap.add_argument(
        "--curve-error-percent", type=float, default=0.25,
        help="maximum normalised curve-fitting p95 error as a percentage of "
             "each object's bounding-box diagonal (default: 0.25)")
    ap.add_argument("--no-geometry", action="store_true",
                    help=argparse.SUPPRESS)   # deprecated alias for --geometry off
    ap.add_argument("--debug", action="store_true",
                    help="show full tracebacks for failed files")
    return ap


def validate_args(ap, args):
    if args.colors and not 2 <= args.colors <= 64:
        ap.error(f"--colors must be 0 (auto) or between 2 and 64, got {args.colors}")
    if not 0 <= args.white_threshold <= 255:
        ap.error(f"--white-threshold must be between 0 and 255, got {args.white_threshold}")
    if args.max_size < 0:
        ap.error(f"--max-size must be 0 (off) or a positive size, got {args.max_size}")
    if args.max_size and args.max_size < 16:
        ap.error(f"--max-size below 16 pixels is not usable, got {args.max_size}")
    if (not math.isfinite(args.curve_error_percent)
            or not 0.05 <= args.curve_error_percent <= 2.0):
        ap.error("--curve-error-percent must be between 0.05 and 2.0")
    if args.input.exists() and not args.input.is_dir():
        ap.error(f"--input is not a folder: {args.input}")


def _run_batch(args):
    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_inputs(input_dir)
    if not images:
        print(f"No images found. Put PNG/JPG/WebP/BMP files in:\n  {input_dir}")
        return 1

    print("=" * 56)
    print("  AI Vector Cleanroom")
    print("=" * 56)
    print("Creates editable SVG vector drafts for each image in the input folder.")

    plan = plan_output_names(images)
    made, failed = [], []
    for img in images:
        try:
            made.append(process_one(img, plan[img], args, output_dir))
        except Exception as e:
            print(f"  [failed] {img.name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            failed.append(img.name)
            # Remove BOTH partial outputs so a failed file leaves nothing
            # that could be mistaken for a fresh result.
            shutil.rmtree(output_dir / f"result_{plan[img]}", ignore_errors=True)
            (output_dir / f"result_{plan[img]}.zip").unlink(missing_ok=True)

    print("\n" + "=" * 56)
    if made:
        print("Done. Result packages:")
        for z in made:
            print(f"   {z.name}")
    if failed:
        print(f"\n{len(failed)} file(s) FAILED: {', '.join(failed)}")
    if not made:
        print("\nNo successful output.")
    print(f"\nOutput: {output_dir}")
    return 1 if failed or not made else 0


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    if args.no_geometry:
        args.geometry = "off"

    output_dir = Path(args.output).resolve(strict=False)
    default_output = (DATA_DIR / "output").resolve(strict=False)
    lock_root = DATA_DIR if output_dir == default_output else output_dir
    try:
        with writer_lock(lock_root, "batch"):
            return _run_batch(args)
    except (DataDirectoryBusyError, DataPathError) as exc:
        print(f"[資料目錄錯誤] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
