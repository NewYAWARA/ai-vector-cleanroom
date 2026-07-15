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

if getattr(sys, "frozen", False):
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TOOL_VERSION = "v3-codex-beta.5"
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
    # while keeping glyphs/arcs whole.  Three points is the measured tea-logo
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


def _select_viable_candidate(viable, requested_options,
                             material_gain=MATERIAL_FALLBACK_GAIN):
    """Choose among already-built candidates without sacrificing editing
    features for an immaterial visual-score fluctuation.

    Candidates within ``material_gain`` points of the best foreground score
    are a visual tie.  Inside that tie, preserve as many requested
    reconstruction stages as possible, then use the ordinary visual/structure
    rank.  A stage is therefore disabled automatically only when it buys a
    material visual improvement (or the requested build failed outright).
    """
    if not viable:
        raise ValueError("no viable candidates")
    # A numerically stronger but explicitly rejected build must never displace
    # a candidate that is at least safe enough for manual review.  Select the
    # best visual-gate tier first; only then compare Pareto trade-offs and
    # requested editing features inside that tier.  Doing this after dominance
    # would be too late because a rejected candidate could already eliminate
    # the safer one.
    status_rank = {"rejected": 0, "manual_review": 1, "accepted": 2}
    visual_status = {
        id(item): _evaluate_visual_gate(item[4])["status"] for item in viable
    }
    best_visual_status_rank = max(
        status_rank.get(visual_status[id(item)], 0) for item in viable)
    visual_tier = [
        item for item in viable
        if status_rank.get(visual_status[id(item)], 0)
        == best_visual_status_rank
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
    # Every survivor represents a real Pareto trade-off.  Re-applying the old
    # scalar foreground window here would undo the safety pruning (for
    # example, a 1.2 foreground gain could still hide a 2.5-point local-detail
    # loss).  Feature retention is therefore allowed only among these
    # non-dominated candidates.
    tied = survivors

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
        "best_visual_status": visual_status[id(visual_tier[0])],
        "selected_visual_status": visual_status[id(selected)],
        "visual_status_counts": visual_status_counts,
        "visual_status_survivor_count": len(visual_tier),
        "policy": "visual_gate_tier_then_safe_dominance_then_preserve_features",
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
        candidates = [first, f"{stem}_{ext}"]
        base = next((c for c in candidates if c.lower() not in used), None)
        if base is None:
            i = 2
            while f"{stem}_{ext}_{i}".lower() in used:
                i += 1
            base = f"{stem}_{ext}_{i}"
        used.add(base.lower())
        plan[p] = base
    return plan


def _paint_gradients(png_path: Path, gradient_info):
    """Paint fitted linear gradients over their placeholder-key pixels.

    svglib/reportlab cannot rasterize <linearGradient>, so the SVG is
    rendered with each gradient's unique key color and the true ramp is
    evaluated here per pixel from the fitted axis and stops."""
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
        vb_w = float(g["viewbox"][0])
        f = w / vb_w if vb_w else 1.0
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
        x1, y1 = g["x1"] * f, g["y1"] * f
        x2, y2 = g["x2"] * f, g["y2"] * f
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 <= 0:
            continue
        tt = ((xs - x1) * dx + (ys - y1) * dy) / L2
        tt = np.clip(tt, 0.0, 1.0)
        offs = [s["offset"] for s in g["stops"]]
        cols = np.array([[int(s["color"][1:3], 16), int(s["color"][3:5], 16),
                          int(s["color"][5:7], 16)] for s in g["stops"]],
                        dtype=np.float64)
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
    """Render the SVG to PNG via svglib. Returns False if deps are missing.

    Gradient fills are substituted with their key colors before rendering
    (svglib cannot draw them) and painted back in afterwards."""
    try:
        import re as _re
        import tempfile as _tf
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        src = str(svg_path)
        text = Path(svg_path).read_text(encoding="utf-8")
        local_gradients = [dict(g) for g in gradient_info] if gradient_info else []
        transformed = False
        if local_gradients:
            text = _re.sub(r"<defs>.*?</defs>", "", text, flags=_re.S)
            for g in local_gradients:
                op_m = _re.search(
                    r'<g\b[^>]*fill="url\(#' + _re.escape(g["id"]) +
                    r'\)"[^>]*fill-opacity="([0-9.]+)"[^>]*>', text)
                if op_m:
                    g["opacity"] = float(op_m.group(1))
                text = text.replace('fill="url(#' + g["id"] + ')"',
                                    'fill="' + g["key"] + '"')
                # Gradient opacity is applied when the true ramp is painted;
                # remove it here so the placeholder key stays detectable.
                text = _re.sub(
                    r'(<g\b[^>]*fill="' + _re.escape(g["key"]) +
                    r'"[^>]*)\sfill-opacity="[0-9.]+"', r'\1', text)
            transformed = True
        if "opacity=" in text:
            text = _flatten_svg_opacity(text, bg=bg)
            transformed = True
        if transformed:
            tf = _tf.NamedTemporaryFile("w", suffix=".svg", delete=False,
                                        encoding="utf-8")
            tf.write(text)
            tf.close()
            src = tf.name
        d = svg2rlg(src)
        if not d or not d.width:
            return False
        s = size / float(d.width)
        d.scale(s, s)
        d.width *= s
        d.height *= s
        renderPM.drawToFile(d, str(png_path), fmt="PNG", bg=bg)
        if local_gradients and png_path.exists():
            _paint_gradients(png_path, local_gradients)
        if src != str(svg_path):
            Path(src).unlink(missing_ok=True)
        return png_path.exists()
    except Exception:
        return False


def validate_svg_stage_renders(before_svg: Path, after_svg: Path, stage: str,
                               gradient_info=None, *, render_cache=None,
                               render_size=None):
    """Renderer-backed transaction guard for SVG post-processing.

    Exact stages must be pixel-identical.  Annulus regularisation is allowed
    a sub-pixel boundary change only when the independent bidirectional ink
    comparison remains above its 99% gate.  When optional rendering packages
    are unavailable, the stage's stricter internal geometry/order invariants
    remain authoritative and the report says so explicitly.
    """

    validation_width = max(1, int(render_size or SELF_CHECK_MAX_SIDE))
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage)
    before_png = before_svg.with_name(before_svg.stem + f"-{safe_stage}-render.png")
    after_png = after_svg.with_name(after_svg.stem + f"-{safe_stage}-render.png")

    def _cache_key(svg_path):
        if render_cache is None:
            return None
        import hashlib
        gradient_payload = json.dumps(
            gradient_info or [], ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), default=str).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(b"ai-vector-cleanroom-stage-render-v1\0")
        digest.update(str(validation_width).encode("ascii"))
        digest.update(b"\0ffffff\0")
        digest.update(gradient_payload)
        digest.update(b"\0")
        # Path.write_text uses CRLF on Windows while the committed candidate
        # is written from UTF-8 bytes with LF.  XML normalises literal line
        # endings before parsing, so canonicalise them in the cache identity;
        # otherwise the same document is needlessly rendered twice.
        svg_bytes = Path(svg_path).read_bytes()
        svg_bytes = svg_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(svg_bytes)
        return digest.hexdigest()

    def _render(svg_path, png_path):
        key = _cache_key(svg_path)
        if key is not None:
            cached = render_cache.get(key)
            if isinstance(cached, bytes) and cached:
                png_path.write_bytes(cached)
                return True, True
        rendered = render_svg_png(
            svg_path, png_path, size=validation_width,
            gradient_info=gradient_info)
        if rendered and key is not None and png_path.is_file():
            render_cache[key] = png_path.read_bytes()
        return rendered, False

    try:
        rendered_before, before_cache_hit = _render(before_svg, before_png)
        rendered_after, after_cache_hit = _render(after_svg, after_png)
        if not (rendered_before and rendered_after):
            return {
                "accepted": True,
                "external_render_check": "unavailable",
                "validation_level": "internal_invariants",
                "reason": "optional SVG renderer unavailable",
                "validation_render_width_px": validation_width,
                "render_cache_hits": {
                    "before": before_cache_hit,
                    "after": after_cache_hit,
                },
            }
        from annulus_detector import compare_rendered_pngs
        exact = stage.endswith("_exact")
        metrics = compare_rendered_pngs(
            before_png, after_png, tolerance_px=0 if exact else 1)
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
        metrics["external_render_check"] = "completed"
        metrics["validation_level"] = "renderer_and_internal_invariants"
        metrics["validation_render_width_px"] = validation_width
        metrics["render_cache_hits"] = {
            "before": before_cache_hit,
            "after": after_cache_hit,
        }
        return metrics
    finally:
        before_png.unlink(missing_ok=True)
        after_png.unlink(missing_ok=True)


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
    ref = Image.open(reference_png).convert("RGBA")
    ren = Image.open(render_png).convert("RGB")
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
    base = Image.new("RGB", (tw, th), (255, 255, 255))
    base.paste(ref, (0, 0), ref)
    a = np.asarray(ren, dtype=np.int16)
    b = np.asarray(base, dtype=np.int16)

    if not foreground_only:
        # Diagnostic whole-canvas score. Candidate selection never relies on
        # this diluted number when a foreground score exists.
        err = np.abs(a - b).max(2).astype(np.float32)
        return float(np.clip(1.0 - err / 128.0, 0.0, 1.0).mean() * 100)

    alpha = np.asarray(ref.getchannel("A"), dtype=np.int16)
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
    src_ink = ((src_rgb_strength >= ink_threshold)
               | (src_alpha_strength >= alpha_threshold))
    ren_ink = np.abs(a - bgc).max(2) >= ink_threshold
    if not src_ink.any():
        details = {"score": None, "recall": None, "precision": None,
                   "coverage_f1": None, "color_fidelity": None,
                   "source_ink_pixels": 0,
                   "render_ink_pixels": int(ren_ink.sum()),
                   "ink_threshold": ink_threshold}
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
        src_support = np.zeros((th, tw), dtype=np.uint8)
        ren_support = np.zeros((th, tw), dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted_a = _shift(a, dy, dx, 255)
            shifted_am = _shift(ren_ink, dy, dx, False)
            src_err = np.abs(shifted_a - b).max(2).astype(np.float32)
            best_src = np.minimum(
                best_src, np.where(shifted_am, src_err, 256.0))

            shifted_b = _shift(b, dy, dx, 255)
            shifted_bm = _shift(src_ink, dy, dx, False)
            ren_err = np.abs(a - shifted_b).max(2).astype(np.float32)
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
        # Render at the comparison resolution when possible.  Rendering every
        # small logo at 2048 px and then shrinking it back blurred an exact
        # one-pixel stroke into grey antialias rows before scoring.
        from PIL import Image
        with Image.open(source_png) as _source:
            sw, sh = _source.size
        scale = min(1.0, SELF_CHECK_MAX_SIDE / max(sw, sh))
        render_width = max(1, int(round(sw * scale)))
        if not render_svg_png(svg_path, tmp, size=render_width,
                              bg=0xffffff, gradient_info=gradient_info):
            return out
        try:
            out["flat"] = _match_percent(tmp, flat_png)
        except Exception:
            pass
        try:
            out["source"] = _match_percent(tmp, source_png)
            fg = _match_percent(tmp, source_png, foreground_only=True,
                                return_details=True)
            out["foreground"] = fg["score"]
            out["foreground_recall"] = fg["recall"]
            out["foreground_precision"] = fg["precision"]
            out["foreground_coverage_f1"] = fg["coverage_f1"]
            out["foreground_color_fidelity"] = fg["color_fidelity"]
            out["source_ink_pixels"] = fg["source_ink_pixels"]
            out["render_ink_pixels"] = fg["render_ink_pixels"]
            out["ink_threshold"] = fg["ink_threshold"]
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
            before_render, after_render)
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
            f"{proposal.get('repair_count', 0)} isolated missing component(s) "
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
        f"{st.get('gradients', 0)} gradients · {st.get('nodes', 0)} nodes")
    rejected = (acceptance_status == "rejected"
                or visual_acceptance_status == "rejected")
    manual_review = (manual_review_required
                     or acceptance_status != "accepted")
    gate_class = ("rejected" if rejected
                  else "manual" if manual_review else "accepted")
    if rejected:
        gate_text = ("未達標：此檔只供診斷，請勿交付設計師或客戶。"
                     "請查看熱區後重跑或改採人工描繪")
    elif manual_review:
        visual_text = ("通過" if visual_acceptance_status == "accepted"
                       else "需檢查")
        edit_text = ("通過" if editability_status == "accepted"
                     else "需檢查")
        gate_text = (f"需人工確認：外觀 {visual_text}；可編輯性 {edit_text}。"
                     "請勿只看總分直接交付")
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
    page = REVIEW_TEMPLATE
    page = page.replace("__TITLE__", html.escape(name))
    page = page.replace(
        "__SCORES__",
        f"source {_s('source')} · foreground {_s('foreground')} · "
        f"flat {_s('flat')}{detail_text}{edit_score_text}"
        f"{automation_text}{human_text}")
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
 li.innerHTML='<span class="sev '+cls+'">'+Math.round(s.severity*100)+'%</span> 區塊 '+(i+1);
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
            layers.append({"name": name, "type": "linearGradient",
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
        gradient_resources.append({
            "name": gradient.get("id", "gradient"),
            "type": "linearGradient",
            "id": gradient.get("id", ""),
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
        "max_size": "處理上限",
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
        acceptance_line = "rejected：未達標，僅保留作診斷，請勿交付"
        acceptance_warning = (
            "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "未達標／勿交付：外觀品質閘門偵測到多項或嚴重失守。\n"
            "請先開啟 review.html 查看差異，再重跑或改採人工描繪。\n"
            "SVG 雖為真正向量路徑，但不代表本次近似結果可用。\n"
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
Beta.5 fidelity, topology and editability enhancements
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
        f"{int(structure.get('nodes', structure.get('nodes_total', 0)) or 0)} nodes.\n"
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


def process_one(img_path: Path, out_base: str, args, output_dir: Path):
    from PIL import Image

    from trace_engine import _prepare_image
    from clean_base import build_clean_base
    from svg_postprocess import atomic_replace_bytes

    warnings = []
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
    }

    # 1) Source reference for review.  Candidate validation must use the same
    # background mode as that candidate: scoring an auto-cleaned logo against
    # the deliberately retained AI paper glow mistakes removable background
    # texture for lost vector detail.  Low-contrast enclosed marks still remain
    # in the auto-cleaned reference and have their ordinary source-ink score.
    clean_img, _sz, removed = _prepare_image(
        img_path, max_size=0, background=args.background,
        white_threshold=args.white_threshold, alpha_threshold=12)
    ref_png = deliver / "source_reference.png"
    clean_img.save(ref_png)
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
            metric_img.save(path)
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
            canonical.save(path)
            metric_refs[hole_key] = path
        return metric_refs[hole_key]

    metric_ref = _metric_reference(requested_options)

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
                              flat_out=flat_chk)
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
        except Exception as exc:
            public["error"] = str(exc)[:240]
        candidates.append(public)

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
    if base_item is not None:
        base_quality = base_item[1]
        base_stats = base_item[3]
        base_visual_status = _evaluate_visual_gate(base_item[4])["status"]
        element_count = max(
            1, base_stats.n_paths + base_stats.n_strokes
            + base_stats.n_native + base_stats.n_gradients)
        free_closed = any(
            detail.get("closed") and not detail.get("primitive")
            for detail in base_stats.stroke_info)
        structure_risk = (
            base_stats.n_nodes > 24 * element_count
            or free_closed)
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
        # measurable visual or structural effect.  The tea logo used to spend half
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
    selection_policy["matrix_strategy"] = matrix_strategy
    selection_policy["evaluated_candidates"] = len(candidates)
    selection_policy["base_structure_risk"] = bool(structure_risk)
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
    clean_img.save(ref_png)
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
        svg_path, validator=render_validator, work_dir=deliver)

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
            svg_path, validator=render_validator, work_dir=deliver,
            enable_annulus=False)
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

    # Portable paint-role resources keep explicit SVG paints authoritative.
    paint_manifest_path = deliver / f"{out_base}_paint_roles.json"
    paint_manifest = None
    paint_role_report = {"status": "unavailable"}
    try:
        paint_manifest, paint_role_report = attach_paint_roles(
            svg_path, paint_manifest_path, validator=render_validator,
            work_dir=deliver)
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
    visual_acceptance_status = visual_gate["status"]
    visual_review_required = visual_acceptance_status != "accepted"
    selected_detail_grid = scores.get("detail_grid") or {}
    detail_p10 = selected_detail_grid.get("p10_score_percent")
    if visual_acceptance_status == "rejected":
        reason = "; ".join(visual_gate["reasons"])
        warnings.append(
            "VISUAL NOT ACCEPTED: this output is diagnostic only and must not "
            f"be handed off without rework ({reason})")
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

    # Visual similarity and designer handoff are separate gates. A file can
    # be a genuine, high-fidelity SVG yet still couple hundreds of unrelated
    # shapes into giant compound paths. Never report that as fully accepted.
    try:
        from editability_audit import audit_editability
        editability = audit_editability(svg_path, {
            **final_structure,
            "designer_operations": designer_operations,
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

    if visual_acceptance_status == "rejected" or editability_status == "rejected":
        acceptance_status = "rejected"
    elif visual_review_required or editability_status != "accepted":
        acceptance_status = "manual_review"
    else:
        acceptance_status = "accepted"
    manual_review_required = acceptance_status != "accepted"

    # Keep the SVG self-describing even when it is copied away from its ZIP.
    # The JSON is XML-escaped text, readable by editors and scripts without
    # introducing proprietary namespaces or non-vector payloads.
    svg_text = svg_path.read_text(encoding="utf-8")
    metadata_payload = {
        "tool": "AI Vector Cleanroom",
        "version": TOOL_VERSION,
        "tool_version": TOOL_VERSION,
        "options_requested": requested_options,
        "options_effective": effective_options,
        "visual_acceptance_status": visual_acceptance_status,
        "visual_gate": visual_gate,
        "editability_status": editability_status,
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
                    else "Vector generated — NOT ACCEPTED / DO NOT HAND OFF")
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
          f"{final_structure['nodes']} nodes total")
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
    hotspots = scores.get("hotspots") or []
    detail_grid = scores.get("detail_grid")
    chk_render.unlink(missing_ok=True)
    for _metric_path in metric_refs.values():
        _metric_path.unlink(missing_ok=True)

    # 3) Preview. Never silently pass off the source as an SVG render.
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
            for detail in stats.gradient_info
        ],
        "nodes_total": final_structure["nodes"],
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
        "acceptance_status": acceptance_status,
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
    zip_path = output_dir / f"result_{out_base}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(deliver.iterdir()):
            z.write(f, arcname=f"{deliver.name}/{f.name}")
    mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  Result package: {zip_path.name} ({mb:.2f} MB)")
    return zip_path


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="AI Vector Cleanroom — bitmap logo/icon to editable SVG draft")
    ap.add_argument("--input", type=Path, default=BASE / "input",
                    help="input folder (default: ./input)")
    ap.add_argument("--output", type=Path, default=BASE / "output",
                    help="output folder (default: ./output)")
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
    if args.input.exists() and not args.input.is_dir():
        ap.error(f"--input is not a folder: {args.input}")


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    if args.no_geometry:
        args.geometry = "off"

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


if __name__ == "__main__":
    sys.exit(main())
