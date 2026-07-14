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
TOOL_VERSION = "v3-codex-beta.3"
MATERIAL_FALLBACK_GAIN = 1.0
RECONSTRUCTION_KEYS = ("strokes", "gradients", "geometry")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_inputs(input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTS)


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
    best_quality = max(item[1] for item in viable)
    tied = [item for item in viable
            if item[1] >= best_quality - material_gain]

    def retention(item):
        options = item[2]
        return sum(
            1 for key in RECONSTRUCTION_KEYS
            if requested_options.get(key) not in (None, "off")
            and options.get(key) == requested_options.get(key)
        )

    selected = max(tied, key=lambda item: (retention(item), item[0], item[1]))
    return selected, {
        "material_visual_gain_required": material_gain,
        "best_visual_quality": best_quality,
        "selected_visual_quality": selected[1],
        "selected_requested_features_retained": retention(selected),
        "requested_features_total": sum(
            1 for key in RECONSTRUCTION_KEYS
            if requested_options.get(key) not in (None, "off")),
        "policy": "preserve_requested_features_within_visual_tie",
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
    for g in gradient_info:
        vb_w = float(g["viewbox"][0])
        f = w / vb_w if vb_w else 1.0
        key = np.array([int(g["key"][1:3], 16), int(g["key"][3:5], 16),
                        int(g["key"][5:7], 16)], dtype=np.int16)
        mask = (np.abs(arr - key).max(axis=2) <= 10)
        if not mask.any():
            continue
        # absorb the antialiased fringe around the keyed region
        near = np.abs(arr - key).max(axis=2) <= 120
        for _ in range(3):
            grow = mask.copy()
            grow[1:, :] |= mask[:-1, :]
            grow[:-1, :] |= mask[1:, :]
            grow[:, 1:] |= mask[:, :-1]
            grow[:, :-1] |= mask[:, 1:]
            mask = (grow & near) | mask
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
                               gradient_info=None):
    """Renderer-backed transaction guard for SVG post-processing.

    Exact stages must be pixel-identical.  Annulus regularisation is allowed
    a sub-pixel boundary change only when the independent bidirectional ink
    comparison remains above its 99% gate.  When optional rendering packages
    are unavailable, the stage's stricter internal geometry/order invariants
    remain authoritative and the report says so explicitly.
    """

    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", stage)
    before_png = before_svg.with_name(before_svg.stem + f"-{safe_stage}-render.png")
    after_png = after_svg.with_name(after_svg.stem + f"-{safe_stage}-render.png")
    try:
        rendered_before = render_svg_png(
            before_svg, before_png, size=SELF_CHECK_MAX_SIDE,
            gradient_info=gradient_info)
        rendered_after = render_svg_png(
            after_svg, after_png, size=SELF_CHECK_MAX_SIDE,
            gradient_info=gradient_info)
        if not (rendered_before and rendered_after):
            return {
                "accepted": True,
                "external_render_check": "unavailable",
                "validation_level": "internal_invariants",
                "reason": "optional SVG renderer unavailable",
            }
        from annulus_detector import compare_rendered_pngs
        exact = stage.endswith("_exact")
        metrics = compare_rendered_pngs(
            before_png, after_png, tolerance_px=0 if exact else 1)
        if exact:
            import hashlib
            from PIL import Image
            with Image.open(before_png) as image:
                before_pixels = image.convert("RGBA").tobytes()
            with Image.open(after_png) as image:
                after_pixels = image.convert("RGBA").tobytes()
            before_hash = hashlib.sha256(before_pixels).hexdigest()
            after_hash = hashlib.sha256(after_pixels).hexdigest()
            metrics["exact_before_pixel_sha256"] = before_hash
            metrics["exact_after_pixel_sha256"] = after_hash
            metrics["exact_pixel_array_equal"] = before_hash == after_hash
            metrics["accepted"] = metrics["exact_pixel_array_equal"]
            metrics["required_equivalence"] = "pixel_array_exact_at_validation_resolution"
        else:
            metrics["required_equivalence"] = "bidirectional_1px_99_percent"
        metrics["external_render_check"] = "completed"
        metrics["validation_level"] = "renderer_and_internal_invariants"
        return metrics
    finally:
        before_png.unlink(missing_ok=True)
        after_png.unlink(missing_ok=True)


SELF_CHECK_MAX_SIDE = 2048


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
           "ink_threshold": None, "detail_grid": None, "hotspots": []}
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
    manual_review = (manual_review_required
                     or acceptance_status == "manual_review")
    gate_class = "manual" if manual_review else "accepted"
    if manual_review:
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
    gradient_index = 0
    for name, value in (getattr(stats, "palette", ()) or ()):
        if str(name).lower().startswith("gradient") and gradient_index < len(gradients):
            gradient = gradients[gradient_index]
            gradient_index += 1
            layers.append({"name": name, "type": "linearGradient",
                           "gradient_id": gradient.get("id", "")})
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
    if acceptance_status == "manual_review":
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
Beta.3 editability enhancements
-------------------------------
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

    # 1) Source reference for review plus a stricter validation reference.
    # The ordinary 220 light-background threshold can classify a deliberate
    # #dddddd/#ebebeb mark as background. Validation therefore keeps the raw
    # background; the chosen candidate still follows the requested option.
    clean_img, _sz, removed = _prepare_image(
        img_path, max_size=0, background=args.background,
        white_threshold=args.white_threshold, alpha_threshold=12)
    ref_png = deliver / "source_reference.png"
    clean_img.save(ref_png)
    msg = " (outer light/checker background removed)" if removed else ""
    print(f"  Source reference OK{msg}")
    metric_ref = deliver / "_metric_reference.png"
    metric_threshold = max(args.white_threshold, 250)
    metric_img, _metric_sz, _metric_removed = _prepare_image(
        img_path, max_size=0, background="keep",
        white_threshold=metric_threshold, alpha_threshold=2)
    metric_img.save(metric_ref)

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
        sc = self_check(svg_path, flat_chk, metric_ref,
                        gradient_info=st.gradient_info,
                        viewbox=st.viewbox)
        return st, sc

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
            st, sc = _build(opts)
            structure = _structure(st)
            quality = _eff(sc)
            rank = ((0.90 * quality + 0.10 * structure["score"])
                    if quality is not None else -1.0)
            public_scores = {key: value for key, value in sc.items()
                             if key != "hotspots"}
            public.update({"status": "ok", "scores": public_scores,
                           "quality_score": quality,
                           "structure": structure,
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
                             svg_snapshot, flat_snapshot))
        except Exception as exc:
            public["error"] = str(exc)[:240]
        candidates.append(public)

    # Staged evaluation keeps normal logos at one render. Expand the full
    # disabling matrix only when fidelity or editability structure is risky.
    _attempt(args.background, {})
    base_item = next((item for item in internal
                      if item[2] == requested_options), None)
    structure_risk = False
    base_quality = None
    if base_item is not None:
        base_quality = base_item[1]
        base_stats = base_item[3]
        element_count = max(
            1, base_stats.n_paths + base_stats.n_strokes
            + base_stats.n_native + base_stats.n_gradients)
        free_closed = any(
            detail.get("closed") and not detail.get("primitive")
            for detail in base_stats.stroke_info)
        structure_risk = (
            base_stats.n_nodes > 80
            or base_stats.n_nodes > 24 * element_count
            or free_closed)
    expand_primary = (base_item is None or base_quality is None
                      or base_quality < 88.0 or structure_risk)
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
        single_quality = max(
            (item[1] for item in internal
             if item[1] is not None and item[2] != requested_options),
            default=-1.0)
        expand_combinations = (
            base_item is None or base_quality is None or base_quality < 88.0
            or single_quality >= base_quality + MATERIAL_FALLBACK_GAIN)
        if expand_combinations:
            matrix_strategy = "full_disable_matrix"
            for ov in variants:
                if len(ov) >= 2:
                    _attempt(args.background, ov)

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
    (_rank, _quality, effective_options, _st, _sc, chosen_public,
     selected_svg, selected_flat) = selected_item
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
    render_validator = lambda before, after, stage: validate_svg_stage_renders(
        before, after, stage, gradient_info=stats.gradient_info)
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

    selected_detail_grid = scores.get("detail_grid") or {}
    detail_p10 = selected_detail_grid.get("p10_score_percent")
    detail_cells = int(selected_detail_grid.get("eligible_cells") or 0)
    detail_review_required = (
        detail_cells >= 2 and detail_p10 is not None and detail_p10 < 80.0)
    visual_review_required = e_final < 80.0 or detail_review_required
    visual_acceptance_status = (
        "manual_review" if visual_review_required else "accepted")
    if e_final < 80.0:
        warnings.append(
            f"VISUAL REVIEW REQUIRED: foreground quality {e_final:.1f}% is "
            "below the 80% automatic acceptance gate")
        print(f"    VISUAL REVIEW REQUIRED: {e_final:.1f}% is below 80%.")
    if detail_review_required:
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

    manual_review_required = (
        visual_review_required or editability_status != "accepted")
    acceptance_status = "manual_review" if manual_review_required else "accepted"

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
        "editability_status": editability_status,
        "acceptance_status": acceptance_status,
        "editability_enhancements": {
            key: value.get("status")
            for key, value in enhancement_report.get("stages", {}).items()
        },
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
    print(f"  Vector OK: {final_structure['groups']} final SVG groups; "
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

    # 2b) Render back and compare against the references.
    chk_render = deliver / "_render_check.png"
    scores = self_check(svg_path, flat_chk, metric_ref,
                        gradient_info=stats.gradient_info,
                        keep_render=chk_render, viewbox=stats.viewbox)
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
    metric_ref.unlink(missing_ok=True)

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
        "hotspots": hotspots,
        "candidates": candidates,
        "candidate_selection_policy": selection_policy,
        "auto_fallback": chosen,
        "visual_acceptance_status": visual_acceptance_status,
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
