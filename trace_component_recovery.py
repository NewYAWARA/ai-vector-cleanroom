"""Recover only small, wholly omitted source compartments after cutout tracing.

This stage preserves admitted source evidence. It does not classify a mark as
intentional, infer transparent holes from white paint, or simplify a design.
Only complete missing connected components are appended; existing contours are
not retraced. The caller must retain its normal source/render/topology gates.
"""
from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from svg_renderer import render_svg_reference
from trace_engine import binary_mask_to_compound_path

NS = "http://www.w3.org/2000/svg"


def _components(mask, diagonal=False):
    height, width = mask.shape
    seen = np.zeros_like(mask)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        queue = deque([(int(sy), int(sx))])
        seen[sy, sx] = True
        points = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        yield np.asarray(points, dtype=np.int32)


def _composite(rgba, background):
    alpha = rgba[:, :, 3:4].astype(float) / 255.0
    return rgba[:, :, :3].astype(float) * alpha + background * (1 - alpha)


def _render(payload, folder, width, background):
    svg, png = folder / "candidate.svg", folder / "candidate.png"
    svg.write_text(payload, encoding="utf-8")
    render_svg_reference(svg, png, width=width, background=None)
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    return _composite(rgba, background)


def recover_missing_source_components(raw_svg, source_rgba, *, excluded_mask=None,
                                      maximum_component_area=256, maximum_recoveries=8):
    """Return (SVG text, evidence) for fully missing coherent small regions.

    ``source_rgba`` must be the exact VTracer input in its pixel coordinate
    system. Gradient and stroke ownership should be excluded by the caller.
    Foreground support uses contrast >=24, with >=2 pixels at contrast >=64.
    Enclosed background compartments are handled separately and retain their
    source paint. One-pixel noise, partial matches and multicolour marks abstain.
    """
    source = np.asarray(source_rgba)
    if source.ndim != 3 or source.shape[2] != 4 or source.dtype != np.uint8:
        raise ValueError("source_rgba must be the uint8 HxWx4 VTracer input")
    height, width = source.shape[:2]
    excluded = np.zeros((height, width), dtype=bool) if excluded_mask is None else np.asarray(excluded_mask, dtype=bool)
    if excluded.shape != (height, width):
        raise ValueError("excluded_mask shape mismatch")
    root = ET.fromstring(raw_svg)
    if int(float(root.get("width", "0"))) != width or int(float(root.get("height", "0"))) != height:
        raise ValueError("raw SVG and source pixel coordinate systems differ")
    report = {"schema": "ai-vector-cleanroom.trace-component-recovery/v1", "status": "no_change",
              "policy": "small_wholly_missing_source_components_only",
              "source_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
              "source_svg_sha256": hashlib.sha256(raw_svg.encode()).hexdigest(),
              "maximum_component_area": int(maximum_component_area), "maximum_recoveries": int(maximum_recoveries),
              "recovered_count": 0, "records": [],
              "hole_scope": "source_background_paint_preserved_without_transparency_inference",
              "downstream_render_validation_required": True}
    if not raw_svg.strip() or min(height, width) < 8 or excluded.all():
        return raw_svg, report
    border = np.concatenate((source[0, :, :3], source[-1, :, :3], source[:, 0, :3], source[:, -1, :3]))
    background = np.median(border.astype(float), axis=0)
    opaque = source[:, :, 3] >= 250
    rgb = _composite(source, background)
    contrast = np.max(np.abs(rgb - background), axis=2)
    support = opaque & (contrast >= 24)
    regions = []
    for kind, mask, diagonal in (("foreground_component", support, False),
                                  ("enclosed_background_compartment", opaque & (contrast < 24), True)):
        for points in _components(mask, diagonal):
            area = len(points)
            if not 4 <= area <= int(maximum_component_area):
                continue
            ys, xs = points[:, 0], points[:, 1]
            if xs.min() < 2 or ys.min() < 2 or xs.max() >= width - 2 or ys.max() >= height - 2:
                continue
            if excluded[ys, xs].any():
                continue
            if kind == "foreground_component" and np.count_nonzero(contrast[ys, xs] >= 64) < 2:
                continue
            regions.append((kind, points))
    if not regions:
        return raw_svg, report
    with tempfile.TemporaryDirectory(prefix="avc-recover-") as temporary:
        folder = Path(temporary)
        rendered = _render(raw_svg, folder, width, background)
        if rendered.shape != rgb.shape:
            raise ValueError("renderer/source shape mismatch")
        for kind, points in regions:
            if report["recovered_count"] >= int(maximum_recoveries):
                break
            ys, xs = points[:, 0], points[:, 1]
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            source_colours = rgb[ys, xs]
            if kind == "foreground_component":
                core_contrast = contrast[ys, xs]
                core = source_colours[core_contrast >= 0.9 * core_contrast.max()]
                colour = np.median(core, axis=0)
                direction = colour - background
                t = np.clip(((source_colours - background) @ direction) / max(float(direction @ direction), 1.0), 0, 1)
                residual = np.max(np.abs(source_colours - (background + t[:, None] * direction)), axis=1)
                if np.quantile(residual, 0.95) > 18:
                    continue
                missing = np.max(np.abs(rendered[ys, xs] - background), axis=1) < 24
            else:
                colour = np.median(source_colours, axis=0)
                if np.max(np.abs(source_colours - colour)) > 24:
                    continue
                missing = np.max(np.abs(rendered[ys, xs] - colour), axis=1) >= 64
            if float(np.mean(missing)) < 0.95:
                continue
            component = np.zeros((height, width), dtype=bool)
            component[ys, xs] = True
            traced = binary_mask_to_compound_path(component, simplify=0.0, min_area=1.0, smooth=0.0, curve=0.0)
            path = traced.get("path") if isinstance(traced, dict) else traced
            if not path:
                continue
            identifier = f'avc-source-recovery-{report["recovered_count"] + 1}'
            while any(node.get("id") == identifier for node in root.iter()):
                identifier += "x"
            paint = "#{:02x}{:02x}{:02x}".format(*np.clip(np.rint(colour), 0, 255).astype(int))
            proposal = ET.SubElement(root, f"{{{NS}}}path", {"id": identifier, "d": path,
                "fill": paint, "fill-rule": "evenodd", "data-avc-source-recovery": kind})
            proposed = ET.tostring(root, encoding="unicode")
            after = _render(proposed, folder, width, background)
            before_error = float(np.mean(np.abs(rendered[y0:y1, x0:x1] - rgb[y0:y1, x0:x1])))
            after_error = float(np.mean(np.abs(after[y0:y1, x0:x1] - rgb[y0:y1, x0:x1])))
            protected = np.ones((height, width), dtype=bool)
            protected[max(0, y0-1):min(height, y1+1), max(0, x0-1):min(width, x1+1)] = False
            outside_changed = bool(np.any(np.abs(after[protected] - rendered[protected]) > 1))
            if outside_changed or after_error > 0.75 * before_error:
                root.remove(proposal)
                continue
            rendered = after
            report["recovered_count"] += 1
            report["records"].append({"id": identifier, "kind": kind, "source_pixels": len(points),
                "bbox_xyxy": [x0, y0, x1, y1], "paint": paint, "missing_share": round(float(np.mean(missing)), 6),
                "before_mean_rgb_error": round(before_error, 6), "after_mean_rgb_error": round(after_error, 6),
                "outside_local_box_unchanged": True, "geometry": "exact_binary_mask_compound_path"})
    if not report["recovered_count"]:
        return raw_svg, report
    result = ET.tostring(root, encoding="unicode")
    report["status"] = "recovered"
    report["output_svg_sha256"] = hashlib.sha256(result.encode()).hexdigest()
    return result, report
