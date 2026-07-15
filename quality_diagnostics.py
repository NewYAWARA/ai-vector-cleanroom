"""Local-detail quality diagnostics for raster-to-vector output.

This module deliberately measures *source ink*, not image alpha.  JPEG and
other opaque, white-background images have alpha=255 everywhere; treating
that as foreground makes a large blank canvas hide a missing hairline or dot.

The public :func:`compute_quality_diagnostics` function estimates the source
background from the full image boundary, builds an adaptive ink ROI, and then
scores only grid cells containing that ink.  Each source-ink pixel may match a
rendered pixel in its 3x3 neighbourhood, so a harmless one-pixel rasterisation
phase shift is tolerated without overlooking genuinely missing detail.

Only Pillow and NumPy are required.  In particular, no SciPy morphology or
distance-transform dependency is used, keeping the portable build small.
"""

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image


ImageInput = Union[str, Path, Image.Image]


def _open_rgba(image: ImageInput) -> Image.Image:
    """Return a detached RGBA image from a path or a Pillow image."""

    if isinstance(image, Image.Image):
        return image.convert("RGBA").copy()
    with Image.open(image) as opened:
        return opened.convert("RGBA")


def _composite_white(image: Image.Image) -> np.ndarray:
    """Return visible RGB values after compositing transparency onto white."""

    rgba = image.convert("RGBA")
    base = Image.new("RGB", rgba.size, (255, 255, 255))
    base.paste(rgba, (0, 0), rgba)
    return np.asarray(base, dtype=np.int16)


def _boundary_pixels(rgb: np.ndarray) -> np.ndarray:
    """Collect the complete perimeter of an HxWx3 RGB array."""

    height, width = rgb.shape[:2]
    if height == 1:
        return rgb[0].reshape(-1, 3)
    if width == 1:
        return rgb[:, 0].reshape(-1, 3)
    return np.concatenate(
        (rgb[0, :], rgb[-1, :], rgb[1:-1, 0], rgb[1:-1, -1]), axis=0
    )


def _neighbour_count(mask: np.ndarray) -> np.ndarray:
    """Count true pixels in each clipped 3x3 neighbourhood."""

    height, width = mask.shape
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    count = np.zeros((height, width), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            count += padded[dy : dy + height, dx : dx + width]
    return count


def source_ink_roi(
    source: ImageInput,
    *,
    min_threshold: float = 6.0,
    max_threshold: float = 24.0,
) -> Dict[str, Any]:
    """Estimate the visible source-ink region from the image boundary.

    The background colour is the per-channel perimeter median.  The ink
    threshold follows measured 95th-percentile boundary noise, bounded so
    normal JPEG ringing is ignored while light-grey logo details remain
    detectable.  Weak isolated noise is removed, but a strongly contrasting
    one-pixel dot is retained.

    The returned ``rgb``, ``strength`` and ``mask`` entries are NumPy arrays;
    metadata entries are JSON-safe scalars/lists.
    """

    rgba = _open_rgba(source)
    rgb = _composite_white(rgba)
    border = _boundary_pixels(rgb)
    background = np.median(border, axis=0).astype(np.float32)
    border_noise = np.abs(border.astype(np.float32) - background).max(axis=1)
    measured_noise = float(np.percentile(border_noise, 95))
    threshold = float(
        max(min_threshold, min(max_threshold, measured_noise + 3.0))
    )

    strength = np.abs(rgb.astype(np.float32) - background).max(axis=2)
    raw_mask = strength >= threshold

    # A thin line has adjacent support even when it is only one pixel wide.
    # A genuinely isolated, high-contrast dot is meaningful too; weak lone
    # pixels are much more likely to be JPEG noise than editable logo detail.
    support = _neighbour_count(raw_mask)
    strong_isolated = strength >= max(threshold * 1.75, threshold + 12.0)
    mask = raw_mask & ((support >= 2) | strong_isolated)

    return {
        "rgb": rgb,
        "strength": strength,
        "mask": mask,
        "background_rgb": [float(v) for v in background],
        "boundary_noise_p95": measured_noise,
        "ink_threshold": threshold,
        "source_ink_pixels": int(mask.sum()),
        "source_ink_fraction": float(mask.mean()) if mask.size else 0.0,
    }


def _shift(array: np.ndarray, dy: int, dx: int, fill: Any) -> np.ndarray:
    """Shift an array without wrapping pixels across the opposite edge."""

    height, width = array.shape[:2]
    out = np.empty_like(array)
    out[...] = fill
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    dy0, dy1 = sy0 + dy, sy1 + dy
    dx0, dx1 = sx0 + dx, sx1 + dx
    out[dy0:dy1, dx0:dx1] = array[sy0:sy1, sx0:sx1]
    return out


def _viewbox_geometry(
    viewbox: Optional[Sequence[float]], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Normalise [width,height] or SVG [x,y,width,height] coordinates."""

    if viewbox is None:
        return 0.0, 0.0, float(width), float(height)
    if len(viewbox) == 2:
        return 0.0, 0.0, float(viewbox[0]), float(viewbox[1])
    if len(viewbox) == 4:
        return tuple(float(v) for v in viewbox)  # type: ignore[return-value]
    raise ValueError("viewbox must be [width, height] or [x, y, width, height]")


def structural_core_threshold(ink_threshold: float) -> float:
    """Return the contrast floor used for structural continuity labels.

    The repair planner must rebuild exactly the same component labels as the
    diagnostic report.  Keeping this policy in one public helper prevents a
    threshold change from invalidating stored component IDs, areas or boxes.
    """

    threshold = float(ink_threshold)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("ink_threshold must be a finite non-negative number")
    return max(threshold * 2.0, threshold + 12.0)


def _component_topology(src_mask: np.ndarray, ren_mask: np.ndarray,
                        *, max_examples: int = 20,
                        failure_score_below: float = 90.0,
                        viewbox: Optional[Sequence[float]] = None
                        ) -> Dict[str, Any]:
    """Measure whether each meaningful source component stays continuous.

    Pixel recall can remain high when one glyph stem or circular arc is split
    into several disconnected vector fragments.  Label source ink normally,
    label a one-pixel-dilated render (the same raster phase tolerance used by
    the local grid), then ask how much of each source component is represented
    by its single largest render component.
    """
    from stroke_engine import connected_components

    height, width = src_mask.shape
    vb_x, vb_y, vb_w, vb_h = _viewbox_geometry(viewbox, width, height)
    scale_x = vb_w / width if width else 1.0
    scale_y = vb_h / height if height else 1.0
    failure_score_below = float(failure_score_below)
    schema = {
        "schema": "ai-vector-cleanroom.component-topology/v1",
        # stroke_engine.connected_components currently labels diagonal run
        # contacts together.  Record the implemented contract explicitly; its
        # historical docstring incorrectly called this 4-connectivity.
        "connectivity": "8-connected",
        "render_tolerance": {
            "pixels": 1,
            "neighbourhood": "3x3",
            "operation": "dilation_before_component_labelling",
        },
        "measurement_size_px": [int(width), int(height)],
        "viewbox": [round(float(vb_x), 3), round(float(vb_y), 3),
                    round(float(vb_w), 3), round(float(vb_h), 3)],
        "failure_score_below": failure_score_below,
        "example_sort": "score_percent_asc_area_px_desc",
    }

    src_labels, src_count = connected_components(src_mask)
    ren_tolerant = _neighbour_count(ren_mask) > 0
    ren_labels, _ = connected_components(ren_tolerant)
    source_ink = int(src_mask.sum())
    min_area = max(16, min(128, int(round(source_ink * 1e-4))))
    areas = np.bincount(src_labels.ravel(), minlength=src_count + 1)
    eligible = np.flatnonzero(areas >= min_area)
    eligible = eligible[eligible != 0]

    if not len(eligible):
        return {
            **schema,
            "eligible_components": 0,
            "minimum_component_area_px": min_area,
            "one_pixel_tolerance": True,
            "p10_score_percent": None,
            "worst_score_percent": None,
            "mean_score_percent": None,
            "coverage_p10_percent": None,
            "connectivity_p10_percent": None,
            "fragmented_components": 0,
            "failed_component_count": 0,
            "examples_total": 0,
            "examples_returned": 0,
            "examples_truncated": False,
            "examples": [],
            "failed_examples": [],
        }

    flat_source = src_labels.ravel()
    foreground_indices = np.flatnonzero(flat_source)
    order = np.argsort(flat_source[foreground_indices], kind="stable")
    grouped = foreground_indices[order]
    grouped_labels = flat_source[grouped]
    starts = np.searchsorted(grouped_labels, eligible, side="left")
    ends = np.searchsorted(grouped_labels, eligible, side="right")
    flat_render = ren_labels.ravel()
    items = []
    scores = []
    coverages = []
    connectivities = []
    fragmented = 0
    for label, start, end in zip(eligible, starts, ends):
        indices = grouped[start:end]
        area = int(len(indices))
        overlapping = flat_render[indices]
        counts = np.bincount(overlapping)
        nonzero = counts[1:] if len(counts) > 1 else np.empty(0, dtype=int)
        largest = int(nonzero.max()) if len(nonzero) else 0
        covered_pixels = int(nonzero.sum()) if len(nonzero) else 0
        coverage = 100.0 * covered_pixels / max(1, area)
        connectivity = 100.0 * largest / max(1, covered_pixels)
        score = 100.0 * largest / max(1, area)
        material_pixels = max(3, int(math.ceil(0.05 * area)))
        fragment_count = int((nonzero >= material_pixels).sum())
        if fragment_count >= 2:
            fragmented += 1
        yy, xx = np.divmod(indices, width)
        bbox_px = [int(xx.min()), int(yy.min()),
                   int(xx.max() - xx.min() + 1),
                   int(yy.max() - yy.min() + 1)]
        below_threshold = score < failure_score_below
        is_fragmented = fragment_count >= 2
        items.append({
            "source_component": int(label),
            "area_px": area,
            "score_percent": round(score, 3),
            "coverage_percent": round(coverage, 3),
            "connectivity_percent": round(connectivity, 3),
            "fragment_count": fragment_count,
            "failure_score_below": failure_score_below,
            "below_failure_threshold": bool(below_threshold),
            "fragmented": bool(is_fragmented),
            "bbox_px_format": "xywh",
            "bbox_px": bbox_px,
            "bbox_viewbox_format": "xywh",
            "bbox_viewbox": [
                round(vb_x + bbox_px[0] * scale_x, 3),
                round(vb_y + bbox_px[1] * scale_y, 3),
                round(bbox_px[2] * scale_x, 3),
                round(bbox_px[3] * scale_y, 3),
            ],
        })
        scores.append(score)
        coverages.append(coverage)
        connectivities.append(connectivity)

    values = np.asarray(scores, dtype=float)
    coverage_values = np.asarray(coverages, dtype=float)
    connectivity_values = np.asarray(connectivities, dtype=float)
    items.sort(key=lambda item: (item["score_percent"], -item["area_px"]))
    examples = items[:max_examples]
    failed_examples = [
        item for item in items
        if item["below_failure_threshold"] or item["fragmented"]
    ]
    return {
        **schema,
        "eligible_components": int(len(values)),
        "minimum_component_area_px": min_area,
        "one_pixel_tolerance": True,
        "p10_score_percent": round(float(np.percentile(values, 10)), 3),
        "worst_score_percent": round(float(values.min()), 3),
        "mean_score_percent": round(float(values.mean()), 3),
        "coverage_p10_percent": round(
            float(np.percentile(coverage_values, 10)), 3),
        "connectivity_p10_percent": round(
            float(np.percentile(connectivity_values, 10)), 3),
        "fragmented_components": int(fragmented),
        "failed_component_count": len(failed_examples),
        "examples_total": len(items),
        "examples_returned": len(examples),
        "examples_truncated": len(examples) < len(items),
        "examples": examples,
        # Unlike the compatibility `examples` sample, this is deliberately
        # complete so a small number of lost glyphs cannot disappear behind
        # the 20-record diagnostic cap.
        "failed_examples": failed_examples,
    }


def compute_quality_diagnostics(
    render: ImageInput,
    source: ImageInput,
    viewbox: Optional[Sequence[float]] = None,
    *,
    cell: int = 48,
    max_spots: int = 40,
    hotspot_score_below: float = 78.0,
) -> Dict[str, Any]:
    """Diagnose lost local detail and return ``hotspots`` + ``detail_grid``.

    Cell scores are source-directed because this diagnostic answers "which
    original details disappeared?"  A source ink pixel receives 65% geometry
    credit when rendered ink exists within one pixel and 35% colour credit
    according to the best RGB match in that same neighbourhood.  Wrong-colour
    geometry therefore cannot pass as a faithful detail, while antialias phase
    changes of one pixel do not create false alarms.

    Blank cells are excluded completely.  Aggregate p10/median/worst scores
    therefore cannot be inflated by adding more white canvas around a logo.
    ``severity`` remains in the 0..1 range expected by the existing review UI.
    """

    if cell < 1:
        raise ValueError("cell must be at least 1 pixel")
    if max_spots < 0:
        raise ValueError("max_spots cannot be negative")

    src_rgba = _open_rgba(source)
    ren_rgba = _open_rgba(render)
    if src_rgba.size != ren_rgba.size:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        src_rgba = src_rgba.resize(ren_rgba.size, resampling)

    roi = source_ink_roi(src_rgba)
    src_rgb = roi["rgb"]
    src_mask = roi["mask"]
    ren_rgb = _composite_white(ren_rgba)
    background = np.asarray(roi["background_rgb"], dtype=np.float32)
    threshold = float(roi["ink_threshold"])
    ren_strength = np.abs(ren_rgb.astype(np.float32) - background).max(axis=2)
    ren_mask = ren_strength >= threshold

    height, width = src_mask.shape
    best_error = np.full((height, width), 256.0, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted_rgb = _shift(ren_rgb, dy, dx, 255)
            shifted_mask = _shift(ren_mask, dy, dx, False)
            error = np.abs(shifted_rgb - src_rgb).max(axis=2).astype(np.float32)
            best_error = np.minimum(
                best_error, np.where(shifted_mask, error, 256.0)
            )

    covered = best_error < 256.0
    colour_similarity = np.clip(1.0 - best_error / 128.0, 0.0, 1.0)
    # Missing ink gets neither geometry nor colour credit.  Wrong-colour ink
    # may keep geometry credit but remains below the default hotspot threshold.
    pixel_score = 100.0 * (
        0.65 * covered.astype(np.float32) + 0.35 * colour_similarity
    )

    vb_x, vb_y, vb_w, vb_h = _viewbox_geometry(viewbox, width, height)
    scale_x = vb_w / width if width else 1.0
    scale_y = vb_h / height if height else 1.0
    cells = []
    hotspots = []
    for cy in range(0, height, cell):
        for cx in range(0, width, cell):
            y1, x1 = min(cy + cell, height), min(cx + cell, width)
            local_mask = src_mask[cy:y1, cx:x1]
            ink_pixels = int(local_mask.sum())
            if ink_pixels == 0:
                continue

            local_covered = covered[cy:y1, cx:x1]
            local_colour = colour_similarity[cy:y1, cx:x1]
            score = float(pixel_score[cy:y1, cx:x1][local_mask].mean())
            recall = float(local_covered[local_mask].mean() * 100.0)
            colour = float(local_colour[local_mask].mean() * 100.0)
            severity = float(np.clip(1.0 - score / 100.0, 0.0, 1.0))
            item = {
                "pixel_x": cx,
                "pixel_y": cy,
                "pixel_w": x1 - cx,
                "pixel_h": y1 - cy,
                "source_ink_pixels": ink_pixels,
                "score_percent": round(score, 3),
                "coverage_percent": round(recall, 3),
                "color_fidelity_percent": round(colour, 3),
                "severity": round(severity, 3),
            }
            cells.append(item)
            if score < hotspot_score_below:
                hotspots.append(
                    {
                        "x": round(vb_x + cx * scale_x, 1),
                        "y": round(vb_y + cy * scale_y, 1),
                        "w": round((x1 - cx) * scale_x, 1),
                        "h": round((y1 - cy) * scale_y, 1),
                        "severity": round(severity, 3),
                        "score_percent": round(score, 1),
                        "source_ink_pixels": ink_pixels,
                    }
                )

    scores = np.asarray([entry["score_percent"] for entry in cells], dtype=float)
    if scores.size:
        aggregates = {
            "p10_score_percent": round(float(np.percentile(scores, 10)), 3),
            "median_score_percent": round(float(np.median(scores)), 3),
            "worst_score_percent": round(float(scores.min()), 3),
            "mean_score_percent": round(float(scores.mean()), 3),
        }
    else:
        aggregates = {
            "p10_score_percent": None,
            "median_score_percent": None,
            "worst_score_percent": None,
            "mean_score_percent": None,
        }

    hotspots.sort(key=lambda item: (-item["severity"], -item["source_ink_pixels"]))
    # Component continuity is a structural question, so measure it on a
    # stable high-contrast core.  The raw ROI deliberately includes very light
    # antialias halos for local colour/detail scoring; labelling those halos as
    # independent components created false topology failures and could also
    # join unrelated dark shapes through a six-level JPEG fringe.
    # On a clean white boundary the adaptive ink threshold bottoms out at six
    # levels.  Merely doubling that value made 13--17-level modelling bands
    # inside white lettering look like independent structural strokes.  A
    # palette flattener may legitimately merge those near-white bands into the
    # light object while preserving its actual glyph silhouette; treating the
    # bands as missing components produced a catastrophic topology score even
    # when every meaningful dark stem remained intact.  The +12 floor applies
    # only while boundary noise is low (once threshold >= 12, the existing 2x
    # rule still dominates), so medium/dark lines on noisier artwork retain the
    # previous structural sensitivity.  Local colour/detail diagnostics still
    # measure the excluded low-contrast pixels.
    core_threshold = structural_core_threshold(threshold)
    core_source = src_mask & (roi["strength"] >= core_threshold)
    core_render = ren_mask & (ren_strength >= core_threshold)
    topology = _component_topology(
        core_source, core_render, viewbox=viewbox)
    topology.update({
        "measurement_mask": "strong_ink_core",
        "core_threshold": round(float(core_threshold), 3),
        "core_source_ink_pixels": int(core_source.sum()),
        "low_contrast_excluded_pixels": int(src_mask.sum() - core_source.sum()),
    })
    detail_grid = {
        "cell_size_px": cell,
        "one_pixel_tolerance": True,
        "eligible_cells": len(cells),
        **aggregates,
        "source_ink_pixels": int(roi["source_ink_pixels"]),
        "source_ink_fraction": float(roi["source_ink_fraction"]),
        "background_rgb": [round(v, 3) for v in roi["background_rgb"]],
        "boundary_noise_p95": round(float(roi["boundary_noise_p95"]), 3),
        "ink_threshold": round(threshold, 3),
        "component_topology": topology,
        "cells": cells,
    }
    return {"hotspots": hotspots[:max_spots], "detail_grid": detail_grid}


def compute_hotspots(
    render: ImageInput,
    source: ImageInput,
    viewbox: Optional[Sequence[float]] = None,
    *,
    cell: int = 48,
    max_spots: int = 40,
) -> list:
    """Compatibility helper returning only the clickable hotspot list."""

    return compute_quality_diagnostics(
        render, source, viewbox, cell=cell, max_spots=max_spots
    )["hotspots"]


__all__ = [
    "compute_hotspots",
    "compute_quality_diagnostics",
    "source_ink_roi",
    "structural_core_threshold",
]
