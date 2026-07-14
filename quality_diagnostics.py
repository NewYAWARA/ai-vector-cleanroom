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
]
