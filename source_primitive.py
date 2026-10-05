"""Narrow source-driven ellipse reconstruction, separate from curve cleanup.

Only an isolated opaque solid silhouette on a plain white original PNG is
eligible. The candidate must improve agreement with that original, satisfy
the selected bidirectional contour budget, preserve holes and salient corners,
and reduce editable anchors. This is a reviewed proposal, not design intent.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from designer_handoff import _geometry, _local, _style, _view_box
from local_refine import GEOMETRY, _by_id, _write_svg


def _opaque_context(root, identifier):
    parents = {child: parent for parent in root.iter() for child in parent}
    shapes = [node for node in root.iter() if _local(node.tag) in
              {'path', 'circle', 'ellipse', 'rect', 'polygon', 'polyline', 'line', 'text'}]
    if len(shapes) != 1 or shapes[0].get('id') != identifier or _local(shapes[0].tag) != 'path':
        raise ValueError('source_primitive_requires_one_isolated_path')
    current = _by_id(root)[identifier]
    if _geometry(current)[0] <= 4:
        raise ValueError('source_primitive_no_anchor_reduction')
    while current is not None:
        attrs = {**current.attrib, **_style(current)}
        if any(key in attrs for key in ('transform', 'filter', 'mask', 'clip-path')):
            raise ValueError('source_primitive_unsupported_effect_context')
        for key in ('opacity', 'fill-opacity'):
            if key in attrs and str(attrs[key]).strip() not in ('1', '1.0', '100%'):
                raise ValueError('source_primitive_transparency_ambiguity')
        if (str(attrs.get('stroke', 'none')).strip().lower() != 'none'
                or 'url(' in str(attrs.get('fill', '')).lower()
                or str(attrs.get('fill', '')).strip().lower() in ('none', 'transparent', 'inherit', 'currentcolor')):
            raise ValueError('source_primitive_requires_plain_solid_fill')
        paint = str(attrs.get('fill', '')).strip().lower()
        if (paint.startswith(('rgba(', 'hsla('))
                or (re.fullmatch(r'#[0-9a-f]{4}', paint) and paint[-1] != 'f')
                or (re.fullmatch(r'#[0-9a-f]{8}', paint) and paint[-2:] != 'ff')):
            raise ValueError('source_primitive_transparency_ambiguity')
        current = parents.get(current)


def _source_contour(source, view):
    from curve_refit import _mask_topology
    with Image.open(source) as image:
        if min(image.size) < 32 or max(image.size) > 2048:
            raise ValueError('source_primitive_image_resolution_budget')
        rgba = np.asarray(image.convert('RGBA'), dtype=np.float64)
    if np.any(rgba[:, :, 3] != 255):
        raise ValueError('source_primitive_transparency_ambiguity')
    rgb = rgba[:, :, :3]
    h, w = rgb.shape[:2]
    if abs((w / h) / (view[2] / view[3]) - 1) > 1e-6:
        raise ValueError('source_primitive_ambiguous_source_alignment')
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    if np.max(np.abs(border - 255)) > 1:
        raise ValueError('source_primitive_requires_plain_white_background')
    delta = 255 - rgb
    distance = np.linalg.norm(delta, axis=2)
    strongest = (distance >= np.percentile(distance, 90)) & (distance > 32)
    if not strongest.any():
        raise ValueError('source_primitive_insufficient_source_contrast')
    foreground = np.median(rgb[strongest], axis=0)
    direction = 255 - foreground
    coverage = np.clip(np.sum(delta * direction, axis=2) / np.sum(direction ** 2), 0, 1)
    reconstructed = 255 - coverage[:, :, None] * direction
    color_residual = np.abs(reconstructed - rgb).max(axis=2)
    if np.percentile(color_residual, 99) > 2 or color_residual.max() > 10:
        raise ValueError('source_primitive_not_single_solid_colour')
    mask = coverage >= 0.5
    if _mask_topology(mask) != (1, 0):
        raise ValueError('source_primitive_multiple_components_or_holes')
    ys, xs = np.where(mask)
    if min(np.ptp(xs), np.ptp(ys)) < 16:
        raise ValueError('source_primitive_shape_too_small_for_intent')
    core = mask.copy()
    for d in (-1, 1):
        core &= np.roll(mask, d, axis=0) & np.roll(mask, d, axis=1)
    if not core.any() or np.any(coverage[core] < 0.97):
        raise ValueError('source_primitive_interior_colour_or_alpha_ambiguity')
    points = []
    # Interpolate the 50% coverage crossing between pixel centres. Binary
    # pixel staircases would bias low-resolution ellipse radii and corners.
    for axis in (0, 1):
        a = coverage[:-1, :] if axis == 0 else coverage[:, :-1]
        b = coverage[1:, :] if axis == 0 else coverage[:, 1:]
        yy, xx = np.where((a > 0.5) != (b > 0.5))
        t = (0.5 - a[yy, xx]) / (b[yy, xx] - a[yy, xx])
        points.extend(zip(xx + 0.5 + (t if axis == 1 else 0),
                          yy + 0.5 + (t if axis == 0 else 0)))
    if not 32 <= len(points) <= 8192:
        raise ValueError('source_primitive_contour_sample_budget')
    points = np.asarray(points) / [w, h] * view[2:] + view[:2]
    return points, rgb, mask, foreground


def _ellipse_geometry(points, budget):
    from curve_refit import _fit_ellipse
    from geometry_error_optimizer import _nearest_distances, _salient_corner_indices
    fitted = _fit_ellipse(points)
    if fitted is None:
        raise ValueError('source_primitive_no_ellipse_fit')
    centre, radii, axes, _ = fitted
    if not np.isfinite(radii).all() or min(radii) <= 0 or max(radii) / min(radii) > 8:
        raise ValueError('source_primitive_ambiguous_ellipse_proportions')
    local = (points - centre) @ axes
    points = points[np.argsort(np.arctan2(local[:, 1] / radii[1], local[:, 0] / radii[0]))]
    scale = float(np.linalg.norm(np.ptp(points, axis=0)))
    tolerance = scale * budget / 100
    theta = np.linspace(0, math.tau, 2049)
    contour = centre + np.column_stack((radii[0] * np.cos(theta), radii[1] * np.sin(theta))) @ axes.T
    forward = _nearest_distances(points, contour)
    backward = _nearest_distances(contour[:-1], np.vstack((points, points[0])))
    errors = np.concatenate((forward, backward))
    p95, maximum = float(np.percentile(errors, 95)), float(errors.max())
    tail = float(np.mean(errors > tolerance))
    corners = _salient_corner_indices(points, closed=True, scale=scale)
    # A primitive would remove these corner editing handles. Even if average
    # contour error is low, a salient corner disqualifies ellipse rebuilding.
    if corners:
        raise ValueError('source_primitive_salient_corners_are_not_an_ellipse')
    if p95 > tolerance or maximum > 3 * tolerance or tail > 0.05:
        raise ValueError('source_primitive_original_contour_error_budget_exceeded')
    angle = math.degrees(math.atan2(axes[1, 0], axes[0, 0]))
    geometry = {'cx': centre[0], 'cy': centre[1], 'rx': radii[0], 'ry': radii[1]}
    geometry = {key: format(float(value), '.9g') for key, value in geometry.items()}
    geometry['transform'] = f'rotate({angle:.9g} {centre[0]:.9g} {centre[1]:.9g})'
    return geometry, {'bidirectional_p95_percent': 100 * p95 / scale,
        'bidirectional_maximum_percent': 100 * maximum / scale, 'over_budget_share': tail,
        'error_budget_percent': budget, 'salient_corner_count': 0,
        'source_contour_samples': len(points), 'ellipse_contour_samples': len(contour),
        'scope': 'subpixel_source_boundary_to_sampled_ellipse_not_design_intent'}


def propose_source_ellipse(root, source, identifier, budget, folder, renderer):
    """Return an independently source-validated candidate or raise to skip."""
    from curve_refit import _mask_topology
    from annulus_detector import _dilate
    _opaque_context(root, identifier)
    points, raw, source_mask, foreground = _source_contour(source, _view_box(root))
    geometry, contour_evidence = _ellipse_geometry(points, budget)
    candidate = copy.deepcopy(root)
    element = _by_id(candidate)[identifier]
    element.tag = '{http://www.w3.org/2000/svg}ellipse'
    for key in GEOMETRY:
        element.attrib.pop(key, None)
    element.attrib.update(geometry)
    arrays, provenance = {}, {}
    for name, tree in (('source-parent', root), ('source-candidate', candidate)):
        path = Path(folder) / (name + '.svg')
        _write_svg(path, tree)
        provenance[name] = renderer(path, path.with_suffix('.png'), raw.shape[1])
        with Image.open(path.with_suffix('.png')) as im:
            arrays[name] = np.asarray(im.convert('RGB'), dtype=np.float64)
    before, after = arrays['source-parent'], arrays['source-candidate']
    if before.shape != raw.shape or after.shape != raw.shape:
        raise ValueError('source_primitive_render_alignment_failed')
    direction = 255 - foreground
    after_coverage = np.clip(np.sum((255 - after) * direction, axis=2) / np.sum(direction ** 2), 0, 1)
    candidate_mask = after_coverage >= 0.5
    if _mask_topology(candidate_mask) != (1, 0):
        raise ValueError('source_primitive_render_topology_failed')
    intersection = source_mask & candidate_mask
    union = source_mask | candidate_mask
    iou = float(intersection.sum() / max(1, union.sum()))
    recall = float(_dilate(candidate_mask, 1)[source_mask].mean())
    precision = float(_dilate(source_mask, 1)[candidate_mask].mean()) if candidate_mask.any() else 0
    before_error, after_error = float(np.abs(before - raw).mean()), float(np.abs(after - raw).mean())
    local_before = float(np.abs(before - raw)[union].mean())
    local_after = float(np.abs(after - raw)[union].mean())
    fill_error = float(np.abs(after - raw)[intersection].mean()) if intersection.any() else math.inf
    if (iou < 0.985 or recall < 0.995 or precision < 0.995 or fill_error > 2.0
            or after_error > before_error - max(0.01, before_error * 0.05)
            or local_after >= local_before or local_after > 2.0):
        raise ValueError('source_primitive_original_image_fidelity_not_improved')
    return candidate, {'proposal_kind': 'source_primitive', 'primitive': 'ellipse',
        'accepted': True, 'external_render_check': 'completed', 'renderer_provenance': provenance,
        'original_png_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'source_contour': contour_evidence, 'source_components_and_holes': [1, 0],
        'render_components_and_holes': [1, 0], 'source_mask_iou': iou,
        'source_ink_recall_tolerance_1px': recall, 'source_ink_precision_tolerance_1px': precision,
        'baseline_mean_absolute_rgb_error': before_error, 'candidate_mean_absolute_rgb_error': after_error,
        'local_baseline_mean_absolute_rgb_error': local_before,
        'local_candidate_mean_absolute_rgb_error': local_after, 'interior_fill_error': fill_error,
        'not_parent_equivalent_simplification': True, 'human_acceptance': 'not_performed'}
