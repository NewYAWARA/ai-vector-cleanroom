"""Bounded, source-pixel handoff hints, never an acceptance certificate.

A removal render measures each existing handoff unit's actual contribution to
the complete scene. Occluded pixels and empty space inside bounding boxes are
not attributed to it. Comparisons are native RGBA/white-composite pixels, so
antialias, translucent paint, groups, clips and paint order remain intact.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SCHEMA = 'aivc.source-object-hints/v1'
MAX_PIXELS = 4 * 1024 * 1024
MAX_UNITS = 256
MAX_RENDER_PIXELS = 96 * 1024 * 1024
MAX_SECONDS = 90
_CACHE = OrderedDict()
_LOCK = threading.RLock()
_CACHE_BYTES = 2 * 1024 * 1024
_NS = 'http://www.w3.org/2000/svg'

LABELS = {
    'appearance_mismatch': '可見部分的顏色或覆蓋範圍與原圖不同；請比對色彩、粗細與透明度。',
    'missing_tonal_variation': '原圖在這部分有較多明暗變化，目前較平；請確認是否需要保留漸層或紋理。',
    'paint_on_source_blank': '此物件有上色落在原圖的近白或透明處；請確認字內留白、細縫與外緣。',
}


def _composite(rgba):
    data = rgba.astype(np.float32)
    return 255.0 + (data[..., :3] - 255.0) * (data[..., 3:4] / 255.0)


def _premul(rgba):
    data = rgba.astype(np.float32)
    data[..., :3] *= data[..., 3:4] / 255.0
    return data


def _quantile(values, weights, probability):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[min(len(order)-1, np.searchsorted(cumulative, probability * cumulative[-1]))]])


def _tone_evidence(source_rgb, scene_rgb, visible, weight_map):
    # A narrow object's antialiased outline is not evidence of a lost ramp.
    # Only this optional tonal test needs a one-pixel neighbourhood; colour
    # and blank-paper hints still measure every visible thin object.
    h, w = visible.shape
    p = np.pad(visible, 1)
    interior = np.logical_and.reduce([p[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    rgb = np.pad(source_rgb, ((1,1), (1,1), (0,0)), mode='edge')
    high = np.maximum.reduce([rgb[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    low = np.minimum.reduce([rgb[y:y+h, x:x+w] for y in range(3) for x in range(3)])
    support = interior & (np.max(high-low, axis=2) <= 24) & (source_rgb.min(axis=2) < 244)
    if int(support.sum()) < 64:
        return {}
    yy, xx = np.nonzero(support)
    weights = weight_map[support]
    if float(weights.sum()) < 48:
        return {}
    src, dst = source_rgb[support], scene_rgb[support]
    src_spread = max(_quantile(src[:,c], weights,.9)-_quantile(src[:,c], weights,.1) for c in range(3))
    dst_spread = max(_quantile(dst[:,c], weights,.9)-_quantile(dst[:,c], weights,.1) for c in range(3))
    # Fit a low-frequency plane only as a diagnostic of spatial organisation.
    # Its colour model is never installed, nor called a verified gradient.
    xy = np.column_stack(((xx-xx.mean())/max(1,float(xx.std())),
                          (yy-yy.mean())/max(1,float(yy.std())), np.ones(len(xx))))
    scale = np.sqrt(weights)
    coefficients = np.linalg.lstsq(xy*scale[:,None], src*scale[:,None], rcond=None)[0]
    explained = 1.0 - float(np.sum(weights[:,None]*(src-xy@coefficients)**2))/max(1,float(np.sum(weights[:,None]*(src-np.average(src,axis=0,weights=weights))**2)))
    mismatch = float(np.average(np.max(np.abs(src-dst),axis=1), weights=weights))
    return {'smooth_source_colour_spread': round(src_spread,3),
            'smooth_scene_colour_spread': round(dst_spread,3),
            'smooth_spatial_explained_fraction': round(explained,4),
            'tone_support_pixels': len(xx),
            'tone_hint': bool(src_spread >= 38 and src_spread-dst_spread >= 26
                              and explained >= .55 and mismatch >= 10)}


def compare_visible(source, scene, without):
    """Return hints only on pixels affected by this unit in the final scene."""
    delta = np.max(np.abs(_premul(scene) - _premul(without)), axis=2)
    visible = delta >= 2
    count = int(visible.sum())
    row = {'visible_pixel_count': count, 'flags': [], 'status': 'checked'}
    if count < 4:
        row['status'] = 'no_measurable_visible_contribution'
        return row
    # Relative contribution retains narrow/translucent objects without a fixed
    # erosion, while low-coverage AA pixels cannot outweigh their centres.
    weight = np.minimum(1.0, delta[visible] / max(8.0, float(np.percentile(delta[visible], 90))))
    total = float(weight.sum())
    if total < 3:
        row['status'] = 'insufficient_visible_evidence'
        return row
    source_rgb, scene_rgb = _composite(source), _composite(scene)
    src, dst = source_rgb[visible], scene_rgb[visible]
    err = np.max(np.abs(src - dst), axis=1)
    raw_src = source[visible]
    # Near-white is not proof that a white object should be deleted. Only
    # non-white candidate pixels differing from source create this hint.
    blank = (raw_src[:, 3] <= 5) | ((raw_src[:, 3] >= 250) & (src.min(axis=1) >= 244))
    on_blank = blank & (dst.min(axis=1) < 239) & (err >= 14)
    blank_mass = float(weight[on_blank].sum())
    paper_fraction = blank_mass / total
    if blank_mass >= 4 and paper_fraction >= .20:
        row['flags'].append('paint_on_source_blank')
    ink = ~blank
    ink_weight = weight[ink]
    ink_mass = float(ink_weight.sum())
    mean = float(np.average(err, weights=weight))
    mismatch_fraction = float(weight[err >= 24].sum()) / total
    if ink_mass >= 3:
        ink_error = float(np.average(err[ink], weights=ink_weight))
        ink_mismatch = float(ink_weight[err[ink] >= 24].sum()) / ink_mass
        if ink_error >= 16 and ink_mismatch >= .35:
            row['flags'].append('appearance_mismatch')
        row.update(source_ink_error=round(ink_error, 3))
    weights = np.zeros(visible.shape, np.float32)
    weights[visible] = weight
    tone = _tone_evidence(source_rgb, scene_rgb, visible, weights)
    if tone.pop('tone_hint', False):
        row['flags'].append('missing_tonal_variation')
    row.update(tone)
    row.update(visible_weight=round(total, 3), mean_max_channel_error=round(mean, 3),
               mismatch_fraction=round(mismatch_fraction, 4), source_blank_paint_fraction=round(paper_fraction, 4),
               priority=round(max(paper_fraction, mismatch_fraction) * min(1.0, mean / 32), 4))
    return row


def _render(root, width, height, box=None):
    import resvg_py
    if box is None:
        output = root
        size = width, height
    else:
        x0, y0, x1, y1 = box
        size = x1-x0, y1-y0
        # The inner SVG retains its native viewport. Cropping the outside
        # does not rescale percentages, gradients, masks or non-square meet.
        output = ET.Element('{' + _NS + '}svg', {'width': str(size[0]), 'height': str(size[1]),
                            'viewBox': f'{x0} {y0} {size[0]} {size[1]}'})
        output.append(root)
    data = resvg_py.svg_to_bytes(svg_string=ET.tostring(output, encoding='unicode'),
        width=size[0], height=size[1], background=None, skip_system_fonts=True,
        log_information=False, shape_rendering='geometric_precision')
    with Image.open(io.BytesIO(data)) as image:
        if image.size != size or image.mode != 'RGBA':
            raise ValueError('unexpected_native_render_dimensions')
        return np.asarray(image).copy()


def _crop_box(box, view_box, width, height, preserve):
    if box is None or preserve not in ('', 'xMidYMid', 'xMidYMid meet'):
        return (0, 0, width, height)
    vx, vy, vw, vh = view_box
    scale = min(width/vw, height/vh)
    ox, oy = (width-vw*scale)/2, (height-vh*scale)/2
    x, y, w, h = box
    return (max(0, min(width, math.floor((x-vx)*scale+ox)-2)),
            max(0, min(height, math.floor((y-vy)*scale+oy)-2)),
            max(0, min(width, math.ceil((x+w-vx)*scale+ox)+2)),
            max(0, min(height, math.ceil((y+h-vy)*scale+oy)+2)))


def audit_source_objects(root, objects, svg_sha256, source_path):
    """Read-only audit, bounded memory/time/work and content-keyed result cache.

    Failed/partial results are never cached as success. No source or SVG is
    rewritten. Cache holds only small JSON evidence, never native image arrays.
    """
    base = {'schema': SCHEMA, 'status': 'unavailable', 'objects': [],
            'svg_sha256': svg_sha256, 'human_acceptance': 'not_performed',
            'scope': 'visible_scene_hints_not_semantic_or_complete_design_validation',
            'native_resolution': True, 'unlocated_topology_is_not_assigned_by_bbox': True}
    try:
        path = Path(source_path)
        if path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError('source_file_budget_exceeded')
        source_bytes = path.read_bytes()
        base['source_sha256'] = hashlib.sha256(source_bytes).hexdigest()
        normalized = ET.tostring(root, encoding='utf-8')
        key = hashlib.sha256(svg_sha256.encode('ascii') + normalized + source_bytes + json.dumps(objects, sort_keys=True, ensure_ascii=False).encode('utf8') + SCHEMA.encode()).hexdigest()
        with _LOCK:
            if key in _CACHE:
                _CACHE.move_to_end(key)
                return copy.deepcopy(_CACHE[key])
        if len(objects) > MAX_UNITS:
            raise ValueError('object_count_budget_exceeded')
        from svg_renderer import _validated_svg, native_canvas_aspect_compatible, svg_with_native_viewport
        svg, sw, sh = _validated_svg(normalized)
        with Image.open(io.BytesIO(source_bytes)) as image:
            width, height = image.size
            if width * height > MAX_PIXELS or max(width, height) > 8192:
                raise ValueError('source_native_pixel_budget_exceeded')
            source = np.asarray(image.convert('RGBA')).copy()
        if not native_canvas_aspect_compatible(sw, sh, width, height):
            raise ValueError('source_svg_aspect_ratio_differs')
        native = ET.fromstring(svg_with_native_viewport(svg, width, height))
        box_values = [float(n) for n in native.get('viewBox').replace(',', ' ').split()]
        boxes = [_crop_box(o.get('bbox'), box_values, width, height, native.get('preserveAspectRatio', '')) for o in objects]
        work = width*height + sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
        if work > MAX_RENDER_PIXELS:
            raise ValueError('native_render_work_budget_exceeded')
        start = time.monotonic()
        scene = _render(native, width, height)
        rows = []
        for obj, box in zip(objects, boxes):
            if time.monotonic() - start > MAX_SECONDS:
                base.update(status='partial_budget_exceeded', objects=rows, checked_unit_count=len(rows))
                return base
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                rows.append({'id': obj['id'], 'status': 'outside_native_canvas', 'flags': []})
                continue
            removed = copy.deepcopy(native)
            members = set(obj['member_ids'])
            for parent in list(removed.iter()):
                for child in list(parent):
                    if child.get('id') in members:
                        parent.remove(child)
            without = _render(removed, width, height, box)
            row = compare_visible(source[y0:y1, x0:x1], scene[y0:y1, x0:x1], without)
            row.update(id=obj['id'], member_ids=list(obj['member_ids']), native_bbox_xyxy=list(box))
            rows.append(row)
        result = {**base, 'status': 'completed', 'objects': rows, 'canvas': [width, height],
                  'checked_unit_count': len(rows), 'flagged_unit_count': sum(bool(r['flags']) for r in rows),
                  'render_pixel_work': work, 'renderer': 'resvg_native_counterfactual_removal',
                  'blank_meaning': 'near_white_or_transparent_not_proof_of_disposable_background'}
        size = len(json.dumps(result, ensure_ascii=False).encode('utf8'))
        if size <= _CACHE_BYTES:
            with _LOCK:
                _CACHE[key] = copy.deepcopy(result)
                while len(_CACHE) > 4 or sum(len(json.dumps(v, ensure_ascii=False).encode('utf8')) for v in _CACHE.values()) > _CACHE_BYTES:
                    _CACHE.popitem(last=False)
        return result
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        base['reason'] = str(exc)[:250]
        return base


def attach_source_hints(manifest, root, source_path):
    if source_path is None:
        manifest['source_object_audit'] = {'schema': SCHEMA, 'status': 'unavailable', 'reason': 'original_source_not_supplied'}
        return manifest
    audit = audit_source_objects(root, manifest['objects'], manifest['svg_sha256'], source_path)
    manifest['source_object_audit'] = audit
    by_id = {row['id']: row for row in audit['objects']}
    for obj in manifest['objects']:
        row = by_id.get(obj['id'], {})
        flags = row.get('flags', [])
        obj['source_hint_flags'] = flags
        obj['source_audit_status'] = row.get('status', audit['status'])
        obj['source_defect_count'] = len(flags)
        obj['source_defect_fraction'] = row.get('priority', 0) if flags else 0
        obj['reasons'] = [LABELS[flag] for flag in flags] + obj['reasons']
        if flags and obj['suggested_action'] == 'keep':
            obj['suggested_action'] = 'review'
    return manifest
