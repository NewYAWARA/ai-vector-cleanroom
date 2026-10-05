"""Read-only native-source structure audit for reliable opaque paper artwork.

This reports visible ink merges, splits and filled narrow paper-coloured
channels. It neither repairs geometry nor resolves white paint versus empty
space. Multiple spatially coincident contrast thresholds are required before
requesting manual review; component counts alone never establish a defect.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time

import numpy as np

from source_scene_guard import _components, _read_rgba, _composite, _source_masks

THRESHOLDS = (32, 64, 96)
MIN_COMPONENT_PIXELS = 4
MIN_STABLE_PIXELS = 2
MAX_REPORTED_DEFECTS = 64


def _shift(array, dy, dx):
    """Zero-padded integer shift, with no wrap-around contacts."""
    result = np.zeros_like(array)
    height, width = array.shape
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    if sy1 > sy0 and sx1 > sx0:
        result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = array[sy0:sy1, sx0:sx1]
    return result


def _pairs(first, second, *, counts=False):
    keep = (first > 0) & (second > 0)
    if not keep.any():
        return []
    radix = int(second.max()) + 1
    # Native images are capped at 8M pixels by _read_rgba: the maximum
    # pixel-label product is < 2**46, safely within signed int64.
    keys, sizes = np.unique(first[keep].astype(np.int64) * radix + second[keep], return_counts=True)
    if counts:
        return [(int(key // radix), int(key % radix), int(size)) for key, size in zip(keys, sizes)]
    return [(int(key // radix), int(key % radix)) for key in keys]


def _label_ink(mask):
    labels, count = _components(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    eligible = sizes >= MIN_COMPONENT_PIXELS
    eligible[0] = False
    return np.where(eligible[labels], labels, 0), sizes, eligible


def _correspondence(source_labels, candidate_labels, source_sizes, candidate_sizes):
    source_to_candidate, candidate_to_source = {}, {}
    rows = []
    for source_id, candidate_id, pixels in _pairs(source_labels, candidate_labels, counts=True):
        smaller = min(int(source_sizes[source_id]), int(candidate_sizes[candidate_id]))
        if pixels < max(2, math.ceil(smaller * .25)):
            continue
        source_to_candidate.setdefault(source_id, set()).add(candidate_id)
        candidate_to_source.setdefault(candidate_id, set()).add(source_id)
        rows.append({'source_component': source_id, 'candidate_component': candidate_id,
                     'overlap_pixels': pixels})
    return source_to_candidate, candidate_to_source, rows


def _contacts(regions, ink):
    touches = {}
    for dy, dx in ((-1,0), (1,0), (0,-1), (0,1)):
        for region, component in _pairs(regions, _shift(ink, dy, dx)):
            touches.setdefault(region, set()).add(component)
    return touches


def _bridge_evidence(changed, touching_ink, owner_labels, owner_to_components,
                     reliable_pixels):
    """A changed bridge/cut must physically contact the corresponding pieces."""
    regions, count = _components(changed)
    touching = _contacts(regions, touching_ink)
    owners = {}
    for region, owner in _pairs(regions, owner_labels):
        owners.setdefault(region, set()).add(owner)
    keep = np.zeros(count + 1, bool)
    events = []
    reliable_counts = np.bincount(regions[reliable_pixels], minlength=count + 1)
    for region, contacts in touching.items():
        if reliable_counts[region] < MIN_STABLE_PIXELS:
            continue
        for owner in owners.get(region, ()):
            matching = contacts & owner_to_components.get(owner, set())
            if len(matching) >= 2:
                keep[region] = True
                events.append({'owner_component': owner, 'matched_components': sorted(matching),
                               'changed_bridge_region': region,
                               'reliable_changed_pixels': int(reliable_counts[region])})
    return keep[regions] & reliable_pixels, events


def _filled_channels(source_ink, candidate_ink, paper):
    """Detect lost native white samples tightly flanked by existing ink.

    Opposing ink within four centre-to-centre pixels covers one/two-pixel
    white channels plus their AA fringe. It is a visual feature, regardless
    of whether the original author meant a white object or negative space.
    """
    lost_paper = paper & candidate_ink
    found = np.zeros(paper.shape, bool)
    ink_on_both = source_ink & candidate_ink
    for dy, dx in ((1,0), (0,1)):
        for before in (1,2,3):
            left = _shift(ink_on_both, before * dy, before * dx)
            for after in range(1, 5 - before):
                right = _shift(ink_on_both, -after * dy, -after * dx)
                found |= lost_paper & left & right
    return found


def _region_record(mask, source, candidate, reference):
    ys, xs = np.nonzero(mask)
    probes = []
    for index in np.unique(np.linspace(0, len(xs)-1, min(5,len(xs)), dtype=int)):
        x, y = int(xs[index]), int(ys[index])
        probe = {'xy':[x,y], 'source_rgba':source[y,x].tolist(),
                 'candidate_rgba':candidate[y,x].tolist()}
        if reference is not None:
            probe['processed_reference_rgba'] = reference[y,x].tolist()
        probes.append(probe)
    return {'bbox_xyxy':[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
            'pixels':int(xs.size), 'probes':probes}


def _paper_core_color_warning(source, candidate, reference, source_distance,
                              candidate_distance, original_rgb, candidate_rgb):
    # Pure array erosion avoids platform-specific native filter behaviour.
    # A 3x3 original-paper core excludes normal antialiased exterior edges;
    # consequently one/two-pixel white slits have no coverage in this branch.
    core = source_distance <= 8
    paper = core.copy()
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            core &= _shift(paper,dy,dx)
    source_error = np.max(np.abs(candidate_rgb-original_rgb),axis=2)
    unexpected = (core & (reference[:,:,3] <= 8) & (candidate[:,:,3] >= 240)
                  & (candidate_distance >= 10) & (source_error >= 10))
    return unexpected, {
        'method':'original_3x3_paper_core_and_processed_clear_with_final_opaque_color',
        'source_paper_max_distance':8, 'processed_alpha_max':8,
        'candidate_alpha_min':240, 'candidate_paper_distance_min':10,
        'candidate_original_rgb_error_min':10,
        'minimum_region_pixels':MIN_COMPONENT_PIXELS,
        'scope':'manual_visual_color_warning_not_alpha_deletion_authorization',
        'coverage_limit':'one_or_two_pixel_white_channels_without_3x3_paper_core_are_not_checked_here',
        'original_paper_core_pixels':int(core.sum()),
        'unexpected_opaque_color_pixels':int(unexpected.sum())}


def _audit_arrays(source, candidate, reference, *, deadline):
    """Array-level deterministic core; callers preserve native dimensions."""
    result = {'schema':'aivc.source-topology-audit/v1', 'status':'not_applicable',
        'manual_review':False, 'native_resolution':True,
        'scope':'visible_ink_structure_not_semantic_objects_or_transparency_recovery',
        'thresholds':[], 'stable_defects':[], 'reasons':[],
        'policy':{'contrast_distance':'maximum_absolute_RGB_channel_distance_from_paper',
            'contrast_thresholds':list(THRESHOLDS), 'connectivity':4,
            'minimum_component_pixels':MIN_COMPONENT_PIXELS,
            'minimum_spatial_overlap_fraction_of_smaller_component':.25,
            'minimum_stable_thresholds':2, 'minimum_stable_pixels':MIN_STABLE_PIXELS,
            'strong_paper_max_distance':8, 'split_source_ink_min_distance':96,
            'counts_alone_are_not_defects':True,
            'white_object_vs_negative_space':'unresolved; visual white feature only'}}
    if source.shape != candidate.shape or (reference is not None and source.shape != reference.shape):
        result.update(status='unavailable', reasons=['native_reference_dimensions_differ'])
        return result
    if np.any(source[:,:,3] < 254):
        result['reasons'] = ['source_not_opaque_paper_artwork']
        return result
    _, _, paper_evidence, paper = _source_masks(source, reference)
    result['paper_evidence'] = paper_evidence
    if not paper_evidence['confident']:
        result['reasons'] = ['reliable_paper_and_processed_reference_evidence_unavailable']
        return result
    original_rgb = _composite(source, paper)
    candidate_rgb = _composite(candidate, paper)
    source_distance = np.max(np.abs(original_rgb-paper), axis=2)
    candidate_distance = np.max(np.abs(candidate_rgb-paper), axis=2)
    reliable_paper = source_distance <= 8
    votes = {kind:np.zeros(source.shape[:2],np.uint8) for kind in ('source_components_merged',
        'source_component_split','paper_colored_channel_filled')}
    per_threshold = []
    for threshold in THRESHOLDS:
        if time.monotonic() >= deadline:
            result.update(status='unavailable', reasons=['native_topology_audit_budget_exhausted'])
            return result
        source_ink, candidate_ink = source_distance >= threshold, candidate_distance >= threshold
        a, sa, ka = _label_ink(source_ink)
        b, sb, kb = _label_ink(candidate_ink)
        a_to_b, b_to_a, correspondence = _correspondence(a,b,sa,sb)
        merge_owners = {key:value for key,value in b_to_a.items() if len(value)>1}
        split_owners = {key:value for key,value in a_to_b.items() if len(value)>1}
        merge_mask, merges = _bridge_evidence(candidate_ink & ~source_ink, a, b,
            merge_owners, reliable_paper & candidate_ink) if merge_owners else (np.zeros(a.shape,bool),[])
        split_mask, splits = _bridge_evidence(source_ink & ~candidate_ink, b, a,
            split_owners, (source_distance >= 96) & ~candidate_ink) if split_owners else (np.zeros(a.shape,bool),[])
        channel = _filled_channels(source_ink, candidate_ink, reliable_paper)
        masks = {'source_components_merged':merge_mask, 'source_component_split':split_mask,
                 'paper_colored_channel_filled':channel}
        for kind, mask in masks.items():
            votes[kind] += mask
        per_threshold.append(masks)
        row = {'contrast_threshold':threshold, 'source_components':int(ka.sum()),
            'candidate_components':int(kb.sum()), 'spatial_correspondence_pairs':len(correspondence),
            'component_merges':merges[:MAX_REPORTED_DEFECTS],
            'component_splits':splits[:MAX_REPORTED_DEFECTS],
            'merge_reliable_pixels':int(merge_mask.sum()), 'split_reliable_pixels':int(split_mask.sum()),
            'filled_channel_reliable_pixels':int(channel.sum()),
            'unmatched_source_components':int(sum(key not in a_to_b for key in np.flatnonzero(ka))),
            'unmatched_candidate_components':int(sum(key not in b_to_a for key in np.flatnonzero(kb))),
            'unmatched_regions_policy':'recorded_only; deletion/noise ambiguity is outside this topology audit'}
        result['thresholds'].append(row)
    for kind, vote in votes.items():
        stable = vote >= 2
        regions, count = _components(stable)
        sizes = np.bincount(regions.ravel(),minlength=count+1)
        for label in np.flatnonzero(sizes >= MIN_STABLE_PIXELS):
            if label == 0:
                continue
            if len(result['stable_defects']) >= MAX_REPORTED_DEFECTS:
                result['stable_defects_truncated'] = True
                break
            region = regions == label
            thresholds = [value for value, masks in zip(THRESHOLDS,per_threshold)
                          if int((region & masks[kind]).sum()) >= MIN_STABLE_PIXELS]
            if len(thresholds)<2:
                continue
            if kind not in result['reasons']:
                result['reasons'].append(kind)
            result['manual_review'] = True
            record = _region_record(region,source,candidate,reference)
            record.update(kind=kind,stable_contrast_thresholds=thresholds)
            result['stable_defects'].append(record)
    warning, warning_policy = _paper_core_color_warning(source,candidate,reference,
        source_distance,candidate_distance,original_rgb,candidate_rgb)
    result['paper_color_warning_policy'] = warning_policy
    regions,count = _components(warning)
    sizes = np.bincount(regions.ravel(),minlength=count+1)
    for label in np.flatnonzero(sizes >= MIN_COMPONENT_PIXELS):
        if label==0:
            continue
        kind='source_paper_unexpected_opaque_color'
        if kind not in result['reasons']:
            result['reasons'].append(kind)
        result['manual_review']=True
        if len(result['stable_defects'])>=MAX_REPORTED_DEFECTS:
            result['stable_defects_truncated']=True
            break
        record=_region_record(regions==label,source,candidate,reference)
        record.update(kind=kind, evidence_method=warning_policy['method'],
            stable_contrast_thresholds=[],
            white_object_vs_negative_space='unresolved; do not remove alpha automatically')
        result['stable_defects'].append(record)
    result['status'] = 'completed'
    result['no_stable_defect_detected'] = not result['manual_review']
    return result


def audit_source_topology(svg_path, source_original_path, processed_reference_path=None,
                          *, maximum_seconds=20):
    """Return structured evidence, without changing inputs or SVG geometry.

    The time limit is checked between bounded native-image stages. It is not
    a subprocess kill deadline. Missing/unsupported evidence is never called
    a successful verification.
    """
    if not math.isfinite(maximum_seconds) or maximum_seconds < 0:
        raise ValueError('invalid_source_topology_audit_budget')
    start = time.monotonic()
    paths = {'svg':Path(svg_path),'source_original':Path(source_original_path)}
    if processed_reference_path is not None:
        paths['processed_reference'] = Path(processed_reference_path)
    hashes = {}
    try:
        hashes = {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in paths.items()}
        source = _read_rgba(paths['source_original'])
        reference = _read_rgba(paths['processed_reference']) if 'processed_reference' in paths else None
        if time.monotonic()-start >= maximum_seconds:
            raise ValueError('native_topology_audit_budget_exhausted')
        from svg_renderer import renderer_info
        from source_scene_guard import _render_native_payload
        height,width = source.shape[:2]
        candidate = _render_native_payload(paths['svg'].read_bytes(),width,height)
        renderer = {**renderer_info(), 'width':width, 'height':height,
            'source_svg_sha256':hashes['svg'], 'background':None,
            'viewport_mode':'explicit_native_viewport_retained_svg_viewbox',
            'rgba_sha256':hashlib.sha256(candidate.tobytes()).hexdigest(),
            'rgba_sha256_scope':'decoded_native_rgba_row_major_dimensions_recorded_separately'}
        result = _audit_arrays(source,candidate,reference,deadline=start+maximum_seconds)
        result['renderer'] = renderer
    except (OSError,ValueError,RuntimeError) as error:
        result = {'schema':'aivc.source-topology-audit/v1','status':'unavailable',
            'manual_review':False,'native_resolution':True,'reasons':[str(error)],'stable_defects':[]}
    result['source_hashes'] = hashes
    result['wall_seconds'] = time.monotonic()-start
    result['inputs_unchanged'] = all(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==hashes.get(key)
                                     for key,path in paths.items())
    if hashes and not result['inputs_unchanged']:
        result.update(status='unavailable',manual_review=False,reasons=['source_topology_inputs_changed'])
    return result
