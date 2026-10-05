"""Bounded partial contour refinement with unchanged scene/source contracts.

Only existing closed L/C compound paths are supported. Unsafe spans and every
other contour retain their exact geometry. This is a verified greedy subset,
not a global minimum or an estimate of a designer's editing time.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

SCHEMA = 'ai-vector-cleanroom.gradient-contour-spans/v1'
SOLVER = 'gradient_contour_spans.verified_bounded_subset'


class SpanSearchRejected(ValueError):
    def __init__(self, reason, diagnostics):
        super().__init__(reason)
        self.diagnostics = diagnostics


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _render_contract_passes(guard):
    """Same topology/missing-ink contract as the main geometry transaction."""
    return (guard.get('external_render_check')=='completed'
        and guard.get('alpha_topology',{}).get('accepted') is True
        and guard.get('composed_alpha',{}).get('accepted') is True
        and all(isinstance(guard.get(k),(int,float)) and math.isfinite(guard[k]) and guard[k]>=99
                for k in ('ink_recall_percent','ink_precision_percent','ink_f1_percent')))


def _path(segments, closed):
    def point(value):
        # Preserve parsed original coordinates exactly, including more than
        # nine decimals on untouched neighbouring spans.
        return ','.join(repr(float(x)) if float(x) else '0' for x in value)
    commands = ['M'+point(segments[0]['start'])]
    for segment in segments:
        if segment['type'] == 'line':
            commands.append('L'+point(segment['end']))
        else:
            commands.append('C'+point(segment['control1'])+' '+point(segment['control2'])+' '+point(segment['end']))
    if closed:
        commands.append('Z')
    return ' '.join(commands)


def _fit(subpath):
    start = list(subpath['start'])
    position, segments = start, []
    for item in subpath['segs']:
        if item[0] == 'L':
            segment = {'type': 'line', 'start': position, 'end': list(item[1:3])}
        elif item[0] == 'C':
            segment = {'type': 'cubic', 'start': position, 'control1': list(item[1:3]),
                       'control2': list(item[3:5]), 'end': list(item[5:7])}
        else:
            raise ValueError('span_refit_requires_existing_line_cubic_contours')
        segments.append(segment)
        position = segment['end']
    if subpath['closed'] and position != start:
        segments.append({'type': 'line', 'start': position, 'end': start})
    if not segments:
        raise ValueError('span_refit_empty_contour')
    return {'closed': bool(subpath['closed']), 'segments': segments,
            'anchor_count': len(segments), 'segment_count': len(segments),
            'input_point_count': len(segments), 'path': _path(segments, subpath['closed'])}


def _source_guard(before_png, after_png, reference, box, folder, label):
    from vector_cleanroom import _match_percent
    rows = {}
    with Image.open(reference) as image:
        origin = image.info.get('avc_reference_alpha_origin')
        for scope in ('whole_scene', 'object_roi'):
            files = [before_png, after_png, reference]
            if scope == 'object_roi':
                files = []
                for index, path in enumerate((before_png, after_png, reference)):
                    target = folder/(label+'-roi-'+str(index)+'.png')
                    with Image.open(path) as crop_image:
                        info = PngInfo()
                        if index == 2 and origin:
                            info.add_text('avc_reference_alpha_origin', origin)
                        crop_image.crop(box).save(target, pnginfo=info)
                    files.append(target)
            old = _match_percent(files[0], files[2], foreground_only=True, return_details=True)
            new = _match_percent(files[1], files[2], foreground_only=True, return_details=True)
            metrics = {key: {'before': float(old[key]), 'after': float(new[key]),
                             'maximum_allowed_regression': .25}
                       for key in ('recall', 'precision', 'coverage_f1')}
            ratio = float(new['render_ink_pixels'])/max(1, float(old['render_ink_pixels']))
            rows[scope] = {'metrics': metrics, 'render_ink_area_ratio': ratio,
                           'source_ink_pixels': int(old['source_ink_pixels']),
                           'accepted': all(v['after']-v['before'] >= -.25 for v in metrics.values())
                                       and .985 <= ratio <= 1.015}
    return {'accepted': all(row['accepted'] for row in rows.values()), 'scopes': rows,
            'object_roi_xyxy': list(box)}


def propose_gradient_contour_spans(svg_text, geometry, source_mask, drawable_id,
                                   original_source_path, processed_rgba, *,
                                   budget_percent=.25, maximum_spans=16,
                                   maximum_probes=8, maximum_seconds=45):
    """Return (actual SVG, geometry, proof), or reject without changing input.

    One work unit may finish after the deadline; no new proposal/probe starts
    after it. Final exact-byte verification is mandatory even at the deadline.
    Callers share the probe/time allowance across their scene's objects.
    """
    from clean_base import _parse_subpaths
    from curve_refit_stage import _sample_subpath
    from geometry_error_optimizer import (optimize_curve, measure_fit_error,
        _error_contract_evidence, _combine_compound_loop_fits)
    from gradient_source_components import _paint_context
    from source_gradient_primitive import gradient_paint_sha256
    from svg_renderer import render_svg_reference
    from vector_cleanroom import validate_svg_stage_renders
    from trace_engine import _mask_to_smooth_loops
    from svg_postprocess import _designer_path_anchors

    started = time.perf_counter()
    maximum_spans = min(16, max(1, int(maximum_spans)))
    maximum_probes = min(8, max(0, int(maximum_probes)))
    maximum_seconds = min(45., max(0., float(maximum_seconds)))
    deadline = started+maximum_seconds
    rows, probes = [], 0
    def performance():
        return {'elapsed_seconds': time.perf_counter()-started,
                'native_probe_count': probes, 'maximum_probes': maximum_probes,
                'search_budget_seconds': maximum_seconds,
                'one_work_unit_and_final_revalidation_may_overrun_budget': True}
    def reject(reason):
        raise SpanSearchRejected(reason, {'performance': performance(), 'trials': rows})
    if maximum_probes == 0 or maximum_seconds == 0:
        reject('contour_span_search_budget_exhausted')
    mask = np.asarray(source_mask, dtype=bool)
    processed = np.asarray(processed_rgba, dtype=np.uint8)
    if mask.ndim != 2 or processed.shape != (*mask.shape, 4) or not mask.any():
        reject('span_source_alignment_invalid')
    root = ET.fromstring(svg_text)
    targets = [node for node in root.iter() if node.get('id') == drawable_id]
    if len(targets) != 1 or targets[0].tag.rsplit('}',1)[-1] != 'path':
        reject('span_target_identity_invalid')
    target = targets[0]
    gradient_id, context = _paint_context(root, target)
    if not gradient_id or not target.get('data-avc-gradient-object'):
        reject('span_source_gradient_identity_required')
    view = [float(v) for v in root.get('viewBox','').replace(',',' ').split()]
    if view != [0., 0., float(mask.shape[1]), float(mask.shape[0])]:
        reject('span_source_pixel_viewbox_required')
    subpaths = _parse_subpaths(target.get('d',''))
    if not subpaths or any(not sub.get('closed') for sub in subpaths):
        reject('span_requires_closed_compound')
    fits = [_fit(sub) for sub in subpaths]
    before_count = _designer_path_anchors(target.get('d',''))
    if before_count != geometry.get('designer_anchor_count'):
        reject('span_baseline_geometry_count_changed')
    ys,xs = np.nonzero(mask)
    x0,y0 = max(0,int(xs.min())-2),max(0,int(ys.min())-2)
    local = mask[y0:int(ys.max())+3,x0:int(xs.max())+3]
    raw = [np.asarray(loop,dtype=float)+[x0,y0] for loop in _mask_to_smooth_loops(
        local,simplify=0,min_area=1,smooth=0)]
    if len(raw) != len(fits):
        reject('span_raw_loop_identity_unavailable')
    base_error = measure_fit_error(raw, _combine_compound_loop_fits(fits), error_budget_percent=budget_percent)
    if not _error_contract_evidence(base_error,budget_percent)['within_budget']:
        reject('span_baseline_not_inside_raw_ownership_contract')
    outer_index = max(range(len(fits)), key=lambda i:len(fits[i]['segments']))
    segments = fits[outer_index]['segments']
    if len(segments) < 32:
        reject('span_baseline_already_economical')
    boundaries = np.linspace(0,len(segments),min(maximum_spans,len(segments)//4)+1,dtype=int)
    chunks = [segments[a:b] for a,b in zip(boundaries[:-1],boundaries[1:])]
    outer_scale = float(np.linalg.norm(np.ptp(raw[outer_index],axis=0)))
    proposals = []
    for index, chunk in enumerate(chunks):
        if time.perf_counter() >= deadline:
            break
        sub = {'start':chunk[0]['start'], 'closed':False, 'segs':[
            ('L',*s['end']) if s['type']=='line' else
            ('C',*s['control1'],*s['control2'],*s['end']) for s in chunk]}
        points = _sample_subpath(sub,.5)
        scale = float(np.linalg.norm(np.ptp(points,axis=0)))
        if scale <= 0:
            continue
        ratio = outer_scale/scale
        result = optimize_curve(points,closed=False,error_budget_percent=budget_percent*ratio,
            tolerance_percents=[v*ratio for v in (.03,.06,.10,.16,.25,.4)],max_segments=4096)
        fitted = result['fit']
        if fitted['segment_count'] >= len(chunk):
            continue
        new_segments = fitted['segments']
        if any(segment.get('type') not in {'line','cubic'} for segment in new_segments):
            continue
        if (new_segments[0]['start'] != chunk[0]['start']
                or new_segments[-1]['end'] != chunk[-1]['end']):
            continue
        proposals.append({'span':index,'segments':new_segments,
                          'predicted_anchors_removed':len(chunk)-len(new_segments)})
    proposals.sort(key=lambda row:(-row['predicted_anchors_removed'],row['span']))
    if not proposals:
        reject('no_economical_contour_span_candidate')
    with Image.open(original_source_path) as image:
        original = np.asarray(image.convert('RGBA'),dtype=np.uint8)
    if max(original.shape[:2]) > 2048 or abs(original.shape[1]/original.shape[0]-mask.shape[1]/mask.shape[0]) > 1e-6:
        reject('span_original_size_or_aspect_unsupported')
    accepted, last_geometry, final_error = [], None, None
    with tempfile.TemporaryDirectory(prefix='avc-span-') as temp:
        folder=Path(temp)
        before=folder/'before.svg';before.write_text(svg_text,encoding='utf8')
        processed_path=folder/'processed.png';info=PngInfo()
        info.add_text('avc_reference_alpha_origin','native' if np.any(original[:,:,3]!=255) else 'opaque_canvas_derived')
        Image.fromarray(processed).save(processed_path,pnginfo=info)
        # Source metadata is untrusted. Only decoded pixels determine native
        # alpha evidence, just as in the pipeline's canonical source snapshot.
        trusted_original=folder/'original.png';native_info=PngInfo()
        native_info.add_text('avc_reference_alpha_origin','native')
        Image.fromarray(original).save(trusted_original,pnginfo=native_info)
        references=[('original',trusted_original,original.shape[1]),('processed',processed_path,mask.shape[1])]
        boxes={};before_pngs={}
        for name,path,width in references:
            png=folder/(name+'-before.png');render_svg_reference(before,png,width,background=None);before_pngs[name]=png
            scale=width/mask.shape[1]
            height=original.shape[0] if name=='original' else mask.shape[0]
            boxes[name]=(max(0,math.floor((xs.min()-2)*scale)),max(0,math.floor((ys.min()-2)*scale)),
                         min(width,math.ceil((xs.max()+3)*scale)),min(height,math.ceil((ys.max()+3)*scale)))
        cache={}
        def validate(candidate):
            guard=validate_svg_stage_renders(before,candidate,'curve_refit',render_cache=cache,
                render_size=max(1200,original.shape[1]),tolerance_px=1)
            source_guards={}
            if _render_contract_passes(guard):
                for name,path,width in references:
                    png=folder/(name+'-after.png');render_svg_reference(candidate,png,width,background=None)
                    source_guards[name]=_source_guard(before_pngs[name],png,path,boxes[name],folder,name)
            passed=(_render_contract_passes(guard)
                and len(source_guards)==2 and all(r['accepted'] for r in source_guards.values()))
            return passed,guard,source_guards
        for proposal in proposals:
            if probes>=maximum_probes or time.perf_counter()>=deadline:
                break
            trial_chunks=chunks.copy();trial_chunks[proposal['span']]=proposal['segments']
            combined_segments=[s for group in trial_chunks for s in group]
            trial_fits=copy.deepcopy(fits)
            trial_fits[outer_index].update(segments=combined_segments,path=_path(combined_segments,True),
                anchor_count=len(combined_segments),segment_count=len(combined_segments))
            compound=_combine_compound_loop_fits(trial_fits)
            error=measure_fit_error(raw,compound,error_budget_percent=budget_percent)
            contract=_error_contract_evidence(error,budget_percent)
            row={'span':proposal['span'],'predicted_anchors_removed':proposal['predicted_anchors_removed'],
                 'raw_geometry_contract':contract,'accepted':False}
            if not contract['within_budget']:
                rows.append(row);continue
            candidate_root=copy.deepcopy(root);node=next(n for n in candidate_root.iter() if n.get('id')==drawable_id)
            node.set('d',compound['path'])
            node.set('data-avc-designer-anchors',str(compound['anchor_count']))
            for attr,value in (('data-avc-error-budget-percent',budget_percent),
                               ('data-avc-p95-error-percent',error['p95_percent']),
                               ('data-avc-max-error-percent',error['max_percent'])):
                node.set(attr,str(value))
            candidate=folder/'candidate.svg'
            candidate.write_bytes(ET.tostring(candidate_root,encoding='unicode').encode('utf8'))
            probes+=1
            passed,guard,source_guards=validate(candidate)
            row.update(accepted=passed,render_guard=guard,source_guards=source_guards)
            rows.append(row)
            if passed:
                chunks=trial_chunks;last_geometry=compound;final_error=error;accepted.append(proposal['span'])
                final_payload=candidate.read_bytes()
        if not accepted:
            reject('no_safe_contour_span')
        final_path=folder/'final.svg';final_path.write_bytes(final_payload)
        passed,guard,source_guards=validate(final_path)
        if not passed:
            reject('final_exact_span_subset_revalidation_failed')
        final_root=ET.fromstring(final_payload)
        final_node=next(n for n in final_root.iter() if n.get('id')==drawable_id)
        _,final_context=_paint_context(final_root,final_node)
        proof={'schema':SCHEMA,'status':'verified_bounded_subset','scope':'same_original_ownership_and_scene_reference_for_every_cumulative_subset',
               'drawable_id':drawable_id,'gradient_object_id':target.get('data-avc-gradient-object'),
               'gradient_id':gradient_id,'paint_sha256':gradient_paint_sha256(final_root,gradient_id),
               'presentation_context':final_context,'path_sha256':_sha(final_node.get('d').encode()),
               'before_path_sha256':_sha(target.get('d').encode()),'before_svg_sha256':_sha(svg_text.encode()),
               'after_svg_sha256':_sha(final_payload),'source_rgba_sha256':_sha(original.tobytes()),
               'source_size':[original.shape[1],original.shape[0]],'ownership_mask_sha256':_sha(mask.tobytes()),
               'selected_spans':accepted,'generated_candidates':len(proposals),'original_span_count':len(chunks),
               'untouched_contours_and_unselected_segments':'exact_existing_geometry',
               'before_designer_anchors':before_count,'after_designer_anchors':last_geometry['anchor_count'],
               'error_budget_percent':budget_percent,'measured_raw_geometry':final_error,
               'raw_geometry_contract':_error_contract_evidence(final_error,budget_percent),
               'render_guard':guard,'source_guards':source_guards,'trials':rows,'performance':performance(),
               'renderer_policy':'existing_main_refit_missing_ink_and_topology_guard_only',
               'minimum_claim':'bounded_verified_greedy_subset_not_global_minimum','human_time_saving_claimed':False}
    updated=copy.deepcopy(geometry)
    updated.update(solver=SOLVER,path=last_geometry['path'],anchor_count=last_geometry['anchor_count'],
        designer_anchor_count=last_geometry['anchor_count'],segment_count=last_geometry['segment_count'],
        source_contour_spans=proof,minimum_claim=proof['minimum_claim'])
    updated['error_budget']={**updated.get('error_budget',{}),'passed':True,'requested_max_percent':budget_percent,
        'actual_p95_error_percent':final_error['p95_percent'],'actual_max_error_percent':final_error['max_percent'],
        'metric':'original_ownership_raw_contours_bidirectional_per_loop_with_relationships'}
    updated['selection_evidence']={'selected_candidate_id':'verified_contour_span_subset',
        'identity_rollback_selected':False,'candidate_search_scope':proof['minimum_claim']}
    # Do not carry old whole-contour frontier counts as if this smaller,
    # scene-verified subset were the same optimiser selection.
    for stale in ('fit','candidates','largest_loop_frontier','refinement_frontier','selected_mixed_identity_summary'):
        updated.pop(stale,None)
    return final_payload.decode('utf8'),updated,proof


def source_contour_spans_certificate_valid(geometry):
    from geometry_error_optimizer import _error_contract_evidence
    try:
        proof=geometry['source_contour_spans']
        if (geometry.get('solver')!=SOLVER or proof.get('schema')!=SCHEMA
                or proof.get('status')!='verified_bounded_subset'
                or proof.get('minimum_claim')!='bounded_verified_greedy_subset_not_global_minimum'
                or geometry['error_budget']['requested_max_percent']!=proof['error_budget_percent']
                or not _error_contract_evidence(proof['measured_raw_geometry'],proof['error_budget_percent'])['within_budget']):return False
        if not 0<len(proof['selected_spans'])<=proof['performance']['native_probe_count']<=8:return False
        if len(set(proof['selected_spans']))!=len(proof['selected_spans']):return False
        if (not 0<geometry['anchor_count']==geometry['designer_anchor_count']==proof['after_designer_anchors']<proof['before_designer_anchors']
                or geometry['error_budget']['actual_p95_error_percent']!=proof['measured_raw_geometry']['p95_percent']
                or geometry['error_budget']['actual_max_error_percent']!=proof['measured_raw_geometry']['max_percent']):return False
        guard=proof['render_guard']
        if (proof.get('renderer_policy')!='existing_main_refit_missing_ink_and_topology_guard_only'
                or not _render_contract_passes(guard)):return False
        for name in ('original','processed'):
            for scope in ('whole_scene','object_roi'):
                row=proof['source_guards'][name]['scopes'][scope]
                if not .985<=row['render_ink_area_ratio']<=1.015:return False
                for key in ('recall','precision','coverage_f1'):
                    metric=row['metrics'][key]
                    if (not all(isinstance(metric[k],(int,float)) and math.isfinite(metric[k]) for k in ('before','after'))
                            or metric['after']-metric['before']<-.25):return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):
        return False


def final_source_contour_spans_matches(root,geometry,source_path):
    from gradient_source_components import _paint_context
    from source_gradient_primitive import gradient_paint_sha256
    from svg_postprocess import _designer_path_anchors
    if not source_contour_spans_certificate_valid(geometry):return False
    try:
        proof=geometry['source_contour_spans'];nodes=[n for n in root.iter() if n.get('id')==proof['drawable_id']]
        if len(nodes)!=1:return False
        node=nodes[0];gradient_id,context=_paint_context(root,node)
        if (gradient_id!=proof['gradient_id'] or context!=proof['presentation_context']
                or node.get('data-avc-gradient-object')!=proof['gradient_object_id']
                or gradient_paint_sha256(root,gradient_id)!=proof['paint_sha256']
                or _sha(node.get('d','').encode())!=proof['path_sha256']
                or _designer_path_anchors(node.get('d',''))!=geometry['designer_anchor_count']):return False
        with Image.open(source_path) as image:source=np.asarray(image.convert('RGBA'),dtype=np.uint8)
        return proof['source_size']==[source.shape[1],source.shape[0]] and proof['source_rgba_sha256']==_sha(source.tobytes())
    except (OSError,KeyError,TypeError,ValueError,OverflowError):return False
