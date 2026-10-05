"""Source-supported enclosed ownership proposals, committed by native rendering.

A palette ownership hole is not necessarily a hole in the artwork.  This module
proposes only small, fully enclosed, opaque coloured source components.  Neither
their colour nor a successful fit authorises an SVG change: the complete scene
must improve against both original and processed pixels without worsening any
pixel in a newly owned component.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

SCHEMA = "ai-vector-cleanroom.source-enclosed-components/v1"
_NEIGHBOURS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))


class _SpanSceneBudget:
    """One bounded span search allowance for every object in this scene."""
    def __init__(self,clock=None):
        self.clock=clock or time.perf_counter;self.started=None;self.probes=0;self.calls=0
        self.elapsed_search=0.;self.call_started=None

    def remaining(self):
        if self.started is None:self.started=self.clock()
        self.call_started=self.clock()
        return max(0,8-self.probes),max(0.,45.-(self.clock()-self.started))

    def consume(self,performance,allocated):
        self.calls+=1
        if self.call_started is not None:self.elapsed_search+=max(0.,self.clock()-self.call_started)
        value=performance.get('native_probe_count') if isinstance(performance,dict) else None
        valid=isinstance(value,int) and not isinstance(value,bool) and 0<=value<=allocated
        self.probes+=value if valid else allocated
        return valid

    def report(self):
        return {'maximum_native_probes':8,'maximum_search_seconds':45,
                'native_probe_count':self.probes,'helper_calls':self.calls,
                'elapsed_seconds':self.elapsed_search,
                'wall_seconds_since_first_request':0. if self.started is None else self.clock()-self.started,
                'scope':'one_allowance_shared_by_all_scene_gradient_objects',
                'deadline_policy':'no_new_work_after_deadline_current_unit_and_final_revalidation_may_finish_later'}


def _digest(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()


def _aligned_original(original, shape):
    array = np.asarray(original)
    if (array.ndim != 3 or array.shape[2] != 4 or array.dtype != np.uint8
            or min(array.shape[:2]) < 8 or max(array.shape[:2]) > 8192):
        raise ValueError("original_decoded_rgba8_required")
    h,w = shape
    if abs((array.shape[1]/array.shape[0])/(w/h)-1) > .005:
        raise ValueError("original_canvas_aspect_ratio_mismatch")
    provenance = {"original_rgba_sha256": _digest(array),
                  "original_size": [int(array.shape[1]),int(array.shape[0])],
                  "aligned_size": [int(w),int(h)],
                  "scope": "decoded_rgba_pixels_not_original_png_file_bytes",
                  "resampling": "none" if array.shape[:2] == shape else "Pillow.LANCZOS",
                  "coordinate_mapping": "same_canvas_pixel_centres"}
    if array.shape[:2] != shape:
        array = np.asarray(Image.fromarray(array,"RGBA").resize((w,h),Image.Resampling.LANCZOS))
    provenance["aligned_rgba_sha256"] = _digest(array)
    return array, provenance


def enclosed_components(mask, *, maximum_pixels=64, maximum_components=64):
    """Eight-connected background compartments; diagonally open gaps stay open."""
    mask = np.asarray(mask,dtype=bool)
    yy,xx = np.where(mask)
    if not len(xx):
        return []
    x0,y0,x1,y1 = max(0,int(xx.min())-1),max(0,int(yy.min())-1),min(mask.shape[1],int(xx.max())+2),min(mask.shape[0],int(yy.max())+2)
    background = np.pad(~mask[y0:y1,x0:x1],1,constant_values=True)
    seen = np.zeros_like(background);h,w = background.shape
    results=[]
    for sy,sx in zip(*np.where(background)):
        if seen[sy,sx]:
            continue
        queue=deque([(int(sy),int(sx))]);seen[sy,sx]=True
        points=[];border=False
        while queue:
            y,x=queue.popleft()
            if y in (0,h-1) or x in (0,w-1):
                border=True
            if len(points)<=maximum_pixels:
                points.append((y,x))
            for dy,dx in _NEIGHBOURS:
                ny,nx=y+dy,x+dx
                if 0<=ny<h and 0<=nx<w and background[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True;queue.append((ny,nx))
        if border or not 1<=len(points)<=maximum_pixels:
            continue
        component=np.zeros_like(mask)
        for y,x in points:
            component[y+y0-1,x+x0-1]=True
        results.append(component)
        if len(results)>maximum_components:
            return []
    return results


def _pure_fill_geometry(proposal,expanded,components,budget):
    """Remove proven hole loops; independently remeasure every retained curve."""
    from clean_base import _parse_subpaths
    from trace_engine import _mask_to_smooth_loops
    from gradient_reconstruction_stage import decode_mask_rle, _mask_topology
    from geometry_error_optimizer import measure_fit_error, _error_contract_evidence
    baseline=proposal['geometry']['path']
    if any(letter.islower() for letter in re.findall(r'[A-Za-z]',baseline)):
        return None
    chunks=re.findall(r'M[^M]*',baseline)
    mask=decode_mask_rle(proposal['mask'])
    old_topology=proposal['geometry'].get('topology',{})
    extraction=old_topology.get('extraction_evidence',{})
    prior_smooth=0. if old_topology.get('smoothing_fallback',False) else float(extraction.get('requested_smooth',0.))
    halo=max(2,int(math.ceil(4*prior_smooth)))
    yy,xx=np.where(mask);x0=max(0,int(xx.min())-halo);y0=max(0,int(yy.min())-halo)
    x1=min(mask.shape[1],int(xx.max())+1+halo);y1=min(mask.shape[0],int(yy.max())+1+halo)
    def extract(smooth):
        return [np.asarray(loop,dtype=float)+[x0,y0] for loop in _mask_to_smooth_loops(
            mask[y0:y1,x0:x1],simplify=0.,min_area=1.,smooth=smooth)]
    source_loops=extract(0.)
    measured_source_loops=extract(prior_smooth) if prior_smooth else source_loops
    if len(measured_source_loops)!=len(source_loops):return None
    if len(source_loops)!=len(chunks):return None
    remove=set()
    for component,_ in components:
        yy,xx=np.where(component);bbox=np.array([xx.min(),yy.min(),xx.max()+1,yy.max()+1],dtype=float)
        matches=[]
        for index,loop in enumerate(source_loops):
            p=np.asarray(loop,dtype=float);q=np.roll(p,-1,axis=0)
            area=float(np.sum(p[:,0]*q[:,1]-q[:,0]*p[:,1])/2)
            bounds=np.r_[p.min(0),p.max(0)]
            if area<0 and np.allclose(bounds,bbox,rtol=0,atol=1e-6):matches.append(index)
        if len(matches)!=1:return None
        remove.add(matches[0])
    retained=[index for index in range(len(chunks)) if index not in remove]
    path=' '.join(chunks[index] for index in retained)
    fitted=[];count=0
    for sub in _parse_subpaths(path):
        if not sub['closed']:return None
        start=np.asarray(sub['start'],dtype=float);current=start;segments=[]
        for segment in sub['segs']:
            kind=segment[0]
            if kind=='L':
                end=np.asarray(segment[1:3],dtype=float)
                segments.append({'type':'line','start':current.tolist(),'end':end.tolist()})
            elif kind=='C':
                end=np.asarray(segment[5:7],dtype=float)
                segments.append({'type':'cubic','start':current.tolist(),'end':end.tolist(),
                                 'control1':segment[1:3],'control2':segment[3:5]})
            elif kind=='A':
                from curve_refit_stage import _svg_arc_center_parameters
                parameters=_svg_arc_center_parameters(current,segment)
                if parameters is None:return None
                cx,cy,rx,ry,phi,theta,delta,x2,y2=parameters;end=np.asarray([x2,y2])
                segments.append({'type':'arc','start':current.tolist(),'end':end.tolist(),
                    'center':[cx,cy],'rx':rx,'ry':ry,'rotation':math.degrees(phi),
                    'large_arc':bool(segment[4]),'sweep':bool(segment[5])})
            else:return None
            current=end
        if not np.allclose(current,start,rtol=0,atol=1e-9):
            segments.append({'type':'line','start':current.tolist(),'end':start.tolist()})
        fitted.append({'closed':True,'segments':segments});count+=len(segments)
    source=[measured_source_loops[index] for index in retained]
    topology=_mask_topology(expanded)
    if sum(topology)!=len(source):return None
    fit={'contours':fitted,'loop_count':len(fitted),'fill_rule':'evenodd','segment_count':count}
    error=measure_fit_error(source,fit,error_budget_percent=budget)
    contract=_error_contract_evidence(error,budget)
    if not contract['within_budget']:return None
    geometry=copy.deepcopy(proposal['geometry']);geometry.pop('fit',None)
    geometry.update(path=path,fill_rule='evenodd',native_primitives=[],native_whole_object_path=None,
        primitive_first=False,anchor_count=count,designer_anchor_count=count,segment_count=count,
        topology={'components':topology[0],'holes':topology[1],'expected_loops':len(source),
                  'actual_loops':len(source),'topology_preserved':True,
                  'scope':'augmented_source_ownership_exact_retained_contours',
                  'retained_reference_contours_from_original_geometry':True,
                  'retained_reference_smooth':prior_smooth,
                  'compound_relationships':error.get('compound_relationships')},
        selection_evidence={'selected_candidate_id':'existing_optimizer_contours_minus_source_supported_holes',
                            'identity_rollback_selected':proposal['geometry'].get('selection_evidence',{}).get('identity_rollback_selected',True),
                            'outer_and_retained_hole_curves_exact':True,
                            'removed_loop_indices':sorted(remove),
                            'independent_geometry_remeasurement':error},
        error_budget={'metric':'bidirectional_geometry_over_source_bbox_diagonal_percent',
                      'requested_max_percent':budget,'actual_p95_error_percent':error['p95_percent'],
                      'actual_max_error_percent':error['max_percent'],'passed':True,'contract':contract})
    return geometry


def _propose_component_variant(proposal, processed_rgb, original_source_rgba,
        visible, other_owners, *, alpha, labels, model_fit_options,
        geometry_error_percent, geometry_smooth, max_segments, model_fitter):
    """Return a pending alternative; the baseline proposal is never mutated."""
    from gradient_reconstruction_stage import (decode_mask_rle, encode_mask_rle,
        _render_discovery_model, _stage_delta_e, _paint_models_stable)
    mask=decode_mask_rle(proposal["mask"])
    original,provenance=_aligned_original(original_source_rgba,mask.shape)
    processed=np.asarray(processed_rgb,dtype=float)[:,:,:3]
    eligible=[];decisions=[]
    enclosed=enclosed_components(mask)
    silhouette=mask.copy()
    for component in enclosed:silhouette|=component
    # Pure ownership completion does not move the exterior at all. A second
    # hypothesis excluding the exterior's entire future error envelope would
    # duplicate work for a motion that belongs to the separate refit phase.
    clearance=1
    for number,component in enumerate(enclosed):
        yy,xx=np.where(component)
        record={"component_id":f"enclosed-{number:04d}","pixels":int(len(xx)),
                "bbox_xyxy":[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]}
        reasons=[]
        neighbourhood=np.asarray(Image.fromarray(component.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(2*clearance+1)))>0
        record['outer_boundary_clearance_pixels']=clearance
        if np.any(neighbourhood&~silhouette):
            reasons.append('component_too_close_to_external_boundary')
        if not np.asarray(visible,dtype=bool)[component].all(): reasons.append("source_not_visible_fill")
        if np.asarray(other_owners,dtype=bool)[component].any(): reasons.append("another_gradient_owns_component")
        if np.any(original[:,:,3][component]!=255): reasons.append("original_alpha_not_opaque")
        if alpha is not None and np.any(np.asarray(alpha)[component]!=255): reasons.append("processed_alpha_not_opaque")
        # A near-white pixel can be an intentional highlight or aperture.  Its
        # colour resemblance elsewhere cannot authorise taking its ownership.
        for name,source in (("original",original[:,:,:3]),("processed",processed)):
            if np.any(np.max(255-source[component].astype(float),axis=1)<32):
                reasons.append(name+"_white_or_light_detail")
            boundary_a=[];boundary_b=[]
            for y,x in zip(yy,xx):
                for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny,nx=int(y)+dy,int(x)+dx
                    if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx]:
                        boundary_a.append(source[y,x]);boundary_b.append(source[ny,nx])
            if boundary_a:
                maximum=float(_stage_delta_e(np.asarray(boundary_a,dtype=float),np.asarray(boundary_b,dtype=float)).max())
                record.setdefault('source_boundary_max_delta_e',{})[name]=maximum
                if maximum>8.0:reasons.append(name+'_material_boundary_at_component')
        prediction=_render_discovery_model(proposal,xx.astype(float),yy.astype(float),extrapolate=False)
        if prediction is None: reasons.append("paint_prediction_unavailable")
        else:
            record["same_model_error"]={}
            for name,source in (("original",original[:,:,:3]),("processed",processed)):
                errors=_stage_delta_e(source[component].astype(float),prediction)
                stats={"mean":float(errors.mean()),"p90":float(np.percentile(errors,90)),"p99":float(np.percentile(errors,99))}
                record["same_model_error"][name]=stats
                if any(stats[key]>min(limit,float(model_fit_options.get(option,limit)))
                       for key,limit,option in (("mean",5.5,"maximum_mean_error"),("p90",8.5,"maximum_p90_error"),("p99",13.,"maximum_p99_error"))):
                    reasons.append(name+"_existing_absolute_paint_budget_exceeded")
        record["reasons"]=reasons
        record["status"]="rejected" if reasons else "source_supported_pending_scene"
        decisions.append(record)
        if not reasons:
            eligible.append((component,record))
    if not eligible or sum(int(component.sum()) for component,_ in eligible)>256:
        return None
    expanded=mask.copy()
    for component,_ in eligible: expanded|=component
    # Refit the augmented ownership on both deterministic holdouts.  The actual
    # emitted paint remains the original one; stability binds both new fits to it.
    fits=[]
    options=dict(model_fit_options)
    for salt in (0x6D39,0x2B71):
        options["validation_seed"]=int(model_fit_options.get("validation_seed",0))^salt
        fit=model_fitter(processed,expanded,alpha=alpha,label_map=labels,**options)
        stable,evidence=_paint_models_stable(proposal,fit)
        fits.append(evidence)
        if not stable: return None
    old_count=int(proposal["geometry"]["designer_anchor_count"])
    pure=_pure_fill_geometry(proposal,expanded,eligible,float(geometry_error_percent))
    if pure is None or int(pure['designer_anchor_count'])>=old_count:return None
    geometry=pure
    return {"schema":SCHEMA,"status":"pending_native_scene_transaction",
            "candidate_id":proposal["candidate_id"],"proposal_id":proposal["proposal_id"],
            "source_provenance":provenance,
            "processed_rgba_sha256":_digest(np.dstack((np.asarray(processed_rgb,dtype=np.uint8)[:,:,:3],
                np.full(mask.shape,255,np.uint8) if alpha is None else np.asarray(alpha,dtype=np.uint8)))),
            "baseline_mask_sha256":_digest(mask),"expanded_mask_sha256":_digest(expanded),
            "baseline_path_sha256":hashlib.sha256(proposal['geometry']['path'].encode()).hexdigest(),
            "mask":encode_mask_rle(expanded),"path":geometry["path"],"geometry":geometry,
            "pure_fill_geometry":pure,
            "geometry_refit_options":{"bbox_xyxy":list(proposal['bbox_xyxy']),
                "smooth_values":list(dict.fromkeys((float(geometry_smooth),0.0))),
                "error_budget_percent":float(geometry_error_percent),"max_segments":int(max_segments)},
            "added_components":[{**record,"mask":encode_mask_rle(component)} for component,record in eligible],
            "component_decisions":decisions,"independent_augmented_paint_revalidation":fits,
            "original_designer_anchor_count":old_count,"paint_unchanged":True,
            "scope":"colours_support_new_ownership_not_binary_mask_equivalence",
            "ownership_variant":"exact_contours_with_enclosed_one_pixel_halo",
            "human_acceptance":"not_performed"}


def propose_enclosed_source_components(proposal,processed_rgb,original_source_rgba,
        visible,other_owners,**kwargs):
    """Propose exact ownership completion; defer refitting until scene approval."""
    started=time.perf_counter();model_fit_calls=0
    original_fitter=kwargs['model_fitter']
    def counted_fitter(*args,**options):
        nonlocal model_fit_calls
        model_fit_calls+=1
        return original_fitter(*args,**options)
    kwargs={**kwargs,'model_fitter':counted_fitter}
    value=_propose_component_variant(proposal,processed_rgb,original_source_rgba,
        visible,other_owners,**kwargs)
    if value is None:return None
    performance={'elapsed_seconds':time.perf_counter()-started,
                 'ownership_hypotheses_considered':1,'independent_paint_fit_calls':model_fit_calls,
                 'distinct_pending_ownership_masks':1,'geometry_fit_calls':0,
                 'geometry_refit_deferred_until_actual_scene_ownership_commit':True,
                 'bounds':{'ownership_hypotheses':1,'independent_paint_fits':2,
                           'geometry_fits':0,'added_pixels_per_candidate':256}}
    value['proposal_generation']=performance
    return value


def _rgba_composite(rgba):
    rgba=np.asarray(rgba,dtype=float)
    return rgba[:,:,:3]*rgba[:,:,3:]/255+255*(1-rgba[:,:,3:]/255)


def _halo(mask):
    h,w=mask.shape;pad=np.pad(mask,1)
    return np.logical_or.reduce([pad[y:y+h,x:x+w] for y in range(3) for x in range(3)])


def _paint_context(root,target):
    parents={child:parent for parent in root.iter() for child in parent}
    effective={}
    current=target
    while current is not None:
        if any(key in current.attrib for key in ('transform','style','mask','clip-path','filter')):
            raise ValueError('unsupported_gradient_presentation_context')
        if current.get('display')=='none' or current.get('visibility') in ('hidden','collapse'):
            raise ValueError('hidden_gradient_context')
        for key in ('opacity','fill-opacity','stroke-opacity'):
            if current.get(key) not in (None,'1','1.0','100%'):
                raise ValueError('gradient_opacity_context_unsupported')
        for key in ('fill','stroke','fill-rule','color','shape-rendering','color-interpolation'):
            if key not in effective and current.get(key) is not None:
                effective[key]=current.get(key)
        current=parents.get(current)
    if effective.get('stroke','none')!='none': raise ValueError('gradient_stroke_context_unsupported')
    paint=effective.get('fill','')
    match=re.fullmatch(r'url\(#([^ )]+)\)',paint)
    if not match: raise ValueError('inherited_gradient_paint_required')
    effective.setdefault('fill-rule','nonzero');effective.setdefault('stroke','none')
    effective['canvas']={key:root.get(key) for key in ('width','height','viewBox','preserveAspectRatio')}
    return match[1],effective


def apply_source_component_candidates(svg_text, grad_regions, original_source_path,
        processed_rgba, *, budget_percent=.25):
    """Commit only complete-scene native-rendered, locally nonregressing changes.

    Returns SVG text, updates keyed by candidate ID, and a public transaction
    report.  Inputs, paths and source files are never mutated.
    """
    from gradient_reconstruction_stage import decode_mask_rle
    from source_gradient_primitive import gradient_paint_sha256
    from svg_renderer import render_svg_reference
    started=time.perf_counter();native_render_calls=0
    processed=np.asarray(processed_rgba)
    if processed.ndim!=3 or processed.shape[2]!=4:
        raise ValueError("processed_rgba_required")
    with Image.open(original_source_path) as image:
        raw=np.asarray(image.convert("RGBA"),dtype=np.uint8)
    original,provenance=_aligned_original(raw,processed.shape[:2])
    root=ET.fromstring(svg_text);updates={};transactions=[];span_budget=_SpanSceneBudget()
    source_arrays={"original":original,"processed":processed}
    with tempfile.TemporaryDirectory(prefix="aivc-source-components-") as temp:
        folder=Path(temp)
        def render(tree,name,*,original_size=False):
            nonlocal native_render_calls
            native_render_calls+=1
            path=folder/(name+".svg")
            text=ET.tostring(tree,encoding="unicode")
            path.write_text(text,encoding="utf8")
            expected=raw if original_size else processed
            renderer=render_svg_reference(path,path.with_suffix('.png'),expected.shape[1],background=None)
            with Image.open(path.with_suffix('.png')) as image:
                rgba=np.asarray(image.convert('RGBA'),dtype=float)
            if rgba.shape!=expected.shape: raise ValueError("scene_source_dimensions_mismatch")
            return rgba,renderer,hashlib.sha256(text.encode()).hexdigest()
        before,before_render,before_hash=render(root,"before")
        native_original_size=raw.shape!=processed.shape
        before_native=None
        if native_original_size:
            before_native,_,_=render(root,'before-original-native',original_size=True)
        work_regions=[]
        for region in grad_regions:
            pending=region.get('enclosed_source_component_candidate')
            if not isinstance(pending,dict):continue
            ownerships=pending.get('ownership_alternatives') or [pending]
            for ownership in ownerships:
                variant=copy.copy(region);candidate=copy.copy(ownership)
                pure=ownership['pure_fill_geometry'];candidate.update(geometry=pure,path=pure['path'])
                variant['enclosed_source_component_candidate']=candidate;work_regions.append(variant)
        for region in work_regions:
            if region.get('candidate_id') in updates:continue
            pending=region.get("enclosed_source_component_candidate")
            if not isinstance(pending,dict): continue
            transaction={"candidate_id":region.get("candidate_id"),"status":"rejected","reasons":[]}
            transactions.append(transaction)
            try:
                if (pending.get("schema")!=SCHEMA or pending.get("status")!="pending_native_scene_transaction"
                        or pending.get("source_provenance")!=provenance
                        or pending.get("processed_rgba_sha256")!=_digest(np.asarray(processed,dtype=np.uint8))
                        or pending.get("candidate_id")!=region.get("candidate_id")
                        or _digest(np.asarray(region["mask"],dtype=bool))!=pending["baseline_mask_sha256"]):
                    raise ValueError("pending_source_or_baseline_identity_mismatch")
                nodes=[n for n in root.iter() if n.get('data-avc-gradient-object')==pending['proposal_id']]
                if len(nodes)!=1 or nodes[0].tag.rsplit('}',1)[-1]!='path':
                    raise ValueError("unique_gradient_path_required")
                target=nodes[0];identifier=target.get('id')
                if not identifier: raise ValueError("stable_drawable_identity_missing")
                if hashlib.sha256(target.get('d','').encode()).hexdigest()!=pending['baseline_path_sha256']:
                    raise ValueError('exact_baseline_path_identity_mismatch')
                gradient_id,presentation=_paint_context(root,target)
                paint_hash=gradient_paint_sha256(root,gradient_id)
                geometry=pending['geometry'];budget=geometry.get('error_budget',{})
                if pending['path']!=geometry['path'] or geometry.get('fill_rule','evenodd')!=presentation.get('fill-rule'):
                    raise ValueError('candidate_path_or_fill_rule_identity_mismatch')
                if (budget.get('passed') is not True or float(budget['requested_max_percent'])!=float(budget_percent)
                        or float(budget['actual_p95_error_percent'])>budget_percent
                        or float(budget['actual_max_error_percent'])>3*budget_percent):
                    raise ValueError("candidate_geometry_budget_not_verified")
                expanded=decode_mask_rle(pending['mask'])
                added=expanded & ~np.asarray(region['mask'],dtype=bool)
                if _digest(expanded)!=pending['expanded_mask_sha256'] or not added.any():
                    raise ValueError("expanded_mask_identity_mismatch")
                for other in grad_regions:
                    if other is not region and (added & np.asarray(other['mask'],dtype=bool)).any():
                        raise ValueError("expanded_mask_conflicts_with_other_owner")
                candidate=copy.deepcopy(root)
                changed=next(n for n in candidate.iter() if n.get('id')==identifier)
                changed.set('d',pending['path'])
                changed.set('data-avc-designer-anchors',str(geometry['designer_anchor_count']))
                changed.set('data-avc-error-budget-percent',str(budget['requested_max_percent']))
                changed.set('data-avc-p95-error-percent',str(budget['actual_p95_error_percent']))
                changed.set('data-avc-max-error-percent',str(budget['actual_max_error_percent']))
                after,after_render,after_hash=render(candidate,"after")
                after_native=None
                if native_original_size:
                    after_native,_,_=render(candidate,'after-original-native',original_size=True)
                if gradient_paint_sha256(candidate,gradient_id)!=paint_hash: raise ValueError("paint_changed")
                changes=np.any(before!=after,axis=2)
                measurements={};components=[]
                comparisons=[(name,source,before,after) for name,source in source_arrays.items()]
                if native_original_size:
                    comparisons.append(('original_native',raw,before_native,after_native))
                for name,source,before_image,after_image in comparisons:
                    source_rgb=_rgba_composite(source)
                    first=np.abs(_rgba_composite(before_image)-source_rgb).mean(2)
                    second=np.abs(_rgba_composite(after_image)-source_rgb).mean(2)
                    local_changes=np.any(before_image!=after_image,axis=2)
                    measurements[name]={'before_global_rgb_mae':float(first.mean()),'after_global_rgb_mae':float(second.mean()),
                        'before_changed_rgb_p95':float(np.percentile(first[local_changes],95)) if local_changes.any() else 0,
                        'after_changed_rgb_p95':float(np.percentile(second[local_changes],95)) if local_changes.any() else 0}
                    if second.mean()>first.mean()+1e-9:
                        raise ValueError(name+'_whole_scene_source_fidelity_worse')
                    for record in pending['added_components']:
                        support=decode_mask_rle(record['mask'])
                        if support.shape!=source.shape[:2]:
                            support=np.asarray(Image.fromarray(support.astype(np.uint8)*255).resize(
                                (source.shape[1],source.shape[0]),Image.Resampling.NEAREST))>0
                        halo=_halo(support)
                        worse=int(np.count_nonzero(second[support]>first[support]+1e-9))
                        alpha_first=np.abs(before_image[:,:,3]-source[:,:,3])
                        alpha_second=np.abs(after_image[:,:,3]-source[:,:,3])
                        alpha_worse=int(np.count_nonzero(alpha_second[support]>alpha_first[support]+1e-9))
                        local={'source':name,'component_id':record['component_id'],'pixels':int(support.sum()),
                            'before_rgb_mae':float(first[support].mean()),'after_rgb_mae':float(second[support].mean()),
                            'before_halo_rgb_mae':float(first[halo].mean()),'after_halo_rgb_mae':float(second[halo].mean()),
                            'worse_pixel_count':worse,'alpha_worse_pixel_count':alpha_worse}
                        components.append(local)
                        if worse or alpha_worse or second[halo].mean()>first[halo].mean()+1e-9:
                            transaction['failed_component']=local
                            raise ValueError(name+'_component_pixel_or_halo_fidelity_worse')
                certificate={k:copy.deepcopy(v) for k,v in pending.items() if k not in ('mask','path','geometry','pure_fill_geometry','geometry_alternatives','geometry_refit_options','ownership_alternatives','added_components')}
                certificate.update(status='committed',drawable_id=identifier,gradient_id=gradient_id,
                    gradient_object_id=pending['proposal_id'],paint_sha256=paint_hash,
                    path_sha256=hashlib.sha256(pending['path'].encode()).hexdigest(),presentation_context=presentation,
                    before_svg_sha256=before_hash,after_svg_sha256=after_hash,
                    validation_sources=[row[0] for row in comparisons],
                    actual_scene_measurements=measurements,actual_component_measurements=components,
                    native_renderer_before=before_render,native_renderer_after=after_render,
                    added_components=[{k:v for k,v in record.items() if k!='mask'} for record in pending['added_components']])
                # Ownership repair is now proven with every surviving contour
                # byte-for-byte intact. Subsequent curve economy uses the same
                # geometry/ink/topology policy as the main refit transaction;
                # it is deliberately not constrained by RGB pixel equality.
                from vector_cleanroom import validate_svg_stage_renders, _match_percent
                from PIL.PngImagePlugin import PngInfo
                ref=folder/'processed-reference.png';info=PngInfo()
                info.add_text('avc_reference_alpha_origin','native' if np.any(raw[:,:,3]!=255) else 'opaque_canvas_derived')
                info.add_text('avc_reference_alpha_provenance','original_pixel_alpha_extrema')
                Image.fromarray(processed.astype(np.uint8),'RGBA').save(ref,pnginfo=info)
                pure_svg=folder/'ownership-committed.svg';pure_svg.write_text(ET.tostring(candidate,encoding='unicode'),encoding='utf8')
                pure_png=folder/'ownership-committed.png'
                native_render_calls+=1
                render_svg_reference(pure_svg,pure_png,processed.shape[1],background=None)
                from gradient_reconstruction_stage import _fit_geometry
                refit_started=time.perf_counter();alternatives=[];seen_paths=set();fit_attempts=0;fit_failures=[]
                options=pending.get('geometry_refit_options')
                if options:
                    if (options['error_budget_percent']!=budget_percent
                            or len(options['smooth_values'])>2):raise ValueError('refit_options_budget_mismatch')
                    for smooth in options['smooth_values']:
                        fit_attempts+=1
                        try:
                            fitted,reasons=_fit_geometry(expanded,tuple(options['bbox_xyxy']),
                                error_budget_percent=budget_percent,smooth=smooth,
                                max_segments=options['max_segments'],geometry_optimizer=None)
                        except (ValueError,KeyError,TypeError,OverflowError) as exc:
                            fit_failures.append({'smooth':smooth,'reason':str(exc)});continue
                        if (fitted is not None and not reasons
                                and fitted['designer_anchor_count']<geometry['designer_anchor_count']
                                and fitted['path'] not in seen_paths):
                            fitted.pop('fit',None);seen_paths.add(fitted['path'])
                            alternatives.append({'smooth':smooth,'geometry':fitted,'path':fitted['path']})
                alternatives.sort(key=lambda row:(row['geometry']['designer_anchor_count'],row['smooth']))
                certificate['deferred_refit_generation']={'elapsed_seconds':time.perf_counter()-refit_started,
                    'geometry_fit_attempts':fit_attempts,'distinct_geometry_candidates':len(alternatives),
                    'failures':fit_failures,
                    'only_after_complete_scene_ownership_commit':True}
                geometry_steps=[];seen_refit_paths=set()
                for alternative in alternatives:
                    next_geometry=alternative['geometry']
                    if next_geometry['designer_anchor_count']>=geometry['designer_anchor_count']:continue
                    if alternative['path'] in seen_refit_paths:continue
                    seen_refit_paths.add(alternative['path'])
                    refined=copy.deepcopy(candidate);node=next(n for n in refined.iter() if n.get('id')==identifier)
                    node.set('d',alternative['path'])
                    node.set('data-avc-designer-anchors',str(next_geometry['designer_anchor_count']))
                    for attr,key in (('data-avc-error-budget-percent','requested_max_percent'),
                                     ('data-avc-p95-error-percent','actual_p95_error_percent'),('data-avc-max-error-percent','actual_max_error_percent')):
                        node.set(attr,str(next_geometry['error_budget'][key]))
                    trial,trial_render,trial_hash=render(refined,'geometry-refit')
                    guard=validate_svg_stage_renders(pure_svg,folder/'geometry-refit.svg',
                        'curve_refit_source_ownership',render_size=int(raw.shape[1]),tolerance_px=1)
                    ink_ok=(guard.get('external_render_check')=='completed'
                        and guard.get('alpha_topology',{}).get('accepted') is True
                        and guard.get('composed_alpha',{}).get('accepted') is True
                        and all(isinstance(guard.get(key),(int,float)) and guard[key]>=99.
                                for key in ('ink_recall_percent','ink_precision_percent','ink_f1_percent')))
                    source_guards={}
                    for name,source_path in (('original',Path(original_source_path)),('processed',ref)):
                        old=_match_percent(pure_png,source_path,foreground_only=True,return_details=True)
                        new=_match_percent(folder/'geometry-refit.png',source_path,foreground_only=True,return_details=True)
                        metrics={key:{'before':float(old[key]),'after':float(new[key]),
                                      'maximum_allowed_regression':.25,'accepted':float(new[key])-float(old[key])>=-.25}
                                 for key in ('recall','precision','coverage_f1')}
                        ratio=float(new['render_ink_pixels'])/max(1,float(old['render_ink_pixels']))
                        source_guards[name]={'accepted':all(row['accepted'] for row in metrics.values()) and .985<=ratio<=1.015,
                                             'metrics':metrics,'render_ink_area_ratio':ratio,'accepted_ratio':[.985,1.015]}
                    approved=ink_ok and all(row['accepted'] for row in source_guards.values())
                    step={'status':'committed' if approved else 'rejected','renderer_topology_guard_accepted':ink_ok,
                          'render_guard':guard,'source_guards':source_guards,'before_svg_sha256':after_hash,
                          'after_svg_sha256':trial_hash,'before_designer_anchors':geometry['designer_anchor_count'],
                          'after_designer_anchors':next_geometry['designer_anchor_count'],
                          'policy':'existing_main_refit_missing_ink_and_topology_guard_only',
                          'rgb_pixel_equality_required':False}
                    geometry_steps.append(step)
                    if approved:
                        candidate=refined;geometry=next_geometry;after=trial;after_hash=trial_hash;after_render=trial_render
                        if native_original_size:after_native,_,_=render(candidate,'geometry-refit-native',original_size=True)
                        break
                if geometry_steps and not any(step['status']=='committed' for step in geometry_steps):
                    available_probes,available_seconds=span_budget.remaining()
                    span_step={'status':'skipped','policy':'gradient_contour_spans.verified_bounded_subset',
                        'before_svg_sha256':after_hash,'before_designer_anchors':geometry['designer_anchor_count'],
                        'allocated_native_probes':available_probes,'allocated_seconds':available_seconds}
                    geometry_steps.append(span_step)
                    if available_probes and available_seconds>0:
                        accounted=False
                        try:
                            from gradient_contour_spans import (propose_gradient_contour_spans,
                                final_source_contour_spans_matches)
                            # Renderer imports may register different XML namespace
                            # prefixes globally. Reuse the exact validated baseline
                            # text rather than reserialising the unchanged DOM.
                            span_baseline=pure_svg.read_text(encoding='utf8')
                            if hashlib.sha256(span_baseline.encode()).hexdigest()!=after_hash:
                                raise ValueError('span_saved_baseline_identity_mismatch')
                            span_svg,span_geometry,span_proof=propose_gradient_contour_spans(
                                span_baseline,geometry,expanded,identifier,
                                original_source_path,processed,budget_percent=budget_percent,
                                maximum_spans=16,maximum_probes=available_probes,maximum_seconds=available_seconds)
                            valid_accounting=span_budget.consume(span_proof.get('performance'),available_probes)
                            accounted=True
                            span_step['performance']=span_proof.get('performance',{})
                            if not valid_accounting:raise ValueError('span_probe_budget_accounting_invalid')
                            span_root=ET.fromstring(span_svg)
                            joins={'final_source_certificate':final_source_contour_spans_matches(span_root,span_geometry,original_source_path),
                                'before_svg_identity':span_proof.get('before_svg_sha256')==after_hash,
                                'before_anchor_identity':span_proof.get('before_designer_anchors')==geometry['designer_anchor_count'],
                                'after_anchor_identity':span_proof.get('after_designer_anchors')==span_geometry['designer_anchor_count']}
                            span_step['join_checks']=joins
                            if not all(joins.values()):
                                raise ValueError('span_final_source_or_baseline_join_failed')
                            span_after,span_renderer,span_hash=render(span_root,'bounded-contour-spans')
                            if span_proof.get('after_svg_sha256')!=span_hash:
                                raise ValueError('span_final_svg_serialization_identity_mismatch')
                            span_native=None
                            if native_original_size:span_native,_,_=render(span_root,'bounded-contour-spans-native',original_size=True)
                            candidate=span_root;geometry=span_geometry;after=span_after;after_hash=span_hash;after_render=span_renderer
                            after_native=span_native
                            span_step.update(status='committed',after_svg_sha256=after_hash,
                                after_designer_anchors=geometry['designer_anchor_count'],
                                performance=span_proof['performance'],minimum_claim=span_proof['minimum_claim'])
                        except (ImportError,ValueError,KeyError,TypeError,OverflowError,OSError) as exc:
                            diagnostics=getattr(exc,'diagnostics',{})
                            if not accounted:span_budget.consume(diagnostics.get('performance'),available_probes)
                            span_step.update(status='rejected',reason=str(exc),diagnostics=diagnostics)
                    else:
                        span_step['reason']='scene_span_search_budget_exhausted'
                certificate['ownership_commit_svg_sha256']=certificate['after_svg_sha256']
                certificate['ownership_measurements_scope']='pure_fill_with_exact_outer_and_retained_hole_curves'
                pure_count=int(pending['geometry']['designer_anchor_count'])
                certificate['designer_anchor_accounting']={
                    'before':int(pending['original_designer_anchor_count']),
                    'after_pure_ownership_completion':pure_count,
                    'after_geometry_refit':int(geometry['designer_anchor_count']),
                    'hole_anchors_removed':int(pending['original_designer_anchor_count'])-pure_count,
                    'curve_anchors_removed':pure_count-int(geometry['designer_anchor_count'])}
                certificate['geometry_transactions']=geometry_steps
                certificate['after_svg_sha256']=after_hash
                certificate['path_sha256']=hashlib.sha256(geometry['path'].encode()).hexdigest()
                compact=copy.deepcopy(geometry);compact['source_ownership_completion']=certificate
                updates[region['candidate_id']]={'mask':expanded,'area':int(expanded.sum()),'path':geometry['path'],
                    'geometry':compact,'source_ownership_completion':certificate,
                    'designer_anchor_delta':int(geometry['designer_anchor_count'])-int(pending['original_designer_anchor_count']),
                    'drawable_id':identifier,'gradient_object_id':pending['proposal_id']}
                root=candidate;before=after;before_hash=after_hash;before_render=after_render;before_native=after_native
                transaction.update(status='committed',drawable_id=identifier,measurements=measurements,
                    added_component_count=len(pending['added_components']),added_pixels=int(added.sum()),
                    designer_anchor_accounting=certificate['designer_anchor_accounting'],
                    geometry_transactions=geometry_steps,
                    designer_anchors_before=pending['original_designer_anchor_count'],designer_anchors_after=geometry['designer_anchor_count'])
            except (ValueError,KeyError,TypeError,OverflowError,OSError) as exc:
                transaction['reasons'].append(str(exc))
    return ET.tostring(root,encoding='unicode'),updates,{'schema':SCHEMA,
        'scene_input_svg_sha256':hashlib.sha256(svg_text.encode()).hexdigest(),
        'status':'committed' if updates else 'no_change','transactions':transactions,
        'performance':{'elapsed_seconds':time.perf_counter()-started,
                       'pending_ownership_transactions':len(work_regions),
                       'attempted_ownership_transactions':len(transactions),
                       'direct_native_render_calls':native_render_calls,
                       'direct_native_render_calls_excludes_standard_refit_guard':True,
                       'bounded_span_search':span_budget.report()},
        'human_acceptance':'not_performed','source_provenance':provenance}


def final_source_component_matches(root,geometry,source_path):
    """Final DOM/source join for a committed source-ownership completion."""
    from source_gradient_primitive import gradient_paint_sha256
    try:
        cert=geometry['source_ownership_completion']
        if cert.get('schema')!=SCHEMA or cert.get('status')!='committed' or cert.get('paint_unchanged') is not True:
            return False
        if cert.get('ownership_measurements_scope')!='pure_fill_with_exact_outer_and_retained_hole_curves':return False
        nodes=[node for node in root.iter() if node.get('id')==cert['drawable_id']]
        if len(nodes)!=1 or nodes[0].tag.rsplit('}',1)[-1]!='path': return False
        target=nodes[0]
        gradient_id,context=_paint_context(root,target)
        if (target.get('data-avc-gradient-object')!=cert['gradient_object_id']
                or gradient_id!=cert['gradient_id'] or context!=cert['presentation_context']
                or gradient_paint_sha256(root,gradient_id)!=cert['paint_sha256']
                or hashlib.sha256(target.get('d','').encode()).hexdigest()!=cert['path_sha256']):
            return False
        with Image.open(source_path) as image: original=np.asarray(image.convert('RGBA'),dtype=np.uint8)
        size=cert['source_provenance']['aligned_size']
        _,provenance=_aligned_original(original,(size[1],size[0]))
        if provenance!=cert['source_provenance']: return False
        components=cert['added_components']
        if not 1<=len(components)<=64 or sum(row['pixels'] for row in components)>256: return False
        accounting=cert['designer_anchor_accounting']
        if (any(not isinstance(accounting[key],int) or isinstance(accounting[key],bool)
                or accounting[key]<0 for key in ('before','after_pure_ownership_completion',
                    'after_geometry_refit','hole_anchors_removed','curve_anchors_removed'))
                or accounting['after_geometry_refit']!=geometry['designer_anchor_count']
                or accounting['hole_anchors_removed']!=accounting['before']-accounting['after_pure_ownership_completion']
                or accounting['curve_anchors_removed']!=accounting['after_pure_ownership_completion']-accounting['after_geometry_refit']):return False
        validation_sources=['original','processed']
        if cert['source_provenance']['original_size']!=cert['source_provenance']['aligned_size']:
            validation_sources.append('original_native')
        if cert.get('validation_sources')!=validation_sources:return False
        expected={(source,row['component_id']) for source in validation_sources for row in components}
        rows=cert['actual_component_measurements']
        if len(rows)!=len(expected) or {(r['source'],r['component_id']) for r in rows}!=expected:return False
        finite=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        for row in rows:
            if row['worse_pixel_count']!=0 or row['alpha_worse_pixel_count']!=0:return False
            for before,after in (('before_rgb_mae','after_rgb_mae'),('before_halo_rgb_mae','after_halo_rgb_mae')):
                if not finite(row[before]) or not finite(row[after]) or not 0<=row[after]<=row[before]+1e-9:return False
        for name in validation_sources:
            row=cert['actual_scene_measurements'][name]
            before,after=row['before_global_rgb_mae'],row['after_global_rgb_mae']
            if not finite(before) or not finite(after) or not 0<=after<=before+1e-9:return False
        for name in ('native_renderer_before','native_renderer_after'):
            row=cert[name]
            if row.get('renderer')!='resvg' or row.get('background') is not None or row.get('paint_model')!='native_svg':return False
        steps=cert.get('geometry_transactions')
        if not isinstance(steps,list) or len(steps)>3:return False
        committed=[step for step in steps if step.get('status')=='committed']
        if len(committed)>1:return False
        if not committed:
            if cert.get('after_svg_sha256')!=cert.get('ownership_commit_svg_sha256'):return False
        else:
            step=committed[0]
            if (step.get('before_svg_sha256')!=cert['ownership_commit_svg_sha256']
                    or step.get('after_svg_sha256')!=cert['after_svg_sha256']
                    or step.get('before_designer_anchors')!=accounting['after_pure_ownership_completion']
                    or step.get('after_designer_anchors')!=accounting['after_geometry_refit']):return False
            if step.get('policy')=='gradient_contour_spans.verified_bounded_subset':
                from gradient_contour_spans import final_source_contour_spans_matches
                proof=geometry['source_contour_spans']
                return (proof.get('before_svg_sha256')==step['before_svg_sha256']
                    and proof.get('after_svg_sha256')==step['after_svg_sha256']
                    and proof.get('before_designer_anchors')==step['before_designer_anchors']
                    and proof.get('after_designer_anchors')==step['after_designer_anchors']
                    and final_source_contour_spans_matches(root,geometry,source_path))
            guard=step['render_guard']
            if (step.get('policy')!='existing_main_refit_missing_ink_and_topology_guard_only'
                    or guard.get('external_render_check')!='completed'
                    or guard.get('alpha_topology',{}).get('accepted') is not True
                    or guard.get('composed_alpha',{}).get('accepted') is not True
                    or any(not finite(guard.get(key)) or guard[key]<99. for key in ('ink_recall_percent','ink_precision_percent','ink_f1_percent'))):return False
            for name in ('original','processed'):
                source=step['source_guards'][name]
                if not finite(source.get('render_ink_area_ratio')) or not .985<=source['render_ink_area_ratio']<=1.015:return False
                for key in ('recall','precision','coverage_f1'):
                    row=source['metrics'][key]
                    if not finite(row['before']) or not finite(row['after']) or row['after']-row['before']<-.25:return False
        return True
    except (ValueError,KeyError,TypeError,OverflowError,OSError):
        return False
