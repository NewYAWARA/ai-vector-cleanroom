"""Preserve exact original contact spans in an uncommitted source proposal.

The reference is explicitly hybrid: source/ownership target on free edges,
original curve geometry at rejected contacts, and bounded new joins. No scene
acceptance is granted here. A caller must re-render and revalidate the result.
"""
from __future__ import annotations

import copy
import math
import time
import xml.etree.ElementTree as ET

import numpy as np

SCHEMA = "ai-vector-cleanroom.source-edge-contacts/v1"
JOIN_DIRECTION_SCREEN = {'threshold_degrees':150.,'tangent_parameter_half_window':.01,
    'scope':'new_nonzero_line_joins_not_existing_corners_or_global_smoothness'}


def _native_box_to_viewbox(box, working_dimensions, native_dimensions):
    """Default SVG xMidYMid meet mapping, including rounding letterboxing."""
    w,h=map(float,working_dimensions);nw,nh=map(float,native_dimensions)
    scale=min(nw/w,nh/h)
    offset=np.array([(nw-w*scale)/2,(nh-h*scale)/2])
    return (np.asarray(box,dtype=float)-np.tile(offset,2))/scale


def _at(segment, t):
    p, q = np.asarray(segment['start']), np.asarray(segment['end'])
    if segment['type'] == 'line':
        return p + (q-p)*t
    u = 1-t
    return (u**3*p + 3*u*u*t*np.asarray(segment['control1'])
            + 3*u*t*t*np.asarray(segment['control2']) + t**3*q)


def _split(segment, t):
    left, right = copy.deepcopy(segment), copy.deepcopy(segment)
    point = _at(segment, t).tolist()
    left['end'], right['start'] = point, point
    if segment['type'] == 'cubic':
        p0,p1,p2,p3 = (np.asarray(segment[key]) for key in ('start','control1','control2','end'))
        a,b,c = p0+(p1-p0)*t, p1+(p2-p1)*t, p2+(p3-p2)*t
        d,e = a+(b-a)*t, b+(c-b)*t
        left['control1'],left['control2'] = a.tolist(),d.tolist()
        right['control1'],right['control2'] = e.tolist(),c.tolist()
    return left,right


def _reverse(segments):
    return [{**s, 'start':s['end'], 'end':s['start'],
             **({'control1':s['control2'],'control2':s['control1']} if s['type']=='cubic' else {})}
            for s in reversed(segments)]


def _signed_area(segments):
    # Orientation, not an area certificate; include interior cubic samples.
    points=np.asarray([_at(s,t) for s in segments for t in (0.,.25,.5,.75)])
    return float(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1)))


def _nearest(segments, point):
    """Analytic line projection; bounded local refinement on cubic segments."""
    point=np.asarray(point,dtype=float)
    starts=np.asarray([s['start'] for s in segments],dtype=float)
    ends=np.asarray([s['end'] for s in segments],dtype=float)
    line_ids=np.asarray([i for i,s in enumerate(segments) if s['type']=='line'],dtype=int)
    best=(float('inf'),-1,0.)
    if len(line_ids):
        delta=ends[line_ids]-starts[line_ids]
        parameters=np.clip(np.sum((point-starts[line_ids])*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-30),0,1)
        distances=np.linalg.norm(starts[line_ids]+parameters[:,None]*delta-point,axis=1)
        j=int(distances.argmin());best=(float(distances[j]),int(line_ids[j]),float(parameters[j]))
    for i,s in enumerate(segments):
        if s['type']=='line':
            continue
        if s['type']!='cubic':
            raise ValueError('source_contacts_require_line_cubic_geometry')
        box=np.asarray([s[k] for k in ('start','control1','control2','end')])
        lower=np.maximum(np.maximum(box.min(0)-point,point-box.max(0)),0)
        if np.linalg.norm(lower)>best[0]:
            continue
        ts=np.linspace(0,1,25)
        distances=[float(np.sum((_at(s,t)-point)**2)) for t in ts]
        j=int(np.argmin(distances));lo,hi=max(0.,(j-1)/24),min(1.,(j+1)/24)
        for _ in range(25):
            a,b=(2*lo+hi)/3,(lo+2*hi)/3
            if np.sum((_at(s,a)-point)**2)<np.sum((_at(s,b)-point)**2):hi=b
            else:lo=a
        t=(lo+hi)/2
        value=(float(np.linalg.norm(_at(s,t)-point)),i,t)
        if value[0]<best[0]:best=value
    return best


def _runs(selected):
    if selected.all():
        raise ValueError('source_contacts_would_preserve_entire_contour')
    records=[];n=len(selected)
    for j in range(n):
        if selected[j] and not selected[j-1]:
            indices=[];k=j
            while selected[k]:
                indices.append(k);k=(k+1)%n
            records.append(indices)
    return records


def _turns_back(a,b):
    norm=float(np.linalg.norm(a)*np.linalg.norm(b))
    return norm>1e-18 and float(np.dot(a,b)/norm)<=math.cos(math.radians(150.))


def _local_tangent(segment,t):
    return _at(segment,min(1.,t+.01))-_at(segment,max(0.,t-.01))


def _join_reversal_flags(new,old,first,last,ai,at,bi,bt):
    """Short joins can still turn backwards; expand the original span first."""
    a=np.asarray(old[first]['start'])-_at(new[ai],at)
    b=_at(new[bi],bt)-old[last]['end']
    return (_turns_back(_local_tangent(new[ai],at),a) or _turns_back(a,_local_tangent(old[first],0.)),
            _turns_back(_local_tangent(old[last],1.),b) or _turns_back(b,_local_tangent(new[bi],bt)))


def _added_joins_reverse(segments,joins):
    for j,segment in enumerate(segments):
        if segment not in joins:continue
        direction=np.asarray(segment['end'])-segment['start']
        if (_turns_back(_local_tangent(segments[j-1],1.),direction)
                or _turns_back(direction,_local_tangent(segments[(j+1)%len(segments)],0.))):
            return True
    return False


def _coalesce_runs(selected, maximum):
    """Reduce splice count by preserving additional short original intervals."""
    selected=selected.copy();added=0
    while len(_runs(selected))>maximum:
        gaps=_runs(~selected)
        shortest=min(gaps,key=lambda indices:(len(indices),indices[0]))
        selected[shortest]=True;added+=len(shortest)
    return selected,added


def _short_forward_arc(segments, ai, at, bi, bt):
    """Check contour direction independently of nonuniform segment density.

    This is only a correspondence screen; exact geometry is certified later.
    A source target may have thousands of samples on one original cubic and a
    handful on another edge, so segment counts cannot identify the short arc.
    """
    # The splice implementation retains the complement of one forward arc.
    # A wrap inside one segment needs a different split order; leave that
    # ambiguous case to contour expansion instead of constructing a wrong arc.
    if ai==bi and bt<=at:return False
    lengths=[]
    for segment in segments:
        if segment['type']=='line':
            lengths.append(float(np.linalg.norm(np.asarray(segment['end'])-segment['start'])))
        else:
            points=np.asarray([_at(segment,t) for t in np.linspace(0,1,17)])
            lengths.append(float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))
    cumulative=np.r_[0.,np.cumsum(lengths)];total=float(cumulative[-1])
    if total<=1e-12:return False
    def position(index,t):
        if segments[index]['type']=='line':return cumulative[index]+t*lengths[index]
        points=np.asarray([_at(segments[index],v) for v in np.linspace(0,t,17)])
        return cumulative[index]+float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    forward=(position(bi,bt)-position(ai,at))%total
    return 1e-12<forward<=total*.5+1e-9


def _splice(segments, original_run, *, maximum_join=None):
    a,ai,at=_nearest(segments,original_run[0]['start'])
    b,bi,bt=_nearest(segments,original_run[-1]['end'])
    n=len(segments);removed=(bi-ai)%n
    if not _short_forward_arc(segments,ai,at,bi,bt):
        raise ValueError('source_contacts_nonmonotone_arc_correspondence')
    if maximum_join is not None and max(a,b)>maximum_join+1e-9:
        raise ValueError('source_contacts_join_exceeds_half_working_pixel')
    if ai==bi:
        left,_=_split(segments[ai],at);_,right=_split(segments[bi],bt)
        keep=[right]+[segments[(bi+1+j)%n] for j in range(n-1)]+[left]
    else:
        _,right=_split(segments[bi],bt);left,_=_split(segments[ai],at)
        keep=[right]+[segments[(bi+1+j)%n] for j in range((ai-bi-1)%n)]+[left]
    join_a={'type':'line','start':keep[-1]['end'],'end':original_run[0]['start']}
    join_b={'type':'line','start':original_run[-1]['end'],'end':keep[0]['start']}
    result=keep+[join_a]+copy.deepcopy(original_run)+[join_b]
    return result, {'join_distances_working_pixels':[a,b],
                    'replaced_candidate_segments':removed+1,
                    'original_segments_preserved':len(original_run),
                    'added_joins':[join_a,join_b]}


def _reference_points(segments, *, step):
    """Keep source-field line vertices; sample only preserved original curves."""
    from geometry_error_optimizer import _sample_segment
    points=[segments[0]['start']]
    for segment in segments:
        if segment['type']=='line':points.append(segment['end'])
        else:points.extend(_sample_segment(segment,step,4096)[1:])
    result=np.asarray(points,dtype=float)
    keep=np.r_[True,np.any(np.diff(result,axis=0)!=0,axis=1)]
    result=result[keep]
    if len(result)>1 and np.array_equal(result[0],result[-1]):result=result[:-1]
    return result


def _boundary_outlier_boxes(scene_guard, target_id):
    """Select repair locations only; this grants no source-error allowance.

    A few reverse-distance outliers can reject an otherwise improved contour.
    Preserve their original spans and require the entire new proposal to pass
    the unchanged boundary and scene gates. Diagnostics may be capped; they
    never prove the remainder is sound.
    """
    evidence=scene_guard.get('source_boundary_evidence',{})
    reasons=set(evidence.get('reasons') or [])
    permitted={'source_boundary_source_to_vector_outside_contract',
               'source_boundary_vector_to_source_outside_contract'}
    if (evidence.get('verified') is not False or evidence.get('target_id')!=target_id
            or not reasons or not reasons<=permitted):
        return []
    tail=evidence.get('native_tail')
    if isinstance(tail,bool) or not isinstance(tail,(int,float)) or not math.isfinite(tail) or tail<=0:
        return []
    boxes=[]
    for row in evidence.get('worst_reverse_points',[])[:12]:
        point=np.asarray(row.get('xy',[]),dtype=float)
        distance=row.get('distance')
        if (point.shape!=(2,) or not np.isfinite(point).all()
                or isinstance(distance,bool) or not isinstance(distance,(int,float))
                or not math.isfinite(distance) or distance<=tail+1e-6):
            continue
        boxes.append(np.r_[point-.5,point+.5])
    return boxes


def preserve_source_edge_contacts(svg_text, proposal, original_rgba, processed_rgba,
                                   scene_guard, *, maximum_spans=8, maximum_seconds=30.,
                                   include_unverified_source_defects=False,
                                   include_boundary_outliers=False):
    """Create a new proposal with exact old contact spans, or fail closed.

    Defects must be native source-scene created/erased-hole observations for this
    exact before/candidate pair. The .5 working-pixel joins are additional new
    geometry, never described as preserved original segments.
    """
    import source_edge_reconstruction as source
    from clean_base import _parse_subpaths
    from gradient_contour_spans import _fit,_path
    from geometry_error_optimizer import (measure_fit_error,_error_contract_evidence,
                                          _combine_compound_loop_fits)
    from svg_postprocess import _designer_path_anchors
    started=time.monotonic()
    maximum_spans=min(8,max(0,int(maximum_spans)))
    seconds=min(30.,max(0.,float(maximum_seconds)))
    if maximum_spans==0 or seconds==0:
        raise ValueError('source_contacts_budget_exhausted')
    deadline=started+seconds
    def budget():
        if time.monotonic()>=deadline:raise ValueError('source_contacts_budget_exhausted')
    geometry=copy.deepcopy(proposal['geometry'])
    if not source.source_edge_certificate_valid(geometry):
        raise ValueError('source_contacts_parent_certificate_invalid')
    cert=geometry['source_edge_reconstruction'];original=source._rgba(original_rgba);processed=source._rgba(processed_rgba)
    if (cert['before_svg_sha256']!=source._sha(svg_text)
            or cert['source_rgba_sha256']!=source.rgba_sha256(original)
            or cert['processed_rgba_sha256']!=source.rgba_sha256(processed)):
        raise ValueError('source_contacts_source_or_parent_changed')
    provenance=scene_guard.get('provenance',{})
    if (scene_guard.get('accepted') is not False
            or provenance.get('before_svg_sha256')!=source._sha(svg_text)
            or provenance.get('after_svg_sha256')!=source._sha(proposal['candidate_svg_text'])):
        raise ValueError('source_contacts_defect_report_not_bound_to_candidate')
    w,h=cert['working_dimensions'];nw,nh=cert['source_dimensions']
    if [scene_guard.get('width'),scene_guard.get('height')]!=[nw,nh]:
        raise ValueError('source_contacts_defect_canvas_changed')
    if (include_unverified_source_defects
            and scene_guard.get('source_boundary_evidence',{}).get('verified') is not True):
        raise ValueError('source_contacts_source_fallback_requires_verified_paper_envelope')
    source_kinds={'new_gap_on_source_ink','new_paint_on_source_empty','coverage_change_source_ambiguous',
                  'new_white_exposure_on_source_ink','new_color_on_source_empty','new_local_color_error',
                  'paint_change_source_ambiguous','source_feature_color_error_increased'}
    regions=[]
    for row in scene_guard.get('localized_defects',[]):
        if (not str(row.get('kind','')).startswith(('created_hole','erased_hole'))
                and not (include_unverified_source_defects and row.get('kind') in source_kinds)):
            continue
        box=np.asarray(row.get('bbox_xyxy',[]),dtype=float)
        if box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[:2]>=box[2:]):
            raise ValueError('source_contacts_invalid_defect_region')
        regions.append(_native_box_to_viewbox(box,(w,h),(nw,nh)))
    if include_boundary_outliers:
        regions.extend(_native_box_to_viewbox(box,(w,h),(nw,nh))
                       for box in _boundary_outlier_boxes(scene_guard,cert['target_id']))
    if not regions or len(regions)>100:
        raise ValueError('source_contacts_no_bounded_hole_regions')
    root,old_target,*_=source._context(svg_text,cert['target_id'],(h,w))
    result_root,target,*_=source._context(proposal['candidate_svg_text'],cert['target_id'],(h,w))
    original_fits=[_fit(s) for s in _parse_subpaths(old_target.get('d',''))]
    fits=[_fit(s) for s in _parse_subpaths(target.get('d',''))]
    loops=[np.asarray(loop,dtype=float) for loop in cert['source_target_loops']]
    if len(fits)!=len(loops):raise ValueError('source_contacts_loop_identity_unavailable')
    outer_index=max(range(len(fits)),key=lambda i:abs(_signed_area(fits[i]['segments'])))
    original_index=max(range(len(original_fits)),key=lambda i:abs(_signed_area(original_fits[i]['segments'])))
    old=original_fits[original_index]['segments'];new=fits[outer_index]['segments']
    if len(old)>4096 or len(new)>4096:raise ValueError('source_contacts_segment_budget_exceeded')
    if _signed_area(old)*_signed_area(new)<0:old=_reverse(old)
    points=loops[outer_index]
    target_segments=[{'type':'line','start':a.tolist(),'end':b.tolist()}
                     for a,b in zip(points,np.roll(points,-1,axis=0))]
    if _signed_area(target_segments)*_signed_area(new)<0:target_segments=_reverse(target_segments)
    selected=np.zeros(len(old),bool)
    prior=cert.get('contact_preservation',{})
    prior_segments={source._json_sha(s) for s in prior.get('preserved_segments',[])}
    for j,segment in enumerate(old):
        if source._json_sha(segment) in prior_segments:selected[j]=True
    for box in regions:
        lo,hi=box[:2]-3.,box[2:]+3.
        for j,segment in enumerate(old):
            samples=np.asarray([_at(segment,t) for t in np.linspace(0,1,13)])
            if np.any(np.all((samples>=lo)&(samples<=hi),axis=1)):selected[j]=True
    if not selected.any():raise ValueError('source_contacts_no_original_boundary_at_defects')
    for _ in range(48):
        budget();expand=[]
        runs=_runs(selected)
        # Neighbouring tiny runs can coalesce during bounded contour expansion.
        # The published span cap applies to the final preserved intervals.
        if len(runs)>64:raise ValueError('source_contacts_too_many_initial_fragments')
        for indices in runs:
            first,last=indices[0],indices[-1]
            da,ai,at=_nearest(new,old[first]['start']);db,bi,bt=_nearest(new,old[last]['end'])
            order=not _short_forward_arc(new,ai,at,bi,bt)
            reverse_start,reverse_end=_join_reversal_flags(new,old,first,last,ai,at,bi,bt)
            if da>.5 or order or reverse_start:expand.append((first-1)%len(old))
            if db>.5 or order or reverse_end:expand.append((last+1)%len(old))
        if not expand:break
        selected[expand]=True
    records=[];preserved=[]
    selected,coalesced_count=_coalesce_runs(selected,maximum_spans)
    final_runs=_runs(selected)
    if len(final_runs)>maximum_spans:raise ValueError('source_contacts_too_many_separate_spans')
    for indices in final_runs:
        budget();run=[old[i] for i in indices]
        new,record=_splice(new,run,maximum_join=.5)
        target_segments,reference_record=_splice(target_segments,run)
        record.update(original_oriented_indices=indices,
                      reference_join_distances_working_pixels=reference_record['join_distances_working_pixels'])
        records.append(record);preserved.extend(copy.deepcopy(run))
    # A later splice may not silently consume an earlier preserved segment.
    segment_keys={source._json_sha(s) for s in new}
    if any(source._json_sha(s) not in segment_keys for s in preserved):
        raise ValueError('source_contacts_overlapping_spans_lost_original_segments')
    if _added_joins_reverse(new,[join for row in records for join in row['added_joins']]):
        raise ValueError('source_contacts_new_join_turns_back')
    fits[outer_index]=_fit(_parse_subpaths(_path(new,True))[0])
    combined=_combine_compound_loop_fits(fits)
    scale=float(np.linalg.norm(np.ptp(loops[outer_index],axis=0)))
    loops[outer_index]=_reference_points(target_segments,step=min(.1,scale*.0001))
    budget()
    error=measure_fit_error(loops,combined,error_budget_percent=cert['error_budget_percent'])
    contract=_error_contract_evidence(error,cert['error_budget_percent'])
    if not contract['within_budget']:
        raise ValueError('source_contacts_hybrid_target_error_contract_failed')
    path=combined['path'];count=_designer_path_anchors(path)
    original_count=_designer_path_anchors(old_target.get('d',''))
    if count>=original_count:
        raise ValueError('source_contacts_no_net_anchor_reduction')
    loop_values=[loop.tolist() for loop in loops]
    contact={'schema':SCHEMA,'status':'proposal_only','scope':'source_free_edges_plus_exact_original_contact_segments_and_new_bounded_joins',
             'original_path_sha256':source._sha(old_target.get('d','')),
             'parent_candidate_path_sha256':cert['after_path_sha256'],
             'parent_source_target_loops_sha256':cert['source_target_loops_sha256'],
             'defect_report_sha256':source._json_sha(scene_guard),
             'defect_coordinate_mapping':'native_canvas_to_default_xMidYMid_meet_viewBox',
             'preserves_unverified_source_regions_after_verified_paper_envelope':bool(include_unverified_source_defects),
             'preserves_original_spans_at_unverified_boundary_outliers':bool(include_boundary_outliers),
             'prior_preserved_segment_count':len(prior_segments),
             'preserved_segments':preserved,'preserved_segments_sha256':source._json_sha(preserved),
             'preserved_segment_count':len(preserved),'original_outer_loop_index':original_index,
             'additional_original_segments_preserved_to_coalesce_spans':coalesced_count,
             'candidate_outer_loop_index':outer_index,'spans':records,
             'maximum_join_working_pixels':.5,'added_joins_are_not_original_geometry':True,
             'new_join_direction_screen':copy.deepcopy(JOIN_DIRECTION_SCREEN),
             'maximum_spans':maximum_spans,'search_budget_seconds':seconds,
             'elapsed_seconds':time.monotonic()-started,
             'one_measurement_and_final_scene_validation_may_overrun_budget':True,
             'hybrid_reference_is_not_entirely_source_derived':True,
             'full_scene_revalidation_required':True}
    cert.update(after_path_sha256=source._sha(path),source_target_loops=loop_values,
                source_target_loops_sha256=source._json_sha(loop_values),
                source_target_point_count=sum(len(loop) for loop in loop_values),
                measurement_sha256=source._json_sha(error),error_contract=contract,
                error_reference='hybrid_source_coverage_ownership_proposal_and_exact_original_contacts',
                contact_preservation=contact)
    candidate_id='source-contact-preservation-v1'
    geometry.update(path=path,fit=combined,actual_error=error,anchor_count=count,designer_anchor_count=count,
                    segment_count=combined['segment_count'],anchors_after=count,
                    segment_count_after=combined['segment_count'],actual_p95_error_percent=error['p95_percent'],
                    actual_max_error_percent=error['max_percent'],
                    anchors_before=original_count,anchors_removed=original_count-count,
                    anchor_reduction_ratio=(original_count-count)/original_count,
                    normalization={'basis':'hybrid_target_bbox_diagonal','scale':error['normalization_scale'],
                                   'source_bbox':error['source_bbox']},
                    status='selected_source_proposal_with_exact_original_contacts',
                    selected_candidate_id=candidate_id,selected_source='source_edge_contacts.bounded_original_contact_preservation',
                    candidate_count=1,eligible_candidate_count=1,
                    candidates=[{'candidate_id':candidate_id,'eligible':True,'fit_succeeded':True,
                                 'within_error_budget':True,'anchors_after':count,
                                 'designer_anchor_count':count,'not_global_minimum':True}],
                    selection_evidence={'identity_rollback_selected':False,'selected_candidate_id':candidate_id,
                        'candidate_count':1,'selection_basis':'source_proposal_with_exact_original_contact_rollback',
                        'not_global_minimum':True})
    geometry.pop('source_edge_scene_commit',None)
    geometry['error_budget'].update(actual_p95_error_percent=error['p95_percent'],actual_max_error_percent=error['max_percent'],
                                   reference='explicit_hybrid_target_with_exact_original_contacts')
    geometry['topology']['reference']='explicit_hybrid_target_not_prior_mask_or_author_intent'
    target.set('d',path)
    for key,value in {'data-avc-designer-anchors':count,'data-avc-p95-error-percent':error['p95_percent'],
                      'data-avc-max-error-percent':error['max_percent']}.items():target.set(key,str(value))
    return {'path':path,'geometry':geometry,'certificate':cert,'source_target_loops':loop_values,
            'candidate_svg_text':ET.tostring(result_root,encoding='unicode')}


def contact_certificate_valid(geometry):
    """Verify exact retained curve coordinates and bounded additional joins."""
    import source_edge_reconstruction as source
    from clean_base import _parse_subpaths
    from gradient_contour_spans import _fit
    try:
        cert=geometry['source_edge_reconstruction'];proof=cert['contact_preservation']
        if proof.get('schema')!=SCHEMA or proof.get('full_scene_revalidation_required') is not True:
            return False
        if proof.get('hybrid_reference_is_not_entirely_source_derived') is not True or proof.get('added_joins_are_not_original_geometry') is not True:
            return False
        preserved=proof['preserved_segments']
        if not preserved or len(preserved)!=proof['preserved_segment_count'] or source._json_sha(preserved)!=proof['preserved_segments_sha256']:
            return False
        subpaths=[_fit(sub)['segments'] for sub in _parse_subpaths(geometry['path'])]
        all_segments=[segment for segments in subpaths for segment in segments]
        available={source._json_sha(s) for s in all_segments}
        if any(source._json_sha(s) not in available for s in preserved):return False
        if proof.get('original_path_sha256')!=cert['before_path_sha256'] or proof.get('maximum_join_working_pixels')!=.5:
            return False
        spans=proof['spans']
        if not 1<=len(spans)<=8 or sum(s['original_segments_preserved'] for s in spans)!=len(preserved):return False
        for row in spans:
            joins=row['added_joins'];lengths=row['join_distances_working_pixels']
            if len(joins)!=2 or len(lengths)!=2:return False
            for join,length in zip(joins,lengths):
                actual=float(np.linalg.norm(np.asarray(join['end'])-join['start']))
                if (not math.isfinite(actual) or actual>.5+1e-9 or abs(actual-float(length))>1e-8
                        or source._json_sha(join) not in available):return False
        if 'new_join_direction_screen' in proof:
            if proof['new_join_direction_screen']!=JOIN_DIRECTION_SCREEN:return False
            joins=[join for row in spans for join in row['added_joins']]
            if any(_added_joins_reverse(segments,joins) for segments in subpaths):return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):
        return False
