"""Independent native-source evidence for a *scoped* free-paper edge allowance.

An optimizer's certificate only binds the proposal. This module remeasures the
original image. It cannot certify contacts, holes, thin features or unknown
interiors. An incomplete certificate returns an empty allowance, never a pass.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import io
import math
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

SCHEMA = "aivc.source-boundary-evidence/v1"


def _sha(value):
    return hashlib.sha256(value.encode("utf8") if isinstance(value, str) else value).hexdigest()


def _render(root, width, height):
    import resvg_py
    from svg_renderer import svg_with_native_viewport
    svg = svg_with_native_viewport(ET.tostring(root, encoding="unicode"), width, height)
    png = resvg_py.svg_to_bytes(svg_string=svg,
                               width=width, height=height, background=None,
                               skip_system_fonts=True, log_information=False,
                               shape_rendering="geometric_precision")
    with Image.open(io.BytesIO(png)) as image:
        if image.size != (width, height):
            raise ValueError("source_boundary_native_viewport_required")
        return np.asarray(image.convert("RGBA")).copy()


def _segments_index(segments):
    from geometry_error_optimizer import _build_segment_bvh
    segments = np.asarray(segments, dtype=float)
    if not len(segments):
        raise ValueError("source_boundary_no_segments")
    starts, ends = segments[:, 0], segments[:, 1]
    delta = ends - starts
    tree = _build_segment_bvh(np.nextafter(np.minimum(starts, ends), -np.inf),
                             np.nextafter(np.maximum(starts, ends), np.inf),
                             (starts + ends) / 2, np.arange(len(starts)), 16)
    return starts, delta, np.sum(delta * delta, axis=1), tree


def _distances(points, index):
    from geometry_error_optimizer import _bvh_nearest_squared
    return np.concatenate([np.sqrt(_bvh_nearest_squared(points[i:i+512], index))
                           for i in range(0, len(points), 512)]) if len(points) else np.empty(0)


def _supported_normal_projection(points, segments):
    """Exclude extrapolation past a fragment's unsupported dangling endpoint.

    Reliable source arcs can end where paper/paint evidence ends. Distance to
    that endpoint alone does not make the following unknown curve a free edge.
    This membership is determined for the BEFORE arcs, never by excluding an
    AFTER arc that has moved away from the certified source.
    """
    starts,ends=segments[:,0],segments[:,1]
    delta=ends-starts; denominator=np.sum(delta*delta,axis=1)
    degree=Counter(tuple(np.round(p,7)) for p in segments.reshape(-1,2))
    connected_start=np.array([degree[tuple(np.round(p,7))]>1 for p in starts])
    connected_end=np.array([degree[tuple(np.round(p,7))]>1 for p in ends])
    out=[]
    for offset in range(0,len(points),128):
        block=points[offset:offset+128]
        t=np.sum((block[:,None,:]-starts)*delta,axis=2)/np.maximum(denominator,1e-12)
        projected=starts+np.clip(t,0,1)[:,:,None]*delta
        dist=np.sum((block[:,None,:]-projected)**2,axis=2)
        nearest=dist.min(axis=1)
        supported=((t>1e-7)&(t<1-1e-7)) | ((t<=1e-7)&connected_start) | ((t>=1-1e-7)&connected_end)
        out.append(np.any(supported & (dist<=nearest[:,None]+1e-9),axis=1))
    return np.concatenate(out) if out else np.empty(0,bool)


def _normal_strip_footprint_overlap(points,segments,radius):
    """Exact pixel-square overlap with certified finite normal strips.

    A native pixel samples a square of side one, not its centre alone. Each
    reliable source segment contributes its oriented rectangle (segment times
    the normal interval [-radius,radius]). The separating-axis theorem tests
    intersection of that rectangle with the pixel square, without extrapolating
    a dangling source endpoint. Connected source vertices additionally supply
    their normal-cone disk; isolated endpoints receive no disk. Confidence and
    protected-feature masks are applied separately at the actual source pixel.
    This changes raster sampling scope only, never the fixed continuous arcs
    used for before/after geometry measurements.
    """
    segments=np.asarray(segments,float)
    delta=segments[:,1]-segments[:,0]
    lengths=np.linalg.norm(delta,axis=1)
    use=lengths>1e-12
    segments,delta,lengths=segments[use],delta[use],lengths[use]
    if not len(segments):
        return np.zeros(len(points),bool)
    tangent=delta/lengths[:,None]
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    middle=segments.mean(axis=1)
    half_box=np.abs(delta)/2+radius*np.abs(normal)
    square_tangent_extent=.5*np.sum(np.abs(tangent),axis=1)
    square_normal_extent=.5*np.sum(np.abs(normal),axis=1)
    degree=Counter(tuple(np.round(p,7)) for p in segments.reshape(-1,2))
    vertices=np.array([p for p,count in degree.items() if count>1],float)
    answers=[]
    for offset in range(0,len(points),128):
        block=np.asarray(points[offset:offset+128],float)
        displacement=block[:,None,:]-middle
        # Axes of the pixel square and of the oriented source rectangle are
        # all required. Projection onto only the tangent would be insufficient
        # for diagonal endpoint cases.
        overlap=np.all(np.abs(displacement)<=half_box+.5+1e-9,axis=2)
        overlap &= np.abs(np.sum(displacement*tangent,axis=2))<=lengths/2+square_tangent_extent+1e-9
        overlap &= np.abs(np.sum(displacement*normal,axis=2))<=radius+square_normal_extent+1e-9
        supported=np.any(overlap,axis=1)
        if len(vertices):
            outside=np.maximum(np.abs(block[:,None,:]-vertices)-.5,0)
            supported |= np.any(np.sum(outside*outside,axis=2)<=radius*radius+1e-9,axis=1)
        answers.append(supported)
    return np.concatenate(answers) if answers else np.empty(0,bool)


def _path_segments(path, scale):
    """Bounded dense sampling; no artificial segments between closed subpaths."""
    from clean_base import _parse_subpaths
    result = []
    for subpath in _parse_subpaths(path):
        if not subpath["closed"]:
            raise ValueError("source_boundary_open_path")
        start = np.asarray(subpath["start"], dtype=float) * scale
        first = start.copy()
        for segment in subpath["segs"]:
            kind = segment[0]
            if kind == "L":
                end = np.asarray(segment[1:3], dtype=float) * scale
                steps = max(1, int(math.ceil(np.linalg.norm(end-start) / .2)))
                if steps>100_000:
                    raise ValueError("source_boundary_sampling_budget")
                points = start + np.linspace(0, 1, steps+1)[:, None] * (end-start)
            elif kind == "C":
                controls = np.asarray(segment[1:], dtype=float).reshape(3, 2) * scale
                p1, p2, end = controls
                length = np.linalg.norm(p1-start)+np.linalg.norm(p2-p1)+np.linalg.norm(end-p2)
                steps = max(2, int(math.ceil(length / .2)))
                if steps > 100_000:
                    raise ValueError("source_boundary_sampling_budget")
                t = np.linspace(0, 1, steps+1)[:, None]
                points = (1-t)**3*start+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*end
            else:
                raise ValueError("source_boundary_unsupported_path_command")
            result.extend(np.stack((points[:-1], points[1:]), axis=1))
            start = end
        if np.linalg.norm(start-first) > 1e-8:
            steps = max(1, int(math.ceil(np.linalg.norm(first-start) / .2)))
            points = start+np.linspace(0, 1, steps+1)[:, None]*(first-start)
            result.extend(np.stack((points[:-1], points[1:]), axis=1))
    if len(result) > 250_000:
        raise ValueError("source_boundary_sampling_budget")
    return np.asarray(result)


def _crossing_segments(field, reliable):
    """Native pixel-centre marching squares, excluding ambiguous saddles/gaps."""
    from trace_engine import _interp
    pairs = {1:(3,0),2:(0,1),3:(3,1),4:(1,2),6:(0,2),7:(3,2),
             8:(2,3),9:(0,2),11:(1,2),12:(3,1),13:(0,1),14:(3,0)}
    v = (field[:-1,:-1], field[:-1,1:], field[1:,1:], field[1:,:-1])
    case = sum((a >= .5).astype(np.uint8) << i for i,a in enumerate(v))
    use = (reliable[:-1,:-1] & reliable[:-1,1:] & reliable[1:,1:] & reliable[1:,:-1]
           & (case != 0) & (case != 15) & (case != 5) & (case != 10))
    out = []
    for y,x in zip(*np.nonzero(use)):
        corners = ((x+.5,y+.5),(x+1.5,y+.5),(x+1.5,y+1.5),(x+.5,y+1.5))
        edges = ((0,1),(1,2),(3,2),(0,3))
        endpoints = []
        for edge in pairs[int(case[y,x])]:
            a,b = edges[edge]
            endpoints.append(_interp(.5,corners[a],corners[b],v[a][y,x],v[b][y,x]))
        out.append(endpoints)
    return np.asarray(out, dtype=float)


def _shift(mask, dy, dx):
    h,w = mask.shape
    result = np.zeros_like(mask)
    y0,y1,x0,x1 = max(0,-dy),min(h,h-dy),max(0,-dx),min(w,w-dx)
    result[y0:y1,x0:x1] = mask[y0+dy:y1+dy,x0+dx:x1+dx]
    return result


def _thin(region, opposite, width):
    """Protect opposed banks within a geometry-derived width, incl. diagonals."""
    out = np.zeros_like(region)
    for dy,dx in ((0,1),(1,0),(1,1),(1,-1)):
        a = np.full(region.shape, 255, np.uint16)
        b = a.copy()
        for radius in range(1,width+2):
            a[(a == 255) & _shift(opposite,dy*radius,dx*radius)] = radius
            b[(b == 255) & _shift(opposite,-dy*radius,-dx*radius)] = radius
        out |= region & (a+b <= width+1)
    return out


def _stats(values):
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("source_boundary_insufficient_distance_samples")
    return {"count":len(values),"mean":float(values.mean()),
            "p95":float(np.percentile(values,95)),"max":float(values.max())}


def _preserved_contact_segments(certificate,scale):
    """Explicit hybrid contacts/joins must never be described as free edges."""
    proof=certificate.get("contact_preservation",{})
    primitives=list(proof.get("preserved_segments",[]))
    for span in proof.get("spans",[]):
        primitives.extend(span.get("added_joins",[]))
    out=[]
    for primitive in primitives:
        start=np.asarray(primitive["start"],float)*scale
        end=np.asarray(primitive["end"],float)*scale
        if primitive["type"]=="line":
            length=np.linalg.norm(end-start)
            controls=None
        elif primitive["type"]=="cubic":
            c1=np.asarray(primitive["control1"],float)*scale
            c2=np.asarray(primitive["control2"],float)*scale
            length=np.linalg.norm(c1-start)+np.linalg.norm(c2-c1)+np.linalg.norm(end-c2)
            controls=(c1,c2)
        else:
            raise ValueError("source_boundary_unsupported_contact_geometry")
        steps=max(2,int(math.ceil(length/.2)))
        if steps>100_000:
            raise ValueError("source_boundary_sampling_budget")
        t=np.linspace(0,1,steps+1)[:,None]
        points=start+t*(end-start) if controls is None else (
            (1-t)**3*start+3*(1-t)**2*t*controls[0]+3*(1-t)*t*t*controls[1]+t**3*end)
        out.extend(np.stack((points[:-1],points[1:]),axis=1))
    return np.asarray(out)


def _scene_signature(root, target_id):
    root = copy.deepcopy(root)
    target = next(n for n in root.iter() if n.get("id") == target_id)
    for key in list(target.attrib):
        if key == "d" or key.startswith("data-avc-"):
            target.attrib.pop(key)
    # Ignore serialization-only namespace/attribute ordering, retain all text,
    # structural and presentation changes (including every other drawable).
    def signature(node):
        return (node.tag, sorted(node.attrib.items()), (node.text or "").strip(),
                (node.tail or "").strip(), [signature(n) for n in node])
    return signature(root)


def _crossfit_source_material(rgb, owner, paper, query):
    """Independent held-out material witnesses from native original pixels.

    Each fixed spatial split supplies four observed high-contrast samples; its
    RGB medoid must explain the other split's witnesses within the existing
    12-RGB-RMS unmix limit. The queried pixel never trains either model. A
    15-pixel window permits four witnesses per split along a narrow dark rim;
    it grants no authority over thin features or other-object contacts.
    """
    h,w=owner.shape
    models=[np.zeros_like(rgb),np.zeros_like(rgb)]
    counts=np.zeros((h,w),dtype=np.uint8)
    ys,xs=np.nonzero(query)
    rejected=Counter()
    maximum_spread=0.
    for y,x in zip(ys.tolist(),xs.tolist()):
        y0,y1=max(0,y-7),min(h,y+8);x0,x1=max(0,x-7),min(w,x+8)
        valid=owner[y0:y1,x0:x1].copy();valid[y-y0,x-x0]=False
        yy,xx=np.nonzero(valid)
        if len(xx)<8:
            rejected['fewer_than_eight_independent_samples']+=1;continue
        colours=rgb[y0+yy,x0+xx]
        hashed=((x0+xx).astype(np.uint64)*73856093)^((y0+yy).astype(np.uint64)*19349663)
        split=((hashed>>np.uint64(7))&np.uint64(1)).astype(bool)
        strength=np.linalg.norm(colours-paper,axis=1)
        endpoints=[];witnesses=[]
        for side in (False,True):
            values=colours[split==side];norms=strength[split==side]
            if len(values)<4:break
            top=values[np.argsort(norms)[-4:]]
            median=np.median(top,axis=0)
            endpoints.append(top[np.argmin(np.sum((top-median)**2,axis=1))])
            witnesses.append(top)
        if len(endpoints)!=2:
            rejected['insufficient_witnesses_in_one_split']+=1;continue
        directions=[endpoint-paper for endpoint in endpoints]
        norms=[float(d@d) for d in directions]
        if min(norms)<2500:
            rejected['insufficient_material_contrast']+=1;continue
        if np.sqrt(np.mean((endpoints[0]-endpoints[1])**2))>12:
            rejected['split_material_disagreement']+=1;continue
        credible=True
        for side in (0,1):
            heldout=witnesses[1-side];direction=directions[side]
            coverage=np.clip((heldout-paper)@direction/norms[side],0,1)
            residual=np.sqrt(np.mean((heldout-(paper+coverage[:,None]*direction))**2,axis=1))
            if np.max(residual)>12:credible=False;break
        if not credible:
            rejected['heldout_material_unexplained']+=1;continue
        observed=rgb[y,x]-paper
        coverage=[float(np.clip(observed@d/n,0,1)) for d,n in zip(directions,norms)]
        if max(np.sqrt(np.mean((observed-c*d)**2)) for c,d in zip(coverage,directions))>12:
            rejected['query_unexplained_by_both_splits']+=1;continue
        for side in (0,1):models[side][y,x]=endpoints[side]
        counts[y,x]=4
        maximum_spread=max(maximum_spread,abs(coverage[0]-coverage[1]))
    return models,counts,{'method':'native_spatial_split_observed_rgb_medoids',
        'window_side_native':15,'witnesses_per_split':4,'query_excluded_from_training':True,
        'cross_heldout_rms_maximum':12,'supported_pixels':int((counts>3).sum()),
        'query_pixels':len(xs),'rejections':dict(rejected),
        'maximum_supported_coverage_disagreement':maximum_spread}


def _protected_feature_footprints(mask, native_tail):
    """Protect both banks of a true thin feature, not just its centre pixels.

    Otherwise a one-pixel stroke could gain new ink on adjacent source paper
    while its protected centre stayed unchanged. The radius is derived from
    the existing geometric envelope plus the native sampling footprint.
    """
    from source_edge_reconstruction import _morph
    radius=int(math.ceil(native_tail+math.sqrt(.5)))
    return _morph(mask,2*radius+1)


def _source_feature_regions(features, affected, native_tail):
    """Bound exact local supports for necessary per-feature source-cost gates.

    These masks are evidence, not an expanded frozen region. Real feature
    centres remain hard-protected; their surrounding banks may improve within
    the existing geometry contract, but cannot spend another feature's gain.
    """
    from source_scene_guard import _components, MAX_HOLES, MAX_SCENE_PIXELS
    near_affected=_protected_feature_footprints(affected,native_tail)
    h,w=affected.shape;radius=int(math.ceil(native_tail+math.sqrt(.5)))
    rows=[];area=0
    for kind,feature_mask in features.items():
        labels,_=_components(feature_mask)
        for label in np.unique(labels[near_affected]):
            if not label:continue
            yy,xx=np.nonzero(labels==label)
            x0=max(0,int(xx.min())-radius);x1=min(w,int(xx.max())+radius+1)
            y0=max(0,int(yy.min())-radius);y1=min(h,int(yy.max())+radius+1)
            support=_protected_feature_footprints(labels[y0:y1,x0:x1]==label,native_tail)
            area+=support.size
            if len(rows)>=MAX_HOLES or area>MAX_SCENE_PIXELS*4:
                raise ValueError('source_boundary_feature_support_budget_exceeded')
            rows.append({'kind':kind,'bbox_xyxy':[x0,y0,x1,y1],
                'feature_pixels':len(xx),
                'support_pixels':int(support.sum()),
                'support_mask_packbits_hex':np.packbits(support).tobytes().hex()})
    return rows


def _build_source_boundary_evidence_once(before_svg_text, after_svg_text, source_rgba,
                                   processed_rgba, geometry, split_model, material_cache):
    """Return ``(native_boolean_allowance, JSON evidence)``; never scene approval.

    Budget is the bound in the original independently validated geometry proof.
    Both directions must improve mean/P95, P95 must fit the original budget and
    maximum distance its 3x tail. Candidate samples outside reliable source
    bands are measured, not silently dropped. Uncovered pixels remain strict.
    """
    from source_scene_guard import _array, _source_masks, _components
    from source_edge_reconstruction import (_context, _render_mask, _morph, _box_mean,
                                            _json_sha, rgba_sha256, _native_mapping,
                                            source_edge_certificate_valid, DRAWABLES)
    source = _array(source_rgba)
    h,w = source.shape[:2]
    empty = np.zeros((h,w), dtype=bool)
    evidence = {"schema":SCHEMA,"verified":False,"scope":"native_free_paper_edges_only",
                "reasons":[],"contacts_authorized":False,"holes_authorized":False}
    try:
        if not source_edge_certificate_valid(geometry):
            raise ValueError("source_boundary_geometry_certificate_invalid")
        if processed_rgba is None:
            raise ValueError("source_boundary_processed_reference_required")
        processed = _array(processed_rgba)
        if processed.shape != source.shape:
            raise ValueError("source_boundary_native_dimensions_required")
        cert = geometry["source_edge_reconstruction"]
        if (cert["source_rgba_sha256"] != rgba_sha256(source)
                or cert["processed_rgba_sha256"] != rgba_sha256(processed)
                or cert["before_svg_sha256"] != _sha(before_svg_text)):
            raise ValueError("source_boundary_source_or_before_binding_mismatch")
        ww,wh = cert["working_dimensions"]
        if (any(isinstance(v,bool) or not isinstance(v,int) or not 1<=v<=8192 for v in (ww,wh))
                or cert.get("source_dimensions") != [w,h]
                or cert.get("processed_dimensions") != [w,h]):
            raise ValueError("source_boundary_certificate_dimensions_mismatch")
        scale, offset = _native_mapping((ww,wh),(w,h))
        old,target,gid,context,paint = _context(before_svg_text,cert["target_id"],(wh,ww))
        new,new_target,new_gid,new_context,new_paint = _context(after_svg_text,cert["target_id"],(wh,ww))
        if (cert["before_path_sha256"] != _sha(target.get("d",""))
                or cert["after_path_sha256"] != _sha(new_target.get("d",""))
                or new_target.get("d") != geometry["path"]
                or paint != new_paint or paint != cert["paint_sha256"]
                or gid != new_gid or context != new_context
                or _json_sha(context) != cert["presentation_context_sha256"]
                or target.get("data-avc-gradient-object") != new_target.get("data-avc-gradient-object")
                or _scene_signature(old,cert["target_id"]) != _scene_signature(new,cert["target_id"])):
            raise ValueError("source_boundary_path_paint_context_or_scene_binding_mismatch")
        fg,bg,support,paper = _source_masks(source,processed)
        if not support["confident"] or paper is None or np.any(source[:,:,3] != 255):
            raise ValueError("source_boundary_requires_confident_opaque_paper")
        budget = float(geometry["actual_error"]["normalization_scale"]) * scale * float(cert["error_budget_percent"]) / 100
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("source_boundary_invalid_native_budget")
        tail = 3*budget
        feature_width=max(1,int(math.ceil(2*tail)))
        if feature_width>16:
            # The bounded native feature search below has been validated up to
            # 16 pixels. Capping the search while allowing a larger envelope
            # would silently expose wider true channels and thin strokes.
            raise ValueError("source_boundary_feature_envelope_exceeds_supported_native_width")
        # Never claim a contact with a different object. Keep surrounding paint
        # in its true stacking order; no white=transparent shortcut.
        others = copy.deepcopy(old)
        for parent in list(others.iter()):
            for child in list(parent):
                if child.get("id") == cert["target_id"]:
                    parent.remove(child)
        others_alpha = _render(others,w,h)[:,:,3]
        contact = _morph(others_alpha >= 32, 5)
        mask = _render_mask(old,target,w,h)
        rgb = source[:,:,:3].astype(float)
        core = _morph(mask,5,erode=True) & fg & ~contact & (processed[:,:,3] >= 240)
        inside,count = _box_mean(rgb,core,5)
        base_direction=inside-paper
        base_norm=np.sum(base_direction*base_direction,axis=2)
        base_coverage=np.clip(np.sum((rgb-paper)*base_direction,axis=2)/np.maximum(base_norm,1e-9),0,1)
        base_residual=np.sqrt(np.mean((rgb-(paper+base_coverage[:,:,None]*base_direction))**2,axis=2))
        base_credible=(count>3)&(base_norm>=2500)&(base_residual<=12)
        near_paper = _morph(bg & ~mask, 11)
        if not material_cache:
            query=near_paper & _morph(mask,11) & ~contact & ~base_credible & ~bg
            material_cache.update(zip(('models','counts','metadata'),_crossfit_source_material(
                rgb,mask & fg & ~contact & (processed[:,:,3]>=240),paper,query)))
        supplement=material_cache['counts']>3
        inside[~base_credible]=material_cache['models'][split_model][~base_credible]
        count[~base_credible]=material_cache['counts'][~base_credible]
        direction = inside-paper
        norm = np.sum(direction*direction,axis=2)
        coverage = np.clip(np.sum((rgb-paper)*direction,axis=2)/np.maximum(norm,1e-9),0,1)
        residual = np.sqrt(np.mean((rgb-(paper+coverage[:,:,None]*direction))**2,axis=2))
        # Confident original paper already establishes zero coverage. It does
        # not require foreground colour witnesses at every paper pixel; only
        # finite neighbouring certified source edges can give it an allowance.
        coverage[bg]=0
        credible = (((count>3) & (norm>=2500) & (residual<=12)) | bg) & ~contact
        reliable = credible & near_paper & _morph(mask,11)
        source_segments = _crossing_segments(coverage,reliable)
        before_segments = _path_segments(target.get("d"),scale) + offset
        after_segments = _path_segments(new_target.get("d"),scale) + offset
        before_index,after_index = map(_segments_index,(before_segments,after_segments))
        contact_segments=_preserved_contact_segments(cert,scale)
        contact_index=None
        if len(contact_segments):
            contact_segments = contact_segments + offset
            contact_index=_segments_index(contact_segments)
            middles=source_segments.mean(axis=1)
            # Fixed source and before geometry decide membership. Removing a
            # contact here never grants an allowance there; it remains strict.
            contact_distance=_distances(middles,contact_index)
            original_distance=_distances(middles,before_index)
            source_segments=source_segments[contact_distance>original_distance+1e-5]
        if len(source_segments) < 16:
            raise ValueError("source_boundary_insufficient_native_paper_edges")
        source_index = _segments_index(source_segments)
        # Uniform samples make both directions comparable across different node
        # counts, while exact segment-distance queries avoid inter-fragment joins.
        source_points = np.concatenate([s[0]+np.linspace(0,1,max(2,int(math.ceil(np.linalg.norm(s[1]-s[0])/.2))))[:,None]*(s[1]-s[0]) for s in source_segments])
        before_points = before_segments.mean(axis=1)
        after_points = after_segments.mean(axis=1)
        before_source_distance = _distances(before_points,source_index)
        # Certification membership comes from the fixed before/source geometry.
        # Once a candidate sample belongs to a free arc, it is measured even if
        # it leaves the source band. All other arcs receive no allowance.
        free = before_source_distance <= tail + math.sqrt(.5)
        free &= _supported_normal_projection(before_points,source_segments)
        if contact_index is not None:
            free &= _distances(before_points,contact_index)>.01
        iy = np.clip(before_points[:,1].astype(int),0,h-1)
        ix = np.clip(before_points[:,0].astype(int),0,w-1)
        free &= ~contact[iy,ix]
        if not np.any(free):
            raise ValueError("source_boundary_no_fixed_free_arcs")
        free_index = _segments_index(before_segments[free])
        after_free = np.ones(len(after_points),dtype=bool)
        if np.any(~free):
            other_index = _segments_index(before_segments[~free])
            after_free = _distances(after_points,free_index)+1e-8 < _distances(after_points,other_index)
        if contact_index is not None:
            after_free &= _distances(after_points,contact_index)>.01
        after_reverse_distances = _distances(after_points[after_free],source_index)
        measurements = {
            "source_to_vector":{"before":_stats(_distances(source_points,before_index)),
                                "after":_stats(_distances(source_points,after_index))},
            "vector_to_source":{"before":_stats(before_source_distance[free]),
                                "after":_stats(after_reverse_distances)}}
        evidence.update(target_id=cert["target_id"],native_budget=budget,native_tail=tail,
                        material_validation=material_cache['metadata'],material_split_model=split_model,
                        supplemental_material_pixels=int(supplement.sum()),
                        measurements=measurements,source_segments=len(source_segments),
                        native_size=[w,h],mean_p95_nonregression_epsilon=1e-6,
                        native_sampling_mapping={"preserve_aspect_ratio":"xMidYMid meet",
                                                 "uniform_scale":float(scale),
                                                 "offset_xy":offset.tolist()},
                        worst_reverse_points=[{"xy":after_points[after_free][j].tolist(),
                                               "distance":float(after_reverse_distances[j])}
                                              for j in np.argsort(-after_reverse_distances)[:12]])
        for direction_name,values in measurements.items():
            a,b=values["after"],values["before"]
            if (a["mean"]>b["mean"]+1e-6 or a["p95"]>b["p95"]+1e-6
                    or a["p95"]>budget+1e-6 or a["max"]>tail+1e-6):
                evidence["reasons"].append("source_boundary_"+direction_name+"_outside_contract")
        if evidence["reasons"]:
            return empty,evidence
        labels,n = _components(bg)
        border = np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
        enclosed = np.ones(n+1,dtype=bool); enclosed[border]=False; enclosed[0]=False
        source_holes = enclosed[labels]
        width = feature_width
        # AA pixels are not a solid thin stroke; a genuine opaque 1px stroke
        # lacking any interior evidence gets no credible allowance in any case.
        solid = fg & credible & (coverage >= .92)
        thin_white = _thin(bg,solid,width)
        # Opposed diagonal paper samples also occur at an ordinary convex
        # corner. Require lack of a 2-D solid neighbourhood, so that corners of
        # broad shapes are not mislabeled as one-pixel strokes. A 1px line or
        # spur has no 3x3 solid support and remains protected.
        broad_ink = _morph(_morph(solid,3,erode=True),3)
        thin_ink = _thin(solid,bg,width) & ~broad_ink
        retained_white = fg & (np.max(np.abs(rgb-paper),axis=2)<=8)
        feature_regions=_source_feature_regions(
            {'source_hole':source_holes,'thin_white':thin_white,'thin_ink':thin_ink},
            mask | _render_mask(new,new_target,w,h),tail)
        protected = source_holes | thin_white | thin_ink | contact | retained_white
        region = reliable & ~protected
        yy,xx = np.nonzero(region)
        pixel_points = np.column_stack((xx+.5,yy+.5))
        # Pixel footprint radius covers sampling AA; geometric edge movement
        # itself remains constrained by the independently measured bound above.
        footprint = math.sqrt(.5)
        distance = _distances(pixel_points,source_index)
        edge_distance = np.minimum(_distances(pixel_points,before_index),_distances(pixel_points,after_index))
        allow = empty.copy()
        eligible = (distance <= tail+footprint) & (edge_distance <= tail+footprint)
        eligible &= _normal_strip_footprint_overlap(pixel_points,source_segments,tail)
        if contact_index is not None:
            eligible &= _distances(pixel_points,contact_index)>footprint
        allow[yy[eligible],xx[eligible]] = True
        # No opaque interior goes missing through a distant band claim. The
        # envelope must touch both original paper and original target coverage.
        ys,xs = np.nonzero(mask)
        pad = int(math.ceil(tail+2))
        roi = [max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),min(w,int(xs.max())+pad+1),min(h,int(ys.max())+pad+1)]
        evidence.update(verified=True,allowance_pixels=int(allow.sum()),
                        allowance_sha256=_sha(allow.tobytes()),roi_xyxy=roi,
                        before_svg_sha256=_sha(before_svg_text),after_svg_sha256=_sha(after_svg_text),
                        source_rgba_sha256=rgba_sha256(source),processed_rgba_sha256=rgba_sha256(processed),
                        source_hole_pixels=int(source_holes.sum()),thin_white_pixels=int(thin_white.sum()),
                        thin_ink_pixels=int(thin_ink.sum()),contact_pixels=int(contact.sum()),
                        source_feature_regions=feature_regions,
                        feature_bank_policy='each_source_feature_support_must_not_increase_source_error',
                        retained_white_object_pixels=int(retained_white.sum()),
                        feature_width_native=width,unmix_rms_maximum=12,unmix_contrast_norm_minimum=50,
                        explicitly_excluded_contact_segments=len(contact_segments),
                        envelope_distance=tail,pixel_footprint_radius=footprint,
                        pixel_scope_method="unit_square_overlap_finite_normal_strips_connected_vertex_disks",
                        uncovered_policy="strict_no_blanket_path_allowance")
        return allow,evidence
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,RuntimeError,ImportError,MemoryError,ET.ParseError) as exc:
        evidence["reasons"].append(str(exc) or type(exc).__name__)
        return empty,evidence


def build_source_boundary_evidence(before_svg_text, after_svg_text, source_rgba,
                                   processed_rgba, geometry):
    """Certify both independent material models and intersect their scope.

    Geometry budgets, source topology and colour nonregression are unchanged.
    The ephemeral cache belongs to this exact transaction only. It contains no
    candidate acceptance flags and is never reused across different SVG hashes.
    """
    cache={}
    mask0,evidence0=_build_source_boundary_evidence_once(before_svg_text,after_svg_text,
        source_rgba,processed_rgba,geometry,0,cache)
    if not evidence0['verified']:
        return mask0,evidence0
    if not evidence0.get('supplemental_material_pixels'):
        evidence0['material_models_required']=1
        return mask0,evidence0
    mask1,evidence1=_build_source_boundary_evidence_once(before_svg_text,after_svg_text,
        source_rgba,processed_rgba,geometry,1,cache)
    evidence=copy.deepcopy(evidence0)
    evidence['material_models_required']=2
    evidence['material_model_measurements']=[
        {'model':index,'verified':item['verified'],'reasons':item['reasons'],
         'measurements':item.get('measurements')} for index,item in enumerate((evidence0,evidence1))]
    if not evidence1['verified']:
        evidence['verified']=False
        evidence['reasons']=['source_boundary_second_material_model_unverified']+evidence1['reasons']
        return np.zeros_like(mask0),evidence
    mask=mask0 & mask1
    regions={repr((r['kind'],r['bbox_xyxy'],r['support_mask_packbits_hex'])):r
             for r in evidence0['source_feature_regions']+evidence1['source_feature_regions']}
    evidence.update(allowance_pixels=int(mask.sum()),allowance_sha256=_sha(mask.tobytes()),
                    material_scope_combination='intersection_of_independently_certified_models',
                    source_feature_regions=list(regions.values()))
    return mask,evidence
