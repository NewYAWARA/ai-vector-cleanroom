"""Source-supported edge proposals for opaque gradients on a white canvas.

This is reconstruction against a raster-derived target, not equivalence to a
tracer's ownership staircase.  A proposal never authorises its own scene commit.
The caller must validate source appearance, alpha topology and neighbouring
objects, then attach ``source_edge_scene_commit`` before using the certificate.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageFilter

SCHEMA = "ai-vector-cleanroom.source-edge-reconstruction/v1"
SOLVER = "source_edge_reconstruction.source_coverage"
DRAWABLES = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon", "text", "image", "use"}


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def rgba_sha256(array):
    """Bind decoded RGBA values and dimensions; this is not a PNG file hash."""
    array = _rgba(array)
    return _sha(str(array.shape).encode("ascii") + array.tobytes())


def _rgba(value):
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            value = np.asarray(image.convert("RGBA"))
    value = np.asarray(value)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 4:
        raise ValueError("source_edge_requires_rgba8")
    if (min(value.shape[:2]) < 8 or max(value.shape[:2]) > 8192
            or value.shape[0]*value.shape[1] > 8*1024*1024):
        raise ValueError("source_edge_unsupported_dimensions")
    return np.ascontiguousarray(value)


def _local(node):
    return node.tag.rsplit("}", 1)[-1]


def _context(svg_text, drawable_id, shape):
    if not isinstance(svg_text, str) or "<!DOCTYPE" in svg_text.upper() or "<!ENTITY" in svg_text.upper():
        raise ValueError("source_edge_unsafe_xml")
    root = ET.fromstring(svg_text)
    for node in root.iter():
        if _local(node) in {"script", "style", "foreignObject", "animate", "animateTransform", "set", "switch", "use", "image", "text"}:
            raise ValueError("source_edge_unsupported_active_or_external_svg")
        for key, value in node.attrib.items():
            if key.lower().startswith("on") or "href" in key.lower() or re.search(r"url\(\s*[^#]", value):
                raise ValueError("source_edge_external_or_active_attribute")
    identities = [node.get("id") for node in root.iter() if node.get("id")]
    if len(identities) != len(set(identities)):
        raise ValueError("source_edge_duplicate_identity")
    nodes = [node for node in root.iter() if node.get("id") == drawable_id]
    if len(nodes) != 1 or _local(nodes[0]) != "path" or not nodes[0].get("data-avc-gradient-object"):
        raise ValueError("source_edge_requires_identified_gradient_path")
    from gradient_source_components import _paint_context
    from source_gradient_primitive import gradient_paint_sha256
    target = nodes[0]
    gradient_id, context = _paint_context(root, target)
    vb = [float(value) for value in root.get("viewBox", "").replace(",", " ").split()]
    h, w = shape
    if root.get('preserveAspectRatio','xMidYMid meet') not in ('xMidYMid','xMidYMid meet'):
        raise ValueError('source_edge_requires_default_meet_canvas')
    if len(vb) != 4 or not np.all(np.isfinite(vb)) or not np.allclose(vb, [0, 0, w, h], atol=1e-8):
        raise ValueError("source_edge_requires_working_pixel_viewbox")
    paint_sha = gradient_paint_sha256(root, gradient_id)
    gradient=next(node for node in root.iter() if node.get("id")==gradient_id)
    from PIL import ImageColor
    stops=list(gradient)
    if not stops or any(_local(stop)!="stop" or stop.get("stop-opacity","1") not in ("1","1.0","100%")
                        or "style" in stop.attrib for stop in stops):
        raise ValueError("source_edge_requires_explicit_opaque_gradient_stops")
    for stop in stops:
        try:
            colour=ImageColor.getcolor(stop.get("stop-color","black"),"RGBA")
        except (ValueError,TypeError):
            raise ValueError("source_edge_requires_explicit_opaque_gradient_stops") from None
        if colour[3]!=255:
            raise ValueError("source_edge_requires_explicit_opaque_gradient_stops")
    from clean_base import _parse_subpaths
    loops = _parse_subpaths(target.get("d", ""))
    if not loops or any(not loop.get("closed") for loop in loops):
        raise ValueError("source_edge_requires_closed_contours")
    if len(loops) > 1 and context.get("fill-rule", "nonzero") != "evenodd":
        raise ValueError("source_edge_compound_requires_evenodd")
    return root, target, gradient_id, context, paint_sha


def _native_rgba(root, width, height):
    """Render the explicit native viewport with SVG's default meet mapping."""
    import resvg_py
    from svg_renderer import svg_with_native_viewport
    svg=svg_with_native_viewport(ET.tostring(root,encoding='unicode'),width,height)
    data=resvg_py.svg_to_bytes(svg_string=svg,
        width=int(width),height=int(height),background=None,skip_system_fonts=True,
        log_information=False,shape_rendering='geometric_precision')
    with Image.open(io.BytesIO(data)) as image:return np.asarray(image.convert('RGBA')).copy()


def _native_mapping(working_dimensions, native_dimensions):
    w,h=map(float,working_dimensions);nw,nh=map(float,native_dimensions)
    if not all(math.isfinite(value) and value>0 for value in (w,h,nw,nh)):
        raise ValueError('source_edge_invalid_mapping_dimensions')
    scale=min(nw/w,nh/h)
    return scale,np.array([(nw-w*scale)/2,(nh-h*scale)/2])


def _render_mask(root, target, width, height=None):
    """Native alpha for this drawable, retaining its ancestor presentation."""
    isolated = copy.deepcopy(root)
    identity = target.get("id")
    for parent in list(isolated.iter()):
        for child in list(parent):
            if _local(child) in DRAWABLES and child.get("id") != identity:
                parent.remove(child)
    if height is not None:
        return _native_rgba(isolated,width,height)[:,:,3]>=128
    from svg_renderer import render_svg_reference
    with tempfile.TemporaryDirectory(prefix="avc-source-edge-") as tmp:
        source, output = Path(tmp) / "mask.svg", Path(tmp) / "mask.png"
        source.write_text(ET.tostring(isolated, encoding="unicode"), encoding="utf-8")
        render_svg_reference(source, output, width=width, background=None)
        with Image.open(output) as image:
            alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    return alpha >= 128


def _later_paint_alpha(root, target, width, height=None):
    """Native coverage of actual later paint; unsupported reusable drawing fails closed."""
    if any(_local(node) in {'clipPath','mask','pattern','marker','symbol'} for node in root.iter()):
        raise ValueError('source_edge_hidden_underlap_unsupported_reusable_drawing')
    isolated=copy.deepcopy(root)
    drawables=[node for node in isolated.iter() if _local(node) in DRAWABLES]
    index=next(i for i,node in enumerate(drawables) if node.get('id')==target.get('id'))
    keep={id(node) for node in drawables[index+1:]}
    for parent in list(isolated.iter()):
        for child in list(parent):
            if _local(child) in DRAWABLES and id(child) not in keep:parent.remove(child)
    if height is not None:return _native_rgba(isolated,width,height)[:,:,3]
    from svg_renderer import render_svg_reference
    with tempfile.TemporaryDirectory(prefix='avc-source-later-') as temp:
        svg,png=Path(temp)/'later.svg',Path(temp)/'later.png'
        svg.write_bytes(ET.tostring(isolated,encoding='utf-8'))
        render_svg_reference(svg,png,width=width,background=None)
        with Image.open(png) as image:return np.asarray(image.convert('RGBA'))[:,:,3].copy()


def _morph(mask, size, *, erode=False):
    # A one-pixel kernel is the identity. Do not enter Pillow's native
    # rank-filter implementation, which crashes on some Windows builds.
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1 or size % 2 != 1:
        raise ValueError('source_morph_requires_positive_odd_kernel')
    if size == 1:
        return np.asarray(mask, dtype=bool).copy()
    operation = ImageFilter.MinFilter(size) if erode else ImageFilter.MaxFilter(size)
    return np.asarray(Image.fromarray(np.uint8(mask) * 255).filter(operation)) > 0


def _box_mean(array, weight, radius=5):
    k = 2 * radius + 1
    def total(value):
        pad = ((radius + 1, radius), (radius + 1, radius)) + (((0, 0),) if value.ndim == 3 else ())
        summed = np.pad(value, pad).cumsum(0).cumsum(1)
        return summed[k:, k:] - summed[:-k, k:] - summed[k:, :-k] + summed[:-k, :-k]
    count = total(weight.astype(np.float64))
    mean = total(array * weight[..., None]) / np.maximum(count, 1e-9)[..., None]
    return mean, count


def _native_material_colour(rgb, owner, paper, radius=5):
    """Source-only colour hypothesis from spatially repeated strong samples.

    Candidate paint and candidate geometry never determine these samples.
    The independent scene validator must still accept every resulting change.
    """
    contrast=np.linalg.norm(rgb-paper,axis=2)
    values=np.where(owner,contrast,-np.inf)
    h,w=values.shape
    pad=np.pad(values,((0,0),(radius,radius)),constant_values=-np.inf)
    horizontal=np.full_like(values,-np.inf)
    for j in range(2*radius+1):np.maximum(horizontal,pad[:,j:j+w],out=horizontal)
    pad=np.pad(horizontal,((radius,radius),(0,0)),constant_values=-np.inf)
    maximum=np.full_like(values,-np.inf)
    for j in range(2*radius+1):np.maximum(maximum,pad[j:j+h],out=maximum)
    selected=owner&(contrast>=maximum-12*math.sqrt(3))
    inside,count=_box_mean(rgb,selected,radius)
    return inside,count,selected


def _field_loops(field):
    """Marching squares with raster samples located at SVG pixel centres."""
    from trace_engine import _interp, _segment_key, _segments_to_loops
    h, w = field.shape
    f = np.pad(np.asarray(field, dtype=float), 1)
    level = 0.5
    pairs = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
             5: [(0, 3), (1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)],
             9: [(0, 2)], 10: [(0, 1), (3, 2)], 11: [(1, 2)],
             12: [(3, 1)], 13: [(0, 1)], 14: [(3, 0)]}
    cases = ((f[:-1, :-1] >= level).astype(np.uint8)
             | ((f[:-1, 1:] >= level).astype(np.uint8) << 1)
             | ((f[1:, 1:] >= level).astype(np.uint8) << 2)
             | ((f[1:, :-1] >= level).astype(np.uint8) << 3))
    segments = []
    for y, x in zip(*np.nonzero((cases != 0) & (cases != 15))):
        y, x = int(y), int(x)
        v0, v1, v2, v3 = f[y, x], f[y, x + 1], f[y + 1, x + 1], f[y + 1, x]
        # Padding offset -1 plus pixel centre +.5, not the old integer-sample grid.
        p0, p1 = (x - .5, y - .5), (x + .5, y - .5)
        p2, p3 = (x + .5, y + .5), (x - .5, y + .5)
        edges = {0: _interp(level, p0, p1, v0, v1), 1: _interp(level, p1, p2, v1, v2),
                 2: _interp(level, p3, p2, v3, v2), 3: _interp(level, p0, p3, v0, v3)}
        for a, b in pairs[int(cases[y, x])]:
            pa, pb = edges[a], edges[b]
            pa = (max(0., min(float(w), pa[0])), max(0., min(float(h), pa[1])))
            pb = (max(0., min(float(w), pb[0])), max(0., min(float(h), pb[1])))
            if _segment_key(pa) != _segment_key(pb):
                segments.append((pa, pb))
    return _segments_to_loops(segments, 0.0, 0.01)


def _topology(mask):
    """Marching-squares convention: four-connected ink, eight-connected paper."""
    from source_scene_guard import _components
    from stroke_engine import connected_components
    _, components = _components(mask)
    labels,count=connected_components(~mask)
    exterior=set(np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1]))).tolist())
    holes=sum(index not in exterior for index in range(1,count+1))
    return {"components":int(components),"holes":int(holes)}


def _propose_source_ink_hole_fills(field, original, processed, ownership_mask):
    """Complete small missing-ink regions, never merely seal their throat.

    A prior ownership crack may become enclosed when a free edge is repaired.
    Keeping that crack solely to match the tracer is not source reconstruction.
    Both native references must show opaque, non-paper colour throughout the
    proposed fill. This does NOT establish which paint owns the region: the
    complete rendered scene must independently prove colour and topology.
    """
    from gradient_source_components import enclosed_components
    if original.shape[:2]!=field.shape or processed.shape[:2]!=field.shape:
        raise ValueError('source_edge_hole_fill_requires_native_aligned_references')
    # Existing unexplained holes retain their identity. Their source may show
    # a deliberately different ink colour, rather than this gradient's paint.
    existing=np.zeros(field.shape,bool)
    for component in enclosed_components(ownership_mask,maximum_pixels=field.size,maximum_components=128):
        existing|=component
    records=[]
    for component in enclosed_components(field>=.5,maximum_pixels=256,maximum_components=128):
        if np.any(component&existing):continue
        pixels=original[component]
        supported=bool(len(pixels) and np.all(pixels[:,3]==255)
            and np.all(pixels[:,:3].min(axis=1)<235)
            and np.all(processed[component,3]>=224))
        if not supported:continue
        ys,xs=np.nonzero(component)
        field[component]=1.
        records.append({'pixels':int(component.sum()),
            'bbox':[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],
            'source_and_processed_ink':True,'paint_ownership_proven':False,
            'full_scene_colour_and_topology_validation_required':True})
    return records


def _source_field(mask, original, processed, paper, *, underlap_pixels=0.0,
                  source_material_owner=None):
    """Unmix only explained paper edges; ambiguous boundaries retain ownership."""
    from gradient_source_components import enclosed_components
    h, w = mask.shape
    image = np.asarray(Image.fromarray(original).resize((w,h), Image.Resampling.LANCZOS), dtype=float)
    rgb = image[:,:,:3]
    interior = _morph(mask, 5, erode=True)
    material_samples=None
    if source_material_owner is None:
        inside, count = _box_mean(rgb, interior, 5)
    else:
        owner=np.asarray(source_material_owner,dtype=bool)
        if owner.shape!=mask.shape:raise ValueError('source_edge_material_owner_dimensions_mismatch')
        inside,count,material_samples=_native_material_colour(rgb,owner,paper,5)
    direction = inside - paper
    norm = np.sum(direction * direction, axis=2)
    alpha = np.clip(np.sum((rgb-paper)*direction, axis=2) / np.maximum(norm,1e-9), 0, 1)
    explained = paper + alpha[:,:,None] * direction
    residual = np.sqrt(np.mean((rgb-explained)**2,axis=2))
    confidence = (count > 3) & (norm >= 50**2) & (residual <= 12)
    adjusted = mask.copy()
    protected = np.zeros(mask.shape, bool)
    hole_records = []
    for component in enclosed_components(mask, maximum_pixels=64, maximum_components=128):
        ys, xs = np.nonzero(component)
        native_component = np.asarray(Image.fromarray(np.uint8(component)*255).resize(
            (original.shape[1],original.shape[0]),Image.Resampling.NEAREST)) > 0
        pixels = original[native_component]
        source_ink = bool(len(pixels) and np.all(pixels[:,3] == 255)
                          and np.all(np.min(pixels[:,:3],axis=1) < 235))
        reference_ink = bool(np.all(processed[component,3] >= 224))
        explained_ink = bool(np.all(confidence[component]) and np.all(alpha[component] >= .92))
        remove = source_ink and reference_ink and explained_ink
        if remove:
            adjusted[component] = True
        else:
            protected |= _morph(component,5)
        hole_records.append({"bbox":[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)],
                             "pixels":int(component.sum()),"removed":remove,
                             "original_ink":source_ink,"processed_ink":reference_ink,
                             "locally_explained_ink":explained_ink})
    interior = _morph(adjusted,5,erode=True)
    dilated = _morph(adjusted,7)
    band = dilated & ~interior
    outside_paper = (rgb.min(axis=2) >= 235) & ~adjusted
    near_paper = _morph(outside_paper,7)
    use = band & near_paper & confidence & ~protected
    # This fallback is a hypothesis at ownership/colour interfaces, not source
    # coverage evidence. It uses the existing trace smoother's fixed .55 radius;
    # the independent scene validator must prove it against original pixels.
    fallback=np.asarray(Image.fromarray(np.uint8(adjusted)*255).filter(
        ImageFilter.GaussianBlur(.55)),dtype=float)/255.
    field=fallback.copy()
    field[protected]=adjusted[protected].astype(float)
    field[use] = alpha[use]
    # Source pixels can belong to another object immediately outside this path.
    # A positive island without any eroded ownership core is not a new claimed
    # part of this gradient. Restore the old local interface proposal instead.
    from source_scene_guard import _components
    labels,count_labels=_components(field>=.5)
    anchored=set(np.unique(labels[interior]).tolist())
    unanchored=np.isin(labels,[i for i in range(1,count_labels+1) if i not in anchored])
    island_fallback=_morph(unanchored,3) if unanchored.any() else unanchored
    field[island_fallback]=fallback[island_fallback]
    use[island_fallback]=False
    field[protected]=adjusted[protected].astype(float)
    for row in hole_records:
        if row["removed"]:
            x,y,bw,bh = row["bbox"]
            # Filled source-supported holes stay ink rather than being reintroduced
            # by colour estimation at the old missing ownership pixels.
            region = adjusted[y:y+bh,x:x+bw] & ~mask[y:y+bh,x:x+bw]
            field[y:y+bh,x:x+bw][region] = 1.
    underlap = np.zeros(mask.shape,bool)
    if underlap_pixels:
        # This field helper supports one sampling-grid pixel of underlap. Do not expand around any
        # enclosed component (including large holes, which are never fill candidates).
        outside = ~adjusted
        from collections import deque
        reachable = np.zeros(mask.shape,bool)
        queue = deque()
        for y,x in [(0,x) for x in range(w)]+[(h-1,x) for x in range(w)]+[(y,0) for y in range(h)]+[(y,w-1) for y in range(h)]:
            if outside[y,x] and not reachable[y,x]:
                reachable[y,x]=True; queue.append((y,x))
        while queue:
            y,x=queue.popleft()
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                a,b=y+dy,x+dx
                if 0<=a<h and 0<=b<w and outside[a,b] and not reachable[a,b]:
                    reachable[a,b]=True; queue.append((a,b))
        underlap = (_morph(adjusted,3) & ~adjusted & reachable & ~_morph(outside_paper,3)
                    & (rgb.min(axis=2)<225) & (processed[:,:,3]>=224) & ~protected)
        field[underlap] = 1.
        use[underlap]=False
    # Local colour unmixing can wrap around an unrelated dark neighbour or
    # misread its paint as low coverage. Do not invent little paper holes there.
    # Restore only those newly enclosed local regions whose original/native
    # pixels and processed reference both say ink. Real paper remains available
    # to the independent source-scene validator as a reconstructed source hole.
    topology_fallbacks=[]
    for component in enclosed_components(field>=.5,maximum_pixels=256,maximum_components=128):
        # A hole already present in adjusted ownership has its own preserved
        # identity. This operation never fills that pre-existing hole.
        if np.any(component & protected):
            continue
        native=np.asarray(Image.fromarray(np.uint8(component)*255).resize(
            (original.shape[1],original.shape[0]),Image.Resampling.NEAREST))>0
        pixels=original[native]
        ink=bool(len(pixels) and np.all(pixels[:,3]==255)
                 and np.all(pixels[:,:3].min(axis=1)<235) and np.all(processed[component,3]>=224))
        if not ink:
            continue
        region=_morph(component,5)
        field[region]=fallback[region]
        use[region]=False
        field[protected]=adjusted[protected].astype(float)
        topology_fallbacks.append({"pixels":int(component.sum()),"scope":"new_source_field_hole_only",
                                   "reason":"both_sources_show_ink_restore_ambiguous_ownership_locally"})
    source_hole_fills=(_propose_source_ink_hole_fills(field,original,processed,adjusted)
        if original.shape[:2]==field.shape and processed.shape[:2]==field.shape else [])
    if int(use.sum()) < 8:
        raise ValueError("source_edge_insufficient_explained_paper_boundary")
    return field, adjusted, {"band_pixels":int(band.sum()),"source_unmixed_pixels":int(use.sum()),
                            "source_material_policy":('native_same_owner_repeated_strong_source_samples' if material_samples is not None else 'interior_source_mean'),
                            "source_material_sample_count":None if material_samples is None else int(material_samples.sum()),
                            "source_material_unmix_rms_budget":12,
                            "ambiguous_boundary_pixels_proposed_from_ownership":int((band & ~use).sum()),
                            "underlap_pixels":int(underlap.sum()),"hole_decisions":hole_records,
                            "ambiguous_interface_proposal":"ownership_gaussian_sigma_0.55_not_source_proven",
                            "ambiguous_interface_smoothing_sigma_sampling_pixels":.55,
                            "unanchored_source_island_fallback_pixels":int(island_fallback.sum()),
                            "removed_holes":sum(row["removed"] for row in hole_records),
                            "removed_hole_pixels":sum(row["pixels"] for row in hole_records if row["removed"]),
                            "source_ink_hole_fill_proposals":source_hole_fills,
                            "source_ink_hole_fill_scope":"native_source_supported_geometry_proposal_not_paint_ownership_or_scene_acceptance",
                            "local_topology_fallbacks":topology_fallbacks}


def propose_source_edge_reconstruction(svg_text, drawable_id, original_rgba, processed_rgba, *,
                                       ownership_mask=None, error_budget_percent=.25, underlap_pixels=0.0):
    """Return an uncommitted source-derived path proposal or raise ValueError.

    Original and processed references remain on the same native pixel canvas.
    Native contours are mapped to the viewBox through SVG's uniform default
    meet transform, including rounding letterboxes. No reference is stretched.
    Only an opaque, uniform near-white original canvas is supported. No
    caller-supplied alpha-provenance metadata is trusted.
    """
    if not math.isfinite(float(error_budget_percent)) or not 0 < float(error_budget_percent) <= .25:
        raise ValueError("source_edge_invalid_or_relaxed_error_budget")
    if underlap_pixels not in (0,0.,1,1.,2,2.):
        raise ValueError("source_edge_underlap_must_be_zero_one_or_two_working_pixels")
    original, processed_input = _rgba(original_rgba), _rgba(processed_rgba)
    if not isinstance(svg_text,str) or "<!DOCTYPE" in svg_text.upper() or "<!ENTITY" in svg_text.upper():
        raise ValueError("source_edge_unsafe_xml")
    dimensions = [float(value) for value in ET.fromstring(svg_text).get("viewBox", "").replace(",", " ").split()]
    if (len(dimensions) != 4 or not np.all(np.isfinite(dimensions)) or dimensions[:2] != [0.,0.]
            or any(value != int(value) or not 8 <= value <= 8192 for value in dimensions[2:])):
        raise ValueError("source_edge_requires_integer_working_viewbox")
    working_w,working_h = map(int,dimensions[2:])
    processed=processed_input
    if original.shape!=processed.shape:
        raise ValueError('source_edge_requires_native_aligned_reference_dimensions')
    native_h,native_w=original.shape[:2]
    sampling_scale,sampling_offset=_native_mapping((working_w,working_h),(native_w,native_h))
    if np.any(original[:,:,3] != 255):
        raise ValueError("source_edge_requires_opaque_original")
    border = np.concatenate((original[0,:,:3],original[-1,:,:3],original[:,0,:3],original[:,-1,:3])).astype(float)
    paper = np.median(border,axis=0)
    if paper.min()<245 or np.max(np.abs(border-paper))>8:
        raise ValueError("source_edge_requires_plain_white_canvas")
    root,target,gradient_id,context,paint_sha = _context(svg_text,drawable_id,(working_h,working_w))
    mask = _render_mask(root,target,native_w,native_h) if ownership_mask is None else np.asarray(ownership_mask)
    if mask.dtype != bool or mask.shape != processed.shape[:2] or int(mask.sum()) < 16:
        raise ValueError("source_edge_invalid_ownership_mask")
    ys,xs = np.nonzero(mask)
    if xs.min()<4 or ys.min()<4 or xs.max()>=mask.shape[1]-4 or ys.max()>=mask.shape[0]-4:
        raise ValueError("source_edge_boundary_touches_canvas")
    # Source material is inferred only inside this original drawable's native
    # ownership, excluding actual other paint and ambiguous source/reference.
    from source_scene_guard import _source_masks
    foreground,_,support,_=_source_masks(original,processed)
    if support.get('confident') is not True:raise ValueError('source_edge_unverified_source_material')
    others=copy.deepcopy(root)
    for parent in list(others.iter()):
        for child in list(parent):
            if child.get('id')==drawable_id:parent.remove(child)
    contact=_morph(_native_rgba(others,native_w,native_h)[:,:,3]>=32,5)
    material_owner=mask&foreground&~contact&(processed[:,:,3]>=240)
    field,adjusted,field_evidence = _source_field(mask,original,processed,paper,
        underlap_pixels=min(underlap_pixels,1),source_material_owner=material_owner)
    field_evidence['coordinate_grid']='native_source_pixels'
    if underlap_pixels==2:
        # Extra margin is restricted to existing later paint and known native
        # source ink. It cannot expand into true enclosed holes or paper.
        later=_later_paint_alpha(root,target,native_w,native_h)
        rgb=original[:,:,:3]
        from source_scene_guard import _components
        labels,_=_components(~adjusted)
        exterior=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
        outside=np.isin(labels,exterior)&~adjusted
        extra=(_morph(adjusted,5)&outside&(later>0)
               &~_morph((rgb.min(axis=2)>=235)&~adjusted,3)
               &(rgb.min(axis=2)<225)&(processed[:,:,3]>=224))
        field[extra]=1.
        field_evidence['hidden_contact_additional_field_pixels']=int(extra.sum())
        field_evidence['hidden_contact_maximum_native_pixels']=2
        field_evidence['later_native_paint_alpha_sha256']=_sha(str(later.shape).encode()+later.tobytes())
        field_evidence['hidden_underlap_scope']='external_source_ink_where_actual_later_paint_has_native_alpha'
    before_topology, target_topology = _topology(adjusted), _topology(field>=.5)
    # A source reconstruction may differ from faulty ownership. This is only a
    # proposal: source-supported spatial topology validation is mandatory before
    # commit and cannot be replaced by matching old mask component/hole counts.
    field_evidence["adjusted_ownership_topology"]=before_topology
    field_evidence["source_target_topology"]=target_topology
    # Crop extraction without altering the pixel-centre convention or viewBox.
    x0,y0=max(0,int(xs.min())-5),max(0,int(ys.min())-5)
    x1,y1=min(mask.shape[1],int(xs.max())+6),min(mask.shape[0],int(ys.max())+6)
    contours=[(np.asarray(loop,float)+[x0,y0]-sampling_offset)/sampling_scale
              for loop in _field_loops(field[y0:y1,x0:x1])]
    expected_loops = target_topology["components"]+target_topology["holes"]
    if not contours:
        raise ValueError("source_edge_empty_source_target")
    field_evidence["raster_expected_loop_count"]=expected_loops
    field_evidence["extracted_target_loop_count"]=len(contours)
    field_evidence["topology_scope"]="extracted_candidate_contours_not_author_intent_or_old_mask_equivalence"
    if len(contours)>1 and context.get("fill-rule","nonzero")!="evenodd":
        raise ValueError("source_edge_compound_requires_evenodd")
    from geometry_error_optimizer import optimize_compound_contours, _error_contract_evidence
    fit = optimize_compound_contours(contours,error_budget_percent=float(error_budget_percent),
                                    include_independently_valid_loop_candidates=True)
    if not _error_contract_evidence(fit["actual_error"],float(error_budget_percent))["within_budget"]:
        raise ValueError("source_edge_fit_outside_source_target_budget")
    path=fit["path"]
    loops=[np.asarray(loop,float).tolist() for loop in contours]
    certificate={"schema":SCHEMA,"status":"proposal_only","target_id":drawable_id,
                 "gradient_object_id":target.get("data-avc-gradient-object"),"gradient_id":gradient_id,
                 "source_rgba_sha256":rgba_sha256(original),"processed_rgba_sha256":rgba_sha256(processed_input),
                 "aligned_processed_rgba_sha256":rgba_sha256(processed),
                 "processed_dimensions":[processed_input.shape[1],processed_input.shape[0]],
                 "source_dimensions":[original.shape[1],original.shape[0]],
                 "working_dimensions":[working_w,working_h],"sha_scope":"decoded_RGBA_shape_and_bytes",
                 "native_sampling_dimensions":[native_w,native_h],
                 "native_sampling_mapping":{"preserve_aspect_ratio":"xMidYMid meet",
                     "viewbox_to_native_scale":float(sampling_scale),
                     "native_letterbox_offset":sampling_offset.tolist(),
                     "no_reference_resampling":True},
                 "before_svg_sha256":_sha(svg_text),"before_path_sha256":_sha(target.get("d","")),
                 "after_path_sha256":_sha(path),"paint_sha256":paint_sha,"presentation_context_sha256":_json_sha(context),
                 "ownership_mask_sha256":_sha(mask.tobytes()),"source_target_loops_sha256":_json_sha(loops),
                 "source_target_loops":loops,
                 "measurement_sha256":_json_sha(fit["actual_error"]),
                 "source_target_point_count":sum(len(loop) for loop in loops),"source_target_loop_count":len(loops),
                 "pixel_coordinates":"native_pixel_centres_x_plus_0.5_y_plus_0.5_then_inverse_SVG_meet",
                 "error_reference":"source_unmixed_paper_edges_with_scene_unverified_ownership_interface_proposals",
                 "not_binary_ownership_equivalence":True,"full_scene_validation_required":True,
                 "source_target_topology":target_topology,"field":field_evidence,
                 "underlap_policy":("none" if not underlap_pixels else
                     "one_native_pixel_external_nonpaper_source_ink_only" if underlap_pixels==1 else
                     "two_native_pixel_external_nonpaper_source_ink_with_native_later_paint_coverage"),
                 "error_budget_percent":float(error_budget_percent),
                 "error_contract":_error_contract_evidence(fit["actual_error"],float(error_budget_percent))}
    geometry=copy.deepcopy(fit)
    geometry.update({"solver":SOLVER,"path":path,"anchor_count":fit["anchors_after"],
                     "segment_count":fit["segment_count_after"],
                     "topology":{"preserved":True,"topology_preserved":True,"reference":"source_target_not_prior_svg","loop_count":len(loops)},
                     "error_budget":{"max_error_percent":float(error_budget_percent),"within_budget":True,
                                     "requested_max_percent":float(error_budget_percent),"passed":True,
                                     "actual_p95_error_percent":fit["actual_p95_error_percent"],
                                     "actual_max_error_percent":fit["actual_max_error_percent"],
                                     "reference":"source_target_contours"},
                     "selection_evidence":copy.deepcopy(fit),"source_edge_reconstruction":certificate})
    for key in list(target.attrib):
        if key.startswith("data-avc-") and key not in {"data-avc-gradient-object","data-avc-role","data-avc-paint-role"}:
            del target.attrib[key]
    target.set("d",path)
    target.set("data-avc-source-edge",SCHEMA)
    target.set("data-avc-designer-anchors",str(fit["designer_anchor_count"]))
    target.set("data-avc-error-budget-percent",str(float(error_budget_percent)))
    target.set("data-avc-p95-error-percent",str(fit["actual_p95_error_percent"]))
    target.set("data-avc-max-error-percent",str(fit["actual_max_error_percent"]))
    return {"path":path,"geometry":geometry,"certificate":certificate,
            "source_target_loops":loops,"candidate_svg_text":ET.tostring(root,encoding="unicode")}


def source_edge_certificate_valid(geometry, *, require_scene_commit=False):
    """Validate the scoped proof; a geometry-only proposal cannot certify a scene."""
    try:
        cert=geometry["source_edge_reconstruction"]
        if geometry.get("solver")!=SOLVER or cert.get("schema")!=SCHEMA:
            return False
        if cert.get("not_binary_ownership_equivalence") is not True or cert.get("full_scene_validation_required") is not True:
            return False
        if (cert.get("after_path_sha256")!=_sha(geometry["path"])
                or geometry.get("fit",{}).get("path") != geometry["path"]):
            return False
        for key in ("source_rgba_sha256","processed_rgba_sha256","before_svg_sha256","before_path_sha256",
                    "after_path_sha256","paint_sha256","presentation_context_sha256","source_target_loops_sha256","ownership_mask_sha256","measurement_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}",str(cert.get(key,""))):
                return False
        budget=float(cert["error_budget_percent"])
        if not math.isfinite(budget) or not 0<budget<=.25:
            return False
        loops=cert["source_target_loops"]
        if (not loops or len(loops)!=cert["source_target_loop_count"]
                or sum(len(loop) for loop in loops)!=cert["source_target_point_count"]
                or _json_sha(loops)!=cert["source_target_loops_sha256"]
                or _json_sha(geometry["actual_error"])!=cert["measurement_sha256"]):
            return False
        if any(np.asarray(loop).ndim!=2 or np.asarray(loop).shape[1]!=2
               or len(loop)<3 or not np.all(np.isfinite(np.asarray(loop,dtype=float))) for loop in loops):
            return False
        from geometry_error_optimizer import _error_contract_evidence
        measured=_error_contract_evidence(geometry["actual_error"],budget)
        if not measured["within_budget"] or measured!=cert.get("error_contract"):
            return False
        if not cert.get("target_id") or not cert.get("gradient_object_id"):
            return False
        if 'native_sampling_mapping' in cert:
            if cert.get('native_sampling_dimensions')!=cert.get('source_dimensions'):
                return False
            scale,offset=_native_mapping(cert['working_dimensions'],cert['source_dimensions'])
            mapping=cert['native_sampling_mapping']
            if (mapping.get('preserve_aspect_ratio')!='xMidYMid meet'
                    or mapping.get('no_reference_resampling') is not True
                    or mapping.get('viewbox_to_native_scale')!=scale
                    or mapping.get('native_letterbox_offset')!=offset.tolist()
                    or cert.get('processed_dimensions')!=cert.get('source_dimensions')
                    or cert.get('aligned_processed_rgba_sha256')!=cert.get('processed_rgba_sha256')):
                return False
        if "contact_preservation" in cert:
            from source_edge_contacts import contact_certificate_valid
            if not contact_certificate_valid(geometry):
                return False
        if require_scene_commit:
            commit=geometry.get("source_edge_scene_commit",{})
            if commit.get("status")!="committed" or commit.get("accepted") is not True:
                return False
            if any(commit.get(key)!=cert.get(key) for key in ("before_svg_sha256","after_path_sha256","source_rgba_sha256")):
                return False
            if commit.get("source_guard",{}).get("accepted") is not True:
                return False
            alpha=commit.get("alpha_guard",{})
            if alpha.get("accepted") is not True or alpha.get("scope")!="source_supported_alpha_not_prior_svg_topology":
                return False
        return True
    except (KeyError,TypeError,ValueError,OverflowError):
        return False


def final_source_edge_matches(root, geometry, original_rgba, processed_rgba=None):
    """Bind a committed certificate to final path, paint, context and source pixels."""
    if not source_edge_certificate_valid(geometry,require_scene_commit=True):
        return False
    try:
        cert=geometry["source_edge_reconstruction"]
        original=_rgba(original_rgba)
        if rgba_sha256(original)!=cert["source_rgba_sha256"]:
            return False
        if processed_rgba is not None and rgba_sha256(processed_rgba)!=cert["processed_rgba_sha256"]:
            return False
        text=ET.tostring(root,encoding="unicode") if hasattr(root,"tag") else str(root)
        _,target,gradient_id,context,paint_sha=_context(text,cert["target_id"],tuple(reversed(cert["working_dimensions"])))
        return (target.get("data-avc-gradient-object")==cert["gradient_object_id"]
                and gradient_id==cert["gradient_id"] and _sha(target.get("d",""))==cert["after_path_sha256"]
                and paint_sha==cert["paint_sha256"] and _json_sha(context)==cert["presentation_context_sha256"])
    except (KeyError,TypeError,ValueError,ET.ParseError):
        return False
