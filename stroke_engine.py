# -*- coding: utf-8 -*-
"""
Monoline stroke reconstruction engine.

Detects near-uniform-width line work (heartbeat lines, field lines, frame
outlines, simple line art) in a color-labeled image and rebuilds each as a
real SVG stroke: a fitted center-line path with `fill="none"`,
`stroke-width`, source-validated straight caps and round curve joins — instead of a high-node filled
outline pair.

Conservative by design: a component is converted only when it passes strict
uniform-width and simple-topology tests; everything else stays with the
fill tracer. Pure numpy, no OpenCV dependency.
"""

from __future__ import annotations

import math
import io
import re
from dataclasses import dataclass, field, replace

import numpy as np

MAX_HALF_WIDTH = 13          # baseline for small images
MAX_SCALED_HALF_WIDTH = 32   # bounded high-resolution stroke search
MIN_STROKE_LEN = 10.0        # px; shorter skeletons are blobs, not strokes
JUNCTION_ARM_STRAIGHTNESS_MIN = 0.90
MULTICOLOR_CURVE_STRAIGHTNESS_MIN = 0.95
MULTICOLOR_SPLIT_DISTANCE = 80.0
MAX_MULTICOLOR_STRAIGHT_RUNS = 8
SHORT_CURVE_MIN_LENGTH_WIDTH_RATIO = 8.0
SHORT_CURVE_STRAIGHTNESS_MIN = 0.97


@dataclass
class Stroke:
    color: tuple                 # (r, g, b)
    width: float                 # stroke width in px
    d: str                       # SVG path data of the center line
    closed: bool
    length: float
    n_nodes: int
    pixels: int = 0
    opacity: float = 1.0
    primitive: str = ""           # "circle" / "rect" / empty path
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    x: float = 0.0
    y: float = 0.0
    shape_width: float = 0.0
    height: float = 0.0
    sample_points: list = field(default_factory=list, repr=False)
    source_fit: dict = field(default_factory=dict, repr=False)
    linecap: str = "round"       # straight paths require native-source cap proof


# ---------- connected components (run-based union-find, 4-connectivity) ----

def connected_components(mask):
    """Label 4-connected components. Returns (labels int32, count)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = []

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
        return min(ra, rb)

    prev_runs = []
    for y in range(h):
        row = mask[y]
        idx = np.flatnonzero(row)
        if idx.size == 0:
            prev_runs = []
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([idx[0]], idx[splits + 1]))
        ends = np.concatenate((idx[splits], [idx[-1]]))
        runs = []
        for s, e in zip(starts, ends):
            lab = -1
            for ps, pe, pl in prev_runs:
                if ps <= e + 1 and pe >= s - 1:     # 8-connectivity
                    lab = union(lab, pl) if lab != -1 else find(pl)
            if lab == -1:
                lab = len(parent)
                parent.append(lab)
            labels[y, s:e + 1] = lab + 1
            runs.append((s, e, lab))
        prev_runs = runs

    if not parent:
        return labels, 0
    # flatten unions and renumber densely
    roots = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    uniq = np.unique(roots)
    remap = np.zeros(len(parent) + 1, dtype=np.int32)
    remap[1:] = np.searchsorted(uniq, roots) + 1
    return remap[labels], len(uniq)


# ---------- distance transform (iterative erosion, capped) ----------

def _erode4(m):
    out = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    out[0, :] = False
    out[-1, :] = False
    out[:, 0] = False
    out[:, -1] = False
    return out


def dist_transform_capped(mask, cap=MAX_HALF_WIDTH + 2):
    """Approximate 4-connected distance to background, capped at `cap`.
    Border pixels count as adjacent to background (padded view)."""
    pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    d = np.zeros(pad.shape, dtype=np.float32)
    cur = pad
    for i in range(cap):
        d[cur] = i + 1
        cur = _erode4(cur)
        if not cur.any():
            break
    else:
        d[cur] = cap + 1
    return d[1:-1, 1:-1]


# ---------- thinning (Zhang-Suen, vectorized) ----------

def _neighbors(img):
    p = np.pad(img, 1)
    P2 = p[:-2, 1:-1]
    P3 = p[:-2, 2:]
    P4 = p[1:-1, 2:]
    P5 = p[2:, 2:]
    P6 = p[2:, 1:-1]
    P7 = p[2:, :-2]
    P8 = p[1:-1, :-2]
    P9 = p[:-2, :-2]
    return P2, P3, P4, P5, P6, P7, P8, P9


def thin(mask, max_iter=200):
    """Zhang-Suen thinning to a 1-px skeleton."""
    img = mask.astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            P2, P3, P4, P5, P6, P7, P8, P9 = _neighbors(img)
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros_like(img)
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            rem = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if rem.any():
                img[rem] = 0
                changed = True
        if not changed:
            break
    return img.astype(bool)


# ---------- skeleton graph ----------

_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _remove_staircase(sk):
    """Remove redundant staircase pixels left by Zhang-Suen.

    A pixel whose neighbors are few and mutually 8-connected without it
    (e.g. the corner of an L-turn whose two neighbors already touch
    diagonally) adds phantom junctions; drop it."""
    sk = sk.copy()
    for _ in range(5):
        removed = False
        for y, x in np.argwhere(sk):
            nbrs = [(y + dy, x + dx) for dy, dx in _OFFS
                    if 0 <= y + dy < sk.shape[0] and 0 <= x + dx < sk.shape[1]
                    and sk[y + dy, x + dx]]
            k = len(nbrs)
            if k < 2 or k > 3:
                continue
            # connected among themselves without the center?
            if k == 2:
                ok = max(abs(nbrs[0][0] - nbrs[1][0]),
                         abs(nbrs[0][1] - nbrs[1][1])) <= 1
            else:
                pairs = [(a, b) for i, a in enumerate(nbrs)
                         for b in nbrs[i + 1:]]
                adj = sum(1 for a, b in pairs
                          if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1)
                ok = adj >= 2      # chain of three
            if ok:
                sk[y, x] = False
                removed = True
        if not removed:
            break
    return sk


def _degree_map(sk):
    p = np.pad(sk.astype(np.uint8), 1)
    deg = np.zeros_like(sk, dtype=np.uint8)
    for dy, dx in _OFFS:
        deg += p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
    return np.where(sk, deg, 0)


def _walk(sk, deg, start, first):
    """Walk from `start` through `first` until endpoint/junction/loop."""
    path = [start, first]
    visited = {start, first}
    cur = first
    while True:
        if deg[cur] != 2 and cur != start:
            return path
        nxt = None
        for dy, dx in _OFFS:
            q = (cur[0] + dy, cur[1] + dx)
            if not (0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1]):
                continue
            if not sk[q]:
                continue
            if q == path[0] and len(path) > 3:
                path.append(q)         # closed the loop
                return path
            if q in visited:
                continue
            nxt = q
            break
        if nxt is None:
            return path
        path.append(nxt)
        visited.add(nxt)
        cur = nxt


def skeleton_to_polyline(sk):
    """Extract a single open polyline or closed loop from a skeleton.

    Returns (points list [(x, y)], closed bool) or None when the topology
    is not a simple line (junctions, multiple branches)."""
    deg = _degree_map(sk)
    n_pix = int(sk.sum())
    if n_pix < 3:
        return None
    junctions = int(((deg >= 3) & sk).sum())
    endpoints = np.argwhere((deg == 1) & sk)

    if junctions == 0 and len(endpoints) == 2:
        start = tuple(endpoints[0])
        for dy, dx in _OFFS:
            q = (start[0] + dy, start[1] + dx)
            if 0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1] and sk[q]:
                path = _walk(sk, deg, start, q)
                break
        else:
            return None
        if len(path) < 0.85 * n_pix:      # didn't cover the skeleton: odd shape
            return None
        return [(float(x), float(y)) for y, x in path], False

    if junctions == 0 and len(endpoints) == 0:
        ys, xs = np.nonzero(sk)
        start = (int(ys[0]), int(xs[0]))
        first = None
        for dy, dx in _OFFS:
            q = (start[0] + dy, start[1] + dx)
            if 0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1] and sk[q]:
                first = q
                break
        if first is None:
            return None
        path = _walk(sk, deg, start, first)
        if path[-1] != path[0] or len(path) < 0.8 * n_pix:
            return None
        return [(float(x), float(y)) for y, x in path[:-1]], True

    return None


def skeleton_to_junction_edges(sk):
    """Split a simple T/Y/X skeleton into endpoint-to-junction edges.

    The junction pixel cluster is collapsed to one centroid so every arm
    meets at exactly the same SVG coordinate.  Complex graphs are rejected
    and left to the fill tracer.
    """
    deg = _degree_map(sk)
    endpoints = [tuple(p) for p in np.argwhere((deg == 1) & sk)]
    jmask = (deg >= 3) & sk
    if len(endpoints) < 3 or not jmask.any():
        return None
    jl, jn = connected_components(jmask)
    if jn != 1:
        return None
    jp = np.argwhere(jmask)
    jy, jx = float(jp[:, 0].mean()), float(jp[:, 1].mean())
    edges = []
    covered = set()
    for ep in endpoints:
        path = [ep]
        prev = None
        cur = ep
        seen = {ep}
        for _ in range(int(sk.sum()) + 2):
            if jmask[cur]:
                break
            nxts = []
            for dy, dx in _OFFS:
                q = (cur[0] + dy, cur[1] + dx)
                if q == prev or q in seen:
                    continue
                if 0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1] and sk[q]:
                    nxts.append(q)
            if not nxts:
                break
            # Prefer the continuation with the smallest degree; a junction
            # cluster may expose several equivalent neighboring pixels.
            nxt = min(nxts, key=lambda q: _degree_map_at(sk, q))
            path.append(nxt)
            seen.add(nxt)
            covered.add(nxt)
            prev, cur = cur, nxt
            if jmask[cur]:
                break
        if not jmask[cur] or len(path) < 3:
            return None
        pts = [(float(x), float(y)) for y, x in path[:-1]]
        pts.append((jx, jy))
        edges.append(pts)
    if len(covered) < 0.65 * (int(sk.sum()) - int(jmask.sum())):
        return None
    return edges


def prune_spurs(sk, max_len):
    """Iteratively remove endpoint branches shorter than max_len."""
    sk = sk.copy()
    for _ in range(4):
        deg = _degree_map(sk)
        endpoints = np.argwhere((deg == 1) & sk)
        if len(endpoints) == 0:
            break
        removed_any = False
        for ep in endpoints:
            path = [tuple(ep)]
            cur = tuple(ep)
            prev = None
            ok = False
            for _step in range(int(max_len) + 1):
                nxt = None
                for dy, dx in _OFFS:
                    q = (cur[0] + dy, cur[1] + dx)
                    if q == prev:
                        continue
                    if 0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1] and sk[q]:
                        nxt = q
                        break
                if nxt is None:
                    break
                if _degree_map_at(sk, nxt) >= 3:
                    ok = True          # reached a junction: this is a spur
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            if ok and len(path) <= max_len:
                for p in path:
                    sk[p] = False
                removed_any = True
        if not removed_any:
            break
    return sk


def _degree_map_at(sk, p):
    c = 0
    for dy, dx in _OFFS:
        q = (p[0] + dy, p[1] + dx)
        if 0 <= q[0] < sk.shape[0] and 0 <= q[1] < sk.shape[1] and sk[q]:
            c += 1
    return c


# ---------- polyline fitting ----------

def _rdp(pts, eps):
    if len(pts) <= 2:
        return list(pts)
    ax, ay = pts[0]
    bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    norm = math.hypot(dx, dy)
    best, bi = -1.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        dist = abs(dy * px - dx * py + bx * ay - by * ax) / norm if norm else \
            math.hypot(px - ax, py - ay)
        if dist > best:
            best, bi = dist, i
    if best > eps:
        left = _rdp(pts[:bi + 1], eps)
        right = _rdp(pts[bi:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def _f(v):
    v = float(v)
    if abs(v) < 1e-6:
        v = 0.0
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fit_path_d(pts, closed, width):
    """Fit the ordered center-line points into a compact SVG path.

    Straight runs become L segments; smooth runs become Catmull-Rom cubics;
    sharp direction changes stay as corners."""
    eps = max(1.0, 0.22 * width)
    keep = _rdp(pts, eps)
    if closed and len(keep) > 2 and keep[0] == keep[-1]:
        keep = keep[:-1]

    # whole line straight?
    if not closed and len(keep) == 2:
        return (f"M{_f(keep[0][0])} {_f(keep[0][1])} "
                f"L{_f(keep[1][0])} {_f(keep[1][1])}"), 2

    # corner detection on simplified points
    n = len(keep)
    corner = [False] * n
    rng = range(n) if closed else range(1, n - 1)
    for i in rng:
        p0 = keep[(i - 1) % n]
        p1 = keep[i]
        p2 = keep[(i + 1) % n]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1 = math.hypot(*v1)
        l2 = math.hypot(*v2)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
        # direction change > 40 deg is a corner: straight = cos 1,
        # right angle = cos 0 — both a 90° elbow and a hairpin must be kept
        # sharp, or Catmull-Rom overshoots them into arcs (review P0-1)
        if cosang < math.cos(math.radians(40)):
            corner[i] = True
    if not closed:
        corner[0] = corner[-1] = True

    c = 1.0 / 6.0
    parts = [f"M{_f(keep[0][0])} {_f(keep[0][1])}"]
    idx_range = range(n) if closed else range(n - 1)
    for i in idx_range:
        p1 = keep[i]
        p2 = keep[(i + 1) % n]
        p0 = keep[(i - 1) % n] if (closed or i > 0) else p1
        p3 = keep[(i + 2) % n] if (closed or i + 2 < n) else p2
        if corner[i]:
            p0 = p1
        if corner[(i + 1) % n]:
            p3 = p2
        if corner[i] and corner[(i + 1) % n]:
            parts.append(f"L{_f(p2[0])} {_f(p2[1])}")
            continue
        c1 = (p1[0] + (p2[0] - p0[0]) * c, p1[1] + (p2[1] - p0[1]) * c)
        c2 = (p2[0] - (p3[0] - p1[0]) * c, p2[1] - (p3[1] - p1[1]) * c)
        parts.append(f"C{_f(c1[0])} {_f(c1[1])} {_f(c2[0])} {_f(c2[1])} "
                     f"{_f(p2[0])} {_f(p2[1])}")
    if closed:
        parts.append("Z")
    return " ".join(parts), len(keep)


def _skeleton_has_real_junction(sk):
    """Return True for a branched line graph, not a closed-loop artifact."""
    ys, xs = np.nonzero(sk)
    if len(xs) < 4:
        return False
    endpoints = 0
    branches = 0
    h, w = sk.shape
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 1), min(h, y + 2)
        x0, x1 = max(0, x - 1), min(w, x + 2)
        deg = int(sk[y0:y1, x0:x1].sum()) - 1
        if deg == 1:
            endpoints += 1
        elif deg >= 3:
            branches += 1
    # A real T/Y/X graph has at least three terminal arms.  Closed rings can
    # contain thinning artifacts with degree 3 but have no such endpoints.
    return branches > 0 and endpoints >= 3


def _fit_closed_primitive(points, width, length):
    """Fit a clean circle or axis-aligned rectangle to a closed centerline.

    Returning a native primitive avoids the seam bump and excess control
    points produced by a periodic Catmull-Rom fit.
    """
    if len(points) < 8:
        return None
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]

    # Algebraic least-squares circle: x²+y² = 2cx*x + 2cy*y + c.
    try:
        mat = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
        cx, cy, c0 = np.linalg.lstsq(mat, x * x + y * y, rcond=None)[0]
        r2 = c0 + cx * cx + cy * cy
        if r2 > 0:
            radius = math.sqrt(float(r2))
            resid = np.abs(np.hypot(x - cx, y - cy) - radius)
            coverage = length / max(1e-6, 2.0 * math.pi * radius)
            tol = max(1.1, 0.12 * width)
            if (radius >= max(3.0, 1.5 * width)
                    and 0.82 <= coverage <= 1.18
                    and float(np.quantile(resid, 0.90)) <= tol):
                return {"primitive": "circle", "cx": float(cx),
                        "cy": float(cy), "radius": radius, "nodes": 1}
    except Exception:
        pass

    # Axis-aligned frames are common in logos.  Fit to the four bbox edges
    # and require all sides to be represented before emitting <rect>.
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    rw, rh = x1 - x0, y1 - y0
    if rw >= 2.5 * width and rh >= 2.5 * width:
        edge_dist = np.minimum.reduce([np.abs(x - x0), np.abs(x - x1),
                                       np.abs(y - y0), np.abs(y - y1)])
        tol = max(1.1, 0.22 * width)
        side_tol = max(2.0, 0.6 * width)
        sides = [np.any(np.abs(x - x0) <= side_tol),
                 np.any(np.abs(x - x1) <= side_tol),
                 np.any(np.abs(y - y0) <= side_tol),
                 np.any(np.abs(y - y1) <= side_tol)]
        coverage = length / max(1e-6, 2.0 * (rw + rh))
        if (all(sides) and 0.78 <= coverage <= 1.22
                and float(np.quantile(edge_dist, 0.90)) <= tol):
            return {"primitive": "rect", "x": x0, "y": y0,
                    "shape_width": rw, "height": rh, "nodes": 4}
    return None


def _polyline_straightness(points, closed=False):
    """Chord/polyline ratio used to distinguish a line from a glyph/arc."""
    if closed or len(points) < 2:
        return 0.0
    length = sum(math.hypot(points[i + 1][0] - points[i][0],
                            points[i + 1][1] - points[i][1])
                 for i in range(len(points) - 1))
    if length <= 1e-9:
        return 0.0
    chord = math.hypot(points[-1][0] - points[0][0],
                       points[-1][1] - points[0][1])
    return chord / length


def _dilate_one(mask):
    """One-pixel guard dilation, used only to prevent Tier-B re-extraction."""
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


# ---------- main entry ----------

def _group_eligible_component_pixels(labels, n, min_area, max_area):
    """Bucket eligible dense-label pixels once, preserving argwhere order.

    Returns ``(areas, eligible, grouped_flat, starts, ends)``.  For component
    label ``li``, its row-major flat pixel indices are
    ``grouped_flat[starts[li - 1]:ends[li - 1]]``.  Keeping this small helper
    separate makes the performance-critical equivalence regression-testable.
    """
    flat_labels = labels.ravel()
    areas = np.bincount(flat_labels, minlength=n + 1)
    eligible = ((areas >= min_area) & (areas <= max_area))
    eligible[0] = False
    eligible_flat = np.flatnonzero(eligible[flat_labels])
    if len(eligible_flat):
        eligible_labels = flat_labels[eligible_flat]
        order = np.argsort(eligible_labels, kind="stable")
        grouped_flat = eligible_flat[order]
    else:
        grouped_flat = np.empty(0, dtype=np.intp)
    grouped_counts = np.where(eligible, areas, 0)
    ends = np.cumsum(grouped_counts[1:], dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    return areas, eligible, grouped_flat, starts, ends


def _render_native_strokes(strokes, native_roi, working_shape, native_shape):
    """Render exact four-decimal emitted stroke parameters into a native ROI."""
    import resvg_py
    from PIL import Image
    height,width=working_shape
    nh,nw=native_shape
    scale=min(nw/float(width),nh/float(height))
    ox,oy=(nw-width*scale)/2,(nh-height*scale)/2
    x0,y0,x1,y1=map(int,native_roi)
    w,h=x1-x0,y1-y0
    if w*h>262144 or min(w,h)<1:
        raise ValueError('native_stroke_group_render_budget_exceeded')
    paths=[]
    for stroke in strokes:
        if stroke.primitive:
            raise ValueError('native_stroke_group_requires_paths')
        color='#{:02x}{:02x}{:02x}'.format(*stroke.color)
        paths.append(f'<path d="{stroke.d}" fill="none" stroke="{color}" '
                     f'stroke-width="{stroke.width:.4f}" stroke-opacity="{stroke.opacity:.4f}" '
                     f'stroke-linecap="{stroke.linecap}" stroke-linejoin="round"/>')
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="{(x0-ox)/scale:.9f} {(y0-oy)/scale:.9f} {w/scale:.9f} {h/scale:.9f}">'
         +''.join(paths)+'</svg>')
    data=resvg_py.svg_to_bytes(svg_string=svg,width=w,height=h,skip_system_fonts=True,
        log_information=False,shape_rendering='geometric_precision')
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert('RGBA'))


def _refine_transparent_straight(stroke, den, palette, bg, component, labels,
                                 component_id, alpha, source_rgba):
    """Use alpha coverage for a single-colour line on transparent surroundings.

    RGB hidden in transparent pixels is not paper and cannot determine line
    geometry. Normalised alpha supplies that geometry, then the original RGBA
    and exact emitted opacity are checked independently.
    """
    if source_rgba is None or stroke.n_nodes!=2 or stroke.primitive or stroke.closed:
        return None
    original=np.asarray(source_rgba)
    nh,nw=original.shape[:2];h,w=den.shape[:2]
    scale=min(nw/float(w),nh/float(h));offset=np.array([(nw-w*scale)/2,(nh-h*scale)/2])
    yy,xx=component[:,0],component[:,1]
    x0=max(0,int(math.floor(xx.min()*scale+offset[0]))-4)
    y0=max(0,int(math.floor(yy.min()*scale+offset[1]))-4)
    x1=min(nw,int(math.ceil((xx.max()+1)*scale+offset[0]))+4)
    y1=min(nh,int(math.ceil((yy.max()+1)*scale+offset[1]))+4)
    if (x1-x0)*(y1-y0)>32768:
        return None
    source=original[y0:y1,x0:x1]
    a=source[:,:,3]
    border=np.r_[a[0],a[-1],a[:,0],a[:,-1]]
    if np.any(border>0) or not np.any(a>=16):
        return None
    peak=int(a.max())
    core=source[:,:,:3][a>=max(16,peak*.95)].astype(float)
    pigment=tuple(map(int,np.rint(np.median(core,axis=0))))
    if np.max(np.abs(core-pigment))>3:
        return None
    # No RGB or alpha mutation of the user's source. This is only a synthetic
    # coverage measurement plane, never exported as the source reference.
    fake=np.empty_like(original)
    coverage=np.clip(original[:,:,3].astype(float)/peak,0,1)
    fake[:,:,:3]=np.rint(255*(1-coverage))[:,:,None]
    fake[:,:,3]=255
    proposal=replace(stroke,color=(0,0,0),opacity=1.,source_fit={})
    fitted=_refine_straight_aa_stroke(proposal,den,np.array([[0,0,0]],dtype=np.uint8),
        np.array([255,255,255]),component,labels,component_id,None,source_rgba=fake)
    if fitted is None:
        return None
    fitted.color=pigment
    fitted.opacity=round(peak/255.,4)
    preview=_render_native_strokes([fitted],[x0,y0,x1,y1],den.shape[:2],original.shape[:2])
    aa=preview[:,:,3].astype(float)
    union=(a>=max(4,peak*.05)) | (aa>=max(4,peak*.05))
    if not union.any():
        return None
    alpha_error=float(np.abs(a.astype(float)-aa)[union].mean())
    pm_source=source[:,:,:3].astype(float)*a[:,:,None]/255
    pm_candidate=preview[:,:,:3].astype(float)*aa[:,:,None]/255
    premult_error=float(np.abs(pm_source-pm_candidate).max(2)[union].mean())
    sa=a>=peak*.5;ca=aa>=peak*.5
    iou=float((sa&ca).sum()/max(1,(sa|ca).sum()))
    if alpha_error>3 or premult_error>3 or iou<.97:
        return None
    fitted.source_fit.update(policy='straight_caps_native_alpha_and_premultiplied_rgb',
        source_alpha_peak=peak,serialized_opacity=fitted.opacity,
        native_alpha_mean_error=alpha_error,native_premultiplied_rgb_mean_error=premult_error,
        native_alpha_iou=iou,original_alpha_jointly_validated=True)
    return fitted


def _refine_native_component_strokes(strokes, den, component, source_rgba):
    """Validate a bounded complete straight graph against its native colour ink.

    This is a separate component proof, not a relaxation of the thin capsule
    error thresholds. Only the one-native-pixel AA boundary may differ;
    original solid cores, paint seams and component/hole topology must survive.
    """
    if not 1<=len(strokes)<=8 or any(s.closed or s.primitive or s.n_nodes!=2 or s.opacity!=1 for s in strokes):
        return None
    if source_rgba is None:
        source_rgba=np.dstack((np.asarray(den,dtype=np.uint8),np.full(den.shape[:2],255,np.uint8)))
    original=np.asarray(source_rgba)
    nh,nw=original.shape[:2];h,w=den.shape[:2]
    scale=min(nw/float(w),nh/float(h));offset=np.array([(nw-w*scale)/2,(nh-h*scale)/2])
    yy,xx=component[:,0],component[:,1]
    x0=max(0,int(math.floor(xx.min()*scale+offset[0]))-4)
    y0=max(0,int(math.floor(yy.min()*scale+offset[1]))-4)
    x1=min(nw,int(math.ceil((xx.max()+1)*scale+offset[0]))+4)
    y1=min(nh,int(math.ceil((yy.max()+1)*scale+offset[1]))+4)
    if (x1-x0)*(y1-y0)>262144:
        return None
    source=original[y0:y1,x0:x1]
    if np.any(source[:,:,3]!=255):
        return None  # alpha has its own stricter coverage/PM-RGB route
    raw=source[:,:,:3].astype(float)
    border=np.concatenate((raw[0],raw[-1],raw[:,0],raw[:,-1]))
    background=np.median(border,axis=0)
    if np.max(np.abs(border-background))>3:
        return None
    colors=sorted({s.color for s in strokes})
    delta=raw-background
    residual=np.full(raw.shape[:2],np.inf)
    source_coverage=np.zeros(raw.shape[:2],float)
    for color in colors:
        pigment=np.asarray(color,dtype=float)-background
        magnitude=float(pigment@pigment)
        if magnitude<24**2:
            return None
        coverage=np.sum(delta*pigment,axis=2)/magnitude
        error=np.max(np.abs(delta-coverage[:,:,None]*pigment),axis=2)
        best=error<residual
        source_coverage[best]=coverage[best]
        residual[best]=error[best]
    signal=np.max(np.abs(delta),axis=2)>8
    if (not signal.any() or np.percentile(residual[signal],95)>3
            or np.max(source_coverage)>1.025):
        return None
    source_coverage=np.clip(source_coverage,0,1)
    def near(mask):
        pad=np.pad(mask,1);rh,rw=mask.shape
        return np.logical_or.reduce([pad[dy:dy+rh,dx:dx+rw] for dy in range(3) for dx in range(3)])
    def core(mask):
        pad=np.pad(mask,1);rh,rw=mask.shape
        return np.logical_and.reduce([pad[dy:dy+rh,dx:dx+rw] for dy in range(3) for dx in range(3)])
    source_core=core(source_coverage>=.5)&(source_coverage>=.9)
    if int(source_core.sum())<16:
        return None  # a thin AA colour fringe is not an independent pigment
    endpoints=[]
    for stroke in strokes:
        numbers=[float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',stroke.d)]
        if len(numbers)!=4 or re.sub(r'[-+]?(?:\d*\.\d+|\d+)','',stroke.d).strip().replace(' ','')!='ML':
            return None
        endpoints.append(np.array(numbers).reshape(2,2))
    shared=[]
    for index,points in enumerate(endpoints):
        shared.append([any(np.linalg.norm(point-other)<2/scale for j,ends in enumerate(endpoints)
                          if j!=index for other in ends) for point in points])
    original_endpoints=[points.copy() for points in endpoints]
    # Skeleton junction centroids can be 1--2 pixels off the straight arms.
    # Estimate each arm from its interior, then intersect those observed axes.
    # This fixes the geometry rather than declaring the old kink equivalent.
    axes=[];centres=[]
    for stroke,points in zip(strokes,endpoints):
        samples=np.asarray(stroke.sample_points,dtype=float)
        if len(samples)<8:
            return None
        distances=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(samples,axis=0),axis=1))]
        inner=samples[(distances>stroke.width*2)&(distances<distances[-1]-stroke.width*2)]
        if len(inner)<4:
            inner=samples
        centre=np.mean(inner,axis=0)
        _,vectors=np.linalg.eigh((inner-centre).T@(inner-centre))
        axis=vectors[:,-1]
        if (points[1]-points[0])@axis<0:axis=-axis
        # Recover the centre of the original ink cross-section. Skeleton
        # pixels can be biased by half a pixel for even-width raster lines.
        gy,gx=np.nonzero(source_coverage>=.5)
        cloud=(np.column_stack((gx+.5+x0,gy+.5+y0))-offset)/scale
        along=(cloud-centre)@axis
        normal=np.array([-axis[1],axis[0]])
        across=(cloud-centre)@normal
        span=(inner-centre)@axis
        selected=cloud[(along>=span.min())&(along<=span.max())&(np.abs(across)<stroke.width)]
        if len(selected)>=16:
            centre=np.mean(selected,axis=0)
            _,vectors=np.linalg.eigh((selected-centre).T@(selected-centre))
            observed=vectors[:,-1]
            if observed@axis<0:observed=-observed
            if abs(float(observed@axis))>.995:axis=observed
        axes.append(axis);centres.append(centre)
    normals=np.array([[-axis[1],axis[0]] for axis in axes])
    constants=np.array([normal@centre for normal,centre in zip(normals,centres)])
    junction=None
    if np.linalg.matrix_rank(normals)==2 and all(any(flags) for flags in shared):
        junction=np.linalg.lstsq(normals,constants,rcond=None)[0]
        if any(np.linalg.norm(junction-point)>3/scale for points,flags in zip(endpoints,shared)
               for point,is_shared in zip(points,flags) if is_shared):
            return None
    for index,(points,axis,centre) in enumerate(zip(endpoints,axes,centres)):
        for endpoint in range(2):
            endpoints[index][endpoint]=(junction if shared[index][endpoint] and junction is not None
                else centre+axis*((points[endpoint]-centre)@axis))
    native_endpoints=[points.copy() for points in endpoints]
    gy,gx=np.nonzero(source_coverage>=.5)
    cloud=(np.column_stack((gx+.5+x0,gy+.5+y0))-offset)/scale
    for index,(points,axis,centre) in enumerate(zip(native_endpoints,axes,centres)):
        normal=np.array([-axis[1],axis[0]])
        selected=cloud[np.abs((cloud-centre)@normal)<strokes[index].width*.65]
        projections=(selected-centre)@axis
        edge=.5*(abs(axis[0])+abs(axis[1]))/scale
        for endpoint in range(2):
            if not shared[index][endpoint]:
                value=(projections.min()-edge if endpoint==0 else projections.max()+edge)
                points[endpoint]=centre+axis*value
    median_width=float(np.median([s.width for s in strokes]))*scale
    widths=sorted({round(median_width*f,4) for f in (.90,.95,1.,1.05,1.10)}
                  | {float(round(median_width)),float(math.floor(median_width)),float(math.ceil(median_width))})
    best=None;evaluations=0
    # Keep the old axes in the candidate set: fitting a new centre must not
    # silently remove a better original proposal. Square caps cover the
    # shared junction; their outer endpoints are inset by half their width.
    families=[(original_endpoints,False),(endpoints,False),(native_endpoints,True)]
    for family,measured_tips in families:
      for cap in ('butt','round','square'):
        for extension in ((0.,) if measured_tips else (0.,.5)):
            for native_width in widths:
                if native_width<=0:
                    continue
                trial=[]
                for index,(stroke,points) in enumerate(zip(strokes,family)):
                    axis=(points[1]-points[0]);length=float(np.linalg.norm(axis))
                    if length<=0:
                        return None
                    axis/=length
                    first,last=points.copy()
                    if not shared[index][0]:first-=axis*native_width*extension/scale
                    if not shared[index][1]:last+=axis*native_width*extension/scale
                    if measured_tips and cap in ('round','square'):
                        if not shared[index][0]:first+=axis*native_width*.5/scale
                        if not shared[index][1]:last-=axis*native_width*.5/scale
                    proposal=replace(stroke,width=round(native_width/scale,4),linecap=cap,
                        d=f'M{first[0]:.4f} {first[1]:.4f} L{last[0]:.4f} {last[1]:.4f}',source_fit={})
                    trial.append(proposal)
                preview=_render_native_strokes(trial,[x0,y0,x1,y1],den.shape[:2],original.shape[:2])
                evaluations+=1
                coverage=preview[:,:,3].astype(float)/255
                predicted=preview[:,:,:3]*coverage[:,:,None]+background*(1-coverage[:,:,None])
                if (np.any((coverage>=.125)&~near(source_coverage>=.125))
                        or np.any((source_coverage>=.125)&~near(coverage>=.125))
                        or np.any(source_core&(coverage<.5))):
                    continue
                protected=source_core | core(coverage>=.5)
                protected_error=float(np.abs(predicted-raw).max(2)[protected].max())
                if protected_error>6:
                    continue
                union=(source_coverage>.05)|(coverage>.05)
                score=float(np.abs(predicted-raw).max(2)[union].mean())
                if best is None or score<best[0]:
                    best=(score,trial,preview,protected_error)
    if best is None:
        return None
    score,trial,preview,protected_error=best
    if score>12:
        return None
    from source_scene_guard import _components, _holes
    source_alpha=np.rint(source_coverage*255).astype(np.uint8)
    topology=[]
    for threshold in (32,128,224):
        _,a=_components(source_alpha>=threshold);_,b=_components(preview[:,:,3]>=threshold)
        _,ha=_holes(source_alpha,threshold);_,hb=_holes(preview[:,:,3],threshold)
        if a!=b or int(ha.sum())!=int(hb.sum()):
            return None
        topology.append({'alpha_threshold':threshold,'components':a,'holes':int(ha.sum())})
    proof={'policy':'native_whole_straight_component_core_and_aa_boundary',
        'serialized_geometry_validated':True,'source_color_jointly_validated':True,
        'native_roi':[x0,y0,x1,y1],'source_dimensions':[nw,nh],
        'maximum_boundary_tolerance_native_pixels':1,'boundary_neighbourhood':'Chebyshev',
        'original_solid_core_max_rgb_error':protected_error,'ink_mean_max_channel_error':score,
        'topology':topology,'component_strokes':len(trial),'render_evaluations':evaluations,
        'scope':'native_raster_equivalence_with_one_pixel_AA_boundary_not_original_authoring_intent'}
    for stroke in trial:
        stroke.source_fit={**proof,'linecap':stroke.linecap}
    return trial


def _refine_opaque_palette_straight(stroke, den, component, source_rgba):
    """Prove a solid line on varying flat fills without claiming their paint.

    This narrow branch requires native opaque, discrete paint pixels and a
    substantial solid pigment core. A palette fringe around a large fill has
    no such core and cannot use this exception to the whole-ink ownership rule.
    Only the stroke's observed colour occupancy is reconstructed; the ordinary
    composed scene validation still checks the underlying fill representation.
    """
    if source_rgba is None or stroke.n_nodes!=2 or stroke.closed or stroke.primitive:
        return None
    source=np.asarray(source_rgba)
    nh,nw=source.shape[:2];h,w=den.shape[:2]
    scale=min(nw/float(w),nh/float(h));offset=np.array([(nw-w*scale)/2,(nh-h*scale)/2])
    yy,xx=component[:,0],component[:,1]
    x0=max(0,int(math.floor(xx.min()*scale+offset[0]))-4)
    y0=max(0,int(math.floor(yy.min()*scale+offset[1]))-4)
    x1=min(nw,int(math.ceil((xx.max()+1)*scale+offset[0]))+4)
    y1=min(nh,int(math.ceil((yy.max()+1)*scale+offset[1]))+4)
    patch=source[y0:y1,x0:x1]
    if patch.shape[0]*patch.shape[1]>262144 or np.any(patch[:,:,3]!=255):
        return None
    colors=np.unique(patch[:,:,:3].reshape(-1,3),axis=0)
    if len(colors)>8:
        return None  # no unsupported alpha/paint unmixing on continuous tones
    distance=np.max(np.abs(colors.astype(float)-stroke.color),axis=1)
    if not np.any(distance<=3) or np.any((distance>3)&(distance<24)):
        return None
    pigment=colors[int(distance.argmin())]
    owned=np.max(np.abs(patch[:,:,:3].astype(float)-pigment),axis=2)<=3
    from source_scene_guard import _components
    _,count=_components(owned)
    if count!=1:
        return None
    solid=owned.copy()
    for _ in range(2):
        p=np.pad(solid,1);hh,ww=solid.shape
        solid=np.logical_and.reduce([p[dy:dy+hh,dx:dx+ww] for dy in range(3) for dx in range(3)])
    if int(solid.sum())<16:
        return None
    measurement=np.full_like(source,255)
    measurement[y0:y1,x0:x1,:3][owned]=pigment
    model=replace(stroke,color=tuple(map(int,pigment)),source_fit={})
    fitted=_refine_native_component_strokes([model],den,component,measurement)
    if fitted is None:
        return None
    result=fitted[0]
    result.color=tuple(map(int,pigment))
    result.source_fit.update(policy='native_discrete_palette_line_occupancy',
        source_paint_core_pixels=int(solid.sum()),native_paint_colors=len(colors),
        underlying_fill_requires_composed_scene_validation=True)
    return result


def infer_occluded_flat_paint(strokes, source_rgba, den, visible, vis_fill, palette):
    """Infer flat support beneath proven opaque strokes, never source pixels.

    Equal native opaque paint on both normal rays is required for every added
    working pixel. The caller uses this only in its tracing surrogate; original
    references and visibility/alpha evidence remain untouched.
    """
    h,w=den.shape[:2];nh,nw=source_rgba.shape[:2]
    scale=min(nw/float(w),nh/float(h));offset=np.array([(nw-w*scale)/2,(nh-h*scale)/2])
    restored=np.zeros((h,w),bool);paint=np.zeros((h,w,3),np.uint8)
    yy,xx=np.nonzero(visible & ~vis_fill)
    if not len(xx):return restored,paint
    coords=np.column_stack((xx+.5,yy+.5))
    for stroke in strokes[:24]:
        if (stroke.opacity!=1 or stroke.source_fit.get('policy')!='native_discrete_palette_line_occupancy'):
            continue
        values=[float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',stroke.d)]
        if len(values)!=4:continue
        first,last=np.array(values).reshape(2,2)
        direction=last-first;length=float(np.linalg.norm(direction));direction/=length
        normal=np.array([-direction[1],direction[0]])
        along=(coords-first)@direction;across=np.abs((coords-first)@normal)
        candidates=(across<=stroke.width*.5+1)&(along>=-stroke.width)&(along<=length+stroke.width)
        indices=np.flatnonzero(candidates)
        if len(indices)>65536:continue
        points=coords[indices]*scale+offset
        xi=np.clip(points[:,0].astype(int),0,nw-1);yi=np.clip(points[:,1].astype(int),0,nh-1)
        core=source_rgba[yi,xi]
        legitimate=(core[:,3]==255)&(np.max(np.abs(core[:,:3].astype(float)-stroke.color),axis=1)<=3)
        side_colors=[];side_valid=[]
        for sign in (-1,1):
            found=np.zeros(len(indices),bool);colors=np.zeros((len(indices),3),np.uint8)
            for distance in np.arange(1.,min(68.,stroke.width*scale*2+4),1.):
                sample=points+normal*distance*sign
                sx=sample[:,0].astype(int);sy=sample[:,1].astype(int)
                inside=(sx>=0)&(sx<nw)&(sy>=0)&(sy<nh)
                rgba=source_rgba[np.clip(sy,0,nh-1),np.clip(sx,0,nw-1)]
                hit=inside&~found&(rgba[:,3]==255)&(np.max(np.abs(rgba[:,:3].astype(float)-stroke.color),axis=1)>24)
                colors[hit]=rgba[hit,:3];found|=hit
            side_colors.append(colors);side_valid.append(found)
        good=legitimate&side_valid[0]&side_valid[1]
        good&=np.max(np.abs(side_colors[0].astype(float)-side_colors[1]),axis=1)<=3
        # Only a surviving fill paint can be bridged, never inferred paper.
        distances=np.max(np.abs(side_colors[0][:,None,:].astype(float)-palette[None,:,:]),axis=2)
        good&=distances.min(axis=1)<=3
        target=indices[good]
        restored[yy[target],xx[target]]=True
        paint[yy[target],xx[target]]=palette[distances.argmin(axis=1)[good]]
    return restored,paint


def _validate_native_open_curve(stroke, den, component, source_rgba):
    """Prove an isolated uniformly painted curve before consuming its fill."""
    if stroke.closed or stroke.primitive or stroke.opacity!=1 or stroke.n_nodes>20 or len(stroke.sample_points)>4096:
        return None
    if source_rgba is None:
        source_rgba=np.dstack((np.asarray(den,dtype=np.uint8),np.full(den.shape[:2],255,np.uint8)))
    source=np.asarray(source_rgba);nh,nw=source.shape[:2];h,w=den.shape[:2]
    scale=min(nw/float(w),nh/float(h));offset=np.array([(nw-w*scale)/2,(nh-h*scale)/2])
    yy,xx=component[:,0],component[:,1]
    x0=max(0,int(math.floor(xx.min()*scale+offset[0]))-4)
    y0=max(0,int(math.floor(yy.min()*scale+offset[1]))-4)
    x1=min(nw,int(math.ceil((xx.max()+1)*scale+offset[0]))+4)
    y1=min(nh,int(math.ceil((yy.max()+1)*scale+offset[1]))+4)
    patch=source[y0:y1,x0:x1]
    if patch.shape[0]*patch.shape[1]>262144 or np.any(patch[:,:,3]!=255):return None
    rgb=patch[:,:,:3].astype(float)
    border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]))
    bg=np.median(border,axis=0)
    if np.max(np.abs(border-bg))>3:return None
    pigment=np.asarray(stroke.color,dtype=float)-bg
    magnitude=float(pigment@pigment)
    if magnitude<24**2:return None
    coverage=np.sum((rgb-bg)*pigment,axis=2)/magnitude
    residual=np.max(np.abs(rgb-bg-coverage[:,:,None]*pigment),axis=2)
    ink=coverage>.05
    if not ink.any() or residual[ink].max()>6 or coverage.max()>1.025:return None
    coverage=np.clip(coverage,0,1)
    def morph(mask,erode=False):
        p=np.pad(mask,1);hh,ww=mask.shape
        parts=[p[dy:dy+hh,dx:dx+ww] for dy in range(3) for dx in range(3)]
        return np.logical_and.reduce(parts) if erode else np.logical_or.reduce(parts)
    core=morph(coverage>=.5,True)&(coverage>=.9)
    if not core.any():return None
    hard_binary=(len(np.unique(patch[:,:,:3].reshape(-1,3),axis=0))==2
                 and np.all((coverage<1e-9)|(coverage>1-1e-9)))
    contrast=float(np.max(np.abs(pigment)))
    source_occupancy=coverage>=.5
    source_boundary=morph(source_occupancy)&~morph(source_occupancy,True)
    pixel_y,pixel_x=np.indices(coverage.shape)
    endpoint_regions=[]
    for point in (stroke.sample_points[0],stroke.sample_points[-1]):
        native=np.asarray(point)*scale+offset-[x0,y0]
        endpoint_regions.append((pixel_x+.5-native[0])**2+(pixel_y+.5-native[1])**2
                                <=(stroke.width*scale*1.5)**2)
    if set(re.findall('[A-Za-z]',stroke.d))-set('MLC'):return None
    best=None;evaluations=0
    from curve_refit import fit_curve
    proposals=[(stroke.d,stroke.n_nodes)]
    proposal_widths={}
    # A thinned L can end half a width before each native tip and move its
    # corner into the turn. Whole-path shifts cannot fix those three points.
    # Recover a single axis-aligned elbow from the observed two straight ink
    # bands; this remains a proposal, subject to every native render guard.
    # No corner is rounded into a cubic and no source pixels are consumed yet.
    if re.findall('[A-Za-z]',stroke.d)==['M','L','L']:
        values=re.findall(r'[-+]?(?:\d*\.\d+|\d+)',stroke.d)
        if len(values)==6:
            p=np.asarray([float(v) for v in values]).reshape(3,2)*scale+offset-[x0,y0]
            directions=np.diff(p,axis=0)
            axes=np.argmax(np.abs(directions),axis=1)
            lengths=np.linalg.norm(directions,axis=1)
            native_width=stroke.width*scale
            axis_supported=(axes[0]!=axes[1] and np.all(lengths>native_width*4)
                and all(abs(directions[k,1-axes[k]])<=lengths[k]*.05 for k in range(2)))
            if axis_supported:
                grid=np.stack([pixel_x+.5,pixel_y+.5],axis=2)
                centres=[];widths=[];tips=[]
                for k in range(2):
                    major=int(axes[k]);minor=1-major
                    low,high=sorted((p[k,major],p[k+1,major]))
                    middle=(grid[:,:,major]>=low+native_width*1.5)&(grid[:,:,major]<=high-native_width*1.5)
                    middle&=np.abs(grid[:,:,minor]-(p[k,minor]+p[k+1,minor])/2)<=native_width
                    weights=coverage*middle
                    mass=float(weights.sum())
                    slices=int(np.count_nonzero(np.any(middle,axis=0 if major==0 else 1)))
                    if mass<=0 or slices<4:break
                    centre=float((weights*grid[:,:,minor]).sum()/mass)
                    band_width=mass/slices
                    # Endpoint is measured on its own straight arm, away from
                    # the other arm. Pixel boundary proposals suit binary
                    # sources; AA sources must still pass their stricter RGB.
                    end=p[0 if k==0 else 2,major]
                    joint=p[1,major]
                    arm=(np.abs(grid[:,:,minor]-centre)<band_width*.6)&(coverage>=.5)
                    arm&=(grid[:,:,major]<(joint-native_width) if end<joint else grid[:,:,major]>(joint+native_width))
                    positions=grid[:,:,major][arm]
                    if not len(positions):break
                    tip=float(positions.min()-.5 if end<joint else positions.max()+.5)
                    centres.append(centre);widths.append(band_width);tips.append(tip)
                if len(centres)==2 and abs(widths[0]-widths[1])<=max(.25,native_width*.03):
                    joint=np.zeros(2)
                    for k in range(2):joint[1-axes[k]]=centres[k]
                    ends=[joint.copy(),joint.copy()]
                    for k in range(2):ends[k][axes[k]]=tips[k]
                    points=(np.vstack((ends[0],joint,ends[1]))+[x0,y0]-offset)/scale
                    path='M'+' L'.join(' '.join(f'{v:.4f}' for v in point) for point in points)
                    proposals.append((path,3))
                    proposal_widths[path]=float(np.mean(widths))/scale
    try:
        points=np.asarray(stroke.sample_points,dtype=float)
        if len(points)>8:
            padded=np.pad(points,((2,2),(0,0)),mode='edge')
            smoothed=sum(padded[k:k+len(points)]*weight for k,weight in enumerate((1,2,3,2,1)))/9
            smoothed[:2]=points[:2];smoothed[-2:]=points[-2:]
        else:smoothed=points
        gy,gx=np.nonzero(coverage>=.5)
        cloud=(np.column_stack((gx+.5+x0,gy+.5+y0))-offset)/scale
        # Thinning stops inside a rounded endpoint. Recover the cap centre
        # from its observed native tip rather than shortening the true line.
        for end in (0,-1):
            inner=smoothed[min(len(smoothed)-1,8) if end==0 else max(0,len(smoothed)-9)]
            axis=smoothed[end]-inner;axis/=np.linalg.norm(axis)
            normal=np.array([-axis[1],axis[0]])
            cloud_along=(cloud-smoothed[end])@axis
            nearby=(cloud_along>-stroke.width)&(np.abs((cloud-smoothed[end])@normal)<stroke.width)
            if nearby.any():
                extension=float(cloud_along[nearby].max())+.5*(abs(axis[0])+abs(axis[1]))/scale-stroke.width*.475
                if abs(extension)<stroke.width*.3:smoothed[end]+=axis*extension
        fit=fit_curve(smoothed,closed=False,tolerance=.4*max(1,min(nw,nh)/128)/scale,line_tolerance=.15/scale,
            corner_angle=80,corner_window=3/scale,allow_primitives=False,max_segments=8)
        if fit['anchor_count']<=10:proposals.append((fit['path'],fit['anchor_count']))
    except (ValueError,RuntimeError,ArithmeticError):pass
    # Circular monoline arcs have a stronger low-anchor proposal than a
    # staircase-derived Catmull path. Native angular support supplies butt
    # endpoints; both cap models still face the same pixel/core/topology gates.
    try:
        points=np.asarray(stroke.sample_points,dtype=float)
        system=np.column_stack((2*points[:,0],2*points[:,1],np.ones(len(points))))
        solved=np.linalg.lstsq(system,np.sum(points*points,axis=1),rcond=None)[0]
        centre=solved[:2];radial=np.linalg.norm(points-centre,axis=1)
        if np.quantile(np.abs(radial-np.median(radial)),.95)<.75/scale:
            gy,gx=np.nonzero(coverage>=.5)
            cloud=(np.column_stack((gx+.5+x0,gy+.5+y0))-offset)/scale
            radius=float(np.median(np.linalg.norm(cloud-centre,axis=1)))
            angles=np.unwrap(np.arctan2(points[:,1]-centre[1],points[:,0]-centre[0]))
            middle=float((angles[0]+angles[-1])/2)
            ca=np.arctan2(cloud[:,1]-centre[1],cloud[:,0]-centre[0])
            ca+=2*math.pi*np.round((middle-ca)/(2*math.pi))
            low,high=float(ca.min()),float(ca.max())
            start,end=(low,high) if angles[-1]>angles[0] else (high,low)
            count=int(math.ceil(abs(end-start)/(math.pi/2)))
            if 1<=count<=4:
              for candidate_radius in sorted({radius,round(radius*scale*2)/(scale*2)}):
                parts=[]
                for a,b in zip(np.linspace(start,end,count+1)[:-1],np.linspace(start,end,count+1)[1:]):
                    p0=centre+candidate_radius*np.array([math.cos(a),math.sin(a)])
                    p3=centre+candidate_radius*np.array([math.cos(b),math.sin(b)])
                    k=4/3*math.tan((b-a)/4)
                    p1=p0+candidate_radius*k*np.array([-math.sin(a),math.cos(a)])
                    p2=p3-candidate_radius*k*np.array([-math.sin(b),math.cos(b)])
                    if not parts:parts.append(f'M{p0[0]:.4f} {p0[1]:.4f}')
                    parts.append('C'+' '.join(f'{v:.4f}' for v in (*p1,*p2,*p3)))
                proposals.append((' '.join(parts),count+1))
    except (ValueError,ArithmeticError,np.linalg.LinAlgError):pass
    shifts=((0,0),(-.25,0),(.25,0),(0,-.25),(0,.25),(-.25,-.25),(.25,.25),
            (-.25,.25),(.25,-.25),(-.5,0),(.5,0),(0,-.5),(0,.5))
    factors=sorted({.90,.925,.95,.975,1.,1.025,round(stroke.width*scale)/(stroke.width*scale)})
    for path,nodes in proposals:
        path_factors=factors
        if path in proposal_widths:
            # The long straight bands directly measure their common width.
            # Do not shrink the whole line merely to fit a few quantized
            # corner pixels when the native band width already passes.
            path_factors=[proposal_widths[path]/stroke.width]
        for shift_x,shift_y in shifts:
            coordinates=iter([shift_x/scale,shift_y/scale]*64)
            shifted=re.sub(r'[-+]?(?:\d*\.\d+|\d+)',
                lambda m:f'{float(m.group())+next(coordinates):.4f}',path)
            for cap in ('round','butt'):
                for factor in path_factors:
                    trial=replace(stroke,d=shifted,n_nodes=nodes,linecap=cap,
                                  width=round(stroke.width*factor,4),source_fit={})
                    rendered=_render_native_strokes([trial],[x0,y0,x1,y1],den.shape[:2],source.shape[:2])
                    evaluations+=1
                    a=rendered[:,:,3].astype(float)/255
                    predicted=rendered[:,:,:3]*a[:,:,None]+bg*(1-a[:,:,None])
                    if (np.any((a>=.125)&~morph(coverage>=.125))
                            or np.any((coverage>=.125)&~morph(a>=.125))):
                        continue
                    protected=core|morph(a>=.5,True)
                    error=np.abs(predicted-rgb).max(2)
                    if error[protected].max()>6:
                        continue
                    union=ink|(a>.05)
                    score=float(error[union].mean())
                    endpoint_checks=[]
                    if hard_binary:
                        occupied=a>=.5
                        iou=float((source_occupancy&occupied).sum()/max(1,(source_occupancy|occupied).sum()))
                        if iou<.97 or np.any((error>1)&~source_boundary) or score>contrast*.06:
                            continue
                        for region in endpoint_regions:
                            local=region&union
                            endpoint_error=float(error[local].mean()) if local.any() else math.inf
                            endpoint_iou=float((region&source_occupancy&occupied).sum()/max(1,(region&(source_occupancy|occupied)).sum()))
                            endpoint_checks.append({'mean_max_rgb_error':endpoint_error,'occupancy_iou':endpoint_iou})
                        # Caps have a local occupancy proof; a whole-arc score
                        # cannot hide a shortened or rounded endpoint. The
                        # one-pixel boundary and solid-core gates also apply.
                        if any(v['occupancy_iou']<.95 for v in endpoint_checks):
                            continue
                    elif score>12:
                        continue
                    if best is None or score<best[0]:
                        best=(score,trial,rendered,float(error[protected].max()),endpoint_checks,
                              path in proposal_widths)
    if best is None:return None
    score,result,rendered,core_error,endpoint_checks,native_elbow=best
    from source_scene_guard import _components,_holes
    original_alpha=np.rint(coverage*255).astype(np.uint8)
    for threshold in (32,128,224):
        _,a=_components(original_alpha>=threshold);_,b=_components(rendered[:,:,3]>=threshold)
        _,ha=_holes(original_alpha,threshold);_,hb=_holes(rendered[:,:,3],threshold)
        if a!=b or int(ha.sum())!=int(hb.sum()):return None
    result.source_fit={'policy':'native_uniform_curve_core_boundary_and_topology',
        'serialized_geometry_validated':True,'source_color_jointly_validated':True,
        'native_roi':[x0,y0,x1,y1],'source_dimensions':[nw,nh],
        'original_solid_core_max_rgb_error':core_error,'ink_mean_max_channel_error':score,
        'maximum_boundary_tolerance_native_pixels':1,'render_evaluations':evaluations,
        'observation_model':'two_opaque_paints_pixel_occupancy' if hard_binary else 'antialiased_native_rgb',
        'hard_binary_endpoint_checks':endpoint_checks,
        'hard_binary_max_normalized_rgb_error':.06 if hard_binary else None,
        'hard_binary_minimum_global_occupancy_iou':.97 if hard_binary else None,
        'hard_binary_minimum_endpoint_occupancy_iou':.95 if hard_binary else None,
        'hard_binary_outside_boundary_quantization_allowance_rgb':1 if hard_binary else None,
        'original_opaque_paint_count':2 if hard_binary else None,'linecap':result.linecap,
        'native_axis_aligned_elbow_proposal_used':native_elbow,
        'scope':'native_raster_equivalence_not_original_authoring_intent'}
    return result


def _refine_straight_aa_stroke(stroke, den, palette, bg, component, labels,
                               component_id, alpha, source_rgba=None):
    """Jointly fit straight-line colour, width and butt/round caps to source.

    Return None when the bounded native-source comparison cannot validate
    the representation. The caller must defer the *whole* component, including Tier B. The
    old rounded fallback could consume rectangular bars and their gaps.
    """
    if (stroke.closed or stroke.primitive or stroke.opacity != 1.0
            or stroke.n_nodes != 2 or len(stroke.sample_points) < 2):
        return None
    try:
        import resvg_py
        from PIL import Image
    except ImportError:
        return None
    H, W = den.shape[:2]
    if source_rgba is None:
        native_rgb = den
        native_alpha = alpha
    else:
        original = np.asarray(source_rgba)
        if original.ndim != 3 or original.shape[2] != 4:
            return None
        native_rgb, native_alpha = original[:, :, :3], original[:, :, 3]
    nh, nw = native_rgb.shape[:2]
    # Serialized SVG uses the default xMidYMid meet viewBox mapping. Work in
    # that exact native coordinate system; never validate a resized proxy.
    scale = min(nw / float(W), nh / float(H))
    offset = np.array([(nw - W * scale) / 2, (nh - H * scale) / 2])
    if not .4 <= stroke.width * scale <= 16.1:
        return None
    yy, xx = component[:, 0], component[:, 1]
    margin = max(1, int(math.ceil(4 / scale)))
    ly0, ly1 = max(0, int(yy.min()) - margin), min(H, int(yy.max()) + margin + 1)
    lx0, lx1 = max(0, int(xx.min()) - margin), min(W, int(xx.max()) + margin + 1)
    nearby = labels[ly0:ly1, lx0:lx1]
    if np.any((nearby != 0) & (nearby != component_id)):
        return None
    y0 = max(0, int(math.floor(yy.min() * scale + offset[1])) - 4)
    y1 = min(nh, int(math.ceil((yy.max() + 1) * scale + offset[1])) + 4)
    x0 = max(0, int(math.floor(xx.min() * scale + offset[0])) - 4)
    x1 = min(nw, int(math.ceil((xx.max() + 1) * scale + offset[0])) + 4)
    h, w = y1 - y0, x1 - x0
    if 32768 < w*h <= 262144:
        # A 3000px one-pixel rule may occupy two downsampled mask rows. Locate
        # the native ink inside that bounded window before allocating renders;
        # keep the original 32768-pixel render budget and all observed ink.
        window=np.asarray(native_rgb[y0:y1,x0:x1],dtype=float)
        border=np.concatenate((window[0],window[-1],window[:,0],window[:,-1]))
        paper=np.median(border,axis=0)
        if np.max(np.abs(border-paper))<=3:
            iy,ix=np.nonzero(np.linalg.norm(window-paper,axis=2)>10)
            if len(ix):
                ax=max(0,int(ix.min())-4);ay=max(0,int(iy.min())-4)
                bx=min(w,int(ix.max())+5);by=min(h,int(iy.max())+5)
                x1=x0+bx;y1=y0+by;x0+=ax;y0+=ay
                h,w=y1-y0,x1-x0
    if w * h > 32768 or min(h, w) < 3:
        return None
    raw = np.asarray(native_rgb[y0:y1, x0:x1], dtype=np.float64)
    border = np.concatenate((raw[0], raw[-1], raw[:, 0], raw[:, -1]))
    if source_rgba is not None:
        bg = np.median(border, axis=0)
    bg = np.asarray(bg, dtype=np.float64)
    if np.max(np.abs(border - bg)) > 3:
        return None
    delta = bg - raw
    strength = np.linalg.norm(delta, axis=2)
    signal = strength > 10
    if not np.any(signal):
        return None
    # This model is opaque ink on uniform paper, not alpha-colour unmixing.
    if native_alpha is not None and np.any(np.asarray(native_alpha)[y0:y1, x0:x1][signal] != 255):
        return None
    py, px = np.indices((h, w), dtype=np.float64)
    coords = np.column_stack((px.ravel() + .5, py.ravel() + .5))
    weights = strength.ravel()
    centre = (coords * weights[:, None]).sum(axis=0) / weights.sum()
    centered = coords - centre
    covariance = (centered * weights[:, None]).T @ centered / weights.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    if eigenvalues[-1] < 100 * max(.01, eigenvalues[0]):
        return None
    projected = centered @ direction
    variance = float(np.sum(projected ** 2 * weights) / weights.sum())
    colors = {tuple(stroke.color)}
    ranked = sorted((tuple(map(int, color)) for color in palette),
                    key=lambda color: -float(np.linalg.norm(np.asarray(color) - bg)))
    colors.update(ranked[:8])
    if source_rgba is not None:
        core = raw[strength >= np.quantile(strength[signal], .85)]
        core_color = tuple(map(int, np.rint(np.median(core, axis=0))))
        # Do not promote one-level antialiasing variants into new pigments;
        # original-resolution evidence still adds a genuinely missing colour.
        if min(np.max(np.abs(np.asarray(color) - core_color)) for color in colors) > 3:
            colors.add(core_color)
    candidates = {"butt": [], "round": []}
    for color in sorted(colors):
        pigment = bg - color
        magnitude = float(pigment @ pigment)
        if magnitude < 24 ** 2:
            continue
        coverage = np.sum(delta * pigment, axis=2) / magnitude
        residual = np.abs(delta - coverage[:, :, None] * pigment).max(axis=2)
        if (float(coverage.max()) > 1.025 or np.percentile(residual[signal], 95) > 2
                or residual[signal].max() > 6):
            continue
        mass = float(np.clip(coverage, 0, 1).sum())
        for cap in candidates:
            if cap == "butt":
                length = math.sqrt(12 * variance)
                width = mass / length
            else:
                # Continuous capsule area/axial moment gives a starting point;
                # all proposals are then compared to native rendered pixels.
                low, high = .45, min(16., math.sqrt(4 * mass / math.pi) * .95)
                for _ in range(35):
                    width = (low + high) / 2
                    radius = width / 2
                    length = max(.01, (mass - math.pi * radius ** 2) / width)
                    moment = (radius * length ** 3 / 6 + math.pi * radius ** 2 * length ** 2 / 4
                              + 4 * length * radius ** 3 / 3 + math.pi * radius ** 4 / 4)
                    if moment / mass > variance:
                        low = width
                    else:
                        high = width
                width = (low + high) / 2
                length = (mass - math.pi * width ** 2 / 4) / width
            if length > MIN_STROKE_LEN and .45 <= width <= 16:
                candidates[cap].append((color, np.array([*centre, length, width]), mass))
    if not any(candidates.values()):
        return None

    render_cache = {}
    def render(parameters, cap, axis=direction):
        cx, cy, length, width = parameters
        first, last = np.array([cx, cy]) - axis * length / 2, np.array([cx, cy]) + axis * length / 2
        key = (cap, *(round(float(v), 7) for v in (*first, *last, width)))
        if key not in render_cache:
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
                   f'<path d="M{first[0]:.7f} {first[1]:.7f} L{last[0]:.7f} {last[1]:.7f}" '
                   f'fill="none" stroke="black" stroke-width="{width:.7f}" stroke-linecap="{cap}"/></svg>')
            png = resvg_py.svg_to_bytes(svg_string=svg, width=w, height=h,
                skip_system_fonts=True, log_information=False, shape_rendering="geometric_precision")
            with Image.open(io.BytesIO(png)) as image:
                render_cache[key] = np.asarray(image.convert('RGBA'), dtype=np.float64)[:, :, 3] / 255
        return render_cache[key]

    def evaluate(parameters, color, cap, axis=direction, source_mass=None):
        if parameters[2] <= 0 or not .40 <= parameters[3] <= 16.1:
            return math.inf, None
        geometric_mass = parameters[2] * parameters[3]
        if cap == "round":
            geometric_mass += math.pi * parameters[3] ** 2 / 4
        if source_mass is not None and abs(geometric_mass - source_mass) > max(.05, source_mass * .01):
            return math.inf, None
        coverage = render(parameters, cap, axis)
        predicted = bg + coverage[:, :, None] * (np.asarray(color) - bg)
        return float(np.abs(predicted - raw).mean()), predicted

    old_first, old_last = [np.asarray(p) * scale + offset for p in
                           (stroke.sample_points[0], stroke.sample_points[-1])]
    old_length = float(np.linalg.norm(old_last - old_first))
    if old_length <= 0:
        return None
    old_axis = (old_last - old_first) / old_length
    old_center = (old_first + old_last) / 2 - [x0, y0]
    baseline, _ = evaluate([*old_center, old_length, stroke.width * scale], stroke.color, "round", old_axis)
    best_by_cap = {}
    # Maximum 2 cap styles x 4 compatible colours x 25 small-ROI renders.
    for cap, hypotheses in candidates.items():
        for color, initial, source_mass in hypotheses[:4]:
            parameters = initial.copy()
            score, predicted = evaluate(parameters, color, cap, source_mass=source_mass)
            for step in (.2, .05, .01):
                for index in (0, 1, 2, 3):
                    for sign in (-1, 1):
                        trial = parameters.copy(); trial[index] += sign * step
                        value, picture = evaluate(trial, color, cap, source_mass=source_mass)
                        if value < score:
                            parameters, score, predicted = trial, value, picture
            if predicted is not None and (cap not in best_by_cap or score < best_by_cap[cap][0]):
                best_by_cap[cap] = (score, parameters, color, predicted, source_mass)
    if len(best_by_cap) != 2:
        return None  # absence of the competing model is not cap evidence
    cap = min(best_by_cap, key=lambda key: best_by_cap[key][0])
    # A one-pixel line can have two genuinely identical native rasters with
    # different cap/length parameterizations. Keep it editable using a stable
    # round tie-break, but explicitly retain the unresolved cap identity.
    cap_ambiguity = bool(np.max(np.abs(best_by_cap['butt'][3]
                                      - best_by_cap['round'][3])) <= 1.0)
    if cap_ambiguity:
        cap = 'round'
    score, parameters, color, predicted, source_mass = best_by_cap[cap]
    other = best_by_cap["butt" if cap == "round" else "round"]
    # Test the endpoint region separately: a long line's interior must not
    # hide rounded-off rectangle corners or a wrong extension of the ends.
    axial = (coords - parameters[:2]) @ direction
    cross = np.abs((coords - parameters[:2]) @ np.array([-direction[1], direction[0]]))
    endpoint = ((np.abs(axial) >= parameters[2] / 2 - max(2., parameters[3]))
                & (cross <= parameters[3] + 2)).reshape(h, w)
    endpoint_error = float(np.abs(predicted - raw).max(axis=2)[endpoint].mean())
    competing_endpoint_error = float(np.abs(other[3] - raw).max(axis=2)[endpoint].mean())
    if not cap_ambiguity and (other[0] - score <= max(.001, score * .10)
            or competing_endpoint_error - endpoint_error <= .05):
        return None
    union = signal | (np.linalg.norm(bg - predicted, axis=2) > 10)
    ink_error = float(np.abs(predicted - raw).max(axis=2)[union].mean())
    source_ink = np.linalg.norm(bg - raw, axis=2) > 20
    candidate_ink = np.linalg.norm(bg - predicted, axis=2) > 20
    iou = float((source_ink & candidate_ink).sum() / max(1, (source_ink | candidate_ink).sum()))
    if score > baseline + .001 or ink_error > 3.0 or endpoint_error > 3.0 or iou < .97:
        return None
    if cap_ambiguity:
        other_union = signal | (np.linalg.norm(bg - other[3], axis=2) > 10)
        other_error = float(np.abs(other[3] - raw).max(axis=2)[other_union].mean())
        other_ink = np.linalg.norm(bg - other[3], axis=2) > 20
        other_iou = float((source_ink & other_ink).sum() / max(1, (source_ink | other_ink).sum()))
        if (other[0] > baseline + .001 or other_error > 3.0
                or competing_endpoint_error > 3.0 or other_iou < .97):
            return None
    cx, cy, length, width = parameters
    first, last = np.array([cx + x0, cy + y0]) - direction * length / 2, np.array([cx + x0, cy + y0]) + direction * length / 2
    first, last = [(point - offset) / scale for point in (first, last)]
    # Validate the exact rounded numbers that will be serialized, not just
    # the higher-precision optimizer state.
    first, last = np.round(first, 4), np.round(last, 4)
    output_width = round(float(width / scale), 4)
    nf, nl = first * scale + offset, last * scale + offset
    final_length = float(np.linalg.norm(nl - nf))
    final_axis = (nl - nf) / final_length
    final_params = [*((nf + nl) / 2 - [x0, y0]), final_length, output_width * scale]
    final_score, final_picture = evaluate(final_params, color, cap, final_axis, source_mass)
    # Select the neighbouring serialized width on the same strict objective.
    # Rounding a subpixel working width down can turn every native black pixel
    # into alpha254; do not discard the exact one-pixel solution for that.
    for delta_width in (-.0001,.0001):
        serialized_width=round(output_width+delta_width,4)
        params=[*final_params[:3],serialized_width*scale]
        value,picture=evaluate(params,color,cap,final_axis,source_mass)
        if picture is not None and value<final_score:
            final_score,final_picture=value,picture
            output_width=serialized_width
    if final_picture is None or final_score > score + .01:
        return None
    final_union = signal | (np.linalg.norm(bg - final_picture, axis=2) > 10)
    ink_error = float(np.abs(final_picture - raw).max(axis=2)[final_union].mean())
    endpoint_error = float(np.abs(final_picture - raw).max(axis=2)[endpoint].mean())
    final_ink = np.linalg.norm(bg - final_picture, axis=2) > 20
    iou = float((source_ink & final_ink).sum() / max(1, (source_ink | final_ink).sum()))
    if ink_error > 3.0 or endpoint_error > 3.0 or iou < .97:
        return None
    stroke.width = output_width
    stroke.color = color
    stroke.linecap = cap
    stroke.d = f'M{first[0]:.4f} {first[1]:.4f} L{last[0]:.4f} {last[1]:.4f}'
    stroke.length = final_length / scale
    stroke.n_nodes = 2
    stroke.sample_points = [tuple(first + t * (last - first)) for t in
        np.linspace(0, 1, min(256, max(16, len(stroke.sample_points))))]
    stroke.source_fit = {'policy': 'straight_caps_native_render_source_coverage',
        'linecap': cap, 'cap_models_compared': ['butt', 'round'],
        'cap_ambiguity': cap_ambiguity,
        'equivalent_at_source_resolution': cap_ambiguity,
        'baseline_rgb_mae': baseline, 'candidate_rgb_mae': final_score,
        'source_ink_mean_max_channel_error': ink_error, 'source_ink_iou': iou,
        'endpoint_max_channel_mae': endpoint_error,
        'competing_endpoint_max_channel_mae': competing_endpoint_error,
        'coverage_not_binary_area': True, 'source_color_jointly_validated': True,
        'source_dimensions': [nw, nh], 'native_roi': [x0, y0, x1, y1],
        'serialized_geometry_validated': True, 'render_evaluations': len(render_cache)}
    return stroke


def extract_strokes(ink_mask, den, palette, bg_color=(255, 255, 255),
                    alpha=None, source_rgba=None, audit=None):
    """Find monoline ink components and rebuild them as strokes.

    ink_mask: bool mask of "ink" pixels (visible foreground, background
              excluded). A whole antialiased line — core plus fringe — forms
              ONE component here, which is what makes thin AA lines
              recoverable at all.
    den:      float32 RGB for color sampling (unfiltered image)
    palette:  uint8 [K,3] palette colors (for optional color snapping)
    bg_color: the color the ink sits on; the stroke color is sampled from
              the pixels farthest from it (the line's true core color,
              uncontaminated by antialiasing blends).

    Returns ``(strokes, stroke_mask, deferred_mask)``.  Deferred pixels are
    deliberately left for the fill/gradient tracer, but must be excluded from
    the later per-palette stroke pass.  This prevents a rejected multicolour
    ring or glyph from being resurrected as disconnected colour fragments.
    """
    H, W = ink_mask.shape
    def record_defer(reason, component):
        if audit is None:
            return
        counts = audit.setdefault('deferred_component_reasons', {})
        counts[reason] = counts.get(reason, 0) + 1
        examples = audit.setdefault('deferred_component_examples', [])
        if len(examples) < 32:
            examples.append({'reason': reason, 'pixels': int(len(component)),
                'bbox': [int(component[:, 1].min()), int(component[:, 0].min()),
                         int(component[:, 1].max()) + 1, int(component[:, 0].max()) + 1]})
    # A 24 px uniform stroke at 384 px is the same design as an 8 px
    # stroke at 128 px. A fixed 13 px half-width cap changed its editability
    # with input resolution. Scale only the search bound; all width variation,
    # length/width, junction and multicolour guards remain in force.
    max_half_width = min(MAX_SCALED_HALF_WIDTH,
                         max(MAX_HALF_WIDTH, int(math.ceil(min(H, W) / 16.0))))
    stroke_mask = np.zeros((H, W), dtype=bool)
    deferred_mask = np.zeros((H, W), dtype=bool)
    strokes = []
    aa_fit_attempts = int(audit.get('native_cap_search_attempts',0)) if audit is not None else 0
    if not ink_mask.any():
        return strokes, stroke_mask, deferred_mask
    bg = np.asarray(bg_color, dtype=np.float32)

    labels, n = connected_components(ink_mask)

    # Do not rescan the whole label image once per component.  Halftone logo
    # art can contain hundreds of dots; ``argwhere(labels == li)`` made that
    # common case O(component_count * H * W) and turned one conversion into a
    # 30+ minute job.  Group eligible foreground pixels once by their dense
    # component label.  The stable sort preserves the exact row-major order
    # returned by np.argwhere, so all downstream geometry remains unchanged.
    areas, eligible, grouped_flat, grouped_starts, grouped_ends = (
        _group_eligible_component_pixels(labels, n, 24, 0.35 * H * W)
    )

    for li in np.flatnonzero(eligible):
        start = grouped_starts[li - 1]
        end = grouped_ends[li - 1]
        comp_flat = grouped_flat[start:end]
        yy, xx = np.divmod(comp_flat, W)
        comp_idx = np.column_stack((yy, xx))
        area = int(areas[li])
        y0, x0 = comp_idx.min(0)
        y1, x1 = comp_idx.max(0)
        bh, bw = y1 - y0 + 1, x1 - x0 + 1
        if max(bh, bw) < MIN_STROKE_LEN:
            continue
        comp = np.zeros((bh + 4, bw + 4), dtype=bool)
        comp[comp_idx[:, 0] - y0 + 2, comp_idx[:, 1] - x0 + 2] = True

        dt = dist_transform_capped(comp, cap=max_half_width + 2)
        max_half = float(dt.max())
        if max_half > max_half_width:
            continue                       # too fat: a shape, not a line

        # Try thinning the raw ribbon first; if the skeleton topology is not
        # a simple line (Zhang-Suen leaves phantom junctions on wide ribbons,
        # especially closed rings), retry on the ridge band of the distance
        # field, which is already 1-2 px wide and thins cleanly.
        got = None
        real_junction = False
        junction_edges = None
        for attempt in ("comp", "ridge"):
            if attempt == "comp":
                src_mask = comp
            else:
                # the capped-erosion transform is integer-valued: take the
                # top TWO levels so straight runs and corner bumps stay
                # connected as one 1-2 px band
                if max_half < 3.0:
                    break
                src_mask = dt >= (max_half - 1.0)
                if not src_mask.any():
                    break
            sk = thin(src_mask)
            if not sk.any():
                continue
            sk = _remove_staircase(sk)
            if not sk.any():
                continue
            rough_w = max(1.0, 2.0 * float(dt[sk].mean()) - 1.0)
            sk = prune_spurs(sk, max_len=max(3, int(1.8 * rough_w)))
            sk = _remove_staircase(sk)
            if not sk.any():
                continue
            if attempt == "comp" and _skeleton_has_real_junction(sk):
                junction_edges = skeleton_to_junction_edges(sk)
                # If graph decomposition is uncertain, leave the complete
                # component to the fill tracer instead of losing an arm.
                real_junction = junction_edges is None
                break
            got = skeleton_to_polyline(sk)
            if got is not None:
                break
        if real_junction or (got is None and not junction_edges):
            dm = np.zeros((H, W), dtype=bool)
            dm[comp_idx[:, 0], comp_idx[:, 1]] = True
            deferred_mask |= _dilate_one(dm)
            continue
        raw_polys = ([(edge, False) for edge in junction_edges]
                     if junction_edges else [got])

        # A simple T/X/Y has straight arms.  Bent arms are much more commonly
        # glyph outlines or illustration shapes; rebuilding them with round
        # caps changes corners and counters.  Keep those components as fills.
        if junction_edges and (
                len(junction_edges) not in (3, 4)
                or any(_polyline_straightness(edge) <
                       JUNCTION_ARM_STRAIGHTNESS_MIN
                       for edge in junction_edges)):
            dm = np.zeros((H, W), dtype=bool)
            dm[comp_idx[:, 0], comp_idx[:, 1]] = True
            deferred_mask |= _dilate_one(dm)
            continue

        def _poly_length(poly, is_closed):
            val = sum(math.hypot(poly[i + 1][0] - poly[i][0],
                                 poly[i + 1][1] - poly[i][1])
                      for i in range(len(poly) - 1))
            if is_closed:
                val += math.hypot(poly[0][0] - poly[-1][0],
                                  poly[0][1] - poly[-1][1])
            return val

        length = sum(_poly_length(poly, is_closed)
                     for poly, is_closed in raw_polys)
        if length < MIN_STROKE_LEN:
            continue

        # width: area over center-line length is robust for thin ribbons
        # (the capped erosion transform quantizes hard at 1-2 px widths)
        width = area / length if length else 0.0
        if width <= 0.5 or width > 2.2 * max_half_width:
            continue
        if length < 2.5 * width:
            continue
        # Short bent ribbons are far more often detached glyph strokes or
        # illustration fragments than intentional centre-line artwork.  Round
        # caps/joins visibly deform them.  Preserve the filled silhouette;
        # long arcs and genuinely straight dashes still qualify as strokes.
        if (not junction_edges and len(raw_polys) == 1
                and not raw_polys[0][1]
                and length < SHORT_CURVE_MIN_LENGTH_WIDTH_RATIO * width
                and _polyline_straightness(raw_polys[0][0])
                    < SHORT_CURVE_STRAIGHTNESS_MIN):
            dm = np.zeros((H, W), dtype=bool)
            dm[comp_idx[:, 0], comp_idx[:, 1]] = True
            deferred_mask |= _dilate_one(dm)
            continue
        # uniformity along the skeleton
        dvals = dt[sk]
        cv = float(dvals.std()) / float(dvals.mean()) if dvals.mean() else 9.9
        if cv > 0.35:
            continue

        def _seg_color(seg_pts):
            ys2 = np.clip(np.round([p[1] - 0.5 for p in seg_pts]).astype(int), 0, H - 1)
            xs2 = np.clip(np.round([p[0] - 0.5 for p in seg_pts]).astype(int), 0, W - 1)
            samples = den[ys2, xs2]
            dist_bg = ((samples - bg) ** 2).sum(1)
            order = np.argsort(dist_bg)
            core = samples[order[int(0.5 * len(order)):]]
            col = np.median(core if len(core) else samples, axis=0)
            dists = ((palette.astype(np.float32) - col) ** 2).sum(1)
            j = int(dists.argmin())
            if dists[j] <= 60 ** 2:
                col = palette[j].astype(np.float32)
            return tuple(int(round(v)) for v in col)

        def _seg_opacity(seg_pts):
            if alpha is None:
                return 1.0
            ys2 = np.clip(np.round([p[1] - 0.5 for p in seg_pts]).astype(int), 0, H - 1)
            xs2 = np.clip(np.round([p[0] - 0.5 for p in seg_pts]).astype(int), 0, W - 1)
            vals = np.asarray(alpha, dtype=np.float32)[ys2, xs2]
            if not len(vals):
                return 1.0
            # Upper-half median ignores transparent antialiasing fringe but
            # preserves a genuinely uniform semi-transparent line.
            vals = np.sort(vals)
            core = vals[len(vals) // 2:]
            op = float(np.median(core if len(core) else vals)) / 255.0
            return 1.0 if op >= 0.985 else round(max(0.02, op), 3)

        # +0.5: a skeleton pixel represents the CENTER of that pixel.  A
        # junction graph becomes one editable stroke per arm, all sharing
        # the same collapsed junction coordinate.
        pieces = []
        all_gpts = []
        defer_component = False
        defer_reason = "unsupported_closed_stroke_geometry"
        for pts, closed in raw_polys:
            gpts = [(x + x0 - 2 + 0.5, y + y0 - 2 + 0.5)
                    for x, y in pts]
            all_gpts.extend(gpts)
            local_pieces = [(gpts, closed)]

            # Closed non-primitive ribbons include glyph counters and
            # irregular outline art.  A centre-line with round joins is not an
            # equivalent representation, so let the fill tracer preserve it.
            local_length = _poly_length(gpts, closed)
            if closed and _fit_closed_primitive(
                    gpts, width, local_length) is None:
                defer_component = True
                break

            # multicolor polylines (e.g. red touching blue) are split at
            # sustained color changes instead of painted one color end to end.
            if len(palette) >= 2 and len(gpts) >= 12:
                lab_pts = []
                for px_, py_ in gpts:
                    yy = min(max(int(py_ - 0.5), 0), H - 1)
                    xx = min(max(int(px_ - 0.5), 0), W - 1)
                    dpx = ((palette.astype(np.float32) - den[yy, xx]) ** 2).sum(1)
                    lab_pts.append(int(dpx.argmin()))
                lab_s = list(lab_pts)
                for ii in range(2, len(lab_pts) - 2):
                    win = lab_pts[ii - 2:ii + 3]
                    lab_s[ii] = max(set(win), key=win.count)
                runs = []
                s0 = 0
                for ii in range(1, len(lab_s) + 1):
                    if ii == len(lab_s) or lab_s[ii] != lab_s[s0]:
                        runs.append((s0, ii))
                        s0 = ii
                min_run = max(6, int(2 * width))
                big_runs = [rr for rr in runs if rr[1] - rr[0] >= min_run]
                distinct = {lab_s[rr[0]] for rr in big_runs}
                if len(big_runs) >= 2 and len(distinct) >= 2:
                    pal_f = palette.astype(np.float32)
                    far = any(((pal_f[a] - pal_f[b]) ** 2).sum()
                              > MULTICOLOR_SPLIT_DISTANCE ** 2
                              for a in distinct for b in distinct if a != b)
                    if far:
                        if (closed or junction_edges or _polyline_straightness(gpts)<.99
                                or len(runs)!=len(big_runs)
                                or len(runs)>MAX_MULTICOLOR_STRAIGHT_RUNS):
                            defer_component = True
                            defer_reason = "multicolour_caps_and_seam_not_source_validated"
                            break
                        local_pieces=[]
                        for start,end in runs:
                            segment=list(gpts[start:end])
                            if start:
                                segment.insert(0,tuple((np.asarray(gpts[start-1])+gpts[start])/2))
                            if end<len(gpts):
                                segment.append(tuple((np.asarray(gpts[end-1])+gpts[end])/2))
                            local_pieces.append((segment,False))
            pieces.extend(local_pieces)

        if defer_component:
            record_defer(defer_reason, comp_idx)
            dm = np.zeros((H, W), dtype=bool)
            dm[comp_idx[:, 0], comp_idx[:, 1]] = True
            deferred_mask |= _dilate_one(dm)
            continue

        component_strokes = []
        for seg_pts, seg_closed in pieces:
            if len(seg_pts) < 2:
                continue
            seg_length = _poly_length(seg_pts, seg_closed)
            primitive = (_fit_closed_primitive(seg_pts, width, seg_length)
                         if seg_closed else None)
            if primitive:
                d, nodes = "", primitive["nodes"]
            else:
                d, nodes = _fit_path_d(seg_pts, seg_closed, width)
            kw = dict(primitive or {})
            primitive_name = kw.pop("primitive", "")
            kw.pop("nodes", None)
            stroke = Stroke(color=_seg_color(seg_pts),
                                  width=round(max(0.45, width), 2),
                                  d=d, closed=seg_closed,
                                  length=seg_length, n_nodes=nodes, pixels=area,
                                  opacity=_seg_opacity(seg_pts),
                                  primitive=primitive_name,
                                  sample_points=list(seg_pts), **kw)
            if (not junction_edges and len(pieces)==1 and not stroke.closed and not stroke.primitive
                    and (stroke.n_nodes == 2 or _polyline_straightness(seg_pts) >= .97)):
                fitted = None
                fit_search_allowed = len(pieces) == 1 and aa_fit_attempts < 24
                if fit_search_allowed:
                    aa_fit_attempts += 1
                    if audit is not None:
                        audit['native_cap_search_attempts']=aa_fit_attempts
                    try:
                        fitted = _refine_straight_aa_stroke(stroke, den, palette, bg,
                            comp_idx, labels, li, alpha, source_rgba=source_rgba)
                        if fitted is None:
                            fitted = _refine_transparent_straight(stroke, den, palette, bg,
                                comp_idx, labels, li, alpha, source_rgba)
                        if fitted is None:
                            native_group = _refine_native_component_strokes([stroke],den,comp_idx,source_rgba)
                            fitted = native_group[0] if native_group else None
                        if fitted is None:
                            fitted = _refine_opaque_palette_straight(stroke,den,comp_idx,source_rgba)
                    except (ValueError, ArithmeticError, RuntimeError):
                        pass
                if fitted is None:
                    defer_component = True
                    defer_reason = ('native_straight_caps_not_source_validated' if fit_search_allowed
                                    else 'native_straight_cap_search_budget_exhausted')
                    break
                stroke = fitted
            component_strokes.append(stroke)

        if defer_component:
            record_defer(defer_reason, comp_idx)
            dm = np.zeros((H, W), dtype=bool)
            dm[comp_idx[:, 0], comp_idx[:, 1]] = True
            deferred_mask |= _dilate_one(dm)
            continue
        if junction_edges or len(pieces)>1:
            native_group=None
            if aa_fit_attempts<24:
                aa_fit_attempts+=1
                if audit is not None:
                    audit['native_cap_search_attempts']=aa_fit_attempts
                native_group = _refine_native_component_strokes(component_strokes,den,comp_idx,source_rgba)
            if native_group is not None:
                component_strokes = native_group
            elif len(pieces)>1 and not junction_edges:
                record_defer('multicolour_caps_and_seam_not_source_validated',comp_idx)
                dm=np.zeros((H,W),dtype=bool)
                dm[comp_idx[:,0],comp_idx[:,1]]=True
                deferred_mask|=_dilate_one(dm)
                continue
        unproven=[s for s in component_strokes if not s.closed and not s.primitive
                  and not s.source_fit.get('serialized_geometry_validated')]
        if unproven:
            proven=None
            if len(component_strokes)==1 and aa_fit_attempts<24:
                aa_fit_attempts+=1
                if audit is not None:audit['native_cap_search_attempts']=aa_fit_attempts
                proven=_validate_native_open_curve(component_strokes[0],den,comp_idx,source_rgba)
            if proven is None:
                record_defer('native_open_component_not_source_validated',comp_idx)
                dm=np.zeros((H,W),dtype=bool);dm[comp_idx[:,0],comp_idx[:,1]]=True
                deferred_mask|=_dilate_one(dm)
                continue
            component_strokes=[proven]
        strokes.extend(component_strokes)

        gm = np.zeros((H, W), dtype=bool)
        gm[comp_idx[:, 0], comp_idx[:, 1]] = True
        # grow to absorb the antialiasing fringe, but ONLY into pixels that
        # look like this stroke or its blend toward the background — an
        # unconditional grow eats neighbouring fills across 1 px gaps
        # (review P0-5)
        col_ref = np.asarray(_seg_color(all_gpts), dtype=np.float32)
        # Grow only one pixel. Two unconditional ownership hops can cross a
        # one-pixel background gap and consume a neighbouring dark fill.
        # One hop is enough to absorb the normal antialiasing fringe.
        near_stroke = (np.abs(den - col_ref).max(axis=2) <= 72)
        near_bg = (np.abs(den - bg).max(axis=2) <= 72)
        allow = near_stroke | near_bg
        for _ in range(1):
            grown = gm.copy()
            grown[1:, :] |= gm[:-1, :]
            grown[:-1, :] |= gm[1:, :]
            grown[:, 1:] |= gm[:, :-1]
            grown[:, :-1] |= gm[:, 1:]
            gm = gm | (grown & allow)
        stroke_mask |= gm

    return strokes, stroke_mask, deferred_mask
