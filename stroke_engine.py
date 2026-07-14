# -*- coding: utf-8 -*-
"""
Monoline stroke reconstruction engine.

Detects near-uniform-width line work (heartbeat lines, field lines, frame
outlines, simple line art) in a color-labeled image and rebuilds each as a
real SVG stroke: a fitted center-line path with `fill="none"`,
`stroke-width`, and round caps/joins — instead of a high-node filled
outline pair.

Conservative by design: a component is converted only when it passes strict
uniform-width and simple-topology tests; everything else stays with the
fill tracer. Pure numpy, no OpenCV dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_HALF_WIDTH = 13          # px at trace scale; wider shapes are not strokes
MIN_STROKE_LEN = 10.0        # px; shorter skeletons are blobs, not strokes


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


def extract_strokes(ink_mask, den, palette, bg_color=(255, 255, 255),
                    alpha=None):
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

    Returns (strokes list[Stroke], stroke_mask bool array).
    """
    H, W = ink_mask.shape
    stroke_mask = np.zeros((H, W), dtype=bool)
    strokes = []
    if not ink_mask.any():
        return strokes, stroke_mask
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

        dt = dist_transform_capped(comp)
        max_half = float(dt.max())
        if max_half > MAX_HALF_WIDTH:
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
            continue
        raw_polys = ([(edge, False) for edge in junction_edges]
                     if junction_edges else [got])

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
        if width <= 0.5 or width > 2.2 * MAX_HALF_WIDTH:
            continue
        if length < 2.5 * width:
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
        for pts, closed in raw_polys:
            gpts = [(x + x0 - 2 + 0.5, y + y0 - 2 + 0.5)
                    for x, y in pts]
            all_gpts.extend(gpts)
            local_pieces = [(gpts, closed)]

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
                    far = any(((pal_f[a] - pal_f[b]) ** 2).sum() > 80 ** 2
                              for a in distinct for b in distinct if a != b)
                    if far:
                        local_pieces = [(gpts[rr[0]:rr[1]], False)
                                        for rr in big_runs
                                        if rr[1] - rr[0] >= 2]
            pieces.extend(local_pieces)

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
            strokes.append(Stroke(color=_seg_color(seg_pts),
                                  width=round(max(0.45, width), 2),
                                  d=d, closed=seg_closed,
                                  length=seg_length, n_nodes=nodes, pixels=area,
                                  opacity=_seg_opacity(seg_pts),
                                  primitive=primitive_name,
                                  sample_points=list(seg_pts), **kw))

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

    return strokes, stroke_mask
