# -*- coding: utf-8 -*-
"""
AI Vector Cleanroom engine.

Pipeline: background cleanup -> automatic palette detection -> color
flattening (removes gradient/antialiasing noise) -> VTracer spline tracing
-> transforms baked into absolute coordinates -> optional geometry
regularization (perfect circles, ring/band arcs, rivet alignment,
concentric centers) -> grouping by actual stacking order (same-color runs
merge only when nothing in between overlaps, so rendering never changes).

The output SVG contains vector paths only and does not embed bitmap images.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from PIL import Image, ImageFilter

import vtracer

from trace_engine import _prepare_image


@dataclass
class CleanBaseStats:
    width: int
    height: int
    colors: int
    palette: list                       # [(group_name, hex), ...]
    removed_background: bool
    geometry_notes: list = field(default_factory=list)
    n_paths: int = 0


# ---------- Palette detection ----------

def _unique_colors(pixels):
    """Return (unique_colors float32 [U,3], weights float64 [U])."""
    q = np.clip(np.asarray(pixels).round(), 0, 255).astype(np.uint8)
    uniq, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    return uniq.astype(np.float32), counts.astype(np.float64)


def _kmeans_pp_init(uniq, w, k, rng):
    """k-means++ seeding over unique colors, starting from the dominant color."""
    idx = [int(np.argmax(w))]
    d2 = ((uniq - uniq[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        probs = w * d2
        s = probs.sum()
        if s <= 0:
            break
        nxt = int(rng.choice(len(uniq), p=probs / s))
        idx.append(nxt)
        d2 = np.minimum(d2, ((uniq - uniq[nxt]) ** 2).sum(1))
    return uniq[list(dict.fromkeys(idx))].copy()


def _kmeans(pixels, k, iters=30, seed=0):
    """Weighted k-means over unique colors with k-means++ init.

    Robust to tiny images (k is clamped to the number of unique colors) and
    to dominant-color images (weighted seeding cannot collapse clusters).
    Empty clusters are re-seeded at the point of largest weighted error.
    """
    uniq, w = _unique_colors(pixels)
    k = max(1, min(int(k), len(uniq)))
    if len(uniq) <= k:
        return uniq.copy()
    rng = np.random.default_rng(seed)
    if len(uniq) > 60000:
        keep = np.argsort(w)[-60000:]
        uniq, w = uniq[keep], w[keep]
    cent = _kmeans_pp_init(uniq, w, k, rng)
    for _ in range(iters):
        d = ((uniq[:, None] - cent[None]) ** 2).sum(2)
        lab = d.argmin(1)
        new = np.empty_like(cent)
        reseeded = False
        for i in range(len(cent)):
            m = lab == i
            if not np.any(m):
                err = w * d.min(1)
                new[i] = uniq[int(np.argmax(err))]
                reseeded = True
                continue
            wi = w[m]
            new[i] = (uniq[m] * wi[:, None]).sum(0) / wi.sum()
        if not reseeded and np.allclose(new, cent, atol=0.3):
            cent = new
            break
        cent = new
    return cent


def _cluster_shares(cent, uniq, w):
    lab = ((uniq[:, None] - cent[None]) ** 2).sum(2).argmin(1)
    shares = np.array([w[lab == i].sum() for i in range(len(cent))])
    total = shares.sum()
    return shares / total if total > 0 else shares


def _merge_close(cent, uniq, w, thresh):
    cent = cent.astype(np.float32)
    while len(cent) > 1:
        lab = ((uniq[:, None] - cent[None]) ** 2).sum(2).argmin(1)
        counts = np.array([w[lab == i].sum() for i in range(len(cent))])
        best = None
        for i in range(len(cent)):
            for j in range(i + 1, len(cent)):
                d = float(((cent[i] - cent[j]) ** 2).sum() ** 0.5)
                if d < thresh and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        wi, wj = max(counts[i], 1.0), max(counts[j], 1.0)
        merged = (cent[i] * wi + cent[j] * wj) / (wi + wj)
        cent = np.delete(cent, [i, j], axis=0)
        cent = np.vstack([cent, merged])
    return cent


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = float((ab * ab).sum())
    t = 0.0 if denom == 0 else float(np.clip(((p - a) * ab).sum() / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _prune_blend_clusters(cent, uniq, w, share_limit=0.015, seg_dist=28.0):
    """Drop tiny clusters that sit on the RGB segment between two larger
    clusters — those are antialiasing blends, not design colors. Small but
    chromatically distinct accent colors are kept."""
    cent = cent.astype(np.float32)
    while len(cent) > 2:
        shares = _cluster_shares(cent, uniq, w)
        victim = None
        for i in np.argsort(shares):
            if shares[i] >= share_limit:
                break
            for a in range(len(cent)):
                if a == i or shares[a] < 3 * shares[i]:
                    continue
                for b in range(len(cent)):
                    if b in (i, a) or shares[b] < 3 * shares[i]:
                        continue
                    if _point_segment_dist(cent[i], cent[a], cent[b]) < seg_dist:
                        victim = i
                        break
                if victim is not None:
                    break
            if victim is not None:
                break
        if victim is None:
            break
        cent = np.delete(cent, victim, axis=0)
    return cent


def detect_palette(den, visible, forced=0, max_k=8, merge_thresh=70):
    pix = den[visible].astype(np.float32)
    uniq, w = _unique_colors(pix)
    if forced and forced >= 2:
        cent = _kmeans(pix, int(forced))
    else:
        cent = _kmeans(pix, min(int(max_k), max(2, len(uniq))))
        cent = _merge_close(cent, uniq, w, merge_thresh)
        cent = _prune_blend_clusters(cent, uniq, w)
    cent_i = np.clip(np.round(cent), 0, 255).astype(np.uint8)
    H, W, _ = den.shape
    lab_all = (((den.reshape(-1, 3)[:, None].astype(np.float32) - cent_i[None]) ** 2)
               .sum(2).argmin(1).reshape(H, W))
    return cent_i, lab_all


def color_name(rgb):
    r, g, b = (int(v) for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx / 255.0
    s = 0 if mx == 0 else (mx - mn) / mx
    if v < 0.18:
        return "black"
    if s < 0.12 and v > 0.9:
        return "white"
    if s < 0.15:
        return "gray"
    if mx == r:
        h = (60 * ((g - b) / (mx - mn))) % 360
    elif mx == g:
        h = 60 * ((b - r) / (mx - mn)) + 120
    else:
        h = 60 * ((r - g) / (mx - mn)) + 240
    names = [(15, "red"), (45, "orange"), (70, "yellow"), (95, "yellow-green"),
             (160, "green"), (200, "cyan"), (255, "blue"), (290, "purple"),
             (340, "magenta"), (360, "red")]
    for lim, nm in names:
        if h < lim:
            return nm
    return "color"


# ---------- 路徑解析（含相對指令，全部轉絕對座標） ----------

_TOKEN_RE = re.compile(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7}


def _f(v):
    v = float(v)
    if abs(v) < 1e-6:
        v = 0.0
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _parse_subpaths(d):
    """把 d 解析成子路徑清單，所有座標轉為絕對值。"""
    tokens = _TOKEN_RE.findall(d)
    subs, sub = [], None
    cur = (0.0, 0.0)
    last_c2 = None
    last_q = None
    i = 0

    def finish(closed):
        nonlocal sub
        if sub and sub["segs"]:
            sub["closed"] = closed
            subs.append(sub)
        sub = None

    while i < len(tokens):
        t = tokens[i]
        if not t[0].isalpha():
            i += 1
            continue
        cmd = t
        i += 1
        up = cmd.upper()
        rel = cmd.islower()
        if up == "Z":
            if sub:
                cur = sub["start"]
            finish(True)
            last_c2 = last_q = None
            continue
        if up not in _ARITY:
            raise ValueError(f"unsupported path command: {cmd}")
        n = _ARITY[up]
        first = True
        while i + n <= len(tokens) and not tokens[i][0].isalpha():
            vals = [float(tokens[i + k]) for k in range(n)]
            i += n
            if up == "M":
                x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                if first:
                    finish(False)
                    sub = {"start": (x, y), "segs": []}
                else:
                    sub["segs"].append(["L", x, y])
                cur = (x, y)
                last_c2 = last_q = None
            elif up == "L":
                x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                sub["segs"].append(["L", x, y]); cur = (x, y)
                last_c2 = last_q = None
            elif up == "H":
                x = vals[0] + (cur[0] if rel else 0)
                sub["segs"].append(["L", x, cur[1]]); cur = (x, cur[1])
                last_c2 = last_q = None
            elif up == "V":
                y = vals[0] + (cur[1] if rel else 0)
                sub["segs"].append(["L", cur[0], y]); cur = (cur[0], y)
                last_c2 = last_q = None
            elif up == "C":
                c1x, c1y, c2x, c2y, x, y = vals
                if rel:
                    c1x += cur[0]; c1y += cur[1]; c2x += cur[0]; c2y += cur[1]
                    x += cur[0]; y += cur[1]
                sub["segs"].append(["C", c1x, c1y, c2x, c2y, x, y])
                cur = (x, y); last_c2 = (c2x, c2y); last_q = None
            elif up == "S":
                c2x, c2y, x, y = vals
                if rel:
                    c2x += cur[0]; c2y += cur[1]; x += cur[0]; y += cur[1]
                c1 = (2 * cur[0] - last_c2[0], 2 * cur[1] - last_c2[1]) if last_c2 else cur
                sub["segs"].append(["C", c1[0], c1[1], c2x, c2y, x, y])
                cur = (x, y); last_c2 = (c2x, c2y); last_q = None
            elif up == "Q":
                qx, qy, x, y = vals
                if rel:
                    qx += cur[0]; qy += cur[1]; x += cur[0]; y += cur[1]
                sub["segs"].append(["Q", qx, qy, x, y])
                cur = (x, y); last_q = (qx, qy); last_c2 = None
            elif up == "T":
                x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                q = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) if last_q else cur
                sub["segs"].append(["Q", q[0], q[1], x, y])
                cur = (x, y); last_q = q; last_c2 = None
            elif up == "A":
                rx, ry, rot, laf, sf, x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                sub["segs"].append(["A", rx, ry, rot, laf, sf, x, y])
                cur = (x, y); last_c2 = last_q = None
            first = False
    finish(False)
    return subs


def _offset_subs(subs, tx, ty):
    if tx == 0 and ty == 0:
        return
    for sub in subs:
        sub["start"] = (sub["start"][0] + tx, sub["start"][1] + ty)
        for s in sub["segs"]:
            if s[0] == "L":
                s[1] += tx; s[2] += ty
            elif s[0] == "C":
                s[1] += tx; s[2] += ty; s[3] += tx; s[4] += ty; s[5] += tx; s[6] += ty
            elif s[0] == "Q":
                s[1] += tx; s[2] += ty; s[3] += tx; s[4] += ty
            elif s[0] == "A":
                s[6] += tx; s[7] += ty


def _emit_sub(sub):
    parts = [f"M{_f(sub['start'][0])} {_f(sub['start'][1])}"]
    for s in sub["segs"]:
        c = s[0]
        if c == "L":
            parts.append(f"L{_f(s[1])} {_f(s[2])}")
        elif c == "C":
            parts.append("C" + " ".join(_f(v) for v in s[1:]))
        elif c == "Q":
            parts.append("Q" + " ".join(_f(v) for v in s[1:]))
        elif c == "A":
            parts.append(f"A{_f(s[1])} {_f(s[2])} {_f(s[3])} {int(s[4])} {int(s[5])} "
                         f"{_f(s[6])} {_f(s[7])}")
    if sub.get("closed", True):
        parts.append("Z")
    return " ".join(parts)


def _sub_bbox_accumulate(sub, xs, ys):
    xs.append(sub["start"][0]); ys.append(sub["start"][1])
    for s in sub["segs"]:
        if s[0] == "L":
            xs.append(s[1]); ys.append(s[2])
        elif s[0] == "C":
            xs += [s[1], s[3], s[5]]; ys += [s[2], s[4], s[6]]
        elif s[0] == "Q":
            xs += [s[1], s[3]]; ys += [s[2], s[4]]
        elif s[0] == "A":
            # 圓弧保守外框：端點 ± 2r 必包住整段弧
            xs += [s[6] - 2 * s[1], s[6] + 2 * s[1]]
            ys += [s[7] - 2 * s[2], s[7] + 2 * s[2]]


# ---------- 幾何規則化 ----------

def _anchors_of(sub):
    pts = []
    for s in sub["segs"]:
        if s[0] == "L":
            pts.append((s[1], s[2]))
        elif s[0] == "C":
            pts.append((s[5], s[6]))
        elif s[0] == "Q":
            pts.append((s[3], s[4]))
        elif s[0] == "A":
            pts.append((s[6], s[7]))
    return pts


def _samples_of(sub):
    pts = []
    prev = sub["start"]
    for s in sub["segs"]:
        if s[0] == "C":
            mx = (prev[0] + 3 * s[1] + 3 * s[3] + s[5]) / 8.0
            my = (prev[1] + 3 * s[2] + 3 * s[4] + s[6]) / 8.0
            pts.append((mx, my))
            prev = (s[5], s[6])
        elif s[0] == "Q":
            mx = (prev[0] + 2 * s[1] + s[3]) / 4.0
            my = (prev[1] + 2 * s[2] + s[4]) / 4.0
            pts.append((mx, my))
            prev = (s[3], s[4])
        elif s[0] == "L":
            pts.append(((prev[0] + s[1]) / 2.0, (prev[1] + s[2]) / 2.0))
            prev = (s[1], s[2])
        elif s[0] == "A":
            prev = (s[6], s[7])
        pts.append(prev)
    return pts


def _kasa(pts):
    A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy = float(sol[0]), float(sol[1])
    r2 = float(sol[2]) + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _fit_circle(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 4:
        return None
    fit = _kasa(pts)
    for _ in range(3):
        if not fit:
            return None
        cx, cy, r = fit
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        tol = max(0.025 * r, 1.2)
        inl = np.abs(d - r) <= tol
        if inl.sum() < max(4, int(0.3 * len(pts))):
            return None
        fit = _kasa(pts[inl])
    return fit


def _ang(p, c):
    return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _coverage_ok(pts, cx, cy, max_gap=75.0):
    angs = sorted(_ang(p, (cx, cy)) for p in pts)
    gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)]
    gaps.append(360.0 - (angs[-1] - angs[0]))
    return max(gaps) <= max_gap


def _snap_to_circle(p, cx, cy, r):
    dx, dy = p[0] - cx, p[1] - cy
    d = math.hypot(dx, dy) or 1.0
    return (cx + dx / d * r, cy + dy / d * r)


def _make_circle_sub(cx, cy, r):
    return {"start": (cx + r, cy),
            "segs": [["A", r, r, 0, 1, 1, cx - r, cy],
                     ["A", r, r, 0, 1, 1, cx + r, cy]],
            "closed": True}


def _try_full_circle(sub, anchors, samples):
    fit = _fit_circle(samples)
    if not fit:
        return None
    cx, cy, r = fit
    if r < 3.0:
        return None
    a = np.asarray(anchors, dtype=np.float64)
    xs, ys = a[:, 0], a[:, 1]
    w = xs.max() - xs.min(); h = ys.max() - ys.min()
    if not (xs.min() - 0.2 * w <= cx <= xs.max() + 0.2 * w
            and ys.min() - 0.2 * h <= cy <= ys.max() + 0.2 * h):
        return None
    d = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
    tol = max(0.02 * r, 0.9)
    maxdev = max(0.03 * r, 1.5)
    if np.all(np.abs(d - r) <= tol * 1.6) and np.abs(d - r).max() <= maxdev \
            and _coverage_ok(samples, cx, cy):
        return (cx, cy, r)
    return None


def _distance_clusters(anchors, cx, cy):
    """依「到中心距離」分群，找出環帶外緣/內緣的半徑候選。"""
    d = sorted(math.hypot(p[0] - cx, p[1] - cy) for p in anchors)
    clusters, cur = [], [d[0]]
    for v in d[1:]:
        if v - cur[-1] <= max(2.5, 0.008 * v):
            cur.append(v)
        else:
            clusters.append(cur); cur = [v]
    clusters.append(cur)
    out = []
    for c in clusters:
        if len(c) >= 8:
            med = c[len(c) // 2]
            if med >= 20 and (c[-1] - c[0]) <= max(0.035 * med, 4.0):
                out.append(med)
    return out


def _arc_replace(sub, anchors, candidates):
    """把貼合候選圓的連續節點段換成數學圓弧；凸出的設計細節原樣保留。"""
    n = len(anchors)
    labels = np.full(n, -1, dtype=int)
    for ci, (cx, cy, r) in enumerate(candidates):
        tol = max(0.02 * r, 1.8)
        for i, p in enumerate(anchors):
            if labels[i] == -1 and abs(math.hypot(p[0] - cx, p[1] - cy) - r) <= tol:
                labels[i] = ci
    if (labels == -1).all():
        return 0

    def _monotonic(idx_from, idx_to, cx, cy):
        """節點沿圓的行進方向是否單調（細長筆畫會去了又回，必須排除）。"""
        angs = [_ang(anchors[i], (cx, cy)) for i in range(idx_from, idx_to + 1)]
        pos = neg = 0.0
        for i in range(1, len(angs)):
            dd = _wrap(angs[i] - angs[i - 1])
            if dd >= 0:
                pos += dd
            else:
                neg -= dd
        return min(pos, neg) <= 3.0

    # 全部貼合同一顆圓 → 換成正圓。但必須：繞滿一圈、方向單調、
    # 圓心落在形狀外框內（細長弧形筆畫三者都不會通過）。
    if (labels != -1).all() and len(set(labels.tolist())) == 1:
        cx, cy, r = candidates[int(labels[0])]
        xs = [p[0] for p in anchors]; ys = [p[1] for p in anchors]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        center_inside = (min(xs) - 0.2 * w <= cx <= max(xs) + 0.2 * w
                         and min(ys) - 0.2 * h <= cy <= max(ys) + 0.2 * h)
        if center_inside and _monotonic(0, n - 1, cx, cy) \
                and _coverage_ok(anchors, cx, cy):
            new = _make_circle_sub(cx, cy, r)
            sub["start"] = new["start"]; sub["segs"] = new["segs"]; sub["closed"] = True
            return 1
        return 0   # 細長弧形筆畫：保持原樣（vtracer 的曲線已足夠平滑）

    # 旋轉起點，避免弧段跨過路徑起點
    if (labels == -1).any():
        k = (int(np.where(labels == -1)[0][0]) + 1) % n
    else:
        ks = [i for i in range(n) if labels[i] != labels[i - 1]]
        k = ks[0] if ks else 0
    segs = sub["segs"][k:] + sub["segs"][:k]
    anchors = anchors[k:] + anchors[:k]
    labels = np.roll(labels, -k)
    start = anchors[-1]

    runs = []
    s = None
    for i in range(n):
        if labels[i] != -1 and s is None:
            s = i
        elif s is not None and (labels[i] == -1 or labels[i] != labels[s]):
            runs.append((s, i - 1))
            s = i if labels[i] != -1 else None
    if s is not None:
        runs.append((s, n - 1))

    run_map = {rs: re_ for rs, re_ in runs}
    new_segs = []
    replaced = 0
    idx = 0
    while idx < n:
        if idx in run_map:
            re_ = run_map[idx]
            count = re_ - idx + 1
            ci = int(labels[idx])
            cx, cy, r = candidates[ci]
            angs = [_ang(anchors[i], (cx, cy)) for i in range(idx, re_ + 1)]
            span = 0.0
            for i in range(1, len(angs)):
                span += _wrap(angs[i] - angs[i - 1])
            if count >= 6 and abs(span) >= 25.0 and _monotonic(idx, re_, cx, cy):
                # 弧只涵蓋「落在圓上」的節點區間：anchors[idx] → anchors[re_]。
                # segs[idx]（從前一節點進入圓的過渡段）原樣保留，
                # 只把它的終點貼到圓上，弧從那裡接手。
                p_first = _snap_to_circle(anchors[idx], cx, cy, r)
                p_end = _snap_to_circle(anchors[re_], cx, cy, r)
                lead = list(segs[idx])
                if lead[0] == "L":
                    lead[1], lead[2] = p_first
                elif lead[0] == "C":
                    lead[5], lead[6] = p_first
                elif lead[0] == "Q":
                    lead[3], lead[4] = p_first
                elif lead[0] == "A":
                    lead[6], lead[7] = p_first
                new_segs.append(lead)
                laf = 1 if abs(span) > 180.0 else 0
                sf = 1 if span > 0 else 0
                new_segs.append(["A", r, r, 0, laf, sf, p_end[0], p_end[1]])
                replaced += 1
                idx = re_ + 1
                continue
            else:
                for i in range(idx, re_ + 1):
                    new_segs.append(segs[i])
                idx = re_ + 1
                continue
        new_segs.append(segs[idx])
        idx += 1

    if replaced:
        sub["start"] = start
        sub["segs"] = new_segs
    return replaced


def _uniform_fit(angles):
    k = len(angles)
    step = 360.0 / k
    res = [a - i * step for i, a in enumerate(angles)]
    s = sum(math.sin(math.radians(x)) for x in res)
    c = sum(math.cos(math.radians(x)) for x in res)
    phase = math.degrees(math.atan2(s, c))
    devs = [_wrap(a - (phase + i * step)) for i, a in enumerate(angles)]
    snapped = [phase + i * step for i in range(k)]
    return snapped, max(abs(v) for v in devs)


def _regularize(all_paths, level="normal"):
    """Geometry regularization.

    level:
      "conservative" — snap near-circles to perfect circles, align concentric
                       rings, unify rivet size/orbit/spacing. Ring-band edge
                       arc replacement is skipped (the most aggressive step).
      "normal"       — everything above plus ring/band edge arc replacement.
    """
    notes = []
    circles = []
    subs_todo = []

    for entry in all_paths:
        for sub in entry.get("subs", []):
            if "segs" not in sub or not sub.get("closed", False):
                continue
            anchors = _anchors_of(sub)
            if len(anchors) < 4:
                continue
            fit = _try_full_circle(sub, anchors, _samples_of(sub))
            if fit:
                circles.append({"sub": sub, "cx": fit[0], "cy": fit[1],
                                "r": fit[2], "color": entry["color"]})
            else:
                subs_todo.append((sub, anchors))

    main = None
    mx = my = rmax = 0.0
    if circles:
        biggest = max(circles, key=lambda c: c["r"])
        main = biggest
        mx, my, rmax = biggest["cx"], biggest["cy"], biggest["r"]

    # Ring/band edge arc replacement (normal level only; protruding design
    # details are preserved, but this is the most aggressive transform).
    arc_count = 0
    if level == "normal":
        for sub, anchors in subs_todo:
            if len(anchors) < 10:
                continue
            candidates = []
            fit = _fit_circle(_samples_of(sub))
            if fit and fit[2] >= 20:
                a = np.asarray(anchors)
                d = np.hypot(a[:, 0] - fit[0], a[:, 1] - fit[1])
                if (np.abs(d - fit[2]) <= np.maximum(0.02 * fit[2], 1.8)).mean() >= 0.35:
                    candidates.append(fit)
            if not candidates and main is not None:
                for r in _distance_clusters(anchors, mx, my):
                    candidates.append((mx, my, r))
            if candidates:
                try:
                    arc_count += _arc_replace(sub, anchors, candidates)
                except Exception:
                    pass
    if arc_count:
        notes.append(f"{arc_count} ring/band edge segments regularized into "
                     "mathematical circular arcs (protruding details preserved)")

    if not circles:
        return notes

    # 大圓同心
    big = [c for c in circles if c["r"] >= 0.35 * rmax]
    if len(big) >= 2:
        cxs = [c["cx"] for c in big]; cys = [c["cy"] for c in big]
        spread = max(max(cxs) - min(cxs), max(cys) - min(cys))
        if spread <= max(0.02 * rmax, 3.0):
            ux, uy = sum(cxs) / len(big), sum(cys) / len(big)
            for c in big:
                c["cx"], c["cy"] = ux, uy
            mx, my = ux, uy
            notes.append(f"{len(big)} large circles aligned to a shared concentric center")

    # 鉚釘：小圓、繞著主中心的軌道上
    rivets = []
    for c in circles:
        if c["r"] <= 0.18 * rmax:
            orbit = math.hypot(c["cx"] - mx, c["cy"] - my)
            if 0.25 * rmax <= orbit <= 1.05 * rmax:
                c["orbit"] = orbit
                c["ang"] = _ang((c["cx"], c["cy"]), (mx, my))
                rivets.append(c)

    if len(rivets) >= 5:
        rivets.sort(key=lambda c: c["ang"])
        clusters = [[rivets[0]]]
        for c in rivets[1:]:
            if c["ang"] - clusters[-1][-1]["ang"] <= 6.0:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        if len(clusters) >= 2 and (clusters[0][0]["ang"] + 360.0 - clusters[-1][-1]["ang"]) <= 6.0:
            clusters[0] = clusters.pop() + clusters[0]

        k = len(clusters)
        if k >= 5:
            cl_ang = [sum(c["ang"] for c in cl) / len(cl) for cl in clusters]
            order = sorted(range(k), key=lambda i: cl_ang[i])
            cl_ang_sorted = [cl_ang[i] for i in order]
            snapped, max_dev = _uniform_fit(cl_ang_sorted)
            aligned = max_dev <= 4.0

            orbits = [c["orbit"] for c in rivets]
            om = sum(orbits) / len(orbits)
            orbit_uni = (max(orbits) - min(orbits)) / om <= 0.06 if om else False

            by_color = defaultdict(list)
            for c in rivets:
                by_color[c["color"]].append(c)
            r_notes = 0
            for cc in by_color.values():
                rs = [c["r"] for c in cc]
                rm = sum(rs) / len(rs)
                if rm > 0 and (max(rs) - min(rs)) / rm <= 0.18:
                    for c in cc:
                        c["r"] = rm
                    r_notes += len(cc)

            if aligned or orbit_uni:
                for pos, ci in enumerate(order):
                    ang = snapped[pos] if aligned else cl_ang_sorted[pos]
                    for c in clusters[ci]:
                        o = om if orbit_uni else c["orbit"]
                        rad = math.radians(ang)
                        c["cx"] = mx + o * math.cos(rad)
                        c["cy"] = my + o * math.sin(rad)
            msg = []
            if aligned:
                msg.append("equal angular spacing")
            if orbit_uni:
                msg.append("unified orbit radius")
            if r_notes:
                msg.append("unified size")
            if msg:
                notes.append(f"{k} rivet-like dots: " + ", ".join(msg))

    notes.append(f"{len(circles)} near-circular shapes replaced with perfect circles")

    for c in circles:
        new = _make_circle_sub(c["cx"], c["cy"], c["r"])
        c["sub"]["start"] = new["start"]
        c["sub"]["segs"] = new["segs"]
        c["sub"]["closed"] = True

    return notes


# ---------- 舊版位移烘焙（解析失敗時的備援） ----------

_PAIR_CMDS = set("MLCSQT")


def _bake_translate(d, tx, ty):
    tokens = re.findall(r"[A-Za-z]|-?\d+\.?\d*(?:e-?\d+)?", d)
    result = []
    k = 0
    while k < len(tokens):
        t = tokens[k]
        if t.isalpha():
            cmd = t
            result.append(cmd)
            k += 1
            if cmd in "Zz":
                continue
            nums = []
            while k < len(tokens) and not tokens[k].isalpha():
                nums.append(float(tokens[k]))
                k += 1
            up = cmd.upper()
            if up in _PAIR_CMDS:
                for n in range(0, len(nums), 2):
                    nums[n] += tx
                    if n + 1 < len(nums):
                        nums[n + 1] += ty
            elif up == "H":
                nums = [v + tx for v in nums]
            elif up == "V":
                nums = [v + ty for v in nums]
            result.append(" ".join(_f(v) for v in nums))
        else:
            k += 1
    return " ".join(result)


# ---------- Main pipeline ----------

def _iter_svg_paths(raw):
    """Yield (d, fill, tx, ty) for every <path> in document order.

    Prefers a real XML parser; falls back to a regex scan if the document
    cannot be parsed (so a vtracer output format change degrades gracefully
    instead of silently producing an empty result).
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        found = False
        for el in root.iter():
            if not el.tag.rsplit("}", 1)[-1] == "path":
                continue
            d = el.get("d")
            fill = el.get("fill", "")
            if not d or not re.fullmatch(r"#[0-9a-fA-F]{6}", fill):
                continue
            tm = re.search(r"translate\(\s*([-\d.]+)[,\s]+([-\d.]+)\s*\)",
                           el.get("transform", "") or "")
            tx, ty = (float(tm.group(1)), float(tm.group(2))) if tm else (0.0, 0.0)
            found = True
            yield d, fill, tx, ty
        if found:
            return
    except Exception:
        pass
    for m in re.finditer(r"<path\b[^>]*?/>", raw):
        tag = m.group(0)
        dm = re.search(r'\bd="([^"]+)"', tag)
        fm = re.search(r'\bfill="(#[0-9a-fA-F]{6})"', tag)
        if not (dm and fm):
            continue
        tm = re.search(r'transform="translate\(([-\d.]+),([-\d.]+)\)"', tag)
        tx, ty = (float(tm.group(1)), float(tm.group(2))) if tm else (0.0, 0.0)
        yield dm.group(1), fm.group(1), tx, ty


def build_clean_base(src, dst, forced_colors=0, white_threshold=220,
                     regularize=True, flat_out=None,
                     background="auto", max_size=0, geometry=None):
    """Convert a bitmap into a grouped, optionally regularized SVG.

    background: "auto" (heuristic removal of light border-connected
                background), "keep" (never remove), "transparent" (force
                removal of light border-connected regions).
    max_size:   downscale the longest image side before tracing (0 = off).
                The SVG width/height still reflect the original size.
    geometry:   "off" | "conservative" | "normal". If None, falls back to
                the legacy boolean `regularize` (True -> "normal").
    """
    src, dst = Path(src), Path(dst)
    if geometry is None:
        geometry = "normal" if regularize else "off"
    if geometry not in ("off", "conservative", "normal"):
        raise ValueError(f"invalid geometry level: {geometry!r}")

    im, (orig_w, orig_h), removed = _prepare_image(
        src, max_size=max_size, background=background,
        white_threshold=white_threshold, alpha_threshold=12,
    )
    A = np.asarray(im)
    visible = A[:, :, 3] >= 128
    H, W = visible.shape
    if not np.any(visible):
        raise ValueError("image has no traceable visible pixels")

    rgb_img = Image.fromarray(A[:, :, :3])
    if min(W, H) >= 3:
        rgb_img = rgb_img.filter(ImageFilter.MedianFilter(3))
    den = np.asarray(rgb_img).astype(np.float32)
    palette, lab_all = detect_palette(den, visible, forced=forced_colors)
    flat = palette[lab_all]
    flat_rgba = np.dstack([flat, np.where(visible, 255, 0).astype(np.uint8)])
    if flat_out:
        Image.fromarray(flat_rgba, "RGBA").save(flat_out)

    with tempfile.TemporaryDirectory() as td:
        flat_png = Path(td) / "flat.png"
        raw_svg = Path(td) / "raw.svg"
        Image.fromarray(flat_rgba, "RGBA").save(flat_png)
        vtracer.convert_image_to_svg_py(
            str(flat_png), str(raw_svg),
            colormode="color", hierarchical="stacked", mode="spline",
            filter_speckle=10, color_precision=8, layer_difference=0,
            corner_threshold=58, length_threshold=5.0, splice_threshold=45,
            path_precision=6,
        )
        raw = raw_svg.read_text(encoding="utf-8")

    pal_rgb = [tuple(int(v) for v in c) for c in palette]
    pal_hex = ["#{:02x}{:02x}{:02x}".format(*c) for c in pal_rgb]

    def nearest_idx(hx):
        r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
        return min(range(len(pal_rgb)),
                   key=lambda i: (pal_rgb[i][0] - r) ** 2 + (pal_rgb[i][1] - g) ** 2
                                 + (pal_rgb[i][2] - b) ** 2)

    # Parse every path, preserving vtracer's stacking order (bottom to top).
    entries = []
    for d, fill, tx, ty in _iter_svg_paths(raw):
        ci = nearest_idx(fill.lower())
        try:
            subs = _parse_subpaths(d)
            _offset_subs(subs, tx, ty)
            entries.append({"color": ci, "subs": subs})
        except Exception:
            entries.append({"color": ci, "raw": _bake_translate(d, tx, ty)})

    geometry_notes = []
    if geometry != "off":
        try:
            geometry_notes = _regularize(entries, level=geometry)
        except Exception:
            geometry_notes = []

    # 幾何規則化後再算外框（座標可能被調整過）
    for e in entries:
        if "subs" in e:
            xs, ys = [], []
            for sub in e["subs"]:
                _sub_bbox_accumulate(sub, xs, ys)
            e["bbox"] = (min(xs), min(ys), max(xs), max(ys))
        else:
            e["bbox"] = None   # 無法判斷 → 視為與所有東西重疊

    # 依堆疊順序切成同色連續段
    runs = []
    for e in entries:
        if runs and runs[-1]["color"] == e["color"]:
            runs[-1]["items"].append(e)
        else:
            runs.append({"color": e["color"], "items": [e]})

    def _overlap(ra, rb):
        for x in ra["items"]:
            for y in rb["items"]:
                bx, by = x["bbox"], y["bbox"]
                if bx is None or by is None:
                    return True
                if not (bx[2] < by[0] or by[2] < bx[0] or bx[3] < by[1] or by[3] < bx[1]):
                    return True
        return False

    # 同色段不與中間形狀重疊時「下沉」合併（減少群組數，且渲染不變）。
    # 每一輪：把每個 run 盡量往下沉，碰到同色就合併；j 單調遞減、i 單調前進，
    # 保證終止（先前的互相讓位寫法會無限交換）。
    for _pass in range(4):
        merged_any = False
        i = 1
        while i < len(runs):
            j = i
            while (j > 0
                   and runs[j]["color"] != runs[j - 1]["color"]
                   and any(r2["color"] == runs[j]["color"] for r2 in runs[:j - 1])
                   and not _overlap(runs[j], runs[j - 1])):
                runs[j - 1], runs[j] = runs[j], runs[j - 1]
                j -= 1
            if j > 0 and runs[j]["color"] == runs[j - 1]["color"]:
                runs[j - 1]["items"] += runs[j]["items"]
                runs.pop(j)
                merged_any = True
                continue    # i 指到的內容已變，原地重試
            i += 1
        if not merged_any:
            break

    used_names = {}
    title = xml_escape(f"{src.stem} vector cleanroom result")
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
         f'width="{orig_w}" height="{orig_h}" viewBox="0 0 {W} {H}" version="1.1" '
         f'shape-rendering="geometricPrecision">'),
        f"  <title>{title}</title>",
        "  <desc>Editable vector approximation grouped by stack order and color. "
        "Group contents remain separate editable paths. No bitmap embedded.</desc>",
    ]
    final_palette = []
    n_paths = 0
    for run in runs:
        ci = run["color"]
        base = color_name(pal_rgb[ci])
        used_names[base] = used_names.get(base, 0) + 1
        nm = base if used_names[base] == 1 else f"{base}{used_names[base]}"
        nm = xml_escape(nm)
        parts.append(f'  <g id="{nm}" inkscape:label="{nm}" fill="{pal_hex[ci]}" '
                     f'fill-rule="evenodd">')
        for e in run["items"]:
            if "raw" in e:
                d = e["raw"]
            else:
                d = " ".join(_emit_sub(s) for s in e["subs"])
            parts.append(f'    <path d="{d}"/>')
            n_paths += 1
        parts.append("  </g>")
        final_palette.append((nm, pal_hex[ci]))
    parts += ["</svg>", ""]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(parts), encoding="utf-8")

    return CleanBaseStats(
        width=orig_w, height=orig_h, colors=len(final_palette),
        palette=final_palette, removed_background=removed,
        geometry_notes=geometry_notes, n_paths=n_paths,
    )
