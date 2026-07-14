# -*- coding: utf-8 -*-
"""
Raster logo/icon tracer.

輸出是純 SVG path/rect 等向量元素，不嵌入原始點陣圖。
適合 logo、icon、平面插畫；照片會被簡化成色塊，通常不建議。
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class VectorizeStats:
    width: int
    height: int
    trace_width: int
    trace_height: int
    colors: int
    paths: int
    points: int
    removed_background: bool


def _fmt(n):
    if abs(n) < 0.00001:
        n = 0
    if float(n).is_integer():
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(v) for v in rgb))


def _border_light_ratio(alpha, light, alpha_threshold):
    h, w = alpha.shape
    if h == 0 or w == 0:
        return 0.0
    border_alpha = np.concatenate([
        alpha[0, :],
        alpha[-1, :],
        alpha[:, 0],
        alpha[:, -1],
    ])
    border_light = np.concatenate([
        light[0, :],
        light[-1, :],
        light[:, 0],
        light[:, -1],
    ])
    visible = border_alpha >= alpha_threshold
    if not np.any(visible):
        return 0.0
    return float(np.count_nonzero(border_light & visible) / np.count_nonzero(visible))


def _border_alpha_ratio(alpha, alpha_threshold):
    h, w = alpha.shape
    if h == 0 or w == 0:
        return 0.0
    border_alpha = np.concatenate([
        alpha[0, :],
        alpha[-1, :],
        alpha[:, 0],
        alpha[:, -1],
    ])
    return float(np.count_nonzero(border_alpha < alpha_threshold) / border_alpha.size)


def _border_connected(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    q = deque()

    def push(y, x):
        if 0 <= y < h and 0 <= x < w and mask[y, x] and not seen[y, x]:
            seen[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        push(y - 1, x)
        push(y + 1, x)
        push(y, x - 1)
        push(y, x + 1)
    return seen


def _prepare_image(src, max_size, background, white_threshold, alpha_threshold):
    im = Image.open(src).convert("RGBA")
    orig_w, orig_h = im.size

    if max_size and max(orig_w, orig_h) > max_size:
        r = float(max_size) / max(orig_w, orig_h)
        im = im.resize((max(1, round(orig_w * r)), max(1, round(orig_h * r))), Image.LANCZOS)

    arr = np.asarray(im).copy()
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    light = (rgb.min(axis=2) >= white_threshold) & (spread <= 35)
    # 很多 AI 圖會把「透明棋盤格」烤進圖裡，角落常是亮灰/白色低飽和格子。
    # 這裡用較寬的 neutral mask 只移除連到圖片外緣的背景，不會碰到內部白字。
    neutral_background = (rgb.min(axis=2) >= min(white_threshold, 190)) & (spread <= 30)

    removed_background = False
    if background != "keep":
        should_remove = background == "transparent"
        if background == "auto":
            should_remove = (
                _border_alpha_ratio(alpha, alpha_threshold) >= 0.2
                or _border_light_ratio(alpha, neutral_background, alpha_threshold) >= 0.55
            )
        if should_remove:
            if background == "auto":
                # Only remove neutral colors that are actually represented on
                # the outer border.  The previous broad >=220 mask connected
                # a deliberate #dddddd hairline to a white canvas and erased
                # both.  Quantized border colors still recognise baked
                # white/#ccc checkerboards because both tones touch an edge.
                border_rgb = np.concatenate(
                    [rgb[0, :], rgb[-1, :], rgb[:, 0], rgb[:, -1]], axis=0)
                border_spread = border_rgb.max(1) - border_rgb.min(1)
                bn = border_rgb[(border_rgb.min(1) >= min(white_threshold, 190))
                                & (border_spread <= 30)]
                candidate = np.zeros(alpha.shape, dtype=bool)
                if len(bn):
                    q = (bn // 8).astype(np.uint8)
                    uq, cnt = np.unique(q, axis=0, return_counts=True)
                    order = np.argsort(cnt)[::-1]
                    keep = [idx for idx in order[:6]
                            if cnt[idx] >= max(2, int(0.01 * len(bn)))]
                    for idx in keep:
                        members = bn[(q == uq[idx]).all(1)].astype(np.float32)
                        center = np.median(members, axis=0)
                        candidate |= (np.abs(rgb.astype(np.float32) - center)
                                      .max(axis=2) <= 14)
                candidate &= neutral_background
            else:
                candidate = light | neutral_background
            removable = _border_connected(candidate & (alpha >= alpha_threshold))
            if np.any(removable):
                arr[removable, 3] = 0
                removed_background = True

    return Image.fromarray(arr, "RGBA"), (orig_w, orig_h), removed_background


def _quantize(im, colors):
    colors = max(2, min(256, int(colors)))
    try:
        method = Image.Quantize.FASTOCTREE
        dither = Image.Dither.NONE
    except AttributeError:
        method = 2
        dither = 0
    return im.quantize(colors=colors, method=method, dither=dither)


def _add_edge(edges, a, b):
    edges[a].append(b)


def _choose_next(cur, prev_dir, candidates):
    order = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    prev_i = order.get(prev_dir, 0)
    best_index = 0
    best_rank = 99
    # 在螢幕座標(y 向下)中，右轉優先可沿著同一色塊外緣穩定繞行。
    turn_rank = {1: 0, 0: 1, 3: 2, 2: 3}
    for i, nxt in enumerate(candidates):
        d = (nxt[0] - cur[0], nxt[1] - cur[1])
        rank = turn_rank.get((order.get(d, 0) - prev_i) % 4, 9)
        if rank < best_rank:
            best_index = i
            best_rank = rank
    return best_index


def _remove_collinear(points):
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) <= 3:
        return points
    out = []
    n = len(points)
    for i, cur in enumerate(points):
        prev = points[(i - 1) % n]
        nxt = points[(i + 1) % n]
        v1 = (cur[0] - prev[0], cur[1] - prev[1])
        v2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        if v1[0] * v2[1] == v1[1] * v2[0] and v1[0] * v2[0] + v1[1] * v2[1] >= 0:
            continue
        out.append(cur)
    return out if len(out) >= 3 else points


def _point_line_distance(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    return abs(dy * px - dx * py + bx * ay - by * ax) / ((dx * dx + dy * dy) ** 0.5)


def _rdp(points, tolerance):
    if len(points) <= 2:
        return points
    a, b = points[0], points[-1]
    max_dist = -1.0
    index = 0
    for i in range(1, len(points) - 1):
        dist = _point_line_distance(points[i], a, b)
        if dist > max_dist:
            max_dist = dist
            index = i
    if max_dist > tolerance:
        left = _rdp(points[:index + 1], tolerance)
        right = _rdp(points[index:], tolerance)
        return left[:-1] + right
    return [a, b]


def _simplify_loop(points, tolerance):
    points = _remove_collinear(points)
    if len(points) <= 3 or tolerance <= 0:
        return points
    start_i = min(range(len(points)), key=lambda i: (points[i][1], points[i][0]))
    rotated = points[start_i:] + points[:start_i] + [points[start_i]]
    simplified = _rdp(rotated, tolerance)
    if simplified and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    simplified = _remove_collinear(simplified)
    return simplified if len(simplified) >= 3 else points


def _polygon_area(points):
    if len(points) < 3:
        return 0.0
    total = 0
    for a, b in zip(points, points[1:] + points[:1]):
        total += a[0] * b[1] - b[0] * a[1]
    return abs(total) / 2.0


def _drop_close_points(points, min_dist=0.08):
    if not points:
        return points
    out = [points[0]]
    min_dist2 = min_dist * min_dist
    for p in points[1:]:
        dx = p[0] - out[-1][0]
        dy = p[1] - out[-1][1]
        if dx * dx + dy * dy >= min_dist2:
            out.append(p)
    if len(out) > 1:
        dx = out[0][0] - out[-1][0]
        dy = out[0][1] - out[-1][1]
        if dx * dx + dy * dy < min_dist2:
            out.pop()
    return out


def _mask_to_loops(mask, simplify, min_area):
    h, w = mask.shape
    edges = defaultdict(list)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        if y == 0 or not mask[y - 1, x]:
            _add_edge(edges, (x, y), (x + 1, y))
        if x == w - 1 or not mask[y, x + 1]:
            _add_edge(edges, (x + 1, y), (x + 1, y + 1))
        if y == h - 1 or not mask[y + 1, x]:
            _add_edge(edges, (x + 1, y + 1), (x, y + 1))
        if x == 0 or not mask[y, x - 1]:
            _add_edge(edges, (x, y + 1), (x, y))

    loops = []
    max_steps = sum(len(v) for v in edges.values()) + 5
    for start in list(edges.keys()):
        while edges[start]:
            cur_start = start
            nxt = edges[start].pop()
            loop = [cur_start, nxt]
            prev_dir = (nxt[0] - cur_start[0], nxt[1] - cur_start[1])
            cur = nxt
            steps = 0
            while cur != cur_start and steps < max_steps:
                candidates = edges.get(cur)
                if not candidates:
                    break
                i = _choose_next(cur, prev_dir, candidates)
                nxt = candidates.pop(i)
                loop.append(nxt)
                prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
                cur = nxt
                steps += 1

            if loop[-1] == cur_start:
                pts = _simplify_loop(loop[:-1], simplify)
                if _polygon_area(pts) >= min_area:
                    loops.append(pts)
    return loops


def _interp(level, a, b, va, vb):
    if abs(vb - va) < 1e-9:
        t = 0.5
    else:
        t = (level - va) / (vb - va)
    t = max(0.0, min(1.0, float(t)))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _segment_key(p, precision=1000):
    return (round(p[0] * precision), round(p[1] * precision))


def _segments_to_loops(segments, simplify, min_area):
    adjacency = defaultdict(list)
    for i, (a, b) in enumerate(segments):
        adjacency[_segment_key(a)].append((i, 0))
        adjacency[_segment_key(b)].append((i, 1))

    used = [False] * len(segments)
    loops = []
    for i, (a, b) in enumerate(segments):
        if used[i]:
            continue
        used[i] = True
        points = [a, b]
        start_key = _segment_key(a)
        cur_key = _segment_key(b)
        guard = 0
        while cur_key != start_key and guard < len(segments) + 5:
            candidates = [(idx, side) for idx, side in adjacency.get(cur_key, []) if not used[idx]]
            if not candidates:
                break
            idx, _side = candidates[0]
            used[idx] = True
            s_a, s_b = segments[idx]
            next_point = s_b if _segment_key(s_a) == cur_key else s_a
            points.append(next_point)
            cur_key = _segment_key(next_point)
            guard += 1

        if cur_key == start_key and len(points) >= 4:
            if _segment_key(points[-1]) == start_key:
                points = points[:-1]
            points = _drop_close_points(points)
            if len(points) < 3:
                continue
            points = _simplify_loop(points, simplify)
            points = _drop_close_points(points)
            if len(points) >= 3 and _polygon_area(points) >= min_area:
                loops.append(points)
    return loops


def _mask_to_smooth_loops(mask, simplify, min_area, smooth):
    if smooth <= 0:
        return _mask_to_loops(mask, simplify, min_area)

    h, w = mask.shape
    img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    img = img.filter(ImageFilter.GaussianBlur(float(smooth)))
    field = np.pad(np.asarray(img).astype(np.float32) / 255.0, 1, mode="constant")
    level = 0.5
    segments = []

    edge_pairs = {
        1: [(3, 0)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(1, 2)],
        5: [(0, 3), (1, 2)],
        6: [(0, 2)],
        7: [(3, 2)],
        8: [(2, 3)],
        9: [(0, 2)],
        10: [(0, 1), (3, 2)],
        11: [(1, 2)],
        12: [(3, 1)],
        13: [(0, 1)],
        14: [(3, 0)],
    }

    fh, fw = field.shape
    for y in range(fh - 1):
        for x in range(fw - 1):
            v0 = field[y, x]
            v1 = field[y, x + 1]
            v2 = field[y + 1, x + 1]
            v3 = field[y + 1, x]
            case = (
                (1 if v0 >= level else 0)
                | (2 if v1 >= level else 0)
                | (4 if v2 >= level else 0)
                | (8 if v3 >= level else 0)
            )
            if case == 0 or case == 15:
                continue
            p0 = (x - 1.0, y - 1.0)
            p1 = (x, y - 1.0)
            p2 = (x, y)
            p3 = (x - 1.0, y)
            edge_points = {
                0: _interp(level, p0, p1, v0, v1),
                1: _interp(level, p1, p2, v1, v2),
                2: _interp(level, p3, p2, v3, v2),
                3: _interp(level, p0, p3, v0, v3),
            }
            for ea, eb in edge_pairs.get(case, []):
                a = edge_points[ea]
                b = edge_points[eb]
                a = (min(max(a[0], 0.0), float(w)), min(max(a[1], 0.0), float(h)))
                b = (min(max(b[0], 0.0), float(w)), min(max(b[1], 0.0), float(h)))
                if _segment_key(a) != _segment_key(b):
                    segments.append((a, b))

    return _segments_to_loops(segments, simplify, min_area)


def _line_path(pts):
    x0, y0 = pts[0]
    parts = [f"M{_fmt(x0)} {_fmt(y0)}"]
    prev_x, prev_y = x0, y0
    for x, y in pts[1:]:
        if abs(y - prev_y) < 0.001:
            parts.append(f"H{_fmt(x)}")
        elif abs(x - prev_x) < 0.001:
            parts.append(f"V{_fmt(y)}")
        else:
            parts.append(f"L{_fmt(x)} {_fmt(y)}")
        prev_x, prev_y = x, y
    parts.append("Z")
    return parts


def _curve_path(pts, curve):
    if len(pts) < 4 or curve <= 0:
        return _line_path(pts)
    c = max(0.0, min(float(curve), 1.0)) / 6.0
    parts = [f"M{_fmt(pts[0][0])} {_fmt(pts[0][1])}"]
    n = len(pts)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = (p1[0] + (p2[0] - p0[0]) * c, p1[1] + (p2[1] - p0[1]) * c)
        c2 = (p2[0] - (p3[0] - p1[0]) * c, p2[1] - (p3[1] - p1[1]) * c)
        parts.append(
            f"C{_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} {_fmt(p2[0])} {_fmt(p2[1])}"
        )
    parts.append("Z")
    return parts


def _loops_to_path(loops, curve=0.0):
    parts = []
    point_count = 0
    for pts in loops:
        if len(pts) < 3:
            continue
        point_count += len(pts)
        parts.extend(_curve_path(pts, curve))
    return " ".join(parts), point_count


def vectorize_file(
    src,
    dst,
    colors=12,
    simplify=1.2,
    background="auto",
    white_threshold=245,
    alpha_threshold=12,
    min_area=8,
    max_size=1800,
    smooth=1.2,
    curve=0.35,
):
    src = Path(src)
    dst = Path(dst)
    im, (orig_w, orig_h), removed_background = _prepare_image(
        src, max_size, background, white_threshold, alpha_threshold
    )
    arr = np.asarray(im)
    visible = arr[:, :, 3] >= alpha_threshold
    if not np.any(visible):
        raise ValueError("圖片沒有可追蹤的可見像素")

    quant = _quantize(im, colors)
    labels = np.asarray(quant)
    palette = quant.getpalette() or []

    used = []
    for idx in np.unique(labels[visible]):
        mask = visible & (labels == idx)
        count = int(np.count_nonzero(mask))
        if count < min_area:
            continue
        base = int(idx) * 3
        if base + 2 >= len(palette):
            continue
        rgb = tuple(palette[base:base + 3])
        avg_alpha = float(np.mean(arr[:, :, 3][mask]) / 255.0)
        used.append((count, int(idx), rgb, avg_alpha))

    used.sort(reverse=True, key=lambda item: item[0])

    path_elements = []
    total_points = 0
    for count, idx, rgb, avg_alpha in used:
        mask = visible & (labels == idx)
        loops = _mask_to_smooth_loops(mask, float(simplify), int(min_area), float(smooth))
        if not loops:
            continue
        d, point_count = _loops_to_path(loops, float(curve))
        if not d:
            continue
        total_points += point_count
        opacity = "" if avg_alpha >= 0.995 else f' fill-opacity="{_fmt(avg_alpha)}"'
        path_elements.append(
            f'  <path fill="{_hex(rgb)}"{opacity} fill-rule="evenodd" d="{d}"/>'
        )

    if not path_elements:
        raise ValueError("沒有產生向量路徑；可試著降低 --vector-min-area 或增加色塊數")

    trace_w, trace_h = im.size
    title = escape(src.name)
    svg = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_w}" height="{orig_h}" '
            f'viewBox="0 0 {trace_w} {trace_h}" version="1.1" shape-rendering="geometricPrecision">'
        ),
        f"  <title>{title}</title>",
        "  <desc>Full-vector trace generated from raster artwork. No bitmap image is embedded.</desc>",
        *path_elements,
        "</svg>",
        "",
    ])
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(svg, encoding="utf-8")

    return VectorizeStats(
        width=orig_w,
        height=orig_h,
        trace_width=trace_w,
        trace_height=trace_h,
        colors=len(path_elements),
        paths=len(path_elements),
        points=total_points,
        removed_background=removed_background,
    )
