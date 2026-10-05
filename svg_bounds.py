"""Conservative SVG paint bounds for selection, navigation and handoff guides.

Bounds are not a certificate of visible area or independent object ownership.
SVG transform lists use post-multiplied affine matrices; ancestor transforms
compose outside child transforms. Filters, markers and non-scaling strokes
remain unknown instead of receiving an underestimated box.
https://www.w3.org/TR/SVG2/coords.html#BoundingBoxes
"""
import math
import re

NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
IDENTITY = (1., 0., 0., 1., 0., 0.)


def multiply(a, b):
    return (a[0]*b[0]+a[2]*b[1], a[1]*b[0]+a[3]*b[1],
            a[0]*b[2]+a[2]*b[3], a[1]*b[2]+a[3]*b[3],
            a[0]*b[4]+a[2]*b[5]+a[4], a[1]*b[4]+a[3]*b[5]+a[5])


def parse_transform(raw):
    if not raw or raw.strip() == "none":
        return IDENTITY
    matrix, end = IDENTITY, 0
    for match in re.finditer(r"([A-Za-z]+)\s*\(([^()]*)\)", raw):
        if raw[end:match.start()].strip(" ,\t\r\n"):
            raise ValueError("Unsupported SVG transform")
        name, text = match.group(1, 2)
        if re.sub(NUMBER, "", text).strip(" ,\t\r\n"):
            raise ValueError("Unsupported transform units")
        n = [float(value) for value in re.findall(NUMBER, text)]
        if not all(math.isfinite(value) for value in n):
            raise ValueError("Nonfinite transform")
        if name == "matrix" and len(n) == 6:
            item = tuple(n)
        elif name == "translate" and len(n) in (1, 2):
            item = (1, 0, 0, 1, n[0], n[1] if len(n) == 2 else 0)
        elif name == "scale" and len(n) in (1, 2):
            item = (n[0], 0, 0, n[1] if len(n) == 2 else n[0], 0, 0)
        elif name == "rotate" and len(n) in (1, 3):
            c, s = math.cos(math.radians(n[0])), math.sin(math.radians(n[0]))
            item = (c, s, -s, c, 0, 0)
            if len(n) == 3:
                item = multiply(multiply((1,0,0,1,n[1],n[2]), item), (1,0,0,1,-n[1],-n[2]))
        elif name in ("skewX", "skewY") and len(n) == 1:
            t = math.tan(math.radians(n[0]))
            item = (1, 0, t, 1, 0, 0) if name == "skewX" else (1, t, 0, 1, 0, 0)
        else:
            raise ValueError("Unsupported SVG transform")
        matrix = multiply(matrix, item)
        end = match.end()
    if end == 0 or raw[end:].strip(" ,\t\r\n") or not all(math.isfinite(value) for value in matrix):
        raise ValueError("Invalid SVG transform")
    return matrix


def arc_extrema(start, values):
    """Exact local endpoints/extrema of an absolute SVG elliptical arc."""
    from curve_refit_stage import _svg_arc_center_parameters
    parameters = _svg_arc_center_parameters(start, ["A", *values])
    endpoint = (values[-2], values[-1])
    if parameters is None:
        if values[0] == 0 or values[1] == 0 or tuple(start) == endpoint:
            return [start, endpoint]
        raise ValueError("Invalid elliptical arc")
    cx, cy, rx, ry, phi, theta, delta, _x, _y = parameters
    c, s = math.cos(phi), math.sin(phi)
    angles = [theta, theta + delta]
    x_extreme = math.atan2(-ry*s, rx*c)
    y_extreme = math.atan2(ry*c, rx*s)
    for angle in (x_extreme, x_extreme+math.pi, y_extreme, y_extreme+math.pi):
        distance = ((angle-theta) if delta > 0 else (theta-angle)) % (2*math.pi)
        if distance <= abs(delta)+1e-10:
            angles.append(angle)
    return [(cx+rx*c*math.cos(t)-ry*s*math.sin(t),
             cy+rx*s*math.cos(t)+ry*c*math.sin(t)) for t in angles]


def paint_bounds(element, parents, local_box, style_reader):
    if local_box is None:
        return None
    chain, current = [], element
    while current is not None:
        chain.append(current)
        current = parents.get(current)
    matrix, presentation = IDENTITY, {}
    try:
        for node in reversed(chain):
            style = style_reader(node)
            if "transform" in style:
                return None  # CSS transform-origin and transform-box need layout.
            attrs = {**node.attrib, **style}
            for key in ("filter", "marker-start", "marker-mid", "marker-end"):
                if attrs.get(key, "none").strip().lower() != "none":
                    return None
            if attrs.get("vector-effect", "none") != "none":
                return None
            matrix = multiply(matrix, parse_transform(attrs.get("transform", "")))
            for key in ("stroke", "stroke-width", "stroke-linejoin", "stroke-linecap", "stroke-miterlimit"):
                if key in attrs:
                    presentation[key] = attrs[key]
        x, y, w, h = local_box
        if presentation.get("stroke", "none").strip().lower() != "none":
            width = presentation.get("stroke-width", "1")
            if not re.fullmatch(NUMBER, width):
                return None
            half = max(0, float(width)) / 2
            join = presentation.get("stroke-linejoin", "miter")
            factor = max(1, float(presentation.get("stroke-miterlimit", "4"))) if join in {"miter", "miter-clip"} else 1
            if presentation.get("stroke-linecap") == "square":
                factor = max(factor, math.sqrt(2))
            padding = half * factor
            x, y, w, h = x-padding, y-padding, w+2*padding, h+2*padding
        a,b,c,d,e,f = matrix
        points = [(a*px+c*py+e,b*px+d*py+f) for px,py in ((x,y),(x+w,y),(x,y+h),(x+w,y+h))]
        if not all(math.isfinite(value) and abs(value) < 1e12 for point in points for value in point):
            return None
        xs, ys = zip(*points)
        return [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)]
    except (ValueError, OverflowError, TypeError):
        return None
