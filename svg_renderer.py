"""Deterministic offline SVG rendering for geometry and paint validation.

Use the actual SVG paint model: gradients, clipping, transforms and opacity
are rendered by resvg rather than reconstructed from placeholder colors.
Reference: https://resvg-py.readthedocs.io/en/latest/api.html
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

MAX_SVG_BYTES = 32 * 1024 * 1024
MAX_RENDER_SIDE = 8192
MAX_RENDER_PIXELS = 32 * 1024 * 1024
_NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def renderer_info():
    try:
        import resvg_py
        return {"available": True, "renderer": "resvg", "binding_version": resvg_py.__version__,
                "engine_version": getattr(resvg_py, "__resvg_version__", "unknown"),
                "paint_model": "native_svg", "system_fonts": False}
    except ImportError:
        return {"available": False, "renderer": "resvg", "reason": "resvg-py is not installed"}


def _length(raw):
    match = re.fullmatch(rf"\s*({_NUMBER})(px|pt|pc|mm|cm|in)?\s*", str(raw or ""))
    if not match:
        return None
    value = float(match[1]) * {None: 1, "px": 1, "pt": 96 / 72, "pc": 16,
                              "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96}[match[2]]
    return value if math.isfinite(value) and value > 0 else None


def _validated_svg(payload):
    if len(payload) > MAX_SVG_BYTES or re.search(br"<!\s*(?:DOCTYPE|ENTITY)", payload, re.I):
        raise ValueError("SVG is too large or declares external entities")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("Invalid SVG XML") from exc
    if root.tag.split("}")[-1] != "svg":
        raise ValueError("Expected an SVG document")
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag in {"script", "foreignObject", "text", "tspan", "textPath"}:
            raise ValueError("Active content or unoutlined text is not supported by the deterministic renderer")
        for key, value in node.attrib.items():
            local = key.split("}")[-1].lower()
            if local.startswith("on") or local in {"base", "src"}:
                raise ValueError("Active or external SVG attributes are unsupported")
            if local == "href":
                if not value.startswith("#") and not re.fullmatch(
                        r"data:image/(?:png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=\s]+", value):
                    raise ValueError("Only local SVG resources and embedded raster references are supported")
            _validate_css_urls(value)
        if tag == "style":
            _validate_css_urls(node.text or "")
    width, height = _length(root.get("width")), _length(root.get("height"))
    if width is None or height is None:
        values = [float(value) for value in re.split(r"[\s,]+", root.get("viewBox", "").strip()) if value]
        if len(values) != 4 or not all(math.isfinite(value) for value in values) or min(values[2:]) <= 0:
            raise ValueError("SVG requires finite positive dimensions or viewBox")
        width, height = values[2:]
    return payload.decode("utf-8"), width, height


def _validate_css_urls(value):
    if any(item in value.lower() for item in ("@import", "@font-face", "javascript:", "expression(")):
        raise ValueError("External or active CSS is unsupported")
    if re.search(r"url\s*\(", value, re.I):
        matches = re.findall(r"url\(\s*['\"]?#[^\s)'\"]+['\"]?\s*\)", value, re.I)
        if len(matches) != len(re.findall(r"url\s*\(", value, re.I)):
            raise ValueError("External SVG paint resources are unsupported")
    if "\\" in value or "/*" in value:
        # No CSS escapes may turn an apparently local URL into a file URL.
        raise ValueError("Escaped SVG CSS or comments are unsupported")


def native_canvas_aspect_compatible(svg_width, svg_height, native_width, native_height):
    """Accept exact proportions or this app's integer-rounded downsize only.

    This never resizes source pixels or changes SVG's preserveAspectRatio.
    The caller still renders both explicit native dimensions and compares all
    native pixels. A rounded short side is not an unrelated source canvas.
    """
    dimensions = (svg_width, svg_height, native_width, native_height)
    if not all(math.isfinite(v) and v > 0 for v in dimensions):
        return False
    if abs(svg_width / svg_height - native_width / native_height) <= 1e-6:
        return True
    if any(float(v) != int(v) for v in dimensions):
        return False
    scale = max(svg_width, svg_height) / max(native_width, native_height)
    if scale > 1:
        return False
    return (int(svg_width), int(svg_height)) == (
        max(1, round(native_width * scale)), max(1, round(native_height * scale)))


def svg_with_native_viewport(svg, width, height):
    """Set an exact output viewport; retain the original SVG coordinate box.

    resvg's width/height output options fit within a box and may otherwise
    produce one pixel less than requested on rounded nonsquare inputs.
    Setting the SVG viewport itself keeps the default uniform meet transform
    and any letterbox padding; it does not stretch or resample source pixels.
    """
    _, old_width, old_height = _validated_svg(svg.encode('utf8'))
    root = ET.fromstring(svg)
    if 'viewBox' not in root.attrib:
        root.set('viewBox', f'0 0 {old_width:g} {old_height:g}')
    root.set('width', str(int(width)))
    root.set('height', str(int(height)))
    return ET.tostring(root, encoding='unicode')


def render_svg_reference(svg_path, png_path, width=2048, *, background="#ffffff"):
    """Render safely or raise; never substitute a flat color for a gradient.

    Returns provenance, dimensions and byte fingerprints. The PNG is written
    only after successful native rendering and dimension verification.
    ``background=None`` preserves alpha for independent compositing checks.
    """
    if isinstance(width, bool) or not isinstance(width, int) or not 1 <= width <= MAX_RENDER_SIDE:
        raise ValueError("Invalid render width")
    payload = Path(svg_path).read_bytes()
    svg, original_width, original_height = _validated_svg(payload)
    height = max(1, int(math.ceil(width * original_height / original_width)))
    if height > MAX_RENDER_SIDE or width * height > MAX_RENDER_PIXELS:
        raise ValueError("SVG render exceeds the bounded pixel budget")
    import resvg_py
    from PIL import Image
    png = resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height,
                                background=background, skip_system_fonts=True,
                                log_information=False, shape_rendering="geometric_precision")
    with Image.open(io.BytesIO(png)) as rendered:
        if rendered.size != (width, height):
            raise ValueError("Native renderer returned unexpected dimensions")
    Path(png_path).write_bytes(png)
    return {**renderer_info(), "width": width, "height": height,
            "source_svg_sha256": hashlib.sha256(payload).hexdigest(),
            "png_sha256": hashlib.sha256(png).hexdigest(), "background": background}
