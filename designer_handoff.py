"""Conservative, object-level designer handoff. No human acceptance is inferred.

The input is never rewritten. SVG paint order, ancestor styling/transforms,
compound paths and resource definitions survive selection by tree pruning.
Unsupported active content and dependencies are rejected before preview/export.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import tempfile
import xml.etree.ElementTree as ET

SVG = "http://www.w3.org/2000/svg"
INK = "http://www.inkscape.org/namespaces/inkscape"
XLINK = "http://www.w3.org/1999/xlink"
SCHEMA = "aivc.designer-handoff/v1"
DRAWABLE = {"path", "rect", "circle", "ellipse", "line", "polygon", "polyline", "text"}
RESOURCES = {"defs", "linearGradient", "radialGradient", "stop", "clipPath", "mask"}
ALLOWED = DRAWABLE | RESOURCES | {"svg", "g", "tspan", "title", "desc", "metadata"}
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
NUM_RE = re.compile(NUMBER)
TOKEN_RE = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|" + NUMBER)
ET.register_namespace("", SVG)
ET.register_namespace("inkscape", INK)
ET.register_namespace("xlink", XLINK)


def _local(name):
    return name.rsplit("}", 1)[-1]


def _style(element):
    result = {}
    for field in (element.get("style") or "").split(";"):
        if ":" in field:
            key, value = field.split(":", 1)
            result[key.strip().lower()] = value.strip()
    return result


def _css_value(value):
    return str(value).split("!", 1)[0].strip().lower()


def _number(value, default=0.0):
    if value is None:
        return default
    match = re.fullmatch(r"\s*(" + NUMBER + r")(?:px)?\s*", str(value))
    if not match:
        raise ValueError("Unsupported SVG length: " + str(value)[:80])
    result = float(match.group(1))
    if not math.isfinite(result):
        raise ValueError("Non-finite SVG length")
    return result


def _view_box(root):
    raw = root.get("viewBox")
    if raw:
        values = [_number(value) for value in re.split(r"[\s,]+", raw.strip())]
        if len(values) != 4:
            raise ValueError("SVG viewBox must contain four numbers")
    else:
        values = [0.0, 0.0, _number(root.get("width")), _number(root.get("height"))]
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError("SVG needs a positive viewBox or numeric width and height")
    return values


def _validate_tree(root):
    if root.tag not in {"svg", "{" + SVG + "}svg"}:
        raise ValueError("Input root must be SVG")
    root_style = _style(root)
    # A transformed or clipped root would also transform/clip the source image
    # inserted for handoff. Reject that uncommon viewport case instead of
    # claiming that an incorrectly aligned reference is usable.
    for key in ("transform", "clip-path", "mask", "filter"):
        if root.get(key) or root_style.get(key):
            raise ValueError("Root SVG effects are unsupported for reference alignment: " + key)
    if _css_value(root_style.get("display", root.get("display", ""))) == "none":
        raise ValueError("Hidden root SVG is unsupported")
    if _css_value(root_style.get("visibility", root.get("visibility", ""))) in {"hidden", "collapse"}:
        raise ValueError("Hidden root SVG is unsupported")
    root_opacity = root_style.get("opacity", root.get("opacity"))
    if root_opacity is not None and _css_value(root_opacity) not in {"1", "1.0", "100%"}:
        raise ValueError("Root SVG opacity is unsupported for reference alignment")
    parents = {child: parent for parent in root.iter() for child in parent}
    ids = {}
    references = []
    for element in root.iter():
        tag = _local(element.tag)
        ancestor = element
        in_metadata = False
        in_defs = False
        while ancestor is not None:
            in_metadata |= _local(ancestor.tag) == "metadata"
            in_defs |= _local(ancestor.tag) == "defs"
            ancestor = parents.get(ancestor)
        if ("}" in element.tag and not element.tag.startswith("{" + SVG + "}")) or tag not in ALLOWED:
            raise ValueError("Unsupported SVG element: " + tag)
        if tag in {"script", "foreignObject", "style", "use", "image", "a"}:
            raise ValueError("Active, raster, linked or unsupported SVG element: " + tag)
        if tag == "svg" and element is not root:
            raise ValueError("Nested SVG viewports are unsupported")
        if tag in RESOURCES - {"defs"} and not in_defs:
            raise ValueError("SVG resources must be inside defs")
        if element.get("id"):
            if element.get("id") in ids:
                raise ValueError("Duplicate SVG ID: " + element.get("id"))
            ids[element.get("id")] = (element, in_defs)
        for name, value in element.attrib.items():
            key = _local(name).lower()
            if key.startswith("on") or key in {"base", "src"}:
                raise ValueError("Active or external SVG attribute: " + key)
            if any(token in value.lower() for token in ("javascript:", "expression(", "@import", "@font-face")):
                raise ValueError("Active SVG attribute content")
            if "\\" in value and key == "style":
                raise ValueError("Escaped CSS is unsupported")
            if key == "href":
                if tag not in {"linearGradient", "radialGradient"} or not re.fullmatch(r"#[^\s#]+", value):
                    raise ValueError("Only local gradient href references are supported")
                references.append(value[1:])
            if re.search(r"url\s*\(", value, re.I):
                # CSS escapes/comments and external URLs are deliberately unsupported.
                matches = list(re.finditer(r"url\(\s*['\"]?(#[^\s)'\"]+)['\"]?\s*\)", value, re.I))
                if len(matches) != len(re.findall(r"url\s*\(", value, re.I)) or not matches:
                    raise ValueError("Unsupported or external SVG URL")
                references.extend(match.group(1)[1:] for match in matches)
            if key == "style" and ("/*" in value or "*/" in value):
                raise ValueError("CSS comments are unsupported")
    for identifier in references:
        if identifier not in ids or not ids[identifier][1]:
            raise ValueError("SVG reference must resolve inside defs: " + identifier)
    return parents


def _load(svg_path):
    payload = Path(svg_path).read_bytes()
    if len(payload) > 50 * 1024 * 1024:
        raise ValueError("SVG exceeds the 50 MiB handoff limit")
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)", payload, re.I):
        raise ValueError("SVG DTDs and entities are unsupported")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError("Invalid SVG XML") from exc
    parents = _validate_tree(root)
    view_box = _view_box(root)
    used = {element.get("id") for element in root.iter() if element.get("id")}
    serial = 0
    for element in root.iter():
        if _local(element.tag) not in DRAWABLE or _inside_defs(element, parents):
            continue
        if not element.get("id"):
            serial += 1
            identifier = f"handoff-item-{serial:04d}"
            while identifier in used:
                serial += 1
                identifier = f"handoff-item-{serial:04d}"
            element.set("id", identifier)
            used.add(identifier)
    return root, parents, view_box, hashlib.sha256(payload).hexdigest()


def _inside_defs(element, parents):
    while element is not None:
        if _local(element.tag) in RESOURCES | {"metadata", "title", "desc"}:
            return True
        element = parents.get(element)
    return False


def _visible(element, parents):
    chain = []
    while element is not None:
        chain.append(element)
        element = parents.get(element)
    visibility = "visible"
    for item in reversed(chain):
        style = _style(item)
        if _css_value(style.get("display", item.get("display", ""))) == "none":
            return False
        opacity = style.get("opacity", item.get("opacity"))
        if opacity is not None:
            try:
                if float(_css_value(opacity)) == 0:
                    return False
            except ValueError:
                pass
        visibility = style.get("visibility", item.get("visibility", visibility))
    return _css_value(visibility) not in {"hidden", "collapse"}


def _path_geometry(data):
    """Return anchor count and conservative control-point box, or unknown box.

    Arcs include their rotated ellipse extrema, not just their endpoints.
    Cubic/quadratic control hulls are conservative, not exact painted bounds.
    """
    tokens = TOKEN_RE.findall(data or "")
    residue = TOKEN_RE.sub("", data or "")
    if residue.strip(" ,\t\r\n") or not tokens:
        return 0, None
    arities = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}
    index, count = 0, 0
    command = None
    x = y = sx = sy = 0.0
    points = []
    uncertain = False
    previous_control = None
    previous_command = None
    try:
        while index < len(tokens):
            if tokens[index].isalpha():
                command = tokens[index]
                index += 1
                if command.upper() == "Z":
                    if (x, y) == (sx, sy) and previous_command not in {None, "M", "Z"}:
                        count = max(0, count - 1)
                    x, y = sx, sy
                    previous_control = None
                    previous_command = "Z"
                    command = None
                    continue
            if not command or command.upper() not in arities:
                return count, None
            kind = command.upper()
            arity = arities[kind]
            values = [float(value) for value in tokens[index:index + arity]]
            if len(values) != arity or not all(math.isfinite(value) for value in values):
                return count, None
            index += arity
            relative = command.islower()
            ox, oy = (x, y) if relative else (0.0, 0.0)
            if kind == "H":
                x = values[0] + ox
            elif kind == "V":
                y = values[0] + oy
            elif kind == "A":
                from svg_bounds import arc_extrema
                absolute = [*values[:-2], values[-2] + ox, values[-1] + oy]
                points.extend(arc_extrema((x, y), absolute))
                x, y = absolute[-2:]
            else:
                converted = [(values[j] + ox, values[j + 1] + oy) for j in range(0, arity, 2)]
                if kind in {"S", "T"}:
                    compatible = {"C", "S"} if kind == "S" else {"Q", "T"}
                    reflected = ((2 * x - previous_control[0], 2 * y - previous_control[1])
                                 if previous_control is not None and previous_command in compatible else (x, y))
                    points.append(reflected)
                    if kind == "T":
                        previous_control = reflected
                points.extend(converted)
                x, y = converted[-1]
                if kind in {"C", "S", "Q"}:
                    previous_control = converted[-2]
            points.append((x, y))
            count += 1
            if kind == "M":
                sx, sy = x, y
                command = "l" if relative else "L"
            if kind not in {"C", "S", "Q", "T"}:
                previous_control = None
            previous_command = kind
    except (ValueError, IndexError):
        return count, None
    return count, None if uncertain else _point_box(points)


def _point_box(points):
    if not points:
        return None
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _geometry(element):
    tag = _local(element.tag)
    try:
        if tag == "path":
            return _path_geometry(element.get("d", ""))
        if tag == "rect":
            from svg_postprocess import rect_designer_anchors
            x, y = _number(element.get("x")), _number(element.get("y"))
            w, h = _number(element.get("width")), _number(element.get("height"))
            return rect_designer_anchors(element), [x, y, w, h]
        if tag in {"circle", "ellipse"}:
            cx, cy = _number(element.get("cx")), _number(element.get("cy"))
            rx = _number(element.get("r" if tag == "circle" else "rx"))
            ry = rx if tag == "circle" else _number(element.get("ry"))
            return 4, [cx - rx, cy - ry, 2 * rx, 2 * ry]
        if tag == "line":
            return 2, _point_box([(_number(element.get("x1")), _number(element.get("y1"))),
                                  (_number(element.get("x2")), _number(element.get("y2")))])
        if tag in {"polygon", "polyline"}:
            raw = element.get("points", "")
            if NUM_RE.sub("", raw).strip(" ,\t\r\n"):
                return 0, None
            values = [float(value) for value in NUM_RE.findall(raw)]
            if len(values) % 2:
                return 0, None
            points = list(zip(values[::2], values[1::2]))
            return len(points), _point_box(points)
    except ValueError:
        pass
    return 0, None


def _safe_box(element, parents):
    from svg_bounds import paint_bounds
    box = _geometry(element)[1]
    if box and (not all(math.isfinite(value) for value in box) or box[2] < 0 or box[3] < 0):
        return None
    return paint_bounds(element, parents, box, _style)


def _units(root, parents):
    units = []
    by_owner = {}
    for element in root.iter():
        if (_local(element.tag) not in DRAWABLE or _inside_defs(element, parents)
                or not _visible(element, parents)):
            continue
        owner = element
        item = parents.get(element)
        while item is not None:
            if _local(item.tag) == "g" and (item.get("id") or "").startswith("object-"):
                owner = item
            item = parents.get(item)
        if owner not in by_owner:
            by_owner[owner] = []
            units.append((owner, by_owner[owner]))
        by_owner[owner].append(element)
    return units


def _reference_kind(report):
    kind = report.get("_handoff_reference_kind") if isinstance(report, dict) else None
    return kind if kind in {"original", "processed_reference"} else "unspecified_reference"


def _display_label(owner, members, bbox, view_box, index):
    label = owner.get("{" + INK + "}label") or owner.get("aria-label")
    # Preserve author labels; replace generated English labels and opaque IDs
    # with location and shape, without claiming to infer the design's meaning.
    if label and not re.fullmatch(r"[A-Za-z-]+ layered object \(\d+ parts\)", label):
        return label
    region = ""
    if bbox is not None:
        x, y, width, height = view_box
        cx = (bbox[0] + bbox[2] / 2 - x) / width
        cy = (bbox[1] + bbox[3] / 2 - y) / height
        horizontal = "左" if cx < 1 / 3 else "右" if cx > 2 / 3 else "中"
        vertical = "上" if cy < 1 / 3 else "下" if cy > 2 / 3 else ""
        region = (horizontal + vertical if vertical else {"左": "左側", "中": "中央", "右": "右側"}[horizontal]) + " · "
    kinds = {"path": "輪廓", "circle": "圓形", "ellipse": "橢圓", "rect": "矩形",
             "line": "線段", "polyline": "折線", "polygon": "多邊形", "text": "文字"}
    kind = kinds.get(_local(members[0].tag), "物件") if len(members) == 1 else f"群組（{len(members)} 部件）"
    return f"{region}{kind} {index + 1:03d}"


def _manifest(root, parents, view_box, digest, report=None):
    objects = []
    source_report_supplied = report is not None
    report = report or {}
    stages = ((report.get('editability_enhancements') or {}).get('stages') or {})
    from quality_diagnostics import source_structure_hotspots
    topology = stages.get('source_topology_audit') or {}
    # Geometry can be edited independently of the saved report. Do not locate
    # an old concern on a new result merely because an ID or bbox survived.
    topology_matches = (isinstance(topology, dict)
                        and (topology.get('source_hashes') or {}).get('svg') == digest
                        and topology.get('inputs_unchanged') is True)
    defects = source_structure_hotspots(topology, view_box) if topology_matches else []
    quality = report.get('designer_quality') or {}
    quality_matches = (isinstance(quality, dict)
                       and (quality.get('source') or {}).get('sha256') == digest)
    curve_diagnostic = ((quality.get('curve_economy_gate') or {}).get(
        'short_counterturn_diagnostic') or {}) if quality_matches else {}
    curve_reviews = {item.get('id'): item for item in curve_diagnostic.get('paths', [])
                     if isinstance(item, dict) and item.get('requires_review') is True}
    paint_evidence = ((quality.get('gradient_object_gate') or {}).get(
        'source_space_field_evidence') or {}) if quality_matches else {}
    partial_paint_members = {identifier for item in paint_evidence.get('objects', [])
        if isinstance(item, dict) and item.get('passed') is True
        and item.get('partial_paint_requires_manual_review') is True
        for identifier in item.get('final_drawable_ids', [])}
    for owner, members in _units(root, parents):
        anchors = sum(_geometry(member)[0] for member in members)
        paths = sum(_local(member.tag) == "path" for member in members)
        boxes = [_safe_box(member, parents) for member in members]
        bbox = None
        if boxes and all(box is not None for box in boxes):
            bbox = _point_box([(box[0], box[1]) for box in boxes] +
                              [(box[0] + box[2], box[1] + box[3]) for box in boxes])
        reasons = []
        action = "review"
        if anchors >= 80 or paths >= 8:
            action = "redraw"
            reasons.append("高節點或多路徑負擔；建議比較局部重畫與修整時間，並非要求刪除設計細節。")
        elif anchors > 40 or len(members) >= 5:
            reasons.append("節點或組件較多，請檢查是否適合整組編輯。")
        elif all(_local(member.tag) in DRAWABLE - {"path", "text"} for member in members) and anchors <= 16:
            action = "keep"
            reasons.append("原生幾何且結構負擔低；仍需人工確認造型與遮擋。")
        if any(member.get("data-avc-source-recovery") for member in members):
            action = "review"
            reasons.insert(0, "原圖中曾被描圖器漏掉的小細節已補回；請確認它是設計細節或雜點，尚未猜成理想圓形。")
        if bbox is None:
            reasons.append("未提供可保守推算的全圖邊界；請以畫面定位，未產生猜測框。")
        # A region's bounding rectangle cannot identify its responsible unit.
        # Keep topology concerns at scene level until actual visible-pixel
        # evidence can attribute a colour/coverage hint to this object.
        curve_review_count = sum(int(curve_reviews.get(member.get('id'), {}).get(
            'short_counterturn_count', 0)) for member in members)
        partial_paint_count = sum(member.get('id') in partial_paint_members for member in members)
        if partial_paint_count:
            if action == 'keep':
                action = 'review'
            reasons.append('這部分只改善漸層填色，原有多條路徑仍保留；請確認輪廓與接縫。')
        if curve_review_count:
            if action == 'keep':
                action = 'review'
            reasons.append(f'有 {curve_review_count} 段短曲線反覆轉彎；請放大確認是否為不需要的毛刺，機器未判定它一定畫錯。')
        identifier = owner.get("id")
        objects.append({"id": identifier,
                        "label": _display_label(owner, members, bbox, view_box, len(objects)),
                        "member_ids": [member.get("id") for member in members],
                        "anchor_count": anchors, "path_count": paths, "bbox": bbox,
                        "suggested_action": action, "reasons": reasons,
                        "source_defect_count": 0,
                        "source_defect_fraction": 0,
                        "partial_paint_member_count": partial_paint_count,
                        "curve_review_count": curve_review_count})
    return {"schema": SCHEMA, "svg_sha256": digest, "view_box": view_box,
            "reference_kind": _reference_kind(report),
            "objects": objects, "default_decisions": {item["id"]: "review" for item in objects},
            "status": "suggestions_only", "human_acceptance": "not_performed",
            "source_report_supplied": source_report_supplied,
            "source_topology_locations_current": topology_matches,
            "scene_source_concerns": defects,
            "curve_review_locations_current": quality_matches,
            "bbox_kind": "conservative_geometry_bounds_not_exact_paint_bounds",
            "limitations": ["建議不是驗收、校準信心或省工承諾。", "原有 object-* 最外層群組視為單一交接單位，無語意重新分組。",
                            "必須檢查局部刪除後的遮擋與露底，保留的幾何不會被改造。"]}


def build_handoff_manifest(svg_path, report=None, *, source_png=None):
    """Describe disjoint existing objects; all initial decisions require review."""
    root, parents, view_box, digest = _load(svg_path)
    manifest = _manifest(root, parents, view_box, digest, report)
    from source_object_audit import attach_source_hints
    return attach_source_hints(manifest, root, source_png if _reference_kind(report) == 'original' else None)


def normalized_svg_text(svg_path):
    """Safe, deterministic preview SVG with IDs identical to the manifest."""
    root, _, _, _ = _load(svg_path)
    return ET.tostring(root, encoding="unicode")


def validate_decisions(manifest, decisions, expected_sha256):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported handoff manifest schema")
    if not isinstance(expected_sha256, str) or expected_sha256 != manifest.get("svg_sha256"):
        raise ValueError("SVG changed or fingerprint is missing; rebuild the handoff before exporting")
    if not isinstance(decisions, dict):
        raise ValueError("Decisions must be an object ID to action mapping")
    identifiers = {item["id"] for item in manifest["objects"]}
    unknown = set(decisions) - identifiers
    missing = identifiers - set(decisions)
    if unknown:
        raise ValueError("Unknown object IDs: " + ", ".join(sorted(str(value) for value in unknown)))
    if missing:
        raise ValueError("Missing object decisions: " + ", ".join(sorted(missing)))
    if any(not isinstance(value, str) or value not in {"keep", "review", "redraw"} for value in decisions.values()):
        raise ValueError("Actions must be keep, review or redraw")
    return {item["id"]: decisions[item["id"]] for item in manifest["objects"]}


def _svg_bytes(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _tag(name):
    return "{" + SVG + "}" + name


def _add_note(root, identifier, text):
    node = ET.Element(_tag("desc"), {"id": identifier})
    node.text = text
    root.insert(0, node)


def _handoff_metadata(root, digest, decisions, role, reference_kind):
    # Input quality reports refer to the complete original image; they must
    # never masquerade as a validation of a newly selected subset.
    for parent in list(root.iter()):
        for child in list(parent):
            if _local(child.tag) == "metadata":
                parent.remove(child)
    metadata = ET.SubElement(root, _tag("metadata"), {"id": _unique_id(root, "designer-handoff-metadata")})
    metadata.text = json.dumps({"schema": "aivc.designer-handoff-export/v1", "source_svg_sha256": digest,
                                "role": role, "not_finished_artwork": True, "decisions": decisions,
                                "reference_kind": reference_kind,
                                "source_validation_applies_to": "original_input_only",
                                "human_final_design_acceptance": "not_performed"}, ensure_ascii=False)


def _png_data(path):
    payload = Path(path).read_bytes()
    if (len(payload) < 33 or len(payload) > 100 * 1024 * 1024
            or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR"):
        raise ValueError("A valid local PNG reference is required (maximum 100 MiB)")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("PNG reference has invalid dimensions")
    return payload, [width, height]


def _unique_id(root, wanted):
    used = {element.get("id") for element in root.iter()}
    candidate, suffix = wanted, 2
    while candidate in used:
        candidate = f"{wanted}-{suffix}"
        suffix += 1
    return candidate


def _atomic_write(path, payload):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".handoff-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_handoff(svg_path, source_png, decisions, output_dir, *, expected_sha256, report=None):
    """Write partial accepted vectors, a raster-backed draft, and JSON decisions.

    Deleting an occluder may expose a kept object. Geometry/order are preserved;
    no claim is made that arbitrary subsets reproduce the original appearance.
    """
    root, parents, view_box, digest = _load(svg_path)
    manifest = _manifest(root, parents, view_box, digest, report)
    from source_object_audit import attach_source_hints
    attach_source_hints(manifest, root, source_png if _reference_kind(report) == 'original' else None)
    decisions = validate_decisions(manifest, decisions, expected_sha256)
    reference_kind = manifest["reference_kind"]
    reference_label = {"original": "原圖參考（點陣，非成品）",
                       "processed_reference": "清理後參考圖（非原圖，點陣，非成品）",
                       "unspecified_reference": "來源參考圖（點陣，未驗證原始性）"}[reference_kind]
    png, png_size = _png_data(source_png)
    output_dir = Path(output_dir).resolve()
    filenames = {"working_svg": "working.svg", "accepted_svg": "accepted.svg", "draft_svg": "draft.svg",
                 "manifest_json": "handoff-manifest.json", "decisions_json": "handoff-decisions.json",
                 "handoff_json": "handoff.json"}
    paths = {key: output_dir / name for key, name in filenames.items()}
    inputs = {Path(svg_path).resolve(), Path(source_png).resolve()}
    if any(path in inputs for path in paths.values()):
        raise ValueError("Handoff outputs must not overwrite the source SVG or reference PNG")
    accepted = copy.deepcopy(root)
    accepted_parents = {child: parent for parent in accepted.iter() for child in parent}
    ownership = {}
    for owner, members in _units(accepted, accepted_parents):
        for member in members:
            ownership[member] = owner.get("id")

    def prune(element, in_defs=False, retained_owner=False):
        tag = _local(element.tag)
        if in_defs or tag == "defs":
            return True
        if tag == "g" and element.get("id") in decisions:
            if decisions[element.get("id")] != "keep":
                return False
            retained_owner = True
        if tag in DRAWABLE:
            return retained_owner or decisions.get(ownership.get(element)) == "keep"
        for child in list(element):
            if not prune(child, False, retained_owner):
                element.remove(child)
        return tag != "g" or bool(list(element))

    prune(accepted)
    omitted = [identifier for identifier, action in decisions.items() if action != "keep"]
    note = ("PARTIAL DESIGNER HANDOFF — selected vector objects only; not a complete approved artwork. "
            "Omitted objects and decisions are listed in handoff.json. Removing an occluder can expose kept geometry.")
    accepted.set("data-handoff-status", "partial-selected-vectors")
    _add_note(accepted, _unique_id(accepted, "handoff-accepted-note"), note)
    _handoff_metadata(accepted, digest, decisions, "selected-vector-objects", reference_kind)

    draft = copy.deepcopy(root)
    draft.set("data-handoff-status", "draft-with-raster-reference-not-final")
    draft_ids = {element.get("id"): element for element in draft.iter() if element.get("id")}
    for identifier, action in decisions.items():
        element = draft_ids[identifier]
        element.set("data-handoff-action", action)
        if action != "keep":
            declarations = [field for field in (element.get("style") or "").split(";")
                            if field.strip() and field.split(":", 1)[0].strip().lower() != "display"]
            element.set("style", ";".join(declarations + ["display:none!important"]))
    reference_id = _unique_id(draft, "handoff-source-reference")
    reference = ET.Element(_tag("g"), {"id": reference_id, "{" + INK + "}groupmode": "layer",
                                       "{" + INK + "}label": reference_label, "data-handoff-role": "raster-reference",
                                       "data-handoff-reference-kind": reference_kind})
    x, y, width, height = view_box
    ET.SubElement(reference, _tag("image"), {"x": str(x), "y": str(y), "width": str(width), "height": str(height),
                                             "preserveAspectRatio": "none", "opacity": "0.35",
                                             "{" + XLINK + "}href": "data:image/png;base64," + base64.b64encode(png).decode("ascii")})
    draft.insert(0, reference)
    guide_id = _unique_id(draft, "handoff-redraw-guides")
    guides = ET.SubElement(draft, _tag("g"), {"id": guide_id, "{" + INK + "}groupmode": "layer",
                                             "{" + INK + "}label": "待處理區域提示（非成品，輸出前移除）",
                                             "data-handoff-role": "guides", "fill": "none", "stroke": "#d94828",
                                             "stroke-width": str(max(width, height) / 500.0)})
    guide_ids = []
    for item in manifest["objects"]:
        if decisions[item["id"]] != "keep" and item["bbox"] is not None:
            bx, by, bw, bh = item["bbox"]
            ET.SubElement(guides, _tag("rect"), {"x": str(bx), "y": str(by), "width": str(bw), "height": str(bh),
                                                "stroke-dasharray": "5 3", "data-handoff-object": item["id"]})
            guide_ids.append(item["id"])
    _add_note(draft, _unique_id(draft, "handoff-draft-note"),
              "DRAFT, NOT FINISHED ARTWORK. Embedded raster reference below vectors; non-kept candidates are hidden "
              "in their original tree/order. Guide rectangles are approximate and must not be delivered as artwork.")
    _handoff_metadata(draft, digest, decisions, "draft-with-raster-reference", reference_kind)
    # Immediate Illustrator entry point: keep every candidate visible so an
    # unreviewed document can be edited without hundreds of adoption clicks.
    # Decisions remain annotations; this complete candidate is never labelled
    # an accepted subset or a verified final design.
    working = copy.deepcopy(root)
    working.set("data-handoff-status", "complete-candidate-for-editing-not-final")
    for element in working.iter():
        if element.get("id") in decisions:
            element.set("data-handoff-action", decisions[element.get("id")])
    working_reference = copy.deepcopy(reference)
    working_reference.set("id", _unique_id(working, "handoff-source-reference"))
    working_reference.set("style", "display:none")
    working.insert(0, working_reference)
    _add_note(working, _unique_id(working, "handoff-working-note"),
              "COMPLETE CANDIDATE FOR EDITING, NOT APPROVED ARTWORK. All candidate vectors remain visible, "
              "including review/redraw decisions. The aligned raster reference is hidden. "
              "Consult handoff.json for decisions. Remove the reference before delivery.")
    _handoff_metadata(working, digest, decisions, "complete-candidate-for-editing", reference_kind)
    summary = {"total_objects": len(decisions), **{action: sum(value == action for value in decisions.values())
                                                 for action in ("keep", "review", "redraw")},
               "is_partial": bool(omitted), "omitted_object_ids": omitted,
               "human_acceptance": "explicit_selection_only_not_final_design_validation"}
    handoff = {"schema": "aivc.designer-handoff-export/v1", "svg_sha256": digest,
               "reference_kind": reference_kind,
               "source_png_sha256": hashlib.sha256(png).hexdigest(), "source_png_size": png_size,
               "summary": summary, "decisions": decisions,
               "accepted_status": "partial_selected_objects_not_finished_artwork",
               "working_status": "all_candidates_visible_not_finally_accepted",
               "working_reference_layer_id": working_reference.get("id"),
               "working_has_hidden_raster_reference": True,
               "draft_has_raster_reference": True, "draft_has_guides": bool(guide_ids),
               "draft_reference_layer_id": reference_id, "draft_guide_layer_id": guide_id,
               "guide_object_ids": guide_ids,
               "unlocated_omitted_object_ids": [identifier for identifier in omitted if identifier not in guide_ids],
               "reference_alignment": "entire_source_png_mapped_to_svg_viewBox",
               "reference_opacity": 0.35,
               "not_finished_artwork": True,
               "omitted_objects": [{**item, "decision": decisions[item["id"]]} for item in manifest["objects"] if item["id"] in omitted],
               "warnings": ["局部移除可能露出底下物件，請人工確認遮擋與露底。", "draft.svg 含點陣參考圖、隱藏候選與提示框，不是純向量完稿。",
                            "參考圖永遠保留；未知邊界不產生猜測提示框。",
                            ("參考圖已去背景或清理，不能用來證明未處理原圖的全部細節。"
                             if reference_kind == "processed_reference" else "參考圖來源種類已記錄於 reference_kind。")],
               "filenames": filenames}
    decision_record = {"schema": "aivc.designer-handoff-decisions/v1", "svg_sha256": digest,
                       "reference_kind": reference_kind, "decisions": decisions}
    payloads = {"working_svg": _svg_bytes(working), "accepted_svg": _svg_bytes(accepted), "draft_svg": _svg_bytes(draft),
                "manifest_json": json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                "decisions_json": json.dumps(decision_record, ensure_ascii=False, indent=2).encode("utf-8"),
                "handoff_json": json.dumps(handoff, ensure_ascii=False, indent=2).encode("utf-8")}
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in payloads.items():
        _atomic_write(paths[key], payload)
    return {"schema": handoff["schema"], "files": {key: str(path) for key, path in paths.items()},
            "summary": summary, "svg_sha256": digest}
