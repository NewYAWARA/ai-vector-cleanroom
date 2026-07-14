# -*- coding: utf-8 -*-
"""Portable paint-role resources for generated SVG files.

The conversion engine intentionally writes ordinary SVG ``fill``, ``stroke``
and ``stop-color`` presentation attributes.  Those explicit values remain the
rendering authority because Illustrator, Inkscape, browsers and lightweight
SVG readers all understand them.  This module adds a *non-rendering* role
manifest and inert ``data-paint-role-*`` annotations, then offers a helper that
can recolour an entire role while writing explicit presentation attributes
back out.

No CSS custom properties, SVG 2 ``solidColor`` resources or editor-specific
swatches are required for rendering.  Editors that discard metadata still
open the artwork normally; the companion tool merely loses its one-control
role information and can rebuild it from the explicit paints.

The clustering is deliberately content-agnostic.  It uses OKLCH lightness,
chroma and hue, never filenames, layer names, language, or known logo colours.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
MANIFEST_SCHEMA = "ai-vector-cleanroom.paint-roles/v1"
MANIFEST_METADATA_ID = "ai-vector-cleanroom-paint-roles"
RECOLOR_METADATA_ID = "ai-vector-cleanroom-paint-recolor"

ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)

_PAINT_PROPERTIES = ("fill", "stroke", "stop-color")
_DRAWABLE_TAGS = {
    "path", "circle", "rect", "ellipse", "line", "polyline", "polygon",
    "text", "use",
}
_GRADIENT_TAGS = {"linearGradient", "radialGradient"}
_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3,8})$")
_RGB_RE = re.compile(
    r"^(rgba?)\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,\)]+)"
    r"(?:\s*,\s*([^\)]+))?\s*\)$",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"^url\(\s*#([^\)\s]+)\s*\)$", re.IGNORECASE)


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParsedColor:
    """A parsed RGB colour plus enough information to preserve alpha."""

    hex_rgb: str
    alpha: float
    source_format: str
    alpha_text: str = ""


@dataclass
class PaintOccurrence:
    """One explicit paint declaration in an SVG element."""

    element: ET.Element
    property_name: str
    location: str  # ``attribute`` or ``style``
    parsed: ParsedColor
    gradient_id: str | None


def _parse_channel(value: str) -> int | None:
    text = value.strip()
    try:
        if text.endswith("%"):
            return round(_clamp(float(text[:-1]), 0.0, 100.0) * 2.55)
        return round(_clamp(float(text), 0.0, 255.0))
    except ValueError:
        return None


def parse_color(value: str) -> ParsedColor | None:
    """Parse common SVG/CSS RGB forms; return ``None`` for URLs and names."""

    raw = (value or "").strip()
    important = re.sub(r"\s*!important\s*$", "", raw, flags=re.IGNORECASE)
    match = _HEX_RE.fullmatch(important)
    if match:
        digits = match.group(1)
        if len(digits) in (3, 4):
            expanded = "".join(character * 2 for character in digits)
        elif len(digits) in (6, 8):
            expanded = digits
        else:
            return None
        rgb = expanded[:6].lower()
        alpha_hex = expanded[6:8]
        alpha = int(alpha_hex, 16) / 255.0 if alpha_hex else 1.0
        return ParsedColor("#" + rgb, alpha, "hex", alpha_hex.lower())

    match = _RGB_RE.fullmatch(important)
    if not match:
        return None
    function, red, green, blue, alpha_text = match.groups()
    channels = [_parse_channel(item) for item in (red, green, blue)]
    if any(item is None for item in channels):
        return None
    alpha = 1.0
    if function.lower() == "rgba":
        if alpha_text is None:
            return None
        try:
            alpha = (float(alpha_text[:-1]) / 100.0
                     if alpha_text.strip().endswith("%")
                     else float(alpha_text))
        except ValueError:
            return None
        alpha = _clamp(alpha, 0.0, 1.0)
    return ParsedColor(
        "#{:02x}{:02x}{:02x}".format(*channels), alpha,
        function.lower(), (alpha_text or "").strip(),
    )


def _format_replacement(hex_rgb: str, original: ParsedColor) -> str:
    red = int(hex_rgb[1:3], 16)
    green = int(hex_rgb[3:5], 16)
    blue = int(hex_rgb[5:7], 16)
    if original.source_format == "rgba":
        alpha = original.alpha_text or f"{original.alpha:.4g}"
        return f"rgba({red},{green},{blue},{alpha})"
    if original.source_format == "rgb":
        return f"rgb({red},{green},{blue})"
    if original.alpha_text:
        return hex_rgb + original.alpha_text
    return hex_rgb


def _parse_style(raw: str) -> list[tuple[str, str]]:
    declarations: list[tuple[str, str]] = []
    for fragment in (raw or "").split(";"):
        if ":" not in fragment:
            continue
        name, value = fragment.split(":", 1)
        name = name.strip()
        if name:
            declarations.append((name, value.strip()))
    return declarations


def _style_value(element: ET.Element, property_name: str) -> str | None:
    result = None
    for name, value in _parse_style(element.attrib.get("style", "")):
        if name.lower() == property_name:
            result = value
    return result


def _set_style_value(element: ET.Element, property_name: str,
                     replacement: str) -> None:
    declarations = _parse_style(element.attrib.get("style", ""))
    rewritten: list[tuple[str, str]] = []
    found = False
    for name, value in declarations:
        if name.lower() == property_name:
            value = replacement
            found = True
        rewritten.append((name, value))
    if not found:
        rewritten.append((property_name, replacement))
    element.set("style", ";".join(f"{name}:{value}" for name, value in rewritten))


def _collect_occurrences(root: ET.Element) -> tuple[
        list[PaintOccurrence], dict[str, int], list[str]]:
    occurrences: list[PaintOccurrence] = []
    gradient_references: dict[str, int] = {}
    unsupported: set[str] = set()

    def walk(element: ET.Element, current_gradient: str | None = None) -> None:
        tag = _local_name(element.tag)
        gradient_id = current_gradient
        if tag in _GRADIENT_TAGS:
            gradient_id = element.attrib.get("id") or None

        for property_name in _PAINT_PROPERTIES:
            values: list[tuple[str, str]] = []
            if property_name in element.attrib:
                values.append(("attribute", element.attrib[property_name]))
            style = _style_value(element, property_name)
            if style is not None:
                # CSS style overrides the presentation attribute, but both are
                # retained as resources so an editor flattening styles cannot
                # reveal an untracked fallback colour.
                values.append(("style", style))
            for location, value in values:
                url_match = _URL_RE.fullmatch(value.strip())
                if url_match:
                    gradient_references[url_match.group(1)] = (
                        gradient_references.get(url_match.group(1), 0) + 1)
                    continue
                parsed = parse_color(value)
                if parsed is not None:
                    occurrences.append(PaintOccurrence(
                        element, property_name, location, parsed, gradient_id,
                    ))
                elif value.strip().lower() not in {
                        "", "none", "inherit", "currentcolor", "transparent"}:
                    unsupported.add(value.strip())

        for child in element:
            walk(child, gradient_id)

    walk(root)
    return occurrences, gradient_references, sorted(unsupported)


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(value: float) -> float:
    return 12.92 * value if value <= 0.0031308 else 1.055 * value ** (1 / 2.4) - 0.055


def hex_to_oklch(hex_rgb: str) -> tuple[float, float, float]:
    """Convert ``#rrggbb`` to (OKLab L, OKLCH C, hue degrees)."""

    red = _srgb_to_linear(int(hex_rgb[1:3], 16) / 255.0)
    green = _srgb_to_linear(int(hex_rgb[3:5], 16) / 255.0)
    blue = _srgb_to_linear(int(hex_rgb[5:7], 16) / 255.0)
    light = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    medium = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    short = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    light_cbrt = math.copysign(abs(light) ** (1 / 3), light)
    medium_cbrt = math.copysign(abs(medium) ** (1 / 3), medium)
    short_cbrt = math.copysign(abs(short) ** (1 / 3), short)
    lab_l = 0.2104542553 * light_cbrt + 0.7936177850 * medium_cbrt - 0.0040720468 * short_cbrt
    lab_a = 1.9779984951 * light_cbrt - 2.4285922050 * medium_cbrt + 0.4505937099 * short_cbrt
    lab_b = 0.0259040371 * light_cbrt + 0.7827717662 * medium_cbrt - 0.8086757660 * short_cbrt
    chroma = math.hypot(lab_a, lab_b)
    hue = math.degrees(math.atan2(lab_b, lab_a)) % 360.0 if chroma else 0.0
    return lab_l, chroma, hue


def _oklab_to_linear_rgb(lightness: float, chroma: float,
                         hue: float) -> tuple[float, float, float]:
    radians = math.radians(hue)
    lab_a = chroma * math.cos(radians)
    lab_b = chroma * math.sin(radians)
    light_cbrt = lightness + 0.3963377774 * lab_a + 0.2158037573 * lab_b
    medium_cbrt = lightness - 0.1055613458 * lab_a - 0.0638541728 * lab_b
    short_cbrt = lightness - 0.0894841775 * lab_a - 1.2914855480 * lab_b
    light = light_cbrt ** 3
    medium = medium_cbrt ** 3
    short = short_cbrt ** 3
    return (
        +4.0767416621 * light - 3.3077115913 * medium + 0.2309699292 * short,
        -1.2684380046 * light + 2.6097574011 * medium - 0.3413193965 * short,
        -0.0041960863 * light - 0.7034186147 * medium + 1.7076147010 * short,
    )


def oklch_to_hex(lightness: float, chroma: float, hue: float) -> str:
    """Convert OKLCH to an in-gamut sRGB hex, reducing chroma if needed."""

    lightness = _clamp(lightness, 0.0, 1.0)
    chroma = max(0.0, chroma)

    def in_gamut(candidate: float) -> bool:
        return all(-1e-7 <= channel <= 1.0000001
                   for channel in _oklab_to_linear_rgb(lightness, candidate, hue))

    if not in_gamut(chroma):
        low, high = 0.0, chroma
        for _ in range(24):
            middle = (low + high) / 2.0
            if in_gamut(middle):
                low = middle
            else:
                high = middle
        chroma = low
    linear = _oklab_to_linear_rgb(lightness, chroma, hue)
    channels = [round(_clamp(_linear_to_srgb(channel), 0.0, 1.0) * 255)
                for channel in linear]
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _circular_delta(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _circular_mean(entries: Iterable[Mapping[str, object]]) -> float:
    sine = 0.0
    cosine = 0.0
    for entry in entries:
        weight = float(entry.get("usage_count", 1) or 1)
        radians = math.radians(float(entry["hue"]))
        sine += math.sin(radians) * weight
        cosine += math.cos(radians) * weight
    return math.degrees(math.atan2(sine, cosine)) % 360.0


def _cluster_chromatic(entries: list[dict[str, object]],
                       max_hue_radius: float = 42.0) -> list[list[dict[str, object]]]:
    """Deterministic agglomeration constrained by a circular hue radius."""

    clusters: list[list[dict[str, object]]] = [[entry] for entry in entries]
    while True:
        choice: tuple[float, int, int] | None = None
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                merged = clusters[left] + clusters[right]
                centre = _circular_mean(merged)
                radius = max(abs(_circular_delta(float(item["hue"]), centre))
                             for item in merged)
                if radius <= max_hue_radius:
                    candidate = (radius, left, right)
                    if choice is None or candidate < choice:
                        choice = candidate
        if choice is None:
            break
        _radius, left, right = choice
        clusters[left] = clusters[left] + clusters[right]
        del clusters[right]
    return clusters


def _weighted_average(entries: Iterable[Mapping[str, object]], key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for entry in entries:
        weight = float(entry.get("usage_count", 1) or 1)
        numerator += float(entry[key]) * weight
        denominator += weight
    return numerator / denominator if denominator else 0.0


def _paint_signature(occurrences: Iterable[PaintOccurrence]) -> str:
    counts: dict[tuple[str, str, str], int] = {}
    for occurrence in occurrences:
        key = (occurrence.parsed.hex_rgb, occurrence.property_name,
               occurrence.location)
        counts[key] = counts.get(key, 0) + 1
    payload = [[*key, count] for key, count in sorted(counts.items())]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _role_from_entries(role_id: str, label: str, kind: str,
                       entries: list[dict[str, object]]) -> dict[str, object]:
    anchor_l = _weighted_average(entries, "lightness")
    anchor_c = _weighted_average(entries, "chroma")
    anchor_h = _circular_mean(entries) if kind == "chromatic" else 0.0

    def distance(entry: Mapping[str, object]) -> float:
        hue_term = (abs(_circular_delta(float(entry["hue"]), anchor_h)) / 180.0
                    if kind == "chromatic" else 0.0)
        return (float(entry["lightness"]) - anchor_l) ** 2 + (
            float(entry["chroma"]) - anchor_c) ** 2 + hue_term ** 2

    representative = min(entries, key=lambda item: (distance(item), item["hex"]))
    members: list[dict[str, object]] = []
    gradient_ids: set[str] = set()
    properties: set[str] = set()
    for entry in sorted(entries, key=lambda item: item["hex"]):
        gradient_ids.update(entry["gradient_ids"])
        properties.update(entry["channels"])
        member: dict[str, object] = {
            "hex": entry["hex"],
            "usage_count": entry["usage_count"],
            "channels": entry["channels"],
            "gradient_ids": entry["gradient_ids"],
            "relative": {
                "lightness_delta": round(float(entry["lightness"]) - anchor_l, 6),
                "chroma_ratio": round(
                    float(entry["chroma"]) / anchor_c if anchor_c > 1e-7 else 0.0,
                    6,
                ),
                "hue_delta_degrees": round(
                    _circular_delta(float(entry["hue"]), anchor_h), 3,
                ) if kind == "chromatic" else 0.0,
            },
        }
        members.append(member)
    return {
        "id": role_id,
        "label": label,
        "kind": kind,
        "control_count": 1,
        "control": {
            "default_hex": representative["hex"],
            "transform": "oklch-relative-v1",
            "anchor_lightness": round(anchor_l, 6),
            "anchor_chroma": round(anchor_c, 6),
            "anchor_hue_degrees": round(anchor_h, 3) if kind == "chromatic" else None,
        },
        "member_count": len(members),
        "usage_count": sum(int(entry["usage_count"]) for entry in entries),
        "properties": sorted(properties),
        "gradient_ids": sorted(gradient_ids),
        "members": members,
    }


def _build_manifest(path: Path, root: ET.Element) -> dict[str, object]:
    occurrences, gradient_references, unsupported = _collect_occurrences(root)
    inventory: dict[str, dict[str, object]] = {}
    for occurrence in occurrences:
        colour = occurrence.parsed.hex_rgb
        entry = inventory.setdefault(colour, {
            "hex": colour,
            "usage_count": 0,
            "channels": {},
            "gradient_ids": set(),
        })
        entry["usage_count"] = int(entry["usage_count"]) + 1
        channels = entry["channels"]
        channel = f"{occurrence.location}:{occurrence.property_name}"
        channels[channel] = channels.get(channel, 0) + 1
        if occurrence.gradient_id:
            entry["gradient_ids"].add(occurrence.gradient_id)

    prepared: list[dict[str, object]] = []
    for entry in inventory.values():
        lightness, chroma, hue = hex_to_oklch(str(entry["hex"]))
        prepared.append({
            **entry,
            "channels": dict(sorted(entry["channels"].items())),
            "gradient_ids": sorted(entry["gradient_ids"]),
            "lightness": lightness,
            "chroma": chroma,
            "hue": hue,
        })

    neutral_buckets: dict[str, list[dict[str, object]]] = {
        "dark": [], "mid": [], "light": [],
    }
    chromatic: list[dict[str, object]] = []
    for entry in prepared:
        if float(entry["chroma"]) < 0.035:
            lightness = float(entry["lightness"])
            bucket = "dark" if lightness < 0.38 else (
                "light" if lightness >= 0.94 else "mid")
            neutral_buckets[bucket].append(entry)
        else:
            chromatic.append(entry)

    role_specs: list[tuple[str, str, str, list[dict[str, object]]]] = []
    for name in ("dark", "mid", "light"):
        if neutral_buckets[name]:
            role_specs.append((f"neutral-{name}", f"Neutral {name}",
                               "neutral", neutral_buckets[name]))
    chromatic_clusters = _cluster_chromatic(chromatic)
    chromatic_clusters.sort(key=lambda group: (
        -sum(int(entry["usage_count"]) for entry in group),
        round(_circular_mean(group), 6),
        min(str(entry["hex"]) for entry in group),
    ))
    for index, group in enumerate(chromatic_clusters, start=1):
        role_specs.append((f"accent-{index}", f"Accent {index}",
                           "chromatic", group))

    roles = [_role_from_entries(*specification) for specification in role_specs]
    role_by_color = {
        member["hex"]: role["id"]
        for role in roles for member in role["members"]
    }

    gradients: list[dict[str, object]] = []
    for element in root.iter():
        if _local_name(element.tag) not in _GRADIENT_TAGS:
            continue
        gradient_id = element.attrib.get("id") or ""
        stop_colours: list[str] = []
        for descendant in element.iter():
            if _local_name(descendant.tag) != "stop":
                continue
            value = _style_value(descendant, "stop-color")
            if value is None:
                value = descendant.attrib.get("stop-color", "")
            parsed = parse_color(value)
            if parsed is not None:
                stop_colours.append(parsed.hex_rgb)
        gradients.append({
            "id": gradient_id or None,
            "type": _local_name(element.tag),
            "usage_count": gradient_references.get(gradient_id, 0),
            "stop_colors": stop_colours,
            "role_ids": sorted({role_by_color[colour]
                                for colour in stop_colours
                                if colour in role_by_color}),
        })

    solid_colours = {
        occurrence.parsed.hex_rgb for occurrence in occurrences
        if occurrence.property_name in {"fill", "stroke"}
    }
    gradient_stop_colours = {
        occurrence.parsed.hex_rgb for occurrence in occurrences
        if occurrence.property_name == "stop-color"
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "filename": path.name,
            "sha256": _sha256(path),
            "sha256_scope": "input_svg_before_role_annotations",
            "paint_signature": _paint_signature(occurrences),
        },
        "compatibility": {
            "rendering_authority": "explicit SVG presentation attributes",
            "css_required": False,
            "css_custom_properties_required": False,
            "editor_specific_swatches_required": False,
            "annotation_attributes_are_inert": True,
        },
        "resource_counts": {
            "solid_paint_resources": len(solid_colours),
            "gradient_resources": len(gradients),
            "paint_resources_total": len(solid_colours) + len(gradients),
            "unique_gradient_stop_colors": len(gradient_stop_colours),
            "unique_color_tokens": len(inventory),
            "role_controls": len(roles),
            "chromatic_role_controls": sum(role["kind"] == "chromatic"
                                            for role in roles),
        },
        "roles": roles,
        "role_by_color": dict(sorted(role_by_color.items())),
        "gradients": gradients,
        "unsupported_paints": unsupported,
        "scope_note": (
            "Role controls automate colour-family changes while preserving "
            "explicit SVG paints. They do not establish semantic object "
            "grouping or designer time savings."
        ),
    }


def build_paint_role_manifest(svg_path: str | Path) -> dict[str, object]:
    """Build a deterministic role/swatch manifest without changing the SVG."""

    path = Path(svg_path)
    root = ET.parse(path).getroot()
    manifest = _build_manifest(path, root)
    json.dumps(manifest, ensure_ascii=False)
    return manifest


def write_paint_role_manifest(svg_path: str | Path,
                              manifest_path: str | Path) -> dict[str, object]:
    manifest = build_paint_role_manifest(svg_path)
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temp_path.unlink(missing_ok=True)
    return manifest


def _load_manifest(value: Mapping[str, object] | str | Path) -> dict[str, object]:
    if isinstance(value, Mapping):
        manifest = copy.deepcopy(dict(value))
    else:
        manifest = json.loads(Path(value).read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported paint-role manifest schema: {manifest.get('schema')!r}")
    return manifest


def _validate_manifest(root: ET.Element, manifest: Mapping[str, object]) -> None:
    occurrences, _references, _unsupported = _collect_occurrences(root)
    expected = manifest.get("source", {}).get("paint_signature")
    actual = _paint_signature(occurrences)
    if expected != actual:
        raise ValueError(
            "paint-role manifest does not match this SVG's explicit paint inventory"
        )


def _metadata_element(root: ET.Element, metadata_id: str) -> ET.Element | None:
    for child in root:
        if _local_name(child.tag) == "metadata" and child.attrib.get("id") == metadata_id:
            return child
    return None


def _set_metadata(root: ET.Element, metadata_id: str,
                  payload: Mapping[str, object]) -> None:
    element = _metadata_element(root, metadata_id)
    if element is None:
        element = ET.Element(f"{{{SVG_NS}}}metadata", {"id": metadata_id})
        # Keep the tool's main metadata first when it already exists.
        insert_at = (1 if len(root) and
                     _local_name(root[0].tag) == "metadata" else 0)
        root.insert(insert_at, element)
    element.text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _remove_metadata(root: ET.Element, metadata_id: str) -> None:
    element = _metadata_element(root, metadata_id)
    if element is not None:
        root.remove(element)


def _write_tree(tree: ET.ElementTree, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True,
               short_empty_elements=True)


def annotate_svg_with_paint_roles(
        svg_path: str | Path,
        manifest: Mapping[str, object] | str | Path,
        output_path: str | Path) -> dict[str, object]:
    """Write an equivalent SVG with inert role annotations and metadata."""

    loaded = _load_manifest(manifest)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    _validate_manifest(root, loaded)
    role_by_color = loaded.get("role_by_color", {})
    occurrences, _references, _unsupported = _collect_occurrences(root)
    annotated = 0
    role_ids: set[str] = set()
    for occurrence in occurrences:
        role_id = role_by_color.get(occurrence.parsed.hex_rgb)
        if not role_id:
            continue
        key = "data-paint-role-" + occurrence.property_name
        occurrence.element.set(key, str(role_id))
        role_ids.add(str(role_id))
        annotated += 1
    _set_metadata(root, MANIFEST_METADATA_ID, loaded)
    _write_tree(tree, output_path)
    return {
        "status": "ok",
        "annotated_paint_declarations": annotated,
        "role_ids": sorted(role_ids),
        "rendering_attributes_changed": 0,
        "css_required": False,
    }


def _role_mapping(role: Mapping[str, object], target_hex: str,
                  preserve_hue_spread: float = 0.5) -> dict[str, str]:
    parsed_target = parse_color(target_hex)
    if parsed_target is None:
        raise ValueError(f"invalid target RGB colour: {target_hex!r}")
    target_l, target_c, target_h = hex_to_oklch(parsed_target.hex_rgb)
    control = role.get("control", {})
    anchor_l = float(control.get("anchor_lightness", target_l))
    anchor_c = float(control.get("anchor_chroma", target_c))
    kind = role.get("kind")
    mapping: dict[str, str] = {}
    for member in role.get("members", []):
        source = str(member["hex"]).lower()
        relative = member.get("relative", {})
        new_l = _clamp(target_l + float(relative.get("lightness_delta", 0.0)),
                       0.03, 0.99)
        if kind == "chromatic":
            ratio = float(relative.get("chroma_ratio", 1.0))
            # An achromatic target is allowed and intentionally neutralises
            # the whole role.  Otherwise preserve relative saturation.
            new_c = target_c * ratio if target_c > 1e-6 else 0.0
            hue_delta = float(relative.get("hue_delta_degrees", 0.0))
            new_h = (target_h + preserve_hue_spread * hue_delta) % 360.0
        else:
            new_c = target_c * (
                float(relative.get("chroma_ratio", 0.0))
                if anchor_c > 1e-7 else 1.0)
            new_h = target_h
        mapping[source] = oklch_to_hex(new_l, new_c, new_h)
    return mapping


def _set_occurrence_value(occurrence: PaintOccurrence, replacement: str) -> None:
    formatted = _format_replacement(replacement, occurrence.parsed)
    if occurrence.location == "attribute":
        occurrence.element.set(occurrence.property_name, formatted)
    else:
        _set_style_value(occurrence.element, occurrence.property_name, formatted)


def structure_signature(svg_or_root: str | Path | ET.Element) -> str:
    """Hash non-paint structure so a recolour can prove geometry is unchanged."""

    root = (ET.parse(svg_or_root).getroot()
            if isinstance(svg_or_root, (str, Path)) else svg_or_root)
    records: list[object] = []
    ignored_attributes = {
        "fill", "stroke", "stop-color", "class",
        "data-paint-role-fill", "data-paint-role-stroke",
        "data-paint-role-stop_color",
    }
    for element in root.iter():
        tag = _local_name(element.tag)
        if tag == "metadata" and element.attrib.get("id") in {
                MANIFEST_METADATA_ID, RECOLOR_METADATA_ID}:
            continue
        attributes: list[tuple[str, str]] = []
        for name, value in element.attrib.items():
            if name in ignored_attributes or name.startswith("data-paint-role-"):
                continue
            if name == "style":
                declarations = [(key, item) for key, item in _parse_style(value)
                                if key.lower() not in _PAINT_PROPERTIES]
                value = ";".join(f"{key}:{item}" for key, item in declarations)
                if not value:
                    continue
            attributes.append((_local_name(name), value))
        records.append((tag, sorted(attributes)))
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def apply_role_recolor(
        svg_path: str | Path,
        manifest: Mapping[str, object] | str | Path,
        replacements: Mapping[str, str],
        output_path: str | Path,
        *,
        preserve_hue_spread: float = 0.5) -> dict[str, object]:
    """Apply one target colour per role and write ordinary explicit paints."""

    loaded = _load_manifest(manifest)
    tree = ET.parse(svg_path)
    root = tree.getroot()
    _validate_manifest(root, loaded)
    before_structure = structure_signature(root)
    roles = {str(role["id"]): role for role in loaded.get("roles", [])}
    unknown = sorted(set(replacements) - set(roles))
    if unknown:
        raise KeyError(f"unknown paint role(s): {', '.join(unknown)}")

    colour_mapping: dict[str, tuple[str, str]] = {}
    role_summaries: list[dict[str, object]] = []
    for role_id, target in replacements.items():
        mapping = _role_mapping(roles[role_id], target, preserve_hue_spread)
        for source, destination in mapping.items():
            colour_mapping[source] = (destination, role_id)
        role_summaries.append({
            "role_id": role_id,
            "target_hex": parse_color(target).hex_rgb,
            "source_color_count": len(mapping),
            "output_color_count": len(set(mapping.values())),
            "mapping": dict(sorted(mapping.items())),
        })

    occurrences, _references, _unsupported = _collect_occurrences(root)
    changed = 0
    by_property: dict[str, int] = {}
    for occurrence in occurrences:
        replacement = colour_mapping.get(occurrence.parsed.hex_rgb)
        if replacement is None:
            continue
        new_hex, _role_id = replacement
        if new_hex != occurrence.parsed.hex_rgb:
            _set_occurrence_value(occurrence, new_hex)
            changed += 1
            by_property[occurrence.property_name] = (
                by_property.get(occurrence.property_name, 0) + 1)

    _remove_metadata(root, MANIFEST_METADATA_ID)
    operation = {
        "schema": "ai-vector-cleanroom.paint-recolor/v1",
        "base_manifest_paint_signature": loaded["source"]["paint_signature"],
        "roles": role_summaries,
        "changed_declarations": changed,
        "preserve_hue_spread": preserve_hue_spread,
        "rendering_authority": "explicit SVG presentation attributes",
    }
    _set_metadata(root, RECOLOR_METADATA_ID, operation)
    after_structure = structure_signature(root)
    if before_structure != after_structure:
        raise RuntimeError("role recolour changed non-paint SVG structure")
    _write_tree(tree, output_path)
    return {
        "status": "ok",
        "changed_declarations": changed,
        "changed_by_property": dict(sorted(by_property.items())),
        "roles": role_summaries,
        "structure_signature_before": before_structure,
        "structure_signature_after": after_structure,
        "structure_unchanged": True,
        "css_required": False,
        "manifest_regeneration_recommended": True,
    }


def _parse_role_target(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("target must be ROLE=#rrggbb")
    role, colour = value.split("=", 1)
    if not role or parse_color(colour) is None:
        raise argparse.ArgumentTypeError("target must be ROLE=#rrggbb")
    return role, colour


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Build and apply portable SVG paint-role resources")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="write a role manifest")
    analyze.add_argument("svg")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--annotated")

    recolor = subparsers.add_parser("recolor", help="recolour one or more roles")
    recolor.add_argument("svg")
    recolor.add_argument("--manifest", required=True)
    recolor.add_argument("--target", action="append", type=_parse_role_target,
                         required=True)
    recolor.add_argument("--output", required=True)
    recolor.add_argument("--output-manifest")
    recolor.add_argument("--hue-spread", type=float, default=0.5)

    arguments = parser.parse_args()
    if arguments.command == "analyze":
        manifest = write_paint_role_manifest(arguments.svg, arguments.manifest)
        summary: dict[str, object] = {
            "status": "ok", "manifest": arguments.manifest,
            "resource_counts": manifest["resource_counts"],
            "roles": [{"id": role["id"], "members": role["member_count"]}
                      for role in manifest["roles"]],
        }
        if arguments.annotated:
            summary["annotation"] = annotate_svg_with_paint_roles(
                arguments.svg, manifest, arguments.annotated)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    replacements = dict(arguments.target)
    result = apply_role_recolor(
        arguments.svg, arguments.manifest, replacements, arguments.output,
        preserve_hue_spread=arguments.hue_spread,
    )
    if arguments.output_manifest:
        write_paint_role_manifest(arguments.output, arguments.output_manifest)
        result["output_manifest"] = arguments.output_manifest
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "MANIFEST_SCHEMA",
    "annotate_svg_with_paint_roles",
    "apply_role_recolor",
    "build_paint_role_manifest",
    "hex_to_oklch",
    "oklch_to_hex",
    "parse_color",
    "structure_signature",
    "write_paint_role_manifest",
]
