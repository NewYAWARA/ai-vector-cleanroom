# -*- coding: utf-8 -*-
"""Renderer-gated, mathematically exact SVG native-shape substitutions.

This module is intentionally independent from the production post-processing
pipeline.  It recognises only a narrow equivalence that can be proven without
curve fitting: one open, single-subpath path made exclusively from moveto and
straight-line commands.  Such a path can be represented by an SVG ``line`` or
``polyline`` with exactly the same vertices and presentation attributes.

The writer is transactional and deliberately refuses to publish a proposal
unless the caller supplies a renderer result whose pixel arrays are identical.
It does not approximate rectangles, polygons, circles, or ellipses.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping
import xml.etree.ElementTree as ET


SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
NATIVEIZER_VERSION = "exact-native-shapes/0.2"
EXACT_STAGE = "exact_linear_nativeization_exact"
EXACT_PIXEL_EQUIVALENCE = "pixel_array_exact_at_validation_resolution"

ET.register_namespace("", SVG_NS)
ET.register_namespace("inkscape", INKSCAPE_NS)

RenderValidator = Callable[[Path, Path, str], Mapping[str, object] | bool]

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_TOKEN_RE = re.compile(rf"[MmLlHhVvZz]|{_NUMBER}")
_COMMAND_RE = re.compile(r"^[MmLlHhVvZz]$")
_SEPARATOR_RE = re.compile(r"^\s*,?\s*$")
_TRAILING_SPACE_RE = re.compile(r"^\s*$")
_PIXEL_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_URL_FRAGMENT_RE = re.compile(
    r'''url\(\s*(["']?)#([^\s\)"']+)\1\s*\)''', re.IGNORECASE)
_UNSAFE_PATH_ATTRIBUTES = {
    "class", "marker-start", "marker-mid", "marker-end", "pathLength",
    "points", "style", "transform", "x1", "x2", "y1", "y2",
}
_ACTIVE_OR_EMBEDDED_ELEMENTS = {
    "animate", "animateColor", "animateMotion", "animateTransform",
    "discard", "foreignObject", "script", "set", "style",
}


def _local(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temp_path.unlink(missing_ok=True)


def _decimal(value: str) -> Decimal | None:
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _lex_path(data: str) -> list[str] | None:
    tokens: list[str] = []
    position = 0
    for match in _TOKEN_RE.finditer(data or ""):
        if not _SEPARATOR_RE.fullmatch(data[position:match.start()]):
            return None
        tokens.append(match.group(0))
        position = match.end()
    if not _TRAILING_SPACE_RE.fullmatch((data or "")[position:]) or not tokens:
        return None
    return tokens


def _read_numbers(tokens: list[str], index: int, count: int) -> (
        tuple[list[Decimal], int] | None):
    values: list[Decimal] = []
    for _ in range(count):
        if index >= len(tokens) or _COMMAND_RE.fullmatch(tokens[index]):
            return None
        value = _decimal(tokens[index])
        if value is None:
            return None
        values.append(value)
        index += 1
    return values, index


def parse_open_linear_path(data: str) -> tuple[tuple[Decimal, Decimal], ...] | None:
    """Return exact vertices for one open M/L/H/V subpath, else ``None``.

    Relative commands and repeated parameter sets are evaluated with
    ``Decimal`` so a coordinate is never perturbed by binary floating point.
    Closed paths and a second moveto are rejected rather than reinterpreted.
    """

    tokens = _lex_path(data)
    if not tokens:
        return None
    index = 0
    active = ""
    current = (Decimal(0), Decimal(0))
    points: list[tuple[Decimal, Decimal]] = []
    saw_move = False

    while index < len(tokens):
        token = tokens[index]
        if _COMMAND_RE.fullmatch(token):
            active = token
            index += 1
        elif not active:
            return None

        upper = active.upper()
        relative = active.islower()
        if upper == "Z":
            return None

        if upper == "M":
            if saw_move:
                return None
            read = _read_numbers(tokens, index, 2)
            if read is None:
                return None
            values, index = read
            x, y = values
            if relative:
                x += current[0]
                y += current[1]
            current = (x, y)
            points.append(current)
            saw_move = True
            # Extra coordinate pairs after moveto are implicit lineto pairs.
            active = "l" if relative else "L"
            continue

        if not saw_move or upper not in {"L", "H", "V"}:
            return None
        count = 2 if upper == "L" else 1
        read = _read_numbers(tokens, index, count)
        if read is None:
            return None
        values, index = read
        if upper == "L":
            x, y = values
            if relative:
                x += current[0]
                y += current[1]
        elif upper == "H":
            x = values[0] + current[0] if relative else values[0]
            y = current[1]
        else:
            x = current[0]
            y = values[0] + current[1] if relative else values[0]
        current = (x, y)
        points.append(current)

    if len(points) < 2:
        return None
    if any(first == second for first, second in zip(points, points[1:])):
        return None
    return tuple(points)


@dataclass(frozen=True)
class LinearCandidate:
    element_id: str
    native_tag: str
    points: tuple[tuple[Decimal, Decimal], ...]


def _document_is_selector_free(text: str, root: ET.Element) -> bool:
    """Return whether static tag substitution can be audited locally.

    Stylesheets can select ``path`` by element name, while scripts, event
    handlers, animation, and embedded foreign content can change the document
    after the validator's static render.  Refusing those documents prevents a
    pixel-identical snapshot from being mistaken for durable DOM equivalence.
    """

    if "<!DOCTYPE" in text.upper() or "<?xml-stylesheet" in text.lower():
        return False
    for element in root.iter():
        if _local(element.tag) in _ACTIVE_OR_EMBEDDED_ELEMENTS:
            return False
        if any(_local(name).lower().startswith("on")
               for name in element.attrib):
            return False
    return True


def _referenced_ids(root: ET.Element) -> frozenset[str]:
    """Collect URI-fragment targets whose element kind must stay stable.

    Some references (notably ``textPath``/``mpath``) require a real path;
    others may be interpreted differently by authoring software.  A blanket
    refusal is intentionally conservative and still leaves unreferenced
    geometry available for exact nativeization.
    """

    referenced: set[str] = set()
    for element in root.iter():
        for name, raw_value in element.attrib.items():
            value = raw_value.strip()
            if _local(name) == "href" and value.startswith("#") and len(value) > 1:
                referenced.add(value[1:])
            referenced.update(match.group(2)
                              for match in _URL_FRAGMENT_RE.finditer(value))
    return frozenset(referenced)


def _candidate_for(element: ET.Element,
                   parents: Mapping[ET.Element, ET.Element],
                   referenced_ids: frozenset[str] = frozenset(),
                   ) -> LinearCandidate | None:
    if _local(element.tag) != "path" or list(element):
        return None
    if any(attribute in element.attrib for attribute in _UNSAFE_PATH_ATTRIBUTES):
        return None
    if element.get("id") and element.get("id") in referenced_ids:
        return None
    if (element.get("fill") or "").strip().lower() != "none":
        return None
    stroke = (element.get("stroke") or "").strip().lower()
    if not stroke or stroke == "none":
        return None
    ancestor = parents.get(element)
    while ancestor is not None:
        if ancestor.get("transform") or ancestor.get("style"):
            return None
        ancestor = parents.get(ancestor)
    points = parse_open_linear_path(element.get("d", ""))
    if points is None:
        return None
    return LinearCandidate(
        element_id=element.get("id", ""),
        native_tag="line" if len(points) == 2 else "polyline",
        points=points,
    )


def find_exact_linear_candidates(svg_path: str | Path) -> list[LinearCandidate]:
    """Find conservative path-to-line/polyline candidates without writing."""

    source = Path(svg_path)
    text = source.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    if not _document_is_selector_free(text, root):
        return []
    parents = {child: parent for parent in root.iter() for child in parent}
    referenced_ids = _referenced_ids(root)
    candidates = []
    for element in root.iter():
        candidate = _candidate_for(element, parents, referenced_ids)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _structure_counts(root: ET.Element) -> dict[str, int]:
    names = [
        _local(element.tag) for element in root.iter()
        if _local(element.tag) in {
            "path", "circle", "ellipse", "rect", "line", "polyline",
            "polygon",
        }
    ]
    return {name: names.count(name) for name in sorted(set(names))}


def _native_geometry_attributes(candidate: LinearCandidate) -> dict[str, str]:
    if candidate.native_tag == "line":
        (x1, y1), (x2, y2) = candidate.points
        return {
            "x1": _decimal_text(x1),
            "y1": _decimal_text(y1),
            "x2": _decimal_text(x2),
            "y2": _decimal_text(y2),
        }
    return {
        "points": " ".join(
            f"{_decimal_text(x)},{_decimal_text(y)}"
            for x, y in candidate.points
        )
    }


def _replace_candidate(element: ET.Element, candidate: LinearCandidate) -> None:
    namespace = element.tag[:-len("path")] if element.tag.endswith("path") else ""
    element.tag = namespace + candidate.native_tag
    element.attrib.pop("d", None)
    element.attrib.update(_native_geometry_attributes(candidate))


def _exact_guard_result(value: Mapping[str, object] | bool | None) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {
            "accepted": False,
            "reason": "renderer mapping with exact_pixel_array_equal=true is required",
            "validator_return_type": type(value).__name__,
        }
    result = dict(value)
    before_hash = result.get("exact_before_pixel_sha256")
    after_hash = result.get("exact_after_pixel_sha256")
    hash_proof_valid = bool(
        isinstance(before_hash, str)
        and isinstance(after_hash, str)
        and _PIXEL_HASH_RE.fullmatch(before_hash)
        and _PIXEL_HASH_RE.fullmatch(after_hash)
        and before_hash.lower() == after_hash.lower()
    )
    result["pixel_hash_proof_valid"] = hash_proof_valid
    result["accepted"] = bool(
        result.get("accepted")
        and result.get("external_render_check") == "completed"
        and result.get("exact_pixel_array_equal") is True
        and result.get("required_equivalence") == EXACT_PIXEL_EQUIVALENCE
        and hash_proof_valid
    )
    if not result["accepted"]:
        result.setdefault(
            "reason", "renderer did not prove exact pixel-array equivalence")
    return result


def nativeize_exact_linear_paths(
    source_svg: str | Path,
    output_svg: str | Path,
    *,
    validator: RenderValidator | None,
) -> dict[str, object]:
    """Publish an exact native-shape proposal only after pixel equality.

    The source is never modified.  A pre-existing output is also left intact
    when parsing, internal invariants, the validator, or the exact pixel gate
    fails.
    """

    source = Path(source_svg)
    output = Path(output_svg)
    if source.resolve() == output.resolve():
        raise ValueError("source and output must be different files")
    report: dict[str, object] = {
        "schema": NATIVEIZER_VERSION,
        "source": str(source),
        "output": str(output),
        "status": "rejected",
        "output_written": False,
        "safety_policy": "static_unreferenced_open_linear_stroke_paths_only",
        "limitations": [
            "only open single-subpath M/L/H/V stroke paths are eligible",
            "filled, closed, curved, compound, styled, transformed, marked, or referenced paths are retained",
            "documents with stylesheets, scripts, event handlers, animation, doctype, or foreignObject are rejected",
            "pixel equality is proven only at the external validator's stated resolution and renderer",
        ],
    }
    try:
        source_text = source.read_text(encoding="utf-8")
        report["source_sha256"] = _sha256(source)
        root = ET.fromstring(source_text)
    except (OSError, UnicodeError, ET.ParseError) as exc:
        report["reason"] = (
            f"source SVG could not be parsed: {type(exc).__name__}: {exc}"[:300])
        return report
    if not _document_is_selector_free(source_text, root):
        report["reason"] = (
            "stylesheet, active content, doctype, embedded content, or "
            "selector-dependent SVG is unsupported")
        return report

    before_counts = _structure_counts(root)
    parents = {child: parent for parent in root.iter() for child in parent}
    referenced_ids = _referenced_ids(root)
    elements_by_candidate: list[tuple[ET.Element, LinearCandidate]] = []
    for element in root.iter():
        candidate = _candidate_for(element, parents, referenced_ids)
        if candidate is not None:
            elements_by_candidate.append((element, candidate))
    candidates = [candidate for _, candidate in elements_by_candidate]
    report.update({
        "candidate_count": len(candidates),
        "line_count": sum(item.native_tag == "line" for item in candidates),
        "polyline_count": sum(item.native_tag == "polyline" for item in candidates),
        "candidate_ids": [item.element_id for item in candidates],
        "referenced_id_count": len(referenced_ids),
        "uri_fragment_target_count": len(referenced_ids),
        "before_counts": before_counts,
    })
    if not candidates:
        report.update({"status": "no_candidates", "reason": "no exact open linear paths"})
        return report
    if validator is None:
        report["reason"] = "exact renderer proof is required before publishing"
        return report

    before_ids = [element.get("id", "") for element in root.iter()
                  if _local(element.tag) in {
                      "path", "circle", "ellipse", "rect", "line",
                      "polyline", "polygon",
                  }]
    before_candidate_attributes = [
        dict(element.attrib) for element, _candidate in elements_by_candidate]
    for element, candidate in elements_by_candidate:
        _replace_candidate(element, candidate)
    after_counts = _structure_counts(root)
    after_ids = [element.get("id", "") for element in root.iter()
                 if _local(element.tag) in {
                     "path", "circle", "ellipse", "rect", "line",
                     "polyline", "polygon",
                 }]
    presentation_attributes_preserved = all(
        {
            name: value for name, value in element.attrib.items()
            if name not in _native_geometry_attributes(candidate)
        }
        == {name: value for name, value in before.items() if name != "d"}
        for (element, candidate), before
        in zip(elements_by_candidate, before_candidate_attributes)
    )
    native_geometry_exact = all(
        all(element.get(name) == value for name, value in
            _native_geometry_attributes(candidate).items())
        for element, candidate in elements_by_candidate
    )
    internal_checks = {
        "drawable_order_and_ids_preserved": before_ids == after_ids,
        "drawable_count_preserved": len(before_ids) == len(after_ids),
        "candidate_non_geometry_attributes_preserved": (
            presentation_attributes_preserved),
        "candidate_native_geometry_exact": native_geometry_exact,
        "path_delta_exact": (
            after_counts.get("path", 0)
            == before_counts.get("path", 0) - len(candidates)
        ),
        "native_delta_exact": (
            after_counts.get("line", 0)
            == before_counts.get("line", 0) + report["line_count"]
            and after_counts.get("polyline", 0)
            == before_counts.get("polyline", 0) + report["polyline_count"]
        ),
    }
    report["after_counts"] = after_counts
    report["internal_checks"] = internal_checks
    if not all(internal_checks.values()):
        report["reason"] = "internal structure invariant failed"
        return report

    with tempfile.TemporaryDirectory(prefix="exact-native-shapes-") as directory:
        proposal = Path(directory) / "proposal.svg"
        ET.ElementTree(root).write(
            proposal, encoding="utf-8", xml_declaration=True)
        try:
            guard_value = validator(source, proposal, EXACT_STAGE)
        except Exception as exc:
            guard_value = {
                "accepted": False,
                "external_render_check": "error",
                "reason": f"validator raised {type(exc).__name__}: {exc}"[:300],
            }
        guard = _exact_guard_result(guard_value)
        report["render_guard"] = guard
        if not guard.get("accepted"):
            report["reason"] = str(guard.get("reason", "exact renderer guard rejected"))
            return report
        _atomic_write(output, proposal.read_bytes())

    report.update({
        "status": "applied",
        "output_written": True,
        "output_sha256": _sha256(output),
        "required_equivalence": EXACT_PIXEL_EQUIVALENCE,
    })
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert exact open linear paths to native line/polyline elements.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    from vector_cleanroom import validate_svg_stage_renders

    report = nativeize_exact_linear_paths(
        args.source, args.output, validator=validate_svg_stage_renders)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        _atomic_write(args.report, (rendered + "\n").encode("utf-8"))
    print(rendered)
    return 0 if report.get("status") == "applied" else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "EXACT_PIXEL_EQUIVALENCE", "EXACT_STAGE", "LinearCandidate",
    "NATIVEIZER_VERSION",
    "find_exact_linear_candidates", "nativeize_exact_linear_paths",
    "parse_open_linear_path",
]
