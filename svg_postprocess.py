"""Transactional SVG editability enhancements.

Each stage is conservative and independently reversible.  The caller may
provide a renderer-backed validator; internal geometry/ordering invariants
remain mandatory even when optional rendering packages are unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping
import xml.etree.ElementTree as ET


RenderValidator = Callable[[Path, Path, str], Mapping[str, object] | bool]
_DRAWABLES = {"path", "circle", "rect", "ellipse", "line", "polyline", "polygon"}
_NATIVES = _DRAWABLES - {"path"}
_STROKE_ID = re.compile(r"^(?:stroke-\d+|annulus-)")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def atomic_replace_bytes(target: str | Path, data: bytes) -> None:
    """Durably replace one file without exposing a half-written target."""

    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temp_path.unlink(missing_ok=True)


def _guard_result(value: Mapping[str, object] | bool | None,
                  *, internal_only_reason: str) -> dict[str, object]:
    if value is None:
        return {
            "accepted": True,
            "external_render_check": "unavailable",
            "validation_level": "internal_invariants",
            "reason": internal_only_reason,
        }
    if isinstance(value, Mapping):
        result = dict(value)
        result["accepted"] = bool(result.get("accepted"))
        result.setdefault("external_render_check", "completed")
        result.setdefault("validation_level", "renderer_and_internal_invariants")
        return result
    return {
        "accepted": bool(value),
        "external_render_check": "completed",
        "validation_level": "renderer_and_internal_invariants",
    }


def _validate_files(validator: RenderValidator | None,
                    before: Path, after: Path, stage: str,
                    internal_only_reason: str) -> dict[str, object]:
    if validator is None:
        return _guard_result(None, internal_only_reason=internal_only_reason)
    try:
        return _guard_result(validator(before, after, stage),
                             internal_only_reason=internal_only_reason)
    except Exception as exc:
        return {
            "accepted": False,
            "external_render_check": "error",
            "validation_level": "renderer_failed",
            "reason": f"render validator raised {type(exc).__name__}: {exc}"[:300],
        }


def _text_validator(validator: RenderValidator | None, directory: Path,
                    stage: str, captures: list[dict[str, object]]):
    def validate(before_text: str, after_text: str) -> bool:
        before = directory / f"{stage}-before.svg"
        after = directory / f"{stage}-after.svg"
        before.write_text(before_text, encoding="utf-8")
        after.write_text(after_text, encoding="utf-8")
        result = _validate_files(
            validator, before, after, stage,
            "stage invariants passed; optional full SVG renderer unavailable",
        )
        captures.append(result)
        return bool(result.get("accepted"))
    return validate


def _path_anchors(data: str) -> int:
    # The audit parser already handles repeated parameter sets and implicit
    # lineto commands.  Keeping one implementation prevents report drift.
    from editability_audit import _path_metrics
    return int(_path_metrics(data)[2])


def _points_count(value: str) -> int:
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value or "")
    return len(numbers) // 2


def measure_svg_structure(svg_path: str | Path) -> dict[str, object]:
    """Measure final delivered DOM rather than stale tracer bookkeeping."""

    root = ET.parse(svg_path).getroot()
    drawables: list[ET.Element] = []
    groups: list[ET.Element] = []
    gradients = 0

    def walk(element: ET.Element, in_defs: bool = False) -> None:
        nonlocal gradients
        tag = _local(element.tag)
        now_defs = in_defs or tag == "defs"
        if tag in {"linearGradient", "radialGradient"}:
            gradients += 1
        if not now_defs:
            if tag == "g":
                groups.append(element)
            if tag in _DRAWABLES:
                drawables.append(element)
        for child in element:
            walk(child, now_defs)

    walk(root)
    paths = [item for item in drawables if _local(item.tag) == "path"]
    native = [item for item in drawables if _local(item.tag) in _NATIVES]
    breakdown = {tag: sum(_local(item.tag) == tag for item in native)
                 for tag in sorted(_NATIVES)}
    nodes = sum(_path_anchors(item.get("d", "")) for item in paths)
    for item in native:
        tag = _local(item.tag)
        if tag == "rect":
            nodes += 4
        elif tag == "line":
            nodes += 2
        elif tag in {"polygon", "polyline"}:
            nodes += max(1, _points_count(item.get("points", "")))
        else:  # circle / ellipse are one native editor object
            nodes += 1
    ids = [item.get("id") for item in drawables if item.get("id")]
    rebuilt_strokes = sum(bool(_STROKE_ID.match(item.get("id", "")))
                          for item in drawables)
    semantic_groups = [item for item in groups
                       if item.get("data-group-mode") == "actual-dom"]
    return {
        "paths": len(paths),
        "native_primitives": len(native),
        "native_circles": breakdown.get("circle", 0),
        "native_rectangles": breakdown.get("rect", 0),
        "native_ellipses": breakdown.get("ellipse", 0),
        "native_lines": breakdown.get("line", 0),
        "native_polylines": breakdown.get("polyline", 0),
        "native_polygons": breakdown.get("polygon", 0),
        "native_polygonal_shapes": (breakdown.get("polygon", 0)
                                    + breakdown.get("polyline", 0)),
        "strokes": rebuilt_strokes,
        "gradients": gradients,
        "nodes": nodes,
        "nodes_total": nodes,
        "drawables": len(drawables),
        "groups": len(groups),
        "semantic_groups": len(semantic_groups),
        "object_ids": len(ids),
        "object_id_coverage": round(len(ids) / len(drawables), 6)
        if drawables else 0.0,
        "count_source": "final_svg_dom",
    }


def enhance_svg_structure(svg_path: str | Path, *,
                          validator: RenderValidator | None = None,
                          work_dir: str | Path | None = None,
                          enable_annulus: bool = True,
                          enable_native_shapes: bool = True,
                          enable_compound_paths: bool = True,
                          enable_scene_graph: bool = True,
                          ) -> dict[str, object]:
    """Apply native geometry, compound-path and scene stages transactionally."""

    target = Path(svg_path)
    before_structure = measure_svg_structure(target)
    stages: dict[str, object] = {}
    base_dir = Path(work_dir) if work_dir else target.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="svg-postprocess-", dir=base_dir) as temp:
        scratch = Path(temp)

        # 1. Large co-circular stroke fragments -> one native dashed circle.
        try:
            if not enable_annulus:
                stages["annulus"] = {"status": "disabled"}
                raise StopIteration
            from annulus_detector import apply_candidate, detect_svg_annuli
            detected = detect_svg_annuli(target)
            annulus_records = []
            for index, candidate in enumerate(detected, 1):
                record = candidate.to_dict()
                record["status"] = "rejected_internal_gate"
                if not candidate.safe_to_replace:
                    annulus_records.append(record)
                    continue
                proposal = scratch / f"annulus-{index}.svg"
                try:
                    apply_candidate(target, candidate, proposal)
                    guard = _validate_files(
                        validator, target, proposal, "annulus_approximate",
                        "shared-circle geometry and bidirectional 1px raster gate passed",
                    )
                    record["render_guard"] = guard
                    if guard.get("accepted"):
                        atomic_replace_bytes(target, proposal.read_bytes())
                        record["status"] = "applied"
                    else:
                        record["status"] = "rolled_back_render_guard"
                except Exception as exc:
                    record["status"] = "rolled_back_error"
                    record["error"] = f"{type(exc).__name__}: {exc}"[:300]
                annulus_records.append(record)
            stages["annulus"] = {
                "status": ("applied" if any(item["status"] == "applied"
                                             for item in annulus_records)
                           else "no_change"),
                "detected_candidates": len(detected),
                "applied_candidates": sum(item["status"] == "applied"
                                           for item in annulus_records),
                "candidates": annulus_records,
            }
        except StopIteration:
            pass
        except Exception as exc:
            stages["annulus"] = {
                "status": "rolled_back_error",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            }

        # 2. Disjoint compound-path families -> separately selectable paths.
        try:
            if not enable_compound_paths:
                stages["compound_paths"] = {"status": "disabled"}
                raise StopIteration
            from compound_path_splitter import process_compound_paths
            captures: list[dict[str, object]] = []
            compound_before = measure_svg_structure(target)
            text = target.read_text(encoding="utf-8")
            result = process_compound_paths(
                text,
                validator=_text_validator(
                    validator, scratch, "compound_exact", captures),
            )
            if result.changed:
                atomic_replace_bytes(target, result.svg_text.encode("utf-8"))
            compound_report = {
                "status": result.status,
                **result.report,
                "render_guards": captures,
            }
            if not result.changed and result.status == "rolled_back":
                # The splitter keeps proposed counts as audit evidence.  The
                # pipeline's top-level counts must describe only committed DOM.
                compound_report["attempted"] = dict(result.report)
                committed_input_paths = result.report.get(
                    "input_paths", compound_before["paths"])
                committed_input_subpaths = result.report.get(
                    "input_subpaths", 0)
                compound_report.update({
                    "input_paths": committed_input_paths,
                    "output_paths": committed_input_paths,
                    "input_subpaths": committed_input_subpaths,
                    "output_subpaths": committed_input_subpaths,
                    "source_paths_split": 0,
                    "split_paths": 0,
                    "new_paths_added": 0,
                    "selectable_path_delta": 0,
                    "subpaths_redistributed": 0,
                    "paths": [],
                })
            stages["compound_paths"] = compound_report
        except StopIteration:
            pass
        except Exception as exc:
            stages["compound_paths"] = {
                "status": "rolled_back_error",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            }

        # 3. Cross-paint spatial fragments -> safe actual DOM object groups.
        try:
            if not enable_scene_graph:
                stages["scene_graph"] = {"status": "disabled"}
                raise StopIteration
            from scene_graph_postprocess import build_scene_graph
            captures = []
            text = target.read_text(encoding="utf-8")
            result = build_scene_graph(
                text,
                validator=_text_validator(
                    validator, scratch, "scene_graph_exact", captures),
            )
            if result.changed:
                atomic_replace_bytes(target, result.svg_text.encode("utf-8"))
            stages["scene_graph"] = {
                "status": result.status,
                **result.report,
                "render_guards": captures,
            }
        except StopIteration:
            pass
        except Exception as exc:
            stages["scene_graph"] = {
                "status": "rolled_back_error",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            }

        # 4. Exact open linear stroke paths -> native line/polyline objects.
        # Run this after scene grouping.  That stage materializes inherited
        # paint attributes on drawable leaves, so eligible tracer strokes are
        # not silently missed.  Publication still requires positive external
        # pixel-array equality in addition to the structural invariants.
        try:
            if not enable_native_shapes:
                stages["exact_native_shapes"] = {
                    "status": "disabled",
                    "committed_candidate_count": 0,
                    "committed_line_count": 0,
                    "committed_polyline_count": 0,
                }
                raise StopIteration
            from exact_native_shapes import nativeize_exact_linear_paths
            proposal = scratch / "exact-native-shapes.svg"
            native_report = nativeize_exact_linear_paths(
                target, proposal, validator=validator)
            native_report["source"] = target.name
            native_report["proposal_scope"] = "transaction_temp_not_delivered"
            native_report.pop("output", None)
            committed = False
            if (native_report.get("status") == "applied"
                    and native_report.get("output_written")
                    and proposal.is_file()):
                try:
                    atomic_replace_bytes(target, proposal.read_bytes())
                    committed = True
                except Exception as exc:
                    native_report = {
                        "schema": native_report.get("schema"),
                        "status": "rolled_back_error",
                        "reason": f"{type(exc).__name__}: {exc}"[:300],
                        "candidate_count": native_report.get("candidate_count", 0),
                        "committed_candidate_count": 0,
                        "committed_line_count": 0,
                        "committed_polyline_count": 0,
                        "attempted": native_report,
                    }
            native_report.setdefault("committed_candidate_count", (
                int(native_report.get("candidate_count", 0) or 0)
                if committed else 0))
            native_report.setdefault("committed_line_count", (
                int(native_report.get("line_count", 0) or 0)
                if committed else 0))
            native_report.setdefault("committed_polyline_count", (
                int(native_report.get("polyline_count", 0) or 0)
                if committed else 0))
            native_report["committed"] = committed
            stages["exact_native_shapes"] = native_report
        except StopIteration:
            pass
        except Exception as exc:
            stages["exact_native_shapes"] = {
                "status": "rolled_back_error",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
                "committed_candidate_count": 0,
                "committed_line_count": 0,
                "committed_polyline_count": 0,
                "committed": False,
            }

    after_structure = measure_svg_structure(target)
    return {
        "schema": "ai-vector-cleanroom.editability-enhancements/v1",
        "stages": stages,
        "structure_before": before_structure,
        "structure_after": after_structure,
        "scope_note": (
            "These stages improve native geometry, selection and grouping. "
            "They do not recover original fonts/layers or prove designer time savings."
        ),
    }


def attach_paint_roles(svg_path: str | Path, manifest_path: str | Path, *,
                       validator: RenderValidator | None = None,
                       work_dir: str | Path | None = None
                       ) -> tuple[dict[str, object], dict[str, object]]:
    """Write a sidecar role manifest and inert annotations transactionally."""

    from paint_roles import (annotate_svg_with_paint_roles,
                             write_paint_role_manifest)

    target = Path(svg_path)
    manifest_file = Path(manifest_path)
    manifest = write_paint_role_manifest(target, manifest_file)
    base_dir = Path(work_dir) if work_dir else target.parent
    with tempfile.TemporaryDirectory(prefix="paint-roles-", dir=base_dir) as temp:
        proposal = Path(temp) / "annotated.svg"
        annotation = annotate_svg_with_paint_roles(
            target, manifest, proposal)
        guard = _validate_files(
            validator, target, proposal, "paint_roles_exact",
            "only inert data attributes/metadata were added; explicit paints unchanged",
        )
        if guard.get("accepted"):
            atomic_replace_bytes(target, proposal.read_bytes())
            status = "applied"
            annotation_committed = True
        else:
            # The sidecar manifest remains a complete, portable recolour
            # resource even if inert DOM annotations are conservatively
            # rejected.  Name that state explicitly instead of implying that
            # the controls were lost or that the proposal was committed.
            status = "manifest_only"
            annotation = dict(annotation)
            annotation["attempted_status"] = annotation.get("status")
            annotation["status"] = "proposal_only_not_committed"
            annotation_committed = False
    report = {
        "status": status,
        "manifest_file": manifest_file.name,
        "resource_counts": manifest.get("resource_counts", {}),
        "roles": [{
            "id": role.get("id"),
            "label": role.get("label"),
            "kind": role.get("kind"),
            "default_hex": (role.get("control") or {}).get("default_hex"),
            "member_count": role.get("member_count"),
            "usage_count": role.get("usage_count"),
        } for role in manifest.get("roles", [])],
        "annotation": annotation,
        "manifest_committed": True,
        "annotation_committed": annotation_committed,
        "render_guard": guard,
        "rendering_authority": "explicit SVG presentation attributes",
    }
    json.dumps(report, ensure_ascii=False)
    return manifest, report


__all__ = [
    "RenderValidator", "atomic_replace_bytes", "attach_paint_roles", "enhance_svg_structure",
    "measure_svg_structure",
]
