"""Deterministic design fixtures and structural benchmark evidence.

Generate references with the product's real resvg renderer. Conversion is a
separate explicit step so fixture generation never runs an older converter.
These synthetic cases measure known design structure, not human editing time.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from svg_renderer import render_svg_reference, renderer_info

NS = "http://www.w3.org/2000/svg"
DRAWABLE = {"path", "circle", "ellipse", "rect", "line", "polyline", "polygon"}
SVG_HEAD = f'<svg xmlns="{NS}" width="256" height="256" viewBox="0 0 256 256">'


def _case(identifier, label, bodies, *, defs="", holes=0, properties=(), ambiguity="none"):
    wrapped = []
    for index, body in enumerate(bodies, 1):
        wrapped.append(f'<g id="object-{index:02d}">{body}</g>')
    return {"id": identifier, "label": label, "svg": SVG_HEAD + defs + "".join(wrapped) + "</svg>",
            "expected_objects": len(bodies), "expected_holes": holes,
            "required_properties": list(properties), "inference_limit": ambiguity}


def fixture_cases():
    """Fourteen deliberately small, independently inspectable reference designs."""
    return [
        _case("01_circle", "Circle", ['<circle cx="128" cy="128" r="78" fill="#14624d"/>'],
              properties=["one native circle", "one independently selectable object"]),
        _case("02_ellipse", "Rotated ellipse", ['<ellipse cx="128" cy="128" rx="88" ry="48" transform="rotate(-25 128 128)" fill="#286e9b"/>'],
              properties=["one editable ellipse", "rotation preserved"]),
        _case("03_round_rect", "Rounded rectangle", ['<rect x="40" y="62" width="176" height="132" rx="24" fill="#bd583a"/>'],
              properties=["four consistent corner radii", "one shape"]),
        _case("04_sharp_corners", "Sharp concave corners", ['<path d="M128 28 L152 94 L224 96 L167 140 L188 214 L128 171 L68 214 L89 140 L32 96 L104 94 Z" fill="#c29220"/>'],
              properties=["ten sharp corners", "concave notches remain open"]),
        _case("05_compound_holes", "Compound three holes", ['<path fill="#24604c" fill-rule="evenodd" d="M34 38 H222 V218 H34 Z M64 69 H110 V112 H64 Z M145 143 H192 V188 H145 Z M176 80 A7 7 0 1 0 162 80 A7 7 0 1 0 176 80 Z"/>'],
              holes=3, properties=["three holes", "one compound editable object", "small round hole must not replace exterior"],
              ambiguity="White-composited input cannot uniquely distinguish transparent holes from white paint; both visual and alpha topology are measured."),
        _case("06_thin_lines", "Thin strokes", [
            '<path d="M36 64 H220" fill="none" stroke="#26394c" stroke-width="2" stroke-linecap="round"/>',
            '<path d="M36 126 H220" fill="none" stroke="#26394c" stroke-width="4" stroke-linecap="round"/>',
            '<path d="M36 188 H220" fill="none" stroke="#26394c" stroke-width="6" stroke-linecap="round"/>'],
              properties=["three independent strokes", "2 4 6 unit widths", "rounded caps"]),
        _case("07_uniform_stroke", "Uniform curved stroke", ['<path d="M36 168 C70 32 174 36 220 150" fill="none" stroke="#286e9b" stroke-width="16" stroke-linecap="round"/>'],
              properties=["one editable centreline", "uniform width", "two endpoint anchors"]),
        _case("08_variable_width", "Variable width silhouette", ['<path d="M32 171 C82 109 143 78 221 58 C171 106 110 170 32 171 Z" fill="#713f80"/>'],
              properties=["taper preserved", "filled shape remains editable"],
              ambiguity="Raster silhouette does not identify a unique original brush or centreline."),
        _case("09_linear_gradient", "Linear gradient", ['<rect x="36" y="56" width="184" height="144" rx="18" fill="url(#linear)"/>'],
              defs='<defs><linearGradient id="linear" x1="36" y1="0" x2="220" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#176258"/><stop offset="1" stop-color="#e3b340"/></linearGradient></defs>',
              properties=["one native linear gradient", "two stops", "no palette-band fragments"]),
        _case("10_radial_gradient", "Radial gradient", ['<circle cx="128" cy="128" r="86" fill="url(#radial)"/>'],
              defs='<defs><radialGradient id="radial" cx="0.42" cy="0.38" r="0.64"><stop offset="0" stop-color="#f2d27a"/><stop offset="1" stop-color="#b43f32"/></radialGradient></defs>',
              properties=["one native radial gradient", "one circle", "off-centre colour field"]),
        _case("11_touching_colours", "Touching colours", [
            '<path d="M36 52 H128 V204 H36 Z" fill="#23695a"/>',
            '<path d="M128 52 H220 V204 H128 Z" fill="#d39533"/>'],
              properties=["two independent colour objects", "shared edge has no gap", "recolour either side"]),
        _case("12_occlusion", "Occlusion", [
            '<circle cx="105" cy="126" r="75" fill="#397d9d"/>',
            '<rect x="114" y="78" width="109" height="129" rx="14" fill="#ba533b"/>'],
              properties=["two selectable objects", "correct stacking"],
              ambiguity="The hidden full circle cannot be uniquely inferred from the raster; visible geometry is the automatic target."),
        _case("13_intentional_details", "Intentional tiny details", [
            '<path d="M39 85 H195 V144 C195 187 156 209 117 209 C78 209 39 187 39 144 Z" fill="#2e6850"/>',
            '<circle cx="69" cy="48" r="3" fill="#2e6850"/>',
            '<circle cx="116" cy="44" r="5" fill="#2e6850"/>',
            '<path d="M165 32 V57" stroke="#2e6850" stroke-width="4" stroke-linecap="round"/>'],
              properties=["four intentional objects", "3-unit radius dot survives", "short stroke survives"],
              ambiguity="Detail versus compression noise needs source context; the synthetic reference declares these marks intentional."),
        _case("14_symmetric_logo", "Symmetric logo", [
            '<path d="M128 33 C82 40 45 83 48 133 C51 181 89 214 128 223 L128 184 C103 174 85 151 86 126 C87 99 104 80 128 73 Z" fill="#285d76"/>',
            '<path d="M128 33 C174 40 211 83 208 133 C205 181 167 214 128 223 L128 184 C153 174 171 151 170 126 C169 99 152 80 128 73 Z" fill="#b7832f"/>'],
              holes=1, properties=["mirror geometry about x=128", "two independently recolourable halves", "central opening"]),
    ]


def adversarial_cases():
    """Source-authoritative counterexamples absent from the original 28 rasters."""
    return [
        _case("15_mixed_caps", "Butt, round and square ends", [
            f'<path d="M36 {y} H220" fill="none" stroke="#26394c" stroke-width="12" stroke-linecap="{cap}"/>'
            for y, cap in ((60, "butt"), (128, "round"), (196, "square"))],
            properties=["three editable strokes or economical equivalent shapes", "flat ends remain flat", "rounded end stays rounded"],
            ambiguity="A square-capped line can equal a longer butt-capped line; compare visible ends, not the unknowable original cap label."),
        _case("16_narrow_channels", "Separate straight bars and narrow gaps", [
            f'<rect x="{x}" y="{y}" width="60" height="24" fill="#164d34"/>'
            for x, gap in ((22, 2), (98, 4), (174, 8)) for y in (90, 114 + gap)],
            properties=["six separate bars", "three open paper channels", "no invented gradient", "square corners"]),
        _case("17_true_micro_holes", "Intentional small holes", [
            '<path fill="#24604c" fill-rule="evenodd" d="M32 40 H224 V216 H32 Z '
            'M67 93 A2 2 0 1 0 63 93 A2 2 0 1 0 67 93 Z '
            'M132 128 A4 4 0 1 0 124 128 A4 4 0 1 0 132 128 Z '
            'M200 168 A8 8 0 1 0 184 168 A8 8 0 1 0 200 168 Z"/>'],
            holes=3, properties=["all three source-visible holes survive", "small holes cannot be discarded as noise by size alone"],
            ambiguity="The opaque raster does not distinguish white paint from transparency; preserve the visible holes either way."),
        _case("18_subtle_gradient", "Real low contrast colour ramp", [
            '<rect x="40" y="56" width="176" height="144" fill="url(#subtle)"/>'],
            defs='<defs><linearGradient id="subtle" x1="40" y1="0" x2="216" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#527a64"/><stop offset="1" stop-color="#638871"/></linearGradient></defs>',
            properties=["genuine continuous ramp is not flattened into one colour", "no palette-band fragments"]),
    ]


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def code_snapshot():
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(CODE_DIR.glob("*.py"))}
    for path in (Path(__file__), Path(__file__).with_name("test_designer_benchmark.py")):
        files["tests/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"aggregate_sha256": digest, "files": files, "python": sys.version,
            "renderer": renderer_info(), "scope": "product_root_python_and_benchmark_source_files"}


def svg_structure(svg):
    """Measure SVG representation without calling raster similarity editability."""
    root = ET.fromstring(svg)
    parents = {child: parent for parent in root.iter() for child in parent}

    def inherited(node, name, default=""):
        while node is not None:
            if name in node.attrib:
                return node.get(name)
            node = parents.get(node)
        return default

    drawables = [node for node in root.iter() if _local(node.tag) in DRAWABLE]
    counts = Counter(_local(node.tag) for node in drawables)
    path_commands = Counter()
    for node in drawables:
        if _local(node.tag) == "path":
            path_commands.update(command.upper() for command in re.findall(r"[AaCcHhLlMmQqSsTtVvZz]", node.get("d", "")))
    object_groups = [node for node in root.iter() if _local(node.tag) == "g" and node.get("id", "").startswith("object-")]
    top_objects = [node for node in object_groups if not any(ancestor in object_groups for ancestor in _ancestors(node, parents))]
    owned = set(node for group in top_objects for node in group.iter() if _local(node.tag) in DRAWABLE)
    selection_units = len(top_objects) + sum(node not in owned for node in drawables)
    gradients = [node for node in root.iter() if _local(node.tag) in {"linearGradient", "radialGradient"}]
    strokes = [node for node in drawables if inherited(node, "stroke", "none") != "none"]
    return {"drawable_count": len(drawables), "path_count": counts["path"],
            "native_primitive_count": sum(counts[k] for k in DRAWABLE - {"path"}),
            "element_counts": dict(sorted(counts.items())), "explicit_path_command_count": sum(path_commands.values()),
            "path_command_counts": dict(sorted(path_commands.items())),
            "stroke_count": len(strokes), "stroke_widths": sorted({inherited(n, "stroke-width", "1") for n in strokes}),
            "linear_gradient_count": sum(_local(g.tag) == "linearGradient" for g in gradients),
            "radial_gradient_count": sum(_local(g.tag) == "radialGradient" for g in gradients),
            "gradient_stop_count": sum(_local(n.tag) == "stop" for g in gradients for n in g),
            "object_group_count": len(top_objects), "selection_unit_count": selection_units,
            "selection_unit_scope": "outer_object_groups_plus_ungrouped_drawables_not_semantic_correspondence",
            "node_metric_scope": "explicit_svg_commands_not_control_points_or_designer_handles"}


def _ancestors(node, parents):
    node = parents.get(node)
    while node is not None:
        yield node
        node = parents.get(node)


def _selection_masks(svg_path, scratch_dir, width):
    """Render one handoff-style unit at a time; compare only visible ownership."""
    root = ET.fromstring(Path(svg_path).read_text(encoding="utf-8"))
    parents = {child: parent for parent in root.iter() for child in parent}
    resources = {"defs", "clipPath", "mask", "marker", "pattern", "symbol"}
    visible = [n for n in root.iter() if _local(n.tag) in DRAWABLE and not any(_local(a.tag) in resources for a in _ancestors(n, parents))]
    groups = [n for n in root.iter() if _local(n.tag) == "g" and n.get("id", "").startswith("object-")]
    groups = [n for n in groups if not any(a in groups for a in _ancestors(n, parents))]
    owned = set(n for g in groups for n in g.iter())
    units = set(groups + [n for n in visible if n not in owned])
    ordered = [n for n in root.iter() if n in units]
    if len(ordered) > 200:
        return None
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    original_nodes = list(root.iter())
    masks = []
    for index, unit in enumerate(ordered):
        chosen = set(unit.iter())
        clone = copy.deepcopy(root)
        for original, node in zip(original_nodes, clone.iter()):
            if original in visible and original not in chosen:
                node.set("display", "none")
        svg = scratch_dir / f"unit-{index:03d}.svg"
        png = scratch_dir / f"unit-{index:03d}.png"
        svg.write_text(ET.tostring(clone, encoding="unicode"), encoding="utf-8")
        render_svg_reference(svg, png, width=width, background=None)
        masks.append(np.asarray(Image.open(png).convert("RGBA"))[:, :, 3] >= 128)
    later = np.zeros((width, width), dtype=bool)
    for index in range(len(masks) - 1, -1, -1):
        full = masks[index]
        masks[index] = full & ~later
        later |= full
    return masks


def selection_correspondence(reference_svg, output_svg, scratch_dir, width):
    """A selection coverage proxy, deliberately separate from visual quality."""
    reference = _selection_masks(reference_svg, Path(scratch_dir) / "reference", width)
    output = _selection_masks(output_svg, Path(scratch_dir) / "output", width)
    if output is None or reference is None:
        return {"status": "not_measured_too_many_units", "maximum_units": 200}
    details = []
    for index, expected in enumerate(reference):
        area = int(expected.sum())
        ranked = []
        for output_index, actual in enumerate(output):
            intersection = int((expected & actual).sum())
            coverage = intersection / area if area else 0.0
            precision = intersection / int(actual.sum()) if actual.any() else 0.0
            ranked.append((min(coverage, precision), coverage, precision, output_index))
        _, coverage, precision, output_index = max(ranked, default=(0.0, 0.0, 0.0, -1))
        details.append({"reference_object": index + 1, "best_output_unit": output_index + 1,
                        "visible_area_px": area, "visible_coverage": round(coverage, 6),
                        "selection_precision": round(precision, 6),
                        "one_unit_matches_visible_object": area > 0 and coverage >= 0.95 and precision >= 0.95})
    return {"status": "measured", "scope": "opaque_visible_pixel_ownership_selection_proxy_not_semantic_restoration",
            "reference_units": len(reference), "output_units": len(output),
            "matched_reference_objects": sum(d["one_unit_matches_visible_object"] for d in details),
            "coverage_threshold": 0.95, "precision_threshold": 0.95,
            "alpha_threshold": 128, "objects": details}


def raster_topology(png_path, *, mode="contrast"):
    """8-connected background holes and 4-connected foreground components."""
    rgba = np.asarray(Image.open(png_path).convert("RGBA"))
    rgb = rgba[:, :, :3].astype(float) * (rgba[:, :, 3:4] / 255) + 255 * (1 - rgba[:, :, 3:4] / 255)
    if mode not in {"contrast", "alpha"}:
        raise ValueError("Unknown topology mode")
    foreground = np.max(255 - rgb, axis=2) > 32 if mode == "contrast" else rgba[:, :, 3] >= 128

    def regions(mask, diagonal):
        seen = np.zeros(mask.shape, dtype=bool)
        height, width = mask.shape
        count = interior = 0
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if diagonal:
            offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for y, x in zip(*np.nonzero(mask)):
            if seen[y, x]:
                continue
            count += 1
            seen[y, x] = True
            queue = deque([(int(y), int(x))])
            touches_edge = False
            while queue:
                yy, xx = queue.popleft()
                touches_edge |= yy == 0 or xx == 0 or yy == height - 1 or xx == width - 1
                for dy, dx in offsets:
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            interior += not touches_edge
        return count, interior
    components, _ = regions(foreground, False)
    _, holes = regions(~foreground, True)
    return {"foreground_components": components, "holes": holes,
            "foreground_pixels": int(foreground.sum()), "threshold_from_white": 32,
            "alpha_threshold": 128, "mode": mode, "connectivity": "foreground_4_background_8"}


def generate(destination, widths=(128, 384), *, suite="standard"):
    destination = Path(destination)
    source_dir, input_dir = destination / "groundtruth", destination / "inputs"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    records = []
    cases = {"standard": fixture_cases, "adversarial": adversarial_cases}.get(suite)
    if cases is None:
        raise ValueError("unknown benchmark suite")
    for case in cases():
        path = source_dir / (case["id"] + ".svg")
        path.write_text(case["svg"], encoding="utf-8")
        record = {key: value for key, value in case.items() if key != "svg"}
        record["svg"] = path.relative_to(destination).as_posix()
        record["svg_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["structure"] = svg_structure(case["svg"])
        record["rasters"] = []
        for width in widths:
            png = input_dir / f'{case["id"]}_{width}px.png'
            render = render_svg_reference(path, png, width=int(width), background="#ffffff")
            topology = raster_topology(png)
            alpha_dir = destination / "groundtruth-alpha-renders"
            alpha_dir.mkdir(exist_ok=True)
            alpha_png = alpha_dir / png.name
            alpha_render = render_svg_reference(path, alpha_png, width=int(width), background=None)
            if topology["holes"] != case["expected_holes"]:
                raise AssertionError(f'{case["id"]} at {width}px has {topology["holes"]} holes, expected {case["expected_holes"]}')
            record["rasters"].append({"path": png.relative_to(destination).as_posix(),
                                      "resolution": "low" if width == min(widths) else "medium",
                                      "render": render, "topology": topology,
                                      "transparent_truth_path": alpha_png.relative_to(destination).as_posix(),
                                      "transparent_truth_render": alpha_render,
                                      "transparent_truth_topology": raster_topology(alpha_png, mode="alpha")})
        records.append(record)
    manifest = {"schema": "designer-groundtruth-benchmark/v1", "renderer": renderer_info(),
                "suite": suite,
                "generation_source_snapshot": code_snapshot(),
                "case_count": len(records), "raster_count": sum(len(c["rasters"]) for c in records),
                "human_edit_time": "not_measured", "semantic_object_matching": "requires_manual_review",
                "acceptance_policy": "No overall pass from pixel similarity; inspect object ownership, holes, strokes, gradients and edit structure independently.",
                "cases": records}
    (destination / "truth-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _contact_sheet(destination, records)
    return manifest


def _contact_sheet(destination, records):
    cell_w, cell_h = 232, 242
    sheet = Image.new("RGB", (cell_w * 4, cell_h * ((len(records) + 3) // 4)), "#ededed")
    draw = ImageDraw.Draw(sheet)
    for index, case in enumerate(records):
        image = Image.open(destination / case["rasters"][-1]["path"]).convert("RGB")
        image.thumbnail((208, 208))
        x, y = (index % 4) * cell_w + 12, (index // 4) * cell_h + 8
        sheet.paste(image, (x, y))
        draw.text((x, y + 213), case["id"], fill="#202020")
    sheet.save(destination / "groundtruth-contact-sheet.png")


def evaluate(destination, results_dir):
    """Summarize delivered SVGs without treating structural counts as success."""
    destination, results_dir = Path(destination), Path(results_dir)
    manifest = json.loads((destination / "truth-manifest.json").read_text(encoding="utf-8"))
    rows = []
    for case in manifest["cases"]:
        for raster in case["rasters"]:
            name = Path(raster["path"]).stem
            result_dir = results_dir / f"result_{name}"
            report_path = result_dir / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            svgs = sorted(result_dir.glob("*_vector.svg")) if result_dir.exists() else []
            row = {"input": name, "case": case["id"], "width": raster["render"]["width"],
                   "conversion_available": bool(svgs), "pipeline_acceptance": report.get("acceptance_status", "missing"),
                   "visual_acceptance": report.get("visual_acceptance_status", "missing"),
                   "designer_readiness": report.get("designer_readiness_status", "missing"),
                   "background_requested": (report.get("options_requested") or {}).get("background"),
                   "background_effective": (report.get("options_effective") or {}).get("background"),
                   "background_removed": report.get("background_removed"),
                   "expected_objects": case["expected_objects"], "expected_holes": case["expected_holes"],
                   "truth_structure": case["structure"], "output_structure": None,
                   "semantic_selectability": "not_verified", "human_edit_time": "not_measured"}
            if svgs:
                row["output_structure"] = svg_structure(svgs[0].read_text(encoding="utf-8"))
                row["output_svg_sha256"] = hashlib.sha256(svgs[0].read_bytes()).hexdigest()
                row["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest() if report_path.exists() else None
                delivered = destination / "converted" / name
                delivered.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(svgs[0], delivered / "output.svg")
                if report_path.exists():
                    shutil.copyfile(report_path, delivered / "report.json")
                row["delivered_svg"] = (delivered / "output.svg").relative_to(destination).as_posix()
                row["reported_nodes"] = report.get("nodes_total")
                row["reported_designer_anchors"] = report.get("designer_anchors_total")
                rendered = destination / "output-renders"
                rendered.mkdir(exist_ok=True)
                png = rendered / f"{name}.png"
                row["render"] = render_svg_reference(svgs[0], png, width=raster["render"]["width"])
                row["output_topology"] = raster_topology(png)
                row["hole_count_matches"] = row["output_topology"]["holes"] == case["expected_holes"]
                alpha_dir = destination / "output-alpha-renders"
                alpha_dir.mkdir(exist_ok=True)
                alpha_png = alpha_dir / f"{name}.png"
                row["alpha_render"] = render_svg_reference(svgs[0], alpha_png, width=raster["render"]["width"], background=None)
                row["output_alpha_topology"] = raster_topology(alpha_png, mode="alpha")
                row["transparent_hole_count_matches"] = row["output_alpha_topology"]["holes"] == case["expected_holes"]
                row["inference_limit"] = case["inference_limit"]
                scratch = CODE_DIR.parents[1] / "work" / "designer-benchmark" / "selection-masks" / name
                row["selection_correspondence"] = selection_correspondence(destination / case["svg"], svgs[0], scratch, raster["render"]["width"])
                row["source_svg"] = str(svgs[0].resolve())
            rows.append(row)
    payload = {"schema": "designer-benchmark-observations/v1", "result_count": len(rows),
               "evaluation_source_snapshot": code_snapshot(),
               "hole_comparison_scope": "White-composited visible openings and actual transparent openings are separate observations. Opaque raster input alone cannot prove whether an enclosed white region was intended as white paint or negative space; retained white paint is not silently reported as a transparent hole.",
               "overall_quality_pass": "not_assessed", "cases": rows}
    (destination / "benchmark-observations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["input", "width", "conversion_available", "pipeline_acceptance", "designer_readiness", "expected_objects", "expected_holes", "reported_nodes", "reported_designer_anchors", "hole_count_matches", "transparent_hole_count_matches", "output_paths", "output_strokes", "output_gradients", "output_selection_units", "matched_visible_objects", "semantic_selectability"]
    with (destination / "benchmark-observations.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            structure = row["output_structure"] or {}
            writer.writerow({**row, "output_paths": structure.get("path_count"), "output_strokes": structure.get("stroke_count"),
                             "output_gradients": structure.get("linear_gradient_count", 0) + structure.get("radial_gradient_count", 0),
                             "output_selection_units": structure.get("selection_unit_count"),
                             "matched_visible_objects": (row.get("selection_correspondence") or {}).get("matched_reference_objects")})
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--evaluate", type=Path, help="Existing conversion output root; never triggers conversion.")
    parser.add_argument("--snapshot-only", action="store_true", help="Record conversion-start source hashes, without running conversion.")
    parser.add_argument("--suite", choices=("standard", "adversarial"), default="standard")
    args = parser.parse_args()
    if args.snapshot_only:
        args.destination.mkdir(parents=True, exist_ok=True)
        result = code_snapshot()
        (args.destination / "conversion-start-snapshot.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate_sha256": result["aggregate_sha256"]}))
    else:
        result = evaluate(args.destination, args.evaluate) if args.evaluate else generate(args.destination, suite=args.suite)
        print(json.dumps({key: value for key, value in result.items() if key not in {"cases", "generation_source_snapshot", "evaluation_source_snapshot"}}, ensure_ascii=False, indent=2))
