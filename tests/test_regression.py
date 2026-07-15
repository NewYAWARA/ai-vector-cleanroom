"""Black-box acceptance tests for vector_cleanroom.py.

The suite intentionally describes the quality floor expected of a releasable
build. A known-bad baseline is allowed to fail; the failure names are the
handoff checklist for the next implementation pass.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET


TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
FIXTURES = TESTS / "fixtures"
RUN = TESTS / "_last_run"
OUTPUT = RUN / "output"
PYTHON = ROOT / "python" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

HARNESS_VERSION = "v3-codex-regression.4"
MANIFEST = RUN / "reuse_manifest.json"
FIXTURE_NAMES = (
    "line_over_fill_darkgray",
    "low_contrast_ddd",
    "mixed_alpha",
    "multicolor_touch",
    "one_px_black",
    "one_px_black_3000",
    "right_angle",
    "ring",
    "soft_alpha_100",
    "square_frame_5px",
    "t_junction",
    "x_junction",
    "y_junction",
)
HASHED_TEST_FILES = (
    ROOT / "vector_cleanroom.py",
    ROOT / "clean_base.py",
    ROOT / "trace_engine.py",
    ROOT / "stroke_engine.py",
    ROOT / "quality_diagnostics.py",
    ROOT / "editability_audit.py",
    TESTS / "generate_fixtures.py",
    TESTS / "test_regression.py",
)
RESULT_FILES = (
    "report.json",
    "review.html",
    "OUTPUT_README.txt",
    "source_reference.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_hashes(paths):
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in paths
    }


def _expected_result_paths(name: str):
    result = OUTPUT / f"result_{name}"
    paths = [result / item for item in RESULT_FILES]
    paths.extend((
        result / f"{name}_vector.svg",
        result / f"{name}_preview.png",
        OUTPUT / f"result_{name}.zip",
    ))
    return paths


def _current_output_hashes():
    return {
        name: {
            str(path.relative_to(RUN)).replace("\\", "/"): _sha256(path)
            for path in _expected_result_paths(name)
        }
        for name in FIXTURE_NAMES
    }


def _validate_reuse_manifest():
    problems = []
    if not MANIFEST.is_file():
        return [f"missing {MANIFEST.name}"]
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid {MANIFEST.name}: {exc}"]

    if manifest.get("schema") != 1:
        problems.append("manifest schema mismatch")
    if manifest.get("harness_version") != HARNESS_VERSION:
        problems.append("test harness version mismatch")
    if manifest.get("process_returncode") != 0:
        problems.append("cached CLI run did not exit successfully")

    missing_sources = [path for path in HASHED_TEST_FILES if not path.is_file()]
    if missing_sources:
        problems.extend(f"missing source: {path}" for path in missing_sources)
    else:
        current = _relative_hashes(HASHED_TEST_FILES)
        if manifest.get("source_hashes") != current:
            problems.append("core/generator/test hashes changed")

    fixture_paths = [FIXTURES / f"{name}.png" for name in FIXTURE_NAMES]
    missing_fixtures = [path for path in fixture_paths if not path.is_file()]
    if missing_fixtures:
        problems.extend(f"missing fixture: {path.name}" for path in missing_fixtures)
    else:
        current = {path.name: _sha256(path) for path in fixture_paths}
        if manifest.get("fixture_hashes") != current:
            problems.append("fixture hashes changed")

    missing_outputs = [
        path for name in FIXTURE_NAMES for path in _expected_result_paths(name)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_outputs:
        problems.extend(f"missing output: {path.relative_to(RUN)}"
                        for path in missing_outputs)
    else:
        try:
            if manifest.get("output_hashes") != _current_output_hashes():
                problems.append("cached output hashes changed")
        except OSError as exc:
            problems.append(f"could not hash cached outputs: {exc}")
    return problems


def _write_reuse_manifest(process_returncode: int):
    fixture_paths = [FIXTURES / f"{name}.png" for name in FIXTURE_NAMES]
    manifest = {
        "schema": 1,
        "harness_version": HARNESS_VERSION,
        "process_returncode": process_returncode,
        "source_hashes": _relative_hashes(HASHED_TEST_FILES),
        "fixture_hashes": {path.name: _sha256(path) for path in fixture_paths},
        "output_hashes": _current_output_hashes() if process_returncode == 0 else {},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                        encoding="utf-8")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _number(value: str) -> float:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value or "")
    if not match:
        raise ValueError(f"not a number: {value!r}")
    return float(match.group(0))


def _hex_rgb(value: str):
    value = (value or "").strip().lower()
    if re.fullmatch(r"#[0-9a-f]{6}", value):
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    return None


def _near_color(actual: str, expected, tolerance=20) -> bool:
    rgb = _hex_rgb(actual)
    return bool(rgb and max(abs(a - b) for a, b in zip(rgb, expected)) <= tolerance)


def _elements(svg_path: Path):
    return list(ET.parse(svg_path).getroot().iter())


def _drawables_with_inherited_fill(svg_path: Path):
    root = ET.parse(svg_path).getroot()
    out = []

    def visit(element, inherited_fill="black"):
        fill = element.attrib.get("fill", inherited_fill)
        if _local(element.tag) in {"path", "circle", "ellipse", "rect", "polygon"}:
            out.append((element, fill.lower()))
        for child in element:
            visit(child, fill)

    visit(root)
    return out


def _drawables_with_inherited_style(svg_path: Path):
    root = ET.parse(svg_path).getroot()
    out = []

    def visit(element, inherited_fill="black", inherited_opacity=1.0):
        fill = element.attrib.get("fill", inherited_fill).lower()
        opacity = inherited_opacity
        for key in ("opacity", "fill-opacity"):
            if key in element.attrib:
                opacity *= float(element.attrib[key])
        if _local(element.tag) in {"path", "circle", "ellipse", "rect", "polygon"}:
            out.append((element, fill, opacity))
        for child in element:
            visit(child, fill, opacity)

    visit(root)
    return out


def _stroke_primitives(svg_path: Path):
    return [
        element for element in _elements(svg_path)
        if element.attrib.get("id", "").startswith("stroke-")
        and element.attrib.get("stroke", "").lower() not in {"", "none"}
    ]


def _stroke_opacities(svg_path: Path):
    root = ET.parse(svg_path).getroot()
    out = []

    def visit(element, inherited_opacity=1.0):
        opacity = inherited_opacity * float(element.attrib.get("opacity", "1"))
        if (element.attrib.get("id", "").startswith("stroke-")
                and element.attrib.get("stroke", "").lower() not in {"", "none"}):
            out.append(opacity * float(element.attrib.get("stroke-opacity", "1")))
        for child in element:
            visit(child, opacity)

    visit(root)
    return out


def _line_endpoints(element):
    if _local(element.tag) == "line":
        return ((float(element.attrib["x1"]), float(element.attrib["y1"])),
                (float(element.attrib["x2"]), float(element.attrib["y2"])))
    if _local(element.tag) != "path":
        return None
    match = re.fullmatch(
        r"\s*M\s*([-+.\d]+)\s+([-+.\d]+)\s+L\s*([-+.\d]+)\s+([-+.\d]+)\s*",
        element.attrib.get("d", ""), re.IGNORECASE,
    )
    if not match:
        return None
    values = [float(value) for value in match.groups()]
    return ((values[0], values[1]), (values[2], values[3]))


def _has_shared_junction(svg_path: Path, branch_count: int, tolerance=2.0):
    endpoints = [_line_endpoints(element) for element in _stroke_primitives(svg_path)]
    if len(endpoints) != branch_count or any(pair is None for pair in endpoints):
        return False
    for candidate in endpoints[0]:
        if all(any(math.dist(candidate, endpoint) <= tolerance for endpoint in pair)
               for pair in endpoints[1:]):
            return True
    return False


class VectorRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reuse = os.environ.get("VECTOR_TEST_REUSE") == "1"
        if reuse:
            problems = _validate_reuse_manifest()
            if problems:
                detail = "\n  - ".join(problems)
                raise RuntimeError(
                    "VECTOR_TEST_REUSE refused stale or incomplete output:\n"
                    f"  - {detail}\nRun tests\\run_tests.bat for a fresh run."
                )
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            cls.process = subprocess.CompletedProcess(
                args=[], returncode=manifest["process_returncode"],
                stdout=(RUN / "process_stdout.txt").read_text(
                    encoding="utf-8", errors="replace")
                if (RUN / "process_stdout.txt").exists() else "",
                stderr=(RUN / "process_stderr.txt").read_text(
                    encoding="utf-8", errors="replace")
                if (RUN / "process_stderr.txt").exists() else "",
            )
            return
        if RUN.exists():
            shutil.rmtree(RUN)
        if FIXTURES.exists():
            shutil.rmtree(FIXTURES)
        FIXTURES.mkdir(parents=True, exist_ok=True)
        RUN.mkdir(parents=True, exist_ok=True)

        generated = subprocess.run(
            [str(PYTHON), str(TESTS / "generate_fixtures.py"),
             "--output", str(FIXTURES)],
            cwd=str(ROOT), text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=60,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        if generated.returncode:
            raise RuntimeError(generated.stdout + generated.stderr)
        expected_fixtures = {f"{name}.png" for name in FIXTURE_NAMES}
        actual_fixtures = {path.name for path in FIXTURES.glob("*.png")}
        if actual_fixtures != expected_fixtures:
            raise RuntimeError(
                "fixture set mismatch: expected "
                f"{sorted(expected_fixtures)}, got {sorted(actual_fixtures)}"
            )

        cls.process = subprocess.run(
            [str(PYTHON), str(ROOT / "vector_cleanroom.py"),
             "--input", str(FIXTURES), "--output", str(OUTPUT)],
            cwd=str(ROOT), text=True, encoding="utf-8", errors="replace",
            capture_output=True, timeout=600,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        (RUN / "process_stdout.txt").write_text(
            cls.process.stdout, encoding="utf-8")
        (RUN / "process_stderr.txt").write_text(
            cls.process.stderr, encoding="utf-8")
        if cls.process.returncode == 0 and all(
            path.is_file() and path.stat().st_size > 0
            for name in FIXTURE_NAMES for path in _expected_result_paths(name)
        ):
            _write_reuse_manifest(cls.process.returncode)

    def result_dir(self, name: str) -> Path:
        return OUTPUT / f"result_{name}"

    def report(self, name: str):
        path = self.result_dir(name) / "report.json"
        self.assertTrue(
            path.exists(),
            f"{name} did not produce report.json. See "
            f"{RUN / 'process_stdout.txt'} and {RUN / 'process_stderr.txt'}",
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def svg(self, name: str) -> Path:
        path = self.result_dir(name) / f"{name}_vector.svg"
        self.assertTrue(path.exists(), f"missing SVG for {name}")
        return path

    def assert_foreground(self, name: str, floor=80.0):
        report = self.report(name)
        score = report.get("foreground_match_percent")
        self.assertIsNotNone(score, f"{name}: foreground score is unavailable")
        self.assertGreaterEqual(score, floor, f"{name}: foreground match too low")
        return report

    def assert_no_white_negative_space(self, name: str):
        report = self.report(name)
        self.assertTrue(report.get("background_removed"),
                        f"{name}: expected white background removal")
        painted_white = [
            element for element, fill in _drawables_with_inherited_fill(self.svg(name))
            if fill in {"#fff", "#ffffff", "white"}
        ]
        self.assertFalse(
            painted_white,
            f"{name}: removed white background/negative space reappeared as "
            f"{len(painted_white)} editable white object(s)",
        )

    def test_one_pixel_line_is_real_black_stroke(self):
        report = self.assert_foreground("one_px_black", 95)
        self.assertEqual(report["strokes"], 1)
        self.assertLessEqual(report["nodes_total"], 3)
        detail = report["stroke_details"][0]
        self.assertTrue(_near_color(detail["color"], (0, 0, 0), 5))
        self.assertAlmostEqual(detail["width"], 1.0, delta=0.25)

    def test_low_contrast_ddd_survives_default_auto_background(self):
        report = self.assert_foreground("low_contrast_ddd", 85)
        self.assertGreaterEqual(report["strokes"], 1)
        self.assertTrue(any(_near_color(item["color"], (221, 221, 221), 10)
                            for item in report["stroke_details"]))
        self.assert_no_white_negative_space("low_contrast_ddd")

    def test_right_angle_does_not_bow(self):
        report = self.assert_foreground("right_angle", 90)
        self.assertEqual(report["strokes"], 1)
        self.assertLessEqual(report["nodes_total"], 6)

    def test_square_has_no_white_fill_object(self):
        report = self.assert_foreground("square_frame_5px", 90)
        self.assertEqual(report["strokes"], 1)
        self.assertEqual(report["native_primitives"], 1)
        self.assertEqual(report["native_circles"], 0)
        self.assertEqual(report["native_rectangles"], 1)
        self.assertEqual(report["stroke_details"][0].get("primitive"), "rect")
        self.assertEqual(report["nodes_total"], 4)
        primitives = _stroke_primitives(self.svg("square_frame_5px"))
        self.assertEqual(len(primitives), 1)
        self.assertEqual(_local(primitives[0].tag), "rect")
        self.assert_no_white_negative_space("square_frame_5px")

    def test_ring_is_clean_and_transparent_inside(self):
        report = self.assert_foreground("ring", 90)
        self.assertEqual(report["strokes"], 1)
        self.assertEqual(report["native_primitives"], 1)
        self.assertEqual(report["native_circles"], 1)
        self.assertEqual(report["native_rectangles"], 0)
        self.assertEqual(report["stroke_details"][0].get("primitive"), "circle")
        self.assertEqual(report["nodes_total"], 1)
        primitives = _stroke_primitives(self.svg("ring"))
        self.assertEqual(len(primitives), 1)
        self.assertEqual(_local(primitives[0].tag), "circle")
        self.assert_no_white_negative_space("ring")

    def test_t_junction_fidelity(self):
        report = self.assert_foreground("t_junction", 90)
        self.assertEqual(report["acceptance_status"], "accepted")
        self.assertEqual(report["strokes"], 3)
        self.assertEqual(report["nodes_total"], 6)
        self.assertTrue(_has_shared_junction(self.svg("t_junction"), 3),
                        "T branches do not share one editable junction")

    def test_x_junction_fidelity(self):
        report = self.assert_foreground("x_junction", 90)
        self.assertEqual(report["acceptance_status"], "accepted")
        self.assertEqual(report["strokes"], 4)
        self.assertEqual(report["nodes_total"], 8)
        self.assertTrue(_has_shared_junction(self.svg("x_junction"), 4),
                        "X branches do not share one editable junction")

    def test_y_junction_fidelity(self):
        report = self.assert_foreground("y_junction", 90)
        self.assertEqual(report["acceptance_status"], "accepted")
        self.assertFalse(report["manual_review_required"])
        self.assertEqual(report["strokes"], 3)
        self.assertEqual(report["nodes_total"], 6)
        self.assertTrue(_has_shared_junction(self.svg("y_junction"), 3),
                        "Y branches do not share one editable junction")

    def test_complex_overlap_candidate_search_is_auditable(self):
        report = self.report("line_over_fill_darkgray")
        candidates = report.get("candidates", [])
        self.assertGreaterEqual(
            len(candidates), 2,
            "complex overlap should expand the candidate matrix",
        )
        selected = [item for item in candidates if item.get("selected")]
        self.assertEqual(len(selected), 1,
                         "candidate report must identify exactly one winner")
        self.assertEqual(selected[0].get("options"), report["options_effective"])
        self.assertIn("selection_score", selected[0])
        self.assertIn("structure", selected[0])

    def test_multicolor_touch_keeps_both_colors(self):
        report = self.assert_foreground("multicolor_touch", 85)
        colors = [item["color"] for item in report["stroke_details"]]
        self.assertTrue(any(_near_color(c, (220, 20, 30), 15) for c in colors))
        self.assertTrue(any(_near_color(c, (20, 70, 220), 15) for c in colors))
        self.assertGreaterEqual(report["strokes"], 2)

    def test_dark_gray_line_over_fill_survives(self):
        report = self.assert_foreground("line_over_fill_darkgray", 90)
        self.assertGreaterEqual(report["paths"], 1)
        self.assertGreaterEqual(report["strokes"], 1)
        self.assertTrue(any(_near_color(item["color"], (51, 51, 51), 20)
                            for item in report["stroke_details"]))

    def test_soft_alpha_is_not_flattened_to_opaque(self):
        report = self.assert_foreground("soft_alpha_100", 90)
        self.assertEqual(report["strokes"], 1)
        expected = 100 / 255
        self.assertAlmostEqual(report["stroke_details"][0]["opacity"], expected,
                               delta=0.03)
        opacities = _stroke_opacities(self.svg("soft_alpha_100"))
        self.assertEqual(len(opacities), 1)
        self.assertAlmostEqual(opacities[0], expected, delta=0.03)

    def test_mixed_alpha_keeps_both_shapes_and_opacity(self):
        report = self.assert_foreground("mixed_alpha", 90)
        self.assertGreaterEqual(report["paths"] + report["native_primitives"], 2,
                                "one alpha tier was dropped")
        styled = _drawables_with_inherited_style(self.svg("mixed_alpha"))
        red = [(fill, opacity) for _, fill, opacity in styled
               if _near_color(fill, (225, 35, 45), 10)]
        blue = [(fill, opacity) for _, fill, opacity in styled
                if _near_color(fill, (25, 75, 225), 10)]
        self.assertTrue(red, "missing red alpha tier")
        self.assertTrue(blue, "missing blue alpha tier")
        self.assertTrue(any(abs(opacity - 80 / 255) <= 0.04
                            for _, opacity in red),
                        "red alpha tier was not preserved")
        self.assertTrue(any(abs(opacity - 200 / 255) <= 0.04
                            for _, opacity in blue),
                        "blue alpha tier was not preserved")

    def test_3000px_one_pixel_line_preserves_semantics(self):
        report = self.assert_foreground("one_px_black_3000", 95)
        self.assertEqual(report["strokes"], 1)
        self.assertLessEqual(report["nodes_total"], 3)
        svg = self.svg("one_px_black_3000")
        root = ET.parse(svg).getroot()
        canvas_width = _number(root.attrib["width"])
        viewbox_width = float(root.attrib["viewBox"].split()[2])
        stroke = next(e for e in root.iter() if "stroke-width" in e.attrib)
        displayed_width = float(stroke.attrib["stroke-width"]) * canvas_width / viewbox_width
        self.assertAlmostEqual(displayed_width, 1.0, delta=0.25)
        self.assertTrue(_near_color(stroke.attrib.get("stroke"), (0, 0, 0), 5),
                        "downsampling changed the semantic stroke color")

    def test_report_contract(self):
        self.assertEqual(
            self.process.returncode, 0,
            "full CLI run failed:\n" + self.process.stdout + self.process.stderr,
        )
        report = self.report("one_px_black")
        required = {
            "tool_version", "input", "size", "groups", "paths",
            "native_primitives", "native_circles", "native_rectangles",
            "native_ellipses", "native_lines", "native_polylines",
            "native_polygons",
            "strokes",
            "stroke_details", "gradients", "nodes_total", "background_removed",
            "flat_match_percent", "source_match_percent",
            "foreground_match_percent", "preview_is_svg_render", "hotspots",
            "detail_grid", "candidates", "candidate_selection_policy",
            "palette_detection", "visual_gate",
            "transparent_light_fidelity",
            "auto_fallback", "visual_acceptance_status",
            "editability_status", "editability_score", "editability_reasons",
            "editability_details", "editability_schema",
            "editability_audit_model", "automation_readiness",
            "redraw_complexity", "workflow_friction",
            "editability_acceptance_gate", "named_operation_evidence",
            "human_validation", "acceptance_status",
            "manual_review_required", "warnings", "options",
            "options_requested", "options_effective",
            "paint_resources", "solid_paints", "gradient_paints",
            "unique_paints_total",
            "final_structure", "engine_structure_before_postprocess",
            "editability_enhancements", "paint_roles",
            "paint_role_manifest", "recolor_page", "designer_operations",
        }
        self.assertEqual(set(), required - set(report),
                         f"missing report fields: {sorted(required - set(report))}")
        self.assertIsInstance(report["candidates"], list)
        self.assertIsInstance(report["candidate_selection_policy"], dict)
        self.assertIsInstance(report["auto_fallback"], dict)
        self.assertIsInstance(report["warnings"], list)
        self.assertIsInstance(report["options"], dict)
        self.assertIsInstance(report["editability_reasons"], list)
        self.assertIsInstance(report["editability_details"], dict)
        self.assertIsInstance(report["detail_grid"], dict)
        self.assertIsInstance(report["palette_detection"], dict)
        self.assertIsInstance(report["visual_gate"], dict)
        self.assertIsInstance(report["transparent_light_fidelity"], dict)
        self.assertIsInstance(report["paint_resources"], list)
        self.assertIsInstance(report["final_structure"], dict)
        self.assertIsInstance(report["editability_enhancements"], dict)
        self.assertIsInstance(report["paint_roles"], dict)
        self.assertIsInstance(report["designer_operations"], dict)
        self.assertTrue(report["preview_is_svg_render"])
        self.assertEqual(report["tool_version"], "v3-codex-beta.5")
        self.assertEqual(
            report["editability_schema"],
            "ai-vector-cleanroom.editability/v2")
        self.assertEqual(
            report["human_validation"]["status"], "not_performed")
        self.assertIsNone(
            report["human_validation"]["original_human_tasks_passed"])
        self.assertEqual(
            report["named_operation_evidence"]["status"],
            "reported_by_separate_structural_audit")
        self.assertEqual(report["options"], report["options_effective"])
        self.assertEqual(
            report["native_primitives"],
            sum(report[key] for key in (
                "native_circles", "native_rectangles", "native_ellipses",
                "native_lines", "native_polylines", "native_polygons")),
        )

        # Beta.5 keeps independent visual and structural/editability gates.  The
        # legacy aggregate status must be derived from those two gates instead
        # of silently treating visual fidelity as proof of easy handoff.
        self.assertIn(report["visual_acceptance_status"],
                      {"accepted", "manual_review"})
        self.assertIn(report["editability_status"],
                      {"accepted", "manual_review"})
        expected_manual = (
            report["visual_acceptance_status"] != "accepted"
            or report["editability_status"] != "accepted"
        )
        self.assertEqual(report["manual_review_required"], expected_manual)
        self.assertEqual(
            report["acceptance_status"],
            "manual_review" if expected_manual else "accepted",
        )
        self.assertIsInstance(report["editability_score"], (int, float))
        self.assertGreaterEqual(report["editability_score"], 0.0)
        self.assertLessEqual(report["editability_score"], 100.0)
        self.assertIn("scope_note", report["editability_details"])
        self.assertIn("risk_penalties", report["editability_details"])
        self.assertIn("review_triggers", report["editability_details"])

        # The local-detail metric must be based on actual source-ink cells and
        # explicitly record the one-pixel tolerance used by the gate.
        detail = report["detail_grid"]
        detail_required = {
            "cell_size_px", "one_pixel_tolerance", "eligible_cells",
            "p10_score_percent", "median_score_percent",
            "worst_score_percent", "mean_score_percent",
            "source_ink_pixels", "source_ink_fraction", "background_rgb",
            "boundary_noise_p95", "ink_threshold",
        }
        self.assertEqual(set(), detail_required - set(detail))
        self.assertIs(detail["one_pixel_tolerance"], True)
        self.assertGreater(detail["eligible_cells"], 0)
        for key in ("p10_score_percent", "median_score_percent",
                    "worst_score_percent", "mean_score_percent"):
            self.assertIsInstance(detail[key], (int, float))
            self.assertGreaterEqual(detail[key], 0.0)
            self.assertLessEqual(detail[key], 100.0)

        topology = detail["component_topology"]
        topology_required = {
            "eligible_components", "minimum_component_area_px",
            "one_pixel_tolerance", "p10_score_percent",
            "worst_score_percent", "mean_score_percent",
            "coverage_p10_percent", "connectivity_p10_percent",
            "fragmented_components", "measurement_mask", "core_threshold",
            "core_source_ink_pixels", "low_contrast_excluded_pixels",
        }
        self.assertEqual(set(), topology_required - set(topology))
        self.assertIs(topology["one_pixel_tolerance"], True)
        self.assertEqual(topology["measurement_mask"], "strong_ink_core")
        self.assertGreaterEqual(topology["eligible_components"], 0)
        self.assertGreaterEqual(topology["fragmented_components"], 0)

        palette_detection = report["palette_detection"]
        accent = palette_detection["initial_accent_retention"]
        self.assertEqual(
            accent["policy"],
            "initial_palette_connected_residual_retention")
        self.assertEqual(accent["colors_retained"], len(accent["records"]))
        linear = palette_detection["linear_detail_stabilization"]
        self.assertEqual(
            linear["policy"],
            "small_strong_ink_pca_linear_multicolour_only")
        self.assertEqual(linear["components_stabilized"],
                         len(linear["components"]))
        self.assertEqual(
            linear["pixels_relabelled"],
            sum(item["pixels_relabelled"] for item in linear["components"]),
        )

        visual_gate = report["visual_gate"]
        visual_gate_required = {
            "status", "metrics", "applicability", "acceptance_thresholds",
            "catastrophic_rejection_thresholds",
            "multi_metric_rejection_thresholds", "acceptance_breaches",
            "catastrophic_breaches", "soft_breaches",
            "compound_local_failure", "reasons", "policy",
        }
        self.assertEqual(set(), visual_gate_required - set(visual_gate))
        self.assertEqual(visual_gate["status"],
                         report["visual_acceptance_status"])
        self.assertIn("topology_p10", visual_gate["metrics"])
        self.assertIn("light_object_coverage", visual_gate["metrics"])

        light = report["transparent_light_fidelity"]
        light_required = {
            "applicable", "source_pixels", "core_pixels",
            "measurement_pixels", "measurement_mask",
            "spatial_tolerance_px", "coverage_percent",
            "non_background_coverage_percent", "mean_color_error",
            "p90_color_error", "match_tolerance_rgb", "error_metric",
            "background_rgb",
        }
        self.assertEqual(set(), light_required - set(light))
        self.assertIsInstance(light["applicable"], bool)
        self.assertIsInstance(light["source_pixels"], int)
        self.assertIsInstance(light["core_pixels"], int)
        self.assertEqual(light["match_tolerance_rgb"], 48)
        self.assertEqual(light["error_metric"], "max_channel_rgb")
        if light["applicable"]:
            self.assertEqual(light["measurement_mask"],
                             "one_pixel_eroded_light_core")
            self.assertIsInstance(light["coverage_percent"], (int, float))
            self.assertGreaterEqual(light["coverage_percent"], 0.0)
            self.assertLessEqual(light["coverage_percent"], 100.0)

        # Report paint resources, not color-stack layer count, as the palette
        # a designer actually sees in an SVG editor.
        self.assertEqual(report["unique_paints_total"],
                         len(report["paint_resources"]))
        self.assertEqual(report["unique_paints_total"],
                         report["solid_paints"] + report["gradient_paints"])
        self.assertEqual(report["solid_paints"], len(report["palette"]))
        self.assertEqual(report["gradient_paints"], report["gradients"])
        self.assertTrue(all(item.get("type") in {"solid", "linearGradient"}
                            for item in report["paint_resources"]))

        final = report["final_structure"]
        self.assertEqual(report["paths"], final["paths"])
        self.assertEqual(report["groups"], final["groups"])
        self.assertEqual(report["strokes"], final["strokes"])
        self.assertEqual(report["nodes_total"], final["nodes"])
        self.assertEqual(report["paint_roles"]["status"], "applied")
        self.assertEqual(report["paint_role_manifest"],
                         "one_px_black_paint_roles.json")
        self.assertEqual(report["recolor_page"], "色彩調整.html")
        self.assertTrue((self.result_dir("one_px_black") /
                         report["paint_role_manifest"]).is_file())
        self.assertTrue((self.result_dir("one_px_black") /
                         report["recolor_page"]).is_file())
        self.assertEqual(report["designer_operations"]["summary"]["total_operations"], 5)
        delivered_sha = hashlib.sha256(
            self.svg("one_px_black").read_bytes()).hexdigest()
        self.assertEqual(report["designer_operations"]["svg"]["sha256"],
                         delivered_sha)
        self.assertEqual(
            report["designer_operations"]["svg"]["sha256_scope"],
            "delivered_svg_after_inert_metadata")

        policy = report["candidate_selection_policy"]
        policy_required = {
            "material_visual_gain_required", "best_visual_quality",
            "selected_visual_quality",
            "selected_requested_features_retained",
            "requested_features_total", "policy", "matrix_strategy",
            "evaluated_candidates", "best_visual_status",
            "selected_visual_status", "visual_status_counts",
            "visual_status_survivor_count", "dominance_budgets",
            "survivor_count", "candidate_count", "selected_metric_vector",
            "base_structure_risk",
        }
        self.assertEqual(set(), policy_required - set(policy))
        self.assertEqual(policy["policy"],
                         "visual_gate_tier_then_safe_dominance_then_preserve_features")
        self.assertEqual(policy["evaluated_candidates"],
                         len(report["candidates"]))
        self.assertLessEqual(policy["selected_requested_features_retained"],
                              policy["requested_features_total"])
        self.assertEqual(policy["candidate_count"], len(report["candidates"]))
        self.assertIn(policy["selected_visual_status"],
                      {"accepted", "manual_review", "rejected"})
        self.assertEqual(
            set(policy["selected_metric_vector"]),
            {"foreground", "color_fidelity", "detail_p10", "detail_mean",
             "topology_p10", "light_object_coverage"},
        )
        selected = [item for item in report["candidates"]
                    if item.get("selected")]
        self.assertEqual(len(selected), 1)
        self.assertAlmostEqual(selected[0]["quality_score"],
                               policy["selected_visual_quality"], places=6)

        render_guards = []
        stages = report["editability_enhancements"]["stages"]
        for stage in stages.values():
            for key in ("render_guard", "render_guards"):
                guard = stage.get(key)
                if isinstance(guard, dict) and guard.get(
                        "external_render_check") == "completed":
                    render_guards.append(guard)
        paint_guard = report["paint_roles"].get("render_guard")
        if isinstance(paint_guard, dict) and paint_guard.get(
                "external_render_check") == "completed":
            render_guards.append(paint_guard)
        self.assertTrue(render_guards)
        for guard in render_guards:
            hits = guard["render_cache_hits"]
            self.assertIsInstance(hits["before"], bool)
            self.assertIsInstance(hits["after"], bool)

        readme = (self.result_dir("one_px_black") / "OUTPUT_README.txt").read_text(
            encoding="utf-8")
        self.assertTrue(readme.startswith("AI 向量清稿工具｜本次輸出摘要\n"))
        self.assertIn("工具版本：v3-codex-beta.5", readme)
        self.assertIn("驗收狀態：accepted", readme)
        self.assertIn("前景符合度：", readme)
        self.assertIn("外觀閘門：accepted", readme)
        self.assertIn("可編輯性閘門：accepted", readme)
        self.assertIn("local detail p10", readme)
        self.assertIn("editability", readme)
        self.assertIn("請求設定：", readme)
        self.assertIn("實際設定：", readme)
        self.assertIn("自動回退：", readme)
        self.assertIn(
            "Beta.5 fidelity, topology and editability enhancements", readme)
        self.assertIn("色彩調整.html", readme)

        review = (self.result_dir("one_px_black") / "review.html").read_text(
            encoding="utf-8")
        self.assertIn("AI Vector Cleanroom v3-codex-beta.5", review)
        self.assertIn("accepted：外觀與可編輯性均通過自動品質閘門", review)
        self.assertIn("局部細節 p10", review)
        self.assertIn("可編輯性", review)
        self.assertIn("native primitives", review)
        self.assertIn("全域換色", review)

        metadata = next(
            element for element in _elements(self.svg("one_px_black"))
            if _local(element.tag) == "metadata"
            and element.attrib.get("id") == "ai-vector-cleanroom-metadata"
        )
        embedded = json.loads(metadata.text)
        self.assertEqual(embedded["tool_version"], "v3-codex-beta.5")
        self.assertEqual(embedded["options_requested"], report["options_requested"])
        self.assertEqual(embedded["options_effective"], report["options_effective"])
        self.assertEqual(embedded["visual_acceptance_status"],
                         report["visual_acceptance_status"])
        self.assertEqual(embedded["editability_status"],
                         report["editability_status"])
        self.assertEqual(embedded["acceptance_status"], report["acceptance_status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
