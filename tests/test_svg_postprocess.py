from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from PIL import Image

from svg_postprocess import (atomic_replace_bytes, attach_paint_roles,
                             enhance_svg_structure, measure_svg_structure)
from vector_cleanroom import validate_svg_stage_renders


SVG_NS = "http://www.w3.org/2000/svg"


def _arc(cx, cy, radius, start, end, count=80):
    points = []
    for index in range(count + 1):
        angle = math.radians(start + (end - start) * index / count)
        points.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    return "M" + " L".join(f"{x:.4f} {y:.4f}" for x, y in points)


def _source() -> str:
    return f'''<svg xmlns="{SVG_NS}" viewBox="0 0 320 320" width="320" height="320">
  <g id="dark" fill="#222222" fill-rule="evenodd">
    <path d="M20 20L80 20L80 80L20 80Z M200 200L260 200L260 260L200 260Z"/>
  </g>
  <g id="light" fill="#eeeeee"><rect x="30" y="30" width="30" height="30"/></g>
  <g id="strokes" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <path id="stroke-1" stroke="#71ff00" stroke-width="8" d="{_arc(160,160,112,-25,-155)}"/>
    <path id="stroke-2" stroke="#71ff00" stroke-width="8.2" d="{_arc(160.5,159.7,112.2,25,155)}"/>
  </g>
</svg>'''


class SvgPostprocessTests(unittest.TestCase):
    def test_structure_reports_disjoint_native_element_types(self):
        with tempfile.TemporaryDirectory() as temp:
            svg = Path(temp) / "native-types.svg"
            svg.write_text(f'''<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">
              <circle cx="10" cy="10" r="4"/><rect x="20" y="5" width="8" height="8"/>
              <ellipse cx="40" cy="10" rx="5" ry="3"/><line x1="5" y1="30" x2="20" y2="30"/>
              <polyline points="30,30 40,35 50,30"/><polygon points="60,30 70,35 65,25"/>
            </svg>''', encoding="utf-8")
            report = measure_svg_structure(svg)
        self.assertEqual(report["native_primitives"], 6)
        self.assertEqual(report["native_circles"], 1)
        self.assertEqual(report["native_rectangles"], 1)
        self.assertEqual(report["native_ellipses"], 1)
        self.assertEqual(report["native_lines"], 1)
        self.assertEqual(report["native_polylines"], 1)
        self.assertEqual(report["native_polygons"], 1)
        self.assertEqual(report["native_polygonal_shapes"], 2)

    def test_pipeline_applies_all_structural_stages(self):
        calls = []

        def validator(_before, _after, stage):
            calls.append(stage)
            return {"accepted": True, "score_percent": 100.0}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            svg.write_text(_source(), encoding="utf-8")
            report = enhance_svg_structure(svg, validator=validator,
                                           work_dir=root)
            final = measure_svg_structure(svg)
            tree = ET.parse(svg).getroot()
        self.assertEqual(report["stages"]["annulus"]["applied_candidates"], 1)
        self.assertEqual(report["stages"]["compound_paths"]["status"], "applied")
        self.assertEqual(report["stages"]["scene_graph"]["status"], "applied")
        self.assertIn("annulus_approximate", calls)
        self.assertIn("compound_exact", calls)
        self.assertIn("scene_graph_exact", calls)
        circles = list(tree.iter(f"{{{SVG_NS}}}circle"))
        self.assertTrue(any(item.get("data-detector") for item in circles))
        self.assertGreaterEqual(final["semantic_groups"], 1)
        self.assertGreater(final["object_id_coverage"], 0.5)

    def test_exact_linear_stage_commits_native_lines_only_with_pixel_proof(self):
        def exact_validator(_before, _after, _stage):
            return {
                "accepted": True,
                "external_render_check": "completed",
                "exact_pixel_array_equal": True,
                "exact_before_pixel_sha256": "a" * 64,
                "exact_after_pixel_sha256": "a" * 64,
                "required_equivalence": "pixel_array_exact_at_validation_resolution",
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "linear.svg"
            svg.write_text(f'''<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">
              <path id="one" d="M10 10 L90 10" fill="none" stroke="#111"/>
              <path id="two" d="M10 20 L50 40 L90 20" fill="none" stroke="#222"/>
            </svg>''', encoding="utf-8")
            report = enhance_svg_structure(
                svg, validator=exact_validator, work_dir=root,
                enable_annulus=False, enable_compound_paths=False,
                enable_scene_graph=False)
            structure = measure_svg_structure(svg)
        stage = report["stages"]["exact_native_shapes"]
        self.assertEqual(stage["status"], "applied")
        self.assertTrue(stage["committed"])
        self.assertEqual(stage["committed_candidate_count"], 2)
        self.assertEqual(stage["committed_line_count"], 1)
        self.assertEqual(stage["committed_polyline_count"], 1)
        self.assertEqual(structure["paths"], 0)
        self.assertEqual(structure["native_lines"], 1)
        self.assertEqual(structure["native_polylines"], 1)

    def test_scene_graph_materializes_inherited_stroke_before_nativeization(self):
        calls = []

        def exact_validator(_before, _after, stage):
            calls.append(stage)
            return {
                "accepted": True,
                "external_render_check": "completed",
                "exact_pixel_array_equal": True,
                "exact_before_pixel_sha256": "c" * 64,
                "exact_after_pixel_sha256": "c" * 64,
                "required_equivalence":
                    "pixel_array_exact_at_validation_resolution",
            }

        source = f'''<svg xmlns="{SVG_NS}" viewBox="0 0 200 100">
          <g id="base" fill="#16324f">
            <rect x="10" y="10" width="30" height="30"/>
          </g>
          <g id="ink" fill="none" stroke="#111" stroke-width="2">
            <path id="inherited-line" d="M120 20L140 20"/>
          </g>
          <g id="highlight" fill="#fff">
            <rect x="18" y="18" width="13" height="13"/>
          </g>
        </svg>'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "inherited.svg"
            svg.write_text(source, encoding="utf-8")
            report = enhance_svg_structure(
                svg, validator=exact_validator, work_dir=root,
                enable_annulus=False, enable_compound_paths=False)
            tree = ET.parse(svg).getroot()

        self.assertEqual(report["stages"]["scene_graph"]["status"], "applied")
        native = report["stages"]["exact_native_shapes"]
        self.assertEqual(native["status"], "applied")
        self.assertEqual(native["committed_line_count"], 1)
        self.assertLess(calls.index("scene_graph_exact"),
                        calls.index("exact_linear_nativeization_exact"))
        line = tree.find(f".//{{{SVG_NS}}}line[@id='inherited-line']")
        self.assertIsNotNone(line)
        self.assertEqual(line.get("stroke"), "#111")
        self.assertEqual(line.get("fill"), "none")

    def test_exact_linear_stage_refuses_internal_only_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "linear.svg"
            original = (f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">'
                        '<path id="one" d="M10 10 L90 10" '
                        'fill="none" stroke="#111"/></svg>').encode("utf-8")
            svg.write_bytes(original)
            report = enhance_svg_structure(
                svg, validator=lambda _a, _b, _s: {
                    "accepted": True,
                    "external_render_check": "unavailable",
                    "exact_pixel_array_equal": True,
                    "exact_before_pixel_sha256": "a" * 64,
                    "exact_after_pixel_sha256": "a" * 64,
                }, work_dir=root, enable_annulus=False,
                enable_compound_paths=False, enable_scene_graph=False)
            delivered = svg.read_bytes()
        stage = report["stages"]["exact_native_shapes"]
        self.assertEqual(stage["status"], "rejected")
        self.assertFalse(stage["committed"])
        self.assertEqual(stage["committed_candidate_count"], 0)
        self.assertEqual(delivered, original)

    def test_exact_linear_commit_failure_keeps_original_bytes(self):
        def exact_validator(_before, _after, _stage):
            return {
                "accepted": True,
                "external_render_check": "completed",
                "exact_pixel_array_equal": True,
                "exact_before_pixel_sha256": "b" * 64,
                "exact_after_pixel_sha256": "b" * 64,
                "required_equivalence": "pixel_array_exact_at_validation_resolution",
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "linear.svg"
            original = (f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">'
                        '<path id="one" d="M10 10 L90 10" '
                        'fill="none" stroke="#111"/></svg>').encode("utf-8")
            svg.write_bytes(original)
            with mock.patch("svg_postprocess.atomic_replace_bytes",
                            side_effect=OSError("injected exact commit failure")):
                report = enhance_svg_structure(
                    svg, validator=exact_validator, work_dir=root,
                    enable_annulus=False, enable_compound_paths=False,
                    enable_scene_graph=False)
            delivered = svg.read_bytes()
        stage = report["stages"]["exact_native_shapes"]
        self.assertEqual(stage["status"], "rolled_back_error")
        self.assertEqual(stage["committed_candidate_count"], 0)
        self.assertFalse(stage["committed"])
        self.assertEqual(delivered, original)

    def test_render_rejection_rolls_back_only_that_stage(self):
        def validator(_before, _after, stage):
            return {"accepted": stage != "annulus_approximate"}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            svg.write_text(_source(), encoding="utf-8")
            report = enhance_svg_structure(svg, validator=validator,
                                           work_dir=root)
            text = svg.read_text(encoding="utf-8")
        annulus = report["stages"]["annulus"]
        self.assertEqual(annulus["applied_candidates"], 0)
        self.assertIn("rolled_back_render_guard",
                      {item["status"] for item in annulus["candidates"]})
        self.assertNotIn("data-detector=", text)
        self.assertEqual(report["stages"]["compound_paths"]["status"], "applied")

    def test_rejected_compound_reports_only_committed_selectability(self):
        def validator(_before, _after, stage):
            return {"accepted": stage != "compound_exact"}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            svg.write_text(_source(), encoding="utf-8")
            report = enhance_svg_structure(svg, validator=validator,
                                           work_dir=root,
                                           enable_annulus=False,
                                           enable_scene_graph=False)
        compound = report["stages"]["compound_paths"]
        self.assertEqual(compound["status"], "rolled_back")
        self.assertEqual(compound["selectable_path_delta"], 0)
        self.assertEqual(compound["paths"], [])
        self.assertGreater(compound["attempted"]["selectable_path_delta"], 0)
        self.assertTrue(compound["attempted"]["paths"])

    def test_paint_roles_are_attached_without_css_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            manifest_path = root / "roles.json"
            svg.write_text(_source(), encoding="utf-8")
            manifest, report = attach_paint_roles(
                svg, manifest_path, validator=lambda _a, _b, _s: True,
                work_dir=root)
            text = svg.read_text(encoding="utf-8")
        self.assertEqual(report["status"], "applied")
        self.assertGreaterEqual(len(manifest["roles"]), 2)
        self.assertIn("data-paint-role-", text)
        self.assertIn("ai-vector-cleanroom-paint-roles", text)
        self.assertFalse(manifest["compatibility"]["css_required"])

    def test_rejected_annotations_keep_honest_manifest_only_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            manifest_path = root / "roles.json"
            source = _source()
            svg.write_text(source, encoding="utf-8")
            manifest, report = attach_paint_roles(
                svg, manifest_path, validator=lambda _a, _b, _s: False,
                work_dir=root)
            delivered = svg.read_text(encoding="utf-8")
        self.assertEqual(report["status"], "manifest_only")
        self.assertTrue(report["manifest_committed"])
        self.assertFalse(report["annotation_committed"])
        self.assertEqual(report["annotation"]["status"],
                         "proposal_only_not_committed")
        self.assertEqual(delivered, source)
        self.assertTrue(manifest["roles"])
        self.assertEqual(manifest["source"]["sha256_scope"],
                         "input_svg_before_role_annotations")

    def test_atomic_replace_failure_preserves_original_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "logo.svg"
            target.write_bytes(b"original")
            with mock.patch("svg_postprocess.os.replace",
                            side_effect=OSError("injected disk failure")):
                with self.assertRaises(OSError):
                    atomic_replace_bytes(target, b"replacement")
            self.assertEqual(target.read_bytes(), b"original")
            self.assertEqual(list(root.glob(".logo.svg.*.tmp")), [])

    def test_all_four_stage_commit_failures_preserve_svg_bytes(self):
        cases = [
            {"enable_compound_paths": False, "enable_scene_graph": False},
            {"enable_annulus": False, "enable_scene_graph": False},
            {"enable_annulus": False, "enable_compound_paths": False},
        ]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                svg = root / "logo.svg"
                original = _source().encode("utf-8")
                svg.write_bytes(original)
                with mock.patch("svg_postprocess.atomic_replace_bytes",
                                side_effect=OSError("injected commit failure")):
                    enhance_svg_structure(
                        svg, validator=lambda _a, _b, _s: True,
                        work_dir=root, **options)
                self.assertEqual(svg.read_bytes(), original)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            original = _source().encode("utf-8")
            svg.write_bytes(original)
            with mock.patch("svg_postprocess.atomic_replace_bytes",
                            side_effect=OSError("injected paint commit failure")):
                with self.assertRaises(OSError):
                    attach_paint_roles(
                        svg, root / "roles.json",
                        validator=lambda _a, _b, _s: True, work_dir=root)
            self.assertEqual(svg.read_bytes(), original)

    def test_exact_render_guard_rejects_one_channel_level_difference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            before = root / "before.svg"
            after = root / "after.svg"
            before.write_text("<svg/>", encoding="utf-8")
            after.write_text("<svg/>", encoding="utf-8")

            def fake_render(svg_path, png_path, **_kwargs):
                image = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
                if Path(svg_path) == after:
                    image.putpixel((4, 4), (254, 255, 255, 255))
                image.save(png_path)
                return True

            with mock.patch("vector_cleanroom.render_svg_png",
                            side_effect=fake_render):
                result = validate_svg_stage_renders(
                    before, after, "compound_exact")
        self.assertFalse(result["accepted"])
        self.assertFalse(result["exact_pixel_array_equal"])
        self.assertEqual(
            result["required_equivalence"],
            "pixel_array_exact_at_validation_resolution")


if __name__ == "__main__":
    unittest.main()
