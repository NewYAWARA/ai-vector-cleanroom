"""Unit tests for the independent SVG editability audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editability_audit import AUDIT_SCHEMA, audit_editability  # noqa: E402


class EditabilityAuditTests(unittest.TestCase):
    def write_svg(self, directory: Path, body: str) -> Path:
        path = directory / "fixture.svg"
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            'viewBox="0 0 100 100">' + body + "</svg>",
            encoding="utf-8",
        )
        return path

    def test_simple_semantic_svg_is_accepted_and_json_serializable(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(directory, """
              <defs>
                <linearGradient id="brand-gradient">
                  <stop offset="0" stop-color="#00ff00"/>
                  <stop offset="1" stop-color="#008800"/>
                </linearGradient>
              </defs>
              <g id="logo-mark" inkscape:label="Logo mark">
                <rect id="tile" x="2" y="2" width="96" height="96" fill="#fff"/>
                <path id="wave" fill="url(#brand-gradient)"
                      d="M10 55 C30 20 60 20 90 55 Z"/>
                <path id="accent" fill="#f00" d="M20 70 L80 70 L50 90 Z"/>
              </g>
            """)
            result = audit_editability(svg, {
                "paths": 2,
                "native_primitives": 1,
                "strokes": 0,
                "gradients": 1,
                "nodes_total": 9,
            })

        details = result["editability_details"]
        self.assertEqual(result["status"], "accepted")
        self.assertGreaterEqual(result["score"], 75)
        self.assertEqual(details["path_count"], 2)
        self.assertEqual(details["native_primitive_count"], 1)
        self.assertEqual(details["gradient_resource_count"], 1)
        self.assertEqual(details["group_count"], 1)
        self.assertEqual(details["semantic_group_count"], 1)
        self.assertEqual(details["unique_solid_paint_count"], 2)
        self.assertEqual(details["object_id_count"], 3)
        self.assertTrue(details["has_object_ids"])
        self.assertFalse(details["only_color_layers_without_semantic_groups"])
        self.assertIn("does not prove an 80%", details["scope_note"])
        self.assertEqual(result["schema"], AUDIT_SCHEMA)
        self.assertEqual(result["audit_model"], "layered-v2")
        self.assertEqual(result["score_axis"], "redraw_ease")
        self.assertEqual(result["status_scope"], "structural_editability_gate")
        self.assertEqual(result["automation_readiness"]["status"],
                         "ready_for_common_operations")
        self.assertEqual(
            result["automation_readiness"]["evidence_class"],
            "generic_structural_heuristic",
        )
        self.assertFalse(
            result["automation_readiness"]["score_is_operation_pass_count"])
        self.assertEqual(result["redraw_complexity"]["level"], "low")
        self.assertEqual(result["workflow_friction"]["level"], "low")
        self.assertTrue(result["acceptance_gate"]["passed"])
        self.assertEqual(result["human_validation"]["status"], "not_performed")
        self.assertIsNone(
            result["human_validation"]["original_human_tasks_passed"])
        json.dumps(result, ensure_ascii=False)

    def test_subpaths_and_implicit_commands_are_counted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(directory, """
              <g id="symbol">
                <path id="compound" fill="#123456"
                      d="M0 0 L10 0 10 10 Z M2 2 L3 3 Z"/>
                <path id="curve" fill="#abcdef" d="M20 20 C30 0 40 0 50 20 Z"/>
              </g>
            """)
            result = audit_editability(svg, {"nodes_total": 8})

        details = result["editability_details"]
        self.assertEqual(details["total_subpaths"], 3)
        self.assertEqual(details["multi_subpath_path_count"], 1)
        self.assertEqual(details["max_subpaths_per_path"], 2)
        self.assertEqual(details["path_command_count_max"], 7)
        self.assertEqual(details["total_path_commands"], 10)
        self.assertEqual(details["max_path_command_share"], 0.7)
        self.assertEqual(details["explicit_bezier_control_point_count"], 2)

    def test_bezier_control_handles_are_reported_separately_from_anchors(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            curved = self.write_svg(directory, (
                '<path id="p" fill="none" stroke="#111" '
                'd="M0 0 C3 0 7 0 10 0 C10 3 10 7 10 10"/>'))
            curved_result = audit_editability(curved)
            straight = directory / "straight.svg"
            straight.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<path id="p" fill="none" stroke="#111" '
                'd="M0 0 L10 0 L10 10"/></svg>', encoding="utf-8")
            straight_result = audit_editability(straight)

        curved_details = curved_result["editability_details"]
        straight_details = straight_result["editability_details"]
        self.assertEqual(curved_details["total_path_commands"], 3)
        self.assertEqual(straight_details["total_path_commands"], 3)
        self.assertEqual(curved_details["explicit_bezier_control_point_count"], 4)
        self.assertEqual(straight_details["explicit_bezier_control_point_count"], 0)
        self.assertEqual(
            curved_details["outline_handle_count_estimate"]
            - straight_details["outline_handle_count_estimate"], 4)

    def test_report_json_path_supplies_engine_counts(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(
                directory,
                '<circle id="dot" cx="20" cy="20" r="10" fill="#000"/>',
            )
            report = directory / "counts.json"
            report.write_text(json.dumps({
                "native_circles": 1,
                "native_rectangles": 2,
                "native_lines": 3,
                "native_polylines": 4,
                "native_polygons": 5,
                "native_polygonal_shapes": 9,
                "strokes": 4,
                "gradients": 3,
                "nodes_total": 17,
            }), encoding="utf-8")
            result = audit_editability(svg, report)

        details = result["editability_details"]
        # The polygonal aggregate is informational and is not double counted.
        self.assertEqual(details["native_primitive_count"], 15)
        self.assertEqual(details["stroke_count"], 4)
        self.assertEqual(details["gradient_count"], 3)
        self.assertEqual(details["node_count"], 17)
        self.assertEqual(details["count_sources"]["nodes"], "report_or_stats")

    def test_stale_report_cannot_hide_svg_node_complexity(self):
        path_data = "M0 0 " + " ".join(
            f"L{index % 100} {(index * 3) % 100}" for index in range(600)
        ) + " Z"
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw), f'<path id="dense" fill="#000" d="{path_data}"/>')
            result = audit_editability(svg, {"nodes_total": 1})

        details = result["editability_details"]
        self.assertEqual(details["node_count"], details["svg_estimated_node_count"])
        self.assertGreater(details["node_count"], 1)
        self.assertEqual(
            details["count_sources"]["nodes"],
            "conservative_svg_estimate_over_report",
        )

    def test_structural_five_of_five_is_not_human_task_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw),
                '<g id="mark"><circle id="dot" cx="20" cy="20" r="10" '
                'fill="#65ee22" data-paint-role-fill="accent-1"/></g>',
            )
            result = audit_editability(svg, {
                "designer_operations": {
                    "schema": "ai-vector-cleanroom.designer-operations/v1",
                    "summary": {"passed": 5, "total_operations": 5},
                },
            })

        evidence = result["named_operation_evidence"]
        self.assertEqual(
            evidence["status"], "reported_by_separate_structural_audit")
        self.assertEqual(evidence["structural_checks_passed"], 5)
        self.assertEqual(evidence["structural_checks_total"], 5)
        self.assertTrue(evidence["all_structural_checks_passed"])
        human = result["human_validation"]
        self.assertEqual(human["status"], "not_performed")
        self.assertIsNone(human["original_human_tasks_passed"])
        self.assertIsNone(human["original_human_tasks_total"])
        self.assertFalse(human["timed_editing_test_performed"])

    def test_complex_color_layer_fixture_requires_manual_review(self):
        # 318 paths, 8,310 reported nodes, one 854-command path, 37 colour
        # layers and no drawable IDs reproduce the structural risk profile of
        # the real-world Chiayi logo without embedding that large artifact.
        long_path = "M0 0 " + " ".join(
            f"L{index % 100} {(index * 7) % 100}" for index in range(852)
        ) + " Z"
        simple_paths = [
            f'<path d="M{i % 90} {i % 80} L{(i + 3) % 90} {(i + 5) % 80} Z"/>'
            for i in range(317)
        ]
        buckets = [[] for _ in range(37)]
        buckets[0].append(f'<path d="{long_path}"/>')
        for index, item in enumerate(simple_paths):
            buckets[index % len(buckets)].append(item)
        layers = []
        for index, items in enumerate(buckets, start=1):
            color = f"#{index:02x}{(index * 3) % 256:02x}{(index * 5) % 256:02x}"
            layers.append(
                f'<g id="color-layer-{index}" inkscape:groupmode="layer" '
                f'fill="{color}">' + "".join(items) + "</g>"
            )

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(directory, "".join(layers))
            result = audit_editability(svg, {
                "paths": 318,
                "native_primitives": 83,
                "strokes": 0,
                "gradients": 2,
                "nodes_total": 8310,
            })

        details = result["editability_details"]
        self.assertEqual(result["status"], "manual_review")
        self.assertLess(result["score"], 75)
        self.assertEqual(details["path_count"], 318)
        self.assertEqual(details["node_count"], 8310)
        self.assertEqual(details["group_count"], 37)
        self.assertEqual(details["path_command_count_max"], 854)
        self.assertFalse(details["has_object_ids"])
        self.assertTrue(details["only_color_layers_without_semantic_groups"])
        self.assertIn("path_count_at_least_200", details["review_triggers"])
        self.assertIn("node_count_at_least_4000", details["review_triggers"])
        self.assertIn("one_path_at_least_500_commands", details["review_triggers"])
        raw = details["risk_penalties"]
        families = details["applied_penalty_families"]
        self.assertEqual(
            families["geometry_volume"],
            max(raw["many_paths"], raw["many_nodes"]),
        )
        self.assertLess(
            families["geometry_volume"],
            raw["many_paths"] + raw["many_nodes"],
        )
        self.assertEqual(result["redraw_complexity"]["level"], "high")
        self.assertEqual(result["workflow_friction"]["level"], "very_high")
        self.assertEqual(result["automation_readiness"]["status"], "limited")
        json.dumps(result, ensure_ascii=False)

    def test_semantic_groups_are_selection_handles_not_a_navigation_penalty(self):
        groups = []
        for index in range(30):
            groups.append(
                f'<g id="object-{index}" data-group-mode="actual-dom" '
                f'data-group-reasons="cross-paint-overlay">'
                f'<path id="shape-{index}" fill="#123456" '
                f'd="M{index} 0 L{index + 1} 0 L{index + 1} 1 Z"/>'
                '</g>'
            )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), "".join(groups))
            result = audit_editability(svg, {"nodes_total": 90})

        details = result["editability_details"]
        self.assertEqual(details["actual_dom_group_count"], 30)
        self.assertEqual(details["semantic_group_coverage"], 1.0)
        self.assertNotIn("excessive_group_navigation", details["risk_penalties"])
        self.assertEqual(
            result["automation_readiness"]["components"]["semantic_selection"],
            25.0,
        )
        self.assertEqual(result["status"], "accepted")

    def test_texture_readiness_does_not_hide_a_giant_compound_outline(self):
        circles = "".join(
            f'<circle id="dot-{index}" cx="{index % 20}" cy="{index // 20}" '
            f'r=".3" fill="#65ee22" data-paint-role-fill="accent-1"/>'
            for index in range(80)
        )
        brush = "M0 20 " + " ".join(
            f"L{index % 100} {20 + (index * 7) % 30}" for index in range(620)
        ) + " Z"
        body = (
            '<g id="halftone" data-group-mode="actual-dom" '
            'data-group-reasons="repeated-dot-proximity">' + circles + '</g>'
            '<g id="brush-mark" data-group-mode="actual-dom" '
            'data-group-reasons="cross-paint-overlay">'
            f'<path id="brush" fill="#65ee22" data-paint-role-fill="accent-1" '
            f'd="{brush}"/></g>'
        )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), body)
            result = audit_editability(svg, {"nodes_total": 5000})

        details = result["editability_details"]
        self.assertEqual(result["automation_readiness"]["status"],
                         "ready_for_common_operations")
        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(result["redraw_complexity"]["level"], "high")
        self.assertIn("one_path_at_least_500_commands",
                      details["review_triggers"])
        self.assertIn("single_very_complex_path", details["risk_penalties"])
        self.assertIn("No visual-style or brush-texture discount",
                      details["penalty_combination"])

if __name__ == "__main__":
    unittest.main()
