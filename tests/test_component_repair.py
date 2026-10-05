"""Pure unit tests for conservative phase-one component repair proposals."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from component_repair import (
    append_repair_fragment,
    propose_missing_component_repairs,
    validate_repair_transaction,
)
from quality_diagnostics import compute_quality_diagnostics
from vector_cleanroom import _attempt_isolated_component_repair


SIZE = (64, 64)


def _image(rectangles=(), *, alpha=255):
    width, height = SIZE
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    for x0, y0, x1, y1, colour in rectangles:
        rgba[y0:y1, x0:x1, :3] = colour
        rgba[y0:y1, x0:x1, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def _example(label=1, *, x=20, y=20, width=10, height=4,
             score=0.0, coverage=0.0, fragments=0):
    return {
        "source_component": label,
        "area_px": width * height,
        "score_percent": score,
        "coverage_percent": coverage,
        "connectivity_percent": 0.0 if coverage == 0 else 100.0,
        "fragment_count": fragments,
        "bbox_px": [x, y, width, height],
    }


def _scores(failed=(), *, foreground=90.0, color=90.0,
            detail_p10=90.0, detail_mean=95.0,
            topology_p10=0.0, topology_worst=0.0):
    return {
        "foreground": foreground,
        "foreground_color_fidelity": color,
        "detail_grid": {
            "p10_score_percent": detail_p10,
            "mean_score_percent": detail_mean,
            "component_topology": {
                "p10_score_percent": topology_p10,
                "worst_score_percent": topology_worst,
                "failed_examples": list(failed),
            },
        },
        "transparent_light_fidelity": {
            "applicable": False,
            "coverage_percent": None,
        },
    }


def _grid_component_case(count):
    positions = [
        (6, 6), (26, 6), (46, 6),
        (6, 26), (26, 26), (46, 26),
        (6, 46), (26, 46), (46, 46),
    ]
    rectangles = tuple(
        (x, y, x + 6, y + 4, (0, 0, 0))
        for x, y in positions[:count])
    source = _image(rectangles)
    blank = _image()
    topology = compute_quality_diagnostics(
        blank, source)["detail_grid"]["component_topology"]
    return source, blank, topology["failed_examples"]


class MissingComponentRepairTests(unittest.TestCase):
    def _propose(self, source, render, flat, examples, **kwargs):
        return propose_missing_component_repairs(
            source, render, flat, examples, viewbox=[64, 64], **kwargs)

    def test_proposes_deterministic_append_only_path_for_safe_component(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        render = _image()
        flat = source.copy()
        example = _example()

        first = self._propose(source, render, flat, [example])
        second = self._propose(source, render, flat, [example])

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "proposed")
        self.assertEqual(first["repair_count"], 1)
        self.assertEqual(first["bbox"], [20.0, 20.0, 10.0, 4.0])
        self.assertIn('id="component-repairs"', first["svg_fragment"])
        self.assertIn('fill-rule="evenodd"', first["svg_fragment"])
        self.assertIn('fill="#000000"', first["svg_fragment"])
        self.assertEqual(first["repairs"][0]["source_component"], 1)
        self.assertEqual(
            first["audit"]["records"][0]["status"], "proposed")

    def test_shifted_nearby_render_is_not_treated_as_missing(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        # One blank column separates the shifted copy from the source bbox;
        # the three-pixel render moat must still see it.
        render = _image(((31, 20, 41, 24, (0, 0, 0)),))

        result = self._propose(source, render, source, [_example()])

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["audit"]["records"][0]["reason"],
            "render_moat_not_clear")

    def test_source_supported_render_neighbour_is_not_a_shifted_duplicate(self):
        source = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            # A blank column keeps this a distinct topology label while the
            # conservative three-pixel audit moat still reaches it.
            (31, 20, 35, 24, (0, 0, 0)),
        ))
        render = _image(((31, 20, 35, 24, (0, 0, 0)),))

        result = self._propose(source, render, source, [_example()])

        self.assertEqual(result["status"], "proposed")
        record = result["audit"]["records"][0]
        self.assertEqual(record["source_gap_pixels"], 0)
        self.assertGreater(record["source_moat_pixels"], 0)
        self.assertGreater(record["render_moat_pixels"], 0)
        self.assertEqual(record["render_moat_unexpected_pixels"], 0)
        self.assertEqual(record["render_moat_supported_pixels"],
                         record["render_moat_pixels"])
        self.assertEqual(record["render_inner_gap_pixels"], 0)
        self.assertEqual(
            record["touching_render_component_unsupported_pixels"], 0)
        self.assertEqual(record["source_support_components"], [2])

    def test_source_supported_neighbour_cannot_bridge_inner_gap(self):
        source = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            (31, 20, 35, 24, (0, 0, 0)),
        ))
        # x=30 is the target's required blank inner ring.  It is also within
        # the neighbour's alignment halo, so support ownership alone must not
        # be allowed to erase the gap.
        render = _image(((30, 20, 35, 24, (0, 0, 0)),))

        result = self._propose(source, render, source, [_example()])

        self.assertEqual(result["status"], "skipped")
        record = result["audit"]["records"][0]
        self.assertEqual(record["reason"], "separation_carve_limit")
        self.assertGreater(record["render_inner_gap_pixels"], 0)

    def test_complete_touching_render_component_must_be_source_supported(self):
        source = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            (31, 20, 35, 24, (0, 0, 0)),
        ))
        # The leading pixels inside the three-pixel moat are source-supported,
        # but the connected tail extends well beyond the support halo.
        render = _image(((31, 20, 41, 24, (0, 0, 0)),))

        result = self._propose(source, render, source, [_example()])

        self.assertEqual(result["status"], "skipped")
        record = result["audit"]["records"][0]
        self.assertEqual(
            record["reason"], "shifted_duplicate_detected")
        self.assertEqual(record["render_moat_unexpected_pixels"], 0)
        self.assertEqual(
            record["shifted_duplicate_scan"]["status"], "detected")
        self.assertLess(
            record["shifted_duplicate_scan"]["detected_match"]
            ["source_support"], 0.9)

    def test_below_topology_area_source_noise_cannot_own_render(self):
        source = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            (31, 20, 33, 22, (0, 0, 0)),
        ))
        render = _image(((31, 20, 33, 22, (0, 0, 0)),))

        result = self._propose(source, render, source, [_example()])

        self.assertEqual(result["status"], "skipped")
        record = result["audit"]["records"][0]
        self.assertEqual(
            record["reason"], "render_moat_not_clear")
        self.assertEqual(record["source_support_components"], [])

    def test_small_source_owned_gap_intrusion_is_carved_fail_closed(self):
        source = _image((
            (20, 20, 30, 30, (0, 0, 0)),
            (31, 23, 35, 27, (0, 0, 0)),
        ))
        render_array = np.asarray(_image((
            (31, 23, 35, 27, (0, 0, 0)),
        ))).copy()
        render_array[25, 30, :3] = (0, 0, 0)
        render = Image.fromarray(render_array, "RGBA")
        example = _example(width=10, height=10)

        result = self._propose(source, render, source, [example])

        self.assertEqual(result["status"], "proposed")
        record = result["audit"]["records"][0]
        self.assertEqual(record["render_inner_gap_pixels"], 1)
        self.assertEqual(record["render_inner_gap_unexpected_pixels"], 0)
        self.assertEqual(record["separation_carve_pixels"], 3)
        self.assertEqual(record["trace_area_px"], 97)
        self.assertEqual(result["repairs"][0]["trace_area_px"], 97)

    def test_multicolour_or_translucent_flat_component_is_skipped(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        multicolour = np.asarray(source).copy()
        multicolour[20:24, 25:30, :3] = (220, 0, 0)
        translucent = np.asarray(source).copy()
        translucent[20:24, 20:30, 3] = 200

        with self.subTest("multicolour"):
            result = self._propose(
                source, _image(), Image.fromarray(multicolour, "RGBA"),
                [_example()])
            self.assertEqual(
                result["audit"]["records"][0]["reason"],
                "flat_component_not_single_colour")
        with self.subTest("translucent"):
            result = self._propose(
                source, _image(), Image.fromarray(translucent, "RGBA"),
                [_example()])
            self.assertEqual(
                result["audit"]["records"][0]["reason"],
                "flat_component_not_opaque")

    def test_fragmented_failure_is_skipped_before_tracing(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))

        result = self._propose(
            source, _image(), source, [_example(fragments=2)])

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["audit"]["records"][0]["reason"],
            "fragmented_component")

    def test_component_inside_edge_guard_is_skipped(self):
        source = _image(((1, 20, 11, 24, (0, 0, 0)),))
        example = _example(x=1)

        result = self._propose(source, _image(), source, [example])

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["audit"]["records"][0]["reason"],
            "component_touches_edge_guard")

    def test_component_count_and_area_limits_skip_the_whole_proposal(self):
        source = _image((
            (10, 10, 20, 14, (0, 0, 0)),
            (35, 35, 45, 39, (0, 0, 0)),
        ))
        examples = [
            _example(1, x=10, y=10),
            _example(2, x=35, y=35),
        ]

        with self.subTest("count"):
            result = self._propose(
                source, _image(), source, examples, max_components=1)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(
                result["audit"]["skipped_reason"],
                "component_count_limit")
        with self.subTest("component-area"):
            result = self._propose(
                source, _image(), source, [examples[0]],
                max_component_area=39)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(
                result["audit"]["records"][0]["reason"],
                "component_area_limit")
        with self.subTest("total-area"):
            result = self._propose(
                source, _image(), source, examples, max_total_area=79)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(
                result["audit"]["skipped_reason"], "total_area_limit")

    def test_default_batch_proposes_five_safe_components_deterministically(self):
        source, blank, examples = _grid_component_case(5)

        first = self._propose(source, blank, source, examples)
        second = self._propose(source, blank, source, examples)

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "proposed")
        self.assertEqual(first["repair_count"], 5)
        self.assertEqual(first["audit"]["limits"]["max_components"], 8)
        self.assertEqual(
            [item["source_component"] for item in first["repairs"]],
            [1, 2, 3, 4, 5])

    def test_default_batch_still_fails_closed_above_eight_components(self):
        source, blank, examples = _grid_component_case(9)

        result = self._propose(source, blank, source, examples)

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["svg_fragment"], "")
        self.assertEqual(
            result["audit"]["skipped_reason"], "component_count_limit")
        self.assertEqual(len(result["audit"]["records"]), 9)
        self.assertTrue(all(
            record.get("reason") == "component_count_limit"
            for record in result["audit"]["records"]))

    def test_non_one_to_one_viewbox_is_skipped_with_reason(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))

        result = propose_missing_component_repairs(
            source, _image(), source, [_example()], viewbox=[128, 64])

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(
            result["audit"]["skipped_reason"], "viewbox_not_one_to_one")

    def test_node_limits_reject_overcomplex_proposals(self):
        source = _image((
            (10, 10, 20, 14, (0, 0, 0)),
            (35, 35, 45, 39, (0, 0, 0)),
        ))
        examples = [
            _example(1, x=10, y=10),
            _example(2, x=35, y=35),
        ]

        with self.subTest("per-component"):
            result = self._propose(
                source, _image(), source, [examples[0]],
                max_component_nodes=3)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(
                result["audit"]["records"][0]["reason"],
                "component_node_limit")
        with self.subTest("total"):
            result = self._propose(
                source, _image(), source, examples,
                max_component_nodes=8, max_total_nodes=7)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(
                result["audit"]["skipped_reason"], "total_node_limit")

    def test_high_resolution_source_uses_diagnostic_lanczos_coordinates(self):
        measurement_source = _image(
            ((20, 20, 30, 24, (0, 0, 0)),))
        high_resolution = measurement_source.resize(
            (128, 128), Image.Resampling.NEAREST)
        render = _image()
        topology = compute_quality_diagnostics(
            render, high_resolution)["detail_grid"]["component_topology"]

        result = self._propose(
            high_resolution, render, measurement_source,
            topology["failed_examples"])

        self.assertEqual(result["status"], "proposed")
        self.assertEqual(
            result["audit"]["source_normalisation"]["status"],
            "resampled")
        self.assertEqual(
            result["audit"]["source_normalisation"]["method"],
            "Pillow.LANCZOS")

    def test_incompatible_measurement_geometry_is_skipped(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))

        with self.subTest("render-flat"):
            result = propose_missing_component_repairs(
                source, _image(), Image.new("RGBA", (32, 32), "white"),
                [_example()], viewbox=[64, 64])
            self.assertEqual(
                result["audit"]["skipped_reason"],
                "render_flat_size_mismatch")
        with self.subTest("source-aspect"):
            result = propose_missing_component_repairs(
                Image.new("RGBA", (128, 64), "white"), _image(), source,
                [_example()], viewbox=[64, 64])
            self.assertEqual(
                result["audit"]["skipped_reason"],
                "source_render_aspect_mismatch")

    def test_fragment_append_is_pure_and_rejects_unsafe_xml(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        proposal = self._propose(source, _image(), source, [_example()])
        base = (
            b'<svg xmlns="http://www.w3.org/2000/svg" '
            b'width="64" height="64"></svg>\n')

        result = append_repair_fragment(base, proposal["svg_fragment"])

        self.assertEqual(base, (
            b'<svg xmlns="http://www.w3.org/2000/svg" '
            b'width="64" height="64"></svg>\n'))
        self.assertIn(b'id="component-repairs"', result)
        self.assertTrue(result.endswith(b'</svg>\n'))
        with self.assertRaisesRegex(ValueError, "already contains"):
            append_repair_fragment(result, proposal["svg_fragment"])
        with self.assertRaisesRegex(ValueError, "safe path"):
            append_repair_fragment(
                base,
                '<g id="component-repairs"><image href="x"/></g>')

    def test_transaction_accepts_local_non_regressing_repair(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        before = _image()
        proposal = self._propose(source, before, source, [_example()])
        before_scores = _scores((_example(),))
        after_scores = _scores(
            (), foreground=91.0, color=91.0, detail_p10=92.0,
            detail_mean=96.0, topology_p10=100.0,
            topology_worst=100.0)

        audit = validate_repair_transaction(
            proposal, before_scores, after_scores,
            {"status": "rejected"}, {"status": "accepted"},
            before, source, source_reference=source)

        self.assertEqual(audit["status"], "accepted")
        self.assertEqual(audit["reasons"], [])
        self.assertEqual(audit["outside_allowed_pixels"], 0)
        self.assertEqual(audit["unresolved_target_components"], [])

    def test_transaction_accepts_bounded_carve_that_preserves_separation(self):
        source = _image((
            (20, 20, 30, 30, (0, 0, 0)),
            (31, 23, 35, 27, (0, 0, 0)),
        ))
        before_array = np.asarray(_image((
            (31, 23, 35, 27, (0, 0, 0)),
        ))).copy()
        before_array[25, 30, :3] = (0, 0, 0)
        before = Image.fromarray(before_array, "RGBA")
        example = _example(width=10, height=10)
        proposal = self._propose(source, before, source, [example])
        after_array = before_array.copy()
        after_array[20:30, 20:30, :3] = (0, 0, 0)
        # This is the authenticated three-pixel carve around the one existing
        # source-owned inner-gap pixel at (30, 25).
        after_array[24:27, 29, :3] = (255, 255, 255)
        after = Image.fromarray(after_array, "RGBA")

        audit = validate_repair_transaction(
            proposal, _scores((example,)),
            _scores((), foreground=91.0, color=91.0,
                    detail_p10=92.0, detail_mean=96.0,
                    topology_p10=100.0, topology_worst=100.0),
            {"status": "rejected"}, {"status": "accepted"},
            before, after, source_reference=source)

        self.assertEqual(audit["status"], "accepted")
        record = audit["post_render_gap_guard"]["records"][0]
        self.assertEqual(record["before_render_inner_gap_core_pixels"], 1)
        self.assertEqual(record["after_render_inner_gap_core_pixels"], 1)
        self.assertEqual(record["new_render_inner_gap_core_pixels"], 0)
        self.assertEqual(
            record["new_render_to_preexisting_connection_pixels"], 0)

    def test_transaction_rejects_outside_changes_and_unresolved_targets(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        before = _image()
        proposal = self._propose(source, before, source, [_example()])
        outside = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            (45, 45, 50, 50, (0, 0, 0)),
        ))

        with self.subTest("outside"):
            audit = validate_repair_transaction(
                proposal, _scores((_example(),)),
                _scores((), topology_p10=100.0, topology_worst=100.0),
                {"status": "rejected"}, {"status": "accepted"},
                before, outside, source_reference=source)
            self.assertEqual(audit["status"], "rejected")
            self.assertIn(
                "render_changed_outside_repair_bbox", audit["reasons"])
        with self.subTest("unresolved"):
            audit = validate_repair_transaction(
                proposal, _scores((_example(),)), _scores((_example(),)),
                {"status": "rejected"}, {"status": "rejected"},
                before, source, source_reference=source)
            self.assertEqual(audit["status"], "rejected")
            self.assertIn("target_component_not_repaired", audit["reasons"])

    def test_transaction_rejects_metric_or_gate_regression(self):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        before = _image()
        proposal = self._propose(source, before, source, [_example()])
        degraded = _scores(
            (), foreground=80.0, color=80.0, detail_p10=80.0,
            detail_mean=80.0, topology_p10=100.0,
            topology_worst=100.0)

        audit = validate_repair_transaction(
            proposal, _scores((_example(),)), degraded,
            {"status": "accepted"}, {"status": "rejected"},
            before, source, source_reference=source)

        self.assertEqual(audit["status"], "rejected")
        self.assertIn("visual_gate_regressed", audit["reasons"])
        self.assertIn("foreground_regressed", audit["reasons"])

    def test_transaction_rejects_post_render_inner_gap_bridge(self):
        source = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            (31, 20, 35, 24, (0, 0, 0)),
        ))
        before = _image(((31, 20, 35, 24, (0, 0, 0)),))
        proposal = self._propose(source, before, source, [_example()])
        bridged = _image((
            (20, 20, 30, 24, (0, 0, 0)),
            (30, 20, 35, 24, (0, 0, 0)),
        ))

        audit = validate_repair_transaction(
            proposal, _scores((_example(),)),
            _scores((), topology_p10=100.0, topology_worst=100.0),
            {"status": "rejected"}, {"status": "accepted"},
            before, bridged, source_reference=source)

        self.assertEqual(audit["status"], "rejected")
        self.assertIn("post_render_added_inner_gap_core", audit["reasons"])
        self.assertGreater(
            audit["post_render_gap_guard"]["records"][0]
            ["new_render_inner_gap_core_pixels"], 0)


class ComponentRepairPipelineTests(unittest.TestCase):
    def _run_pipeline(self, *, resolved):
        source = _image(((20, 20, 30, 24, (0, 0, 0)),))
        before_render_image = _image()
        before_scores = _scores((_example(),))
        after_scores = _scores(
            () if resolved else (_example(),),
            foreground=91.0, color=91.0, detail_p10=92.0,
            detail_mean=96.0,
            topology_p10=100.0 if resolved else 0.0,
            topology_worst=100.0 if resolved else 0.0)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            svg = td / "candidate.svg"
            flat = td / "flat.png"
            source_path = td / "source.png"
            before_render = td / "before.png"
            original = (
                b'<svg xmlns="http://www.w3.org/2000/svg" '
                b'width="64" height="64" viewBox="0 0 64 64"></svg>\n')
            svg.write_bytes(original)
            source.save(source_path)
            source.save(flat)
            before_render_image.save(before_render)
            stats = SimpleNamespace(
                viewbox=[64, 64], gradient_info=[], n_paths=0, n_nodes=0,
                geometry_notes=[])

            def fake_self_check(_svg, _flat, _source, **kwargs):
                source.save(kwargs["keep_render"])
                return after_scores

            with mock.patch(
                    "vector_cleanroom.self_check",
                    side_effect=fake_self_check), mock.patch(
                    "vector_cleanroom._evaluate_visual_gate",
                    side_effect=[
                        {"status": "rejected"},
                        {"status": "accepted" if resolved else "rejected"},
                    ]):
                scores, audit = _attempt_isolated_component_repair(
                    svg, flat, source_path, stats, before_scores,
                    before_render)
            return {
                "scores": scores,
                "audit": audit,
                "bytes": svg.read_bytes(),
                "original": original,
                "stats": stats,
                "temp_artifacts": sorted(path.name for path in td.iterdir()
                                         if path.name.startswith(
                                             "_component_repair")),
            }

    def test_pipeline_commits_only_after_render_transaction_accepts(self):
        result = self._run_pipeline(resolved=True)

        self.assertEqual(result["audit"]["status"], "committed")
        self.assertIn(b'id="component-repairs"', result["bytes"])
        self.assertEqual(result["stats"].n_paths, 1)
        self.assertGreater(result["stats"].n_nodes, 0)
        self.assertEqual(result["temp_artifacts"], [])
        self.assertEqual(result["scores"]["foreground"], 91.0)

    def test_pipeline_rollback_leaves_live_svg_byte_identical(self):
        result = self._run_pipeline(resolved=False)

        self.assertEqual(result["audit"]["status"], "rolled_back")
        self.assertEqual(result["bytes"], result["original"])
        self.assertTrue(result["audit"]["live_svg_unchanged"])
        self.assertEqual(result["stats"].n_paths, 0)
        self.assertEqual(result["temp_artifacts"], [])

    def test_pipeline_commits_five_only_after_all_targets_resolve(self):
        source, blank, failed = _grid_component_case(5)
        before_scores = _scores(failed)
        after_scores = _scores(
            (), foreground=91.0, color=91.0, detail_p10=92.0,
            detail_mean=96.0, topology_p10=100.0,
            topology_worst=100.0)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            svg = td / "candidate.svg"
            flat = td / "flat.png"
            source_path = td / "source.png"
            before_render = td / "before.png"
            svg.write_bytes(
                b'<svg xmlns="http://www.w3.org/2000/svg" '
                b'width="64" height="64" viewBox="0 0 64 64"></svg>\n')
            source.save(source_path)
            source.save(flat)
            blank.save(before_render)
            stats = SimpleNamespace(
                viewbox=[64, 64], gradient_info=[], n_paths=0, n_nodes=0,
                geometry_notes=[])

            def fake_self_check(_svg, _flat, _source, **kwargs):
                source.save(kwargs["keep_render"])
                return after_scores

            with mock.patch(
                    "vector_cleanroom.self_check",
                    side_effect=fake_self_check), mock.patch(
                    "vector_cleanroom._evaluate_visual_gate",
                    side_effect=[{"status": "rejected"},
                                 {"status": "accepted"}]):
                scores, audit = _attempt_isolated_component_repair(
                    svg, flat, source_path, stats, before_scores,
                    before_render)

            self.assertEqual(scores, after_scores)
            self.assertEqual(audit["status"], "committed")
            self.assertEqual(audit["repair_count"], 5)
            self.assertEqual(stats.n_paths, 5)
            self.assertEqual(
                audit["transaction"]["target_components"], [1, 2, 3, 4, 5])
            self.assertEqual(
                audit["transaction"]["unresolved_target_components"], [])
            self.assertFalse(any(
                path.name.startswith("_component_repair")
                for path in td.iterdir()))

    def test_pipeline_prefilter_skips_partial_failures_without_io(self):
        partial = _example(score=50.0, coverage=50.0, fragments=1)
        scores, audit = _attempt_isolated_component_repair(
            Path("does-not-exist.svg"), Path("does-not-exist-flat.png"),
            Path("does-not-exist-source.png"),
            SimpleNamespace(viewbox=[64, 64], gradient_info=[]),
            _scores((partial,)), Path("does-not-exist-render.png"))

        self.assertEqual(scores["foreground"], 90.0)
        self.assertEqual(audit["status"], "skipped")
        self.assertEqual(
            audit["reason"], "no_completely_missing_components")
        self.assertEqual(audit["completely_missing_examples"], 0)


if __name__ == "__main__":
    unittest.main()
