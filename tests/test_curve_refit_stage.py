# -*- coding: utf-8 -*-

import copy
import hashlib
import json
import tempfile
import unittest
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

from curve_refit_stage import (
    _sample_subpath,
    apply_svg_curve_refit_path_frontier_candidate,
    build_svg_curve_refit_path_frontier,
    propose_svg_curve_refit,
)


class CurveRefitStageTests(unittest.TestCase):
    def _write(self, folder, path_data):
        source = Path(folder) / "source.svg"
        source.write_text(
            '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 100 100"><g fill="#126b45"><path id="shape" d="'
            + path_data + '"/></g></svg>', encoding="utf-8")
        return source

    def _path_from_points(self, points):
        values = [f"M{points[0][0]:.6f} {points[0][1]:.6f}"]
        values.extend(f"L{point[0]:.6f} {point[1]:.6f}"
                      for point in points[1:])
        values.append("Z")
        return " ".join(values)

    def _elements(self, svg_path, local_name):
        root = ET.parse(svg_path).getroot()
        return [item for item in root.iter()
                if item.tag.rsplit("}", 1)[-1] == local_name]

    def _identity_optimizer_result(self, anchors=3, loops=1):
        return {
            "schema_version": 1,
            "status": "identity_rollback_no_safe_reduction",
            "identity_rollback_selected": True,
            "safe_refit_selected": False,
            "selection_outcome": "source_identity_rollback",
            "optimization_basis": "geometry_only",
            "uses_colour_or_pixel_similarity": False,
            "error_budget_percent": 0.25,
            "anchors_before": anchors,
            "anchors_after": anchors,
            "designer_anchor_count": anchors,
            "segment_count_after": anchors,
            "fit": {"source_identity": True},
            "path": "M10 10 L90 10 L50 90 Z",
            "actual_p95_error_percent": 0.0,
            "actual_max_error_percent": 0.0,
            "over_budget_share": 0.0,
            "salient_corner_max_percent": 0.0,
            "selected_candidate_id": "source_identity",
            "selected_source": "source_identity_rollback",
            "candidate_count": 9,
            "eligible_candidate_count": 1,
            "lexicographic_objective": [
                "preserve_topology_hard_constraint",
                ("p95_bidirectional_geometric_error_percent_within_budget_"
                 "hard_constraint"),
                ("maximum_error_within_three_times_budget_hard_tail_"
                 "constraint"),
                ("over_budget_share_at_most_5_percent_hard_tail_"
                 "constraint"),
                ("salient_corner_error_within_two_times_budget_hard_"
                 "constraint"),
                "minimize_designer_anchor_count",
                "minimize_anchor_count",
                "minimize_fragment_count",
                "minimize_segment_count",
                "prefer_simpler_primitive_category_when_economy_equal",
            ],
            "loop_count": loops,
            "primitive_complexity": {
                "category": "source_polyline_rollback",
                "native_primitives": [],
            },
        }

    def test_single_path_frontier_is_geometry_only_and_applies_authoritative_row(self):
        source_d = " ".join(
            f"M{x} 0 L{x + 3} 0 L{x + 3} 3 L{x} 3 Z"
            for x in (0, 10, 20, 30, 40))
        candidate_d = " ".join((
            "M0 0 L3 0 L0 3 Z",
            "M10 0 L13 0 L10 3 Z",
            "M20 0 L23 0 L23 3 L20 3 Z",
            "M30 0 L33 0 L30 3 Z",
            "M40 0 L43 0 L43 3 L40 3 Z",
        ))
        row = {
            "candidate_id": "compound_refinement_loop_02_next",
            "candidate_kind": "single_loop_refinement",
            "eligible": True,
            "changed_loop_index": 2,
            "replacement_candidate_id": "loop_02_curve_refit_02",
            "path": candidate_d,
            "anchors_after": 17,
            "designer_anchor_count": 17,
            "segment_count": 17,
            "actual_p95_error_percent": 0.20,
            "actual_max_error_percent": 0.30,
            "over_budget_share": 0.01,
            "salient_corner_max_percent": 0.20,
            "primitive_complexity": {
                "category": "bezier_geometry", "native_primitives": []},
        }
        optimizer_result = {
            "anchors_after": 16,
            "designer_anchor_count": 16,
            "selected_candidate_id": "curve_refit_mixed_loops",
            "path": "base-path",
            "actual_p95_error_percent": 0.21,
            "actual_max_error_percent": 0.31,
            "safe_refit_selected": True,
            "lexicographic_objective": [
                "preserve_topology_hard_constraint",
                "minimize_designer_anchor_count",
            ],
            "refinement_frontier": {
                "candidates": [row, {
                    **row,
                    "candidate_id": "identity",
                    "candidate_kind": "source_identity_control",
                    "anchors_after": 400,
                    "designer_anchor_count": 400,
                }],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.svg"
            source.write_text(
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg">'
                '<g fill="#fefefe" fill-rule="evenodd">'
                f'<path id="target" d="{source_d}"/>'
                '<path id="decoy" d="M90 0 L91 1 Z"/>'
                '</g></svg>', encoding="utf-8")
            with patch(
                    "curve_refit_stage.optimize_compound_contours",
                    return_value=optimizer_result) as optimizer:
                frontier = build_svg_curve_refit_path_frontier(
                    source, "target", error_budget_percent=0.25)

            self.assertEqual(frontier["status"], "candidates_available")
            self.assertFalse(frontier["uses_colour_or_pixel_similarity"])
            self.assertEqual(frontier["source_anchor_count"], 20)
            self.assertEqual(frontier["candidate_count"], 1)
            self.assertEqual(
                frontier["candidates"][0]["path_data_sha256"],
                hashlib.sha256(candidate_d.encode("utf-8")).hexdigest())
            optimizer.assert_called_once()
            self.assertTrue(
                optimizer.call_args.kwargs["include_refinement_frontier"])
            applied = apply_svg_curve_refit_path_frontier_candidate(
                source.read_bytes(), target_id="target",
                frontier=frontier, candidate=frontier["candidates"][0])
            source_bytes = source.read_bytes()

        root = ET.fromstring(applied["bytes"])
        by_id = {item.get("id"): item for item in root.iter()
                 if item.get("id")}
        self.assertEqual(by_id["target"].get("d"), candidate_d)
        self.assertEqual(by_id["decoy"].get("d"), "M90 0 L91 1 Z")
        self.assertEqual(
            by_id["target"].get("data-avc-anchors-after"), "17")
        detail = applied["detail"]
        self.assertTrue(detail["economy_certified"])
        self.assertEqual(detail["selected_candidate_id"], row["candidate_id"])
        self.assertEqual(detail["anchors_before"], 20)
        self.assertEqual(detail["anchors_after"], 17)
        self.assertEqual(
            detail["path_data_sha256"],
            hashlib.sha256(candidate_d.encode("utf-8")).hexdigest())

        tampered = source_bytes.replace(
            source_d.encode("utf-8"), b"M0 0 L1 1 Z")
        with self.assertRaisesRegex(RuntimeError, "source geometry changed"):
            apply_svg_curve_refit_path_frontier_candidate(
                tampered, target_id="target", frontier=frontier,
                candidate=frontier["candidates"][0])

        digest_tamper = copy.deepcopy(frontier)
        digest_tamper["candidates"][0]["path_data_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "candidate digest mismatch"):
            apply_svg_curve_refit_path_frontier_candidate(
                source_bytes, target_id="target", frontier=digest_tamper,
                candidate=digest_tamper["candidates"][0])

    def test_reduces_redundant_closed_curve(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(
                folder,
                "M10 10 C12 10 14 10 16 10 C18 10 20 10 22 10 "
                "C24 10 26 10 28 10 C30 10 32 10 34 10 "
                "L34 34 C30 34 26 34 22 34 C18 34 14 34 10 34 Z")
            candidate = Path(folder) / "candidate.svg"
            report = propose_svg_curve_refit(
                source, candidate, tolerance=0.4, sample_step=1.5,
                minimum_nodes=6, minimum_reduction_ratio=0.1)
            self.assertEqual(report["status"], "proposed")
            self.assertGreater(report["anchors_removed"], 0)
            self.assertIn("data-avc-curve-refit", candidate.read_text("utf-8"))
            final_path = self._elements(candidate, "path")[0]
            detail = report["details"][0]
            self.assertEqual(detail["final_element"], "path")
            self.assertEqual(detail["final_drawable_id"], "shape")
            self.assertEqual(
                detail["path_data_sha256"],
                hashlib.sha256(
                    final_path.get("d").encode("utf-8")).hexdigest())
            self.assertEqual(
                detail["path_data_digest_scope"],
                "utf8_svg_path_d_attribute")

    def test_idless_evaluated_paths_receive_stable_unique_svg_ids(self):
        dense = (
            "M10 10 C12 10 14 10 16 10 C18 10 20 10 22 10 "
            "C24 10 26 10 28 10 C30 10 32 10 34 10 "
            "L34 34 C30 34 26 34 22 34 C18 34 14 34 10 34 Z"
        )
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.svg"
            source.write_text(
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 100 100"><g fill="#126b45">'
                f'<path d="{dense}"/><path d="{dense}"/>'
                '</g></svg>', encoding="utf-8")
            first = Path(folder) / "first.svg"
            second = Path(folder) / "second.svg"
            first_report = propose_svg_curve_refit(
                source, first, tolerance=0.4, sample_step=1.5,
                minimum_nodes=6, minimum_reduction_ratio=0.1)
            second_report = propose_svg_curve_refit(
                source, second, tolerance=0.4, sample_step=1.5,
                minimum_nodes=6, minimum_reduction_ratio=0.1)

            self.assertEqual(first_report["status"], "proposed")
            first_ids = [item["final_drawable_id"]
                         for item in first_report["details"]]
            second_ids = [item["final_drawable_id"]
                          for item in second_report["details"]]
            self.assertEqual(first_ids, second_ids)
            self.assertEqual(len(first_ids), 2)
            self.assertEqual(len(first_ids), len(set(first_ids)))
            self.assertTrue(all(identifier.startswith("avc-refit-path-")
                                for identifier in first_ids))
            final_paths = self._elements(first, "path")
            self.assertEqual(
                [item.get("id") for item in final_paths], first_ids)
            self.assertEqual(
                [item["id"] for item in first_report["details"]], first_ids)
            stable = first_report["stable_id_normalization"]
            self.assertEqual(stable["source_svg_sha256"], hashlib.sha256(
                source.read_bytes()).hexdigest())
            self.assertEqual(stable["assigned_id_count"], 2)
            self.assertEqual(
                [item["global_path_ordinal_1_based"]
                 for item in stable["records"]], [1, 2])
            self.assertEqual(
                [item["source_path_data_sha256"]
                 for item in stable["records"]],
                [hashlib.sha256(dense.encode("utf-8")).hexdigest()] * 2)
            self.assertTrue(all(
                item["original_id_state"] == "missing_assigned"
                and item["assignment_applied"] is True
                for item in stable["records"]))

    def test_stable_id_assignment_is_collision_safe_and_preserves_existing_ids(self):
        dense = (
            "M10 10 C12 10 14 10 16 10 C18 10 20 10 22 10 "
            "C24 10 26 10 28 10 C30 10 32 10 34 10 "
            "L34 34 C30 34 26 34 22 34 C18 34 14 34 10 34 Z"
        )
        digest = hashlib.sha256(dense.encode("utf-8")).hexdigest()
        collision = f"avc-refit-path-2-{digest[:12]}"
        optimizer_result = {
            "anchors_after": 1,
            "designer_anchor_count": 1,
            "path": "",
            "primitive_complexity": {},
        }
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.svg"
            source.write_text(
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg">'
                f'<path id="kept" fill="#126b45" d="{dense}"/>'
                f'<g id="{collision}" fill="#126b45"><path d="{dense}"/>'
                '</g></svg>', encoding="utf-8")
            candidate = Path(folder) / "candidate.svg"
            with patch("curve_refit_stage.optimize_compound_contours",
                       return_value=optimizer_result):
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=4)

            records = report["stable_id_normalization"]["records"]
            self.assertEqual(records[0]["original_id_state"],
                             "existing_preserved")
            self.assertEqual(records[0]["assigned_id"], "kept")
            self.assertFalse(records[0]["assignment_applied"])
            self.assertEqual(records[1]["original_id_state"],
                             "missing_assigned")
            self.assertEqual(records[1]["assigned_id"], collision + "-2")
            ids = [item.get("id") for item in ET.parse(candidate).getroot().iter()
                   if item.get("id")]
            self.assertEqual(len(ids), len(set(ids)))

    def test_duplicate_existing_svg_ids_fail_closed_before_candidate_write(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.svg"
            source.write_text(
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg">'
                '<path id="duplicate" d="M0 0L10 0L10 10Z"/>'
                '<path id="duplicate" d="M20 0L30 0L30 10Z"/>'
                '</svg>', encoding="utf-8")
            candidate = Path(folder) / "candidate.svg"
            with self.assertRaisesRegex(RuntimeError,
                                        "duplicate existing SVG IDs"):
                propose_svg_curve_refit(source, candidate, minimum_nodes=3)
            self.assertFalse(candidate.exists())

    def test_rotated_elliptical_arc_sampling_follows_svg_geometry(self):
        rotation_degrees = 31.0
        rotation = math.radians(rotation_degrees)
        cx, cy, rx, ry = 50.0, 40.0, 30.0, 12.0
        start = (
            cx + math.cos(rotation) * rx,
            cy + math.sin(rotation) * rx,
        )
        endpoint = (
            cx - math.sin(rotation) * ry,
            cy + math.cos(rotation) * ry,
        )
        subpath = {
            "start": start,
            "segs": [[
                "A", rx, ry, rotation_degrees, 0.0, 1.0,
                endpoint[0], endpoint[1],
            ]],
            "closed": False,
        }
        points = _sample_subpath(subpath, sample_step=1.0)

        self.assertIsNotNone(points)
        self.assertGreater(len(points), 20)
        self.assertEqual(points[0], start)
        self.assertEqual(points[-1], endpoint)
        for x, y in points:
            dx, dy = x - cx, y - cy
            local_x = math.cos(rotation) * dx + math.sin(rotation) * dy
            local_y = -math.sin(rotation) * dx + math.cos(rotation) * dy
            self.assertAlmostEqual(
                (local_x / rx) ** 2 + (local_y / ry) ** 2,
                1.0, places=10)

        large_reverse = _sample_subpath({
            **subpath,
            "segs": [[
                "A", rx, ry, rotation_degrees, 1.0, 0.0,
                endpoint[0], endpoint[1],
            ]],
        }, sample_step=1.0)
        self.assertIsNotNone(large_reverse)
        self.assertGreater(len(large_reverse), 2 * len(points))
        self.assertEqual(large_reverse[-1], endpoint)

        corrected = _sample_subpath({
            "start": (0.0, 0.0),
            "segs": [["A", 10.0, 5.0, 0.0, 0.0, 1.0, 100.0, 0.0]],
            "closed": False,
        }, sample_step=2.0)
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected[0], (0.0, 0.0))
        self.assertEqual(corrected[-1], (100.0, 0.0))
        for x, y in corrected:
            self.assertAlmostEqual(
                ((x - 50.0) / 50.0) ** 2 + (y / 25.0) ** 2,
                1.0, places=10)
        self.assertLess(min(point[1] for point in corrected), -24.99)

    def test_degenerate_or_invalid_arcs_fail_closed(self):
        cases = {
            "zero-radius": ["A", 0.0, 10.0, 0.0, 0.0, 1.0, 30.0, 10.0],
            "same-endpoint": ["A", 10.0, 10.0, 0.0, 0.0, 1.0, 10.0, 10.0],
            "invalid-flag": ["A", 10.0, 10.0, 0.0, 2.0, 1.0, 30.0, 10.0],
            "non-finite": ["A", float("inf"), 10.0, 0.0, 0.0, 1.0,
                           30.0, 10.0],
        }
        for label, segment in cases.items():
            with self.subTest(label=label):
                self.assertIsNone(_sample_subpath({
                    "start": (10.0, 10.0),
                    "segs": [segment],
                    "closed": False,
                }, sample_step=1.0))

    def test_degenerate_arc_path_is_preserved_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(
                folder, "M10 10 A0 20 0 1 0 50 10 L50 50 L10 50 Z")
            original_d = self._elements(source, "path")[0].get("d")
            candidate = Path(folder) / "candidate.svg"
            report = propose_svg_curve_refit(
                source, candidate, minimum_nodes=3)
            self.assertEqual(report["status"], "no_change")
            self.assertIn("unsupported_or_degenerate_geometry", report["skipped"])
            candidate_path = self._elements(candidate, "path")[0]
            self.assertEqual(candidate_path.get("d"), original_d)
            self.assertIsNone(candidate_path.get("data-avc-curve-refit"))

    def test_arc_only_path_and_frontier_remain_ineligible(self):
        path_data = (
            "M10 50 A40 25 17 0 1 90 50 "
            "A40 25 17 0 1 10 50 Z"
        )
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(folder, path_data)
            candidate = Path(folder) / "candidate.svg"
            with patch("curve_refit_stage.optimize_compound_contours") as optimizer:
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=3)
                frontier = build_svg_curve_refit_path_frontier(
                    source, "shape")

            optimizer.assert_not_called()
            self.assertEqual(report["status"], "no_change")
            self.assertEqual(report["optimizer_evaluated_path_count"], 0)
            self.assertEqual(
                report["skipped"]["unsupported_or_degenerate_geometry"], 1)
            self.assertEqual(frontier["status"], "ineligible")
            self.assertFalse(frontier["eligibility"]["arc_sampling_policy"])

    def test_arc_and_non_arc_commands_in_one_loop_fail_closed(self):
        path_data = (
            "M8 8 C25 3 70 3 88 8 A40 35 0 0 1 88 88 "
            "C65 94 30 94 8 88 L8 8 Z "
            "M38 50 A12 8 0 0 1 62 50 A12 8 0 0 1 38 50 Z"
        )
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(folder, path_data)
            candidate = Path(folder) / "candidate.svg"
            with patch("curve_refit_stage.optimize_compound_contours") as optimizer:
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=3)

            optimizer.assert_not_called()
            self.assertEqual(report["optimizer_evaluated_path_count"], 0)
            self.assertEqual(
                report["skipped"]["unsupported_or_degenerate_geometry"], 1)

    def test_mixed_arc_compound_is_optimizer_evaluated_with_evidence(self):
        path_data = (
            "M8 8 C18 5 30 5 42 8 C54 11 66 11 78 8 "
            "C88 16 91 29 88 42 C85 55 86 70 78 84 "
            "C64 91 48 90 34 88 C20 86 10 78 7 65 "
            "C4 51 5 35 8 8 Z "
            "M38 50 A12 8 27 0 1 62 50 A12 8 27 0 1 38 50 Z"
        )
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(folder, path_data)
            candidate = Path(folder) / "candidate.svg"
            fake = self._identity_optimizer_result(anchors=9, loops=2)
            fake["path"] = path_data
            with patch("curve_refit_stage.optimize_compound_contours",
                       return_value=fake) as optimizer:
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=9,
                    minimum_reduction_ratio=0.0)

            optimizer.assert_called_once()
            contours = optimizer.call_args.args[0]
            self.assertEqual(len(contours), 2)
            self.assertGreater(len(contours[1]), 20)
            self.assertEqual(report["eligible_path_count"], 1)
            self.assertEqual(report["optimizer_evaluated_path_count"], 1)
            self.assertEqual(
                report["stable_id_normalization"]
                ["optimizer_evaluated_path_count"], 1)
            self.assertTrue(
                report["stable_id_normalization"]
                ["all_optimizer_evaluations_authenticated"])
            self.assertTrue(
                report["evaluation_evidence_integrity"]
                ["all_optimizer_results_accounted"])
            self.assertFalse(report["uses_colour_or_pixel_similarity"])

            frontier_result = {
                "anchors_after": 9,
                "designer_anchor_count": 9,
                "selected_candidate_id": "source_identity",
                "path": path_data,
                "actual_p95_error_percent": 0.0,
                "actual_max_error_percent": 0.0,
                "safe_refit_selected": False,
                "lexicographic_objective": [],
                "refinement_frontier": {"candidates": []},
            }
            with patch("curve_refit_stage.optimize_compound_contours",
                       return_value=frontier_result) as frontier_optimizer:
                frontier = build_svg_curve_refit_path_frontier(
                    source, "shape")
            frontier_optimizer.assert_called_once()
            self.assertEqual(frontier["status"], "no_candidate")
            self.assertTrue(frontier["eligibility"]["arc_sampling_policy"])
            self.assertTrue(frontier["eligibility"]["supported_geometry"])

    def test_gradient_object_is_untouched_while_solid_path_is_refit(self):
        with tempfile.TemporaryDirectory() as folder:
            dense = (
                "M10 10 C12 10 14 10 16 10 C18 10 20 10 22 10 "
                "C24 10 26 10 28 10 C30 10 32 10 34 10 "
                "L34 34 C30 34 26 34 22 34 C18 34 14 34 10 34 Z"
            )
            source = Path(folder) / "source.svg"
            source.write_text(
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 100 100"><defs><linearGradient id="field">'
                '<stop offset="0" stop-color="#9adbea"/><stop offset="1" '
                'stop-color="#083c24"/></linearGradient></defs>'
                '<path id="gradient" fill="url(#field)" '
                'data-avc-gradient-object="mountain" d="' + dense + '"/>'
                '<path id="solid" fill="#126b45" d="' + dense + '"/>'
                '</svg>', encoding="utf-8")
            candidate = Path(folder) / "candidate.svg"
            report = propose_svg_curve_refit(
                source, candidate, tolerance=0.4, sample_step=1.5,
                minimum_nodes=6, minimum_reduction_ratio=0.1)

            self.assertEqual(report["status"], "proposed")
            paths = {item.get("id"): item
                     for item in self._elements(candidate, "path")}
            self.assertEqual(paths["gradient"].get("d"), dense)
            self.assertIsNone(
                paths["gradient"].get("data-avc-curve-refit"))
            self.assertNotEqual(paths["solid"].get("d"), dense)
            self.assertEqual(
                paths["solid"].get("data-avc-curve-refit"),
                "geometry-budgeted")
            reason = (
                "gradient_object_requires_source_ownership_revalidation")
            self.assertEqual(report["skipped"][reason], 1)
            self.assertEqual(
                report["protected_gradient_objects"]["skipped_path_count"],
                1)
            self.assertFalse(
                report["protected_gradient_objects"]
                ["ownership_mask_revalidation_performed"])

    def test_noisy_low_resolution_circle_becomes_true_native_circle(self):
        with tempfile.TemporaryDirectory() as folder:
            theta = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
            points = np.column_stack((
                np.round(50.0 + 35.0 * np.cos(theta)),
                np.round(50.0 + 35.0 * np.sin(theta)),
            ))
            direction = points[9] - np.array([50.0, 50.0])
            points[9] += 0.7 * direction / np.linalg.norm(direction)
            source = Path(folder) / "source.svg"
            source.write_text(
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 100 100"><path id="shape" fill="#126b45" '
                'style="opacity:.8" clip-path="url(#keep)" data-owner="tea" d="'
                + self._path_from_points(points) + '"/></svg>', encoding="utf-8")
            candidate = Path(folder) / "candidate.svg"
            report = propose_svg_curve_refit(
                source, candidate, error_budget_percent=1.0,
                sample_step=0.5, minimum_nodes=10,
                minimum_reduction_ratio=0.05)
            self.assertEqual(report["status"], "proposed")
            circles = self._elements(candidate, "circle")
            self.assertEqual(len(circles), 1)
            circle = circles[0]
            self.assertEqual(circle.get("id"), "shape")
            self.assertEqual(circle.get("fill"), "#126b45")
            self.assertEqual(circle.get("style"), "opacity:.8")
            self.assertEqual(circle.get("clip-path"), "url(#keep)")
            self.assertEqual(circle.get("data-owner"), "tea")
            self.assertEqual(circle.get("data-avc-designer-anchors"), "4")
            self.assertIsNone(circle.get("d"))
            detail = report["details"][0]
            self.assertEqual(detail["primitive"]["emitted_element"], "circle")
            self.assertEqual(detail["final_element"], "circle")
            self.assertEqual(detail["final_drawable_id"], "shape")
            self.assertEqual(
                set(detail["native_parameters"]), {"cx", "cy", "r"})
            canonical = json.dumps({
                "element": "circle",
                "id": "shape",
                "parameters": detail["native_parameters"],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self.assertEqual(
                detail["geometry_sha256"],
                hashlib.sha256(canonical.encode("utf-8")).hexdigest())
            self.assertEqual(
                detail["geometry_digest_scope"],
                "canonical_json_element_id_and_svg_geometry_attributes")
            self.assertLessEqual(detail["actual_p95_error_percent"], 1.0)
            self.assertLessEqual(detail["actual_max_error_percent"], 3.0)
            self.assertLessEqual(detail["over_budget_share"], 0.05)
            self.assertIn("salient_corner_max_percent", detail)
            self.assertFalse(report["uses_colour_or_pixel_similarity"])
            self.assertIn("not a claim of a globally analytic minimum",
                          report["optimality_note"])

    def test_rotated_ellipse_becomes_native_ellipse_with_legal_transform(self):
        with tempfile.TemporaryDirectory() as folder:
            theta = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
            rotation = math.radians(27.0)
            x = 36.0 * np.cos(theta)
            y = 19.0 * np.sin(theta)
            points = np.column_stack((
                50.0 + math.cos(rotation) * x - math.sin(rotation) * y,
                50.0 + math.sin(rotation) * x + math.cos(rotation) * y,
            ))
            source = self._write(folder, self._path_from_points(points))
            candidate = Path(folder) / "candidate.svg"
            report = propose_svg_curve_refit(
                source, candidate, error_budget_percent=0.30,
                sample_step=0.5, minimum_nodes=10,
                minimum_reduction_ratio=0.05)
            self.assertEqual(report["status"], "proposed")
            ellipses = self._elements(candidate, "ellipse")
            self.assertEqual(len(ellipses), 1)
            ellipse = ellipses[0]
            self.assertEqual(ellipse.get("id"), "shape")
            self.assertTrue(ellipse.get("transform", "").startswith("rotate("))
            self.assertEqual(ellipse.get("data-avc-designer-anchors"), "4")
            detail = report["details"][0]
            self.assertEqual(detail["final_element"], "ellipse")
            self.assertEqual(
                detail["native_parameters"]["transform"],
                ellipse.get("transform"))
            self.assertEqual(len(detail["geometry_sha256"]), 64)

    def test_irregular_shape_is_not_forced_to_circle_when_over_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            theta = np.linspace(0.0, 2.0 * math.pi, 100, endpoint=False)
            radius = 32.0 * (1.0 + 0.24 * np.cos(3.0 * theta))
            points = np.column_stack((
                50.0 + radius * np.cos(theta),
                50.0 + radius * np.sin(theta),
            ))
            source = self._write(folder, self._path_from_points(points))
            candidate = Path(folder) / "candidate.svg"
            report = propose_svg_curve_refit(
                source, candidate, error_budget_percent=0.25,
                sample_step=0.5, minimum_nodes=10,
                minimum_reduction_ratio=0.0)
            self.assertEqual(self._elements(candidate, "circle"), [])
            if report["details"]:
                self.assertNotIn(
                    "circle", report["details"][0]["primitive"]["native"])

    def test_replacement_requires_designer_anchor_reduction(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(folder, "M10 10 L90 10 L50 90 Z")
            candidate = Path(folder) / "candidate.svg"
            fake = {
                "anchors_after": 2,
                "designer_anchor_count": 4,
                "path": "M90 50 A40 40 0 1 1 10 50 A40 40 0 1 1 90 50 Z",
                "fit": {"loop_count": 1, "contours": [{
                    "primitive": "circle",
                    "native_primitive": {"element": "circle", "cx": 50.0,
                                         "cy": 50.0, "r": 40.0},
                }]},
                "primitive_complexity": {
                    "category": "native_primitive",
                    "native_primitives": ["circle"],
                },
                "actual_p95_error_percent": 0.0,
                "actual_max_error_percent": 0.0,
                "over_budget_share": 0.0,
                "salient_corner_max_percent": 0.0,
                "selected_candidate_id": "fake",
                "lexicographic_objective": [],
                "loop_count": 1,
            }
            with patch("curve_refit_stage.optimize_compound_contours",
                       return_value=fake):
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=3,
                    minimum_reduction_ratio=0.0)
            self.assertEqual(report["status"], "no_change")
            self.assertIn("insufficient_reduction", report["skipped"])
            self.assertEqual(report["evaluated_but_retained"], [])
            self.assertEqual(len(report["uncertified_evaluations"]), 1)
            self.assertEqual(
                report["uncertified_evaluations"][0]["stage_reason"],
                "designer_anchor_count_exceeds_source")
            self.assertEqual(len(self._elements(candidate, "path")), 1)
            self.assertEqual(self._elements(candidate, "circle"), [])

    def test_identity_minimum_is_reported_per_id_without_mutating_path(self):
        path_data = "M10 10 L90 10 L50 90 Z"
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(folder, path_data)
            candidate = Path(folder) / "candidate.svg"
            fake = self._identity_optimizer_result()
            fake.update({
                "status": "selected_within_geometry_budget",
                "identity_rollback_selected": False,
                "safe_refit_selected": True,
                "selection_outcome": "certified_geometry_refit",
                "anchors_before": 24,
                "anchors_after": 4,
                "designer_anchor_count": 4,
                "segment_count_after": 4,
                "selected_candidate_id": "curve_refit_03",
                "selected_source": "curve_refit",
                "fit": {"source_identity": False},
            })
            with patch("curve_refit_stage.optimize_compound_contours",
                       return_value=fake):
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=3,
                    minimum_reduction_ratio=0.0)

            self.assertEqual(
                report["schema"],
                "ai-vector-cleanroom.curve-refit-proposal/v3")
            self.assertEqual(report["status"], "no_change")
            self.assertEqual(report["optimizer_evaluated_path_count"], 1)
            self.assertEqual(report["retained_identity_path_count"], 1)
            self.assertEqual(report["details"], [])
            self.assertEqual(report["uncertified_evaluations"], [])
            retained = report["evaluated_but_retained"]
            self.assertEqual(len(retained), 1)
            evidence = retained[0]
            self.assertEqual(evidence["id"], "shape")
            self.assertEqual(evidence["outcome"],
                             "retained_identity_minimum")
            self.assertTrue(evidence["economy_certified"])
            self.assertTrue(evidence["geometry_unchanged"])
            self.assertEqual(evidence["anchors_before"], 3)
            self.assertEqual(evidence["anchors_after"], 3)
            self.assertEqual(evidence["designer_anchors_before"], 3)
            self.assertEqual(evidence["designer_anchors_after"], 3)
            self.assertEqual(evidence["selected_candidate_id"],
                             "curve_refit_03")
            self.assertFalse(evidence["identity_rollback_selected"])
            self.assertEqual(evidence["optimizer_input_anchor_count"], 24)
            self.assertEqual(evidence["optimizer_selected_anchor_count"], 4)
            self.assertEqual(
                evidence["optimizer_selected_designer_anchor_count"], 4)
            self.assertEqual(
                evidence["optimizer_selected_segment_count"], 4)
            self.assertEqual(
                evidence["source_baseline_economy"], {
                    "designer_anchor_count": 3,
                    "anchor_count": 3,
                    "fragment_count": 1,
                    "segment_count": 3,
                })
            self.assertEqual(
                evidence["optimizer_selected_economy"]
                ["designer_anchor_count"], 4)
            self.assertEqual(evidence["loops"], 1)
            self.assertEqual(
                evidence["path_data_sha256"],
                hashlib.sha256(path_data.encode("utf-8")).hexdigest())
            final_path = self._elements(candidate, "path")[0]
            self.assertEqual(final_path.get("d"), path_data)
            self.assertIsNone(final_path.get("data-avc-curve-refit"))

            integrity = report["evaluation_evidence_integrity"]
            self.assertTrue(integrity["all_optimizer_results_accounted"])
            self.assertTrue(integrity["evidence_ids_unique"])
            self.assertTrue(
                integrity["committed_and_retained_ids_disjoint"])
            self.assertEqual(integrity["committed_detail_count"], 0)
            self.assertEqual(
                integrity["retained_identity_detail_count"], 1)
            self.assertEqual(integrity["uncertified_evaluation_count"], 0)

    def test_tampered_or_incomplete_identity_evidence_fails_closed(self):
        cases = {
            "selection-state": "optimizer_selection_state_inconsistent",
            "objective": "lexicographic_objective_incomplete",
            "p95": "p95_error_exceeds_budget",
            "candidate-count": "candidate_count_missing_or_invalid",
        }
        for label, expected_failure in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as folder:
                source = self._write(folder, "M10 10 L90 10 L50 90 Z")
                candidate = Path(folder) / "candidate.svg"
                fake = copy.deepcopy(self._identity_optimizer_result())
                if label == "selection-state":
                    fake["safe_refit_selected"] = True
                elif label == "objective":
                    fake["lexicographic_objective"].remove(
                        "minimize_designer_anchor_count")
                elif label == "p95":
                    fake["actual_p95_error_percent"] = 0.251
                elif label == "candidate-count":
                    fake["candidate_count"] = None
                with patch("curve_refit_stage.optimize_compound_contours",
                           return_value=fake):
                    report = propose_svg_curve_refit(
                        source, candidate, minimum_nodes=3,
                        minimum_reduction_ratio=0.0)

                self.assertEqual(report["evaluated_but_retained"], [])
                self.assertEqual(report["retained_identity_path_count"], 0)
                self.assertEqual(len(report["uncertified_evaluations"]), 1)
                self.assertIn(
                    expected_failure,
                    report["uncertified_evaluations"][0]
                    ["integrity_failures"])
                self.assertTrue(
                    report["evaluation_evidence_integrity"]
                    ["all_optimizer_results_accounted"])

    def test_open_path_is_not_reported_as_optimizer_evaluated(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(folder, "M10 10 L90 10 L50 90")
            candidate = Path(folder) / "candidate.svg"
            with patch("curve_refit_stage.optimize_compound_contours") as optimizer:
                report = propose_svg_curve_refit(
                    source, candidate, minimum_nodes=3,
                    minimum_reduction_ratio=0.0)

            optimizer.assert_not_called()
            self.assertEqual(report["eligible_path_count"], 0)
            self.assertEqual(report["optimizer_evaluated_path_count"], 0)
            self.assertEqual(report["retained_identity_path_count"], 0)
            self.assertEqual(report["evaluated_but_retained"], [])
            self.assertEqual(report["uncertified_evaluations"], [])
            self.assertEqual(report["skipped"]["open_path"], 1)
            integrity = report["evaluation_evidence_integrity"]
            self.assertEqual(integrity["accounted_evaluation_count"], 0)
            self.assertTrue(integrity["all_optimizer_results_accounted"])

    def test_is_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            source = self._write(
                folder,
                "M10 10 C15 8 20 8 25 10 C30 12 35 12 40 10 "
                "L40 40 C30 42 20 42 10 40 Z")
            first = Path(folder) / "first.svg"
            second = Path(folder) / "second.svg"
            one = propose_svg_curve_refit(
                source, first, minimum_nodes=5, minimum_reduction_ratio=0.0)
            two = propose_svg_curve_refit(
                source, second, minimum_nodes=5, minimum_reduction_ratio=0.0)
            self.assertEqual(one, {**two, "candidate": first.name})
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
