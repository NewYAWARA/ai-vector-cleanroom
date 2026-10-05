# -*- coding: utf-8 -*-
"""Regressions for geometry-budgeted, economy-first anchor minimisation."""

import copy
import json
import math
import unittest
from unittest import mock

import numpy as np

from geometry_error_optimizer import (
    _identity_path,
    _is_strictly_more_conservative_loop_option,
    _measure_refinement_candidate,
    _nearest_distances,
    _primitive_complexity,
    _selection_key,
    _salient_corner_indices,
    fit_compound_contours,
    measure_fit_error,
    optimize_compound_contours,
    optimize_curve,
)


def _reference_nearest_distances(points, polyline, *, point_chunk=256,
                                 segment_chunk=256):
    """Frozen pre-BVH all-segment implementation."""
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    starts, ends = polyline[:-1], polyline[1:]
    if not len(starts):
        raise ValueError("distance polyline must contain a segment")
    output = np.empty(len(points), dtype=np.float64)
    for point_start in range(0, len(points), point_chunk):
        point_block = points[point_start:point_start + point_chunk]
        best = np.full(len(point_block), np.inf, dtype=np.float64)
        for segment_start in range(0, len(starts), segment_chunk):
            a = starts[segment_start:segment_start + segment_chunk]
            b = ends[segment_start:segment_start + segment_chunk]
            delta = b - a
            denominator = np.sum(delta * delta, axis=1)
            offset = point_block[:, None, :] - a[None, :, :]
            numerator = np.sum(offset * delta[None, :, :], axis=2)
            parameter = np.divide(
                numerator, denominator[None, :],
                out=np.zeros_like(numerator),
                where=denominator[None, :] > 1.0e-12)
            parameter = np.clip(parameter, 0.0, 1.0)
            projected = (
                a[None, :, :]
                + parameter[:, :, None] * delta[None, :, :])
            distances2 = np.sum(
                (point_block[:, None, :] - projected) ** 2, axis=2)
            best = np.minimum(best, distances2.min(axis=1))
        output[point_start:point_start + len(point_block)] = np.sqrt(best)
    return output


FAST = {
    "sampling_step_percent": 0.10,
    "max_samples_per_segment": 1024,
}
GRID = (0.01, 0.03, 0.06, 0.10, 0.16, 0.25, 0.40, 0.63, 1.0, 1.6, 2.5)


class ArtificialSeamCandidateTests(unittest.TestCase):
    def test_rejected_hole_primitive_does_not_discard_safe_outer_frontier(self):
        from geometry_error_optimizer import _error_contract_evidence

        vertices = np.asarray([[0., 0.], [100., 0.], [100., 100.]])
        outer = np.concatenate([
            a + (b-a)*np.arange(100)[:, None]/100
            for a, b in zip(vertices, np.roll(vertices, -1, axis=0))])
        # Its diamond is fully inside the triangle; its attempted circle
        # crosses the diagonal. The outer triangle itself is exact at 3 nodes.
        hole = np.asarray([[50.5,49.4], [50.,49.9], [49.5,49.4], [50.,48.9]])
        safe_default = optimize_compound_contours([outer, hole], error_budget_percent=.25)
        self.assertTrue(safe_default['identity_rollback_selected'])
        self.assertEqual(safe_default['fit']['anchor_count'], 304)
        result = optimize_compound_contours(
            [outer, hole], error_budget_percent=.25,
            include_independently_valid_loop_candidates=True)
        self.assertFalse(result['identity_rollback_selected'])
        self.assertEqual(result['fit']['contours'][0]['anchor_count'], 3)
        self.assertEqual(result['fit']['contours'][1]['anchor_count'], 4)
        self.assertEqual(result['fit']['anchor_count'], 7)
        measured = measure_fit_error([outer, hole], result['fit'], error_budget_percent=.25)
        self.assertTrue(_error_contract_evidence(measured, .25)['within_budget'])
        self.assertTrue(measured['compound_relationships']['preserved'])
        self.assertTrue(any('topology_not_preserved' in row.get('rejection_reasons', [])
                            for row in result['candidates']))

    def test_tiny_hole_error_is_not_diluted_by_large_parent_bbox(self):
        from geometry_error_optimizer import _error_contract_evidence

        outer = np.asarray([[0., 0.], [10000., 0.], [10000., 10000.], [0., 10000.]])
        hole = np.asarray([[5000., 5000.], [5010., 5000.], [5010., 5010.], [5000., 5010.]])
        fitted = fit_compound_contours([outer, hole + [.6, 0]], tolerance=.01,
                                       allow_primitives=False)
        measured = measure_fit_error([outer, hole], fitted, error_budget_percent=.25)
        self.assertLess(measured["max_percent"], .25)
        self.assertGreater(measured["loops"][1]["local_max_percent"], .75)
        contract = _error_contract_evidence(measured, .25)
        self.assertFalse(contract["each_loop_local_scale_within_budget"])
        self.assertFalse(contract["within_budget"])

    def test_hole_crossing_parent_fails_even_inside_local_distance_budget(self):
        from geometry_error_optimizer import _error_contract_evidence

        outer = np.asarray([[0., 0.], [10000., 0.], [10000., 10000.], [0., 10000.]])
        hole = np.asarray([[.01, 5000.], [100.01, 5000.], [100.01, 5100.], [.01, 5100.]])
        fitted = fit_compound_contours([outer, hole - [.05, 0]], tolerance=.001,
                                       allow_primitives=False)
        measured = measure_fit_error([outer, hole], fitted, error_budget_percent=.25)
        self.assertLess(measured["loops"][1]["local_max_percent"], .25)
        self.assertFalse(measured["compound_relationships"]["preserved"])
        self.assertEqual(measured["compound_relationships"]["source"]["pair_relations"],
                         [[0, 1, "second_inside_first"]])
        self.assertEqual(measured["compound_relationships"]["candidate"]["pair_relations"],
                         [[0, 1, "crossing"]])
        self.assertFalse(_error_contract_evidence(measured, .25)["within_budget"])

    @staticmethod
    def smooth_outline():
        theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        radius = 50 + 2 * np.cos(4 * theta + .3)
        return np.column_stack((70 + radius * np.cos(theta),
                                70 + .8 * radius * np.sin(theta)))

    def test_artificial_seam_removal_reduces_anchors_without_replacing_old_frontier(self):
        points = self.smooth_outline()
        baseline = optimize_curve(points, closed=True, merge_artificial_seams=False)
        improved = optimize_curve(points, closed=True)
        self.assertEqual(baseline["anchors_after"], 8)
        self.assertEqual(improved["anchors_after"], 6)
        self.assertTrue(improved["selected_candidate_id"].endswith("_seam"))
        self.assertLess(improved["actual_p95_error_percent"], .25)
        self.assertLess(improved["actual_max_error_percent"], .75)
        old_rows = {row["candidate_id"]: row for row in baseline["candidates"]}
        new_rows = {row["candidate_id"]: row for row in improved["candidates"]}
        for identifier, row in old_rows.items():
            self.assertEqual(row, new_rows[identifier])

    def test_failed_seam_geometry_cannot_displace_safe_baseline(self):
        points = self.smooth_outline()
        baseline = optimize_curve(points, closed=True, merge_artificial_seams=False)
        original_measure = measure_fit_error

        def measured(source, fitted, **kwargs):
            result = original_measure(source, fitted, **kwargs)
            if fitted.get("parameters", {}).get("merge_artificial_seams"):
                result.update(p95_percent=20., max_percent=20., over_budget_share=1.,
                              salient_corner_max_percent=20.)
            return result

        with mock.patch("geometry_error_optimizer.measure_fit_error", side_effect=measured):
            result = optimize_curve(points, closed=True)
        self.assertEqual(result["path"], baseline["path"])
        self.assertEqual(result["selected_candidate_id"], baseline["selected_candidate_id"])
        self.assertTrue(any(row.get("artificial_seam_merge_candidate")
                            and not row["eligible"] for row in result["candidates"]))

    def test_real_corners_are_never_offered_as_removable_seams(self):
        vertices = np.asarray([[0., 0.], [100., 0.], [100., 80.], [0., 80.]])
        points = np.vstack([np.linspace(a, b, 40, endpoint=False)
                            for a, b in zip(vertices, np.roll(vertices, -1, axis=0))])
        result = optimize_curve(points, closed=True)
        self.assertEqual(result["anchors_after"], 4)
        self.assertFalse(any(row.get("artificial_seam_merge_candidate")
                             for row in result["candidates"]))


def _circle(radius=50.0, count=96, centre=(0.0, 0.0)):
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.column_stack((
        centre[0] + radius * np.cos(theta),
        centre[1] + radius * np.sin(theta),
    ))


def _s_curve(scale=1.0):
    x = np.linspace(0.0, 100.0, 121)
    y = 8.0 * np.sin(x / 13.0) + 2.0 * np.sin(x / 4.5)
    return scale * np.column_stack((x, y))


def _mixed_scale_compound(count=96, noise=0.05):
    """Large almost-ellipse plus a tiny square hole on one global scale."""
    theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    radius = 80.0 + noise * np.sin(17.0 * theta)
    outer = np.column_stack((
        100.0 + radius * np.cos(theta),
        100.0 + 0.7 * radius * np.sin(theta),
    ))
    hole = np.array([
        [96.0, 96.0], [104.0, 96.0],
        [104.0, 104.0], [96.0, 104.0],
    ])
    return outer, hole


def _zero_compound_error(loop_count):
    loop = {
        "max_percent": 0.0,
        "p95_percent": 0.0,
        "over_budget_share": 0.0,
        "salient_corner_max_percent": 0.0,
    }
    return {
        "method": "test_zero_error",
        "approximate": False,
        "normalization_basis": "source_bbox_diagonal",
        "normalization_scale": 1.0,
        "source_bbox": [0.0, 0.0, 1.0, 1.0],
        "max_absolute": 0.0,
        "p95_absolute": 0.0,
        "max_percent": 0.0,
        "p95_percent": 0.0,
        "over_budget_share": 0.0,
        "over_budget_sample_count": 0,
        "salient_corner_count": 0,
        "salient_corner_max_absolute": 0.0,
        "salient_corner_max_percent": 0.0,
        "source_to_fit": {
            "max_absolute": 0.0, "p95_absolute": 0.0, "sample_count": 1,
        },
        "fit_to_source": {
            "max_absolute": 0.0, "p95_absolute": 0.0, "sample_count": 1,
        },
        "sampling": None,
        "loops": [dict(loop) for _ in range(loop_count)],
    }


FIVE_LOOP_SOURCE_ANCHORS = (46, 18, 127, 70, 58)
FIVE_LOOP_BASE_ANCHORS = (14, 7, 32, 21, 18)
FIVE_LOOP_REFINED_ANCHORS = (15, 8, 33, 22, 19)


def _five_loop_contours():
    return [
        _circle(
            radius=10.0 + index,
            count=count,
            centre=(40.0 * index, 20.0 * index))
        for index, count in enumerate(FIVE_LOOP_SOURCE_ANCHORS)
    ]


def _mock_compound_fit(anchor_counts, generation):
    contours = []
    for index, (source_count, anchor_count) in enumerate(zip(
            FIVE_LOOP_SOURCE_ANCHORS, anchor_counts)):
        x = 1000 * generation + 10 * index
        path = (
            f"M{x} 0 C{x + 1} 0 {x + 1} 1 {x} 1 "
            f"C{x - 1} 1 {x - 1} 0 {x} 0 Z")
        contours.append({
            "path": path,
            "closed": True,
            "segments": [
                {"type": "cubic"} for _ in range(anchor_count)
            ],
            "segment_count": anchor_count,
            "anchor_count": anchor_count,
            "input_point_count": source_count,
        })
    return {
        "path": " ".join(item["path"] for item in contours),
        "fill_rule": "evenodd",
        "loop_count": len(contours),
        "contours": contours,
        "segment_count": sum(anchor_counts),
        "anchor_count": sum(anchor_counts),
        "input_point_count": sum(FIVE_LOOP_SOURCE_ANCHORS),
    }


def _five_loop_fitter():
    fits = (
        _mock_compound_fit(FIVE_LOOP_REFINED_ANCHORS, 0),
        _mock_compound_fit(FIVE_LOOP_BASE_ANCHORS, 1),
    )
    calls = {"count": 0}

    def fit(*_args, **_kwargs):
        result = fits[calls["count"] % len(fits)]
        calls["count"] += 1
        return copy.deepcopy(result)

    return fit


def _loop_options_from_fit(compound_fit, tolerance_percent=0.20):
    options = []
    for loop_index, fit in enumerate(compound_fit["contours"]):
        primitive = _primitive_complexity(fit)
        options.append({
            "candidate_id": f"loop_{loop_index:02d}_fixture",
            "source": "curve_refit_per_loop_tolerance",
            "loop_index": loop_index,
            "tolerance_percent_of_bbox_diagonal": tolerance_percent,
            "tolerance_absolute": tolerance_percent,
            "fit": fit,
            "eligible": True,
            "within_error_budget": True,
            "anchors_after": fit["anchor_count"],
            "segment_count": fit["segment_count"],
            "fragment_count": 1,
            "primitive_complexity": primitive,
            "designer_anchor_count": primitive["designer_anchor_count"],
            "actual_max_error_percent": 0.0,
            "actual_p95_error_percent": 0.0,
            "over_budget_share": 0.0,
            "salient_corner_max_percent": 0.0,
            "error_contract": {
                "p95_within_budget": True,
                "max_within_three_times_budget": True,
                "over_budget_share_at_most_5_percent": True,
                "salient_corner_max_within_two_times_budget": True,
            },
            "rejection_reasons": [],
        })
    return options


class GeometryErrorOptimizerTests(unittest.TestCase):
    def test_bvh_nearest_distances_match_frozen_full_scan_exactly(self):
        rng = np.random.default_rng(20260719)
        random_walk = np.cumsum(
            rng.normal(size=(513, 2)), axis=0).astype(np.float64)
        repeated = np.vstack((
            random_walk[:170], random_walk[169], random_walk[170:]))
        crossing = np.asarray([
            [-1000.0, 0.0], [1000.0, 0.0],
            [0.0, -1000.0], [0.0, 1000.0],
            [-1000.0, -1000.0], [1000.0, 1000.0],
        ])
        point_sets = [
            rng.normal(size=(377, 2)),
            np.vstack((random_walk[::7], [[0.0, 0.0], [1.0e-12, 0.0]])),
        ]
        for polyline in (random_walk, repeated, crossing):
            for points in point_sets:
                for point_chunk, segment_chunk in ((17, 13), (64, 32),
                                                    (256, 256)):
                    with self.subTest(
                            segments=len(polyline) - 1,
                            points=len(points), point_chunk=point_chunk,
                            segment_chunk=segment_chunk):
                        expected = _reference_nearest_distances(
                            points, polyline, point_chunk=point_chunk,
                            segment_chunk=segment_chunk)
                        actual = _nearest_distances(
                            points, polyline, point_chunk=point_chunk,
                            segment_chunk=segment_chunk)
                        np.testing.assert_array_equal(actual, expected)

    def test_refinement_option_is_strictly_conservative_and_never_replaces_identity(self):
        base = _loop_options_from_fit(
            _mock_compound_fit(FIVE_LOOP_BASE_ANCHORS, 1),
            tolerance_percent=0.20)[0]
        refined = _loop_options_from_fit(
            _mock_compound_fit(FIVE_LOOP_REFINED_ANCHORS, 0),
            tolerance_percent=0.10)[0]
        self.assertTrue(
            _is_strictly_more_conservative_loop_option(refined, base))

        worse_error = copy.deepcopy(refined)
        worse_error["actual_p95_error_percent"] = 0.01
        self.assertFalse(
            _is_strictly_more_conservative_loop_option(worse_error, base))

        identity_base = copy.deepcopy(base)
        identity_base["source"] = "source_identity_per_loop_rollback"
        identity_base["fit"]["source_identity"] = True
        self.assertFalse(
            _is_strictly_more_conservative_loop_option(refined, identity_base))

    def test_optimise_fit_context_is_call_scoped_and_exact_true_gated(self):
        contours = _five_loop_contours()
        fit_fixture = _mock_compound_fit(FIVE_LOOP_BASE_ANCHORS, 1)
        absent = object()

        def capture(capability, *, optimise_calls=1, tolerances=(0.10,)):
            calls = []

            def fitter(*_args, **kwargs):
                calls.append(kwargs)
                return copy.deepcopy(fit_fixture)

            if capability is not absent:
                fitter._supports_exact_fit_context = capability
            with mock.patch(
                    "geometry_error_optimizer.fit_compound_contours",
                    new=fitter), mock.patch(
                        "geometry_error_optimizer.measure_fit_error",
                        return_value=_zero_compound_error(5)):
                for _ in range(optimise_calls):
                    optimize_compound_contours(
                        contours, error_budget_percent=0.25,
                        tolerance_percents=tolerances, **FAST)
            return calls

        bundled_capability = getattr(
            fit_compound_contours,
            "_supports_exact_fit_context", False)
        self.assertIs(bundled_capability, True)

        capable_calls = capture(
            bundled_capability, optimise_calls=2,
            tolerances=(0.10, 0.20))
        self.assertEqual(len(capable_calls), 4)
        contexts = [call["_fit_context"] for call in capable_calls]
        self.assertTrue(all(isinstance(item, dict) for item in contexts))
        self.assertIs(contexts[0], contexts[1])
        self.assertIs(contexts[2], contexts[3])
        self.assertIsNot(contexts[0], contexts[2])

        for label, capability in (
                ("false", False),
                ("truthy_non_bool", 1),
                ("absent", absent)):
            with self.subTest(capability=label):
                calls = capture(capability)
                self.assertEqual(len(calls), 1)
                self.assertNotIn("_fit_context", calls[0])

    def test_refinement_frontier_flag_off_is_byte_for_byte_result_compatible(self):
        contours = _five_loop_contours()
        fitter = _five_loop_fitter()
        with mock.patch(
                "geometry_error_optimizer.fit_compound_contours",
                side_effect=fitter), mock.patch(
                    "geometry_error_optimizer.measure_fit_error",
                    return_value=_zero_compound_error(5)), mock.patch(
                        "geometry_error_optimizer."
                        "_build_compound_refinement_frontier",
                        side_effect=AssertionError(
                            "flag-off must not build a frontier")):
            omitted = optimize_compound_contours(
                contours, error_budget_percent=0.25,
                tolerance_percents=(0.10, 0.20), **FAST)
            explicit_false = optimize_compound_contours(
                contours, error_budget_percent=0.25,
                tolerance_percents=(0.10, 0.20),
                include_refinement_frontier=False, **FAST)

        self.assertEqual(omitted, explicit_false)
        self.assertNotIn("refinement_frontier", omitted)
        self.assertEqual(
            json.dumps(omitted, ensure_ascii=False, sort_keys=True),
            json.dumps(explicit_false, ensure_ascii=False, sort_keys=True))

    def test_five_loop_frontier_changes_exactly_one_loop_deterministically(self):
        contours = _five_loop_contours()
        fitter = _five_loop_fitter()
        with mock.patch(
                "geometry_error_optimizer.fit_compound_contours",
                side_effect=fitter), mock.patch(
                    "geometry_error_optimizer.measure_fit_error",
                    return_value=_zero_compound_error(5)):
            first = optimize_compound_contours(
                contours, error_budget_percent=0.25,
                tolerance_percents=(0.10, 0.20),
                include_refinement_frontier=True, **FAST)
            second = optimize_compound_contours(
                contours, error_budget_percent=0.25,
                tolerance_percents=(0.10, 0.20),
                include_refinement_frontier=True, **FAST)

        frontier = first["refinement_frontier"]
        self.assertEqual(frontier, second["refinement_frontier"])
        self.assertFalse(frontier["uses_colour_or_pixel_similarity"])
        self.assertEqual(frontier["base_path"], first["path"])
        self.assertEqual(frontier["candidate_count"], 7)
        self.assertEqual(
            [row["candidate_id"] for row in frontier["candidates"]],
            [
                "compound_refinement_base",
                "compound_refinement_loop_00_next",
                "compound_refinement_loop_01_next",
                "compound_refinement_loop_02_next",
                "compound_refinement_loop_03_next",
                "compound_refinement_loop_04_next",
                "compound_refinement_source_identity",
            ])
        base = next(
            row for row in frontier["candidates"]
            if row["candidate_kind"] == "base_mixed")
        self.assertEqual(base["anchors_after"], 92)
        base_ids = [
            row["selected_candidate_id"]
            for row in base["per_loop_selection"]
        ]
        refinements = [
            row for row in frontier["candidates"]
            if row["candidate_kind"] == "single_loop_refinement"
        ]
        self.assertEqual(len(refinements), 5)
        for expected_loop, row in enumerate(refinements):
            self.assertEqual(row["changed_loop_index"], expected_loop)
            self.assertEqual(row["anchors_after"], 93)
            self.assertEqual(row["designer_anchor_count"], 93)
            self.assertEqual(row["anchor_cost"], {
                "anchors_added_from_base": 1,
                "designer_anchors_added_from_base": 1,
            })
            self.assertTrue(row["eligible"])
            self.assertTrue(row["topology"]["preserved"])
            self.assertTrue(all(row["error_contract"].values()))
            self.assertLess(
                row["replacement_tolerance_percent"], 0.20)
            candidate_ids = [
                item["selected_candidate_id"]
                for item in row["per_loop_selection"]
            ]
            changed = [
                index for index, (before, after) in enumerate(zip(
                    base_ids, candidate_ids)) if before != after
            ]
            self.assertEqual(changed, [expected_loop])
        identity = next(
            row for row in frontier["candidates"]
            if row["candidate_kind"] == "source_identity_control")
        self.assertTrue(identity["eligible"])
        self.assertTrue(identity["exact_source_identity"])
        self.assertEqual(identity["anchors_after"], 319)
        self.assertEqual(
            identity["path"], _identity_path(contours, [True] * 5))
        self.assertTrue(all(
            item["source_identity"]
            for item in identity["per_loop_selection"]))

    def test_refinement_candidate_failures_keep_evidence_and_fail_closed(self):
        contours = _five_loop_contours()
        options = _loop_options_from_fit(
            _mock_compound_fit(FIVE_LOOP_BASE_ANCHORS, 1))
        common = {
            "candidate_id": "fixture_refinement",
            "candidate_kind": "single_loop_refinement",
            "selected_options": options,
            "changed_loop_index": 2,
            "replacement_option": options[2],
            "contours": contours,
            "closed_flags": [True] * 5,
            "budget": 0.25,
            "scale": 100.0,
            "sampling_step_percent": FAST["sampling_step_percent"],
            "max_samples_per_segment": FAST["max_samples_per_segment"],
            "base_anchor_count": 91,
            "base_designer_anchor_count": 91,
        }

        with mock.patch(
                "geometry_error_optimizer._combine_compound_loop_fits",
                side_effect=ValueError("fixture combine failure")):
            combined = _measure_refinement_candidate(
                metric_cache={}, **common)
        self.assertFalse(combined["eligible"])
        self.assertFalse(combined["fit_succeeded"])
        self.assertEqual(combined["failure_stage"], "combine")
        self.assertEqual(
            combined["rejection_reasons"],
            ["compound_loop_combine_failed"])

        failed_topology = {
            "preserved": False,
            "loop_count_preserved": True,
            "closure_preserved": False,
            "evenodd_preserved": True,
            "nonempty_geometry": True,
        }
        with mock.patch(
                "geometry_error_optimizer._topology_evidence",
                return_value=failed_topology), mock.patch(
                    "geometry_error_optimizer.measure_fit_error",
                    return_value=_zero_compound_error(5)):
            topology = _measure_refinement_candidate(
                metric_cache={}, **common)
        self.assertFalse(topology["eligible"])
        self.assertTrue(topology["global_remeasurement_succeeded"])
        self.assertEqual(
            topology["rejection_reasons"], ["topology_not_preserved"])

        over_budget = _zero_compound_error(5)
        over_budget.update({
            "max_percent": 1.0,
            "p95_percent": 0.30,
            "over_budget_share": 0.06,
            "salient_corner_max_percent": 0.60,
        })
        with mock.patch(
                "geometry_error_optimizer.measure_fit_error",
                return_value=over_budget):
            contract = _measure_refinement_candidate(
                metric_cache={}, **common)
        self.assertFalse(contract["eligible"])
        self.assertFalse(contract["within_error_budget"])
        self.assertEqual(
            contract["rejection_reasons"],
            ["geometric_error_budget_exceeded"])
        self.assertFalse(all(contract["error_contract"].values()))

    def test_budget_relaxation_never_increases_anchor_count(self):
        points = _s_curve()
        tight = optimize_curve(
            points, error_budget_percent=0.08,
            tolerance_percents=GRID, **FAST)
        loose = optimize_curve(
            points, error_budget_percent=0.65,
            tolerance_percents=GRID, **FAST)
        self.assertLessEqual(loose["anchors_after"], tight["anchors_after"])
        self.assertLessEqual(tight["anchors_after"], len(points))
        self.assertLessEqual(loose["anchors_after"], len(points))
        self.assertLessEqual(
            tight["actual_p95_error_percent"], tight["error_budget_percent"])
        self.assertLessEqual(
            loose["actual_p95_error_percent"], loose["error_budget_percent"])

    def test_selected_candidate_is_lexicographic_minimum_of_eligible_set(self):
        result = optimize_curve(
            _s_curve(), error_budget_percent=0.35,
            tolerance_percents=GRID, **FAST)
        eligible = [row for row in result["candidates"] if row["eligible"]]

        def key(row):
            return (
                row["designer_anchor_count"],
                row["anchors_after"],
                row["fragment_count"],
                row["segment_count"],
                row["primitive_complexity"]["rank"],
                row["actual_max_error_percent"],
                row["actual_p95_error_percent"],
                row["candidate_id"],
            )

        expected = min(eligible, key=key)
        self.assertEqual(result["selected_candidate_id"],
                         expected["candidate_id"])
        self.assertEqual(result["anchors_after"], expected["anchors_after"])
        self.assertFalse(result["uses_colour_or_pixel_similarity"])
        objective = result["lexicographic_objective"]
        self.assertLess(objective.index("minimize_designer_anchor_count"),
                        objective.index(
                            "prefer_simpler_primitive_category_when_economy_equal"))
        self.assertLess(objective.index("minimize_segment_count"),
                        objective.index(
                            "prefer_simpler_primitive_category_when_economy_equal"))

    def test_dense_compound_outer_uses_economy_before_primitive_tiebreak(self):
        outer = _circle(radius=80.0, count=64, centre=(100.0, 100.0))
        hole = np.array([
            [96.0, 96.0], [104.0, 96.0],
            [104.0, 104.0], [96.0, 104.0],
        ])
        outer_fit = {
            "path": "M20 100 C20 20 180 20 180 100 C180 180 20 180 20 100 Z",
            "closed": True,
            "segments": [{"type": "cubic"} for _ in range(4)],
            "segment_count": 4,
            "anchor_count": 6,
            "input_point_count": len(outer),
        }
        # Deliberately uneconomic fit for the tiny hole: per-loop selection
        # should retain its four-point source identity while simplifying the
        # dense outer loop.
        hole_fit = {
            "path": "M96 96 C98 95 100 95 104 96 C105 98 105 100 104 104 "
                    "C102 105 100 105 96 104 C95 102 95 100 96 96 Z",
            "closed": True,
            "segments": [{"type": "cubic"} for _ in range(8)],
            "segment_count": 8,
            "anchor_count": 8,
            "input_point_count": len(hole),
        }
        compound_fit = {
            "path": f"{outer_fit['path']} {hole_fit['path']}",
            "fill_rule": "evenodd",
            "loop_count": 2,
            "contours": [outer_fit, hole_fit],
            "segment_count": 12,
            "anchor_count": 14,
            "input_point_count": len(outer) + len(hole),
        }

        with mock.patch(
                "geometry_error_optimizer.fit_compound_contours",
                return_value=compound_fit), mock.patch(
                    "geometry_error_optimizer.measure_fit_error",
                    return_value=_zero_compound_error(2)):
            result = optimize_compound_contours(
                [outer, hole], error_budget_percent=0.25,
                tolerance_percents=(0.10,), **FAST)

        self.assertEqual(result["selected_candidate_id"],
                         "curve_refit_mixed_loops")
        selected = next(
            row for row in result["candidates"]
            if row["candidate_id"] == result["selected_candidate_id"])
        choices = selected["per_loop_selection"]
        self.assertFalse(choices[0]["identity_rollback"])
        self.assertEqual(choices[0]["source_point_count"], len(outer))
        self.assertEqual(choices[0]["designer_anchor_count"], 6)
        self.assertEqual(choices[0]["primitive_rank"], 3)
        self.assertEqual(choices[0]["primitive_category"], "bezier_geometry")
        self.assertTrue(choices[1]["identity_rollback"])
        self.assertEqual(choices[1]["primitive_category"],
                         "source_polyline_rollback")

        frontier = result["largest_loop_frontier"]
        self.assertEqual(len(frontier), 2)
        self.assertTrue(all(
            row["source_point_count"] == len(outer) for row in frontier))
        identity = next(
            row for row in frontier if row["identity_rollback"])
        self.assertEqual(identity["primitive_rank"], 3)
        self.assertEqual(identity["primitive_category"],
                         "source_polyline_rollback")
        self.assertEqual(identity["primitive"], {
            "rank": 3, "category": "source_polyline_rollback",
        })
        self.assertEqual(identity["tolerance_percent"], 0.0)
        self.assertEqual(identity["segment_type_counts"],
                         {"line": len(outer)})

        summary = result["selected_mixed_identity_summary"]
        self.assertTrue(summary["is_mixed_loop_candidate"])
        self.assertEqual(summary["identity_loop_count"], 1)
        self.assertEqual(summary["identity_anchor_count"], len(hole))
        self.assertEqual(summary["largest_identity_loop"]["loop_index"], 1)
        self.assertEqual(
            summary["largest_identity_loop"]["source_point_count"],
            len(hole))

        def economy_key(row):
            return (
                row["designer_anchor_count"], row["anchors_after"],
                row["fragment_count"], row["segment_count"],
                row["primitive_complexity"]["rank"],
                row["actual_max_error_percent"],
                row["actual_p95_error_percent"], row["candidate_id"],
            )

        eligible = [row for row in result["candidates"] if row["eligible"]]
        self.assertEqual(selected["candidate_id"],
                         min(eligible, key=economy_key)["candidate_id"])
        self.assertTrue(selected["topology"]["preserved"])
        self.assertTrue(all(selected["error_contract"].values()))
        self.assertLessEqual(result["actual_p95_error_percent"],
                             result["error_budget_percent"])
        self.assertLessEqual(result["actual_max_error_percent"],
                             3.0 * result["error_budget_percent"])

    def test_dense_lines_are_not_analytic_but_native_ovals_remain_native(self):
        dense_polyline = {
            "segments": [{"type": "line"} for _ in range(12)],
            "segment_count": 12,
            "anchor_count": 12,
        }
        complexity = _primitive_complexity(dense_polyline)
        self.assertEqual(complexity["rank"], 3)
        self.assertEqual(complexity["category"],
                         "general_polyline_geometry")

        for primitive in ("circle", "ellipse"):
            with self.subTest(primitive=primitive):
                native = _primitive_complexity({
                    "primitive": primitive,
                    "segments": [{"type": "arc"} for _ in range(4)],
                    "segment_count": 4,
                    "anchor_count": 4,
                })
                self.assertEqual(native["rank"], 0)
                self.assertEqual(native["category"], "native_primitive")

    def test_primitive_rank_only_breaks_an_exact_economy_tie(self):
        def row(candidate_id, *, designer, anchors, segments, rank):
            return {
                "candidate_id": candidate_id,
                "designer_anchor_count": designer,
                "anchors_after": anchors,
                "fragment_count": 1,
                "segment_count": segments,
                "primitive_complexity": {"rank": rank},
                "actual_max_error_percent": 0.1,
                "actual_p95_error_percent": 0.05,
            }

        smaller_bezier = row(
            "smaller_bezier", designer=5, anchors=5, segments=4, rank=3)
        larger_native = row(
            "larger_native", designer=6, anchors=6, segments=4, rank=0)
        self.assertEqual(
            min([larger_native, smaller_bezier],
                key=_selection_key)["candidate_id"],
            "smaller_bezier")

        tied_bezier = row(
            "tied_bezier", designer=5, anchors=5, segments=4, rank=3)
        tied_native = row(
            "tied_native", designer=5, anchors=5, segments=4, rank=0)
        self.assertEqual(
            min([tied_bezier, tied_native],
                key=_selection_key)["candidate_id"],
            "tied_native")

    def test_gross_over_simplification_is_rejected_by_external_error_gate(self):
        zigzag = np.array([
            [0.0, 0.0], [10.0, 35.0], [20.0, -30.0], [30.0, 38.0],
            [40.0, -32.0], [50.0, 34.0], [60.0, 0.0],
        ])
        result = optimize_curve(
            zigzag, error_budget_percent=0.10,
            tolerance_percents=(100.0,), corner_angle=179.0, **FAST)
        attempted = result["candidates"][0]
        self.assertFalse(attempted["eligible"])
        self.assertIn("geometric_error_budget_exceeded",
                      attempted["rejection_reasons"])
        self.assertEqual(result["selected_candidate_id"], "source_identity")
        self.assertEqual(result["status"],
                         "identity_rollback_no_safe_reduction")
        self.assertTrue(result["identity_rollback_selected"])
        self.assertFalse(result["safe_refit_selected"])
        self.assertEqual(result["selection_outcome"],
                         "source_identity_rollback")
        self.assertEqual(result["anchors_after"], len(zigzag))
        identity_error = measure_fit_error(
            zigzag, result["fit"], closed=False,
            error_budget_percent=0.10, **FAST)
        self.assertEqual(identity_error["max_percent"], 0.0)

    def test_mixed_scale_loops_are_selected_then_globally_remeasured(self):
        outer, hole = _mixed_scale_compound()
        result = optimize_compound_contours(
            [outer, hole], error_budget_percent=0.25,
            tolerance_percents=(0.01, 0.10, 0.40), **FAST)
        self.assertEqual(result["selected_candidate_id"],
                         "curve_refit_mixed_loops")
        self.assertEqual(result["status"],
                         "selected_within_geometry_budget")
        self.assertFalse(result["identity_rollback_selected"])
        self.assertTrue(result["safe_refit_selected"])
        self.assertFalse(result["uses_colour_or_pixel_similarity"])
        self.assertLess(result["designer_anchor_count"],
                        result["anchors_before"])

        selected = next(
            row for row in result["candidates"]
            if row["candidate_id"] == result["selected_candidate_id"])
        self.assertTrue(selected["global_remeasurement_after_loop_merge"])
        self.assertTrue(selected["topology"]["preserved"])
        self.assertTrue(selected["designer_anchor_reduction_achieved"])
        self.assertEqual(result["fit"]["loop_count"], 2)
        self.assertEqual(result["fit"]["fill_rule"], "evenodd")
        self.assertTrue(all(item["closed"]
                            for item in result["fit"]["contours"]))

        choices = selected["per_loop_selection"]
        self.assertEqual(len(choices), 2)
        self.assertFalse(choices[0]["identity_rollback"])
        self.assertTrue(choices[1]["identity_rollback"])
        self.assertGreater(
            choices[0]["selected_tolerance_percent_of_bbox_diagonal"],
            choices[1]["selected_tolerance_percent_of_bbox_diagonal"])

        budget = result["error_budget_percent"]
        self.assertLessEqual(result["actual_p95_error_percent"], budget)
        self.assertLessEqual(result["actual_max_error_percent"], 3.0 * budget)
        self.assertLessEqual(result["over_budget_share"], 0.05)
        self.assertLessEqual(result["salient_corner_max_percent"],
                             2.0 * budget)
        self.assertTrue(all(selected["error_contract"].values()))

    def test_mixed_scale_budget_relaxation_is_monotone(self):
        outer, hole = _mixed_scale_compound(noise=0.20)
        grid = (0.01, 0.03, 0.10, 0.25, 0.40, 0.80)
        tight = optimize_compound_contours(
            [outer, hole], error_budget_percent=0.12,
            tolerance_percents=grid, **FAST)
        loose = optimize_compound_contours(
            [outer, hole], error_budget_percent=0.35,
            tolerance_percents=grid, **FAST)
        self.assertLessEqual(loose["designer_anchor_count"],
                             tight["designer_anchor_count"])
        self.assertLessEqual(loose["anchors_after"], tight["anchors_after"])
        self.assertLessEqual(tight["actual_p95_error_percent"],
                             tight["error_budget_percent"])
        self.assertLessEqual(loose["actual_p95_error_percent"],
                             loose["error_budget_percent"])
        self.assertEqual(tight["fit"]["loop_count"], 2)
        self.assertEqual(loose["fit"]["loop_count"], 2)
        self.assertEqual(tight["fit"]["fill_rule"], "evenodd")
        self.assertEqual(loose["fit"]["fill_rule"], "evenodd")

    def test_normalised_error_and_anchor_choice_are_scale_invariant(self):
        small = optimize_curve(
            _s_curve(), error_budget_percent=0.30,
            tolerance_percents=GRID, **FAST)
        large = optimize_curve(
            _s_curve(17.0), error_budget_percent=0.30,
            tolerance_percents=GRID, **FAST)
        self.assertEqual(small["anchors_after"], large["anchors_after"])
        self.assertEqual(small["segment_count_after"],
                         large["segment_count_after"])
        self.assertAlmostEqual(small["actual_max_error_percent"],
                               large["actual_max_error_percent"], places=5)
        self.assertAlmostEqual(small["actual_p95_error_percent"],
                               large["actual_p95_error_percent"], places=5)

    def test_open_closed_and_compound_hole_topology(self):
        open_line = np.column_stack((
            np.linspace(0.0, 100.0, 101),
            0.08 * np.sin(np.linspace(0.0, 12.0, 101)),
        ))
        line = optimize_curve(
            open_line, closed=False, error_budget_percent=0.15,
            tolerance_percents=GRID, **FAST)
        self.assertFalse(line["closed"])
        self.assertEqual(line["fit"]["primitive"], "line")
        self.assertEqual(line["anchors_after"], 2)

        outer = _circle(50.0, centre=(60.0, 60.0))
        hole = _circle(20.0, centre=(60.0, 60.0))[::-1]
        compound = optimize_compound_contours(
            [outer, hole], error_budget_percent=0.08,
            tolerance_percents=(0.01, 0.03, 0.06, 0.10), **FAST)
        self.assertTrue(compound["compound"])
        self.assertEqual(compound["loop_count"], 2)
        self.assertEqual(compound["fill_rule"], "evenodd")
        self.assertEqual(compound["fit"]["loop_count"], 2)
        self.assertTrue(all(item["closed"]
                            for item in compound["fit"]["contours"]))
        self.assertLessEqual(compound["anchors_after"], 4)
        one_loop_compound = optimize_compound_contours(
            [outer], error_budget_percent=0.08,
            tolerance_percents=(0.03, 0.10), **FAST)
        self.assertTrue(one_loop_compound["compound"])
        self.assertEqual(one_loop_compound["fill_rule"], "evenodd")
        self.assertEqual(one_loop_compound["fit"]["loop_count"], 1)

    def test_noisy_low_resolution_circle_prefers_native_circle(self):
        theta = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
        points = np.column_stack((
            np.round(100.0 + 80.0 * np.cos(theta)),
            np.round(100.0 + 80.0 * np.sin(theta)),
        ))
        # A single, sub-tail sampling blemish must not force dozens of anchors.
        direction = points[7] - np.array([100.0, 100.0])
        points[7] += 1.6 * direction / np.linalg.norm(direction)
        result = optimize_curve(
            points, closed=True, error_budget_percent=0.50,
            tolerance_percents=GRID, **FAST)
        self.assertEqual(result["fit"]["primitive"], "circle")
        self.assertEqual(result["primitive_complexity"]["category"],
                         "native_primitive")
        self.assertLessEqual(result["designer_anchor_count"], 4)
        self.assertLessEqual(result["anchors_after"], 4)
        self.assertLessEqual(result["actual_p95_error_percent"], 0.50)
        self.assertLessEqual(result["actual_max_error_percent"], 1.50)
        self.assertLessEqual(result["over_budget_share"], 0.05)

    def test_irregular_shape_over_budget_is_not_forced_to_circle(self):
        theta = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
        radius = 70.0 * (1.0 + 0.20 * np.cos(3.0 * theta))
        points = np.column_stack((radius * np.cos(theta),
                                  radius * np.sin(theta)))
        result = optimize_curve(
            points, closed=True, error_budget_percent=0.25,
            tolerance_percents=GRID, **FAST)
        self.assertNotEqual(result["fit"].get("primitive"), "circle")
        self.assertLessEqual(result["actual_p95_error_percent"], 0.25)
        circle_rows = [
            row for row in result["candidates"]
            if row.get("primitive_complexity", {}).get("native_primitives")
            and "circle" in row["primitive_complexity"]["native_primitives"]
        ]
        self.assertTrue(all(not row["eligible"] for row in circle_rows))

    def test_measurement_and_optimizer_evidence_are_json_safe_and_deterministic(self):
        points = _s_curve()
        first = optimize_curve(
            points, error_budget_percent=0.30,
            tolerance_percents=GRID, **FAST)
        second = optimize_curve(
            points, error_budget_percent=0.30,
            tolerance_percents=GRID, **FAST)
        self.assertEqual(first, second)
        encoded = json.dumps(first, ensure_ascii=False, sort_keys=True)
        self.assertIn("geometry_only", encoded)
        measured = measure_fit_error(
            points, first["fit"], closed=False,
            error_budget_percent=0.30, **FAST)
        json.dumps(measured, ensure_ascii=False, sort_keys=True)
        self.assertIn("source_to_fit", measured)
        self.assertIn("fit_to_source", measured)
        self.assertIn("over_budget_share", measured)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            optimize_curve(np.ones((1, 2)), error_budget_percent=0.2)
        with self.assertRaises(ValueError):
            optimize_curve(np.ones((4, 2)), closed=True,
                           error_budget_percent=0.0)
        with self.assertRaises(ValueError):
            optimize_compound_contours([], error_budget_percent=0.2)
        with self.assertRaises(ValueError):
            optimize_curve(np.ones((4, 2)), closed=True,
                           error_budget_percent=0.2,
                           tolerance_percents=[])


class SalientCornerSamplingTests(unittest.TestCase):
    @staticmethod
    def reference(points, *, closed, scale, window_percent=1.5, minimum_turn_degrees=45.):
        count=len(points);window=max(scale*window_percent/100.,scale*1e-9)
        def distant(start,direction):
            current=start;travelled=0.
            for _ in range(count-1):
                nxt=current+direction
                if closed:nxt%=count
                elif nxt<0 or nxt>=count:return current
                travelled+=float(np.linalg.norm(points[nxt]-points[current]));current=nxt
                if travelled>=window:break
            return current
        result=[]
        for i in range(0 if closed else 1,count if closed else count-1):
            previous,following=distant(i,-1),distant(i,1)
            if previous==i or following==i:continue
            incoming,outgoing=points[i]-points[previous],points[following]-points[i]
            denominator=float(np.linalg.norm(incoming)*np.linalg.norm(outgoing))
            if denominator<=1e-12:continue
            cosine=float(np.clip(np.dot(incoming,outgoing)/denominator,-1.,1.))
            if math.acos(cosine)>=math.radians(minimum_turn_degrees):result.append(i)
        return result

    def test_fast_corner_indices_match_old_walk_for_open_closed_zero_and_uneven_edges(self):
        rng=np.random.default_rng(2105)
        random=np.cumsum(rng.normal(size=(113,2))*rng.uniform(.0001,7,(113,1)),axis=0)
        fixtures=[random,np.repeat(random,2,axis=0),np.zeros((30,2)),
                  np.array([[0,0],[10,0],[10,10],[0,10]],float),
                  np.vstack((np.column_stack((np.linspace(0,100,300),np.zeros(300))),[[100,30],[0,30]]))]
        for points in fixtures:
            for closed in (False,True):
                for window in (.015,1.5,35.,300.):
                    with self.subTest(count=len(points),closed=closed,window=window):
                        args=dict(closed=closed,scale=120.,window_percent=window)
                        self.assertEqual(_salient_corner_indices(points,**args),self.reference(points,**args))

    def test_floating_window_boundary_keeps_original_accumulation_tiebreak(self):
        points=np.array([[0,0],[.1,0],[.3,0],[.6,0],[.6,.3],[.6,.5],[.6,.6],[.5,.6],[.3,.6],[0,.6],[0,.3]],float)
        for closed in (False,True):
            for scale in (.3,np.nextafter(.3,0),np.nextafter(.3,1),.6):
                args=dict(closed=closed,scale=scale,window_percent=100.)
                self.assertEqual(_salient_corner_indices(points,**args),self.reference(points,**args))

    def test_dense_curve_avoids_repeated_norm_walk_without_reducing_source_points(self):
        # Counting scalar norm evaluations verifies linear work without a
        # wall-clock assertion dependent on the machine running the test.
        angles=np.linspace(0,2*math.pi,10000,endpoint=False)
        points=np.column_stack((120*np.cos(angles),80*np.sin(angles)))
        with mock.patch('geometry_error_optimizer.np.linalg.norm',wraps=np.linalg.norm) as norm:
            actual=_salient_corner_indices(points,closed=True,scale=300.)
        self.assertEqual(actual,[])
        self.assertLess(norm.call_count,6*len(points))
        self.assertEqual(len(points),10000)


if __name__ == "__main__":
    unittest.main()
