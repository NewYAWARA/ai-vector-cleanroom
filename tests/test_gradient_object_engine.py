# -*- coding: utf-8 -*-

import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from gradient_object_engine import (
    SCHEMA,
    _clip01,
    _delta_e,
    _gradient_acceptance,
    _gradient_stop_offset_families,
    _stable_bounded_argsort,
    fit_gradient_object_proposal,
    propose_gradient_object,
)


def _paint_stops(t, offsets, colours):
    t = np.asarray(t, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    colours = np.asarray(colours, dtype=np.float64)
    result = np.empty(t.shape + (3,), dtype=np.float64)
    for channel in range(3):
        result[..., channel] = np.interp(t, offsets, colours[:, channel])
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _linear_ramp(height=72, width=108, angle_degrees=31.0):
    yy, xx = np.indices((height, width), dtype=np.float64)
    radians = math.radians(angle_degrees)
    direction = np.asarray((math.cos(radians), math.sin(radians)))
    projection = xx * direction[0] + yy * direction[1]
    t = (projection - projection.min()) / (projection.max() - projection.min())
    offsets = [0.0, 0.18, 0.47, 0.73, 1.0]
    colours = [
        [18, 42, 82],
        [25, 92, 145],
        [63, 151, 174],
        [157, 204, 126],
        [244, 188, 54],
    ]
    return _paint_stops(t, offsets, colours), np.ones((height, width), bool), direction


def _radial_sun(size=121):
    yy, xx = np.indices((size, size), dtype=np.float64)
    cx, cy = 61.0, 57.0
    angle = math.radians(24.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    dx, dy = xx - cx, yy - cy
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy
    radius_x, radius_y = 49.0, 37.0
    radius = np.sqrt((xr / radius_x) ** 2 + (yr / radius_y) ** 2)
    mask = radius <= 1.0
    t = np.clip(radius, 0.0, 1.0)
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    paint = _paint_stops(
        t,
        [0.0, 0.42, 1.0],
        [[255, 247, 132], [252, 207, 42], [219, 132, 5]],
    )
    image[mask] = paint[mask]
    return image, mask, (cx, cy), (radius_x, radius_y)


def _symmetric_linear_ramp(height=71, width=109, angle_degrees=17.0):
    yy, xx = np.indices((height, width), dtype=np.float64)
    radians = math.radians(angle_degrees)
    direction = np.asarray((math.cos(radians), math.sin(radians)))
    projection = xx * direction[0] + yy * direction[1]
    t = (projection - projection.min()) / (projection.max() - projection.min())
    image = _paint_stops(
        t,
        [0.0, 0.5, 1.0],
        [[22, 45, 70], [218, 196, 72], [22, 45, 70]],
    )
    return image, np.ones((height, width), bool), direction


def _boomerang_five_stop_ramp(height=101, width=141):
    yy, xx = np.indices((height, width), dtype=np.float64)
    radians = math.radians(37.0)
    direction = np.asarray((math.cos(radians), math.sin(radians)))
    projection = xx * direction[0] + yy * direction[1]
    t = ((projection - projection.min())
         / (projection.max() - projection.min()))
    image = _paint_stops(
        t,
        [0.0, 0.22, 0.48, 0.75, 1.0],
        [
            [30, 40, 130],
            [220, 50, 40],
            [40, 210, 75],
            [235, 190, 35],
            [30, 40, 130],
        ],
    )
    mask = (
        (
            (yy > 10.0 + 0.15 * xx)
            & (yy < 32.0 + 0.50 * xx)
            & (xx < 100.0)
        )
        | (
            (yy > 50.0 - 0.15 * xx)
            & (yy < 82.0 - 0.35 * np.maximum(xx - 70.0, 0.0))
            & (xx > 55.0)
        )
    )
    return image, mask, direction


def _reference_gradient_stop_offset_families(
    t, colours, training, *, maximum_stops
):
    """Frozen pre-optimisation implementation for byte-level equivalence."""
    maximum_stops = max(2, min(5, int(maximum_stops)))
    train_t = _clip01(t[training])
    train_colours = np.asarray(colours[training], dtype=np.float64)
    bin_count = 33
    bin_index = np.minimum(bin_count - 1, (train_t * bin_count).astype(int))
    profile_t = []
    profile_colour = []
    for index in range(bin_count):
        members = bin_index == index
        if int(members.sum()) < 3:
            continue
        profile_t.append(float(np.median(train_t[members])))
        profile_colour.append(np.median(train_colours[members], axis=0))

    families = [np.asarray((0.0, 1.0), dtype=np.float64)]
    if len(profile_t) >= 3:
        profile_t_array = np.asarray(profile_t, dtype=np.float64)
        profile_rgb = np.asarray(profile_colour, dtype=np.float64)
        selected = [0, len(profile_t_array) - 1]
        while len(selected) < maximum_stops:
            selected.sort()
            best_index = None
            best_error = -1.0
            for left_index, right_index in zip(selected[:-1], selected[1:]):
                if right_index <= left_index + 1:
                    continue
                left_t = profile_t_array[left_index]
                right_t = profile_t_array[right_index]
                fraction = (
                    (profile_t_array[left_index + 1 : right_index] - left_t)
                    / max(1e-9, right_t - left_t)
                )[:, None]
                prediction = (
                    profile_rgb[left_index][None, :] * (1.0 - fraction)
                    + profile_rgb[right_index][None, :] * fraction
                )
                errors = _delta_e(
                    profile_rgb[left_index + 1 : right_index], prediction
                )
                local = int(np.argmax(errors))
                value = float(errors[local])
                candidate_index = left_index + 1 + local
                if value > best_error + 1e-12 or (
                    abs(value - best_error) <= 1e-12
                    and (best_index is None or candidate_index < best_index)
                ):
                    best_error = value
                    best_index = candidate_index
            if best_index is None:
                break
            selected.append(best_index)
            offsets = [0.0]
            offsets.extend(
                float(profile_t_array[i]) for i in sorted(selected)[1:-1]
            )
            offsets.append(1.0)
            offsets_array = np.asarray(offsets, dtype=np.float64)
            if np.all(np.diff(offsets_array) >= 0.04):
                families.append(offsets_array)

    for stop_count in range(3, maximum_stops + 1):
        families.append(np.linspace(0.0, 1.0, stop_count, dtype=np.float64))

    unique = {}
    for offsets in families:
        key = tuple(round(float(value), 5) for value in offsets)
        unique.setdefault(key, offsets)
    return sorted(unique.values(), key=lambda item: (len(item), tuple(item)))


class GradientObjectEngineTests(unittest.TestCase):
    def test_stable_bounded_argsort_exactly_matches_full_reference(self):
        cases = [
            np.asarray([], dtype=np.uint64),
            np.asarray([7, 7, 7, 7], dtype=np.uint64),
            np.asarray([9, 2, 7, 2, 5, 2, 9, 1], dtype=np.uint64),
            np.random.default_rng(8147).integers(
                0, 23, size=4097, dtype=np.uint64),
        ]
        for values in cases:
            for limit in sorted({0, 1, 2, 7, len(values) // 2,
                                 len(values), len(values) + 3}):
                with self.subTest(size=len(values), limit=limit):
                    bounded = max(0, min(limit, len(values)))
                    expected = np.argsort(values, kind="stable")[:bounded]
                    actual = _stable_bounded_argsort(values, limit)
                    np.testing.assert_array_equal(actual, expected)

    def test_stop_offset_grouping_matches_preoptimisation_reference_exactly(self):
        rng = np.random.default_rng(20260719)
        random_t = rng.uniform(-0.2, 1.2, size=4096)
        # Exact bin edges, repeated values and sparsely occupied edge bins
        # exercise the membership semantics that the stable grouping must keep.
        boundary_t = np.repeat(np.linspace(0.0, 1.0, 34), (1, 2, 3, 4) * 8 + (1, 2))
        t = np.concatenate((random_t, boundary_t)).astype(np.float64)
        colours = rng.uniform(-20.0, 275.0, size=(len(t), 3)).astype(np.float64)
        training = rng.permutation(len(t))[:3777]

        for maximum_stops in (1, 2, 3, 4, 5, 8):
            with self.subTest(maximum_stops=maximum_stops):
                expected = _reference_gradient_stop_offset_families(
                    t, colours, training, maximum_stops=maximum_stops)
                actual = _gradient_stop_offset_families(
                    t, colours, training, maximum_stops=maximum_stops)
                self.assertEqual(len(actual), len(expected))
                for actual_offsets, expected_offsets in zip(actual, expected):
                    np.testing.assert_array_equal(actual_offsets, expected_offsets)

    def test_dominant_improvement_tail_exception_is_narrow_and_auditable(self):
        candidate = {
            "heldout_error": {"mean": 2.4, "p90": 5.2, "p99": 12.6},
            "comparison_to_solid": {
                "mean_improvement": 6.5,
                "p90_improvement": 16.0,
                "p99_improvement": 11.0,
                "improved_share": 0.82,
                "degraded_share": 0.085,
                "materially_degraded_share": 0.061,
                "p99_sample_degradation": 7.0,
            },
        }
        solid = {"heldout_error": {"mean": 8.9, "p90": 21.2, "p99": 23.6}}
        edges = {
            "material_internal_hard_edge": False,
            "material_hard_label_boundary": False,
        }
        kwargs = dict(
            maximum_mean_error=5.5, maximum_p90_error=8.5,
            maximum_p99_error=13.0, maximum_degraded_share=0.28,
            maximum_materially_degraded_share=0.10,
            maximum_p99_sample_degradation=3.0,
        )

        accepted, reasons = _gradient_acceptance(
            candidate, solid, edges, 40.0, **kwargs)
        self.assertTrue(accepted, reasons)
        self.assertTrue(candidate[
            "dominant_improvement_tail_exception"]["used"])

        candidate["comparison_to_solid"]["materially_degraded_share"] = 0.071
        accepted, reasons = _gradient_acceptance(
            candidate, solid, edges, 40.0, **kwargs)
        self.assertFalse(accepted)
        self.assertIn("heldout_sample_tail_degradation_too_high", reasons)

        candidate["comparison_to_solid"]["materially_degraded_share"] = 0.061
        kwargs["maximum_p99_sample_degradation"] = 0.1
        accepted, reasons = _gradient_acceptance(
            candidate, solid, edges, 40.0, **kwargs)
        self.assertFalse(accepted)
        self.assertIn("heldout_sample_tail_degradation_too_high", reasons)

    def test_three_stop_linear_ramp_needs_more_than_endpoints(self):
        height, width = 60, 94
        yy, xx = np.indices((height, width), dtype=np.float64)
        direction = np.asarray((math.cos(-0.37), math.sin(-0.37)))
        projection = xx * direction[0] + yy * direction[1]
        t = (projection - projection.min()) / (projection.max() - projection.min())
        image = _paint_stops(
            t,
            [0.0, 0.46, 1.0],
            [[12, 35, 92], [35, 202, 196], [224, 184, 52]],
        )
        proposal = propose_gradient_object(
            image, np.ones((height, width), bool), validation_seed=71
        )

        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["model"]["type"], "linear")
        self.assertGreaterEqual(proposal["model"]["stop_count"], 3)
        self.assertLessEqual(proposal["model"]["stop_count"], 5)
        self.assertLess(proposal["error"]["heldout"]["p90"], 4.0)

    def test_symmetric_ramp_escalates_when_first_order_slope_cancels(self):
        image, mask, expected_direction = _symmetric_linear_ramp()
        proposal = propose_gradient_object(
            image, mask, allow_radial=False, validation_seed=29)

        self.assertEqual(proposal["status"], "proposed", proposal.get("reasons"))
        model = proposal["model"]
        self.assertTrue(model["linear_axis_regression_fallback_used"])
        self.assertEqual(model["linear_axis_requested_full_fit_count"], 1)
        self.assertEqual(model["linear_axis_full_fit_count"], 5)
        fitted = np.asarray(model["direction"])
        self.assertGreater(abs(float(fitted @ expected_direction)), 0.98)
        self.assertLess(proposal["error"]["heldout"]["mean"], 2.5)

    def test_nonconvex_multihue_rank_is_diagnostic_not_a_rejection(self):
        image, mask, expected_direction = _boomerang_five_stop_ramp()
        proposal = propose_gradient_object(
            image,
            mask,
            allow_radial=False,
            validation_seed=17,
            linear_direction_candidates=1,
        )

        self.assertEqual(proposal["status"], "proposed", proposal)
        linear = next(item for item in proposal["candidates"]
                      if item["type"] == "linear")
        model = linear["model"]
        self.assertGreater(model["colour_field_rank_ratio"], 0.25)
        self.assertTrue(model["linear_axis_regression_fallback_used"])
        self.assertEqual(model["linear_axis_requested_full_fit_count"], 1)
        self.assertEqual(model["linear_axis_full_fit_count"], 5)
        self.assertEqual(model["stop_count"], 5)
        fitted = np.asarray(model["direction"], dtype=np.float64)
        self.assertGreater(abs(float(fitted @ expected_direction)), 0.995)
        self.assertLess(linear["heldout_error"]["mean"], 1.0)
        self.assertLess(linear["heldout_error"]["p99"], 3.0)
        self.assertGreater(
            linear["comparison_to_solid"]["improved_share"], 0.99)
        self.assertNotIn("colour_field_not_one_dimensional", linear["reasons"])
        self.assertFalse(
            proposal["validation"]["model_selection_uses_outer_heldout"])
        json.dumps(proposal, allow_nan=False, sort_keys=True)

    def test_five_stop_arbitrary_angle_linear_ramp(self):
        image, mask, expected_direction = _linear_ramp()
        proposal = propose_gradient_object(image, mask, validation_seed=17)

        self.assertEqual(proposal["schema"], SCHEMA)
        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["model"]["type"], "linear")
        self.assertGreaterEqual(proposal["model"]["stop_count"], 3)
        self.assertLessEqual(proposal["model"]["stop_count"], 5)
        fitted_direction = np.asarray(proposal["model"]["direction"])
        self.assertGreater(abs(float(fitted_direction @ expected_direction)), 0.97)
        self.assertLess(proposal["error"]["heldout"]["mean"], 2.5)
        self.assertGreater(proposal["error"]["heldout_mean_improvement"], 4.0)
        comparison = proposal["error"]["comparison_to_solid"]
        self.assertLessEqual(
            comparison["degraded_share"],
            proposal["validation"]["maximum_degraded_share"],
        )
        self.assertLessEqual(
            comparison["materially_degraded_share"],
            proposal["validation"]["maximum_materially_degraded_share"],
        )
        evidence = proposal["selection_evidence"]
        self.assertFalse(evidence["geometry_selection_authorised"])
        self.assertFalse(evidence["model_selection_uses_outer_heldout"])
        self.assertEqual(
            proposal["validation"]["model_selection_strategy"],
            "inner_training_model_selection_then_single_outer_accept_or_skip",
        )
        self.assertEqual(evidence["paint_model_complexity"]["native_gradient_count"], 1)

    def test_elliptical_radial_sun_is_paint_only(self):
        image, mask, center, radii = _radial_sun()
        proposal = propose_gradient_object(image, mask, validation_seed=3)

        self.assertEqual(proposal["status"], "proposed")
        self.assertEqual(proposal["model"]["type"], "radial")
        self.assertEqual(proposal["model"]["svg_type"], "radialGradient")
        self.assertGreaterEqual(proposal["model"]["stop_count"], 2)
        fitted_center = proposal["model"]["center"]
        self.assertLess(abs(fitted_center[0] - center[0]), 3.0)
        self.assertLess(abs(fitted_center[1] - center[1]), 3.0)
        fitted_radii = sorted(
            [proposal["model"]["radius_x"], proposal["model"]["radius_y"]]
        )
        self.assertLess(abs(fitted_radii[0] - min(radii)), 5.0)
        self.assertLess(abs(fitted_radii[1] - max(radii)), 6.0)
        self.assertFalse(
            proposal["selection_evidence"]["geometry_selection_authorised"]
        )

    def test_solid_fill_never_becomes_a_gradient(self):
        image = np.full((48, 64, 3), [37, 119, 83], dtype=np.uint8)
        mask = np.ones(image.shape[:2], dtype=bool)
        proposal = propose_gradient_object(image, mask)

        self.assertEqual(proposal["status"], "skipped")
        self.assertIsNone(proposal["model"])
        self.assertIn("solid_baseline_preferred", proposal["reasons"])
        self.assertIn("no_gradient_model_passed_validation", proposal["reasons"])

    def test_two_object_hard_edge_union_fails_closed(self):
        height, width = 64, 96
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, : width // 2] = [24, 69, 111]
        image[:, width // 2 :] = [230, 172, 34]
        mask = np.ones((height, width), dtype=bool)
        labels = np.zeros((height, width), dtype=np.int32)
        labels[:, width // 2 :] = 1

        proposal = propose_gradient_object(
            image, mask, label_map=labels, validation_seed=11
        )

        self.assertEqual(proposal["status"], "skipped")
        self.assertIsNone(proposal["model"])
        self.assertIn("material_internal_hard_edge", proposal["reasons"])
        self.assertIn("material_hard_label_boundary", proposal["reasons"])
        edges = proposal["validation"]["internal_edges"]
        self.assertTrue(edges["material_internal_hard_edge"])
        self.assertTrue(edges["material_hard_label_boundary"])
        self.assertFalse(proposal["validation"]["passed"])

    def test_degraded_share_and_sample_tail_budgets_are_enforced(self):
        height, width = 72, 108
        yy, xx = np.indices((height, width))
        t = xx.astype(np.float64) / (width - 1)
        start = np.asarray([20.0, 50.0, 80.0])
        end = np.asarray([220.0, 180.0, 60.0])
        low_amplitude_texture = 5.0 * (((xx + yy) % 2) * 2 - 1)
        image = (
            start + (end - start) * t[..., None]
            + low_amplitude_texture[..., None]
        )
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
        mask = np.ones((height, width), dtype=bool)

        ordinary = propose_gradient_object(
            image, mask, allow_radial=False, validation_seed=0
        )
        self.assertEqual(ordinary["status"], "proposed")
        comparison = ordinary["error"]["comparison_to_solid"]
        self.assertGreater(comparison["degraded_share"], 0.01)
        self.assertGreater(comparison["p99_sample_degradation"], 0.1)
        self.assertEqual(
            ordinary["validation"]["internal_edges"]["hard_pair_count"], 0
        )

        degraded_rejected = propose_gradient_object(
            image,
            mask,
            allow_radial=False,
            validation_seed=0,
            maximum_degraded_share=0.01,
        )
        self.assertEqual(degraded_rejected["status"], "skipped")
        self.assertIn(
            "degraded_share_exceeds_safety_budget",
            degraded_rejected["candidates"][1]["reasons"],
        )

        tail_rejected = propose_gradient_object(
            image,
            mask,
            allow_radial=False,
            validation_seed=0,
            maximum_p99_sample_degradation=0.1,
        )
        self.assertEqual(tail_rejected["status"], "skipped")
        self.assertIn(
            "heldout_sample_tail_degradation_too_high",
            tail_rejected["candidates"][1]["reasons"],
        )

    def test_direction_search_is_bounded_inner_validated_and_deterministic(self):
        image, mask, expected_direction = _linear_ramp(
            height=61, width=97, angle_degrees=37.0)
        options = dict(
            validation_seed=31415,
            allow_radial=False,
            linear_direction_candidates=5,
        )
        first = propose_gradient_object(image, mask, **options)
        second = propose_gradient_object(image, mask, **options)

        self.assertEqual(first, second)
        self.assertEqual(first["status"], "proposed")
        model = first["model"]
        self.assertEqual(model["type"], "linear")
        self.assertEqual(model["linear_axis_full_fit_count"], 5)
        self.assertGreater(model["linear_axis_prescreen_count"], 5)
        self.assertFalse(model["linear_axis_selection_uses_outer_heldout"])
        fitted = np.asarray(model["direction"])
        self.assertGreater(abs(float(fitted @ expected_direction)), 0.96)
        search = first["selection_evidence"]["linear_direction_search"]
        self.assertEqual(search["requested_full_fit_candidates"], 5)
        self.assertEqual(search["actual_full_fit_candidates"], 5)
        self.assertFalse(search["outer_heldout_used_for_axis_selection"])

        invalid = propose_gradient_object(
            image, mask, linear_direction_candidates=9)
        self.assertEqual(invalid["status"], "error")
        self.assertIn("linear_direction_candidates", invalid["reasons"][0])

    def test_outer_rejection_is_single_gate_without_model_fallback(self):
        image, mask, _direction = _linear_ramp(height=58, width=91)
        with patch(
            "gradient_object_engine._gradient_acceptance",
            return_value=(False, ["forced_outer_rejection"]),
        ) as gate:
            proposal = propose_gradient_object(
                image, mask, allow_radial=True, validation_seed=111)

        self.assertEqual(gate.call_count, 1)
        self.assertEqual(proposal["status"], "skipped")
        self.assertIn("forced_outer_rejection", proposal["reasons"])
        self.assertEqual(
            proposal["validation"]["outer_holdout_role"],
            "selected_model_accept_or_skip_no_fallback",
        )
        self.assertFalse(
            proposal["validation"]["model_selection_uses_outer_heldout"])
        self.assertIn("No fallback model", proposal["scope_note"])

    def test_coordinate_hash_seed_and_json_are_deterministic(self):
        image, mask, _direction = _linear_ramp(height=56, width=87)
        first = fit_gradient_object_proposal(
            image, mask, validation_seed=90210, max_samples=2048
        )
        second = fit_gradient_object_proposal(
            image, mask, validation_seed=90210, max_samples=2048
        )

        self.assertEqual(first, second)
        self.assertEqual(first["validation"]["validation_seed"], 90210)
        encoded = json.dumps(first, allow_nan=False, sort_keys=True)
        self.assertIn("deterministic_coordinate_hash_holdout", encoded)
        other_seed = fit_gradient_object_proposal(
            image, mask, validation_seed=90211, max_samples=2048
        )
        self.assertEqual(other_seed["status"], "proposed")
        self.assertEqual(other_seed["model"]["type"], "linear")
        self.assertEqual(other_seed["validation"]["validation_seed"], 90211)
        self.assertLess(
            abs(first["error"]["heldout"]["mean"]
                - other_seed["error"]["heldout"]["mean"]),
            0.5,
        )

    def test_scale_change_preserves_model_class_and_error_budget(self):
        small_image, small_mask, small_direction = _linear_ramp(
            height=42, width=63, angle_degrees=38.0
        )
        large_image, large_mask, large_direction = _linear_ramp(
            height=84, width=126, angle_degrees=38.0
        )
        small = propose_gradient_object(small_image, small_mask, validation_seed=5)
        large = propose_gradient_object(large_image, large_mask, validation_seed=5)

        for proposal, direction in (
            (small, small_direction), (large, large_direction)
        ):
            self.assertEqual(proposal["status"], "proposed")
            self.assertEqual(proposal["model"]["type"], "linear")
            fitted = np.asarray(proposal["model"]["direction"])
            self.assertGreater(abs(float(fitted @ direction)), 0.96)
            self.assertLess(
                proposal["error"]["heldout"]["mean"],
                proposal["validation"]["maximum_mean_error"],
            )
        self.assertLess(
            abs(small["error"]["heldout"]["mean"]
                - large["error"]["heldout"]["mean"]),
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
