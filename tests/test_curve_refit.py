"""Focused regressions for deterministic low-anchor contour refitting."""

from __future__ import annotations

import json
import math
import unittest

import numpy as np

from curve_refit import (
    _bezier_derivative,
    _bezier_point,
    _bezier_second_derivative,
    _reparameterize,
    fit_compound_contours,
    fit_curve,
    fit_mask,
)
from trace_engine import binary_mask_to_compound_path


def _reference_reparameterize(points, parameters, control):
    updated = np.asarray(parameters, dtype=np.float64).copy()
    for index, (point, value) in enumerate(zip(points, updated)):
        curve_point = _bezier_point(control, value)
        first = _bezier_derivative(control, value)
        second = _bezier_second_derivative(control, value)
        difference = curve_point - point
        denominator = float(np.dot(first, first) + np.dot(difference, second))
        if abs(denominator) <= 1.0e-12:
            continue
        updated[index] = float(np.clip(
            value - np.dot(difference, first) / denominator, 0.0, 1.0))
    updated[0] = 0.0
    updated[-1] = 1.0
    if np.any(np.diff(updated) <= 1.0e-9):
        return np.asarray(parameters, dtype=np.float64)
    return updated


def _dense_polygon(vertices, samples_per_edge=50):
    points = []
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        points.extend(np.linspace(start, end, samples_per_edge, endpoint=False))
    return np.asarray(points, dtype=np.float64)


class CurveRefitTests(unittest.TestCase):
    def test_vectorized_reparameterize_matches_scalar_reference_exactly(self):
        rng = np.random.default_rng(20260719)
        fixtures = []
        for count in (2, 3, 9, 64, 257):
            points = np.cumsum(rng.normal(size=(count, 2)), axis=0)
            parameters = np.linspace(0.0, 1.0, count, dtype=np.float64)
            control = rng.normal(size=(4, 2))
            fixtures.append((points, parameters, control))
        fixtures.extend((
            (np.zeros((7, 2)), np.linspace(0.0, 1.0, 7),
             np.zeros((4, 2))),
            (rng.normal(size=(11, 2)),
             np.asarray([0.0, 0.3, 0.2, 0.4, 0.5, 0.6,
                         0.7, 0.8, 0.9, 0.95, 1.0]),
             rng.normal(size=(4, 2))),
        ))
        for points, parameters, control in fixtures:
            with self.subTest(count=len(points)):
                expected = _reference_reparameterize(
                    points, parameters, control)
                actual = _reparameterize(points, parameters, control)
                np.testing.assert_array_equal(actual, expected)

    def test_near_straight_open_contour_becomes_one_line(self):
        x = np.linspace(0.0, 120.0, 241)
        points = np.column_stack((x, 0.4 * x + 7.0))

        result = fit_curve(points, tolerance=0.05)

        self.assertEqual(result["primitive"], "line")
        self.assertEqual(result["segment_type_counts"],
                         {"line": 1, "cubic": 0, "arc": 0})
        self.assertEqual(result["anchor_count"], 2)
        self.assertIn("L", result["path"])
        self.assertNotIn("C", result["path"])
        self.assertLessEqual(result["error"]["max"], 0.05)
        self.assertGreater(result["economy"]["reduction_ratio"], 0.98)

    def test_circle_and_rotated_ellipse_use_two_arc_compound_geometry(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
        circle = np.column_stack((
            40.0 + 24.0 * np.cos(theta),
            30.0 + 24.0 * np.sin(theta),
        ))
        circle_result = fit_curve(
            circle, closed=True, tolerance=0.02)

        angle = 0.47
        rotation = np.array([
            [math.cos(angle), math.sin(angle)],
            [-math.sin(angle), math.cos(angle)],
        ])
        local = np.column_stack((
            42.0 * np.cos(theta), 17.0 * np.sin(theta)))
        ellipse = local @ rotation + np.array([75.0, 51.0])
        ellipse_result = fit_curve(
            ellipse, closed=True, tolerance=0.03)

        self.assertEqual(circle_result["primitive"], "circle")
        self.assertEqual(circle_result["path"].count("A"), 2)
        self.assertEqual(circle_result["anchor_count"], 2)
        self.assertEqual(circle_result["native_primitive"]["element"], "circle")
        self.assertLessEqual(circle_result["error"]["max"], 0.02)

        self.assertEqual(ellipse_result["primitive"], "ellipse")
        self.assertEqual(ellipse_result["path"].count("A"), 2)
        self.assertEqual(ellipse_result["anchor_count"], 2)
        self.assertEqual(
            ellipse_result["native_primitive"]["element"], "ellipse")
        self.assertLessEqual(ellipse_result["error"]["max"], 0.03)

    def test_open_circular_arc_is_recognised_without_cubics(self):
        theta = np.linspace(-0.35, 2.7, 180)
        points = np.column_stack((
            15.0 + 38.0 * np.cos(theta),
            22.0 + 38.0 * np.sin(theta),
        ))

        result = fit_curve(points, tolerance=0.03)

        self.assertEqual(result["primitive"], "circular_arc")
        self.assertEqual(result["segment_type_counts"],
                         {"line": 0, "cubic": 0, "arc": 1})
        self.assertEqual(result["path"].count("A"), 1)
        self.assertEqual(result["anchor_count"], 2)
        self.assertLessEqual(result["error"]["max"], 0.03)

    def test_s_curve_uses_few_error_bounded_cubics(self):
        x = np.linspace(0.0, 120.0, 321)
        points = np.column_stack((
            x, 18.0 * np.sin(x / 120.0 * 2.0 * np.pi)))
        tolerance = 0.45

        result = fit_curve(
            points, tolerance=tolerance, primitive_tolerance=0.1)

        self.assertIsNone(result["primitive"])
        self.assertGreater(result["segment_type_counts"]["cubic"], 0)
        self.assertLessEqual(result["segment_count"], 6)
        self.assertLessEqual(result["error"]["max"], tolerance + 1e-9)
        self.assertGreater(result["economy"]["reduction_ratio"], 0.95)

    def test_anchor_removal_pass_merges_a_redundant_recursive_join(self):
        parameter = np.linspace(0.0, 1.0, 301)
        points = np.column_stack((
            100.0 * parameter,
            20.0 * np.sin(2.0 * np.pi * parameter)
            + 3.0 * np.sin(6.0 * np.pi * parameter),
        ))

        result = fit_curve(
            points, tolerance=0.05, allow_primitives=False)

        self.assertGreaterEqual(result["economy"]["merge_attempts"], 1)
        self.assertGreaterEqual(result["economy"]["merges_accepted"], 1)
        self.assertTrue(result["economy"]["locally_merge_minimal"])
        self.assertLessEqual(result["error"]["max"], 0.05 + 1e-9)
        self.assertLess(result["anchor_count"], len(points) / 10)

    def test_sharp_rectangle_corners_are_preserved_as_four_lines(self):
        points = _dense_polygon([
            (8.0, 11.0), (92.0, 11.0),
            (92.0, 67.0), (8.0, 67.0),
        ], samples_per_edge=70)

        result = fit_curve(
            points, closed=True, tolerance=0.2,
            primitive_tolerance=0.05)

        self.assertIsNone(result["primitive"])
        self.assertEqual(result["corner_count"], 4)
        self.assertEqual(result["segment_type_counts"],
                         {"line": 4, "cubic": 0, "arc": 0})
        self.assertEqual(result["anchor_count"], 4)
        self.assertEqual(result["path"].count("L"), 4)
        self.assertEqual(result["path"].count("Z"), 1)
        self.assertEqual(result["error"]["max"], 0.0)

    def test_mask_hole_stays_one_evenodd_compound_path(self):
        mask = np.zeros((90, 120), dtype=bool)
        mask[8:82, 9:111] = True
        mask[31:59, 41:79] = False
        before = mask.copy()

        result = fit_mask(mask, tolerance=0.85, smooth=0.55)

        np.testing.assert_array_equal(mask, before)
        self.assertEqual(result["fill_rule"], "evenodd")
        self.assertEqual(result["loop_count"], 2)
        self.assertEqual(result["path"].count("M"), 2)
        self.assertEqual(result["path"].count("Z"), 2)
        self.assertEqual(result["topology"]["components"], 1)
        self.assertEqual(result["topology"]["holes"], 1)
        self.assertTrue(result["topology"]["topology_preserved"])
        self.assertLessEqual(result["error"]["max"], 0.85)

    def test_raster_circle_drops_hundreds_of_raw_anchors(self):
        y, x = np.mgrid[:171, :171]
        mask = (x - 85) ** 2 + (y - 85) ** 2 <= 58 ** 2
        raw = binary_mask_to_compound_path(
            mask, simplify=0.0, min_area=1.0, smooth=0.55, curve=0.35)

        result = fit_mask(
            mask, tolerance=1.0, primitive_tolerance=1.0, smooth=0.55)

        self.assertEqual(result["contours"][0]["primitive"], "circle")
        self.assertEqual(result["anchor_count"], 2)
        self.assertGreater(raw["node_count"], 250)
        self.assertLess(result["anchor_count"], raw["node_count"] / 100)
        self.assertGreater(result["economy"]["reduction_ratio"], 0.98)
        self.assertLessEqual(result["error"]["max"], 1.0)

    def test_noisy_round_contour_refits_deterministically_within_budget(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        radial_noise = (
            0.20 * np.sin(37.0 * theta)
            + 0.12 * np.sin(83.0 * theta + 0.4))
        radius = 44.0 + radial_noise
        points = np.column_stack((
            60.0 + radius * np.cos(theta),
            58.0 + radius * np.sin(theta),
        ))

        expected = fit_curve(
            points, closed=True, tolerance=0.45,
            primitive_tolerance=0.45)
        for _ in range(4):
            self.assertEqual(
                fit_curve(points, closed=True, tolerance=0.45,
                          primitive_tolerance=0.45),
                expected,
            )

        self.assertEqual(expected["primitive"], "circle")
        self.assertLessEqual(expected["error"]["max"], 0.45)
        self.assertGreater(expected["economy"]["reduction_ratio"], 0.99)
        # The public result must remain serializable for report/receipt use.
        json.dumps(expected, sort_keys=True)

    def test_compound_contours_keep_every_closed_subpath(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        outer = np.column_stack((
            50.0 + 35.0 * np.cos(theta),
            50.0 + 35.0 * np.sin(theta),
        ))
        inner = np.column_stack((
            50.0 + 12.0 * np.cos(theta[::-1]),
            50.0 + 12.0 * np.sin(theta[::-1]),
        ))

        result = fit_compound_contours(
            [outer, inner], tolerance=0.05)

        self.assertEqual(result["fill_rule"], "evenodd")
        self.assertEqual(result["loop_count"], 2)
        self.assertEqual(result["path"].count("M"), 2)
        self.assertEqual(result["path"].count("Z"), 2)
        self.assertEqual(result["segment_count"], 4)
        self.assertEqual(result["anchor_count"], 4)

    def test_exact_fit_context_preserves_every_tolerance_in_any_order(self):
        x = np.linspace(0.0, 40.0, 161)
        points = np.column_stack((
            x,
            4.0 * np.sin(x / 3.0) + 0.35 * np.sin(x * 1.7),
        ))
        tolerances = (0.45, 0.025, 0.20, 0.06, 0.80)
        expected = [fit_curve(
            points, tolerance=value, line_tolerance=value,
            primitive_tolerance=value, allow_primitives=False)
            for value in tolerances]
        context = {}
        actual = [fit_curve(
            points, tolerance=value, line_tolerance=value,
            primitive_tolerance=value, allow_primitives=False,
            _fit_context=context)
            for value in tolerances]

        self.assertEqual(
            json.dumps(actual, sort_keys=True, separators=(",", ":")),
            json.dumps(expected, sort_keys=True, separators=(",", ":")))
        self.assertTrue(context.get("lines"))
        self.assertTrue(context.get("profiles"))
        self.assertTrue(getattr(
            fit_compound_contours, "_supports_exact_fit_context", False))

    def test_rejects_invalid_contours_masks_and_budgets(self):
        with self.assertRaises(ValueError):
            fit_curve(np.ones((1, 2)), tolerance=0.5)
        with self.assertRaises(ValueError):
            fit_curve(np.ones((4, 3)), tolerance=0.5)
        with self.assertRaises(ValueError):
            fit_curve(np.array([[0.0, 0.0], [math.nan, 1.0]]),
                      tolerance=0.5)
        with self.assertRaises(ValueError):
            fit_curve(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      tolerance=0.0)
        with self.assertRaises(TypeError):
            fit_mask([[True, False]], tolerance=0.5)
        with self.assertRaises(TypeError):
            fit_mask(np.ones((3, 3), dtype=np.uint8), tolerance=0.5)


if __name__ == "__main__":
    unittest.main()
