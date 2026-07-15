"""Focused tests for the public binary-mask compound-path primitive."""

from __future__ import annotations

import math
import unittest

import numpy as np

from trace_engine import binary_mask_to_compound_path


class BinaryMaskCompoundPathTests(unittest.TestCase):
    def test_outer_contour_and_inner_hole_form_one_compound_path(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:10, 1:10] = True
        mask[4:7, 4:7] = False

        result = binary_mask_to_compound_path(
            mask, simplify=0.0, min_area=1.0, smooth=0.0, curve=0.0)

        self.assertEqual(result["loop_count"], 2)
        self.assertEqual(result["path"].count("M"), 2)
        self.assertEqual(result["path"].count("Z"), 2)
        self.assertEqual(result["bbox"], [1.0, 1.0, 9.0, 9.0])
        self.assertGreaterEqual(result["node_count"], 8)
        self.assertEqual(result["mask_pixels"], 72)
        self.assertEqual(result["mask_size"], [12, 12])
        self.assertEqual(result["fill_rule"], "evenodd")

    def test_single_pixel_island_is_retained_at_unit_minimum_area(self):
        mask = np.zeros((12, 14), dtype=bool)
        mask[2, 3] = True
        mask[7:9, 9:11] = True

        retained = binary_mask_to_compound_path(
            mask, simplify=0.0, min_area=1.0, smooth=0.0, curve=0.0)
        filtered = binary_mask_to_compound_path(
            mask, simplify=0.0, min_area=1.01, smooth=0.0, curve=0.0)

        self.assertEqual(retained["loop_count"], 2)
        self.assertEqual(retained["bbox"], [3.0, 2.0, 8.0, 7.0])
        self.assertEqual(retained["mask_pixels"], 5)
        self.assertEqual(filtered["loop_count"], 1)
        self.assertEqual(filtered["bbox"], [9.0, 7.0, 2.0, 2.0])

    def test_result_is_deterministic_and_does_not_mutate_mask(self):
        mask = np.zeros((31, 37), dtype=bool)
        mask[3:25, 5:11] = True
        mask[19:27, 10:30] = True
        mask[7:13, 20:33] = True
        before = mask.copy()

        expected = binary_mask_to_compound_path(
            mask, simplify=0.45, min_area=1.0, smooth=0.6, curve=0.35)
        for _ in range(5):
            self.assertEqual(
                binary_mask_to_compound_path(
                    mask, simplify=0.45, min_area=1.0,
                    smooth=0.6, curve=0.35),
                expected,
            )
        np.testing.assert_array_equal(mask, before)

    def test_curved_bbox_conservatively_contains_straight_contour(self):
        mask = np.zeros((18, 20), dtype=bool)
        mask[2:15, 3:7] = True
        mask[11:16, 6:17] = True

        straight = binary_mask_to_compound_path(
            mask, simplify=0.0, min_area=1.0, smooth=0.0, curve=0.0)
        curved = binary_mask_to_compound_path(
            mask, simplify=0.0, min_area=1.0, smooth=0.0, curve=1.0)

        sx, sy, sw, sh = straight["bbox"]
        cx, cy, cw, ch = curved["bbox"]
        self.assertLessEqual(cx, sx)
        self.assertLessEqual(cy, sy)
        self.assertGreaterEqual(cx + cw, sx + sw)
        self.assertGreaterEqual(cy + ch, sy + sh)

    def test_empty_mask_has_explicit_empty_result(self):
        result = binary_mask_to_compound_path(
            np.zeros((4, 7), dtype=bool))

        self.assertEqual(result["path"], "")
        self.assertEqual(result["node_count"], 0)
        self.assertEqual(result["loop_count"], 0)
        self.assertIsNone(result["bbox"])
        self.assertEqual(result["mask_pixels"], 0)

    def test_rejects_non_boolean_or_non_two_dimensional_masks(self):
        with self.assertRaises(TypeError):
            binary_mask_to_compound_path([[True, False]])
        with self.assertRaises(TypeError):
            binary_mask_to_compound_path(np.ones((2, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            binary_mask_to_compound_path(np.ones((2, 2, 1), dtype=bool))
        with self.assertRaises(ValueError):
            binary_mask_to_compound_path(np.zeros((0, 2), dtype=bool))

    def test_rejects_invalid_numeric_parameters(self):
        mask = np.ones((2, 2), dtype=bool)
        invalid = (
            ("simplify", -0.01),
            ("simplify", math.nan),
            ("min_area", math.inf),
            ("min_area", -1),
            ("smooth", "0.5"),
            ("smooth", True),
            ("curve", 1.01),
        )
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with self.assertRaises((TypeError, ValueError)):
                    binary_mask_to_compound_path(mask, **{name: value})


if __name__ == "__main__":
    unittest.main()
