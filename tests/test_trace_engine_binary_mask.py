"""Focused tests for the public binary-mask compound-path primitive."""

from __future__ import annotations

import math
import unittest

import numpy as np

import trace_engine as te
from trace_engine import binary_mask_to_compound_path


def _legacy_mask_to_smooth_loops(mask, simplify, min_area, smooth):
    """Reference implementation preserving the former nested cell scan."""
    if smooth <= 0:
        return te._mask_to_loops(mask, simplify, min_area)

    h, w = mask.shape
    img = te.Image.fromarray((mask.astype(np.uint8) * 255), "L")
    img = img.filter(te.ImageFilter.GaussianBlur(float(smooth)))
    field = np.pad(
        np.asarray(img).astype(np.float32) / 255.0,
        1, mode="constant")
    level = 0.5
    segments = []
    edge_pairs = {
        1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)],
        5: [(0, 3), (1, 2)], 6: [(0, 2)], 7: [(3, 2)],
        8: [(2, 3)], 9: [(0, 2)], 10: [(0, 1), (3, 2)],
        11: [(1, 2)], 12: [(3, 1)], 13: [(0, 1)], 14: [(3, 0)],
    }
    fh, fw = field.shape
    for y in range(fh - 1):
        for x in range(fw - 1):
            v0 = field[y, x]
            v1 = field[y, x + 1]
            v2 = field[y + 1, x + 1]
            v3 = field[y + 1, x]
            case = (
                (1 if v0 >= level else 0)
                | (2 if v1 >= level else 0)
                | (4 if v2 >= level else 0)
                | (8 if v3 >= level else 0)
            )
            if case == 0 or case == 15:
                continue
            p0 = (x - 1.0, y - 1.0)
            p1 = (x, y - 1.0)
            p2 = (x, y)
            p3 = (x - 1.0, y)
            edge_points = {
                0: te._interp(level, p0, p1, v0, v1),
                1: te._interp(level, p1, p2, v1, v2),
                2: te._interp(level, p3, p2, v3, v2),
                3: te._interp(level, p0, p3, v0, v3),
            }
            for edge_a, edge_b in edge_pairs.get(case, []):
                a = edge_points[edge_a]
                b = edge_points[edge_b]
                a = (min(max(a[0], 0.0), float(w)),
                     min(max(a[1], 0.0), float(h)))
                b = (min(max(b[0], 0.0), float(w)),
                     min(max(b[1], 0.0), float(h)))
                if te._segment_key(a) != te._segment_key(b):
                    segments.append((a, b))
    return te._segments_to_loops(segments, simplify, min_area)


class BinaryMaskCompoundPathTests(unittest.TestCase):
    def test_vectorized_smooth_case_scan_matches_legacy_row_major_output(self):
        structured = np.zeros((27, 31), dtype=bool)
        structured[2:23, 3:12] = True
        structured[16:25, 10:28] = True
        structured[6:12, 18:27] = True
        structured[18:21, 14:18] = False
        random_mask = np.random.default_rng(20260719).random((23, 29)) > 0.58

        for mask in (structured, random_mask):
            for smooth in (0.35, 0.6, 1.0):
                with self.subTest(shape=mask.shape, smooth=smooth):
                    expected = _legacy_mask_to_smooth_loops(
                        mask, simplify=0.0, min_area=1.0, smooth=smooth)
                    actual = te._mask_to_smooth_loops(
                        mask, simplify=0.0, min_area=1.0, smooth=smooth)
                    self.assertEqual(actual, expected)

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
