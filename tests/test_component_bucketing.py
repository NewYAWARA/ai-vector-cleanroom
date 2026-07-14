"""Equivalence and scale checks for stroke component pixel bucketing."""

from __future__ import annotations

import time
import unittest

import numpy as np

from stroke_engine import (connected_components,
                           _group_eligible_component_pixels)


class ComponentBucketingRegression(unittest.TestCase):
    def assert_matches_argwhere(self, mask, min_area=1, max_area=None):
        labels, n = connected_components(mask)
        if max_area is None:
            max_area = mask.size
        areas, eligible, grouped, starts, ends = (
            _group_eligible_component_pixels(
                labels, n, min_area=min_area, max_area=max_area)
        )

        expected_labels = []
        for li in range(1, n + 1):
            expected = np.argwhere(labels == li)
            should_keep = min_area <= len(expected) <= max_area
            self.assertEqual(bool(eligible[li]), should_keep)
            self.assertEqual(int(areas[li]), len(expected))
            if not should_keep:
                continue
            flat = grouped[starts[li - 1]:ends[li - 1]]
            actual = np.column_stack(np.divmod(flat, mask.shape[1]))
            np.testing.assert_array_equal(actual, expected)
            expected_labels.append(li)

        self.assertEqual(np.flatnonzero(eligible).tolist(), expected_labels)

    def test_matches_argwhere_for_edge_cases_and_thresholds(self):
        cases = []
        cases.append(np.zeros((20, 30), dtype=bool))

        mixed = np.zeros((24, 40), dtype=bool)
        mixed[0:4, 0:6] = True       # exactly 24 pixels
        mixed[0:1, 10:33] = True     # 23 pixels
        mixed[10:16, 4:10] = True
        mixed[18:23, 30:35] = True
        cases.append(mixed)

        diagonal = np.zeros((31, 31), dtype=bool)
        np.fill_diagonal(diagonal, True)  # one component under 8-connectivity
        diagonal[3:9, 20:26] = True
        cases.append(diagonal)

        nested = np.zeros((48, 64), dtype=bool)
        nested[2:46, 2:62] = True
        nested[8:40, 8:56] = False
        nested[15:33, 15:49] = True
        cases.append(nested)

        for mask in cases:
            with self.subTest(shape=mask.shape, ink=int(mask.sum())):
                self.assert_matches_argwhere(mask, min_area=24,
                                             max_area=0.35 * mask.size)

    def test_thousands_of_components_group_without_full_image_rescans(self):
        mask = np.zeros((512, 512), dtype=bool)
        mask[::4, ::4] = True
        labels, n = connected_components(mask)
        self.assertEqual(n, 128 * 128)

        started = time.perf_counter()
        areas, eligible, grouped, starts, ends = (
            _group_eligible_component_pixels(
                labels, n, min_area=1, max_area=mask.size)
        )
        elapsed = time.perf_counter() - started

        self.assertEqual(len(grouped), n)
        self.assertEqual(int(eligible.sum()), n)
        self.assertTrue(np.all(areas[1:] == 1))
        self.assertEqual(int(starts[0]), 0)
        self.assertEqual(int(ends[-1]), n)
        self.assertLess(elapsed, 2.0,
                        "component bucketing regressed toward per-label scans")


if __name__ == "__main__":
    unittest.main()
