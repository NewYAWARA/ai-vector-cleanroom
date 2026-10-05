# -*- coding: utf-8 -*-

import unittest

import numpy as np

from gradient_candidate_groups import (
    _connected_components_4,
    propose_gradient_candidates,
)


def _legacy_connected_components_4(mask):
    """Reference implementation preserving the former all-runs scan."""
    source = np.asarray(mask, dtype=np.bool_)
    height, width = source.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = []

    def find(value):
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != root:
            parent[value], value = root, parent[value]
        return root

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
        return min(left, right)

    previous_runs = []
    for y in range(height):
        indices = np.flatnonzero(source[y])
        if indices.size == 0:
            previous_runs = []
            continue
        splits = np.flatnonzero(np.diff(indices) > 1)
        starts = np.concatenate(([indices[0]], indices[splits + 1]))
        ends = np.concatenate((indices[splits], [indices[-1]]))
        runs = []
        for start, end in zip(starts, ends):
            label = -1
            for old_start, old_end, old_label in previous_runs:
                if old_start <= end and old_end >= start:
                    label = (union(label, old_label) if label != -1
                             else find(old_label))
            if label == -1:
                label = len(parent)
                parent.append(label)
            labels[y, start:end + 1] = label + 1
            runs.append((start, end, label))
        previous_runs = runs
    if not parent:
        return labels, 0
    roots = np.asarray([find(index) for index in range(len(parent))],
                       dtype=np.int32)
    unique = np.unique(roots)
    remap = np.zeros(len(parent) + 1, dtype=np.int32)
    remap[1:] = np.searchsorted(unique, roots) + 1
    return remap[labels], int(len(unique))


def _smooth_bands(values, *, height=24, band_width=20, palette_values=None):
    """Make labels with hard bands over a genuinely smooth source ramp."""
    width = len(values) * band_width
    x = np.linspace(float(values[0]), float(values[-1]), width, dtype=np.float32)
    den = np.repeat(x[None, :, None], height, axis=0)
    den = np.repeat(den, 3, axis=2)
    labels = np.repeat(np.arange(len(values), dtype=np.int32), band_width)
    labels = np.repeat(labels[None, :], height, axis=0)
    if palette_values is None:
        palette_values = values
    palette = np.asarray([[value, value, value] for value in palette_values],
                         dtype=np.float32)
    return den, labels, np.ones((height, width), dtype=bool), palette


class GradientCandidateGroupsTests(unittest.TestCase):
    def test_real_source_ramp_survives_a_single_quantized_palette_label(self):
        values = np.linspace(90, 110, 96, dtype=np.float32)
        rgb = np.broadcast_to(values[None, :, None], (40, 96, 3)).copy()
        labels = np.zeros((40, 96), dtype=np.int32)
        visible = np.ones(labels.shape, dtype=bool)
        result = propose_gradient_candidates(rgb, labels, visible, np.array([[100, 100, 100]]))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['kind'], 'source_single_component')
        np.testing.assert_array_equal(result[0]['mask'], visible)
        self.assertTrue(result[0]['evidence']['proposal_only_requires_heldout_paint_fit'])

    def test_a_solid_interior_with_antialias_fringe_does_not_invent_gradient(self):
        rgb = np.full((40, 96, 3), (30, 80, 50), dtype=np.float32)
        rgb[:3] = 245
        rgb[-3:] = 245
        rgb[:, :3] = 245
        rgb[:, -3:] = 245
        labels = np.zeros((40, 96), dtype=np.int32)
        result = propose_gradient_candidates(rgb, labels, np.ones(labels.shape, bool),
                                             np.array([[30, 80, 50]]))
        self.assertEqual(result, [])

    def test_interval_sweep_preserves_legacy_labels_and_union_order(self):
        bridge = np.zeros((9, 17), dtype=bool)
        bridge[1, 1:5] = True
        bridge[1, 8:12] = True
        bridge[2, 3:10] = True
        bridge[4, 2:15:2] = True
        bridge[5, 1:16] = True
        checker = (np.indices((13, 19)).sum(axis=0) % 2) == 0
        random_mask = np.random.default_rng(20260719).random((31, 37)) > 0.64

        for mask in (bridge, checker, random_mask):
            with self.subTest(shape=mask.shape):
                expected_labels, expected_count = (
                    _legacy_connected_components_4(mask))
                labels, count = _connected_components_4(mask)
                self.assertEqual(count, expected_count)
                np.testing.assert_array_equal(labels, expected_labels)

    def test_diagonal_palette_islands_remain_separate_vector_objects(self):
        den = np.zeros((12, 12, 3), dtype=np.float32)
        labels = np.zeros((12, 12), dtype=np.int32)
        visible = np.zeros((12, 12), dtype=bool)
        visible[2:5, 2:5] = True
        visible[5:8, 5:8] = True
        den[visible] = (30, 120, 60)
        candidates = propose_gradient_candidates(
            den, labels, visible,
            np.asarray([[30, 120, 60]], dtype=np.float32),
            min_component_area=1, min_candidate_area=1,
            min_shared_boundary=1, min_chromatic_area=1,
        )
        # One palette colour cannot form a band-pair proposal, while strict-4
        # source ownership must also keep the two diagonal islands separate.
        self.assertEqual(candidates, [])

    def test_three_band_smooth_ramp_proposes_whole_chain(self):
        den, labels, visible, palette = _smooth_bands([40, 100, 170])
        candidates = propose_gradient_candidates(
            den, labels, visible, palette,
            min_component_area=8, min_candidate_area=32,
            min_shared_boundary=6, min_chromatic_area=10_000)
        whole = [candidate for candidate in candidates
                 if len(candidate["component_ids"]) == 3]
        self.assertTrue(whole)
        self.assertIn(whole[0]["kind"], {
            "smooth_field", "community", "monotonic_chain"})
        self.assertEqual(whole[0]["area"], visible.size)
        self.assertEqual(whole[0]["mask"].dtype, np.bool_)
        self.assertTrue(whole[0]["evidence"][
            "ownership_closed_over_smooth_graph"])
        partial = [candidate for candidate in candidates
                   if len(candidate["component_ids"]) == 2]
        self.assertTrue(partial)
        self.assertTrue(all(candidate["evidence"]["external_smooth_edges"] > 0
                            for candidate in partial))
        context = whole[0]["_component_context"]
        self.assertEqual(context["component_map"].shape, visible.shape)
        self.assertEqual(len(context["eligible_edges"]), 2)
        self.assertIs(context, partial[0]["_component_context"])

    def test_hard_edge_is_not_a_gradient_candidate(self):
        height, width = 24, 48
        den = np.zeros((height, width, 3), dtype=np.float32)
        den[:, :24] = 35
        den[:, 24:] = 220
        labels = np.zeros((height, width), dtype=np.int32)
        labels[:, 24:] = 1
        visible = np.ones((height, width), dtype=bool)
        palette = np.asarray([[35, 35, 35], [220, 220, 220]],
                             dtype=np.float32)
        candidates = propose_gradient_candidates(
            den, labels, visible, palette,
            min_component_area=8, min_candidate_area=32,
            min_shared_boundary=6, min_chromatic_area=10_000)
        self.assertEqual(candidates, [])

    def test_transitive_bridge_does_not_force_two_ramps_into_one(self):
        # The source is smooth across all five boundaries, deliberately making
        # the middle edge graph-eligible.  Palette progression reverses at that
        # bridge, so a raw union-find would incorrectly return all six bands.
        palette_values = [30, 80, 130, 35, 85, 135]
        den, labels, visible, palette = _smooth_bands(
            [30, 135, 135, 135, 135, 135],
            palette_values=palette_values)
        candidates = propose_gradient_candidates(
            den, labels, visible, palette,
            min_component_area=8, min_candidate_area=32,
            min_shared_boundary=6, min_chromatic_area=10_000,
            mutual_support=0.20)
        component_sets = {candidate["component_ids"] for candidate in candidates}
        self.assertNotIn((1, 2, 3, 4, 5, 6), component_sets)
        self.assertIn((1, 2, 3), component_sets)
        self.assertIn((4, 5, 6), component_sets)

    def test_order_and_metadata_are_deterministic(self):
        den, labels, visible, palette = _smooth_bands([25, 75, 140, 205])
        kwargs = dict(min_component_area=8, min_candidate_area=32,
                      min_shared_boundary=6, min_chromatic_area=10_000)
        first = propose_gradient_candidates(den, labels, visible, palette,
                                            **kwargs)
        second = propose_gradient_candidates(den, labels, visible, palette,
                                             **kwargs)

        def summary(items):
            return [(item["candidate_id"], item["kind"],
                     item["component_ids"], item["bbox"], item["area"],
                     item["score"], item["parent_ids"], item["overlap_info"])
                    for item in items]

        self.assertEqual(summary(first), summary(second))
        self.assertEqual(len({item["candidate_id"] for item in first}),
                         len(first))
        for item in first:
            self.assertEqual(item["area"], int(item["mask"].sum()))
            self.assertIn("smooth_fraction", item["evidence"])


if __name__ == "__main__":
    unittest.main()
