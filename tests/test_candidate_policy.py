"""Candidate fallback policy must protect requested editing features."""

from __future__ import annotations

import unittest

from vector_cleanroom import (_rolled_back_stage_report,
                              _select_viable_candidate)


REQUESTED = {
    "strokes": "on",
    "gradients": "on",
    "geometry": "conservative",
    "background": "auto",
}


def _item(name, quality, rank, **overrides):
    options = dict(REQUESTED)
    options.update(overrides)
    return (rank, quality, options, None, {}, {"name": name})


class CandidatePolicyRegression(unittest.TestCase):
    def test_visual_tie_keeps_requested_strokes(self):
        requested = _item("requested", 96.889, 92.022)
        strokes_off = _item("strokes-off", 97.047, 92.239,
                            strokes="off")
        selected, policy = _select_viable_candidate(
            [requested, strokes_off], REQUESTED)
        self.assertEqual(selected[5]["name"], "requested")
        self.assertEqual(policy["selected_requested_features_retained"], 3)
        self.assertEqual(policy["material_visual_gain_required"], 1.0)

    def test_material_visual_gain_can_disable_a_feature(self):
        requested = _item("requested", 88.0, 84.0)
        strokes_off = _item("strokes-off", 90.2, 86.0, strokes="off")
        selected, _policy = _select_viable_candidate(
            [requested, strokes_off], REQUESTED)
        self.assertEqual(selected[5]["name"], "strokes-off")

    def test_failed_requested_build_still_selects_best_viable_tie(self):
        one_feature = _item("one-feature", 94.1, 90.0,
                            strokes="off", geometry="off")
        two_features = _item("two-features", 93.4, 89.0, strokes="off")
        selected, policy = _select_viable_candidate(
            [one_feature, two_features], REQUESTED)
        self.assertEqual(selected[5]["name"], "two-features")
        self.assertEqual(policy["requested_features_total"], 3)

    def test_rank_breaks_tie_after_equal_feature_retention(self):
        lower_rank = _item("lower", 96.8, 91.0, strokes="off")
        higher_rank = _item("higher", 96.7, 92.0, strokes="off")
        selected, _policy = _select_viable_candidate(
            [lower_rank, higher_rank], REQUESTED)
        self.assertEqual(selected[5]["name"], "higher")

    def test_global_rollback_reports_only_committed_structure(self):
        compound = _rolled_back_stage_report("compound_paths", {
            "status": "applied", "input_paths": 4, "output_paths": 7,
            "input_subpaths": 9, "output_subpaths": 9,
            "selectable_path_delta": 3, "source_paths_split": 2,
            "source_paths_simplified": 2,
            "linear_cubics_simplified": 40,
            "path_data_bytes_saved": 1200,
        })
        scene = _rolled_back_stage_report("scene_graph", {
            "status": "applied", "drawable_count": 20,
            "actual_dom_group_count": 3, "manifest_only_group_count": 2,
            "grouped_drawables": 11, "actual_dom_groups": [{"id": "x"}],
        })
        annulus = _rolled_back_stage_report("annulus", {
            "status": "applied", "applied_candidates": 1,
        })
        exact_native = _rolled_back_stage_report("exact_native_shapes", {
            "status": "applied", "committed": True,
            "committed_candidate_count": 8,
            "committed_line_count": 5, "committed_polyline_count": 3,
        })
        self.assertEqual(compound["selectable_path_delta"], 0)
        self.assertEqual(compound["output_paths"], 4)
        self.assertEqual(compound["linear_cubics_simplified"], 0)
        self.assertEqual(compound["path_data_bytes_saved"], 0)
        self.assertEqual(scene["actual_dom_group_count"], 0)
        self.assertEqual(scene["ungrouped_drawables"], 20)
        self.assertEqual(annulus["applied_candidates"], 0)
        self.assertFalse(exact_native["committed"])
        self.assertEqual(exact_native["committed_candidate_count"], 0)
        self.assertEqual(exact_native["committed_line_count"], 0)
        self.assertEqual(exact_native["committed_polyline_count"], 0)
        self.assertEqual(compound["attempted_status"], "applied")


if __name__ == "__main__":
    unittest.main()
