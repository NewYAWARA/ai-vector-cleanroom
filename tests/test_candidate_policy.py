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


def _item(name, quality, rank, scores=None, **overrides):
    options = dict(REQUESTED)
    options.update(overrides)
    return (rank, quality, options, None, scores or {}, {"name": name})


class CandidatePolicyRegression(unittest.TestCase):
    @staticmethod
    def _gate_scores(foreground, color, detail_p10, detail_mean,
                     topology_p10):
        return {
            "foreground": foreground,
            "foreground_color_fidelity": color,
            "detail_grid": {
                "eligible_cells": 20,
                "p10_score_percent": detail_p10,
                "mean_score_percent": detail_mean,
                "component_topology": {
                    "eligible_components": 20,
                    "p10_score_percent": topology_p10,
                },
            },
        }

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

    def test_local_detail_can_safely_override_requested_strokes(self):
        # Real tea-logo failure signature: the centre-line reconstruction buys
        # 0.2 foreground points but visibly damages glyphs and the outer arc.
        requested = _item(
            "strokes-on", 90.640, 87.0,
            scores={
                "foreground_color_fidelity": 83.634,
                "detail_grid": {"p10_score_percent": 70.249,
                                "mean_score_percent": 83.201},
            },
        )
        strokes_off = _item(
            "strokes-off", 90.439, 86.8, strokes="off",
            scores={
                "foreground_color_fidelity": 81.217,
                "detail_grid": {"p10_score_percent": 71.907,
                                "mean_score_percent": 85.328},
            },
        )
        selected, policy = _select_viable_candidate(
            [requested, strokes_off], REQUESTED)
        self.assertEqual(selected[5]["name"], "strokes-off")
        self.assertEqual(policy["survivor_count"], 1)
        self.assertEqual(policy["selected_metric_vector"]["detail_p10"],
                         71.907)

    def test_high_quality_dark_wordmark_tie_still_keeps_requested_strokes(self):
        requested = _item(
            "requested", 96.871, 92.006,
            scores={
                "foreground_color_fidelity": 90.966,
                "detail_grid": {"p10_score_percent": 93.387,
                                "mean_score_percent": 97.005},
            },
        )
        strokes_off = _item(
            "strokes-off", 97.047, 92.239, strokes="off",
            scores={
                "foreground_color_fidelity": 91.363,
                "detail_grid": {"p10_score_percent": 93.731,
                                "mean_score_percent": 97.092},
            },
        )
        selected, policy = _select_viable_candidate(
            [requested, strokes_off], REQUESTED)
        self.assertEqual(selected[5]["name"], "requested")
        self.assertEqual(policy["survivor_count"], 2)

    def test_component_continuity_selects_intact_ali_glyphs(self):
        strokes_on = _item(
            "strokes-on", 91.820, 86.85,
            scores={
                "foreground_color_fidelity": 85.663,
                "detail_grid": {
                    "p10_score_percent": 76.319,
                    "mean_score_percent": 86.788,
                    "component_topology": {
                        "eligible_components": 53,
                        "p10_score_percent": 97.522,
                    },
                },
            },
        )
        strokes_off = _item(
            "strokes-off", 91.777, 86.81, strokes="off",
            scores={
                "foreground_color_fidelity": 84.906,
                "detail_grid": {
                    "p10_score_percent": 74.001,
                    "mean_score_percent": 87.094,
                    "component_topology": {
                        "eligible_components": 53,
                        "p10_score_percent": 99.977,
                    },
                },
            },
        )

        selected, policy = _select_viable_candidate(
            [strokes_on, strokes_off], REQUESTED)

        self.assertEqual(selected[5]["name"], "strokes-off")
        self.assertEqual(policy["survivor_count"], 1)
        self.assertEqual(policy["selected_metric_vector"]["topology_p10"],
                         99.977)

    def test_visual_gain_cannot_hide_excess_local_regression(self):
        stable = _item(
            "stable", 90.0, 85.0,
            scores={
                "foreground_color_fidelity": 90.0,
                "detail_grid": {"p10_score_percent": 90.0,
                                "mean_score_percent": 90.0},
            },
        )
        damaged = _item(
            "damaged", 91.2, 86.0, strokes="off",
            scores={
                "foreground_color_fidelity": 90.1,
                "detail_grid": {"p10_score_percent": 87.5,
                                "mean_score_percent": 88.5},
            },
        )
        selected, policy = _select_viable_candidate(
            [stable, damaged], REQUESTED)
        self.assertEqual(selected[5]["name"], "stable")
        self.assertEqual(policy["survivor_count"], 2)

    def test_manual_review_candidate_outranks_rejected_requested_build(self):
        # Real tea-logo opaque/cutout failure: feature retention previously let the
        # rejected base beat the safer strokes-off candidate.
        rejected = _item(
            "requested-rejected", 95.983, 90.0,
            scores=self._gate_scores(95.983, 91.448, 89.047, 92.346,
                                     69.062),
        )
        manual = _item(
            "strokes-off-manual", 95.875, 89.0, strokes="off",
            scores=self._gate_scores(95.875, 90.983, 85.390, 92.532,
                                     87.571),
        )

        selected, policy = _select_viable_candidate(
            [rejected, manual], REQUESTED)

        self.assertEqual(selected[5]["name"], "strokes-off-manual")
        self.assertEqual(policy["best_visual_status"], "manual_review")
        self.assertEqual(policy["selected_visual_status"], "manual_review")
        self.assertEqual(policy["visual_status_counts"]["rejected"], 1)
        self.assertEqual(policy["visual_status_survivor_count"], 1)

    def test_accepted_candidate_outranks_higher_foreground_manual_review(self):
        manual = _item(
            "requested-manual", 99.0, 99.0,
            scores=self._gate_scores(99.0, 84.9, 92.0, 94.0, 96.0),
        )
        accepted = _item(
            "strokes-off-accepted", 96.0, 90.0, strokes="off",
            scores=self._gate_scores(96.0, 90.0, 90.0, 94.0, 96.0),
        )

        selected, policy = _select_viable_candidate(
            [manual, accepted], REQUESTED)

        self.assertEqual(selected[5]["name"], "strokes-off-accepted")
        self.assertEqual(policy["best_visual_status"], "accepted")
        self.assertEqual(policy["selected_visual_status"], "accepted")
        self.assertEqual(
            policy["policy"],
            "visual_gate_tier_then_safe_dominance_then_preserve_features",
        )

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
