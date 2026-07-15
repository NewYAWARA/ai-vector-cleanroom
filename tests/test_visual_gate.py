import unittest

from vector_cleanroom import _evaluate_visual_gate


def _scores(fg, color, p10, mean, topology=100.0,
            cells=50, components=20, light_coverage=None):
    scores = {
        "foreground": fg,
        "foreground_color_fidelity": color,
        "detail_grid": {
            "eligible_cells": cells,
            "p10_score_percent": p10,
            "mean_score_percent": mean,
            "component_topology": {
                "eligible_components": components,
                "p10_score_percent": topology,
            },
        },
    }
    if light_coverage is not None:
        scores["transparent_light_fidelity"] = {
            "applicable": True,
            "source_pixels": 500,
            "coverage_percent": light_coverage,
        }
    return scores


class VisualGateTests(unittest.TestCase):
    def test_high_quality_dark_wordmark_is_accepted(self):
        gate = _evaluate_visual_gate(
            _scores(96.900, 91.030, 93.387, 97.012, 99.9))
        self.assertEqual(gate["status"], "accepted")
        self.assertEqual(gate["acceptance_breaches"], [])

    def test_visibly_damaged_ali_is_rejected_by_multiple_axes(self):
        gate = _evaluate_visual_gate(
            _scores(91.777, 84.906, 74.001, 87.094, 99.977,
                    cells=304, components=53))
        self.assertEqual(gate["status"], "rejected")
        self.assertIn("color_fidelity", gate["soft_breaches"])
        self.assertIn("detail_p10", gate["soft_breaches"])
        self.assertIn("detail_mean", gate["soft_breaches"])

    def test_one_borderline_axis_requires_review_not_rejection(self):
        gate = _evaluate_visual_gate(
            _scores(90.0, 88.0, 78.0, 91.0, 99.0))
        self.assertEqual(gate["status"], "manual_review")
        self.assertEqual(gate["soft_breaches"], [])

    def test_low_tail_and_low_local_mean_reject_together(self):
        gate = _evaluate_visual_gate(
            _scores(92.0, 87.0, 76.0, 87.0, 99.0))
        self.assertEqual(gate["status"], "rejected")
        self.assertTrue(gate["compound_local_failure"])
        self.assertEqual(gate["soft_breaches"], ["detail_mean"])

    def test_catastrophic_colour_is_rejected_alone(self):
        gate = _evaluate_visual_gate(
            _scores(91.0, 65.0, 90.0, 94.0, 99.0))
        self.assertEqual(gate["status"], "rejected")
        self.assertEqual(gate["catastrophic_breaches"], ["color_fidelity"])

    def test_sparse_line_does_not_invent_local_or_topology_failures(self):
        gate = _evaluate_visual_gate({
            "foreground": 90.0,
            "foreground_color_fidelity": 90.0,
            "detail_grid": {
                "eligible_cells": 2,
                "p10_score_percent": 30.0,
                "mean_score_percent": 40.0,
                "component_topology": {
                    "eligible_components": 1,
                    "p10_score_percent": 20.0,
                },
            },
        })
        self.assertEqual(gate["status"], "accepted")
        self.assertFalse(gate["applicability"]["local_detail"])
        self.assertFalse(gate["applicability"]["component_topology"])

    def test_transparent_white_objects_missing_on_colour_are_rejected(self):
        gate = _evaluate_visual_gate(
            _scores(96.0, 91.0, 92.0, 95.0, 99.0,
                    light_coverage=84.4))
        self.assertEqual(gate["status"], "rejected")
        self.assertIn("light_object_coverage", gate["catastrophic_breaches"])
        self.assertTrue(
            gate["applicability"]["transparent_light_objects"])

    def test_sparse_light_pixels_do_not_activate_transparency_gate(self):
        scores = _scores(96.0, 91.0, 92.0, 95.0, 99.0)
        scores["transparent_light_fidelity"] = {
            "applicable": True,
            "source_pixels": 20,
            "coverage_percent": 10.0,
        }
        gate = _evaluate_visual_gate(scores)
        self.assertEqual(gate["status"], "accepted")
        self.assertFalse(
            gate["applicability"]["transparent_light_objects"])


if __name__ == "__main__":
    unittest.main()
