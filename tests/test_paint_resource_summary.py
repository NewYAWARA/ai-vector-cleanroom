"""Focused contracts for designer-facing paint resource reporting."""

from types import SimpleNamespace
import unittest

from vector_cleanroom import _paint_resource_summary


class PaintResourceSummaryTests(unittest.TestCase):
    def test_reused_gradient_layer_is_not_reported_as_a_solid_paint(self):
        stats = SimpleNamespace(
            palette=[
                ("gradient", "#334455"),
                ("blue", "#112233"),
                ("gradient2", "#8899aa"),
                # The same grad1 resource appears in a later stack run.
                ("gradient3", "#334455"),
                ("blue2", "#112233"),
            ],
            gradient_info=[
                {
                    "id": "grad1",
                    "stops": [
                        {"offset": 0.0, "color": "#102030"},
                        {"offset": 0.5, "color": "#334455"},
                        {"offset": 1.0, "color": "#506070"},
                    ],
                },
                {
                    "id": "grad2",
                    "stops": [
                        {"offset": 0.0, "color": "#667788"},
                        {"offset": 0.5, "color": "#8899aa"},
                        {"offset": 1.0, "color": "#aabbcc"},
                    ],
                },
            ],
            stroke_info=[
                {"color": "#abcdef"},
                {"color": "#112233"},
            ],
        )

        summary = _paint_resource_summary(stats)

        self.assertEqual(summary["solid_paints"], 2)
        self.assertEqual(summary["gradient_paints"], 2)
        self.assertEqual(summary["unique_paints_total"], 4)
        self.assertEqual(len(summary["palette"]), summary["solid_paints"])
        self.assertEqual(
            len(summary["paint_resources"]),
            summary["unique_paints_total"],
        )
        self.assertEqual(
            summary["solid_paints"] + summary["gradient_paints"],
            summary["unique_paints_total"],
        )
        self.assertEqual(
            {item["hex"] for item in summary["palette"]},
            {"#112233", "#abcdef"},
        )
        gradient_layers = [
            item for item in summary["layers"]
            if item["type"] == "linearGradient"
        ]
        self.assertEqual(
            [item["gradient_id"] for item in gradient_layers],
            ["grad1", "grad2", "grad1"],
        )
        self.assertNotIn(
            "#334455",
            {item["hex"] for item in summary["palette"]},
        )


if __name__ == "__main__":
    unittest.main()
