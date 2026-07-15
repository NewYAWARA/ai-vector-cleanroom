"""Regression tests for source-ink local-detail diagnostics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from quality_diagnostics import (
    _component_topology,
    compute_quality_diagnostics,
    source_ink_roi,
)


class QualityDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _save(self, image, name):
        path = self.tmp / name
        image.save(path)
        return path

    @staticmethod
    def _line_logo(size=(256, 256), offset=(0, 0)):
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)
        ox, oy = offset
        draw.line((96 + ox, 128 + oy, 160 + ox, 128 + oy), fill="black", width=1)
        draw.ellipse((126 + ox, 119 + oy, 130 + ox, 123 + oy), fill="black")
        return image

    def test_opaque_white_background_is_not_foreground(self):
        source = self._line_logo()
        roi = source_ink_roi(source)

        self.assertLess(roi["source_ink_fraction"], 0.002)
        self.assertGreater(roi["source_ink_pixels"], 60)
        self.assertLess(roi["source_ink_pixels"], 100)

    def test_large_white_canvas_cannot_dilute_missing_thin_detail(self):
        source = Image.new("RGB", (1024, 1024), "white")
        draw = ImageDraw.Draw(source)
        draw.line((480, 512, 544, 512), fill="black", width=1)
        draw.ellipse((510, 500, 514, 504), fill="black")
        render = Image.new("RGB", source.size, "white")

        result = compute_quality_diagnostics(render, source, cell=48)
        grid = result["detail_grid"]

        self.assertGreater(grid["eligible_cells"], 0)
        self.assertLessEqual(grid["eligible_cells"], 4)
        self.assertEqual(grid["worst_score_percent"], 0.0)
        self.assertEqual(grid["median_score_percent"], 0.0)
        self.assertTrue(result["hotspots"])
        self.assertGreaterEqual(result["hotspots"][0]["severity"], 0.99)

    def test_correct_copy_scores_high(self):
        source = self._line_logo()
        source_path = self._save(source, "source.png")
        render_path = self._save(source.copy(), "render.png")

        result = compute_quality_diagnostics(
            render_path, source_path, viewbox=[512, 512], cell=48
        )
        grid = result["detail_grid"]

        self.assertGreater(grid["eligible_cells"], 0)
        self.assertGreaterEqual(grid["worst_score_percent"], 99.0)
        self.assertGreaterEqual(grid["p10_score_percent"], 99.0)
        self.assertEqual(result["hotspots"], [])

    def test_one_pixel_shift_is_tolerated(self):
        source = self._line_logo()
        shifted = self._line_logo(offset=(1, 0))

        result = compute_quality_diagnostics(shifted, source, cell=48)
        grid = result["detail_grid"]

        self.assertGreaterEqual(grid["worst_score_percent"], 98.0)
        self.assertGreaterEqual(grid["median_score_percent"], 99.0)
        self.assertEqual(result["hotspots"], [])

    def test_missing_small_dot_is_materially_worse_than_correct_copy(self):
        source = self._line_logo()
        correct = compute_quality_diagnostics(source.copy(), source, cell=24)

        without_dot = Image.new("RGB", source.size, "white")
        ImageDraw.Draw(without_dot).line((96, 128, 160, 128), fill="black", width=1)
        missing = compute_quality_diagnostics(without_dot, source, cell=24)

        correct_worst = correct["detail_grid"]["worst_score_percent"]
        missing_worst = missing["detail_grid"]["worst_score_percent"]
        self.assertGreaterEqual(correct_worst, 99.0)
        self.assertLess(missing_worst, correct_worst - 20.0)
        self.assertTrue(missing["hotspots"])

    @staticmethod
    def _component_logo(*, split=False, shift=0):
        image = Image.new("RGB", (280, 260), "white")
        draw = ImageDraw.Draw(image)
        for row in range(10):
            y = 16 + row * 23
            if split:
                draw.rectangle((40 + shift, y, 92 + shift, y + 3), fill="black")
                draw.rectangle((108 + shift, y, 160 + shift, y + 3), fill="black")
            else:
                draw.rectangle((40 + shift, y, 160 + shift, y + 3), fill="black")
        return image

    def test_component_topology_scores_intact_components_high(self):
        source = self._component_logo()
        topology = compute_quality_diagnostics(
            source.copy(), source)["detail_grid"]["component_topology"]

        self.assertEqual(topology["eligible_components"], 10)
        self.assertGreaterEqual(topology["p10_score_percent"], 99.0)
        self.assertEqual(topology["fragmented_components"], 0)

    def test_component_topology_detects_split_glyph_stems(self):
        source = self._component_logo()
        split = self._component_logo(split=True)
        topology = compute_quality_diagnostics(
            split, source, viewbox=[10, 20, 560, 520]
        )["detail_grid"]["component_topology"]

        self.assertEqual(topology["eligible_components"], 10)
        self.assertLess(topology["p10_score_percent"], 60.0)
        self.assertEqual(topology["fragmented_components"], 10)
        self.assertEqual(
            topology["schema"],
            "ai-vector-cleanroom.component-topology/v1")
        self.assertEqual(topology["connectivity"], "8-connected")
        self.assertEqual(topology["render_tolerance"], {
            "pixels": 1,
            "neighbourhood": "3x3",
            "operation": "dilation_before_component_labelling",
        })
        self.assertEqual(topology["measurement_size_px"], [280, 260])
        self.assertEqual(topology["viewbox"], [10.0, 20.0, 560.0, 520.0])
        self.assertEqual(topology["failure_score_below"], 90.0)
        self.assertEqual(topology["failed_component_count"], 10)
        self.assertEqual(topology["examples_total"], 10)
        self.assertEqual(topology["examples_returned"], 10)
        self.assertFalse(topology["examples_truncated"])
        self.assertEqual(
            topology["example_sort"], "score_percent_asc_area_px_desc")
        self.assertEqual(len(topology["failed_examples"]), 10)

        example = topology["examples"][0]
        self.assertEqual(example["failure_score_below"], 90.0)
        self.assertTrue(example["below_failure_threshold"])
        self.assertTrue(example["fragmented"])
        self.assertEqual(example["bbox_px_format"], "xywh")
        self.assertEqual(example["bbox_viewbox_format"], "xywh")
        x, y, width, height = example["bbox_px"]
        self.assertEqual(
            example["bbox_viewbox"],
            [10.0 + 2.0 * x, 20.0 + 2.0 * y,
             2.0 * width, 2.0 * height],
        )

    def test_component_topology_keeps_complete_failures_when_examples_truncate(self):
        source = np.zeros((200, 200), dtype=bool)
        for row in range(5):
            for column in range(5):
                y = 5 + row * 35
                x = 5 + column * 35
                source[y:y + 4, x:x + 4] = True
        render = np.zeros_like(source)

        topology = _component_topology(
            source, render, max_examples=3, viewbox=[0, 0, 400, 400])

        self.assertEqual(topology["eligible_components"], 25)
        self.assertEqual(topology["failed_component_count"], 25)
        self.assertEqual(topology["examples_total"], 25)
        self.assertEqual(topology["examples_returned"], 3)
        self.assertTrue(topology["examples_truncated"])
        self.assertEqual(len(topology["examples"]), 3)
        self.assertEqual(len(topology["failed_examples"]), 25)
        self.assertTrue(all(
            item["below_failure_threshold"]
            for item in topology["failed_examples"]))
        self.assertEqual(
            topology["failed_examples"][0]["bbox_viewbox"],
            [10.0, 10.0, 8.0, 8.0],
        )

    def test_component_topology_empty_schema_is_complete(self):
        empty = np.zeros((24, 32), dtype=bool)

        topology = _component_topology(
            empty, empty, viewbox=[4, 6, 64, 48])

        self.assertEqual(topology["measurement_size_px"], [32, 24])
        self.assertEqual(topology["viewbox"], [4.0, 6.0, 64.0, 48.0])
        self.assertTrue(topology["one_pixel_tolerance"])
        self.assertEqual(topology["failed_component_count"], 0)
        self.assertEqual(topology["examples_total"], 0)
        self.assertEqual(topology["examples_returned"], 0)
        self.assertFalse(topology["examples_truncated"])
        self.assertEqual(topology["examples"], [])
        self.assertEqual(topology["failed_examples"], [])

    def test_component_topology_tolerates_one_pixel_shift(self):
        source = self._component_logo()
        shifted = self._component_logo(shift=1)
        topology = compute_quality_diagnostics(
            shifted, source)["detail_grid"]["component_topology"]

        self.assertGreaterEqual(topology["p10_score_percent"], 99.0)

    def test_light_antialias_halo_is_not_a_topology_component(self):
        source = Image.new("RGB", (220, 160), "white")
        draw = ImageDraw.Draw(source)
        draw.line((40, 70, 180, 70), fill="black", width=1)
        draw.line((40, 72, 180, 72), fill=(249, 249, 249), width=1)
        render = Image.new("RGB", source.size, "white")
        ImageDraw.Draw(render).line((40, 70, 180, 70), fill="black", width=1)

        result = compute_quality_diagnostics(render, source, cell=48)
        topology = result["detail_grid"]["component_topology"]

        self.assertEqual(topology["measurement_mask"], "strong_ink_core")
        self.assertEqual(topology["eligible_components"], 1)
        self.assertGreaterEqual(topology["p10_score_percent"], 99.0)
        self.assertGreater(topology["low_contrast_excluded_pixels"], 100)
        self.assertLess(result["detail_grid"]["worst_score_percent"], 90.0)

    def test_near_white_glyph_modelling_bands_are_not_structural_components(self):
        source = Image.new("RGB", (260, 180), "white")
        draw = ImageDraw.Draw(source)
        draw.rectangle((35, 40, 80, 135), fill="black")
        # These broad bands are large enough to pass the component-area guard,
        # but at only 13 RGB levels from white they are modelling/antialias
        # detail, not glyph stems whose continuity should reject an SVG.
        for y in (52, 78, 104):
            draw.rectangle((125, y, 215, y + 3), fill=(242, 242, 242))
        render = Image.new("RGB", source.size, "white")
        ImageDraw.Draw(render).rectangle((35, 40, 80, 135), fill="black")

        result = compute_quality_diagnostics(render, source, cell=48)
        topology = result["detail_grid"]["component_topology"]

        self.assertEqual(topology["core_threshold"], 18.0)
        self.assertEqual(topology["eligible_components"], 1)
        self.assertGreaterEqual(topology["p10_score_percent"], 99.0)
        self.assertGreater(topology["low_contrast_excluded_pixels"], 1000)
        self.assertLess(result["detail_grid"]["worst_score_percent"], 90.0)

    def test_medium_gray_missing_line_remains_a_topology_failure(self):
        source = Image.new("RGB", (220, 140), "white")
        ImageDraw.Draw(source).line(
            (30, 70, 190, 70), fill=(230, 230, 230), width=3)
        render = Image.new("RGB", source.size, "white")

        topology = compute_quality_diagnostics(
            render, source)["detail_grid"]["component_topology"]

        self.assertEqual(topology["core_threshold"], 18.0)
        self.assertEqual(topology["eligible_components"], 1)
        self.assertEqual(topology["p10_score_percent"], 0.0)

    def test_light_bridge_does_not_fuse_two_dark_components(self):
        source = Image.new("RGB", (220, 140), "white")
        draw = ImageDraw.Draw(source)
        draw.rectangle((30, 45, 75, 90), fill="black")
        draw.rectangle((145, 45, 190, 90), fill="black")
        draw.line((76, 67, 144, 67), fill=(249, 249, 249), width=1)
        render = Image.new("RGB", source.size, "white")
        rdraw = ImageDraw.Draw(render)
        rdraw.rectangle((30, 45, 75, 90), fill="black")
        rdraw.rectangle((145, 45, 190, 90), fill="black")

        topology = compute_quality_diagnostics(
            render, source)["detail_grid"]["component_topology"]

        self.assertEqual(topology["eligible_components"], 2)
        self.assertGreaterEqual(topology["p10_score_percent"], 99.0)
        self.assertEqual(topology["fragmented_components"], 0)

    def test_missing_one_pixel_black_line_still_fails_topology(self):
        source = Image.new("RGB", (220, 140), "white")
        ImageDraw.Draw(source).line((30, 70, 190, 70), fill="black", width=1)
        render = Image.new("RGB", source.size, "white")

        topology = compute_quality_diagnostics(
            render, source)["detail_grid"]["component_topology"]

        self.assertEqual(topology["eligible_components"], 1)
        self.assertEqual(topology["p10_score_percent"], 0.0)
        self.assertEqual(topology["coverage_p10_percent"], 0.0)

    def test_missing_dark_bridge_still_reports_true_fragmentation(self):
        source = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(source)
        draw.rectangle((35, 50, 85, 110), fill="black")
        draw.rectangle((155, 50, 205, 110), fill="black")
        draw.rectangle((86, 77, 154, 83), fill="black")
        render = Image.new("RGB", source.size, "white")
        rdraw = ImageDraw.Draw(render)
        rdraw.rectangle((35, 50, 85, 110), fill="black")
        rdraw.rectangle((155, 50, 205, 110), fill="black")

        topology = compute_quality_diagnostics(
            render, source)["detail_grid"]["component_topology"]

        self.assertEqual(topology["eligible_components"], 1)
        self.assertLess(topology["p10_score_percent"], 60.0)
        self.assertLess(topology["connectivity_p10_percent"], 60.0)
        self.assertEqual(topology["fragmented_components"], 1)


if __name__ == "__main__":
    unittest.main()
