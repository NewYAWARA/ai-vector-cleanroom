"""Regression tests for source-ink local-detail diagnostics."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from quality_diagnostics import compute_quality_diagnostics, source_ink_roi


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


if __name__ == "__main__":
    unittest.main()
