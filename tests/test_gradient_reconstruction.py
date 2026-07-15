"""Safety regressions for gradient reconstruction and preview painting."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import clean_base
from vector_cleanroom import _paint_gradients


class GradientReconstructionTests(unittest.TestCase):
    @staticmethod
    def _chebyshev(a, b):
        return max(abs(int(x) - int(y)) for x, y in zip(a, b))

    def test_gradient_keys_are_isolated_from_palette_and_each_other(self):
        palette = np.asarray([
            (241, 3, 247), (239, 3, 243), (250, 15, 230),
            (0, 0, 0), (255, 255, 255),
        ], dtype=np.uint8)

        keys = clean_base._allocate_gradient_keys(palette, 5)
        again = clean_base._allocate_gradient_keys(palette, 5)

        self.assertEqual(keys, again)
        self.assertEqual(len(keys), 5)
        self.assertEqual(len(set(keys)), 5)
        for key in keys:
            self.assertGreaterEqual(
                min(self._chebyshev(key, colour) for colour in palette), 48)
        for i in range(len(keys)):
            for j in range(i):
                self.assertGreaterEqual(
                    self._chebyshev(keys[i], keys[j]), 48)

    def test_paint_gradients_assigns_close_legacy_keys_to_own_roi(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "keys.png"
            pixels = np.full((8, 20, 3), 255, dtype=np.uint8)
            pixels[2:6, 2:6] = (241, 3, 247)
            pixels[2:6, 14:18] = (239, 3, 243)
            Image.fromarray(pixels, "RGB").save(path)
            info = [
                {
                    "key": "#f103f7", "viewbox": [20, 8],
                    "x1": 0.0, "y1": 0.0, "x2": 19.0, "y2": 0.0,
                    "stops": [
                        {"offset": 0.0, "color": "#dc1e1e"},
                        {"offset": 1.0, "color": "#dc1e1e"},
                    ],
                },
                {
                    "key": "#ef03f3", "viewbox": [20, 8],
                    "x1": 0.0, "y1": 0.0, "x2": 19.0, "y2": 0.0,
                    "stops": [
                        {"offset": 0.0, "color": "#1446dc"},
                        {"offset": 1.0, "color": "#1446dc"},
                    ],
                },
            ]

            _paint_gradients(path, info)
            out = np.asarray(Image.open(path).convert("RGB"))

            np.testing.assert_array_equal(
                out[2:6, 2:6],
                np.full((4, 4, 3), (220, 30, 30), dtype=np.uint8))
            np.testing.assert_array_equal(
                out[2:6, 14:18],
                np.full((4, 4, 3), (20, 70, 220), dtype=np.uint8))
            np.testing.assert_array_equal(
                out[:, 8:12], np.full((8, 4, 3), 255, dtype=np.uint8))

    def test_paint_gradients_recovers_fully_antialiased_key_sliver(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "thin-key.png"
            pixels = np.full((8, 8, 3), 255, dtype=np.uint8)
            # No pixel is within the former 10-level seed tolerance.  This is
            # how a sub-pixel placeholder path appears after rasterisation.
            pixels[3:5, 3:5] = (231, 23, 227)
            Image.fromarray(pixels, "RGB").save(path)
            info = [{
                "key": "#f103f7", "viewbox": [8, 8],
                "x1": 0.0, "y1": 0.0, "x2": 7.0, "y2": 0.0,
                "stops": [
                    {"offset": 0.0, "color": "#287850"},
                    {"offset": 1.0, "color": "#287850"},
                ],
            }]

            _paint_gradients(path, info)
            out = np.asarray(Image.open(path).convert("RGB"))

            np.testing.assert_array_equal(
                out[3:5, 3:5],
                np.full((2, 2, 3), (40, 120, 80), dtype=np.uint8))

    def test_antialiased_key_flood_follows_a_long_connected_sliver(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "long-thin-key.png"
            pixels = np.full((8, 32, 3), 255, dtype=np.uint8)
            # Starts close enough to seed, then becomes progressively more
            # antialiased for far more than the former three growth steps.
            for x in range(2, 30):
                delta = min(90, 20 + 3 * (x - 2))
                pixels[4, x] = (241 - delta, 3 + delta, 247 - delta)
            Image.fromarray(pixels, "RGB").save(path)
            info = [{
                "key": "#f103f7", "key_distance": 140,
                "viewbox": [32, 8],
                "x1": 0.0, "y1": 0.0, "x2": 31.0, "y2": 0.0,
                "stops": [
                    {"offset": 0.0, "color": "#287850"},
                    {"offset": 1.0, "color": "#287850"},
                ],
            }]

            _paint_gradients(path, info)
            out = np.asarray(Image.open(path).convert("RGB"))

            np.testing.assert_array_equal(
                out[4, 2:30],
                np.full((28, 3), (40, 120, 80), dtype=np.uint8))

    def test_gradient_flood_stops_before_nearest_real_palette_colour(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "guarded-key.png"
            pixels = np.full((8, 12, 3), 255, dtype=np.uint8)
            pixels[3:5, 2:5] = (241, 3, 247)
            nearest_real = (193, 51, 199)  # exactly 48 Chebyshev levels away
            pixels[3:5, 5:10] = nearest_real
            Image.fromarray(pixels, "RGB").save(path)
            info = [{
                "key": "#f103f7", "key_distance": 48,
                "viewbox": [12, 8],
                "x1": 0.0, "y1": 0.0, "x2": 11.0, "y2": 0.0,
                "stops": [
                    {"offset": 0.0, "color": "#287850"},
                    {"offset": 1.0, "color": "#287850"},
                ],
            }]

            _paint_gradients(path, info)
            out = np.asarray(Image.open(path).convert("RGB"))

            np.testing.assert_array_equal(
                out[3:5, 5:10],
                np.full((2, 5, 3), nearest_real, dtype=np.uint8))

    def test_well_isolated_key_recovers_seedless_antialiased_island(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "seedless-key.png"
            pixels = np.full((8, 12, 3), 255, dtype=np.uint8)
            # Sixty levels from the placeholder: deliberately no <=47 seed.
            pixels[3:5, 4:8] = (181, 63, 187)
            Image.fromarray(pixels, "RGB").save(path)
            info = [{
                "key": "#f103f7", "key_distance": 140,
                "viewbox": [12, 8],
                "x1": 0.0, "y1": 0.0, "x2": 11.0, "y2": 0.0,
                "stops": [
                    {"offset": 0.0, "color": "#287850"},
                    {"offset": 1.0, "color": "#287850"},
                ],
            }]

            _paint_gradients(path, info)
            out = np.asarray(Image.open(path).convert("RGB"))

            np.testing.assert_array_equal(
                out[3:5, 4:8],
                np.full((2, 4, 3), (40, 120, 80), dtype=np.uint8))

    def test_detect_gradients_rejects_ramp_worse_than_flat_palette(self):
        height = width = 64
        x = np.arange(width)
        values = np.where(
            x < 30, 50,
            np.where(x < 34, 50 + (x - 29) * 15, 130),
        ).astype(np.float32)
        den = np.repeat(np.repeat(values[None, :, None], height, axis=0),
                        3, axis=2)
        labels = np.zeros((height, width), dtype=np.int32)
        labels[:, 32:] = 1
        palette = np.asarray(((50, 50, 50), (130, 130, 130)),
                             dtype=np.uint8)

        regions = clean_base._detect_gradients(
            den, labels, np.ones((height, width), dtype=bool), palette)

        self.assertEqual(regions, [])

    def test_detect_gradients_keeps_true_linear_ramp(self):
        height = width = 64
        values = (40.0 + 140.0 * np.arange(width) / (width - 1)).astype(
            np.float32)
        den = np.repeat(np.repeat(values[None, :, None], height, axis=0),
                        3, axis=2)
        labels = np.zeros((height, width), dtype=np.int32)
        labels[:, 32:] = 1
        palette = np.asarray(((75, 75, 75), (145, 145, 145)),
                             dtype=np.uint8)

        regions = clean_base._detect_gradients(
            den, labels, np.ones((height, width), dtype=bool), palette)

        self.assertEqual(len(regions), 1)
        audit = regions[0]["validation"]
        self.assertLess(audit["gradient_mean_error"],
                        audit["flat_mean_error"])
        self.assertLessEqual(audit["degraded_over_3_share"], 0.10)

    def test_detect_gradients_keeps_safe_low_span_ramp(self):
        height = width = 64
        values = (80.0 + 35.0 * np.arange(width) / (width - 1)).astype(
            np.float32)
        den = np.repeat(np.repeat(values[None, :, None], height, axis=0),
                        3, axis=2)
        labels = np.zeros((height, width), dtype=np.int32)
        labels[:, 32:] = 1
        palette = np.asarray(((88, 88, 88), (107, 107, 107)),
                             dtype=np.uint8)

        regions = clean_base._detect_gradients(
            den, labels, np.ones((height, width), dtype=bool), palette)

        self.assertEqual(len(regions), 1)
        audit = regions[0]["validation"]
        self.assertGreaterEqual(
            audit["flat_mean_error"] - audit["gradient_mean_error"], 2.0)
        self.assertLessEqual(
            audit["gradient_p90_error"], audit["flat_p90_error"])
        self.assertLessEqual(audit["degraded_over_3_share"], 0.10)


if __name__ == "__main__":
    unittest.main()
