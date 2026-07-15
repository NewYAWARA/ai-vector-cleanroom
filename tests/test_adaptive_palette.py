import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from clean_base import _allow_palette_tier_b_strokes, detect_palette


class AdaptivePaletteTests(unittest.TestCase):
    def test_flat_logo_keeps_a_compact_palette(self):
        image = np.zeros((96, 96, 3), dtype=np.float32)
        image[:48, :48] = (12, 73, 45)
        image[:48, 48:] = (123, 162, 50)
        image[48:, :48] = (236, 196, 20)
        image[48:, 48:] = (116, 190, 220)
        visible = np.ones(image.shape[:2], dtype=bool)

        palette, labels, audit = detect_palette(
            image, visible, return_audit=True)

        self.assertLessEqual(len(palette), 4)
        self.assertEqual(audit["mode"], "compact")
        self.assertEqual(labels.shape, visible.shape)

    def test_multiaxis_ramps_expand_only_as_far_as_needed(self):
        size = 144
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        x = xx / (size - 1)
        y = yy / (size - 1)
        # A two-dimensional colour surface is representative of AI-generated
        # logo texture: eight flat swatches are visibly inadequate, while a
        # bounded 12/16/20-colour palette is a useful tracing compromise.
        image = np.stack([
            12 + 112 * x + 24 * np.sin(2 * np.pi * y),
            58 + 118 * y + 30 * np.sin(np.pi * x),
            24 + 82 * (1 - x) * y + 18 * np.cos(2 * np.pi * x),
        ], axis=2)
        image = np.clip(image, 0, 255).astype(np.float32)
        visible = np.ones((size, size), dtype=bool)

        palette, _, audit = detect_palette(
            image, visible, return_audit=True, return_labels=False)

        self.assertGreater(len(palette), 8)
        self.assertLessEqual(len(palette), 20)
        self.assertEqual(audit["mode"], "adaptive")
        self.assertLess(audit["fit_after"]["mean"],
                        audit["fit_before"]["mean"])
        self.assertLess(audit["fit_after"]["p90"],
                        audit["fit_before"]["p90"])

    def test_adaptive_palette_is_deterministic(self):
        rng = np.random.default_rng(90210)
        image = rng.integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
        visible = np.ones(image.shape[:2], dtype=bool)

        first, _, first_audit = detect_palette(
            image, visible, return_audit=True, return_labels=False)
        second, _, second_audit = detect_palette(
            image, visible, return_audit=True, return_labels=False)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_audit, second_audit)

    def test_forced_palette_still_obeys_the_user(self):
        rng = np.random.default_rng(12)
        image = rng.integers(0, 256, size=(72, 72, 3), dtype=np.uint8)
        visible = np.ones(image.shape[:2], dtype=bool)

        palette, _, audit = detect_palette(
            image, visible, forced=6, return_audit=True,
            return_labels=False)

        self.assertEqual(len(palette), 6)
        self.assertEqual(audit["mode"], "forced")

    def test_nine_real_flat_colours_are_not_forced_into_eight(self):
        colours = np.asarray([
            (10, 10, 10), (245, 10, 10), (10, 245, 10),
            (10, 10, 245), (245, 245, 10), (245, 10, 245),
            (10, 245, 245), (128, 128, 128), (245, 245, 245),
        ], dtype=np.float32)
        image = np.zeros((90, 90, 3), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                image[row * 30:(row + 1) * 30,
                      col * 30:(col + 1) * 30] = colours[row * 3 + col]
        visible = np.ones(image.shape[:2], dtype=bool)

        palette, _, audit = detect_palette(
            image, visible, return_audit=True, return_labels=False)

        self.assertEqual(len(palette), 9)
        self.assertEqual(audit["mode"], "adaptive")

    def test_jpeg_antialiasing_does_not_expand_a_flat_logo(self):
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[:64, :64] = (12, 73, 45)
        image[:64, 64:] = (123, 162, 50)
        image[64:, :64] = (236, 196, 20)
        image[64:, 64:] = (116, 190, 220)
        buffer = BytesIO()
        Image.fromarray(image, "RGB").save(
            buffer, format="JPEG", quality=78, subsampling=2)
        buffer.seek(0)
        jpeg = np.asarray(Image.open(buffer).convert("RGB"))
        visible = np.ones(jpeg.shape[:2], dtype=bool)

        palette, _, audit = detect_palette(
            jpeg, visible, return_audit=True, return_labels=False)

        self.assertLessEqual(len(palette), 8)
        self.assertEqual(audit["mode"], "compact")

    def test_tiny_distinct_accent_colour_survives(self):
        image = np.empty((200, 200, 3), dtype=np.uint8)
        image[:, :100] = (22, 82, 54)
        image[:, 100:] = (116, 165, 67)
        image[97:102, 97:102] = (235, 24, 35)
        visible = np.ones(image.shape[:2], dtype=bool)

        palette, _, _ = detect_palette(
            image, visible, return_audit=True, return_labels=False)

        accent = np.asarray((235, 24, 35), dtype=np.float32)
        self.assertLess(float(np.linalg.norm(
            palette.astype(np.float32) - accent, axis=1).min()), 30.0)

    def test_continuous_tone_palette_disables_fragment_level_strokes(self):
        self.assertFalse(_allow_palette_tier_b_strokes({
            "mode": "adaptive", "selected_colors": 20,
        }))
        self.assertFalse(_allow_palette_tier_b_strokes({
            "mode": "adaptive", "selected_colors": 12,
        }))

    def test_flat_or_user_forced_palette_keeps_tier_b_available(self):
        self.assertTrue(_allow_palette_tier_b_strokes({
            "mode": "compact", "selected_colors": 8,
        }))
        self.assertTrue(_allow_palette_tier_b_strokes({
            "mode": "adaptive", "selected_colors": 9,
        }))
        self.assertTrue(_allow_palette_tier_b_strokes({
            "mode": "forced", "selected_colors": 20,
        }))


if __name__ == "__main__":
    unittest.main()
