"""Safety regressions for conservative stroke reconstruction."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from stroke_engine import extract_strokes


class StrokeSafetyTests(unittest.TestCase):
    @staticmethod
    def _extract(image, palette):
        den = np.asarray(image.convert("RGB"), dtype=np.float32)
        mask = np.abs(den - 255.0).max(axis=2) > 20.0
        strokes, owned, deferred = extract_strokes(
            mask, den, np.asarray(palette, dtype=np.uint8), (255, 255, 255))
        return mask, strokes, owned, deferred

    def test_multicolour_closed_ring_is_deferred_without_gaps(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        dark = (10, 75, 35)
        olive = (165, 165, 20)
        draw.arc((16, 16, 112, 112), 90, 270, fill=dark, width=8)
        draw.arc((16, 16, 112, 112), 270, 450, fill=olive, width=8)

        mask, strokes, owned, deferred = self._extract(
            image, (dark, olive))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        self.assertGreaterEqual(int((deferred & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_multicolour_curved_open_arc_is_deferred_as_one_component(self):
        image = Image.new("RGB", (160, 120), "white")
        draw = ImageDraw.Draw(image)
        dark = (8, 70, 35)
        olive = (170, 165, 15)
        draw.arc((20, 10, 140, 130), 185, 270, fill=dark, width=8)
        draw.arc((20, 10, 140, 130), 270, 355, fill=olive, width=8)

        mask, strokes, owned, deferred = self._extract(
            image, (dark, olive))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        self.assertGreaterEqual(int((deferred & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_bent_glyph_like_junction_is_not_rounded_into_three_arms(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        ink = (12, 70, 42)
        # 山-like topology: the side arms turn 90 degrees before reaching the
        # central junction, unlike a genuine straight T/Y/X line diagram.
        draw.line((20, 35, 20, 102, 108, 102, 108, 35),
                  fill=ink, width=12, joint="curve")
        draw.line((64, 18, 64, 102), fill=ink, width=12)

        mask, strokes, owned, deferred = self._extract(image, (ink,))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        self.assertGreaterEqual(int((deferred & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_straight_two_colour_rule_remains_editable_and_gap_free(self):
        image = Image.new("RGB", (160, 64), "white")
        draw = ImageDraw.Draw(image)
        red = (220, 20, 30)
        blue = (20, 70, 220)
        draw.line((12, 32, 80, 32), fill=red, width=10)
        draw.line((80, 32, 148, 32), fill=blue, width=10)

        mask, strokes, owned, deferred = self._extract(image, (red, blue))

        self.assertGreaterEqual(len(strokes), 2)
        self.assertFalse(deferred.any())
        # Ownership includes the complete source ribbon; shared run endpoints
        # ensure the colour transition is never turned into a white gap.
        self.assertGreaterEqual(int((owned & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_short_curved_glyph_fragment_stays_a_fill(self):
        image = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(image)
        ink = (12, 70, 42)
        draw.arc((28, 28, 68, 68), 35, 155, fill=ink, width=9)

        mask, strokes, owned, deferred = self._extract(image, (ink,))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        # Whether rejected before or at the explicit short-curve guard, all
        # source pixels remain available to the fill tracer.
        self.assertEqual(int((mask & ~owned).sum()), int(mask.sum()))

    def test_long_single_colour_arc_remains_an_editable_stroke(self):
        image = Image.new("RGB", (180, 180), "white")
        draw = ImageDraw.Draw(image)
        ink = (12, 70, 42)
        draw.arc((20, 20, 160, 160), 15, 215, fill=ink, width=8)

        mask, strokes, owned, deferred = self._extract(image, (ink,))

        self.assertGreaterEqual(len(strokes), 1)
        self.assertFalse(deferred.any())
        self.assertGreaterEqual(int((owned & mask).sum()),
                                int(mask.sum() * 0.98))


if __name__ == "__main__":
    unittest.main()
