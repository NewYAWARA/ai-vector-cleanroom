"""Focused regressions for the real-world logo P0 clean-base fixes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import clean_base


def _fake_vtracer(_src, dst, **_kwargs):
    Path(dst).write_text("<svg/>", encoding="utf-8")


class CleanBaseP0Tests(unittest.TestCase):
    def test_stroke_sample_snaps_only_to_near_initial_palette_color(self):
        palette = np.asarray(((33, 24, 21), (140, 254, 1)), dtype=np.uint8)
        self.assertEqual(
            clean_base._snap_stroke_color_to_palette((39, 19, 25), palette),
            (33, 24, 21),
        )
        # Downsampling can turn a true 1 px black line into mid-gray.  That
        # distant trace color must not undo original-resolution black recovery.
        self.assertEqual(
            clean_base._snap_stroke_color_to_palette(
                (0, 0, 0), np.asarray(((148, 148, 148),), dtype=np.uint8)),
            (0, 0, 0),
        )

    def test_original_resolution_jpeg_variant_is_canonicalized(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "jpeg-like.png"
            dst = td / "out.svg"

            original = np.full((100, 100, 4), 255, dtype=np.uint8)
            original[49:52, 49:52, :3] = (38, 20, 24)
            Image.fromarray(original, "RGBA").save(src)

            trace = np.empty((10, 10, 4), dtype=np.uint8)
            trace[..., :3] = (33, 24, 21)
            trace[..., 3] = 255
            stroke = SimpleNamespace(
                sample_points=[(5.0, 5.0)], color=(99, 99, 99),
                opacity=1.0, width=1.0, primitive="",
                d="M 0 5 L 9 5", n_nodes=2, closed=False,
            )

            def fake_extract(_mask, _den, _palette, _bg, alpha=None):
                return [stroke], np.ones((10, 10), dtype=bool)

            with patch.object(
                    clean_base, "_prepare_image",
                    return_value=(Image.fromarray(trace, "RGBA"),
                                  (100, 100), False)), \
                    patch("stroke_engine.extract_strokes",
                          side_effect=fake_extract):
                stats = clean_base.build_clean_base(
                    src, dst, geometry="off", gradients="off")

            self.assertEqual(stats.stroke_info[0]["color"], "#211815")
            self.assertIn('stroke="#211815"', dst.read_text(encoding="utf-8"))

    def test_final_palette_uses_real_middle_gradient_stop(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "gradient.png"
            dst = td / "out.svg"
            rgba = np.full((32, 32, 4), 255, dtype=np.uint8)
            rgba[:, :16, :3] = (0, 102, 0)
            rgba[:, 16:, :3] = (204, 255, 34)
            Image.fromarray(rgba, "RGBA").save(src)

            region = {
                "mask": np.ones((32, 32), dtype=bool),
                "area": 32 * 32,
                "x1": 0.0, "y1": 16.0, "x2": 31.0, "y2": 16.0,
                "stops": [
                    (0.0, (0, 102, 0)),
                    (0.5, (51, 170, 51)),
                    (1.0, (204, 255, 34)),
                ],
            }

            def fake_paths(_raw):
                key = "#{:02x}{:02x}{:02x}".format(*region["key"])
                return iter((("M 0 0 L 31 0 L 31 31 L 0 31 Z",
                              key, 0.0, 0.0),))

            with patch.object(clean_base, "_detect_gradients",
                              return_value=[region]), \
                    patch.object(clean_base.vtracer,
                                 "convert_image_to_svg_py",
                                 side_effect=_fake_vtracer), \
                    patch.object(clean_base, "_iter_svg_paths",
                                 side_effect=fake_paths):
                stats = clean_base.build_clean_base(
                    src, dst, background="keep", geometry="off",
                    strokes="off", gradients="on")

            sentinel = stats.gradient_info[0]["key"]
            self.assertEqual(stats.palette, [("gradient", "#33aa33")])
            self.assertNotEqual(stats.palette[0][1], sentinel)
            self.assertIn('fill="url(#grad1)"', dst.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "no real color stops"):
                clean_base._gradient_palette_hex({"stops": []})

    def test_circle_note_counts_emitted_fill_native_circles(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "ring.png"
            dst = td / "out.svg"
            rgba = np.zeros((32, 32, 4), dtype=np.uint8)
            rgba[..., 3] = 255
            Image.fromarray(rgba, "RGBA").save(src)

            compound = (
                "M 2 2 L 30 2 L 30 30 L 2 30 Z "
                "M 6 6 L 26 6 L 26 26 L 6 26 Z")

            def fake_regularize(entries, level="normal"):
                subs = entries[0]["subs"]
                subs[0]["is_circle"] = (16.0, 16.0, 14.0)
                subs[1]["is_circle"] = (16.0, 16.0, 10.0)
                return ["2 near-circular shapes replaced with perfect circles"]

            with patch.object(clean_base.vtracer,
                              "convert_image_to_svg_py",
                              side_effect=_fake_vtracer), \
                    patch.object(clean_base, "_iter_svg_paths",
                                 return_value=iter(((compound, "#000000",
                                                    0.0, 0.0),))), \
                    patch.object(clean_base, "_regularize",
                                 side_effect=fake_regularize):
                stats = clean_base.build_clean_base(
                    src, dst, background="keep", geometry="conservative",
                    strokes="off", gradients="off")

            self.assertEqual(stats.n_native, 1)
            self.assertIn(
                "1 near-circular fill shapes emitted as native SVG circles",
                stats.geometry_notes,
            )
            self.assertNotIn(
                "2 near-circular shapes replaced with perfect circles",
                stats.geometry_notes,
            )
            root = ET.parse(dst).getroot()
            circles = [node for node in root.iter()
                       if node.tag.rsplit("}", 1)[-1] == "circle"]
            self.assertEqual(len(circles), 1)


if __name__ == "__main__":
    unittest.main()
