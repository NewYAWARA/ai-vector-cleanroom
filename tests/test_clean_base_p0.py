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
import trace_engine


def _fake_vtracer(_src, dst, **_kwargs):
    Path(dst).write_text("<svg/>", encoding="utf-8")


class CleanBaseP0Tests(unittest.TestCase):
    def test_fragmented_linear_detail_is_unified_but_compact_shape_is_not(self):
        palette = np.asarray(((213, 200, 147), (225, 179, 47),
                              (243, 247, 244)), dtype=np.uint8)
        den = np.full((80, 80, 3), 255, dtype=np.float32)
        visible = np.zeros((80, 80), dtype=bool)
        labels = np.zeros((80, 80), dtype=np.int16)
        line_pixels = []
        for x in range(6, 58):
            y = 10 + (x - 6) // 2
            for yy in (y, y + 1):
                line_pixels.append((yy, x))
                visible[yy, x] = True
                label = (x // 5) % 3
                labels[yy, x] = label
                den[yy, x] = palette[label]
        visible[60:72, 60:72] = True
        labels[60:72, 60:72] = np.indices((12, 12)).sum(0) % 2
        den[60:72, 60:72] = palette[labels[60:72, 60:72]]

        stabilized, audit = clean_base._stabilize_fragmented_linear_details(
            den, visible, palette, labels, (255, 255, 255))

        yy, xx = zip(*line_pixels)
        self.assertEqual(audit["components_stabilized"], 1)
        self.assertEqual(len(np.unique(stabilized[yy, xx])), 1)
        np.testing.assert_array_equal(
            stabilized[60:72, 60:72], labels[60:72, 60:72])

    def test_fragmented_antialiased_line_uses_strong_colour_core(self):
        pale_neutral = (189, 203, 180)
        gold = (188, 130, 11)
        palette = np.asarray((pale_neutral, gold), dtype=np.uint8)
        den = np.full((40, 80, 3), 255, dtype=np.float32)
        visible = np.zeros((40, 80), dtype=bool)
        labels = np.zeros((40, 80), dtype=np.int16)
        # A faded one-pixel gold ray: most samples are pale antialiasing, while
        # a minority near the centre reveals the intended opaque paint.
        line_pixels = []
        for x in range(8, 70):
            y = 8 + (x - 8) // 3
            alpha = 0.18 + 0.75 * ((x % 9) / 8.0)
            colour = np.rint(255 + alpha * (np.asarray(gold) - 255))
            for yy in (y, y + 1):
                line_pixels.append((yy, x))
                den[yy, x] = colour
                visible[yy, x] = True
                labels[yy, x] = x % 2

        stabilized, audit = clean_base._stabilize_fragmented_linear_details(
            den, visible, palette, labels, (255, 255, 255))

        yy, xx = zip(*line_pixels)
        self.assertEqual(audit["components_stabilized"], 1)
        self.assertTrue(np.all(stabilized[yy, xx] == 1))
        self.assertEqual(audit["components"][0]["paint"], "#bc820b")
        self.assertEqual(audit["components"][0]["colour_core_quantile"], 0.85)

    def test_initial_accent_retention_restores_only_material_component(self):
        initial = np.asarray(((13, 70, 44), (138, 96, 51)), dtype=np.uint8)
        fill = np.asarray(((13, 70, 44), (118, 129, 76)), dtype=np.uint8)
        den = np.full((64, 64, 3), 255, dtype=np.float32)
        visible = np.zeros((64, 64), dtype=bool)
        labels = np.zeros((64, 64), dtype=np.int16)
        # A coherent 16x12 brown emblem is large enough to be a real accent.
        visible[8:20, 8:24] = True
        den[8:20, 8:24] = (136, 93, 48)
        labels[8:20, 8:24] = 1
        # A tiny brown fleck must remain with the compact fill palette.
        visible[40:45, 40:45] = True
        den[40:45, 40:45] = (136, 93, 48)
        labels[40:45, 40:45] = 1

        palette, retained, audit = clean_base._retain_initial_accent_colors(
            den, visible, initial, fill, labels,
            minimum_foreground_share=0.0)

        self.assertEqual(audit["colors_retained"], 1)
        self.assertEqual(tuple(palette[-1]), (138, 96, 51))
        self.assertTrue(np.all(retained[8:20, 8:24] == 2))
        self.assertTrue(np.all(retained[40:45, 40:45] == 1))

    def test_broad_circle_and_rectangle_background_pockets_are_removed(self):
        yy, xx = np.ogrid[:80, :80]
        circle = (xx - 40) ** 2 + (yy - 40) ** 2 <= 30 ** 2
        rectangle = np.zeros((80, 80), dtype=bool)
        rectangle[10:70, 12:68] = True

        for name, enclosure in (("circle", circle),
                                ("rectangle", rectangle)):
            with self.subTest(enclosure=name):
                removed = clean_base._broad_enclosed_background_mask(
                    enclosure, enclosure, enclosure)
                np.testing.assert_array_equal(removed, enclosure)

    def test_enclosed_white_letters_and_thin_highlight_are_preserved(self):
        enclosure = np.zeros((80, 80), dtype=bool)
        enclosure[10:70, 10:70] = True
        visible = enclosure.copy()
        light_paint = np.zeros_like(enclosure)
        # Two disconnected block-letter fragments.
        light_paint[23:39, 21:27] = True
        light_paint[23:29, 27:36] = True
        light_paint[33:39, 27:36] = True
        light_paint[23:39, 45:51] = True
        light_paint[23:29, 51:59] = True
        # A one-pixel highlight may span widely, but is not broad in both axes.
        light_paint[52, 19:61] = True

        removed = clean_base._broad_enclosed_background_mask(
            visible, light_paint, enclosure)

        self.assertFalse(removed.any())

    def test_enclosed_twenty_percent_white_emblem_is_preserved(self):
        enclosure = np.zeros((80, 80), dtype=bool)
        enclosure[10:70, 10:70] = True
        visible = enclosure.copy()
        emblem = np.zeros_like(enclosure)
        # 20 x 36 = 720 pixels, exactly 20% of the 60 x 60 enclosure.
        emblem[30:50, 22:58] = True

        removed = clean_base._broad_enclosed_background_mask(
            visible, emblem, enclosure)

        self.assertFalse(removed.any())

    def test_partitioned_cross_background_pockets_are_removed(self):
        enclosure = np.zeros((80, 80), dtype=bool)
        enclosure[10:70, 10:70] = True
        visible = enclosure.copy()
        background = enclosure.copy()
        # Opaque artwork touching all four sides partitions one negative-space
        # background into four individually sub-30% components.
        background[38:42, 10:70] = False
        background[10:70, 38:42] = False

        removed = clean_base._broad_enclosed_background_mask(
            visible, background, enclosure)

        np.testing.assert_array_equal(removed, background)

    def test_stroke_proven_ring_hole_canonicalizes_scaled_reference(self):
        from vector_cleanroom import _apply_validation_hole_mask

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "ring.png"
            dst = td / "ring.svg"
            yy, xx = np.ogrid[:128, :128]
            radius = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2)
            rgba = np.full((128, 128, 4), 255, dtype=np.uint8)
            rgba[np.abs(radius - 40) <= 4, :3] = 0
            Image.fromarray(rgba, "RGBA").save(src)

            # Force a trace/native resolution difference: the private evidence
            # must stay compact and still clear the correct native pixels.
            stats = clean_base.build_clean_base(
                src, dst, background="auto", max_size=64,
                geometry="conservative", strokes="on", gradients="off")
            hole = stats._validation_hole_mask()

            self.assertTrue(stats.removed_background)
            self.assertIsNotNone(hole)
            self.assertEqual(hole.shape, (64, 64))
            self.assertTrue(hole[32, 32])
            self.assertLess(len(stats._validation_hole_bits), hole.nbytes // 4)

            prepared, _size, _removed = trace_engine._prepare_image(
                src, max_size=0, background="auto", white_threshold=220,
                alpha_threshold=12)
            canonical = _apply_validation_hole_mask(prepared, hole)
            alpha = np.asarray(canonical)[:, :, 3]
            self.assertEqual(int(alpha[64, 64]), 0)   # ring interior
            self.assertEqual(int(alpha[2, 2]), 0)     # exterior canvas
            self.assertEqual(int(alpha[64, 104]), 255)  # black ring itself

    def test_clustered_glyph_counters_are_removed_from_canonical_alpha(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "counter-row.png"
            rgba = np.full((120, 220, 4), 255, dtype=np.uint8)
            # Three separate dark glyph-like frames on opaque white paper.
            # Their trapped white centres are negative space, not white paint.
            for x0 in (22, 94, 166):
                rgba[38:82, x0:x0 + 32, :3] = (13, 70, 44)
                rgba[40:80, x0 + 2:x0 + 30, :3] = 255
            Image.fromarray(rgba, "RGBA").save(src)

            prepared, _size, removed = trace_engine._prepare_image(
                src, max_size=0, background="auto", white_threshold=220,
                alpha_threshold=12)
            alpha = np.asarray(prepared)[:, :, 3]
            counter_mask = np.asarray(
                prepared._avc_background_counter_mask, dtype=bool)

            self.assertTrue(removed)
            self.assertEqual(int(counter_mask.sum()), 3 * 28 * 40)
            for x0 in (22, 94, 166):
                self.assertEqual(int(alpha[60, x0 + 16]), 0)
                self.assertEqual(int(alpha[39, x0 + 1]), 255)
                self.assertTrue(counter_mask[60, x0 + 16])

    def test_clustered_real_white_marks_in_solid_field_are_preserved(self):
        visible = np.zeros((120, 220), dtype=bool)
        white_marks = np.zeros_like(visible)
        # The marks form a horizontal group and sit only eight pixels from
        # transparency.  A component-relative search radius would misclassify
        # them; the strict canvas-scale distance must keep each coloured tile.
        for x0 in (10, 82, 154):
            visible[20:86, x0:x0 + 46] = True
            white_marks[28:78, x0 + 8:x0 + 38] = True

        removed = trace_engine._clustered_enclosed_background_mask(
            visible, white_marks)

        self.assertFalse(removed.any())

    def test_prepare_image_keeps_near_edge_white_wordmark_row(self):
        """Real white text near a cutout edge must not become counters.

        This exercises the full background-removal route rather than only the
        component helper.  The eight-pixel dark margin is intentionally small:
        a component-relative search radius reaches the transparent canvas and
        mistakes the three white glyphs for paper-coloured holes, while the
        canvas-scale counter guard stays local enough to preserve them.
        """
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "near-edge-white-wordmark.png"
            rgba = np.full((120, 220, 4), 255, dtype=np.uint8)
            rgba[20:100, 10:210, :3] = (31, 22, 22)
            for x0 in (18, 90, 162):
                rgba[28:78, x0:x0 + 30, :3] = 255
            Image.fromarray(rgba, "RGBA").save(src)

            prepared, _size, removed = trace_engine._prepare_image(
                src, max_size=0, background="auto", white_threshold=220,
                alpha_threshold=12)
            alpha = np.asarray(prepared)[:, :, 3]
            counter_mask = np.asarray(
                prepared._avc_background_counter_mask, dtype=bool)

            self.assertTrue(removed)
            self.assertEqual(int(alpha[2, 2]), 0)
            self.assertFalse(counter_mask.any())
            for x0 in (18, 90, 162):
                self.assertEqual(int(alpha[50, x0 + 15]), 255)

    def test_opaque_trace_surround_is_removed_but_internal_white_is_kept(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "transparent-logo.png"
            dst = td / "out.svg"
            flat = td / "flat.png"
            rgba = np.zeros((32, 32, 4), dtype=np.uint8)
            rgba[4:28, 4:28, :3] = (13, 70, 44)
            rgba[4:28, 4:28, 3] = 255
            rgba[11:16, 11:16, :3] = 255
            Image.fromarray(rgba, "RGBA").save(src)
            captured = {}

            def fake_trace(trace_src, trace_dst, **kwargs):
                work = Image.open(trace_src).convert("RGBA")
                is_light = Path(trace_dst).stem.startswith("light-")
                if not is_light:
                    captured["alpha_extrema"] = work.getchannel("A").getextrema()
                    route = work.getpixel((0, 0))[:3]
                    captured["route"] = route
                    # Real VTracer may shift the routing fill by a level.  The
                    # removal guard must use colour tolerance plus geometry.
                    captured["raw_background"] = tuple(max(0, value - 1)
                                                        for value in route)
                    captured["hierarchical"] = kwargs.get("hierarchical")
                    captured["filter_speckle"] = kwargs.get("filter_speckle")
                marker = "light" if is_light else "main"
                Path(trace_dst).write_text(
                    f'<svg data-test-kind="{marker}"/>', encoding="utf-8")

            def fake_paths(raw):
                if 'data-test-kind="light"' in raw:
                    return iter((
                        ("M 11 11 L 16 11 L 16 16 L 11 16 Z",
                         "#000000", 0.0, 0.0),
                    ))
                bg = "#{:02x}{:02x}{:02x}".format(*captured["raw_background"])
                return iter((
                    ("M 0 0 L 32 0 L 32 32 L 0 32 Z", bg, 0.0, 0.0),
                    ("M 4 4 L 28 4 L 28 28 L 4 28 Z",
                     "#0d462c", 0.0, 0.0),
                    ("M 11 11 L 16 11 L 16 16 L 11 16 Z",
                     "#ffffff", 0.0, 0.0),
                ))

            with patch.object(clean_base.vtracer, "convert_image_to_svg_py",
                              side_effect=fake_trace), \
                    patch.object(clean_base, "_iter_svg_paths",
                                 side_effect=fake_paths):
                stats = clean_base.build_clean_base(
                    src, dst, flat_out=flat, background="keep",
                    geometry="off", strokes="off", gradients="off")

            svg = dst.read_text(encoding="utf-8")
            self.assertEqual(captured["alpha_extrema"], (255, 255))
            self.assertEqual(captured["hierarchical"], "cutout")
            self.assertEqual(captured["filter_speckle"], 2)
            self.assertLessEqual(
                max(abs(captured["route"][i] - 255) for i in range(3)), 31)
            self.assertEqual(stats.n_paths, 2)
            self.assertNotIn("M 0 0 L 32 0", svg)
            self.assertIn("M11 11", svg)
            self.assertEqual(Image.open(flat).getpixel((0, 0))[3], 0)
            self.assertTrue(any("tracer-only routing background" in note
                                for note in stats.geometry_notes))

    def test_real_trace_internal_white_remains_white_on_coloured_canvas(self):
        from vector_cleanroom import render_svg_png, self_check

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "white-mark.png"
            dst = td / "white-mark.svg"
            flat = td / "flat.png"
            rendered = td / "coloured.png"
            rgba = np.zeros((64, 64, 4), dtype=np.uint8)
            rgba[8:56, 8:56, :3] = (13, 70, 44)
            rgba[8:56, 8:56, 3] = 255
            rgba[18:46, 27:37, :3] = 255
            Image.fromarray(rgba, "RGBA").save(src)

            stats = clean_base.build_clean_base(
                src, dst, flat_out=flat, background="auto", geometry="off",
                strokes="off", gradients="off")
            self.assertTrue(any("internal light-paint" in note
                                for note in stats.geometry_notes))
            self.assertTrue(render_svg_png(
                dst, rendered, size=64, bg=0x5b4b8a))

            image = Image.open(rendered).convert("RGB")
            white = image.getpixel((32, 32))
            outside = image.getpixel((2, 2))
            self.assertGreater(min(white), 240)
            self.assertLess(sum(abs(outside[i] - (91, 75, 138)[i])
                                for i in range(3)), 12)
            scores = self_check(dst, flat, src, viewbox=(64, 64))
            transparency = scores["transparent_light_fidelity"]
            self.assertTrue(transparency["applicable"])
            self.assertGreaterEqual(transparency["coverage_percent"], 95.0)

    def test_light_fidelity_ignores_matte_ring_but_checks_white_core_colour(self):
        from vector_cleanroom import (
            _source_has_transparent_light_objects,
            _transparent_light_fidelity,
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source_path = td / "source.png"
            correct_path = td / "correct.png"
            wrong_path = td / "wrong.png"
            background = np.asarray((91, 75, 138), dtype=np.uint8)

            source = np.zeros((48, 48, 4), dtype=np.uint8)
            # A one-pixel white matte surrounds the coloured square.  It must
            # not be mistaken for a white design object.
            source[3:45, 3:45, :3] = 255
            source[3:45, 3:45, 3] = 255
            source[4:44, 4:44, :3] = (13, 70, 44)
            # This thick white mark is a genuine object and must stay white.
            source[16:32, 16:32, :3] = 255
            Image.fromarray(source, "RGBA").save(source_path)

            correct = np.empty((48, 48, 3), dtype=np.uint8)
            correct[:] = background
            correct[4:44, 4:44] = (13, 70, 44)
            correct[16:32, 16:32] = 255
            Image.fromarray(correct, "RGB").save(correct_path)
            wrong = correct.copy()
            wrong[16:32, 16:32] = (13, 70, 44)
            Image.fromarray(wrong, "RGB").save(wrong_path)

            self.assertTrue(_source_has_transparent_light_objects(source_path))
            good = _transparent_light_fidelity(correct_path, source_path)
            bad = _transparent_light_fidelity(wrong_path, source_path)
            self.assertTrue(good["applicable"])
            self.assertEqual(good["measurement_mask"],
                             "one_pixel_eroded_light_core")
            self.assertGreaterEqual(good["coverage_percent"], 99.0)
            # Merely being different from the purple background is no longer
            # enough: the rendered object must match the expected light paint.
            self.assertLess(bad["coverage_percent"], 1.0)
            self.assertGreater(bad["non_background_coverage_percent"], 99.0)

    def test_light_fidelity_does_not_activate_for_only_one_pixel_halo(self):
        from vector_cleanroom import (
            _source_has_transparent_light_objects,
            _transparent_light_fidelity,
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source_path = td / "source.png"
            render_path = td / "render.png"
            source = np.zeros((48, 48, 4), dtype=np.uint8)
            source[3:45, 3:45, :3] = 255
            source[3:45, 3:45, 3] = 255
            source[4:44, 4:44, :3] = (13, 70, 44)
            Image.fromarray(source, "RGBA").save(source_path)
            rendered = np.empty((48, 48, 3), dtype=np.uint8)
            rendered[:] = (91, 75, 138)
            rendered[4:44, 4:44] = (13, 70, 44)
            Image.fromarray(rendered, "RGB").save(render_path)

            self.assertFalse(_source_has_transparent_light_objects(source_path))
            result = _transparent_light_fidelity(render_path, source_path)
            self.assertFalse(result["applicable"])
            self.assertEqual(
                result["inapplicable_reason"],
                "fewer_than_64_stable_light_core_pixels")

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
