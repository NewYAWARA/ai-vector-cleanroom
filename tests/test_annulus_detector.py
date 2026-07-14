from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from annulus_detector import (
    apply_candidate,
    compare_rendered_pngs,
    detect_svg_annuli,
)


ROOT = Path(__file__).resolve().parents[1]
SVG_NS = "http://www.w3.org/2000/svg"


def _arc_path(cx, cy, radius, start_degrees, end_degrees, count=80):
    points = []
    for index in range(count + 1):
        angle = math.radians(
            start_degrees + (end_degrees - start_degrees) * index / count)
        points.append((cx + radius * math.cos(angle),
                       cy + radius * math.sin(angle)))
    return "M" + " L".join(f"{x:.4f} {y:.4f}" for x, y in points)


def _ellipse_path(cx, cy, rx, ry, start_degrees, end_degrees, count=80):
    points = []
    for index in range(count + 1):
        angle = math.radians(
            start_degrees + (end_degrees - start_degrees) * index / count)
        points.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))
    return "M" + " L".join(f"{x:.4f} {y:.4f}" for x, y in points)


def _write_svg(path: Path, paths: list[dict]):
    body = []
    for attrs in paths:
        serialized = " ".join(f'{key}="{value}"' for key, value in attrs.items())
        body.append(f"    <path {serialized}/>")
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="320" '
        'viewBox="0 0 320 320">\n'
        '  <g id="strokes" fill="none" stroke-linecap="round" '
        'stroke-linejoin="round">\n'
        + "\n".join(body)
        + "\n  </g>\n</svg>\n",
        encoding="utf-8",
    )


class AnnulusDetectorTests(unittest.TestCase):
    def test_two_matching_arcs_become_one_safe_native_circle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.svg"
            _write_svg(source, [
                {"id": "top", "stroke": "#71ff00", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, -25, -155)},
                {"id": "bottom", "stroke": "#71ff00", "stroke-width": "8.2",
                 "d": _arc_path(160.5, 159.7, 112.2, 25, 155)},
            ])

            candidates = detect_svg_annuli(source)
            self.assertEqual(len(candidates), 1)
            candidate = candidates[0]
            self.assertTrue(candidate.safe_to_replace)
            self.assertEqual(set(candidate.source_ids), {"top", "bottom"})
            self.assertAlmostEqual(candidate.cx, 160.25, delta=1.0)
            self.assertAlmostEqual(candidate.cy, 159.85, delta=1.0)
            self.assertAlmostEqual(candidate.radius, 112.1, delta=1.0)
            self.assertGreater(candidate.coverage_degrees, 250.0)
            self.assertGreaterEqual(candidate.raster_recall, 0.985)
            self.assertGreaterEqual(candidate.raster_precision, 0.985)
            self.assertEqual(candidate.linecap, "round")
            self.assertTrue(candidate.dasharray)

            output = apply_candidate(source, candidate, root / "proposal.svg")
            svg_root = ET.parse(output).getroot()
            paths = list(svg_root.iter(f"{{{SVG_NS}}}path"))
            circles = list(svg_root.iter(f"{{{SVG_NS}}}circle"))
            self.assertEqual(paths, [])
            self.assertEqual(len(circles), 1)
            circle = circles[0]
            self.assertEqual(circle.attrib["stroke"], "#71ff00")
            self.assertEqual(circle.attrib["stroke-linecap"], "round")
            self.assertIn("stroke-dasharray", circle.attrib)
            self.assertEqual(circle.attrib["data-merged-from"], "bottom,top")

    def test_different_paint_or_width_does_not_merge(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.svg"
            _write_svg(source, [
                {"id": "a", "stroke": "#71ff00", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, -25, -155)},
                {"id": "b", "stroke": "#70ff00", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, 25, 155)},
                {"id": "c", "stroke": "#71ff00", "stroke-width": "14",
                 "d": _arc_path(160, 160, 112, 25, 155)},
            ])
            self.assertEqual(detect_svg_annuli(source), [])

    def test_short_or_overlapping_arcs_do_not_fake_an_annulus(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            short = root / "short.svg"
            _write_svg(short, [
                {"id": "a", "stroke": "#000000", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, -10, -55)},
                {"id": "b", "stroke": "#000000", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, 10, 55)},
            ])
            self.assertEqual(detect_svg_annuli(short), [])

            overlap = root / "overlap.svg"
            _write_svg(overlap, [
                {"id": "a", "stroke": "#000000", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, -20, -160)},
                {"id": "b", "stroke": "#000000", "stroke-width": "8",
                 "d": _arc_path(160, 160, 112, -22, -158)},
            ])
            self.assertEqual(detect_svg_annuli(overlap), [])

    def test_ellipse_is_rejected_by_radial_residual_gate(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "ellipse.svg"
            _write_svg(source, [
                {"id": "top", "stroke": "#000000", "stroke-width": "8",
                 "d": _ellipse_path(160, 160, 118, 96, -20, -160)},
                {"id": "bottom", "stroke": "#000000", "stroke-width": "8",
                 "d": _ellipse_path(160, 160, 118, 96, 20, 160)},
            ])
            self.assertEqual(detect_svg_annuli(source), [])

    def test_render_pair_gate_detects_identical_and_damaged_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            before = root / "before.png"
            same = root / "same.png"
            damaged = root / "damaged.png"
            image = Image.new("RGB", (200, 200), "white")
            draw = ImageDraw.Draw(image)
            draw.ellipse((30, 30, 170, 170), outline="black", width=8)
            image.save(before)
            image.save(same)
            broken = image.copy()
            ImageDraw.Draw(broken).rectangle((0, 0, 100, 200), fill="white")
            broken.save(damaged)

            accepted = compare_rendered_pngs(before, same)
            rejected = compare_rendered_pngs(before, damaged)
            self.assertTrue(accepted["accepted"])
            self.assertAlmostEqual(accepted["score_percent"], 100.0)
            self.assertFalse(rejected["accepted"])
            self.assertLess(rejected["ink_recall_percent"], 80.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
