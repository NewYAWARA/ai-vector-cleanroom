import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from designer_handoff import build_handoff_manifest
from svg_bounds import parse_transform
from svg_renderer import render_svg_reference


class SvgBoundsTests(unittest.TestCase):
    def check_paint_enclosed(self, body):
        with tempfile.TemporaryDirectory() as directory:
            svg, png = Path(directory)/'test.svg', Path(directory)/'test.png'
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'+body+'</svg>')
            manifest = build_handoff_manifest(svg)
            self.assertEqual(len(manifest['objects']), 1)
            x,y,w,h = manifest['objects'][0]['bbox']
            render_svg_reference(svg,png,width=800,background=None)
            painted = Image.open(png).getchannel('A').point(lambda value: 255 if value > 10 else 0).getbbox()
            self.assertIsNotNone(painted)
            left,top,right,bottom = [value/4 for value in painted]
            self.assertLessEqual(x, left+.25)
            self.assertLessEqual(y, top+.25)
            self.assertGreaterEqual(x+w, right-.25)
            self.assertGreaterEqual(y+h, bottom-.25)

    def test_transform_order_and_center_rotation(self):
        self.assertEqual(parse_transform('translate(10 20) scale(2,3)'), (2,0,0,3,10,20))
        matrix = parse_transform('rotate(90,10,20)')
        for actual, expected in zip(matrix,(0,1,-1,0,30,10)):
            self.assertAlmostEqual(actual,expected)

    def test_rotated_arcs_relative_radii_correction_and_sweeps(self):
        for flags in ('0 0','0 1','1 0','1 1'):
            with self.subTest(flags=flags):
                self.check_paint_enclosed(f'<path d="M65 75 a20 30 35 {flags} 55 50 Z" fill="red"/>')

    def test_nested_transform_and_inherited_stroke(self):
        self.check_paint_enclosed('<g transform="translate(80 40) rotate(28)" stroke="black" stroke-width="8" fill="none"><g transform="scale(1.2 .8)"><path d="M0 0L40 20L0 50" stroke-linejoin="miter" stroke-linecap="square"/></g></g>')
        self.check_paint_enclosed('<g stroke="black" stroke-width="40"><rect x="10" y="20" width="30" height="50" stroke="none"/></g>')

    def test_reflected_bezier_controls(self):
        self.check_paint_enclosed('<path d="M20 70Q40 10 60 70T100 70T140 70Z"/>')
        self.check_paint_enclosed('<g transform="translate(10 10) skewX(10)"><path d="M20 70C25 20 45 20 60 70S90 120 110 70Z"/></g>')

    def test_invalid_transform_is_not_estimated(self):
        for raw in ('translate(1px)', 'scale(NaN)', 'matrix(1 0 0)', 'rotate(1) garbage'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_transform(raw)


if __name__ == '__main__':
    unittest.main()
