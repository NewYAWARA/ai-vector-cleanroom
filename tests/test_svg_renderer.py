import tempfile
from pathlib import Path
import unittest
from PIL import Image
from svg_renderer import render_svg_reference, renderer_info


class NativeRendererTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.svg = Path(self.temp.name) / "in.svg"
        self.png = Path(self.temp.name) / "out.png"

    def render(self, body, background="#ffffff"):
        self.svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">' + body + '</svg>', encoding="utf-8")
        info = render_svg_reference(self.svg, self.png, width=100, background=background)
        self.assertEqual(info["paint_model"], "native_svg")
        with Image.open(self.png) as image:
            return image.convert("RGBA")

    def test_real_linear_and_radial_paint_are_not_flat_placeholders(self):
        image = self.render('<defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="100" height="100" fill="url(#g)"/>')
        self.assertGreater(image.getpixel((10,50))[0], 220)
        self.assertGreater(image.getpixel((90,50))[2], 220)
        image = self.render('<defs><radialGradient id="g"><stop stop-color="white"/><stop offset="1" stop-color="black"/></radialGradient></defs><circle cx="50" cy="50" r="45" fill="url(#g)"/>')
        self.assertGreater(image.getpixel((50,50))[0], 240)
        self.assertLess(image.getpixel((91,50))[0], 30)

    def test_opacity_is_composited_in_paint_order(self):
        image = self.render('<rect x="10" y="10" width="70" height="70" fill="red" opacity="0.5"/><rect x="30" y="30" width="60" height="60" fill="blue" opacity="0.5"/>')
        pixel = image.getpixel((40,40))
        for actual, expected in zip(pixel, (127,63,191,255)):
            self.assertLessEqual(abs(actual-expected), 2)

    def test_clipped_transformed_shape_and_evenodd_hole(self):
        image = self.render('<defs><clipPath id="c"><rect width="40" height="100"/></clipPath></defs><g transform="translate(10 0)"><path clip-path="url(#c)" fill-rule="evenodd" d="M0 0H80V90H0Z M20 20H30V30H20Z" fill="black"/></g>', None)
        self.assertEqual(image.getpixel((15,10))[3],255)
        self.assertEqual(image.getpixel((35,25))[3],0)
        self.assertEqual(image.getpixel((70,10))[3],0)

    def test_external_resources_and_unoutlined_text_fail_closed(self):
        for body in ('<image href="file:///secret.png"/>', '<rect style="fill:url(https://example.com/x)"/>',
                     '<text x="1" y="1">hello</text>', '<script>alert(1)</script>'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.render(body)

    def test_failed_render_does_not_replace_existing_output(self):
        self.png.write_bytes(b"old-file")
        self.svg.write_text('<svg width="1" height="1000000000"/>')
        with self.assertRaises(ValueError):
            render_svg_reference(self.svg,self.png,width=100)
        self.assertEqual(self.png.read_bytes(),b"old-file")


if __name__ == "__main__":
    unittest.main()
