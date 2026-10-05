"""Actual SVG paints must use symmetric alpha evidence in foreground scoring."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from svg_renderer import render_svg_reference
from vector_cleanroom import _match_percent, self_check
import vector_cleanroom


class ForegroundAlphaMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def render(self, name, body, *, width=96, background=None):
        svg = self.root / f'{name}.svg'
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="96" '
                       'height="96" viewBox="0 0 96 96">' + body + '</svg>',
                       encoding='utf-8')
        png = svg.with_suffix('.png')
        render_svg_reference(svg, png, width=width, background=background)
        return svg, png

    def score(self, candidate, reference):
        return _match_percent(candidate, reference, foreground_only=True,
                              return_details=True)

    def mark(self, image_path, origin):
        with Image.open(image_path) as image:
            rgba = image.convert('RGBA')
        metadata = PngInfo()
        metadata.add_text('avc_reference_alpha_origin', origin)
        rgba.save(image_path, pnginfo=metadata)

    def pipeline(self, source, name):
        options = SimpleNamespace(strokes='on', gradients='on', geometry='conservative',
                                  background='auto', colors=0, white_threshold=220,
                                  max_size=2048, curve_error_percent=0.25)
        output = self.root / 'output'
        output.mkdir(exist_ok=True)
        with redirect_stdout(io.StringIO()):
            vector_cleanroom.process_one(source, name, options, output)
        result = output / f'result_{name}'
        return result, json.loads((result / 'report.json').read_text(encoding='utf-8'))

    def test_real_white_object_on_transparency_is_preserved_and_missing_is_rejected(self):
        dark = '<rect x="12" y="24" width="24" height="48" fill="#285d76"/>'
        white = '<rect x="60" y="24" width="24" height="48" fill="white"/>'
        _, source = self.render('source', dark + white)
        _, same = self.render('same', dark + white)
        _, lost = self.render('lost', dark)
        preserved = self.score(same, source)
        self.assertGreater(preserved['score'], 99.9)
        self.assertTrue(preserved['alpha_comparison_applied'])
        self.assertEqual(preserved['source_ink_pixels'], preserved['render_ink_pixels'])
        self.assertLess(self.score(lost, source)['score'], 70)

    def test_filling_transparent_canvas_white_cannot_pass_as_equivalent(self):
        shape = '<circle cx="48" cy="48" r="18" fill="#285d76"/>'
        _, source = self.render('source', shape)
        _, filled = self.render('filled', '<rect width="96" height="96" fill="white"/>' + shape)
        self.assertLess(self.score(filled, source)['score'], 55)
        self.assertLess(_match_percent(filled, source), 30)
        # A PNG without an alpha channel explicitly means opaque paint.
        with Image.open(filled) as image:
            image.convert('RGB').save(filled)
        result = self.score(filled, source)
        self.assertFalse(result['render_has_alpha_channel'])
        self.assertLess(result['score'], 55)

    def test_opaque_source_does_not_forbid_deliberate_white_canvas_removal(self):
        shape = '<rect x="24" y="24" width="48" height="48" fill="#285d76"/>'
        _, source = self.render('opaque-source', shape, background='#ffffff')
        _, transparent = self.render('transparent-candidate', shape)
        result = self.score(transparent, source)
        self.assertFalse(result['alpha_comparison_applied'])
        self.assertGreater(result['score'], 99.9)

    def test_white_alpha_strength_is_checked_at_native_and_resized_resolution(self):
        half = '<rect x="24" y="24" width="48" height="48" fill="white" fill-opacity="0.5"/>'
        full = '<rect x="24" y="24" width="48" height="48" fill="white"/>'
        _, source = self.render('half-source', half)
        for width in (96, 48):
            with self.subTest(width=width):
                _, same = self.render(f'half-{width}', half, width=width)
                _, wrong = self.render(f'opaque-{width}', full, width=width)
                self.assertGreater(self.score(same, source)['score'], 99)
                self.assertLess(self.score(wrong, source)['score'], 75)

    def test_self_check_uses_native_alpha_but_keeps_white_composite_for_consumers(self):
        body = ('<rect x="12" y="24" width="24" height="48" fill="#285d76"/>'
                '<rect x="60" y="24" width="24" height="48" fill="white"/>')
        svg, source = self.render('source', body)
        preview = self.root / 'kept-render.png'
        scores = self_check(svg, source, source, keep_render=preview, viewbox=(96, 96))
        self.assertGreater(scores['foreground'], 99.9)
        self.assertTrue(scores['alpha_comparison']['alpha_comparison_applied'])
        self.assertGreater(scores['transparent_light_fidelity']['coverage_percent'], 99.9)
        with Image.open(preview) as image:
            self.assertEqual(image.mode, 'RGB')
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertFalse((self.root / '_selfcheck_alpha.png').exists())

    def test_derived_alpha_still_rejects_missing_white_and_filled_canvas(self):
        body = '<rect x="24" y="24" width="48" height="48" fill="white"/>'
        _, source = self.render('source', body)
        self.mark(source, 'opaque_canvas_derived')
        _, same = self.render('same', body)
        _, missing = self.render('missing', '')
        _, half = self.render('half', body.replace('fill="white"', 'fill="white" opacity="0.5"'))
        _, full = self.render('full', '<rect width="96" height="96" fill="white"/>')
        self.assertGreater(self.score(same, source)['score'], 99.9)
        self.assertLess(self.score(missing, source)['score'], 1)
        self.assertLess(self.score(half, source)['score'], 70)
        self.assertLess(self.score(full, source)['score'], 70)

    def test_complete_pipeline_keeps_true_thin_strokes_after_opaque_source_cleanup(self):
        from generate_designer_benchmark import fixture_cases
        case = next(c for c in fixture_cases() if c['id'] == '06_thin_lines')
        svg = self.root / 'lines.svg'
        svg.write_text(case['svg'], encoding='utf-8')
        source = self.root / 'lines.png'
        render_svg_reference(svg, source, width=128, background='#ffffff')
        # Untrusted metadata cannot contradict the original opaque pixels.
        self.mark(source, 'native')
        result, report = self.pipeline(source, 'thin')
        self.assertEqual(report['strokes'], 3)
        self.assertEqual(report['options_effective']['strokes'], 'on')
        self.assertGreater(report['foreground_match_percent'], 99)
        policy = report['foreground_alpha_comparison']
        self.assertEqual(policy['reference_alpha_origin'], 'opaque_canvas_derived')
        with Image.open(result / 'source_reference.png') as image:
            self.assertEqual(image.info['avc_reference_alpha_origin'], 'opaque_canvas_derived')

    def test_complete_pipeline_does_not_trust_fake_opaque_provenance_on_native_alpha(self):
        body = ('<rect x="12" y="24" width="24" height="48" fill="#285d76"/>'
                '<rect x="60" y="24" width="24" height="48" fill="white" opacity="0.5"/>')
        _, source = self.render('native', body)
        self.mark(source, 'opaque_canvas_derived')
        result, report = self.pipeline(source, 'native')
        policy = report['foreground_alpha_comparison']
        self.assertEqual(policy['reference_alpha_origin'], 'native')
        self.assertEqual(policy['alpha_policy'], 'symmetric_support_and_strict_alpha_magnitude')
        with Image.open(result / 'source_reference.png') as image:
            self.assertEqual(image.info['avc_reference_alpha_origin'], 'native')

    def test_hole_adjusted_metric_reference_keeps_original_pixel_provenance(self):
        _, source = self.render('ring', '<circle cx="48" cy="48" r="30" fill="none" stroke="#285d76" stroke-width="8"/>', background='#ffffff')
        calls = []
        actual_check = vector_cleanroom.self_check
        def record(svg, flat, reference, **kwargs):
            with Image.open(reference) as image:
                calls.append((Path(reference).name, image.info.get('avc_reference_alpha_origin')))
            return actual_check(svg, flat, reference, **kwargs)
        with patch.object(vector_cleanroom, 'self_check', side_effect=record):
            result, report = self.pipeline(source, 'ring')
        self.assertTrue(any('_holes_' in name for name, _ in calls))
        self.assertTrue(all(origin == 'opaque_canvas_derived' for _, origin in calls))
        with Image.open(result / 'source_reference.png') as image:
            self.assertEqual(image.info['avc_reference_alpha_origin'], 'opaque_canvas_derived')


if __name__ == '__main__':
    unittest.main()
