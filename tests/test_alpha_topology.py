from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from alpha_topology import compare_alpha_topology, _unique_label_pairs
from vector_cleanroom import validate_svg_stage_renders


class AlphaTopologyTests(unittest.TestCase):
    def test_component_check_does_not_authorize_erasing_holes(self):
        from alpha_topology import compare_alpha_components, compare_composed_alpha
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.png', Path(directory) / 'b.png'
            before = Image.new('RGBA', (60, 60), (40, 100, 70, 255))
            ImageDraw.Draw(before).rectangle((20, 20, 25, 25), fill=(0, 0, 0, 0))
            before.save(a)
            Image.new('RGBA', (60, 60), (40, 100, 70, 255)).save(b)
            result = compare_alpha_components(a, b)
            self.assertTrue(result['accepted'])
            self.assertIn('not_hole_or_coverage_authorization', result['scope'])
            self.assertFalse(compare_alpha_topology(a, b)['accepted'])
            with self.assertRaisesRegex(ValueError, 'holes'):
                compare_composed_alpha(a, b, check_coverage=False)

    def test_separate_component_check_rejects_white_merge_and_split(self):
        from alpha_topology import compare_alpha_components
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.png', Path(directory) / 'b.png'
            before = Image.new('RGBA', (80, 60))
            draw = ImageDraw.Draw(before)
            draw.rectangle((10, 10, 30, 50), fill='white')
            draw.rectangle((33, 10, 60, 50), fill='white')
            after = before.copy()
            ImageDraw.Draw(after).rectangle((30, 28, 33, 32), fill='white')
            before.save(a); after.save(b)
            for first, second in ((a, b), (b, a)):
                with self.assertRaisesRegex(ValueError, 'components'):
                    compare_alpha_components(first, second)

    def test_separate_component_check_requires_rgba_and_same_canvas(self):
        from alpha_topology import compare_alpha_components
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.png', Path(directory) / 'b.png'
            Image.new('RGBA', (30, 30), 'white').save(a)
            Image.new('RGB', (30, 30), 'white').save(b)
            with self.assertRaisesRegex(ValueError, 'rgba_renderer'):
                compare_alpha_components(a, b)
            Image.new('RGBA', (31, 30), 'white').save(b)
            with self.assertRaisesRegex(ValueError, 'dimensions'):
                compare_alpha_components(a, b)

    def test_geometry_mode_omits_alpha_magnitude_but_keeps_component_guard(self):
        from alpha_topology import compare_composed_alpha
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.png', Path(directory) / 'b.png'
            before = Image.new('RGBA', (80, 80))
            ImageDraw.Draw(before).rectangle((20, 20, 60, 60), fill=(255, 255, 255, 240))
            before.save(a)
            after = Image.new('RGBA', (80, 80))
            ImageDraw.Draw(after).rectangle((20, 20, 60, 60), fill=(255, 255, 255, 255))
            after.save(b)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                compare_composed_alpha(a, b)
            result = compare_composed_alpha(a, b, check_coverage=False)
            self.assertTrue(result['accepted'])
            self.assertFalse(result['coverage_checked'])
            ImageDraw.Draw(after).rectangle((39, 20, 40, 60), fill=(0, 0, 0, 0))
            after.save(b)
            with self.assertRaisesRegex(ValueError, 'components'):
                compare_composed_alpha(a, b, check_coverage=False)

    def test_integer_label_keys_equal_pair_sort_and_check_overflow(self):
        import numpy as np
        generator = np.random.default_rng(1729)
        for size, maximum in ((0, 10), (3000, 151), (500, 32 * 1024 * 1024)):
            a = generator.integers(0, maximum, size=size, dtype=np.int64)
            b = generator.integers(0, maximum, size=size, dtype=np.int64)
            expected = np.unique(np.column_stack((a, b)), axis=0)
            np.testing.assert_array_equal(_unique_label_pairs(a, b), expected)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            _unique_label_pairs(np.array([2**62]), np.array([2**62]))
        with self.assertRaisesRegex(ValueError, 'negative'):
            _unique_label_pairs(np.array([-1]), np.array([1]))

    def test_unavailable_renderer_never_approves_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b = Path(directory)/'a.svg',Path(directory)/'b.svg'
            a.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            b.write_bytes(a.read_bytes())
            with patch('vector_cleanroom.render_svg_png', return_value=False):
                result = validate_svg_stage_renders(a,b,'curve_refit')
            self.assertFalse(result['accepted'])
            self.assertEqual(result['validation_level'],'unverified_fail_closed')

    def compare(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            a,b = Path(directory)/'a.png', Path(directory)/'b.png'
            before.save(a);after.save(b)
            return compare_alpha_topology(a,b)

    def test_new_interior_hairline_gap_is_not_averaged_away(self):
        before = Image.new('RGBA',(200,200),(40,90,70,255))
        after=before.copy()
        ImageDraw.Draw(after).rectangle((98,80,99,100), fill=(40,90,70,40))
        result=self.compare(before,after)
        self.assertFalse(result['accepted'])
        self.assertTrue(any(row['created_regions'] for row in result['thresholds']))

    def test_lost_hole_is_rejected_but_small_boundary_shift_is_allowed(self):
        a=Image.new('RGBA',(60,60),(20,30,40,255))
        ImageDraw.Draw(a).ellipse((20,20,35,35), fill=(0,0,0,0))
        b=Image.new('RGBA',(60,60),(20,30,40,255))
        ImageDraw.Draw(b).ellipse((21,20,36,35), fill=(0,0,0,0))
        self.assertTrue(self.compare(a,b)['accepted'])
        self.assertFalse(self.compare(a,Image.new('RGBA',(60,60),(20,30,40,255)))['accepted'])

    def test_no_coverage_change_does_not_reject_recolor(self):
        self.assertTrue(self.compare(Image.new('RGBA',(10,10),'red'),Image.new('RGBA',(10,10),'blue'))['accepted'])

    def test_two_existing_holes_cannot_be_merged(self):
        a=Image.new('RGBA',(60,60),'red')
        draw=ImageDraw.Draw(a)
        draw.rectangle((10,20,20,30),fill=(0,0,0,0))
        draw.rectangle((30,20,40,30),fill=(0,0,0,0))
        b=a.copy()
        ImageDraw.Draw(b).rectangle((20,24,30,26),fill=(0,0,0,0))
        self.assertFalse(self.compare(a,b)['accepted'])

    def test_white_shape_cannot_disappear_under_exact_white_background_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b = Path(directory)/'a.svg',Path(directory)/'b.svg'
            a.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><rect x="4" y="4" width="12" height="12" fill="white"/></svg>')
            b.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"/>')
            result=validate_svg_stage_renders(a,b,'shape_exact',render_size=100)
            self.assertEqual(result['validation_background'],'transparent')
            self.assertFalse(result['accepted'])

    def test_workbench_port_cannot_be_shared_by_two_workspaces(self):
        from workbench import ThreadingHTTPServer, Handler
        first=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        try:
            with self.assertRaises(OSError):
                second=ThreadingHTTPServer(first.server_address,Handler)
                second.server_close()
        finally:
            first.server_close()


if __name__=='__main__':
    unittest.main()
