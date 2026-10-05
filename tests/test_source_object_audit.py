import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import source_object_audit as audit
from designer_handoff import build_handoff_manifest
from source_scene_guard import _render_native_payload
from svg_renderer import svg_with_native_viewport

NS = 'http://www.w3.org/2000/svg'


class SourceObjectAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.svg = self.folder/'art.svg'
        self.png = self.folder/'source_original.png'
        audit._CACHE.clear()

    def run_case(self, original, candidate=None, *, view='0 0 100 100', size=(100, 100)):
        wrap = lambda body: f'<svg xmlns="{NS}" viewBox="{view}">{body}</svg>'
        source = _render_native_payload(wrap(original).encode(), *size)
        Image.fromarray(source).save(self.png)
        self.svg.write_text(wrap(candidate if candidate is not None else original), encoding='utf8')
        return self.manifest()

    def manifest(self):
        return build_handoff_manifest(self.svg, {'_handoff_reference_kind':'original'}, source_png=self.png)

    def rows(self, result):
        self.assertEqual(result['source_object_audit']['status'], 'completed', result['source_object_audit'])
        return {row['id']:row for row in result['source_object_audit']['objects']}

    def test_correct_subpixel_translucent_thin_line_needs_no_eroded_core(self):
        result = self.run_case('<path id="thin" d="M12.3 13.2L88.1 83.6" stroke="#bb792e" stroke-width=".7" opacity=".45"/>')
        row = self.rows(result)['thin']
        self.assertGreater(row['visible_pixel_count'], 40)
        self.assertEqual(row['status'], 'checked')
        self.assertEqual(row['flags'], [])
        self.assertEqual(result['default_decisions']['thin'], 'review')

    def test_wrong_colour_thin_line_is_not_lost_for_lack_of_core(self):
        original = '<path id="thin" d="M10 25.5H90" fill="none" stroke="#b86f24" stroke-width="1"/>'
        result = self.run_case(original, original.replace('#b86f24', '#247fb8'))
        self.assertIn('appearance_mismatch', self.rows(result)['thin']['flags'])

    def test_wide_trace_of_flat_thin_line_is_not_called_missing_gradient(self):
        original = '<path id="thin" d="M10 25.2H90" fill="none" stroke="#b86f24" stroke-width="2"/>'
        result = self.run_case(original, original.replace('stroke-width="2"', 'stroke-width="5"'))
        flags = self.rows(result)['thin']['flags']
        self.assertIn('paint_on_source_blank', flags)
        self.assertNotIn('missing_tonal_variation', flags)

    def test_real_white_object_and_translucent_overlay_on_coloured_background(self):
        body = '<rect id="back" width="100" height="100" fill="#346397"/><rect id="white" x="10" y="10" width="30" height="40" fill="white"/><g opacity=".4"><rect id="overlay" x="25" y="25" width="50" height="50" fill="#e38854"/></g>'
        rows = self.rows(self.run_case(body))
        self.assertTrue(all(row['flags'] == [] for row in rows.values()))
        self.assertEqual(rows['white']['status'], 'checked')

    def test_fully_occluded_wrong_colour_is_not_attributed(self):
        body = '<rect id="hidden" x="10" y="10" width="50" height="50" fill="red"/><rect id="front" width="100" height="100" fill="blue"/>'
        rows = self.rows(self.run_case(body))
        self.assertEqual(rows['hidden']['status'], 'no_measurable_visible_contribution')
        self.assertFalse(rows['hidden']['flags'])

    def test_empty_hole_inside_object_bbox_does_not_inherit_other_objects_error(self):
        body = '<path id="ring" d="M5 5H95V95H5Z M20 20V80H80V20Z" fill-rule="evenodd" fill="#336633"/><rect id="middle" x="35" y="35" width="30" height="30" fill="blue"/>'
        rows = self.rows(self.run_case(body, body.replace('fill="blue"', 'fill="red"')))
        self.assertFalse(rows['ring']['flags'])
        self.assertIn('appearance_mismatch', rows['middle']['flags'])

    def test_partial_occlusion_ignores_error_outside_visible_support(self):
        body = '<rect id="back" x="10" y="10" width="80" height="80" fill="red"/><rect id="front" x="50" y="10" width="40" height="80" fill="blue"/>'
        rows = self.rows(self.run_case(body, body.replace('fill="blue"', 'fill="green"')))
        self.assertFalse(rows['back']['flags'])
        self.assertIn('appearance_mismatch', rows['front']['flags'])

    def test_group_mapping_and_clip_preserve_visible_members(self):
        body = '<defs><clipPath id="clip"><circle cx="50" cy="50" r="30"/></clipPath></defs><g id="object-pair" clip-path="url(#clip)" opacity=".7"><rect id="a" width="50" height="100" fill="red"/><rect id="b" x="50" width="50" height="100" fill="blue"/></g>'
        result = self.run_case(body, body.replace('fill="red"', 'fill="green"'))
        rows = self.rows(result)
        self.assertEqual(list(rows), ['object-pair'])
        self.assertEqual(rows['object-pair']['member_ids'], ['a', 'b'])
        self.assertIn('appearance_mismatch', rows['object-pair']['flags'])

    def test_gradient_is_hint_only_when_visible_tone_was_lost(self):
        defs = '<defs><linearGradient id="ramp"><stop stop-color="#246020"/><stop offset="1" stop-color="#aed950"/></linearGradient></defs>'
        original = defs + '<rect id="leaf" x="10" y="10" width="80" height="80" fill="url(#ramp)"/>'
        self.assertFalse(self.rows(self.run_case(original))['leaf']['flags'])
        result = self.run_case(original, defs + '<rect id="leaf" x="10" y="10" width="80" height="80" fill="#699c38"/>')
        self.assertIn('missing_tonal_variation', self.rows(result)['leaf']['flags'])
        self.assertEqual(result['default_decisions']['leaf'], 'review')

    def test_nonsquare_rounded_native_canvas_crop_is_identical_to_native_render(self):
        body = '<g transform="translate(7.2,3.1)"><path id="thin" d="M4 12L80 49" stroke="#87392a" stroke-width="1.2"/></g>'
        result = self.run_case(body, view='0 0 100 53', size=(211, 112))
        self.assertFalse(self.rows(result)['thin']['flags'])
        native = ET.fromstring(svg_with_native_viewport(self.svg.read_text(), 211, 112))
        full = audit._render(native, 211, 112)
        box = (9, 5, 187, 103)
        part = audit._render(copy.deepcopy(native), 211, 112, box)
        self.assertTrue(np.array_equal(full[5:103, 9:187], part))

    def test_paint_on_transparent_source_is_not_equated_with_real_white_paint(self):
        original = '<rect id="good" x="10" y="10" width="30" height="20" fill="white"/>'
        candidate = original + '<rect id="extra" x="60" y="10" width="20" height="20" fill="#a0b29c"/>'
        rows = self.rows(self.run_case(original, candidate))
        self.assertFalse(rows['good']['flags'])
        self.assertIn('paint_on_source_blank', rows['extra']['flags'])

    def test_content_cache_reuses_exact_input_but_rechecks_changed_source(self):
        body = '<rect id="box" x="10" y="10" width="40" height="40" fill="red"/>'
        result = self.run_case(body)
        with mock.patch.object(audit, '_render', side_effect=AssertionError('cache should avoid render')):
            self.assertEqual(self.manifest(), result)
        Image.new('RGBA', (100, 100), 'blue').save(self.png)
        changed = self.manifest()
        self.assertNotEqual(changed['source_object_audit']['source_sha256'], result['source_object_audit']['source_sha256'])
        self.assertIn('appearance_mismatch', self.rows(changed)['box']['flags'])

    def test_semantically_identical_svg_byte_change_does_not_reuse_old_fingerprint(self):
        first = self.run_case('<rect id="box" width="100" height="100" fill="red"/>')
        self.svg.write_text(self.svg.read_text().replace('<svg ', '<svg  '), encoding='utf8')
        second = self.manifest()
        self.assertNotEqual(first['svg_sha256'], second['svg_sha256'])
        self.assertEqual(second['svg_sha256'], second['source_object_audit']['svg_sha256'])

    def test_mutating_returned_evidence_cannot_poison_cached_result(self):
        result = self.run_case('<rect id="box" width="100" height="100" fill="red"/>')
        result['source_object_audit']['objects'][0]['flags'].append('invented')
        self.assertNotIn('invented', self.rows(self.manifest())['box']['flags'])

    def test_aspect_mismatch_and_render_failure_stay_unchecked(self):
        self.run_case('<rect id="box" width="100" height="100" fill="red"/>')
        Image.new('RGBA', (100, 80), 'red').save(self.png)
        self.assertEqual(self.manifest()['source_object_audit']['reason'], 'source_svg_aspect_ratio_differs')
        Image.new('RGBA', (100, 100), 'red').save(self.png)
        audit._CACHE.clear()
        with mock.patch.object(audit, '_render', side_effect=RuntimeError('render failed')):
            failed = self.manifest()
        self.assertEqual(failed['source_object_audit']['status'], 'unavailable')
        self.assertEqual(failed['objects'][0]['source_audit_status'], 'unavailable')
        self.assertEqual(self.manifest()['source_object_audit']['status'], 'completed')

    def test_failed_budget_audit_not_reported_as_no_defects_or_cached_success(self):
        body = '<rect id="box" width="100" height="100" fill="red"/>'
        self.run_case(body)
        audit._CACHE.clear()
        with mock.patch.object(audit, 'MAX_PIXELS', 20):
            failed = self.manifest()
            self.assertEqual(failed['source_object_audit']['status'], 'unavailable')
            self.assertEqual(failed['objects'][0]['source_audit_status'], 'unavailable')
        self.assertEqual(self.manifest()['source_object_audit']['status'], 'completed')

    def test_processed_reference_is_never_claimed_as_original_audit(self):
        self.run_case('<rect id="box" width="100" height="100" fill="red"/>')
        result = build_handoff_manifest(self.svg, {'_handoff_reference_kind':'processed_reference'}, source_png=self.png)
        self.assertEqual(result['source_object_audit']['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
