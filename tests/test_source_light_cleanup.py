"""Native-rendered source-light proposals; opacity alone never deletes white."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from source_light_cleanup import propose_light_fill_cleanup, apply_light_fill_candidate
from svg_renderer import render_svg_reference


class SourceLightCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def document(self, body):
        return '<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">'+body+'</svg>'

    def band(self, identifier='artifact', fill='#dde8e2', y=42):
        return f'<path id="{identifier}" fill="{fill}" d="M30 {y}H92V{y+2}H30Z"/>'

    def native(self, text, name, background=None):
        svg = self.folder/(name+'.svg')
        svg.write_text(text, encoding='utf8')
        png = svg.with_suffix('.png')
        render_svg_reference(svg, png, width=128, background=background)
        with Image.open(png) as image:
            return png, np.asarray(image.convert('RGBA'))

    def reference(self, text):
        source, pixels = self.native(text, 'source', '#ffffff')
        processed = pixels.copy()
        processed[:, :, 3] = np.where(pixels[:, :, :3].min(2) >= 235, 0, 255)
        return source, pixels, processed

    def proposals(self, candidate, truth, **kwargs):
        source, original, processed = self.reference(truth)
        report = propose_light_fill_cleanup(candidate, source, processed, **kwargs)
        return report, source, original, processed

    def test_open_paper_thin_false_tint_is_removable_and_neighbor_is_unchanged(self):
        neighbor = '<path id="ink" fill="#204830" d="M20 60H108V64H20Z"/>'
        truth = self.document(neighbor)
        candidate = self.document(neighbor+self.band())
        report, source, _, processed = self.proposals(candidate, truth)
        self.assertEqual(len(report['proposals']), 1, report)
        proposal = report['proposals'][0]
        self.assertEqual(proposal['operation'], 'remove_drawable')
        self.assertTrue(proposal['downstream_source_scene_guard_required'])
        before_source = source.read_bytes()
        before_processed = processed.copy()
        output = apply_light_fill_candidate(candidate, proposal)
        self.assertNotIn('id="artifact"', output)
        a = next(n for n in ET.fromstring(candidate).iter() if n.get('id') == 'ink')
        b = next(n for n in ET.fromstring(output).iter() if n.get('id') == 'ink')
        self.assertEqual(ET.tostring(a), ET.tostring(b))
        self.assertEqual(source.read_bytes(), before_source)
        self.assertTrue(np.array_equal(processed, before_processed))

    def test_closed_counter_is_recolored_without_deleting_white_geometry(self):
        dark = '<path id="body" fill="#204830" d="M20 20H108V100H20Z"/>'
        truth = self.document(dark+self.band(fill='#ffffff', y=60))
        candidate = self.document(dark+self.band(y=60))
        report, _, _, _ = self.proposals(candidate, truth)
        self.assertEqual(len(report['proposals']), 1, report)
        proposal = report['proposals'][0]
        self.assertEqual(proposal['operation'], 'recolor_paper')
        self.assertEqual(proposal['reason'], 'white_object_vs_negative_space_unresolved')
        self.assertGreater(proposal['source_classification']['enclosed_paper_core_pixels'], 0)
        output = apply_light_fill_candidate(candidate, proposal)
        _, before = self.native(candidate, 'before')
        _, after = self.native(output, 'after')
        self.assertTrue(np.array_equal(before[:, :, 3], after[:, :, 3]))
        old = next(n for n in ET.fromstring(candidate).iter() if n.get('id') == 'artifact')
        new = next(n for n in ET.fromstring(output).iter() if n.get('id') == 'artifact')
        self.assertEqual(old.get('d'), new.get('d'))
        self.assertEqual(new.get('fill'), '#ffffff')

    def test_real_white_highlight_and_background_connected_white_object_survive(self):
        dark = '<path id="body" fill="#204830" d="M20 20H108V100H20Z"/>'
        for truth in (self.document(dark+self.band(fill='#ffffff', y=60)),
                      self.document(dark+self.band(fill='#ffffff', y=19))):
            with self.subTest(truth=truth):
                report, _, _, _ = self.proposals(truth, truth)
                self.assertEqual(report['proposals'], [])
                self.assertTrue(any(row['reason'] == 'white_or_paper_paint_preserved' for row in report['retained']))

    def test_processed_transparency_cannot_delete_real_pastel_source(self):
        truth = self.document(self.band())
        source, _, processed = self.reference(truth)
        processed[:, :, 3] = 0  # Simulate erroneous background removal.
        report = propose_light_fill_cleanup(truth, source, processed)
        self.assertEqual(report['proposals'], [])
        self.assertTrue(any(row['reason'] == 'original_supports_real_pastel_content' for row in report['retained']))

    def test_paper_majority_cannot_override_a_real_pastel_segment(self):
        truth = self.document('<path id="real" fill="#dde8e2" d="M48 42H68V44H48Z"/>')
        candidate = self.document(self.band())
        source, _, processed = self.reference(truth)
        processed[:, :, 3] = 0
        report = propose_light_fill_cleanup(candidate, source, processed)
        self.assertEqual(report['proposals'], [])
        self.assertTrue(any(row['reason'] == 'original_supports_real_pastel_content' for row in report['retained']))

    def test_absence_of_global_paper_or_native_transparent_source_abstains(self):
        truth = self.document('<rect width="128" height="128" fill="#233344"/>'+self.band())
        report, _, _, _ = self.proposals(truth, truth)
        self.assertEqual(report['proposals'], [])
        self.assertEqual(report['reason'], 'no_reliable_opaque_neutral_paper')
        source, pixels = self.native(self.document(self.band(fill='#ffffff')), 'transparent')
        report = propose_light_fill_cleanup(self.document(self.band()), source, pixels)
        self.assertEqual(report['proposals'], [])
        self.assertEqual(report['reason'], 'no_reliable_opaque_neutral_paper')

    def test_transform_reference_and_ambiguous_identity_abstain(self):
        source, _, processed = self.reference(self.document(''))
        cases = [self.document('<g transform="translate(0 0)">'+self.band()+'</g>'),
                 self.document(self.band()+'<use href="#artifact"/>'),
                 self.document(self.band()+self.band())]
        for candidate in cases:
            self.assertEqual(propose_light_fill_cleanup(candidate, source, processed)['proposals'], [])

    def test_stale_target_or_presentation_is_rejected_and_independent_units_can_apply(self):
        candidate = self.document('<g id="light">'+self.band('first')+self.band('second', y=52)+'</g>')
        report, _, _, _ = self.proposals(candidate, self.document(''))
        self.assertEqual(len(report['proposals']), 2, report)
        first, second = report['proposals']
        once = apply_light_fill_candidate(candidate, first)
        twice = apply_light_fill_candidate(once, second)
        self.assertNotIn('id="first"', twice)
        self.assertNotIn('id="second"', twice)
        for changed in (candidate.replace('M30 42', 'M31 42'), candidate.replace('id="light"', 'id="new-parent"')):
            with self.assertRaises(ValueError):
                apply_light_fill_candidate(changed, first)
        with self.assertRaises(ValueError):
            apply_light_fill_candidate(once, first)

    def test_zero_budget_and_alignment_errors_fail_closed(self):
        source, _, processed = self.reference(self.document(''))
        candidate = self.document(self.band())
        for options in ({'maximum_candidates': 0}, {'maximum_seconds': 0}):
            report = propose_light_fill_cleanup(candidate, source, processed, **options)
            self.assertEqual(report['proposals'], [])
            self.assertEqual(report['examined_candidates'], 0)
        with self.assertRaises(ValueError):
            propose_light_fill_cleanup(candidate, source, processed[:80])
        with self.assertRaises(ValueError):
            propose_light_fill_cleanup(candidate, source, processed, maximum_seconds=float('nan'))

    def test_processed_alpha_can_be_aligned_without_replacing_original_rgb(self):
        candidate = self.document(self.band())
        source, _, processed = self.reference(self.document(''))
        smaller = np.asarray(Image.fromarray(processed).resize((64, 64), Image.Resampling.NEAREST))
        report = propose_light_fill_cleanup(candidate, source, smaller)
        self.assertEqual(report['processed_alignment'], 'same_canvas_nearest_alpha')
        self.assertEqual(report['proposals'][0]['operation'], 'remove_drawable')

    def test_real_tea_six_reviewed_units_are_independently_triaged(self):
        root = Path(__file__).resolve().parents[4]
        fixture = root/'outputs/designer-verification-v2/release-tea-r4/result_tea'
        if not (fixture/'tea_vector.svg').is_file():
            self.skipTest('Private real tea release fixture is not distributed')
        vector = fixture/'tea_vector.svg'
        source = fixture/'source_original.png'
        original_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (vector, source)}
        with Image.open(fixture/'source_reference.png') as image:
            processed = np.asarray(image.convert('RGBA'))
        report = propose_light_fill_cleanup(vector.read_text(encoding='utf8'), source, processed)
        proposed = {row['drawable_id']: row for row in report['proposals']}
        retained = {row['drawable_id']: row for row in report['retained']}
        for identifier in ('sg-node-909da5c1df48', 'sg-node-dc4f618177cb', 'sg-node-741693beb12d'):
            self.assertEqual(proposed[identifier]['operation'], 'recolor_paper')
            self.assertTrue(proposed[identifier]['source_local_guard']['recolor_alpha_identical'])
            self.assertEqual(proposed[identifier]['reason'], 'white_object_vs_negative_space_unresolved')
        for identifier in ('sg-node-0e29185b33a7', 'sg-node-458034b54768', 'sg-node-6cff8e92662e'):
            self.assertIn('not_strictly_improved', retained[identifier]['reason'])
        self.assertFalse(any(row['operation'] == 'remove_drawable' for row in report['proposals']))
        self.assertEqual(original_hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (vector, source)})


if __name__ == '__main__':
    unittest.main()
