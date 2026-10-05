"""A failed or stale native audit must not become a designer-ready claim."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from vector_cleanroom import _bind_delivered_source_audits

from designer_quality import audit_designer_quality


class SourceReadinessBinding(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.svg = Path(self.tmp.name) / 'shape.svg'
        self.svg.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="20" fill="#164d34"/></svg>')
        self.evidence = {'schema': 'aivc.source-topology-audit/v1',
            'status': 'completed', 'native_resolution': True, 'manual_review': False,
            'inputs_unchanged': True,
            'stable_defects': [], 'reasons': [],
            'source_hashes': {'svg': hashlib.sha256(self.svg.read_bytes()).hexdigest()}}

    def audit(self, report, *, changed=False):
        proposal = {'source_topology_report': report}
        if changed:
            proposal['editability_enhancements'] = {'stages': {
                'source_reconstruction': {'status': 'committed'}}}
        return audit_designer_quality(self.svg, proposal_metadata=proposal)

    def test_verified_original_bytes_pass_but_edited_shape_cannot_reuse_evidence(self):
        self.assertEqual(self.audit(self.evidence)['source_topology_gate']['status'], 'passed')
        self.svg.write_bytes(self.svg.read_bytes().replace(b'r="20"', b'r="18"'))
        result = self.audit(self.evidence)
        self.assertFalse(result['designer_ready'])
        self.assertIn('source_topology_svg_evidence_stale', result['source_topology_gate']['reasons'])

    def test_unavailable_timeout_and_partial_reports_require_review(self):
        for report in ({'status': 'unavailable', 'manual_review': False,
                        'reasons': ['native_topology_audit_budget_exhausted']},
                       {'status': 'completed', 'manual_review': False},
                       dict(self.evidence, native_resolution=False)):
            with self.subTest(report=report):
                self.assertFalse(self.audit(report)['designer_ready'])

    def test_source_reconstruction_requires_its_final_source_audit(self):
        for report in (None, {'status': 'not_applicable'}):
            with self.subTest(report=report):
                self.assertFalse(self.audit(report, changed=True)['designer_ready'])

    def test_not_applicable_does_not_claim_paper_verification(self):
        result = self.audit({'status': 'not_applicable', 'reasons': ['native_alpha_source']})
        self.assertEqual(result['source_topology_gate']['status'], 'not_applicable')

    def test_inconsistent_clean_flag_cannot_hide_local_defects(self):
        report = copy.deepcopy(self.evidence)
        report['stable_defects'] = [{'kind': 'source_components_merged'}]
        self.assertFalse(self.audit(report)['designer_ready'])

    def test_missing_native_stroke_count_is_not_zero(self):
        for evidence in ({}, {'complex_strokes_without_native_cap_proof': False},
                         {'complex_strokes_without_native_cap_proof': -1}):
            with self.subTest(evidence=evidence):
                result = audit_designer_quality(self.svg, proposal_metadata={
                    'stroke_reconstruction_report': evidence})
                self.assertFalse(result['designer_ready'])


class FinalDeliveryBinding(unittest.TestCase):
    def setUp(self):
        self.before = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="20" height="10" fill="#164d34"/></svg>'
        self.sha = hashlib.sha256(self.before).hexdigest()
        self.audit = {'source_hashes': {'svg': self.sha},
                      'renderer': {'source_svg_sha256': self.sha}}
        self.quality = {'source': {'sha256': self.sha},
                        'source_topology_gate': {'evidence': self.audit}}
        self.enhancements = {'stages': {'source_topology_audit': self.audit}}
        self.after = self.before.replace(b'<rect', b'\n<metadata id="ai-vector-cleanroom-metadata">{}</metadata>\n<rect')

    def test_inert_annotation_rebinds_final_bytes_but_keeps_actual_render_input(self):
        binding = _bind_delivered_source_audits(self.before, self.after,
                                               self.quality, self.enhancements)
        final = hashlib.sha256(self.after).hexdigest()
        self.assertEqual(self.quality['source']['sha256'], final)
        self.assertEqual(self.audit['source_hashes']['svg'], final)
        self.assertEqual(self.audit['renderer']['source_svg_sha256'], self.sha)
        self.assertEqual(binding['audit_input_svg_sha256'], self.sha)

    def test_geometry_or_paint_change_cannot_reuse_source_check(self):
        for changed in (self.after.replace(b'width="20"', b'width="19"'),
                        self.after.replace(b'#164d34', b'#ffffff')):
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, 'artwork_changed'):
                _bind_delivered_source_audits(self.before, changed, self.quality, self.enhancements)

    def test_already_stale_report_is_not_rebound(self):
        self.quality['source']['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'designer_audit_does_not_match'):
            _bind_delivered_source_audits(self.before, self.after, self.quality, self.enhancements)


if __name__ == '__main__':
    unittest.main()
