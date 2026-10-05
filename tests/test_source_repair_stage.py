"""Transactions must never publish a source repair rejected by scene evidence."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

from source_repair_stage import attempt_source_repairs, project_gradient_report, _small_source_contact_tail


class SourceRepairTransactions(unittest.TestCase):
    @staticmethod
    def _tail_fixture(counts):
        return {'accepted': False, 'source_boundary_evidence': {'verified': True},
                'reasons': list(counts),
                'defect_totals': [{'kind': kind, 'pixels': count} for kind, count in counts.items()],
                'localized_defects': [{'kind': kind, 'pixels': count} for kind, count in counts.items()]}

    def test_small_tail_allows_only_four_cumulative_local_colour_or_empty_paint_pixels(self):
        kinds=('new_local_color_error','new_paint_on_source_empty','new_color_on_source_empty')
        for kind in kinds:
            for count in (1,4):
                guard=self._tail_fixture({kind:count});saved=copy.deepcopy(guard)
                with self.subTest(kind=kind,count=count):
                    self.assertTrue(_small_source_contact_tail(guard))
                    self.assertEqual(guard,saved)
                    self.assertIs(guard['accepted'],False)  # eligibility never changes approval
        mixed=self._tail_fixture(dict(zip(kinds,(2,1,1))))
        self.assertTrue(_small_source_contact_tail(mixed))
        mixed['defect_totals'][0]['pixels']=3
        mixed['localized_defects'][0]['pixels']=3
        self.assertFalse(_small_source_contact_tail(mixed))  # five overall, though each kind <= 4
        self.assertFalse(_small_source_contact_tail(self._tail_fixture({kinds[0]:5})))
        # Multiple reported regions may partition one total without requiring
        # one region per kind; every reported pixel must still be accounted for.
        partitioned=self._tail_fixture({kinds[1]:4})
        partitioned['localized_defects']=[{'kind':kinds[1],'pixels':1},{'kind':kinds[1],'pixels':3}]
        self.assertTrue(_small_source_contact_tail(partitioned))

    def test_small_tail_rejects_unknown_topology_truncated_and_unverified_diagnostics(self):
        good=self._tail_fixture({'new_paint_on_source_empty':1})
        for reason in ('created_hole_on_source_ink','erased_hole_on_source_empty',
                       'source_component_merge','source_feature_color_error_increased','unknown'):
            for field in ('reasons','defect_totals','localized_defects'):
                guard=copy.deepcopy(good)
                if field=='reasons':guard[field].append(reason)
                else:guard[field][0]['kind']=reason
                with self.subTest(reason=reason,field=field):
                    self.assertFalse(_small_source_contact_tail(guard))
        for changes in ({'reasons':[]},{'localized_defects_truncated':True},
                        {'defect_totals':[]},{'localized_defects':[]},
                        {'source_boundary_evidence':{}},
                        {'source_boundary_evidence':{'verified':False}},
                        {'source_boundary_evidence':{'verified':'true'}}):
            with self.subTest(changes=changes):
                self.assertFalse(_small_source_contact_tail({**good,**changes}))

    def test_small_tail_requires_positive_integer_counts_and_complete_localization(self):
        good=self._tail_fixture({'new_color_on_source_empty':2})
        for field in ('defect_totals','localized_defects'):
            for count in (True,False,0,-1,2.,'2',None,float('nan'),float('inf')):
                guard=copy.deepcopy(good);guard[field][0]['pixels']=count
                with self.subTest(field=field,count=count):
                    self.assertFalse(_small_source_contact_tail(guard))
        for count in (1,3):
            guard=copy.deepcopy(good);guard['localized_defects'][0]['pixels']=count
            with self.subTest(localized_count=count):
                self.assertFalse(_small_source_contact_tail(guard))

    def test_extra_tail_attempt_never_includes_topology_or_truncated_regions(self):
        remaining = {'source_boundary_evidence': {'verified': True},
                     'reasons': ['new_local_color_error'],
                     'defect_totals': [{'kind': 'new_local_color_error', 'pixels': 2}],
                     'localized_defects': [{'kind': 'new_local_color_error', 'pixels': 2}]}
        self.assertTrue(_small_source_contact_tail(remaining))
        for overrides in ({'reasons': ['new_local_color_error', 'created_hole_on_source_ink']},
                          {'source_boundary_evidence': {'verified': False}},
                          {'localized_defects_truncated': True},
                          {'defect_totals': [{'kind': 'new_local_color_error', 'pixels': 8}]},
                          {'localized_defects': []}):
            with self.subTest(overrides=overrides):
                self.assertFalse(_small_source_contact_tail({**remaining, **overrides}))

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.svg = self.root / 'candidate.svg'
        self.svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
            '<defs><linearGradient id="g"><stop stop-color="#105020"/><stop offset="1" stop-color="#207030"/></linearGradient></defs>'
            '<path id="p" data-avc-gradient-object="owner" data-avc-designer-anchors="90" '
            'fill="url(#g)" d="M10 10 L50 10 L50 50 L10 50 Z"/></svg>',encoding='utf-8')
        self.original = self.svg.read_bytes()
        self.source = self.root / 'source.png'
        self.ref = self.root / 'reference.png'
        Image.new('RGBA',(64,64),'white').save(self.source)
        Image.new('RGBA',(64,64)).save(self.ref)
        self.details = [{'id':'g', 'candidate_id':'owner', 'validation':{
            'geometry':{'anchor_count':90}, 'paint':{'preserved':'yes'},
            'selection':{'colour_used_for_geometry':False}}}]

    def proposal(self,*args,**kwargs):
        root = ET.fromstring(args[0])
        path = next(n for n in root.iter() if n.get('id') == 'p')
        path.set('d','M10 10 H50 V50 H10 Z')
        path.set('data-avc-designer-anchors','4')
        return {'path':path.get('d'), 'candidate_svg_text':ET.tostring(root,encoding='unicode'),
            'geometry':{'anchor_count':4,'designer_anchor_count':4,
                'source_edge_reconstruction':{'gradient_id':'g','source_rgba_sha256':'source-proof'}}}

    def run_stage(self,guards):
        outcomes = iter(guards)
        with patch('source_edge_reconstruction.propose_source_edge_reconstruction',side_effect=self.proposal), \
             patch('source_flat_paint.propose_source_flat_paint',return_value={'proposals':[]}), \
             patch('source_light_cleanup.propose_light_fill_cleanup',return_value={'proposals':[]}), \
             patch('source_scene_guard.validate_source_scene',side_effect=lambda *a,**kw: next(outcomes)), \
             patch('source_scene_guard.validate_source_scene_chain',side_effect=lambda *a,**kw: next(outcomes)):
            return attempt_source_repairs(self.svg,self.source,self.ref,self.details)

    def test_aggregate_rejection_restores_exact_file_and_original_proof(self):
        saved = copy.deepcopy(self.details)
        report, proof = self.run_stage([
            {'accepted':True}, {'accepted':False,'reasons':['new_ink_hole']}])
        self.assertEqual(report['status'],'rolled_back')
        self.assertEqual(report['committed'],[])
        self.assertEqual(self.svg.read_bytes(),self.original)
        self.assertEqual(proof,saved)
        self.assertEqual(self.details,saved)

    def test_missing_scene_approval_is_not_a_pass(self):
        report, proof = self.run_stage([{}, {}, {}])
        self.assertEqual(report['status'],'unchanged')
        self.assertEqual(self.svg.read_bytes(),self.original)
        self.assertEqual(proof,self.details)

    def test_scene_approved_commit_updates_geometry_and_keeps_paint_proof(self):
        report, proof = self.run_stage([{'accepted':True},{'accepted':True}])
        self.assertEqual(report['status'],'committed')
        self.assertNotEqual(self.svg.read_bytes(),self.original)
        geometry = proof[0]['validation']['geometry']
        self.assertEqual(geometry['anchor_count'],4)
        self.assertTrue(geometry['source_edge_scene_commit']['accepted'])
        self.assertEqual(proof[0]['validation']['paint'],{'preserved':'yes'})
        self.assertEqual(self.details[0]['validation']['geometry'],{'anchor_count':90})

    def test_exception_does_not_replace_original(self):
        report, proof = self.run_stage([RuntimeError('renderer unavailable')])
        self.assertEqual(self.svg.read_bytes(),self.original)
        self.assertEqual(proof,self.details)
        self.assertEqual(report['attempts'][0]['status'],'error')

    def test_later_simplification_does_not_inherit_source_geometry_claim(self):
        from auto_prepare import _invalidate_metadata
        from local_refine import _derived_report
        tree = ET.fromstring('<svg><path id="changed" data-avc-source-edge="proof"/>'
                             '<path id="untouched" data-avc-source-edge="proof"/></svg>')
        _invalidate_metadata(tree,['changed'],'parent',.25)
        nodes = {n.get('id'):n for n in tree.iter() if n.get('id')}
        self.assertNotIn('data-avc-source-edge',nodes['changed'].attrib)
        self.assertEqual(nodes['untouched'].get('data-avc-source-edge'),'proof')
        stale = {'gradient_details':[{'proof':'old'}],
                 'gradient_geometry_consistency':{'status':'verified_unchanged'}}
        manifest = {'svg_sha256':'parent','objects':[]}
        result = _derived_report(stale,manifest,dict(manifest,svg_sha256='derived'),
                                 {'changed_member_ids':['changed']},.25,'source')
        self.assertIsNone(result['gradient_details'])
        self.assertIsNone(result['gradient_geometry_consistency'])
        self.assertEqual(result['acceptance_status'],'manual_review')

    def test_flat_projection_keeps_selection_history_and_remaining_identity(self):
        original = {'summary': {'objects_selected': 2}, 'decisions': [
            {'candidate_id': 'flat', 'status': 'selected'},
            {'candidate_id': 'real-gradient', 'status': 'selected'}]}
        result = project_gradient_report(original, [{'candidate_id': 'real-gradient'}],
            {'replaced_gradient_details': [{'candidate_id': 'flat'}]})
        self.assertEqual(result['summary']['objects_selected'], 1)
        self.assertEqual(result['source_repair_projection']['original_summary']['objects_selected'], 2)
        self.assertEqual(result['decisions'][0]['status'], 'replaced_with_source_solid')
        self.assertEqual(original['decisions'][0]['status'], 'selected')
        with self.assertRaisesRegex(ValueError, 'projection_mismatch'):
            project_gradient_report(original, [],
                {'replaced_gradient_details': [{'candidate_id': 'flat'}]})

    def test_flat_projection_never_promotes_retained_partial_paint_to_complete(self):
        partial = {'id': 'partial-g', 'candidate_id': 'partial-owner', 'validation': {
            'geometry': {'source_paint_only': {'partial_selection': True,
                'manual_review_required': True, 'drawable_records': [{'id': 'p'}]}}}}
        original = {'summary': {'objects_selected': 1, 'partial_paint_fields': 1},
                    'decisions': [{'candidate_id': 'flat', 'status': 'selected'},
                        {'candidate_id': 'partial-owner', 'status': 'geometry_rejected'},
                        {'candidate_id': 'partial-owner', 'status': 'partial_paint_selected'}]}
        saved = copy.deepcopy(partial)
        projected = project_gradient_report(original, [partial],
            {'replaced_gradient_details': [{'candidate_id': 'flat'}]})
        self.assertEqual(projected['summary']['objects_selected'], 0)
        self.assertEqual(projected['summary']['partial_paint_fields'], 1)
        self.assertEqual(projected['decisions'][1]['status'], 'geometry_rejected')
        self.assertEqual(projected['decisions'][2]['status'], 'partial_paint_selected')
        self.assertTrue(projected['source_repair_projection']['partial_paint_remains_manual_review'])
        self.assertEqual(partial, saved)
        self.assertEqual(original['summary']['objects_selected'], 1)
        invalid = copy.deepcopy(original)
        invalid['decisions'][2]['status'] = 'selected'
        with self.assertRaisesRegex(ValueError, 'projection_mismatch'):
            project_gradient_report(invalid, [partial],
                {'replaced_gradient_details': [{'candidate_id': 'flat'}]})

    def test_retiring_partial_definition_preserves_rejected_geometry_history(self):
        original = {'summary': {'objects_selected': 0, 'partial_paint_fields': 1},
                    'decisions': [{'candidate_id': 'partial-owner', 'status': 'geometry_rejected'},
                        {'candidate_id': 'partial-owner', 'status': 'partial_paint_selected'}]}
        projected = project_gradient_report(original, [],
            {'replaced_gradient_details': [{'candidate_id': 'partial-owner'}]})
        self.assertEqual(projected['summary']['objects_selected'], 0)
        self.assertEqual(projected['summary']['partial_paint_fields'], 0)
        self.assertEqual(projected['decisions'][0]['status'], 'geometry_rejected')
        self.assertEqual(projected['decisions'][1]['status'], 'replaced_with_source_solid')
        self.assertEqual(projected['decisions'][1]['previous_status'], 'partial_paint_selected')

    def test_shared_partial_paths_keep_exact_proof_while_other_gradient_becomes_solid(self):
        root = ET.fromstring(self.original)
        ns = '{http://www.w3.org/2000/svg}'
        defs = root.find(ns+'defs')
        partial_gradient = copy.deepcopy(defs[0]); partial_gradient.set('id', 'partial-g')
        defs.append(partial_gradient)
        for index in range(4):
            ET.SubElement(root, ns+'path', {'id': 'partial-'+str(index),
                'data-avc-gradient-object': 'partial-owner', 'data-avc-designer-anchors': '90',
                'fill': 'url(#partial-g)', 'd': f'M{index} 0L{index+1} 0L{index+1} 1Z'})
        before_partial = [ET.tostring(n) for n in root.iter() if (n.get('id') or '').startswith('partial-')]
        self.svg.write_text(ET.tostring(root, encoding='unicode'), encoding='utf8')
        original = self.svg.read_text(encoding='utf8')
        partial = {'id': 'partial-g', 'candidate_id': 'partial-owner', 'validation': {
            'geometry': {'source_paint_only': {'partial_selection': True,
                'gradient_object_id': 'partial-owner', 'manual_review_required': True,
                'drawable_records': [{'id': 'partial-'+str(i), 'anchors': 90} for i in range(4)]}}}}
        details = self.details + [partial]
        candidate_root = ET.fromstring(original)
        target = next(n for n in candidate_root.iter() if n.get('id') == 'p')
        target.set('fill', '#105020'); target.attrib.pop('data-avc-gradient-object')
        proposal = {'drawable_id': 'p', 'gradient_id': 'g', 'replacement_fill': '#105020',
                    'evidence': {}, 'roi_xyxy': [10,10,50,50],
                    'before_svg_sha256': hashlib.sha256(original.encode()).hexdigest(),
                    'svg_text': ET.tostring(candidate_root, encoding='unicode')}
        # An attractive sibling recolour must be deferred: its existing
        # four-path certificate cannot silently shrink to three records.
        unsafe_partial = {'drawable_id': 'partial-0', 'gradient_id': 'partial-g'}
        with patch('source_flat_paint.propose_source_flat_paint', side_effect=[
                {'proposals': [unsafe_partial, proposal]}, {'proposals': []}]), \
             patch('source_edge_reconstruction.propose_source_edge_reconstruction') as edge, \
             patch('source_light_cleanup.propose_light_fill_cleanup', return_value={'proposals': []}), \
             patch('source_scene_guard.validate_source_scene', return_value={'accepted': True}), \
             patch('source_scene_guard.validate_source_scene_chain', return_value={'accepted': True}):
            report, retained = attempt_source_repairs(self.svg,self.source,self.ref,details)
        edge.assert_not_called()
        self.assertEqual(report['status'], 'committed')
        self.assertEqual(retained, [partial])
        self.assertEqual(len(report['contour_deferred']), 4)
        self.assertTrue(all(item['reason'] == 'partial_existing_paths_not_complete_contour_ownership'
                            for item in report['contour_deferred']))
        self.assertEqual(report['attempts'][0]['reason'],
                         'partial_existing_paths_paint_certificate_requires_revalidation')
        after_partial = [ET.tostring(n) for n in ET.parse(self.svg).getroot().iter()
                         if (n.get('id') or '').startswith('partial-')]
        self.assertEqual(before_partial, after_partial)
        stage = {'summary': {'objects_selected': 1, 'partial_paint_fields': 1}, 'decisions': [
            {'candidate_id': 'owner', 'status': 'selected'},
            {'candidate_id': 'partial-owner', 'status': 'partial_paint_selected'}]}
        projected = project_gradient_report(stage, retained, report)
        self.assertEqual(projected['summary']['objects_selected'], 0)
        self.assertEqual(projected['summary']['partial_paint_fields'], 1)
        self.assertTrue(retained[0]['validation']['geometry']['source_paint_only']['manual_review_required'])

    def test_unverified_complex_stroke_is_not_claimed_designer_ready(self):
        from designer_quality import audit_designer_quality
        self.svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<path d="M10 10 L20 10 L20 20" fill="none" stroke="black"/></svg>', encoding='utf8')
        audit = audit_designer_quality(self.svg, proposal_metadata={
            'stroke_reconstruction_report': {'complex_strokes_without_native_cap_proof': 1}})
        self.assertFalse(audit['designer_ready'])
        self.assertEqual(audit['stroke_source_gate']['status'], 'manual_review')


if __name__ == '__main__':
    unittest.main()
