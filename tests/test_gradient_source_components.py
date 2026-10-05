"""Native-rendered regression cases for source-supported enclosed ownership."""
import copy
import math
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from gradient_object_engine import fit_gradient_object_proposal
from gradient_reconstruction_stage import encode_mask_rle, decode_mask_rle, _fit_geometry
from gradient_source_components import (enclosed_components,
    propose_enclosed_source_components, apply_source_component_candidates,
    final_source_component_matches, _SpanSceneBudget)
from svg_renderer import render_svg_reference


class GradientSourceComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        points=[(63.5+42.4*math.cos(i*math.tau/128),63.5+42.4*math.sin(i*math.tau/128)) for i in range(128)]
        self.outer='M'+' L'.join(f'{x:.6f} {y:.6f}' for x,y in points)+' Z'
        self.hole='M63 60 H65 V62 H63 Z'
        self.model={'type':'linear','x1':16.,'y1':64.,'x2':112.,'y2':64.,
                    'direction':[1.,0.],'stop_count':2}
        self.stops=[{'offset':0.,'color':'#208060','rgb':[32,128,96]},
                    {'offset':1.,'color':'#70b090','rgb':[112,176,144]}]
        self.truth=self.document('M22 64 A42 42 0 1 1 106 64 A42 42 0 1 1 22 64 Z')
        path=self.folder/'truth.svg';path.write_text(self.truth,encoding='utf8')
        alpha=self.folder/'alpha.png';render_svg_reference(path,alpha,128,background=None)
        self.mask=np.asarray(Image.open(alpha).convert('RGBA'))[:,:,3]>=128
        self.mask[60:62,63:65]=False
        self.source=self.folder/'source.png';render_svg_reference(path,self.source,128,background='#ffffff')
        self.raw=np.asarray(Image.open(self.source).convert('RGBA'))
        self.rgb=self.raw[:,:,:3]
        geometry,reasons=_fit_geometry(self.mask,(22,22,106,106),error_budget_percent=.25,
            smooth=.55,max_segments=4096,geometry_optimizer=None)
        self.assertIsNotNone(geometry,reasons)
        self.baseline_path=geometry['path']
        self.proposal={'candidate_id':'candidate-one','proposal_id':'object-one',
                       'mask':encode_mask_rle(self.mask),'model':self.model,'stops':self.stops,
                       'bbox_xyxy':[22,22,106,106],'geometry':geometry}

    def document(self,path,extra=''):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">'
                '<defs><linearGradient id="paint" gradientUnits="userSpaceOnUse" x1="16" y1="64" x2="112" y2="64">'
                '<stop offset="0" stop-color="#208060"/><stop offset="1" stop-color="#70b090"/>'
                '</linearGradient></defs><g fill="url(#paint)">'
                f'<path id="drawable-one" data-avc-gradient-object="object-one" fill-rule="evenodd" d="{path}"/>'
                '</g>'+extra+'</svg>')

    def propose(self,raw=None,rgb=None,other=None):
        return propose_enclosed_source_components(self.proposal,self.rgb if rgb is None else rgb,
            self.raw if raw is None else raw,np.ones_like(self.mask),
            np.zeros_like(self.mask) if other is None else other,
            alpha=np.full(self.mask.shape,255,np.uint8),labels=np.zeros(self.mask.shape,np.int32),
            model_fit_options={},geometry_error_percent=.25,geometry_smooth=.55,max_segments=4096,
            model_fitter=fit_gradient_object_proposal)

    def test_eight_connected_background_preserves_diagonal_openings_and_external_marks(self):
        mask=np.zeros((12,12),bool);mask[2:10,2:10]=True;mask[5,5]=False
        self.assertEqual([int(m.sum()) for m in enclosed_components(mask)],[1])
        for index in range(2,6):mask[index,index]=False
        self.assertEqual(enclosed_components(mask),[])
        mask[0,11]=True
        self.assertEqual(enclosed_components(mask),[])

    def test_span_search_budget_is_shared_across_objects_and_elapsed_work(self):
        now=[100.];budget=_SpanSceneBudget(clock=lambda:now[0])
        self.assertEqual(budget.remaining(),(8,45.))
        self.assertTrue(budget.consume({'native_probe_count':3},8))
        now[0]+=10.
        self.assertEqual(budget.remaining(),(5,35.))
        # Time spent on later object geometry does not restart the allowance.
        now[0]+=36.
        self.assertEqual(budget.remaining(),(5,0.))
        exhausted=_SpanSceneBudget(clock=lambda:now[0]);exhausted.remaining()
        self.assertFalse(exhausted.consume({},8))
        self.assertEqual(exhausted.remaining()[0],0)

    def test_original_white_alpha_and_other_owner_are_protected(self):
        white=self.raw.copy();white[60:62,63:65,:3]=255
        self.assertIsNone(self.propose(raw=white))
        transparent=self.raw.copy();transparent[60:62,63:65,3]=0
        self.assertIsNone(self.propose(raw=transparent))
        owner=np.zeros_like(self.mask);owner[60:62,63:65]=True
        self.assertIsNone(self.propose(other=owner))
        coloured=self.raw.copy();coloured[60:62,63:65,:3]=[250,20,20]
        self.assertIsNone(self.propose(raw=coloured))

    def test_real_proposal_keeps_baseline_and_passes_two_independent_holdouts(self):
        before=copy.deepcopy(self.proposal)
        pending=self.propose()
        self.assertIsNotNone(pending)
        self.assertEqual(pending['status'],'pending_native_scene_transaction')
        self.assertEqual(self.proposal,before)
        self.assertEqual(len(pending['independent_augmented_paint_revalidation']),2)
        self.assertTrue(all(r['passed'] for r in pending['independent_augmented_paint_revalidation']))
        self.assertEqual(int((decode_mask_rle(pending['mask'])&~self.mask).sum()),4)
        self.assertLess(pending['geometry']['designer_anchor_count'],self.proposal['geometry']['designer_anchor_count'])
        original_loops=re.findall(r'M[^M]*',self.baseline_path)
        retained_loops=re.findall(r'M[^M]*',pending['pure_fill_geometry']['path'])
        self.assertEqual(retained_loops,[original_loops[0]])
        self.assertTrue(pending['pure_fill_geometry']['selection_evidence']['outer_and_retained_hole_curves_exact'])
        self.assertLessEqual(pending['proposal_generation']['independent_paint_fit_calls'],2)
        self.assertEqual(pending['proposal_generation']['geometry_fit_calls'],0)

    def test_actual_scene_commit_binds_inherited_paint_source_geometry_and_canvas(self):
        pending=self.propose();self.assertIsNotNone(pending)
        region={'candidate_id':'candidate-one','mask':self.mask,'enclosed_source_component_candidate':pending}
        before=self.document(self.baseline_path)
        after,updates,audit=apply_source_component_candidates(before,[region],self.source,self.raw)
        self.assertEqual(audit['status'],'committed',audit)
        geometry=updates['candidate-one']['geometry'];root=ET.fromstring(after)
        self.assertTrue(final_source_component_matches(root,geometry,self.source))
        certificate=geometry['source_ownership_completion']
        accounting=certificate['designer_anchor_accounting']
        self.assertEqual(accounting['hole_anchors_removed'],8)
        self.assertEqual(accounting['curve_anchors_removed'],0)
        self.assertEqual(certificate['geometry_transactions'],[])
        self.assertTrue(all(row['worse_pixel_count']==0 for row in certificate['actual_component_measurements']))
        for key,value in [('viewBox','0 0 256 256'),('display','none')]:
            modified=ET.fromstring(after);modified.set(key,value)
            self.assertFalse(final_source_component_matches(modified,geometry,self.source))
        modified=ET.fromstring(after)
        target=next(n for n in modified.iter() if n.get('id')=='drawable-one')
        target.set('d',target.get('d')+' M1 1L2 1L2 2Z')
        self.assertFalse(final_source_component_matches(modified,geometry,self.source))
        changed=self.raw.copy();changed[61,64,0]+=1
        wrong=self.folder/'different.png';Image.fromarray(changed).save(wrong)
        self.assertFalse(final_source_component_matches(root,geometry,wrong))
        modified=copy.deepcopy(geometry)
        modified['source_ownership_completion']['actual_component_measurements'][0]['worse_pixel_count']=1
        self.assertFalse(final_source_component_matches(root,modified,self.source))
        modified=copy.deepcopy(geometry)
        modified['source_ownership_completion']['designer_anchor_accounting']['curve_anchors_removed']=2
        self.assertFalse(final_source_component_matches(root,modified,self.source))

    def test_changed_processed_source_cannot_reuse_pending_source_proof(self):
        pending=self.propose();self.assertIsNotNone(pending)
        region={'candidate_id':'candidate-one','mask':self.mask,'enclosed_source_component_candidate':pending}
        processed=self.raw.copy();processed[60:62,63:65,:3]=255
        with patch('gradient_reconstruction_stage._fit_geometry',side_effect=ValueError('unexpected refit')) as fitter:
            _,updates,audit=apply_source_component_candidates(self.document(self.baseline_path),
                [region],self.source,processed)
            fitter.assert_not_called()
        self.assertEqual(updates,{})
        self.assertEqual(audit['status'],'no_change')
        self.assertTrue(all('pending_source_or_baseline_identity_mismatch' in row['reasons'] for row in audit['transactions']))

    def test_optional_refit_failure_preserves_the_proven_exact_ownership_repair(self):
        pending=self.propose();self.assertIsNotNone(pending)
        region={'candidate_id':'candidate-one','mask':self.mask,'enclosed_source_component_candidate':pending}
        with patch('gradient_reconstruction_stage._fit_geometry',side_effect=ValueError('bounded fitting unavailable')):
            after,updates,audit=apply_source_component_candidates(self.document(self.baseline_path),
                [region],self.source,self.raw)
        self.assertEqual(audit['status'],'committed')
        geometry=updates['candidate-one']['geometry']
        self.assertEqual(geometry['path'],pending['pure_fill_geometry']['path'])
        self.assertEqual(geometry['source_ownership_completion']['designer_anchor_accounting']['curve_anchors_removed'],0)
        self.assertTrue(final_source_component_matches(ET.fromstring(after),geometry,self.source))

    def test_pure_fill_cannot_claim_exact_contours_for_a_different_scene_path(self):
        pending=self.propose();self.assertIsNotNone(pending)
        region={'candidate_id':'candidate-one','mask':self.mask,'enclosed_source_component_candidate':pending}
        with patch('gradient_reconstruction_stage._fit_geometry',side_effect=ValueError('unexpected refit')) as fitter:
            _,updates,audit=apply_source_component_candidates(
                self.document(self.baseline_path+' M1 1L2 1L2 2Z'),[region],self.source,self.raw)
            fitter.assert_not_called()
        self.assertFalse(updates)
        self.assertIn('exact_baseline_path_identity_mismatch',audit['transactions'][0]['reasons'])

    def test_span_fallback_reuses_validated_bytes_after_namespace_registration_changes(self):
        import hashlib
        from gradient_contour_spans import SpanSearchRejected
        pending=self.propose();self.assertIsNotNone(pending)
        region={'candidate_id':'candidate-one','mask':self.mask,'enclosed_source_component_candidate':pending}
        trial=copy.deepcopy(pending['pure_fill_geometry'])
        trial['designer_anchor_count']-=1
        captured=[]
        def reject_whole(*args,**kwargs):
            ET.register_namespace('changedsvg','http://www.w3.org/2000/svg')
            return {'external_render_check':'completed','alpha_topology':{'accepted':False},
                    'composed_alpha':{'accepted':False},'ink_recall_percent':100.,
                    'ink_precision_percent':100.,'ink_f1_percent':100.}
        def reject_span(svg_text,*args,**kwargs):
            captured.append(hashlib.sha256(svg_text.encode()).hexdigest())
            raise SpanSearchRejected('no_safe_contour_span',{'performance':{'native_probe_count':1}})
        with patch.dict(ET._namespace_map,dict(ET._namespace_map),clear=True), \
                patch('gradient_reconstruction_stage._fit_geometry',return_value=(trial,[])), \
                patch('vector_cleanroom.validate_svg_stage_renders',side_effect=reject_whole), \
                patch('gradient_contour_spans.propose_gradient_contour_spans',side_effect=reject_span):
            _,updates,audit=apply_source_component_candidates(self.document(self.baseline_path),
                [region],self.source,self.raw)
        cert=updates['candidate-one']['source_ownership_completion']
        self.assertEqual(captured,[cert['ownership_commit_svg_sha256']])
        self.assertEqual(audit['performance']['bounded_span_search']['native_probe_count'],1)

    def test_already_correct_baseline_is_not_replaced_with_worse_geometry(self):
        pending=self.propose();self.assertIsNotNone(pending)
        # A real source-coloured overlay already covers the nominal hole. The
        # ownership proposal may not use the hypothetical white gap as baseline.
        overlay='<path fill="url(#paint)" d="M62 59 H66 V63 H62 Z"/>'
        before=self.document(self.baseline_path,overlay)
        region={'candidate_id':'candidate-one','mask':self.mask,'enclosed_source_component_candidate':pending}
        _,updates,audit=apply_source_component_candidates(before,[region],self.source,self.raw)
        for update in updates.values():
            for row in update['source_ownership_completion']['actual_scene_measurements'].values():
                self.assertLessEqual(row['after_global_rgb_mae'],row['before_global_rgb_mae']+1e-9)
        # Either a pixel-identical fill is harmless, or a contour change must be rejected.
        self.assertIn(audit['status'],('committed','no_change'))


if __name__=='__main__':unittest.main()
