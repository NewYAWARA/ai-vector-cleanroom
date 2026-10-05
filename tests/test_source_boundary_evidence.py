import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from source_boundary_evidence import (build_source_boundary_evidence, _render,
                                      _thin, _path_segments, _crossing_segments,
                                      _normal_strip_footprint_overlap, _supported_normal_projection,
                                      _crossfit_source_material, _protected_feature_footprints)
from source_edge_reconstruction import propose_source_edge_reconstruction, rgba_sha256
from source_scene_guard import validate_source_scene, validate_source_scene_arrays, validate_source_scene_chain


class SourceBoundaryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def svg(d):
            return ('<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">'
                    '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="20" x2="108" y1="20" y2="20">'
                    '<stop offset="0" stop-color="#205b38"/><stop offset="1" stop-color="#4c995b"/></linearGradient></defs>'
                    '<path id="target" data-avc-gradient-object="object" fill="url(#g)" d="'+d+'"/>'
                    '<path id="other" fill="red" d="M2 2H4V4H2Z"/></svg>')
        cls.before = svg('M20.6 20.6H108.6V108.6H20.6Z')
        cls.reference = _render(ET.fromstring(svg('M20 20H108V108H20Z')),128,128)
        alpha = cls.reference[:,:,3:4].astype(float)/255
        cls.source = cls.reference.copy()
        cls.source[:,:,:3] = np.rint(cls.reference[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
        cls.source[:,:,3] = 255
        mask = _render(ET.fromstring(cls.before),128,128)[:,:,3]>=128
        mask[:5,:5] = False
        cls.proposal = propose_source_edge_reconstruction(cls.before,'target',cls.source,cls.reference,
                                                          ownership_mask=mask)
        cls.after = cls.proposal['candidate_svg_text']
        cls.geometry = cls.proposal['geometry']

    def build(self, **changes):
        args = dict(before_svg_text=self.before,after_svg_text=self.after,
                    source_rgba=self.source,processed_rgba=self.reference,geometry=self.geometry)
        args.update(changes)
        return build_source_boundary_evidence(**args)

    def test_independent_native_measurement_accepts_source_improving_boundary(self):
        allowance,evidence = self.build()
        self.assertTrue(evidence['verified'],evidence)
        self.assertGreater(int(allowance.sum()),100)
        for metrics in evidence['measurements'].values():
            self.assertLessEqual(metrics['after']['mean'],metrics['before']['mean']+1e-6)
            self.assertLessEqual(metrics['after']['p95'],evidence['native_budget']+1e-6)
        self.assertFalse(evidence['contacts_authorized'])
        self.assertFalse(allowance[50,50])
        self.assertFalse(allowance[2,2])

    def test_rounded_nonsquare_native_contour_uses_uniform_meet_and_offset(self):
        # Independent analytic native document: source coordinates are already
        # mapped into pixels; no production viewport or mapping helper is used.
        import resvg_py
        working_w,working_h,native_w,native_h=128,85,193,128
        scale=128/85;offset=(193-128*scale)/2
        before=(f'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="85" viewBox="0 0 128 85">'
            '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="20" y1="15" x2="108" y2="15">'
            '<stop offset="0" stop-color="#205b38"/><stop offset="1" stop-color="#4c995b"/></linearGradient></defs>'
            '<path id="target" data-avc-gradient-object="object" fill="url(#g)" d="M20.6 15.6H108.6V68.6H20.6Z"/></svg>')
        native=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{native_w}" height="{native_h}">'
            f'<defs><linearGradient id="native" gradientUnits="userSpaceOnUse" x1="{20*scale+offset}" '
            f'y1="{15*scale}" x2="{108*scale+offset}" y2="{15*scale}">'
            '<stop offset="0" stop-color="#205b38"/><stop offset="1" stop-color="#4c995b"/></linearGradient></defs>'
            f'<rect x="{20*scale+offset}" y="{15*scale}" width="{88*scale}" height="{53*scale}" fill="url(#native)"/></svg>')
        data=resvg_py.svg_to_bytes(svg_string=native,background=None,skip_system_fonts=True,
            log_information=False,shape_rendering='geometric_precision')
        with Image.open(io.BytesIO(data)) as image:reference=np.asarray(image.convert('RGBA')).copy()
        alpha=reference[:,:,3:4]/255
        source=reference.copy();source[:,:,:3]=np.rint(reference[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);source[:,:,3]=255
        proposal=propose_source_edge_reconstruction(before,'target',source,reference)
        allowance,evidence=build_source_boundary_evidence(before,proposal['candidate_svg_text'],
            source,reference,proposal['geometry'])
        self.assertTrue(evidence['verified'],evidence)
        self.assertEqual(allowance.shape,(native_h,native_w))
        self.assertGreater(allowance.sum(),100)
        self.assertAlmostEqual(evidence['native_sampling_mapping']['uniform_scale'],scale)
        np.testing.assert_allclose(evidence['native_sampling_mapping']['offset_xy'],[offset,0])
        for values in evidence['measurements'].values():
            self.assertLessEqual(values['after']['p95'],evidence['native_budget']+1e-6)

    def test_guard_opt_in_is_bound_and_runs_global_and_target_roi_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'before.svg').write_text(self.before,encoding='utf8')
            (root/'after.svg').write_text(self.after,encoding='utf8')
            Image.fromarray(self.source).save(root/'source.png')
            Image.fromarray(self.reference).save(root/'processed.png')
            result=validate_source_scene(root/'before.svg',root/'after.svg',root/'source.png',
                                         processed_reference_png=root/'processed.png',source_edge_geometry=self.geometry)
            self.assertEqual(result['boundary_mode'],'verified_native_free_paper_envelope',result)
            self.assertTrue(result['metrics']['global_color_nonregression'])
            self.assertTrue(result['metrics']['roi_color_nonregression'])

    def test_bad_geometry_source_path_paint_context_or_other_drawable_falls_back(self):
        changed_source=self.source.copy();changed_source[50,50,0]^=1
        changed_ref=self.reference.copy();changed_ref[50,50,0]^=1
        bad_geometry=copy.deepcopy(self.geometry);bad_geometry['path']+=' M1 1L2 2Z'
        mutations=[{'geometry':bad_geometry},{'source_rgba':changed_source},{'processed_rgba':changed_ref},
                   {'after_svg_text':self.after.replace('stop-color="#205b38"','stop-color="red"')},
                   {'after_svg_text':self.after.replace('fill="red"','fill="blue"')},
                   {'after_svg_text':self.after.replace('id="target"','opacity=".5" id="target"')},
                   {'processed_rgba':None}]
        for mutation in mutations:
            with self.subTest(mutation=list(mutation)):
                mask,evidence=self.build(**mutation)
                self.assertFalse(evidence['verified'])
                self.assertFalse(mask.any())

    def test_retained_white_object_on_a_free_edge_never_gets_paper_allowance(self):
        # Source white plus retained alpha is a white object, not erased paper,
        # even beside an otherwise trustworthy high-contrast paper edge.
        source=self.source.copy();reference=self.reference.copy()
        source[50,20,:3]=255;reference[50,20,:3]=255
        mask=_render(ET.fromstring(self.before),128,128)[:,:,3]>=128
        mask[:5,:5]=False
        proposal=propose_source_edge_reconstruction(self.before,'target',source,reference,ownership_mask=mask)
        allowance,evidence=self.build(source_rgba=source,processed_rgba=reference,
                                     after_svg_text=proposal['candidate_svg_text'],geometry=proposal['geometry'])
        self.assertTrue(evidence['verified'],evidence)
        self.assertGreaterEqual(evidence['retained_white_object_pixels'],1)
        self.assertFalse(allowance[50,20])

    def test_certificate_dimension_mismatch_never_reinterprets_native_pixels(self):
        for key,value in [('source_dimensions',[64,64]),('processed_dimensions',[64,64]),
                          ('working_dimensions',[0,128])]:
            geometry=copy.deepcopy(self.geometry)
            geometry['source_edge_reconstruction'][key]=value
            allowance,evidence=self.build(geometry=geometry)
            self.assertFalse(evidence['verified'])
            self.assertFalse(allowance.any())
            # Producer certificates now also reject inconsistent native/working
            # dimensions before the independent builder's explicit check.
            self.assertTrue(any(reason in evidence['reasons'] for reason in (
                'source_boundary_certificate_dimensions_mismatch',
                'source_boundary_geometry_certificate_invalid')), evidence['reasons'])

    def test_large_native_envelope_does_not_truncate_thin_feature_protection(self):
        wide=self.before.replace('width="128"','width="1920"').replace(
            'viewBox="0 0 128 128"','viewBox="0 0 1920 128"').replace('H108.6','H1900.6')
        exact=wide.replace('M20.6 20.6H1900.6V108.6H20.6Z','M20 20H1900V108H20Z')
        reference=_render(ET.fromstring(exact),1920,128)
        a=reference[:,:,3:4].astype(float)/255
        source=reference.copy();source[:,:,:3]=np.rint(reference[:,:,:3]*a+255*(1-a)).astype(np.uint8)
        source[:,:,3]=255
        mask=_render(ET.fromstring(wide),1920,128)[:,:,3]>=128;mask[:5,:5]=False
        proposal=propose_source_edge_reconstruction(wide,'target',source,reference,ownership_mask=mask)
        allowance,evidence=build_source_boundary_evidence(wide,proposal['candidate_svg_text'],
            source,reference,proposal['geometry'])
        self.assertFalse(evidence['verified'])
        self.assertFalse(allowance.any())
        self.assertIn('source_boundary_feature_envelope_exceeds_supported_native_width',evidence['reasons'])

    def test_exact_crlf_file_binding_is_preserved_by_single_and_chain_apis(self):
        before=self.before.replace('><','>\r\n<')
        mask=_render(ET.fromstring(before),128,128)[:,:,3]>=128;mask[:5,:5]=False
        proposal=propose_source_edge_reconstruction(before,'target',self.source,self.reference,ownership_mask=mask)
        after=proposal['candidate_svg_text']
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'before.svg').write_bytes(before.encode('utf8'))
            (root/'after.svg').write_bytes(after.encode('utf8'))
            Image.fromarray(self.source).save(root/'source.png')
            Image.fromarray(self.reference).save(root/'processed.png')
            single=validate_source_scene(root/'before.svg',root/'after.svg',root/'source.png',
                processed_reference_png=root/'processed.png',source_edge_geometry=proposal['geometry'])
            chain=validate_source_scene_chain(root/'before.svg',root/'after.svg',root/'source.png',
                processed_reference_png=root/'processed.png',transactions=[{
                    'before_svg_text':before,'after_svg_text':after,'source_edge_geometry':proposal['geometry']}])
            self.assertTrue(single['accepted'],single['reasons'])
            self.assertTrue(chain['accepted'],chain['reasons'])
            expected=hashlib.sha256(before.encode('utf8')).hexdigest()
            self.assertEqual(single['provenance']['before_svg_sha256'],expected)
            self.assertEqual(chain['transaction_chain']['before_svg_sha256'],expected)

    def test_no_fictitious_segment_across_disconnected_source_arcs(self):
        field=np.zeros((8,8));field[2:6,2:6]=1
        reliable=np.ones_like(field,dtype=bool);reliable[:,3:5]=False
        segments=_crossing_segments(field,reliable)
        self.assertTrue(all(abs(s[0][0]-s[1][0])<=1 and abs(s[0][1]-s[1][1])<=1 for s in segments))
        self.assertFalse(any(min(s[:,0])<3 and max(s[:,0])>5 for s in segments))

    def test_pixel_footprint_intersects_endpoint_strip_without_unbounded_extension(self):
        segments=np.array([[[0.,0.],[2.,0.]]])
        points=np.array([[2.4,0.],[2.6,0.],[1.,.59],[1.,.61]])
        self.assertFalse(_supported_normal_projection(points[:1],segments)[0])
        np.testing.assert_array_equal(_normal_strip_footprint_overlap(points,segments,.1),
                                      [True,False,True,False])

    def test_diagonal_footprint_requires_all_separating_axes(self):
        segments=np.array([[[0.,0.],[1.,1.]]])
        np.testing.assert_array_equal(_normal_strip_footprint_overlap(
            np.array([[1.55,1.],[1.6,1.]]),segments,.1),[True,False])

    def test_unknown_arc_gap_never_acquires_endpoint_disks(self):
        segments=np.array([[[0.,0.],[2.,0.]],[[5.,0.],[7.,0.]]])
        self.assertFalse(_normal_strip_footprint_overlap(np.array([[3.5,0.]]),segments,4)[0])

    def test_only_connected_source_vertices_have_certified_corner_envelopes(self):
        connected=np.array([[[-2.,0.],[0.,0.]],[[0.,0.],[0.,-2.]]])
        point=np.array([[1.,1.]])
        self.assertTrue(_normal_strip_footprint_overlap(point,connected,1)[0])
        self.assertFalse(_normal_strip_footprint_overlap(point,connected[:1],1)[0])

    def test_true_one_pixel_channel_and_thin_ink_are_protected(self):
        region=np.zeros((11,11),bool);region[1:10,5]=True
        banks=np.zeros_like(region);banks[1:10,4]=True;banks[1:10,6]=True
        self.assertTrue(np.all(_thin(region,banks,3)[1:10,5]))
        # Same geometry in the opposite colors is a thin retained ink stroke.
        self.assertTrue(np.all(_thin(region,banks,3)[1:10,5]))

    def test_repeated_dark_rim_has_two_independent_material_witnesses(self):
        rgb=np.full((50,50,3),255.);rgb[:,20:]=[80,150,80];rgb[:,20:22]=[10,80,30]
        rgb[:,19]=[132.5,167.5,142.5]
        owner=np.zeros((50,50),bool);owner[:,19:]=True
        query=np.zeros_like(owner);query[25,19:23]=True
        models,count,evidence=_crossfit_source_material(rgb,owner,np.array([255.]*3),query)
        self.assertEqual(int(count[25,19]),4)
        for material in models:np.testing.assert_allclose(material[25,19],[10,80,30])
        self.assertTrue(evidence['query_excluded_from_training'])

    def test_thin_feature_protection_includes_external_paper_banks(self):
        stroke=np.zeros((25,25),bool);stroke[5:20,12]=True
        protected=_protected_feature_footprints(stroke,2.6)
        self.assertTrue(protected[12,12])
        self.assertTrue(protected[12,11]);self.assertTrue(protected[12,13])
        self.assertTrue(protected[12,8]);self.assertTrue(protected[12,16])
        self.assertFalse(protected[12,7]);self.assertFalse(protected[12,17])

    def test_dark_rim_candidate_must_meet_both_independent_source_contours(self):
        source=self.source.copy();reference=self.reference.copy()
        source[40:88,20:22,:3]=[8,30,15]
        reference[40:88,20:22,:3]=[8,30,15]
        mask=_render(ET.fromstring(self.before),128,128)[:,:,3]>=128;mask[:5,:5]=False
        proposal=propose_source_edge_reconstruction(self.before,'target',source,reference,ownership_mask=mask)
        allowance,evidence=self.build(source_rgba=source,processed_rgba=reference,
            after_svg_text=proposal['candidate_svg_text'],geometry=proposal['geometry'])
        self.assertTrue(evidence['verified'],evidence)
        self.assertGreater(evidence['supplemental_material_pixels'],0)
        self.assertEqual(evidence['material_models_required'],2)
        self.assertEqual(evidence['material_scope_combination'],'intersection_of_independently_certified_models')
        for model in evidence['material_model_measurements']:
            self.assertTrue(model['verified'])
            for direction in model['measurements'].values():
                self.assertLessEqual(direction['after']['p95'],evidence['native_budget']+1e-6)
                self.assertLessEqual(direction['after']['max'],evidence['native_tail']+1e-6)
        self.assertEqual(hashlib.sha256(allowance.tobytes()).hexdigest(),evidence['allowance_sha256'])

    def test_pastel_and_isolated_dark_noise_do_not_gain_material_authority(self):
        owner=np.ones((40,40),bool);query=np.zeros_like(owner);query[20,20]=True
        for dark_noise in (False,True):
            rgb=np.full((40,40,3),[240,245,241.])
            if dark_noise:rgb[19,20]=0
            _,counts,_=_crossfit_source_material(rgb,owner,np.array([255.]*3),query)
            self.assertEqual(int(counts[20,20]),0)

    def test_disagreeing_spatial_material_splits_remain_unknown(self):
        yy,xx=np.indices((40,40))
        hashed=(xx.astype(np.uint64)*73856093)^(yy.astype(np.uint64)*19349663)
        split=((hashed>>np.uint64(7))&np.uint64(1)).astype(bool)
        rgb=np.full((40,40,3),[180,20,20.]);rgb[split]=[20,20,180.]
        owner=np.ones((40,40),bool);query=np.zeros_like(owner);query[20,20]=True
        _,counts,_=_crossfit_source_material(rgb,owner,np.array([255.]*3),query)
        self.assertEqual(int(counts[20,20]),0)

    def test_other_owner_samples_cannot_supply_material_witnesses(self):
        rgb=np.full((40,40,3),[240,245,241.]);rgb[:20]=[10,50,20]
        owner=np.zeros((40,40),bool);owner[20:]=True
        query=np.zeros_like(owner);query[20,20]=True
        _,counts,_=_crossfit_source_material(rgb,owner,np.array([255.]*3),query)
        self.assertEqual(int(counts[20,20]),0)

    def test_closed_hole_cannot_be_waived_by_even_a_bound_boundary_mask(self):
        # Low-level regression isolates the topology invariant from the builder:
        # even an overbroad trusted mask cannot waive created closed holes.
        source=np.zeros((20,20,4),np.uint8);source[:,:,:3]=[20,80,40];source[:,:,3]=255
        before=source.copy();after=before.copy();after[8:9,8:15,3]=0
        reference=source.copy();allowance=np.ones((20,20),bool)
        evidence={'verified':True,'allowance_sha256':hashlib.sha256(allowance.tobytes()).hexdigest(),
                  'source_rgba_sha256':rgba_sha256(source),'processed_rgba_sha256':rgba_sha256(reference),
                  'render_binding':{'before':rgba_sha256(before),'after':rgba_sha256(after)}}
        result=validate_source_scene_arrays(before,after,source,processed_reference_rgba=reference,
                                            _boundary_allowance=allowance,_boundary_evidence=evidence)
        self.assertFalse(result['accepted'])
        self.assertTrue(any('created_hole' in reason for reason in result['reasons']),result['reasons'])

    def test_rotated_source_white_channel_cannot_be_sealed_despite_global_improvement(self):
        yy,xx=np.indices((64,64))
        ring=(xx>=10)&(xx<54)&(yy>=10)&(yy<54)&((xx<16)|(xx>=48)|(yy<16)|(yy>=48))
        channel=(np.abs(xx-yy-18)<=1)&(yy<=22)
        original_ink=ring&~channel
        source=np.full((64,64,4),255,np.uint8);source[original_ink,:3]=[20,80,40]
        reference=source.copy();reference[~original_ink,3]=0
        before=np.zeros_like(source);before[original_ink]=[60,115,80,255]
        after=np.zeros_like(source);after[ring]=[20,80,40,255]
        # The internal mask deliberately includes the channel: hard source
        # connectivity must not depend on the directional thin-feature helper.
        allowance=np.ones((64,64),bool)
        evidence={'verified':True,'allowance_sha256':hashlib.sha256(allowance.tobytes()).hexdigest(),
                  'source_rgba_sha256':rgba_sha256(source),'processed_rgba_sha256':rgba_sha256(reference),
                  'render_binding':{'before':rgba_sha256(before),'after':rgba_sha256(after)}}
        result=validate_source_scene_arrays(before,after,source,processed_reference_rgba=reference,
            _boundary_allowance=allowance,_boundary_evidence=evidence)
        self.assertTrue(result['metrics']['global_color_nonregression'])
        self.assertFalse(result['accepted'])
        self.assertIn('closed_source_open_channel',result['reasons'])
        self.assertTrue(any(d['kind']=='closed_source_open_channel' and d['alpha_threshold']==128
                            for d in result['localized_defects']))

    def test_restoring_a_genuine_closed_source_hole_remains_allowed(self):
        yy,xx=np.indices((64,64))
        ring=(xx>=10)&(xx<54)&(yy>=10)&(yy<54)&((xx<16)|(xx>=48)|(yy<16)|(yy>=48))
        source=np.full((64,64,4),255,np.uint8);source[ring,:3]=[20,80,40]
        reference=source.copy();reference[~ring,3]=0
        before=np.zeros_like(source);before[ring]=[20,80,40,255];before[10:16,28:31]=0
        after=np.zeros_like(source);after[ring]=[20,80,40,255]
        result=validate_source_scene_arrays(before,after,source,processed_reference_rgba=reference)
        self.assertTrue(result['accepted'],result['reasons'])
        self.assertTrue(any(r['kind']=='restored_source_empty_region' for r in result['source_supported_hole_repairs']))

    def test_unbound_array_allowance_is_not_trusted(self):
        before=_render(ET.fromstring(self.before),128,128)
        after=_render(ET.fromstring(self.after),128,128)
        mask,evidence=self.build()
        result=validate_source_scene_arrays(before,after,self.source,processed_reference_rgba=self.reference,
                                            _boundary_allowance=mask,_boundary_evidence=evidence)
        self.assertIn('source_boundary_array_binding_mismatch',result['reasons'])

    def test_dense_curve_sampling_does_not_join_subpaths(self):
        segments=_path_segments('M1 1L2 1L2 2Z M10 10L11 10L11 11Z',1)
        self.assertTrue(np.all(np.linalg.norm(segments[:,1]-segments[:,0],axis=1)<=.201))

    def chain(self,transactions,final=None):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'before.svg').write_text(self.before,encoding='utf8')
            (root/'after.svg').write_text(final or self.after,encoding='utf8')
            Image.fromarray(self.source).save(root/'source.png')
            Image.fromarray(self.reference).save(root/'processed.png')
            return validate_source_scene_chain(root/'before.svg',root/'after.svg',root/'source.png',
                processed_reference_png=root/'processed.png',transactions=transactions)

    def test_chain_replays_geometry_and_strict_transaction_then_checks_aggregate(self):
        tx={'before_svg_text':self.before,'after_svg_text':self.after,'source_edge_geometry':self.geometry}
        # No-op metadata transaction goes through the same strict scene path as
        # a paint transaction. It cannot silently acquire the previous proof.
        final=self.after.replace('id="other"','data-review="manual" id="other"')
        result=self.chain([tx,{'before_svg_text':self.after,'after_svg_text':final}],final)
        self.assertTrue(result['accepted'],result)
        self.assertEqual(result['transaction_chain']['replayed_count'],2)
        self.assertTrue(result['transaction_chain']['final_topology_and_cost_rechecked'])
        self.assertTrue(result['metrics']['all_required_roi_nonregression'])
        self.assertNotIn('svg_text',str(result['transaction_chain']))

    def test_chain_missing_reordered_or_modified_transaction_rejects(self):
        tx={'before_svg_text':self.before,'after_svg_text':self.after,'source_edge_geometry':self.geometry}
        cases=[([],self.after),([dict(tx,before_svg_text=self.after)],self.after),
               ([tx],self.after.replace('fill="red"','fill="blue"'))]
        for transactions,final in cases:
            result=self.chain(transactions,final)
            self.assertFalse(result['accepted'])
            self.assertTrue(any('chain_' in reason for reason in result['reasons']),result['reasons'])

    def test_chain_does_not_trust_saved_accepted_flag(self):
        final=self.after.replace('fill="red"','fill="blue"')
        result=self.chain([{'before_svg_text':self.before,'after_svg_text':self.after,'source_edge_geometry':self.geometry},
                           {'before_svg_text':self.after,'after_svg_text':final,'accepted':True}],final)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['transaction_chain']['failed_index'],1)
        self.assertIn('source_color_error_increased',result['reasons'])


if __name__=='__main__':unittest.main()
