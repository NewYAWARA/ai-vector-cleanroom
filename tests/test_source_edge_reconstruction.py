import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from source_edge_reconstruction import (
    _field_loops, _source_field, _topology, _later_paint_alpha,_native_mapping,_native_material_colour,_propose_source_ink_hole_fills,propose_source_edge_reconstruction,
    source_edge_certificate_valid, final_source_edge_matches, rgba_sha256,
)
from svg_renderer import render_svg_reference


class SourceEdgeReconstructionTests(unittest.TestCase):
    def test_identity_morphology_never_enters_native_rank_filter(self):
        from source_edge_reconstruction import _morph
        mask = np.array([[True, False], [False, True]])
        with patch('source_edge_reconstruction.ImageFilter.MinFilter', side_effect=AssertionError('native filter called')), \
             patch('source_edge_reconstruction.ImageFilter.MaxFilter', side_effect=AssertionError('native filter called')):
            for erode in (False, True):
                result = _morph(mask, 1, erode=erode)
                np.testing.assert_array_equal(result, mask)
                self.assertIsNot(result, mask)
            for invalid in (0, -1, 2, True, 1.5):
                with self.assertRaises(ValueError):
                    _morph(mask, invalid)

    @staticmethod
    def reference(source):
        reference=source.copy()
        reference[np.min(reference[:,:,:3],axis=2)>=245,3]=0
        return reference

    def fixture(self, hole=False):
        mask=np.zeros((64,64),bool);mask[10:54,10:54]=True
        path='M10 10H54V54H10Z'
        if hole:
            mask[30:33,30:33]=False;path+=' M30 30H33V33H30Z'
        svg=('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
             '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="64" y2="0">'
             '<stop offset="0" stop-color="#205b38"/><stop offset="1" stop-color="#4c995b"/></linearGradient></defs>'
             '<g fill-rule="evenodd"><path id="target" data-avc-gradient-object="object" data-avc-old-proof="old" fill="url(#g)" d="'+path+'"/></g>'
             '<path id="unrelated" fill="red" d="M2 2H4V4H2Z"/></svg>')
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source.svg';source.write_text(svg,encoding='utf8')
            render_svg_reference(source,source.with_suffix('.png'),64,background='white')
            rgba=np.array(Image.open(source.with_suffix('.png')).convert('RGBA'))
        return svg,mask,rgba

    def test_pixel_centres_do_not_shift_rectangle_half_pixel(self):
        field=np.zeros((30,40));field[5:20,10:30]=1
        loops=_field_loops(field)
        self.assertEqual(len(loops),1)
        points=np.asarray(loops[0])
        np.testing.assert_equal(points.min(0),[10,5])
        np.testing.assert_equal(points.max(0),[30,20])

    def test_native_mapping_rejects_invalid_dimensions_without_arithmetic_failure(self):
        for dimensions in ((0,64),(-1,64),(float('nan'),64),(64,float('inf'))):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError,'invalid_mapping_dimensions'):
                _native_mapping(dimensions,(128,128))
            with self.subTest(native=dimensions), self.assertRaisesRegex(ValueError,'invalid_mapping_dimensions'):
                _native_mapping((64,64),dimensions)

    def test_real_native_candidate_is_proposal_until_scene_commit_and_binds_source(self):
        svg,mask,rgba=self.fixture()
        reference=self.reference(rgba)
        out=propose_source_edge_reconstruction(svg,'target',rgba,reference,ownership_mask=mask)
        geometry=out['geometry'];cert=out['certificate']
        self.assertTrue(source_edge_certificate_valid(geometry))
        self.assertFalse(source_edge_certificate_valid(geometry,require_scene_commit=True))
        self.assertEqual(cert['source_rgba_sha256'],rgba_sha256(rgba))
        self.assertEqual(cert['pixel_coordinates'],'native_pixel_centres_x_plus_0.5_y_plus_0.5_then_inverse_SVG_meet')
        self.assertTrue(cert['not_binary_ownership_equivalence'])
        before=ET.fromstring(svg);after=ET.fromstring(out['candidate_svg_text'])
        self.assertEqual(ET.tostring(next(n for n in before.iter() if n.get('id')=='unrelated')),
                         ET.tostring(next(n for n in after.iter() if n.get('id')=='unrelated')))
        target=next(n for n in after.iter() if n.get('id')=='target')
        self.assertNotIn('data-avc-old-proof',target.attrib)
        geometry['source_edge_scene_commit']={
            'status':'committed','accepted':True,'source_guard':{'accepted':True},
            'alpha_guard':{'accepted':True,'scope':'source_supported_alpha_not_prior_svg_topology'},
            **{key:cert[key] for key in ('before_svg_sha256','after_path_sha256','source_rgba_sha256')}}
        self.assertTrue(final_source_edge_matches(after,geometry,rgba,reference))
        for mutation in ('path','source','paint','context','guard'):
            tree=copy.deepcopy(after);new=copy.deepcopy(geometry);source=rgba.copy()
            if mutation=='path': next(n for n in tree.iter() if n.get('id')=='target').set('d','M0 0H60V60Z')
            if mutation=='source': source[20,20,0]^=1
            if mutation=='paint': next(n for n in tree.iter() if n.tag.endswith('stop')).set('stop-color','red')
            if mutation=='context': next(n for n in tree.iter() if n.get('id')=='target').set('opacity','.5')
            if mutation=='guard': new['source_edge_scene_commit']['alpha_guard']['accepted']=False
            with self.subTest(mutation=mutation):
                self.assertFalse(final_source_edge_matches(tree,new,source,reference))

    def test_true_hole_preserved_and_false_hole_requires_both_references(self):
        svg,mask,source=self.fixture(hole=True)
        field,adjusted,proof=_source_field(mask,source,source,np.array([255.,255.,255.]))
        self.assertEqual(proof['removed_holes'],0)
        self.assertEqual(_topology(field>=.5)['holes'],1)
        _,_,without_hole=self.fixture(hole=False)
        field,adjusted,proof=_source_field(mask,without_hole,without_hole,np.array([255.,255.,255.]))
        self.assertEqual(proof['removed_holes'],1)
        self.assertEqual(proof['removed_hole_pixels'],9)
        self.assertEqual(_topology(field>=.5)['holes'],0)
        transparent=without_hole.copy();transparent[30:33,30:33,3]=0
        _,_,proof=_source_field(mask,without_hole,transparent,np.array([255.,255.,255.]))
        self.assertEqual(proof['removed_holes'],0)

    def test_source_supported_crack_is_completed_not_left_as_a_sealed_hole(self):
        # A tracer's exterior-connected crack becomes enclosed when its mouth
        # is reconstructed. The source is continuous ink in the entire crack.
        original=np.full((40,40,4),[80,140,45,255],dtype=np.uint8)
        old=np.ones((40,40),bool);old[:24,19:21]=False
        field=old.astype(float);field[:10,19:21]=1.
        self.assertEqual(_topology(old)['holes'],0)
        self.assertEqual(_topology(field>=.5)['holes'],1)
        rows=_propose_source_ink_hole_fills(field,original,original,old)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['pixels'],28)
        self.assertTrue(np.all(field==1.))
        self.assertFalse(rows[0]['paint_ownership_proven'])
        self.assertTrue(rows[0]['full_scene_colour_and_topology_validation_required'])

    def test_hole_proposal_preserves_real_paper_vein_and_ambiguous_reference(self):
        source=np.full((40,40,4),[80,140,45,255],dtype=np.uint8)
        field=np.ones((40,40));field[10:24,19:21]=0.
        old=np.ones((40,40),bool);old[:24,19:21]=False
        for variant in ('paper_vein','one_paper_pixel','reference_missing','transparent_source','large','existing_coloured_hole'):
            actual=source.copy();reference=source.copy();target=field.copy()
            ownership=old
            if variant=='paper_vein':actual[10:24,19:21,:3]=255
            elif variant=='one_paper_pixel':actual[17,20,:3]=255
            elif variant=='reference_missing':reference[17,20,3]=0
            elif variant=='transparent_source':actual[17,20,3]=128
            elif variant=='large':target[10:30,10:30]=0.
            else:
                ownership=target>=.5
                actual[10:24,19:21,:3]=[180,10,50]
            before=target.copy()
            with self.subTest(variant=variant):
                self.assertEqual(_propose_source_ink_hole_fills(target,actual,reference,ownership),[])
                np.testing.assert_equal(target,before)

    def test_native_reference_size_is_bound_and_working_viewbox_is_separate(self):
        svg,mask,source=self.fixture()
        native=np.asarray(Image.fromarray(source).resize((128,128),Image.Resampling.NEAREST))
        reference=self.reference(native)
        native_mask=np.repeat(np.repeat(mask,2,axis=0),2,axis=1)
        result=propose_source_edge_reconstruction(svg,'target',native,reference,ownership_mask=native_mask)
        cert=result['certificate']
        self.assertEqual(cert['source_dimensions'],[128,128])
        self.assertEqual(cert['processed_dimensions'],[128,128])
        self.assertEqual(cert['working_dimensions'],[64,64])
        self.assertEqual(cert['processed_rgba_sha256'],rgba_sha256(reference))
        self.assertEqual(cert['aligned_processed_rgba_sha256'],cert['processed_rgba_sha256'])
        self.assertEqual(cert['native_sampling_dimensions'],[128,128])
        self.assertEqual(cert['native_sampling_mapping']['viewbox_to_native_scale'],2.)
        with self.assertRaisesRegex(ValueError,'ownership_mask'):
            propose_source_edge_reconstruction(svg,'target',native,reference,ownership_mask=mask)
        target=next(n for n in ET.fromstring(result['candidate_svg_text']).iter() if n.get('id')=='target')
        self.assertEqual(int(target.get('data-avc-designer-anchors')),result['geometry']['designer_anchor_count'])

    def test_rounded_non_square_native_canvas_uses_uniform_meet_without_reference_resize(self):
        svg,_,_=self.fixture()
        svg=svg.replace('width="64" height="64" viewBox="0 0 64 64"',
                        'width="100" height="70" viewBox="0 0 100 69"')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.svg';path.write_text(svg,encoding='utf8')
            render_svg_reference(path,path.with_suffix('.png'),100,background='white')
            image=np.asarray(Image.open(path.with_suffix('.png')).convert('RGBA'))
        self.assertEqual(image.shape[:2],(70,100))
        reference=self.reference(image)
        result=propose_source_edge_reconstruction(svg,'target',image,reference)
        cert=result['certificate'];mapping=cert['native_sampling_mapping']
        self.assertTrue(source_edge_certificate_valid(result['geometry']))
        self.assertEqual(mapping['viewbox_to_native_scale'],1.)
        self.assertEqual(mapping['native_letterbox_offset'],[0.,.5])
        self.assertEqual(cert['processed_rgba_sha256'],rgba_sha256(reference))
        scale,offset=_native_mapping((1024,341),(1254,418))
        point=np.array([517.25,173.875])
        np.testing.assert_allclose((point*scale+offset-offset)/scale,point,rtol=0,atol=1e-12)
        changed=copy.deepcopy(result['geometry'])
        changed['source_edge_reconstruction']['native_sampling_mapping']['native_letterbox_offset']=[0.,0.]
        self.assertFalse(source_edge_certificate_valid(changed))
        with self.assertRaisesRegex(ValueError,'native_aligned'):
            propose_source_edge_reconstruction(svg,'target',image,reference[:-1])

    def test_native_material_colour_does_not_borrow_cross_owner_dark_pixels(self):
        rgb=np.full((20,20,3),220.,float);owner=np.zeros((20,20),bool);owner[5:15,5:10]=True
        rgb[owner]=[80.,140.,65.];rgb[5:15,10:15]=[10.,10.,10.]
        inside,count,selected=_native_material_colour(rgb,owner,np.array([255.,255.,255.]))
        np.testing.assert_allclose(inside[10,9],[80.,140.,65.])
        self.assertGreater(count[10,9],3)
        self.assertFalse(selected[:,10:].any())

    def test_source_target_and_measured_error_cannot_be_replaced_after_fit(self):
        svg,mask,source=self.fixture()
        result=propose_source_edge_reconstruction(svg,'target',source,self.reference(source),ownership_mask=mask)
        for kind in ('point','measurement','empty_contract','fit_path'):
            geometry=copy.deepcopy(result['geometry'])
            if kind=='point': geometry['source_edge_reconstruction']['source_target_loops'][0][0][0]+=1
            if kind=='measurement': geometry['actual_error']['p95_percent']=0.001
            if kind=='empty_contract': geometry['source_edge_reconstruction']['error_contract']={}
            if kind=='fit_path': geometry['fit']['path']='M0 0H5V5Z'
            with self.subTest(kind=kind): self.assertFalse(source_edge_certificate_valid(geometry))

    def test_coloured_interior_mark_is_not_erased_as_false_hole(self):
        _,mask,_=self.fixture(hole=True)
        _,_,source=self.fixture(hole=False)
        source[30:33,30:33,:3]=[180,10,50]
        _,_,proof=_source_field(mask,source,source,np.array([255.,255.,255.]))
        self.assertEqual(proof['removed_holes'],0)
        self.assertEqual(proof['source_ink_hole_fill_proposals'],[])

    def test_one_native_white_pixel_blocks_false_hole_removal_after_downsampling(self):
        _,mask,_=self.fixture(hole=True)
        _,_,solid=self.fixture(hole=False)
        native=np.array(Image.fromarray(solid).resize((128,128),Image.Resampling.NEAREST))
        native[61,61,:3]=255
        _,_,proof=_source_field(mask,native,solid,np.array([255.,255.,255.]))
        self.assertEqual(proof['removed_holes'],0)
        self.assertFalse(proof['hole_decisions'][0]['original_ink'])

    def test_nonwhite_coloured_interface_is_explicitly_only_a_guarded_hypothesis(self):
        _,mask,source=self.fixture()
        source[12:52,54:57,:3]=[200,15,60]
        field,_,proof=_source_field(mask,source,source,np.array([255.,255.,255.]))
        self.assertIn('not_source_proven',proof['ambiguous_interface_proposal'])
        self.assertEqual(proof['ambiguous_interface_smoothing_sigma_sampling_pixels'],.55)
        self.assertGreater(proof['ambiguous_boundary_pixels_proposed_from_ownership'],0)
        self.assertLess(proof['source_unmixed_pixels'],proof['band_pixels'])
        self.assertTrue(np.all(np.isfinite(field)))

    def test_underlap_never_expands_true_enclosed_hole(self):
        _,mask,source=self.fixture(hole=True)
        field,_,proof=_source_field(mask,source,source,np.array([255.,255.,255.]),underlap_pixels=1)
        self.assertEqual(proof['removed_holes'],0)
        self.assertTrue(np.all(field[30:33,30:33]<.5))

    def test_later_native_paint_coverage_excludes_target_and_earlier_objects(self):
        svg,_,_=self.fixture()
        root=ET.fromstring(svg);target=next(n for n in root.iter() if n.get('id')=='target')
        alpha=_later_paint_alpha(root,target,64)
        self.assertEqual(int(alpha[3,3]),255)
        self.assertEqual(int(alpha[20,20]),0)
        # Moving that same little object before the target removes its coverage
        # from the underlap permission mask, although the scene still contains it.
        child=next(n for n in root if n.get('id')=='unrelated');root.remove(child);root.insert(0,child)
        alpha=_later_paint_alpha(root,target,64)
        self.assertEqual(int(alpha.max()),0)

    def test_reject_low_contrast_transparency_background_and_transform(self):
        svg,mask,source=self.fixture()
        for kind in ('low_contrast','transparent','background','transform','budget','external','stop_alpha','stop_style'):
            image=source.copy();text=svg;kwargs={}
            if kind=='low_contrast': image[mask,:3]=248
            if kind=='transparent': image[0,0,3]=0
            if kind=='background': image[0,0,:3]=[20,20,20]
            if kind=='transform': text=svg.replace('<g ','<g transform="translate(1 0)" ')
            if kind=='budget': kwargs['error_budget_percent']=.5
            if kind=='external': text=svg.replace('<defs>','<image href="https://example.invalid/x"/><defs>')
            if kind=='stop_alpha': text=svg.replace('stop-color="#205b38"','stop-color="#205b38" stop-opacity=".5"')
            if kind=='stop_style': text=svg.replace('stop-color="#205b38"','stop-color="#205b38" style="stop-opacity:.5"')
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                propose_source_edge_reconstruction(text,'target',image,self.reference(image),ownership_mask=mask,**kwargs)

    def test_field_preserves_deliberate_sharp_sawtooth_above_budget(self):
        # A dark, opaque pixel stair with repeated long teeth is source evidence,
        # not licence for smoothing the whole ownership mask with a Gaussian.
        mask=np.zeros((80,80),bool);mask[20:60,15:65]=True
        for x in range(18,60,8): mask[12:20,x:x+3]=True
        image=np.full((80,80,4),255,np.uint8);image[mask,:3]=[20,90,45]
        field,_,_= _source_field(mask,image,image,np.array([255.,255.,255.]))
        np.testing.assert_equal(field>=.5,mask)


if __name__=='__main__':
    unittest.main()
