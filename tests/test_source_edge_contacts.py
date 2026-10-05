import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from source_edge_contacts import (_nearest,_splice,_reference_points,_coalesce_runs,_runs,_native_box_to_viewbox,
    _join_reversal_flags,_added_joins_reverse,_boundary_outlier_boxes,JOIN_DIRECTION_SCREEN,
    contact_certificate_valid,preserve_source_edge_contacts,SCHEMA)
from source_edge_reconstruction import _json_sha,_sha
from gradient_contour_spans import _path


def line(start,end):
    return {'type':'line','start':list(map(float,start)),'end':list(map(float,end))}


class SourceEdgeContactTests(unittest.TestCase):
    def fixture(self):
        candidate=[line((5,5),(75,5)),line((75,5),(75,75)),line((75,75),(5,75)),line((5,75),(5,5))]
        original=[{'type':'cubic','start':[20.,5.],'control1':[25.,5.],
                   'control2':[28.,15.],'end':[35.,15.]},
                  {'type':'cubic','start':[35.,15.],'control1':[42.,15.],
                   'control2':[45.,5.],'end':[50.,5.]}]
        return candidate,original

    def test_analytic_line_projection_does_not_depend_on_sampling_grid(self):
        rows=[line((i,0),(i+1,0)) for i in range(1000)]
        distance,index,t=_nearest(rows,[832.123456789,.25])
        self.assertAlmostEqual(distance,.25,places=12)
        self.assertEqual(index,832)
        self.assertAlmostEqual(t,.123456789,places=10)

    def test_native_box_mapping_uses_meet_letterbox_not_independent_axis_stretch(self):
        np.testing.assert_allclose(_native_box_to_viewbox([0,.5,100,69.5],(100,69),(100,70)),[0,0,100,69])
        np.testing.assert_allclose(_native_box_to_viewbox([.5,0,100.5,70],(100,70),(101,70)),[0,0,100,70])

    def test_splice_preserves_exact_original_cubics_and_bounded_new_joins(self):
        candidate,original=self.fixture()
        result,record=_splice(candidate,original,maximum_join=.5)
        self.assertEqual(record['join_distances_working_pixels'],[0.,0.])
        self.assertTrue(all(row in result for row in original))
        self.assertEqual(result[-3:-1],original)
        self.assertTrue(all(row in result for row in candidate[1:]))
        # The sampled hybrid target retains these curves and their local bay,
        # rather than declaring the failed straight source target equivalent.
        points=_reference_points(result,step=.05)
        self.assertTrue(np.any(np.all(np.isclose(points,[35.,15.]),axis=1)))

    def test_long_connection_and_wrong_contour_direction_fail_closed(self):
        candidate,original=self.fixture()
        far=copy.deepcopy(original);far[0]['start'][1]=7.
        with self.assertRaisesRegex(ValueError,'half_working_pixel'):
            _splice(candidate,far,maximum_join=.5)
        backwards=[line((50,5),(20,5))]
        with self.assertRaisesRegex(ValueError,'nonmonotone'):
            _splice(candidate,backwards,maximum_join=.5)

    def test_short_arc_correspondence_is_invariant_to_segment_density(self):
        # Dense sampling of one short side must not turn a legitimate forward
        # arc into a long backward arc according to segment indices.
        candidate=[line((0,0),(100,0)),line((100,0),(100,100)),line((100,100),(0,100))]
        candidate += [line((0,100-i/10),(0,100-(i+1)/10)) for i in range(1000)]
        original=[line((0,90),(0,10))]
        result,record=_splice(candidate,original,maximum_join=.5)
        self.assertIn(original[0],result)
        self.assertGreater(record['replaced_candidate_segments'],len(candidate)//2)
        self.assertTrue(all(row in result for row in candidate[:3]))
        with self.assertRaisesRegex(ValueError,'nonmonotone'):
            _splice(candidate,[line((0,10),(0,90))],maximum_join=.5)

    def test_subpixel_short_join_must_not_introduce_a_backward_corner(self):
        candidate,_=self.fixture()
        old=[{'type':'cubic','start':[20.,5.],'control1':[25.,5.],
              'control2':[50.,4.],'end':[50.,5.25]}]
        _,ai,at=_nearest(candidate,old[0]['start']);_,bi,bt=_nearest(candidate,old[0]['end'])
        self.assertEqual(_join_reversal_flags(candidate,old,0,0,ai,at,bi,bt),(False,True))
        result,record=_splice(candidate,old,maximum_join=.5)
        self.assertAlmostEqual(record['join_distances_working_pixels'][1],.25)
        self.assertTrue(_added_joins_reverse(result,record['added_joins']))
        old[0]['control2']=[50.,6.]
        self.assertEqual(_join_reversal_flags(candidate,old,0,0,ai,at,bi,bt),(False,False))
        result,record=_splice(candidate,old,maximum_join=.5)
        self.assertFalse(_added_joins_reverse(result,record['added_joins']))

    def test_contact_proof_rechecks_preserved_coordinates_and_added_joins(self):
        candidate,original=self.fixture()
        result,record=_splice(candidate,original,maximum_join=.5)
        before=_path(candidate,True);path=_path(result,True)
        geometry={'path':path,'source_edge_reconstruction':{
            'before_path_sha256':_sha(before),'contact_preservation':{
                'schema':SCHEMA,'full_scene_revalidation_required':True,
                'hybrid_reference_is_not_entirely_source_derived':True,
                'added_joins_are_not_original_geometry':True,
                'preserved_segments':original,'preserved_segments_sha256':_json_sha(original),
                'preserved_segment_count':2,'original_path_sha256':_sha(before),
                'maximum_join_working_pixels':.5,'spans':[record]}}}
        self.assertTrue(contact_certificate_valid(geometry))
        geometry['source_edge_reconstruction']['contact_preservation']['new_join_direction_screen']=copy.deepcopy(JOIN_DIRECTION_SCREEN)
        self.assertTrue(contact_certificate_valid(geometry))
        altered=copy.deepcopy(geometry)
        altered['source_edge_reconstruction']['contact_preservation']['new_join_direction_screen']['threshold_degrees']=180
        self.assertFalse(contact_certificate_valid(altered))
        changed=copy.deepcopy(geometry)
        changed['path']=path.replace('28.0,15.0','28.0,16.0')
        self.assertFalse(contact_certificate_valid(changed))
        changed=copy.deepcopy(geometry)
        changed['source_edge_reconstruction']['contact_preservation']['spans'][0]['join_distances_working_pixels'][0]=.25
        self.assertFalse(contact_certificate_valid(changed))

    def test_zero_search_budget_does_not_read_or_mutate_proposal(self):
        proposal={'sentinel':[1,2]};before=copy.deepcopy(proposal)
        with self.assertRaisesRegex(ValueError,'budget_exhausted'):
            preserve_source_edge_contacts('',proposal,None,None,{},maximum_seconds=0)
        self.assertEqual(before,proposal)

    def test_span_cap_coalesces_by_preserving_more_original_geometry(self):
        selected=np.array([False,True,True,False,True,False,False,False,True,False])
        merged,added=_coalesce_runs(selected,2)
        self.assertTrue(np.all(merged[selected]))
        self.assertEqual(len(_runs(merged)),2)
        self.assertEqual(added,1)
        self.assertTrue(merged[3])


class SourceContactDiagnosticBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from source_edge_reconstruction import propose_source_edge_reconstruction
        from svg_renderer import render_svg_reference
        points = ([(x,10) for x in range(10,54)] + [(54,y) for y in range(10,54)]
            + [(x,54) for x in range(54,10,-1)] + [(10,y) for y in range(54,10,-1)])
        path = 'M'+' L'.join(f'{x} {y}' for x,y in points)+' Z'
        cls.svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
            '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x2="64">'
            '<stop stop-color="#205b38"/><stop offset="1" stop-color="#4c995b"/></linearGradient></defs>'
            '<path id="target" data-avc-gradient-object="object" data-avc-designer-anchors="176" '
            'fill="url(#g)" d="'+path+'"/><path id="unrelated" fill="red" d="M2 2H8V8H2Z"/></svg>')
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'source.svg';file.write_text(cls.svg,encoding='utf8')
            render_svg_reference(file,file.with_suffix('.png'),64,background='white')
            cls.source=np.array(Image.open(file.with_suffix('.png')).convert('RGBA'))
        cls.reference=cls.source.copy()
        cls.reference[np.min(cls.reference[:,:,:3],axis=2)>=245,3]=0
        mask=np.zeros((64,64),bool);mask[10:54,10:54]=True
        cls.proposal=propose_source_edge_reconstruction(cls.svg,'target',cls.source,cls.reference,ownership_mask=mask)
        cls.guard={'accepted':False,'width':64,'height':64,
            'provenance':{'before_svg_sha256':_sha(cls.svg),'after_svg_sha256':_sha(cls.proposal['candidate_svg_text'])},
            'source_boundary_evidence':{'verified':False,'target_id':'target',
                'reasons':['source_boundary_source_to_vector_outside_contract'],
                'native_tail':.5,'worst_reverse_points':[{'xy':[32.,10.],'distance':1.}]}}

    def test_diagnostic_is_target_scoped_finite_strict_tail_and_bounded(self):
        good=copy.deepcopy(self.guard)
        np.testing.assert_array_equal(_boundary_outlier_boxes(good,'target'),[[31.5,9.5,32.5,10.5]])
        saved=copy.deepcopy(good)
        for field,value in (
                ('verified',True),('verified',None),('target_id','another'),
                ('reasons',[]),('reasons',['source_boundary_reference_unavailable']),
                ('reasons',['source_boundary_source_to_vector_outside_contract','unknown']),
                ('native_tail',True),('native_tail',0),('native_tail',float('nan')),
                ('native_tail',float('inf'))):
            invalid=copy.deepcopy(good);invalid['source_boundary_evidence'][field]=value
            with self.subTest(field=field,value=value):
                self.assertEqual(_boundary_outlier_boxes(invalid,'target'),[])
        for row in ({'xy':[32,10],'distance':.5}, {'xy':[32,10],'distance':True},
                    {'xy':[32,10],'distance':float('inf')}, {'xy':[32,10],'distance':float('nan')},
                    {'xy':[32,float('nan')],'distance':1}, {'xy':[32],'distance':1}, {}):
            invalid=copy.deepcopy(good);invalid['source_boundary_evidence']['worst_reverse_points']=[row]
            with self.subTest(row=row):
                self.assertEqual(_boundary_outlier_boxes(invalid,'target'),[])
        self.assertEqual(good,saved)
        good['source_boundary_evidence']['worst_reverse_points']*=100
        self.assertEqual(len(_boundary_outlier_boxes(good,'target')),12)

    def test_current_source_parent_and_failed_pair_are_required_before_splicing(self):
        for kind in ('source','processed','parent','guard_before','guard_after','guard_accepted','canvas','certificate'):
            svg=self.svg;source=self.source.copy();reference=self.reference.copy()
            proposal=copy.deepcopy(self.proposal);guard=copy.deepcopy(self.guard)
            reason='source_or_parent_changed'
            if kind=='source':source[0,0,0]=254
            elif kind=='processed':reference[0,0,3]=1
            elif kind=='parent':svg+='\n'
            elif kind.startswith('guard_'):
                reason='defect_report_not_bound_to_candidate'
                if kind=='guard_before':guard['provenance']['before_svg_sha256']='0'*64
                elif kind=='guard_after':guard['provenance']['after_svg_sha256']='0'*64
                else:guard['accepted']=True
            elif kind=='canvas':guard['width']=128;reason='defect_canvas_changed'
            else:proposal['geometry']['path']+=' L1 1';reason='parent_certificate_invalid'
            saved=copy.deepcopy(proposal)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,reason):
                preserve_source_edge_contacts(svg,proposal,source,reference,guard,include_boundary_outliers=True)
            self.assertEqual(proposal,saved)

    def test_boundary_repair_is_opt_in_and_unsupported_regions_cannot_select_spans(self):
        with self.assertRaisesRegex(ValueError,'no_bounded_hole_regions'):
            preserve_source_edge_contacts(self.svg,self.proposal,self.source,self.reference,self.guard)
        for point in ([500.,500.],[-500.,-500.]):
            guard=copy.deepcopy(self.guard)
            guard['source_boundary_evidence']['worst_reverse_points'][0]['xy']=point
            with self.subTest(point=point),self.assertRaisesRegex(ValueError,'no_original_boundary_at_defects'):
                preserve_source_edge_contacts(self.svg,self.proposal,self.source,self.reference,guard,include_boundary_outliers=True)
        guard=copy.deepcopy(self.guard)
        guard['localized_defects']=[{'kind':'source_feature_color_error_increased','bbox_xyxy':[31,9,33,11]}]
        with self.assertRaisesRegex(ValueError,'requires_verified_paper_envelope'):
            preserve_source_edge_contacts(self.svg,self.proposal,self.source,self.reference,guard,
                                          include_unverified_source_defects=True)
        guard['source_boundary_evidence']={'verified':True}
        # A proven source envelope permits this diagnostic only as a location.
        amended=preserve_source_edge_contacts(self.svg,self.proposal,self.source,self.reference,guard,
                                             include_unverified_source_defects=True)
        self.assertTrue(amended['certificate']['contact_preservation']['full_scene_revalidation_required'])

    def test_real_amended_geometry_preserves_exact_spans_without_granting_scene_approval(self):
        from source_edge_reconstruction import source_edge_certificate_valid,final_source_edge_matches
        from clean_base import _parse_subpaths
        from gradient_contour_spans import _fit
        saved=copy.deepcopy(self.proposal)
        # A parent approval must never survive amendment of its geometry.
        parent=copy.deepcopy(self.proposal)
        parent['geometry']['source_edge_scene_commit']={'status':'committed','accepted':True}
        amended=preserve_source_edge_contacts(self.svg,parent,self.source,self.reference,self.guard,
                                             include_boundary_outliers=True)
        proof=amended['certificate']['contact_preservation']
        self.assertEqual(proof['status'],'proposal_only')
        self.assertTrue(proof['hybrid_reference_is_not_entirely_source_derived'])
        self.assertNotIn('source_edge_scene_commit',amended['geometry'])
        self.assertTrue(source_edge_certificate_valid(amended['geometry']))
        self.assertFalse(source_edge_certificate_valid(amended['geometry'],require_scene_commit=True))
        self.assertFalse(final_source_edge_matches(ET.fromstring(amended['candidate_svg_text']),
                                                  amended['geometry'],self.source,self.reference))
        self.assertLess(amended['geometry']['anchor_count'],176)
        old_path=next(n for n in ET.fromstring(self.svg).iter() if n.get('id')=='target').get('d')
        old=_fit(_parse_subpaths(old_path)[0])['segments']
        original_undirected={tuple(sorted((tuple(s['start']),tuple(s['end'])))) for s in old}
        self.assertTrue(proof['preserved_segments'])
        for segment in proof['preserved_segments']:
            self.assertEqual(segment['type'],'line')
            self.assertIn(tuple(sorted((tuple(segment['start']),tuple(segment['end'])))),original_undirected)
        for span in proof['spans']:
            self.assertLessEqual(max(span['join_distances_working_pixels']),.5+1e-9)
        before_other=next(n for n in ET.fromstring(self.svg).iter() if n.get('id')=='unrelated')
        after_other=next(n for n in ET.fromstring(amended['candidate_svg_text']).iter() if n.get('id')=='unrelated')
        self.assertEqual(ET.tostring(before_other),ET.tostring(after_other))
        self.assertEqual(self.proposal,saved)

    def test_real_geometry_certificate_cannot_excuse_unrelated_source_colour_damage(self):
        from source_scene_guard import validate_source_scene
        from source_edge_reconstruction import source_edge_certificate_valid
        parent=copy.deepcopy(self.proposal)
        tree=ET.fromstring(parent['candidate_svg_text'])
        next(n for n in tree.iter() if n.get('id')=='unrelated').set('fill','blue')
        parent['candidate_svg_text']=ET.tostring(tree,encoding='unicode')
        guard=copy.deepcopy(self.guard)
        guard['provenance']['after_svg_sha256']=_sha(parent['candidate_svg_text'])
        amended=preserve_source_edge_contacts(self.svg,parent,self.source,self.reference,guard,
                                             include_boundary_outliers=True)
        self.assertTrue(source_edge_certificate_valid(amended['geometry']))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);before=folder/'before.svg';after=folder/'after.svg'
            source=folder/'source.png';reference=folder/'reference.png'
            before.write_text(self.svg,encoding='utf8');after.write_text(amended['candidate_svg_text'],encoding='utf8')
            Image.fromarray(self.source).save(source);Image.fromarray(self.reference).save(reference)
            evidence=validate_source_scene(before,after,source,processed_reference_png=reference,
                                           source_edge_geometry=amended['geometry'])
        self.assertFalse(evidence['accepted'])
        self.assertTrue(any('color' in reason or 'colour' in reason for reason in evidence['reasons']),evidence['reasons'])

    def test_stage_rechecks_amended_candidate_and_rejected_scene_cannot_publish(self):
        from source_repair_stage import attempt_source_repairs
        checked=[]
        def guard(before,after,*args,**kwargs):
            candidate=Path(after).read_text(encoding='utf8');checked.append(candidate)
            if candidate==self.proposal['candidate_svg_text']:
                return copy.deepcopy(self.guard)
            return {'accepted':False,'reasons':['source_feature_color_error_increased']}
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);svg=folder/'before.svg';source=folder/'source.png';reference=folder/'reference.png'
            svg.write_text(self.svg,encoding='utf8');original=svg.read_bytes()
            Image.fromarray(self.source).save(source);Image.fromarray(self.reference).save(reference)
            details=[{'id':'g','candidate_id':'object','validation':{'geometry':{'anchor_count':176},'paint':{'retained':True}}}]
            saved=copy.deepcopy(details)
            with patch('source_edge_reconstruction.propose_source_edge_reconstruction',side_effect=lambda *a,**kw:copy.deepcopy(self.proposal)), \
                 patch('source_flat_paint.propose_source_flat_paint',return_value={'proposals':[]}), \
                 patch('source_light_cleanup.propose_light_fill_cleanup',return_value={'proposals':[]}), \
                 patch('source_scene_guard.validate_source_scene',side_effect=guard), \
                 patch('source_scene_guard.validate_source_scene_chain') as chain, \
                 patch('source_edge_contacts.preserve_source_edge_contacts',wraps=preserve_source_edge_contacts) as preserve:
                result,final_details=attempt_source_repairs(svg,source,reference,details)
            self.assertGreater(preserve.call_count,0)
            self.assertTrue(any(candidate!=self.proposal['candidate_svg_text'] for candidate in checked))
            self.assertEqual(result['status'],'unchanged')
            self.assertEqual(result['committed'],[])
            self.assertEqual(svg.read_bytes(),original)
            self.assertEqual(final_details,saved);self.assertEqual(details,saved)
            chain.assert_not_called()


if __name__=='__main__':
    unittest.main()
