import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from gradient_contour_spans import (propose_gradient_contour_spans, SpanSearchRejected,
    source_contour_spans_certificate_valid, final_source_contour_spans_matches, _path)
from svg_renderer import render_svg_reference


class GradientContourSpanTests(unittest.TestCase):
    def fixture(self, folder):
        from trace_engine import _mask_to_smooth_loops
        from svg_postprocess import _designer_path_anchors
        mask=np.zeros((80,80),dtype=bool);mask[10:70,10:70]=True
        mask[37:42,37:42]=False
        loops=_mask_to_smooth_loops(mask,simplify=0,min_area=1,smooth=0)
        commands=[]
        for index,loop in enumerate(loops):
            points=np.asarray(loop,dtype=float)
            if index==0:
                points=np.concatenate([a+(b-a)*np.arange(16)[:,None]/16
                    for a,b in zip(points,np.roll(points,-1,axis=0))])
            commands.append('M'+','.join(map(str,points[0])))
            commands.extend('L'+','.join(map(str,p)) for p in points[1:])
            commands.append('Z')
        path=' '.join(commands)
        svg='<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80"><defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="80" y2="0"><stop offset="0" stop-color="#174f39"/><stop offset="1" stop-color="#73b57c"/></linearGradient></defs><path id="target" fill="url(#g)" fill-rule="evenodd" data-avc-gradient-object="object" d="'+path+'"/></svg>'
        svg_path=folder/'source.svg';svg_path.write_text(svg,encoding='utf8');source=folder/'source_original.png'
        render_svg_reference(svg_path,source,80,background='white')
        rgba=np.asarray(Image.open(source).convert('RGBA'))
        count=_designer_path_anchors(path)
        geometry={'path':path,'anchor_count':count,'designer_anchor_count':count,'segment_count':count,
                  'topology':{'topology_preserved':True},'error_budget':{'passed':True,'requested_max_percent':.25}}
        return svg,geometry,mask,source,rgba

    def test_actual_native_subset_retains_hole_and_binds_final_source_paint(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);svg,geometry,mask,source,rgba=self.fixture(folder)
            result,new,proof=propose_gradient_contour_spans(svg,geometry,mask,'target',source,rgba,
                                                          maximum_spans=8,maximum_probes=2)
            self.assertLess(new['designer_anchor_count'],geometry['designer_anchor_count'])
            self.assertLessEqual(proof['performance']['native_probe_count'],2)
            self.assertTrue(proof['render_guard']['alpha_topology']['accepted'])
            self.assertTrue(proof['render_guard']['composed_alpha']['accepted'])
            self.assertTrue(source_contour_spans_certificate_valid(new))
            self.assertTrue(final_source_contour_spans_matches(ET.fromstring(result),new,source))
            from vector_cleanroom import (_final_gradient_report_details,
                _gradient_geometry_snapshot,_gradient_geometry_digest)
            target=folder/'final.svg';target.write_text(result,encoding='utf8')
            guard={'gradient_geometry_guard':{'before_geometry_sha256':
                _gradient_geometry_digest(_gradient_geometry_snapshot(target))}}
            details,_=_final_gradient_report_details(target,[{'id':'g','validation':{'geometry':new}}],guard)
            self.assertTrue(details[0]['validation']['geometry']['final_svg_consistency']['source_contour_spans_verified'])
            for field in ('recall','precision','coverage_f1'):
                changed=copy.deepcopy(new)
                values=changed['source_contour_spans']['source_guards']['original']['scopes']['object_roi']['metrics'][field]
                values['after']=values['before']-.251
                self.assertFalse(source_contour_spans_certificate_valid(changed))
            changed=copy.deepcopy(new);changed['source_contour_spans']['render_guard']['composed_alpha']['accepted']=False
            self.assertFalse(source_contour_spans_certificate_valid(changed))
            root=ET.fromstring(result);next(n for n in root.iter() if n.get('id')=='target').set('d','M0 0H80V80Z')
            self.assertFalse(final_source_contour_spans_matches(root,new,source))
            changed_rgba=rgba.copy();changed_rgba[20,20,:3]=255;Image.fromarray(changed_rgba).save(source)
            self.assertFalse(final_source_contour_spans_matches(ET.fromstring(result),new,source))

    def test_zero_budget_returns_no_candidate_and_leaves_input_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            svg,geometry,mask,source,rgba=self.fixture(Path(temp));before=copy.deepcopy(geometry)
            with self.assertRaises(SpanSearchRejected) as caught:
                propose_gradient_contour_spans(svg,geometry,mask,'target',source,rgba,maximum_seconds=0)
            self.assertEqual(caught.exception.diagnostics['performance']['native_probe_count'],0)
            self.assertEqual(geometry,before)

    def test_untouched_coordinates_round_trip_full_float_precision(self):
        from clean_base import _parse_subpaths
        x=0.12345678912345678
        path=_path([{'type':'line','start':[x,1.],'end':[x,2.]}],False)
        parsed=_parse_subpaths(path)[0]
        self.assertEqual(parsed['start'][0],x)
        self.assertEqual(parsed['segs'][0][1],x)

    def test_original_alpha_metadata_is_replaced_from_pixels(self):
        from unittest.mock import patch
        import gradient_contour_spans as spans
        with tempfile.TemporaryDirectory() as temp:
            svg,geometry,mask,source,rgba=self.fixture(Path(temp))
            # A user-provided alpha-origin assertion is not trusted.
            info=PngInfo();info.add_text('avc_reference_alpha_origin','opaque_canvas_derived')
            Image.fromarray(rgba).save(source,pnginfo=info)
            real=spans._source_guard;seen=[]
            def checked(before,after,reference,*args):
                with Image.open(reference) as image:seen.append((Path(reference).name,image.info.get('avc_reference_alpha_origin')))
                return real(before,after,reference,*args)
            with patch.object(spans,'_source_guard',side_effect=checked):
                propose_gradient_contour_spans(svg,geometry,mask,'target',source,rgba,maximum_spans=8,maximum_probes=1)
            self.assertIn(('original.png','native'),seen)


if __name__=='__main__':
    unittest.main()
