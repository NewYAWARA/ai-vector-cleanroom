from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

import auto_prepare as prepare
from designer_handoff import build_handoff_manifest, normalized_svg_text
from source_primitive import propose_source_ellipse
from svg_renderer import render_svg_reference


class SourcePrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'result_source'; self.folder.mkdir()
        self.svg = self.folder / 'source_vector.svg'
        points = [(64 + (40 + .5 * math.sin(i * 1.9)) * math.cos(i * math.tau / 24),
                   64 + (24 + .5 * math.cos(i * 1.3)) * math.sin(i * math.tau / 24)) for i in range(24)]
        data = 'M' + ' L'.join(f'{x:.8f} {y:.8f}' for x,y in points) + ' Z'
        self.svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><path id="p" fill="#255c3e" d="{data}"/></svg>', encoding='utf8')
        self.source = self.folder / 'source_original.png'
        self._source('<ellipse cx="64" cy="64" rx="40" ry="24" fill="#255c3e"/>')
        (self.folder / 'report.json').write_text('{}')
        self.tree = ET.fromstring(normalized_svg_text(self.svg))

    def _source(self, shape):
        raw = self.root / 'source-authoring.svg'
        raw.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'+shape+'</svg>', encoding='utf8')
        render_svg_reference(raw, self.source, width=128)

    def _propose(self, tree=None):
        return propose_source_ellipse(self.tree if tree is None else tree, self.source, 'p', .25, self.root, render_svg_reference)

    def _payload(self):
        manifest=build_handoff_manifest(self.svg)
        return {'result': self.folder.name, 'revision': None, 'svg_sha256':manifest['svg_sha256'],
                'decisions':{'p':'review'}, 'error_budget_percent':.25}

    def test_real_source_fit_is_native_four_anchors_and_strictly_improves_original(self):
        candidate, gate = self._propose()
        self.assertEqual(prepare._local(prepare._by_id(candidate)['p'].tag), 'ellipse')
        self.assertEqual(prepare._geometry(prepare._by_id(candidate)['p'])[0], 4)
        self.assertEqual(gate['source_components_and_holes'], [1,0])
        self.assertEqual(gate['source_contour']['salient_corner_count'], 0)
        self.assertLess(gate['candidate_mean_absolute_rgb_error'], gate['baseline_mean_absolute_rgb_error'])
        self.assertTrue(gate['not_parent_equivalent_simplification'])

    def test_actual_bounded_auto_prepare_uses_distinct_source_proposal_and_preserves_original(self):
        original = self.svg.read_bytes()
        result=prepare.prepare_result(self.root,self._payload(),lock=threading.RLock())
        self.assertEqual(result['summary']['source_reconstructed_paths'],1,result)
        self.assertEqual(result['summary']['curve_simplified_paths'],0)
        self.assertEqual(result['units'][0]['proposal_kind'],'source_primitive')
        self.assertEqual(self.svg.read_bytes(),original)
        derived=self.root/result['url'].split('result=',1)[1]
        report=json.loads((derived/'report.json').read_text(encoding='utf8'))
        self.assertEqual(report['acceptance_status'],'manual_review')
        self.assertEqual(report['local_refine_validation']['scope'],'source_primitive_reconstruction_original_image_fit')
        self.assertFalse(report['auto_prepare']['evidence']['fixed_original_parent_baseline'])
        self.assertEqual(report['local_refine_history'][0]['member_ids'],['p'])
        self.assertTrue((derived/report['recolor_page']).is_file())

    def test_keep_lock_prevents_source_reconstruction(self):
        payload=self._payload();payload['decisions']['p']='keep'
        result=prepare.prepare_result(self.root,payload,lock=threading.RLock())
        self.assertIsNone(result['url']);self.assertEqual(result['summary']['attempted_paths'],0)

    def test_processed_reference_cannot_be_used_as_original_primitive_evidence(self):
        self.source.rename(self.folder/'source_reference.png')
        def worker(request, timeout):
            self.assertFalse(request['allow_source_primitive'])
            return {'status':'skipped','reason':'no_safe_reduction'}
        with patch.object(prepare,'_invoke_worker',side_effect=worker):
            result=prepare.prepare_result(self.root,self._payload(),lock=threading.RLock())
        self.assertEqual(result['summary']['source_reconstructed_paths'],0)

    def test_even_one_faint_interior_detail_prevents_solid_primitive_claim(self):
        with Image.open(self.source) as image:
            changed=image.convert('RGB')
        changed.putpixel((64,64), (146,174,158))
        changed.save(self.source)
        with self.assertRaisesRegex(ValueError,'colour_or_alpha_ambiguity'):
            self._propose()

    def test_non_ellipses_corners_holes_and_multiple_shapes_are_not_invented_as_ellipse(self):
        pear='M'+' L'.join(f'{64+40*math.cos(t)*(1+.23*math.sin(t))} {64+24*math.sin(t)}' for t in [i*math.tau/128 for i in range(128)])+' Z'
        shapes=[
            '<rect x="24" y="40" width="80" height="48" fill="#255c3e"/>',
            '<rect x="24" y="40" width="80" height="48" rx="8" fill="#255c3e"/>',
            f'<path d="{pear}" fill="#255c3e"/>',
            '<ellipse cx="64" cy="64" rx="40" ry="24" fill="#255c3e"/><circle cx="64" cy="64" r="5" fill="white"/>',
            '<ellipse cx="64" cy="64" rx="40" ry="24" fill="#255c3e"/><circle cx="12" cy="12" r="3" fill="#255c3e"/>',
        ]
        for shape in shapes:
            with self.subTest(shape=shape[:60]):
                self._source(shape)
                with self.assertRaises(ValueError):self._propose()

    def test_gradient_texture_and_transparency_are_rejected(self):
        self._source('<defs><linearGradient id="g"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><ellipse cx="64" cy="64" rx="40" ry="24" fill="url(#g)"/>')
        with self.assertRaisesRegex(ValueError,'colour'):self._propose()
        self._source('<ellipse cx="64" cy="64" rx="40" ry="24" fill="#255c3e"/>')
        rgba=Image.open(self.source).convert('RGBA');rgba.putalpha(200);rgba.save(self.source)
        with self.assertRaisesRegex(ValueError,'transparency'):self._propose()

    def test_overlap_context_opacity_and_transform_are_rejected(self):
        for attrs in ({'opacity':'.5'},{'transform':'translate(0 0)'},{'stroke':'black'},
                      {'fill':'rgba(20,50,20,0.5)'},{'fill':'#255c3e80'}):
            tree=copy.deepcopy(self.tree);prepare._by_id(tree)['p'].attrib.update(attrs)
            with self.subTest(attrs=attrs):
                with self.assertRaises(ValueError):self._propose(tree)
        tree=copy.deepcopy(self.tree);ET.SubElement(tree,'{http://www.w3.org/2000/svg}circle',{'r':'1'})
        with self.assertRaisesRegex(ValueError,'isolated'):self._propose(tree)


if __name__=='__main__':unittest.main()
