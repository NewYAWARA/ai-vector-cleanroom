"""Actual native resvg evidence, not mocked component counts."""
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import numpy as np

from source_topology_audit import audit_source_topology
from svg_renderer import render_svg_reference


def document(body, width=128, height=96):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{body}</svg>'


class SourceTopologyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, original, candidate, *, background='#ffffff'):
        reference_svg = self.folder/'reference.svg'
        reference_svg.write_text(original, encoding='utf8')
        original_png = self.folder/'source.png'
        processed_png = self.folder/'processed.png'
        render_svg_reference(reference_svg, original_png, width=128, background=background)
        render_svg_reference(reference_svg, processed_png, width=128, background=None)
        output = self.folder/'candidate.svg'
        output.write_text(candidate, encoding='utf8')
        hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (original_png,processed_png,output)}
        report = audit_source_topology(output,original_png,processed_png)
        self.assertEqual(hashes,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (original_png,processed_png,output)})
        self.assertTrue(report['inputs_unchanged'])
        return report

    def test_two_pixel_channel_merge_is_spatially_localized_at_multiple_thresholds(self):
        original = document('<rect x="20" y="20" width="80" height="10" fill="#164d34"/>'
                            '<rect x="20" y="32" width="80" height="10" fill="#164d34"/>')
        candidate = document('<rect x="20" y="20" width="80" height="22" fill="#164d34"/>')
        result = self.run_case(original,candidate)
        self.assertEqual(result['status'],'completed')
        self.assertTrue(result['manual_review'])
        for kind in ('source_components_merged','paper_colored_channel_filled'):
            defect = next(row for row in result['stable_defects'] if row['kind']==kind)
            self.assertEqual(defect['bbox_xyxy'],[20,30,100,32])
            self.assertEqual(defect['pixels'],160)
            self.assertEqual(defect['stable_contrast_thresholds'],[32,64,96])

    def test_one_pixel_white_highlight_is_visual_content_even_when_processed_alpha_is_opaque(self):
        original = document('<rect x="20" y="20" width="80" height="30" fill="#164d34"/>'
                            '<rect x="24" y="34" width="72" height="1" fill="white"/>')
        candidate = document('<rect x="20" y="20" width="80" height="30" fill="#164d34"/>')
        result = self.run_case(original,candidate)
        self.assertTrue(result['manual_review'])
        channel = next(row for row in result['stable_defects'] if row['kind']=='paper_colored_channel_filled')
        self.assertEqual(channel['bbox_xyxy'],[24,34,96,35])
        self.assertTrue(all(p['processed_reference_rgba'][3]==255 for p in channel['probes']))
        self.assertNotIn('source_components_merged',result['reasons'])  # surrounding ink is one object

    def test_split_and_merge_cannot_cancel_by_equal_global_component_counts(self):
        original = document('<rect x="10" y="12" width="40" height="10" fill="#164d34"/>'
                            '<rect x="10" y="24" width="40" height="10" fill="#164d34"/>'
                            '<rect x="70" y="12" width="40" height="22" fill="#164d34"/>')
        candidate = document('<rect x="10" y="12" width="40" height="22" fill="#164d34"/>'
                             '<rect x="70" y="12" width="40" height="10" fill="#164d34"/>'
                             '<rect x="70" y="24" width="40" height="10" fill="#164d34"/>')
        result = self.run_case(original,candidate)
        self.assertTrue(result['manual_review'])
        self.assertIn('source_components_merged',result['reasons'])
        self.assertIn('source_component_split',result['reasons'])
        self.assertTrue(all(row['source_components']==row['candidate_components']==3 for row in result['thresholds']))

    def test_single_threshold_light_bridge_is_recorded_but_not_declared_stable(self):
        bodies = '<rect x="20" y="20" width="80" height="10" fill="#164d34"/><rect x="20" y="32" width="80" height="10" fill="#164d34"/>'
        result = self.run_case(document(bodies),document(bodies+'<rect x="20" y="30" width="80" height="2" fill="#d2d2d2"/>'))
        self.assertEqual(result['status'],'completed')
        self.assertFalse(result['manual_review'])
        self.assertGreater(result['thresholds'][0]['merge_reliable_pixels'],0)
        self.assertEqual(result['thresholds'][1]['merge_reliable_pixels'],0)

    def test_exact_round_butt_and_curved_strokes_do_not_false_fail(self):
        for body in (
            '<path d="M20 32 H110" stroke="#26394c" stroke-width="1" stroke-linecap="round"/>',
            '<path d="M20 32 H110" stroke="#26394c" stroke-width="6" stroke-linecap="butt"/>',
            '<path d="M20 70 C40 10 80 10 110 70" fill="none" stroke="#26394c" stroke-width="8" stroke-linecap="round"/>'):
            with self.subTest(body=body):
                result = self.run_case(document(body),document(body))
                self.assertEqual(result['status'],'completed')
                self.assertFalse(result['manual_review'])

    def test_opaque_pale_color_on_original_paper_is_local_warning_not_alpha_authorization(self):
        result=self.run_case(document(''),document('<rect x="20" y="20" width="70" height="4" fill="#dde8e2"/>'))
        self.assertTrue(result['manual_review'])
        warning=next(row for row in result['stable_defects'] if row['kind']=='source_paper_unexpected_opaque_color')
        self.assertEqual(warning['bbox_xyxy'],[20,20,90,24])
        self.assertEqual(warning['stable_contrast_thresholds'],[])
        self.assertIn('do not remove alpha',warning['white_object_vs_negative_space'])
        self.assertTrue(all(p['source_rgba']==[255,255,255,255] for p in warning['probes']))

    def test_paper_color_warning_preserves_white_pastel_and_gray_original_content(self):
        for paint in ('#ffffff','#dde8e2','#dddddd'):
            with self.subTest(paint=paint):
                svg=document(f'<rect x="20" y="20" width="70" height="4" fill="{paint}"/>')
                self.run_case(svg,svg)
                # Deliberately erase the processed reference: its alpha alone
                # cannot turn legitimate original paint into unwanted content.
                Image.new('RGBA',(128,96),(255,255,255,0)).save(self.folder/'processed.png')
                result=audit_source_topology(self.folder/'candidate.svg',self.folder/'source.png',self.folder/'processed.png')
                self.assertEqual(result['status'],'completed')
                self.assertNotIn('source_paper_unexpected_opaque_color',result['reasons'])

    def test_paper_core_warning_explicitly_abstains_on_sub_three_pixel_white_slit(self):
        ink='<rect x="20" y="20" width="80" height="10" fill="#164d34"/><rect x="20" y="32" width="80" height="10" fill="#164d34"/>'
        result=self.run_case(document(ink),document(ink+'<rect x="20" y="30" width="80" height="2" fill="#eeeeee"/>'))
        # A 17-level pale fill misses the contrast-topology thresholds, and the
        # source slit has no 3x3 white core. Report the known coverage limit.
        self.assertFalse(result['manual_review'])
        self.assertIn('one_or_two_pixel',result['paper_color_warning_policy']['coverage_limit'])

    def test_small_antialias_edge_shift_and_count_change_alone_are_not_topology_failures(self):
        original = document('<circle cx="60" cy="40" r="20" fill="#164d34"/>')
        candidate = document('<circle cx="60.3" cy="40.2" r="20" fill="#164d34"/>')
        result = self.run_case(original,candidate)
        self.assertFalse(result['manual_review'])
        # Extra isolated noise is reported as unmatched, never as a merge.
        result = self.run_case(original,document('<circle cx="60" cy="40" r="20" fill="#164d34"/><rect x="110" y="80" width="3" height="3" fill="#164d34"/>'))
        self.assertTrue(any(r['source_components']!=r['candidate_components'] for r in result['thresholds']))
        self.assertNotIn('source_components_merged',result['reasons'])
        self.assertNotIn('source_component_split',result['reasons'])
        # Independently, nine fully opaque dark pixels on proven blank paper
        # are a real colour warning, not a component-count inference.
        self.assertEqual(result['reasons'],['source_paper_unexpected_opaque_color'])

    def test_diagonal_contact_is_four_connected_not_eight_connected(self):
        svg = document('<rect x="20" y="20" width="8" height="8" fill="#164d34"/><rect x="28" y="28" width="8" height="8" fill="#164d34"/>')
        result = self.run_case(svg,svg)
        self.assertTrue(all(row['source_components']==2 for row in result['thresholds']))
        self.assertFalse(result['manual_review'])

    def test_nonpaper_transparent_and_missing_reference_are_not_claimed_verified(self):
        svg = document('<rect x="20" y="20" width="80" height="10" fill="#164d34"/>')
        for background in ('#303840',None):
            with self.subTest(background=background):
                result = self.run_case(svg,svg,background=background)
                self.assertEqual(result['status'],'not_applicable')
                self.assertNotIn('no_stable_defect_detected',result)
        self.run_case(svg,svg)
        result = audit_source_topology(self.folder/'candidate.svg',self.folder/'source.png')
        self.assertEqual(result['status'],'not_applicable')

    def test_reference_mismatch_and_zero_budget_do_not_silently_resize_or_pass(self):
        svg = document('<rect x="20" y="20" width="80" height="10" fill="#164d34"/>')
        self.run_case(svg,svg)
        with Image.open(self.folder/'processed.png') as image:
            image.resize((64,48)).save(self.folder/'smaller.png')
        result = audit_source_topology(self.folder/'candidate.svg',self.folder/'source.png',self.folder/'smaller.png')
        self.assertEqual(result['status'],'unavailable')
        result = audit_source_topology(self.folder/'candidate.svg',self.folder/'source.png',self.folder/'processed.png',maximum_seconds=0)
        self.assertEqual(result['status'],'unavailable')
        with self.assertRaises(ValueError):
            audit_source_topology('unused','unused',maximum_seconds=float('nan'))

    def test_rounded_native_viewport_matches_independent_analytic_source(self):
        import resvg_py
        from source_scene_guard import _render_native_payload
        # Old width-only rendering returned 96x65 (or a 95px fit-box result),
        # despite an actual 96x64 source. Both orientations require a centred
        # letterbox and uniform scaling, never independent x/y stretching.
        for ww,wh,nw,nh in ((64,43,96,64),(43,64,64,96)):
            scale=min(nw/ww,nh/wh);dx=(nw-ww*scale)/2;dy=(nh-wh*scale)/2
            body='<rect x="10" y="10" width="20" height="8" fill="#164d34"/><rect x="10" y="20" width="20" height="8" fill="#164d34"/>'
            text=f'<svg xmlns="http://www.w3.org/2000/svg" width="{ww}" height="{wh}" viewBox="0 0 {ww} {wh}">{body}</svg>'
            analytic=f'<svg xmlns="http://www.w3.org/2000/svg" width="{nw}" height="{nh}"><g transform="translate({dx} {dy}) scale({scale})">{body}</g></svg>'
            payload=resvg_py.svg_to_bytes(svg_string=analytic,background=None,skip_system_fonts=True,
                log_information=False,shape_rendering='geometric_precision')
            with Image.open(io.BytesIO(payload)) as image:reference=np.asarray(image.convert('RGBA')).copy()
            actual=_render_native_payload(text.encode('utf8'),nw,nh)
            np.testing.assert_array_equal(actual,reference)
            alpha=reference[:,:,3:4]/255
            source=reference.copy();source[:,:,:3]=np.rint(reference[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);source[:,:,3]=255
            Image.fromarray(source).save(self.folder/'source.png');Image.fromarray(reference).save(self.folder/'processed.png')
            (self.folder/'candidate.svg').write_text(text,encoding='utf8')
            result=audit_source_topology(self.folder/'candidate.svg',self.folder/'source.png',self.folder/'processed.png')
            self.assertEqual(result['status'],'completed',result)
            self.assertFalse(result['manual_review'],result['reasons'])
            self.assertEqual([result['renderer']['width'],result['renderer']['height']],[nw,nh])
            self.assertEqual(result['renderer']['rgba_sha256'],hashlib.sha256(reference.tobytes()).hexdigest())

    def test_private_full_e3_and_thin_machine_evidence(self):
        root = Path(__file__).resolve().parents[4]
        e3 = root/'work/p3/e3-full/result_gaps'
        if not (e3/'gaps_vector.svg').is_file():
            self.skipTest('Private full E3 candidate is not distributed')
        result = audit_source_topology(e3/'gaps_vector.svg',e3/'source_original.png',e3/'source_reference.png')
        self.assertEqual(result['status'],'completed')
        self.assertTrue(result['manual_review'])
        self.assertTrue(any(row['kind']=='source_components_merged' and row['bbox_xyxy'][1]==73
                            and row['pixels']>=100 for row in result['stable_defects']))
        evidence = root/'work/p3/data'
        for size in (128,384):
            result = audit_source_topology(evidence/f'caps-verification/06_thin_lines-{size}/actual.svg',
                evidence/f'caps-verification/06_thin_lines-{size}.png',evidence/f'source-topology/thin-{size}-reference.png')
            self.assertEqual(result['status'],'completed')
            self.assertFalse(result['manual_review'])

    def test_private_tea_gray_bands_have_original_paper_core_evidence(self):
        root=Path(__file__).resolve().parents[4]
        folder=root/'work/p3/t2'
        if not (folder/'tea_vector.svg').is_file():
            self.skipTest('Private development tea fixture is not distributed')
        result=audit_source_topology(folder/'tea_vector.svg',folder/'source_original.png',folder/'source_reference.png')
        warnings=[row for row in result['stable_defects'] if row['kind']=='source_paper_unexpected_opaque_color']
        self.assertTrue(any(row['bbox_xyxy']==[690,922,725,928] and row['pixels']>=100 for row in warnings))
        self.assertTrue(any(row['bbox_xyxy']==[695,952,726,959] and row['pixels']>=100 for row in warnings))


if __name__=='__main__':
    unittest.main()
