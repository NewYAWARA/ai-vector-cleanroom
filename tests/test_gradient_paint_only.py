import copy
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from gradient_paint_only import (apply_paint_only_gradient,
                                 apply_partial_paint_only_gradient,
                                 apply_pending_paint_alternatives,
                                 _native_field_mask,
                                 paint_only_certificate_valid,
                                 final_paint_only_matches)
from svg_renderer import render_svg_reference


class GradientPaintOnlyTests(unittest.TestCase):
    def test_native_field_mask_uses_meet_pixel_cells_and_keeps_padding_outside(self):
        mask = np.zeros((43, 64), dtype=bool)
        mask[10:30, :41] = True
        old = np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((96,64), Image.Resampling.NEAREST)) > 0
        mapped = _native_field_mask(mask, 96, 64)
        self.assertTrue(np.all(old[15:45,61]))
        self.assertFalse(np.any(mapped[15:45,61]))
        self.assertEqual(int((old & ~mapped).sum()), 30)
        self.assertFalse(np.any(mapped & ~old))
        np.testing.assert_array_equal(_native_field_mask(mask.T,64,96),mapped.T)
        np.testing.assert_array_equal(_native_field_mask(mask,64,43),mask)
        padded = _native_field_mask(np.ones((2,2),bool),6,2)
        np.testing.assert_array_equal(padded,np.array([[False,False,True,True,False,False]]*2))

    def test_nonsquare_partial_paint_outside_field_matches_native_source_guard(self):
        import resvg_py
        from source_scene_guard import _render_native_payload
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            head='<svg xmlns="http://www.w3.org/2000/svg" width="64" height="43" viewBox="0 0 64 43">'
            gradient='<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="8" y1="0" x2="56" y2="0"><stop offset="0" stop-color="#184838"/><stop offset="1" stop-color="#98a878"/></linearGradient></defs>'
            paths='<path id="left" d="M8 8H32V35H8Z" fill="#386048"/><path id="right" d="M32 8H56V35H32Z" fill="#789068"/>'
            before=head+paths+'</svg>'
            proposal=head+gradient+'<path d="M8 8H56V35H8Z" fill="url(#g)"/></svg>'
            scale=64/43;offset=(96-64*scale)/2
            # This source has explicit native coordinates, independently of
            # the production field mapper or viewport replacement helper.
            native=(f'<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64"><defs>'
                f'<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="{8*scale+offset}" y1="0" x2="{56*scale+offset}" y2="0">'
                '<stop offset="0" stop-color="#184838"/><stop offset="1" stop-color="#98a878"/></linearGradient></defs>'
                f'<rect x="{8*scale+offset}" y="{8*scale}" width="{48*scale}" height="{27*scale}" fill="url(#g)"/></svg>')
            png=resvg_py.svg_to_bytes(svg_string=native,background=None,skip_system_fonts=True,
                                     log_information=False,shape_rendering='geometric_precision')
            with Image.open(io.BytesIO(png)) as image:reference=np.asarray(image.convert('RGBA')).copy()
            source=reference.copy();alpha=reference[:,:,3:4]/255
            source[:,:,:3]=np.rint(reference[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8);source[:,:,3]=255
            original=folder/'source.png';Image.fromarray(source).save(original)
            processed=_render_native_payload(proposal.encode(),64,43)
            mask=np.zeros((43,64),bool);mask[8:35,8:45]=True
            region={'mask':mask,'id':'g','candidate_id':'nonsquare-field','proposal_id':'nonsquare-object'}
            svg,geometry,proof=apply_partial_paint_only_gradient(before,proposal,region,original,processed,
                                                               native_reference_rgba=reference)
            self.assertTrue(paint_only_certificate_valid(geometry))
            self.assertTrue(final_paint_only_matches(ET.fromstring(svg),geometry,original))
            self.assertTrue(all(row['accepted'] for row in proof['source_scene_checks']))
            native_measurement=next(row for row in proof['measurements'] if row['source_reference_kind']=='unmodified_input')
            yy,xx=np.indices((64,96));wx=(xx+.5-offset)/scale;wy=(yy+.5)/scale
            expected=(wx>=8)&(wx<45)&(wy>=8)&(wy<35)
            self.assertEqual(native_measurement['field_mask_sha256'],hashlib.sha256(expected.tobytes()).hexdigest())
            self.assertEqual(native_measurement['field_mask_sampling'],'inverse_default_meet_native_pixel_centres_half_open_working_cells')
            outside=next(row for row in native_measurement['local'] if row['scope']=='right:outside_field')
            self.assertGreater(outside['pixels'],0)
            self.assertLessEqual(outside['measurements']['after']['rgb_mae'],outside['measurements']['before']['rgb_mae'])

    def fixture(self, folder):
        head = '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">'
        paths = '<path id="left" d="M8 8H48V88H8Z" fill="#386048"/><path id="right" d="M48 8H88V88H48Z" fill="#789068"/>'
        before = head + paths + '</svg>'
        gradient = '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" x1="8" y1="0" x2="88" y2="0"><stop offset="0" stop-color="#184838"/><stop offset="1" stop-color="#98a878"/></linearGradient></defs>'
        proposal = head+gradient+'<path fill="url(#g)" d="M8 8H88V88H8Z"/></svg>'
        source_svg = folder/'source.svg'
        source_svg.write_text(proposal, encoding='utf8')
        source = folder/'source_original.png'
        render_svg_reference(source_svg, source, 96, background='white')
        rgba = np.asarray(Image.open(source).convert('RGBA')).copy()
        mask = np.zeros((96, 96), dtype=bool)
        mask[8:88, 8:88] = True
        rgba[~mask, 3] = 0
        region = {'mask': mask, 'id': 'g', 'candidate_id': 'source-field-1', 'proposal_id': 'gradient-object-1'}
        return before, proposal, source, rgba, region

    def test_native_same_geometry_paint_restore_and_final_join(self):
        from vector_cleanroom import (_gradient_geometry_digest,
                                      _gradient_geometry_snapshot,
                                      _final_gradient_report_details)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, proposal, source, rgba, region = self.fixture(folder)
            svg, geometry, proof = apply_paint_only_gradient(before, proposal, region, source, rgba)
            self.assertTrue(paint_only_certificate_valid(geometry))
            self.assertEqual([r['id'] for r in proof['drawable_records']], ['left', 'right'])
            self.assertEqual(geometry['anchor_count'], 8)
            self.assertFalse(proof['geometry_optimized'])
            self.assertTrue(final_paint_only_matches(ET.fromstring(svg), geometry, source))
            target = folder/'final.svg'
            target.write_text(svg, encoding='utf8')
            guard = {'gradient_geometry_guard': {'before_geometry_sha256': _gradient_geometry_digest(_gradient_geometry_snapshot(target))}}
            details, summary = _final_gradient_report_details(target, [{'id': proof['gradient_id'], 'validation': {'geometry': geometry}}], guard)
            self.assertEqual(summary['gradient_drawable_count'], 2)
            self.assertEqual(details[0]['validation']['geometry']['final_svg_consistency']['final_drawable_ids'], ['left', 'right'])
            from designer_quality import _source_space_gradient_field_evidence
            from tests.test_designer_quality import _source_certified_field_metadata
            report = _source_certified_field_metadata()
            detail = report['gradient_details'][0]
            detail['id'] = proof['gradient_id']
            detail['validation']['geometry'] = details[0]['validation']['geometry']
            detail['validation']['selection']['geometry_source'] = 'unchanged_existing_svg_geometry'
            source_evidence = _source_space_gradient_field_evidence(report, 2)
            self.assertTrue(source_evidence['authoritative'], source_evidence)
            self.assertFalse(source_evidence['objects'][0]['economy_certificate_available'])
            for mutation in ('geometry', 'paint', 'context'):
                root = ET.fromstring(svg)
                left = next(n for n in root.iter() if n.get('id') == 'left')
                if mutation == 'geometry':
                    left.set('d', 'M9 8H48V88H9Z')
                elif mutation == 'paint':
                    next(n for n in root.iter() if n.tag.endswith('stop')).set('stop-color', '#ffffff')
                else:
                    root.set('viewBox', '0 0 192 96')
                self.assertFalse(final_paint_only_matches(root, geometry, source), mutation)
            damaged = rgba.copy(); damaged[20,20,:3] = 255
            Image.fromarray(damaged).save(source)
            self.assertFalse(final_paint_only_matches(ET.fromstring(svg), geometry, source))

    def test_original_local_regression_rejected_even_when_other_path_improves(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, proposal, source, rgba, region = self.fixture(folder)
            # Original left object already matches its retained solid fill;
            # the overall scene still benefits, but local source must win.
            changed = rgba.copy(); changed[8:88,8:48,:3] = (56,96,72)
            Image.fromarray(changed).save(source)
            with self.assertRaisesRegex(ValueError, 'unmodified_input_paint_only_local_source_fidelity_worse:left'):
                apply_paint_only_gradient(before, proposal, region, source, rgba)

    def test_palette_band_tail_policy_is_explicit_and_cannot_exempt_original(self):
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            _, geometry, _ = apply_paint_only_gradient(before, proposal, region, source, rgba)
            changed = copy.deepcopy(geometry)
            processed = changed['source_paint_only']['measurements'][1]['local'][1]
            processed['measurements']['after']['rgb_p95'] = processed['measurements']['before']['rgb_p95'] + .75
            self.assertTrue(paint_only_certificate_valid(changed))
            original = changed['source_paint_only']['measurements'][0]['local'][1]
            original['measurements']['after']['rgb_p95'] = original['measurements']['before']['rgb_p95'] + .01
            self.assertFalse(paint_only_certificate_valid(changed))

    def test_ownership_cannot_repaint_unrelated_path_or_ignore_missing_half(self):
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'complete_existing_path_ownership'):
                apply_paint_only_gradient(before.replace('M48 8H88V88H48Z', 'M48 8H94V88H48Z'), proposal, region, source, rgba)

    def test_partial_route_retains_geometry_alpha_and_requires_review(self):
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            svg, geometry, proof = apply_partial_paint_only_gradient(before, proposal, region, source, rgba)
            self.assertTrue(paint_only_certificate_valid(geometry))
            self.assertTrue(final_paint_only_matches(ET.fromstring(svg), geometry, source))
            self.assertEqual([row['id'] for row in proof['drawable_records']], ['left', 'right'])
            self.assertTrue(proof['manual_review_required'])
            self.assertFalse(proof['complete_field_reconstructed'])
            self.assertFalse(proof['geometry_optimized'])
            original_paths = [(n.get('id'), n.get('d')) for n in ET.fromstring(before).iter() if n.tag.endswith('path')]
            final_paths = [(n.get('id'), n.get('d')) for n in ET.fromstring(svg).iter() if n.tag.endswith('path')]
            self.assertEqual(original_paths, final_paths)
            altered = copy.deepcopy(geometry)
            altered['source_paint_only']['manual_review_required'] = False
            self.assertFalse(paint_only_certificate_valid(altered))

    def test_partial_route_preserves_path_whose_original_source_would_worsen(self):
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            changed = rgba.copy(); changed[8:88,8:48,:3] = (56,96,72)
            Image.fromarray(changed).save(source)
            svg, geometry, proof = apply_partial_paint_only_gradient(before, proposal, region, source, rgba)
            self.assertEqual([r['id'] for r in proof['drawable_records']], ['right'])
            self.assertEqual(next(n for n in ET.fromstring(svg).iter() if n.get('id') == 'left').get('fill'), '#386048')
            self.assertTrue(paint_only_certificate_valid(geometry))

    def test_partial_route_checks_outside_field_and_rejects_forged_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            # The right path is only partly covered by the discovered field.
            # Its exterior still has to match the actual source independently.
            region['mask'][:, 68:] = False
            svg, geometry, proof = apply_partial_paint_only_gradient(before, proposal, region, source, rgba)
            self.assertEqual(len(proof['drawable_records']), 2)
            original = geometry['source_paint_only']['measurements'][0]
            outside = next(r for r in original['local'] if r['scope'] == 'right:outside_field')
            self.assertGreater(outside['pixels'], 0)
            damaged = copy.deepcopy(geometry)
            row = next(r for r in damaged['source_paint_only']['measurements'][0]['local'] if r['scope'] == 'right:outside_field')
            row['measurements']['after']['rgb_p95'] = row['measurements']['before']['rgb_p95'] + .1
            self.assertFalse(paint_only_certificate_valid(damaged))
            source_bad = rgba.copy(); source_bad[8:88,68:88,:3] = (120,144,104)
            Image.fromarray(source_bad).save(source)
            _, _, proof_bad = apply_partial_paint_only_gradient(before, proposal, region, source, rgba)
            self.assertNotIn('right', [r['id'] for r in proof_bad['drawable_records']])

    def test_partial_route_preserves_existing_gradient_owner_and_search_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            owned = before.replace('<path id="left"', '<path data-avc-gradient-object="earlier" id="left"')
            svg, _, proof = apply_partial_paint_only_gradient(owned, proposal, region, source, rgba)
            self.assertEqual([r['id'] for r in proof['drawable_records']], ['right'])
            with self.assertRaisesRegex(ValueError, 'search_budget'):
                apply_partial_paint_only_gradient(before, proposal, region, source, rgba, maximum_trials=17)

    def test_pending_partial_route_final_join_is_valid_but_never_designer_ready(self):
        from gradient_reconstruction_stage import encode_mask_rle
        from vector_cleanroom import (_gradient_geometry_digest, _gradient_geometry_snapshot,
                                      _final_gradient_report_details)
        from designer_quality import _source_space_gradient_field_evidence, audit_designer_quality
        from tests.test_designer_quality import _source_certified_field_metadata
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            before, proposal, source, rgba, region = self.fixture(folder)
            template = _source_certified_field_metadata()
            option = {'schema': 'ai-vector-cleanroom.paint-ready-alternative/v1',
                'status': 'pending_native_source_and_existing_path_validation',
                'geometry_certified': False, 'original_source_verified': False,
                'candidate_id': region['candidate_id'], 'candidate_family': 'fixture',
                'mask': encode_mask_rle(region['mask']),
                'model': {'type': 'linear', 'x1': 8, 'y1': 0, 'x2': 88, 'y2': 0},
                'stops': [{'offset': 0, 'color': '#184838'}, {'offset': 1, 'color': '#98a878'}],
                'heldout_evidence': template['gradient_details'][0]['validation']['paint']}
            second_option = copy.deepcopy(option); second_option['candidate_id'] = 'second-unexamined-field'
            svg, details, decisions, proofs = apply_pending_paint_alternatives(before, [option, second_option], source, rgba)
            self.assertEqual(len(details), 1)
            self.assertEqual(len(decisions), 1)
            with self.assertRaisesRegex(ValueError, 'one_field_transaction_limit'):
                apply_pending_paint_alternatives(before, [option, second_option], source, rgba, maximum_fields=2)
            self.assertEqual(decisions[0]['status'], 'partial_paint_selected')
            target = folder/'final.svg'; target.write_text(svg, encoding='utf8')
            guard = {'gradient_geometry_guard': {'before_geometry_sha256': _gradient_geometry_digest(_gradient_geometry_snapshot(target))}}
            joined, summary = _final_gradient_report_details(target, details, guard)
            template['gradient_details'] = joined
            report = template['gradient_reconstruction_report']
            report['status'] = 'skipped'
            report['summary']['objects_selected'] = 0
            report['summary']['partial_paint_fields'] = 1
            report['decisions'] = decisions
            evidence = _source_space_gradient_field_evidence(template, 2)
            self.assertFalse(evidence['authoritative'], evidence)
            self.assertFalse(evidence['failure_reasons'], evidence)
            result = audit_designer_quality(target, proposal_metadata=template)
            self.assertFalse(result['designer_ready'], result)
            self.assertIn('partial_source_paint_requires_manual_review', result['gradient_object_gate']['warning_reasons'])

    def test_partial_scene_guard_rejects_seven_white_pixels_despite_mean_tail_gain(self):
        from source_scene_guard import _render_native_payload, _composite, validate_source_scene_arrays
        with tempfile.TemporaryDirectory() as temp:
            before, proposal, source, rgba, region = self.fixture(Path(temp))
            # One white-filled rectangle already matches seven source-paper
            # pixels. Repainting it improves thousands of gradient pixels but
            # must not bury that true white feature under opaque green paint.
            head = '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">'
            before = head+'<path id="body" d="M8 8H88V88H8Z" fill="#ffffff"/></svg>'
            raw = np.asarray(Image.open(source).convert('RGBA')).copy()
            raw[40, 40:47, :3] = 255
            Image.fromarray(raw).save(source)
            reference = rgba.copy(); reference[40, 40:47, :3] = 255; reference[40, 40:47, 3] = 0
            a = _render_native_payload(before.encode(), 96, 96)
            b = _render_native_payload(proposal.encode(), 96, 96)
            errors = [np.abs(_composite(image)-_composite(raw)).mean(2) for image in (a, b)]
            self.assertLess(errors[1].mean(), errors[0].mean())
            self.assertLess(np.percentile(errors[1], 95), np.percentile(errors[0], 95))
            self.assertTrue(np.array_equal(a[:, :, 3], b[:, :, 3]))
            guard = validate_source_scene_arrays(a, b, raw, processed_reference_rgba=reference)
            self.assertFalse(guard['accepted'], guard)
            with self.assertRaisesRegex(ValueError, 'no_path_passed_original_source_checks'):
                apply_partial_paint_only_gradient(before, proposal, region, source, reference)


if __name__ == '__main__':
    unittest.main()
