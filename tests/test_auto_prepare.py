from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import auto_prepare as prepare
from designer_handoff import build_handoff_manifest, normalized_svg_text, _geometry
from handoff_service import HandoffConflict, save_decisions

DENSE = 'M10 10 L30 10 L50 10 L70 10 L90 10 L90 50 L90 90 L50 90 L10 90 L10 50 Z'
SIMPLE = 'M10 10 L90 10 L90 90 L10 90 Z'
PASSED = {'external_render_check': 'completed', 'accepted': True}


class AutoPrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / 'result_demo'
        self.folder.mkdir()
        self.svg = self.folder / 'demo_vector.svg'
        self.source = self.folder / 'source_original.png'
        self.source.write_bytes(b'original-source-retained')
        self.reference = self.folder / 'source_reference.png'
        self.reference.write_bytes(b'processed-reference-retained')
        self.report = self.folder / 'report.json'
        self.report.write_text(json.dumps({'acceptance_status': 'accepted', 'foreground_match_percent': 99}), encoding='utf-8')
        self._svg()

    def _svg(self, attrs='', extra='', defs=''):
        self.svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 110">'
            + defs + f'<path id="target" d="{DENSE}" fill="#126b45" {attrs}/>'
            + '<g id="object-locked"><circle id="locked" cx="180" cy="50" r="30" fill="red"/></g>'
            + extra + '</svg>', encoding='utf-8')

    def _payload(self):
        manifest = build_handoff_manifest(self.svg)
        return {'result': self.folder.name, 'svg_sha256': manifest['svg_sha256'], 'revision': None,
                'decisions': {unit['id']: ('keep' if unit['id'] == 'object-locked' else 'redraw')
                              for unit in manifest['objects']}, 'error_budget_percent': 0.25}

    def _worker(self, request, timeout):
        tree = ET.parse(request['current_svg']).getroot()
        element = prepare._by_id(tree)[request['member_id']]
        before = _geometry(element)[0]
        element.set('d', SIMPLE)
        prepare._write_svg(Path(request['work']) / 'candidate.svg', tree)
        return {'status': 'improved', 'anchors_before': before, 'anchors_after': 4,
                'target_gate': {'id': request['member_id'], 'target_silhouette_topology': 'passed',
                                'actual_paint': {'accepted': True}, 'render': dict(PASSED)},
                'whole_svg_render': {**PASSED, 'composed_alpha': dict(PASSED)}}

    def _run(self, payload=None, worker=None):
        with patch.object(prepare, '_invoke_worker', side_effect=worker or self._worker):
            return prepare.prepare_result(self.root, payload or self._payload(), lock=threading.RLock())

    def _derived(self, result):
        return self.root / result['url'].split('result=', 1)[1]

    def test_preserves_sources_keep_and_paints_then_invalidates_final_acceptance(self):
        originals = {path: path.read_bytes() for path in self.folder.iterdir()}
        result = self._run()
        derived = self._derived(result)
        for path, data in originals.items():
            self.assertEqual(path.read_bytes(), data)
        self.assertEqual((derived / self.source.name).read_bytes(), originals[self.source])
        self.assertEqual((derived / self.reference.name).read_bytes(), originals[self.reference])
        report = json.loads((derived / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['acceptance_status'], 'manual_review')
        self.assertIsNone(report['foreground_match_percent'])
        self.assertEqual(report['local_refine_history'][0]['member_ids'], ['target'])
        self.assertFalse(result['summary']['human_time_saving_validated'])
        self.assertEqual(result['summary']['anchors_removed'], 6)
        state = json.loads((derived / 'handoff_state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['decisions'], {'target': 'review', 'object-locked': 'keep'})
        a = ET.fromstring(normalized_svg_text(self.svg))
        b = ET.parse(next(derived.glob('*_vector.svg'))).getroot()
        self.assertEqual(ET.tostring(prepare._by_id(a)['object-locked']), ET.tostring(prepare._by_id(b)['object-locked']))
        self.assertEqual(prepare._by_id(b)['target'].get('fill'), '#126b45')

    def test_original_only_identity_is_retained(self):
        self.reference.unlink()
        derived = self._derived(self._run())
        self.assertTrue((derived / 'source_original.png').is_file())
        self.assertFalse((derived / 'source_reference.png').exists())

    def test_recolor_uses_exact_derived_geometry_ids_and_manual_review_metadata(self):
        self.svg.write_text(self.svg.read_text(encoding='utf-8').replace('fill="red"', 'fill="#ff0000"'), encoding='utf-8')
        derived = self._derived(self._run())
        report = json.loads((derived / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['recolor_artifacts']['status'], 'available_not_designer_validated')
        svg = next(derived.glob('*_vector.svg'))
        paint = json.loads((derived / report['paint_role_manifest']).read_text(encoding='utf-8'))
        self.assertEqual(paint['source']['sha256'], prepare._sha(svg.read_bytes()))
        self.assertEqual(paint['source']['sha256_scope'], 'exact_delivered_derived_svg')
        html = (derived / report['recolor_page']).read_text(encoding='utf-8')
        embedded = json.loads(re.search(r'<script id="svg-source" type="application/json">(.*?)</script>', html, re.S)[1])
        self.assertEqual(embedded, svg.read_text(encoding='utf-8'))
        self.assertEqual(prepare._by_id(ET.fromstring(embedded))['target'].get('d'), SIMPLE)
        self.assertEqual(report['acceptance_status'], 'manual_review')

    def test_recolor_unsupported_paint_is_explicit_and_does_not_lose_derivative(self):
        derived = self._derived(self._run())
        report = json.loads((derived / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['recolor_artifacts']['status'], 'unavailable')
        self.assertIn('not_supported', report['recolor_artifacts']['reason'])
        self.assertIsNone(report['recolor_page'])
        self.assertFalse((derived / '色彩調整.html').exists())

    def test_unsupported_paths_skip_while_other_paths_improve(self):
        self._svg(extra=f'<path id="transformed" d="{DENSE}" transform="translate(100 0)"/>')
        result = self._run()
        unit = next(unit for unit in result['units'] if unit['id'] == 'transformed')
        self.assertEqual(unit['status'], 'skipped')
        self.assertIn('unsupported_transform', unit['reason'])
        self.assertEqual(result['summary']['paths_improved'], 1)

    def test_per_path_failure_rolls_back_but_later_path_succeeds(self):
        self._svg(extra=f'<path id="second" d="{DENSE}" fill="purple"/>')
        def worker(request, timeout):
            if request['member_id'] == 'second':
                return {'status': 'skipped', 'reason': 'per_path_time_budget_exhausted'}
            return self._worker(request, timeout)
        result = self._run(worker=worker)
        self.assertEqual(result['summary']['paths_improved'], 1)
        second = prepare._by_id(ET.parse(next(self._derived(result).glob('*_vector.svg'))).getroot())['second']
        self.assertEqual(second.get('d'), DENSE)

    def test_incomplete_render_evidence_cannot_publish(self):
        def worker(request, timeout):
            result = self._worker(request, timeout)
            result['whole_svg_render']['external_render_check'] = 'unavailable'
            return result
        result = self._run(worker=worker)
        self.assertIsNone(result['url'])
        self.assertEqual(result['summary']['anchors_removed'], 0)
        self.assertEqual([path.name for path in self.root.iterdir()], ['result_demo'])

    def test_missing_or_failed_composed_alpha_evidence_cannot_publish(self):
        for alpha in (None, {'external_render_check': 'completed', 'accepted': False},
                      {'external_render_check': 'unavailable', 'accepted': True}):
            with self.subTest(alpha=alpha):
                def worker(request, timeout):
                    result = self._worker(request, timeout)
                    if alpha is None:
                        result['whole_svg_render'].pop('composed_alpha')
                    else:
                        result['whole_svg_render']['composed_alpha'] = alpha
                    return result
                self.assertIsNone(self._run(worker=worker)['url'])

    def test_boundary_rejects_paint_defs_keep_order_and_unchanged_geometry(self):
        for change in ('paint', 'keep', 'order', 'unchanged'):
            with self.subTest(change=change):
                def worker(request, timeout):
                    result = self._worker(request, timeout)
                    path = Path(request['work']) / 'candidate.svg'
                    tree = ET.parse(path).getroot()
                    ids = prepare._by_id(tree)
                    if change == 'paint':
                        ids['target'].set('fill', 'blue')
                    elif change == 'keep':
                        ids['locked'].set('r', '40')
                    elif change == 'order':
                        tree.append(tree[0]); tree.remove(tree[0])
                    else:
                        ids['target'].set('d', DENSE)
                    prepare._write_svg(path, tree)
                    return result
                self.assertIsNone(self._run(worker=worker)['url'])

    def test_missing_stale_revision_and_changed_svg_rejected(self):
        payload = self._payload()
        payload.pop('revision')
        with self.assertRaises(HandoffConflict):
            self._run(payload)
        payload = self._payload(); payload['svg_sha256'] = '0' * 64
        with self.assertRaises(HandoffConflict):
            self._run(payload)
        save_decisions(self.root, self._payload())
        with self.assertRaises(HandoffConflict):
            self._run()

    def test_change_during_work_cannot_publish(self):
        def worker(request, timeout):
            result = self._worker(request, timeout)
            self.source.write_bytes(b'new-original')
            return result
        with self.assertRaises(HandoffConflict):
            self._run(worker=worker)
        self.assertEqual([path.name for path in self.root.iterdir()], ['result_demo'])

    def test_decision_change_during_work_cannot_publish(self):
        def worker(request, timeout):
            result = self._worker(request, timeout)
            save_decisions(self.root, self._payload())
            return result
        with self.assertRaises(HandoffConflict):
            self._run(worker=worker)

    def test_prior_refinement_is_not_repeated(self):
        self.report.write_text(json.dumps({'local_refine_history': [{'member_ids': ['target']}]}), encoding='utf-8')
        with patch.object(prepare, '_invoke_worker') as worker:
            result = prepare.prepare_result(self.root, self._payload(), lock=threading.RLock())
        worker.assert_not_called()
        self.assertIsNone(result['url'])
        self.assertIn('already_refined', result['units'][0]['reason'])

    def test_budgets_stop_without_dropping_units(self):
        self._svg(extra=f'<path id="second" d="{DENSE}" fill="purple"/>')
        with patch.object(prepare, 'MAX_ATTEMPTS', 1):
            result = self._run()
        self.assertEqual(len(result['units']), 3)
        self.assertEqual(result['summary']['attempted_paths'], 1)
        self.assertTrue(any('budget_exhausted' in unit['reason'] for unit in result['units']))

    def test_gradient_preflight_retains_local_paint_support(self):
        tree = ET.fromstring(f'<svg><defs><linearGradient id="paint"><stop offset="0" stop-color="red"/></linearGradient></defs><g fill="url(#paint)"><path id="p" d="{DENSE}"/></g></svg>')
        self.assertEqual(prepare._path_support(tree, 'p', set()), (None, True))
        paint = prepare._target_document(tree, 'p', _geometry(prepare._by_id(tree)['p'])[1])
        self.assertEqual(len([e for e in paint.iter() if e.tag == 'linearGradient']), 1)

    def test_nested_gradient_dependency_and_cycles_are_explicit_skips(self):
        tree = ET.fromstring(f'<svg><defs><linearGradient id="paint" href="#nested"/></defs>'
            f'<g color="red"><defs><linearGradient id="nested"/></defs></g><path id="p" d="{DENSE}" fill="url(#paint)"/></svg>')
        self.assertEqual(prepare._path_support(tree, 'p', set())[0], 'unsupported_nested_gradient_context')
        prepare._by_id(tree)['paint'].set('href', '#paint')
        self.assertEqual(prepare._path_support(tree, 'p', set())[0], 'unsupported_cyclic_gradient_dependency')

    def test_timeout_is_explicit_skip(self):
        with patch.object(prepare.subprocess, 'run', side_effect=subprocess.TimeoutExpired('worker', 1)):
            result = prepare._invoke_worker({}, 1)
        self.assertEqual(result['reason'], 'per_path_time_budget_exhausted')

    def test_real_target_gate_rejects_removed_hole_and_changed_gradient_color(self):
        before = ET.fromstring('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<defs><linearGradient id="paint"><stop offset="0" stop-color="red"/>'
            '<stop offset="1" stop-color="blue"/></linearGradient></defs>'
            f'<path id="p" d="{SIMPLE} M30 30 L70 30 L70 70 L30 70 Z" fill="url(#paint)" fill-rule="evenodd"/></svg>')
        after = copy.deepcopy(before)
        prepare._by_id(after)['p'].set('d', SIMPLE)
        with self.assertRaisesRegex(ValueError, 'components_or_holes'):
            prepare._target_gate(before, after, 'p', self.root)
        after = copy.deepcopy(before)
        prepare._by_id(after)['paint'][0].set('stop-color', 'green')
        with self.assertRaisesRegex(ValueError, 'actual_paint'):
            prepare._target_gate(before, after, 'p', self.root)

    def test_local_source_crop_prevents_canvas_average_hiding_small_damage(self):
        import numpy as np
        from PIL import Image
        svg = self.root / 'base.svg'
        svg.write_text('<svg viewBox="0 0 100 100"/>')
        a = np.full((100, 100, 4), 255, dtype=np.uint8)
        a[40:50, 40:50, :3] = 100
        b = a.copy(); b[40:50, 40:50, :3] = 99
        Image.fromarray(a).save(self.root / 'parent-reference.png')
        Image.fromarray(a).save(self.root / 'raw.png')
        def render(src, png, width):
            Image.fromarray(b).save(png)
        with patch.object(prepare, '_render_reference', side_effect=render):
            with self.assertRaisesRegex(ValueError, 'local_original_reference_regression'):
                prepare._global_gate(svg, svg, self.root / 'raw.png', self.root, [40, 40, 10, 10])

    def test_worker_runs_outside_publish_lock(self):
        class Lock:
            held = False
            def __enter__(self): self.held = True
            def __exit__(self, *args): self.held = False
        lock = Lock()
        def worker(request, timeout):
            self.assertFalse(lock.held)
            return self._worker(request, timeout)
        with patch.object(prepare, '_invoke_worker', side_effect=worker):
            prepare.prepare_result(self.root, self._payload(), lock=lock)

    def test_real_worker_reduces_gradient_and_solid_preserves_keep(self):
        try:
            import resvg_py
        except ImportError:
            self.skipTest('native renderer dependency unavailable')
        self._svg(defs='<defs><linearGradient id="paint"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs>',
                  extra=f'<path id="gradient" d="{DENSE}" fill="url(#paint)" opacity="0.6"/>')
        prepare._render_reference(self.svg, self.source, 880)
        result = prepare.prepare_result(self.root, self._payload(), lock=threading.RLock())
        self.assertEqual(result['summary']['paths_improved'], 2, result['units'])
        self.assertEqual(result['summary']['anchors_removed'], 12)
        derived = self._derived(result)
        tree = ET.parse(next(derived.glob('*_vector.svg'))).getroot()
        self.assertEqual(prepare._by_id(tree)['gradient'].get('fill'), 'url(#paint)')
        self.assertEqual(prepare._by_id(tree)['gradient'].get('opacity'), '0.6')

    def test_real_worker_rejects_white_near_contact_hole_and_component_changes(self):
        from PIL import Image
        import numpy as np
        from alpha_topology import compare_alpha_topology
        from stroke_engine import connected_components
        from svg_renderer import render_svg_reference
        points = [(20 + 10 * math.cos((i + .5) * math.tau / 32),
                   60 + 30 * math.sin((i + .5) * math.tau / 32)) for i in range(32)]
        path = 'M' + ' L'.join(f'{x:.8f} {y:.8f}' for x, y in points) + ' Z'
        # These are genuine curve-fit proposals: the formerly accepted oval
        # becomes four anchors and individually retains its silhouette. Only
        # the complete composition closes the 0.1-unit white-on-white gap.
        cases = {
            'holes': ('<path id="fixed" fill="white" d="M10 10 H110 V110 H10 V89.9 H90 V30.1 H10 Z"/>',
                      'whole_alpha_holes_changed'),
            'components': ('<rect id="fixed" fill="white" x="10" y="10" width="100" height="20.1"/>',
                           'whole_alpha_components_changed'),
        }
        for name, (fixed, reason) in cases.items():
            with self.subTest(case=name):
                folder = self.root / name; folder.mkdir()
                svg = folder / 'before.svg'
                svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
                               + fixed + f'<path id="oval" fill="white" d="{path}"/></svg>')
                source = folder / 'source.png'
                render_svg_reference(svg, source, width=1200, background=None)
                work = folder / 'worker'; work.mkdir()
                result = prepare._invoke_worker({'work': str(work), 'shared_work': str(folder),
                    'current_svg': str(svg), 'original_svg': str(svg), 'source': str(source),
                    'member_id': 'oval', 'gradient': False, 'allow_source_primitive': False,
                    'error_budget_percent': .25}, prepare.PATH_SECONDS)
                self.assertEqual(result, {'status': 'skipped', 'reason': reason})
                candidate = work / 'candidate.svg'
                self.assertTrue(candidate.is_file())
                self.assertEqual(_geometry(prepare._by_id(ET.parse(candidate).getroot())['oval'])[0], 4)
                after = folder / 'candidate-reference.png'
                with Image.open(source) as image:
                    a = np.asarray(image.convert('RGBA'))
                with Image.open(after) as image:
                    b = np.asarray(image.convert('RGBA'))
                def white(rgba):
                    picture = Image.new('RGBA', (rgba.shape[1], rgba.shape[0]), 'white')
                    picture.alpha_composite(Image.fromarray(rgba))
                    return np.asarray(picture.convert('RGB'))
                self.assertTrue(np.array_equal(white(a), white(b)))
                topology = compare_alpha_topology(source, after)
                if name == 'holes':
                    self.assertFalse(topology['accepted'])
                    self.assertEqual(topology['thresholds'][-1]['created_regions'], 1)
                else:
                    self.assertTrue(topology['accepted']) # hole-only protection is insufficient
                    counts = []
                    for rgba in (a, b):
                        labels, count = connected_components(rgba[:, :, 3] >= 224)
                        counts.append(int((np.bincount(labels.ravel(), minlength=count + 1)[1:] >= 2).sum()))
                    self.assertEqual(counts, [2, 1])

    def test_composed_alpha_requires_rgba_and_limits_invisible_white_coverage_change(self):
        from PIL import Image, ImageDraw
        before = self.root / 'coverage-before.png'
        after = self.root / 'coverage-after.png'
        image = Image.new('RGBA', (120, 120), (255, 255, 255, 0))
        ImageDraw.Draw(image).rectangle((20, 20, 100, 100), fill=(255, 255, 255, 255))
        image.save(before)
        self.assertTrue(prepare._composed_alpha_gate(before, before)['identical_alpha'])
        ImageDraw.Draw(image).rectangle((20, 20, 30, 100), fill=(255, 255, 255, 0))
        image.save(after)
        with self.assertRaisesRegex(ValueError, 'alpha_coverage'):
            prepare._composed_alpha_gate(before, after)
        image.convert('RGB').save(after)
        with self.assertRaisesRegex(ValueError, 'requires_rgba'):
            prepare._composed_alpha_gate(before, after)

    def test_composed_alpha_one_pixel_unmatched_region_tolerance_is_explicit(self):
        from PIL import Image, ImageDraw
        before = self.root / 'region-before.png'
        after = self.root / 'region-after.png'
        image = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 490, 490), fill=(255, 255, 255, 255))
        draw.line((5, 100, 5, 101), fill=(255, 255, 255, 255))
        image.save(before)
        draw.line((5, 100, 5, 101), fill=(255, 255, 255, 0))
        draw.line((6, 100, 6, 101), fill=(255, 255, 255, 255))
        image.save(after)
        result = prepare._composed_alpha_gate(before, after)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['component_minimum_region_pixels'], 2)
        self.assertEqual(result['component_correspondence_tolerance_pixels'], 1)
        self.assertEqual([(row['components_before'], row['components_after'])
                          for row in result['component_thresholds']], [(2, 2)] * 3)


if __name__ == '__main__':
    unittest.main()
