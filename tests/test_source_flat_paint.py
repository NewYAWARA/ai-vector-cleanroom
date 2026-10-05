from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from source_flat_paint import propose_source_flat_paint, _rgba_render
from source_scene_guard import validate_source_scene_arrays


def svg(stops=None, *, height=40, second=False, claims=True):
    stops = stops or '<stop offset="0" stop-color="#16472d"/><stop offset="1" stop-color="#387850"/>'
    proof = ' data-avc-gradient-object="g1" data-avc-designer-anchors="4" data-avc-p95-error-percent="0.1"' if claims else ''
    paths = f'<path id="one" d="M10 10H130V{10 + height}H10Z" fill="url(#g)"{proof}/>'
    if second:
        paths += '<path id="two" d="M10 70H130V110H10Z" fill="url(#g)"/>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="160" height="120" viewBox="0 0 160 120"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient></defs>{paths}</svg>'


def render(text):
    return _rgba_render(ET.fromstring(text), 160, 120)


def flat_source(text, color="#24603c", only_id=None):
    root = ET.fromstring(text)
    for n in root.iter():
        if n.tag.endswith("}path") and (only_id is None or n.get("id") == only_id):
            n.set("fill", color)
    return _rgba_render(root, 160, 120)


class SourceFlatPaintTests(unittest.TestCase):
    def test_zero_radius_returns_identity_without_native_rank_filter(self):
        from source_flat_paint import _erode
        mask = np.array([[True, False], [False, True]])
        with patch('source_flat_paint.ImageFilter.MinFilter', side_effect=AssertionError('native filter called')):
            result = _erode(mask, 0)
            np.testing.assert_array_equal(result, mask)
            self.assertIsNot(result, mask)
            for invalid in (-1, True, 0.5):
                with self.assertRaises(ValueError):
                    _erode(mask, invalid)

    def test_rounded_nonsquare_source_can_propose_without_resizing_original(self):
        text = svg(height=70).replace('height="120"', 'height="119"').replace('0 0 160 120', '0 0 160 119')
        root = ET.fromstring(text)
        for node in root.iter():
            if node.get('id') == 'one':
                node.set('fill', '#24603c')
        original = _rgba_render(root, 201, 150)
        result = propose_source_flat_paint(text, original)
        self.assertEqual(result['status'], 'proposed', result)
        self.assertEqual(result['native_size'], [201, 150])
        self.assertEqual(result['proposals'][0]['replacement_fill'], '#24603c')

    def test_original_flat_source_proposes_paint_only_and_requires_scene_guard(self):
        text = svg()
        source = flat_source(text)
        result = propose_source_flat_paint(text, source)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(len(result["proposals"]), 1)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["replacement_fill"], "#24603c")
        self.assertEqual(proposal["status"], "proposed_pending_scene_guard")
        self.assertTrue(result["requires_whole_scene_source_guard"])
        self.assertEqual(proposal["evidence"]["source_core"]["exact_median_share"], 1)
        self.assertGreater(proposal["evidence"]["core_holdout"]["mean_improvement"], 0)
        self.assertGreater(proposal["evidence"]["edge_holdout"]["mean_improvement"], 0)
        self.assertTrue(validate_source_scene_arrays(render(text), render(proposal["svg_text"]), source,
                                                    roi_xyxy=proposal["roi_xyxy"])["accepted"])

    def test_geometry_stack_and_shared_resource_are_unchanged_and_old_claims_removed(self):
        text = svg(second=True)
        result = propose_source_flat_paint(text, flat_source(text, only_id="one"))
        self.assertEqual(len(result["proposals"]), 1)
        proposal = result["proposals"][0]
        before, after = ET.fromstring(text), ET.fromstring(proposal["svg_text"])
        bnodes = {n.get("id"): n for n in before.iter() if n.get("id")}
        anodes = {n.get("id"): n for n in after.iter() if n.get("id")}
        self.assertEqual(list(bnodes), list(anodes))
        self.assertEqual(bnodes["one"].get("d"), anodes["one"].get("d"))
        self.assertEqual(ET.tostring(bnodes["two"]), ET.tostring(anodes["two"]))
        self.assertEqual(ET.tostring(bnodes["g"]), ET.tostring(anodes["g"]))
        self.assertIsNone(anodes["one"].get("data-avc-gradient-object"))
        self.assertIsNone(anodes["one"].get("data-avc-p95-error-percent"))
        self.assertIsNotNone(anodes["one"].get("data-avc-source-flat-paint"))
        np.testing.assert_array_equal(render(text)[:, :, 3], render(proposal["svg_text"])[:, :, 3])

    def test_legitimate_subtle_gradient_is_retained(self):
        text = svg('<stop offset="0" stop-color="#446a54"/><stop offset="1" stop-color="#486e58"/>')
        result = propose_source_flat_paint(text, render(text))
        self.assertEqual(result["proposals"], [])
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["decisions"][0]["reason"], "native_core_has_colour_variation_preserve_gradient")

    def test_real_edge_gradient_with_flat_core_is_retained_by_separate_edge_holdout(self):
        text = svg('<stop offset="0" stop-color="#3b7853"/><stop offset="0.02" stop-color="#24603c"/>'
                   '<stop offset="0.98" stop-color="#24603c"/><stop offset="1" stop-color="#3b7853"/>')
        result = propose_source_flat_paint(text, render(text))
        self.assertEqual(result["proposals"], [])
        decision = result["decisions"][0]
        self.assertEqual(decision["source_core"]["exact_median_share"], 1)
        self.assertEqual(decision["reason"], "native_core_and_edge_do_not_both_favour_solid")
        self.assertLessEqual(decision["evidence"]["edge_holdout"]["mean_improvement"], 0)

    def test_narrow_shape_without_native_core_is_retained(self):
        text = svg(height=6)
        result = propose_source_flat_paint(text, flat_source(text))
        self.assertEqual(result["proposals"], [])
        self.assertEqual(result["decisions"][0]["reason"], "insufficient_native_core_or_edge_preserve_gradient")

    def test_thin_stable_core_can_equal_baseline_while_false_edge_gradient_improves(self):
        stops = ('<stop offset="0" stop-color="#002809"/>'
                 '<stop offset="0.2" stop-color="#0e472d"/>'
                 '<stop offset="0.8" stop-color="#0e472d"/>'
                 '<stop offset="1" stop-color="#002809"/>')
        text = svg(stops, height=10).replace('x2="1" y2="0"', 'x2="0" y2="1"')
        source = flat_source(text, '#0e472d')
        result = propose_source_flat_paint(text, source)
        self.assertEqual(result['status'], 'proposed', result)
        proposal = result['proposals'][0]
        evidence = proposal['evidence']
        self.assertGreaterEqual(evidence['source_core']['pixels'], 64)
        self.assertLess(evidence['source_core']['support_fraction_diagnostic_only'], .2)
        self.assertEqual(evidence['source_core']['exact_median_share'], 1)
        self.assertEqual(evidence['core_holdout']['mean_improvement'], 0)
        self.assertGreater(evidence['edge_holdout']['mean_improvement'], .05)
        self.assertTrue(validate_source_scene_arrays(render(text), render(proposal['svg_text']), source,
                                                    roi_xyxy=proposal['roi_xyxy'])['accepted'])

    def test_same_thin_constant_middle_with_real_edge_colour_is_not_flattened(self):
        stops = ('<stop offset="0" stop-color="#3b7853"/>'
                 '<stop offset="0.2" stop-color="#0e472d"/>'
                 '<stop offset="0.8" stop-color="#0e472d"/>'
                 '<stop offset="1" stop-color="#3b7853"/>')
        text = svg(stops, height=10).replace('x2="1" y2="0"', 'x2="0" y2="1"')
        result = propose_source_flat_paint(text, render(text))
        self.assertEqual(result['proposals'], [])
        decision = result['decisions'][0]
        self.assertEqual(decision['source_core']['exact_median_share'], 1)
        self.assertLess(decision['source_core']['support_fraction_diagnostic_only'], .2)
        self.assertEqual(decision['reason'], 'native_core_and_edge_do_not_both_favour_solid')
        self.assertLess(decision['evidence']['edge_holdout']['mean_improvement'], 0)

    def test_gradient_opacity_must_not_be_flattened_away(self):
        text = svg('<stop offset="0" stop-color="#16472d" stop-opacity="0.99"/>'
                   '<stop offset="1" stop-color="#387850"/>')
        result = propose_source_flat_paint(text, flat_source(text))
        self.assertEqual(result["proposals"], [])
        self.assertEqual(result["decisions"][0]["reason"], "solid_would_change_gradient_alpha_preserve_gradient")

    def test_candidate_budget_is_bounded_and_other_paths_are_retained(self):
        text = svg(second=True)
        result = propose_source_flat_paint(text, flat_source(text), max_candidates=1)
        self.assertEqual(len(result["proposals"]), 1)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["decisions"][-1]["reason"], "candidate_budget_exhausted")
        for budget in (0, 17, True):
            result = propose_source_flat_paint(text, flat_source(text), max_candidates=budget)
            self.assertEqual(result["status"], "unverified_fail_closed")

    def test_css_stroke_and_hidden_ownership_are_not_guessed(self):
        for text in (svg().replace('id="one"', 'id="one" style="fill:url(#g)"'),
                     svg().replace('id="one"', 'id="one" stroke="#111111"')):
            result = propose_source_flat_paint(text, flat_source(svg()))
            self.assertEqual(result["proposals"], [])
            self.assertEqual(result["decisions"][0]["reason"], "css_or_stroke_ownership_not_supported")
        hidden = svg().replace('</svg>', '<path id="cover" fill="#e1a131" d="M0 0H160V60H0Z"/></svg>')
        result = propose_source_flat_paint(hidden, render(hidden))
        self.assertEqual(result["proposals"], [])

    def test_inherited_gradient_owner_is_not_left_as_false_proof(self):
        text = svg().replace('<path id="one"', '<g data-avc-gradient-object="shared-owner"><path id="one"')
        text = text.replace('</svg>', '</g></svg>')
        result = propose_source_flat_paint(text, flat_source(text))
        self.assertEqual(result["proposals"], [])
        self.assertEqual(result["decisions"][0]["reason"], "inherited_gradient_owner_requires_manual_projection")

    def test_invalid_input_and_renderer_failure_fail_closed(self):
        text = svg()
        source = flat_source(text)
        for bad in (source.astype(float), source[:, :, :3], source[:, :-1]):
            result = propose_source_flat_paint(text, bad)
            self.assertEqual(result["status"], "unverified_fail_closed")
        with patch("source_flat_paint._rgba_render", side_effect=RuntimeError("renderer failed")):
            result = propose_source_flat_paint(text, source)
        self.assertEqual(result["status"], "unverified_fail_closed")
        self.assertEqual(result["proposals"], [])

    def test_e3_native_false_gradients_have_safe_flat_alternatives_when_fixture_available(self):
        workspace = Path(__file__).resolve().parents[4]
        base = workspace / "work/cr-1005101832/exp/e3_gaps/out_in_1254/result_gaps"
        if not (base / "gaps_vector.svg").exists():
            self.skipTest("External Claude E3 forensic fixture is not packaged")
        text = (base / "gaps_vector.svg").read_text(encoding="utf-8")
        source = np.asarray(Image.open(base / "source_original.png").convert("RGBA"))
        reference = np.asarray(Image.open(base / "source_reference.png").convert("RGBA"))
        result = propose_source_flat_paint(text, source)
        self.assertEqual(len(result["proposals"]), 4)
        before = _rgba_render(ET.fromstring(text), 1254, 418)
        for proposal in result["proposals"]:
            self.assertEqual(proposal["replacement_fill"], "#0e472d")
            self.assertEqual(proposal["evidence"]["source_core"]["rgb_standard_deviation"], [0., 0., 0.])
            after = _rgba_render(ET.fromstring(proposal["svg_text"]), 1254, 418)
            guard = validate_source_scene_arrays(before, after, source, processed_reference_rgba=reference,
                                                  roi_xyxy=proposal["roi_xyxy"])
            self.assertTrue(guard["accepted"], guard["reasons"])

    def test_e3_thin_false_edge_gradient_has_safe_flat_alternative_when_fixture_available(self):
        workspace = Path(__file__).resolve().parents[4]
        base = workspace / 'work/p3/verify-check2/gaps/results/result_gaps_1024'
        if not (base / 'gaps_1024_vector.svg').exists():
            self.skipTest('External E3 frozen-check2 forensic fixture is not packaged')
        text = (base / 'gaps_1024_vector.svg').read_text(encoding='utf8')
        source = np.asarray(Image.open(base / 'source_original.png').convert('RGBA'))
        reference = np.asarray(Image.open(base / 'source_reference.png').convert('RGBA'))
        result = propose_source_flat_paint(text, source)
        self.assertEqual(len(result['proposals']), 1, result)
        proposal = result['proposals'][0]
        self.assertEqual(proposal['replacement_fill'], '#0e472d')
        self.assertEqual(proposal['evidence']['source_core']['pixels'], 228)
        self.assertEqual(proposal['evidence']['core_holdout']['mean_improvement'], 0)
        before = _rgba_render(ET.fromstring(text), 1024, 341)
        after = _rgba_render(ET.fromstring(proposal['svg_text']), 1024, 341)
        guard = validate_source_scene_arrays(before, after, source, processed_reference_rgba=reference,
                                            roi_xyxy=proposal['roi_xyxy'])
        self.assertTrue(guard['accepted'], guard['reasons'])


if __name__ == "__main__":
    unittest.main()
