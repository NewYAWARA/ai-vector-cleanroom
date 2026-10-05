# -*- coding: utf-8 -*-
"""Source-space gradient proposals through the clean-base SVG assembler."""

from pathlib import Path
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import clean_base
from gradient_reconstruction_stage import encode_mask_rle


SENTINEL = (241, 3, 247)
SENTINEL_HEX = "#f103f7"


class _VirtualTemporaryDirectory:
    def __enter__(self):
        return "virtual-gradient-temp"

    def __exit__(self, *_args):
        return False


def _geometry(path, *, native_primitives=None, anchors=4):
    return {
        "path": path,
        "fill_rule": "evenodd",
        "native_primitives": list(native_primitives or []),
        "primitive_first": True,
        "primitive_complexity": {
            "anchor_count": anchors,
            "segment_count": anchors,
            "object_fragments": 1,
        },
        "topology": {
            "components": 1,
            "holes": 0,
            "topology_preserved": True,
        },
        "anchor_count": anchors,
        "designer_anchor_count": anchors,
        "segment_count": anchors,
        "anchors_before": 128,
        "error_budget": {
            "passed": True,
            "requested_max_percent": 0.5,
            "actual_p95_error_percent": 0.08,
            "actual_max_error_percent": 0.1,
        },
        "lexicographic_objective": [
            "hard:geometry_error_budget",
            "minimise:anchor_count",
        ],
    }


def _proposal(mask, model, stops, geometry, *, proposal_id="object-one"):
    return {
        "proposal_id": proposal_id,
        "candidate_id": "candidate-one",
        "candidate_family": "community",
        "component_ids": [1, 2, 3],
        "area": int(mask.sum()),
        "bbox_xyxy": [0, 0, int(mask.shape[1]), int(mask.shape[0])],
        "mask": encode_mask_rle(mask),
        "path": geometry["path"],
        "fill_rule": geometry["fill_rule"],
        "model": model,
        "stops": stops,
        "confidence": 0.91,
        "heldout_evidence": {
            "error": {
                "heldout": {"mean": 1.0, "p90": 2.0, "p99": 4.0},
                "solid_baseline": {"mean": 8.0, "p90": 12.0, "p99": 16.0},
            },
            "validation": {"passed": True},
            "reasons": ["heldout_validation_passed"],
        },
        "geometry": geometry,
        "selection": {"colour_used_for_geometry": False},
    }


def _stage(proposal):
    return {
        "schema": "ai-vector-cleanroom.gradient-reconstruction-stage/v1",
        "status": "proposed",
        "proposals": [proposal],
        "summary": {"candidates_generated": 1, "objects_selected": 1},
        "objective": {"pixel_boundary_is_guardrail_not_curve_target": True},
        "parameters": {"max_candidates": 48},
        "decisions": [],
    }


def _run_clean_base(stage, visible_mask, *, trace_fill=SENTINEL_HEX,
                    gradient_stage_cache=None, gradients="on",
                    proposer=None):
    """Run the real assembler with only filesystem/renderer edges virtualised."""
    visible_mask = np.asarray(visible_mask, dtype=bool)
    height, width = visible_mask.shape
    yy, xx = np.indices((height, width))
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    first = np.asarray((20, 40, 60), dtype=np.uint8)
    second = np.asarray((30, 50, 70), dtype=np.uint8)
    source_rgb = np.where(((xx + yy) & 1)[..., None], first, second)
    rgba[visible_mask, :3] = source_rgb[visible_mask]
    rgba[visible_mask, 3] = 255
    prepared = Image.fromarray(rgba, "RGBA")
    captured = {}
    raw_path = (
        f"M0 0 L{width - 1} 0 L{width - 1} {height - 1} "
        f"L0 {height - 1} Z"
    )

    def fake_paths(_raw):
        return iter(((raw_path, trace_fill, 0.0, 0.0),))

    def fake_write_text(path, text, *args, **kwargs):
        captured["svg"] = str(text)
        return len(str(text))

    if proposer is None:
        def proposer(*_args, **_kwargs):
            return stage

    with patch.object(
            clean_base, "_prepare_image",
            return_value=(prepared, (width, height), bool((~visible_mask).any()))), \
            patch("gradient_residual_provenance.prove_preexisting_residuals",
                  side_effect=ValueError("virtual test renderer unavailable")), \
            patch(
                "gradient_reconstruction_stage.propose_gradient_reconstruction",
                new=proposer), \
            patch.object(clean_base, "_allocate_gradient_keys",
                         side_effect=lambda _palette, count, **_kwargs:
                         [SENTINEL][:count]), \
            patch.object(clean_base.vtracer, "convert_image_to_svg_py",
                         return_value=None), \
            patch("trace_component_recovery.recover_missing_source_components",
                  side_effect=lambda raw, source, **kwargs: (raw, {
                      "status": "no_change", "recovered_count": 0})), \
            patch.object(clean_base, "_iter_svg_paths", side_effect=fake_paths), \
            patch.object(clean_base.tempfile, "TemporaryDirectory",
                         _VirtualTemporaryDirectory), \
            patch.object(Image.Image, "save", return_value=None), \
            patch.object(Path, "read_text", autospec=True,
                         side_effect=lambda path, *args, **kwargs: captured.get("svg", "<svg/>")
                         if path.name == "baseline.svg" else "<svg/>"), \
            patch.object(Path, "mkdir", return_value=None), \
            patch.object(Path, "write_text", new=fake_write_text):
        stats = clean_base.build_clean_base(
            Path("virtual-source.png"), Path("virtual-output.svg"),
            background="keep", geometry="off", strokes="off",
            gradients=gradients, gradient_stage_cache=gradient_stage_cache)
    return stats, captured["svg"]


def _elements(svg, local_name):
    root = ET.fromstring(svg)
    return [node for node in root.iter()
            if node.tag.rsplit("}", 1)[-1] == local_name]


class CleanBaseGradientIntegrationTests(unittest.TestCase):
    def test_pending_paint_assistance_is_not_counted_as_complete_geometry(self):
        stage = {'status': 'skipped', 'proposals': [], 'decisions': [],
                 'summary': {'objects_selected': 0}, 'paint_ready_alternatives': [{'candidate_id': 'pending'}]}
        detail = {'id': 'partial', 'candidate_id': 'pending', 'validation': {'geometry': {
            'source_paint_only': {'partial_selection': True, 'manual_review_required': True}}}}
        def assisted(svg, _alternatives, _source, _processed, **kwargs):
            return svg, [detail], [{'candidate_id': 'pending', 'status': 'partial_paint_selected'}], [{'partial_selection': True}]
        with patch('gradient_paint_only.apply_pending_paint_alternatives', side_effect=assisted) as applied:
            stats, _ = _run_clean_base(stage, np.ones((12, 12), bool))
        applied.assert_called_once()
        self.assertEqual(stats.n_gradients, 1)
        self.assertEqual(stats.gradient_info, [detail])
        summary = stats.palette_audit['gradient_reconstruction']['summary']
        self.assertEqual(summary['objects_selected'], 0)
        self.assertEqual(summary['partial_paint_fields'], 1)
        self.assertTrue(any('manual review required' in note for note in stats.geometry_notes))

    def test_fragmented_residual_withdraws_only_consuming_gradient(self):
        labels = np.zeros((20, 20), dtype=int)
        labels[:, :10] = 1
        owned_left = np.zeros_like(labels, dtype=bool)
        owned_left[:, :8] = True
        owned_right = np.zeros_like(owned_left)
        owned_right[:, 12:] = True
        many = " ".join(f"M9 {y}L10 {y}L10 {y+1}L9 {y+1}Z"
                        for y in range(0, 20, 2))
        record = clean_base._gradient_residual_fragmentation(
            [{"color": 1, "raw": many}], labels, np.ones_like(owned_left),
            [{"candidate_id": "left", "mask": owned_left},
             {"candidate_id": "right", "mask": owned_right}])
        self.assertEqual(record["candidate_ids"], ["left"])
        self.assertEqual(record["residual_fragments"], 10)
        self.assertFalse(record["human_time_saving_claimed"])
        self.assertIsNone(clean_base._gradient_residual_fragmentation(
            [{"color": 1, "raw": "M9 0L10 0L10 1L9 1Z"}], labels,
            np.ones_like(owned_left), [{"candidate_id": "left", "mask": owned_left}]))

    def test_fragmentation_retry_reuses_stage_and_reports_actual_selected_set(self):
        mask = np.ones((20, 24), dtype=bool)
        proposal = _proposal(mask,
            {"type": "linear", "x1": 0, "y1": 0, "x2": 23, "y2": 19},
            [{"offset": 0, "color": "#14283c", "rgb": [20, 40, 60]},
             {"offset": 1, "color": "#1e3246", "rgb": [30, 50, 70]}],
            _geometry("M0 0L23 0L23 19L0 19Z"))
        stage = _stage(proposal)
        stage["decisions"] = [{"candidate_id": "candidate-one", "status": "selected",
                               "reasons": ["passed"]}]
        calls = []
        def propose(*args, **kwargs):
            calls.append(1)
            return stage
        withdrawal = {
            "reason": "gradient_replacement_requires_fragmented_palette_residuals",
            "candidate_ids": ["candidate-one"], "residual_fragments": 40}
        with patch.object(clean_base, "_gradient_residual_fragmentation",
                          side_effect=[withdrawal, None]):
            stats, svg = _run_clean_base(stage, mask, trace_fill="#192d41", proposer=propose)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stats.n_gradients, 0)
        self.assertGreater(stats.n_paths, 0)
        audit = stats.palette_audit["gradient_reconstruction"]
        self.assertEqual(audit["summary"]["objects_selected"], 0)
        self.assertEqual(audit["decisions"][0]["status"], "withdrawn")
        self.assertEqual(audit["fragmentation_withdrawals"], [withdrawal])
        self.assertEqual(stage["decisions"][0]["status"], "selected")
        self.assertEqual(len(stage["proposals"]), 1)
        self.assertNotIn("linearGradient", svg)

    def test_light_marker_recovery_cannot_resurrect_unproven_edge_crumbs(self):
        self.assertTrue(clean_base._proven_light_compartment_exclusions((20, 30), {}).all())
        excluded = clean_base._proven_light_compartment_exclusions((20, 30), {"records": [
            {"kind": "foreground_component", "bbox_xyxy": [1, 1, 3, 3]},
            {"kind": "enclosed_background_compartment", "bbox_xyxy": [10, 5, 14, 8]}]})
        self.assertEqual(np.count_nonzero(~excluded), 12)
        self.assertTrue(excluded[1:3, 1:3].all())
        self.assertFalse(excluded[5:8, 10:14].any())

    def test_gradient_residual_recovery_emits_missing_ink_without_claiming_owned_pixels(self):
        labels = np.zeros((12, 12), dtype=int)
        labels[1:11, 1:11] = 1
        labels[0, 0] = 2
        visible = np.ones_like(labels, dtype=bool)
        owned = np.zeros_like(visible)
        owned[1:11, 1:11] = True
        # Palette 0 is already represented; palette 1 belongs to the gradient.
        entries = [{"color": 0, "raw": "M0 0L12 0L12 12Z"}]
        recovered, count = clean_base._recover_unowned_gradient_residuals(
            entries, labels, visible, [{"mask": owned}], 3)
        self.assertEqual(count, 1)
        self.assertEqual([entry["color"] for entry in recovered], [2])
        self.assertTrue(recovered[0]["raw"])
        subpaths = clean_base._parse_subpaths(recovered[0]["raw"])
        self.assertEqual(len(subpaths), 1)
        self.assertTrue(subpaths[0]["closed"])
        self.assertEqual(clean_base._recover_unowned_gradient_residuals(
            entries, labels, visible, [], 3), ([], 0))

    def test_unrelated_missing_palette_noise_cannot_withdraw_every_gradient(self):
        labels = np.zeros((32, 32), dtype=int)
        labels[1::3, 1::3] = 2
        owned = np.zeros_like(labels, dtype=bool)
        owned[8:24, 8:24] = True
        labels[owned] = 1
        recovered, pixels = clean_base._recover_unowned_gradient_residuals(
            [{"color": 0}], labels, np.ones_like(owned),
            [{"candidate_id": "independent", "mask": owned}], 3)
        self.assertEqual(recovered, [])
        self.assertEqual(pixels, 0)
        self.assertIsNone(clean_base._gradient_residual_fragmentation(
            [], labels, np.ones_like(owned), [{"candidate_id": "independent", "mask": owned}]))

    def test_lone_native_hole_does_not_replace_whole_gradient_object(self):
        mask = np.ones((40, 40), dtype=bool)
        mask[18:23, 18:23] = False
        hole = "M18 20 A2 2 0 1 0 22 20 A2 2 0 1 0 18 20 Z"
        outer = "M0 0 C10 1 30 1 39 0 L39 39 L0 39 Z"
        for path, native in [
            (outer + " " + hole, [{"element": "circle", "cx": 20, "cy": 20, "r": 2}]),
            ("M0 20 A20 20 0 1 0 40 20 A20 20 0 1 0 0 20 Z " + hole,
             [{"element": "circle", "cx": 20, "cy": 20, "r": 20},
              {"element": "circle", "cx": 20, "cy": 20, "r": 2}]),
        ]:
            with self.subTest(native_count=len(native)):
                geometry = _geometry(path, native_primitives=native, anchors=6)
                geometry["topology"].update(holes=1, expected_loops=2, actual_loops=2)
                proposal = _proposal(mask, {
                    "type": "linear", "x1": 0, "y1": 0, "x2": 39, "y2": 39},
                    [{"offset": 0, "color": "#204020", "rgb": [32, 64, 32]},
                     {"offset": 1, "color": "#80a040", "rgb": [128, 160, 64]}], geometry)
                stats, svg = _run_clean_base(_stage(proposal), mask)
                self.assertEqual(stats.n_native, 0)
                self.assertEqual(len(_elements(svg, "circle")), 0)
                self.assertEqual(_elements(svg, "path")[0].get("d"), path)
                self.assertIsNone(stats.gradient_info[0]["validation"]["geometry"]["native_whole_object_path"])

    def test_five_stop_linear_emits_one_low_anchor_path_and_real_palette(self):
        mask = np.ones((24, 32), dtype=bool)
        path = "M0 0 L31 0 L31 23 L0 23 Z"
        model = {
            "type": "linear",
            "svg_type": "linearGradient",
            "gradient_units": "userSpaceOnUse",
            "x1": 0.0,
            "y1": 0.0,
            "x2": 31.0,
            "y2": 23.0,
            "stop_count": 5,
            "bounded_maximum_stops": 5,
        }
        stops = [
            {"offset": 0.0, "color": "#102030", "rgb": [16, 32, 48]},
            {"offset": 0.2, "color": "#204860", "rgb": [32, 72, 96]},
            {"offset": 0.5, "color": "#3c7880", "rgb": [60, 120, 128]},
            {"offset": 0.8, "color": "#80a060", "rgb": [128, 160, 96]},
            {"offset": 1.0, "color": "#d0b040", "rgb": [208, 176, 64]},
        ]
        geometry = _geometry(path, anchors=4)
        geometry["selection_evidence"] = {
            "selected_candidate_id": "curve_refit_04",
            "identity_loop_count": 0,
        }
        proposal = _proposal(mask, model, stops, geometry)

        stats, svg = _run_clean_base(_stage(proposal), mask)

        self.assertEqual(stats.n_gradients, 1)
        self.assertEqual(stats.palette_audit["trace_component_recovery"]["status"], "no_change")
        self.assertEqual(stats.n_paths, 1)
        self.assertEqual(stats.n_native, 0)
        self.assertEqual(stats.n_nodes, 4)
        self.assertEqual(len(_elements(svg, "linearGradient")), 1)
        self.assertEqual(len(_elements(svg, "radialGradient")), 0)
        self.assertEqual(len(_elements(svg, "stop")), 5)
        paths = _elements(svg, "path")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].get("d"), path)
        self.assertEqual(paths[0].get("id"), "avc-gradient-drawable-grad1")
        self.assertEqual(paths[0].get("data-avc-designer-anchors"), "4")
        self.assertEqual(paths[0].get("data-avc-error-budget-percent"), "0.5")
        self.assertEqual(paths[0].get("data-avc-p95-error-percent"), "0.08")
        self.assertEqual(paths[0].get("data-avc-max-error-percent"), "0.1")
        self.assertNotIn(SENTINEL_HEX, svg.lower())
        self.assertEqual(stats.palette, [("gradient", "#3c7880")])
        self.assertNotEqual(stats.palette[0][1], stats.gradient_info[0]["key"])

        info = stats.gradient_info[0]
        self.assertEqual(info["type"], "linear")
        self.assertEqual(info["model"]["stop_count"], 5)
        self.assertEqual(len(info["stops"]), 5)
        self.assertEqual(info["stops"][2]["color"], "#3c7880")
        self.assertEqual(info["validation"]["engine"],
                         "source_space_heldout_gradient_object")
        self.assertEqual(
            info["validation"]["geometry"]["selection_evidence"],
            geometry["selection_evidence"])

    def test_radial_gradient_emits_native_circle_and_model_stats(self):
        yy, xx = np.indices((41, 41))
        mask = (xx - 20) ** 2 + (yy - 20) ** 2 <= 16 ** 2
        path = "M4 20 A16 16 0 1 0 36 20 A16 16 0 1 0 4 20 Z"
        model = {
            "type": "radial",
            "svg_type": "radialGradient",
            "gradient_units": "userSpaceOnUse",
            "center": [20.0, 20.0],
            "radius_x": 16.0,
            "radius_y": 16.0,
            "rotation_degrees": 0.0,
            "stop_count": 3,
            "bounded_maximum_stops": 5,
        }
        stops = [
            {"offset": 0.0, "color": "#fff080", "rgb": [255, 240, 128]},
            {"offset": 0.5, "color": "#f0b020", "rgb": [240, 176, 32]},
            {"offset": 1.0, "color": "#a05008", "rgb": [160, 80, 8]},
        ]
        native = [{"element": "circle", "cx": 20.0, "cy": 20.0,
                   "r": 16.0}]
        geometry = _geometry(path, native_primitives=native, anchors=1)
        proposal = _proposal(mask, model, stops, geometry, proposal_id="sun")

        stats, svg = _run_clean_base(_stage(proposal), mask)

        self.assertEqual(stats.n_gradients, 1)
        self.assertEqual(stats.n_paths, 0)
        self.assertEqual(stats.n_native, 1)
        self.assertEqual(stats.n_nodes, 1)
        self.assertEqual(len(_elements(svg, "radialGradient")), 1)
        self.assertEqual(len(_elements(svg, "linearGradient")), 0)
        self.assertEqual(len(_elements(svg, "circle")), 1)
        self.assertEqual(len(_elements(svg, "path")), 0)
        radial = _elements(svg, "radialGradient")[0]
        self.assertIn("scale(16 16)", radial.get("gradientTransform"))
        circle = _elements(svg, "circle")[0]
        self.assertEqual(circle.get("id"), "avc-gradient-drawable-grad1")
        self.assertEqual(circle.get("data-avc-gradient-object"), "sun")
        self.assertEqual(circle.get("data-avc-designer-anchors"), "4")
        self.assertEqual(circle.get("data-avc-error-budget-percent"), "0.5")
        self.assertEqual(circle.get("data-avc-p95-error-percent"), "0.08")
        self.assertEqual(circle.get("data-avc-max-error-percent"), "0.1")
        self.assertNotIn(SENTINEL_HEX, svg.lower())
        self.assertEqual(stats.palette, [("gradient", "#f0b020")])
        self.assertEqual(stats.gradient_info[0]["type"], "radial")
        self.assertEqual(stats.gradient_info[0]["model"]["radius_x"], 16.0)
        self.assertEqual(len(stats.gradient_info[0]["stops"]), 3)

    def test_malformed_proposal_fails_closed_to_solid_trace(self):
        mask = np.ones((20, 24), dtype=bool)
        path = "M0 0 L23 0 L23 19 L0 19 Z"
        malformed = _proposal(
            mask,
            {"type": "mesh", "stop_count": 3},
            [
                {"offset": 0.0, "color": "#102030", "rgb": [16, 32, 48]},
                {"offset": 0.5, "color": "#405060", "rgb": [64, 80, 96]},
                {"offset": 1.0, "color": "#8090a0", "rgb": [128, 144, 160]},
            ],
            _geometry(path, anchors=4),
        )

        stats, svg = _run_clean_base(
            _stage(malformed), mask, trace_fill="#192d41")

        self.assertEqual(stats.n_gradients, 0)
        self.assertEqual(stats.n_paths, 1)
        self.assertEqual(len(_elements(svg, "linearGradient")), 0)
        self.assertEqual(len(_elements(svg, "radialGradient")), 0)
        self.assertNotIn(SENTINEL_HEX, svg.lower())
        audit = stats.palette_audit["gradient_reconstruction"]
        self.assertEqual(audit["status"], "error")
        self.assertIn("supported native model", audit["error"])
        self.assertTrue(any("failed closed" in note
                            for note in stats.geometry_notes))


if __name__ == "__main__":
    unittest.main()
