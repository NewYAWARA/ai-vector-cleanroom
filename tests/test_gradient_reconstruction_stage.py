# -*- coding: utf-8 -*-
import json
import os
import threading
import unittest
from unittest.mock import patch

import numpy as np

from gradient_reconstruction_stage import (
    _fit_geometry,
    _mask_topology,
    _optimise_mask_geometry,
    _paint_models_stable,
    decode_mask_rle,
    propose_gradient_reconstruction,
)


def _fast_geometry(mask, **_kwargs):
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    path = f"M{x0} {y0} L{x1} {y0} L{x1} {y1} L{x0} {y1} Z"
    return {
        "path": path,
        "fill_rule": "evenodd",
        "topology": {"components": 1, "holes": 0,
                     "topology_preserved": True},
        "anchors_before": int(2 * ((x1 - x0) + (y1 - y0))),
        "anchors_after": 4,
        "segment_count_after": 4,
        "actual_max_error_percent": 0.0,
    }


def _fake_paint(_rgb, _mask, **_kwargs):
    return {
        "status": "proposed",
        "model": {"type": "linear", "svg_type": "linearGradient",
                  "x1": 0, "y1": 0, "x2": 1, "y2": 0,
                  "direction": [1.0, 0.0], "stop_count": 2},
        "stops": [
            {"offset": 0.0, "color": "#102030", "rgb": [16, 32, 48]},
            {"offset": 1.0, "color": "#8090a0", "rgb": [128, 144, 160]},
        ],
        "confidence": 0.9,
        "error": {
            "heldout": {"mean": 2.0, "p90": 4.0, "p99": 6.0},
            "solid_baseline": {"mean": 10.0, "p90": 14.0, "p99": 18.0},
            "heldout_mean_improvement": 8.0,
        },
        "validation": {"passed": True},
        "reasons": ["heldout_validation_passed"],
    }


class GradientReconstructionStageTests(unittest.TestCase):
    def test_failed_geometry_preserves_only_uncommitted_processed_paint_option(self):
        shape = (32, 44)
        mask = np.zeros(shape, dtype=bool)
        mask[5:27, 7:36] = True
        candidate = {'candidate_id': 'paint-without-geometry', 'kind': 'pair',
                     'mask': mask, 'component_ids': [1], 'score': 1.0}
        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8), np.zeros(shape, dtype=np.int32),
            np.ones(shape, dtype=bool), np.asarray([[10, 20, 30]], dtype=np.uint8),
            candidate_provider=lambda *_a, **_k: [candidate], model_fitter=_fake_paint,
            geometry_optimizer=lambda *_a, **_k: {'path': ''}, max_objects=1)
        self.assertEqual(result['proposals'], [])
        self.assertEqual(result['summary']['objects_selected'], 0)
        pending = result['paint_ready_alternatives']
        self.assertEqual(len(pending), 1)
        self.assertFalse(pending[0]['geometry_certified'])
        self.assertFalse(pending[0]['original_source_verified'])
        self.assertEqual(pending[0]['heldout_evidence']['scope'], 'processed_reference_only')
        self.assertEqual(pending[0]['status'], 'pending_native_source_and_existing_path_validation')
        self.assertNotIn('path', pending[0])
        self.assertNotIn('economy', pending[0])
        np.testing.assert_array_equal(decode_mask_rle(pending[0]['mask']), mask)
        json.dumps(result)

    def test_shared_geometry_fit_cache_reuses_exact_mask_across_stage_calls(self):
        shape = (32, 44)
        mask = np.zeros(shape, dtype=bool)
        mask[5:27, 7:36] = True

        def candidates(*_args, **_kwargs):
            return [{
                "candidate_id": "shared-mask",
                "kind": "pair",
                "mask": mask.copy(),
                "component_ids": [1],
                "score": 1.0,
            }]

        calls = []

        def counted_default(candidate_mask, bbox, **kwargs):
            calls.append(int(np.asarray(candidate_mask, dtype=bool).sum()))
            options = dict(kwargs)
            options["geometry_optimizer"] = _fast_geometry
            return _fit_geometry(candidate_mask, bbox, **options)

        args = (
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32),
            np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
        )
        shared_cache = {}
        kwargs = dict(
            candidate_provider=candidates,
            model_fitter=_fake_paint,
            max_candidates=4,
            max_geometry_candidates=4,
            max_objects=1,
            geometry_fit_cache=shared_cache,
        )
        with patch(
                "gradient_reconstruction_stage._fit_geometry",
                new=counted_default), patch.dict(os.environ, {
                    "AVC_GRADIENT_PROCESS_POOL_SAFE": "0",
                }):
            first = propose_gradient_reconstruction(*args, **kwargs)
            second = propose_gradient_reconstruction(*args, **kwargs)
            second["proposals"][0]["geometry"]["path"] = "tampered"
            third = propose_gradient_reconstruction(*args, **kwargs)

        self.assertEqual(calls, [int(mask.sum())])
        self.assertEqual(first["proposals"], third["proposals"])
        self.assertNotEqual(
            second["proposals"][0]["geometry"]["path"],
            third["proposals"][0]["geometry"]["path"],
        )
        self.assertEqual(
            first["summary"]["geometry_optimizer_calls"], 1)
        self.assertEqual(
            third["summary"]["geometry_optimizer_calls"], 0)
        self.assertEqual(
            third["summary"]["shared_geometry_fit_cache_hits"], 1)
        self.assertEqual(shared_cache["audit"]["requests"], 3)
        self.assertEqual(shared_cache["audit"]["hits"], 2)
        self.assertEqual(shared_cache["audit"]["misses"], 1)
        self.assertEqual(shared_cache["audit"]["stores"], 1)

    def test_default_geometry_parallelism_matches_serial_exactly(self):
        shape = (48, 72)
        masks = []
        for index in range(4):
            mask = np.zeros(shape, dtype=bool)
            x0 = 3 + index * 16
            mask[7:34, x0:x0 + 12] = True
            masks.append(mask)

        def candidates(*_args, **_kwargs):
            return [
                {"candidate_id": f"geometry-{index}", "kind": "pair",
                 "mask": mask, "component_ids": [index + 1],
                 "score": 1.0 - index / 10.0}
                for index, mask in enumerate(masks)
            ]

        args = (
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32),
            np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
        )
        kwargs = dict(
            candidate_provider=candidates,
            model_fitter=_fake_paint,
            max_candidates=8,
            max_geometry_candidates=8,
            max_objects=4,
        )
        with patch.dict(
                os.environ, {
                    "AVC_GRADIENT_GEOMETRY_WORKERS": "1",
                    "AVC_GRADIENT_PROCESS_POOL_SAFE": "1",
                    "AVC_GRADIENT_PROCESS_POOL_OWNER_PID": "-1",
                }):
            serial = propose_gradient_reconstruction(*args, **kwargs)
        with patch.dict(
                os.environ, {
                    "AVC_GRADIENT_GEOMETRY_WORKERS": "4",
                    "AVC_GRADIENT_PROCESS_POOL_SAFE": "1",
                    "AVC_GRADIENT_PROCESS_POOL_OWNER_PID": str(os.getpid()),
                }):
            parallel = propose_gradient_reconstruction(*args, **kwargs)
        self.assertEqual(
            json.dumps(serial, ensure_ascii=False, sort_keys=True),
            json.dumps(parallel, ensure_ascii=False, sort_keys=True))

    def test_default_paint_cache_and_parallel_fit_match_serial_exactly(self):
        shape = (24, 36)
        first = np.zeros(shape, dtype=bool)
        first[3:18, 3:17] = True
        second = np.zeros(shape, dtype=bool)
        second[5:20, 22:33] = True

        def candidates(*_args, **_kwargs):
            return [
                {"candidate_id": "a", "kind": "pair", "mask": first,
                 "component_ids": [1]},
                {"candidate_id": "b-duplicate", "kind": "pair",
                 "mask": first.copy(), "component_ids": [1]},
                {"candidate_id": "c", "kind": "pair", "mask": second,
                 "component_ids": [2]},
            ]

        calls = []
        call_lock = threading.Lock()

        def counting_paint(rgb, mask, **kwargs):
            with call_lock:
                calls.append((int(mask.sum()), tuple(sorted(kwargs))))
            return _fake_paint(rgb, mask, **kwargs)

        args = (
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32),
            np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
        )
        kwargs = {
            "candidate_provider": candidates,
            "geometry_optimizer": _fast_geometry,
            "max_candidates": 8,
            "max_geometry_candidates": 8,
            "max_objects": 3,
        }
        with patch(
                "gradient_reconstruction_stage.fit_gradient_object_proposal",
                new=counting_paint), patch.dict(
                    os.environ, {"AVC_GRADIENT_WORKERS": "1"}):
            serial = propose_gradient_reconstruction(
                *args, model_fitter=counting_paint, **kwargs)
        serial_calls = list(calls)
        calls.clear()
        with patch(
                "gradient_reconstruction_stage.fit_gradient_object_proposal",
                new=counting_paint), patch.dict(
                    os.environ, {"AVC_GRADIENT_WORKERS": "4"}):
            parallel = propose_gradient_reconstruction(
                *args, model_fitter=counting_paint, **kwargs)

        self.assertEqual(len(serial_calls), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            json.dumps(serial, sort_keys=True),
            json.dumps(parallel, sort_keys=True),
        )

    def test_bundled_geometry_identity_rollback_is_not_deliverable(self):
        mask = np.zeros((12, 16), dtype=bool)
        mask[2:10, 3:13] = True
        identity = {
            "status": "identity_rollback_no_safe_reduction",
            "identity_rollback_selected": True,
            "path": "M3 2 L13 2 L13 10 L3 10 Z",
            "fill_rule": "evenodd",
            "topology": {"topology_preserved": True},
            "anchor_count": 36,
            "segment_count": 36,
            "anchors_before": 36,
            "actual_max_error_percent": 0.0,
        }
        with (
            patch("gradient_reconstruction_stage.optimize_compound_contours",
                  new=object()),
            patch("gradient_reconstruction_stage._optimise_mask_geometry",
                  return_value=dict(identity)),
        ):
            geometry, reasons = _fit_geometry(
                mask, (3, 2, 13, 10), error_budget_percent=0.35,
                smooth=0.55, max_segments=512, geometry_optimizer=None)
        self.assertIsNone(geometry)
        self.assertIn(
            "geometry_identity_rollback_no_safe_anchor_reduction", reasons)

        # An injected legacy mock owns its own contract and is deliberately
        # unaffected by the bundled-only rollback interpretation.
        def injected(_mask, **_kwargs):
            return dict(identity)

        geometry, reasons = _fit_geometry(
            mask, (3, 2, 13, 10), error_budget_percent=0.35,
            smooth=0.55, max_segments=512, geometry_optimizer=injected)
        self.assertIsNotNone(geometry, reasons)
        self.assertEqual(reasons, [])

    def test_dominant_mixed_identity_outer_loop_fails_closed(self):
        mask = np.zeros((20, 24), dtype=bool)
        mask[2:18, 3:21] = True
        raw = {
            "status": "selected_within_geometry_budget",
            "identity_rollback_selected": False,
            "selected_candidate_id": "curve_refit_mixed_loops",
            "path": "M3 2 L21 2 L21 18 L3 18 Z",
            "fill_rule": "evenodd",
            "topology": {"topology_preserved": True},
            "anchor_count": 104,
            "designer_anchor_count": 104,
            "segment_count": 104,
            "anchors_before": 110,
            "actual_max_error_percent": 0.1,
            "actual_p95_error_percent": 0.05,
            "over_budget_share": 0.0,
            "salient_corner_max_percent": 0.1,
            "error_contract": {"primary": "p95"},
            "candidates": [{
                "candidate_id": "curve_refit_mixed_loops",
                "eligible": True,
                "per_loop_selection": [
                    {"loop_index": 0, "source_point_count": 100,
                     "identity_rollback": True},
                    {"loop_index": 1, "source_point_count": 10,
                     "identity_rollback": False},
                ],
            }],
        }
        with (
            patch("gradient_reconstruction_stage.optimize_compound_contours",
                  new=object()),
            patch("gradient_reconstruction_stage._optimise_mask_geometry",
                  return_value=raw),
        ):
            geometry, reasons = _fit_geometry(
                mask, (3, 2, 21, 18), error_budget_percent=0.25,
                smooth=0.55, max_segments=512, geometry_optimizer=None)
        self.assertIsNone(geometry)
        self.assertIn("geometry_dominant_source_polyline_rollback", reasons)

    def test_partial_native_compound_is_not_claimed_as_primitive_first(self):
        mask = np.zeros((20, 24), dtype=bool)
        mask[2:18, 3:21] = True
        selection = {"identity_loop_count": 0,
                     "selected_candidate_id": "curve_refit_04"}
        raw = {
            "status": "selected_within_geometry_budget",
            "identity_rollback_selected": False,
            "selected_candidate_id": "curve_refit_04",
            "path": "M3 2 C8 1 16 1 21 2 L21 18 L3 18 Z",
            "fill_rule": "evenodd",
            "topology": {"topology_preserved": True},
            "fit": {"contours": [
                {"native_primitive": {"element": "circle", "cx": 8,
                                      "cy": 8, "r": 1}},
                {"segments": [{"type": "cubic"}]},
            ]},
            "primitive_complexity": {"rank": 3,
                                     "category": "bezier_geometry"},
            "anchor_count": 12,
            "designer_anchor_count": 14,
            "segment_count": 12,
            "anchors_before": 80,
            "actual_max_error_percent": 0.1,
            "actual_p95_error_percent": 0.05,
            "over_budget_share": 0.0,
            "salient_corner_max_percent": 0.1,
            "error_contract": {"primary": "p95"},
            "selection_evidence": selection,
            "candidates": [{"candidate_id": "curve_refit_04",
                            "eligible": True}],
        }
        with (
            patch("gradient_reconstruction_stage.optimize_compound_contours",
                  new=object()),
            patch("gradient_reconstruction_stage._optimise_mask_geometry",
                  return_value=raw),
        ):
            geometry, reasons = _fit_geometry(
                mask, (3, 2, 21, 18), error_budget_percent=0.25,
                smooth=0.55, max_segments=512, geometry_optimizer=None)
        self.assertEqual(reasons, [])
        self.assertIsNotNone(geometry)
        self.assertFalse(geometry["primitive_first"])
        self.assertEqual(len(geometry["native_primitives"]), 1)
        self.assertEqual(geometry["selection_evidence"], selection)

    def test_independent_model_stability_fails_closed_on_invalid_evidence(self):
        first = _fake_paint(None, None)
        second = _fake_paint(None, None)
        stable, evidence = _paint_models_stable(first, second)
        self.assertTrue(stable, evidence)

        for invalid_direction in ([], [0.0, 0.0], [float("nan"), 0.0],
                                  ["not-a-number", 0.0]):
            bad = _fake_paint(None, None)
            bad["model"]["direction"] = invalid_direction
            stable, evidence = _paint_models_stable(first, bad)
            self.assertFalse(stable)
            self.assertIn("linear_direction_missing_or_invalid",
                          evidence["reasons"])

        bad_type = _fake_paint(None, None)
        bad_type["model"] = {"type": "mesh", "stop_count": 2}
        stable, evidence = _paint_models_stable(bad_type, bad_type)
        self.assertFalse(stable)
        self.assertIn("discovery_model_type_missing_or_invalid",
                      evidence["reasons"])

        bad_stops = _fake_paint(None, None)
        bad_stops["model"]["stop_count"] = True
        stable, evidence = _paint_models_stable(first, bad_stops)
        self.assertFalse(stable)
        self.assertIn("stop_count_missing_or_invalid", evidence["reasons"])

    def test_default_geometry_scans_local_bbox_and_restores_coordinates(self):
        mask = np.zeros((120, 160), dtype=bool)
        mask[70:92, 110:142] = True
        result = _optimise_mask_geometry(
            mask, error_budget_percent=1.0, smooth=0.55,
            max_segments=512)
        crop = result["topology"]["contour_crop"]
        self.assertLess(crop["cells_scanned"], mask.size // 5)
        source_bbox = result["normalization"]["source_bbox"]
        self.assertGreaterEqual(source_bbox[0], 100)
        self.assertGreaterEqual(source_bbox[1], 60)
        self.assertTrue(result["topology"]["topology_preserved"])
        extraction = result["topology"]["extraction_evidence"]
        self.assertEqual(extraction["requested_smooth"], 0.55)
        self.assertEqual(extraction["expected_loop_count"], 1)
        self.assertEqual(extraction["unsmoothed_loop_count"], 1)
        self.assertEqual(len(extraction["unsmoothed_loops"]), 1)
        self.assertIn("smoothed_loop_count_before_fallback", extraction)
        self.assertEqual(result["selection_evidence"][
            "selected_candidate_id"], result["selected_candidate_id"])

    def test_smoothing_fallback_keeps_pre_fallback_loop_evidence(self):
        mask = np.zeros((12, 14), dtype=bool)
        mask[2:10, 2:12] = True
        mask[5, 6] = False
        outer = [(0, 0), (10, 0), (10, 8), (0, 8)]
        hole = [(4, 3), (4, 4), (5, 4), (5, 3)]
        optimiser_result = {
            "selected_candidate_id": "curve_refit_mixed_loops",
            "selected_source": "curve_refit_per_loop_mixed_tolerance",
            "identity_rollback_selected": False,
            "fit": {"mixed_loop_tolerances": True},
            "loop_count": 2,
            "anchors_before": 8,
            "designer_anchor_count": 8,
            "candidates": [{
                "candidate_id": "curve_refit_mixed_loops",
                "topology": {"preserved": True},
                "per_loop_selection": [],
            }],
        }

        def extracted(_mask, *, smooth, **_kwargs):
            return [outer] if smooth > 0 else [outer, hole]

        with (
            patch("trace_engine._mask_to_smooth_loops",
                  side_effect=extracted),
            patch("gradient_reconstruction_stage.optimize_compound_contours",
                  return_value=optimiser_result),
        ):
            result = _optimise_mask_geometry(
                mask, error_budget_percent=0.25, smooth=0.55,
                max_segments=512)
        extraction = result["topology"]["extraction_evidence"]
        self.assertTrue(extraction["fallback_used"])
        self.assertEqual(extraction["expected_loop_count"], 2)
        self.assertEqual(extraction["smoothed_loop_count_before_fallback"], 1)
        self.assertEqual(extraction["unsmoothed_loop_count"], 2)
        self.assertEqual(len(extraction["smoothed_loops"]), 1)
        self.assertEqual(len(extraction["unsmoothed_loops"]), 2)

    def test_diagonal_foreground_uses_contour_compatible_topology(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2, 2] = True
        mask[3, 3] = True
        self.assertEqual(_mask_topology(mask), (2, 0))

        # A diagonal background contact remains connected and therefore is not
        # fabricated into a vector hole by the complementary 4/8 convention.
        ring = np.ones((5, 5), dtype=bool)
        ring[0, 0] = False
        ring[1, 1] = False
        ring[2, 2] = False
        self.assertEqual(_mask_topology(ring), (1, 0))

    def _two_ramps(self):
        height, width = 48, 96
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        labels = np.zeros((height, width), dtype=np.int32)
        visible = np.zeros((height, width), dtype=bool)
        palette = []
        specifications = [
            (4, 42, (30, 80, 45), (130, 180, 105)),
            (54, 92, (25, 70, 150), (100, 170, 225)),
        ]
        for object_index, (x0, x1, start, end) in enumerate(specifications):
            start = np.asarray(start, dtype=float)
            end = np.asarray(end, dtype=float)
            visible[8:40, x0:x1] = True
            for x in range(x0, x1):
                t = (x - x0) / float(x1 - x0 - 1)
                rgb[8:40, x] = np.rint(start * (1 - t) + end * t)
                labels[8:40, x] = object_index * 5 + min(4, int(t * 5))
            for band in range(5):
                t = (band + 0.5) / 5.0
                palette.append(start * (1 - t) + end * t)
        return rgb, labels, visible, np.asarray(palette)

    def test_two_adjacent_ramps_become_two_disjoint_objects(self):
        rgb, labels, visible, palette = self._two_ramps()
        result = propose_gradient_reconstruction(
            rgb, labels, visible, palette,
            geometry_error_percent=0.8,
            max_candidates=30,
            max_objects=4,
            geometry_optimizer=_fast_geometry,
            candidate_options={
                "min_component_area": 20,
                "min_candidate_area": 80,
                "min_chromatic_area": 100,
                "min_shared_boundary": 6,
                "smooth_delta": 20,
            },
            model_options={
                "min_pixels": 64,
                "maximum_mean_error": 8,
                "maximum_p90_error": 12,
                "maximum_p99_error": 18,
            },
        )
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["summary"]["objects_selected"], 2)
        self.assertEqual(result["summary"]["overlap_pixels_between_selected"], 0)
        masks = [decode_mask_rle(item["mask"]) for item in result["proposals"]]
        self.assertFalse(np.logical_and(masks[0], masks[1]).any())
        self.assertTrue(all(item["model"]["type"] == "linear"
                            for item in result["proposals"]))
        self.assertTrue(all(item["selection"]["colour_used_for_geometry"] is False
                            for item in result["proposals"]))

    def test_overlap_chooses_broad_ownership_and_rejects_pair_overlay(self):
        shape = (32, 48)
        broad = np.zeros(shape, dtype=bool)
        broad[4:24, 4:32] = True
        pair = np.zeros(shape, dtype=bool)
        pair[4:24, 4:18] = True
        independent = np.zeros(shape, dtype=bool)
        independent[8:24, 36:45] = True

        def candidates(*_args, **_kwargs):
            return [
                {"candidate_id": "broad", "kind": "community",
                 "mask": broad, "component_ids": [1, 2, 3], "score": 0.8},
                {"candidate_id": "pair", "kind": "pair",
                 "mask": pair, "component_ids": [1, 2], "score": 0.99},
                {"candidate_id": "other", "kind": "pair",
                 "mask": independent, "component_ids": [4, 5], "score": 0.7},
            ]

        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0, max_candidates=8, max_objects=3,
            candidate_provider=candidates, model_fitter=_fake_paint,
            geometry_optimizer=_fast_geometry,
        )
        selected = {item["candidate_id"] for item in result["proposals"]}
        self.assertEqual(selected, {"broad", "other"})
        rejected = next(item for item in result["decisions"]
                        if item["candidate_id"] == "pair")
        self.assertEqual(rejected["status"], "overlap_rejected")
        self.assertTrue(rejected["overlaps"])
        broad_proposal = next(item for item in result["proposals"]
                              if item["candidate_id"] == "broad")
        self.assertIn("pair", broad_proposal["selection"]["overlap_rejections"])

        # Even under a two-call geometry cap, the overlap-aware shortlist
        # keeps the broad object plus the independent object, not its nested
        # pair alias.
        tight = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0,
            max_candidates=8,
            max_geometry_candidates=2,
            max_objects=2,
            candidate_provider=candidates,
            model_fitter=_fake_paint,
            geometry_optimizer=_fast_geometry,
        )
        self.assertEqual(
            {item["candidate_id"] for item in tight["proposals"]},
            {"broad", "other"},
        )
        pair_decision = next(item for item in tight["decisions"]
                             if item["candidate_id"] == "pair")
        self.assertEqual(pair_decision["status"],
                         "geometry_shortlist_deferred")

    def test_incomplete_smooth_pair_is_seed_only(self):
        shape = (28, 60)
        pair = np.zeros(shape, dtype=bool)
        pair[4:24, 4:28] = True
        whole = np.zeros(shape, dtype=bool)
        whole[4:24, 4:48] = True

        def candidates(*_args, **_kwargs):
            return [
                {"candidate_id": "pair-seed", "kind": "pair",
                 "mask": pair, "component_ids": [1, 2], "score": 0.95,
                 "evidence": {"external_smooth_edges": 1,
                              "external_smooth_shared_boundary": 20}},
                {"candidate_id": "whole-field", "kind": "smooth_field",
                 "mask": whole, "component_ids": [1, 2, 3], "score": 0.8,
                 "evidence": {"external_smooth_edges": 0,
                              "ownership_closed_over_smooth_graph": True}},
            ]

        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0,
            candidate_provider=candidates, model_fitter=_fake_paint,
            geometry_optimizer=_fast_geometry,
        )
        self.assertEqual(
            [item["candidate_id"] for item in result["proposals"]],
            ["whole-field"],
        )
        decision = next(item for item in result["decisions"]
                        if item["candidate_id"] == "pair-seed")
        self.assertEqual(decision["status"], "ownership_seed_deferred")
        self.assertEqual(result["summary"]["ownership_seed_deferred"], 1)

    def _community_partition_fixture(self, external_support=0.20):
        shape = (20, 80)
        component_map = np.zeros(shape, dtype=np.int32)
        for component_id in range(1, 5):
            component_map[:, (component_id - 1) * 20:component_id * 20] = component_id
        mask = component_map <= 3
        context = {
            "component_map": component_map,
            "mutual_support_base": 0.38,
            "eligible_edges": (
                {"a": 1, "b": 2, "shared_boundary": 20,
                 "smooth_fraction": 1.0, "mutual_support": 1.0},
                {"a": 2, "b": 3, "shared_boundary": 20,
                 "smooth_fraction": 1.0, "mutual_support": 0.8},
                {"a": 3, "b": 4, "shared_boundary": 5,
                 "smooth_fraction": 1.0,
                 "mutual_support": external_support},
            ),
        }

        def candidates(*_args, **_kwargs):
            return [{
                "candidate_id": "community-field",
                "kind": "community",
                "mask": mask,
                "component_ids": [1, 2, 3],
                "score": 0.8,
                "evidence": {
                    "internal_edges": 2,
                    "external_smooth_edges": 1,
                    "external_smooth_shared_boundary": 5,
                    "ownership_closed_over_smooth_graph": False,
                },
                "_component_context": context,
            }]
        return shape, candidates

    def _model_guided_expansion_fixture(self, *, hard_edge=False,
                                        wrong_neighbour=False):
        height, width = 16, 60
        component_map = np.zeros((height, width), dtype=np.int32)
        for component_id in range(1, 6):
            x0 = (component_id - 1) * 12
            component_map[:, x0:x0 + 12] = component_id
        xs = np.linspace(0.0, 1.0, width)
        start = np.asarray((24.0, 72.0, 96.0))
        end = np.asarray((174.0, 202.0, 222.0))
        row = np.rint(start[None, :] * (1.0 - xs[:, None])
                      + end[None, :] * xs[:, None]).astype(np.uint8)
        rgb = np.repeat(row[None, :, :], height, axis=0)
        if wrong_neighbour:
            rgb[:, 36:] = np.asarray((225, 35, 170), dtype=np.uint8)
        labels = component_map - 1
        visible = np.ones((height, width), dtype=bool)
        palette = np.asarray([
            np.median(rgb[component_map == component_id], axis=0)
            for component_id in range(1, 6)
        ], dtype=np.uint8)
        edges = []
        for component_id in range(1, 5):
            is_hard = hard_edge and component_id == 3
            edges.append({
                "a": component_id,
                "b": component_id + 1,
                "shared_boundary": height,
                "source_delta_mean": 2.0,
                "source_delta_p90": 3.0,
                "smooth_fraction": 0.10 if is_hard else 1.0,
                "palette_distance": 30.0,
                "mutual_support": 1.0,
            })
        context = {
            "schema": "ai-vector-cleanroom.gradient-component-context/v1",
            "component_map": component_map,
            "eligible_edges": tuple(edges),
            "mutual_support_base": 0.38,
        }
        seed_mask = np.isin(component_map, (2, 3))

        def candidates(*_args, **_kwargs):
            return [{
                "candidate_id": "chain-seed",
                "kind": "monotonic_chain",
                "mask": seed_mask,
                "component_ids": [2, 3],
                "score": 0.8,
                "evidence": {
                    "internal_edges": 1,
                    "external_smooth_edges": 2,
                    "external_smooth_shared_boundary": 2 * height,
                    "ownership_closed_over_smooth_graph": False,
                },
                "_component_context": context,
            }]

        def paint(_rgb, _mask, **kwargs):
            if kwargs.get("validation_seed") == "reject_confirmation":
                return {"status": "skipped",
                        "reasons": ["forced_confirmation_rejection"]}
            return {
                "status": "proposed",
                "model": {
                    "type": "linear", "svg_type": "linearGradient",
                    "x1": 0.0, "y1": 0.0,
                    "x2": float(width - 1), "y2": 0.0,
                    "direction": [1.0, 0.0], "stop_count": 2,
                },
                "stops": [
                    {"offset": 0.0, "color": "#184860",
                     "rgb": [24, 72, 96]},
                    {"offset": 1.0, "color": "#aecaDE".lower(),
                     "rgb": [174, 202, 222]},
                ],
                "confidence": 0.95,
                "error": {
                    "heldout": {"mean": 0.2, "p90": 0.4, "p99": 0.8},
                    "solid_baseline": {
                        "mean": 8.0, "p90": 11.0, "p99": 14.0},
                    "heldout_mean_improvement": 7.8,
                },
                "validation": {"passed": True},
                "reasons": ["heldout_validation_passed"],
            }
        return rgb, labels, visible, palette, candidates, paint

    def _run_model_guided_expansion_fixture(self, **fixture_options):
        rgb, labels, visible, palette, candidates, paint = (
            self._model_guided_expansion_fixture(**fixture_options))
        result = propose_gradient_reconstruction(
            rgb, labels, visible, palette,
            geometry_error_percent=1.0,
            candidate_provider=candidates,
            model_fitter=paint,
            geometry_optimizer=_fast_geometry,
        )
        return result

    def test_model_guided_seed_expands_to_complete_multiband_object(self):
        result = self._run_model_guided_expansion_fixture()
        self.assertEqual(len(result["proposals"]), 1, result)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["candidate_family"], "model_guided_field")
        self.assertEqual(proposal["component_ids"], [1, 2, 3, 4, 5])
        expansion = proposal["candidate_evidence"]["model_guided_expansion"]
        self.assertEqual(expansion["before_component_ids"], [2, 3])
        self.assertEqual(expansion["after_component_ids"], [1, 2, 3, 4, 5])
        self.assertGreater(expansion["after_area"], expansion["before_area"])
        self.assertEqual(
            result["summary"]["model_guided_expansion_components_added"], 3)

    def test_model_guided_expansion_never_crosses_material_hard_edge(self):
        result = self._run_model_guided_expansion_fixture(hard_edge=True)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["component_ids"], [1, 2, 3])
        rejected = proposal["candidate_evidence"][
            "model_guided_expansion"]["rejected_components"]
        component_four = next(item for item in rejected
                              if item["component_id"] == 4)
        self.assertIn("material_hard_edge_smooth_fraction",
                      component_four["reasons"])

    def test_model_guided_expansion_rejects_wrong_colour_neighbour(self):
        result = self._run_model_guided_expansion_fixture(
            wrong_neighbour=True)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["component_ids"], [1, 2, 3])
        rejected = proposal["candidate_evidence"][
            "model_guided_expansion"]["rejected_components"]
        component_four = next(item for item in rejected
                              if item["component_id"] == 4)
        self.assertIn("robust_delta_e_threshold_failed",
                      component_four["reasons"])

    def test_model_guided_expansion_is_deterministic(self):
        first = self._run_model_guided_expansion_fixture()
        second = self._run_model_guided_expansion_fixture()
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )

    def test_expanded_field_confirmation_failure_rolls_back_and_defers(self):
        rgb, labels, visible, palette, candidates, base_paint = (
            self._model_guided_expansion_fixture())

        def unstable_paint(source, mask, **kwargs):
            if kwargs.get("validation_seed") == 0x5A17:
                return {"status": "skipped",
                        "reasons": ["forced_confirmation_rejection"]}
            return base_paint(source, mask, **kwargs)

        result = propose_gradient_reconstruction(
            rgb, labels, visible, palette,
            geometry_error_percent=1.0,
            candidate_provider=candidates,
            model_fitter=unstable_paint,
            geometry_optimizer=_fast_geometry,
        )
        self.assertEqual(result["proposals"], [])
        statuses = {item["status"] for item in result["decisions"]
                    if item["candidate_id"] == "chain-seed"}
        self.assertIn("independent_paint_revalidation_rejected", statuses)
        self.assertIn(
            "ownership_seed_deferred_after_expansion_revalidation_failure",
            statuses)
        self.assertEqual(
            result["summary"][
                "model_guided_expansion_revalidation_rejected"], 1)

    def test_paint_valid_community_can_close_only_across_certified_weak_bridge(self):
        shape, candidates = self._community_partition_fixture(0.20)
        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0,
            candidate_provider=candidates, model_fitter=_fake_paint,
            geometry_optimizer=_fast_geometry,
        )

        self.assertEqual(len(result["proposals"]), 1, result)
        proposal = result["proposals"][0]
        self.assertEqual(proposal["candidate_family"], "model_guided_field")
        ownership = proposal["candidate_evidence"]
        self.assertTrue(ownership["ownership_closed_after_certified_cuts"])
        self.assertEqual(ownership["unresolved_external_smooth_edges"], 0)
        self.assertEqual(len(ownership["certified_cut_edges"]), 1)
        self.assertEqual(
            result["summary"]["independent_paint_revalidation_passed"], 1)

    def test_direction_search_options_are_reused_for_independent_confirmation(self):
        shape, candidates = self._community_partition_fixture(0.20)
        calls = []

        def recording_paint(rgb, mask, **kwargs):
            calls.append(dict(kwargs))
            return _fake_paint(rgb, mask, **kwargs)

        with patch(
                "gradient_reconstruction_stage.fit_gradient_object_proposal",
                new=recording_paint):
            result = propose_gradient_reconstruction(
                np.zeros(shape + (3,), dtype=np.uint8),
                np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
                np.asarray([[10, 20, 30]], dtype=np.uint8),
                geometry_error_percent=1.0,
                candidate_provider=candidates,
                model_fitter=recording_paint,
                geometry_optimizer=_fast_geometry,
            )

        self.assertEqual(len(result["proposals"]), 1, result)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [call.get("linear_direction_candidates") for call in calls],
            [5, 5],
        )
        self.assertEqual(
            [call.get("validation_seed", 0) for call in calls],
            [0, 0x5A17],
        )
        self.assertEqual(result["summary"]["direction_search_candidates"], 1)

    def test_strong_unresolved_external_edge_stays_deferred(self):
        shape, candidates = self._community_partition_fixture(0.80)
        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0,
            candidate_provider=candidates, model_fitter=_fake_paint,
            geometry_optimizer=_fast_geometry,
        )

        self.assertEqual(result["proposals"], [])
        decision = next(item for item in result["decisions"]
                        if item["candidate_id"] == "community-field")
        self.assertEqual(decision["status"], "ownership_seed_deferred")

    def test_independent_holdout_must_confirm_certified_partition(self):
        shape, candidates = self._community_partition_fixture(0.20)

        def unstable_paint(rgb, mask, **kwargs):
            if kwargs.get("validation_seed") == 0x5A17:
                return {"status": "skipped",
                        "reasons": ["confirmation_seed_rejected"]}
            return _fake_paint(rgb, mask, **kwargs)

        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0,
            candidate_provider=candidates, model_fitter=unstable_paint,
            geometry_optimizer=_fast_geometry,
        )

        self.assertEqual(result["proposals"], [])
        decision = next(item for item in result["decisions"]
                        if item["candidate_id"] == "community-field")
        self.assertEqual(
            decision["status"], "independent_paint_revalidation_rejected")

    def test_material_hard_edge_fails_closed(self):
        height = width = 48
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        visible = np.zeros((height, width), dtype=bool)
        visible[8:40, 8:40] = True
        rgb[8:40, 8:24] = (220, 30, 30)
        rgb[8:40, 24:40] = (20, 40, 220)
        labels = np.zeros((height, width), dtype=np.int32)
        labels[8:40, 24:40] = 1
        result = propose_gradient_reconstruction(
            rgb, labels, visible,
            np.asarray(((220, 30, 30), (20, 40, 220))),
            geometry_error_percent=1.0, max_candidates=10, max_objects=2,
            geometry_optimizer=_fast_geometry,
            candidate_options={
                "min_component_area": 20,
                "min_candidate_area": 80,
                "min_chromatic_area": 100,
                "min_shared_boundary": 4,
                "smooth_delta": 30,
            },
            model_options={
                "min_pixels": 64,
                "maximum_mean_error": 20,
                "maximum_p90_error": 30,
                "maximum_p99_error": 40,
            },
        )
        self.assertEqual(result["status"], "skipped")
        reasons = [reason for item in result["decisions"]
                   for reason in item["reasons"]]
        self.assertTrue(any("hard_edge" in reason or "hard_label" in reason
                            for reason in reasons))

    def test_radial_model_retains_native_circle_evidence(self):
        height = width = 96
        yy, xx = np.mgrid[:height, :width]
        radius = np.hypot(xx - 48.0, yy - 48.0)
        circle = radius <= 34.0
        start = np.asarray((255.0, 225.0, 50.0))
        end = np.asarray((185.0, 90.0, 5.0))
        t = np.clip(radius / 34.0, 0.0, 1.0)
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[circle] = np.rint(start + (end - start) * t[..., None])[circle]
        labels = np.zeros((height, width), dtype=np.int32)
        labels[circle] = np.minimum(5, (t[circle] * 6).astype(int))
        palette = np.asarray([
            start + (end - start) * ((index + 0.5) / 6.0)
            for index in range(6)
        ])

        def candidates(*_args, **_kwargs):
            return [{"candidate_id": "sun", "kind": "source_chromatic",
                     "mask": circle, "component_ids": list(range(6)),
                     "score": 1.0}]

        def circle_geometry(mask, **kwargs):
            result = _fast_geometry(mask, **kwargs)
            result["path"] = "M14 48 A34 34 0 1 0 82 48 A34 34 0 1 0 14 48 Z"
            result["anchors_after"] = 2
            result["segment_count_after"] = 2
            result["native_primitive"] = {
                "element": "circle", "cx": 48.0, "cy": 48.0, "r": 34.0}
            return result

        result = propose_gradient_reconstruction(
            rgb, labels, circle, palette,
            geometry_error_percent=0.8, max_candidates=4, max_objects=1,
            candidate_provider=candidates,
            geometry_optimizer=circle_geometry,
            model_options={
                "min_pixels": 64,
                "maximum_mean_error": 7,
                "maximum_p90_error": 10,
                "maximum_p99_error": 14,
            },
        )
        proposal = result["proposals"][0]
        self.assertEqual(proposal["model"]["type"], "radial")
        self.assertTrue(proposal["geometry"]["primitive_first"])
        self.assertEqual(proposal["geometry"]["native_primitives"][0]["element"],
                         "circle")

    def test_output_is_strict_json_and_deterministic(self):
        shape = (24, 32)
        mask = np.zeros(shape, dtype=bool)
        mask[4:20, 5:27] = True

        def candidates(*_args, **_kwargs):
            return [{"candidate_id": "one", "kind": "community",
                     "mask": mask, "component_ids": (2, 1), "score": 0.7}]

        args = (
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32),
            np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
        )
        kwargs = dict(
            geometry_error_percent=1.0,
            candidate_provider=candidates,
            model_fitter=_fake_paint,
            geometry_optimizer=_fast_geometry,
        )
        first = propose_gradient_reconstruction(*args, **kwargs)
        second = propose_gradient_reconstruction(*args, **kwargs)
        encoded_first = json.dumps(first, ensure_ascii=False, sort_keys=True,
                                   allow_nan=False)
        encoded_second = json.dumps(second, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False)
        self.assertEqual(encoded_first, encoded_second)
        restored = decode_mask_rle(first["proposals"][0]["mask"])
        np.testing.assert_array_equal(restored, mask)
        self.assertEqual(first["objective"]["colour_used_for_geometry"], False)

    def test_twenty_four_paint_candidates_use_bounded_deterministic_geometry(self):
        shape = (96, 144)
        families = ("source_chromatic", "community", "monotonic_chain", "pair")
        specifications = []
        for index in range(24):
            row, column = divmod(index, 6)
            mask = np.zeros(shape, dtype=bool)
            y0, x0 = 3 + row * 22, 3 + column * 22
            mask[y0:y0 + 14, x0:x0 + 14] = True
            specifications.append({
                "candidate_id": f"candidate-{index:02d}",
                "kind": families[index % len(families)],
                "mask": mask,
                "component_ids": [index * 2 + 1, index * 2 + 2],
                "score": 1.0 - index / 100.0,
            })

        def candidates(*_args, **_kwargs):
            return specifications

        calls = {"count": 0}

        def counted_geometry(mask, **kwargs):
            calls["count"] += 1
            return _fast_geometry(mask, **kwargs)

        args = (
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32),
            np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
        )
        kwargs = dict(
            geometry_error_percent=1.0,
            max_candidates=48,
            max_geometry_candidates=7,
            max_objects=4,
            candidate_provider=candidates,
            model_fitter=_fake_paint,
            geometry_optimizer=counted_geometry,
        )
        first = propose_gradient_reconstruction(*args, **kwargs)
        first_calls = calls["count"]
        calls["count"] = 0
        second = propose_gradient_reconstruction(*args, **kwargs)
        second_calls = calls["count"]

        self.assertEqual(first_calls, 7)
        self.assertEqual(second_calls, 7)
        self.assertEqual(first["summary"]["paint_eligible"], 24)
        self.assertEqual(first["summary"]["geometry_shortlisted"], 7)
        self.assertEqual(first["summary"]["geometry_shortlist_deferred"], 17)
        self.assertEqual(first["summary"]["unique_geometry_masks_evaluated"], 7)
        deferred = [item for item in first["decisions"]
                    if item["status"] == "geometry_shortlist_deferred"]
        self.assertEqual(len(deferred), 17)
        self.assertTrue(all(item["paint_status"] == "proposed"
                            for item in deferred))
        self.assertEqual(
            first["geometry_shortlist"]["candidate_ids"],
            second["geometry_shortlist"]["candidate_ids"],
        )
        family_by_id = {item["candidate_id"]: item["kind"]
                        for item in specifications}
        shortlisted_families = {
            family_by_id[candidate_id]
            for candidate_id in first["geometry_shortlist"]["candidate_ids"]
        }
        self.assertEqual(shortlisted_families, set(families))
        shortlist_reasons = [
            reason
            for values in first["geometry_shortlist"][
                "reasons_by_candidate"].values()
            for reason in values
        ]
        self.assertTrue(any(reason.startswith("disjoint_area_coverage_seed")
                            for reason in shortlist_reasons))
        self.assertTrue(any(reason.startswith("family_paint_gain_reserve")
                            for reason in shortlist_reasons))
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_identical_shortlisted_masks_share_geometry_digest_cache(self):
        shape = (36, 52)
        shared = np.zeros(shape, dtype=bool)
        shared[4:28, 4:28] = True
        separate = np.zeros(shape, dtype=bool)
        separate[8:28, 34:48] = True

        def candidates(*_args, **_kwargs):
            return [
                {"candidate_id": "same-broad", "kind": "source_chromatic",
                 "mask": shared, "component_ids": [1, 2], "score": 0.9},
                {"candidate_id": "same-pair", "kind": "pair",
                 "mask": shared.copy(), "component_ids": [1, 2], "score": 0.8},
                {"candidate_id": "separate", "kind": "community",
                 "mask": separate, "component_ids": [3, 4], "score": 0.7},
            ]

        calls = {"count": 0}

        def counted_geometry(mask, **kwargs):
            calls["count"] += 1
            return _fast_geometry(mask, **kwargs)

        result = propose_gradient_reconstruction(
            np.zeros(shape + (3,), dtype=np.uint8),
            np.zeros(shape, dtype=np.int32), np.ones(shape, dtype=bool),
            np.asarray([[10, 20, 30]], dtype=np.uint8),
            geometry_error_percent=1.0,
            max_geometry_candidates=3,
            max_objects=3,
            candidate_provider=candidates,
            model_fitter=_fake_paint,
            geometry_optimizer=counted_geometry,
        )
        self.assertEqual(result["summary"]["geometry_shortlisted"], 3)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(result["summary"]["unique_geometry_masks_evaluated"], 2)
        self.assertEqual(result["summary"]["geometry_mask_cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
