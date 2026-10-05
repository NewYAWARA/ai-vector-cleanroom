# -*- coding: utf-8 -*-
"""Process-local raw gradient-stage cache contracts."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, PngImagePlugin

import clean_base
from tests.test_clean_base_gradient_integration import (
    _geometry,
    _proposal,
    _run_clean_base,
    _stage,
)


def _cache_key_callable(*_args, **_kwargs):
    """Stable callable identity for key-contract tests."""
    return None


def _linear_stage(mask):
    height, width = mask.shape
    path = (
        f"M0 0 L{width - 1} 0 L{width - 1} {height - 1} "
        f"L0 {height - 1} Z"
    )
    model = {
        "type": "linear",
        "svg_type": "linearGradient",
        "gradient_units": "userSpaceOnUse",
        "x1": 0.0,
        "y1": 0.0,
        "x2": float(width - 1),
        "y2": float(height - 1),
        "stop_count": 3,
        "bounded_maximum_stops": 5,
    }
    stops = [
        {"offset": 0.0, "color": "#102030", "rgb": [16, 32, 48]},
        {"offset": 0.5, "color": "#406070", "rgb": [64, 96, 112]},
        {"offset": 1.0, "color": "#90b0c0", "rgb": [144, 176, 192]},
    ]
    return _stage(_proposal(mask, model, stops, _geometry(path, anchors=4)))


class CleanBaseGradientCacheTests(unittest.TestCase):
    def test_store_and_hit_are_isolated_from_nested_result_mutation(self):
        cache = clean_base.new_gradient_stage_cache()
        produced = {
            "schema": "stage/v1",
            "proposals": [{"component_ids": [1, 2], "stops": ["#102030"]}],
            "summary": {"objects_selected": 1},
        }
        calls = []

        def producer():
            calls.append("called")
            return produced

        first, first_evidence = clean_base._gradient_stage_cache_get_or_compute(
            cache, "MUTATION-KEY", producer)
        self.assertEqual(first_evidence["status"], "miss")

        # The miss result is the producer's object.  Mutating it must not alter
        # the detached copy stored for later candidate builds.
        first["proposals"][0]["component_ids"].append(99)
        first["summary"]["objects_selected"] = 9
        second, second_evidence = (
            clean_base._gradient_stage_cache_get_or_compute(
                cache, "MUTATION-KEY", producer))
        self.assertEqual(second_evidence["status"], "hit")
        self.assertEqual(second["proposals"][0]["component_ids"], [1, 2])
        self.assertEqual(second["summary"]["objects_selected"], 1)

        # A hit must also be detached: downstream assembly mutation from one
        # candidate cannot contaminate any following candidate.
        second["proposals"][0]["stops"].append("#ffffff")
        third, third_evidence = clean_base._gradient_stage_cache_get_or_compute(
            cache, "MUTATION-KEY", producer)
        self.assertEqual(third_evidence["status"], "hit")
        self.assertEqual(third["proposals"][0]["stops"], ["#102030"])
        self.assertEqual(calls, ["called"])

        audit = clean_base.gradient_stage_cache_audit(cache)
        self.assertEqual(audit["requests"], 3)
        self.assertEqual(audit["misses"], 1)
        self.assertEqual(audit["hits"], 2)
        self.assertEqual(audit["entry_count"], 1)

    def test_stage_parameter_difference_gets_a_distinct_cache_key_and_miss(self):
        den = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        arrays = (("den", den),)
        base_parameters = {
            "geometry_error_percent": 0.25,
            "max_candidates": 48,
        }
        changed_parameters = {
            "geometry_error_percent": 0.50,
            "max_candidates": 48,
        }
        base_key = clean_base._gradient_stage_cache_key(
            arrays, base_parameters, _cache_key_callable, "stage/v1")
        changed_key = clean_base._gradient_stage_cache_key(
            arrays, changed_parameters, _cache_key_callable, "stage/v1")

        self.assertNotEqual(base_key, changed_key)
        cache = clean_base.new_gradient_stage_cache()
        calls = []

        def producer():
            calls.append("called")
            return {"proposals": []}

        _, base_evidence = clean_base._gradient_stage_cache_get_or_compute(
            cache, base_key, producer)
        _, changed_evidence = clean_base._gradient_stage_cache_get_or_compute(
            cache, changed_key, producer)
        self.assertEqual(base_evidence["status"], "miss")
        self.assertEqual(changed_evidence["status"], "miss")
        self.assertEqual(calls, ["called", "called"])

        audit = clean_base.gradient_stage_cache_audit(cache)
        self.assertEqual(audit["requests"], 2)
        self.assertEqual(audit["misses"], 2)
        self.assertEqual(audit["hits"], 0)
        self.assertEqual(audit["entry_count"], 2)

    def test_cache_disabled_miss_and_hit_emit_byte_identical_svg(self):
        mask = np.ones((18, 24), dtype=bool)
        stage = _linear_stage(mask)
        calls = []

        def proposer(*_args, **_kwargs):
            calls.append("called")
            return stage

        uncached_stats, uncached_svg = _run_clean_base(
            stage, mask, gradient_stage_cache=None, proposer=proposer)
        cache = clean_base.new_gradient_stage_cache()
        miss_stats, miss_svg = _run_clean_base(
            stage, mask, gradient_stage_cache=cache, proposer=proposer)
        hit_stats, hit_svg = _run_clean_base(
            stage, mask, gradient_stage_cache=cache, proposer=proposer)

        self.assertEqual(uncached_svg.encode("utf-8"),
                         miss_svg.encode("utf-8"))
        self.assertEqual(uncached_svg.encode("utf-8"),
                         hit_svg.encode("utf-8"))
        self.assertEqual(calls, ["called", "called"])
        self.assertEqual(
            uncached_stats.palette_audit["gradient_reconstruction"]
            ["cache"]["status"], "disabled")
        self.assertEqual(
            miss_stats.palette_audit["gradient_reconstruction"]
            ["cache"]["status"], "miss")
        self.assertEqual(
            hit_stats.palette_audit["gradient_reconstruction"]
            ["cache"]["status"], "hit")

        audit = clean_base.gradient_stage_cache_audit(cache)
        self.assertEqual(audit["requests"], 2)
        self.assertEqual(audit["misses"], 1)
        self.assertEqual(audit["hits"], 1)
        self.assertEqual(audit["entry_count"], 1)

    def test_gradients_off_bypasses_stage_without_cache_request(self):
        mask = np.ones((18, 24), dtype=bool)
        stage = _linear_stage(mask)
        cache = clean_base.new_gradient_stage_cache()

        def unexpected_proposer(*_args, **_kwargs):
            self.fail("gradient proposer must not run when gradients are off")

        stats, svg = _run_clean_base(
            stage, mask, gradient_stage_cache=cache, gradients="off",
            trace_fill="#192d41", proposer=unexpected_proposer)

        reconstruction = stats.palette_audit["gradient_reconstruction"]
        self.assertEqual(reconstruction["status"], "disabled")
        self.assertEqual(reconstruction["cache"]["status"], "bypassed")
        self.assertEqual(reconstruction["cache"]["reason"], "gradients_off")
        self.assertNotIn("<linearGradient", svg)
        self.assertNotIn("<radialGradient", svg)

        audit = clean_base.gradient_stage_cache_audit(cache)
        self.assertEqual(audit["requests"], 0)
        self.assertEqual(audit["hits"], 0)
        self.assertEqual(audit["misses"], 0)
        self.assertEqual(audit["bypasses"], 1)
        self.assertEqual(audit["entry_count"], 0)


class PostFitSceneCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.source = Path(temp.name) / "source.png"
        self.rgba = np.full((8, 8, 4), 255, np.uint8)
        Image.fromarray(self.rgba).save(self.source)
        self.cache = clean_base.new_gradient_stage_cache()
        self.options = dict(operation="ownership", svg_text='<svg><path d="M0 0Z"/></svg>',
            source_path=self.source, processed_rgba=self.rgba.copy(),
            native_reference_rgba=self.rgba.copy(),
            fields=[{"mask": np.ones((8, 8), bool), "paint": {"stop": "#208060"}}],
            parameters={"budget_percent": .25}, stage_callable=_cache_key_callable)

    def call(self, producer, **changed):
        return clean_base._cached_post_fit_scene(self.cache, producer=producer,
                                                **{**self.options, **changed})

    def test_full_scene_key_changes_for_every_source_paint_and_geometry_input(self):
        calls = []
        def produce():
            calls.append(1)
            return {"status": "no_change"}
        _, first = self.call(produce)
        _, repeated = self.call(produce)
        self.assertEqual((first["status"], repeated["status"]), ("miss", "hit"))
        self.assertFalse(repeated["producer_executed"])
        self.assertTrue(repeated["validation_reused_for_exact_inputs"])
        processed = self.rgba.copy(); processed[0, 0, 0] -= 1
        native = self.rgba.copy(); native[1, 1, 3] -= 1
        mask = copy.deepcopy(self.options["fields"]); mask[0]["mask"][0, 0] = False
        paint = copy.deepcopy(self.options["fields"]); paint[0]["paint"]["stop"] = "#218060"
        changes = [dict(operation="partial"), dict(svg_text=self.options["svg_text"]+" "),
            dict(processed_rgba=processed), dict(native_reference_rgba=native),
            dict(fields=mask), dict(fields=paint), dict(parameters={"budget_percent": .5}),
            dict(stage_callable=lambda: None)]
        keys = {first["key_sha256"]}
        for changed in changes:
            with self.subTest(changed=list(changed)):
                _, evidence = self.call(produce, **changed)
                self.assertEqual(evidence["status"], "miss")
                self.assertNotIn(evidence["key_sha256"], keys)
                keys.add(evidence["key_sha256"])
        # Source byte metadata and decoded pixels are both identity-bearing.
        info = PngImagePlugin.PngInfo(); info.add_text("avc_reference_alpha_origin", "native")
        Image.fromarray(self.rgba).save(self.source, pnginfo=info)
        _, metadata_change = self.call(produce)
        self.assertEqual(metadata_change["status"], "miss")
        changed = self.rgba.copy(); changed[2, 2, 1] -= 1
        Image.fromarray(changed).save(self.source)
        _, pixel_change = self.call(produce)
        self.assertEqual(pixel_change["status"], "miss")
        self.assertEqual(len(calls), 11)

    def test_result_reports_and_numpy_masks_are_detached_on_miss_and_every_hit(self):
        made = ("<svg/>", {"one": {"mask": np.ones((8, 8), bool),
                                   "geometry": {"proof": ["original"]}}})
        first, _ = self.call(lambda: made)
        first[1]["one"]["mask"][0, 0] = False
        first[1]["one"]["geometry"]["proof"].append("mutated")
        hit, _ = self.call(lambda: self.fail("repeated transaction"))
        self.assertTrue(hit[1]["one"]["mask"].all())
        self.assertEqual(hit[1]["one"]["geometry"]["proof"], ["original"])
        hit[1]["one"]["geometry"]["proof"].clear()
        again, _ = self.call(lambda: self.fail("repeated transaction"))
        self.assertEqual(again[1]["one"]["geometry"]["proof"], ["original"])
        self.assertEqual(clean_base.gradient_stage_cache_audit(self.cache)
                         ["post_fit_scene_cache"]["hits"], 2)

    def test_failed_or_concurrently_changed_transactions_are_never_stored(self):
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.call(lambda: (_ for _ in ()).throw(RuntimeError("cancelled")))
        self.assertEqual(clean_base._post_fit_scene_cache_audit(self.cache)["entry_count"], 0)
        def changed_source():
            altered = self.rgba.copy(); altered[3, 3, 2] = 0
            Image.fromarray(altered).save(self.source)
            return {"status": "no_change"}
        _, evidence = self.call(changed_source)
        self.assertFalse(evidence["stored"])
        self.assertEqual(evidence["storage_reason"], "inputs_changed_during_transaction")
        self.assertEqual(clean_base._post_fit_scene_cache_audit(self.cache)["entry_count"], 0)
        _, retried = self.call(lambda: {"status": "no_change"})
        self.assertEqual(retried["status"], "miss")

    def test_per_job_cache_is_bounded_by_entries_and_payload_bytes(self):
        with patch.object(clean_base, "POST_FIT_SCENE_CACHE_MAX_ENTRIES", 2):
            for index in range(3):
                self.call(lambda: {"proof": [1]}, svg_text=f"<svg id='{index}'/>")
            audit = clean_base._post_fit_scene_cache_audit(self.cache)
            self.assertEqual((audit["entry_count"], audit["evictions"]), (2, 1))
            _, first_again = self.call(lambda: {"proof": [1]}, svg_text="<svg id='0'/>")
            self.assertEqual(first_again["status"], "miss")
        fresh = clean_base.new_gradient_stage_cache()
        with patch.object(clean_base, "POST_FIT_SCENE_CACHE_MAX_BYTES", 100):
            _, evidence = clean_base._cached_post_fit_scene(fresh, producer=lambda: "large"*100,
                                                          **self.options)
        self.assertFalse(evidence["stored"])
        self.assertEqual(clean_base._post_fit_scene_cache_audit(fresh)["entry_count"], 0)
        _, separate_job = clean_base._cached_post_fit_scene(fresh, producer=lambda: {}, **self.options)
        self.assertEqual(separate_job["status"], "miss")

    def test_assembler_reuses_both_transaction_routes_and_counts_no_repeated_probes(self):
        png = self.source.read_bytes()
        mask = np.ones((18, 24), bool)
        stage = _linear_stage(mask)
        stage["proposals"][0]["enclosed_source_component_candidate"] = {"schema": "fixture"}
        stage["paint_ready_alternatives"] = [{"candidate_id": "pending"}]
        def proposer(*args, **kwargs):
            return stage
        def ownership(svg, *args, **kwargs):
            return svg, {}, {"status": "no_change", "performance": {
                "bounded_span_search": {"native_probe_count": 8, "helper_calls": 1,
                                        "elapsed_seconds": 32., "wall_seconds_since_first_request": 33.}}}
        def partial(svg, *args, **kwargs):
            return svg, [], [{"status": "partial_paint_rejected"}], []
        with patch.object(Path, "read_bytes", return_value=png), \
                patch("gradient_source_components.apply_source_component_candidates", new=ownership), \
                patch("gradient_paint_only.apply_pending_paint_alternatives", new=partial):
            first, first_svg = _run_clean_base(stage, mask, proposer=proposer, gradient_stage_cache=self.cache)
            second, second_svg = _run_clean_base(stage, mask, proposer=proposer, gradient_stage_cache=self.cache)
        self.assertEqual(first_svg, second_svg)
        a, b = first.palette_audit, second.palette_audit
        self.assertEqual(a["gradient_source_ownership_transaction"]["execution_cache"]["status"], "miss")
        self.assertEqual(b["gradient_source_ownership_transaction"]["execution_cache"]["status"], "hit")
        self.assertEqual(b["gradient_partial_paint_execution_cache"]["status"], "hit")
        self.assertEqual(a["gradient_contour_span_scene_searches"]["native_probe_count"], 8)
        summary = b["gradient_contour_span_scene_searches"]
        self.assertEqual((summary["native_probe_count"], summary["helper_calls"]), (0, 0))
        self.assertEqual(summary["exact_scene_cache_hits"], 1)
        self.assertEqual(summary["assembled_scene_search_count"], 0)
        self.assertEqual(summary["assembled_scene_evidence_count"], 1)
        self.assertEqual(summary["scenes"][0]["wall_seconds_since_first_request"], 0.)
        self.assertEqual(summary["scenes"][0]["original_validation_search"]["native_probe_count"], 8)

    def test_actual_native_source_hole_transaction_reuses_identical_certified_svg(self):
        from tests.test_gradient_source_components import GradientSourceComponentTests
        from gradient_source_components import apply_source_component_candidates, final_source_component_matches
        fixture = GradientSourceComponentTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        pending = fixture.propose()
        regions = [{"candidate_id": "candidate-one", "mask": fixture.mask,
                    "enclosed_source_component_candidate": pending}]
        before = fixture.document(fixture.baseline_path)
        calls = []
        def producer():
            calls.append(1)
            return apply_source_component_candidates(before, regions, fixture.source, fixture.raw)
        options = dict(operation="source_ownership_and_bounded_spans", svg_text=before,
            source_path=fixture.source, processed_rgba=fixture.raw, fields=regions,
            parameters={"budget_percent": .25}, stage_callable=apply_source_component_candidates,
            producer=producer)
        miss, _ = clean_base._cached_post_fit_scene(self.cache, **options)
        hit, evidence = clean_base._cached_post_fit_scene(self.cache, **options)
        self.assertEqual(len(calls), 1)
        self.assertEqual(evidence["status"], "hit")
        self.assertEqual(miss[0], hit[0])
        self.assertEqual(hit[2]["status"], "committed")
        self.assertTrue(final_source_component_matches(ET.fromstring(hit[0]),
                       hit[1]["candidate-one"]["geometry"], fixture.source))

    def test_actual_partial_paint_scene_checks_and_final_identity_survive_cache(self):
        from tests.test_gradient_paint_only import GradientPaintOnlyTests
        from tests.test_designer_quality import _source_certified_field_metadata
        from gradient_reconstruction_stage import encode_mask_rle
        from gradient_paint_only import apply_pending_paint_alternatives, final_paint_only_matches
        before, _, source, rgba, region = GradientPaintOnlyTests().fixture(self.source.parent)
        option = {"schema": "ai-vector-cleanroom.paint-ready-alternative/v1",
            "status": "pending_native_source_and_existing_path_validation", "geometry_certified": False,
            "candidate_id": region["candidate_id"], "mask": encode_mask_rle(region["mask"]),
            "model": {"type": "linear", "x1": 8, "y1": 0, "x2": 88, "y2": 0},
            "stops": [{"offset": 0, "color": "#184838"}, {"offset": 1, "color": "#98a878"}],
            "heldout_evidence": _source_certified_field_metadata()["gradient_details"][0]["validation"]["paint"]}
        calls = []
        def producer():
            calls.append(1)
            return apply_pending_paint_alternatives(before, [option], source, rgba)
        options = dict(operation="pending_partial_paint", svg_text=before, source_path=source,
            processed_rgba=rgba, fields=[option], parameters={"budget_percent": .25},
            stage_callable=apply_pending_paint_alternatives, producer=producer)
        miss, _ = clean_base._cached_post_fit_scene(self.cache, **options)
        hit, evidence = clean_base._cached_post_fit_scene(self.cache, **options)
        self.assertEqual(len(calls), 1)
        self.assertEqual(evidence["status"], "hit")
        self.assertEqual(miss, hit)
        self.assertEqual(len(hit[1]), 1)
        geometry = hit[1][0]["validation"]["geometry"]
        self.assertTrue(final_paint_only_matches(ET.fromstring(hit[0]), geometry, source))
        self.assertTrue(all(row["accepted"] for row in geometry["source_paint_only"]["source_scene_checks"]))


if __name__ == "__main__":
    unittest.main()
