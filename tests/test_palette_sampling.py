"""Sampling is not ownership: real source pixels, caps and gaps stay intact."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from palette_sampling import (
    independent_sampling_mask, discover_with_sampling_hypothesis)


def _scene(kind="multicolour"):
    image = Image.new("RGB", (512, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 25, 380, 230), fill=(90, 170, 105))
    if kind == "subtle_gradient":
        array = np.asarray(image).copy()
        ramp = np.linspace(0, 15, 351).astype(np.uint8)
        array[25:231, 30:381, 0] += ramp[None, :]
        image = Image.fromarray(array)
        draw = ImageDraw.Draw(image)
    elif kind == "narrow_gaps":
        draw.rectangle((150, 25, 151, 230), fill="white")
    draw.line((30, 290, 170, 290), fill=(20, 50, 130), width=2)
    if kind == "multicolour":
        draw.line((100, 290, 170, 290), fill=(200, 50, 20), width=2)
    if kind == "mixed_caps":
        draw.line((30, 330, 170, 330), fill=(90, 20, 140), width=2)
        draw.ellipse((29, 329, 31, 331), fill=(90, 20, 140))
        draw.ellipse((169, 329, 171, 331), fill=(90, 20, 140))
    return image


class PaletteSamplingTests(unittest.TestCase):
    def test_sampling_only_exclusion_preserves_counter_and_all_inputs(self):
        rgb = np.asarray(_scene()).copy()
        vis = (rgb < 250).any(axis=2)
        counter = np.zeros_like(vis)
        counter[280:300, 90:105] = True
        before = [v.copy() for v in (rgb, vis, counter)]
        mask, audit = independent_sampling_mask(rgb, vis, counter)
        self.assertEqual(audit["status"], "proposed")
        self.assertGreater(audit["excluded_pixels"], 100)
        self.assertTrue(mask[counter].all())
        self.assertFalse(audit["renderable_mask_changed"])
        for original, expected in zip((rgb, vis, counter), before):
            np.testing.assert_array_equal(original, expected)
        json.dumps(audit, allow_nan=False)

    def test_line_only_insufficient_samples_falls_back(self):
        rgb = np.full((512, 512, 3), 255, dtype=np.uint8)
        rgb[100:102, 20:220] = (10, 10, 10)
        rgb[200:202, 20:220] = (100, 20, 10)
        vis = (rgb < 250).any(axis=2)
        mask, audit = independent_sampling_mask(rgb, vis)
        self.assertEqual(audit["reason"], "insufficient_independent_fill_samples")
        np.testing.assert_array_equal(mask, vis)

    def test_wide_fill_edges_never_become_sampling_components(self):
        rgb = np.asarray(_scene("narrow_gaps")).copy()
        vis = (rgb < 250).any(axis=2)
        mask, audit = independent_sampling_mask(rgb, vis)
        self.assertEqual(audit["status"], "proposed")
        np.testing.assert_array_equal(mask[25:231, 30:381], vis[25:231, 30:381])
        self.assertFalse(mask[25:231, 150:152].any())

    def test_scale_adaptive_area_budget_and_determinism(self):
        for size in (256, 512, 1024):
            rgb = np.full((size, size, 3), 255, dtype=np.uint8)
            rgb[10:size // 2, 10:size // 2] = (100, 140, 200)
            rgb[3 * size // 4, 10:10 + size // 3] = (10, 50, 80)
            vis = (rgb < 250).any(axis=2)
            first, audit = independent_sampling_mask(rgb, vis)
            second, repeat = independent_sampling_mask(rgb, vis)
            self.assertEqual(audit["status"], "proposed")
            self.assertEqual(audit, repeat)
            np.testing.assert_array_equal(first, second)

    def test_fragmented_noise_falls_back_before_component_search(self):
        rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
        checker = np.indices((256, 256)).sum(axis=0) % 2 == 0
        rgb[checker] = 0
        with patch("stroke_engine.connected_components", side_effect=AssertionError("unbounded search")):
            mask, audit = independent_sampling_mask(rgb, checker)
        self.assertEqual(audit["reason"], "source_fragmentation_exceeds_sampling_budget")
        np.testing.assert_array_equal(mask, checker)

    def test_paired_context_and_baseline_reserved_under_fixed_budget(self):
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        labels = np.zeros((24, 32), dtype=np.int32)
        alternative = np.ones_like(labels)
        vis = np.ones_like(labels, dtype=bool)
        pal = np.array([[20, 40, 60], [80, 90, 100]])
        calls = []
        def discovery(source, lab, visible, palette, max_candidates, **kwargs):
            calls.append((lab, visible, palette, max_candidates))
            start = 0 if lab is labels else 12
            result = []
            for n in range(max_candidates):
                mask = np.zeros_like(vis); mask[start + n, :8] = True
                result.append({"candidate_id": str(n), "kind": "pair", "mask": mask,
                               "_component_context": {"palette": palette}})
            return result
        result = discover_with_sampling_hypothesis(
            rgb, labels, vis, pal, hypothesis={"palette": pal[::-1], "lab_all": alternative},
            discovery=discovery, max_candidates=12)
        self.assertEqual(len(result), 12)
        self.assertEqual(sum(v["candidate_id"].startswith("baseline:") for v in result), 8)
        self.assertEqual(sum(v["candidate_id"].startswith("sampling:") for v in result), 4)
        self.assertTrue(all(call[1] is vis for call in calls))
        for candidate in result[8:]:
            self.assertIs(candidate["_sampling_labels"], alternative)
            np.testing.assert_array_equal(candidate["_component_context"]["palette"], pal[::-1])

    def test_duplicate_masks_restore_baseline_capacity(self):
        rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        labels = np.zeros((24, 32), dtype=np.int32)
        vis = np.ones_like(labels, dtype=bool)
        pal = np.array([[10, 20, 30]])
        def discovery(*args, max_candidates, **kwargs):
            rows = []
            for n in range(max_candidates):
                mask = np.zeros_like(vis); mask[n, :8] = True
                rows.append({"candidate_id": str(n), "mask": mask, "kind": "pair"})
            return rows
        result = discover_with_sampling_hypothesis(
            rgb, labels, vis, pal, hypothesis={"palette": pal, "lab_all": labels.copy()},
            discovery=discovery, max_candidates=12)
        self.assertEqual(len(result), 12)
        self.assertTrue(all(v["candidate_id"].startswith("baseline:") for v in result))

    def test_broad_alternative_is_not_starved_by_high_ranked_pair_edges(self):
        rgb = np.zeros((60, 32, 3), dtype=np.uint8)
        labels = np.zeros((60, 32), dtype=np.int32)
        alternate = np.ones_like(labels)
        vis = np.ones_like(labels, dtype=bool)
        pal = np.array([[10, 20, 30], [40, 50, 60]])
        def discovery(source, lab, visible, palette, max_candidates, **kwargs):
            rows = []
            for n in range(max_candidates):
                mask = np.zeros_like(vis)
                mask[n if lab is labels else 25 + n, :4] = True
                rows.append({"candidate_id": str(n), "mask": mask, "kind": "pair"})
            if lab is alternate:
                mask = np.zeros_like(vis); mask[40:59, 5:30] = True
                rows[-1] = {"candidate_id": "broad", "mask": mask, "kind": "smooth_field"}
            return rows
        result = discover_with_sampling_hypothesis(
            rgb, labels, vis, pal, hypothesis={"palette": pal, "lab_all": alternate},
            discovery=discovery, max_candidates=12)
        self.assertEqual(len(result), 12)
        self.assertTrue(any(v["kind"] == "smooth_field" for v in result))
        self.assertEqual(len({v["candidate_id"] for v in result}), len(result))

    def test_actual_prefix_heldout_cases_keep_flat_ownership_and_strokes(self):
        from clean_base import _produce_pre_gradient_state
        with tempfile.TemporaryDirectory() as folder:
            for kind in ("mixed_caps", "subtle_gradient", "narrow_gaps", "multicolour"):
                with self.subTest(kind=kind):
                    source = Path(folder) / (kind + ".png")
                    _scene(kind).save(source)
                    kwargs = dict(forced_colors=0, white_threshold=220, background="auto",
                                  max_size=0, strokes="on", checkpoint=lambda *a: None)
                    new = _produce_pre_gradient_state(source, **kwargs)
                    def unchanged(rgb, visible, counter):
                        return visible | counter, {"status": "baseline", "reason": "test_control"}
                    with patch("palette_sampling.independent_sampling_mask", side_effect=unchanged):
                        old = _produce_pre_gradient_state(source, **kwargs)
                    for name in ("den", "palette", "lab_all", "visible", "source_alpha", "vis_fill", "flat"):
                        np.testing.assert_array_equal(new[name], old[name], err_msg=kind + ":" + name)
                    self.assertEqual(new["flat_png_bytes"], old["flat_png_bytes"])
                    self.assertEqual([(v.d, v.color, v.width, v.linecap) for v in new["stroke_list"]],
                                     [(v.d, v.color, v.width, v.linecap) for v in old["stroke_list"]])
                    audit = new["palette_audit"]["independent_gradient_sampling"]
                    self.assertGreater(audit["excluded_pixels"], 100)
                    if audit["status"] == "baseline":
                        self.assertEqual(audit["reason"], "identical_palette_and_labels")
                        self.assertIsNone(new["sampling_hypothesis"])
                    else:
                        self.assertIsNotNone(new["sampling_hypothesis"])

    def test_stage_uses_paired_labels_for_discovery_fit_and_salted_confirmation(self):
        from gradient_reconstruction_stage import propose_gradient_reconstruction
        from tests.test_gradient_reconstruction_stage import _fake_paint, _fast_geometry
        rgb = np.zeros((32, 40, 3), dtype=np.uint8)
        labels = np.zeros((32, 40), dtype=np.int32)
        alternate = np.ones_like(labels)
        mask = np.ones_like(labels, dtype=bool)
        pal = np.array([[10, 20, 30], [40, 50, 60]])
        calls = []
        def provider(*args, **kwargs):
            return [{"candidate_id": "alternate", "kind": "smooth_field", "mask": mask,
                     "_sampling_labels": alternate, "_sampling_palette": pal,
                     "_component_context": {"palette": pal}}]
        def fit(source, owned, **kwargs):
            calls.append(kwargs["label_map"])
            return _fake_paint(source, owned, **kwargs)
        result = propose_gradient_reconstruction(
            rgb, labels, mask, pal, candidate_provider=provider, model_fitter=fit,
            geometry_optimizer=_fast_geometry, max_geometry_candidates=1)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(v is alternate for v in calls))
        self.assertEqual(result["summary"]["independent_paint_revalidation_passed"], 1)
        json.dumps(result, allow_nan=False)

    def test_actual_svg_render_preserves_caps_gaps_and_multicolour_without_gradients(self):
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for kind in ("mixed_caps", "subtle_gradient", "narrow_gaps", "multicolour"):
                with self.subTest(kind=kind):
                    source = root / (kind + ".png"); _scene(kind).save(source)
                    kwargs = dict(strokes="on", gradients="off", geometry="off", max_size=0)
                    new_svg = root / (kind + "-new.svg")
                    old_svg = root / (kind + "-baseline.svg")
                    new_stats = build_clean_base(source, new_svg, **kwargs)
                    def unchanged(rgb, visible, counter):
                        return visible | counter, {"status": "baseline", "reason": "test_control"}
                    with patch("palette_sampling.independent_sampling_mask", side_effect=unchanged):
                        old_stats = build_clean_base(source, old_svg, **kwargs)
                    self.assertEqual(new_svg.read_bytes(), old_svg.read_bytes())
                    for svg in (new_svg, old_svg):
                        render_svg_reference(svg, svg.with_suffix(".png"), width=512, background=None)
                    with Image.open(new_svg.with_suffix(".png")) as one, Image.open(old_svg.with_suffix(".png")) as two:
                        np.testing.assert_array_equal(np.asarray(one), np.asarray(two))

    def test_same_mask_different_labels_do_not_share_default_paint_fit_cache(self):
        import gradient_reconstruction_stage as stage
        from tests.test_gradient_reconstruction_stage import _fake_paint, _fast_geometry
        rgb = np.zeros((32, 40, 3), dtype=np.uint8)
        labels = np.zeros((32, 40), dtype=np.int32)
        alternate = np.ones_like(labels)
        mask = np.ones_like(labels, dtype=bool)
        pal = np.array([[10, 20, 30], [40, 50, 60]])
        calls = []
        def provider(*args, **kwargs):
            return [{"candidate_id": str(index), "kind": "pair", "mask": mask,
                     "_sampling_labels": lab, "_sampling_palette": pal}
                    for index, lab in enumerate((labels, alternate))]
        def fit(source, owned, **kwargs):
            calls.append(int(kwargs["label_map"][0, 0]))
            return _fake_paint(source, owned, **kwargs)
        with patch.object(stage, "fit_gradient_object_proposal", fit):
            stage.propose_gradient_reconstruction(
                rgb, labels, mask, pal, candidate_provider=provider, model_fitter=fit,
                geometry_optimizer=_fast_geometry, max_geometry_candidates=1)
        self.assertCountEqual(calls, [0, 1])


if __name__ == "__main__":
    unittest.main()
