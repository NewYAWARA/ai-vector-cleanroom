"""Local source evidence must dominate attractive global error averages."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from source_scene_guard import validate_source_scene, validate_source_scene_arrays


INK = (40, 110, 70, 255)


def icon(width=80, height=60):
    image = np.zeros((height, width, 4), np.uint8)
    image[5:height - 5, 5:width - 5] = INK
    return image


def opaque_source(reference):
    alpha = reference[:, :, 3:4].astype(float) / 255
    source = reference.copy()
    source[:, :, :3] = np.rint(reference[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    source[:, :, 3] = 255
    return source


class SourceSceneGuardTests(unittest.TestCase):
    def check(self, before, after, source, reference=None):
        return validate_source_scene_arrays(before, after, source, processed_reference_rgba=reference)

    def check_feature(self, before, after, reference, feature):
        # Deliberately broad scope isolates the necessary feature-cost gate:
        # neither global improvement nor a boundary allowance can bypass it.
        from source_boundary_evidence import _source_feature_regions
        from source_edge_reconstruction import rgba_sha256
        source=opaque_source(reference);allowance=np.ones(feature.shape,bool)
        regions=_source_feature_regions({'thin_ink':feature},feature,2.6)
        evidence={'verified':True,'allowance_sha256':hashlib.sha256(allowance.tobytes()).hexdigest(),
            'render_binding':{'before':rgba_sha256(before),'after':rgba_sha256(after)},
            'source_rgba_sha256':rgba_sha256(source),'processed_rgba_sha256':rgba_sha256(reference),
            'source_feature_regions':regions}
        return validate_source_scene_arrays(before,after,source,processed_reference_rgba=reference,
            _boundary_allowance=allowance,_boundary_evidence=evidence)

    def test_one_pixel_line_growth_and_branch_loss_cannot_spend_large_body_improvement(self):
        reference=np.zeros((64,64,4),np.uint8)
        reference[10:35,10:35]=INK;reference[10:30,50]=INK
        feature=np.zeros((64,64),bool);feature[10:30,50]=True
        before=reference.copy();before[10:35,10:35,:3]=[90,150,110]
        for mutation in ('widen','extend_endpoint','shorten_tiny_branch'):
            after=reference.copy()
            if mutation=='widen':after[10:30,49]=INK
            elif mutation=='extend_endpoint':after[30:32,50]=INK
            else:after[28:30,50]=0
            with self.subTest(mutation=mutation):
                result=self.check_feature(before,after,reference,feature)
                self.assertTrue(result['metrics']['global_color_nonregression'])
                self.assertFalse(result['accepted'])
                self.assertIn('source_feature_color_error_increased',result['reasons'])
                self.assertFalse(result['metrics']['all_source_feature_nonregression'])

    def test_subpixel_stroke_shift_toward_source_is_not_frozen_by_feature_bank(self):
        reference=np.zeros((64,64,4),np.uint8);reference[10:30,50]=INK
        feature=np.zeros((64,64),bool);feature[10:30,50]=True
        before=reference.copy();before[10:30,50,3]=127;before[10:30,49]=INK;before[10:30,49,3]=128
        after=reference.copy();after[10:30,50,3]=191;after[10:30,49]=INK;after[10:30,49,3]=64
        result=self.check_feature(before,after,reference,feature)
        self.assertTrue(result['accepted'],result['reasons'])
        self.assertTrue(result['metrics']['all_source_feature_nonregression'])
        cost=result['metrics']['required_source_feature_costs'][0]
        self.assertLess(cost['after'],cost['before'])

    def test_false_hole_can_be_filled_when_original_supports_ink(self):
        source = icon()
        before = source.copy()
        before[20:24, 30:33] = 0
        result = self.check(before, source, source)
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertEqual(len(result["multi_alpha"]), 3)
        self.assertTrue(any(r["kind"] == "filled_source_supported_false_hole"
                            for r in result["source_supported_hole_repairs"]))

    def test_true_hole_cannot_be_filled_by_better_global_mae(self):
        source = icon(400, 300)
        source[120:122, 170:172] = 0
        before = source.copy()
        before[15:100, 15:100, :3] = (70, 130, 80)
        after = source.copy()
        after[120:122, 170:172] = INK
        result = self.check(before, after, source)
        self.assertFalse(result["accepted"])
        self.assertLess(result["metrics"]["white_composite_rgb_mae_after"],
                        result["metrics"]["white_composite_rgb_mae_before"])
        self.assertIn("filled_source_true_hole", result["reasons"])
        self.assertTrue(any(d["bbox_xyxy"] == [170, 120, 172, 122]
                            for d in result["localized_defects"]))

    def test_hole_count_equality_does_not_authorize_hole_relocation(self):
        source = icon()
        source[20:23, 20:23] = 0
        before = source.copy()
        after = icon()
        after[20:23, 40:43] = 0
        result = self.check(before, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("created_hole_on_source_ink", result["reasons"])
        self.assertIn("filled_source_true_hole", result["reasons"])
        self.assertTrue(all(row["holes_before"] == row["holes_after"] == 1
                            for row in result["multi_alpha"]))

    def test_overlapping_true_hole_cannot_be_partly_filled(self):
        source = icon()
        source[20:27, 20:27] = 0
        after = source.copy()
        after[20:27, 20:22] = INK
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("new_paint_on_source_empty", result["reasons"])
        self.assertTrue(all(row["pairs"] for row in result["multi_alpha"]))

    def test_old_open_crack_becoming_new_closed_seven_pixel_hole_is_rejected(self):
        source = icon()
        before = source.copy()
        before[5:27, 30] = 0
        after = source.copy()
        after[20:27, 30] = 0
        result = self.check(before, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("created_hole_on_source_ink", result["reasons"])
        # No newly emptied pixels: a novelty-only alpha mask would miss this.
        self.assertFalse(np.any((before[:, :, 3] > 0) & (after[:, :, 3] == 0)))
        defect = next(d for d in result["localized_defects"]
                      if d["alpha_threshold"] == 128 and d["kind"] == "created_hole_on_source_ink")
        self.assertEqual(defect["pixels"], 7)
        self.assertEqual(defect["bbox_xyxy"], [30, 20, 31, 27])
        self.assertEqual(defect["probes"][0]["source_rgba"], list(INK))

    def test_true_open_channel_cannot_be_closed(self):
        source = icon()
        source[5:35, 30] = 0
        after = source.copy()
        after[15, 30] = INK
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("new_paint_on_source_empty", result["reasons"])

    def test_new_open_channel_on_source_ink_is_rejected_without_holes(self):
        source = icon()
        after = source.copy()
        after[5:35, 30] = 0
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("new_gap_on_source_ink", result["reasons"])
        self.assertTrue(all(row["holes_after"] == 0 for row in result["multi_alpha"]))

    def test_source_supported_reopening_channel_is_allowed(self):
        source = icon()
        source[5:35, 30] = 0
        before = source.copy()
        before[15, 30] = INK
        result = self.check(before, source, source)
        self.assertTrue(result["accepted"], result["reasons"])
        self.assertTrue(any(row["erased"] for row in result["multi_alpha"]))

    def test_white_object_on_transparent_source_cannot_disappear(self):
        source = icon()
        source[20:30, 30:40] = (255, 255, 255, 255)
        after = source.copy()
        after[20:30, 30:40] = 0
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["metrics"]["white_composite_rgb_mae_after"], 0)
        self.assertIn("new_gap_on_source_ink", result["reasons"])

    def test_opaque_white_object_requires_and_is_protected_by_reference(self):
        reference = icon()
        reference[20:30, 30:40] = (255, 255, 255, 255)
        source = opaque_source(reference)
        after = reference.copy()
        after[20:30, 30:40] = 0
        result = self.check(reference, after, source, reference)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["source_support"]["kind"], "opaque_paper_with_processed_reference")
        self.assertIn("new_gap_on_source_ink", result["reasons"])

    def test_opaque_white_is_ambiguous_without_processed_reference(self):
        reference = icon()
        reference[20:30, 30:40] = (255, 255, 255, 255)
        source = opaque_source(reference)
        after = reference.copy()
        after[20:30, 30:40] = 0
        result = self.check(reference, after, source)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["source_support"]["kind"], "opaque_ambiguous")
        self.assertIn("coverage_change_source_ambiguous", result["reasons"])

    def test_nonpaper_border_does_not_authorize_white_removal(self):
        reference = icon()
        source = opaque_source(reference)
        source[0] = (90, 40, 20, 255)
        after = reference.copy()
        after[20:22, 20:22] = 0
        result = self.check(reference, after, source, reference)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["source_support"]["confident"])

    def test_new_opaque_white_fill_on_ink_is_caught_without_alpha_change(self):
        source = icon()
        after = source.copy()
        after[20:22, 30:32] = (255, 255, 255, 255)
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("new_white_exposure_on_source_ink", result["reasons"])

    def test_lighter_paint_on_empty_source_is_caught_by_alpha_not_rgb_threshold(self):
        source = icon()
        after = source.copy()
        after[1:3, 20:24] = (252, 252, 252, 255)
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("new_paint_on_source_empty", result["reasons"])

    def test_multiple_alpha_thresholds_catch_partial_opacity_gap(self):
        source = icon()
        after = source.copy()
        after[20:24, 30, 3] = 160
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["multi_alpha"][0]["created"])
        self.assertFalse(result["multi_alpha"][1]["created"])
        self.assertTrue(result["multi_alpha"][2]["created"])

    def test_source_supported_alpha_edge_improvement_is_not_blanket_rejected(self):
        source = icon()
        source[5:55, 5, 3] = 100
        before = source.copy()
        before[5:55, 5, 3] = 150
        result = self.check(before, source, source)
        self.assertTrue(result["accepted"], result["reasons"])

    def test_opaque_edge_improvement_with_real_source_agreement_is_allowed(self):
        reference = icon()
        reference[5:55, 5] = 0
        source = opaque_source(reference)
        # The independently processed edge was binary; original has partial AA.
        source[5:55, 5, :3] = (171, 198, 182)
        before = reference.copy()
        after = reference.copy()
        after[5:55, 5] = (40, 110, 70, 100)
        result = self.check(before, after, source, reference)
        self.assertTrue(result["accepted"], result["reasons"])

    def test_identical_scene_is_allowed_without_inventing_source_confidence(self):
        source = np.full((20, 30, 4), 255, np.uint8)
        result = self.check(source, source, source)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["source_support"]["confident"])

    def test_dimensions_and_invalid_array_fail_closed(self):
        source = icon()
        self.assertFalse(self.check(source[:, :-1], source, source)["accepted"])
        self.assertFalse(self.check(source.astype(float), source, source)["accepted"])
        self.assertFalse(self.check(source, source, source, source[:, :-1])["accepted"])

    def test_mild_uniform_colour_worsening_cannot_escape_severe_error_threshold(self):
        source = icon()
        after = source.copy()
        after[5:-5, 5:-5, :3] += 5
        result = self.check(source, after, source)
        self.assertFalse(result["accepted"])
        self.assertIn("source_color_error_increased", result["reasons"])
        self.assertNotIn("new_local_color_error", result["reasons"])
        self.assertFalse(result["metrics"]["global_color_nonregression"])

    def test_local_baseline_cannot_be_compensated_by_distant_improvement(self):
        source = icon(100, 100)
        before = source.copy()
        before[50:90, 50:90, :3] += 15
        after = source.copy()
        after[10:20, 10:20, :3] += 5
        result = validate_source_scene_arrays(before, after, source, roi_xyxy=[10, 10, 20, 20])
        self.assertFalse(result["accepted"])
        self.assertTrue(result["metrics"]["global_color_nonregression"])
        self.assertFalse(result["metrics"]["roi_color_nonregression"])
        self.assertEqual(result["metrics"]["roi_rgb_mae_before"], 0)
        self.assertEqual(result["metrics"]["roi_rgb_mae_after"], 5)
        self.assertIn("source_color_error_increased", result["reasons"])

    def test_invalid_local_roi_fails_closed(self):
        source = icon()
        for roi in ([-1, 0, 5, 5], [0, 0, 500, 5], [0, 0, 0, 5], [0., 0, 5, 5]):
            result = validate_source_scene_arrays(source, source, source, roi_xyxy=roi)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["status"], "unverified_fail_closed")

    def test_native_nonsquare_render_keeps_dimensions_and_checks_svg_aspect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = icon(96, 64)
            Image.fromarray(source).save(root / "source.png")
            svg = '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="64"><path fill="#286e46" d="M5 5H91V59H5Z"/></svg>'
            (root / "a.svg").write_text(svg, encoding="utf-8")
            result = validate_source_scene(root / "a.svg", root / "a.svg", root / "source.png")
            self.assertTrue(result["accepted"], result["reasons"])
            self.assertEqual(result["provenance"]["canvas"], [96, 64])
            (root / "bad.svg").write_text(svg.replace('height="64"', 'height="96"'), encoding="utf-8")
            bad = validate_source_scene(root / "a.svg", root / "bad.svg", root / "source.png")
            self.assertFalse(bad["accepted"])
            self.assertIn("source_scene_svg_aspect_ratio_differs", bad["reasons"])

    def test_render_failure_and_pixel_budget_never_authorize_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.png"
            Image.fromarray(icon()).save(path)
            with patch("source_scene_guard._render_native", side_effect=RuntimeError("renderer unavailable")):
                result = validate_source_scene("before.svg", "after.svg", path)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["status"], "unverified_fail_closed")
        with patch("source_scene_guard.MAX_SCENE_PIXELS", 10):
            result = self.check(icon(), icon(), icon())
        self.assertFalse(result["accepted"])
        self.assertIn("source_scene_native_pixel_budget_exceeded", result["reasons"])

    def test_integer_rounded_downsize_uses_actual_native_canvas_without_stretching_source(self):
        from source_scene_guard import _render_native_payload
        from svg_renderer import native_canvas_aspect_compatible
        for dims in ((1024, 341, 1254, 418), (341, 1024, 418, 1254), (64, 43, 96, 64)):
            self.assertTrue(native_canvas_aspect_compatible(*dims))
        for dims in ((1024, 342, 1254, 418), (64, 44, 96, 64), (96, 63, 96, 64),
                     (64.1, 43, 96, 64), (1, 1, 0, 3)):
            self.assertFalse(native_canvas_aspect_compatible(*dims))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="43" viewBox="0 0 64 43"><rect x="10" y="10" width="40" height="20" fill="#286e46"/></svg>'
            baseline = _render_native_payload(good.encode(), 96, 64)
            self.assertEqual(baseline.shape, (64, 96, 4))
            Image.fromarray(baseline).save(root/'source.png')
            (root/'a.svg').write_text(good, encoding='utf8')
            result = validate_source_scene(root/'a.svg', root/'a.svg', root/'source.png')
            self.assertTrue(result['accepted'], result['reasons'])
            (root/'b.svg').write_text(good.replace('width="40"', 'width="39"'), encoding='utf8')
            result = validate_source_scene(root/'a.svg', root/'b.svg', root/'source.png')
            self.assertFalse(result['accepted'])

    def test_claude_d2e_native_seven_pixel_gap_regression_when_fixture_available(self):
        # Optional external forensic fixture; synthetic tests above remain part
        # of the source release and require no user image. No conversion rerun.
        workspace = Path(__file__).resolve().parents[4]
        tea = workspace / "outputs/designer-verification-v2/release-tea-r4/result_tea"
        candidate = workspace / "work/cr-1005101832/exp/e2d/scene_D2E.svg"
        if not candidate.exists() or not (tea / "source_original.png").exists():
            self.skipTest("External Claude D2E forensic fixture is not packaged")
        result = validate_source_scene(tea / "tea_vector.svg", candidate, tea / "source_original.png",
                                       processed_reference_png=tea / "source_reference.png")
        self.assertFalse(result["accepted"])
        self.assertEqual((result["width"], result["height"]), (1254, 1254))
        self.assertTrue(any(d["kind"] == "created_hole_on_source_ink" and d["alpha_threshold"] == 128
                            and d["pixels"] == 7 and d["bbox_xyxy"] == [445, 714, 447, 719]
                            for d in result["localized_defects"]))
        self.assertTrue(result["source_supported_hole_repairs"])
        self.assertLess(result["metrics"]["white_composite_rgb_mae_after"],
                        result["metrics"]["white_composite_rgb_mae_before"])


if __name__ == "__main__":
    unittest.main()
