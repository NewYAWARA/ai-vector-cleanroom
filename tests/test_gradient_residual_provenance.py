from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import vtracer

from gradient_residual_provenance import (unchanged_remote_residual_support,
                                         prove_preexisting_residuals,
                                         prove_unchanged_scene_residuals)


class GradientResidualProvenanceTests(unittest.TestCase):
    def test_complete_native_scene_proof_keeps_remote_details_and_rejects_one_lost_dot(self):
        from svg_renderer import render_svg_reference
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            head = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
            dot = '<rect x="4" y="4" width="1" height="1" fill="#147850"/>'
            main = '<rect x="15" y="15" width="40" height="40" fill="{}"/>'
            before = head+dot+main.format('#709070')+'</svg>'
            after = head+dot+main.format('#147850')+'</svg>'
            src_svg = folder/'original.svg'; src_svg.write_text(after, encoding='utf8')
            source = folder/'original.png'
            render_svg_reference(src_svg, source, 64, background='white')
            rgba = np.asarray(Image.open(source).convert('RGBA'))
            residual = np.zeros((64,64), dtype=bool); residual[4,4] = True
            proof = prove_unchanged_scene_residuals(before, after, residual, source, rgba)
            self.assertEqual(proof['status'], 'verified_no_new_residual_damage')
            for row in proof['measurements']:
                self.assertEqual(row['residual_halo_maximum_rgba_difference'], 0)
            with self.assertRaisesRegex(ValueError, 'residual_halo_rgba_changed'):
                prove_unchanged_scene_residuals(before, after.replace(dot,''), residual, source, rgba)

    def test_remote_exact_support_cannot_hide_changed_colour_alpha_or_neighbour(self):
        before = np.full((12, 20, 4), 255, dtype=np.uint8)
        after = before.copy()
        residual = np.zeros((12, 20), dtype=bool)
        residual[4, (3, 8, 14)] = True
        owned = np.zeros_like(residual)
        owned[4, 4] = True
        after[4, 9, 3] = 254
        proven = unchanged_remote_residual_support(before, after, residual, owned)
        self.assertEqual(np.argwhere(proven).tolist(), [[4, 14]])
        after[4, 14, 0] = 254
        self.assertFalse(unchanged_remote_residual_support(before, after, residual, owned).any())

    def test_real_ordinary_trace_proves_old_remote_scrap_not_new_gradient_damage(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = np.full((64, 80, 4), 255, dtype=np.uint8)
            source[20:45, 25:55, :3] = (40, 100, 70)
            source[5, 5, :3] = (40, 100, 70)
            png, svg = folder/"input.png", folder/"trace.svg"
            Image.fromarray(source, "RGBA").save(png)
            vtracer.convert_image_to_svg_py(
                str(png), str(svg), colormode="color", hierarchical="cutout", mode="spline",
                filter_speckle=2, color_precision=8, layer_difference=0, corner_threshold=58,
                length_threshold=5.0, splice_threshold=45, path_precision=6)
            root = ET.parse(svg).getroot()
            ET.SubElement(root, "{http://www.w3.org/2000/svg}rect", {
                "x": "30", "y": "25", "width": "10", "height": "10", "fill": "#ff00ff"})
            residual = np.zeros((64, 80), dtype=bool)
            residual[5, 5] = True
            owned = np.zeros_like(residual)
            owned[20:45, 25:55] = True
            visible = owned | residual
            proven, report = prove_preexisting_residuals(
                ET.tostring(root, encoding="unicode"), source, residual, owned,
                visible=visible, source_rgb=source[:, :, :3])
            self.assertTrue(proven[5, 5])
            self.assertEqual(report["unchanged_residual_pixels"], 1)
            errors = report["source_rgb_mae_on_proven_support"]
            self.assertEqual(errors["before"], errors["after"])
            self.assertGreater(errors["after"], 0)  # admitted old error, not claimed repaired
            ET.SubElement(root, "{http://www.w3.org/2000/svg}rect", {
                "x": "5", "y": "5", "width": "1", "height": "1", "fill": "#000000"})
            proven, _ = prove_preexisting_residuals(
                ET.tostring(root, encoding="unicode"), source, residual, owned,
                visible=visible, source_rgb=source[:, :, :3])
            self.assertFalse(proven.any())


if __name__ == "__main__":
    unittest.main()
