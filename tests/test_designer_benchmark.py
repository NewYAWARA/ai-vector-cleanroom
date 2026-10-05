"""Reference correctness and evidence scope, independent of tracing choices."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from generate_designer_benchmark import fixture_cases, generate, raster_topology, selection_correspondence, svg_structure


class DesignerBenchmarkTests(unittest.TestCase):
    def test_cases_have_independent_truth_and_unique_object_units(self):
        cases = fixture_cases()
        self.assertGreaterEqual(len(cases), 12)
        self.assertEqual(len({c["id"] for c in cases}), len(cases))
        for case in cases:
            with self.subTest(case=case["id"]):
                root = ET.fromstring(case["svg"])
                ids = [node.get("id") for node in root.iter() if node.get("id")]
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(svg_structure(case["svg"])["selection_unit_count"], case["expected_objects"])
                self.assertTrue(case["required_properties"])

    def test_structure_distinguishes_stroke_geometry_and_paint_resources(self):
        structures = {c["id"]: svg_structure(c["svg"]) for c in fixture_cases()}
        self.assertEqual(structures["06_thin_lines"]["stroke_count"], 3)
        self.assertEqual(structures["06_thin_lines"]["stroke_widths"], ["2", "4", "6"])
        self.assertEqual(structures["09_linear_gradient"]["linear_gradient_count"], 1)
        self.assertEqual(structures["10_radial_gradient"]["radial_gradient_count"], 1)
        self.assertEqual(structures["05_compound_holes"]["path_command_counts"]["M"], 4)
        self.assertEqual(structures["05_compound_holes"]["path_count"], 1)

    def test_native_renderer_fixtures_preserve_holes_details_and_true_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = generate(Path(folder), widths=(128,))
            self.assertEqual(manifest["raster_count"], 14)
            cases = {case["id"]: case for case in manifest["cases"]}
            self.assertEqual(cases["05_compound_holes"]["rasters"][0]["topology"]["holes"], 3)
            self.assertEqual(cases["13_intentional_details"]["rasters"][0]["topology"]["foreground_components"], 4)
            image = np.asarray(Image.open(Path(folder) / cases["09_linear_gradient"]["rasters"][0]["path"]).convert("RGB"))
            self.assertGreater(np.linalg.norm(image[64, 30].astype(float) - image[64, 96]), 50)
            self.assertEqual(manifest["human_edit_time"], "not_measured")
            disk = json.loads((Path(folder) / "truth-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(disk["cases"], manifest["cases"])

    def test_identical_pixels_do_not_hide_fused_selection_units(self):
        case = next(c for c in fixture_cases() if c["id"] == "11_touching_colours")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            reference = folder / "reference.svg"
            fused = folder / "fused.svg"
            reference.write_text(case["svg"], encoding="utf-8")
            fused.write_text(case["svg"].replace('</g><g id="object-02">', ""), encoding="utf-8")
            same = selection_correspondence(reference, reference, folder / "same", 128)
            merged = selection_correspondence(reference, fused, folder / "merged", 128)
            self.assertEqual(same["matched_reference_objects"], 2)
            self.assertEqual(merged["matched_reference_objects"], 0)
            self.assertEqual(merged["output_units"], 1)

    def test_white_paint_and_transparent_hole_are_reported_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            image = np.zeros((20, 20, 4), dtype=np.uint8)
            image[3:17, 3:17] = (0, 80, 40, 255)
            image[8:12, 8:12] = (255, 255, 255, 255)
            path = Path(folder) / "white-fill.png"
            Image.fromarray(image).save(path)
            self.assertEqual(raster_topology(path)["holes"], 1)
            self.assertEqual(raster_topology(path, mode="alpha")["holes"], 0)


if __name__ == "__main__":
    unittest.main()
