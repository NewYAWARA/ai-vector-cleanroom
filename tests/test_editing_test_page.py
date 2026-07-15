from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from editing_test_page import TASKS, build_editing_test_page


class EditingTestPageTests(unittest.TestCase):
    def test_page_requires_actual_timing_for_saving_metric(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = output / "result_demo"
            result.mkdir()
            svg = result / "demo_vector.svg"
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>',
                           encoding="utf-8")
            (result / "source_reference.png").write_bytes(b"png")
            (result / "report.json").write_text(json.dumps({
                "tool_version": "v3-codex-beta.5",
                "input": "demo.png",
            }), encoding="utf-8")
            path = build_editing_test_page(output, [{
                "dir": "result_demo", "base": "demo", "input": "demo.png",
                "svg": "result_demo/demo_vector.svg",
                "visual_acceptance_status": "accepted",
                "editability_status": "manual_review",
            }], tool_version="v3-codex-beta.5")
            body = path.read_text(encoding="utf-8")

        self.assertIn("Stage 2", body)
        self.assertIn("actual_timed_weighted_saving_percent", body)
        self.assertIn("baselineKind==='actual'", body)
        self.assertIn("product_claim_validated:false", body)
        self.assertIn("multiple designers and representative logos", body)
        self.assertEqual(body.count('data-task="'), len(TASKS))
        self.assertIn("demo_vector.svg", body)

    def test_missing_files_are_skipped_without_broken_case(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            path = build_editing_test_page(output, [{
                "dir": "result_missing", "svg": "result_missing/no.svg",
            }], tool_version="v3-codex-beta.5")
            body = path.read_text(encoding="utf-8")
        self.assertIn("尚無同時具備 SVG", body)
        self.assertNotIn('class="case"', body)

    def test_result_paths_cannot_escape_output_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            output.mkdir()
            outside = Path(folder) / "outside.svg"
            outside.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>',
                               encoding="utf-8")
            path = build_editing_test_page(output, [{
                "dir": "..", "svg": "../outside.svg",
            }], tool_version="v3-codex-beta.5")
            body = path.read_text(encoding="utf-8")
        self.assertIn("尚無同時具備 SVG", body)
        self.assertNotIn(str(outside), body)


if __name__ == "__main__":
    unittest.main()
