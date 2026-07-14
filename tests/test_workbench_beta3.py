from __future__ import annotations

import http.client
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workbench  # noqa: E402


def _write_result(output: Path, name: str, report: dict, *, recolor: bool = False,
                  images: bool = False) -> Path:
    directory = output / f"result_{name}"
    directory.mkdir(parents=True)
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (directory / f"{name}_vector.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    if recolor:
        (directory / "色彩調整.html").write_text(
            "<!doctype html><title>recolour</title>", encoding="utf-8")
    if images:
        # build_blind_test only needs the paths to exist because data_url is
        # replaced by a deterministic stub in the test.
        (directory / "source_reference.png").write_bytes(b"source")
        (directory / f"{name}_preview.png").write_bytes(b"preview")
    return directory


def _base_report(name: str) -> dict:
    return {
        "input": f"{name}.png",
        "source_match_percent": 98.0,
        "foreground_match_percent": 97.0,
        "paths": 12,
        "native_primitives": 8,
        "native_circles": 2,
        "native_rectangles": 1,
        "native_ellipses": 1,
        "native_lines": 2,
        "native_polylines": 1,
        "native_polygons": 1,
        "strokes": 4,
        "gradients": 2,
        "nodes_total": 90,
        "hotspots": [],
        "preview_is_svg_render": True,
        "options": {"background": "auto", "geometry": "conservative"},
        "acceptance_status": "accepted",
    }


class WorkbenchBeta3Tests(unittest.TestCase):
    def _patch_output(self, output: Path):
        return mock.patch.multiple(
            workbench,
            OUTPUT_DIR=output,
            HISTORY_DIR=output / "_history",
        )

    def test_beta3_report_fields_are_exposed_and_recolor_needs_real_file(self):
        report = _base_report("new")
        report.update({
            "editability_schema": "ai-vector-cleanroom.editability/v2",
            "automation_readiness": {"score": 84.6, "status": "strong"},
            "redraw_complexity": {"ease_score": 64.0, "level": "high"},
            "human_validation": {"status": "not_performed"},
            "scene": {
                "status": "applied",
                "actual_dom_group_count": 27,
                "manifest_only_group_count": 8,
            },
            "paint": {
                "status": "applied",
                "manifest_file": "new_paint_roles.json",
                "resource_counts": {
                    "role_controls": 4,
                    "paint_resources_total": 11,
                },
                "roles": [{"id": "accent-1"}],
            },
            "designer_operations": {
                "acceptance_scope": "generic_machine_detectable_structural_handles",
                "semantic_task_validation": "not_performed",
                "timed_human_editing_validation": "not_performed",
                "human_acceptance": "not_tested",
                "summary": {
                    "total_operations": 5,
                    "passed": 5,
                    "partial": 0,
                    "failed": 0,
                    "manual_review": 0,
                    "automatable": 5,
                },
                "passed": ["a", "b", "c", "d", "e"],
                "partial": [],
                "failed": [],
                "manual_review": [],
                "automatable": ["a", "b", "c", "d", "e"],
            },
        })
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            directory = _write_result(output, "new", report, recolor=True)
            with self._patch_output(output):
                item = workbench._list_results()[0]

            self.assertEqual(item["scene"], {
                "status": "applied",
                "actual_dom_group_count": 27,
                "manifest_only_group_count": 8,
            })
            self.assertEqual(item["paint"], {
                "status": "applied",
                "role_controls": 4,
                "paint_resources_total": 11,
                "manifest_file": "new_paint_roles.json",
            })
            self.assertEqual(item["designer_operations"], {
                "status": "passed",
                "acceptance_scope": "generic_machine_detectable_structural_handles",
                "semantic_task_validation": "not_performed",
                "timed_human_editing_validation": "not_performed",
                "human_acceptance": "not_tested",
                "total_operations": 5,
                "passed": 5,
                "partial": 0,
                "failed": 0,
                "manual_review": 0,
                "automatable": 5,
            })
            self.assertEqual(item["recolor"], "result_new/色彩調整.html")
            self.assertEqual(item["automation_readiness_score"], 84.6)
            self.assertEqual(item["redraw_ease_score"], 64.0)
            self.assertEqual(item["human_validation_status"], "not_performed")
            self.assertEqual(item["native_primitives"], 8)
            self.assertEqual(item["native_lines"], 2)
            self.assertEqual(item["native_polylines"], 1)

            (directory / "色彩調整.html").unlink()
            with self._patch_output(output):
                without_file = workbench._list_results()[0]
            self.assertEqual(without_file["recolor"], "")

    def test_legacy_beta2_report_remains_compatible(self):
        report = _base_report("legacy")
        for key in (
                "native_circles", "native_rectangles", "native_ellipses",
                "native_lines", "native_polylines", "native_polygons"):
            report.pop(key)
        report["native_primitives"] = 3
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _write_result(output, "legacy", report)
            with self._patch_output(output):
                item = workbench._list_results()[0]

        self.assertEqual(item["native_primitives"], 3)
        self.assertEqual(item["native_circles"], 3)
        self.assertEqual(item["scene"]["status"], "not_audited")
        self.assertIsNone(item["scene"]["actual_dom_group_count"])
        self.assertEqual(item["paint"]["status"], "not_audited")
        self.assertIsNone(item["paint"]["role_controls"])
        self.assertEqual(item["designer_operations"]["status"], "not_audited")
        self.assertEqual(
            item["designer_operations"]["human_acceptance"], "not_audited")
        self.assertIsNone(item["designer_operations"]["total_operations"])
        self.assertEqual(item["recolor"], "")

    def test_fallback_audit_counts_and_intermediate_scene_layout_are_supported(self):
        report = _base_report("fallback")
        report.update({
            "editability_enhancements": {
                "stages": {
                    "scene_graph": {
                        "status": "no_change",
                        "actual_dom_group_count": 0,
                        "manifest_only_group_count": 3,
                    },
                },
            },
            "paint_roles": {
                "status": "applied",
                "roles": [{"id": "one"}, {"id": "two"}],
                "resource_counts": {"paint_resources_total": "7"},
            },
            # This is the fail-safe shape used when the operation audit itself
            # cannot run; unlike the normal result it has scalar top-level counts.
            "designer_operations": {
                "status": "manual_review",
                "passed": 0,
                "partial": 0,
                "failed": 0,
                "manual_review": 5,
                "automatable": 0,
            },
        })
        summary = workbench._beta3_report_summary(report)
        self.assertEqual(summary["scene"]["manifest_only_group_count"], 3)
        self.assertEqual(summary["paint"]["role_controls"], 2)
        self.assertEqual(summary["paint"]["paint_resources_total"], 7)
        self.assertEqual(summary["designer_operations"]["total_operations"], 5)
        self.assertEqual(summary["designer_operations"]["status"], "manual_review")

    def test_blind_payload_and_visible_workbench_version_are_beta3(self):
        report = _base_report("blind")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _write_result(output, "blind", report, images=True)
            with self._patch_output(output), mock.patch.object(
                    workbench.vc, "data_url", return_value="data:image/png;base64,AA=="):
                page = workbench.build_blind_test()
                body = page.read_text(encoding="utf-8")

        self.assertIn("version:'v3-codex-beta.3'", body)
        self.assertNotIn("v3-codex-beta.2", body)
        self.assertIn("v3 Codex Beta.3", workbench.APP_HTML)
        self.assertIn('r.recolor', workbench.APP_HTML)
        self.assertIn('>換色</a>', workbench.APP_HTML)
        self.assertIn('beta3Text(r)', workbench.APP_HTML)
        self.assertIn("通用結構把手", workbench.APP_HTML)
        self.assertIn("真人未驗", workbench.APP_HTML)
        self.assertIn("自動化準備", workbench.APP_HTML)
        self.assertIn("描點收尾", workbench.APP_HTML)
        self.assertIn("Stage 2 實作計時", workbench.APP_HTML)
        self.assertIn("/api/editingtest", workbench.APP_HTML)

    def test_stage2_page_is_generated_from_real_result_files(self):
        report = _base_report("timed")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _write_result(output, "timed", report, images=True)
            with self._patch_output(output):
                page = workbench.build_editing_test()
                body = page.read_text(encoding="utf-8")
        self.assertIn("timed_vector.svg", body)
        self.assertIn("timed-designer-editing-stage2", body)
        self.assertIn("不計入 80% 省工驗收", body)

    def test_two_uploads_are_both_accepted_and_stay_visible_in_queue(self):
        """Regression: a later waiting file must not look like it vanished."""
        payload = io.BytesIO()
        Image.new("RGBA", (1, 1), (20, 120, 240, 255)).save(
            payload, format="PNG")
        png = payload.getvalue()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local_queue = queue.Queue()
            with mock.patch.multiple(
                    workbench,
                    INPUT_DIR=root / "input",
                    OUTPUT_DIR=root / "output",
                    HISTORY_DIR=root / "output" / "_history",
                    WB_TOKEN="test-token",
                    _queue=local_queue,
                    _jobs=[],
                    _job_sequence=0):
                server = workbench.ThreadingHTTPServer(
                    ("127.0.0.1", 0), workbench.Handler)
                thread = threading.Thread(target=server.serve_forever,
                                          daemon=True)
                thread.start()
                try:
                    for name in ("first.png", "second.png"):
                        connection = http.client.HTTPConnection(
                            "127.0.0.1", server.server_address[1], timeout=3)
                        connection.request(
                            "POST", "/api/upload?name=" + name, body=png,
                            headers={"X-WB-Token": "test-token",
                                     "Content-Length": str(len(png))})
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertIn("job_id", json.loads(
                            response.read().decode("utf-8")))
                        connection.close()

                    self.assertEqual(local_queue.qsize(), 2)
                    self.assertEqual([j["status"] for j in workbench._jobs],
                                     ["queued", "queued"])
                    self.assertEqual([j["name"] for j in workbench._jobs],
                                     ["first.png", "second.png"])
                    self.assertIn("Array.from(e.dataTransfer.files||[])",
                                  workbench.APP_HTML)
                    self.assertIn("j.status==='queued'||j.status==='running'",
                                  workbench.APP_HTML)
                    self.assertIn("// A completed earlier item must appear immediately",
                                  workbench.APP_HTML)
                    self.assertIn("refresh();\n if(busy)", workbench.APP_HTML)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
