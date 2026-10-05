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
        "designer_anchors_total": 104,
        "hotspots": [],
        "preview_is_svg_render": True,
        "options": {"background": "auto", "geometry": "conservative",
                    "curve_error_percent": 0.25},
        "acceptance_status": "accepted",
    }


class WorkbenchBeta5Tests(unittest.TestCase):
    def _patch_output(self, output: Path):
        return mock.patch.multiple(
            workbench,
            OUTPUT_DIR=output,
            HISTORY_DIR=output / "_history",
        )

    def test_job_timeout_default_and_override_are_bounded(self):
        with mock.patch.dict(workbench.os.environ, {}, clear=True):
            self.assertEqual(workbench._job_timeout_seconds(), 1200.0)
        for invalid in ("", "invalid", "nan", "inf", "-inf"):
            with self.subTest(invalid=invalid), mock.patch.dict(
                    workbench.os.environ,
                    {"AVC_JOB_TIMEOUT_SECONDS": invalid}, clear=True):
                self.assertEqual(workbench._job_timeout_seconds(), 1200.0)
        with mock.patch.dict(workbench.os.environ,
                             {"AVC_JOB_TIMEOUT_SECONDS": "900"}, clear=True):
            self.assertEqual(workbench._job_timeout_seconds(), 900.0)
        with mock.patch.dict(
                workbench.os.environ,
                {"AVC_JOB_TIMEOUT_SECONDS": "1"}, clear=True):
            self.assertEqual(workbench._job_timeout_seconds(), 30.0)
        with mock.patch.dict(
                workbench.os.environ,
                {"AVC_JOB_TIMEOUT_SECONDS": "99999"}, clear=True):
            self.assertEqual(workbench._job_timeout_seconds(), 1800.0)

    def test_current_result_exposes_zip_and_explicit_download_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _write_result(output, "download", _base_report("download"))
            (output / "result_download.zip").write_bytes(b"zip")
            with self._patch_output(output):
                item = workbench._list_results()[0]

        self.assertEqual(item["zip"], "result_download.zip")
        self.assertIn("function outputUrl", workbench.APP_HTML)
        self.assertIn("split('/').map(encodeURIComponent)", workbench.APP_HTML)
        self.assertIn("開啟校稿", workbench.APP_HTML)
        self.assertIn("下載 SVG", workbench.APP_HTML)
        self.assertIn("下載完整 ZIP", workbench.APP_HTML)

    def test_default_io_uses_external_data_root_and_bounds_upload_names(self):
        self.assertEqual(workbench.INPUT_DIR, workbench.DATA_DIR / "input")
        self.assertEqual(workbench.OUTPUT_DIR, workbench.DATA_DIR / "output")
        self.assertNotEqual(workbench.DATA_DIR, workbench.BASE)
        safe = workbench._safe_name("很長的圖片名稱" * 30 + ".png")
        self.assertTrue(safe.endswith(".png"))
        self.assertRegex(Path(safe).stem, r"_[0-9a-f]{12}$")

        long_stem = "同名而且非常長" * 20
        paths = [Path(long_stem + ".png"), Path(long_stem + ".jpg")]
        planned = workbench.vc.plan_output_names(paths)
        self.assertEqual(len(set(planned.values())), 2)
        self.assertTrue(all(len(value) <= 48 for value in planned.values()))

    def test_report_features_are_exposed_and_recolor_needs_real_file(self):
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
            self.assertEqual(item["nodes"], 90)
            self.assertEqual(item["designer_anchors"], 104)
            self.assertEqual(
                item["designer_anchor_source"], "canonical_svg_geometry")

            (directory / "色彩調整.html").unlink()
            with self._patch_output(output):
                without_file = workbench._list_results()[0]
            self.assertEqual(without_file["recolor"], "")

    def test_legacy_beta2_report_remains_compatible(self):
        report = _base_report("legacy")
        report.pop("designer_anchors_total")
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
        self.assertEqual(item["designer_anchors"], 90)
        self.assertEqual(item["designer_anchor_source"], "legacy_nodes_total")

    def test_rejected_result_is_visible_but_never_presented_as_done(self):
        report = _base_report("rejected")
        report.update({
            "acceptance_status": "rejected",
            "output_base": "rejected",
            "visual_acceptance_status": "rejected",
            "manual_review_required": True,
            "visual_gate": {"reasons": ["多項失守：顏色、局部低分區"]},
        })
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            image = input_dir / "rejected.png"
            Image.new("RGB", (4, 4), "white").save(image)

            def fake_execute(_job_id, _image, out_base, _overrides,
                             staging_root, _timeout):
                staging_output = staging_root / "output"
                _write_result(staging_output, out_base, report)
                (staging_output / f"result_{out_base}.zip").write_bytes(b"zip")
                return {"staging_output": staging_output, "report": report,
                        "receipt": {"status": "complete"}}

            jobs = [{"id": 1, "name": image.name, "status": "queued",
                     "detail": "", "t": "00:00:00"}]
            with mock.patch.multiple(
                    workbench, INPUT_DIR=input_dir, OUTPUT_DIR=output,
                    HISTORY_DIR=output / "_history", JOB_ROOT=root / ".jobs",
                    _jobs=jobs,
                    _job_runtime={1: {"cancel_requested": False,
                                      "process": None,
                                      "started_monotonic": None}}), \
                    mock.patch.object(workbench, "_execute_job_subprocess",
                                      side_effect=fake_execute):
                workbench._run_one(image, {}, requested_base="rejected",
                                   job_id=1)
                listed = workbench._list_results()[0]

            self.assertEqual(jobs[0]["status"], "rejected")
            self.assertIn("尚需人工修整", jobs[0]["detail"])
            self.assertTrue(listed["manual_review_required"])
            self.assertEqual(listed["visual_acceptance_status"], "rejected")
            self.assertIn("外觀未達標", workbench.APP_HTML)
            self.assertIn("完成但需檢查", workbench.APP_HTML)

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
        summary = workbench._report_feature_summary(report)
        self.assertEqual(summary["scene"]["manifest_only_group_count"], 3)
        self.assertEqual(summary["paint"]["role_controls"], 2)
        self.assertEqual(summary["paint"]["paint_resources_total"], 7)
        self.assertEqual(summary["designer_operations"]["total_operations"], 5)
        self.assertEqual(summary["designer_operations"]["status"], "manual_review")

    def test_blind_payload_and_visible_workbench_version_are_beta5(self):
        report = _base_report("blind")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            _write_result(output, "blind", report, images=True)
            with self._patch_output(output), mock.patch.object(
                    workbench.vc, "data_url", return_value="data:image/png;base64,AA=="):
                page = workbench.build_blind_test()
                body = page.read_text(encoding="utf-8")

        self.assertEqual(workbench.vc.TOOL_VERSION, "v0.6.0-alpha")
        self.assertIn(
            f"version:'{workbench.vc.TOOL_VERSION}'", body)
        self.assertNotIn("v3-codex-beta.3", body)
        self.assertNotIn("v3-codex-beta.4", body)
        self.assertIn(workbench.vc.TOOL_VERSION, workbench.APP_HTML)
        self.assertIn('r.recolor', workbench.APP_HTML)
        self.assertIn('>換色</a>', workbench.APP_HTML)
        self.assertIn('featureText(r)', workbench.APP_HTML)
        self.assertIn("通用結構把手", workbench.APP_HTML)
        self.assertIn("真人未驗", workbench.APP_HTML)
        self.assertIn("自動化準備", workbench.APP_HTML)
        self.assertIn("描點收尾", workbench.APP_HTML)
        self.assertIn("Stage 2 實作計時", workbench.APP_HTML)
        self.assertIn("/api/editingtest", workbench.APP_HTML)
        self.assertIn('data-k="curve_error_percent"', workbench.APP_HTML)
        self.assertIn("先限制幾何誤差，再最小化錨點", workbench.APP_HTML)
        self.assertIn("/api/cancel", workbench.APP_HTML)
        self.assertIn("timed_out:'已逾時'", workbench.APP_HTML)

    def test_timed_out_rerun_never_touches_previous_success(self):
        report = _base_report("stable")
        report["output_base"] = "stable"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_dir = root / "input"
            output = root / "output"
            input_dir.mkdir()
            image = input_dir / "stable.png"
            Image.new("RGB", (2, 2), "white").save(image)
            old_dir = _write_result(output, "stable", report)
            marker = old_dir / "old-success.txt"
            marker.write_text("keep me", encoding="utf-8")
            old_zip = output / "result_stable.zip"
            old_zip.write_bytes(b"old zip")
            jobs = [{"id": 7, "name": image.name, "status": "queued",
                     "detail": "", "t": "00:00:00"}]
            runtime = {7: {"cancel_requested": False, "process": None,
                           "started_monotonic": None}}
            with mock.patch.multiple(
                    workbench, INPUT_DIR=input_dir, OUTPUT_DIR=output,
                    HISTORY_DIR=output / "_history", JOB_ROOT=root / ".jobs",
                    _jobs=jobs, _job_runtime=runtime), \
                    mock.patch.object(
                        workbench, "_execute_job_subprocess",
                        side_effect=workbench._JobTimedOut("bounded timeout")):
                workbench._run_one(image, {}, requested_base="stable",
                                   job_id=7)

            self.assertEqual(jobs[0]["status"], "timed_out")
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")
            self.assertEqual(old_zip.read_bytes(), b"old zip")
            self.assertFalse((output / "_history").exists())

    def test_transactional_commit_archives_old_result_only_after_success(self):
        old_report = _base_report("swap")
        old_report["output_base"] = "swap"
        new_report = dict(old_report, source_match_percent=99.5)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "output"
            staging_output = root / ".jobs" / "one" / "output"
            old_dir = _write_result(output, "swap", old_report)
            (old_dir / "old-only.txt").write_text("old", encoding="utf-8")
            (output / "result_swap.zip").write_bytes(b"old zip")
            _write_result(staging_output, "swap", new_report)
            (staging_output / "result_swap.zip").write_bytes(b"new zip")

            with mock.patch.multiple(
                    workbench, OUTPUT_DIR=output,
                    HISTORY_DIR=output / "_history"):
                archive = workbench._commit_staged_result(
                    staging_output, "swap")

            committed = json.loads((output / "result_swap" / "report.json")
                                   .read_text(encoding="utf-8"))
            self.assertEqual(committed["source_match_percent"], 99.5)
            self.assertEqual((output / "result_swap.zip").read_bytes(),
                             b"new zip")
            self.assertFalse((output / "result_swap" / "old-only.txt").exists())
            self.assertTrue(archive.is_file())

    def test_queued_job_can_be_cancelled_before_worker_start(self):
        jobs = [{"id": 9, "name": "queued.png", "status": "queued",
                 "detail": "", "t": "00:00:00", "cancellable": True}]
        runtime = {9: {"cancel_requested": False, "process": None,
                       "started_monotonic": None}}
        with mock.patch.multiple(workbench, _jobs=jobs, _job_runtime=runtime):
            self.assertEqual(workbench._cancel_job(9), "cancelled")
            self.assertTrue(runtime[9]["cancel_requested"])
            self.assertEqual(jobs[0]["status"], "cancelled")

    def test_supervisor_hard_timeout_terminates_worker_process(self):
        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.terminated = False

            def poll(self):
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "timeout.png"
            Image.new("RGB", (1, 1), "white").save(image)
            staging = root / ".jobs" / "timeout"
            staging.mkdir(parents=True)
            process = FakeProcess()
            runtime = {3: {"cancel_requested": False, "process": None,
                           "started_monotonic": None}}
            jobs = [{"id": 3, "name": image.name, "status": "running",
                     "elapsed_seconds": 0.0}]
            with mock.patch.multiple(
                    workbench, _job_runtime=runtime, _jobs=jobs,
                    JOB_WORKER=root / "job_worker.py"), \
                    mock.patch.object(workbench.subprocess, "Popen",
                                      return_value=process), \
                    mock.patch.object(workbench.time, "monotonic",
                                      side_effect=[0.0, 31.0]), \
                    mock.patch.object(workbench.time, "sleep"):
                with self.assertRaises(workbench._JobTimedOut):
                    workbench._execute_job_subprocess(
                        3, image, "timeout", {}, staging, 30.0)
            self.assertTrue(process.terminated)
            self.assertEqual(jobs[0]["elapsed_seconds"], 31.0)

    def test_tampered_completion_receipt_cannot_be_committed(self):
        report = _base_report("receipt")
        report["output_base"] = "receipt"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "receipt.png"
            Image.new("RGB", (1, 1), "white").save(image)
            staging = root / "job"
            output = staging / "output"
            _write_result(output, "receipt", report)
            zip_path = output / "result_receipt.zip"
            zip_path.write_bytes(b"verified")
            input_sha = workbench._sha256(image)
            token = "a" * 32
            receipt = {
                "schema": workbench.JOB_SCHEMA, "token": token,
                "job_id": 4, "status": "complete",
                "input_name": image.name, "input_sha256": input_sha,
                "out_base": "receipt", "acceptance_status": "accepted",
                "report_sha256": workbench._sha256(
                    output / "result_receipt" / "report.json"),
                "zip_sha256": workbench._sha256(zip_path),
            }
            (staging / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8")
            zip_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "雜湊"):
                workbench._validate_receipt(
                    staging, token, 4, image, input_sha, "receipt")

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
        self.assertIn("空白、0、估算、未完成與失敗", body)

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
                    _job_runtime={},
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
