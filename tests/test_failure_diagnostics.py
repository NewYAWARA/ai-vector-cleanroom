from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import job_worker  # noqa: E402
import workbench  # noqa: E402


def _write_trace(path: Path, events) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8")


class WorkerFailureDiagnosticTests(unittest.TestCase):
    def test_candidate_summary_normaliser_is_bounded_and_finite(self):
        source = {
            "status": "failed",
            "options": {
                "background": "auto",
                "strokes": "off",
                "colors": 20,
                "unknown": "must not survive",
            },
            "quality_score": float("nan"),
            "selection_score": -1.0,
            "structure_score": float("inf"),
            "visual_gate_status": "rejected",
            "error": "x" * 400,
            "unknown": {"nested": "must not survive"},
        }

        result = job_worker._normalise_candidate_summary(source)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["options"], {
            "background": "auto", "strokes": "off"})
        self.assertNotIn("quality_score", result)
        self.assertEqual(result["selection_score"], -1.0)
        self.assertNotIn("structure_score", result)
        self.assertEqual(result["visual_gate_status"], "rejected")
        self.assertEqual(len(result["error"]), 240)
        self.assertEqual(source["error"], "x" * 400)

    def test_progress_trace_is_one_linear_append_per_event(self):
        stream = mock.Mock()
        stream.write.side_effect = lambda data: len(data)

        for sequence in range(1, 6):
            job_worker._append_progress_trace(stream, {
                "seq": sequence,
                "stage": "candidate_build",
                "elapsed_seconds": float(sequence),
            })

        self.assertEqual(stream.write.call_count, 5)
        rows = [
            json.loads(bytes(call.args[0]).decode("utf-8"))
            for call in stream.write.call_args_list
        ]
        self.assertEqual([row["seq"] for row in rows], [1, 2, 3, 4, 5])

    def test_runtime_fingerprint_is_observed_and_excludes_token(self):
        manifest = {
            "job_id": 9,
            "token": "secret-token-must-not-leak",
            "input_sha256": "A" * 64,
            "out_base": "sample",
            "overrides": {"gradients": "strong"},
            "budget_seconds": 600.0,
            "candidate_cap": 16,
        }
        with mock.patch.object(
                job_worker, "_installed_distribution_versions",
                return_value={"numpy": "observed-version"}), \
                mock.patch.object(
                    job_worker, "_fingerprinted_files",
                    side_effect=[[{"relative_path": "job_worker.py",
                                   "sha256": "SOURCE"}],
                                 [{"relative_path": "lock.txt",
                                   "sha256": "LOCK"}]]):
            payload = job_worker._build_runtime_fingerprint(
                manifest, Path("input.png"), process_tree_guard=True)

        self.assertEqual(payload["python"]["executable"], sys.executable)
        self.assertTrue(payload["process"]["process_tree_guard"])
        self.assertEqual(payload["distributions"]["numpy"],
                         "observed-version")
        self.assertEqual(payload["source_files"][0]["sha256"], "SOURCE")
        self.assertEqual(payload["lock_files"][0]["sha256"], "LOCK")
        self.assertNotIn("secret-token", json.dumps(payload))

    def test_run_manifest_writes_fingerprint_exactly_once_before_input_check(self):
        with tempfile.TemporaryDirectory() as folder:
            staging = Path(folder)
            input_path = staging / "input.png"
            input_path.write_bytes(b"input")
            manifest_path = staging / "manifest.json"
            manifest_path.write_text(json.dumps({
                "schema": job_worker.SCHEMA,
                "token": "a" * 32,
                "job_id": 3,
                "staging_root": str(staging),
                "input_path": str(input_path),
                "input_sha256": "B" * 64,
                "out_base": "sample",
                "overrides": {},
                "candidate_cap": 16,
                "budget_seconds": 30.0,
            }), encoding="utf-8")
            with mock.patch.object(
                    job_worker, "_atomic_json") as atomic, \
                    mock.patch.object(
                        job_worker, "_build_runtime_fingerprint",
                        return_value={"runtime": "observed"}), \
                    mock.patch.object(
                        job_worker, "_sha256", return_value="C" * 64):
                with self.assertRaisesRegex(ValueError, "digest changed"):
                    job_worker.run_manifest(manifest_path)

        fingerprint_calls = [
            call for call in atomic.call_args_list
            if call.args[0].name == "runtime_fingerprint.json"
        ]
        self.assertEqual(len(fingerprint_calls), 1)


class WorkbenchFailureDiagnosticTests(unittest.TestCase):
    def _events(self):
        return [
            {"seq": 1, "elapsed_seconds": 1.0,
             "stage": "candidate_search", "substage": "candidate",
             "candidate_current": 1, "candidate_started": 1,
             "candidate_evaluated": 0,
             "candidate_summary": {
                 "status": "running",
                 "options": {"strokes": "on", "gradients": "strong"},
             }},
            {"seq": 2, "elapsed_seconds": 2.0,
             "stage": "candidate_build", "substage": "main_trace",
             "candidate_started": 1, "candidate_evaluated": 0},
            {"seq": 3, "elapsed_seconds": 5.0,
             "stage": "candidate_search", "substage": "candidate",
             "candidate_current": 1, "candidate_started": 1,
             "candidate_evaluated": 1,
             "candidate_summary": {
                 "status": "ok", "quality_score": 91.5,
                 "selection_score": 88.25, "structure_score": 59.0,
                 "visual_gate_status": "accepted",
                 "options": {"strokes": "on", "gradients": "strong"},
             }},
            {"seq": 4, "elapsed_seconds": 5.5,
             "stage": "candidate_search", "substage": "candidate",
             "candidate_current": 2, "candidate_started": 2,
             "candidate_evaluated": 1,
             "candidate_summary": {
                 "status": "running",
                 "options": {"strokes": "off", "gradients": "strong"},
             }},
            {"seq": 5, "elapsed_seconds": 7.0,
             "stage": "gradient_reconstruction", "substage": "optimise",
             "candidate_started": 2, "candidate_evaluated": 1,
             "gradient_candidate_current": 3,
             "gradient_candidate_total": 8},
        ]

    def test_trace_summary_has_candidate_and_stage_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            staging = Path(folder)
            _write_trace(staging / "progress_trace.ndjson", self._events())
            summary = workbench._summarize_progress_trace(
                staging, {"stall_since_last_progress_seconds": 3.0})

        complete = next(row for row in summary["candidate_timings"]
                        if row["candidate"] == 1)
        incomplete = next(row for row in summary["candidate_timings"]
                          if row["candidate"] == 2)
        self.assertEqual(complete["duration_seconds"], 4.0)
        self.assertEqual(
            complete["candidate_summary"]["quality_score"], 91.5)
        self.assertEqual(
            complete["candidate_summary"]["options"]["gradients"],
            "strong")
        self.assertEqual(incomplete["duration_seconds_lower_bound"], 4.5)
        self.assertFalse(incomplete["complete"])
        self.assertEqual(
            incomplete["candidate_summary"]["options"]["strokes"], "off")
        self.assertEqual(
            summary["active_at_terminal"]["stage"],
            "gradient_reconstruction")
        self.assertEqual(
            summary["active_at_terminal"]["observed_stall_seconds"], 3.0)
        self.assertEqual(
            summary["active_at_terminal"]["candidate_summary"]["status"],
            "running")
        self.assertEqual(
            summary["gradient_candidate_timings"][0][
                "gradient_candidate_current"], 3)
        self.assertGreaterEqual(len(summary["stage_spans"]), 4)

    def test_failed_staging_is_renamed_with_all_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_root = root / ".jobs"
            staging = job_root / "job-4-original"
            staging.mkdir(parents=True)
            (staging / "manifest.json").write_text(json.dumps({
                "input_sha256": "B" * 64,
                "budget_seconds": 30.0,
                "candidate_cap": 16,
            }), encoding="utf-8")
            (staging / "runtime_fingerprint.json").write_text(
                "{}", encoding="utf-8")
            (staging / "gpu_runtime.json").write_text(json.dumps({
                "provider": "wgpu", "device": "Test GPU",
                "parity": "passed",
            }), encoding="utf-8")
            (staging / "stdout.log").write_text("out", encoding="utf-8")
            (staging / "stderr.log").write_text("err", encoding="utf-8")
            _write_trace(staging / "progress_trace.ndjson", self._events())
            runtime = {4: {"stall_since_last_progress_seconds": 2.0}}

            with mock.patch.multiple(
                    workbench, JOB_ROOT=job_root, _job_runtime=runtime):
                destination, error = workbench._preserve_failed_staging(
                    staging, job_id=4, status="timed_out",
                    detail="bounded timeout", img_path=root / "input.png",
                    out_base="sample", overrides={})

            self.assertIsNone(error)
            self.assertFalse(staging.exists())
            self.assertEqual(destination.parent, root / ".failed_jobs")
            for name in (
                    "manifest.json", "runtime_fingerprint.json",
                    "gpu_runtime.json", "progress_trace.ndjson",
                    "stdout.log", "stderr.log",
                    "failure_summary.json"):
                self.assertTrue((destination / name).is_file(), name)
            summary = json.loads((destination / "failure_summary.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(summary["terminal_status"], "timed_out")
            self.assertEqual(summary["timings"]["trace_event_count"], 5)
            self.assertEqual(summary["gpu_runtime"]["device"], "Test GPU")
            self.assertEqual(summary["actual_diagnostic_path"],
                             str(destination))
            self.assertEqual(
                summary["timings"]["candidate_timings"][0]
                ["candidate_summary"]["visual_gate_status"], "accepted")

    def test_archive_rename_failure_keeps_original_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_root = root / ".jobs"
            staging = job_root / "job-5-original"
            staging.mkdir(parents=True)
            (staging / "stderr.log").write_text("keep", encoding="utf-8")
            with mock.patch.multiple(
                    workbench, JOB_ROOT=job_root, _job_runtime={5: {}}), \
                    mock.patch.object(
                        Path, "replace", side_effect=OSError("rename denied")):
                destination, error = workbench._preserve_failed_staging(
                    staging, job_id=5, status="failed", detail="failure",
                    img_path=root / "input.png", out_base="sample",
                    overrides={})

            self.assertEqual(destination, staging)
            self.assertIn("rename denied", error)
            self.assertTrue(staging.is_dir())
            self.assertEqual((staging / "stderr.log").read_text(
                encoding="utf-8"), "keep")
            self.assertTrue((staging / "failure_summary.json").is_file())

    def test_run_one_timeout_exposes_diagnostic_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_root = root / ".jobs"
            image = root / "input.png"
            image.write_bytes(b"input")
            jobs = [{"id": 7, "name": image.name, "status": "queued",
                     "detail": "", "t": "00:00:00"}]
            runtime = {7: {"cancel_requested": False, "process": None,
                           "started_monotonic": None}}

            def fail_with_evidence(_job_id, _image, _out_base, _overrides,
                                   staging_root, _timeout):
                (staging_root / "stderr.log").write_text(
                    "timed out", encoding="utf-8")
                raise workbench._JobTimedOut("bounded timeout")

            with mock.patch.multiple(
                    workbench, JOB_ROOT=job_root, _jobs=jobs,
                    _job_runtime=runtime), \
                    mock.patch.object(
                        workbench, "_execute_job_subprocess",
                        side_effect=fail_with_evidence), \
                    mock.patch.object(
                        workbench, "_cleanup_staging") as cleanup:
                workbench._run_one(
                    image, {}, requested_base="sample", job_id=7)

            diagnostic_path = Path(jobs[0]["diagnostic_path"])
            self.assertEqual(jobs[0]["status"], "timed_out")
            self.assertEqual(diagnostic_path.parent, root / ".failed_jobs")
            self.assertTrue((diagnostic_path / "stderr.log").is_file())
            self.assertTrue(
                (diagnostic_path / "failure_summary.json").is_file())
            cleanup.assert_not_called()

    def test_success_still_cleans_staging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_root = root / ".jobs"
            image = root / "input.png"
            image.write_bytes(b"input")
            jobs = [{"id": 8, "name": image.name, "status": "queued",
                     "detail": "", "t": "00:00:00"}]
            runtime = {8: {"cancel_requested": False, "process": None,
                           "started_monotonic": None}}

            def successful(_job_id, _image, _out_base, _overrides,
                           staging_root, _timeout):
                staging_output = staging_root / "output"
                staging_output.mkdir()
                return {
                    "staging_output": staging_output,
                    "report": {"acceptance_status": "accepted"},
                    "receipt": {"status": "complete"},
                }

            with mock.patch.multiple(
                    workbench, JOB_ROOT=job_root, _jobs=jobs,
                    _job_runtime=runtime), \
                    mock.patch.object(
                        workbench, "_execute_job_subprocess",
                        side_effect=successful), \
                    mock.patch.object(
                        workbench, "_commit_staged_result_locked",
                        return_value=None):
                workbench._run_one(
                    image, {}, requested_base="sample", job_id=8)

            self.assertEqual(jobs[0]["status"], "done")
            self.assertFalse(Path(runtime[8]["staging_root"]).exists())
            self.assertFalse((root / ".failed_jobs").exists())

    def test_timeout_kill_rereads_progress_before_raising(self):
        class FakeProcess:
            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "input.png"
            image.write_bytes(b"input")
            staging = root / ".jobs" / "job"
            staging.mkdir(parents=True)
            process = FakeProcess()
            jobs = [{"id": 11, "status": "running",
                     "elapsed_seconds": 0.0}]
            runtime = {11: {"cancel_requested": False, "process": None}}
            with mock.patch.multiple(
                    workbench, _jobs=jobs, _job_runtime=runtime,
                    JOB_WORKER=root / "job_worker.py"), \
                    mock.patch.object(
                        workbench.subprocess, "Popen", return_value=process), \
                    mock.patch.object(
                        workbench, "_read_json", return_value=None) as read, \
                    mock.patch.object(
                        workbench, "_publish_progress") as publish, \
                    mock.patch.object(
                        workbench.time, "monotonic",
                        side_effect=[0.0, 31.0]), \
                    mock.patch.object(workbench.time, "sleep"):
                with self.assertRaises(workbench._JobTimedOut):
                    workbench._execute_job_subprocess(
                        11, image, "sample", {}, staging, 30.0)

            self.assertEqual(read.call_count, 2)
            self.assertEqual(publish.call_count, 2)
            self.assertEqual(runtime[11]["termination_mode"], "terminate")
            self.assertEqual(runtime[11]["process_returncode"], -15)


if __name__ == "__main__":
    unittest.main()
