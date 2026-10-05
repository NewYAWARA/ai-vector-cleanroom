import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import job_worker


class JobWorkerAtomicProgressTests(unittest.TestCase):
    def _job_api(self, *, set_ok=True, assign_ok=True):
        api = mock.MagicMock()
        api.CreateJobObjectW.return_value = 123
        api.SetInformationJobObject.return_value = set_ok
        api.AssignProcessToJobObject.return_value = assign_ok
        api.GetCurrentProcess.return_value = 456
        return api

    def test_process_tree_guard_sets_owner_only_after_full_success(self):
        api = self._job_api()
        with mock.patch.object(job_worker.os, "name", "nt"), \
                mock.patch("ctypes.WinDLL", return_value=api), \
                mock.patch.object(job_worker, "_PROCESS_TREE_JOB_HANDLE", None), \
                mock.patch.dict(os.environ, {}, clear=False):
            self.assertTrue(job_worker._install_windows_process_tree_guard())
            self.assertEqual(
                os.environ.get("AVC_GRADIENT_PROCESS_POOL_SAFE"), "1")
            self.assertEqual(
                os.environ.get("AVC_GRADIENT_PROCESS_POOL_OWNER_PID"),
                str(os.getpid()))
            api.CloseHandle.assert_not_called()

    def test_process_tree_guard_closes_unassigned_handle_and_stays_serial(self):
        for set_ok, assign_ok in ((False, True), (True, False)):
            api = self._job_api(set_ok=set_ok, assign_ok=assign_ok)
            with self.subTest(set_ok=set_ok, assign_ok=assign_ok), \
                    mock.patch.object(job_worker.os, "name", "nt"), \
                    mock.patch("ctypes.WinDLL", return_value=api), \
                    mock.patch.object(
                        job_worker, "_PROCESS_TREE_JOB_HANDLE", None), \
                    mock.patch.dict(os.environ, {}, clear=False):
                self.assertFalse(
                    job_worker._install_windows_process_tree_guard())
                self.assertEqual(
                    os.environ.get("AVC_GRADIENT_PROCESS_POOL_SAFE"), "0")
                self.assertNotIn(
                    "AVC_GRADIENT_PROCESS_POOL_OWNER_PID", os.environ)
                api.CloseHandle.assert_called_once_with(123)

    def test_main_guard_exception_falls_back_to_serial_geometry(self):
        with mock.patch.object(
                job_worker, "_install_windows_process_tree_guard",
                side_effect=OSError("guard unavailable")), \
                mock.patch.object(job_worker, "run_manifest", return_value=0), \
                mock.patch.dict(os.environ, {
                    "AVC_GRADIENT_PROCESS_POOL_SAFE": "1",
                    "AVC_GRADIENT_PROCESS_POOL_OWNER_PID": "999",
                }, clear=False):
            self.assertEqual(job_worker.main(["manifest.json"]), 0)
            self.assertEqual(
                os.environ.get("AVC_GRADIENT_PROCESS_POOL_SAFE"), "0")
            self.assertNotIn(
                "AVC_GRADIENT_PROCESS_POOL_OWNER_PID", os.environ)

    def test_atomic_json_retries_windows_share_violation_then_commits(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "progress.json"
            real_replace = job_worker.os.replace
            attempts = {"count": 0}

            def flaky_replace(source, destination):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    error = PermissionError("reader holds progress path")
                    error.winerror = 5
                    raise error
                return real_replace(source, destination)

            with mock.patch.object(
                    job_worker.os, "replace", side_effect=flaky_replace), \
                    mock.patch.object(job_worker.time, "sleep"):
                committed = job_worker._atomic_json(
                    target, {"stage": "paint_model_fit"})

            self.assertTrue(committed)
            self.assertEqual(attempts["count"], 2)
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8"))["stage"],
                "paint_model_fit")

    def test_best_effort_progress_drop_never_claims_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "progress.json"
            error = PermissionError("persistent reader race")
            error.winerror = 5
            with mock.patch.object(
                    job_worker.os, "replace", side_effect=error), \
                    mock.patch.object(job_worker.time, "sleep"):
                committed = job_worker._atomic_json(
                    target, {"stage": "candidate_search"}, best_effort=True)

            self.assertFalse(committed)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
