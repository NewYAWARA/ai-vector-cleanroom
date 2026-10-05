# -*- coding: utf-8 -*-
from __future__ import annotations

import contextlib
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import setup_windows as setup


class SetupWindowsTests(unittest.TestCase):
    def _temporary_path(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def _managed_venv(
        self,
        path: Path,
        *,
        sentinel: str = "old",
        marker: bool = True,
        lock_sha256: str | None = None,
    ) -> Path:
        (path / "Scripts").mkdir(parents=True, exist_ok=True)
        (path / "pyvenv.cfg").write_text(
            "version = 3.12.0\n", encoding="utf-8")
        (path / "Scripts" / "python.exe").write_bytes(b"stub")
        (path / "sentinel.txt").write_text(sentinel, encoding="utf-8")
        if marker:
            payload = setup.expected_venv_marker()
            if lock_sha256 is not None:
                payload["lock_sha256"] = lock_sha256
            (path / setup.VENV_MARKER_NAME).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
        return path

    @staticmethod
    def _result(argv, code=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(
            list(argv), code, stdout=stdout, stderr=stderr)

    def test_default_runtime_key_isolated_for_gpu_lock_revision(self):
        path = setup.resolve_venv_dir({"LOCALAPPDATA": r"C:\LocalData"})
        self.assertEqual(path.name, "v3-designer-preview.4")

    def test_target_path_rejects_source_tree_and_reparse_root(self):
        with self.assertRaisesRegex(setup.SetupError, "原始碼目錄之外"):
            setup.validate_target_path(setup.PROJECT_ROOT / ".venv")

    def test_metadata_permission_failure_rebuilds_once_and_keeps_backup(self):
        root = self._temporary_path()
        target = self._managed_venv(root / "runtime")
        calls = []

        def runner(argv, *, capture=False, **_kwargs):
            args = [str(value) for value in argv]
            calls.append(args)
            if len(args) >= 3 and args[1:3] == ["-I", "-c"]:
                return self._result(
                    args, 13, stderr="PermissionError: [WinError 5]")
            if "-m" in args and "venv" in args:
                self._managed_venv(Path(args[-1]), sentinel="new")
            return self._result(args)

        outcome = setup.ensure_validated_environment(
            target,
            runner=runner,
            now=datetime(2026, 7, 20, 15, 30, 0),
        )

        self.assertEqual(outcome.action, "rebuilt")
        self.assertIsNotNone(outcome.backup_dir)
        self.assertEqual(
            (target / "sentinel.txt").read_text(encoding="utf-8"), "new")
        self.assertEqual(
            (outcome.backup_dir / "sentinel.txt").read_text(encoding="utf-8"),
            "old",
        )
        self.assertEqual(
            sum("-m" in args and "venv" in args for args in calls), 1)

    def test_failed_rebuild_restores_old_environment_and_keeps_partial(self):
        root = self._temporary_path()
        target = self._managed_venv(root / "runtime")

        def runner(argv, *, capture=False, **_kwargs):
            args = [str(value) for value in argv]
            if len(args) >= 3 and args[1:3] == ["-I", "-c"]:
                return self._result(args, 13, stderr="metadata unreadable")
            if "-m" in args and "venv" in args:
                self._managed_venv(Path(args[-1]), sentinel="partial")
                return self._result(args)
            if "pip" in args:
                return self._result(args, 1, stderr="wheel unavailable")
            return self._result(args)

        with self.assertRaisesRegex(setup.SetupError, "舊環境已原位還原"):
            setup.ensure_validated_environment(
                target,
                runner=runner,
                now=datetime(2026, 7, 20, 15, 31, 0),
            )

        self.assertEqual(
            (target / "sentinel.txt").read_text(encoding="utf-8"), "old")
        failed = list(root.glob("runtime.failed-rebuild-20260720-153100*"))
        self.assertEqual(len(failed), 1)
        self.assertEqual(
            (failed[0] / "sentinel.txt").read_text(encoding="utf-8"),
            "partial",
        )
        self.assertFalse(any(root.glob("runtime.quarantine-*")))

    def test_readable_metadata_and_valid_preflight_reuses_without_pip(self):
        root = self._temporary_path()
        target = self._managed_venv(root / "runtime")
        calls = []

        def runner(argv, *, capture=False, **_kwargs):
            args = [str(value) for value in argv]
            calls.append(args)
            if len(args) >= 3 and args[1:3] == ["-I", "-c"]:
                return self._result(args, stdout="metadata-ok\n")
            return self._result(args, stdout="[環境檢查通過]\n")

        outcome = setup.ensure_validated_environment(target, runner=runner)

        self.assertEqual(outcome.action, "reused")
        self.assertFalse(any("pip" in args for args in calls))
        self.assertFalse(any("-m" in args and "venv" in args for args in calls))

    def test_normal_pip_failure_does_not_quarantine_readable_environment(self):
        root = self._temporary_path()
        target = self._managed_venv(root / "runtime")

        def runner(argv, *, capture=False, **_kwargs):
            args = [str(value) for value in argv]
            if len(args) >= 3 and args[1:3] == ["-I", "-c"]:
                return self._result(args, stdout="metadata-ok\n")
            if "environment_preflight.py" in " ".join(args):
                return self._result(args, 2, stderr="missing package")
            if "pip" in args:
                return self._result(args, 1, stderr="network unavailable")
            return self._result(args)

        with self.assertRaisesRegex(setup.SetupError, "依賴安裝失敗"):
            setup.ensure_validated_environment(target, runner=runner)

        self.assertTrue((target / "sentinel.txt").is_file())
        self.assertFalse(any(root.glob("runtime.quarantine-*")))

    def test_foreign_existing_directory_fails_closed_without_commands(self):
        root = self._temporary_path()
        target = root / "not-a-venv"
        self._managed_venv(target, marker=False)
        calls = []

        def runner(argv, *, capture=False, **_kwargs):
            calls.append(list(argv))
            return self._result(argv)

        with self.assertRaisesRegex(setup.SetupError, "ownership marker"):
            setup.ensure_validated_environment(target, runner=runner)
        self.assertEqual(calls, [])

    def test_owned_venv_with_old_lock_is_rebuilt_not_updated_in_place(self):
        root = self._temporary_path()
        target = self._managed_venv(
            root / "runtime", lock_sha256="0" * 64)
        calls = []

        def runner(argv, *, capture=False, **_kwargs):
            args = [str(value) for value in argv]
            calls.append(args)
            if "-m" in args and "venv" in args:
                self._managed_venv(Path(args[-1]), sentinel="new")
            return self._result(args)

        outcome = setup.ensure_validated_environment(
            target,
            runner=runner,
            now=datetime(2026, 7, 20, 15, 33, 0),
        )

        self.assertEqual(outcome.action, "rebuilt")
        self.assertEqual(
            (target / "sentinel.txt").read_text(encoding="utf-8"), "new")
        self.assertFalse(any("-I" in args and "-c" in args for args in calls))

    def test_failed_first_install_preserves_partial_and_leaves_target_clear(self):
        root = self._temporary_path()
        target = root / "runtime"

        def runner(argv, *, capture=False, **_kwargs):
            args = [str(value) for value in argv]
            if "-m" in args and "venv" in args:
                self._managed_venv(Path(args[-1]), sentinel="partial")
                return self._result(args)
            if "pip" in args:
                return self._result(args, 1, stderr="download failed")
            return self._result(args)

        with self.assertRaisesRegex(setup.SetupError, "初次建立外部環境失敗"):
            setup.ensure_validated_environment(
                target,
                runner=runner,
                now=datetime(2026, 7, 20, 15, 32, 0),
            )

        self.assertFalse(target.exists())
        failed = list(root.glob("runtime.failed-setup-20260720-153200*"))
        self.assertEqual(len(failed), 1)
        self.assertEqual(
            (failed[0] / "sentinel.txt").read_text(encoding="utf-8"),
            "partial",
        )

    def test_child_environment_removes_python_path_and_home(self):
        result = setup.child_environment({
            "PATH": "kept",
            "PYTHONHOME": "poisoned-home",
            "PYTHONPATH": "poisoned-path",
        })
        self.assertEqual(result["PATH"], "kept")
        self.assertNotIn("PYTHONHOME", result)
        self.assertNotIn("PYTHONPATH", result)

    def test_setup_lock_failure_stops_before_any_environment_command(self):
        root = self._temporary_path()
        target = root / "runtime"
        calls = []

        def runner(argv, *, capture=False, **_kwargs):
            calls.append(list(argv))
            return self._result(argv)

        with mock.patch.object(
            setup, "_lock_handle", side_effect=OSError("lock busy")
        ), self.assertRaisesRegex(setup.SetupError, "另一個 setup"):
            setup.ensure_validated_environment(target, runner=runner)

        self.assertEqual(calls, [])
        self.assertFalse(target.exists())

    def test_main_reports_localized_setup_error_without_replacement_character(self):
        stderr = io.StringIO()
        with mock.patch.object(
            setup, "ensure_validated_environment",
            side_effect=setup.SetupError("依賴安裝失敗，測試訊息完整。"),
        ), mock.patch.object(setup, "validate_host_runtime"), mock.patch.object(
            setup, "resolve_venv_dir", return_value=Path(r"C:\runtime")
        ), contextlib.redirect_stderr(stderr):
            result = setup.main()

        self.assertEqual(result, 2)
        self.assertIn("依賴安裝失敗，測試訊息完整。", stderr.getvalue())
        self.assertNotIn("\ufffd", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
