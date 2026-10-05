# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from release import package_source_beta6 as source_release


ROOT = Path(__file__).resolve().parents[1]


class SourceReleaseTests(unittest.TestCase):
    @staticmethod
    def _source_tree_snapshot(root: Path):
        """Return a byte/type snapshot that also notices new empty folders."""
        snapshot = {}
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[relative] = ("symlink", os.readlink(path))
            elif path.is_dir():
                snapshot[relative] = ("directory",)
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                snapshot[relative] = (
                    "file", path.stat().st_size, digest.hexdigest())
            else:
                snapshot[relative] = ("other",)
        return snapshot

    @staticmethod
    def _validated_external_venv():
        """Find an existing external venv; the launcher performs validation."""
        candidates = []
        configured = os.environ.get("AVC_VENV_DIR")
        if configured:
            candidates.append(Path(configured))

        executable = Path(sys.executable).resolve()
        if executable.parent.name.casefold() == "scripts":
            candidates.append(executable.parent.parent)

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(
                Path(local_app_data)
                / "AI-Vector-Cleanroom"
                / "venvs"
                / "v0.6.0-alpha"
            )

        seen = set()
        for candidate in candidates:
            key = os.path.normcase(os.path.abspath(candidate))
            if key in seen:
                continue
            seen.add(key)
            if (candidate / "Scripts" / "python.exe").is_file():
                return candidate.resolve()
        return None

    @unittest.skipUnless(os.name == "nt", "Windows batch integration smoke")
    def targeted_long_source_path_external_data_conversion(self):
        """Run only by explicit method name; normal unittest discovery skips it.

        Command::

            python -m unittest tests.test_source_release.SourceReleaseTests.targeted_long_source_path_external_data_conversion -v
        """
        venv = self._validated_external_venv()
        if venv is None:
            self.skipTest(
                "no existing validated external venv; run setup_windows.bat first")
        override = os.environ.get("AVC_TEST_PYTHON")
        runtime_python = (Path(override).resolve() if override
                          else venv / "Scripts" / "python.exe")
        if not runtime_python.is_file():
            self.skipTest(f"test runtime is unavailable: {runtime_python}")

        with tempfile.TemporaryDirectory(prefix="avc-b6-source-") as source_raw, \
                tempfile.TemporaryDirectory(prefix="avc-b6-data-") as data_raw:
            source_stage = Path(source_raw).resolve()
            data_root = Path(data_raw).resolve()
            if len(str(data_root)) > 120:
                self.skipTest(
                    f"temporary data root is not short enough: {data_root}")

            # Match the user's deep source layout without crossing legacy
            # MAX_PATH, which would test cmd.exe itself instead of our launcher.
            long_source = source_stage
            target_length = 182
            while len(str(long_source)) < target_length:
                remaining = target_length - len(str(long_source)) - 1
                component_length = min(48, max(1, remaining))
                long_source /= "s" * component_length
            self.assertGreaterEqual(len(str(long_source)), 175)
            self.assertLessEqual(len(str(long_source)), 190)

            for relative in source_release.PUBLIC_FILES:
                parts = PurePosixPath(relative).parts
                source = ROOT.joinpath(*parts)
                destination = long_source.joinpath(*parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            self.assertFalse((long_source / "input").exists())
            self.assertFalse((long_source / "output").exists())
            self.assertFalse(any(
                path.name == "__pycache__"
                for path in long_source.rglob("__pycache__")
            ))
            before = self._source_tree_snapshot(long_source)

            input_dir = data_root / "input"
            output_dir = data_root / "output"
            input_dir.mkdir()
            shutil.copy2(
                ROOT / "tests" / "fixtures" / "right_angle.png",
                input_dir / "smoke.png",
            )

            environment = os.environ.copy()
            environment["AVC_VENV_DIR"] = str(venv)
            environment["AVC_DATA_DIR"] = str(data_root)
            environment["PYTHONUTF8"] = "1"
            environment["PYTHONNOUSERSITE"] = "1"
            environment.pop("PYTHONHOME", None)
            if override:
                # CI/Codex may be unable to execute the venv launcher itself.
                # A CPython 3.12 x64 override can still consume only the exact
                # validated venv packages by disabling its own site module.
                environment["PYTHONPATH"] = str(venv / "Lib" / "site-packages")
            else:
                environment.pop("PYTHONPATH", None)

            smoke_code = """
import json
from app_paths import writer_lock
import workbench

image = workbench.INPUT_DIR / "smoke.png"
workbench._jobs.clear()
with writer_lock(workbench.DATA_DIR, "targeted-workbench-smoke"):
    workbench._run_one(image, {}, None)
job = workbench._jobs[-1]
print("AIVC_WORKBENCH_JOB=" + json.dumps(job, ensure_ascii=False))
raise SystemExit(0 if job.get("status") != "failed" else 4)
"""
            runtime_args = [str(runtime_python)]
            if override:
                runtime_args.append("-S")
            process = subprocess.run(
                runtime_args + ["-B", "-c", smoke_code],
                cwd=str(long_source),
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=240,
            )

            after = self._source_tree_snapshot(long_source)
            self.assertEqual(
                before, after,
                "conversion wrote into or changed the source tree")
            self.assertFalse((long_source / "input").exists())
            self.assertFalse((long_source / "output").exists())
            self.assertFalse(any(
                path.name == "__pycache__"
                for path in long_source.rglob("__pycache__")
            ))

            detail = process.stdout + process.stderr
            self.assertEqual(process.returncode, 0, detail)
            self.assertIn("AIVC_WORKBENCH_JOB=", process.stdout)
            self.assertEqual(output_dir.resolve().parent, data_root)
            self.assertFalse((data_root / ".aivc-writer.lock").exists())
            self.assertFalse(any(data_root.glob(".aivc-write-probe-*")))

            result = output_dir / "result_smoke"
            report_path = result / "report.json"
            svg_path = result / "smoke_vector.svg"
            zip_path = output_dir / "result_smoke.zip"
            for path in (report_path, svg_path, zip_path):
                self.assertTrue(path.is_file(), f"missing result: {path}\n{detail}")
                self.assertGreater(path.stat().st_size, 0, f"empty result: {path}")

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report.get("input"), "smoke.png")
            ET.parse(svg_path)
            with zipfile.ZipFile(zip_path, "r") as archive:
                self.assertIsNone(archive.testzip())
            evidence = {
                "data_root_chars": len(str(data_root)),
                "report_sha256": hashlib.sha256(
                    report_path.read_bytes()).hexdigest().upper(),
                "source_root_chars": len(str(long_source)),
                "source_tree_unchanged": before == after,
                "svg_sha256": hashlib.sha256(
                    svg_path.read_bytes()).hexdigest().upper(),
                "workbench_status": "not_failed",
                "zip_sha256": hashlib.sha256(
                    zip_path.read_bytes()).hexdigest().upper(),
            }
            print("AIVC_TARGETED_EVIDENCE=" + json.dumps(
                evidence, ensure_ascii=False, sort_keys=True))

    def test_windows_launchers_are_utf8_no_bom_and_crlf_only(self):
        batch_paths = [
            ROOT.joinpath(*Path(relative).parts)
            for relative in source_release.PUBLIC_FILES
            if Path(relative).suffix.casefold() == ".bat"
        ]

        self.assertGreaterEqual(len(batch_paths), 6)
        for path in batch_paths:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                data = path.read_bytes()
                self.assertFalse(data.startswith(b"\xef\xbb\xbf"))
                self.assertTrue(data.isascii())
                self.assertTrue(data.endswith(b"\r\n"))
                self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))
                self.assertNotIn(b"\r", data.replace(b"\r\n", b""))

    def test_user_launchers_are_ascii_trampolines_without_goto(self):
        for relative in ("setup_windows.bat", "工作台.bat", "清稿.bat", "clean.bat"):
            with self.subTest(relative=relative):
                data = (ROOT / relative).read_bytes()
                self.assertTrue(data.isascii())
                self.assertNotIn(b"goto ", data.lower())

    @unittest.skipUnless(os.name == "nt", "Windows cmd.exe integration")
    def test_workbench_missing_venv_branch_has_no_stray_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment["AVC_VENV_DIR"] = str(
                Path(temporary) / "missing (中文)&venv")
            environment["AVC_NO_PAUSE"] = "1"
            result = subprocess.run(
                [
                    os.environ.get("ComSpec", r"C:\Windows\System32\cmd.exe"),
                    "/d", "/q", "/c", "call", str(ROOT / "工作台.bat"),
                ],
                cwd=str(ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 2, output)
        self.assertIn("external Python environment is missing", output)
        self.assertNotIn("\ufffd", output)
        self.assertNotIn("is not recognized", output.casefold())
        self.assertNotIn("was unexpected at this time", output.casefold())
        self.assertNotIn("syntax of the command", output.casefold())

    def test_source_packager_enforces_windows_batch_byte_contract(self):
        valid = b"@echo off\r\necho Environment check\r\n"
        source_release._validate_windows_batch_bytes("valid.bat", valid)

        invalid_cases = {
            "utf8_bom": b"\xef\xbb\xbf@echo off\r\n",
            "lf_only": b"@echo off\nexit /b 0\n",
            "mixed": b"@echo off\r\nexit /b 0\n",
            "bare_cr": b"@echo off\rexit /b 0\r\n",
            "missing_final_crlf": b"@echo off\r\nexit /b 0",
            "utf8_text": "@echo off\r\necho 環境檢查\r\n".encode("utf-8"),
            "utf8_with_goto": "@echo off\r\ngoto :錯誤\r\n:錯誤\r\n".encode("utf-8"),
        }
        for case, data in invalid_cases.items():
            with self.subTest(case=case):
                with self.assertRaises(source_release.SourcePackageError):
                    source_release._validate_windows_batch_bytes(
                        f"{case}.bat", data)

    def test_public_allowlist_has_no_runtime_or_private_evidence(self):
        result = source_release.audit(ROOT)
        paths = set(source_release.PUBLIC_FILES)

        self.assertEqual(result["status"], "source_allowlist_ready")
        self.assertEqual(result["file_count"], len(paths))
        self.assertIn("README.md", paths)
        self.assertIn("app_paths.py", paths)
        self.assertIn("compute_backend.py", paths)
        self.assertIn("environment_preflight.py", paths)
        self.assertIn("execution_control.py", paths)
        self.assertIn("job_worker.py", paths)
        self.assertIn("tests/test_clean_base_prefix_cache.py", paths)
        self.assertIn("tests/test_compute_backend.py", paths)
        self.assertIn("tests/test_failure_diagnostics.py", paths)
        self.assertNotIn("beta6_release_runner.py", paths)
        self.assertNotIn("release/package_beta6.py", paths)
        self.assertFalse(any(path.startswith("python/") for path in paths))
        self.assertFalse(any(path.startswith("validation/") for path in paths))
        self.assertFalse(any("private_perf" in path for path in paths))
        self.assertFalse(any("14_13_12" in path for path in paths))
        self.assertFalse(any("PAUSE_CHECKPOINT" in path for path in paths))

    def test_checked_in_manifest_matches_current_public_sources(self):
        expected = source_release._manifest_bytes(
            source_release.collect_entries(ROOT))
        self.assertEqual(
            (ROOT / source_release.MANIFEST_NAME).read_bytes(), expected)

    def test_build_verify_and_no_overwrite_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            archive = directory / (source_release.PACKAGE_NAME + ".zip")
            receipt = directory / source_release.RECEIPT_NAME

            built = source_release.build(archive, receipt, ROOT)
            verified = source_release.verify(archive, receipt)

            self.assertEqual(built["status"], "source_archive_verified")
            self.assertEqual(verified["receipt"], "verified")
            self.assertEqual(built["zip_sha256"], verified["zip_sha256"])
            self.assertEqual(built["file_count"], len(source_release.PUBLIC_FILES))
            with zipfile.ZipFile(archive, "r") as package:
                manifest = json.loads(package.read(
                    f"{source_release.PACKAGE_NAME}/"
                    f"{source_release.MANIFEST_NAME}"))
            self.assertEqual(
                tuple(record["path"] for record in manifest["files"]),
                source_release.PUBLIC_FILES,
            )
            with zipfile.ZipFile(archive) as package:
                self.assertFalse(any(
                    name.startswith(source_release.PACKAGE_NAME + "/python/")
                    for name in package.namelist()
                ))
            with self.assertRaisesRegex(
                    source_release.SourcePackageError, "refuses to overwrite"):
                source_release.build(archive, receipt, ROOT)


if __name__ == "__main__":
    unittest.main()
