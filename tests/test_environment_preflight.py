from __future__ import annotations

import contextlib
import importlib.metadata
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import environment_preflight as preflight


class EnvironmentPreflightTests(unittest.TestCase):
    def _lock_file(self, text: str):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "validated.lock.txt"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(temporary.cleanup)
        return path

    def test_parse_locked_requirements_accepts_pip_hash_continuations(self):
        path = self._lock_file(
            "# validated\n"
            "NumPy==2.5.1 \\\n"
            "    --hash=sha256:abc\n"
            "pillow==12.3.0 \\\n"
            "    --hash=sha256:def\n"
        )

        pins = preflight.parse_locked_requirements(path)

        self.assertEqual(pins["numpy"], ("NumPy", "2.5.1"))
        self.assertEqual(pins["pillow"], ("pillow", "12.3.0"))

    def test_parse_locked_requirements_rejects_unpinned_range(self):
        path = self._lock_file("numpy>=2\n")

        with self.assertRaisesRegex(preflight.PreflightError, "精確版本鎖定"):
            preflight.parse_locked_requirements(path)

    def test_parse_locked_requirements_rejects_conflicting_duplicate(self):
        path = self._lock_file("rlPyCairo==0.4.0\nrlpycairo==0.5.0\n")

        with self.assertRaisesRegex(preflight.PreflightError, "衝突版本"):
            preflight.parse_locked_requirements(path)

    def test_python_runtime_requires_cpython_312_64_bit_contract(self):
        result = preflight.validate_python_runtime(
            version_info=(3, 12, 8), pointer_bits=64)
        self.assertEqual(result["version"], (3, 12, 8))
        self.assertEqual(result["pointer_bits"], 64)

        with self.assertRaisesRegex(preflight.PreflightError, "Python 3.12"):
            preflight.validate_python_runtime(
                version_info=(3, 11, 9), pointer_bits=64)
        with self.assertRaisesRegex(preflight.PreflightError, "64 位"):
            preflight.validate_python_runtime(
                version_info=(3, 12, 8), pointer_bits=32)

    def test_locked_versions_fail_fast_on_missing_distribution(self):
        pins = {"numpy": ("numpy", "2.5.1")}

        def missing(_name):
            raise importlib.metadata.PackageNotFoundError("numpy")

        with self.assertRaisesRegex(preflight.PreflightError, "尚未安裝"):
            preflight.validate_locked_versions(pins, version_getter=missing)

    def test_locked_versions_fail_fast_on_version_mismatch(self):
        pins = {"numpy": ("numpy", "2.5.1")}

        with self.assertRaisesRegex(preflight.PreflightError, "版本不符"):
            preflight.validate_locked_versions(
                pins, version_getter=lambda _name: "2.4.0")

    def test_full_import_check_includes_native_core_and_renderer_modules(self):
        loaded = []

        def importer(module_name):
            loaded.append(module_name)
            return object()

        result = preflight.validate_imports(full=True, importer=importer)

        self.assertIn("numpy", loaded)
        self.assertIn("PIL.Image", loaded)
        self.assertIn("vtracer", loaded)
        self.assertIn("svglib.svglib", loaded)
        self.assertIn("reportlab.graphics.renderPM", loaded)
        self.assertIn("rlPyCairo", loaded)
        self.assertIn("cairo", loaded)
        self.assertEqual(len(result), len(loaded))

    def test_import_failure_is_converted_to_actionable_preflight_error(self):
        def importer(module_name):
            if module_name == "numpy":
                raise ImportError("DLL load failed")
            return object()

        with self.assertRaisesRegex(
                preflight.PreflightError, "無法載入必要套件 NumPy"):
            preflight.validate_imports(importer=importer)

    def test_main_returns_nonzero_and_traditional_chinese_on_failure(self):
        stderr = io.StringIO()
        with mock.patch.object(
                preflight, "run_preflight",
                side_effect=preflight.PreflightError("測試依賴錯誤")):
            with contextlib.redirect_stderr(stderr):
                result = preflight.main(["--full", "--strict-versions"])

        self.assertEqual(result, 2)
        self.assertIn("環境檢查失敗", stderr.getvalue())
        self.assertIn("setup_windows.bat", stderr.getvalue())

    def test_main_reports_strict_success_summary(self):
        summary = {
            "runtime": {"version": (3, 12, 8), "pointer_bits": 64},
            "loaded": ("NumPy", "Pillow", "vtracer"),
            "strict_versions": True,
            "locked_distribution_count": 3,
            "lock_path": Path("validated.lock.txt"),
        }
        stdout = io.StringIO()
        with mock.patch.object(preflight, "run_preflight", return_value=summary):
            with contextlib.redirect_stdout(stdout):
                result = preflight.main(["--full", "--strict-versions"])

        self.assertEqual(result, 0)
        self.assertIn("環境檢查通過", stdout.getvalue())
        self.assertIn("版本鎖定通過", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
