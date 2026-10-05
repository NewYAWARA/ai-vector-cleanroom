from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import app_paths


class AppPathTests(unittest.TestCase):
    def test_default_data_root_is_short_versioned_and_outside_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve()
            source = root / "source" / "deep"
            source.mkdir(parents=True)

            resolved = app_paths.resolve_data_dir(
                env={"LOCALAPPDATA": str(root / "local")},
                code_dir=source,
            )

            self.assertEqual(resolved, root / "local" / "AIVC" / "designer4")
            self.assertNotIn(source, resolved.parents)

    def test_override_must_be_absolute_local_and_outside_source(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw).resolve() / "source"
            source.mkdir()
            with self.assertRaisesRegex(app_paths.DataPathError, "絕對路徑"):
                app_paths.resolve_data_dir(
                    env={"AVC_DATA_DIR": "relative-data"},
                    code_dir=source,
                )
            with self.assertRaisesRegex(app_paths.DataPathError, "原始碼目錄"):
                app_paths.resolve_data_dir(
                    env={"AVC_DATA_DIR": str(source / "data")},
                    code_dir=source,
                )

    def test_long_names_use_stable_distinct_sha_suffixes(self):
        first = "很長的上傳名稱" * 20
        second = first + "不同"

        first_base = app_paths.bounded_output_base(first)
        second_base = app_paths.bounded_output_base(second)

        self.assertLessEqual(
            app_paths._utf16_units(first_base),
            app_paths.MAX_OUTPUT_BASE_UNITS,
        )
        self.assertRegex(first_base, r"_[0-9a-f]{12}$")
        self.assertNotEqual(first_base, second_base)
        self.assertEqual(first_base, app_paths.bounded_output_base(first))

        filename = app_paths.bounded_input_filename(first + ".png")
        self.assertTrue(filename.endswith(".png"))
        self.assertRegex(Path(filename).stem, r"_[0-9a-f]{12}$")

    def test_output_budget_fails_before_conversion(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw).resolve() / ("d" * 180) / "output"
            with self.assertRaisesRegex(app_paths.DataPathError, "輸出路徑仍過長"):
                app_paths.ensure_output_path_budget(output, "x" * 48)

    def test_prepare_layout_probes_atomically_without_residue(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw).resolve()
            source = base / "source"
            data = base / "data"
            source.mkdir()

            result = app_paths.prepare_data_layout(data, code_dir=source)

            self.assertEqual(Path(result["root"]), data)
            self.assertTrue((data / "input").is_dir())
            self.assertTrue((data / "output").is_dir())
            self.assertFalse(any(data.glob(".aivc-write-probe-*")))
            self.assertLessEqual(
                result["max_result_path_units"],
                app_paths.MAX_WINDOWS_PATH_UNITS,
            )

    def test_writer_lock_is_exclusive_and_removed_on_release(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw).resolve() / "data"
            with app_paths.writer_lock(root, "first"):
                self.assertTrue((root / app_paths.LOCK_NAME).is_file())
                with self.assertRaises(app_paths.DataDirectoryBusyError):
                    with app_paths.writer_lock(root, "second"):
                        self.fail("second writer unexpectedly acquired the lock")
            self.assertFalse((root / app_paths.LOCK_NAME).exists())


if __name__ == "__main__":
    unittest.main()
