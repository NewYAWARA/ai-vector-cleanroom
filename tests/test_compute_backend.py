import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import compute_backend
import clean_base


class ComputeBackendTests(unittest.TestCase):
    _ENV_KEYS = (
        "AVC_GPU_MODE",
        "AVC_GPU_MIN_WORK",
        "AVC_GPU_AUDIT_PATH",
    )

    def setUp(self):
        self._saved_environment = {
            key: os.environ.get(key) for key in self._ENV_KEYS
        }
        os.environ["AVC_GPU_MODE"] = "cpu"
        os.environ.pop("AVC_GPU_MIN_WORK", None)
        os.environ.pop("AVC_GPU_AUDIT_PATH", None)
        compute_backend._reset_for_tests()

    def tearDown(self):
        for key, value in self._saved_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        compute_backend._reset_for_tests()

    @staticmethod
    def _sample():
        image = np.asarray(
            [
                [[0, 0, 0], [5, 0, 0], [10, 20, 30]],
                [[255, 255, 255], [120, 130, 140], [0, 250, 60]],
            ],
            dtype=np.uint8,
        )
        palette = np.asarray(
            [
                [0, 0, 0],
                [10, 0, 0],
                [10, 20, 30],
                [10, 20, 30],
                [255, 255, 255],
                [0, 255, 64],
            ],
            dtype=np.uint8,
        )
        return image, palette

    def test_cpu_result_matches_float32_reference_and_first_tie(self):
        image, palette = self._sample()
        pixels = image.astype(np.float32).reshape(-1, 3)
        colours = palette.astype(np.float32).reshape(-1, 3)
        expected = (
            ((pixels[:, None] - colours[None, :]) ** 2)
            .sum(2)
            .argmin(1)
            .astype(np.int16)
            .reshape(image.shape[:2])
        )

        actual = compute_backend.nearest_palette_labels(image, palette)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.int16)
        self.assertEqual(int(actual[0, 2]), 2)
        audit = compute_backend.accelerator_audit()
        self.assertEqual(audit["mode"], "cpu")
        self.assertEqual(audit["provider"], "cpu")
        self.assertEqual(audit["parity"], "skipped")
        self.assertEqual(audit["calls"], 1)
        self.assertEqual(audit["successes"], 0)
        self.assertEqual(audit["fallbacks"], 1)

    def test_clean_base_default_palette_labeller_uses_backend_contract(self):
        image, palette = self._sample()
        expected = np.zeros(image.shape[:2], dtype=np.int16)
        with mock.patch.object(
                compute_backend, "nearest_palette_labels",
                return_value=expected) as accelerated:
            actual = clean_base._nearest_palette_labels(image, palette)
        accelerated.assert_called_once_with(image, palette)
        self.assertIs(actual, expected)

    def test_invalid_input_is_rejected_before_backend_state_changes(self):
        with self.assertRaisesRegex(ValueError, "image must have shape"):
            compute_backend.nearest_palette_labels(
                np.zeros((3, 3), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            compute_backend.nearest_palette_labels(
                np.zeros((3, 3, 3), dtype=np.uint8),
                np.zeros((0, 3), dtype=np.uint8),
            )
        self.assertEqual(compute_backend.accelerator_audit()["calls"], 0)

    def test_audit_is_atomically_written_with_required_secret_free_fields(self):
        image, palette = self._sample()
        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / "compute-audit.json"
            os.environ["AVC_GPU_AUDIT_PATH"] = str(audit_path)
            compute_backend._reset_for_tests()

            compute_backend.nearest_palette_labels(image, palette)

            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            required = {
                "mode",
                "provider",
                "device",
                "vendor",
                "adapter_type",
                "backend_type",
                "parity",
                "status",
                "calls",
                "successes",
                "fallbacks",
                "last_error",
            }
            self.assertTrue(required.issubset(payload))
            self.assertNotIn(str(audit_path), json.dumps(payload))
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_audit_write_failure_never_aborts_cpu_conversion(self):
        image, palette = self._sample()
        os.environ["AVC_GPU_AUDIT_PATH"] = "unused.json"
        compute_backend._reset_for_tests()

        with mock.patch.object(
            compute_backend.tempfile,
            "mkstemp",
            side_effect=PermissionError("denied"),
        ):
            actual = compute_backend.nearest_palette_labels(image, palette)

        expected = compute_backend._cpu_nearest_palette_labels(image, palette)
        np.testing.assert_array_equal(actual, expected)

    def test_forced_gpu_success_returns_labels_and_updates_counters(self):
        image, palette = self._sample()
        expected = compute_backend._cpu_nearest_palette_labels(
            image,
            palette,
        ).reshape(-1).astype(np.uint32)
        os.environ["AVC_GPU_MODE"] = "gpu"
        compute_backend._reset_for_tests()
        fake_backend = object()

        with (
            mock.patch.object(
                compute_backend,
                "_ensure_backend_locked",
                return_value=fake_backend,
            ),
            mock.patch.object(
                compute_backend,
                "_run_gpu_locked",
                return_value=expected,
            ) as run_gpu,
        ):
            actual = compute_backend.nearest_palette_labels(image, palette)

        np.testing.assert_array_equal(actual.reshape(-1), expected.astype(np.int16))
        run_gpu.assert_called_once()
        audit = compute_backend.accelerator_audit()
        self.assertEqual(audit["calls"], 1)
        self.assertEqual(audit["successes"], 1)
        self.assertEqual(audit["fallbacks"], 0)
        self.assertEqual(audit["status"], "gpu_active")

    def test_small_auto_workload_does_not_dispatch_to_ready_gpu(self):
        image, palette = self._sample()
        os.environ["AVC_GPU_MODE"] = "auto"
        os.environ["AVC_GPU_MIN_WORK"] = "999999999"
        compute_backend._reset_for_tests()
        fake_backend = object()

        with (
            mock.patch.object(
                compute_backend,
                "_ensure_backend_locked",
                return_value=fake_backend,
            ) as ensure_backend,
            mock.patch.object(compute_backend, "_run_gpu_locked") as run_gpu,
        ):
            actual = compute_backend.nearest_palette_labels(image, palette)

        expected = compute_backend._cpu_nearest_palette_labels(image, palette)
        np.testing.assert_array_equal(actual, expected)
        ensure_backend.assert_not_called()
        run_gpu.assert_not_called()
        audit = compute_backend.accelerator_audit()
        self.assertEqual(audit["status"], "cpu_small_input")
        self.assertEqual(audit["fallbacks"], 1)

    def test_nonintegral_rgb_is_ineligible_and_uses_float32_cpu_contract(self):
        image, palette = self._sample()
        image = image.astype(np.float32)
        image[0, 0, 0] = 0.25
        os.environ["AVC_GPU_MODE"] = "gpu"
        compute_backend._reset_for_tests()

        with (
            mock.patch.object(
                compute_backend,
                "_ensure_backend_locked",
                return_value=object(),
            ),
            mock.patch.object(compute_backend, "_run_gpu_locked") as run_gpu,
        ):
            actual = compute_backend.nearest_palette_labels(image, palette)

        expected = compute_backend._cpu_nearest_palette_labels(image, palette)
        np.testing.assert_array_equal(actual, expected)
        run_gpu.assert_not_called()
        self.assertEqual(
            compute_backend.accelerator_audit()["status"],
            "cpu_ineligible_input",
        )

    def test_gpu_error_circuit_breaks_and_falls_back_to_cpu(self):
        image, palette = self._sample()
        os.environ["AVC_GPU_MODE"] = "gpu"
        compute_backend._reset_for_tests()

        with (
            mock.patch.object(
                compute_backend,
                "_ensure_backend_locked",
                return_value=object(),
            ),
            mock.patch.object(
                compute_backend,
                "_run_gpu_locked",
                side_effect=MemoryError("simulated OOM"),
            ),
        ):
            actual = compute_backend.nearest_palette_labels(image, palette)

        expected = compute_backend._cpu_nearest_palette_labels(image, palette)
        np.testing.assert_array_equal(actual, expected)
        audit = compute_backend.accelerator_audit()
        self.assertEqual(audit["successes"], 0)
        self.assertEqual(audit["fallbacks"], 1)
        self.assertEqual(audit["status"], "cpu_fallback")
        self.assertEqual(audit["last_error"], "dispatch:MemoryError")

    def test_out_of_range_gpu_label_is_a_mismatch_and_falls_back(self):
        image, palette = self._sample()
        labels = np.full(image.shape[0] * image.shape[1], 999, dtype=np.uint32)
        os.environ["AVC_GPU_MODE"] = "gpu"
        compute_backend._reset_for_tests()

        with (
            mock.patch.object(
                compute_backend,
                "_ensure_backend_locked",
                return_value=object(),
            ),
            mock.patch.object(
                compute_backend,
                "_run_gpu_locked",
                return_value=labels,
            ),
        ):
            actual = compute_backend.nearest_palette_labels(image, palette)

        expected = compute_backend._cpu_nearest_palette_labels(image, palette)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            compute_backend.accelerator_audit()["last_error"],
            "dispatch:_GpuResultMismatch",
        )

    def test_hardware_selection_prefers_discrete_over_integrated(self):
        class AdapterInfo(dict):
            is_fallback_adapter = False

        integrated = SimpleNamespace(
            info=AdapterInfo(
                device="Integrated",
                adapter_type="IntegratedGPU",
            )
        )
        discrete = SimpleNamespace(
            info=AdapterInfo(
                device="Discrete",
                adapter_type="DiscreteGPU",
            )
        )
        software = SimpleNamespace(
            info=AdapterInfo(
                device="Software",
                adapter_type="CPU",
            )
        )
        fake_wgpu = SimpleNamespace(
            gpu=SimpleNamespace(
                enumerate_adapters_sync=lambda: [integrated, software, discrete]
            )
        )

        selected, info = compute_backend._select_hardware_adapter(fake_wgpu)

        self.assertIs(selected, discrete)
        self.assertEqual(info["device"], "Discrete")

    def test_synthetic_parity_probe_passes_before_backend_is_ready(self):
        os.environ["AVC_GPU_MODE"] = "auto"
        compute_backend._reset_for_tests()
        fake_backend = SimpleNamespace(
            info={
                "device": "Test Discrete GPU",
                "vendor": "Test Vendor",
                "adapter_type": "DiscreteGPU",
                "backend_type": "Vulkan",
            }
        )

        def exact_fixture_labels(_backend, pixels, palette):
            fixture = pixels.reshape(17, 19, 3)
            return compute_backend._cpu_nearest_palette_labels(
                fixture,
                palette,
            ).reshape(-1).astype(np.uint32)

        with (
            mock.patch.object(
                compute_backend,
                "_create_wgpu_backend",
                return_value=fake_backend,
            ),
            mock.patch.object(
                compute_backend,
                "_run_gpu_locked",
                side_effect=exact_fixture_labels,
            ),
        ):
            audit = compute_backend.accelerator_audit(probe=True)

        self.assertEqual(audit["provider"], "wgpu")
        self.assertEqual(audit["device"], "Test Discrete GPU")
        self.assertEqual(audit["parity"], "passed")
        self.assertEqual(audit["status"], "ready")

    def test_synthetic_parity_mismatch_disables_gpu(self):
        os.environ["AVC_GPU_MODE"] = "auto"
        compute_backend._reset_for_tests()
        fake_backend = SimpleNamespace(
            info={
                "device": "Bad GPU",
                "vendor": "Test Vendor",
                "adapter_type": "DiscreteGPU",
                "backend_type": "Vulkan",
            }
        )

        with (
            mock.patch.object(
                compute_backend,
                "_create_wgpu_backend",
                return_value=fake_backend,
            ),
            mock.patch.object(
                compute_backend,
                "_run_gpu_locked",
                return_value=np.zeros(17 * 19, dtype=np.uint32),
            ),
        ):
            audit = compute_backend.accelerator_audit(probe=True)

        self.assertEqual(audit["parity"], "failed")
        self.assertEqual(audit["status"], "unavailable")
        self.assertEqual(audit["last_error"], "probe:_GpuResultMismatch")

    def test_adapter_selection_uses_direct_discrete_without_enumeration(self):
        class AdapterInfo(dict):
            is_fallback_adapter = False

        info = AdapterInfo(
            device="Direct GPU",
            adapter_type="DiscreteGPU",
            backend_type="Vulkan",
        )
        adapter = SimpleNamespace(info=info)
        gpu = SimpleNamespace(
            request_adapter_sync=mock.Mock(return_value=adapter),
            enumerate_adapters_sync=mock.Mock(
                side_effect=AssertionError("must not enumerate")),
        )

        selected, selected_info = compute_backend._select_hardware_adapter(
            SimpleNamespace(gpu=gpu))

        self.assertIs(selected, adapter)
        self.assertEqual(selected_info["device"], "Direct GPU")
        gpu.enumerate_adapters_sync.assert_not_called()

    def test_fingerprint_omits_mutable_counters(self):
        image, palette = self._sample()
        first = compute_backend.accelerator_fingerprint(probe=True)
        compute_backend.nearest_palette_labels(image, palette)
        second = compute_backend.accelerator_fingerprint()

        self.assertEqual(first, second)
        self.assertTrue(first["id"].startswith("sha256:"))
        self.assertNotIn("calls", first)
        self.assertNotIn("last_error", first)


if __name__ == "__main__":
    unittest.main()
