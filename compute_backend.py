"""Fail-closed compute acceleration for exact RGB palette labelling.

The public operation in this module has one contract: it returns the same
``int16`` nearest-palette labels as the established NumPy/float32 path.  A
hardware WebGPU adapter is used only after an integer WGSL parity check passes;
all unsupported inputs, unavailable drivers, validation failures, allocation
errors, device loss, and malformed results fall back to the CPU in-process.

Environment controls are deliberately small:

``AVC_GPU_MODE``
    ``auto`` (default), ``gpu`` (also accelerate small inputs), or ``cpu``.
``AVC_GPU_MIN_WORK``
    Minimum ``pixel_count * palette_count`` for GPU use in auto mode.
``AVC_GPU_AUDIT_PATH``
    Optional path atomically overwritten with a small, secret-free JSON audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import numpy as np


__all__ = [
    "nearest_palette_labels",
    "accelerator_audit",
    "accelerator_fingerprint",
]


_CPU_CHUNK_PIXELS = 131_072
_DEFAULT_MIN_WORK = 1_000_000
_WORKGROUP_SIZE = 64

_NEAREST_PALETTE_WGSL = r"""
@group(0) @binding(0)
var<storage, read> pixels: array<u32>;

@group(0) @binding(1)
var<storage, read> palette: array<u32>;

@group(0) @binding(2)
var<storage, read> params: array<u32>;

@group(0) @binding(3)
var<storage, read_write> labels: array<u32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params[0]) {
        return;
    }

    let colour = pixels[index];
    let red = i32(colour & 255u);
    let green = i32((colour >> 8u) & 255u);
    let blue = i32((colour >> 16u) & 255u);

    var best_distance = 0xffffffffu;
    var best_index = 0u;
    for (var candidate = 0u; candidate < params[1]; candidate += 1u) {
        let paint = palette[candidate];
        let dr = red - i32(paint & 255u);
        let dg = green - i32((paint >> 8u) & 255u);
        let db = blue - i32((paint >> 16u) & 255u);
        let distance = u32(dr * dr + dg * dg + db * db);
        // Strictly-less preserves NumPy argmin's first-index tie break.
        if (distance < best_distance) {
            best_distance = distance;
            best_index = candidate;
        }
    }
    labels[index] = best_index;
}
"""


class _GpuResultMismatch(RuntimeError):
    """The GPU returned an impossible shape, label, or parity result."""


@dataclass(slots=True)
class _WgpuBackend:
    wgpu: Any
    device: Any
    bind_group_layout: Any
    pipeline: Any
    info: dict[str, Any]


def _configured_mode() -> str:
    value = os.environ.get("AVC_GPU_MODE", "auto").strip().lower()
    if value in {"cpu", "off", "disabled", "disable", "0", "false"}:
        return "cpu"
    if value in {"gpu", "on", "force", "1", "true"}:
        return "gpu"
    return "auto"


def _configured_min_work() -> int:
    raw = os.environ.get("AVC_GPU_MIN_WORK", "").strip()
    if not raw:
        return _DEFAULT_MIN_WORK
    try:
        return max(0, int(raw))
    except (TypeError, ValueError, OverflowError):
        return _DEFAULT_MIN_WORK


def _new_audit() -> dict[str, Any]:
    mode = _configured_mode()
    return {
        "schema": "ai-vector-cleanroom.compute-backend.v1",
        "mode": mode,
        "provider": "cpu",
        "device": None,
        "vendor": None,
        "adapter_type": None,
        "backend_type": None,
        "parity": "skipped" if mode == "cpu" else "not_run",
        "status": "unprobed",
        "calls": 0,
        "successes": 0,
        "fallbacks": 0,
        "last_error": None,
    }


_STATE_LOCK = threading.RLock()
_OWNER_PID = os.getpid()
_PROBED = False
_BACKEND: _WgpuBackend | None = None
_AUDIT = _new_audit()


def _error_marker(stage: str, error: BaseException) -> str:
    """Return useful failure taxonomy without copying secret-bearing messages."""
    name = type(error).__name__ or "Error"
    return f"{stage}:{name}"[:96]


def _persist_audit_locked() -> None:
    """Best-effort atomic audit write; conversion must never depend on it."""
    raw_path = os.environ.get("AVC_GPU_AUDIT_PATH", "").strip()
    if not raw_path:
        return

    fd = -1
    temporary_name: str | None = None
    try:
        destination = Path(raw_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            json.dump(
                _AUDIT,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    except Exception:
        # An invalid/unwritable diagnostic destination is not a conversion error.
        pass
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def _reset_after_fork_locked() -> None:
    global _OWNER_PID, _PROBED, _BACKEND, _AUDIT
    current_pid = os.getpid()
    if current_pid == _OWNER_PID:
        return
    _OWNER_PID = current_pid
    _PROBED = False
    _BACKEND = None
    _AUDIT = _new_audit()


def _normalise_inputs(
    image: Any,
    palette: Any,
) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")

    palette_array = np.asarray(palette)
    try:
        palette_array = palette_array.reshape(-1, 3)
    except ValueError as error:
        raise ValueError("palette must contain RGB triples") from error
    if len(palette_array) == 0:
        raise ValueError("palette must contain at least one RGB colour")
    if len(palette_array) > np.iinfo(np.int16).max + 1:
        raise ValueError("palette has more labels than int16 can represent")
    return image_array, palette_array


def _cpu_nearest_palette_labels(
    image: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """Mirror the established float32/chunked NumPy implementation exactly."""
    pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)
    colours = np.asarray(palette, dtype=np.float32).reshape(-1, 3)
    labels = np.empty(len(pixels), dtype=np.int16)
    for start in range(0, len(pixels), _CPU_CHUNK_PIXELS):
        block = pixels[start : start + _CPU_CHUNK_PIXELS]
        distances = ((block[:, None] - colours[None, :]) ** 2).sum(2)
        labels[start : start + len(block)] = distances.argmin(1).astype(
            np.int16,
            copy=False,
        )
    return labels.reshape(image.shape[:2])


def _as_exact_u8_rgb(values: np.ndarray) -> np.ndarray | None:
    """Return contiguous uint8 RGB data only when conversion is lossless."""
    array = np.asarray(values)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array.reshape(-1, 3))

    kind = array.dtype.kind
    if kind in "uib":
        if array.size and (np.min(array) < 0 or np.max(array) > 255):
            return None
    elif kind == "f":
        if not np.all(np.isfinite(array)):
            return None
        if array.size and (np.min(array) < 0 or np.max(array) > 255):
            return None
        if not np.all(array == np.trunc(array)):
            return None
    else:
        return None
    return np.ascontiguousarray(array.reshape(-1, 3), dtype=np.uint8)


def _pack_rgb(values: np.ndarray) -> np.ndarray:
    expanded = np.asarray(values, dtype=np.uint32)
    return np.ascontiguousarray(
        expanded[:, 0]
        | (expanded[:, 1] << np.uint32(8))
        | (expanded[:, 2] << np.uint32(16))
    )


def _normalised_enum(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _adapter_info(adapter: Any) -> dict[str, Any]:
    try:
        return dict(adapter.info)
    except Exception:
        return {}


def _adapter_is_fallback(adapter: Any, info: dict[str, Any]) -> bool:
    try:
        return bool(adapter.info.is_fallback_adapter)
    except Exception:
        return bool(info.get("is_fallback_adapter", False))


def _select_hardware_adapter(wgpu: Any) -> tuple[Any, dict[str, Any]]:
    # The high-performance request avoids enumerating every Vulkan, D3D12 and
    # OpenGL representation on hybrid laptops.  On the validated RTX 4060
    # system that cuts first-use setup by several seconds.  If the platform
    # still returns an integrated adapter, enumerate once so a discrete device
    # can retain the documented priority.
    request_adapter = getattr(wgpu.gpu, "request_adapter_sync", None)
    requested = (request_adapter(
        power_preference="high-performance",
        force_fallback_adapter=False,
    ) if callable(request_adapter) else None)
    adapters: list[Any] = [] if requested is None else [requested]
    if requested is not None:
        requested_info = _adapter_info(requested)
        requested_type = _normalised_enum(
            requested_info.get("adapter_type", ""))
        if (not _adapter_is_fallback(requested, requested_info)
                and requested_type == "discretegpu"):
            return requested, requested_info

    enumerate_adapters = getattr(wgpu.gpu, "enumerate_adapters_sync", None)
    if callable(enumerate_adapters):
        for adapter in enumerate_adapters():
            if all(adapter is not existing for existing in adapters):
                adapters.append(adapter)

    hardware: list[tuple[Any, dict[str, Any]]] = []
    for adapter in adapters:
        info = _adapter_info(adapter)
        adapter_type = _normalised_enum(info.get("adapter_type", ""))
        if _adapter_is_fallback(adapter, info) or adapter_type == "cpu":
            continue
        hardware.append((adapter, info))
    if not hardware:
        raise RuntimeError("no hardware WebGPU adapter")

    # Stable sort keeps wgpu's own backend preference when adapter classes tie.
    priority = {
        "discretegpu": 0,
        "integratedgpu": 1,
        "virtualgpu": 2,
    }
    return min(
        hardware,
        key=lambda item: priority.get(
            _normalised_enum(item[1].get("adapter_type", "")),
            3,
        ),
    )


def _create_wgpu_backend() -> _WgpuBackend:
    # Lazy import makes a missing optional driver/package an ordinary CPU case.
    import wgpu  # type: ignore[import-not-found]

    adapter, info = _select_hardware_adapter(wgpu)
    device = adapter.request_device_sync(label="AI Vector Cleanroom compute")
    shader = device.create_shader_module(code=_NEAREST_PALETTE_WGSL)

    read_only = wgpu.BufferBindingType.read_only_storage
    read_write = wgpu.BufferBindingType.storage
    binding_layouts = []
    for binding in range(4):
        binding_layouts.append(
            {
                "binding": binding,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": read_write if binding == 3 else read_only,
                    "has_dynamic_offset": False,
                },
            }
        )
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader, "entry_point": "main"},
    )
    return _WgpuBackend(
        wgpu=wgpu,
        device=device,
        bind_group_layout=bind_group_layout,
        pipeline=pipeline,
        info=info,
    )


def _run_gpu_locked(
    backend: _WgpuBackend,
    pixels_u8: np.ndarray,
    palette_u8: np.ndarray,
) -> np.ndarray:
    pixel_count = len(pixels_u8)
    if pixel_count <= 0:
        return np.empty(0, dtype=np.uint32)

    packed_pixels = _pack_rgb(pixels_u8)
    packed_palette = _pack_rgb(palette_u8)
    params = np.asarray([pixel_count, len(palette_u8)], dtype=np.uint32)
    wgpu = backend.wgpu
    device = backend.device

    storage = wgpu.BufferUsage.STORAGE
    pixel_buffer = device.create_buffer_with_data(data=packed_pixels, usage=storage)
    palette_buffer = device.create_buffer_with_data(data=packed_palette, usage=storage)
    params_buffer = device.create_buffer_with_data(data=params, usage=storage)
    label_buffer = device.create_buffer(
        size=pixel_count * np.dtype(np.uint32).itemsize,
        usage=storage | wgpu.BufferUsage.COPY_SRC,
    )
    buffers = (pixel_buffer, palette_buffer, params_buffer, label_buffer)
    entries = [
        {
            "binding": binding,
            "resource": {"buffer": buffer, "offset": 0, "size": buffer.size},
        }
        for binding, buffer in enumerate(buffers)
    ]
    bind_group = device.create_bind_group(
        layout=backend.bind_group_layout,
        entries=entries,
    )

    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(backend.pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(
        math.ceil(pixel_count / _WORKGROUP_SIZE),
        1,
        1,
    )
    compute_pass.end()
    device.queue.submit([encoder.finish()])
    result = device.queue.read_buffer(label_buffer)
    return np.frombuffer(result, dtype=np.uint32, count=pixel_count).copy()


def _parity_fixture() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((17, 19), dtype=np.uint32)
    image = np.stack(
        (
            (x * 17 + y * 3) & 255,
            (x * 5 + y * 29) & 255,
            (x * 41 + y * 7) & 255,
        ),
        axis=2,
    ).astype(np.uint8)
    palette = np.asarray(
        [
            [0, 0, 0],
            [255, 255, 255],
            [10, 20, 30],
            [10, 20, 30],  # duplicate asserts first-index tie behaviour
            [128, 128, 128],
            [255, 0, 127],
            [0, 255, 64],
        ],
        dtype=np.uint8,
    )
    image[0, 0] = palette[2]
    return image, palette


def _audit_adapter_locked(backend: _WgpuBackend) -> None:
    info = backend.info
    _AUDIT.update(
        {
            "provider": "wgpu",
            "device": _audit_text(info.get("device")),
            "vendor": _audit_text(info.get("vendor")),
            "adapter_type": _audit_text(info.get("adapter_type")),
            "backend_type": _audit_text(info.get("backend_type")),
        }
    )


def _audit_text(value: Any) -> str | None:
    if value is None:
        return None
    # Adapter metadata is public hardware identity; keep the audit bounded.
    return str(value).replace("\r", " ").replace("\n", " ")[:160]


def _ensure_backend_locked() -> _WgpuBackend | None:
    global _PROBED, _BACKEND
    if _PROBED:
        return _BACKEND
    _PROBED = True

    if _AUDIT["mode"] == "cpu":
        _AUDIT.update(
            {
                "provider": "cpu",
                "parity": "skipped",
                "status": "cpu_only",
                "last_error": None,
            }
        )
        _persist_audit_locked()
        return None

    try:
        candidate = _create_wgpu_backend()
        _audit_adapter_locked(candidate)
        fixture_image, fixture_palette = _parity_fixture()
        expected = _cpu_nearest_palette_labels(
            fixture_image,
            fixture_palette,
        ).reshape(-1)
        actual = _validate_gpu_result(
            _run_gpu_locked(
                candidate,
                fixture_image.reshape(-1, 3),
                fixture_palette,
            ),
            fixture_image.shape[0] * fixture_image.shape[1],
            len(fixture_palette),
        )
        if actual.shape != expected.shape or not np.array_equal(
            actual.astype(np.int16, copy=False),
            expected,
        ):
            raise _GpuResultMismatch("synthetic parity mismatch")
        _BACKEND = candidate
        _AUDIT.update(
            {
                "parity": "passed",
                "status": "ready",
                "last_error": None,
            }
        )
    except Exception as error:
        _BACKEND = None
        if isinstance(error, _GpuResultMismatch):
            _AUDIT["parity"] = "failed"
        _AUDIT.update(
            {
                "status": "unavailable",
                "last_error": _error_marker("probe", error),
            }
        )
        if _AUDIT["provider"] != "wgpu":
            _AUDIT["provider"] = "cpu"
    _persist_audit_locked()
    return _BACKEND


def _validate_gpu_result(
    labels: np.ndarray,
    pixel_count: int,
    palette_count: int,
) -> np.ndarray:
    flattened = np.asarray(labels, dtype=np.uint32).reshape(-1)
    if len(flattened) != pixel_count:
        raise _GpuResultMismatch("GPU label count mismatch")
    if flattened.size and int(flattened.max()) >= palette_count:
        raise _GpuResultMismatch("GPU label out of palette range")
    return flattened


def nearest_palette_labels(image: Any, palette: Any) -> np.ndarray:
    """Return exact nearest RGB palette indices, using GPU when safe.

    GPU acceleration is an implementation detail.  The returned shape is
    ``(height, width)`` and the dtype is always ``int16``.  The function never
    exposes a GPU failure to callers when the established CPU calculation can
    still complete.
    """
    global _BACKEND
    image_array, palette_array = _normalise_inputs(image, palette)
    pixels_u8 = _as_exact_u8_rgb(image_array)
    palette_u8 = _as_exact_u8_rgb(palette_array)
    pixel_count = int(image_array.shape[0]) * int(image_array.shape[1])
    work = pixel_count * len(palette_array)

    use_cpu = False
    gpu_result: np.ndarray | None = None
    with _STATE_LOCK:
        _reset_after_fork_locked()
        _AUDIT["calls"] += 1
        backend = None

        # Eligibility and the auto workload threshold are decided before
        # importing wgpu or opening a device.  A small logo must not pay a
        # multi-second driver/shader setup cost merely to choose the CPU path.
        if _AUDIT["mode"] == "cpu":
            use_cpu = True
            _AUDIT["fallbacks"] += 1
            _AUDIT["status"] = "cpu_only"
            _AUDIT["last_error"] = None
        elif pixel_count == 0:
            use_cpu = True
            _AUDIT.update(
                {
                    "fallbacks": _AUDIT["fallbacks"] + 1,
                    "status": "cpu_empty_input",
                    "last_error": None,
                }
            )
        elif pixels_u8 is None or palette_u8 is None:
            use_cpu = True
            _AUDIT.update(
                {
                    "fallbacks": _AUDIT["fallbacks"] + 1,
                    "status": "cpu_ineligible_input",
                    "last_error": None,
                }
            )
        elif _AUDIT["mode"] == "auto" and work < _configured_min_work():
            use_cpu = True
            _AUDIT.update(
                {
                    "fallbacks": _AUDIT["fallbacks"] + 1,
                    "status": "cpu_small_input",
                    "last_error": None,
                }
            )
        else:
            backend = _ensure_backend_locked()
            if backend is None:
                use_cpu = True
                _AUDIT.update(
                    {
                        "fallbacks": _AUDIT["fallbacks"] + 1,
                        "status": "cpu_fallback",
                    }
                )
                _persist_audit_locked()
                backend = None
        if not use_cpu and backend is not None:
            try:
                gpu_result = _validate_gpu_result(
                    _run_gpu_locked(backend, pixels_u8, palette_u8),
                    pixel_count,
                    len(palette_array),
                )
                _AUDIT.update(
                    {
                        "successes": _AUDIT["successes"] + 1,
                        "status": "gpu_active",
                        "last_error": None,
                    }
                )
            except Exception as error:
                # Circuit-break a suspect device for the remainder of this process.
                _BACKEND = None
                use_cpu = True
                _AUDIT.update(
                    {
                        "fallbacks": _AUDIT["fallbacks"] + 1,
                        "status": "cpu_fallback",
                        "last_error": _error_marker("dispatch", error),
                    }
                )
        _persist_audit_locked()

    if use_cpu:
        return _cpu_nearest_palette_labels(image_array, palette_array)
    assert gpu_result is not None
    return gpu_result.astype(np.int16, copy=False).reshape(image_array.shape[:2])


def accelerator_audit(*, probe: bool = False) -> dict[str, Any]:
    """Return a detached JSON-safe snapshot of current backend audit state."""
    with _STATE_LOCK:
        _reset_after_fork_locked()
        if probe:
            _ensure_backend_locked()
        return dict(_AUDIT)


def accelerator_fingerprint(*, probe: bool = False) -> dict[str, Any]:
    """Return stable accelerator identity plus a compact SHA-256 identifier."""
    audit = accelerator_audit(probe=probe)
    identity = {
        key: audit[key]
        for key in (
            "mode",
            "provider",
            "device",
            "vendor",
            "adapter_type",
            "backend_type",
            "parity",
        )
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **identity,
        "id": f"sha256:{sha256(canonical).hexdigest()}",
    }


def _reset_for_tests() -> None:
    """Reset process-local state; intentionally private test seam."""
    global _OWNER_PID, _PROBED, _BACKEND, _AUDIT
    with _STATE_LOCK:
        _OWNER_PID = os.getpid()
        _PROBED = False
        _BACKEND = None
        _AUDIT = _new_audit()
