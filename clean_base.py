# -*- coding: utf-8 -*-
"""
AI Vector Cleanroom engine.

Pipeline: background cleanup -> automatic palette detection -> color
flattening (removes gradient/antialiasing noise) -> VTracer spline tracing
-> transforms baked into absolute coordinates -> optional geometry
regularization (perfect circles, ring/band arcs, rivet alignment,
concentric centers) -> grouping by actual stacking order (same-color runs
merge only when nothing in between overlaps, so rendering never changes).

The output SVG contains vector paths only and does not embed bitmap images.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from PIL import Image, ImageFilter

import vtracer

from execution_control import ConversionInterrupted
from trace_engine import _prepare_image


GRADIENT_STAGE_CACHE_SCHEMA = "ai-vector-cleanroom.gradient-stage-cache/v1"
PRE_GRADIENT_STATE_CACHE_SCHEMA = (
    "ai-vector-cleanroom.pre-gradient-state-cache/v1")
POST_FIT_SCENE_CACHE_SCHEMA = "ai-vector-cleanroom.post-fit-scene-cache/v1"
POST_FIT_SCENE_CACHE_MAX_ENTRIES = 16
POST_FIT_SCENE_CACHE_MAX_BYTES = 64 * 1024 * 1024


def new_pre_gradient_state_cache():
    """Return one exact cache shared only by a single ``process_one`` run."""
    return {
        "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
        "entries": {},
        "audit": {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "errors": 0,
        },
    }


def _normalise_pre_gradient_state_cache(cache):
    if cache is None:
        return None
    if not isinstance(cache, dict):
        raise TypeError("pre_gradient_state_cache must be a dict or None")
    if not cache:
        cache.update(new_pre_gradient_state_cache())
    if cache.get("schema") != PRE_GRADIENT_STATE_CACHE_SCHEMA:
        raise ValueError("unsupported pre-gradient-state cache schema")
    entries = cache.get("entries")
    audit = cache.get("audit")
    if not isinstance(entries, dict) or not isinstance(audit, dict):
        raise ValueError("malformed pre-gradient-state cache")
    for name in ("requests", "hits", "misses", "errors"):
        audit[name] = int(audit.get(name, 0) or 0)
    return cache


def pre_gradient_state_cache_audit(cache):
    """Return a JSON-safe audit without retaining cached image state."""
    state = _normalise_pre_gradient_state_cache(cache)
    if state is None:
        return {
            "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
            "scope": "disabled",
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "entry_count": 0,
            "keys_sha256": [],
        }
    return {
        "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
        "scope": "single_process_one",
        **{name: int(state["audit"].get(name, 0) or 0)
           for name in ("requests", "hits", "misses", "errors")},
        "entry_count": len(state["entries"]),
        "keys_sha256": sorted(str(key) for key in state["entries"]),
    }


def _pre_gradient_state_cache_key(src, parameters, stage_callable,
                                  stage_schema):
    """Hash exact source bytes and every option affecting the cached prefix."""
    source_bytes = Path(src).read_bytes()
    callable_identity = {
        "module": str(getattr(stage_callable, "__module__",
                              type(stage_callable).__module__)),
        "qualname": str(getattr(stage_callable, "__qualname__",
                                getattr(stage_callable, "__name__",
                                        type(stage_callable).__qualname__))),
        "process_identity": id(stage_callable),
    }
    source_sha256 = hashlib.sha256(source_bytes).hexdigest().upper()
    contract = {
        "cache_schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
        "stage_schema": str(stage_schema),
        "stage_callable": callable_identity,
        "parameters": parameters,
        "source_size_bytes": len(source_bytes),
        "source_sha256": source_sha256,
    }
    header = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(header).hexdigest().upper()


def _pre_gradient_state_cache_get_or_compute(cache, key, producer):
    """Return isolated state; interrupted or failed producers are not stored."""
    state = _normalise_pre_gradient_state_cache(cache)
    if state is None:
        return producer(), {
            "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
            "status": "disabled",
            "key_sha256": key,
        }
    audit = state["audit"]
    entries = state["entries"]
    audit["requests"] += 1
    if key in entries:
        audit["hits"] += 1
        return copy.deepcopy(entries[key]), {
            "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
            "status": "hit",
            "key_sha256": key,
        }
    audit["misses"] += 1
    try:
        result = producer()
    except BaseException:
        audit["errors"] += 1
        raise
    entries[key] = copy.deepcopy(result)
    return result, {
        "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
        "status": "miss",
        "key_sha256": key,
    }


def new_gradient_stage_cache():
    """Return one process-local cache shared by a single ``process_one`` run.

    The cache deliberately has no disk persistence and is not global.  Its
    entries are raw JSON-safe gradient-stage results, before clean-base turns
    proposal masks into mutable assembly records.
    """
    return {
        "schema": GRADIENT_STAGE_CACHE_SCHEMA,
        "entries": {},
        "geometry_fit_cache": {},
        "audit": {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "errors": 0,
        },
    }


def _normalise_gradient_stage_cache(cache):
    if cache is None:
        return None
    if not isinstance(cache, dict):
        raise TypeError("gradient_stage_cache must be a dict or None")
    if not cache:
        cache.update(new_gradient_stage_cache())
    if cache.get("schema") != GRADIENT_STAGE_CACHE_SCHEMA:
        raise ValueError("unsupported gradient-stage cache schema")
    entries = cache.get("entries")
    geometry_fit_cache = cache.setdefault("geometry_fit_cache", {})
    audit = cache.get("audit")
    if (not isinstance(entries, dict)
            or not isinstance(geometry_fit_cache, dict)
            or not isinstance(audit, dict)):
        raise ValueError("malformed gradient-stage cache")
    for name in ("requests", "hits", "misses", "bypasses", "errors"):
        audit[name] = int(audit.get(name, 0) or 0)
    return cache


def gradient_stage_cache_audit(cache):
    """Return a JSON-safe, entry-free audit for the final conversion report."""
    state = _normalise_gradient_stage_cache(cache)
    if state is None:
        return {
            "schema": GRADIENT_STAGE_CACHE_SCHEMA,
            "scope": "disabled",
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "errors": 0,
            "entry_count": 0,
            "keys_sha256": [],
        }
    result = {
        "schema": GRADIENT_STAGE_CACHE_SCHEMA,
        "scope": "single_process_one",
        **{name: int(state["audit"].get(name, 0) or 0)
           for name in ("requests", "hits", "misses", "bypasses", "errors")},
        "entry_count": len(state["entries"]),
        "keys_sha256": sorted(str(key) for key in state["entries"]),
    }
    geometry_state = state.get("geometry_fit_cache") or {}
    geometry_entries = geometry_state.get("entries") or {}
    geometry_audit = geometry_state.get("audit") or {}
    result["geometry_fit_cache"] = {
        "schema": geometry_state.get("schema"),
        **{name: int(geometry_audit.get(name, 0) or 0)
           for name in ("requests", "hits", "misses", "stores", "errors")},
        "entry_count": len(geometry_entries),
        "keys_sha256": sorted(str(key) for key in geometry_entries),
    }
    result["post_fit_scene_cache"] = _post_fit_scene_cache_audit(state)
    return result


def _gradient_stage_cache_bypass(cache, reason):
    state = _normalise_gradient_stage_cache(cache)
    if state is not None:
        state["audit"]["bypasses"] += 1
    return {
        "schema": GRADIENT_STAGE_CACHE_SCHEMA,
        "status": "bypassed",
        "reason": str(reason),
    }


def _gradient_stage_cache_key(arrays, parameters, stage_callable,
                              stage_schema):
    """Hash every value that can affect the source-space stage result.

    Array layout is normalised to C order, while name, dtype and shape remain
    part of the contract.  Callable identity is intentionally process-local:
    this cache must never be reused after a code reload or persisted to disk.
    """
    digest = hashlib.sha256()
    callable_identity = {
        "module": str(getattr(stage_callable, "__module__",
                              type(stage_callable).__module__)),
        "qualname": str(getattr(stage_callable, "__qualname__",
                                getattr(stage_callable, "__name__",
                                        type(stage_callable).__qualname__))),
        "process_identity": id(stage_callable),
    }
    contract = {
        "cache_schema": GRADIENT_STAGE_CACHE_SCHEMA,
        "stage_schema": str(stage_schema),
        "stage_callable": callable_identity,
        "parameters": parameters,
    }
    header = json.dumps(
        contract, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for name, value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        metadata = json.dumps({
            "name": str(name),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }, ensure_ascii=True, sort_keys=True,
            separators=(",", ":")).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        payload = memoryview(array).cast("B")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def _gradient_stage_cache_get_or_compute(cache, key, producer):
    """Return an isolated stage result; failed producers are never cached."""
    state = _normalise_gradient_stage_cache(cache)
    if state is None:
        return producer(), {
            "schema": GRADIENT_STAGE_CACHE_SCHEMA,
            "status": "disabled",
            "key_sha256": key,
        }
    audit = state["audit"]
    entries = state["entries"]
    audit["requests"] += 1
    if key in entries:
        audit["hits"] += 1
        return copy.deepcopy(entries[key]), {
            "schema": GRADIENT_STAGE_CACHE_SCHEMA,
            "status": "hit",
            "key_sha256": key,
        }
    audit["misses"] += 1
    try:
        result = producer()
    except BaseException:
        audit["errors"] += 1
        raise
    # Store a detached copy.  The caller may subsequently decode masks or add
    # assembly-only fields without mutating the value seen by a later candidate.
    entries[key] = copy.deepcopy(result)
    return result, {
        "schema": GRADIENT_STAGE_CACHE_SCHEMA,
        "status": "miss",
        "key_sha256": key,
    }


def _post_fit_scene_state(cache):
    parent = _normalise_gradient_stage_cache(cache)
    if parent is None:
        return None
    state = parent.setdefault("post_fit_scene_cache", {
        "schema": POST_FIT_SCENE_CACHE_SCHEMA, "entries": {}, "bytes": 0,
        "audit": {name: 0 for name in
                  ("requests", "hits", "misses", "stores", "evictions", "errors", "bypasses")},
    })
    if (state.get("schema") != POST_FIT_SCENE_CACHE_SCHEMA
            or not isinstance(state.get("entries"), dict)
            or not isinstance(state.get("audit"), dict)):
        raise ValueError("malformed post-fit-scene cache")
    return state


def _post_fit_scene_cache_audit(cache):
    state = _post_fit_scene_state(cache)
    return {
        "schema": POST_FIT_SCENE_CACHE_SCHEMA,
        "scope": "single_process_one" if state is not None else "disabled",
        **({**state["audit"]} if state is not None else {}),
        "entry_count": len(state["entries"]) if state is not None else 0,
        "estimated_payload_bytes": state["bytes"] if state is not None else 0,
        "maximum_entries": POST_FIT_SCENE_CACHE_MAX_ENTRIES,
        "maximum_estimated_payload_bytes": POST_FIT_SCENE_CACHE_MAX_BYTES,
        "keys_sha256": sorted(state["entries"]) if state is not None else [],
        "identity_scope": "exact_svg_source_bytes_decoded_rgba_references_fields_parameters_callable",
    }


def _post_fit_cache_value(value):
    """Typed exact fingerprints, including nested proposal masks and proofs."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not cacheable")
        array = np.ascontiguousarray(value)
        return ["array", array.dtype.str, list(array.shape),
                hashlib.sha256(array.tobytes()).hexdigest()]
    if isinstance(value, np.generic):
        return ["numpy_scalar", value.dtype.str, _post_fit_cache_value(value.item())]
    if isinstance(value, dict):
        rows = [[_post_fit_cache_value(k), _post_fit_cache_value(v)] for k, v in value.items()]
        rows.sort(key=lambda row: json.dumps(row[0], sort_keys=True, ensure_ascii=True))
        return ["dict", rows]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_post_fit_cache_value(v) for v in value]]
    if isinstance(value, bytes):
        return ["bytes", len(value), hashlib.sha256(value).hexdigest()]
    if value is None or isinstance(value, (str, bool, int)):
        return [type(value).__name__, value]
    if isinstance(value, float) and math.isfinite(value):
        return ["float", value.hex()]
    raise TypeError("unsupported post-fit cache input type")


def _post_fit_scene_key(operation, svg_text, source_path, processed_rgba,
                        fields, parameters, stage_callable, native_reference_rgba=None):
    raw = Path(source_path).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        original = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    contract = _post_fit_cache_value({
        "schema": POST_FIT_SCENE_CACHE_SCHEMA, "operation": operation,
        "svg_utf8": svg_text.encode("utf-8"),
        "source_path": str(Path(source_path).resolve()),
        "source_file_bytes": raw, "source_decoded_rgba": original,
        "processed_rgba": processed_rgba, "native_reference_rgba": native_reference_rgba,
        "fields": fields, "parameters": parameters,
        "callable": (str(getattr(stage_callable, "__module__", "")),
                     str(getattr(stage_callable, "__qualname__", type(stage_callable).__qualname__)),
                     id(stage_callable)),
    })
    return hashlib.sha256(json.dumps(contract, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False).encode("ascii")).hexdigest()


def _post_fit_result_size(value):
    """Conservative payload accounting, including nested arrays and reports."""
    if isinstance(value, np.ndarray):
        return 256 + value.nbytes
    if isinstance(value, dict):
        return 256 + sum(128 + _post_fit_result_size(k) + _post_fit_result_size(v)
                         for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return 256 + sum(64 + _post_fit_result_size(v) for v in value)
    if isinstance(value, str):
        return 128 + 4 * len(value)
    if isinstance(value, bytes):
        return 128 + len(value)
    if value is None or isinstance(value, (bool, int, float, np.generic)):
        return 128
    raise TypeError("unsupported post-fit cache result type")


def _cached_post_fit_scene(cache, *, operation, svg_text, source_path, processed_rgba,
                           fields, parameters, stage_callable, producer,
                           native_reference_rgba=None):
    """Reuse only this job's identical full-scene transaction, never its neighbours.

    Both successful and normal rejected results may be reused. Exceptions and
    inputs changing during a transaction are not cached. Existing certificates
    remain untouched; the separate cache record distinguishes reused evidence
    from native validation executed during this invocation.
    """
    started = time.perf_counter()
    state = _post_fit_scene_state(cache)
    evidence = {"schema": POST_FIT_SCENE_CACHE_SCHEMA, "operation": operation,
                "status": "disabled", "producer_executed": True,
                "validation_reused_for_exact_inputs": False}
    def finish(result):
        evidence["invocation_elapsed_seconds"] = time.perf_counter() - started
        return result, evidence
    if state is None:
        return finish(producer())
    audit, entries = state["audit"], state["entries"]
    audit["requests"] += 1
    def key_now():
        return _post_fit_scene_key(operation, svg_text, source_path, processed_rgba,
            fields, parameters, stage_callable, native_reference_rgba)
    try:
        key = key_now()
    except (OSError, TypeError, ValueError):
        audit["bypasses"] += 1
        evidence.update(status="bypassed", reason="exact_input_snapshot_unavailable")
        return finish(producer())
    evidence["key_sha256"] = key
    if key in entries:
        audit["hits"] += 1
        entry = entries[key]
        evidence.update(status="hit", producer_executed=False,
            validation_reused_for_exact_inputs=True,
            original_producer_elapsed_seconds=entry["producer_elapsed_seconds"])
        return finish(copy.deepcopy(entry["value"]))
    audit["misses"] += 1
    evidence["status"] = "miss"
    validation_started = time.perf_counter()
    try:
        result = producer()
    except BaseException:
        audit["errors"] += 1
        raise
    elapsed = time.perf_counter() - validation_started
    evidence["original_producer_elapsed_seconds"] = elapsed
    try:
        if key_now() != key:
            raise ValueError("inputs_changed_during_transaction")
        size = _post_fit_result_size(result)
        if size > POST_FIT_SCENE_CACHE_MAX_BYTES:
            raise ValueError("result_exceeds_bounded_cache")
        detached = copy.deepcopy(result)
    except (OSError, TypeError, ValueError) as exc:
        audit["bypasses"] += 1
        evidence.update(stored=False, storage_reason=str(exc)[:160])
        return finish(result)
    while entries and (len(entries) >= POST_FIT_SCENE_CACHE_MAX_ENTRIES
                       or state["bytes"] + size > POST_FIT_SCENE_CACHE_MAX_BYTES):
        oldest = entries.pop(next(iter(entries)))
        state["bytes"] -= oldest["size_bytes"]
        audit["evictions"] += 1
    entries[key] = {"value": detached, "size_bytes": size,
                    "producer_elapsed_seconds": elapsed}
    state["bytes"] += size
    audit["stores"] += 1
    evidence["stored"] = True
    return finish(result)


@dataclass
class CleanBaseStats:
    width: int
    height: int
    colors: int
    palette: list                       # [(group_name, hex), ...]
    removed_background: bool
    geometry_notes: list = field(default_factory=list)
    n_paths: int = 0
    n_native: int = 0            # native <circle> elements
    n_strokes: int = 0           # rebuilt center-line strokes
    n_nodes: int = 0             # total anchor/segment count (editability)
    stroke_info: list = field(default_factory=list)
    n_gradients: int = 0         # banded ramps rebuilt as linearGradient
    gradient_info: list = field(default_factory=list)
    component_repair: dict = field(default_factory=dict)
    viewbox: list = field(default_factory=list)   # [W, H] trace coordinates
    palette_audit: dict = field(default_factory=dict)
    # Private, compact provenance for broad background-coloured pockets that
    # the stroke-aware hole guard classified as negative space.  Candidate
    # validation runs after this function returns; carrying the exact mask lets
    # it canonicalize the metric/source reference without re-guessing from
    # colour alone (which would endanger genuine white lettering).  Packbits
    # keeps a 2048px trace mask small when several candidates are retained.
    _validation_hole_shape: tuple = field(
        default_factory=tuple, repr=False, compare=False)
    _validation_hole_bits: bytes = field(
        default=b"", repr=False, compare=False)

    def _validation_hole_mask(self):
        """Return the private stroke-proven negative-space mask, if any."""
        if len(self._validation_hole_shape) != 2 or not self._validation_hole_bits:
            return None
        height, width = (int(value) for value in self._validation_hole_shape)
        if height <= 0 or width <= 0:
            return None
        packed = np.frombuffer(self._validation_hole_bits, dtype=np.uint8)
        unpacked = np.unpackbits(packed, bitorder="little")
        required = height * width
        if unpacked.size < required:
            return None
        return unpacked[:required].reshape((height, width)).astype(bool)


# ---------- Palette detection ----------

def _unique_colors(pixels):
    """Return (unique_colors float32 [U,3], weights float64 [U])."""
    q = np.clip(np.asarray(pixels).round(), 0, 255).astype(np.uint8)
    uniq, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    return uniq.astype(np.float32), counts.astype(np.float64)


def _kmeans_pp_init(uniq, w, k, rng):
    """k-means++ seeding over unique colors, starting from the dominant color."""
    idx = [int(np.argmax(w))]
    d2 = ((uniq - uniq[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        probs = w * d2
        s = probs.sum()
        if s <= 0:
            break
        nxt = int(rng.choice(len(uniq), p=probs / s))
        idx.append(nxt)
        d2 = np.minimum(d2, ((uniq - uniq[nxt]) ** 2).sum(1))
    return uniq[list(dict.fromkeys(idx))].copy()


def _kmeans_from_unique(uniq, w, k, iters=30, seed=0):
    """Weighted k-means over a prepared unique-colour histogram."""
    uniq = np.asarray(uniq, dtype=np.float32)
    w = np.asarray(w, dtype=np.float64)
    k = max(1, min(int(k), len(uniq)))
    if len(uniq) <= k:
        return uniq.copy()
    rng = np.random.default_rng(seed)
    # Keep clustering memory bounded on photographic inputs while retaining
    # both dominant colours and a deterministic spread of the colour gamut.
    # The former implementation kept only the most frequent colours; with a
    # smooth ramp (many equally frequent colours) that biased the sample to
    # one lexicographic end of the gamut.
    if len(uniq) > 100000:
        ranked = np.argsort(-w, kind="stable")
        dominant = ranked[:50000]
        remainder = np.sort(ranked[50000:])
        spread_at = np.linspace(0, len(remainder) - 1, 50000,
                                dtype=np.int64)
        keep = np.unique(np.concatenate([dominant, remainder[spread_at]]))
        uniq, w = uniq[keep], w[keep]
        k = max(1, min(k, len(uniq)))
    cent = _kmeans_pp_init(uniq, w, k, rng)
    for _ in range(iters):
        d = ((uniq[:, None] - cent[None]) ** 2).sum(2)
        lab = d.argmin(1)
        new = np.empty_like(cent)
        reseeded = False
        for i in range(len(cent)):
            m = lab == i
            if not np.any(m):
                err = w * d.min(1)
                new[i] = uniq[int(np.argmax(err))]
                reseeded = True
                continue
            wi = w[m]
            new[i] = (uniq[m] * wi[:, None]).sum(0) / wi.sum()
        if not reseeded and np.allclose(new, cent, atol=0.3):
            cent = new
            break
        cent = new
    return cent


def _kmeans(pixels, k, iters=30, seed=0):
    """Weighted k-means over unique colors with k-means++ init.

    Robust to tiny images (k is clamped to the number of unique colors) and
    to dominant-color images (weighted seeding cannot collapse clusters).
    Empty clusters are re-seeded at the point of largest weighted error.
    """
    uniq, w = _unique_colors(pixels)
    return _kmeans_from_unique(uniq, w, k, iters=iters, seed=seed)


def _cluster_shares(cent, uniq, w):
    lab = ((uniq[:, None] - cent[None]) ** 2).sum(2).argmin(1)
    shares = np.array([w[lab == i].sum() for i in range(len(cent))])
    total = shares.sum()
    return shares / total if total > 0 else shares


def _merge_close(cent, uniq, w, thresh):
    cent = cent.astype(np.float32)
    while len(cent) > 1:
        lab = ((uniq[:, None] - cent[None]) ** 2).sum(2).argmin(1)
        counts = np.array([w[lab == i].sum() for i in range(len(cent))])
        best = None
        for i in range(len(cent)):
            for j in range(i + 1, len(cent)):
                d = float(((cent[i] - cent[j]) ** 2).sum() ** 0.5)
                if d < thresh and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        wi, wj = max(counts[i], 1.0), max(counts[j], 1.0)
        merged = (cent[i] * wi + cent[j] * wj) / (wi + wj)
        cent = np.delete(cent, [i, j], axis=0)
        cent = np.vstack([cent, merged])
    return cent


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = float((ab * ab).sum())
    t = 0.0 if denom == 0 else float(np.clip(((p - a) * ab).sum() / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _prune_blend_clusters(cent, uniq, w, share_limit=0.035, seg_dist=28.0):
    """Drop tiny clusters that sit on the RGB segment between two larger
    clusters — those are antialiasing blends, not design colors. Small but
    chromatically distinct accent colors are kept."""
    cent = cent.astype(np.float32)
    while len(cent) > 2:
        shares = _cluster_shares(cent, uniq, w)
        victim = None
        for i in np.argsort(shares):
            if shares[i] >= share_limit:
                break
            for a in range(len(cent)):
                if a == i or shares[a] < 3 * shares[i]:
                    continue
                for b in range(len(cent)):
                    if b in (i, a) or shares[b] < 3 * shares[i]:
                        continue
                    if _point_segment_dist(cent[i], cent[a], cent[b]) < seg_dist:
                        victim = i
                        break
                if victim is not None:
                    break
            if victim is not None:
                break
        if victim is None:
            break
        cent = np.delete(cent, victim, axis=0)
    return cent


def _palette_fit_stats(uniq, w, cent):
    """Weighted RGB fitting error without expanding the pixel histogram."""
    uniq = np.asarray(uniq, dtype=np.float32)
    w = np.asarray(w, dtype=np.float64)
    cent = np.asarray(cent, dtype=np.float32).reshape(-1, 3)
    if not len(uniq) or not len(cent) or float(w.sum()) <= 0:
        return {"mean": 0.0, "p90": 0.0, "p95": 0.0,
                "over_25_share": 0.0}
    error = np.empty(len(uniq), dtype=np.float32)
    # A photo can contain millions of unique colours.  Keep the audit itself
    # bounded; otherwise deciding whether to use 16 colours could consume more
    # memory than the trace it is meant to improve.
    for start in range(0, len(uniq), 65536):
        block = uniq[start:start + 65536]
        error[start:start + len(block)] = np.sqrt(
            ((block[:, None] - cent[None]) ** 2).sum(2).min(1))
    total = float(w.sum())
    order = np.argsort(error, kind="stable")
    cumulative = np.cumsum(w[order])

    def _weighted_percentile(q):
        idx = int(np.searchsorted(cumulative, q * total, side="left"))
        return float(error[order[min(idx, len(order) - 1)]])

    return {
        "mean": round(float((error * w).sum() / total), 3),
        "p90": round(_weighted_percentile(0.90), 3),
        "p95": round(_weighted_percentile(0.95), 3),
        "over_25_share": round(float(w[error > 25.0].sum() / total), 4),
    }


def _nearest_palette_labels(img, palette, chunk=131072):
    """Assign palette labels in bounded memory (20 colours can be sizeable)."""
    if int(chunk) == 131072:
        # The operation-level backend uses an integer WebGPU shader only for
        # exactly representable RGB inputs and otherwise executes this same
        # float32 NumPy contract.  Keeping the import local avoids any GPU
        # package/driver startup on code paths that never label a palette.
        from compute_backend import nearest_palette_labels
        return nearest_palette_labels(img, palette)
    pixels = np.asarray(img, dtype=np.float32).reshape(-1, 3)
    palette = np.asarray(palette, dtype=np.float32).reshape(-1, 3)
    out = np.empty(len(pixels), dtype=np.int16)
    for start in range(0, len(pixels), int(chunk)):
        block = pixels[start:start + int(chunk)]
        out[start:start + len(block)] = (
            ((block[:, None] - palette[None]) ** 2).sum(2).argmin(1))
    return out.reshape(np.asarray(img).shape[:2])


def detect_palette(den, visible, forced=0, max_k=8, merge_thresh=45,
                   return_audit=False, return_labels=True):
    pix = den[visible].astype(np.float32)
    uniq, w = _unique_colors(pix)
    if forced and forced >= 2:
        cent = _kmeans_from_unique(uniq, w, int(forced))
        audit = {
            "mode": "forced", "base_colors": int(len(cent)),
            "selected_colors": int(len(cent)),
            "fit_before": _palette_fit_stats(uniq, w, cent),
        }
    else:
        base_k = min(int(max_k), max(2, len(uniq)))
        cent = _kmeans_from_unique(uniq, w, base_k)
        cent = _merge_close(cent, uniq, w, merge_thresh)
        cent = _prune_blend_clusters(cent, uniq, w)
        base_cent = cent.copy()
        base_fit = _palette_fit_stats(uniq, w, base_cent)
        candidates = [{"colors": int(len(base_cent)), **base_fit}]

        # A flat logo should stay compact and easy to edit.  A genuinely
        # continuous/textured logo, however, cannot be represented honestly by
        # the old hard 8-colour ceiling.  Expand only when BOTH mean and tail
        # error say that 8 colours are visibly inadequate, and stop at the
        # smallest richer palette that meets the target.  Twenty is the final
        # bounded step: it can rescue a difficult ramp without jumping to the
        # visibly more fragmented 24-colour result.
        adaptive = (int(max_k) >= 8 and len(uniq) > len(base_cent)
                    and base_fit["mean"] > 13.0
                    and base_fit["p90"] > 25.0)
        selected_fit = base_fit
        if adaptive:
            best_cent, best_fit = base_cent, base_fit
            for richer_k in (12, 16, 20):
                richer_k = min(richer_k, len(uniq))
                if richer_k <= len(base_cent):
                    continue
                richer = _kmeans_from_unique(uniq, w, richer_k)
                # Only collapse virtually duplicate centres.  Ordinary blend
                # pruning would remove the very ramp stops this branch exists
                # to preserve.
                richer = _merge_close(richer, uniq, w, 10.0)
                richer = _prune_blend_clusters(
                    richer, uniq, w, share_limit=0.0015, seg_dist=6.0)
                fit = _palette_fit_stats(uniq, w, richer)
                candidates.append({"colors": int(len(richer)), **fit})
                material = (
                    (fit["mean"] <= best_fit["mean"] - 1.0
                     or fit["p90"] <= best_fit["p90"] - 2.0)
                    and fit["mean"] <= best_fit["mean"] + 0.25
                    and fit["p90"] <= best_fit["p90"] + 0.5)
                if material:
                    best_cent, best_fit = richer, fit
                if fit["mean"] <= 13.0 and fit["p90"] <= 25.0:
                    best_cent, best_fit = richer, fit
                    break
            cent, selected_fit = best_cent, best_fit
        audit = {
            "mode": "adaptive" if len(cent) > len(base_cent) else "compact",
            "base_colors": int(len(base_cent)),
            "selected_colors": int(len(cent)),
            "fit_before": base_fit,
            "fit_after": selected_fit,
            "candidates": candidates,
        }
    cent_i = np.clip(np.round(cent), 0, 255).astype(np.uint8)
    lab_all = (_nearest_palette_labels(den, cent_i)
               if return_labels else None)
    if return_audit:
        return cent_i, lab_all, audit
    return cent_i, lab_all


def _allow_palette_tier_b_strokes(palette_audit):
    """Return whether per-palette stroke recovery is visually trustworthy.

    Tier B is useful for flat logos where a real line crosses another fill.
    On continuous-tone artwork that needed 12+ adaptive colours, however, the
    thin quantisation bands inside glyphs and gradients look exactly like
    one-pixel strokes.  Keep the global Tier-A lines, but leave those colour
    fragments with the fill tracer so letter silhouettes stay intact.
    """
    audit = palette_audit or {}
    return not (
        audit.get("mode") == "adaptive"
        and int(audit.get("selected_colors") or 0) >= 12
    )


def _stabilize_fragmented_linear_details(
        den, visible, palette, labels, bg_color, owned_mask=None,
        *, core_threshold=12.0, min_area=21, max_area=800):
    """Collapse only tiny, ruler-straight multi-palette details to one paint.

    A one- or two-pixel ray can cross several adaptive palette bands.  A
    cutout tracer then sees each band as a separate speck and may drop the
    entire ray even though the original strong-ink component is continuous.
    PCA distinguishes those details from glyphs and illustrated ribbons: the
    perpendicular sigma must stay below 1.5px and the component must be both
    extremely linear and sparsely filled inside its bounding box.
    """

    from stroke_engine import connected_components

    den = np.asarray(den, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    palette = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    result = np.asarray(labels).copy()
    owned = (np.zeros_like(visible) if owned_mask is None
             else np.asarray(owned_mask, dtype=bool))
    bg = np.asarray(bg_color, dtype=np.float32).reshape(1, 1, 3)
    strength = np.abs(den - bg).max(axis=2)
    components, count = connected_components(
        visible & ~owned & (strength >= float(core_threshold)))
    areas = np.bincount(components.ravel(), minlength=count + 1)
    records = []
    changed_pixels = 0

    for component in range(1, count + 1):
        area = int(areas[component])
        if area < int(min_area) or area > int(max_area):
            continue
        yy, xx = np.nonzero(components == component)
        if not len(xx):
            continue
        width = int(xx.max() - xx.min() + 1)
        height = int(yy.max() - yy.min() + 1)
        if max(width, height) < 12:
            continue
        points = np.column_stack((xx, yy)).astype(np.float64)
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        perpendicular_sigma = math.sqrt(max(0.0, float(eigenvalues[0])))
        linearity = float(eigenvalues[-1]) / max(float(eigenvalues[0]), 0.25)
        fill_ratio = area / float(max(1, width * height))
        if (perpendicular_sigma > 1.5 or linearity < 100.0
                or fill_ratio > 0.28):
            continue

        component_labels = result[yy, xx].astype(np.int64)
        counts = np.bincount(component_labels, minlength=len(palette))
        dominant_share = float(counts.max()) / max(1, area)
        if dominant_share >= 0.85:
            continue
        # A one-pixel antialiased line contains many pale edge samples.  The
        # median therefore points toward the paper/background colour and can
        # select a low-chroma grey even when the actual line is gold or blue.
        # Estimate the paint from the strongest 15% of the component instead:
        # those pixels are closest to the opaque centre-line colour while the
        # full component is still used for the geometry decision above.
        component_strength = strength[yy, xx]
        core_cutoff = float(np.quantile(component_strength, 0.85))
        colour_core = den[yy[component_strength >= core_cutoff],
                          xx[component_strength >= core_cutoff]]
        source_median = np.median(
            colour_core if len(colour_core) else den[yy, xx], axis=0)
        palette_index = int(
            ((palette.astype(np.float32) - source_median) ** 2).sum(1).argmin())
        changed = int((component_labels != palette_index).sum())
        if changed <= 0:
            continue
        result[yy, xx] = palette_index
        changed_pixels += changed
        records.append({
            "source_component": int(component),
            "area_px": area,
            "bbox_px": [int(xx.min()), int(yy.min()), width, height],
            "linearity": round(linearity, 3),
            "perpendicular_sigma_px": round(perpendicular_sigma, 3),
            "old_dominant_share": round(dominant_share, 4),
            "colour_core_quantile": 0.85,
            "paint": "#{:02x}{:02x}{:02x}".format(
                *(int(value) for value in palette[palette_index])),
            "pixels_relabelled": changed,
        })

    return result, {
        "policy": "small_strong_ink_pca_linear_multicolour_only",
        "components_stabilized": len(records),
        "pixels_relabelled": int(changed_pixels),
        "core_threshold": float(core_threshold),
        "max_perpendicular_sigma_px": 1.5,
        "minimum_linearity": 100.0,
        "components": records,
    }


def _retain_initial_accent_colors(
        den, visible, initial_palette, fill_palette, labels, *,
        max_colors=2, minimum_improvement=8.0,
        minimum_component_area=128, minimum_foreground_share=0.002):
    """Restore a small, coherent accent lost by fill-only re-clustering.

    Stroke extraction can remove most samples of a real accent (for example
    two brown rules), causing the remaining brown leaf to disappear from the
    re-estimated fill palette.  Only colours already proven by the initial
    whole-art palette are eligible, and only large connected ownership
    regions with a material residual improvement are recoloured.
    """

    from stroke_engine import connected_components

    den = np.asarray(den, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    initial = np.asarray(initial_palette, dtype=np.uint8).reshape(-1, 3)
    palette = np.asarray(fill_palette, dtype=np.uint8).reshape(-1, 3).copy()
    result = np.asarray(labels).copy()
    foreground_pixels = int(visible.sum())
    minimum_total = max(
        int(minimum_component_area),
        int(math.ceil(float(minimum_foreground_share) * foreground_pixels)),
    )
    records = []
    if foreground_pixels and int(max_colors) > 0:
        current_rgb = palette[result].astype(np.float32)
        current_error = np.sqrt(((den - current_rgb) ** 2).sum(axis=2))
        candidate_colours = []
        for colour in initial:
            colour_tuple = tuple(int(v) for v in colour)
            candidate = np.asarray(colour_tuple, dtype=np.float32)
            palette_distance = np.sqrt(
                ((palette.astype(np.float32) - candidate) ** 2).sum(axis=1))
            if len(palette_distance) and float(palette_distance.min()) < 8.0:
                continue
            candidate_colours.append(colour_tuple)

        # Assign every pixel to at most one missing initial colour before any
        # component labelling.  The old implementation flood-filled the full
        # canvas once per candidate (and again after each acceptance), which
        # turned a two-colour guard into a minute-long step on a 1254px logo.
        best_candidate = np.full(visible.shape, -1, dtype=np.int16)
        best_gain = np.zeros(visible.shape, dtype=np.float32)
        for index, colour in enumerate(candidate_colours):
            candidate = np.asarray(colour, dtype=np.float32)
            candidate_error = np.sqrt(((den - candidate) ** 2).sum(axis=2))
            improvement = current_error - candidate_error
            better = visible & (improvement > best_gain)
            best_gain[better] = improvement[better]
            best_candidate[better] = index

        ranked = []
        for index, colour in enumerate(candidate_colours):
            strong = ((best_candidate == index)
                      & (best_gain >= float(minimum_improvement)))
            strong_pixels = int(strong.sum())
            if strong_pixels < minimum_total:
                continue
            ranked.append((float(best_gain[strong].sum()), strong_pixels,
                           index, colour))
        ranked.sort(reverse=True)

        for _rough_benefit, _rough_pixels, index, colour in ranked:
            if len(records) >= int(max_colors):
                break
            candidate = np.asarray(colour, dtype=np.float32)
            if np.sqrt(((palette.astype(np.float32) - candidate) ** 2)
                       .sum(axis=1)).min() < 8.0:
                continue
            owner = (best_candidate == index) & visible & (best_gain > 0)
            strong = owner & (best_gain >= float(minimum_improvement))
            owner_y, owner_x = np.nonzero(owner)
            if not len(owner_y):
                continue
            y0, y1 = int(owner_y.min()), int(owner_y.max()) + 1
            x0, x1 = int(owner_x.min()), int(owner_x.max()) + 1
            owner_crop = owner[y0:y1, x0:x1]
            strong_crop = strong[y0:y1, x0:x1]
            owner_labels, owner_count = connected_components(owner_crop)
            keep_ids = []
            strong_pixels = 0
            for component in range(1, owner_count + 1):
                strong_count = int(
                    (strong_crop & (owner_labels == component)).sum())
                if strong_count >= int(minimum_component_area):
                    keep_ids.append(component)
                    strong_pixels += strong_count
            if strong_pixels < minimum_total or not keep_ids:
                continue
            accepted = np.zeros_like(visible)
            accepted[y0:y1, x0:x1] = np.isin(owner_labels, keep_ids)
            benefit = float(best_gain[accepted].sum())
            mean_gain = float(best_gain[accepted].mean())
            palette_index = len(palette)
            palette = np.vstack([palette, np.asarray(colour, dtype=np.uint8)])
            result[accepted] = palette_index
            records.append({
                "paint": "#{:02x}{:02x}{:02x}".format(*colour),
                "assigned_pixels": int(accepted.sum()),
                "strong_improvement_pixels": int(strong_pixels),
                "mean_rgb_error_improvement": round(mean_gain, 3),
                "total_error_improvement": round(benefit, 3),
            })

    return palette, result, {
        "policy": "initial_palette_connected_residual_retention",
        "maximum_colors": int(max_colors),
        "minimum_rgb_error_improvement": float(minimum_improvement),
        "minimum_component_area_px": int(minimum_component_area),
        "minimum_foreground_share": float(minimum_foreground_share),
        "colors_retained": len(records),
        "records": records,
    }


def color_name(rgb):
    r, g, b = (int(v) for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx / 255.0
    s = 0 if mx == 0 else (mx - mn) / mx
    if v < 0.18:
        return "black"
    if s < 0.12 and v > 0.9:
        return "white"
    if s < 0.15:
        return "gray"
    if mx == r:
        h = (60 * ((g - b) / (mx - mn))) % 360
    elif mx == g:
        h = 60 * ((b - r) / (mx - mn)) + 120
    else:
        h = 60 * ((r - g) / (mx - mn)) + 240
    names = [(15, "red"), (45, "orange"), (70, "yellow"), (95, "yellow-green"),
             (160, "green"), (200, "cyan"), (255, "blue"), (290, "purple"),
             (340, "magenta"), (360, "red")]
    for lim, nm in names:
        if h < lim:
            return nm
    return "color"


_STROKE_PALETTE_SNAP_DISTANCE = 48.0


def _snap_stroke_color_to_palette(rgb, palette,
                                  max_distance=_STROKE_PALETTE_SNAP_DISTANCE):
    """Return a nearby canonical palette color, otherwise keep ``rgb``.

    Original-resolution sampling recovers the real core of a sub-pixel line,
    but a JPEG may give each sampled line a slightly different compression
    color.  Snapping only inside a small Euclidean RGB neighbourhood removes
    those accidental variants.  A genuinely recovered black line stays black
    when the downsampled palette contains only a much lighter gray.
    """
    sampled = np.clip(np.asarray(rgb, dtype=np.float32), 0, 255)
    pal = np.asarray(palette, dtype=np.float32).reshape(-1, 3)
    if pal.size == 0:
        return tuple(int(round(v)) for v in sampled)
    dist2 = ((pal - sampled) ** 2).sum(axis=1)
    nearest = int(np.argmin(dist2))
    if float(dist2[nearest]) <= float(max_distance) ** 2:
        sampled = pal[nearest]
    return tuple(int(round(v)) for v in sampled)


def _gradient_palette_hex(gradient):
    """Choose a real, central stop color for palette/report presentation."""
    stops = gradient.get("stops") or []
    if not stops:
        # A detected gradient is required to have real stops.  Failing here
        # is safer than leaking its routing sentinel into user-facing data.
        raise ValueError("gradient region has no real color stops")
    central = min(
        stops,
        key=lambda stop: abs(float(
            stop.get("offset", 0.0) if isinstance(stop, dict) else stop[0])
            - 0.5),
    )
    if isinstance(central, dict):
        colour = central.get("color")
        if isinstance(colour, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
            return colour.lower()
        rgb = central.get("rgb")
    else:
        _, rgb = central
    rgb = tuple(int(np.clip(round(float(v)), 0, 255)) for v in rgb)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


_NEAR_CIRCLE_NOTE_RE = re.compile(
    r"^\d+ near-circular shapes replaced with perfect circles$")


def _finalize_circle_geometry_note(notes, emitted_fill_circles):
    """Replace the detection-time circle count with the emitted SVG count."""
    replacement = (
        f"{int(emitted_fill_circles)} near-circular fill shapes emitted as "
        "native SVG circles")
    out = []
    replaced = False
    for note in notes:
        if _NEAR_CIRCLE_NOTE_RE.fullmatch(note):
            if not replaced:
                out.append(replacement)
                replaced = True
            continue
        out.append(note)
    return out


# ---------- 路徑解析（含相對指令，全部轉絕對座標） ----------

_TOKEN_RE = re.compile(r"[A-Za-z]|-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "A": 7}


def _f(v):
    v = float(v)
    if abs(v) < 1e-6:
        v = 0.0
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _parse_subpaths(d):
    """把 d 解析成子路徑清單，所有座標轉為絕對值。"""
    tokens = _TOKEN_RE.findall(d)
    subs, sub = [], None
    cur = (0.0, 0.0)
    last_c2 = None
    last_q = None
    i = 0

    def finish(closed):
        nonlocal sub
        if sub and sub["segs"]:
            sub["closed"] = closed
            subs.append(sub)
        sub = None

    while i < len(tokens):
        t = tokens[i]
        if not t[0].isalpha():
            i += 1
            continue
        cmd = t
        i += 1
        up = cmd.upper()
        rel = cmd.islower()
        if up == "Z":
            if sub:
                cur = sub["start"]
            finish(True)
            last_c2 = last_q = None
            continue
        if up not in _ARITY:
            raise ValueError(f"unsupported path command: {cmd}")
        n = _ARITY[up]
        first = True
        while i + n <= len(tokens) and not tokens[i][0].isalpha():
            vals = [float(tokens[i + k]) for k in range(n)]
            i += n
            if up == "M":
                x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                if first:
                    finish(False)
                    sub = {"start": (x, y), "segs": []}
                else:
                    sub["segs"].append(["L", x, y])
                cur = (x, y)
                last_c2 = last_q = None
            elif up == "L":
                x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                sub["segs"].append(["L", x, y]); cur = (x, y)
                last_c2 = last_q = None
            elif up == "H":
                x = vals[0] + (cur[0] if rel else 0)
                sub["segs"].append(["L", x, cur[1]]); cur = (x, cur[1])
                last_c2 = last_q = None
            elif up == "V":
                y = vals[0] + (cur[1] if rel else 0)
                sub["segs"].append(["L", cur[0], y]); cur = (cur[0], y)
                last_c2 = last_q = None
            elif up == "C":
                c1x, c1y, c2x, c2y, x, y = vals
                if rel:
                    c1x += cur[0]; c1y += cur[1]; c2x += cur[0]; c2y += cur[1]
                    x += cur[0]; y += cur[1]
                sub["segs"].append(["C", c1x, c1y, c2x, c2y, x, y])
                cur = (x, y); last_c2 = (c2x, c2y); last_q = None
            elif up == "S":
                c2x, c2y, x, y = vals
                if rel:
                    c2x += cur[0]; c2y += cur[1]; x += cur[0]; y += cur[1]
                c1 = (2 * cur[0] - last_c2[0], 2 * cur[1] - last_c2[1]) if last_c2 else cur
                sub["segs"].append(["C", c1[0], c1[1], c2x, c2y, x, y])
                cur = (x, y); last_c2 = (c2x, c2y); last_q = None
            elif up == "Q":
                qx, qy, x, y = vals
                if rel:
                    qx += cur[0]; qy += cur[1]; x += cur[0]; y += cur[1]
                sub["segs"].append(["Q", qx, qy, x, y])
                cur = (x, y); last_q = (qx, qy); last_c2 = None
            elif up == "T":
                x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                q = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) if last_q else cur
                sub["segs"].append(["Q", q[0], q[1], x, y])
                cur = (x, y); last_q = q; last_c2 = None
            elif up == "A":
                rx, ry, rot, laf, sf, x, y = vals
                if rel:
                    x += cur[0]; y += cur[1]
                sub["segs"].append(["A", rx, ry, rot, laf, sf, x, y])
                cur = (x, y); last_c2 = last_q = None
            first = False
    finish(False)
    return subs


def _offset_subs(subs, tx, ty):
    if tx == 0 and ty == 0:
        return
    for sub in subs:
        sub["start"] = (sub["start"][0] + tx, sub["start"][1] + ty)
        for s in sub["segs"]:
            if s[0] == "L":
                s[1] += tx; s[2] += ty
            elif s[0] == "C":
                s[1] += tx; s[2] += ty; s[3] += tx; s[4] += ty; s[5] += tx; s[6] += ty
            elif s[0] == "Q":
                s[1] += tx; s[2] += ty; s[3] += tx; s[4] += ty
            elif s[0] == "A":
                s[6] += tx; s[7] += ty


def _emit_sub(sub):
    parts = [f"M{_f(sub['start'][0])} {_f(sub['start'][1])}"]
    for s in sub["segs"]:
        c = s[0]
        if c == "L":
            parts.append(f"L{_f(s[1])} {_f(s[2])}")
        elif c == "C":
            parts.append("C" + " ".join(_f(v) for v in s[1:]))
        elif c == "Q":
            parts.append("Q" + " ".join(_f(v) for v in s[1:]))
        elif c == "A":
            parts.append(f"A{_f(s[1])} {_f(s[2])} {_f(s[3])} {int(s[4])} {int(s[5])} "
                         f"{_f(s[6])} {_f(s[7])}")
    if sub.get("closed", True):
        parts.append("Z")
    return " ".join(parts)


def _sub_bbox_accumulate(sub, xs, ys):
    xs.append(sub["start"][0]); ys.append(sub["start"][1])
    for s in sub["segs"]:
        if s[0] == "L":
            xs.append(s[1]); ys.append(s[2])
        elif s[0] == "C":
            xs += [s[1], s[3], s[5]]; ys += [s[2], s[4], s[6]]
        elif s[0] == "Q":
            xs += [s[1], s[3]]; ys += [s[2], s[4]]
        elif s[0] == "A":
            # 圓弧保守外框：端點 ± 2r 必包住整段弧
            xs += [s[6] - 2 * s[1], s[6] + 2 * s[1]]
            ys += [s[7] - 2 * s[2], s[7] + 2 * s[2]]


# ---------- 幾何規則化 ----------

def _anchors_of(sub):
    pts = []
    for s in sub["segs"]:
        if s[0] == "L":
            pts.append((s[1], s[2]))
        elif s[0] == "C":
            pts.append((s[5], s[6]))
        elif s[0] == "Q":
            pts.append((s[3], s[4]))
        elif s[0] == "A":
            pts.append((s[6], s[7]))
    return pts


def _samples_of(sub):
    pts = []
    prev = sub["start"]
    for s in sub["segs"]:
        if s[0] == "C":
            mx = (prev[0] + 3 * s[1] + 3 * s[3] + s[5]) / 8.0
            my = (prev[1] + 3 * s[2] + 3 * s[4] + s[6]) / 8.0
            pts.append((mx, my))
            prev = (s[5], s[6])
        elif s[0] == "Q":
            mx = (prev[0] + 2 * s[1] + s[3]) / 4.0
            my = (prev[1] + 2 * s[2] + s[4]) / 4.0
            pts.append((mx, my))
            prev = (s[3], s[4])
        elif s[0] == "L":
            pts.append(((prev[0] + s[1]) / 2.0, (prev[1] + s[2]) / 2.0))
            prev = (s[1], s[2])
        elif s[0] == "A":
            prev = (s[6], s[7])
        pts.append(prev)
    return pts


def _kasa(pts):
    A = np.c_[2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy = float(sol[0]), float(sol[1])
    r2 = float(sol[2]) + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return cx, cy, math.sqrt(r2)


def _fit_circle(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 4:
        return None
    fit = _kasa(pts)
    for _ in range(3):
        if not fit:
            return None
        cx, cy, r = fit
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        tol = max(0.025 * r, 1.2)
        inl = np.abs(d - r) <= tol
        if inl.sum() < max(4, int(0.3 * len(pts))):
            return None
        fit = _kasa(pts[inl])
    return fit


def _ang(p, c):
    return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _coverage_ok(pts, cx, cy, max_gap=75.0):
    angs = sorted(_ang(p, (cx, cy)) for p in pts)
    gaps = [angs[i + 1] - angs[i] for i in range(len(angs) - 1)]
    gaps.append(360.0 - (angs[-1] - angs[0]))
    return max(gaps) <= max_gap


def _snap_to_circle(p, cx, cy, r):
    dx, dy = p[0] - cx, p[1] - cy
    d = math.hypot(dx, dy) or 1.0
    return (cx + dx / d * r, cy + dy / d * r)


def _make_circle_sub(cx, cy, r):
    return {"start": (cx + r, cy),
            "segs": [["A", r, r, 0, 1, 1, cx - r, cy],
                     ["A", r, r, 0, 1, 1, cx + r, cy]],
            "closed": True}


def _try_full_circle(sub, anchors, samples):
    fit = _fit_circle(samples)
    if not fit:
        return None
    cx, cy, r = fit
    if r < 3.0:
        return None
    a = np.asarray(anchors, dtype=np.float64)
    xs, ys = a[:, 0], a[:, 1]
    w = xs.max() - xs.min(); h = ys.max() - ys.min()
    if not (xs.min() - 0.2 * w <= cx <= xs.max() + 0.2 * w
            and ys.min() - 0.2 * h <= cy <= ys.max() + 0.2 * h):
        return None
    d = np.hypot(a[:, 0] - cx, a[:, 1] - cy)
    tol = max(0.02 * r, 0.9)
    maxdev = max(0.03 * r, 1.5)
    if np.all(np.abs(d - r) <= tol * 1.6) and np.abs(d - r).max() <= maxdev \
            and _coverage_ok(samples, cx, cy):
        return (cx, cy, r)
    return None


def _distance_clusters(anchors, cx, cy):
    """依「到中心距離」分群，找出環帶外緣/內緣的半徑候選。"""
    d = sorted(math.hypot(p[0] - cx, p[1] - cy) for p in anchors)
    clusters, cur = [], [d[0]]
    for v in d[1:]:
        if v - cur[-1] <= max(2.5, 0.008 * v):
            cur.append(v)
        else:
            clusters.append(cur); cur = [v]
    clusters.append(cur)
    out = []
    for c in clusters:
        if len(c) >= 8:
            med = c[len(c) // 2]
            if med >= 20 and (c[-1] - c[0]) <= max(0.035 * med, 4.0):
                out.append(med)
    return out


def _arc_replace(sub, anchors, candidates):
    """把貼合候選圓的連續節點段換成數學圓弧；凸出的設計細節原樣保留。"""
    n = len(anchors)
    labels = np.full(n, -1, dtype=int)
    for ci, (cx, cy, r) in enumerate(candidates):
        tol = max(0.02 * r, 1.8)
        for i, p in enumerate(anchors):
            if labels[i] == -1 and abs(math.hypot(p[0] - cx, p[1] - cy) - r) <= tol:
                labels[i] = ci
    if (labels == -1).all():
        return 0

    def _monotonic(idx_from, idx_to, cx, cy):
        """節點沿圓的行進方向是否單調（細長筆畫會去了又回，必須排除）。"""
        angs = [_ang(anchors[i], (cx, cy)) for i in range(idx_from, idx_to + 1)]
        pos = neg = 0.0
        for i in range(1, len(angs)):
            dd = _wrap(angs[i] - angs[i - 1])
            if dd >= 0:
                pos += dd
            else:
                neg -= dd
        return min(pos, neg) <= 3.0

    # 全部貼合同一顆圓 → 換成正圓。但必須：繞滿一圈、方向單調、
    # 圓心落在形狀外框內（細長弧形筆畫三者都不會通過）。
    if (labels != -1).all() and len(set(labels.tolist())) == 1:
        cx, cy, r = candidates[int(labels[0])]
        xs = [p[0] for p in anchors]; ys = [p[1] for p in anchors]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        center_inside = (min(xs) - 0.2 * w <= cx <= max(xs) + 0.2 * w
                         and min(ys) - 0.2 * h <= cy <= max(ys) + 0.2 * h)
        if center_inside and _monotonic(0, n - 1, cx, cy) \
                and _coverage_ok(anchors, cx, cy):
            new = _make_circle_sub(cx, cy, r)
            sub["start"] = new["start"]; sub["segs"] = new["segs"]; sub["closed"] = True
            return 1
        return 0   # 細長弧形筆畫：保持原樣（vtracer 的曲線已足夠平滑）

    # 旋轉起點，避免弧段跨過路徑起點
    if (labels == -1).any():
        k = (int(np.where(labels == -1)[0][0]) + 1) % n
    else:
        ks = [i for i in range(n) if labels[i] != labels[i - 1]]
        k = ks[0] if ks else 0
    segs = sub["segs"][k:] + sub["segs"][:k]
    anchors = anchors[k:] + anchors[:k]
    labels = np.roll(labels, -k)
    start = anchors[-1]

    runs = []
    s = None
    for i in range(n):
        if labels[i] != -1 and s is None:
            s = i
        elif s is not None and (labels[i] == -1 or labels[i] != labels[s]):
            runs.append((s, i - 1))
            s = i if labels[i] != -1 else None
    if s is not None:
        runs.append((s, n - 1))

    run_map = {rs: re_ for rs, re_ in runs}
    new_segs = []
    replaced = 0
    idx = 0
    while idx < n:
        if idx in run_map:
            re_ = run_map[idx]
            count = re_ - idx + 1
            ci = int(labels[idx])
            cx, cy, r = candidates[ci]
            angs = [_ang(anchors[i], (cx, cy)) for i in range(idx, re_ + 1)]
            span = 0.0
            for i in range(1, len(angs)):
                span += _wrap(angs[i] - angs[i - 1])
            if count >= 6 and abs(span) >= 25.0 and _monotonic(idx, re_, cx, cy):
                # 弧只涵蓋「落在圓上」的節點區間：anchors[idx] → anchors[re_]。
                # segs[idx]（從前一節點進入圓的過渡段）原樣保留，
                # 只把它的終點貼到圓上，弧從那裡接手。
                p_first = _snap_to_circle(anchors[idx], cx, cy, r)
                p_end = _snap_to_circle(anchors[re_], cx, cy, r)
                lead = list(segs[idx])
                if lead[0] == "L":
                    lead[1], lead[2] = p_first
                elif lead[0] == "C":
                    lead[5], lead[6] = p_first
                elif lead[0] == "Q":
                    lead[3], lead[4] = p_first
                elif lead[0] == "A":
                    lead[6], lead[7] = p_first
                new_segs.append(lead)
                laf = 1 if abs(span) > 180.0 else 0
                sf = 1 if span > 0 else 0
                new_segs.append(["A", r, r, 0, laf, sf, p_end[0], p_end[1]])
                replaced += 1
                idx = re_ + 1
                continue
            else:
                for i in range(idx, re_ + 1):
                    new_segs.append(segs[i])
                idx = re_ + 1
                continue
        new_segs.append(segs[idx])
        idx += 1

    if replaced:
        sub["start"] = start
        sub["segs"] = new_segs
    return replaced


def _uniform_fit(angles):
    k = len(angles)
    step = 360.0 / k
    res = [a - i * step for i, a in enumerate(angles)]
    s = sum(math.sin(math.radians(x)) for x in res)
    c = sum(math.cos(math.radians(x)) for x in res)
    phase = math.degrees(math.atan2(s, c))
    devs = [_wrap(a - (phase + i * step)) for i, a in enumerate(angles)]
    snapped = [phase + i * step for i in range(k)]
    return snapped, max(abs(v) for v in devs)


def _regularize(all_paths, level="normal"):
    """Geometry regularization.

    level:
      "conservative" — snap near-circles to perfect circles, align concentric
                       rings, unify rivet size/orbit/spacing. Ring-band edge
                       arc replacement is skipped (the most aggressive step).
      "normal"       — everything above plus ring/band edge arc replacement.
    """
    notes = []
    circles = []
    subs_todo = []

    for entry in all_paths:
        for sub in entry.get("subs", []):
            if "segs" not in sub or not sub.get("closed", False):
                continue
            anchors = _anchors_of(sub)
            if len(anchors) < 4:
                continue
            fit = _try_full_circle(sub, anchors, _samples_of(sub))
            if fit:
                circles.append({"sub": sub, "cx": fit[0], "cy": fit[1],
                                "r": fit[2], "color": entry["color"]})
            else:
                subs_todo.append((sub, anchors))

    main = None
    mx = my = rmax = 0.0
    if circles:
        biggest = max(circles, key=lambda c: c["r"])
        main = biggest
        mx, my, rmax = biggest["cx"], biggest["cy"], biggest["r"]

    # Ring/band edge arc replacement (normal level only; protruding design
    # details are preserved, but this is the most aggressive transform).
    arc_count = 0
    if level == "normal":
        for sub, anchors in subs_todo:
            if len(anchors) < 10:
                continue
            candidates = []
            fit = _fit_circle(_samples_of(sub))
            if fit and fit[2] >= 20:
                a = np.asarray(anchors)
                d = np.hypot(a[:, 0] - fit[0], a[:, 1] - fit[1])
                if (np.abs(d - fit[2]) <= np.maximum(0.02 * fit[2], 1.8)).mean() >= 0.35:
                    candidates.append(fit)
            if not candidates and main is not None:
                for r in _distance_clusters(anchors, mx, my):
                    candidates.append((mx, my, r))
            if candidates:
                try:
                    arc_count += _arc_replace(sub, anchors, candidates)
                except Exception:
                    pass
    if arc_count:
        notes.append(f"{arc_count} ring/band edge segments regularized into "
                     "mathematical circular arcs (protruding details preserved)")

    if not circles:
        return notes

    # 大圓同心
    big = [c for c in circles if c["r"] >= 0.35 * rmax]
    if len(big) >= 2:
        cxs = [c["cx"] for c in big]; cys = [c["cy"] for c in big]
        spread = max(max(cxs) - min(cxs), max(cys) - min(cys))
        if spread <= max(0.02 * rmax, 3.0):
            ux, uy = sum(cxs) / len(big), sum(cys) / len(big)
            for c in big:
                c["cx"], c["cy"] = ux, uy
            mx, my = ux, uy
            notes.append(f"{len(big)} large circles aligned to a shared concentric center")

    # 鉚釘：小圓、繞著主中心的軌道上
    rivets = []
    for c in circles:
        if c["r"] <= 0.18 * rmax:
            orbit = math.hypot(c["cx"] - mx, c["cy"] - my)
            if 0.25 * rmax <= orbit <= 1.05 * rmax:
                c["orbit"] = orbit
                c["ang"] = _ang((c["cx"], c["cy"]), (mx, my))
                rivets.append(c)

    if len(rivets) >= 5:
        rivets.sort(key=lambda c: c["ang"])
        clusters = [[rivets[0]]]
        for c in rivets[1:]:
            if c["ang"] - clusters[-1][-1]["ang"] <= 6.0:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        if len(clusters) >= 2 and (clusters[0][0]["ang"] + 360.0 - clusters[-1][-1]["ang"]) <= 6.0:
            clusters[0] = clusters.pop() + clusters[0]

        k = len(clusters)
        if k >= 5:
            cl_ang = [sum(c["ang"] for c in cl) / len(cl) for cl in clusters]
            order = sorted(range(k), key=lambda i: cl_ang[i])
            cl_ang_sorted = [cl_ang[i] for i in order]
            snapped, max_dev = _uniform_fit(cl_ang_sorted)
            aligned = max_dev <= 4.0

            orbits = [c["orbit"] for c in rivets]
            om = sum(orbits) / len(orbits)
            orbit_uni = (max(orbits) - min(orbits)) / om <= 0.06 if om else False

            by_color = defaultdict(list)
            for c in rivets:
                by_color[c["color"]].append(c)
            r_notes = 0
            for cc in by_color.values():
                rs = [c["r"] for c in cc]
                rm = sum(rs) / len(rs)
                if rm > 0 and (max(rs) - min(rs)) / rm <= 0.18:
                    for c in cc:
                        c["r"] = rm
                    r_notes += len(cc)

            if aligned or orbit_uni:
                for pos, ci in enumerate(order):
                    ang = snapped[pos] if aligned else cl_ang_sorted[pos]
                    for c in clusters[ci]:
                        o = om if orbit_uni else c["orbit"]
                        rad = math.radians(ang)
                        c["cx"] = mx + o * math.cos(rad)
                        c["cy"] = my + o * math.sin(rad)
            msg = []
            if aligned:
                msg.append("equal angular spacing")
            if orbit_uni:
                msg.append("unified orbit radius")
            if r_notes:
                msg.append("unified size")
            if msg:
                notes.append(f"{k} rivet-like dots: " + ", ".join(msg))

    notes.append(f"{len(circles)} near-circular shapes replaced with perfect circles")

    for c in circles:
        new = _make_circle_sub(c["cx"], c["cy"], c["r"])
        c["sub"]["start"] = new["start"]
        c["sub"]["segs"] = new["segs"]
        c["sub"]["closed"] = True
        c["sub"]["is_circle"] = (c["cx"], c["cy"], c["r"])

    return notes


# ---------- 舊版位移烘焙（解析失敗時的備援） ----------

_PAIR_CMDS = set("MLCSQT")


def _bake_translate(d, tx, ty):
    tokens = re.findall(r"[A-Za-z]|-?\d+\.?\d*(?:e-?\d+)?", d)
    result = []
    k = 0
    while k < len(tokens):
        t = tokens[k]
        if t.isalpha():
            cmd = t
            result.append(cmd)
            k += 1
            if cmd in "Zz":
                continue
            nums = []
            while k < len(tokens) and not tokens[k].isalpha():
                nums.append(float(tokens[k]))
                k += 1
            up = cmd.upper()
            if up in _PAIR_CMDS:
                for n in range(0, len(nums), 2):
                    nums[n] += tx
                    if n + 1 < len(nums):
                        nums[n + 1] += ty
            elif up == "H":
                nums = [v + tx for v in nums]
            elif up == "V":
                nums = [v + ty for v in nums]
            result.append(" ".join(_f(v) for v in nums))
        else:
            k += 1
    return " ".join(result)


# ---------- Gradient detection ----------

def _allocate_gradient_keys(palette, count, min_distance=48):
    """Allocate deterministic placeholder colours that cannot alias each other.

    Gradient placeholders are an internal transport between the fill tracer and
    the preview renderer.  They must be well separated from both real palette
    colours and every other placeholder; otherwise one gradient can overwrite
    another during antialias-fringe recovery.
    """
    count = int(count)
    if count <= 0:
        return []
    pal = np.asarray(palette, dtype=np.int16).reshape(-1, 3)
    levels = np.asarray((3, 35, 67, 99, 131, 163, 195, 227, 251),
                        dtype=np.int16)
    candidates = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"),
                          axis=-1).reshape(-1, 3)
    chosen = []
    occupied = pal.copy()
    for _ in range(count):
        if occupied.size:
            # Chebyshev separation matches the renderer's max-channel key
            # tolerance exactly.
            dist = np.abs(candidates[:, None, :] - occupied[None, :, :]).max(2)
            score = dist.min(1)
        else:
            score = np.full(len(candidates), 255, dtype=np.int16)
        idx = int(score.argmax())
        if int(score[idx]) < int(min_distance):
            raise ValueError(
                f"cannot isolate {count} gradient keys by {min_distance} RGB levels")
        key = candidates[idx].copy()
        chosen.append(tuple(int(v) for v in key))
        occupied = (np.vstack([occupied, key]) if occupied.size
                    else key.reshape(1, 3))
        candidates = np.delete(candidates, idx, axis=0)
    return chosen


def _detect_gradients(den, lab_all, vis_fill, palette, max_regions=8,
                      band_color_distance=32.0, minimum_color_span=30.0):
    """Detect palette-quantization banding and rebuild it as linear gradients.

    Adjacent flat-color components whose shared boundary is SMOOTH in the
    original image (no real edge) but differs strongly in the flattened
    palette are quantization bands of one gradient region. Merged regions
    must then fit a linear color ramp with low residual, so a genuine
    edge (e.g. green art on black) can never masquerade as a gradient.

    Returns regions: {mask, area, x1, y1, x2, y2, stops[(offset, rgb)]}.
    """
    from stroke_engine import connected_components
    H, W = vis_fill.shape
    pal = palette.astype(np.float32)
    K = len(pal)

    comp_map = np.zeros((H, W), dtype=np.int32)
    comp_color = [0]
    comp_area = [0]
    for ci in range(K):
        m = vis_fill & (lab_all == ci)
        if not m.any():
            continue
        lb, n = connected_components(m)
        if n == 0:
            continue
        off = len(comp_color) - 1
        comp_map[m] = lb[m] + off
        counts = np.bincount(lb[m], minlength=n + 1)
        for i in range(1, n + 1):
            comp_color.append(ci)
            comp_area.append(int(counts[i]))
    NC = len(comp_color) - 1
    if NC < 2:
        return []
    area_arr = np.asarray(comp_area)

    def pair_stats(A, B, D):
        valid = (A > 0) & (B > 0) & (A != B)
        out = {}
        if not valid.any():
            return out
        a = A[valid].astype(np.int64)
        b = B[valid].astype(np.int64)
        d = D[valid]
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keys = lo * (NC + 1) + hi
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        d = d[order]
        uk, start = np.unique(keys, return_index=True)
        for i, k in enumerate(uk):
            s = start[i]
            e = start[i + 1] if i + 1 < len(start) else len(keys)
            seg = d[s:e]
            out[int(k)] = (len(seg), float((seg < 30).mean()))
        return out

    dh = np.abs(den[:, :-1] - den[:, 1:]).max(axis=2)
    dv = np.abs(den[:-1, :] - den[1:, :]).max(axis=2)
    stats = pair_stats(comp_map[:, :-1], comp_map[:, 1:], dh)
    for k, (n2, f2) in pair_stats(comp_map[:-1, :], comp_map[1:, :], dv).items():
        if k in stats:
            n1, f1 = stats[k]
            stats[k] = (n1 + n2, (f1 * n1 + f2 * n2) / (n1 + n2))
        else:
            stats[k] = (n2, f2)

    parent = list(range(NC + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged_any = False
    for k, (n, frac) in stats.items():
        a, b = divmod(k, NC + 1)
        if n < 40 or frac < 0.55:
            continue
        if area_arr[a] < 400 or area_arr[b] < 400:
            continue
        ca, cb = comp_color[a], comp_color[b]
        if (float(((pal[ca] - pal[cb]) ** 2).sum())
                < float(band_color_distance) ** 2):
            continue
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
            merged_any = True
    if not merged_any:
        return []

    groups = {}
    for c in range(1, NC + 1):
        groups.setdefault(find(c), []).append(c)

    rng = np.random.default_rng(0)
    regions = []
    for comps in groups.values():
        if len(comps) < 2:
            continue
        mask = np.isin(comp_map, comps)
        area = int(mask.sum())
        if area < 1200:
            continue
        ys, xs = np.nonzero(mask)
        if len(ys) > 20000:
            sel = rng.choice(len(ys), 20000, replace=False)
            ys_s, xs_s = ys[sel], xs[sel]
        else:
            ys_s, xs_s = ys, xs
        Xm = np.c_[np.ones(len(ys_s)), xs_s, ys_s].astype(np.float64)
        col = den[ys_s, xs_s].astype(np.float64)
        try:
            sol, *_ = np.linalg.lstsq(Xm, col, rcond=None)
        except Exception:
            continue
        resid = float(np.abs(Xm @ sol - col).max(axis=1).mean())
        if resid > 26.0:
            continue
        gx = 0.299 * sol[1, 0] + 0.587 * sol[1, 1] + 0.114 * sol[1, 2]
        gy = 0.299 * sol[2, 0] + 0.587 * sol[2, 1] + 0.114 * sol[2, 2]
        norm = math.hypot(gx, gy)
        if norm < 1e-4:
            ch = int(np.abs(sol[1:]).sum(axis=0).argmax())
            gx, gy = float(sol[1, ch]), float(sol[2, ch])
            norm = math.hypot(gx, gy)
            if norm < 1e-6:
                continue
        ux, uy = gx / norm, gy / norm
        t = xs * ux + ys * uy
        tmin, tmax = float(t.min()), float(t.max())
        if tmax - tmin < 24:
            continue
        stops = []
        nb = 5
        halfwin = (tmax - tmin) * 0.08
        for i in range(nb):
            tc = tmin + (tmax - tmin) * i / (nb - 1)
            m2 = (t >= tc - halfwin) & (t <= tc + halfwin)
            if int(m2.sum()) < 30:
                continue
            c2 = np.median(den[ys[m2], xs[m2]], axis=0)
            stops.append((i / (nb - 1),
                          tuple(int(v) for v in np.clip(c2, 0, 255))))
        if len(stops) < 2:
            continue
        ded = [stops[0]]
        for off, c2 in stops[1:]:
            if max(abs(c2[j] - ded[-1][1][j]) for j in range(3)) > 6 \
                    or off == stops[-1][0]:
                ded.append((off, c2))
        span = max(abs(ded[0][1][j] - ded[-1][1][j]) for j in range(3))
        if span < float(minimum_color_span):
            continue

        # A smooth-looking fitted ramp is not automatically better than the
        # flat palette it replaces.  Validate every region against that exact
        # baseline, including its error tail, so a misleading fit can never
        # make a large part of the logo visibly worse.
        tt_s = np.clip((xs_s * ux + ys_s * uy - tmin) / (tmax - tmin),
                       0.0, 1.0)
        stop_offs = np.asarray([off for off, _ in ded], dtype=np.float64)
        stop_cols = np.asarray([c for _, c in ded], dtype=np.float64)
        fitted = np.column_stack([
            np.interp(tt_s, stop_offs, stop_cols[:, ch]) for ch in range(3)
        ])
        source_sample = den[ys_s, xs_s].astype(np.float64)
        flat_sample = pal[lab_all[ys_s, xs_s]]
        grad_err = np.abs(fitted - source_sample).max(axis=1)
        flat_err = np.abs(flat_sample - source_sample).max(axis=1)
        grad_mean = float(grad_err.mean())
        flat_mean = float(flat_err.mean())
        grad_p90 = float(np.quantile(grad_err, 0.90))
        flat_p90 = float(np.quantile(flat_err, 0.90))
        degraded_share = float((grad_err > flat_err + 3.0).mean())
        validation = {
            "sample_pixels": int(len(grad_err)),
            "flat_mean_error": round(flat_mean, 3),
            "gradient_mean_error": round(grad_mean, 3),
            "flat_p90_error": round(flat_p90, 3),
            "gradient_p90_error": round(grad_p90, 3),
            "degraded_over_3_share": round(degraded_share, 4),
        }
        if (grad_mean > flat_mean - 2.0
                or grad_p90 > flat_p90
                or degraded_share > 0.10):
            continue
        regions.append({"mask": mask, "area": area,
                        "x1": ux * tmin, "y1": uy * tmin,
                        "x2": ux * tmax, "y2": uy * tmax,
                        "stops": ded, "validation": validation})
    regions.sort(key=lambda r: -r["area"])
    return regions[:max_regions]


def _gradient_region_from_proposal(proposal):
    """Adapt a held-out paint + geometry proposal to the SVG assembler.

    The source-space mask owns the paint object; its colour model and its
    error-budgeted contour remain separate evidence.  Only compact evidence is
    retained here so reports do not embed every optimiser candidate/sample.
    """
    from gradient_reconstruction_stage import decode_mask_rle

    mask = decode_mask_rle(proposal["mask"])
    model = dict(proposal.get("model") or {})
    stops = [dict(stop) for stop in proposal.get("stops") or []]
    geometry = dict(proposal.get("geometry") or {})
    if model.get("type") not in {"linear", "radial"}:
        raise ValueError("gradient proposal has no supported native model")
    if not 2 <= len(stops) <= 5:
        raise ValueError("gradient proposal must contain 2 to 5 stops")
    path = geometry.get("path") or proposal.get("path") or ""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("gradient proposal has no vector geometry")
    error_budget = dict(geometry.get("error_budget") or {})
    budget = error_budget.get("requested_max_percent")
    p95_error = error_budget.get("actual_p95_error_percent")
    max_error = error_budget.get("actual_max_error_percent")
    evidence_values = (budget, p95_error, max_error)
    if (error_budget.get("passed") is not True
            or any(isinstance(value, bool) or not isinstance(
                value, (int, float)) or not math.isfinite(float(value))
                   for value in evidence_values)
            or float(budget) <= 0.0
            or float(p95_error) < 0.0
            or float(max_error) < 0.0
            or float(p95_error) > float(budget) + 1e-7
            or float(max_error) > 3.0 * float(budget) + 1e-7):
        raise ValueError(
            "gradient proposal has incomplete or invalid geometry error evidence")
    from native_geometry_contract import whole_object_native_primitive

    native = list(geometry.get("native_primitives") or [])
    whole_native = whole_object_native_primitive(geometry, path)
    compact_geometry = {
        "solver": geometry.get("solver"),
        "primitive_first": bool(geometry.get("primitive_first")),
        "native_primitives": native,
        "native_whole_object_path": path if whole_native else None,
        "primitive_complexity": geometry.get("primitive_complexity") or {},
        "topology": geometry.get("topology") or {},
        "selection_evidence": geometry.get("selection_evidence") or {},
        "anchor_count": int(geometry.get("anchor_count") or 0),
        "designer_anchor_count": int(
            geometry.get("designer_anchor_count")
            or geometry.get("anchor_count") or 0),
        "segment_count": int(geometry.get("segment_count") or 0),
        "anchors_before": int(geometry.get("anchors_before") or 0),
        "error_budget": error_budget,
        "lexicographic_objective": geometry.get(
            "lexicographic_objective") or [],
    }
    region = {
        "mask": mask,
        "area": int(proposal.get("area") or mask.sum()),
        "model": model,
        "stops": stops,
        "path_override": path,
        "fill_rule": geometry.get("fill_rule") or proposal.get(
            "fill_rule") or "evenodd",
        "native_primitives": native,
        "proposal_id": proposal.get("proposal_id"),
        "candidate_id": proposal.get("candidate_id"),
        "candidate_family": proposal.get("candidate_family"),
        "confidence": float(proposal.get("confidence") or 0.0),
        "enclosed_source_component_candidate": copy.deepcopy(
            proposal.get("enclosed_source_component_candidate")),
        "validation": {
            "engine": "source_space_heldout_gradient_object",
            "paint": proposal.get("heldout_evidence") or {},
            "geometry": compact_geometry,
            "selection": proposal.get("selection") or {},
        },
    }
    # Top-level legacy coordinates remain available to older preview/report
    # readers while the authoritative model stays intact.
    if model["type"] == "linear":
        for name in ("x1", "y1", "x2", "y2"):
            region[name] = float(model[name])
    return region


def _recover_unowned_gradient_residuals(entries, labels, visible, gradient_regions,
                                        palette_count, *, proven_preexisting=None):
    """Explicitly trace palette ink omitted outside all gradient ownership.

    A gradient can absorb virtually the entire image, leaving a handful of
    one-pixel antialias colours that VTracer's speckle filter drops.  Keep the
    ink-loss gate: reconstruct the missing ink instead of waiving that gate or
    pretending that a placeholder routing colour represents it.
    """
    if not gradient_regions:
        return [], 0
    from trace_engine import binary_mask_to_compound_path

    residual = np.asarray(visible, dtype=bool).copy()
    owned = np.zeros_like(residual)
    for region in gradient_regions:
        owned |= np.asarray(region["mask"], dtype=bool)
    residual &= ~owned
    if proven_preexisting is not None:
        preexisting = np.asarray(proven_preexisting, dtype=bool)
        if preexisting.shape != residual.shape:
            raise ValueError("gradient residual provenance shape mismatch")
        residual &= ~preexisting
    emitted = {entry["color"] for entry in entries}
    recovered, pixels, unrelated_fragments = [], 0, 0
    for raw_index in np.unique(labels[residual]):
        index = int(raw_index)
        if index in emitted or index < 0 or index >= palette_count:
            continue
        mask = residual & (labels == index)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        # Pad the crop explicitly even on a source-canvas edge, so a one-pixel
        # residual retains a closed boundary instead of becoming a trace edge.
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        local = np.pad(mask[y0:y1, x0:x1], 1)
        traced = binary_mask_to_compound_path(
            local, simplify=0.0, min_area=0.0, smooth=0.0, curve=0.0)
        if traced.get("path"):
            consumed = int(np.count_nonzero(owned & (labels == index)))
            fragments = len(_parse_subpaths(traced["path"]))
            if not consumed:
                # An unrelated palette colour already omitted by tracing is
                # not a consequence of gradient ownership. Only a bounded
                # handful of scraps may be recovered opportunistically; never
                # turn pre-existing antialias noise into hundreds of paths or
                # withdraw unrelated gradient objects because of that noise.
                if unrelated_fragments + fragments > 8:
                    continue
                unrelated_fragments += fragments
            recovered.append({"color": index,
                              "raw": _bake_translate(traced["path"], x0 - 1, y0 - 1)})
            pixels += int(len(xs))
    return recovered, pixels


def _proven_light_compartment_exclusions(shape, main_recovery):
    excluded = np.ones(shape, dtype=bool)
    height, width = shape
    for record in main_recovery.get("records", []):
        if record.get("kind") != "enclosed_background_compartment":
            continue
        x0, y0, x1, y1 = record["bbox_xyxy"]
        if 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height:
            excluded[y0:y1, x0:x1] = False
    return excluded


def _gradient_residual_fragmentation(entries, labels, visible, regions,
                                     maximum_fragments=8):
    """Withdraw replacements that would require many new disconnected scraps.

    This is a bounded editing-complexity policy, not a claim about human time.
    A source colour which the ordinary tracer omitted must not be recovered as
    hundreds of selectable pixels merely to keep a gradient proposal alive.
    Withdraw only proposals consuming those fragmented source colours, then
    rerun the ordinary tracer with their original palette pixels restored.
    """
    counts = {}
    for entry in entries:
        counts[int(entry["color"])] = len(_parse_subpaths(entry["raw"]))
    excessive = {index: count for index, count in counts.items()
                 if count > maximum_fragments}
    if sum(counts.values()) > maximum_fragments and not excessive:
        excessive = counts
    if not excessive:
        return None
    residual = np.asarray(visible, dtype=bool).copy()
    for region in regions:
        residual &= ~np.asarray(region["mask"], dtype=bool)
    residual &= np.isin(labels, list(excessive))
    # Attribute the scraps locally. The same palette colour may also appear
    # in unrelated gradients; withdrawing every consumer would unnecessarily
    # discard safe objects elsewhere on the canvas.
    neighbourhood = np.asarray(Image.fromarray(
        residual.astype(np.uint8) * 255, "L").filter(ImageFilter.MaxFilter(3))) > 0
    owners = []
    for region in regions:
        candidate_id = region.get("candidate_id")
        mask = np.asarray(region["mask"], dtype=bool)
        consumed = sum(int(np.count_nonzero(mask & (labels == index)))
                       for index in excessive)
        if candidate_id and consumed:
            owners.append({"candidate_id": str(candidate_id),
                           "adjacent_residual_pixels": int(np.count_nonzero(mask & neighbourhood)),
                           "consumed_palette_pixels": consumed})
    if not owners:
        return None
    owners.sort(key=lambda row: (-row["adjacent_residual_pixels"],
                                -row["consumed_palette_pixels"], row["candidate_id"]))
    return {
        "reason": "gradient_replacement_requires_fragmented_palette_residuals",
        "candidate_ids": [owners[0]["candidate_id"]] if owners else [],
        "attribution_candidates": owners,
        "residual_fragments": sum(counts.values()),
        "maximum_recovered_fragments": int(maximum_fragments),
        "palette_fragment_counts": {str(k): v for k, v in sorted(counts.items())},
        "action": "withdraw_gradient_and_retrace_original_palette",
        "human_time_saving_claimed": False,
    }


def _withdraw_gradient_proposals(stage, withdrawals):
    """Keep the final certificate aligned with the actually emitted set."""
    if not withdrawals:
        return stage
    result = copy.deepcopy(stage)
    blocked = {item for record in withdrawals for item in record["candidate_ids"]}
    proposals = [p for p in result.get("proposals", [])
                 if str(p.get("candidate_id")) not in blocked]
    result["proposals"] = proposals
    result["status"] = "proposed" if proposals else "skipped"
    for decision in result.get("decisions", []):
        if (decision.get("status") == "selected"
                and str(decision.get("candidate_id")) in blocked):
            decision["status"] = "withdrawn"
            decision["reasons"] = list(decision.get("reasons") or []) + [
                "gradient_replacement_requires_fragmented_palette_residuals"]
    summary = result.setdefault("summary", {})
    summary.update({
        "objects_selected": len(proposals),
        "covered_pixels": sum(int(p.get("area") or 0) for p in proposals),
        "pair_objects_selected": sum(p.get("candidate_family") == "pair" for p in proposals),
        "total_anchors": sum(int((p.get("economy") or {}).get("anchors_after") or 0) for p in proposals),
        "total_segments": sum(int((p.get("economy") or {}).get("segment_count") or 0) for p in proposals),
        "fragmentation_withdrawn_objects": len(blocked),
    })
    return result


# ---------- Main pipeline ----------

def _iter_svg_paths(raw):
    """Yield (d, fill, tx, ty) for every <path> in document order.

    Prefers a real XML parser; falls back to a regex scan if the document
    cannot be parsed (so a vtracer output format change degrades gracefully
    instead of silently producing an empty result).
    """
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(raw)
        found = False
        for el in root.iter():
            if not el.tag.rsplit("}", 1)[-1] == "path":
                continue
            d = el.get("d")
            fill = el.get("fill", "")
            if not d or not re.fullmatch(r"#[0-9a-fA-F]{6}", fill):
                continue
            tm = re.search(r"translate\(\s*([-\d.]+)[,\s]+([-\d.]+)\s*\)",
                           el.get("transform", "") or "")
            tx, ty = (float(tm.group(1)), float(tm.group(2))) if tm else (0.0, 0.0)
            found = True
            yield d, fill, tx, ty
        if found:
            return
    except Exception:
        pass
    for m in re.finditer(r"<path\b[^>]*?/>", raw):
        tag = m.group(0)
        dm = re.search(r'\bd="([^"]+)"', tag)
        fm = re.search(r'\bfill="(#[0-9a-fA-F]{6})"', tag)
        if not (dm and fm):
            continue
        tm = re.search(r'transform="translate\(([-\d.]+),([-\d.]+)\)"', tag)
        tx, ty = (float(tm.group(1)), float(tm.group(2))) if tm else (0.0, 0.0)
        yield dm.group(1), fm.group(1), tx, ty


def _erode_boolean_mask(mask, iterations=1):
    """Return a small dependency-free 8-neighbourhood erosion."""
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        eroded = np.ones_like(result, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy:dy + result.shape[0],
                                 dx:dx + result.shape[1]]
        result = eroded
    return result


def _broad_enclosed_background_mask(visible, bg_like, enclosure):
    """Select only broad background pockets inside a detected outline.

    A closed circle/rectangle can trap the source canvas as an opaque white
    island after border-connected background removal.  Conversely, real white
    lettering and highlights can live inside that same outline.  Treating all
    background-coloured pixels as a hole therefore destroys real artwork.

    This classifier removes a component only when it has the spatial evidence
    of negative space: it occupies at least 30% of the enclosure, or is both
    broadly spanning and dominant while touching the inner boundary.  A third
    branch handles a cross/bar that partitions one large background into
    several components; it activates only when background-like pixels jointly
    occupy at least 30% of the enclosure and the component clearly owns part
    of the inner boundary.
    """
    visible = np.asarray(visible, dtype=bool)
    bg_like = np.asarray(bg_like, dtype=bool)
    enclosure = np.asarray(enclosure, dtype=bool)
    if not (visible.shape == bg_like.shape == enclosure.shape):
        raise ValueError("visible, bg_like and enclosure must have equal shapes")

    selected = np.zeros_like(enclosure, dtype=bool)
    enclosure_area = int(enclosure.sum())
    candidate = enclosure & visible & bg_like
    candidate_area = int(candidate.sum())
    if enclosure_area == 0 or candidate_area == 0:
        return selected

    from stroke_engine import connected_components

    labels, count = connected_components(candidate)
    if not count:
        return selected
    areas = np.bincount(labels.ravel(), minlength=count + 1)

    ey, ex = np.nonzero(enclosure)
    enclosure_width = max(1, int(ex.max() - ex.min() + 1))
    enclosure_height = max(1, int(ey.max() - ey.min() + 1))
    # A 1--3 px band tolerates antialiasing gaps at the detected stroke edge
    # without making centrally placed white artwork count as boundary-owned.
    band_depth = max(1, min(
        3, int(round(0.01 * min(enclosure_width, enclosure_height)))))
    inner = _erode_boolean_mask(enclosure, band_depth)
    boundary_band = enclosure & ~inner
    boundary_area = max(1, int(boundary_band.sum()))
    total_ratio = candidate_area / float(enclosure_area)

    for component in range(1, count + 1):
        area = int(areas[component])
        if area <= 0:
            continue
        component_mask = labels == component
        cy, cx = np.nonzero(component_mask)
        area_ratio = area / float(enclosure_area)
        span_x = (int(cx.max() - cx.min() + 1) / float(enclosure_width))
        span_y = (int(cy.max() - cy.min() + 1) / float(enclosure_height))
        dominance = area / float(candidate_area)
        boundary_share = float(
            (component_mask & boundary_band).sum()) / boundary_area

        large_single_pocket = area_ratio >= 0.30
        dominant_spanning_pocket = (
            area_ratio >= 0.08
            and span_x >= 0.72
            and span_y >= 0.72
            and dominance >= 0.65
            and boundary_share >= 0.12
        )
        partitioned_background = (
            total_ratio >= 0.30
            and area_ratio >= 0.01
            and boundary_share >= 0.08
        )
        if (large_single_pocket or dominant_spanning_pocket
                or partitioned_background):
            selected |= component_mask

    return selected


def _produce_pre_gradient_state(src, *, forced_colors, white_threshold,
                                background, max_size, strokes, checkpoint):
    """Build the exact stroke-aware state immediately before gradients.

    The returned value is deliberately self-contained: callers may deep-copy
    it between serial candidate builds without sharing any mutable array,
    stroke object, note list or audit dictionary.
    """
    im, (orig_w, orig_h), removed = _prepare_image(
        src, max_size=max_size, background=background,
        white_threshold=white_threshold, alpha_threshold=12,
    )
    A = np.asarray(im)
    source_alpha = A[:, :, 3]
    background_counter_mask = np.asarray(
        getattr(im, "_avc_background_counter_mask",
                np.zeros(source_alpha.shape, dtype=bool)),
        dtype=bool,
    )
    if background_counter_mask.shape != source_alpha.shape:
        background_counter_mask = np.zeros(source_alpha.shape, dtype=bool)
    # Keep genuine semi-transparent artwork in the canonical mask.  The old
    # alpha>=128 split silently deleted mixed-alpha details whenever another
    # opaque object was present.
    visible = source_alpha >= 16
    # Counter pixels are transparent delivery geometry, but retaining their
    # neutral samples solely for adaptive palette estimation avoids changing
    # unrelated colours, gradient bands and thin accents when a wordmark's
    # paper-coloured holes are canonicalized.
    palette_visible = visible | background_counter_mask
    H, W = visible.shape
    pre_notes = []
    if not np.any(visible):
        raise ValueError("image has no traceable visible pixels")
    soft_pixels = visible & (source_alpha < 250)
    if np.any(soft_pixels):
        pre_notes.append(
            "semi-transparent source preserved with SVG opacity where "
            "component colors can be separated")

    # Alpha-aware, detail-preserving preprocessing.
    # Transparent pixels usually carry black RGB, so fill the invisible area
    # with white before any filtering. The median filter is used ONLY to
    # estimate palette centers (denoising), never to assign labels: pixels
    # the median changes a lot are fine detail (thin lines) and are fed to
    # the palette estimator unfiltered, and label assignment always runs on
    # the unfiltered image — a 3x3 median can no longer erase a thin stroke.
    rgb = A[:, :, :3].copy()
    # Preserve the original off-white/antialiased counter samples for palette
    # estimation; only pixels that were never artwork or retained sampling
    # evidence receive the neutral invisible RGB fill.
    rgb[~palette_visible] = 255
    den = rgb.astype(np.float32)                     # labels/colors: unfiltered
    if min(W, H) >= 3 and int(visible.sum()) >= 4096:
        med = np.asarray(Image.fromarray(rgb)
                         .filter(ImageFilter.MedianFilter(3))).astype(np.float32)
        # A 60-level cutoff erased legitimate light-gray 1 px artwork such
        # as #dddddd on white (difference 34).  Keep changes above 18 for
        # palette estimation; ordinary JPEG/background noise is subsequently
        # merged by the 45-distance palette merge.
        fine = np.abs(med - den).max(axis=2) > 18
        den_palette = np.where(fine[..., None], den, med)
    else:
        den_palette = den
    palette, _, palette_audit = detect_palette(
        den_palette, palette_visible, forced=forced_colors,
        return_audit=True, return_labels=False)
    # Stroke colors sampled later from the original-resolution image must
    # snap back to this pre-extraction palette, not to the fill-only palette
    # that is estimated after stroke pixels have been removed.
    initial_palette = palette.copy()

    def _assign(img, pal):
        return _nearest_palette_labels(img, pal)

    lab_all = _assign(den, palette)

    def _palette_opacities(mask, pal, labels):
        out = []
        for ci in range(len(pal)):
            vals = source_alpha[mask & (labels == ci)]
            if vals.size == 0:
                out.append(1.0)
                continue
            # Upper quartile keeps ordinary opaque antialiasing at opacity 1,
            # while a uniformly translucent component retains its true alpha.
            op = float(np.quantile(vals.astype(np.float32), 0.75)) / 255.0
            out.append(1.0 if op >= 0.985 else round(max(0.02, op), 3))
        return out

    palette_opacity = _palette_opacities(visible, palette, lab_all)
    checkpoint("stroke_analysis", "正在辨識可編輯筆畫與保留細節色彩")

    # Monoline stroke reconstruction: uniform-width line work becomes real
    # strokes (center line + stroke-width) instead of filled outline pairs.
    # Tier A works on the global ink mask, so an antialiased line — core
    # plus fringe — is ONE component and yields ONE stroke. Tier B then
    # scans each palette color for lines sitting on other fills.
    stroke_list = []
    stroke_mask = np.zeros((H, W), dtype=bool)
    stroke_deferred_mask = np.zeros((H, W), dtype=bool)
    stroke_safety_audit = {
        "policy": "native_straight_caps_and_whole_ink_palette_components",
        "source_ink_coverage_minimum": .98,
        "palette_fragment_candidate_pixels_deferred": 0,
        "native_cap_validated_strokes": 0,
        "complex_strokes_without_native_cap_proof": 0,
        "closed_strokes_without_native_geometry_fit": 0,
    }
    bg_col = (255.0, 255.0, 255.0)
    if strokes != "off":
        try:
            from stroke_engine import extract_strokes, connected_components, _dilate_one
            # Source cap/width evidence must use native pixels, including when
            # the trace mask was resized or its background made transparent.
            with Image.open(src) as original_image:
                stroke_source_rgba = np.asarray(original_image.convert("RGBA"))

            def _stroke_result(value):
                # Three values are the Beta.4 contract.  Accept the former
                # two-value shape for third-party tests/extensions and treat
                # it as having no explicit deferred guard.
                if len(value) == 3:
                    return value
                found, owned = value
                return found, owned, np.zeros_like(owned)

            if removed:
                # the removed background was light; enclosed pockets of the
                # same light color are still opaque (negative space), so
                # exclude them from the ink mask too
                bg_col = (255.0, 255.0, 255.0)
            else:
                border = np.concatenate([den[0, :], den[-1, :],
                                         den[:, 0], den[:, -1]])
                bg_col = tuple(np.median(border, axis=0))
            ink = visible & (((den - np.asarray(bg_col, dtype=np.float32)) ** 2)
                             .sum(axis=2) > 24 ** 2)
            s_a, m_a, d_a = _stroke_result(
                extract_strokes(ink, den, palette, bg_col,
                                alpha=source_alpha, source_rgba=stroke_source_rgba,
                                audit=stroke_safety_audit))
            stroke_list += s_a
            stroke_mask |= m_a
            stroke_deferred_mask |= d_a
            if _allow_palette_tier_b_strokes(palette_audit):
                whole_ink_labels, _ = connected_components(ink)
                whole_ink_areas = np.bincount(whole_ink_labels.ravel())
                for ci in range(len(palette)):
                    cm = (visible & ~stroke_mask & ~stroke_deferred_mask
                          & (lab_all == ci))
                    if not cm.any() or cm.sum() > 0.35 * H * W:
                        continue
                    # A colour-cluster ribbon may be just the AA outline of a
                    # filled object. It cannot claim stroke ownership unless
                    # it represents practically the entire original ink
                    # component. Otherwise preserve the fill, including its
                    # gaps/corners, instead of rounding one colour fragment.
                    covered = np.bincount(whole_ink_labels[cm], minlength=len(whole_ink_areas))
                    eligible = covered >= np.maximum(1, whole_ink_areas) * .98
                    eligible[0] = False
                    allowed = _dilate_one(eligible[whole_ink_labels])
                    # A real solid line may sit on another filled shape. It
                    # gets a separate bounded attempt only when this palette
                    # component has a two-pixel-eroded solid core; thin AA
                    # outlines cannot qualify. Every recovered stroke must
                    # subsequently carry the native palette-occupancy proof.
                    exceptional=cm & ~allowed
                    if exceptional.any():
                        core=exceptional.copy()
                        for _ in range(2):
                            p=np.pad(core,1)
                            core=np.logical_and.reduce([p[dy:dy+H,dx:dx+W]
                                for dy in range(3) for dx in range(3)])
                        if int(core.sum())>=16:
                            found,owned,deferred=_stroke_result(extract_strokes(
                                exceptional,den,palette,bg_col,alpha=source_alpha,
                                source_rgba=stroke_source_rgba,audit=stroke_safety_audit))
                            if found and all(getattr(s,'source_fit',{}).get('policy')==
                                    'native_discrete_palette_line_occupancy' for s in found):
                                stroke_list+=found
                                stroke_mask|=owned
                                cm &= ~owned
                    rejected = int((cm & ~allowed).sum())
                    stroke_safety_audit["palette_fragment_candidate_pixels_deferred"] += rejected
                    cm &= allowed
                    if not cm.any():
                        continue
                    s_b, m_b, d_b = _stroke_result(
                        extract_strokes(cm, den, palette, bg_col,
                                        alpha=source_alpha, source_rgba=stroke_source_rgba,
                                        audit=stroke_safety_audit))
                    stroke_list += s_b
                    stroke_mask |= m_b
                    stroke_deferred_mask |= d_b
            else:
                pre_notes.append(
                    "per-palette stroke recovery skipped on continuous-tone "
                    "artwork; global line components remain editable")
        except Exception as e:
            stroke_list = []
            stroke_mask[:] = False
            stroke_deferred_mask[:] = False
            pre_notes.append(f"stroke engine disabled by error: {e!r}"[:160])

    stroke_safety_audit["native_cap_validated_strokes"] = sum(
        bool(getattr(s, "source_fit", {}).get("serialized_geometry_validated")) for s in stroke_list)
    stroke_safety_audit["complex_strokes_without_native_cap_proof"] = sum(
        not getattr(s, "closed", False) and not getattr(s, "primitive", "")
        and not getattr(s, "source_fit", {}).get("serialized_geometry_validated") for s in stroke_list)
    # Closed outlines have no endpoints. Their geometry still needs the
    # ordinary scene checks, but an absent cap proof is not a failure there.
    stroke_safety_audit["closed_strokes_without_native_geometry_fit"] = sum(
        (getattr(s, "closed", False) or bool(getattr(s, "primitive", "")))
        and not getattr(s, "source_fit", {}).get("serialized_geometry_validated") for s in stroke_list)
    if stroke_safety_audit["palette_fragment_candidate_pixels_deferred"]:
        pre_notes.append("palette fragments not covering their whole source ink component "
                         "retained as fills; antialias outlines are not editable strokes")
    if stroke_safety_audit["complex_strokes_without_native_cap_proof"]:
        pre_notes.append("complex strokes lack native-source endpoint proof; manual review required")

    checkpoint("stroke_analysis_complete", "筆畫分析完成")

    # Recover the true core color (and sub-pixel trace width) from the
    # original-resolution source when max_size downsampling turned a 1 px
    # black line into a gray trace-scale pixel.
    if stroke_list and (orig_w != W or orig_h != H):
        try:
            orig_a = np.asarray(Image.open(src).convert("RGBA"))
            ob = np.concatenate([orig_a[0, :, :3], orig_a[-1, :, :3],
                                 orig_a[:, 0, :3], orig_a[:, -1, :3]])
            obg = np.median(ob.astype(np.float32), axis=0)
            sx, sy = orig_w / float(W), orig_h / float(H)
            trace_scale = (W / float(orig_w) + H / float(orig_h)) / 2.0
            for s in stroke_list:
                if getattr(s, "source_fit", {}).get("serialized_geometry_validated"):
                    continue  # native width/colour/caps were already jointly proven
                pts = s.sample_points
                if not pts:
                    continue
                step = max(1, len(pts) // 256)
                samples = []
                alphas = []
                for px, py in pts[::step]:
                    ox = min(orig_w - 1, max(0, int(round(px * sx - 0.5))))
                    oy = min(orig_h - 1, max(0, int(round(py * sy - 0.5))))
                    x0, x1 = max(0, ox - 1), min(orig_w, ox + 2)
                    y0, y1 = max(0, oy - 1), min(orig_h, oy + 2)
                    patch = orig_a[y0:y1, x0:x1].reshape(-1, 4)
                    patch = patch[patch[:, 3] >= 16]
                    if len(patch):
                        samples.extend(patch[:, :3].astype(np.float32))
                        alphas.extend(patch[:, 3].astype(np.float32))
                if samples:
                    sm = np.asarray(samples, dtype=np.float32)
                    dist = ((sm - obg) ** 2).sum(1)
                    core = sm[np.argsort(dist)[max(0, int(0.70 * len(sm))):]]
                    col = np.median(core if len(core) else sm, axis=0)
                    s.color = _snap_stroke_color_to_palette(
                        col, initial_palette)
                if alphas:
                    av = np.sort(np.asarray(alphas, dtype=np.float32))
                    op = float(np.median(av[len(av) // 2:])) / 255.0
                    s.opacity = 1.0 if op >= 0.985 else round(max(0.02, op), 3)
                if s.width <= 1.25:
                    s.width = round(max(0.45, s.width * trace_scale), 2)
        except Exception as e:
            pre_notes.append(f"original-resolution stroke sampling skipped: {e!r}"[:160])

    # A detected circular outline encloses negative space.  When the outer
    # background was removed, matching background-colored pixels inside the
    # ring should stay transparent rather than becoming a white disk.
    hole_mask = np.zeros((H, W), dtype=bool)
    if removed:
        yy, xx = np.ogrid[:H, :W]
        bg_arr = np.asarray(bg_col, dtype=np.float32)
        bg_like = np.abs(den - bg_arr).max(axis=2) <= 36
        for s in stroke_list:
            enclosure = None
            if s.primitive == "circle":
                inner = max(0.0, s.radius - s.width / 2.0 - 0.5)
                enclosure = (((xx - s.cx) ** 2 + (yy - s.cy) ** 2)
                             <= inner * inner)
            elif s.primitive == "rect":
                inset = s.width / 2.0 + 0.5
                enclosure = ((xx >= s.x + inset)
                             & (xx <= s.x + s.shape_width - inset)
                             & (yy >= s.y + inset)
                             & (yy <= s.y + s.height - inset))
            if enclosure is not None:
                hole_mask |= _broad_enclosed_background_mask(
                    visible, bg_like, enclosure)

    vis_fill = visible & ~hole_mask
    if stroke_list:
        vis_fill &= ~stroke_mask
        # leftovers too small to trace are antialiasing residue, not shapes
        if vis_fill.sum() < max(16, int(0.001 * visible.sum())):
            vis_fill = np.zeros_like(vis_fill)
        if not forced_colors and vis_fill.any():
            # re-estimate the palette without the stroke pixels so that
            # antialiasing-gray clusters created only by thin lines vanish
            palette_fill_samples = vis_fill | background_counter_mask
            palette, _, fill_palette_audit = detect_palette(
                den_palette, palette_fill_samples, forced=0,
                return_audit=True, return_labels=False)
            palette_audit = {
                "initial": palette_audit,
                "fill_after_strokes": fill_palette_audit,
            }
            lab_all = _assign(den, palette)
            palette_opacity = _palette_opacities(vis_fill, palette, lab_all)

    accent_retention_audit = {
        "policy": "initial_palette_connected_residual_retention",
        "colors_retained": 0,
        "records": [],
    }
    palette_analysis_fill = vis_fill | background_counter_mask
    if vis_fill.any() and len(initial_palette) and len(palette):
        palette, lab_all, accent_retention_audit = _retain_initial_accent_colors(
            den, palette_analysis_fill, initial_palette, palette, lab_all)
        if accent_retention_audit["colors_retained"]:
            paints = ", ".join(
                record["paint"] for record in accent_retention_audit["records"])
            pre_notes.append(
                "small coherent accent paint retained after stroke-aware "
                f"palette re-estimation: {paints}")

    linear_detail_audit = {
        "policy": "small_strong_ink_pca_linear_multicolour_only",
        "components_stabilized": 0,
        "pixels_relabelled": 0,
        "components": [],
    }
    if vis_fill.any() and len(palette) >= 2:
        # The fill-only palette intentionally drops colours represented mostly
        # by extracted strokes.  A fragmented sibling line may nevertheless
        # need one of those paints, so make the original whole-art palette
        # available to this narrowly guarded recovery pass.  Existing colours
        # stay first, preserving every pre-existing label index.
        base_detail_palette_count = len(palette)
        detail_palette = [tuple(int(v) for v in colour) for colour in palette]
        seen_detail_colours = set(detail_palette)
        for colour in initial_palette:
            key = tuple(int(v) for v in colour)
            if key not in seen_detail_colours:
                detail_palette.append(key)
                seen_detail_colours.add(key)
        detail_palette = np.asarray(detail_palette, dtype=np.uint8)
        lab_all, linear_detail_audit = _stabilize_fragmented_linear_details(
            den, palette_analysis_fill, detail_palette, lab_all, bg_col,
            owned_mask=stroke_mask)
        if linear_detail_audit["components_stabilized"]:
            # Keep only supplemental colours that the stabilizer actually
            # selected.  Leaving every initial colour in the nearest-colour
            # map would let unrelated VTracer paths snap to otherwise unused
            # paints and inflate both colour controls and visual noise.
            supplemental_used = sorted(
                int(value) for value in np.unique(lab_all[vis_fill])
                if int(value) >= base_detail_palette_count)
            if supplemental_used:
                compact_palette = np.vstack((
                    palette, detail_palette[supplemental_used]))
                for compact_index, expanded_index in enumerate(
                        supplemental_used, start=base_detail_palette_count):
                    lab_all[lab_all == expanded_index] = compact_index
                palette = compact_palette.astype(np.uint8)
            pre_notes.append(
                f"{linear_detail_audit['components_stabilized']} small "
                "multicolour line detail(s) stabilised before cutout tracing")
    if isinstance(palette_audit, dict):
        palette_audit["initial_accent_retention"] = accent_retention_audit
        palette_audit["linear_detail_stabilization"] = linear_detail_audit
        palette_audit["stroke_reconstruction"] = stroke_safety_audit
    # This alternative estimates colours independently of accepted strokes.
    # It never changes flat tracing, fill ownership, opacity or stroke output.
    # The gradient stage may inspect its paired labels under the same total
    # candidate budget; source paint and geometry gates still decide delivery.
    sampling_hypothesis = None
    from palette_sampling import independent_sampling_mask
    sampling_mask, sampling_audit = independent_sampling_mask(
        den, visible, background_counter_mask)
    if forced_colors:
        sampling_audit.update(status="baseline", reason="explicit_color_count")
    elif sampling_audit["status"] == "proposed":
        sampling_palette, _, sampling_estimator_audit = detect_palette(
            den_palette, sampling_mask, forced=0,
            return_audit=True, return_labels=False)
        sampling_labels = _assign(den, sampling_palette)
        sampling_palette, sampling_labels, sampling_accent_audit = (
            _retain_initial_accent_colors(
                den, sampling_mask, initial_palette,
                sampling_palette, sampling_labels))
        if (np.array_equal(sampling_palette, palette)
                and np.array_equal(sampling_labels, lab_all)):
            sampling_audit.update(status="baseline", reason="identical_palette_and_labels")
        else:
            sampling_hypothesis = {"palette": sampling_palette,
                                   "lab_all": sampling_labels}
        sampling_audit["estimator"] = sampling_estimator_audit
        sampling_audit["accent_retention"] = sampling_accent_audit
    palette_audit["independent_gradient_sampling"] = sampling_audit
    palette_opacity = _palette_opacities(vis_fill, palette, lab_all)
    checkpoint("flat_reference", "正在建立扁平化品質參考")

    # The exact encoded bytes are retained so every cache hit can restore the
    # same candidate-validation artifact without invoking Pillow again.
    flat = palette[lab_all]
    flat_alpha = np.where(visible, source_alpha, 0).astype(np.uint8)
    flat_rgba_full = np.dstack([flat, flat_alpha])
    flat_png = io.BytesIO()
    Image.fromarray(flat_rgba_full, "RGBA").save(flat_png, format="PNG")

    return {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "removed": removed,
        "source_alpha": source_alpha,
        "visible": visible,
        "H": H,
        "W": W,
        "pre_notes": pre_notes,
        "den": den,
        "palette": palette,
        "lab_all": lab_all,
        "palette_opacity": palette_opacity,
        "stroke_list": stroke_list,
        "bg_col": bg_col,
        "hole_mask": hole_mask,
        "vis_fill": vis_fill,
        "palette_audit": palette_audit,
        "flat": flat,
        "flat_png_bytes": flat_png.getvalue(),
        "sampling_hypothesis": sampling_hypothesis,
    }


def build_clean_base(src, dst, forced_colors=0, white_threshold=220,
                     regularize=True, flat_out=None,
                     background="auto", max_size=0, geometry=None,
                     strokes="on", gradients="on",
                     curve_error_percent=0.25,
                     gradient_stage_cache=None,
                     progress=None, control=None,
                     pre_gradient_state_cache=None,
                     _gradient_withdrawals=(), _gradient_retry_state=None,
                     _residual_scene_probe=True):
    """Convert a bitmap into a grouped, optionally regularized SVG.

    background: "auto" (heuristic removal of light border-connected
                background), "keep" (never remove), "transparent" (force
                removal of light border-connected regions).
    max_size:   downscale the longest image side before tracing (0 = off).
                The SVG width/height still reflect the original size.
    geometry:   "off" | "conservative" | "normal". If None, falls back to
                the legacy boolean `regularize` (True -> "normal").
    curve_error_percent: object-bbox-normalised p95 geometry budget used by
                 curve and native gradient-object reconstruction.
    gradient_stage_cache: optional process-local cache created once by
                 ``vector_cleanroom.process_one`` and shared only among that
                 image's candidate builds.
    pre_gradient_state_cache: optional process-local exact cache for the
                 stroke-aware state immediately before gradient reconstruction.
    """
    src, dst = Path(src), Path(dst)

    def _checkpoint(substage, detail=""):
        if control is not None:
            control.checkpoint(f"candidate_build:{substage}")
        if progress is not None:
            event = {
                "stage": "candidate_build",
                "substage": str(substage),
                "detail": str(detail),
            }
            if control is not None and hasattr(control, "snapshot"):
                event.update(control.snapshot())
                event["stage"] = "candidate_build"
                event["substage"] = str(substage)
            progress(event)

    _checkpoint("prepare_source", "正在清理背景與建立來源遮罩")
    if geometry is None:
        geometry = "normal" if regularize else "off"
    if geometry not in ("off", "conservative", "normal"):
        raise ValueError(f"invalid geometry level: {geometry!r}")
    curve_error_percent = float(curve_error_percent)
    if (not math.isfinite(curve_error_percent)
            or not 0.05 <= curve_error_percent <= 2.0):
        raise ValueError("curve_error_percent must be between 0.05 and 2.0")

    pre_gradient_parameters = {
        "forced_colors": int(forced_colors),
        "white_threshold": int(white_threshold),
        "background": str(background),
        "max_size": int(max_size),
        "strokes": str(strokes),
    }
    pre_gradient_key = None
    if pre_gradient_state_cache is not None:
        pre_gradient_key = _pre_gradient_state_cache_key(
            src, pre_gradient_parameters, _produce_pre_gradient_state,
            PRE_GRADIENT_STATE_CACHE_SCHEMA)

    def _produce_prefix():
        return _produce_pre_gradient_state(
            src, forced_colors=forced_colors,
            white_threshold=white_threshold, background=background,
            max_size=max_size, strokes=strokes, checkpoint=_checkpoint)

    if _gradient_retry_state is None:
        pre_gradient_state, pre_gradient_cache_evidence = (
            _pre_gradient_state_cache_get_or_compute(
                pre_gradient_state_cache, pre_gradient_key, _produce_prefix))
    else:
        pre_gradient_state = copy.deepcopy(_gradient_retry_state["prefix"])
        pre_gradient_cache_evidence = {
            "schema": PRE_GRADIENT_STATE_CACHE_SCHEMA,
            "status": "fragmentation_retry_exact_reuse"}
    retry_prefix = copy.deepcopy(pre_gradient_state) if gradients != "off" else None
    original_gradient_stage = None
    orig_w = pre_gradient_state["orig_w"]
    orig_h = pre_gradient_state["orig_h"]
    removed = pre_gradient_state["removed"]
    source_alpha = pre_gradient_state["source_alpha"]
    visible = pre_gradient_state["visible"]
    H = pre_gradient_state["H"]
    W = pre_gradient_state["W"]
    pre_notes = pre_gradient_state["pre_notes"]
    den = pre_gradient_state["den"]
    palette = pre_gradient_state["palette"]
    lab_all = pre_gradient_state["lab_all"]
    palette_opacity = pre_gradient_state["palette_opacity"]
    stroke_list = pre_gradient_state["stroke_list"]
    bg_col = pre_gradient_state["bg_col"]
    hole_mask = pre_gradient_state["hole_mask"]
    vis_fill = pre_gradient_state["vis_fill"]
    palette_audit = pre_gradient_state["palette_audit"]
    flat = pre_gradient_state["flat"]
    sampling_hypothesis = pre_gradient_state.get("sampling_hypothesis")
    if isinstance(palette_audit, dict):
        palette_audit["pre_gradient_state_cache"] = (
            pre_gradient_cache_evidence)
    if flat_out:
        Path(flat_out).write_bytes(pre_gradient_state["flat_png_bytes"])

    _checkpoint("gradient_reconstruction", "正在驗證原生漸層候選")

    # Gradient-object reconstruction is intentionally source-space first:
    # held-out colour evidence selects one bounded native paint model, while a
    # separate normalised geometry contract minimises anchors.  Palette bands
    # are never treated as the desired object boundaries.
    grad_regions = []
    gradient_reconstruction_audit = {
        "status": "disabled" if gradients == "off" else "skipped",
        "summary": {},
        "cache": (_gradient_stage_cache_bypass(
            gradient_stage_cache, "gradients_off")
            if gradients == "off" else {
                "schema": GRADIENT_STAGE_CACHE_SCHEMA,
                "status": "pending",
            }),
    }
    if gradients != "off" and vis_fill.any():
        cache_evidence = {
            "schema": GRADIENT_STAGE_CACHE_SCHEMA,
            "status": "error",
        }
        try:
            from gradient_reconstruction_stage import (
                SCHEMA as GRADIENT_RECONSTRUCTION_SCHEMA,
                propose_gradient_reconstruction,
            )
            from palette_sampling import (
                discover_with_sampling_hypothesis, SCHEMA as SAMPLING_SCHEMA)
            stage_parameters = {
                "geometry_error_percent": curve_error_percent,
                "geometry_smooth": 0.55,
                "max_candidates": 48,
                "max_geometry_candidates": 16,
                "max_objects": 8,
                "max_segments_per_object": 4096,
                "selection_beam_width": 256,
                "candidate_options": None,
                "model_options": None,
            }
            cache_state = _normalise_gradient_stage_cache(
                gradient_stage_cache)
            shared_geometry_fit_cache = (
                cache_state["geometry_fit_cache"]
                if cache_state is not None else None)
            try:
                with Image.open(src) as original_image:
                    original_source_rgba = np.asarray(original_image.convert("RGBA"), dtype=np.uint8)
            except OSError:
                original_source_rgba = None
            # Keep the key tied to the exact arrays presented to the stage,
            # not merely the source path or high-level candidate options.
            cache_key = _gradient_stage_cache_key((
                ("den", den),
                ("lab_all", lab_all),
                ("vis_fill", vis_fill),
                ("palette", palette),
                ("source_alpha", source_alpha),
                ("original_source_rgba", original_source_rgba),
                ("sampling_palette", (sampling_hypothesis or {}).get(
                    "palette", np.zeros(1, dtype=np.uint8))),
                ("sampling_labels", (sampling_hypothesis or {}).get(
                    "lab_all", np.zeros(1, dtype=np.int32))),
            ), stage_parameters, propose_gradient_reconstruction,
                GRADIENT_RECONSTRUCTION_SCHEMA + ":" + SAMPLING_SCHEMA)

            def _produce_gradient_stage():
                discovery_kwargs = {}
                if sampling_hypothesis is not None:
                    def sampling_provider(*args, **kwargs):
                        return discover_with_sampling_hypothesis(
                            *args, hypothesis=sampling_hypothesis, **kwargs)
                    discovery_kwargs["candidate_provider"] = sampling_provider
                return propose_gradient_reconstruction(
                    den,
                    lab_all,
                    vis_fill,
                    palette,
                    alpha=source_alpha,
                    original_source_rgba=original_source_rgba,
                    geometry_fit_cache=shared_geometry_fit_cache,
                    progress=progress,
                    control=control,
                    **discovery_kwargs,
                    **stage_parameters,
                )

            if _gradient_retry_state is None:
                original_gradient_stage, cache_evidence = (
                    _gradient_stage_cache_get_or_compute(
                        gradient_stage_cache, cache_key,
                        _produce_gradient_stage))
            else:
                original_gradient_stage = _gradient_retry_state["stage"]
                cache_evidence = {
                    "schema": GRADIENT_STAGE_CACHE_SCHEMA,
                    "status": "fragmentation_retry_exact_reuse"}
            gradient_stage = _withdraw_gradient_proposals(
                original_gradient_stage, _gradient_withdrawals)
            grad_regions = [
                _gradient_region_from_proposal(proposal)
                for proposal in gradient_stage.get("proposals", [])
            ]
            gradient_reconstruction_audit = {
                "schema": gradient_stage.get("schema"),
                "status": gradient_stage.get("status"),
                "summary": gradient_stage.get("summary") or {},
                "objective": gradient_stage.get("objective") or {},
                "parameters": gradient_stage.get("parameters") or {},
                "decisions": gradient_stage.get("decisions") or [],
                "fragmentation_withdrawals": list(_gradient_withdrawals),
                "cache": cache_evidence,
            }
        except ConversionInterrupted:
            raise
        except Exception as e:
            grad_regions = []
            gradient_reconstruction_audit = {
                "status": "error",
                "summary": {},
                "error": f"{type(e).__name__}: {e}"[:240],
                "cache": cache_evidence,
            }
            pre_notes.append(
                f"source-space gradient reconstruction failed closed: {e!r}"[:200])
    elif gradients != "off":
        gradient_reconstruction_audit["cache"] = (
            _gradient_stage_cache_bypass(
                gradient_stage_cache, "no_visible_fill"))
    if isinstance(palette_audit, dict):
        palette_audit["gradient_reconstruction"] = gradient_reconstruction_audit
    _checkpoint("gradient_reconstruction_complete", "漸層候選驗證完成")
    grad_keys = []
    if grad_regions:
        try:
            grad_keys = _allocate_gradient_keys(palette, len(grad_regions))
        except Exception as e:
            grad_regions = []
            pre_notes.append(
                f"gradient reconstruction disabled: placeholder isolation failed: {e!r}"[:200])
    if grad_regions:
        flat = flat.copy()
        for g, key in zip(grad_regions, grad_keys):
            g["key"] = key
            g["key_distance"] = min(
                int(np.abs(palette.astype(np.int16)
                           - np.asarray(key, dtype=np.int16)).max(axis=1).min()),
                255,
            ) if len(palette) else 255
            avals = source_alpha[g["mask"]]
            gop = (float(np.quantile(avals.astype(np.float32), 0.75)) / 255.0
                   if avals.size else 1.0)
            g["opacity"] = 1.0 if gop >= 0.985 else round(max(0.02, gop), 3)
            flat[g["mask"]] = key

    # VTracer shifts boundary colours when it receives transparent RGB: the
    # transparent edge is averaged into stacked/cutout regions before its
    # fill is reported, which can remap a dark glyph to a lighter palette
    # colour.  Give the *working* raster an isolated opaque routing surround,
    # then remove only that temporary geometry after tracing.  The delivered
    # SVG remains transparent and the separate flat reference above retains
    # the source alpha exactly.
    trace_vis_fill=vis_fill
    trace_rgb = flat.copy()
    if any(getattr(s,'source_fit',{}).get('policy')=='native_discrete_palette_line_occupancy'
           for s in stroke_list):
        from stroke_engine import infer_occluded_flat_paint
        with Image.open(src) as source_image:
            native_rgba=np.asarray(source_image.convert('RGBA'))
        support,support_paint=infer_occluded_flat_paint(
            stroke_list,native_rgba,den,visible,vis_fill,palette)
        if support.any():
            trace_vis_fill=vis_fill|support
            trace_rgb[support]=support_paint[support]
            palette_audit['stroke_underpaint']={
                'policy':'native_equal_flat_paint_on_both_normal_sides',
                'inferred_occluded_working_pixels':int(support.sum()),
                'original_reference_unchanged':True,
                'scope':'trace_surrogate_only_requires_final_composed_scene_validation'}
    trace_opaque_background = bool((~trace_vis_fill).any())
    trace_background_rgb = None
    trace_background_isolated = False
    if trace_opaque_background:
        occupied_trace_colours = {
            tuple(int(channel) for channel in colour) for colour in palette
        }
        occupied_trace_colours.update(
            tuple(int(channel) for channel in region.get("key", ()))
            for region in grad_regions if region.get("key") is not None)
        # Keep the routing surround neutral for fast, stable edge tracing.
        # Genuine white/light objects are reconstructed independently below,
        # so they no longer depend on VTracer distinguishing them from this
        # temporary near-white canvas.
        for level in range(255, 223, -1):
            candidate = (level, level, level)
            if all(max(abs(level - channel) for channel in colour) >= 8
                   for colour in occupied_trace_colours):
                trace_background_rgb = candidate
                trace_background_isolated = True
                break
        if trace_background_rgb is None:
            trace_background_rgb = (255, 255, 255)
        trace_rgb[~trace_vis_fill] = trace_background_rgb
        trace_alpha = np.full((H, W), 255, dtype=np.uint8)
    else:
        trace_alpha = np.where(trace_vis_fill, 255, 0).astype(np.uint8)
    flat_rgba = np.dstack([trace_rgb, trace_alpha])

    raw = ""
    if vis_fill.any():
        # Cutout tracing no longer creates the large cumulative unions that
        # needed aggressive speckle suppression.  Keep small intentional logo
        # details (sun rays, dots, diacritics) while still dropping 1px noise.
        # Logos often use intentional micro-details (halftone dots, star
        # points, registration marks).  A resolution-scaled speckle cutoff
        # silently erased those details on large source images before the
        # native-circle pass could simplify them.  Two pixels remains a small
        # noise guard while preserving design components for later validation.
        speckle = 2
        _checkpoint("main_trace", "正在描摹主要向量輪廓")
        with tempfile.TemporaryDirectory() as td:
            flat_png = Path(td) / "flat.png"
            raw_svg = Path(td) / "raw.svg"
            Image.fromarray(flat_rgba, "RGBA").save(flat_png)
            vtracer.convert_image_to_svg_py(
                str(flat_png), str(raw_svg),
                colormode="color", hierarchical="cutout", mode="spline",
                filter_speckle=speckle, color_precision=8, layer_difference=0,
                corner_threshold=58, length_threshold=5.0, splice_threshold=45,
                path_precision=6,
            )
            raw = raw_svg.read_text(encoding="utf-8")
        try:
            from trace_component_recovery import recover_missing_source_components

            # vis_fill already excludes recovered strokes and removed
            # background in the cached pre-gradient state.
            recovery_excluded = ~np.asarray(vis_fill, dtype=bool)
            for region in grad_regions:
                recovery_excluded |= np.asarray(region["mask"], dtype=bool)
            raw, component_recovery = recover_missing_source_components(
                raw, flat_rgba, excluded_mask=recovery_excluded)
            if component_recovery.get("recovered_count", 0):
                pre_notes.append(
                    f'{component_recovery["recovered_count"]} wholly omitted small '
                    "source compartment(s) restored before normal render validation")
        except ConversionInterrupted:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            component_recovery = {
                "schema": "ai-vector-cleanroom.trace-component-recovery/v1",
                "status": "skipped_error", "recovered_count": 0,
                "error": f"{type(exc).__name__}: {exc}"[:240],
                "downstream_render_validation_required": True}
            pre_notes.append("small source compartment recovery unavailable; "
                             "original trace still requires normal source validation")
        if isinstance(palette_audit, dict):
            palette_audit["trace_component_recovery"] = component_recovery
        _checkpoint("main_trace_complete", "主要向量輪廓完成")

    # Cutout tracing can encode white lettering as holes in a darker parent
    # instead of as white objects.  That looks correct on the normal white
    # review canvas but fails as soon as a designer places the SVG on colour.
    # Trace each *visible internal* light paint independently as a binary mask
    # and later replace any ambiguous main-trace light paths with these real
    # overlay objects.  External background pixels are absent from vis_fill,
    # so they can never leak into this recovery pass.
    light_overlay_entries = {}
    if trace_opaque_background and vis_fill.any():
        _checkpoint("light_paint_trace", "正在保留透明背景上的淺色物件")
        gradient_owned = np.zeros_like(vis_fill)
        for region in grad_regions:
            gradient_owned |= np.asarray(region.get("mask"), dtype=bool)
        background_arr = np.asarray(bg_col, dtype=np.float32)
        light_indices = []
        for ci, colour in enumerate(palette):
            colour_f = colour.astype(np.float32)
            if (float(colour_f.mean()) >= 210.0
                    and float(np.abs(colour_f - background_arr).max()) <= 72.0
                    and int((vis_fill & ~gradient_owned
                             & (lab_all == ci)).sum()) >= 8):
                light_indices.append(ci)
        if light_indices:
            with tempfile.TemporaryDirectory() as light_td:
                light_td = Path(light_td)
                markers = _allocate_gradient_keys(
                    np.asarray(((255, 255, 255),), dtype=np.uint8),
                    len(light_indices), min_distance=48)
                marker_array = np.asarray(markers, dtype=np.int16)
                binary_rgb = np.full((H, W, 3), 255, dtype=np.uint8)
                for ci, marker in zip(light_indices, markers):
                    mask = vis_fill & ~gradient_owned & (lab_all == ci)
                    binary_rgb[mask] = marker
                binary_rgba = np.dstack((
                    binary_rgb, np.full((H, W), 255, dtype=np.uint8)))
                mask_png = light_td / "light-paints.png"
                mask_svg = light_td / "light-paints.svg"
                Image.fromarray(binary_rgba, "RGBA").save(mask_png)
                vtracer.convert_image_to_svg_py(
                    str(mask_png), str(mask_svg), colormode="color",
                    hierarchical="cutout", mode="spline", filter_speckle=2,
                    color_precision=8, layer_difference=0,
                    corner_threshold=58, length_threshold=5.0,
                    splice_threshold=45, path_precision=6,
                )
                light_raw = mask_svg.read_text(encoding="utf-8")
                try:
                    from trace_component_recovery import recover_missing_source_components

                    # Marker colours exaggerate antialias crumbs. Reopen only
                    # source-white compartments already proven missing in the
                    # real-colour main trace; never infer new marks here.
                    light_excluded = _proven_light_compartment_exclusions(
                        vis_fill.shape, component_recovery)
                    light_raw, light_recovery = recover_missing_source_components(
                        light_raw, binary_rgba, excluded_mask=light_excluded)
                    if light_recovery.get("recovered_count", 0):
                        pre_notes.append(
                            f'{light_recovery["recovered_count"]} wholly omitted light '
                            "source compartment(s) restored before normal render validation")
                except ConversionInterrupted:
                    raise
                except (OSError, ValueError, RuntimeError) as exc:
                    light_recovery = {
                        "schema": "ai-vector-cleanroom.trace-component-recovery/v1",
                        "status": "skipped_error", "recovered_count": 0,
                        "error": f"{type(exc).__name__}: {exc}"[:240],
                        "downstream_render_validation_required": True}
                if isinstance(palette_audit, dict):
                    palette_audit["light_trace_component_recovery"] = light_recovery
                for d, fill, tx, ty in _iter_svg_paths(light_raw):
                    rgb = np.asarray(tuple(
                        int(fill[index:index + 2], 16)
                        for index in (1, 3, 5)), dtype=np.int16)
                    if int(np.abs(rgb - 255).max()) <= 32:
                        continue
                    marker_index = int(
                        ((marker_array.astype(np.int32)
                          - rgb.astype(np.int32)) ** 2)
                        .sum(axis=1).argmin())
                    ci = light_indices[marker_index]
                    try:
                        subs = _parse_subpaths(d)
                        _offset_subs(subs, tx, ty)
                        xs, ys = [], []
                        for sub in subs:
                            _sub_bbox_accumulate(sub, xs, ys)
                        covers_canvas = (
                            xs and ys and min(xs) <= 1.0 and min(ys) <= 1.0
                            and max(xs) >= W - 1.0 and max(ys) >= H - 1.0)
                        if not covers_canvas:
                            light_overlay_entries.setdefault(ci, []).append(
                                {"color": ci, "subs": subs})
                    except Exception:
                        light_overlay_entries.setdefault(ci, []).append({
                            "color": ci,
                            "raw": _bake_translate(d, tx, ty),
                        })
        _checkpoint("light_paint_trace_complete", "淺色物件保留完成")

    pal_rgb = [tuple(int(v) for v in c) for c in palette]
    pal_opacity = list(palette_opacity)
    # gradient placeholder keys join the mapping so their traced shapes get
    # their own color index (grad_by_idx routes them at emission time)
    grad_by_idx = {}
    for g in grad_regions:
        grad_by_idx[len(pal_rgb)] = g
        pal_rgb.append(g["key"])
        pal_opacity.append(g.get("opacity", 1.0))
    pal_hex = ["#{:02x}{:02x}{:02x}".format(*c) for c in pal_rgb]

    def nearest_idx(hx):
        r, g, b = int(hx[1:3], 16), int(hx[3:5], 16), int(hx[5:7], 16)
        return min(range(len(pal_rgb)),
                   key=lambda i: (pal_rgb[i][0] - r) ** 2 + (pal_rgb[i][1] - g) ** 2
                                 + (pal_rgb[i][2] - b) ** 2)

    # Parse every path, preserving vtracer's stacking order (bottom to top).
    entries = []
    trace_background_paths_removed = 0
    for d, fill, tx, ty in _iter_svg_paths(raw):
        raw_rgb = tuple(int(fill[i:i + 2], 16) for i in (1, 3, 5))
        try:
            subs = _parse_subpaths(d)
            _offset_subs(subs, tx, ty)
            route_distance = (max(
                abs(raw_rgb[i] - trace_background_rgb[i]) for i in range(3))
                if trace_background_rgb is not None else 256)
            if trace_background_rgb is not None and route_distance <= 24:
                xs, ys = [], []
                for sub in subs:
                    _sub_bbox_accumulate(sub, xs, ys)
                covers_canvas = (
                    xs and ys and min(xs) <= 1.0 and min(ys) <= 1.0
                    and max(xs) >= W - 1.0 and max(ys) >= H - 1.0)
                if ((trace_background_isolated and route_distance <= 4)
                        or covers_canvas):
                    trace_background_paths_removed += 1
                    continue
            ci = nearest_idx(fill.lower())
            if ci in light_overlay_entries:
                continue
            entries.append({"color": ci, "subs": subs})
        except Exception:
            ci = nearest_idx(fill.lower())
            if ci in light_overlay_entries:
                continue
            entries.append({"color": ci, "raw": _bake_translate(d, tx, ty)})

    recovered_light_paths = 0
    for ci in sorted(light_overlay_entries,
                     key=lambda index: float(palette[index].mean())):
        recovered = light_overlay_entries[ci]
        entries.extend(recovered)
        recovered_light_paths += len(recovered)

    residual_entries, residual_pixels = _recover_unowned_gradient_residuals(
        entries, lab_all, vis_fill, grad_regions, len(palette))
    fragmentation = _gradient_residual_fragmentation(
        residual_entries, lab_all, vis_fill, grad_regions)
    if fragmentation:
        # A consumed palette also occurs in unrelated antialias pixels. Before
        # assigning its old trace scraps to any gradient, compare the actual
        # no-gradient trace. Only exact unchanged remote 3x3 RGBA support can
        # leave the recovery debt; changed/source-adjacent details remain.
        from gradient_residual_provenance import prove_preexisting_residuals
        owned = np.zeros_like(vis_fill, dtype=bool)
        for region in grad_regions:
            owned |= np.asarray(region["mask"], dtype=bool)
        indices = [int(entry["color"]) for entry in residual_entries]
        residual_mask = vis_fill & ~owned & np.isin(lab_all, indices)
        baseline_rgb = pre_gradient_state["flat"].copy()
        if trace_opaque_background:
            baseline_rgb[~vis_fill] = trace_background_rgb
        baseline_rgba = np.dstack((baseline_rgb, trace_alpha))
        try:
            preexisting, provenance = prove_preexisting_residuals(
                raw, baseline_rgba, residual_mask, owned, visible=vis_fill, source_rgb=den)
            palette_audit["gradient_residual_provenance"] = provenance
            if preexisting.any():
                residual_entries, residual_pixels = _recover_unowned_gradient_residuals(
                    entries, lab_all, vis_fill, grad_regions, len(palette),
                    proven_preexisting=preexisting)
                fragmentation = _gradient_residual_fragmentation(
                    residual_entries, lab_all, vis_fill, grad_regions)
        except (OSError, ValueError, RuntimeError) as exc:
            palette_audit["gradient_residual_provenance"] = {
                "status": "unavailable_original_recovery_policy_retained",
                "reason": str(exc)[:240]}
    residual_scene_counterfactual = None
    if fragmentation and fragmentation["candidate_ids"]:
        already_withdrawn = {item for record in _gradient_withdrawals
                             for item in record["candidate_ids"]}
        if set(fragmentation["candidate_ids"]) - already_withdrawn:
            _checkpoint("gradient_fragmentation_rollback",
                        "正在比較漸層與原始色層，確認是否真的增加碎片")
            if not _residual_scene_probe:
                return build_clean_base(
                    src, dst, forced_colors=forced_colors,
                    white_threshold=white_threshold, regularize=regularize,
                    flat_out=flat_out, background=background, max_size=max_size,
                    geometry=geometry, strokes=strokes, gradients=gradients,
                    curve_error_percent=curve_error_percent,
                    gradient_stage_cache=gradient_stage_cache, progress=progress,
                    control=control, pre_gradient_state_cache=pre_gradient_state_cache,
                    _gradient_withdrawals=(*_gradient_withdrawals, fragmentation),
                    _gradient_retry_state={"prefix": retry_prefix, "stage": original_gradient_stage},
                    _residual_scene_probe=False)
            with tempfile.TemporaryDirectory() as baseline_dir:
                baseline_svg = Path(baseline_dir) / "baseline.svg"
                baseline_stats = build_clean_base(
                src, baseline_svg, forced_colors=forced_colors,
                white_threshold=white_threshold, regularize=regularize,
                flat_out=flat_out, background=background, max_size=max_size,
                geometry=geometry, strokes=strokes, gradients=gradients,
                curve_error_percent=curve_error_percent,
                gradient_stage_cache=gradient_stage_cache, progress=progress,
                control=control, pre_gradient_state_cache=pre_gradient_state_cache,
                _gradient_withdrawals=(*_gradient_withdrawals, fragmentation),
                _gradient_retry_state={"prefix": retry_prefix,
                                       "stage": original_gradient_stage},
                _residual_scene_probe=False)
                residual_scene_counterfactual = {
                    "svg": baseline_svg.read_text(encoding="utf8"), "stats": baseline_stats,
                    "fragmentation": fragmentation, "residual_mask": residual_mask}
            # The candidate is tentative. It may omit recovery scraps only
            # if its complete final native rendering proves them unchanged;
            # otherwise the counterfactual's exact SVG and stats win below.
            residual_entries, residual_pixels = [], 0

    def _span_scene_search_summary():
        scenes = []
        audits = [("assembled_candidate", palette_audit)]
        if residual_scene_counterfactual is not None:
            audits.insert(0, ("withdrawal_counterfactual", residual_scene_counterfactual["stats"].palette_audit))
        for label, audit in audits:
            transaction = audit.get("gradient_source_ownership_transaction") or {}
            performance = transaction.get("performance") or {}
            search = performance.get("bounded_span_search")
            if isinstance(search, dict):
                row = {"scene": label, **copy.deepcopy(search)}
                cached = transaction.get("execution_cache") or {}
                row["evidence_reused_for_exact_scene"] = cached.get("status") == "hit"
                if row["evidence_reused_for_exact_scene"]:
                    row["original_validation_search"] = copy.deepcopy(search)
                    row.update(native_probe_count=0, helper_calls=0)
                    # Retain timing under its original-validation scope only.
                    row.update(elapsed_seconds=0.0, wall_seconds_since_first_request=0.0)
                scenes.append(row)
        return {
            "scope": "this_clean_base_candidate_residual_transaction_not_all_cli_fallback_variants",
            "assembled_scene_search_count": sum(not row["evidence_reused_for_exact_scene"] for row in scenes),
            "assembled_scene_evidence_count": len(scenes),
            "maximum_assembled_scenes": 2,
            "native_probe_count": sum(int(row.get("native_probe_count", 0)) for row in scenes),
            "helper_calls": sum(int(row.get("helper_calls", 0)) for row in scenes),
            "exact_scene_cache_hits": sum(bool(row.get("evidence_reused_for_exact_scene")) for row in scenes),
            "scenes": scenes,
            "independent_scene_evidence": True,
        }

    def _return_residual_counterfactual(reason):
        baseline_stats = residual_scene_counterfactual["stats"]
        baseline_stats.palette_audit["gradient_residual_scene_transaction"] = {
            "status": "rolled_back_to_verified_counterfactual", "reason": str(reason)[:240]}
        baseline_stats.palette_audit["gradient_contour_span_scene_searches"] = _span_scene_search_summary()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(residual_scene_counterfactual["svg"], encoding="utf8")
        return baseline_stats
    if _gradient_withdrawals:
        pre_notes.append(
            f"{sum(len(record['candidate_ids']) for record in _gradient_withdrawals)} "
            "gradient replacement(s) withdrawn because they required fragmented "
            "palette residuals; original palette retraced without pixel scraps")
    entries.extend(residual_entries)
    if residual_entries:
        pre_notes.append(
            f"{residual_pixels} unowned source pixel(s) in {len(residual_entries)} "
            "omitted palette colour(s) explicitly restored outside gradient ownership")

    if trace_background_paths_removed:
        pre_notes.append(
            f"{trace_background_paths_removed} tracer-only routing background "
            "path(s) removed after opaque-edge colour stabilisation")
    if recovered_light_paths:
        pre_notes.append(
            f"{recovered_light_paths} internal light-paint path(s) explicitly "
            "preserved for transparent-background fidelity")

    geometry_notes = []
    if geometry != "off":
        _checkpoint("geometry_regularization", "正在規則化圓形、線段與輪廓")
        try:
            geometry_notes = _regularize(entries, level=geometry)
        except Exception as e:
            geometry_notes = []
            pre_notes.append(f"geometry regularization disabled by error: {e!r}"[:160])
        _checkpoint("geometry_regularization_complete", "幾何規則化完成")

    # 幾何規則化後再算外框（座標可能被調整過）
    for e in entries:
        if "subs" in e:
            xs, ys = [], []
            for sub in e["subs"]:
                _sub_bbox_accumulate(sub, xs, ys)
            e["bbox"] = (min(xs), min(ys), max(xs), max(ys))
        else:
            e["bbox"] = None   # 無法判斷 → 視為與所有東西重疊

    # 依堆疊順序切成同色連續段
    runs = []
    for e in entries:
        if runs and runs[-1]["color"] == e["color"]:
            runs[-1]["items"].append(e)
        else:
            runs.append({"color": e["color"], "items": [e]})

    def _overlap(ra, rb):
        for x in ra["items"]:
            for y in rb["items"]:
                bx, by = x["bbox"], y["bbox"]
                if bx is None or by is None:
                    return True
                if not (bx[2] < by[0] or by[2] < bx[0] or bx[3] < by[1] or by[3] < bx[1]):
                    return True
        return False

    # 同色段不與中間形狀重疊時「下沉」合併（減少群組數，且渲染不變）。
    # 每一輪：把每個 run 盡量往下沉，碰到同色就合併；j 單調遞減、i 單調前進，
    # 保證終止（先前的互相讓位寫法會無限交換）。
    for _pass in range(4):
        merged_any = False
        i = 1
        while i < len(runs):
            j = i
            while (j > 0
                   and runs[j]["color"] != runs[j - 1]["color"]
                   and any(r2["color"] == runs[j]["color"] for r2 in runs[:j - 1])
                   and not _overlap(runs[j], runs[j - 1])):
                runs[j - 1], runs[j] = runs[j], runs[j - 1]
                j -= 1
            if j > 0 and runs[j]["color"] == runs[j - 1]["color"]:
                runs[j - 1]["items"] += runs[j]["items"]
                runs.pop(j)
                merged_any = True
                continue    # i 指到的內容已變，原地重試
            i += 1
        if not merged_any:
            break

    used_names = {}
    title = xml_escape(f"{src.stem} vector cleanroom result")
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
         f'width="{orig_w}" height="{orig_h}" viewBox="0 0 {W} {H}" version="1.1" '
         f'shape-rendering="geometricPrecision">'),
        f"  <title>{title}</title>",
        "  <desc>Editable vector approximation. Layers follow stack order; "
        "uniform-width line work is rebuilt as real strokes; near-circles "
        "become native circle elements; continuous source colour fields are "
        "rebuilt as bounded native linear or radial gradients. No bitmap "
        "embedded.</desc>",
    ]
    grad_ids = {}
    if grad_by_idx:
        parts.append("  <defs>")
        for n, (idx, g) in enumerate(sorted(grad_by_idx.items()), 1):
            gid = f"grad{n}"
            grad_ids[idx] = gid
            g["id"] = gid
            model = g.get("model") or {
                "type": "linear", "x1": g["x1"], "y1": g["y1"],
                "x2": g["x2"], "y2": g["y2"],
            }
            model_type = model.get("type", "linear")
            if model_type == "radial":
                cx, cy = model["center"]
                rx = float(model["radius_x"])
                ry = float(model["radius_y"])
                angle = float(model.get("rotation_degrees", 0.0))
                parts.append(
                    f'    <radialGradient id="{gid}" '
                    f'gradientUnits="userSpaceOnUse" cx="0" cy="0" r="1" '
                    f'gradientTransform="translate({_f(cx)} {_f(cy)}) '
                    f'rotate({_f(angle)}) scale({_f(rx)} {_f(ry)})">')
                closing_tag = "radialGradient"
            else:
                parts.append(
                    f'    <linearGradient id="{gid}" '
                    f'gradientUnits="userSpaceOnUse" '
                    f'x1="{_f(model["x1"])}" y1="{_f(model["y1"])}" '
                    f'x2="{_f(model["x2"])}" y2="{_f(model["y2"])}">')
                closing_tag = "linearGradient"
            for stop in g["stops"]:
                if isinstance(stop, dict):
                    off = float(stop["offset"])
                    colour = stop.get("color")
                    if not isinstance(colour, str):
                        colour = "#{:02x}{:02x}{:02x}".format(
                            *(int(value) for value in stop["rgb"]))
                else:
                    off, colour_rgb = stop
                    colour = "#{:02x}{:02x}{:02x}".format(*colour_rgb)
                parts.append(f'      <stop offset="{_f(off)}" '
                             f'stop-color="{colour}"/>')
            parts.append(f"    </{closing_tag}>")
        parts.append("  </defs>")
    final_palette = []
    n_paths = 0
    n_native = 0
    n_fill_native_circles = 0
    n_nodes = 0

    def _concentric_ring(subs):
        if len(subs) != 2:
            return None
        a = subs[0].get("is_circle")
        b = subs[1].get("is_circle")
        if not (a and b):
            return None
        if math.hypot(a[0] - b[0], a[1] - b[1]) > 2.5:
            return None
        r_out, r_in = max(a[2], b[2]), min(a[2], b[2])
        if r_out - r_in < 1.0:
            return None
        cx = (a[0] + b[0]) / 2.0
        cy = (a[1] + b[1]) / 2.0
        return cx, cy, (r_out + r_in) / 2.0, r_out - r_in

    emitted_gradient_geometry = set()
    for run in runs:
        ci = run["color"]
        grad = grad_by_idx.get(ci)
        if grad is not None and ci in emitted_gradient_geometry:
            continue
        if grad is not None:
            base = "gradient"
            fill_attr = f'url(#{grad_ids[ci]})'
        else:
            base = color_name(pal_rgb[ci])
            fill_attr = None
        used_names[base] = used_names.get(base, 0) + 1
        nm = base if used_names[base] == 1 else f"{base}{used_names[base]}"
        nm = xml_escape(nm)
        group_fill = fill_attr if fill_attr else pal_hex[ci]
        op = pal_opacity[ci] if ci < len(pal_opacity) else 1.0
        opacity_attr = (f' fill-opacity="{_f(op)}"'
                        if op < 0.995 else "")
        parts.append(f'  <g id="{nm}" inkscape:label="{nm}" '
                     f'inkscape:groupmode="layer" fill="{group_fill}" '
                     f'fill-rule="evenodd"{opacity_attr}>')
        if grad is not None and grad.get("path_override"):
            # The placeholder trace establishes stack position only.  Actual
            # geometry comes from the independent error-budgeted optimiser,
            # so palette band edges cannot become designer anchors.
            from native_geometry_contract import whole_object_native_primitive

            primitive = whole_object_native_primitive(
                (grad.get("validation") or {}).get("geometry") or {},
                grad["path_override"])
            # Keep the native coordinates at the same precision as the
            # independently verified path.  Two-decimal formatting would
            # introduce a new, unmeasured geometry change at emission.
            native_number = lambda value: f'{float(value):.6f}'.rstrip('0').rstrip('.')
            designer_anchors = int(
                ((grad.get("validation") or {}).get("geometry") or {})
                .get("designer_anchor_count") or 0)
            if isinstance(primitive, dict) and primitive.get("element") in {"circle", "ellipse"}:
                designer_anchors = 4
            error_budget = (
                ((grad.get("validation") or {}).get("geometry") or {})
                .get("error_budget") or {})
            budget_percent = float(error_budget["requested_max_percent"])
            p95_percent = float(error_budget["actual_p95_error_percent"])
            max_percent = float(error_budget["actual_max_error_percent"])
            evidence = (
                f' id="avc-gradient-drawable-{xml_escape(str(grad_ids[ci]))}"'
                f' data-avc-gradient-object="{xml_escape(str(grad.get("proposal_id") or "source"))}"'
                f' data-avc-designer-anchors="{designer_anchors}"'
                f' data-avc-error-budget-percent="{_f(budget_percent)}"'
                f' data-avc-p95-error-percent="{_f(p95_percent)}"'
                f' data-avc-max-error-percent="{_f(max_percent)}"')
            if isinstance(primitive, dict) and primitive.get("element") == "circle":
                parts.append(
                    f'    <circle cx="{native_number(primitive["cx"])}" '
                    f'cy="{native_number(primitive["cy"])}" r="{native_number(primitive["r"])}"'
                    f'{evidence}/>')
                n_native += 1
                n_fill_native_circles += 1
                n_nodes += 1
            elif isinstance(primitive, dict) and primitive.get("element") == "ellipse":
                rotation = float(primitive.get("rotation_degrees", 0.0))
                transform = (
                    f' transform="rotate({native_number(rotation)} {native_number(primitive["cx"])} '
                    f'{native_number(primitive["cy"])})"'
                    if abs(rotation) > 1e-9 else "")
                parts.append(
                    f'    <ellipse cx="{native_number(primitive["cx"])}" '
                    f'cy="{native_number(primitive["cy"])}" rx="{native_number(primitive["rx"])}" '
                    f'ry="{native_number(primitive["ry"])}"{transform}{evidence}/>')
                n_native += 1
                n_nodes += 1
            else:
                parts.append(
                    f'    <path d="{grad["path_override"]}"{evidence}/>')
                n_paths += 1
                n_nodes += int(
                    ((grad.get("validation") or {}).get("geometry") or {})
                    .get("anchor_count")
                    or len(re.findall(r"[A-Za-z]", grad["path_override"])))
            emitted_gradient_geometry.add(ci)
            parts.append("  </g>")
            final_palette.append((nm, _gradient_palette_hex(grad)))
            continue
        for e in run["items"]:
            if "raw" in e:
                parts.append(f'    <path d="{e["raw"]}"/>')
                n_paths += 1
                n_nodes += len(re.findall(r"[A-Za-z]", e["raw"]))
                continue
            subs = e["subs"]
            # a lone perfect circle becomes a native <circle>
            if len(subs) == 1 and subs[0].get("is_circle"):
                cx, cy, r = subs[0]["is_circle"]
                parts.append(f'    <circle id="avc-fill-circle-{n_fill_native_circles + 1}" '
                             f'cx="{_f(cx)}" cy="{_f(cy)}" '
                             f'r="{_f(r)}"/>')
                n_native += 1
                n_fill_native_circles += 1
                n_nodes += 1
                continue
            # two concentric circles = a ring = one stroked native circle
            ring = _concentric_ring(subs) if grad is None else None
            if ring:
                cx, cy, r, wd = ring
                parts.append(f'    <circle id="avc-fill-ring-{n_fill_native_circles + 1}" '
                             f'cx="{_f(cx)}" cy="{_f(cy)}" '
                             f'r="{_f(r)}" fill="none" '
                             f'stroke="{pal_hex[ci]}" stroke-width="{_f(wd)}"/>')
                n_native += 1
                n_fill_native_circles += 1
                n_nodes += 1
                continue
            d = " ".join(_emit_sub(s) for s in subs)
            parts.append(f'    <path d="{d}"/>')
            n_paths += 1
            n_nodes += sum(len(s["segs"]) + 1 for s in subs)
        parts.append("  </g>")
        # Reserved magenta exists only to route a traced union back to its
        # gradient.  It is never a user-facing palette color.
        final_palette.append((nm, _gradient_palette_hex(grad)
                              if grad is not None else pal_hex[ci]))

    missing_gradient_geometry = set(grad_by_idx) - emitted_gradient_geometry
    if missing_gradient_geometry:
        if residual_scene_counterfactual is not None:
            return _return_residual_counterfactual("candidate_gradient_placeholder_routing_failed")
        raise ValueError(
            "gradient placeholder routing lost an accepted source-space "
            f"object: {sorted(missing_gradient_geometry)}")

    # rebuilt strokes sit on top (line work visually overlays fills)
    if stroke_list:
        parts.append('  <g id="strokes" inkscape:label="strokes" '
                     'inkscape:groupmode="layer" fill="none" '
                     'stroke-linecap="round" stroke-linejoin="round">')
        for i, s in enumerate(stroke_list, 1):
            hx = "#{:02x}{:02x}{:02x}".format(*s.color)
            op_attr = (f' stroke-opacity="{s.opacity:.4f}"'
                       if getattr(s, "opacity", 1.0) < 0.995 else "")
            if getattr(s, "primitive", "") == "circle":
                parts.append(f'    <circle id="stroke-{i}" stroke="{hx}" '
                             f'stroke-width="{_f(s.width)}" '
                             f'cx="{_f(s.cx)}" cy="{_f(s.cy)}" '
                             f'r="{_f(s.radius)}"{op_attr}/>')
                n_native += 1
            elif getattr(s, "primitive", "") == "rect":
                parts.append(f'    <rect id="stroke-{i}" stroke="{hx}" '
                             f'stroke-width="{_f(s.width)}" '
                             f'x="{_f(s.x)}" y="{_f(s.y)}" '
                             f'width="{_f(s.shape_width)}" '
                             f'height="{_f(s.height)}"{op_attr}/>')
            else:
                parts.append(f'    <path id="stroke-{i}" stroke="{hx}" '
                             f'stroke-width="{s.width:.4f}" '
                             f'stroke-linecap="{getattr(s, "linecap", "round")}" d="{s.d}"'
                             f'{op_attr}/>')
            n_nodes += s.n_nodes
        parts.append("  </g>")

    parts += ["</svg>", ""]

    if n_paths == 0 and n_native == 0 and not stroke_list:
        # An empty SVG must never be reported as success: the foreground was
        # probably smaller than the speckle filter, or everything was removed
        # as background.
        raise ValueError(
            "tracing produced no vector paths; the visible foreground may be "
            "too small, or was removed as background (try --background keep)")

    # Ink-loss guard: every palette color with a meaningful pixel share must
    # have produced at least one element; a silently dropped color region
    # (e.g. a thin detail eaten by the speckle filter) is a hard failure.
    vis_guard = vis_fill
    if grad_regions:
        vis_guard = vis_fill.copy()
        for g in grad_regions:
            vis_guard &= ~g["mask"]
    if min(W, H) >= 8 and vis_guard.any():
        emitted = {run["color"] for run in runs if run["items"]}
        vis_labels = lab_all[vis_guard]
        total = vis_labels.size
        pal_f = np.asarray(pal_rgb, dtype=np.float32)
        # blend endpoints: every emitted color, plus the removed background
        # (antialiasing fringe blends toward it even though it is not drawn)
        endpoints = [pal_f[a] for a in emitted]
        # Rebuilt strokes are emitted outside the fill runs.  Their core
        # colors must participate in the accounting or a tiny leftover gray
        # antialias cluster looks like a lost fill even though the stroke
        # already represents it.
        endpoints.extend(np.asarray(s.color, dtype=np.float32)
                         for s in stroke_list)
        if removed:
            endpoints.append(np.array([255.0, 255.0, 255.0], dtype=np.float32))
        lost = []
        for i in range(len(pal_rgb)):
            share = float((vis_labels == i).sum()) / total
            if share < 0.01 or i in emitted:
                continue
            # antialiasing fringe (a color lying between two endpoint colors
            # in RGB space) may legitimately be absorbed — skip it
            is_blend = any(
                _point_segment_dist(pal_f[i], a, b) < 35
                for ai, a in enumerate(endpoints)
                for b in endpoints[ai + 1:])
            # vtracer's stacked mode averages the colors of covered unions;
            # a "dropped" palette entry whose pixels are visually covered by
            # a near-identical emitted layer (max diff <= 75, i.e. within
            # the self-check tolerance neighbourhood) is represented, not
            # lost — the render still matches the source there
            represented = [pal_f[e] for e in emitted]
            represented.extend(np.asarray(s.color, dtype=np.float32)
                               for s in stroke_list)
            is_covered = any(
                float(np.abs(pal_f[i] - col).max()) <= 75.0
                for col in represented)
            if not (is_blend or is_covered):
                lost.append((pal_hex[i], share))
        if lost:
            detail = ", ".join(f"{hx} ({share:.1%})" for hx, share in lost)
            if residual_scene_counterfactual is not None:
                return _return_residual_counterfactual("candidate_source_ink_loss_guard_failed: " + detail)
            raise ValueError(
                f"vectorizer silently dropped visible color regions: {detail}; "
                "the details may be thinner than the speckle filter")

    dst.parent.mkdir(parents=True, exist_ok=True)
    svg_text = "\n".join(parts)
    if any(isinstance(region.get("enclosed_source_component_candidate"), dict)
           for region in grad_regions):
        from gradient_source_components import apply_source_component_candidates
        processed_scene = np.dstack((den, source_alpha))
        ownership_result, ownership_cache = _cached_post_fit_scene(
            gradient_stage_cache, operation="source_ownership_and_bounded_spans",
            svg_text=svg_text, source_path=src, processed_rgba=processed_scene,
            fields=grad_regions, parameters={"budget_percent": curve_error_percent},
            stage_callable=apply_source_component_candidates,
            producer=lambda: apply_source_component_candidates(
                svg_text, grad_regions, src, processed_scene, budget_percent=curve_error_percent))
        refined_svg, updates, ownership_transaction = ownership_result
        ownership_transaction["execution_cache"] = ownership_cache
        ownership_transaction["performance_scope"] = (
            "original_validation_execution_reused" if ownership_cache["status"] == "hit"
            else "this_invocation")
        for region in grad_regions:
            update = updates.get(region.get("candidate_id"))
            if update is None:
                continue
            previous_geometry = region["validation"]["geometry"]
            n_nodes += int(update["geometry"]["anchor_count"]) - int(previous_geometry["anchor_count"])
            region.update(mask=update["mask"], area=update["area"], path_override=update["path"])
            region["validation"]["geometry"] = update["geometry"]
            region["validation"]["selection"] = {
                **region["validation"].get("selection", {}),
                "ownership_source": "original_and_processed_source_supported_enclosed_components",
                "ownership_colour_evidence_used": True}
        svg_text = refined_svg
        palette_audit["gradient_source_ownership_transaction"] = ownership_transaction
    if len(grad_regions) == 1 and not stroke_list:
        from source_gradient_primitive import propose_isolated_gradient_primitive
        try:
            refined_svg, source_geometry, source_certificate = (
                propose_isolated_gradient_primitive(
                    svg_text, src, budget_percent=curve_error_percent))
            grad = grad_regions[0]
            # The source certificate covers the complete candidate, including
            # every separately verified AA residual it removed. One final
            # native element is distinct from its four/eight designer handles.
            n_paths, n_native, n_nodes = 0, 1, 1
            n_fill_native_circles = int(source_certificate["primitive_kind"] == "circle")
            grad["validation"]["geometry"] = source_geometry
            grad["validation"]["selection"] = {
                **grad["validation"].get("selection", {}),
                "colour_used_for_geometry": True,
                "geometry_source": "unmodified_input_coverage50_with_verified_gradient_paint"}
            palette_audit["source_gradient_primitive"] = source_certificate
            svg_text = refined_svg
            pre_notes.append(
                "1 isolated gradient " + source_certificate["primitive_kind"] +
                " rebuilt from unmodified input coverage; "
                "same paint and original-source contour/render checks passed; "
                "not binary-mask-equivalent simplification")
        except (OSError, ValueError) as exc:
            palette_audit["source_gradient_primitive"] = {
                "status": "not_applied", "reason": str(exc)[:240]}
    def _gradient_public_info(g):
        model = dict(g.get("model") or {
            "type": "linear", "x1": g["x1"], "y1": g["y1"],
            "x2": g["x2"], "y2": g["y2"],
        })
        stops = []
        for stop in g["stops"]:
            if isinstance(stop, dict):
                stops.append({
                    "offset": float(stop["offset"]),
                    "color": str(stop.get("color") or
                                 "#{:02x}{:02x}{:02x}".format(
                                     *(int(v) for v in stop["rgb"]))),
                })
            else:
                off, colour = stop
                stops.append({
                    "offset": float(off),
                    "color": "#{:02x}{:02x}{:02x}".format(*colour),
                })
        info = {
            "id": g.get("id", ""),
            "key": "#{:02x}{:02x}{:02x}".format(*g["key"]),
            "model": model,
            "type": model.get("type", "linear"),
            "opacity": g.get("opacity", 1.0),
            "key_distance": g.get("key_distance", 48),
            "validation": g.get("validation", {}),
            "candidate_id": g.get("candidate_id"),
            "candidate_family": g.get("candidate_family"),
            "confidence": g.get("confidence"),
            "stops": stops,
            "viewbox": [W, H],
        }
        if model.get("type") == "linear":
            info.update({name: float(model[name])
                         for name in ("x1", "y1", "x2", "y2")})
        return info

    if residual_scene_counterfactual is not None:
        from gradient_residual_provenance import prove_unchanged_scene_residuals
        try:
            scene_proof = prove_unchanged_scene_residuals(
                residual_scene_counterfactual["svg"], svg_text,
                residual_scene_counterfactual["residual_mask"], src, np.dstack((den, source_alpha)))
            scene_proof["counterfactual_withdrawal"] = residual_scene_counterfactual["fragmentation"]
            palette_audit["gradient_residual_scene_transaction"] = scene_proof
            pre_notes.append("gradient retained after complete-scene comparison proved all residual halos unchanged")
        except (OSError, ValueError, RuntimeError) as exc:
            # The new ownership outline may fail source edge fidelity even
            # when its independently verified paint is useful. Retain the
            # counterfactual's exact geometry and try only that source paint.
            try:
                from gradient_paint_only import apply_paint_only_gradient
                withdrawn = residual_scene_counterfactual["fragmentation"]["candidate_ids"]
                candidates = [g for g in grad_regions if g.get("candidate_id") in withdrawn]
                if len(withdrawn) != 1 or len(candidates) != 1:
                    raise ValueError("paint_only_requires_one_unambiguous_source_field")
                region = copy.deepcopy(candidates[0])
                paint_svg, paint_geometry, paint_proof = apply_paint_only_gradient(
                    residual_scene_counterfactual["svg"], svg_text, region,
                    src, np.dstack((den, source_alpha)), budget_percent=curve_error_percent)
                # Clone every report before mutation: a later failure must
                # return the exact untouched counterfactual SVG and evidence.
                result = copy.deepcopy(residual_scene_counterfactual["stats"])
                region["id"] = paint_proof["gradient_id"]
                region["validation"]["geometry"] = paint_geometry
                region["validation"]["selection"] = {
                    **region["validation"].get("selection", {}),
                    "colour_used_for_geometry": False,
                    "geometry_source": "unchanged_existing_svg_geometry"}
                result.gradient_info.append(_gradient_public_info(region))
                result.n_gradients += 1
                audit = result.palette_audit["gradient_reconstruction"]
                restored = [d for d in audit["decisions"] if d.get("candidate_id") == region["candidate_id"]]
                if len(restored) != 1 or restored[0].get("status") != "withdrawn":
                    raise ValueError("paint_only_withdrawal_evidence_identity_mismatch")
                restored[0].update(status="selected", reconstruction_route="paint_only_existing_paths")
                partial_count = sum(bool((g.get("validation") or {}).get("geometry", {}).get(
                    "source_paint_only", {}).get("partial_selection")) for g in result.gradient_info)
                audit["summary"]["objects_selected"] = len(result.gradient_info) - partial_count
                audit["summary"]["partial_paint_fields"] = partial_count
                audit["summary"]["paint_only_existing_geometry_objects"] = 1
                audit["summary"]["geometry_replacements_withdrawn"] = len(withdrawn)
                result.palette_audit["gradient_paint_only_transaction"] = paint_proof
                result.palette_audit["gradient_residual_scene_transaction"] = {
                    "status": "new_outline_rejected_existing_geometry_paint_restored",
                    "outline_rejection": str(exc)[:240],
                    "residual_handling": "counterfactual_geometry_and_paint_order_unchanged"}
                result.palette_audit["gradient_contour_span_scene_searches"] = _span_scene_search_summary()
                result.geometry_notes.append(
                    "1 source gradient restored on existing paths; geometry, path count, "
                    "paint order and complete alpha unchanged; no geometry economy claimed")
                dst.write_text(paint_svg, encoding="utf8")
                return result
            except (OSError, ValueError, RuntimeError) as paint_exc:
                return _return_residual_counterfactual(str(exc)+"; paint-only: "+str(paint_exc))
    partial_paint_details = []
    if original_gradient_stage is not None and original_gradient_stage.get("paint_ready_alternatives"):
        from gradient_paint_only import apply_pending_paint_alternatives
        _checkpoint("partial_source_paint", "核對既有路徑的局部漸層；保留原輪廓與堆疊")
        native_paint_reference, _, _ = _prepare_image(
            src, max_size=0, background=background,
            white_threshold=white_threshold, alpha_threshold=12)
        processed_scene = np.dstack((den, source_alpha))
        native_paint_rgba = np.asarray(native_paint_reference.convert("RGBA"))
        pending_paint = original_gradient_stage["paint_ready_alternatives"]
        partial_result, partial_cache = _cached_post_fit_scene(
            gradient_stage_cache, operation="pending_partial_paint",
            svg_text=svg_text, source_path=src, processed_rgba=processed_scene,
            native_reference_rgba=native_paint_rgba, fields=pending_paint,
            parameters={"budget_percent": curve_error_percent, "maximum_fields": 1,
                        "background": background, "white_threshold": white_threshold,
                        "native_reference_alpha_threshold": 12},
            stage_callable=apply_pending_paint_alternatives,
            producer=lambda: apply_pending_paint_alternatives(
                svg_text, pending_paint, src, processed_scene,
                budget_percent=curve_error_percent, maximum_fields=1,
                native_reference_rgba=native_paint_rgba))
        svg_text, partial_paint_details, partial_decisions, partial_proofs = partial_result
        palette_audit["gradient_partial_paint_execution_cache"] = partial_cache
        audit = palette_audit["gradient_reconstruction"]
        audit.setdefault("decisions", []).extend(partial_decisions)
        audit.setdefault("summary", {})["partial_paint_fields"] = len(partial_paint_details)
        audit["summary"]["objects_selected"] = len(grad_regions)
        palette_audit["gradient_partial_paint_transactions"] = partial_proofs
        if partial_paint_details:
            pre_notes.append(
                f"{len(partial_paint_details)} source field(s) partially restored on existing paths; "
                "original geometry, alpha and paint order retained; manual review required; "
                "no complete-object or geometry-economy claim")
    palette_audit["gradient_contour_span_scene_searches"] = _span_scene_search_summary()
    geometry_notes = _finalize_circle_geometry_note(
        geometry_notes, n_fill_native_circles)
    dst.write_text(svg_text, encoding="utf-8")
    _checkpoint("svg_written", "候選 SVG 已寫入，等待渲染品質驗證")

    if grad_regions:
        geometry_notes = geometry_notes + [
            f"{len(grad_regions)} continuous source colour field(s) rebuilt "
            "as single native gradient objects with error-budgeted contours"]
    if pre_notes:
        geometry_notes = pre_notes + geometry_notes

    if np.any(hole_mask):
        validation_hole_shape = tuple(int(value) for value in hole_mask.shape)
        validation_hole_bits = np.packbits(
            hole_mask.reshape(-1), bitorder="little").tobytes()
    else:
        validation_hole_shape = ()
        validation_hole_bits = b""

    return CleanBaseStats(
        width=orig_w, height=orig_h, colors=len(final_palette),
        palette=final_palette, removed_background=removed,
        geometry_notes=geometry_notes, n_paths=n_paths,
        n_native=n_native, n_strokes=len(stroke_list), n_nodes=n_nodes,
        stroke_info=[{"color": "#{:02x}{:02x}{:02x}".format(*s.color),
                      "width": s.width, "closed": s.closed,
                      "nodes": s.n_nodes,
                      "opacity": getattr(s, "opacity", 1.0),
                      "primitive": getattr(s, "primitive", ""),
                      "linecap": getattr(s, "linecap", "round"),
                      "source_fit": getattr(s, "source_fit", {})}
                     for s in stroke_list],
        viewbox=[W, H],
        n_gradients=len(grad_regions) + len(partial_paint_details),
        gradient_info=[_gradient_public_info(g) for g in grad_regions] + partial_paint_details,
        palette_audit=palette_audit,
        _validation_hole_shape=validation_hole_shape,
        _validation_hole_bits=validation_hole_bits,
    )
