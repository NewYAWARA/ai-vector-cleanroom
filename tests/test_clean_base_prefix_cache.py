# -*- coding: utf-8 -*-
"""Exact process-local pre-gradient state cache contracts."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import clean_base
from execution_control import ConversionInterrupted


def _stage_callable_a():
    return None


def _stage_callable_b():
    return None


def _minimal_state(flat_png_bytes=b"exact-flat-png-bytes"):
    return {
        "orig_w": 1,
        "orig_h": 1,
        "removed": False,
        "source_alpha": np.asarray([[255]], dtype=np.uint8),
        "visible": np.asarray([[True]], dtype=bool),
        "H": 1,
        "W": 1,
        "pre_notes": [],
        "den": np.zeros((1, 1, 3), dtype=np.float32),
        "palette": np.asarray([[1, 2, 3]], dtype=np.uint8),
        "lab_all": np.asarray([[0]], dtype=np.int32),
        "palette_opacity": [1.0],
        "stroke_list": [],
        "bg_col": (255.0, 255.0, 255.0),
        "hole_mask": np.asarray([[False]], dtype=bool),
        "vis_fill": np.asarray([[True]], dtype=bool),
        "palette_audit": {"nested": {"values": [1, 2]}},
        "flat": np.asarray([[[1, 2, 3]]], dtype=np.uint8),
        "flat_png_bytes": flat_png_bytes,
    }


class _StopBeforeGradientStage:
    def checkpoint(self, stage):
        if stage == "candidate_build:gradient_reconstruction":
            raise ConversionInterrupted("targeted cache integration stop")


class CleanBasePrefixCacheTests(unittest.TestCase):
    def test_key_covers_source_parameters_callable_and_schema(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source.bin"
            source.write_bytes(b"source-bytes-a")
            parameters = {
                "forced_colors": 0,
                "white_threshold": 220,
                "background": "auto",
                "max_size": 0,
                "strokes": "on",
            }
            base = clean_base._pre_gradient_state_cache_key(
                source, parameters, _stage_callable_a, "stage/v1")

            for name, changed in (
                ("forced_colors", 7),
                ("white_threshold", 219),
                ("background", "keep"),
                ("max_size", 1024),
                ("strokes", "off"),
            ):
                with self.subTest(parameter=name):
                    variant = dict(parameters)
                    variant[name] = changed
                    self.assertNotEqual(
                        base,
                        clean_base._pre_gradient_state_cache_key(
                            source, variant, _stage_callable_a, "stage/v1"))

            self.assertNotEqual(
                base,
                clean_base._pre_gradient_state_cache_key(
                    source, parameters, _stage_callable_b, "stage/v1"))
            self.assertNotEqual(
                base,
                clean_base._pre_gradient_state_cache_key(
                    source, parameters, _stage_callable_a, "stage/v2"))
            source.write_bytes(b"source-bytes-b")
            self.assertNotEqual(
                base,
                clean_base._pre_gradient_state_cache_key(
                    source, parameters, _stage_callable_a, "stage/v1"))

    def test_store_and_hits_are_deeply_isolated(self):
        cache = clean_base.new_pre_gradient_state_cache()
        calls = []

        def producer():
            calls.append("called")
            return _minimal_state()

        first, first_evidence = (
            clean_base._pre_gradient_state_cache_get_or_compute(
                cache, "PREFIX", producer))
        self.assertEqual(first_evidence["status"], "miss")
        first["palette_audit"]["nested"]["values"].append(99)
        first["flat"][0, 0, 0] = 255

        second, second_evidence = (
            clean_base._pre_gradient_state_cache_get_or_compute(
                cache, "PREFIX", producer))
        self.assertEqual(second_evidence["status"], "hit")
        self.assertEqual(
            second["palette_audit"]["nested"]["values"], [1, 2])
        self.assertEqual(int(second["flat"][0, 0, 0]), 1)

        second["palette_audit"]["nested"]["values"].append(77)
        third, third_evidence = (
            clean_base._pre_gradient_state_cache_get_or_compute(
                cache, "PREFIX", producer))
        self.assertEqual(third_evidence["status"], "hit")
        self.assertEqual(
            third["palette_audit"]["nested"]["values"], [1, 2])
        self.assertEqual(third["flat_png_bytes"], b"exact-flat-png-bytes")
        self.assertEqual(calls, ["called"])

        audit = clean_base.pre_gradient_state_cache_audit(cache)
        self.assertEqual(audit["requests"], 3)
        self.assertEqual(audit["misses"], 1)
        self.assertEqual(audit["hits"], 2)
        self.assertEqual(audit["entry_count"], 1)

    def test_failed_and_interrupted_producers_are_never_cached(self):
        for exception in (
                ValueError("failed prefix"),
                ConversionInterrupted("interrupted prefix")):
            with self.subTest(exception=type(exception).__name__):
                cache = clean_base.new_pre_gradient_state_cache()

                def producer():
                    raise exception

                with self.assertRaises(type(exception)):
                    clean_base._pre_gradient_state_cache_get_or_compute(
                        cache, "FAILED", producer)
                audit = clean_base.pre_gradient_state_cache_audit(cache)
                self.assertEqual(audit["requests"], 1)
                self.assertEqual(audit["misses"], 1)
                self.assertEqual(audit["errors"], 1)
                self.assertEqual(audit["entry_count"], 0)

    def test_build_reuses_exact_prefix_across_gradient_geometry_and_curve(self):
        cache = clean_base.new_pre_gradient_state_cache()
        calls = []

        def producer(*_args, **_kwargs):
            calls.append(_kwargs["strokes"])
            return _minimal_state()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.png"
            destination = root / "candidate.svg"
            flat = root / "flat.png"
            source.write_bytes(b"exact-source-identity")
            control = _StopBeforeGradientStage()
            with patch.object(
                    clean_base, "_produce_pre_gradient_state",
                    side_effect=producer):
                attempts = (
                    {"strokes": "on", "gradients": "on",
                     "geometry": "normal", "curve_error_percent": 0.25},
                    {"strokes": "on", "gradients": "off",
                     "geometry": "normal", "curve_error_percent": 0.50},
                    {"strokes": "on", "gradients": "on",
                     "geometry": "off", "curve_error_percent": 1.00},
                    {"strokes": "off", "gradients": "off",
                     "geometry": "normal", "curve_error_percent": 0.25},
                )
                for options in attempts:
                    with self.assertRaises(ConversionInterrupted):
                        clean_base.build_clean_base(
                            source, destination, background="auto",
                            forced_colors=0, white_threshold=220, max_size=0,
                            flat_out=flat,
                            pre_gradient_state_cache=cache,
                            control=control, **options)
                    self.assertEqual(flat.read_bytes(),
                                     b"exact-flat-png-bytes")

        # gradients, geometry and curve budget never split the L2 key;
        # strokes on/off must remain separate exact states.
        self.assertEqual(calls, ["on", "off"])
        audit = clean_base.pre_gradient_state_cache_audit(cache)
        self.assertEqual(audit["requests"], 4)
        self.assertEqual(audit["misses"], 2)
        self.assertEqual(audit["hits"], 2)
        self.assertEqual(audit["entry_count"], 2)

    def test_full_build_hit_is_byte_identical_to_uncached_and_miss(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.png"
            rgba = np.zeros((48, 64, 4), dtype=np.uint8)
            rgba[6:42, 8:56] = (31, 93, 171, 255)
            rgba[16:32, 20:44] = (244, 181, 38, 255)
            Image.fromarray(rgba, "RGBA").save(source)

            cache = clean_base.new_pre_gradient_state_cache()
            results = []
            for name, selected_cache in (
                    ("uncached", None), ("miss", cache), ("hit", cache)):
                svg = root / f"{name}.svg"
                flat = root / f"{name}.png"
                stats = clean_base.build_clean_base(
                    source, svg, background="keep", geometry="off",
                    strokes="off", gradients="off", flat_out=flat,
                    pre_gradient_state_cache=selected_cache)
                stats_without_audit = copy.deepcopy(stats.__dict__)
                stats_without_audit.pop("palette_audit", None)
                results.append((svg.read_bytes(), flat.read_bytes(),
                                stats_without_audit))

            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0], results[2])
            audit = clean_base.pre_gradient_state_cache_audit(cache)
            self.assertEqual(audit["requests"], 2)
            self.assertEqual(audit["misses"], 1)
            self.assertEqual(audit["hits"], 1)


if __name__ == "__main__":
    unittest.main()
