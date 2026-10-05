# -*- coding: utf-8 -*-
"""In-memory regressions for the self-check native-gradient painter."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from vector_cleanroom import _paint_gradients


KEY = (241, 3, 247)
BACKGROUND = (211, 223, 229)


def _paint_in_memory(pixels, gradient_info):
    """Exercise the real file-oriented function without requiring disk I/O."""
    source = Image.fromarray(np.asarray(pixels, dtype=np.uint8), "RGB")
    saved = []

    def fake_open(_path):
        return source.copy()

    def fake_save(image, _path, *args, **kwargs):
        saved.append(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())

    with patch.object(Image, "open", side_effect=fake_open), patch.object(
        Image.Image, "save", new=fake_save
    ):
        _paint_gradients(Path("in-memory-preview.png"), gradient_info)
    if len(saved) != 1:
        raise AssertionError("preview painter did not save exactly one image")
    return saved[0]


class GradientPreviewPainterTests(unittest.TestCase):
    def test_json_roundtripped_arbitrary_angle_five_stop_linear(self):
        pixels = np.full((11, 11, 3), BACKGROUND, dtype=np.uint8)
        for coordinate in range(1, 10):
            pixels[coordinate, coordinate] = KEY
        info = [{
            "key": "#f103f7",
            "key_distance": 140,
            "viewbox": [11, 11],
            "model": {
                "type": "linear",
                "svg_type": "linearGradient",
                "gradient_units": "userSpaceOnUse",
                "x1": 1.0,
                "y1": 1.0,
                "x2": 9.0,
                "y2": 9.0,
                "direction": [0.70710678, 0.70710678],
                "axis_span_px": 11.31371,
                "stop_count": 5,
                "bounded_maximum_stops": 5,
            },
            "stops": [
                {"offset": 0.0, "color": "#000000", "rgb": [0, 0, 0]},
                {"offset": 0.25, "color": "#ff0000", "rgb": [255, 0, 0]},
                {"offset": 0.5, "color": "#00ff00", "rgb": [0, 255, 0]},
                {"offset": 0.75, "color": "#0000ff", "rgb": [0, 0, 255]},
                {"offset": 1.0, "color": "#ffffff", "rgb": [255, 255, 255]},
            ],
        }]
        # Force the exact strict-JSON shape emitted by the paint engine.
        info = json.loads(json.dumps(info, allow_nan=False, sort_keys=True))
        out = _paint_in_memory(pixels, info)

        np.testing.assert_array_equal(out[1, 1], [0, 0, 0])
        np.testing.assert_array_equal(out[3, 3], [255, 0, 0])
        np.testing.assert_array_equal(out[5, 5], [0, 255, 0])
        np.testing.assert_array_equal(out[7, 7], [0, 0, 255])
        np.testing.assert_array_equal(out[9, 9], [255, 255, 255])
        # t=.375: exact stop-offset interpolation, not endpoint-only paint.
        np.testing.assert_array_equal(out[4, 4], [127, 127, 0])
        np.testing.assert_array_equal(out[1, 2], BACKGROUND)
        np.testing.assert_array_equal(out[0, 0], BACKGROUND)

    def test_rotated_elliptical_radial_uses_existing_fill_mask_only(self):
        pixels = np.full((13, 13, 3), BACKGROUND, dtype=np.uint8)
        owned = [(6, 6), (6, 7), (6, 8), (8, 6), (10, 6)]
        for yy, xx in owned:
            pixels[yy, xx] = KEY
        info = [{
            "key": "#f103f7",
            "key_distance": 140,
            "viewbox": [13, 13],
            "model": {
                "type": "radial",
                "svg_type": "radialGradient",
                "gradient_units": "userSpaceOnUse",
                "center": [6.0, 6.0],
                "radius_x": 4.0,
                "radius_y": 2.0,
                "rotation_degrees": 90.0,
                "geometry_fit_rmse": 0.0,
                "polarity": "centre_to_edge",
                "stop_count": 3,
                "bounded_maximum_stops": 5,
            },
            "stops": [
                {"offset": 0.0, "color": "#000000", "rgb": [0, 0, 0]},
                {"offset": 0.5, "color": "#ff0000", "rgb": [255, 0, 0]},
                {"offset": 1.0, "color": "#ffff00", "rgb": [255, 255, 0]},
            ],
        }]
        info = json.loads(json.dumps(info, allow_nan=False, sort_keys=True))
        first = _paint_in_memory(pixels, info)
        second = _paint_in_memory(pixels, info)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[6, 6], [0, 0, 0])
        # A 90-degree ellipse: horizontal distance 1 uses radius_y=2.
        np.testing.assert_array_equal(first[6, 7], [255, 0, 0])
        np.testing.assert_array_equal(first[6, 8], [255, 255, 0])
        # Vertical distance 2 uses radius_x=4; distance 4 reaches the edge.
        np.testing.assert_array_equal(first[8, 6], [255, 0, 0])
        np.testing.assert_array_equal(first[10, 6], [255, 255, 0])
        # Geometric ellipse coverage never authorises new painted pixels.
        np.testing.assert_array_equal(first[6, 5], BACKGROUND)
        np.testing.assert_array_equal(first[7, 6], BACKGROUND)
        np.testing.assert_array_equal(first[0, 0], BACKGROUND)

    def test_legacy_flat_two_stop_linear_record_is_unchanged(self):
        pixels = np.full((3, 7, 3), BACKGROUND, dtype=np.uint8)
        pixels[1, 1:6] = KEY
        legacy = [{
            "key": "#f103f7",
            "key_distance": 140,
            "viewbox": [7, 3],
            "x1": 1.0,
            "y1": 1.0,
            "x2": 5.0,
            "y2": 1.0,
            "stops": [
                {"offset": 0.0, "color": "#204060"},
                {"offset": 1.0, "color": "#e0c0a0"},
            ],
        }]
        out = _paint_in_memory(pixels, legacy)

        np.testing.assert_array_equal(out[1, 1], [32, 64, 96])
        np.testing.assert_array_equal(out[1, 3], [128, 128, 128])
        np.testing.assert_array_equal(out[1, 5], [224, 192, 160])
        np.testing.assert_array_equal(out[0],
                                      np.full((7, 3), BACKGROUND, dtype=np.uint8))
        np.testing.assert_array_equal(out[2],
                                      np.full((7, 3), BACKGROUND, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
