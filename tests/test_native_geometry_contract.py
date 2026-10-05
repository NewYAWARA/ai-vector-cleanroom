import copy
import math
import unittest

import numpy as np

from curve_refit import fit_curve
from native_geometry_contract import whole_object_native_primitive, native_geometry_matches


class NativeGeometryContractTests(unittest.TestCase):
    def geometry(self, rx=20, ry=20, rotation=0):
        angles = np.linspace(0, 2 * math.pi, 96, endpoint=False)
        phi = math.radians(rotation)
        x, y = rx * np.cos(angles), ry * np.sin(angles)
        points = np.column_stack((50 + x * math.cos(phi) - y * math.sin(phi),
                                  40 + x * math.sin(phi) + y * math.cos(phi)))
        fitted = fit_curve(points, closed=True, tolerance=0.05)
        return {"primitive_first": True,
                "native_primitives": [fitted["native_primitive"]],
                "path": fitted["path"],
                "topology": {"components": 1, "holes": 0,
                             "expected_loops": 1, "actual_loops": 1,
                             "topology_preserved": True}}

    def test_real_fitter_circle_and_rotated_ellipse_match_whole_path(self):
        for args in [(20, 20, 0), (30, 15, 37)]:
            with self.subTest(args=args):
                geometry = self.geometry(*args)
                primitive = whole_object_native_primitive(geometry)
                self.assertEqual(primitive, geometry["native_primitives"][0])
                self.assertTrue(native_geometry_matches(primitive, primitive))

    def test_partial_native_hole_never_replaces_compound(self):
        geometry = self.geometry()
        geometry["path"] = "M0 0 L100 0 L100 100 L0 100 Z " + geometry["path"]
        # Even forged/incorrect upstream topology cannot turn two loops into one.
        self.assertIsNone(whole_object_native_primitive(geometry))
        geometry["topology"].update(holes=1, expected_loops=2, actual_loops=2)
        self.assertIsNone(whole_object_native_primitive(geometry))

    def test_metadata_alone_is_insufficient_and_native_numbers_are_checked(self):
        base = self.geometry()
        for key, value in [("cx", 52), ("cy", 42), ("r", 21)]:
            geometry = copy.deepcopy(base)
            geometry["native_primitives"][0][key] = value
            self.assertIsNone(whole_object_native_primitive(geometry))
            self.assertFalse(native_geometry_matches(
                base["native_primitives"][0], geometry["native_primitives"][0]))
        for path in ["M0 0L100 0L100 100Z", base["path"].replace("Z", ""),
                     base["path"].replace("A", "a"), base["path"] + " garbage"]:
            self.assertIsNone(whole_object_native_primitive(base, path))


if __name__ == "__main__":
    unittest.main()
