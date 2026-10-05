"""Actual native rendering for a narrow source-driven gradient-circle route."""
import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from source_gradient_primitive import (SCHEMA, gradient_paint_sha256,
    propose_isolated_gradient_circle, propose_isolated_gradient_roundrect,
    source_circle_certificate_valid, source_primitive_certificate_valid, final_source_primitive_matches)
from svg_renderer import render_svg_reference


def svg(shape, *, colours=("#f2d27a", "#b43f32"), extra="", view="0 0 128 128"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="{view}">'
            f'<defs><radialGradient id="paint" gradientUnits="userSpaceOnUse" cx="58" cy="54" r="55">'
            f'<stop offset="0" stop-color="{colours[0]}"/><stop offset="1" stop-color="{colours[1]}"/>'
            '</radialGradient></defs><g fill="url(#paint)">' + shape + extra + '</g></svg>')


def baseline():
    points = [(63.5 + 42.4 * math.cos(i * math.tau / 32),
               63.5 + 42.4 * math.sin(i * math.tau / 32)) for i in range(32)]
    path = "M" + " L".join(f"{x:.6f} {y:.6f}" for x, y in points) + " Z"
    return f'<path id="object-one" data-avc-gradient-object="proposal-one" data-avc-designer-anchors="32" d="{path}"/>'


class SourceGradientPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def source(self, shape='<circle cx="64" cy="64" r="42"/>', *, colours=("#f2d27a", "#b43f32"), background="#ffffff"):
        path = self.folder / "truth.svg"
        path.write_text(svg(shape, colours=colours), encoding="utf8")
        output = self.folder / "original.png"
        render_svg_reference(path, output, 128, background=background)
        return output

    def test_real_source_circle_reduces_handles_and_improves_same_paint(self):
        source = self.source()
        before = svg(baseline())
        after, geometry, cert = propose_isolated_gradient_circle(before, source)
        self.assertTrue(source_circle_certificate_valid(geometry))
        self.assertEqual(geometry["designer_anchor_count"], 4)
        self.assertEqual(geometry["anchor_count"], 2)  # two arcs in the numeric native proof
        self.assertEqual(cert["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(cert["schema"], SCHEMA)
        self.assertIn("not_binary_mask_equivalence", cert["scope"])
        before_root, after_root = ET.fromstring(before), ET.fromstring(after)
        self.assertEqual(gradient_paint_sha256(before_root, "paint"), gradient_paint_sha256(after_root, "paint"))
        element = next(n for n in after_root.iter() if n.get("id") == "object-one")
        self.assertTrue(element.tag.endswith("circle"))
        self.assertEqual(element.get("data-avc-gradient-object"), "proposal-one")
        self.assertAlmostEqual(float(element.get("cx")), 64, delta=.06)
        for key in ("global_rgb_mae", "object_rgb_mae", "edge_rgb_mae"):
            self.assertLess(cert["measurements"]["after"][key], cert["measurements"]["before"][key])
        changed = copy.deepcopy(geometry)
        changed["source_reconstruction"]["measurements"]["after"]["edge_rgb_mae"] = 999
        self.assertFalse(source_circle_certificate_valid(changed))

    def test_background_alpha_and_near_white_denominator_fail_closed(self):
        for kwargs in ({"background": "#dddddd"}, {"background": None},
                       {"colours": ("#fafafa", "#fefefe")}):
            with self.subTest(kwargs=kwargs):
                source = self.source(**kwargs)
                with self.assertRaises(ValueError):
                    propose_isolated_gradient_circle(svg(baseline(), colours=kwargs.get("colours", ("#f2d27a", "#b43f32"))), source)

    def test_holes_multiple_objects_semicircle_and_cut_corner_are_not_circles(self):
        shapes = [
            '<path fill-rule="evenodd" d="M22 64 A42 42 0 1 1 106 64 A42 42 0 1 1 22 64 Z M58 64 A6 6 0 1 0 70 64 A6 6 0 1 0 58 64 Z"/>',
            '<circle cx="42" cy="64" r="20"/><circle cx="91" cy="64" r="15"/>',
            '<path d="M22 64 A42 42 0 0 1 106 64 Z"/>',
            '<path d="M22 42 L42 22 H86 L106 42 V106 H22 Z"/>',
            '<path d="M22 22 H106 V106 H22 Z"/>',
        ]
        for shape in shapes:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    propose_isolated_gradient_circle(svg(baseline()), self.source(shape))

    def test_multi_drawable_transform_mask_and_compound_context_reject(self):
        source = self.source()
        contexts = [svg(baseline(), extra='<circle cx="30" cy="30" r="2"/>'),
                    svg(baseline().replace('id="object-one"', 'id="object-one" transform="translate(1 0)"')),
                    svg(baseline().replace('id="object-one"', 'id="object-one" mask="url(#m)"')),
                    svg(baseline().replace(' Z"', ' Z M50 50L55 50L55 55Z"'))]
        for document in contexts:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    propose_isolated_gradient_circle(document, source)

    def test_exact_source_parent_is_not_reported_as_an_improvement(self):
        source = self.source()
        exact = svg('<circle id="object-one" data-avc-gradient-object="proposal-one" data-avc-designer-anchors="4" cx="64" cy="64" r="42"/>')
        with self.assertRaisesRegex(ValueError, "fidelity_not_improved"):
            propose_isolated_gradient_circle(exact, source)

    def test_original_pixels_map_to_offset_scaled_viewbox(self):
        def document(shape):
            return svg(shape, view="10 20 256 256").replace(
                'cx="58" cy="54" r="55"', 'cx="126" cy="128" r="110"')
        truth = self.folder / "offset-truth.svg"
        truth.write_text(document('<circle cx="138" cy="148" r="84"/>'), encoding="utf8")
        source = self.folder / "offset-original.png"
        render_svg_reference(truth, source, 128)
        parent = document('<circle id="object-one" data-avc-gradient-object="proposal-one" '
                          'data-avc-designer-anchors="4" cx="137" cy="147" r="84.8"/>')
        _, geometry, cert = propose_isolated_gradient_circle(parent, source)
        native = geometry["native_primitives"][0]
        self.assertAlmostEqual(native["cx"], 138, delta=.12)
        self.assertAlmostEqual(native["cy"], 148, delta=.12)
        self.assertAlmostEqual(native["r"], 84, delta=.15)
        self.assertEqual(cert["source_size"], [128, 128])
        self.assertEqual(cert["view_box"], [10, 20, 256, 256])

    def test_final_native_identity_geometry_and_paint_are_bound(self):
        from vector_cleanroom import (_final_gradient_report_details,
            _gradient_geometry_snapshot, _gradient_geometry_digest)
        source = self.source()
        after, geometry, _ = propose_isolated_gradient_circle(svg(baseline()), source)
        (self.folder / "source_original.png").write_bytes(source.read_bytes())
        path = self.folder / "candidate.svg"
        path.write_text(after, encoding="utf8")
        def certify():
            snapshot = _gradient_geometry_snapshot(path)
            report = {"gradient_geometry_guard": {"before_geometry_sha256": _gradient_geometry_digest(snapshot)}}
            return _final_gradient_report_details(path, [{"id": "paint", "validation": {"geometry": geometry}}], report)
        details, _ = certify()
        self.assertIn("coverage50", details[0]["validation"]["geometry"]["evidence_scope"])
        path.write_text(after.replace('#f2d27a', '#f2d27b'), encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "paint changed"):
            certify()


class SourceGradientRoundrectTests(unittest.TestCase):
    setUp = SourceGradientPrimitiveTests.setUp
    source = SourceGradientPrimitiveTests.source
    def parent(self, extra=""):
        return svg('<path id="object-one" data-avc-gradient-object="proposal-one" data-avc-designer-anchors="8" '
                   'd="M23 27.5863 C20.4367 29.7682 17.5003 31.5927 17.5 35 L17.5863 94 '
                   'C19.8784 96.5086 21.5487 99.4998 25 99.5 L102 99.5 '
                   'C106.0107 99.4997 109.4996 96.013 109.5 92 L109.5 35 '
                   'C109.4996 30.987 106.0107 27.5003 102 27.5 L23 27.5863 Z"/>', extra=extra)

    def roundrect_source(self, extra="", shape=None):
        return self.source((shape or '<rect x="18" y="28" width="92" height="72" rx="9"/>') + extra)

    def test_source_roundrect_is_eight_handles_same_paint_and_bound_to_complete_scene(self):
        source = self.roundrect_source()
        after, geometry, cert = propose_isolated_gradient_roundrect(self.parent(), source)
        self.assertTrue(source_primitive_certificate_valid(geometry))
        self.assertEqual(geometry["designer_anchor_count"], 8)
        self.assertEqual(geometry["native_primitives"][0]["element"], "rect")
        self.assertAlmostEqual(geometry["native_primitives"][0]["rx"], 9, delta=.15)
        self.assertTrue(final_source_primitive_matches(ET.fromstring(after), geometry, source))
        self.assertEqual(cert["removed_drawables"], [])
        for key, value in (("viewBox", "0 0 256 256"), ("display", "none"),
                           ("visibility", "hidden"), ("width", "256"), ("shape-rendering", "crispEdges")):
            changed = ET.fromstring(after)
            changed.set(key, value)
            self.assertFalse(final_source_primitive_matches(changed, geometry, source), key)
        changed = ET.fromstring(after)
        ET.SubElement(changed, "{http://www.w3.org/2000/svg}circle", {"cx": "8", "cy": "8", "r": "1"})
        self.assertFalse(final_source_primitive_matches(changed, geometry, source))
        changed_geometry = copy.deepcopy(geometry)
        changed_geometry["source_reconstruction"]["source_rgba_sha256"] = "1"*64
        self.assertFalse(final_source_primitive_matches(ET.fromstring(after), changed_geometry, source))
        changed_geometry = copy.deepcopy(geometry)
        changed_geometry["native_whole_object_path"] = "M0 0 H1 V1 Z"
        self.assertFalse(source_primitive_certificate_valid(changed_geometry))

    def test_roundrect_does_not_erase_glyph_dot_hole_or_nonuniform_interior(self):
        extras = ['<circle cx="8" cy="8" r="2" fill="#000000"/>',
                  '<path d="M45 55H47V64H45Z" fill="#ffffff"/>',
                  '<circle cx="58" cy="60" r="3" fill="#eeeeee"/>',
                  '<circle cx="80" cy="60" r="4" fill="#187040"/>',
                  '<circle cx="16" cy="58" r="1.3" fill="#000000"/>']
        for extra in extras:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                propose_isolated_gradient_roundrect(self.parent(), self.roundrect_source(extra))
        for shape in ('<path d="M18 40L30 28H98L110 40V88L98 100H30L18 88Z"/>',
                      '<circle cx="64" cy="64" r="38"/>',
                      '<rect x="18" y="28" width="92" height="72"/>'):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                propose_isolated_gradient_roundrect(self.parent(), self.roundrect_source(shape=shape))

    def test_residual_intent_cannot_hide_inside_global_average_improvement(self):
        source = self.roundrect_source()
        for extra in ('<path id="intent" fill="#000000" d="M8 8H9V9H8Z"/>',
                      '<path id="intent" fill="#000000" d="M50 50H51V51H50Z"/>',
                      '<path id="intent" fill="#000000" d="M18 28H50V35H18Z"/>'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                propose_isolated_gradient_roundrect(self.parent(extra), source)

    def test_tiny_aa_removal_is_per_pixel_and_final_snapshot_is_bound(self):
        from vector_cleanroom import (_gradient_geometry_snapshot, _gradient_geometry_digest,
                                      _final_gradient_report_details)
        source = self.roundrect_source()
        extra = '<path id="aa" fill="#777777" d="M21 29H22V30H21Z M21 30V31H20V30Z M20 31V32H19V31Z"/>'
        after, geometry, cert = propose_isolated_gradient_roundrect(self.parent(extra), source)
        self.assertEqual(len(cert["removed_drawables"]), 1)
        self.assertEqual(cert["removed_drawables"][0]["id"], "aa")
        self.assertLessEqual(cert["removed_drawables"][0]["maximum_support_pixel_rgb_mae_regression"], 0)
        self.assertTrue(source_primitive_certificate_valid(geometry))
        path = self.folder / "final.svg"
        path.write_text(after, encoding="utf8")
        # Match production: a re-encoded PNG is checked by exact RGBA bytes.
        from PIL import Image
        with Image.open(source) as image:
            image.convert("RGBA").save(self.folder / "source_original.png")
        snapshot = _gradient_geometry_snapshot(path)
        self.assertEqual(snapshot[0]["anchor_count"], 1)
        self.assertEqual(snapshot[0]["designer_anchor_count"], "8")
        report = {"gradient_geometry_guard": {"before_geometry_sha256": _gradient_geometry_digest(snapshot)}}
        details = [{"id": "paint", "validation": {"geometry": geometry}}]
        final, _ = _final_gradient_report_details(path, details, report)
        self.assertEqual(final[0]["validation"]["geometry"]["final_svg_consistency"]["final_element"], "rect")
        changed = ET.fromstring(after)
        ET.SubElement(changed, "{http://www.w3.org/2000/svg}path", {"id": "aa", "d": "M21 29H22V30H21Z"})
        path.write_text(ET.tostring(changed, encoding="unicode"), encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "source-primitive"):
            _final_gradient_report_details(path, details, report)
        modified = copy.deepcopy(geometry)
        modified["source_reconstruction"]["removed_drawables"][0]["maximum_support_pixel_rgb_mae_regression"] = .1
        self.assertFalse(source_primitive_certificate_valid(modified))


if __name__ == "__main__":
    unittest.main()
