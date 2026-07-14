# -*- coding: utf-8 -*-
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

from paint_roles import (
    MANIFEST_SCHEMA,
    annotate_svg_with_paint_roles,
    apply_role_recolor,
    build_paint_role_manifest,
    hex_to_oklch,
    oklch_to_hex,
    structure_signature,
    write_paint_role_manifest,
)


SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="120"
     viewBox="0 0 200 120">
  <defs>
    <linearGradient id="r1" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#287a3c"/>
      <stop offset="1" style="stop-color:#b5f23a;stop-opacity:1"/>
    </linearGradient>
  </defs>
  <g id="a" fill="#62e53a">
    <path id="p1" d="M0 0L10 0L10 10Z"/>
    <path id="p2" d="M12 0L22 0L22 10Z"/>
  </g>
  <path id="b" fill="url(#r1)" d="M30 0L40 0L40 10Z"/>
  <path id="c" style="fill:none;stroke:#48c85a;stroke-width:2"
        d="M0 20L40 20"/>
  <path id="d" fill="#181a1e" d="M0 30L10 30L10 40Z"/>
  <path id="e" fill="#b8bec4" d="M12 30L22 30L22 40Z"/>
  <path id="f" stroke="#bbc1c6" d="M24 35L34 35"/>
  <path id="g" fill="#f7f8fa" d="M36 30L46 30L46 40Z"/>
  <path id="h" fill="#295cc7" d="M50 30L60 30L60 40Z"/>
</svg>
"""


def local_name(name):
    return name.rsplit("}", 1)[-1]


class PaintRoleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.svg = self.directory / "sample.svg"
        self.svg.write_text(SVG, encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_clusters_by_colour_role_not_names(self):
        manifest = build_paint_role_manifest(self.svg)
        self.assertEqual(manifest["schema"], MANIFEST_SCHEMA)
        self.assertFalse(manifest["compatibility"]["css_required"])
        counts = manifest["resource_counts"]
        self.assertEqual(counts["gradient_resources"], 1)
        self.assertEqual(counts["chromatic_role_controls"], 2)

        role_by_colour = manifest["role_by_color"]
        accent = role_by_colour["#62e53a"]
        self.assertEqual(role_by_colour["#287a3c"], accent)
        self.assertEqual(role_by_colour["#b5f23a"], accent)
        self.assertEqual(role_by_colour["#48c85a"], accent)
        self.assertNotEqual(role_by_colour["#295cc7"], accent)
        self.assertEqual(role_by_colour["#181a1e"], "neutral-dark")
        self.assertEqual(role_by_colour["#f7f8fa"], "neutral-light")
        self.assertEqual(role_by_colour["#b8bec4"], "neutral-mid")
        self.assertEqual(role_by_colour["#bbc1c6"], "neutral-mid")

        encoded = json.dumps(manifest, ensure_ascii=False)
        self.assertNotIn("green", encoded.lower())
        self.assertNotIn("blue", encoded.lower())

    def test_annotation_is_inert_and_rebuildable(self):
        manifest = build_paint_role_manifest(self.svg)
        before = structure_signature(self.svg)
        annotated = self.directory / "annotated.svg"
        result = annotate_svg_with_paint_roles(self.svg, manifest, annotated)
        self.assertEqual(result["rendering_attributes_changed"], 0)
        self.assertFalse(result["css_required"])
        self.assertEqual(before, structure_signature(annotated))

        root = ET.parse(annotated).getroot()
        metadata = [element for element in root
                    if local_name(element.tag) == "metadata"]
        self.assertEqual(len(metadata), 1)
        embedded = json.loads(metadata[0].text)
        self.assertEqual(embedded["schema"], MANIFEST_SCHEMA)
        annotated_elements = [element for element in root.iter()
                              if any(key.startswith("data-paint-role-")
                                     for key in element.attrib)]
        self.assertTrue(annotated_elements)

        rebuilt = build_paint_role_manifest(annotated)
        self.assertEqual(manifest["source"]["paint_signature"],
                         rebuilt["source"]["paint_signature"])
        self.assertEqual(manifest["role_by_color"], rebuilt["role_by_color"])

    def test_one_role_recolours_solids_strokes_and_gradient_stops(self):
        manifest = build_paint_role_manifest(self.svg)
        accent = manifest["role_by_color"]["#62e53a"]
        role = next(item for item in manifest["roles"] if item["id"] == accent)
        source_colours = {item["hex"] for item in role["members"]}
        output = self.directory / "recoloured.svg"
        result = apply_role_recolor(
            self.svg, manifest, {accent: "#d942b8"}, output)
        self.assertTrue(result["structure_unchanged"])
        self.assertGreaterEqual(result["changed_by_property"]["fill"], 1)
        self.assertGreaterEqual(result["changed_by_property"]["stroke"], 1)
        self.assertEqual(result["changed_by_property"]["stop-color"], 2)

        rewritten = output.read_text(encoding="utf-8").lower()
        output_colours = set(build_paint_role_manifest(output)["role_by_color"])
        for colour in source_colours:
            self.assertNotIn(colour, output_colours)
        for colour in ("#181a1e", "#b8bec4", "#bbc1c6", "#f7f8fa", "#295cc7"):
            self.assertIn(colour, rewritten)
        self.assertNotIn("var(--", rewritten)
        self.assertEqual(structure_signature(self.svg), structure_signature(output))
        ET.parse(output)  # still ordinary, parseable SVG

    def test_stale_manifest_is_rejected_before_write(self):
        manifest = build_paint_role_manifest(self.svg)
        changed = self.directory / "changed.svg"
        changed.write_text(SVG.replace("#62e53a", "#ff8400"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            annotate_svg_with_paint_roles(
                changed, manifest, self.directory / "must_not_exist.svg")
        self.assertFalse((self.directory / "must_not_exist.svg").exists())

    def test_oklch_round_trip_is_close(self):
        for colour in ("#181a1e", "#62e53a", "#f7f8fa", "#295cc7"):
            converted = oklch_to_hex(*hex_to_oklch(colour))
            differences = [abs(int(colour[index:index + 2], 16) -
                               int(converted[index:index + 2], 16))
                           for index in (1, 3, 5)]
            self.assertLessEqual(max(differences), 1)

    def test_manifest_commit_failure_preserves_previous_file(self):
        target = self.directory / "roles.json"
        target.write_bytes(b"previous manifest")
        with mock.patch("paint_roles.os.replace",
                        side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                write_paint_role_manifest(self.svg, target)
        self.assertEqual(target.read_bytes(), b"previous manifest")
        self.assertEqual(list(self.directory.glob(".roles.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
