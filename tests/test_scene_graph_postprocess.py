"""Tests for conservative cross-paint SVG object grouping."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene_graph_postprocess import (  # noqa: E402
    _path_bbox,
    build_scene_graph,
    process_svg_file,
)


SVG = "http://www.w3.org/2000/svg"


def _svg(body: str, viewbox: str = "0 0 200 100") -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{viewbox}" width="200" height="100">'
        f"{body}</svg>"
    )


def _objects(text: str):
    root = ET.fromstring(text)
    return [element for element in root.iter(f"{{{SVG}}}g")
            if (element.get("id") or "").startswith("object-")]


def _drawables(element: ET.Element):
    wanted = {f"{{{SVG}}}{name}" for name in
              ("path", "circle", "ellipse", "rect", "line", "polyline", "polygon")}
    return [item for item in element.iter() if item.tag in wanted]


class SceneGraphPostprocessTests(unittest.TestCase):
    def test_cross_paint_overlay_becomes_selectable_group(self):
        source = _svg(
            '<g id="base" fill="#16324f">'
            '<rect x="10" y="10" width="30" height="30"/>'
            '</g>'
            '<g id="unrelated" fill="#00aa66">'
            '<rect x="120" y="10" width="20" height="20"/>'
            '</g>'
            '<g id="highlight" fill="#ffffff">'
            '<rect x="18" y="18" width="13" height="13"/>'
            '</g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "applied", result.report)
        objects = _objects(result.svg_text)
        self.assertEqual(len(objects), 1)
        self.assertEqual(len(_drawables(objects[0])), 2)
        self.assertEqual(objects[0].get("data-member-count"), "2")
        self.assertGreater(float(objects[0].get("data-group-confidence")), 0.8)
        self.assertEqual(objects[0].get("data-group-kind"), "layered-object")
        self.assertIn("layered object", result.report["groups"][0]["label"])
        fills = {element.get("fill") for element in _drawables(objects[0])}
        self.assertEqual(fills, {"#16324f", "#ffffff"})
        self.assertEqual(result.report["paint_order_validation"], "passed")
        self.assertEqual(result.report["ungrouped_drawables"], 1)

    def test_stable_ids_and_output_are_deterministic(self):
        source = _svg(
            '<g id="a" fill="#111"><circle cx="20" cy="20" r="8"/></g>'
            '<g id="b" fill="#eee"><circle cx="20" cy="20" r="5"/></g>'
        )
        first = build_scene_graph(source)
        second = build_scene_graph(source)
        self.assertEqual(first.status, "applied")
        self.assertEqual(first.svg_text, second.svg_text)
        self.assertEqual(first.report["groups"], second.report["groups"])
        group = _objects(first.svg_text)[0]
        ids = [element.get("id") for element in _drawables(group)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(node_id.startswith("sg-node-") for node_id in ids))

    def test_intervening_overlapping_scaffold_prevents_unsafe_move(self):
        source = _svg(
            '<g id="base" fill="#222"><rect x="20" y="20" width="20" height="20"/></g>'
            # This large element is excluded from semantic grouping but its
            # paint order still blocks moving either small part across it.
            '<g id="scaffold" fill="#ccc"><rect x="0" y="0" width="150" height="90"/></g>'
            '<g id="highlight" fill="#fff"><rect x="22" y="22" width="16" height="16"/></g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "no_change")
        self.assertEqual(result.svg_text, source)
        self.assertEqual(result.report["candidate_groups"], 1)
        self.assertEqual(len(result.report["skipped_unsafe_groups"]), 1)
        self.assertEqual(result.report["actual_dom_group_count"], 0)
        self.assertEqual(result.report["manifest_only_group_count"], 1)
        self.assertEqual(result.report["manifest_only_groups"][0]["mode"],
                         "manifest-only")

    def test_two_halftone_clusters_stay_separate(self):
        layers = []
        points = [
            (20, 20, "#8cfe01"), (32, 20, "#cefd2a"),
            (44, 20, "#8cfe01"), (56, 20, "#cefd2a"),
            (140, 70, "#8cfe01"), (152, 70, "#cefd2a"),
            (164, 70, "#8cfe01"), (176, 70, "#cefd2a"),
        ]
        # One circle per paint layer mimics a tracer that splits identical
        # visual clusters across many non-semantic color-stack groups.
        for index, (x, y, fill) in enumerate(points):
            layers.append(
                f'<g id="paint-{index}" fill="{fill}">'
                f'<circle cx="{x}" cy="{y}" r="4"/></g>'
            )
        result = build_scene_graph(_svg("".join(layers)))
        self.assertEqual(result.status, "applied", result.report)
        objects = _objects(result.svg_text)
        self.assertEqual(len(objects), 2)
        self.assertEqual(sorted(len(_drawables(group)) for group in objects), [4, 4])
        boxes = sorted(group["bbox"] for group in result.report["groups"])
        self.assertLess(boxes[0][2], boxes[1][0])
        self.assertTrue(all("repeated-dot-proximity" in group["reasons"]
                            for group in result.report["groups"]))

    def test_adjacent_chromatic_decoration_does_not_join_neutral_object(self):
        source = _svg(
            '<g id="neutral-base" fill="#222">'
            '<rect x="10" y="10" width="30" height="30"/></g>'
            '<g id="green-decoration" fill="#8cfe01">'
            '<rect x="39" y="10" width="25" height="30"/></g>'
            '<g id="neutral-highlight" fill="#fff">'
            '<rect x="18" y="18" width="13" height="13"/></g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "applied", result.report)
        objects = _objects(result.svg_text)
        self.assertEqual(len(objects), 1)
        self.assertEqual(len(_drawables(objects[0])), 2)
        self.assertEqual(result.report["ungrouped_drawables"], 1)
        fills = {element.get("fill") for element in _drawables(objects[0])}
        self.assertEqual(fills, {"#222", "#fff"})

    def test_external_render_validator_can_force_byte_exact_rollback(self):
        source = _svg(
            '<g id="a" fill="#111"><circle cx="20" cy="20" r="8"/></g>'
            '<g id="b" fill="#eee"><circle cx="20" cy="20" r="5"/></g>'
        )
        result = build_scene_graph(source, validator=lambda _old, _new: False)
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.svg_text, source)
        self.assertIn("validator", result.report["reason"])

    def test_unsafe_group_transform_rolls_back_without_partial_changes(self):
        source = _svg(
            '<g id="a" fill="#111" transform="translate(3 2)">'
            '<circle cx="20" cy="20" r="8"/></g>'
            '<g id="b" fill="#eee"><circle cx="20" cy="20" r="5"/></g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.svg_text, source)
        self.assertIn("transform", result.report["reason"])

    def test_drawable_transform_rolls_back_when_bbox_cannot_be_trusted(self):
        source = _svg(
            '<g id="paint" fill="#111">'
            '<circle cx="20" cy="20" r="8" transform="translate(3 2)"/>'
            '</g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.svg_text, source)
        self.assertIn("transform", result.report["reason"])

    def test_circle_rotation_about_own_center_is_bbox_safe(self):
        source = _svg(
            '<g id="ring" fill="none" stroke="#8cfe01">'
            '<circle cx="20" cy="20" r="8" stroke-dasharray="10 5" '
            'transform="rotate(-3.5 20 20)"/></g>'
            '<g id="highlight" fill="#ffffff">'
            '<circle cx="20" cy="20" r="5"/></g>'
        )
        result = build_scene_graph(source)
        self.assertIn(result.status, {"applied", "no_change"}, result.report)
        root = ET.fromstring(result.svg_text)
        rotated = next(element for element in root.iter(f"{{{SVG}}}circle")
                       if element.get("transform"))
        self.assertEqual(rotated.get("transform"), "rotate(-3.5 20 20)")

    def test_arc_bbox_accounts_for_svg_radius_correction(self):
        # SVG scales too-small radii until the arc can reach both endpoints.
        # A 1-unit radius over a 1000-unit chord therefore needs a large bbox.
        bbox = _path_bbox("M 0 0 A 1 1 0 0 0 1000 0")
        self.assertIsNotNone(bbox)
        self.assertLessEqual(bbox.y0, -500)
        self.assertGreaterEqual(bbox.y1, 500)

    def test_group_opacity_rolls_back_because_flattening_changes_compositing(self):
        source = _svg(
            '<g id="transparent-stack" opacity="0.5" fill="#111">'
            '<circle cx="20" cy="20" r="8"/>'
            '<circle cx="24" cy="20" r="8"/></g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.svg_text, source)
        self.assertIn("opacity", result.report["reason"])

    def test_unsupported_root_visual_rolls_back_instead_of_changing_z_order(self):
        source = _svg(
            '<g id="bottom" fill="#111">'
            '<circle cx="20" cy="20" r="8"/></g>'
            '<text x="10" y="25">middle</text>'
            '<g id="top" fill="#eee">'
            '<circle cx="20" cy="20" r="5"/></g>'
        )
        result = build_scene_graph(source)
        self.assertEqual(result.status, "rolled_back")
        self.assertEqual(result.svg_text, source)
        self.assertIn("root scene content", result.report["reason"])

    def test_file_fallback_preserves_original_bom_and_newline_bytes(self):
        source_text = _svg(
            '<g id="unsafe" transform="translate(1 1)" fill="#111">'
            '<circle cx="20" cy="20" r="8"/></g>'
        ).replace("><", ">\r\n<")
        source_bytes = b"\xef\xbb\xbf" + source_text.encode("utf-8")
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.svg"
            destination = Path(directory) / "destination.svg"
            source.write_bytes(source_bytes)
            result = process_svg_file(source, destination)
            self.assertEqual(result.status, "rolled_back")
            self.assertEqual(destination.read_bytes(), source_bytes)

    def test_report_is_embedded_and_scope_is_honest(self):
        source = _svg(
            '<g id="a" fill="#111"><circle cx="20" cy="20" r="8"/></g>'
            '<g id="b" fill="#eee"><circle cx="20" cy="20" r="5"/></g>'
        )
        result = build_scene_graph(source)
        root = ET.fromstring(result.svg_text)
        metadata = next(element for element in root.findall(f"{{{SVG}}}metadata")
                        if element.get("id") == "scene-graph-metadata")
        embedded = json.loads(metadata.text)
        self.assertEqual(embedded["groups"], result.report["groups"])
        self.assertIn("Neither mode proves", embedded["scope_note"])
        self.assertEqual(root.get("data-scene-graph-version"), "beta3-0.1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
