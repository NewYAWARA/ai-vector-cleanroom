"""Selection must not corrupt source geometry or conceal incomplete handoffs."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from designer_handoff import (build_handoff_manifest, export_handoff,
                              normalized_svg_text, validate_decisions)

SVG = "http://www.w3.org/2000/svg"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN3sAAAAASUVORK5CYII=")


class DesignerHandoffTests(unittest.TestCase):
    def test_partial_paint_hint_uses_current_verified_members_only(self):
        self.write('<rect id="partial" x="10" y="10" width="20" height="20"/>'
                   '<rect id="other" x="60" y="60" width="10" height="10"/>')
        report = {'designer_quality': {
            'source': {'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()},
            'gradient_object_gate': {'source_space_field_evidence': {'objects': [
                {'passed': True, 'partial_paint_requires_manual_review': True,
                 'final_drawable_ids': ['partial']}]}}}}
        items = {o['id']: o for o in build_handoff_manifest(self.source, report)['objects']}
        self.assertEqual(items['partial']['partial_paint_member_count'], 1)
        self.assertEqual(items['partial']['suggested_action'], 'review')
        self.assertTrue(any('只改善漸層填色' in reason for reason in items['partial']['reasons']))
        self.assertEqual(items['other']['partial_paint_member_count'], 0)
        report['designer_quality']['source']['sha256'] = '0' * 64
        self.assertEqual(build_handoff_manifest(self.source, report)['objects'][0]['partial_paint_member_count'], 0)

    def test_small_source_defect_requires_review_despite_few_anchors(self):
        from PIL import Image
        self.write('<rect id="suspect" x="20" y="20" width="20" height="10"/>'
                   '<rect id="clear" x="70" y="70" width="10" height="10"/>')
        report = {'editability_enhancements': {'stages': {'source_topology_audit': {
            'status': 'completed', 'renderer': {'width': 100, 'height': 100},
            'inputs_unchanged': True,
            'source_hashes': {'svg': hashlib.sha256(self.source.read_bytes()).hexdigest()},
            'stable_defects': [{'kind': 'source_paper_unexpected_opaque_color',
                                'bbox_xyxy': [22, 22, 38, 28]}]}}}}
        # Region boxes alone do not assign a warning. Supply independent
        # original pixels to locate the visible overpaint precisely.
        Image.new('RGBA', (100, 100), 'white').save(self.png)
        report['_handoff_reference_kind'] = 'original'
        # The clear unit should agree with the source, unlike suspect.
        from PIL import ImageDraw
        source = Image.open(self.png).convert('RGBA')
        ImageDraw.Draw(source).rectangle((70, 70, 79, 79), fill='black')
        source.save(self.png)
        items = {o['id']: o for o in build_handoff_manifest(self.source, report, source_png=self.png)['objects']}
        self.assertEqual(items['suspect']['suggested_action'], 'review')
        self.assertEqual(items['suspect']['source_defect_count'], 1)
        self.assertIn('近白或透明', items['suspect']['reasons'][0])
        self.assertEqual(items['clear']['source_defect_count'], 0)
        self.assertEqual(items['clear']['suggested_action'], 'keep')

    def test_stale_source_locations_are_not_carried_to_edited_artwork(self):
        self.write('<rect id="suspect" x="20" y="20" width="20" height="10"/>')
        report = {'editability_enhancements': {'stages': {'source_topology_audit': {
            'status': 'completed', 'renderer': {'width': 100, 'height': 100},
            'inputs_unchanged': True, 'source_hashes': {'svg': '0' * 64},
            'stable_defects': [{'kind': 'source_paper_unexpected_opaque_color',
                                'bbox_xyxy': [22, 22, 38, 28]}]}}}}
        manifest = build_handoff_manifest(self.source, report)
        self.assertFalse(manifest['source_topology_locations_current'])
        self.assertEqual(manifest['objects'][0]['source_defect_count'], 0)

    def test_curve_warning_follows_exact_member_in_group_and_current_svg(self):
        self.write('<g id="object-group"><path id="edge" d="M0 0L20 0L20 20Z"/>'
                   '<rect id="other" x="30" y="30" width="5" height="5"/></g>')
        quality = {'source': {'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest()},
                   'curve_economy_gate': {'short_counterturn_diagnostic': {'paths': [
                       {'id': 'edge', 'short_counterturn_count': 6, 'requires_review': True}]}}}
        manifest = build_handoff_manifest(self.source, {'designer_quality': quality})
        self.assertEqual(manifest['objects'][0]['curve_review_count'], 6)
        self.assertTrue(any('反覆轉彎' in reason for reason in manifest['objects'][0]['reasons']))
        quality['source']['sha256'] = '0' * 64
        stale = build_handoff_manifest(self.source, {'designer_quality': quality})
        self.assertEqual(stale['objects'][0]['curve_review_count'], 0)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source.svg"
        self.png = self.base / "reference.png"
        self.png.write_bytes(PNG)

    def write(self, body, attributes='viewBox="0 0 100 100"'):
        self.source.write_text(f'<svg xmlns="{SVG}" {attributes}>{body}</svg>', encoding="utf-8")
        return build_handoff_manifest(self.source)

    def export(self, decisions, manifest=None):
        manifest = manifest or build_handoff_manifest(self.source)
        return export_handoff(self.source, self.png, decisions, self.base / "out",
                              expected_sha256=manifest["svg_sha256"])

    def parse(self, result, key):
        return ET.parse(result["files"][key]).getroot()

    def test_topmost_object_groups_have_disjoint_ownership_and_stable_ids(self):
        manifest = self.write('''<g id="object-outer"><circle cx="5" cy="5" r="2"/>
          <g id="object-inner"><rect width="2" height="2"/></g></g>
          <g><path d="M0 0L5 0L5 5Z"/></g>''')
        self.assertEqual(len(manifest["objects"]), 2)
        self.assertEqual(manifest["objects"][0]["id"], "object-outer")
        self.assertEqual(len(manifest["objects"][0]["member_ids"]), 2)
        member_ids = [identifier for item in manifest["objects"] for identifier in item["member_ids"]]
        self.assertEqual(len(member_ids), len(set(member_ids)))
        preview = ET.fromstring(normalized_svg_text(self.source))
        preview_ids = {item.get("id") for item in preview.iter()}
        self.assertTrue(set(member_ids) <= preview_ids)
        self.assertEqual(manifest, build_handoff_manifest(self.source))
        self.assertTrue(all(value == "review" for value in manifest["default_decisions"].values()))

    def test_working_artwork_is_editable_without_manual_adoption_and_preserves_appearance(self):
        from PIL import Image
        from svg_renderer import render_svg_reference
        manifest = self.write('<circle id="back" cx="40" cy="40" r="25" fill="red"/>'
                              '<rect id="front" x="40" y="30" width="25" height="40" fill="white"/>')
        result = self.export({'back':'review','front':'redraw'},manifest)
        working = self.parse(result,'working_svg')
        for identifier in ('back','front'):
            element=next(item for item in working.iter() if item.get('id')==identifier)
            self.assertNotIn('display:none',element.get('style',''))
        reference=next(item for item in working.iter() if item.get('data-handoff-role')=='raster-reference')
        self.assertEqual(reference.get('style'),'display:none')
        a,b=self.base/'original.png',self.base/'working.png'
        render_svg_reference(self.source,a,width=200,background=None)
        render_svg_reference(result['files']['working_svg'],b,width=200,background=None)
        self.assertEqual(Image.open(a).tobytes(),Image.open(b).tobytes())
        metadata=next(item for item in working.iter() if item.tag==f'{{{SVG}}}metadata')
        self.assertTrue(json.loads(metadata.text)['not_finished_artwork'])
        self.assertEqual(result['summary']['keep'],0)

    def test_nested_transforms_styles_holes_and_masks_survive_pruning(self):
        manifest = self.write('''<defs><mask id="m"><rect width="100" height="100" fill="white"/></mask>
          <clipPath id="c"><circle cx="40" cy="40" r="30"/></clipPath>
          <linearGradient id="grad"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs>
          <g id="parent" transform="translate(5,6)" style="opacity:0.8" clip-path="url(#c)">
          <g id="object-ring" fill="url(#grad)" mask="url(#m)">
          <path id="ring" fill-rule="evenodd" d="M0 0L30 0L30 30L0 30ZM5 5L5 25L25 25L25 5Z"/></g>
          <rect id="remove" x="1" y="1" width="4" height="4"/></g>''')
        self.assertEqual(manifest["objects"][0]["bbox"], [5, 6, 30, 30])
        original = self.source.read_bytes()
        result = self.export({"object-ring": "keep", "remove": "redraw"}, manifest)
        root = self.parse(result, "accepted_svg")
        ids = {item.get("id"): item for item in root.iter() if item.get("id")}
        self.assertNotIn("remove", ids)
        self.assertEqual(ids["parent"].get("transform"), "translate(5,6)")
        self.assertEqual(ids["parent"].get("style"), "opacity:0.8")
        self.assertEqual(ids["parent"].get("clip-path"), "url(#c)")
        self.assertEqual(ids["object-ring"].get("mask"), "url(#m)")
        self.assertEqual(ids["ring"].get("fill-rule"), "evenodd")
        self.assertEqual(ids["ring"].get("d"), "M0 0L30 0L30 30L0 30ZM5 5L5 25L25 25L25 5Z")
        self.assertTrue({"m", "c", "grad"} <= set(ids))
        self.assertEqual(self.source.read_bytes(), original)

    def test_accepted_overlapping_members_keep_original_paint_order(self):
        manifest = self.write('''<rect id="back" width="40" height="40" fill="red"/>
          <g id="object-middle" opacity="0.6"><circle id="middle" cx="20" cy="20" r="10"/></g>
          <path id="front" d="M0 0L40 0L20 40Z" fill="blue"/>''')
        result = self.export({"back": "keep", "object-middle": "keep", "front": "review"}, manifest)
        root = self.parse(result, "accepted_svg")
        self.assertEqual([item.get("id") for item in root if item.tag in {f"{{{SVG}}}rect", f"{{{SVG}}}g"}],
                         ["back", "object-middle"])
        draft = self.parse(result, "draft_svg")
        original_order = [item.get("id") for item in draft if item.get("id") in {"back", "object-middle", "front"}]
        self.assertEqual(original_order, ["back", "object-middle", "front"])
        front = next(item for item in draft if item.get("id") == "front")
        self.assertIn("display:none", front.get("style"))

    def test_all_redraw_has_empty_vectors_but_embedded_reference_and_explicit_omissions(self):
        manifest = self.write('<rect id="rect" x="10" y="20" width="30" height="40"/>')
        result = self.export({"rect": "redraw"}, manifest)
        accepted = self.parse(result, "accepted_svg")
        self.assertFalse(any(item.tag == f"{{{SVG}}}rect" for item in accepted))
        draft = self.parse(result, "draft_svg")
        images = list(draft.iter(f"{{{SVG}}}image"))
        self.assertEqual(len(images), 1)
        self.assertTrue(images[0].get("{http://www.w3.org/1999/xlink}href").startswith("data:image/png;base64,"))
        handoff = json.loads(Path(result["files"]["handoff_json"]).read_text(encoding="utf-8"))
        self.assertEqual(handoff["summary"]["omitted_object_ids"], ["rect"])
        self.assertTrue(handoff["draft_has_raster_reference"])
        self.assertTrue(handoff["draft_has_guides"])
        self.assertTrue(handoff["not_finished_artwork"])
        self.assertEqual(handoff["omitted_objects"][0]["decision"], "redraw")
        saved = json.loads(Path(result["files"]["decisions_json"]).read_text(encoding="utf-8"))
        self.assertEqual(saved["decisions"], {"rect": "redraw"})

    def test_source_viewbox_offset_is_used_for_reference_alignment(self):
        manifest = self.write('<circle id="c" cx="20" cy="30" r="4"/>', 'viewBox="10 20 80 60"')
        result = self.export({"c": "review"}, manifest)
        image = next(self.parse(result, "draft_svg").iter(f"{{{SVG}}}image"))
        self.assertEqual([float(image.get(key)) for key in ("x", "y", "width", "height")], [10, 20, 80, 60])

    def test_stale_unknown_missing_and_invalid_decisions_fail_before_write(self):
        manifest = self.write('<rect id="r" width="2" height="2"/>')
        for decisions, digest in [({}, manifest["svg_sha256"]), ({"r": "keep", "x": "keep"}, manifest["svg_sha256"]),
                                  ({"r": "maybe"}, manifest["svg_sha256"]), ({"r": "keep"}, "0" * 64)]:
            with self.subTest(decisions=decisions, digest=digest):
                with self.assertRaises(ValueError):
                    export_handoff(self.source, self.png, decisions, self.base / "out", expected_sha256=digest)
                self.assertFalse((self.base / "out").exists())
        self.source.write_text(self.source.read_text(encoding="utf-8").replace('width="2"', 'width="3"'), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.export({"r": "keep"}, manifest)

    def test_missing_reference_fails_even_when_all_objects_are_kept(self):
        manifest = self.write('<circle id="c" r="4"/>')
        self.png.unlink()
        with self.assertRaises(FileNotFoundError):
            self.export({"c": "keep"}, manifest)
        self.assertFalse((self.base / "out").exists())

    def test_input_cannot_be_overwritten(self):
        manifest = self.write('<circle id="c" r="4"/>')
        accepted_source = self.base / "accepted.svg"
        accepted_source.write_bytes(self.source.read_bytes())
        with self.assertRaises(ValueError):
            export_handoff(accepted_source, self.png, {"c": "keep"}, self.base,
                           expected_sha256=manifest["svg_sha256"])

    def test_active_external_and_unsupported_svg_is_rejected(self):
        bodies = ['<script>alert(1)</script>', '<rect onclick="alert(1)"/>',
                  '<image href="https://example.com/a.png"/>', '<use href="#x"/>',
                  '<style>rect { fill: red; }</style>', '<rect fill="url(https://example.com/x.svg#a)"/>',
                  '<defs><linearGradient id="g" href="https://example.com/x.svg#a"/></defs>',
                  '<rect style="fill:u\\72l(https://example.com/x)"/>',
                  '<rect id="same"/><circle id="same"/>',
                  '<rect id="x"/><rect clip-path="url(#x)"/>']
        for body in bodies:
            with self.subTest(body=body):
                with self.assertRaises(ValueError):
                    self.write(body)
        self.source.write_text('<!DOCTYPE svg [<!ENTITY x "hello">]><svg xmlns="' + SVG + '" viewBox="0 0 10 10"/>', encoding="utf-8")
        with self.assertRaises(ValueError):
            normalized_svg_text(self.source)

    def test_arcs_strokes_and_transforms_have_conservative_bounds_text_stays_unknown(self):
        manifest = self.write('''<path id="arc" d="M10 10A20 20 0 0 1 40 40"/>
          <line id="stroke" x2="50" stroke="black"/><g transform="rotate(20)"><rect id="rot" width="10" height="10"/></g>
          <text id="text" x="10" y="10">Hi</text><rect id="plain" x="1" y="2" width="3" height="4"/>''')
        objects = {item["id"]: item for item in manifest["objects"]}
        self.assertIsNone(objects["text"]["bbox"])
        self.assertTrue(all(objects[key]["bbox"] is not None for key in ("arc", "stroke", "rot")))
        self.assertLess(objects["rot"]["bbox"][0], 0)
        self.assertGreater(objects["stroke"]["bbox"][2], 50)
        self.assertEqual(objects["plain"]["bbox"], [1, 2, 3, 4])

    def test_hidden_drawables_are_not_independent_decisions_and_do_not_leak_into_accepted(self):
        manifest = self.write('<g display="none"><circle id="hidden" r="4"/></g><rect id="visible" width="5" height="5"/>')
        self.assertEqual([item["id"] for item in manifest["objects"]], ["visible"])
        result = self.export({"visible": "keep"}, manifest)
        ids = {item.get("id") for item in self.parse(result, "accepted_svg").iter()}
        self.assertNotIn("hidden", ids)

    def test_foreground_shape_contained_in_defs_is_not_a_decision(self):
        manifest = self.write('<defs><clipPath id="c"><rect id="resource-shape" width="10" height="10"/></clipPath></defs><rect id="r" width="20" height="20" clip-path="url(#c)"/>')
        self.assertEqual([item["id"] for item in manifest["objects"]], ["r"])
        validate_decisions(manifest, {"r": "keep"}, manifest["svg_sha256"])

    def test_css_important_cannot_make_unselected_candidate_visible(self):
        manifest = self.write('<rect id="r" width="10" height="10" style="fill:red;display:inline!important"/>')
        result = self.export({"r": "review"}, manifest)
        candidate = next(item for item in self.parse(result, "draft_svg").iter() if item.get("id") == "r")
        self.assertEqual(candidate.get("style"), "fill:red;display:none!important")

    def test_old_full_image_validation_is_not_reused_for_partial_output(self):
        manifest = self.write('<metadata id="old-validation">{"acceptance_status":"accepted"}</metadata><circle id="c" r="4"/>')
        result = self.export({"c": "review"}, manifest)
        for key in ("accepted_svg", "draft_svg"):
            root = self.parse(result, key)
            self.assertFalse(any(item.get("id") == "old-validation" for item in root.iter()))
            metadata = next(root.iter(f"{{{SVG}}}metadata"))
            record = json.loads(metadata.text)
            self.assertTrue(record["not_finished_artwork"])
            self.assertEqual(record["source_validation_applies_to"], "original_input_only")

    def test_unsupported_root_effects_and_active_metadata_are_rejected(self):
        for attr in ('transform="translate(3,4)"', 'style="opacity:0.5"', 'display="none"'):
            with self.subTest(attr=attr), self.assertRaises(ValueError):
                self.write('<rect width="10" height="10"/>', 'viewBox="0 0 100 100" ' + attr)
        with self.assertRaises(ValueError):
            self.write('<metadata><animate attributeName="opacity" dur="1s" values="0;1"/></metadata>')


if __name__ == "__main__":
    unittest.main()
