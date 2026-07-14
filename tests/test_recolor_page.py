import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

from paint_roles import _role_mapping, build_paint_role_manifest
from recolor_page import make_recolor_html


SVG = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
  <defs><linearGradient id="g"><stop stop-color="#66aa22"/></linearGradient></defs>
  <path fill="#88cc22" stroke="#224400" d="M1 1H19V19H1Z"/>
</svg>'''


PROJECT = Path(__file__).resolve().parents[1]


def json_block(page, element_id):
    marker = f'id="{element_id}" type="application/json">'
    return page.split(marker, 1)[1].split("</script>", 1)[0]


def paint_math(page):
    match = re.search(
        r"/\* PAINT_MATH_START \*/(.*?)/\* PAINT_MATH_END \*/",
        page, re.DOTALL)
    if not match:
        raise AssertionError("paint math marker was not found")
    return match.group(1)


def run_node(page, expression, payload):
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("Node is only used for the development parity test")
    program = paint_math(page) + "\n" + f'''
let input='';
process.stdin.setEncoding('utf8');
process.stdin.on('data',chunk=>input+=chunk);
process.stdin.on('end',()=>{{
  const payload=JSON.parse(input);
  const result={expression};
  process.stdout.write(JSON.stringify(result));
}});
'''
    completed = subprocess.run(
        [node, "-e", program], input=json.dumps(payload), text=True,
        capture_output=True, check=False, timeout=20)
    if completed.returncode:
        raise AssertionError(completed.stderr)
    return json.loads(completed.stdout)


def manifest():
    return {
        "schema": "ai-vector-cleanroom.paint-roles/v1",
        "roles": [{
            "id": "accent-1",
            "label": "Accent 1",
            "kind": "chromatic",
            "member_count": 2,
            "usage_count": 2,
            "control": {
                "default_hex": "#88cc22",
                "anchor_lightness": 0.8,
                "anchor_chroma": 0.2,
            },
            "members": [
                {"hex": "#88cc22", "relative": {
                    "lightness_delta": 0.0, "chroma_ratio": 1.0,
                    "hue_delta_degrees": 0.0}},
                {"hex": "#66aa22", "relative": {
                    "lightness_delta": -0.1, "chroma_ratio": 0.8,
                    "hue_delta_degrees": 2.0}},
            ],
        }],
    }


class RecolorPageTests(unittest.TestCase):
    def make_page(self, svg_text=SVG, paint_manifest=None, **options):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        svg = root / "logo.svg"
        out = root / "colors.html"
        svg.write_text(svg_text, encoding="utf-8")
        make_recolor_html(
            out, svg, manifest() if paint_manifest is None else paint_manifest,
            **options)
        return out.read_text(encoding="utf-8")

    def test_self_contained_page_exports_explicit_svg(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            out = root / "colors.html"
            svg.write_text(SVG, encoding="utf-8")
            make_recolor_html(out, svg, manifest(),
                              download_filename="brand-blue.svg",
                              tool_version="test-version")
            text = out.read_text(encoding="utf-8")
        self.assertIn('data-role="accent-1"', text)
        self.assertIn("主色系 1", text)
        self.assertIn("#88cc22", text)
        self.assertIn("new XMLSerializer", text)
        self.assertIn("new DOMParser", text)
        self.assertIn("explicit SVG presentation attributes", text)
        self.assertIn('a.download="brand-blue.svg"', text)
        self.assertIn("setAttributeNS(XMLNS_NS,'xmlns',SVG_NS)", text)
        self.assertIn("if(c.value.toLowerCase()===defaults", text)
        self.assertNotIn("<script src=", text.lower())
        self.assertNotIn("<link ", text.lower())
        self.assertNotIn("fetch(", text)
        self.assertNotIn('<template id="svg-source">', text)

    def test_manifest_json_cannot_break_script_block(self):
        value = manifest()
        value["scope_note"] = "</script><script>bad()</script>"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            out = root / "colors.html"
            svg.write_text(SVG, encoding="utf-8")
            make_recolor_html(out, svg, value)
            text = out.read_text(encoding="utf-8")
        block = text.split('id="paint-manifest" type="application/json">', 1)[1]
        block = block.split("</script>", 1)[0]
        self.assertNotIn("<script>bad", block)
        decoded = json.loads(block)
        self.assertEqual(decoded["scope_note"], "</script><script>bad()</script>")

    def test_svg_source_is_json_encoded_and_keeps_svg_namespace(self):
        dangerous_text = SVG.replace(
            "<defs>", "<!-- </script><script>bad()</script> --><defs>")
        text = self.make_page(dangerous_text)
        self.assertNotIn("<script>bad()", text)
        decoded = json.loads(json_block(text, "svg-source"))
        self.assertEqual(decoded, dangerous_text)
        root = ET.fromstring(decoded)
        self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
        self.assertFalse(any(
            element.tag.rsplit("}", 1)[-1] in {"image", "foreignObject"}
            for element in root.iter()))

    def test_active_or_external_svg_content_is_rejected(self):
        variants = {
            "script": SVG.replace("</svg>", "<script>bad()</script></svg>"),
            "image": SVG.replace(
                "</svg>", '<image href="data:image/png;base64,AA=="/></svg>'),
            "event": SVG.replace("<path ", '<path onload="bad()" '),
            "external": SVG.replace(
                "</svg>", '<use href="https://example.invalid/a.svg#x"/></svg>'),
            "css": SVG.replace(
                "</svg>", '<style>@import url(https://example.invalid/x.css)</style></svg>'),
        }
        for label, svg_text in variants.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.make_page(svg_text)

    def test_manifest_values_are_validated_before_html_generation(self):
        invalid_default = manifest()
        invalid_default["roles"][0]["control"]["default_hex"] = '"><script>'
        with self.assertRaisesRegex(ValueError, "default colour"):
            self.make_page(paint_manifest=invalid_default)

        duplicate = manifest()
        duplicate["roles"][0]["members"][1]["hex"] = "#88cc22"
        with self.assertRaisesRegex(ValueError, "duplicate paint member"):
            self.make_page(paint_manifest=duplicate)

        non_finite = manifest()
        non_finite["roles"][0]["members"][0]["relative"]["chroma_ratio"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid chroma_ratio"):
            self.make_page(paint_manifest=non_finite)

    def test_download_filename_is_a_safe_svg_basename(self):
        text = self.make_page(
            download_filename='../bad"><script>name')
        self.assertIn('a.download="_bad_script_name.svg"', text)
        self.assertNotIn('a.download="../', text)

    def test_js_oklch_and_gamut_mapping_matches_python(self):
        paint_manifest = manifest()
        text = self.make_page(paint_manifest=paint_manifest)
        role = paint_manifest["roles"][0]
        targets = ["#205cff", "#ff00ff", "#00ffff", "#050505", "#fefefe"]
        cases = [{"role": role, "target": target} for target in targets]
        javascript = run_node(
            text, "payload.map(item=>mappingFor(item.role,item.target))", cases)
        python = [_role_mapping(role, target, 0.5) for target in targets]
        self.assertEqual(javascript, python)

    def test_complete_inline_javascript_passes_node_syntax_check(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node is only used for the development syntax test")
        text = self.make_page()
        scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>",
                             text, re.DOTALL)
        self.assertGreaterEqual(len(scripts), 3)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "recolor-page.js"
            source.write_text(scripts[-1], encoding="utf-8")
            completed = subprocess.run(
                [node, "--check", str(source)], text=True,
                capture_output=True, check=False, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_js_paint_token_replacement_matches_explicit_svg_forms(self):
        text = self.make_page()
        payload = {
            "map": {"#aabbcc": "#112233"},
            "values": [
                "#abc", "#aabbcc80", "rgb(170,187,204)",
                "rgba(170,187,204,0.5)",
                "rgb(66.666%,73.333%,80%)", "url(#gradient)",
            ],
        }
        replaced = run_node(
            text,
            "payload.values.map(value=>replacePaint(value,payload.map))",
            payload,
        )
        self.assertEqual(replaced, [
            "#112233", "#11223380", "rgb(17,34,51)",
            "rgba(17,34,51,0.5)", "rgb(17,34,51)", None,
        ])
        self.assertIn("['fill','stroke','stop-color']", text)
        self.assertIn("el.hasAttribute(attr)", text)
        self.assertIn("el.style.getPropertyValue(attr)", text)

    def test_empty_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            svg = root / "logo.svg"
            svg.write_text(SVG, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no controls"):
                make_recolor_html(root / "colors.html", svg, {"roles": []})


if __name__ == "__main__":
    unittest.main()
