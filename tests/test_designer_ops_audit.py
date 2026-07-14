from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from designer_ops_audit import AUDIT_SCHEMA, audit_designer_operations  # noqa: E402


def _complete_svg() -> str:
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
  <defs>
    <linearGradient id="flow" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#44cc55"/>
      <stop offset="1" stop-color="#aaff44"/>
    </linearGradient>
  </defs>
  <rect x="400" y="260" width="180" height="80" fill="url(#flow)"/>
  <circle id="outer" cx="500" cy="300" r="230" fill="none"
          stroke="#44cc55" stroke-width="10"/>
  <g id="component-a" data-group-mode="actual-dom"
     data-group-reasons="cross-paint-overlay">
    <rect x="40" y="230" width="140" height="90" fill="#112233"/>
    <rect x="62" y="247" width="100" height="56" fill="#fefefe"/>
  </g>
  <g id="component-b" data-group-mode="actual-dom"
     data-group-reasons="cross-paint-overlap">
    <rect x="820" y="230" width="140" height="90" fill="#112233"/>
    <rect x="842" y="247" width="100" height="56" fill="#44cc55"/>
  </g>
  <g id="texture-a" data-group-mode="actual-dom"
     data-group-reasons="repeated-dot-proximity">
    <circle cx="90" cy="80" r="5" fill="#44cc55"/>
    <circle cx="108" cy="80" r="5" fill="#aaff44"/>
    <circle cx="126" cy="80" r="5" fill="#44cc55"/>
    <circle cx="144" cy="80" r="5" fill="#aaff44"/>
  </g>
  <g id="texture-b" data-group-mode="actual-dom"
     data-group-reasons="parallel-stroke-proximity">
    <line x1="720" y1="70" x2="810" y2="50" stroke="#112233"/>
    <line x1="720" y1="85" x2="810" y2="65" stroke="#112233"/>
    <line x1="720" y1="100" x2="810" y2="80" stroke="#112233"/>
    <line x1="720" y1="115" x2="810" y2="95" stroke="#112233"/>
  </g>
</svg>'''


def _manifest() -> dict:
    return {
        "schema": "ai-vector-cleanroom.paint-roles/v1",
        "compatibility": {"css_required": False},
        "roles": [{
            "id": "family-1",
            "control_count": 1,
            "control": {"default_hex": "#44cc55", "transform": "relative"},
            "properties": [
                "attribute:fill", "attribute:stroke", "attribute:stop-color",
            ],
            "gradient_ids": ["flow"],
            "members": [{
                "hex": "#44cc55",
                "channels": {
                    "attribute:fill": 2,
                    "attribute:stroke": 1,
                    "attribute:stop-color": 1,
                },
                "gradient_ids": ["flow"],
            }],
        }],
        "role_by_color": {"#44cc55": "family-1"},
        "unsupported_paints": [],
    }


def _by_id(result: dict) -> dict[str, dict]:
    return {item["id"]: item for item in result["operations"]}


class DesignerOperationsAuditTests(unittest.TestCase):
    def _write(self, folder: str, text: str, name: str = "art.svg") -> Path:
        path = Path(folder) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_complete_native_resources_pass_all_five_operations(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, _complete_svg())
            result = audit_designer_operations(path, _manifest())

        self.assertEqual(result["schema"], AUDIT_SCHEMA)
        self.assertEqual(result["summary"]["passed"], 5, result)
        self.assertEqual(len(result["passed"]), 5)
        self.assertEqual(len(result["automatable"]), 5)
        self.assertEqual(
            result["acceptance_scope"],
            "generic_machine_detectable_structural_handles")
        self.assertEqual(result["semantic_task_validation"], "not_performed")
        self.assertEqual(result["timed_human_editing_validation"], "not_performed")
        self.assertEqual(result["human_acceptance"], "not_tested")
        operations = _by_id(result)
        colour = operations["global-colour-role"]
        self.assertEqual(colour["status"], "passed")
        self.assertEqual(colour["evidence"]["complete_role_ids"], ["family-1"])
        self.assertEqual(
            operations["native-gradient-resource"]["evidence"]["editable_gradient_count"],
            1,
        )
        local = operations["local-multicolour-dom-groups"]
        self.assertEqual(local["evidence"]["covered_regions"], ["left", "right"])
        ring = operations["native-main-outer-ring"]
        self.assertEqual(ring["evidence"]["native_ring_candidates"][0]["id"], "outer")
        decoration = operations["hideable-decoration-groups"]
        self.assertEqual(
            decoration["evidence"]["actual_dom_kinds"],
            ["halftone", "parallel-lines"],
        )
        # Public output must remain directly embeddable in report.json.
        json.dumps(result, ensure_ascii=False, allow_nan=False)

    def test_manifest_only_groups_never_count_as_selectable_or_hideable(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
          <rect x="20" y="20" width="960" height="560" fill="#112233"/>
        </svg>'''
        scene_report = {
            "actual_dom_groups": [],
            "manifest_only_groups": [
                {"id": "proposal-a", "mode": "manifest-only",
                 "reasons": ["cross-paint-overlay"], "paint_count": 2,
                 "bbox": [20, 200, 180, 320]},
                {"id": "proposal-b", "mode": "manifest-only",
                 "reasons": ["repeated-dot-proximity"], "paint_count": 2,
                 "bbox": [780, 40, 940, 120]},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, svg)
            result = audit_designer_operations(path, scene_graph_report=scene_report)
        operations = _by_id(result)
        self.assertEqual(operations["local-multicolour-dom-groups"]["status"], "failed")
        self.assertEqual(operations["hideable-decoration-groups"]["status"], "failed")
        self.assertEqual(result["svg"]["actual_dom_group_count"], 0)

    def test_report_claim_does_not_pass_when_group_is_absent_from_dom(self):
        scene_report = {
            "actual_dom_groups": [{
                "id": "not-in-document", "mode": "actual-dom",
                "reasons": ["cross-paint-overlay"], "paint_count": 2,
                "bbox": [20, 200, 180, 320],
            }],
        }
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">'
               '<rect x="20" y="20" width="50" height="50" fill="#111"/>'
               '</svg>')
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, svg)
            result = audit_designer_operations(path, scene_graph_report=scene_report)
        operation = _by_id(result)["local-multicolour-dom-groups"]
        self.assertNotEqual(operation["status"], "passed")
        self.assertEqual(operation["evidence"]["qualifying_actual_dom_group_count"], 0)

    def test_one_sided_semantic_group_is_only_partial(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
          <g id="only-one" data-group-mode="actual-dom"
             data-group-reasons="cross-paint-overlay">
            <rect x="30" y="200" width="150" height="100" fill="#111111"/>
            <rect x="50" y="220" width="100" height="60" fill="#eeeeee"/>
          </g>
        </svg>'''
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, svg)
            result = audit_designer_operations(path)
        operation = _by_id(result)["local-multicolour-dom-groups"]
        self.assertEqual(operation["status"], "partial")
        self.assertEqual(operation["evidence"]["covered_regions"], ["left"])

    def test_roles_split_across_controls_are_not_reported_as_one_click(self):
        manifest = {
            "schema": "ai-vector-cleanroom.paint-roles/v1",
            "roles": [
                {"id": "fill-only", "control_count": 1, "control": {},
                 "properties": ["attribute:fill"], "members": [{"hex": "#44cc55"}]},
                {"id": "stroke-only", "control_count": 1, "control": {},
                 "properties": ["attribute:stroke"], "members": [{"hex": "#44cc55"}]},
                {"id": "gradient-only", "control_count": 1, "control": {},
                 "properties": ["attribute:stop-color"], "gradient_ids": ["flow"],
                 "members": [{"hex": "#44cc55"}]},
            ],
            "unsupported_paints": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, _complete_svg())
            result = audit_designer_operations(path, manifest)
        operation = _by_id(result)["global-colour-role"]
        self.assertEqual(operation["status"], "partial")
        self.assertFalse(operation["automatable"])
        self.assertEqual(operation["evidence"]["complete_role_ids"], [])

    def test_small_dots_and_large_filled_circle_do_not_pass_as_outer_ring(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
          <circle id="large-disc" cx="500" cy="300" r="230"
                  fill="#44cc55" stroke="#111111" stroke-width="10"/>
          <circle cx="80" cy="80" r="8" fill="#44cc55"/>
          <circle cx="100" cy="80" r="6" fill="#44cc55"/>
        </svg>'''
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, svg)
            result = audit_designer_operations(path)
        operation = _by_id(result)["native-main-outer-ring"]
        self.assertEqual(operation["status"], "partial")
        self.assertEqual(operation["evidence"]["native_ring_candidates"], [])
        self.assertEqual(operation["evidence"]["large_non_ring_circles"][0]["id"],
                         "large-disc")

    def test_absent_gradient_requires_review_instead_of_false_failure(self):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
               '<rect x="10" y="10" width="30" height="30" fill="#123456"/>'
               '</svg>')
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, svg)
            result = audit_designer_operations(path)
        operation = _by_id(result)["native-gradient-resource"]
        self.assertEqual(operation["status"], "manual_review")
        self.assertFalse(operation["automatable"])

    def test_sidecar_files_are_discovered_without_filename_assumptions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self._write(folder, _complete_svg(), "arbitrary-name.svg")
            (Path(folder) / "paint_roles.json").write_text(
                json.dumps(_manifest()), encoding="utf-8")
            result = audit_designer_operations(path)
        colour = _by_id(result)["global-colour-role"]
        self.assertEqual(colour["status"], "passed")
        self.assertTrue(colour["evidence"]["manifest_source"].endswith(
            "paint_roles.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
