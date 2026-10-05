"""Unit tests for the independent SVG editability audit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editability_audit import AUDIT_SCHEMA, audit_editability  # noqa: E402


class EditabilityAuditTests(unittest.TestCase):
    def test_single_visible_circle_and_gradient_are_direct_selection_targets(self):
        for fill, defs in [("#24604c", ""), ("url(#radial)",
                '<defs><radialGradient id="radial"><stop stop-color="red"/>'
                '<stop offset="1" stop-color="yellow"/></radialGradient></defs>')]:
            with self.subTest(fill=fill), tempfile.TemporaryDirectory() as raw:
                svg = self.write_svg(Path(raw), defs +
                    f'<g id="green" fill="{fill}" data-paint-role-fill="paint-1">'
                    '<circle id="single-circle" cx="50" cy="50" r="25"/></g>')
                result = audit_editability(svg)
                details = result['editability_details']
                proof = details['direct_single_drawable_selection']
                self.assertTrue(proof['available'])
                self.assertEqual(proof['basis'], 'single_identifiable_visible_drawable')
                self.assertFalse(proof['semantic_group_created'])
                self.assertEqual(details['semantic_group_coverage'], 0)
                self.assertEqual(details['effective_selection_coverage'], 1)
                self.assertEqual(result['automation_readiness']['components']['semantic_selection'], 25)
                self.assertEqual(result['status'], 'accepted')

    def test_single_selection_credit_does_not_cover_multiple_unidentified_or_hidden_objects(self):
        cases = [
            '<circle id="a" r="10"/><circle id="b" cx="30" r="10"/>',
            '<circle r="10"/>',
            '<circle id="same" r="10"/><metadata id="same"/>',
            '<circle id="a" r="10" display="none"/>',
            '<circle id="a" r="10" opacity="0"/>',
            '<circle id="a" r="10" transform="scale(0)"/>',
            '<circle id="a" r="0"/>',
            '<path id="a" d="M0 0 L10 0 L10 10 Z M20 0 L30 0 L30 10 Z"/>',
        ]
        for body in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as raw:
                svg = self.write_svg(Path(raw), '<g id="green" fill="#24604c">' + body + '</g>')
                result = audit_editability(svg)
                self.assertFalse(result['editability_details']['direct_single_drawable_selection']['available'])
                self.assertEqual(result['automation_readiness']['components']['semantic_selection'], 0)
                self.assertIn('Groups separate paint layers only; no semantic object grouping was detected.', result['reasons'])

    def write_svg(self, directory: Path, body: str) -> Path:
        path = directory / "fixture.svg"
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            'viewBox="0 0 100 100">' + body + "</svg>",
            encoding="utf-8",
        )
        return path

    def compound_path_data(self, *, subpaths: int = 50,
                           extra_commands: int = 0,
                           x_offset: int = 0) -> str:
        parts = []
        for index in range(subpaths):
            x = x_offset + (index % 10) * 3
            y = (index // 10) * 3
            commands = [
                f"M{x} {y}",
                f"L{x + 1} {y}",
                f"L{x + 1} {y + 1}",
            ]
            if index == 0:
                commands.extend(
                    f"L{1 + step % 90} {1 + (step * 7) % 90}"
                    for step in range(extra_commands)
                )
            commands.append("Z")
            parts.append(" ".join(commands))
        return " ".join(parts)

    def gradient_compound_body(self, *, fill_id: str = "field-gradient",
                               extra_commands: int = 0,
                               include_uncertified: bool = False,
                               duplicate_path_id: bool = False) -> str:
        certified = self.compound_path_data(
            subpaths=50, extra_commands=extra_commands)
        uncertified = ""
        if include_uncertified:
            second = self.compound_path_data(subpaths=50, x_offset=40)
            uncertified = (
                f'<path id="uncertified-path" fill="#345678" '
                f'data-paint-role-fill="field-role" d="{second}"/>'
            )
        duplicate = ""
        if duplicate_path_id:
            duplicate = (
                '<path id="field-path" fill="url(#field-gradient)" '
                'data-avc-gradient-object="field-object" '
                'd="M80 80 L81 80 L81 81 Z"/>'
            )
        return f'''
          <defs>
            <linearGradient id="field-gradient">
              <stop offset="0" stop-color="#9adbea"/>
              <stop offset="1" stop-color="#083c24"/>
            </linearGradient>
            <linearGradient id="other-gradient">
              <stop offset="0" stop-color="#ffffff"/>
              <stop offset="1" stop-color="#000000"/>
            </linearGradient>
          </defs>
          <g id="field-group" data-group-mode="actual-dom"
             data-group-reasons="source-field-ownership">
            <path id="field-path" fill="url(#{fill_id})"
                  fill-rule="evenodd"
                  data-avc-gradient-object="field-object"
                  data-paint-role-fill="field-role" d="{certified}"/>
            {uncertified}
            {duplicate}
          </g>
        '''

    def exact_gradient_certificate(self, svg: Path) -> dict:
        return {
            "designer_quality": {
                "source": {
                    "sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
                },
                "gradient_object_gate": {
                    "source_space_field_evidence": {
                        "authoritative": True,
                        "objects": [{
                            "candidate_id": "field-candidate",
                            "gradient_id": "field-gradient",
                            "passed": True,
                            "economy_certificate_passed": True,
                            "final_element": "path",
                            "final_drawable_id": "field-path",
                            "gradient_object_id": "field-object",
                        }],
                    },
                },
                "curve_economy_gate": {
                    "optimizer_economy_evidence": {
                        "gradient": {
                            "authoritative": True,
                            "certified_path_ids": ["field-path"],
                        },
                    },
                },
            },
        }

    def exact_curve_certificate(
            self, svg: Path, *, refit_ids: list[str],
            retained_ids: list[str] | None = None,
            uncertified_ids: list[str] | None = None) -> dict:
        retained = list(retained_ids or [])
        refit = list(refit_ids)
        certified = sorted(refit + retained)
        return {
            "designer_quality": {
                "source": {
                    "sha256": hashlib.sha256(svg.read_bytes()).hexdigest(),
                },
                "curve_economy_gate": {
                    "optimizer_economy_evidence": {
                        "curve_refit": {
                            "available": True,
                            "authoritative": True,
                            "transaction_status": "committed",
                            "proposal_schema": (
                                "ai-vector-cleanroom.curve-refit-proposal/v3"),
                            "proposal_status": "proposed",
                            "certified_refit_path_ids": refit,
                            "certified_retained_identity_path_ids": retained,
                            "certified_path_ids": certified,
                            "certified_path_count": len(certified),
                            "uncertified_evaluation_ids": list(
                                uncertified_ids or []),
                            "failure_reasons": [],
                            "invalid_detail_ids": [],
                            "scope": (
                                "transaction_backed_geometry_only_optimizer_"
                                "economy_certificate"),
                        },
                    },
                },
            },
        }

    def long_line_path_data(self, draw_commands: int) -> str:
        return "M0 0 " + " ".join(
            f"L{index % 100} {(index * 7) % 100}"
            for index in range(draw_commands)
        ) + " Z"

    def test_simple_semantic_svg_is_accepted_and_json_serializable(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(directory, """
              <defs>
                <linearGradient id="brand-gradient">
                  <stop offset="0" stop-color="#00ff00"/>
                  <stop offset="1" stop-color="#008800"/>
                </linearGradient>
              </defs>
              <g id="logo-mark" inkscape:label="Logo mark">
                <rect id="tile" x="2" y="2" width="96" height="96" fill="#fff"/>
                <path id="wave" fill="url(#brand-gradient)"
                      d="M10 55 C30 20 60 20 90 55 Z"/>
                <path id="accent" fill="#f00" d="M20 70 L80 70 L50 90 Z"/>
              </g>
            """)
            result = audit_editability(svg, {
                "paths": 2,
                "native_primitives": 1,
                "strokes": 0,
                "gradients": 1,
                "nodes_total": 9,
            })

        details = result["editability_details"]
        self.assertEqual(result["status"], "accepted")
        self.assertGreaterEqual(result["score"], 75)
        self.assertEqual(details["path_count"], 2)
        self.assertEqual(details["native_primitive_count"], 1)
        self.assertEqual(details["gradient_resource_count"], 1)
        self.assertEqual(details["group_count"], 1)
        self.assertEqual(details["semantic_group_count"], 1)
        self.assertEqual(details["unique_solid_paint_count"], 2)
        self.assertEqual(details["object_id_count"], 3)
        self.assertTrue(details["has_object_ids"])
        self.assertFalse(details["only_color_layers_without_semantic_groups"])
        self.assertIn("does not prove an 80%", details["scope_note"])
        self.assertEqual(result["schema"], AUDIT_SCHEMA)
        self.assertEqual(result["audit_model"], "layered-v2")
        self.assertEqual(result["score_axis"], "redraw_ease")
        self.assertEqual(result["status_scope"], "structural_editability_gate")
        self.assertEqual(result["automation_readiness"]["status"],
                         "ready_for_common_operations")
        self.assertEqual(
            result["automation_readiness"]["evidence_class"],
            "generic_structural_heuristic",
        )
        self.assertFalse(
            result["automation_readiness"]["score_is_operation_pass_count"])
        self.assertEqual(result["redraw_complexity"]["level"], "low")
        self.assertEqual(result["workflow_friction"]["level"], "low")
        self.assertTrue(result["acceptance_gate"]["passed"])
        self.assertEqual(result["human_validation"]["status"], "not_performed")
        self.assertIsNone(
            result["human_validation"]["original_human_tasks_passed"])
        json.dumps(result, ensure_ascii=False)

    def test_subpaths_and_implicit_commands_are_counted(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(directory, """
              <g id="symbol">
                <path id="compound" fill="#123456"
                      d="M0 0 L10 0 10 10 Z M2 2 L3 3 Z"/>
                <path id="curve" fill="#abcdef" d="M20 20 C30 0 40 0 50 20 Z"/>
              </g>
            """)
            result = audit_editability(svg, {"nodes_total": 8})

        details = result["editability_details"]
        self.assertEqual(details["total_subpaths"], 3)
        self.assertEqual(details["multi_subpath_path_count"], 1)
        self.assertEqual(details["max_subpaths_per_path"], 2)
        self.assertEqual(details["path_command_count_max"], 7)
        self.assertEqual(details["total_path_commands"], 10)
        self.assertEqual(details["max_path_command_share"], 0.7)
        self.assertEqual(details["explicit_bezier_control_point_count"], 2)

    def test_generic_fifty_subpath_compound_stays_manual_review(self):
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw), self.gradient_compound_body())
            result = audit_editability(svg)

        details = result["editability_details"]
        self.assertEqual(details["max_subpaths_per_path"], 50)
        self.assertEqual(details["generic_coupling_max_subpaths"], 50)
        self.assertEqual(
            details["source_topology_certified_compound_path_ids"], [])
        self.assertFalse(
            details["source_topology_gradient_evidence"]["available"])
        self.assertIn(
            "one_path_at_least_50_subpaths", details["review_triggers"])
        self.assertEqual(result["status"], "manual_review")
        self.assertFalse(result["acceptance_gate"]["passed"])

    def test_exact_gradient_certificate_scopes_only_generic_subpath_trigger(self):
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw), self.gradient_compound_body())
            result = audit_editability(
                svg, self.exact_gradient_certificate(svg))

        details = result["editability_details"]
        evidence = details["source_topology_gradient_evidence"]
        self.assertEqual(details["max_subpaths_per_path"], 50)
        self.assertEqual(details["generic_coupling_max_subpaths"], 0)
        self.assertEqual(
            details["source_topology_certified_compound_path_ids"],
            ["field-path"],
        )
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["authoritative"], evidence)
        self.assertEqual(evidence["certified_path_ids"], ["field-path"])
        self.assertNotIn(
            "one_path_at_least_50_subpaths", details["review_triggers"])
        self.assertEqual(result["status"], "accepted", result)
        self.assertTrue(result["acceptance_gate"]["passed"])
        self.assertEqual(result["score_axis"], "redraw_ease")
        self.assertEqual(
            result["score"], result["redraw_complexity"]["ease_score"])
        self.assertTrue(any(
            "source-topology-certified gradient compound" in reason
            for reason in result["reasons"]
        ))
        human = result["human_validation"]
        self.assertEqual(human["status"], "not_performed")
        self.assertFalse(human["timed_editing_test_performed"])
        self.assertIsNone(human["designer_acceptance"])
        self.assertIsNone(human["original_human_tasks_passed"])
        self.assertIsNone(human["original_human_tasks_total"])
        self.assertIn("does not prove an 80%", details["scope_note"])
        self.assertIn("timed human editing is required", details["scope_note"])

    def test_gradient_compound_certificate_fields_fail_closed(self):
        cases = (
            "sha256",
            "final_path_id",
            "gradient_owner",
            "gradient_fill",
            "duplicate_path_id",
            "source_space_authoritative",
            "gradient_optimizer_authoritative",
            "gradient_optimizer_certified_ids",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                fill_id = (
                    "other-gradient" if case == "gradient_fill"
                    else "field-gradient")
                svg = self.write_svg(
                    Path(raw), self.gradient_compound_body(
                        fill_id=fill_id,
                        duplicate_path_id=(case == "duplicate_path_id"),
                    ))
                report = self.exact_gradient_certificate(svg)
                quality = report["designer_quality"]
                source_fields = quality["gradient_object_gate"][
                    "source_space_field_evidence"]
                gradient_object = source_fields["objects"][0]
                optimizer = quality["curve_economy_gate"][
                    "optimizer_economy_evidence"]["gradient"]
                if case == "sha256":
                    quality["source"]["sha256"] = "0" * 64
                elif case == "final_path_id":
                    gradient_object["final_drawable_id"] = "different-path"
                elif case == "gradient_owner":
                    gradient_object["gradient_object_id"] = "different-owner"
                elif case == "source_space_authoritative":
                    source_fields["authoritative"] = False
                elif case == "gradient_optimizer_authoritative":
                    optimizer["authoritative"] = False
                elif case == "gradient_optimizer_certified_ids":
                    optimizer["certified_path_ids"] = ["different-path"]

                result = audit_editability(svg, copy.deepcopy(report))

            details = result["editability_details"]
            evidence = details["source_topology_gradient_evidence"]
            self.assertFalse(evidence["authoritative"], evidence)
            self.assertEqual(evidence["certified_path_ids"], [])
            self.assertEqual(details["generic_coupling_max_subpaths"], 50)
            self.assertIn(
                "one_path_at_least_50_subpaths", details["review_triggers"])
            self.assertEqual(result["status"], "manual_review")

    def test_uncertified_fifty_subpath_peer_keeps_manual_review(self):
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw), self.gradient_compound_body(
                    include_uncertified=True))
            result = audit_editability(
                svg, self.exact_gradient_certificate(svg))

        details = result["editability_details"]
        self.assertEqual(details["max_subpaths_per_path"], 50)
        self.assertEqual(details["generic_coupling_max_subpaths"], 50)
        self.assertEqual(
            details["source_topology_certified_compound_path_ids"],
            ["field-path"],
        )
        self.assertIn(
            "one_path_at_least_50_subpaths", details["review_triggers"])
        self.assertEqual(result["status"], "manual_review")

    def test_exact_gradient_certificate_preserves_raw_500_command_trigger(self):
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw), self.gradient_compound_body(extra_commands=300))
            result = audit_editability(
                svg, self.exact_gradient_certificate(svg))

        details = result["editability_details"]
        self.assertEqual(details["max_subpaths_per_path"], 50)
        self.assertEqual(details["generic_coupling_max_subpaths"], 0)
        self.assertGreaterEqual(details["path_command_count_max"], 500)
        self.assertNotIn(
            "one_path_at_least_50_subpaths", details["review_triggers"])
        self.assertNotIn(
            "one_path_at_least_500_commands", details["review_triggers"])
        self.assertIn(
            "one_path_at_least_500_commands", details["raw_review_triggers"])
        self.assertEqual(
            details["generic_uncertified_redraw_scope"][
                "max_commands_in_one_path"], 0)
        self.assertEqual(result["status"], "accepted")

    def test_exact_curve_certificate_scopes_formal_gate_but_keeps_raw_totals(self):
        giant = self.long_line_path_data(4100)
        body = (
            '<g id="logo" data-group-mode="actual-dom" '
            'data-group-reasons="one-object">'
            f'<path id="certified-giant" fill="#123456" '
            'data-paint-role-fill="accent" '
            'data-avc-curve-refit="geometry-budgeted" '
            f'd="{giant}"/>'
            '<path id="uncertified-small" fill="#123456" '
            'data-paint-role-fill="accent" d="M1 1 L2 1 L2 2 Z"/>'
            '</g>'
        )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), body)
            report = self.exact_curve_certificate(
                svg, refit_ids=["certified-giant"],
                uncertified_ids=["uncertified-small"])
            result = audit_editability(svg, report)

        details = result["editability_details"]
        raw_scope = details["raw_scope_metrics"]
        generic = details["generic_uncertified_redraw_scope"]
        certified = details["certified_optimizer_scope"]
        self.assertTrue(certified["curve_evidence"]["authoritative"], certified)
        self.assertEqual(certified["path_ids"], ["certified-giant"])
        self.assertGreaterEqual(raw_scope["node_count"], 4000)
        self.assertGreaterEqual(raw_scope["max_commands_in_one_path"], 500)
        self.assertIn("node_count_at_least_4000",
                      raw_scope["review_triggers"])
        self.assertIn("one_path_at_least_500_commands",
                      raw_scope["review_triggers"])
        self.assertEqual(generic["path_count"], 1)
        self.assertEqual(generic["svg_path_anchor_count"], 3)
        self.assertEqual(generic["max_commands_in_one_path"], 4)
        self.assertEqual(generic["review_triggers"], [])
        self.assertEqual(result["status"], "accepted", result)

    def test_giant_uncertified_peer_still_fails_without_threshold_changes(self):
        giant = self.long_line_path_data(4100)
        body = (
            '<g id="logo" data-group-mode="actual-dom" '
            'data-group-reasons="one-object">'
            '<path id="certified-small" fill="#123456" '
            'data-paint-role-fill="accent" '
            'data-avc-curve-refit="geometry-budgeted" '
            'd="M1 1 L2 1 L2 2 Z"/>'
            f'<path id="uncertified-giant" fill="#123456" '
            'data-paint-role-fill="accent" '
            f'd="{giant}"/>'
            '</g>'
        )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), body)
            report = self.exact_curve_certificate(
                svg, refit_ids=["certified-small"],
                uncertified_ids=["uncertified-giant"])
            result = audit_editability(svg, report)

        generic = result["editability_details"][
            "generic_uncertified_redraw_scope"]
        self.assertGreaterEqual(generic["node_count"], 4000)
        self.assertGreaterEqual(generic["max_commands_in_one_path"], 500)
        self.assertIn("node_count_at_least_4000",
                      generic["review_triggers"])
        self.assertIn("one_path_at_least_500_commands",
                      generic["review_triggers"])
        self.assertEqual(result["status"], "manual_review")
        self.assertFalse(result["acceptance_gate"]["passed"])

    def test_curve_certificate_integrity_failures_exclude_nothing(self):
        giant = self.long_line_path_data(650)
        cases = (
            "sha256",
            "authoritative",
            "transaction_status",
            "proposal_schema",
            "proposal_status",
            "failure_reasons",
            "invalid_detail_ids",
            "certified_union",
            "certified_count",
            "uncertified_overlap",
            "missing_final_id",
            "missing_refit_claim",
            "duplicate_final_id",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as raw:
                claim = "" if case == "missing_refit_claim" else (
                    'data-avc-curve-refit="geometry-budgeted" ')
                duplicate = (
                    '<path id="curve-path" fill="#123456" '
                    'data-avc-curve-refit="geometry-budgeted" '
                    'd="M3 3 L4 3 L4 4 Z"/>'
                    if case == "duplicate_final_id" else "")
                svg = self.write_svg(Path(raw), (
                    f'<path id="curve-path" fill="#123456" {claim}'
                    f'd="{giant}"/>{duplicate}'))
                report = self.exact_curve_certificate(
                    svg, refit_ids=["curve-path"])
                quality = report["designer_quality"]
                curve = quality["curve_economy_gate"][
                    "optimizer_economy_evidence"]["curve_refit"]
                if case == "sha256":
                    quality["source"]["sha256"] = "0" * 64
                elif case == "authoritative":
                    curve["authoritative"] = False
                elif case == "transaction_status":
                    curve["transaction_status"] = "rolled_back"
                elif case == "proposal_schema":
                    curve["proposal_schema"] = "wrong"
                elif case == "proposal_status":
                    curve["proposal_status"] = "no_change"
                elif case == "failure_reasons":
                    curve["failure_reasons"] = ["tampered"]
                elif case == "invalid_detail_ids":
                    curve["invalid_detail_ids"] = ["curve-path"]
                elif case == "certified_union":
                    curve["certified_path_ids"] = []
                elif case == "certified_count":
                    curve["certified_path_count"] = 2
                elif case == "uncertified_overlap":
                    curve["uncertified_evaluation_ids"] = ["curve-path"]
                elif case == "missing_final_id":
                    curve["certified_refit_path_ids"] = ["missing-path"]
                    curve["certified_path_ids"] = ["missing-path"]

                result = audit_editability(svg, copy.deepcopy(report))

            details = result["editability_details"]
            evidence = details["curve_optimizer_certificate_evidence"]
            self.assertFalse(evidence["authoritative"], evidence)
            self.assertEqual(evidence["certified_path_ids"], [])
            self.assertEqual(
                details["certified_optimizer_scope"]["path_count"], 0)
            self.assertGreaterEqual(
                details["generic_uncertified_redraw_scope"][
                    "max_commands_in_one_path"], 500)
            self.assertIn("one_path_at_least_500_commands",
                          details["review_triggers"])
            self.assertEqual(result["status"], "manual_review")

    def test_optional_three_refit_fixture_retains_generic_failure(self):
        diagnostic_root = (
            ROOT / "tests" / "optional_fixtures" / "curve_refit_legacy"
        )
        svg = diagnostic_root / "vector.svg"
        diagnostic = diagnostic_root / "curve_diagnostic.json"
        if not svg.is_file() or not diagnostic.is_file():
            self.skipTest("optional legacy curve diagnostic fixture is not included")
        self.assertEqual(
            hashlib.sha256(diagnostic.read_bytes()).hexdigest().upper(),
            "6182893639AFD6D7890B65CCD7196D77E9E05E8E81B5BC682C50B8B0B5A6B318",
        )
        self.assertEqual(
            hashlib.sha256(svg.read_bytes()).hexdigest().upper(),
            "746EF5E27B7BA619FD396C3398CF9429D87059F29EA1218E726FB9F732047279",
        )
        payload = json.loads(diagnostic.read_text(encoding="utf-8"))
        proposal = payload["transaction_full_evidence"]["proposal"]
        refit_ids = [item["id"] for item in proposal["details"]]
        retained_ids = [
            item["id"] for item in proposal["evaluated_but_retained"]]
        uncertified_ids = [
            item["id"] for item in proposal["uncertified_evaluations"]]
        report = {"designer_quality": copy.deepcopy(
            payload["designer_quality"])}
        curve = self.exact_curve_certificate(
            svg, refit_ids=refit_ids, retained_ids=retained_ids,
            uncertified_ids=uncertified_ids)["designer_quality"][
                "curve_economy_gate"]["optimizer_economy_evidence"][
                    "curve_refit"]
        report["designer_quality"]["curve_economy_gate"][
            "optimizer_economy_evidence"]["curve_refit"] = curve

        result = audit_editability(svg, report)

        details = result["editability_details"]
        certified = details["certified_optimizer_scope"]
        generic = details["generic_uncertified_redraw_scope"]
        self.assertTrue(certified["curve_evidence"]["authoritative"], certified)
        self.assertEqual(certified["curve_path_count"], 58)
        self.assertEqual(certified["gradient_path_count"], 2)
        self.assertEqual(certified["path_count"], 60)
        self.assertEqual(generic["path_count"], 113)
        self.assertEqual(generic["svg_path_anchor_count"], 4271)
        self.assertEqual(generic["command_count"], 4409)
        self.assertEqual(generic["max_commands_in_one_path"], 329)
        self.assertEqual(generic["max_subpaths_in_one_path"], 5)
        self.assertIn("node_count_at_least_4000",
                      generic["review_triggers"])
        self.assertNotIn("one_path_at_least_500_commands",
                         generic["review_triggers"])
        self.assertIn("one_path_at_least_500_commands",
                      details["raw_review_triggers"])
        self.assertEqual(result["status"], "manual_review")
        self.assertFalse(result["acceptance_gate"]["passed"])

    def test_bezier_control_handles_are_reported_separately_from_anchors(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            curved = self.write_svg(directory, (
                '<path id="p" fill="none" stroke="#111" '
                'd="M0 0 C3 0 7 0 10 0 C10 3 10 7 10 10"/>'))
            curved_result = audit_editability(curved)
            straight = directory / "straight.svg"
            straight.write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<path id="p" fill="none" stroke="#111" '
                'd="M0 0 L10 0 L10 10"/></svg>', encoding="utf-8")
            straight_result = audit_editability(straight)

        curved_details = curved_result["editability_details"]
        straight_details = straight_result["editability_details"]
        self.assertEqual(curved_details["total_path_commands"], 3)
        self.assertEqual(straight_details["total_path_commands"], 3)
        self.assertEqual(curved_details["explicit_bezier_control_point_count"], 4)
        self.assertEqual(straight_details["explicit_bezier_control_point_count"], 0)
        self.assertEqual(
            curved_details["outline_handle_count_estimate"]
            - straight_details["outline_handle_count_estimate"], 4)

    def test_report_json_path_supplies_engine_counts(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(
                directory,
                '<circle id="dot" cx="20" cy="20" r="10" fill="#000"/>',
            )
            report = directory / "counts.json"
            report.write_text(json.dumps({
                "native_circles": 1,
                "native_rectangles": 2,
                "native_lines": 3,
                "native_polylines": 4,
                "native_polygons": 5,
                "native_polygonal_shapes": 9,
                "strokes": 4,
                "gradients": 3,
                "nodes_total": 17,
            }), encoding="utf-8")
            result = audit_editability(svg, report)

        details = result["editability_details"]
        # The polygonal aggregate is informational and is not double counted.
        self.assertEqual(details["native_primitive_count"], 15)
        self.assertEqual(details["stroke_count"], 4)
        self.assertEqual(details["gradient_count"], 3)
        self.assertEqual(details["node_count"], 17)
        self.assertEqual(details["count_sources"]["nodes"], "report_or_stats")

    def test_stale_report_cannot_hide_svg_node_complexity(self):
        path_data = "M0 0 " + " ".join(
            f"L{index % 100} {(index * 3) % 100}" for index in range(600)
        ) + " Z"
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw), f'<path id="dense" fill="#000" d="{path_data}"/>')
            result = audit_editability(svg, {"nodes_total": 1})

        details = result["editability_details"]
        self.assertEqual(details["node_count"], details["svg_estimated_node_count"])
        self.assertGreater(details["node_count"], 1)
        self.assertEqual(
            details["count_sources"]["nodes"],
            "conservative_svg_estimate_over_report",
        )

    def test_structural_five_of_five_is_not_human_task_validation(self):
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(
                Path(raw),
                '<g id="mark"><circle id="dot" cx="20" cy="20" r="10" '
                'fill="#65ee22" data-paint-role-fill="accent-1"/></g>',
            )
            result = audit_editability(svg, {
                "designer_operations": {
                    "schema": "ai-vector-cleanroom.designer-operations/v1",
                    "summary": {"passed": 5, "total_operations": 5},
                },
            })

        evidence = result["named_operation_evidence"]
        self.assertEqual(
            evidence["status"], "reported_by_separate_structural_audit")
        self.assertEqual(evidence["structural_checks_passed"], 5)
        self.assertEqual(evidence["structural_checks_total"], 5)
        self.assertTrue(evidence["all_structural_checks_passed"])
        human = result["human_validation"]
        self.assertEqual(human["status"], "not_performed")
        self.assertIsNone(human["original_human_tasks_passed"])
        self.assertIsNone(human["original_human_tasks_total"])
        self.assertFalse(human["timed_editing_test_performed"])

    def test_complex_color_layer_fixture_requires_manual_review(self):
        # 318 paths, 8,310 reported nodes, one 854-command path, 37 colour
        # layers and no drawable IDs reproduce the structural risk profile of
        # a complex multi-colour graphic without embedding an external artifact.
        long_path = "M0 0 " + " ".join(
            f"L{index % 100} {(index * 7) % 100}" for index in range(852)
        ) + " Z"
        simple_paths = [
            f'<path d="M{i % 90} {i % 80} L{(i + 3) % 90} {(i + 5) % 80} Z"/>'
            for i in range(317)
        ]
        buckets = [[] for _ in range(37)]
        buckets[0].append(f'<path d="{long_path}"/>')
        for index, item in enumerate(simple_paths):
            buckets[index % len(buckets)].append(item)
        layers = []
        for index, items in enumerate(buckets, start=1):
            color = f"#{index:02x}{(index * 3) % 256:02x}{(index * 5) % 256:02x}"
            layers.append(
                f'<g id="color-layer-{index}" inkscape:groupmode="layer" '
                f'fill="{color}">' + "".join(items) + "</g>"
            )

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            svg = self.write_svg(directory, "".join(layers))
            result = audit_editability(svg, {
                "paths": 318,
                "native_primitives": 83,
                "strokes": 0,
                "gradients": 2,
                "nodes_total": 8310,
            })

        details = result["editability_details"]
        self.assertEqual(result["status"], "manual_review")
        self.assertLess(result["score"], 75)
        self.assertEqual(details["path_count"], 318)
        self.assertEqual(details["node_count"], 8310)
        self.assertEqual(details["group_count"], 37)
        self.assertEqual(details["path_command_count_max"], 854)
        self.assertFalse(details["has_object_ids"])
        self.assertTrue(details["only_color_layers_without_semantic_groups"])
        self.assertIn("path_count_at_least_200", details["review_triggers"])
        self.assertIn("node_count_at_least_4000", details["review_triggers"])
        self.assertIn("one_path_at_least_500_commands", details["review_triggers"])
        raw = details["risk_penalties"]
        families = details["applied_penalty_families"]
        self.assertEqual(
            families["geometry_volume"],
            max(raw["many_paths"], raw["many_nodes"]),
        )
        self.assertLess(
            families["geometry_volume"],
            raw["many_paths"] + raw["many_nodes"],
        )
        self.assertEqual(result["redraw_complexity"]["level"], "high")
        self.assertEqual(result["workflow_friction"]["level"], "very_high")
        self.assertEqual(result["automation_readiness"]["status"], "limited")
        json.dumps(result, ensure_ascii=False)

    def test_semantic_groups_are_selection_handles_not_a_navigation_penalty(self):
        groups = []
        for index in range(30):
            groups.append(
                f'<g id="object-{index}" data-group-mode="actual-dom" '
                f'data-group-reasons="cross-paint-overlay">'
                f'<path id="shape-{index}" fill="#123456" '
                f'd="M{index} 0 L{index + 1} 0 L{index + 1} 1 Z"/>'
                '</g>'
            )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), "".join(groups))
            result = audit_editability(svg, {"nodes_total": 90})

        details = result["editability_details"]
        self.assertEqual(details["actual_dom_group_count"], 30)
        self.assertEqual(details["semantic_group_coverage"], 1.0)
        self.assertNotIn("excessive_group_navigation", details["risk_penalties"])
        self.assertEqual(
            result["automation_readiness"]["components"]["semantic_selection"],
            25.0,
        )
        self.assertEqual(result["status"], "accepted")

    def test_inherited_paint_role_counts_each_controlled_drawable(self):
        body = (
            '<g id="accent-layer" fill="#65ee22" '
            'data-paint-role-fill="accent-1">'
            '<path id="shape-a" d="M0 0 L10 0 L10 10 Z"/>'
            '<circle id="shape-b" cx="20" cy="20" r="5"/>'
            '</g>'
        )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), body)
            result = audit_editability(svg)

        details = result["editability_details"]
        self.assertEqual(details["paint_role_count"], 1)
        self.assertEqual(details["paint_role_annotated_drawable_count"], 2)
        self.assertEqual(details["paint_role_annotation_coverage"], 1.0)

    def test_local_paint_override_without_role_clears_inherited_control(self):
        body = (
            '<g id="accent-layer" fill="#65ee22" '
            'data-paint-role-fill="accent-1">'
            '<path id="controlled" d="M0 0 L10 0 L10 10 Z"/>'
            '<path id="overridden" fill="#123456" '
            'd="M20 20 L30 20 L30 30 Z"/>'
            '</g>'
        )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), body)
            result = audit_editability(svg)

        details = result["editability_details"]
        self.assertEqual(details["paint_role_count"], 1)
        self.assertEqual(details["paint_role_annotated_drawable_count"], 1)
        self.assertEqual(details["paint_role_annotation_coverage"], 0.5)

    def test_texture_readiness_does_not_hide_a_giant_compound_outline(self):
        circles = "".join(
            f'<circle id="dot-{index}" cx="{index % 20}" cy="{index // 20}" '
            f'r=".3" fill="#65ee22" data-paint-role-fill="accent-1"/>'
            for index in range(80)
        )
        brush = "M0 20 " + " ".join(
            f"L{index % 100} {20 + (index * 7) % 30}" for index in range(620)
        ) + " Z"
        body = (
            '<g id="halftone" data-group-mode="actual-dom" '
            'data-group-reasons="repeated-dot-proximity">' + circles + '</g>'
            '<g id="brush-mark" data-group-mode="actual-dom" '
            'data-group-reasons="cross-paint-overlay">'
            f'<path id="brush" fill="#65ee22" data-paint-role-fill="accent-1" '
            f'd="{brush}"/></g>'
        )
        with tempfile.TemporaryDirectory() as raw:
            svg = self.write_svg(Path(raw), body)
            result = audit_editability(svg, {"nodes_total": 5000})

        details = result["editability_details"]
        self.assertEqual(result["automation_readiness"]["status"],
                         "ready_for_common_operations")
        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(result["redraw_complexity"]["level"], "high")
        self.assertIn("one_path_at_least_500_commands",
                      details["review_triggers"])
        self.assertIn("single_very_complex_path", details["risk_penalties"])
        self.assertIn("No visual-style or brush-texture discount",
                      details["penalty_combination"])

    def test_optional_legacy_fixture_explains_33_score_without_false_acceptance(self):
        run = ROOT / "tests" / "optional_fixtures" / "legacy_beta3"
        reports = list(run.rglob("report.json"))
        svgs = list(run.rglob("*_vector.svg"))
        if not reports or not svgs:
            self.skipTest("optional legacy editability artifact is not included")
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report.get("editability_score"), 33.2)

        result = audit_editability(svgs[0], report)
        details = result["editability_details"]
        self.assertEqual(result["status"], "manual_review")
        self.assertEqual(result["score"], 64.0)
        self.assertEqual(result["automation_readiness"]["score"], 84.6)
        self.assertEqual(result["automation_readiness"]["status"],
                         "ready_for_common_operations")
        self.assertEqual(result["redraw_complexity"], {
            "ease_score": 64.0,
            "burden_score": 36.0,
            "level": "high",
            "penalty_families": {
                "geometry_volume": 24.0,
                "local_reshape": 12.0,
            },
            "scope_note": (
                "Measures freeform point-level reshaping and cleanup burden in "
                "the generic uncertified path scope. Raw final-SVG complexity "
                "remains separately disclosed. Intentional brush edges remain "
                "real redraw complexity even when common automated operations "
                "pass."
            ),
        })
        self.assertEqual(details["actual_dom_group_count"], 25)
        self.assertEqual(details["paint_role_count"], 4)
        self.assertEqual(details["object_id_coverage"], 1.0)
        self.assertFalse(details["visual_style_discount_applied"])
        self.assertIn("one_path_at_least_50_subpaths",
                      details["review_triggers"])
        self.assertEqual(result["workflow_friction"], {
            "ease_score": 100.0,
            "burden_score": 0,
            "level": "low",
            "penalty_families": {},
            "scope_note": (
                "Measures group navigation, stable object identity and semantic "
                "structure friction. It is deliberately excluded from the "
                "freeform outline-cleanup score."
            ),
        })
        evidence = result["named_operation_evidence"]
        self.assertEqual(evidence["structural_checks_passed"], 5)
        self.assertEqual(evidence["structural_checks_total"], 5)
        self.assertTrue(evidence["all_structural_checks_passed"])
        self.assertEqual(result["human_validation"]["status"], "not_performed")
        self.assertIsNone(
            result["human_validation"]["original_human_tasks_passed"])
        gate = result["acceptance_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["observed"]["redraw_ease"], 64.0)
        self.assertEqual(gate["observed"]["automation_readiness"], 84.6)
        self.assertEqual(gate["observed"]["review_trigger_count"], 4)


if __name__ == "__main__":
    unittest.main()
