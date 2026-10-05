# -*- coding: utf-8 -*-
"""Transaction regressions for geometry-first curve refitting."""

import contextlib
import copy
import hashlib
import io
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vector_cleanroom import (
    _attempt_curve_refit_transaction,
    _curve_refit_candidate_identity_guard,
    _curve_refit_final_digest_records,
    _curve_refit_hybrid_bytes,
    _curve_refit_identity_normalized_bytes,
    _final_gradient_report_details,
    _gradient_geometry_digest,
    _gradient_geometry_snapshot,
    _verify_designer_gradient_evidence_join,
    build_arg_parser,
    validate_args,
    validate_svg_stage_renders,
)


ORIGINAL = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
    b'viewBox="0 0 100 100"><path id="old" d="M0 0L10 0L10 10Z"/>'
    b'</svg>'
)
CANDIDATE = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
    b'viewBox="0 0 100 100"><circle id="old" cx="5" cy="5" r="5" '
    b'data-avc-designer-anchors="4"/></svg>'
)

SUBSET_ORIGINAL = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
    b'viewBox="0 0 100 100">'
    b'<path id="large" d="M0 0L40 0L40 40Z"/>'
    b'<path id="medium" d="M50 0L70 0L70 20Z"/>'
    b'<path id="small" d="M80 0L90 0L90 10Z"/>'
    b'</svg>'
)
SUBSET_PATHS = {
    "large": "M0 0L40 0L40 39Z",
    "medium": "M50 0L70 0L70 19Z",
    "small": "M80 0L90 0L90 9Z",
}
SUBSET_COUNTS = {
    "large": (100, 10, 8),
    "medium": (20, 5, 2),
    "small": (10, 4, 1),
}
SUBSET_CANDIDATE = (
    b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
    b'viewBox="0 0 100 100">'
    b'<path id="large" d="M0 0L40 0L40 39Z" '
    b'data-avc-curve-refit="geometry-budgeted"/>'
    b'<path id="medium" d="M50 0L70 0L70 19Z" '
    b'data-avc-curve-refit="geometry-budgeted"/>'
    b'<path id="small" d="M80 0L90 0L90 9Z" '
    b'data-avc-curve-refit="geometry-budgeted"/>'
    b'</svg>'
)


def _proposal(candidate_bytes=CANDIDATE):
    def create(_source, candidate, **_kwargs):
        Path(candidate).write_bytes(candidate_bytes)
        source_bytes = Path(_source).read_bytes()
        source_root = ET.fromstring(source_bytes)
        source_paths = [item for item in source_root.iter()
                        if item.tag.rsplit("}", 1)[-1] == "path"]
        source_path = source_paths[0]
        identifier = source_path.get("id")
        candidate_root = ET.fromstring(candidate_bytes)
        candidate_matches = [item for item in candidate_root.iter()
                             if item.get("id") == identifier]
        if len(candidate_matches) != 1:
            candidate_matches = [item for item in candidate_root.iter()
                                 if item.tag.rsplit("}", 1)[-1] in {
                                     "path", "circle", "ellipse"}]
        from curve_refit_stage import _committed_geometry_evidence
        detail = {
            "id": identifier,
            "outcome": "committed_refit",
            "economy_certified": True,
            **_committed_geometry_evidence(candidate_matches[0]),
            "anchors_before": 23,
            "anchors_after": 3,
            "designer_anchors_before": 23,
            "designer_anchors_after": 3,
        }
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        source_d = source_path.get("d") or ""
        stable_record = {
            "source_svg_sha256": source_sha,
            "global_path_ordinal_1_based": 1,
            "source_element": "path",
            "source_path_data_sha256": hashlib.sha256(
                source_d.encode("utf-8")).hexdigest(),
            "source_path_data_digest_scope": "utf8_svg_path_d_attribute",
            "original_id_state": "existing_preserved",
            "original_id": identifier,
            "assigned_id": identifier,
            "assignment_applied": False,
        }
        return {
            "schema": "ai-vector-cleanroom.curve-refit-proposal/v3",
            "status": "proposed",
            "path_count_refit": 1,
            "eligible_path_count": 1,
            "optimizer_evaluated_path_count": 1,
            "retained_identity_path_count": 0,
            "anchors_before": 23,
            "anchors_after": 3,
            "anchors_removed": 20,
            "designer_anchors_before": 23,
            "designer_anchors_after": 3,
            "designer_anchors_removed": 20,
            "error_budget_percent": 0.25,
            "actual_p95_error_percent": 0.18,
            "actual_max_error_percent": 0.52,
            "over_budget_share": 0.02,
            "salient_corner_max_percent": 0.20,
            "optimization_basis": "geometry_only",
            "uses_colour_or_pixel_similarity": False,
            "details": [detail],
            "evaluated_but_retained": [],
            "uncertified_evaluations": [],
            "stable_id_normalization": {
                "schema": (
                    "ai-vector-cleanroom.curve-refit-stable-id-"
                    "normalization/v1"),
                "source_svg_sha256": source_sha,
                "source_svg_digest_scope": "exact_source_svg_bytes",
                "source_path_count": len(source_paths),
                "optimizer_evaluated_path_count": 1,
                "record_count": 1,
                "assigned_id_count": 0,
                "existing_id_preserved_count": 1,
                "all_optimizer_evaluations_authenticated": True,
                "assigned_ids_unique": True,
                "candidate_all_svg_ids_unique": True,
                "candidate_svg_sha256": hashlib.sha256(
                    candidate_bytes).hexdigest(),
                "records": [stable_record],
            },
            "evaluation_evidence_integrity": {
                "schema": (
                    "ai-vector-cleanroom.curve-refit-evaluation-evidence/v1"),
                "optimizer_evaluated_path_count": 1,
                "committed_detail_count": 1,
                "retained_identity_detail_count": 0,
                "uncertified_evaluation_count": 0,
                "accounted_evaluation_count": 1,
                "all_optimizer_results_accounted": True,
                "evidence_ids_unique": True,
                "committed_and_retained_ids_disjoint": True,
            },
        }
    return create


def _subset_detail(identifier, before, after, loops, path_data):
    return {
        "id": identifier,
        "outcome": "committed_refit",
        "economy_certified": True,
        "final_element": "path",
        "final_drawable_id": identifier,
        "path_data_sha256": hashlib.sha256(
            path_data.encode("utf-8")).hexdigest(),
        "path_data_digest_scope": "utf8_svg_path_d_attribute",
        "anchors_before": before,
        "anchors_after": after,
        "designer_anchors_before": before,
        "designer_anchors_after": after,
        "error_budget_percent": 0.25,
        "actual_p95_error_percent": 0.10,
        "actual_max_error_percent": 0.30,
        "over_budget_share": 0.0,
        "salient_corner_max_percent": 0.20,
        "loops": loops,
    }


def _subset_proposal(candidate_bytes=SUBSET_CANDIDATE, details=None):
    if details is None:
        details = [
            _subset_detail(identifier, *SUBSET_COUNTS[identifier],
                           SUBSET_PATHS[identifier])
            for identifier in ("large", "medium", "small")
        ]

    def create(_source, candidate, **_kwargs):
        Path(candidate).write_bytes(candidate_bytes)
        source_bytes = Path(_source).read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        source_root = ET.fromstring(source_bytes)
        source_paths = [item for item in source_root.iter()
                        if item.tag.rsplit("}", 1)[-1] == "path"]
        path_by_id = {item.get("id"): (index, item)
                      for index, item in enumerate(source_paths, 1)}
        stable_records = []
        for item in details:
            identifier = str(item["id"])
            ordinal, source_path = path_by_id[identifier]
            source_d = source_path.get("d") or ""
            stable_records.append({
                "source_svg_sha256": source_sha,
                "global_path_ordinal_1_based": ordinal,
                "source_element": "path",
                "source_path_data_sha256": hashlib.sha256(
                    source_d.encode("utf-8")).hexdigest(),
                "source_path_data_digest_scope": (
                    "utf8_svg_path_d_attribute"),
                "original_id_state": "existing_preserved",
                "original_id": identifier,
                "assigned_id": identifier,
                "assignment_applied": False,
            })
        anchors_before = sum(item["anchors_before"] for item in details)
        anchors_after = sum(item["anchors_after"] for item in details)
        return {
            "schema": "ai-vector-cleanroom.curve-refit-proposal/v3",
            "status": "proposed",
            "source": Path(_source).name,
            "candidate": Path(candidate).name,
            "path_count_refit": len(details),
            "eligible_path_count": len(details),
            "optimizer_evaluated_path_count": len(details),
            "retained_identity_path_count": 0,
            "anchors_before": anchors_before,
            "anchors_after": anchors_after,
            "anchors_removed": anchors_before - anchors_after,
            "anchor_reduction_ratio": round(
                (anchors_before - anchors_after) / anchors_before, 6),
            "designer_anchors_before": anchors_before,
            "designer_anchors_after": anchors_after,
            "designer_anchors_removed": anchors_before - anchors_after,
            "designer_anchor_reduction_ratio": round(
                (anchors_before - anchors_after) / anchors_before, 6),
            "optimization_basis": "geometry_only",
            "uses_colour_or_pixel_similarity": False,
            "error_budget_percent": 0.25,
            "details": [dict(item) for item in details],
            "evaluated_but_retained": [],
            "uncertified_evaluations": [],
            "stable_id_normalization": {
                "schema": (
                    "ai-vector-cleanroom.curve-refit-stable-id-"
                    "normalization/v1"),
                "source_svg_sha256": source_sha,
                "source_svg_digest_scope": "exact_source_svg_bytes",
                "source_path_count": len(source_paths),
                "optimizer_evaluated_path_count": len(details),
                "record_count": len(stable_records),
                "assigned_id_count": 0,
                "existing_id_preserved_count": len(stable_records),
                "all_optimizer_evaluations_authenticated": True,
                "assigned_ids_unique": True,
                "candidate_all_svg_ids_unique": True,
                "candidate_svg_sha256": hashlib.sha256(
                    candidate_bytes).hexdigest(),
                "records": stable_records,
            },
            "protected_gradient_objects": {
                "ownership_mask_revalidation_performed": False,
                "skipped_path_count": 0,
            },
            "evaluation_evidence_integrity": {
                "schema": (
                    "ai-vector-cleanroom.curve-refit-evaluation-evidence/v1"),
                "optimizer_evaluated_path_count": len(details),
                "committed_detail_count": len(details),
                "retained_identity_detail_count": 0,
                "uncertified_evaluation_count": 0,
                "accounted_evaluation_count": len(details),
                "all_optimizer_results_accounted": True,
                "evidence_ids_unique": True,
                "committed_and_retained_ids_disjoint": True,
            },
        }
    return create


def _render_guard(*, recall=100.0, precision=100.0, f1=100.0,
                  accepted=True, colour=100.0):
    return {
        "accepted": accepted,
        "external_render_check": "completed",
        "alpha_topology": {"accepted": True},
        "composed_alpha": {"external_render_check": "completed", "accepted": True},
        "ink_recall_percent": recall,
        "ink_precision_percent": precision,
        "ink_f1_percent": f1,
        "color_similarity_percent": colour,
        "score_percent": colour,
    }


def _source_scores(*, recall=100.0, precision=100.0, f1=100.0,
                   render_ink=1000, flat=100.0, source=100.0,
                   foreground=100.0, colour=100.0):
    return {
        "flat": flat,
        "source": source,
        "foreground": foreground,
        "foreground_color_fidelity": colour,
        "foreground_recall": recall,
        "foreground_precision": precision,
        "foreground_coverage_f1": f1,
        "source_ink_pixels": 1000,
        "render_ink_pixels": render_ink,
    }


def _idless_proposal(*, status="no_change", replacement=None,
                     mutate_candidate=None):
    def create(source, candidate, **_kwargs):
        source_bytes = Path(source).read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        root = ET.fromstring(source_bytes)
        paths = [item for item in root.iter()
                 if item.tag.rsplit("}", 1)[-1] == "path"]
        element = paths[0]
        source_d = element.get("d") or ""
        source_d_sha = hashlib.sha256(source_d.encode("utf-8")).hexdigest()
        identifier = f"avc-refit-path-1-{source_d_sha[:12]}"
        element.set("id", identifier)
        details = []
        retained = []
        if status == "proposed":
            element.set("d", replacement)
            element.set("data-avc-curve-refit", "geometry-budgeted")
            element.set("data-avc-anchors-before", "4")
            element.set("data-avc-anchors-after", "3")
            element.set("data-avc-designer-anchors", "3")
            element.set("data-avc-error-budget-percent", "0.25")
            element.set("data-avc-p95-error-percent", "0.1")
            element.set("data-avc-max-error-percent", "0.2")
            from curve_refit_stage import _committed_geometry_evidence
            details = [{
                "id": identifier,
                "outcome": "committed_refit",
                "economy_certified": True,
                **_committed_geometry_evidence(element),
                "anchors_before": 4,
                "anchors_after": 3,
                "designer_anchors_before": 4,
                "designer_anchors_after": 3,
                "error_budget_percent": 0.25,
                "actual_p95_error_percent": 0.1,
                "actual_max_error_percent": 0.2,
                "over_budget_share": 0.0,
                "salient_corner_max_percent": 0.1,
                "loops": 1,
            }]
        else:
            retained = [{"id": identifier}]
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        output = io.BytesIO()
        ET.ElementTree(root).write(
            output, encoding="utf-8", xml_declaration=True)
        candidate_bytes = output.getvalue()
        stable_record = {
            "source_svg_sha256": source_sha,
            "global_path_ordinal_1_based": 1,
            "source_element": "path",
            "source_path_data_sha256": source_d_sha,
            "source_path_data_digest_scope": "utf8_svg_path_d_attribute",
            "original_id_state": "missing_assigned",
            "original_id": None,
            "assigned_id": identifier,
            "assignment_applied": True,
        }
        proposal = {
            "schema": "ai-vector-cleanroom.curve-refit-proposal/v3",
            "status": status,
            "source": Path(source).name,
            "candidate": Path(candidate).name,
            "path_count_refit": len(details),
            "eligible_path_count": 1,
            "optimizer_evaluated_path_count": 1,
            "retained_identity_path_count": len(retained),
            "anchors_before": 4,
            "anchors_after": 3 if details else 4,
            "anchors_removed": 1 if details else 0,
            "designer_anchors_before": 4,
            "designer_anchors_after": 3 if details else 4,
            "designer_anchors_removed": 1 if details else 0,
            "optimization_basis": "geometry_only",
            "uses_colour_or_pixel_similarity": False,
            "error_budget_percent": 0.25,
            "details": details,
            "evaluated_but_retained": retained,
            "uncertified_evaluations": [],
            "stable_id_normalization": {
                "schema": (
                    "ai-vector-cleanroom.curve-refit-stable-id-"
                    "normalization/v1"),
                "source_svg_sha256": source_sha,
                "source_svg_digest_scope": "exact_source_svg_bytes",
                "source_path_count": 1,
                "optimizer_evaluated_path_count": 1,
                "record_count": 1,
                "assigned_id_count": 1,
                "existing_id_preserved_count": 0,
                "all_optimizer_evaluations_authenticated": True,
                "assigned_ids_unique": True,
                "candidate_all_svg_ids_unique": True,
                "candidate_svg_sha256": hashlib.sha256(
                    candidate_bytes).hexdigest(),
                "records": [stable_record],
            },
            "evaluation_evidence_integrity": {
                "schema": (
                    "ai-vector-cleanroom.curve-refit-evaluation-evidence/v1"),
                "optimizer_evaluated_path_count": 1,
                "committed_detail_count": len(details),
                "retained_identity_detail_count": len(retained),
                "uncertified_evaluation_count": 0,
                "accounted_evaluation_count": 1,
                "all_optimizer_results_accounted": True,
                "evidence_ids_unique": True,
                "committed_and_retained_ids_disjoint": True,
            },
            "protected_gradient_objects": {
                "ownership_mask_revalidation_performed": False,
                "skipped_path_count": 0,
            },
        }
        written = (mutate_candidate(candidate_bytes, proposal)
                   if mutate_candidate else candidate_bytes)
        Path(candidate).write_bytes(written)
        return proposal
    return create


class CurveRefitTransactionTests(unittest.TestCase):
    def test_real_white_neighbours_cannot_merge_during_main_curve_transaction(self):
        import math
        from svg_renderer import render_svg_reference
        points = [(20 + 10 * math.cos((index + .5) * math.tau / 32),
                   60 + 30 * math.sin((index + .5) * math.tau / 32))
                  for index in range(32)]
        data = 'M' + ' L'.join(f'{x:.8f} {y:.8f}' for x, y in points) + ' Z'
        with tempfile.TemporaryDirectory() as directory:
            svg, source = Path(directory) / 'white.svg', Path(directory) / 'source.png'
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
                '<rect id="fixed" x="10" y="10" width="100" height="20.1" fill="white"/>'
                f'<path id="oval" d="{data}" fill="white"/></svg>', encoding='utf-8')
            original = svg.read_bytes()
            render_svg_reference(svg, source, width=128, background=None)
            stats = SimpleNamespace(viewbox=(128., 128.), gradient_info=[], geometry_notes=[])
            report = _attempt_curve_refit_transaction(svg, source, source, stats)
            self.assertEqual(report['status'], 'rolled_back', report)
            self.assertEqual(svg.read_bytes(), original)
            self.assertEqual(report['render_guard']['validation_render_width_px'], 1200)
            self.assertEqual(report['render_guard']['composed_alpha']['reason'],
                             'whole_alpha_components_changed')

    def _fixture(self, folder):
        svg = Path(folder) / "live.svg"
        svg.write_bytes(ORIGINAL)
        stats = SimpleNamespace(
            viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
        return svg, stats

    def _idless_fixture(self, folder):
        svg = Path(folder) / "live.svg"
        svg.write_bytes(
            b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
            b'viewBox="0 0 100 100"><path fill="#126b45" '
            b'd="M0 0L10 0L10 10L0 10Z"/></svg>')
        stats = SimpleNamespace(
            viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
        return svg, stats

    def test_no_change_id_assignment_is_guarded_committed_and_authoritative(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._idless_fixture(folder)
            original = svg.read_bytes()
            stable = _source_scores()
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_idless_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[stable, stable]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            self.assertEqual(report["commit_scope"],
                             "stable_id_normalization_only")
            self.assertNotEqual(svg.read_bytes(), original)
            path = next(item for item in ET.fromstring(svg.read_bytes()).iter()
                        if item.tag.rsplit("}", 1)[-1] == "path")
            self.assertTrue(path.get("id").startswith("avc-refit-path-1-"))
            self.assertEqual(path.get("d"), "M0 0L10 0L10 10L0 10Z")
            self.assertEqual(report["precommit_final_digest_records"],
                             report["postcommit_final_digest_records"])
            self.assertEqual(
                report["proposal"][
                    "transaction_postcommit_final_digest_records"],
                report["postcommit_final_digest_records"])
            self.assertTrue(report["identity_normalization"]
                            ["baseline_validation"]["accepted"])
            self.assertEqual(render.call_count, 1)
            self.assertFalse((Path(folder) /
                              "_curve_refit_identity_baseline.svg").exists())

    def test_full_idless_refit_has_identical_pre_and_post_digest_records(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._idless_fixture(folder)
            stable = _source_scores()
            replacement = "M0 0L10 0L10 9Z"
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_idless_proposal(
                           status="proposed", replacement=replacement)), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[stable, stable, stable]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            self.assertEqual(report["commit_scope"],
                             "geometry_refit_with_stable_id_normalization")
            self.assertEqual(report["precommit_final_digest_records"],
                             report["postcommit_final_digest_records"])
            record = report["postcommit_final_digest_records"][0]
            self.assertTrue(record["id"].startswith("avc-refit-path-1-"))
            self.assertEqual(record["final_drawable_id"], record["id"])
            self.assertEqual(record["path_data_sha256"], hashlib.sha256(
                replacement.encode("utf-8")).hexdigest())
            self.assertEqual(render.call_count, 2)

    def test_idless_candidate_geometry_tamper_fails_before_commit(self):
        def tamper(candidate_bytes, _proposal):
            return candidate_bytes.replace(b"L10 9Z", b"L10 8Z")

        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._idless_fixture(folder)
            original = svg.read_bytes()
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_idless_proposal(
                           status="proposed", replacement="M0 0L10 0L10 9Z",
                           mutate_candidate=tamper)), \
                    patch("vector_cleanroom.validate_svg_stage_renders") as render:
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)
            self.assertEqual(report["status"], "error")
            self.assertTrue(report["live_svg_unchanged"])
            self.assertEqual(svg.read_bytes(), original)
            render.assert_not_called()

    def test_wrong_stable_id_ordinal_and_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, _stats = self._idless_fixture(folder)
            candidate = Path(folder) / "candidate.svg"
            proposal = _idless_proposal()(svg, candidate)
            bad_ordinal = copy.deepcopy(proposal)
            bad_ordinal["stable_id_normalization"]["records"][0][
                "global_path_ordinal_1_based"] = 2
            with self.assertRaisesRegex(RuntimeError, "locator is invalid"):
                _curve_refit_identity_normalized_bytes(
                    svg.read_bytes(), bad_ordinal)
            bad_digest = copy.deepcopy(proposal)
            bad_digest["stable_id_normalization"]["records"][0][
                "source_path_data_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError,
                                        "source path digest mismatch"):
                _curve_refit_identity_normalized_bytes(
                    svg.read_bytes(), bad_digest)
            duplicate_locator = copy.deepcopy(proposal)
            duplicate_record = copy.deepcopy(
                duplicate_locator["stable_id_normalization"]["records"][0])
            duplicate_record["assigned_id"] += "-second"
            duplicate_locator["stable_id_normalization"]["records"].append(
                duplicate_record)
            duplicate_locator["stable_id_normalization"].update({
                "optimizer_evaluated_path_count": 2,
                "record_count": 2,
                "assigned_id_count": 2,
            })
            duplicate_locator["optimizer_evaluated_path_count"] = 2
            duplicate_locator["evaluated_but_retained"].append({
                "id": duplicate_record["assigned_id"]})
            with self.assertRaisesRegex(RuntimeError, "locator is invalid"):
                _curve_refit_identity_normalized_bytes(
                    svg.read_bytes(), duplicate_locator)

    def test_candidate_id_and_duplicate_id_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, _stats = self._idless_fixture(folder)
            candidate = Path(folder) / "candidate.svg"
            proposal = _idless_proposal(
                status="proposed", replacement="M0 0L10 0L10 9Z")(
                    svg, candidate)
            identifier = proposal["stable_id_normalization"]["records"][0][
                "assigned_id"]
            id_tamper = candidate.read_bytes().replace(
                identifier.encode("utf-8"),
                (identifier + "-tampered").encode("utf-8"), 1)
            with self.assertRaisesRegex(RuntimeError,
                                        "candidate bytes digest mismatch"):
                _curve_refit_candidate_identity_guard(id_tamper, proposal)
            duplicate = candidate.read_bytes().replace(
                b"</svg>",
                (f'<path id="{identifier}" d="M20 20L21 20Z"/>'
                 '</svg>').encode("utf-8"))
            duplicate_proposal = copy.deepcopy(proposal)
            duplicate_proposal["stable_id_normalization"][
                "candidate_svg_sha256"] = hashlib.sha256(duplicate).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "duplicate SVG IDs"):
                _curve_refit_candidate_identity_guard(
                    duplicate, duplicate_proposal)

    def test_idless_subset_restore_keeps_generated_locator(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, _stats = self._idless_fixture(folder)
            candidate = Path(folder) / "candidate.svg"
            proposal = _idless_proposal(
                status="proposed", replacement="M0 0L10 0L10 9Z")(
                    svg, candidate)
            normalized, evidence = _curve_refit_identity_normalized_bytes(
                svg.read_bytes(), proposal)
            identifier = evidence["records"][0]["assigned_id"]
            hybrid = _curve_refit_hybrid_bytes(
                normalized, candidate.read_bytes(), [identifier],
                {identifier: proposal["details"][0]})
            path = next(item for item in ET.fromstring(hybrid).iter()
                        if item.tag.rsplit("}", 1)[-1] == "path")
            self.assertEqual(path.get("id"), identifier)
            self.assertEqual(path.get("d"), "M0 0L10 0L10 10L0 10Z")

    def test_postcommit_digest_failure_restores_exact_frozen_original(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._idless_fixture(folder)
            original = svg.read_bytes()
            stable = _source_scores()
            calls = {"count": 0}

            def fail_postcommit(svg_bytes, details):
                calls["count"] += 1
                if calls["count"] == 3:
                    raise RuntimeError("simulated postcommit digest mismatch")
                return _curve_refit_final_digest_records(svg_bytes, details)

            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_idless_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[stable, stable]), \
                    patch("vector_cleanroom._curve_refit_final_digest_records",
                          side_effect=fail_postcommit):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "rolled_back", report)
            self.assertTrue(report["rollback_to_frozen_original_verified"])
            self.assertTrue(report["live_svg_unchanged"])
            self.assertEqual(svg.read_bytes(), original)

    def test_colour_and_source_pixel_scores_cannot_reject_geometry_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._fixture(folder)
            before = _source_scores()
            # Deliberately catastrophic colour and whole-pixel scores.  Ink
            # coverage changes by only 0.1 percentage point and therefore the
            # geometry-qualified candidate must still commit.
            after = _source_scores(
                recall=99.9, precision=99.9, f1=99.9,
                flat=0.0, source=0.0, foreground=0.0, colour=0.0)
            guard = _render_guard(
                accepted=False, colour=0.0,
                recall=99.9, precision=99.9, f1=99.9)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=guard), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, after]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats,
                    error_budget_percent=0.25)
            self.assertEqual(report["status"], "committed")
            self.assertEqual(svg.read_bytes(), CANDIDATE)
            self.assertTrue(report["renderer_topology_guard"]["accepted"])
            self.assertTrue(report["source_guard"]["accepted"])
            self.assertEqual(
                report["source_guard"]["policy"],
                "missing_ink_and_topology_guard_only")
            self.assertEqual(set(report["source_guard"]["comparisons"]), {
                "foreground_recall", "foreground_precision",
                "foreground_coverage_f1", "render_ink_area_ratio",
            })
            excluded = report["source_guard"]["excluded_from_decision"]
            self.assertIn("whole_canvas_source_similarity", excluded)
            self.assertIn("foreground_colour_fidelity", excluded)

    def test_missing_source_ink_rolls_back_and_preserves_exact_live_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._fixture(folder)
            before = _source_scores()
            after = _source_scores(
                recall=96.0, precision=100.0, f1=97.959,
                render_ink=960, flat=100.0, source=100.0,
                foreground=100.0, colour=100.0)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, after]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)
            self.assertEqual(report["status"], "rolled_back")
            self.assertFalse(report["source_guard"]["accepted"])
            self.assertFalse(
                report["source_guard"]["comparisons"]
                ["foreground_recall"]["accepted"])
            self.assertTrue(report["live_svg_unchanged"])
            self.assertEqual(svg.read_bytes(), ORIGINAL)
            self.assertFalse((Path(folder) / "_curve_refit_proposal.svg").exists())

    def test_missing_rendered_ink_rolls_back_even_if_source_scores_are_stable(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._fixture(folder)
            stable = _source_scores()
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard(
                              recall=97.0, precision=100.0, f1=98.477)), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[stable, stable]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)
            self.assertEqual(report["status"], "rolled_back")
            self.assertFalse(report["renderer_topology_guard"]["accepted"])
            self.assertEqual(svg.read_bytes(), ORIGINAL)

    def test_transparent_gaps_cannot_hide_behind_high_ink_recall(self):
        for evidence in ({"accepted": False, "created_regions": 29}, None):
            with self.subTest(evidence=evidence), tempfile.TemporaryDirectory() as folder:
                svg, stats = self._fixture(folder)
                guard = _render_guard(recall=99.99, precision=99.4, f1=99.6)
                guard["alpha_topology"] = evidence
                with patch("curve_refit_stage.propose_svg_curve_refit", side_effect=_proposal()), \
                        patch("vector_cleanroom.validate_svg_stage_renders", return_value=guard), \
                        patch("vector_cleanroom.self_check", return_value=_source_scores()):
                    report = _attempt_curve_refit_transaction(
                        svg, Path(folder) / "flat.png", Path(folder) / "source.png", stats)
                self.assertEqual(report["status"], "rolled_back")
                self.assertFalse(report["renderer_topology_guard"]["accepted"])
                self.assertEqual(svg.read_bytes(), ORIGINAL)

    def test_full_failure_commits_exact_safe_subset_with_reconciled_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            full_failure = _source_scores(recall=99.70)
            subset_pass = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          side_effect=[_render_guard(), _render_guard(),
                                       _render_guard()]) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, full_failure, subset_pass,
                                       subset_pass]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            final_bytes = svg.read_bytes()
            self.assertEqual(report["status"], "committed", report)
            self.assertEqual(render.call_count, 3)
            self.assertEqual(
                report["after_svg_sha256"],
                hashlib.sha256(final_bytes).hexdigest())
            fallback = report["subset_fallback"]
            self.assertEqual(fallback["status"], "partial_candidate_selected")
            # The transaction preserves the high-value 90-anchor reduction and
            # first restores the cheapest six-anchor path.
            self.assertEqual(
                fallback["rollback_order_ids"],
                ["small", "medium", "large"])
            self.assertEqual(fallback["selected_rollback_ids"], ["small"])
            self.assertEqual(
                fallback["rollback_costs"][-1]["id"], "large")
            self.assertTrue(
                fallback["final_revalidation"]["accepted"])
            self.assertTrue(
                fallback["final_revalidation"][
                    "exact_candidate_bytes_stable"])

            proposal = report["proposal"]
            self.assertEqual(proposal["path_count_refit"], 2)
            self.assertEqual(
                proposal["final_committed_path_ids"], ["large", "medium"])
            self.assertEqual(proposal["uncertified_transaction_rollback_ids"],
                             ["small"])
            self.assertEqual(proposal["anchors_before"], 130)
            self.assertEqual(proposal["anchors_after"], 25)
            self.assertEqual(proposal["anchors_removed"], 105)
            self.assertEqual(
                proposal["evaluation_evidence_integrity"]
                ["committed_detail_count"], 2)
            self.assertEqual(
                proposal["evaluation_evidence_integrity"]
                ["uncertified_evaluation_count"], 1)
            self.assertEqual(
                proposal["evaluation_evidence_integrity"]
                ["accounted_evaluation_count"], 3)
            self.assertEqual(
                [item["id"] for item in
                 proposal["final_committed_geometry_digests"]],
                ["large", "medium"])
            self.assertEqual(
                proposal["transaction_subset_fallback"]
                ["selected_candidate_svg_sha256"],
                hashlib.sha256(final_bytes).hexdigest())

            root = ET.fromstring(final_bytes)
            by_id = {item.get("id"): item for item in root.iter()
                     if item.get("id")}
            self.assertEqual(by_id["small"].get("d"),
                             "M80 0L90 0L90 10Z")
            self.assertIsNone(
                by_id["small"].get("data-avc-curve-refit"))
            self.assertEqual(by_id["medium"].get("d"),
                             SUBSET_PATHS["medium"])
            self.assertEqual(by_id["large"].get("d"),
                             SUBSET_PATHS["large"])
            self.assertFalse(
                (Path(folder) / "_curve_refit_proposal.svg").exists())
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_poisoned_largest_path_does_not_discard_independent_safe_refits(self):
        # Every cheap-to-expensive rollback prefix retains the broken largest
        # path. The original search therefore discards two safe smaller edits.
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(viewbox=(100., 100.), gradient_info=[], geometry_notes=[])
            def guard(_before, after, *_args, **_kwargs):
                root = ET.fromstring(Path(after).read_bytes())
                largest = next(node for node in root.iter() if node.get('id') == 'large')
                result = _render_guard()
                result['alpha_topology'] = {'accepted': largest.get('d') == 'M0 0L40 0L40 40Z'}
                return result
            with patch('curve_refit_stage.propose_svg_curve_refit', side_effect=_subset_proposal()), \
                    patch('vector_cleanroom.validate_svg_stage_renders', side_effect=guard), \
                    patch('vector_cleanroom.self_check', return_value=_source_scores()):
                report = _attempt_curve_refit_transaction(svg, Path(folder)/'flat.png', Path(folder)/'source.png', stats)
            self.assertEqual(report['status'], 'committed', report)
            fallback = report['subset_fallback']
            self.assertEqual(fallback['selected_candidate_strategy'], 'bounded_verified_blocks')
            self.assertEqual(fallback['selected_rollback_ids'], ['large'])
            self.assertTrue(fallback['final_revalidation']['exact_candidate_bytes_stable'])
            self.assertEqual(report['proposal']['final_committed_path_ids'], ['medium', 'small'])
            actual = {node.get('id'): node.get('d') for node in ET.fromstring(svg.read_bytes()).iter() if node.get('id')}
            self.assertEqual(actual['large'], 'M0 0L40 0L40 40Z')
            self.assertEqual(actual['medium'], SUBSET_PATHS['medium'])
            self.assertEqual(actual['small'], SUBSET_PATHS['small'])

    def test_bounded_subset_search_never_commits_an_unverified_block(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder)/'live.svg'; svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(viewbox=(100.,100.), gradient_info=[], geometry_notes=[])
            failed = _render_guard(); failed['alpha_topology'] = {'accepted': False}
            with patch('curve_refit_stage.propose_svg_curve_refit', side_effect=_subset_proposal()), \
                    patch('vector_cleanroom.validate_svg_stage_renders', return_value=failed), \
                    patch('vector_cleanroom.self_check', return_value=_source_scores()):
                report = _attempt_curve_refit_transaction(svg, Path(folder)/'flat.png', Path(folder)/'source.png', stats)
            self.assertEqual(report['status'], 'rolled_back')
            self.assertEqual(svg.read_bytes(), SUBSET_ORIGINAL)
            search = report['subset_fallback']['bounded_subset_search']
            self.assertEqual(search['status'], 'no_verified_subset_within_budget')
            self.assertLessEqual(len(search['attempts']), search['maximum_probes'])

    def test_boundary_only_probe_commits_safer_lower_anchor_candidate(self):
        """A lone boundary culprit may replace, never weaken, a safe prefix."""
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.70)
            passing = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()) as propose, \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure, failure,
                                       passing, passing,
                                       passing, passing]) as source_check:
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            fallback = report["subset_fallback"]
            self.assertEqual(
                fallback["selected_candidate_strategy"],
                "boundary_single_path_probe")
            self.assertEqual(fallback["selected_rollback_ids"], ["medium"])
            self.assertEqual(
                fallback["prefix_candidate"]["rollback_ids"],
                ["small", "medium"])
            boundary = fallback["boundary_single_path_probe"]
            self.assertEqual(boundary["status"], "selected")
            self.assertEqual(boundary["boundary_rank_1_based"], 2)
            self.assertEqual(boundary["boundary_id"], "medium")
            self.assertTrue(boundary["failed_predecessor_verified"])
            self.assertEqual(
                boundary["probe"]["rollback_ids"], ["medium"])
            self.assertTrue(boundary["probe"]["accepted"])
            self.assertTrue(
                boundary["probe"]["integrity_guard"]
                ["exact_source_element_join_authenticated"])
            self.assertTrue(
                boundary["final_revalidation"]["accepted"])
            self.assertTrue(
                boundary["final_revalidation"]
                ["exact_candidate_bytes_stable"])
            self.assertTrue(boundary["economy_improves_prefix"])
            self.assertEqual(
                boundary["boundary_candidate_economy"][
                    "committed_refit_ids"], ["large", "small"])
            self.assertEqual(
                boundary["boundary_candidate_economy"]["anchors_after"], 34)
            self.assertEqual(
                boundary["prefix_candidate_economy"]["anchors_after"], 40)

            proposal = report["proposal"]
            self.assertEqual(proposal["path_count_refit"], 2)
            self.assertEqual(
                proposal["final_committed_path_ids"], ["large", "small"])
            self.assertEqual(
                proposal["uncertified_transaction_rollback_ids"],
                ["medium"])
            self.assertEqual(proposal["anchors_after"], 34)
            self.assertEqual(proposal["anchors_removed"], 96)
            root = ET.fromstring(svg.read_bytes())
            by_id = {item.get("id"): item for item in root.iter()
                     if item.get("id")}
            self.assertEqual(by_id["medium"].get("d"),
                             "M50 0L70 0L70 20Z")
            self.assertEqual(by_id["small"].get("d"),
                             SUBSET_PATHS["small"])
            self.assertEqual(by_id["large"].get("d"),
                             SUBSET_PATHS["large"])
            propose.assert_called_once()
            self.assertEqual(render.call_count, 6)
            self.assertEqual(source_check.call_count, 7)
            self.assertFalse(
                (Path(folder) / "_curve_refit_proposal.svg").exists())
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_boundary_identity_is_replaced_by_generic_per_loop_frontier(self):
        """Pixels guard candidates but never rank the geometry frontier."""
        details = [
            _subset_detail("large", 100, 10, 8, SUBSET_PATHS["large"]),
            _subset_detail("medium", 80, 5, 2, SUBSET_PATHS["medium"]),
            _subset_detail("small", 10, 4, 1, SUBSET_PATHS["small"]),
        ]
        source_medium = "M50 0L70 0L70 20Z"

        def row(identifier, path_data, anchors):
            return {
                "candidate_id": identifier,
                "candidate_kind": "single_loop_refinement",
                "eligible": True,
                "changed_loop_index": 1,
                "replacement_candidate_id": f"{identifier}-replacement",
                "path": path_data,
                "path_data_sha256": hashlib.sha256(
                    path_data.encode("utf-8")).hexdigest(),
                "anchors_after": anchors,
                "designer_anchor_count": anchors,
                "segment_count": anchors,
                "actual_p95_error_percent": 0.20,
                "actual_max_error_percent": 0.30,
                "over_budget_share": 0.01,
                "salient_corner_max_percent": 0.20,
                "primitive_complexity": {
                    "category": "bezier_geometry",
                    "native_primitives": [],
                },
            }

        candidates = [
            row("frontier-06", "M50 0L70 0L70 18Z", 6),
            row("frontier-08", "M50 0L70 0L70 19Z", 8),
        ]
        frontier = {
            "schema": "ai-vector-cleanroom.curve-refit-path-frontier/v1",
            "status": "candidates_available",
            "target_id": "medium",
            "optimization_basis": "geometry_only",
            "uses_colour_or_pixel_similarity": False,
            "source_path_data_sha256": hashlib.sha256(
                source_medium.encode("utf-8")).hexdigest(),
            "source_anchor_count": 80,
            "source_loop_anchor_counts": [40, 40],
            "loop_count": 2,
            "error_budget_percent": 0.25,
            "candidate_tolerance_percents": [0.025, 0.0625],
            "measurement_step_percent": 0.08,
            "measurement_max_samples_per_segment": 1024,
            "selection_basis": (
                "designer_anchors_then_anchors_then_segments_then_geometry_then_id"),
            "base_candidate": {
                "anchors_after": 5, "designer_anchor_count": 5},
            "lexicographic_objective": [
                "preserve_topology_hard_constraint",
                "minimize_designer_anchor_count",
            ],
            "candidate_count": 2,
            "candidates": candidates,
        }
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.70)
            passing = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal(details=details)) as propose, \
                    patch(
                        "curve_refit_stage.build_svg_curve_refit_path_frontier",
                        return_value=frontier) as build_frontier, \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[
                              before, failure,
                              failure, passing, passing,
                              passing, passing,
                              failure, passing, passing,
                          ]) as source_check:
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            fallback = report["subset_fallback"]
            self.assertEqual(
                fallback["selected_candidate_strategy"],
                "per_loop_conservative_refinement_frontier")
            self.assertEqual(fallback["selected_rollback_count"], 0)
            self.assertEqual(fallback["selected_rollback_ids"], [])
            frontier_transaction = fallback["per_loop_refinement_frontier"]
            self.assertEqual(frontier_transaction["status"], "selected")
            self.assertEqual(
                frontier_transaction["selected_candidate_id"], "frontier-08")
            self.assertEqual(len(frontier_transaction["attempts"]), 2)
            self.assertFalse(
                frontier_transaction["attempts"][0]["probe"]["accepted"])
            self.assertTrue(
                frontier_transaction["attempts"][1]["probe"]["accepted"])
            self.assertTrue(
                frontier_transaction["attempts"][1]["final_revalidation"]
                ["exact_candidate_bytes_stable"])

            proposal = report["proposal"]
            self.assertEqual(proposal["path_count_refit"], 3)
            self.assertEqual(
                proposal["final_committed_path_ids"],
                ["large", "medium", "small"])
            self.assertEqual(
                proposal["uncertified_transaction_rollback_ids"], [])
            self.assertEqual(proposal["anchors_after"], 22)
            medium = next(
                item for item in proposal["details"]
                if item["id"] == "medium")
            self.assertEqual(medium["anchors_before"], 80)
            self.assertEqual(medium["anchors_after"], 8)
            self.assertEqual(
                medium["selected_candidate_id"], "frontier-08")
            self.assertTrue(medium["economy_certified"])
            root = ET.fromstring(svg.read_bytes())
            by_id = {item.get("id"): item for item in root.iter()
                     if item.get("id")}
            self.assertEqual(by_id["medium"].get("d"), candidates[1]["path"])
            self.assertEqual(
                by_id["medium"].get("data-avc-anchors-after"), "8")
            propose.assert_called_once()
            build_frontier.assert_called_once_with(
                svg.with_name("_curve_refit_identity_baseline.svg"),
                "medium", error_budget_percent=0.25,
                sample_step=2.0, maximum_segments=4096)
            self.assertEqual(render.call_count, 9)
            self.assertEqual(source_check.call_count, 10)
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_frontier_failure_retains_verified_boundary_identity(self):
        details = [
            _subset_detail("large", 100, 10, 8, SUBSET_PATHS["large"]),
            _subset_detail("medium", 80, 5, 2, SUBSET_PATHS["medium"]),
            _subset_detail("small", 10, 4, 1, SUBSET_PATHS["small"]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.70)
            passing = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal(details=details)), \
                    patch(
                        "curve_refit_stage.build_svg_curve_refit_path_frontier",
                        side_effect=RuntimeError("fixture frontier failure")), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure, failure,
                                       passing, passing,
                                       passing, passing]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            fallback = report["subset_fallback"]
            self.assertEqual(
                fallback["selected_candidate_strategy"],
                "boundary_single_path_probe")
            frontier = fallback["per_loop_refinement_frontier"]
            self.assertEqual(frontier["status"], "unavailable_fail_closed")
            self.assertEqual(
                frontier["reason"], "retained_verified_boundary_identity")
            self.assertEqual(fallback["selected_rollback_ids"], ["medium"])
            self.assertEqual(
                report["proposal"]["uncertified_transaction_rollback_ids"],
                ["medium"])
            root = ET.fromstring(svg.read_bytes())
            by_id = {item.get("id"): item for item in root.iter()
                     if item.get("id")}
            self.assertEqual(by_id["medium"].get("d"),
                             "M50 0L70 0L70 20Z")
            self.assertIsNone(by_id["medium"].get("data-avc-curve-refit"))

    def test_boundary_only_guard_failure_retains_verified_prefix(self):
        """One failed focused probe costs one check and preserves the prefix."""
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.70)
            passing = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()) as propose, \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure, failure,
                                       passing, passing, failure]) as source_check:
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            fallback = report["subset_fallback"]
            self.assertEqual(
                fallback["selected_candidate_strategy"],
                "minimal_passing_prefix")
            self.assertEqual(
                fallback["selected_rollback_ids"], ["small", "medium"])
            boundary = fallback["boundary_single_path_probe"]
            self.assertEqual(boundary["status"], "rejected_by_guards")
            self.assertFalse(boundary["probe"]["accepted"])
            self.assertIsNone(boundary["final_revalidation"])
            self.assertEqual(
                fallback["final_revalidation"]["phase"],
                "final_revalidation")
            self.assertEqual(
                report["proposal"]["final_committed_path_ids"], ["large"])
            self.assertEqual(
                report["proposal"]["uncertified_transaction_rollback_ids"],
                ["small", "medium"])
            self.assertEqual(report["proposal"]["anchors_after"], 40)
            propose.assert_called_once()
            self.assertEqual(render.call_count, 5)
            self.assertEqual(source_check.call_count, 6)
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_boundary_probe_byte_tamper_fails_closed_to_verified_prefix(self):
        """Unstable boundary bytes cannot displace the revalidated prefix."""
        calls = []

        def tamper_boundary_probe(original, candidate, rollback_ids,
                                  detail_by_id):
            data = _curve_refit_hybrid_bytes(
                original, candidate, rollback_ids, detail_by_id)
            calls.append(list(rollback_ids))
            if len(calls) == 4:
                root = ET.fromstring(data)
                for element in root.iter():
                    if element.get("id") == "small":
                        element.set("d", "M80 0L90 0L90 8Z")
                return ET.tostring(
                    root, encoding="utf-8", xml_declaration=True)
            return data

        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.70)
            passing = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()) as propose, \
                    patch("vector_cleanroom._curve_refit_hybrid_bytes",
                          side_effect=tamper_boundary_probe) as hybrid_join, \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure, failure,
                                       passing, passing,
                                       passing, passing]) as source_check:
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "committed", report)
            fallback = report["subset_fallback"]
            self.assertEqual(
                fallback["selected_candidate_strategy"],
                "minimal_passing_prefix")
            self.assertEqual(
                fallback["selected_rollback_ids"], ["small", "medium"])
            boundary = fallback["boundary_single_path_probe"]
            self.assertEqual(
                boundary["status"], "rejected_final_revalidation")
            self.assertTrue(boundary["probe"]["accepted"])
            self.assertTrue(boundary["final_revalidation"]["accepted"])
            self.assertFalse(
                boundary["final_revalidation"]
                ["exact_candidate_bytes_stable"])
            self.assertEqual(
                report["proposal"]["final_committed_path_ids"], ["large"])
            self.assertEqual(report["proposal"]["anchors_after"], 40)
            propose.assert_called_once()
            self.assertEqual(hybrid_join.call_count, 5)
            self.assertEqual(render.call_count, 6)
            self.assertEqual(source_check.call_count, 7)
            self.assertFalse(
                (Path(folder) / "_curve_refit_proposal.svg").exists())
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_all_partial_candidates_fail_and_all_refits_are_reconciled_back(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.60)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          side_effect=[_render_guard(), _render_guard(),
                                       _render_guard()]), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure, failure, failure]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "rolled_back", report)
            self.assertEqual(svg.read_bytes(), SUBSET_ORIGINAL)
            fallback = report["subset_fallback"]
            self.assertEqual(fallback["status"], "all_refits_rolled_back")
            self.assertEqual(
                [item["rollback_count"] for item in fallback["attempts"]],
                [1, 2])
            proposal = report["proposal"]
            self.assertEqual(proposal["status"], "no_change")
            self.assertEqual(proposal["details"], [])
            self.assertEqual(proposal["final_committed_path_ids"], [])
            self.assertEqual(proposal["final_committed_geometry_digests"], [])
            self.assertEqual(proposal["anchors_after"], 130)
            self.assertEqual(proposal["anchors_removed"], 0)
            self.assertEqual(
                proposal["uncertified_transaction_rollback_ids"],
                ["small", "medium", "large"])
            self.assertEqual(
                proposal["evaluation_evidence_integrity"]
                ["uncertified_evaluation_count"], 3)
            self.assertFalse(
                (Path(folder) / "_curve_refit_proposal.svg").exists())
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_selected_subset_final_revalidation_failure_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.60)
            passing_once = _source_scores(recall=99.80)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          side_effect=[_render_guard(), _render_guard(),
                                       _render_guard()]), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure, passing_once, failure]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "rolled_back", report)
            self.assertEqual(svg.read_bytes(), SUBSET_ORIGINAL)
            fallback = report["subset_fallback"]
            self.assertEqual(fallback["status"], "all_refits_rolled_back")
            self.assertEqual(
                fallback["reason"],
                "selected_candidate_final_revalidation_failed")
            self.assertFalse(fallback["final_revalidation"]["accepted"])
            self.assertEqual(report["proposal"]["details"], [])
            self.assertEqual(
                report["proposal"]["uncertified_transaction_rollback_ids"],
                ["small", "medium", "large"])
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_single_refit_failure_has_explicit_all_rollback_evidence(self):
        original = (
            b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
            b'viewBox="0 0 10 10"><path id="one" d="M0 0L9 0L9 9Z"/>'
            b'</svg>')
        path_data = "M0 0L9 0L9 8Z"
        candidate = (
            b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
            b'viewBox="0 0 10 10"><path id="one" d="M0 0L9 0L9 8Z" '
            b'data-avc-curve-refit="geometry-budgeted"/></svg>')
        details = [_subset_detail("one", 10, 4, 1, path_data)]
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(original)
            stats = SimpleNamespace(
                viewbox=(10.0, 10.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.60)
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal(candidate, details)), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "rolled_back", report)
            self.assertEqual(svg.read_bytes(), original)
            fallback = report["subset_fallback"]
            self.assertEqual(fallback["status"], "all_refits_rolled_back")
            self.assertEqual(
                fallback["reason"],
                "single_refit_has_no_nonempty_partial_subset")
            self.assertEqual(fallback["attempts"], [])
            self.assertEqual(
                report["proposal"]["uncertified_transaction_rollback_ids"],
                ["one"])
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_each_hybrid_fails_closed_if_gradient_snapshot_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            svg.write_bytes(SUBSET_ORIGINAL)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            before = _source_scores()
            failure = _source_scores(recall=99.60)
            stable_gradient = [{"id": "field", "geometry_sha256": "stable"}]
            changed_gradient = [{"id": "field", "geometry_sha256": "changed"}]
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_subset_proposal()), \
                    patch("vector_cleanroom._gradient_geometry_snapshot",
                          side_effect=[stable_gradient, stable_gradient,
                                       changed_gradient, changed_gradient]), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()) as render, \
                    patch("vector_cleanroom.self_check",
                          side_effect=[before, failure]):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "rolled_back", report)
            self.assertEqual(svg.read_bytes(), SUBSET_ORIGINAL)
            self.assertEqual(render.call_count, 1)
            attempts = report["subset_fallback"]["attempts"]
            self.assertEqual(len(attempts), 2)
            self.assertTrue(all(
                item["gradient_geometry_guard"]["status"] == "rejected"
                and item["render_guard"] is None
                for item in attempts))
            self.assertFalse(
                (Path(folder) / "_curve_refit_proposal.svg").exists())
            self.assertFalse(
                (Path(folder) / "_curve_refit_hybrid.svg").exists())

    def test_atomic_commit_exception_keeps_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            svg, stats = self._fixture(folder)
            stable = _source_scores()
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_proposal()), \
                    patch("vector_cleanroom.validate_svg_stage_renders",
                          return_value=_render_guard()), \
                    patch("vector_cleanroom.self_check",
                          side_effect=[stable, stable]), \
                    patch("svg_postprocess.atomic_replace_bytes",
                          side_effect=OSError("simulated commit failure")):
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)
            self.assertEqual(report["status"], "error")
            self.assertTrue(report["live_svg_unchanged"])
            self.assertEqual(svg.read_bytes(), ORIGINAL)

    def test_gradient_geometry_change_is_rejected_before_render_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            original = (
                b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
                b'viewBox="0 0 100 100"><defs><linearGradient id="grad1"/>'
                b'</defs><path id="mountain" fill="url(#grad1)" '
                b'data-avc-gradient-object="mountain-field" '
                b'data-avc-designer-anchors="3" '
                b'data-avc-error-budget-percent="0.25" '
                b'data-avc-p95-error-percent="0.12" '
                b'data-avc-max-error-percent="0.3" '
                b'd="M0 0L10 0L10 10Z"/></svg>'
            )
            changed = original.replace(
                b'M0 0L10 0L10 10Z', b'M0 0L5 0L10 0L10 10Z')
            svg.write_bytes(original)
            stats = SimpleNamespace(
                viewbox=(100.0, 100.0), gradient_info=[], geometry_notes=[])
            with patch("curve_refit_stage.propose_svg_curve_refit",
                       side_effect=_proposal(changed)), \
                    patch("vector_cleanroom.validate_svg_stage_renders") as render:
                report = _attempt_curve_refit_transaction(
                    svg, Path(folder) / "flat.png",
                    Path(folder) / "source.png", stats)

            self.assertEqual(report["status"], "rolled_back")
            self.assertEqual(
                report["reason"],
                "gradient_geometry_ownership_revalidation_missing")
            self.assertEqual(
                report["gradient_geometry_guard"]["status"], "rejected")
            self.assertTrue(report["live_svg_unchanged"])
            self.assertEqual(svg.read_bytes(), original)
            render.assert_not_called()

    def test_final_gradient_report_requires_exact_final_svg_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "live.svg"
            original = (
                '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="0 0 100 100"><defs><linearGradient id="grad1"/>'
                '</defs><path id="mountain" fill="url(#grad1)" '
                'data-avc-gradient-object="mountain-field" '
                'data-avc-designer-anchors="3" '
                'data-avc-error-budget-percent="0.25" '
                'data-avc-p95-error-percent="0.12" '
                'data-avc-max-error-percent="0.3" '
                'd="M0 0L10 0L10 10Z"/></svg>'
            )
            svg.write_text(original, encoding="utf-8")
            snapshot = _gradient_geometry_snapshot(svg)
            curve_report = {"gradient_geometry_guard": {
                "before_geometry_sha256": _gradient_geometry_digest(snapshot),
            }}
            gradient_info = [{
                "id": "grad1",
                "validation": {"geometry": {
                    "anchor_count": 3,
                    "designer_anchor_count": 3,
                    "error_budget": {
                        "requested_max_percent": 0.25,
                        "actual_p95_error_percent": 0.12,
                        "actual_max_error_percent": 0.30,
                    },
                }},
            }]
            gradient_info[0]["candidate_id"] = "candidate-1"
            gradient_info[0]["validation"]["geometry"]["segment_count"] = 3

            details, consistency = _final_gradient_report_details(
                svg, gradient_info, curve_report)
            geometry = details[0]["validation"]["geometry"]
            self.assertEqual(
                geometry["evidence_scope"],
                "gradient_reconstruction_against_original_source_ownership_mask")
            self.assertEqual(
                geometry["final_svg_consistency"]["status"],
                "verified_unchanged")
            self.assertEqual(
                geometry["final_svg_consistency"]["final_anchor_count"], 3)
            self.assertEqual(consistency["status"], "verified_unchanged")
            self.assertFalse(
                details[0]["geometry_evidence"]["curve_refit_applied"])

            joined_object = {
                "candidate_id": "candidate-1",
                "gradient_id": "grad1",
                "final_drawable_id": "mountain",
                "gradient_object_id": "mountain-field",
                "final_element": "path",
                "anchor_count": 3,
                "designer_anchor_count": 3,
                "segment_count": 3,
                "economy_certificate_available": True,
            }
            designer_quality = {"gradient_object_gate": {
                "source_space_field_evidence": {
                    "objects": [joined_object],
                },
            }}
            _verify_designer_gradient_evidence_join(
                details, designer_quality)
            joined_object["final_drawable_id"] = None
            with self.assertRaisesRegex(
                    RuntimeError, "did not consume finalized gradient"):
                _verify_designer_gradient_evidence_join(
                    details, designer_quality)

            svg.write_text(
                original.replace(
                    'M0 0L10 0L10 10Z', 'M0 0L5 0L10 0L10 10Z'),
                encoding="utf-8")
            with self.assertRaisesRegex(
                    RuntimeError, "differs from the ownership-validated"):
                _final_gradient_report_details(
                    svg, gradient_info, curve_report)

    def test_final_native_gradient_requires_whole_shape_and_numeric_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            svg = Path(folder) / "native.svg"
            source = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40">'
                      '<defs><linearGradient id="g"/></defs><circle id="disc" '
                      'cx="20" cy="20" r="16" fill="url(#g)" '
                      'data-avc-gradient-object="disc-object" '
                      'data-avc-designer-anchors="4" data-avc-error-budget-percent="0.25" '
                      'data-avc-p95-error-percent="0.12" data-avc-max-error-percent="0.30"/></svg>')
            geometry = {
                "anchor_count": 2, "designer_anchor_count": 4, "segment_count": 2,
                "primitive_first": True,
                "native_primitives": [{"element": "circle", "cx": 20, "cy": 20, "r": 16}],
                "native_whole_object_path": "M4 20 A16 16 0 1 0 36 20 A16 16 0 1 0 4 20 Z",
                "topology": {"components": 1, "holes": 0, "topology_preserved": True},
                "error_budget": {"requested_max_percent": 0.25,
                                 "actual_p95_error_percent": 0.12,
                                 "actual_max_error_percent": 0.30}}
            for mode in ("valid", "partial-native", "moved-native", "missing-path"):
                with self.subTest(mode=mode):
                    evidence = copy.deepcopy(geometry)
                    svg.write_text(source.replace('cx="20"', 'cx="21"')
                                   if mode == "moved-native" else source, encoding="utf-8")
                    if mode == "partial-native":
                        evidence["anchor_count"] = 149
                        evidence["topology"]["holes"] = 3
                    if mode == "missing-path":
                        evidence.pop("native_whole_object_path")
                    guard = {"gradient_geometry_guard": {"before_geometry_sha256":
                        _gradient_geometry_digest(_gradient_geometry_snapshot(svg))}}
                    info = [{"id": "g", "validation": {"geometry": evidence}}]
                    if mode == "valid":
                        details, _ = _final_gradient_report_details(svg, info, guard)
                        final = details[0]["validation"]["geometry"]["final_svg_consistency"]
                        self.assertEqual(final["reconstruction_anchor_count"], 2)
                        self.assertEqual(final["final_anchor_count"], 1)
                        self.assertEqual(final["native_geometry"]["r"], "16")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "whole-object proof"):
                            _final_gradient_report_details(svg, info, guard)

    def test_renderer_validation_forwards_explicit_tolerance(self):
        with tempfile.TemporaryDirectory() as folder:
            before = Path(folder) / "before.svg"
            after = Path(folder) / "after.svg"
            before.write_bytes(ORIGINAL)
            after.write_bytes(CANDIDATE)

            def fake_render(_svg, png, **_kwargs):
                from PIL import Image
                Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(png)
                return True

            metrics = _render_guard(
                accepted=False, colour=0.0,
                recall=100.0, precision=100.0, f1=100.0)
            with patch("vector_cleanroom.render_svg_png",
                       side_effect=fake_render), \
                    patch("annulus_detector.compare_rendered_pngs",
                          return_value=dict(metrics)) as compare:
                report = validate_svg_stage_renders(
                    before, after, "curve_refit", render_size=512,
                    tolerance_px=4)
            compare.assert_called_once()
            self.assertEqual(compare.call_args.kwargs["tolerance_px"], 4)
            self.assertEqual(report["ink_recall_percent"], 100.0)
            self.assertEqual(report["color_similarity_percent"], 0.0)
            # This low-level report retains diagnostics; the transaction's
            # renderer_topology_guard intentionally ignores report.accepted.
            self.assertFalse(report["accepted"])
            self.assertFalse((Path(folder) / "before-curve_refit-render.png").exists())
            self.assertFalse((Path(folder) / "after-curve_refit-render.png").exists())

    def test_curve_error_cli_accepts_closed_interval_and_rejects_outside(self):
        parser = build_arg_parser()
        for value in ("0.05", "0.25", "2"):
            args = parser.parse_args(["--curve-error-percent", value])
            validate_args(parser, args)
            self.assertEqual(args.curve_error_percent, float(value))
        for value in ("0.049999", "2.000001", "nan", "inf", "-inf"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    args = parser.parse_args(
                        [f"--curve-error-percent={value}"])
                    validate_args(parser, args)


if __name__ == "__main__":
    unittest.main()
