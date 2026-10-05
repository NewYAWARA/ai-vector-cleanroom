from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from designer_quality import AUDIT_SCHEMA, audit_designer_quality  # noqa: E402


def _visual_metadata(status: str = "accepted") -> str:
    payload = {
        "visual_acceptance_status": status,
        "visual_gate": {
            "status": status,
            "metrics": {"foreground": 98.5, "color_fidelity": 97.0},
        },
    }
    return (f'<metadata id="ai-vector-cleanroom-metadata">'
            f'{json.dumps(payload)}</metadata>')


def _good_svg() -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
  {_visual_metadata()}
  <defs>
    <linearGradient id="mountain" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#174f39"/>
      <stop offset="0.55" stop-color="#4d8f73"/>
      <stop offset="1" stop-color="#9bc7b7"/>
    </linearGradient>
    <radialGradient id="sun">
      <stop offset="0" stop-color="#ffe86a"/>
      <stop offset="1" stop-color="#d59b16"/>
    </radialGradient>
  </defs>
  <circle id="frame" cx="500" cy="280" r="230" fill="none"
          stroke="#174f39" stroke-width="10"/>
  <circle id="sun-disc" cx="500" cy="170" r="55" fill="url(#sun)"
          data-avc-gradient-object="sun"
          data-avc-error-budget-percent="0.5"
          data-avc-p95-error-percent="0.4"
          data-avc-max-error-percent="1.2"
          data-avc-designer-anchors="4"/>
  <path id="mountain-object"
        d="M180 430 C250 355 310 270 395 330 C450 365 485 250 545 310
           C620 390 700 325 820 430 C650 500 350 500 180 430 Z"
        fill="url(#mountain)"
        data-avc-gradient-object="mountain"
        data-avc-error-budget-percent="0.5"
        data-avc-p95-error-percent="0.4"
        data-avc-max-error-percent="1.4"
        data-avc-designer-anchors="5"/>
  <line x1="420" y1="90" x2="400" y2="45" stroke="#d59b16"/>
  <line x1="580" y1="90" x2="600" y2="45" stroke="#d59b16"/>
</svg>'''


def _micro_rectangle(identifier: str, x: int, colour: str) -> str:
    """A deliberately trace-like rectangle made from two-unit cubic steps."""
    points: list[tuple[int, int]] = []
    points.extend((value, 220) for value in range(x + 2, x + 18, 2))
    points.extend((x + 18, value) for value in range(222, 302, 2))
    points.extend((value, 300) for value in range(x + 16, x - 1, -2))
    points.extend((x, value) for value in range(298, 219, -2))
    current = (x, 220)
    commands = [f"M{current[0]} {current[1]}"]
    for point in points:
        first = (current[0] + (point[0] - current[0]) / 3.0,
                 current[1] + (point[1] - current[1]) / 3.0)
        second = (current[0] + 2.0 * (point[0] - current[0]) / 3.0,
                  current[1] + 2.0 * (point[1] - current[1]) / 3.0)
        commands.append(
            f"C{first[0]:.3f} {first[1]:.3f} {second[0]:.3f} "
            f"{second[1]:.3f} {point[0]} {point[1]}")
        current = point
    commands.append("Z")
    return f'<path id="{identifier}" d="{" ".join(commands)}" fill="{colour}"/>'


def _bad_svg() -> str:
    colours = [
        "#174f39", "#205a42", "#2b6650", "#39745e", "#4a826d", "#5b907c",
        "#6b9e8b", "#7baf9b", "#8dbdac", "#9bc7b7", "#79a68f", "#416f59",
    ]
    fragments = "\n".join(
        _micro_rectangle(f"band-{index}", 120 + index * 18, colour)
        for index, colour in enumerate(colours)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
  {_visual_metadata()}
  <defs>
    <linearGradient id="token-gradient">
      <stop offset="0" stop-color="#174f39"/>
      <stop offset="1" stop-color="#9bc7b7"/>
    </linearGradient>
  </defs>
  <circle cx="760" cy="150" r="50" fill="url(#token-gradient)"/>
  {fragments}
</svg>'''


def _source_certified_field_svg() -> str:
    colours = [
        "#174f39", "#205a42", "#2b6650", "#39745e",
        "#4a826d", "#5b907c", "#6b9e8b", "#7baf9b",
    ]
    unrelated = "".join(
        f'<rect id="flat-{index}" x="{20 + (index % 12) * 12}" '
        f'y="{300 + (index // 12) * 22}" width="11" height="20" '
        f'fill="{colours[index % len(colours)]}"/>'
        for index in range(24)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
      <defs><linearGradient id="field">
        <stop offset="0" stop-color="#9adbea"/>
        <stop offset="1" stop-color="#083c24"/>
      </linearGradient></defs>
      <path id="field-object" d="M250 250 C360 120 560 120 720 250 Z"
            fill="url(#field)" data-avc-gradient-object="main-field"
            data-avc-error-budget-percent="0.25"
            data-avc-p95-error-percent="0.12"
            data-avc-max-error-percent="0.30"
            data-avc-designer-anchors="4"/>
      {unrelated}
    </svg>'''


def _source_certified_field_metadata(*, hard_edge=False, overlap=0,
                                     detail_candidate_id="field-candidate"):
    candidate_id = "field-candidate"
    return {
        "gradient_reconstruction_report": {
            "schema": "ai-vector-cleanroom.gradient-reconstruction-stage/v1",
            "status": "proposed",
            "summary": {
                "objects_selected": 1,
                "overlap_pixels_between_selected": overlap,
                "geometry_shortlist_deferred": 0,
            },
            "objective": {"hard_constraints": [
                "paint_model_beats_solid_on_deterministic_heldout",
                "no_material_internal_hard_edge",
                "topology_preserved",
                "p95_geometry_error_percent_within_budget",
                "maximum_geometry_error_within_three_times_budget_tail",
                "selected_masks_are_pairwise_disjoint",
            ]},
            "decisions": [{
                "candidate_id": candidate_id,
                "status": "selected",
            }],
        },
        "gradient_details": [{
            "id": "field",
            "candidate_id": detail_candidate_id,
            "stops": [
                {"offset": 0.0, "color": "#9adbea"},
                {"offset": 1.0, "color": "#083c24"},
            ],
            "validation": {
                "engine": "source_space_heldout_gradient_object",
                "paint": {
                    "validation": {
                        "passed": True,
                        "internal_edges": {
                            "material_internal_hard_edge": hard_edge,
                            "material_hard_label_boundary": False,
                        },
                    },
                    "independent_revalidation": {"passed": True},
                },
                "geometry": {
                    "topology": {"topology_preserved": True},
                    "error_budget": {"passed": True},
                },
                "selection": {"colour_used_for_geometry": False},
            },
        }],
    }


def _dense_retained_path_data() -> str:
    top = " ".join(f"L{x} 10" for x in range(11, 40))
    bottom = " ".join(f"L{x} 20" for x in range(38, 9, -1))
    return f"M10 10 {top} L39 20 {bottom} Z"


def _economy_objective() -> list[str]:
    return [
        "preserve_topology_hard_constraint",
        "p95_bidirectional_geometric_error_percent_within_budget_hard_constraint",
        "maximum_error_within_three_times_budget_hard_tail_constraint",
        "over_budget_share_at_most_5_percent_hard_tail_constraint",
        "salient_corner_error_within_two_times_budget_hard_constraint",
        "minimize_designer_anchor_count",
        "minimize_anchor_count",
        "minimize_fragment_count",
        "minimize_segment_count",
    ]


def _retained_curve_proposal(path_data: str) -> dict:
    anchors = 60
    retained = {
        "id": "retained",
        "fill": "#174f39",
        "outcome": "retained_identity_minimum",
        "economy_certified": True,
        "economy_reason": (
            "source_path_baseline_is_no_more_complex_than_best_error_eligible_"
            "evaluated_candidate"),
        "retention_basis": "source_svg_path_implicit_zero_error_candidate",
        "geometry_unchanged": True,
        "path_data_sha256": hashlib.sha256(
            path_data.encode("utf-8")).hexdigest(),
        "path_data_digest_scope": "utf8_svg_path_d_attribute",
        "source_baseline_geometry_error": {
            "actual_p95_error_percent": 0.0,
            "actual_max_error_percent": 0.0,
            "over_budget_share": 0.0,
            "salient_corner_max_percent": 0.0,
            "passed": True,
        },
        "economy_comparison_order": [
            "designer_anchor_count", "anchor_count", "fragment_count",
            "segment_count",
        ],
        "source_baseline_economy": {
            "designer_anchor_count": anchors, "anchor_count": anchors,
            "fragment_count": 1, "segment_count": anchors,
        },
        "optimizer_selected_economy": {
            "designer_anchor_count": anchors, "anchor_count": anchors,
            "fragment_count": 1, "segment_count": anchors,
        },
        "anchors_before": anchors,
        "anchors_after": anchors,
        "designer_anchors_before": anchors,
        "designer_anchors_after": anchors,
        "optimizer_input_anchor_count": 180,
        "optimizer_selected_anchor_count": anchors,
        "optimizer_selected_designer_anchor_count": anchors,
        "optimizer_selected_segment_count": anchors,
        "error_budget_percent": 0.25,
        "actual_p95_error_percent": 0.20,
        "actual_max_error_percent": 0.30,
        "over_budget_share": 0.01,
        "salient_corner_max_percent": 0.20,
        "selected_candidate_id": "curve_refit_04",
        "selected_source": "curve_refit_per_loop_tolerance",
        "identity_rollback_selected": False,
        "lexicographic_objective": _economy_objective(),
        "loops": 1,
        "candidate_count": 10,
        "eligible_candidate_count": 8,
        "primitive": {
            "category": "bezier_geometry", "native": [],
            "emitted_element": None,
        },
        "stage_reason": "no_strict_complexity_reduction",
    }
    source_digest = hashlib.sha256(path_data.encode("utf-8")).hexdigest()
    stable_record = {
        "source_svg_sha256": "a" * 64,
        "global_path_ordinal_1_based": 1,
        "source_element": "path",
        "source_path_data_sha256": source_digest,
        "source_path_data_digest_scope": "utf8_svg_path_d_attribute",
        "original_id_state": "existing_preserved",
        "original_id": "retained",
        "assigned_id": "retained",
        "assignment_applied": False,
    }
    stable_ids = {
        "schema": (
            "ai-vector-cleanroom.curve-refit-stable-id-normalization/v1"),
        "source_svg_sha256": "a" * 64,
        "source_svg_digest_scope": "exact_source_svg_bytes",
        "source_path_count": 1,
        "optimizer_evaluated_path_count": 1,
        "record_count": 1,
        "assigned_id_count": 0,
        "existing_id_preserved_count": 1,
        "all_optimizer_evaluations_authenticated": True,
        "assigned_ids_unique": True,
        "candidate_all_svg_ids_unique": True,
        "candidate_svg_sha256": "c" * 64,
        "records": [stable_record],
    }
    return {
        "editability_enhancements": {
            "schema": "ai-vector-cleanroom.editability-enhancements/v1",
            "stages": {
                "curve_refit": {
                    "schema": "ai-vector-cleanroom.curve-refit-transaction/v1",
                    "status": "not_needed",
                    "reason": "no_safe_reductions",
                    "commit_scope": "none",
                    "before_svg_sha256": "a" * 64,
                    "proposal": {
                        "schema": "ai-vector-cleanroom.curve-refit-proposal/v3",
                        "status": "no_change",
                        "path_count_refit": 0,
                        "eligible_path_count": 1,
                        "optimizer_evaluated_path_count": 1,
                        "retained_identity_path_count": 1,
                        "anchors_before": anchors,
                        "anchors_after": anchors,
                        "optimization_basis": "geometry_only",
                        "uses_colour_or_pixel_similarity": False,
                        "details": [],
                        "evaluated_but_retained": [retained],
                        "uncertified_evaluations": [],
                        "stable_id_normalization": stable_ids,
                        "evaluation_evidence_integrity": {
                            "schema": (
                                "ai-vector-cleanroom.curve-refit-evaluation-"
                                "evidence/v1"),
                            "optimizer_evaluated_path_count": 1,
                            "committed_detail_count": 0,
                            "retained_identity_detail_count": 1,
                            "uncertified_evaluation_count": 0,
                            "accounted_evaluation_count": 1,
                            "all_optimizer_results_accounted": True,
                            "evidence_ids_unique": True,
                            "committed_and_retained_ids_disjoint": True,
                        },
                    },
                    "gradient_geometry_guard": {
                        "status": "verified_unchanged",
                        "ownership_mask_revalidation_performed": False,
                    },
                    "identity_normalization": {
                        "schema": (
                            "ai-vector-cleanroom.curve-refit-stable-id-"
                            "transaction/v1"),
                        "status": "authenticated",
                        "source_svg_sha256": "a" * 64,
                        "normalized_baseline_svg_sha256": "a" * 64,
                        "source_path_count": 1,
                        "optimizer_evaluated_path_count": 1,
                        "record_count": 1,
                        "assigned_id_count": 0,
                        "existing_id_preserved_count": 1,
                        "assigned_ids_unique": True,
                        "source_existing_ids_unique": True,
                        "normalized_all_svg_ids_unique": True,
                        "records": [stable_record],
                        "candidate_identity_guard": {
                            "status": "verified",
                            "candidate_all_svg_ids_unique": True,
                            "evaluated_path_count": 1,
                        },
                        "baseline_validation": {
                            "status": "not_needed",
                            "accepted": True,
                        },
                    },
                },
            },
        },
    }


def _committed_curve_detail(
        identifier: str, *, final_element: str, emitted_element: str | None,
        anchors_before: int, anchors_after: int) -> dict:
    return {
        "id": identifier,
        "fill": "#174f39",
        "outcome": "committed_refit",
        "economy_certified": True,
        "final_element": final_element,
        "final_drawable_id": identifier,
        "anchors_before": anchors_before,
        "anchors_after": anchors_after,
        "designer_anchors_before": anchors_before,
        "designer_anchors_after": anchors_after,
        "error_budget_percent": 0.25,
        "actual_p95_error_percent": 0.20,
        "actual_max_error_percent": 0.30,
        "over_budget_share": 0.01,
        "salient_corner_max_percent": 0.20,
        "selected_candidate_id": "curve_refit_01",
        "selected_source": "curve_refit_per_loop_tolerance",
        "lexicographic_objective": _economy_objective(),
        "loops": 1,
        "primitive": {
            "category": "native_geometry" if emitted_element else "bezier_geometry",
            "native": [emitted_element] if emitted_element else [],
            "emitted_element": emitted_element,
        },
    }


def _committed_curve_proposal(detail: dict) -> dict:
    identifier = str(detail["id"])
    digest_keys = (
        ("final_element", "final_drawable_id", "path_data_sha256",
         "path_data_digest_scope")
        if detail.get("final_element") == "path" else
        ("final_element", "final_drawable_id", "native_parameters",
         "geometry_sha256", "geometry_digest_scope")
    )
    final_record = {
        "id": identifier,
        **{key: detail[key] for key in digest_keys},
    }
    stable_record = {
        "source_svg_sha256": "a" * 64,
        "global_path_ordinal_1_based": 1,
        "source_element": "path",
        "source_path_data_sha256": "c" * 64,
        "source_path_data_digest_scope": "utf8_svg_path_d_attribute",
        "original_id_state": "existing_preserved",
        "original_id": identifier,
        "assigned_id": identifier,
        "assignment_applied": False,
    }
    stable_ids = {
        "schema": (
            "ai-vector-cleanroom.curve-refit-stable-id-normalization/v1"),
        "source_svg_sha256": "a" * 64,
        "source_svg_digest_scope": "exact_source_svg_bytes",
        "source_path_count": 1,
        "optimizer_evaluated_path_count": 1,
        "record_count": 1,
        "assigned_id_count": 0,
        "existing_id_preserved_count": 1,
        "all_optimizer_evaluations_authenticated": True,
        "assigned_ids_unique": True,
        "candidate_all_svg_ids_unique": True,
        "candidate_svg_sha256": "d" * 64,
        "records": [stable_record],
    }
    proposal = {
        "schema": "ai-vector-cleanroom.curve-refit-proposal/v3",
        "status": "proposed",
        "path_count_refit": 1,
        "eligible_path_count": 1,
        "optimizer_evaluated_path_count": 1,
        "retained_identity_path_count": 0,
        "anchors_before": detail["anchors_before"],
        "anchors_after": detail["anchors_after"],
        "optimization_basis": "geometry_only",
        "uses_colour_or_pixel_similarity": False,
        "details": [detail],
        "evaluated_but_retained": [],
        "uncertified_evaluations": [],
        "stable_id_normalization": stable_ids,
        "transaction_precommit_final_digest_records": [final_record],
        "transaction_postcommit_final_digest_records": [final_record],
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
    return {
        "editability_enhancements": {
            "schema": "ai-vector-cleanroom.editability-enhancements/v1",
            "stages": {
                "curve_refit": {
                    "schema": "ai-vector-cleanroom.curve-refit-transaction/v1",
                    "status": "committed",
                    "reason": None,
                    "commit_scope": "geometry_refit",
                    "before_svg_sha256": "a" * 64,
                    "after_svg_sha256": "b" * 64,
                    "proposal": proposal,
                    "precommit_final_digest_records": [final_record],
                    "postcommit_final_digest_records": [final_record],
                    "render_guard": {
                        "accepted": True,
                        "external_render_check": "completed",
                        "score_percent": 100.0,
                        "ink_recall_percent": 100.0,
                        "ink_precision_percent": 100.0,
                        "ink_f1_percent": 100.0,
                        "color_similarity_percent": 100.0,
                    },
                    "source_guard": {"accepted": True},
                    "renderer_topology_guard": {
                        "accepted": True,
                        "policy": "bidirectional_ink_topology_guard",
                        "comparisons": {
                            key: {
                                "value": 100.0,
                                "minimum": 99.0,
                                "accepted": True,
                            }
                            for key in (
                                "ink_recall_percent",
                                "ink_precision_percent",
                                "ink_f1_percent",
                            )
                        },
                        "colour_similarity_excluded": True,
                    },
                    "gradient_geometry_guard": {
                        "status": "verified_unchanged",
                        "ownership_mask_revalidation_performed": False,
                    },
                    "identity_normalization": {
                        "schema": (
                            "ai-vector-cleanroom.curve-refit-stable-id-"
                            "transaction/v1"),
                        "status": "committed",
                        "commit_scope": "geometry_refit",
                        "source_svg_sha256": "a" * 64,
                        "normalized_baseline_svg_sha256": "a" * 64,
                        "postcommit_svg_sha256": "b" * 64,
                        "source_path_count": 1,
                        "optimizer_evaluated_path_count": 1,
                        "record_count": 1,
                        "assigned_id_count": 0,
                        "existing_id_preserved_count": 1,
                        "assigned_ids_unique": True,
                        "source_existing_ids_unique": True,
                        "normalized_all_svg_ids_unique": True,
                        "records": [stable_record],
                        "candidate_identity_guard": {
                            "status": "verified",
                            "candidate_all_svg_ids_unique": True,
                            "evaluated_path_count": 1,
                        },
                        "baseline_validation": {
                            "status": "not_needed",
                            "accepted": True,
                        },
                        "precommit_final_digest_records": [final_record],
                        "postcommit_final_digest_records": [final_record],
                    },
                },
            },
        },
    }


def _native_geometry_fields(
        kind: str, identifier: str, parameters: dict[str, str]) -> dict:
    canonical = json.dumps({
        "element": kind,
        "id": identifier,
        "parameters": parameters,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "native_parameters": dict(parameters),
        "geometry_sha256": hashlib.sha256(
            canonical.encode("utf-8")).hexdigest(),
        "geometry_digest_scope": (
            "canonical_json_element_id_and_svg_geometry_attributes"),
    }


def _committed_path_fixture(path_data: str | None = None) -> tuple[str, dict]:
    identifier = "committed-path"
    path_data = path_data or "M10 10 L90 10 L90 90 L10 90 Z"
    detail = _committed_curve_detail(
        identifier, final_element="path", emitted_element=None,
        anchors_before=12, anchors_after=4)
    detail.update({
        "path_data_sha256": hashlib.sha256(
            path_data.encode("utf-8")).hexdigest(),
        "path_data_digest_scope": "utf8_svg_path_d_attribute",
    })
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      {_visual_metadata()}
      <path id="{identifier}" d="{path_data}" fill="#174f39"
            data-avc-curve-refit="geometry-budgeted"
            data-avc-anchors-before="12" data-avc-anchors-after="4"
            data-avc-designer-anchors="4"
            data-avc-error-budget-percent="0.25"
            data-avc-p95-error-percent="0.20"
            data-avc-max-error-percent="0.30"/>
    </svg>'''
    return svg, _committed_curve_proposal(detail)


def _rolled_back_curve_fixture(*, named=False, dense=False) -> tuple[str, dict]:
    path_data = (_dense_retained_path_data() if dense else
                 "M10 10 L90 10 L90 30 L10 30 Z")
    anchors = 60 if dense else 4
    identifier = "committed-path"
    attribute = f'id="{identifier}"' if named else ""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           f'{_visual_metadata()}<path {attribute} d="{path_data}" fill="#174f39"/></svg>')
    _, report = _committed_path_fixture(path_data)
    curve = report["editability_enhancements"]["stages"]["curve_refit"]
    proposal = curve["proposal"]
    digest = hashlib.sha256(svg.encode()).hexdigest()
    stable = proposal["stable_id_normalization"]
    identity = curve["identity_normalization"]
    for evidence in (stable, identity):
        evidence.update(source_svg_sha256=digest, assigned_id_count=0 if named else 1,
                        existing_id_preserved_count=1 if named else 0)
        evidence["records"][0].update(
            source_svg_sha256=digest,
            source_path_data_sha256=hashlib.sha256(path_data.encode()).hexdigest(),
            original_id_state="existing_preserved" if named else "missing_assigned",
            original_id=identifier if named else None, assignment_applied=not named)
    identity["status"] = "authenticated"
    identity["baseline_validation"] = {
        "status": "not_needed" if named else "verified", "accepted": True,
        "reason": "no_missing_ids_required_assignment" if named else None,
        **{key: {"accepted": True} for key in
           ("renderer_topology_guard", "source_guard", "gradient_geometry_guard")}}
    subset = {"schema": "ai-vector-cleanroom.curve-refit-subset-fallback/v1",
              "status": "all_refits_rolled_back", "full_candidate_validation": {
                  "renderer_topology_guard": {"accepted": False},
                  "source_guard": {"accepted": True}}}
    proposal.update(status="no_change", path_count_refit=0, details=[],
        anchors_before=anchors, anchors_after=anchors,
        transaction_subset_fallback=subset,
        uncertified_transaction_rollback_ids=[identifier],
        uncertified_evaluations=[{
            "id": identifier, "outcome": "transaction_guard_rollback_to_source_identity",
            "economy_certified": False, "rollback_basis": "exact_source_svg_element_restoration",
            "stage_reason": "renderer_or_source_guard_rejected_candidate",
            "integrity_failures": ["geometry_candidate_not_committed_after_transaction_guard"],
            "anchors_before": anchors, "anchors_after": anchors,
            "designer_anchors_before": anchors, "designer_anchors_after": anchors}])
    proposal["evaluation_evidence_integrity"].update(
        committed_detail_count=0, uncertified_evaluation_count=1, transaction_rollback_count=1)
    curve.update(status="rolled_back", reason="renderer_or_source_guard_rejected",
        before_svg_sha256=digest, after_svg_sha256=digest, live_svg_unchanged=True,
        commit_scope=None, precommit_final_digest_records=None,
        postcommit_final_digest_records=None, subset_fallback=subset)
    return svg, report


def _frontier_committed_path_fixture() -> tuple[str, dict]:
    identifier = "generic-multi-loop-target"
    points = [
        (50.0 + 35.0 * math.cos(2.0 * math.pi * index / 99.0),
         50.0 + 35.0 * math.sin(2.0 * math.pi * index / 99.0))
        for index in range(99)
    ]
    path_data = " ".join(
        [f"M{points[0][0]:.6f} {points[0][1]:.6f}"]
        + [f"L{x:.6f} {y:.6f}" for x, y in points[1:]]
        + ["Z"])
    detail = _committed_curve_detail(
        identifier, final_element="path", emitted_element=None,
        anchors_before=319, anchors_after=99)
    detail.update({
        "actual_p95_error_percent": 0.213742908,
        "actual_max_error_percent": 0.259724519,
        "path_data_sha256": hashlib.sha256(
            path_data.encode("utf-8")).hexdigest(),
        "path_data_digest_scope": "utf8_svg_path_d_attribute",
        "selected_candidate_id": "compound_refinement_loop_02_next",
        "refinement_frontier": {
            "schema": "ai-vector-cleanroom.curve-refit-path-frontier/v1",
            "selection_basis": (
                "designer_anchors_then_anchors_then_segments_then_geometry_"
                "then_id"),
            "changed_loop_index": 2,
            "replacement_candidate_id": "loop_02_next",
            "base_candidate": {"anchors_after": 92},
        },
    })
    proposal = _committed_curve_proposal(detail)
    proposal["editability_enhancements"]["stages"]["curve_refit"][
        "proposal"]["transaction_frontier_refinement"] = {
            "schema": "ai-vector-cleanroom.curve-refit-path-frontier/v1",
            "target_id": identifier,
            "selected_candidate_id": "compound_refinement_loop_02_next",
            "selection_basis": (
                "designer_anchors_then_anchors_then_segments_then_geometry_"
                "then_id"),
            "optimization_basis": "geometry_only",
            "uses_colour_or_pixel_similarity": False,
        }
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      {_visual_metadata()}
      <path id="{identifier}" d="{path_data}" fill="#174f39"
            data-avc-curve-refit="geometry-budgeted"
            data-avc-anchors-before="319" data-avc-anchors-after="99"
            data-avc-designer-anchors="99"
            data-avc-error-budget-percent="0.25"
            data-avc-p95-error-percent="0.213742908"
            data-avc-max-error-percent="0.259724519"/>
    </svg>'''
    return svg, proposal


def _committed_native_fixture(kind: str) -> tuple[str, dict]:
    identifier = f"committed-{kind}"
    if kind == "circle":
        parameters = {"cx": "50", "cy": "50", "r": "30"}
    elif kind == "ellipse":
        parameters = {
            "cx": "50", "cy": "50", "rx": "34", "ry": "18",
            "transform": "rotate(27 50 50)",
        }
    else:  # pragma: no cover - test helper contract
        raise ValueError(kind)
    attributes = " ".join(
        f'{name}="{value}"' for name, value in parameters.items())
    detail = _committed_curve_detail(
        identifier, final_element=kind, emitted_element=kind,
        anchors_before=48, anchors_after=4)
    detail.update(_native_geometry_fields(kind, identifier, parameters))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      {_visual_metadata()}
      <{kind} id="{identifier}" {attributes} fill="#174f39"
              data-avc-curve-refit="geometry-budgeted"
              data-avc-anchors-before="48" data-avc-anchors-after="4"
              data-avc-designer-anchors="4"
              data-avc-error-budget-percent="0.25"
              data-avc-p95-error-percent="0.20"
              data-avc-max-error-percent="0.30"/>
    </svg>'''
    return svg, _committed_curve_proposal(detail)


class DesignerQualityAuditTests(unittest.TestCase):
    def test_native_gradient_economy_accepts_only_complete_numerical_proof(self):
        from designer_quality import _source_space_gradient_field_evidence

        report = _source_certified_field_metadata()
        geometry = report["gradient_details"][0]["validation"]["geometry"]
        geometry.update({
            "solver": "geometry_error_optimizer.optimize_compound_contours",
            "lexicographic_objective": _economy_objective(),
            "selection_evidence": {"identity_rollback_selected": False,
                                   "selected_candidate_id": "circle-candidate"},
            "evidence_scope": "gradient_reconstruction_against_original_source_ownership_mask",
            "anchor_count": 2, "designer_anchor_count": 4, "segment_count": 2,
            "primitive_first": True,
            "native_primitives": [{"element": "circle", "cx": 20, "cy": 20, "r": 16}],
            "native_whole_object_path": "M4 20 A16 16 0 1 0 36 20 A16 16 0 1 0 4 20 Z",
            "topology": {"topology_preserved": True, "components": 1, "holes": 0},
            "final_svg_consistency": {
                "status": "verified_unchanged", "final_drawable_id": "disc",
                "gradient_object_id": "disc-object", "final_element": "circle",
                "reconstruction_anchor_count": 2, "final_anchor_count": 1,
                "curve_refit_applied": False,
                "anchor_count_semantics": "native_svg_element_vs_designer_handles",
                "native_geometry": {"element": "circle", "cx": 20, "cy": 20, "r": 16}}})
        valid = _source_space_gradient_field_evidence(report, 1)["objects"][0]
        self.assertTrue(valid["economy_certificate_passed"], valid)
        for mutation in ("missing-path", "partial-native", "moved-native"):
            changed = copy.deepcopy(report)
            target = changed["gradient_details"][0]["validation"]["geometry"]
            if mutation == "missing-path":
                target.pop("native_whole_object_path")
            elif mutation == "partial-native":
                target["topology"]["holes"] = 3
            else:
                target["final_svg_consistency"]["native_geometry"]["r"] = 17
            result = _source_space_gradient_field_evidence(changed, 1)["objects"][0]
            self.assertFalse(result["economy_certificate_passed"], mutation)
            self.assertIn("final_svg_native_whole_object_proof_invalid", result["economy_failure_reasons"])

    def _audit(self, svg: str, proposal: dict | None = None) -> dict:
        # Keep the synthetic fixtures in-memory so this structural audit also
        # runs in read-only package verification environments.
        with mock.patch.object(Path, "read_bytes", return_value=svg.encode("utf-8")):
            return audit_designer_quality(
                Path("synthetic-designer-quality.svg"),
                proposal_metadata=proposal,
            )

    def test_clean_native_gradient_and_economical_curves_are_designer_ready(self):
        result = self._audit(_good_svg())

        self.assertEqual(result["schema"], AUDIT_SCHEMA)
        self.assertEqual(result["raster_visual_gate"]["status"], "accepted")
        self.assertEqual(result["gradient_object_gate"]["status"], "passed", result)
        self.assertEqual(result["curve_economy_gate"]["status"], "passed", result)
        self.assertEqual(result["designer_readiness_status"], "designer_ready")
        self.assertTrue(result["designer_ready"])
        self.assertFalse(result["raster_visual_designer_status_divergence"])
        resources = result["gradient_object_gate"]["gradient_resources"]
        self.assertEqual(resources["resource_count"], 2)
        self.assertEqual(resources["usage_count"], 2)
        self.assertEqual(resources["linear_resource_count"], 1)
        self.assertEqual(resources["radial_resource_count"], 1)
        self.assertEqual(
            result["curve_economy_gate"]["primitive_recovery"]
            ["native_designer_anchor_policy"]["circle"], 4)

    def test_source_space_field_proof_scopes_global_colour_heuristics(self):
        result = self._audit(
            _source_certified_field_svg(),
            _source_certified_field_metadata())
        gate = result["gradient_object_gate"]

        self.assertEqual(gate["status"], "passed", gate)
        self.assertTrue(
            gate["source_space_field_evidence"]["authoritative"])
        self.assertGreaterEqual(
            gate["metrics"]["severe_band_cluster_count"], 1)
        self.assertIn(
            "global_colour_clusters_superseded_by_source_space_field_evidence",
            gate["informational_reasons"])
        self.assertNotIn(
            "spatial_colour_band_fragmentation", gate["failure_reasons"])
        self.assertNotIn(
            "low_gradient_usage_among_many_chromatic_fragments",
            gate["failure_reasons"])

    def test_invalid_source_space_field_proof_fails_closed(self):
        cases = {
            "hard-edge": _source_certified_field_metadata(hard_edge=True),
            "overlap": _source_certified_field_metadata(overlap=1),
            "candidate-id": _source_certified_field_metadata(
                detail_candidate_id="different-candidate"),
        }
        for label, metadata in cases.items():
            with self.subTest(label=label):
                gate = self._audit(
                    _source_certified_field_svg(), metadata
                )["gradient_object_gate"]
                self.assertEqual(gate["status"], "failed")
                self.assertFalse(
                    gate["source_space_field_evidence"]["authoritative"])
                self.assertIn(
                    "source_space_gradient_field_evidence_invalid",
                    gate["failure_reasons"])

    def test_missing_geometry_evidence_warns_but_does_not_hard_fail(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">
          <defs><linearGradient id="g">
            <stop offset="0" stop-color="#124d35"/>
            <stop offset="1" stop-color="#8ac7ab"/>
          </linearGradient></defs>
          <path id="legacy" d="M10 80 C50 20 150 20 190 80 Z" fill="url(#g)"/>
        </svg>'''
        result = self._audit(svg)

        self.assertEqual(result["gradient_object_gate"]["status"], "manual_review")
        self.assertEqual(result["curve_economy_gate"]["status"], "manual_review")
        self.assertNotEqual(result["gradient_object_gate"]["status"], "failed")
        self.assertNotEqual(result["curve_economy_gate"]["status"], "failed")
        self.assertIn(
            "geometry_error_budget_evidence_missing",
            result["curve_economy_gate"]["warning_reasons"],
        )
        evidence = result["curve_economy_gate"][
            "geometry_error_budget_evidence"]
        self.assertEqual(evidence["applicable_object_count"], 1)
        self.assertEqual(evidence["missing_object_ids"], ["legacy"])

    def test_plain_solid_identity_path_does_not_require_geometry_evidence(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <path id="identity" d="M10 10 L90 10 L90 90 L10 90 Z" fill="#174f39"/>
        </svg>'''
        result = self._audit(svg)
        gate = result["curve_economy_gate"]

        self.assertEqual(gate["status"], "passed", result)
        self.assertNotIn(
            "geometry_error_budget_evidence_missing", gate["warning_reasons"])
        evidence = gate["geometry_error_budget_evidence"]
        self.assertEqual(evidence["applicable_object_count"], 0)
        self.assertEqual(evidence["missing_object_ids"], [])

    def test_curve_refit_claim_without_error_contract_still_warns(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <path id="claimed" d="M10 10 L90 10 L90 90 L10 90 Z" fill="#174f39"
                data-avc-curve-refit="geometry-budgeted"/>
        </svg>'''
        result = self._audit(svg)
        gate = result["curve_economy_gate"]

        self.assertEqual(gate["status"], "manual_review", result)
        evidence = gate["geometry_error_budget_evidence"]
        self.assertEqual(evidence["applicable_object_count"], 1)
        self.assertEqual(evidence["missing_object_ids"], ["claimed"])

    def test_native_gradient_primitive_without_error_contract_still_warns(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <defs><radialGradient id="g">
            <stop offset="0" stop-color="#f7d84a"/>
            <stop offset="1" stop-color="#c48310"/>
          </radialGradient></defs>
          <circle id="legacy-native" cx="50" cy="50" r="30" fill="url(#g)"/>
        </svg>'''
        result = self._audit(svg)
        gate = result["curve_economy_gate"]

        self.assertEqual(gate["status"], "manual_review", result)
        evidence = gate["geometry_error_budget_evidence"]
        self.assertEqual(evidence["applicable_object_count"], 1)
        self.assertEqual(evidence["missing_object_ids"], ["legacy-native"])

    def test_error_budget_p95_and_tail_are_hard_constraints(self):
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">
          {_visual_metadata()}
          <path id="p95-bad" d="M10 80 C50 10 100 10 140 80 Z" fill="#174f39"
                data-avc-error-budget-percent="0.5"
                data-avc-p95-error-percent="0.51"
                data-avc-max-error-percent="1.0"
                data-avc-designer-anchors="3"/>
          <path id="tail-bad" d="M160 80 C200 10 250 10 290 80 Z" fill="#4d8f73"
                data-avc-error-budget-percent="0.5"
                data-avc-p95-error-percent="0.4"
                data-avc-max-error-percent="1.51"
                data-avc-designer-anchors="3"/>
        </svg>'''
        result = self._audit(svg)
        gate = result["curve_economy_gate"]

        self.assertEqual(gate["status"], "failed")
        self.assertIn("geometry_p95_exceeds_error_budget", gate["failure_reasons"])
        self.assertIn("geometry_max_error_exceeds_tail_budget", gate["failure_reasons"])
        self.assertEqual(
            gate["geometry_error_budget_evidence"]["p95_violation_object_ids"],
            ["p95-bad"],
        )
        self.assertEqual(
            gate["geometry_error_budget_evidence"]["max_violation_object_ids"],
            ["tail-bad"],
        )
        self.assertFalse(
            gate["geometry_error_budget_evidence"]["contract"]
            ["visual_similarity_can_override"])
        self.assertTrue(result["raster_visual_gate"]["accepted"])

    def test_used_gradient_must_have_two_to_five_stops(self):
        stops = "".join(
            f'<stop offset="{index / 5:.1f}" stop-color="#{index + 1:02x}7040"/>'
            for index in range(6)
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <defs><radialGradient id="too-many">{stops}</radialGradient></defs>
          <circle id="disc" cx="50" cy="50" r="40" fill="url(#too-many)"
                  data-avc-error-budget-percent="0.5"
                  data-avc-p95-error-percent="0.4"
                  data-avc-max-error-percent="1.0"/>
        </svg>'''
        result = self._audit(svg)

        self.assertEqual(result["gradient_object_gate"]["status"], "failed")
        self.assertIn(
            "used_gradient_stop_count_outside_2_to_5",
            result["gradient_object_gate"]["failure_reasons"],
        )

    def test_one_gradient_object_cannot_remain_nine_trace_fragments(self):
        fragments = "".join(
            f'''<rect id="fragment-{index}" x="{index * 10}" y="10"
                       width="9" height="70" fill="url(#ramp)"
                       data-avc-gradient-object="one-painted-object"
                       data-avc-error-budget-percent="0.5"
                       data-avc-p95-error-percent="0.4"
                       data-avc-max-error-percent="1.0"
                       data-avc-designer-anchors="4"/>'''
            for index in range(9)
        )
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <defs><linearGradient id="ramp">
            <stop offset="0" stop-color="#174f39"/>
            <stop offset="1" stop-color="#9bc7b7"/>
          </linearGradient></defs>
          {fragments}
        </svg>'''
        result = self._audit(svg)
        gate = result["gradient_object_gate"]

        self.assertEqual(gate["status"], "failed")
        self.assertIn(
            "gradient_fill_split_into_excessive_objects",
            gate["failure_reasons"],
        )
        self.assertEqual(gate["metrics"]["max_drawables_per_gradient_fill"], 9)

    def test_compound_path_is_budgeted_per_loop_not_as_one_curve(self):
        commands = []
        for index in range(25):
            x = 10 + (index % 5) * 60
            y = 10 + (index // 5) * 60
            commands.append(
                f"M{x} {y} L{x + 20} {y} L{x + 20} {y + 20} "
                f"L{x} {y + 20} Z")
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 310 310">
          <path id="compound" d="{' '.join(commands)}" fill="#174f39"
                data-avc-error-budget-percent="0.5"
                data-avc-p95-error-percent="0.4"
                data-avc-max-error-percent="1.0"
                data-avc-designer-anchors="100"/>
        </svg>'''
        result = self._audit(svg)
        gate = result["curve_economy_gate"]

        self.assertEqual(gate["status"], "passed", result)
        metrics = gate["metrics"]
        self.assertEqual(metrics["max_nodes_in_single_path"], 100)
        self.assertEqual(metrics["loop_count"], 25)
        self.assertEqual(metrics["max_nodes_in_single_loop"], 4)
        self.assertEqual(metrics["loops_over_80_nodes"], 0)
        self.assertNotIn(
            "single_path_anchor_count_is_excessive", gate["failure_reasons"])

    def test_transaction_certified_retained_path_uses_geometry_contract(self):
        path_data = _dense_retained_path_data()
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          {_visual_metadata()}
          <path id="retained" d="{path_data}" fill="#174f39"/>
        </svg>'''
        result = self._audit(svg, _retained_curve_proposal(path_data))
        gate = result["curve_economy_gate"]

        self.assertGreater(gate["metrics"]["short_segment_ratio"], 0.90)
        self.assertEqual(gate["status"], "passed", gate)
        self.assertEqual(
            gate["heuristic_scope_metrics"]["path_count"], 0)
        evidence = gate["optimizer_economy_evidence"]["curve_refit"]
        self.assertTrue(evidence["authoritative"], evidence)
        self.assertEqual(
            evidence["certified_retained_identity_path_ids"], ["retained"])
        self.assertEqual(result["designer_readiness_status"], "designer_ready")

    def test_guarded_no_change_id_normalization_is_authoritative(self):
        path_data = _dense_retained_path_data()
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          {_visual_metadata()}
          <path id="retained" d="{path_data}" fill="#174f39"/>
        </svg>'''
        report = _retained_curve_proposal(path_data)
        curve = report["editability_enhancements"]["stages"]["curve_refit"]
        proposal = curve["proposal"]
        stable = proposal["stable_id_normalization"]
        stable["records"][0].update({
            "original_id_state": "missing_assigned",
            "original_id": None,
            "assignment_applied": True,
        })
        stable.update({
            "assigned_id_count": 1,
            "existing_id_preserved_count": 0,
        })
        final_record = {
            "id": "retained",
            "final_element": "path",
            "final_drawable_id": "retained",
            "path_data_sha256": hashlib.sha256(
                path_data.encode("utf-8")).hexdigest(),
            "path_data_digest_scope": "utf8_svg_path_d_attribute",
        }
        topology = {
            "accepted": True,
            "policy": "bidirectional_ink_topology_guard",
            "colour_similarity_excluded": True,
            "comparisons": {
                key: {"value": 100.0, "minimum": 99.0, "accepted": True}
                for key in (
                    "ink_recall_percent", "ink_precision_percent",
                    "ink_f1_percent")
            },
        }
        render = {
            "external_render_check": "completed",
            "ink_recall_percent": 100.0,
            "ink_precision_percent": 100.0,
            "ink_f1_percent": 100.0,
        }
        proposal["transaction_precommit_final_digest_records"] = [
            final_record]
        proposal["transaction_postcommit_final_digest_records"] = [
            final_record]
        curve.update({
            "status": "committed",
            "reason": None,
            "commit_scope": "stable_id_normalization_only",
            "after_svg_sha256": "b" * 64,
            "render_guard": render,
            "renderer_topology_guard": topology,
            "source_guard": {"accepted": True},
            "precommit_final_digest_records": [final_record],
            "postcommit_final_digest_records": [final_record],
        })
        identity = curve["identity_normalization"]
        identity.update({
            "status": "committed",
            "commit_scope": "stable_id_normalization_only",
            "postcommit_svg_sha256": "b" * 64,
            "assigned_id_count": 1,
            "existing_id_preserved_count": 0,
            "records": copy.deepcopy(stable["records"]),
            "precommit_final_digest_records": [final_record],
            "postcommit_final_digest_records": [final_record],
            "baseline_validation": {
                "status": "verified",
                "accepted": True,
                "renderer_topology_guard": {"accepted": True},
                "source_guard": {"accepted": True},
                "gradient_geometry_guard": {"accepted": True},
            },
        })

        gate = self._audit(svg, report)["curve_economy_gate"]
        evidence = gate["optimizer_economy_evidence"]["curve_refit"]
        self.assertEqual(gate["status"], "passed", gate)
        self.assertTrue(evidence["authoritative"], evidence)
        self.assertEqual(evidence[
            "certified_retained_identity_path_ids"], ["retained"])

    def test_curve_economy_certificate_tampering_fails_closed(self):
        path_data = _dense_retained_path_data()
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          {_visual_metadata()}
          <path id="retained" d="{path_data}" fill="#174f39"/>
        </svg>'''
        cases = []
        digest_tamper = _retained_curve_proposal(path_data)
        digest_tamper["editability_enhancements"]["stages"]["curve_refit"][
            "proposal"]["evaluated_but_retained"][0]["path_data_sha256"] = "0" * 64
        cases.append(("path-digest", digest_tamper))
        economy_tamper = _retained_curve_proposal(path_data)
        economy_tamper["editability_enhancements"]["stages"]["curve_refit"][
            "proposal"]["evaluated_but_retained"][0][
                "optimizer_selected_economy"]["designer_anchor_count"] = 59
        cases.append(("better-candidate", economy_tamper))
        integrity_tamper = _retained_curve_proposal(path_data)
        integrity_tamper["editability_enhancements"]["stages"]["curve_refit"][
            "proposal"]["evaluation_evidence_integrity"][
                "all_optimizer_results_accounted"] = False
        cases.append(("integrity", integrity_tamper))

        for label, proposal in cases:
            with self.subTest(label=label):
                gate = self._audit(svg, copy.deepcopy(proposal))[
                    "curve_economy_gate"]
                self.assertEqual(gate["status"], "failed", gate)
                self.assertIn(
                    "curve_optimizer_economy_certificate_invalid",
                    gate["failure_reasons"],
                )
                self.assertFalse(
                    gate["optimizer_economy_evidence"]["curve_refit"]
                    ["authoritative"])

    def test_complete_rollback_is_manual_with_no_economy_certificate(self):
        for named in (False, True):
            with self.subTest(named=named):
                svg, report = _rolled_back_curve_fixture(named=named)
                gate = self._audit(svg, report)["curve_economy_gate"]
                evidence = gate["optimizer_economy_evidence"]["curve_refit"]
                self.assertEqual(gate["status"], "manual_review", gate)
                self.assertFalse(evidence["available"])
                self.assertFalse(evidence["authoritative"])
                self.assertEqual(evidence["certified_path_count"], 0)
                self.assertEqual(evidence["failure_reasons"], [])
                self.assertTrue(evidence["source_path_data_restored"])
                self.assertEqual(gate["heuristic_scope_metrics"]["path_count"], 1)
                self.assertIn("curve_refit_retained_source_paths_economy_unavailable",
                              gate["warning_reasons"])

    def test_rollback_does_not_exempt_dense_paths_or_claim_presentation_identity(self):
        svg, report = _rolled_back_curve_fixture(dense=True)
        gate = self._audit(svg, report)["curve_economy_gate"]
        self.assertEqual(gate["status"], "failed", gate)
        self.assertNotIn("curve_optimizer_economy_certificate_invalid", gate["failure_reasons"])
        self.assertGreater(len(gate["failure_reasons"]), 0)
        self.assertEqual(gate["heuristic_scope_metrics"]["certified_path_count_excluded"], 0)
        svg, report = _rolled_back_curve_fixture()
        for attribute in ('transform="translate(10,0)"', 'display="none"', 'opacity="0.3"'):
            result = self._audit(svg.replace('<path ', '<path '+attribute+' ', 1), report)
            evidence = result["curve_economy_gate"]["optimizer_economy_evidence"]["curve_refit"]
            self.assertEqual(evidence["certified_path_count"], 0)
            self.assertTrue(evidence["presentation_not_verified"])
            self.assertEqual(evidence["rollback_identity_scope"],
                "restored_source_path_data_and_original_ids_not_full_presentation")
            self.assertFalse(result["designer_ready"])

    def test_rollback_missing_or_tampered_path_identity_stays_invalid(self):
        svg, report = _rolled_back_curve_fixture()
        cases = [
            ("changed-d", svg.replace("L90 30", "L89 30"), copy.deepcopy(report)),
            ("assigned-id-kept", svg.replace('<path ', '<path id="committed-path" ', 1), copy.deepcopy(report)),
        ]
        for label in ("after-hash", "missing-live-proof", "ordinal", "source-digest",
                      "missing-uncertified", "missing-rejection", "all-guards-pass",
                      "malformed-rollback-ids"):
            changed = copy.deepcopy(report)
            curve = changed["editability_enhancements"]["stages"]["curve_refit"]
            if label == "after-hash": curve["after_svg_sha256"] = "0"*64
            elif label == "missing-live-proof": curve.pop("live_svg_unchanged")
            elif label == "ordinal":
                for key in (curve["identity_normalization"], curve["proposal"]["stable_id_normalization"]):
                    key["records"][0]["global_path_ordinal_1_based"] = 2
            elif label == "source-digest":
                for key in (curve["identity_normalization"], curve["proposal"]["stable_id_normalization"]):
                    key["records"][0]["source_path_data_sha256"] = "0"*64
            elif label == "missing-uncertified": curve["proposal"]["uncertified_evaluations"] = []
            elif label == "missing-rejection": curve["subset_fallback"].pop("full_candidate_validation")
            elif label == "all-guards-pass":
                curve["subset_fallback"]["full_candidate_validation"]["renderer_topology_guard"]["accepted"] = True
            else: curve["proposal"]["uncertified_transaction_rollback_ids"] = [{}]
            cases.append((label, svg, changed))
        for label, changed_svg, changed_report in cases:
            with self.subTest(label=label):
                gate = self._audit(changed_svg, changed_report)["curve_economy_gate"]
                self.assertIn("curve_optimizer_economy_certificate_invalid", gate["failure_reasons"], gate)
                self.assertFalse(gate["optimizer_economy_evidence"]["curve_refit"]["source_path_data_restored"])

    def test_committed_path_certificate_matches_exact_final_svg_path(self):
        svg, proposal = _committed_path_fixture()
        result = self._audit(svg, proposal)
        gate = result["curve_economy_gate"]
        evidence = gate["optimizer_economy_evidence"]["curve_refit"]

        self.assertEqual(gate["status"], "passed", gate)
        self.assertTrue(evidence["authoritative"], evidence)
        self.assertEqual(
            evidence["certified_refit_path_ids"], ["committed-path"])
        self.assertEqual(result["designer_readiness_status"], "designer_ready")

    def test_frontier_commit_is_certified_and_provenance_tampering_fails_closed(
            self):
        svg, proposal = _frontier_committed_path_fixture()
        result = self._audit(svg, proposal)
        gate = result["curve_economy_gate"]
        evidence = gate["optimizer_economy_evidence"]["curve_refit"]

        self.assertEqual(gate["metrics"]["max_nodes_in_single_path"], 99)
        self.assertEqual(gate["status"], "passed", gate)
        self.assertTrue(evidence["authoritative"], evidence)
        self.assertEqual(
            evidence["certified_refit_path_ids"],
            ["generic-multi-loop-target"])
        self.assertEqual(gate["heuristic_scope_metrics"]["path_count"], 0)

        tampered = copy.deepcopy(proposal)
        tampered["editability_enhancements"]["stages"]["curve_refit"][
            "proposal"]["transaction_frontier_refinement"][
                "target_id"] = "identity-path"
        tampered_gate = self._audit(svg, tampered)["curve_economy_gate"]
        tampered_evidence = tampered_gate[
            "optimizer_economy_evidence"]["curve_refit"]
        self.assertEqual(tampered_gate["status"], "failed", tampered_gate)
        self.assertFalse(tampered_evidence["authoritative"])
        self.assertIn(
            "curve_refit_frontier_refinement_invalid",
            tampered_evidence["failure_reasons"])

    def test_committed_certificate_excludes_colour_only_render_rejection(self):
        svg, proposal = _committed_path_fixture()
        curve = proposal["editability_enhancements"]["stages"]["curve_refit"]
        curve["render_guard"].update({
            "accepted": False,
            "score_percent": 98.44,
            "color_similarity_percent": 95.55,
        })

        gate = self._audit(svg, proposal)["curve_economy_gate"]
        evidence = gate["optimizer_economy_evidence"]["curve_refit"]

        self.assertFalse(curve["render_guard"]["accepted"])
        self.assertTrue(curve["renderer_topology_guard"]["accepted"])
        self.assertTrue(curve["source_guard"]["accepted"])
        self.assertEqual(gate["status"], "passed", gate)
        self.assertTrue(evidence["authoritative"], evidence)
        self.assertNotIn(
            "curve_refit_render_guard_invalid", evidence["failure_reasons"])

    def test_committed_certificate_guards_remain_fail_closed(self):
        cases = []

        topology_svg, topology_proposal = _committed_path_fixture()
        topology_curve = topology_proposal[
            "editability_enhancements"]["stages"]["curve_refit"]
        topology_curve["render_guard"]["ink_recall_percent"] = 98.99
        topology_curve["renderer_topology_guard"]["comparisons"][
            "ink_recall_percent"].update({
                "value": 98.99,
                # Keep the claimed pass bits forged high: the certificate
                # must independently enforce the 99% evidence value.
                "accepted": True,
            })
        topology_curve["renderer_topology_guard"]["accepted"] = True
        cases.append((
            "topology-below-99", topology_svg, topology_proposal,
            "curve_refit_topology_guard_invalid"))

        source_svg, source_proposal = _committed_path_fixture()
        source_proposal["editability_enhancements"]["stages"]["curve_refit"][
            "source_guard"]["accepted"] = False
        cases.append((
            "source-rejected", source_svg, source_proposal,
            "curve_refit_source_guard_invalid"))

        renderer_svg, renderer_proposal = _committed_path_fixture()
        renderer_proposal["editability_enhancements"]["stages"]["curve_refit"][
            "render_guard"]["external_render_check"] = "unavailable"
        cases.append((
            "external-renderer-unavailable", renderer_svg, renderer_proposal,
            "curve_refit_render_guard_invalid"))

        for label, candidate_svg, candidate_proposal, expected_failure in cases:
            with self.subTest(label=label):
                gate = self._audit(candidate_svg, candidate_proposal)[
                    "curve_economy_gate"]
                evidence = gate[
                    "optimizer_economy_evidence"]["curve_refit"]
                self.assertEqual(gate["status"], "failed", gate)
                self.assertFalse(evidence["authoritative"], evidence)
                self.assertIn(
                    expected_failure, evidence["failure_reasons"], evidence)

    def test_optimizer_contract_accepts_zero_without_relaxing_nonzero_bounds(self):
        svg, zero_proposal = _committed_path_fixture()
        svg = svg.replace(
            'data-avc-p95-error-percent="0.20"',
            'data-avc-p95-error-percent="0.0"').replace(
                'data-avc-max-error-percent="0.30"',
                'data-avc-max-error-percent="0.0"')
        zero_detail = zero_proposal["editability_enhancements"]["stages"][
            "curve_refit"]["proposal"]["details"][0]
        for field in (
                "actual_p95_error_percent", "actual_max_error_percent",
                "over_budget_share", "salient_corner_max_percent"):
            zero_detail[field] = 0.0

        zero_gate = self._audit(svg, zero_proposal)["curve_economy_gate"]
        zero_evidence = zero_gate[
            "optimizer_economy_evidence"]["curve_refit"]
        self.assertEqual(zero_gate["status"], "passed", zero_gate)
        self.assertTrue(zero_evidence["authoritative"], zero_evidence)

        cases = (
            ("actual_p95_error_percent", 0.26,
             "p95_error_contract_failed"),
            ("actual_max_error_percent", 0.76,
             "maximum_error_contract_failed"),
            ("over_budget_share", 0.06,
             "over_budget_share_contract_failed"),
            ("salient_corner_max_percent", 0.51,
             "salient_corner_contract_failed"),
        )
        for field, value, expected_failure in cases:
            with self.subTest(field=field):
                invalid_proposal = copy.deepcopy(zero_proposal)
                invalid_proposal["editability_enhancements"]["stages"][
                    "curve_refit"]["proposal"]["details"][0][field] = value
                gate = self._audit(svg, invalid_proposal)[
                    "curve_economy_gate"]
                evidence = gate[
                    "optimizer_economy_evidence"]["curve_refit"]
                self.assertEqual(gate["status"], "failed", gate)
                self.assertFalse(evidence["authoritative"], evidence)
                self.assertTrue(
                    any(expected_failure in reason
                        for reason in evidence["failure_reasons"]),
                    evidence,
                )

    def test_committed_path_identity_tampering_fails_closed(self):
        svg, proposal = _committed_path_fixture()
        cases: list[tuple[str, str, dict, str]] = []

        missing_digest = copy.deepcopy(proposal)
        missing_digest["editability_enhancements"]["stages"]["curve_refit"][
            "proposal"]["details"][0].pop("path_data_sha256")
        cases.append((
            "missing-digest", svg, missing_digest,
            "committed_path_digest_mismatch"))

        changed_path = svg.replace("L90 90", "L80 90")
        cases.append((
            "changed-final-d", changed_path, copy.deepcopy(proposal),
            "committed_path_digest_mismatch"))

        changed_final_id = copy.deepcopy(proposal)
        changed_final_id["editability_enhancements"]["stages"][
            "curve_refit"]["proposal"]["details"][0][
                "final_drawable_id"] = "another-path"
        cases.append((
            "changed-final-id", svg, changed_final_id,
            "committed_final_drawable_id_mismatch"))

        changed_element = copy.deepcopy(proposal)
        changed_element["editability_enhancements"]["stages"]["curve_refit"][
            "proposal"]["details"][0]["final_element"] = "circle"
        cases.append((
            "changed-final-element", svg, changed_element,
            "committed_final_element_mismatch"))

        duplicate_id_svg = svg.replace(
            "</svg>",
            '<rect id="committed-path" x="1" y="1" width="2" height="2"/>'
            "</svg>")
        cases.append((
            "duplicate-final-id", duplicate_id_svg, copy.deepcopy(proposal),
            "committed_final_drawable_id_not_unique"))

        for label, candidate_svg, candidate_proposal, expected in cases:
            with self.subTest(label=label):
                gate = self._audit(candidate_svg, candidate_proposal)[
                    "curve_economy_gate"]
                evidence = gate["optimizer_economy_evidence"]["curve_refit"]
                self.assertEqual(gate["status"], "failed", gate)
                self.assertFalse(evidence["authoritative"], evidence)
                self.assertTrue(
                    any(expected in reason
                        for reason in evidence["failure_reasons"]),
                    evidence,
                )

    def test_committed_native_certificates_match_exact_final_geometry(self):
        for kind in ("circle", "ellipse"):
            with self.subTest(kind=kind):
                svg, proposal = _committed_native_fixture(kind)
                result = self._audit(svg, proposal)
                gate = result["curve_economy_gate"]
                evidence = gate["optimizer_economy_evidence"]["curve_refit"]

                self.assertEqual(gate["status"], "passed", gate)
                self.assertTrue(evidence["authoritative"], evidence)
                self.assertEqual(evidence["invalid_detail_ids"], [])
                self.assertEqual(
                    result["designer_readiness_status"], "designer_ready")

    def test_committed_native_geometry_tampering_fails_closed(self):
        for kind in ("circle", "ellipse"):
            svg, proposal = _committed_native_fixture(kind)
            detail_path = [
                "editability_enhancements", "stages", "curve_refit",
                "proposal", "details",
            ]

            parameter_tamper = copy.deepcopy(proposal)
            details = parameter_tamper
            for key in detail_path:
                details = details[key]
            parameter_name = "r" if kind == "circle" else "rx"
            details[0]["native_parameters"][parameter_name] = "999"

            digest_tamper = copy.deepcopy(proposal)
            details = digest_tamper
            for key in detail_path:
                details = details[key]
            details[0]["geometry_sha256"] = "0" * 64

            if kind == "circle":
                transformed_svg = svg.replace(
                    'r="30"', 'r="30" transform="translate(1 0)"')
            else:
                transformed_svg = svg.replace(
                    'transform="rotate(27 50 50)"',
                    'transform="rotate(28 50 50)"')
            renamed_svg = svg.replace(
                f'id="committed-{kind}"', f'id="replayed-{kind}"')

            cases = [
                ("parameters", svg, parameter_tamper,
                 "committed_native_parameters_mismatch"),
                ("transform", transformed_svg, copy.deepcopy(proposal),
                 ("committed_circle_transform_outside_digest_scope"
                  if kind == "circle" else
                  "committed_native_parameters_mismatch")),
                ("digest", svg, digest_tamper,
                 "committed_native_digest_mismatch"),
                ("id", renamed_svg, copy.deepcopy(proposal),
                 "committed_final_drawable_id_not_unique"),
            ]
            for label, candidate_svg, candidate_proposal, expected in cases:
                with self.subTest(kind=kind, tamper=label):
                    gate = self._audit(candidate_svg, candidate_proposal)[
                        "curve_economy_gate"]
                    evidence = gate[
                        "optimizer_economy_evidence"]["curve_refit"]
                    self.assertEqual(gate["status"], "failed", gate)
                    self.assertFalse(evidence["authoritative"], evidence)
                    self.assertTrue(
                        any(expected in reason
                            for reason in evidence["failure_reasons"]),
                        evidence,
                    )

    def test_native_circle_always_counts_as_four_designer_anchors(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <circle id="native" cx="50" cy="50" r="30" fill="#174f39"
                  data-avc-curve-refit="geometry-budgeted"
                  data-avc-error-budget-percent="0.5"
                  data-avc-p95-error-percent="0.4"
                  data-avc-max-error-percent="1.0"/>
        </svg>'''
        result = self._audit(svg)
        gate = result["curve_economy_gate"]

        self.assertEqual(gate["status"], "passed", result)
        self.assertEqual(
            gate["primitive_recovery"]["native_circle_designer_anchor_count"], 4)
        self.assertEqual(
            gate["geometry_error_budget_evidence"]["complete_object_count"], 1)

    def test_visual_acceptance_does_not_override_fragment_and_anchor_failures(self):
        result = self._audit(_bad_svg())

        self.assertTrue(result["raster_visual_gate"]["accepted"])
        self.assertEqual(result["gradient_object_gate"]["status"], "failed")
        self.assertEqual(result["curve_economy_gate"]["status"], "failed")
        self.assertEqual(
            result["designer_readiness_status"], "manual_rework_required")
        self.assertFalse(result["designer_ready"])
        self.assertTrue(result["raster_visual_designer_status_divergence"])
        gradient_metrics = result["gradient_object_gate"]["metrics"]
        self.assertGreaterEqual(gradient_metrics["severe_band_cluster_count"], 1)
        curve_metrics = result["curve_economy_gate"]["metrics"]
        self.assertGreater(curve_metrics["short_segment_ratio"], 0.90)
        self.assertGreater(curve_metrics["near_collinear_cubic_ratio"], 0.90)

    def test_flat_native_art_is_not_falsely_failed_for_having_no_gradient(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 200">
          <rect x="20" y="20" width="80" height="60" fill="#163f30"/>
          <circle cx="210" cy="90" r="45" fill="#d8ab32"/>
          <line x1="40" y1="160" x2="260" y2="160" stroke="#163f30"/>
        </svg>'''
        result = self._audit(svg)

        gate = result["gradient_object_gate"]
        self.assertEqual(gate["status"], "passed", result)
        self.assertFalse(gate["applicable"])
        self.assertEqual(result["designer_readiness_status"], "designer_ready")

    def test_overanchored_circle_fails_even_with_another_native_circle(self):
        points = []
        for index in range(16):
            angle = 2.0 * math.pi * index / 16.0
            points.append((
                100.0 + 60.0 * math.cos(angle),
                100.0 + 60.0 * math.sin(angle),
            ))
        path = "M" + " L".join(
            f"{x:.6f} {y:.6f}" for x, y in points) + " Z"
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 200">
          <circle id="already-native" cx="25" cy="25" r="10" fill="#163d2b"/>
          <path id="pixel-circle" d="{path}" fill="#d8ab32"/>
        </svg>'''
        result = self._audit(svg)

        gate = result["curve_economy_gate"]
        self.assertEqual(gate["status"], "failed")
        self.assertIn(
            "circle_like_paths_have_excessive_anchors",
            gate["failure_reasons"])
        primitive = gate["primitive_recovery"]
        self.assertEqual(primitive["overanchored_circle_path_count"], 1)
        self.assertEqual(
            primitive["overanchored_circle_paths"], ["pixel-circle"])

    def test_leaf_silhouettes_are_not_misreported_as_circles(self):
        right_leaf = (
            "M663 1058 C663.66 1058.33 664.32 1058.66 665 1059 "
            "C664.03 1059.79 663.07 1060.58 662.07 1061.39 "
            "C651.37 1070.27 641.35 1079.86 636 1093 "
            "C636 1093.99 636 1094.98 636 1096 "
            "C635.01 1095.67 634.02 1095.34 633 1095 "
            "C633.99 1092.36 634.98 1089.72 636 1087 "
            "C634.68 1088.32 633.36 1089.64 632 1091 "
            "C631 1090 631 1090 630.94 1086.94 "
            "C630.96 1085.97 630.98 1085 631 1084 "
            "C631.66 1084.99 632.32 1085.98 633 1087 "
            "C633.87 1085.66 634.73 1084.32 635.62 1082.94 "
            "C642.9 1072.11 651.86 1064.6 663 1058 Z")
        left_leaf = (
            "M590 1061 C602.46 1064.65 615.4 1074.79 622 1086 "
            "C621.67 1086.99 621.34 1087.98 621 1089 "
            "C621.33 1089.99 621.66 1090.98 622 1092 "
            "C621.34 1092.66 620.68 1093.32 620 1094 "
            "C619.34 1093.34 618.68 1092.68 618 1092 "
            "C618.66 1092 619.32 1092 620 1092 "
            "C613.81 1080.76 606.57 1071.9 595.35 1065.44 "
            "C593 1064 593 1064 590 1061 Z")
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 1254 1254">
          <path id="right-leaf" d="{right_leaf}" fill="#d7e3dc"/>
          <path id="left-leaf" d="{left_leaf}" fill="#d7e3dc"/>
        </svg>'''
        result = self._audit(svg)
        primitive = result["curve_economy_gate"]["primitive_recovery"]
        self.assertEqual(primitive["near_circle_path_candidate_count"], 0)
        self.assertNotIn(
            "circle_like_paths_have_excessive_anchors",
            result["curve_economy_gate"]["failure_reasons"],
        )

    def test_arc_radii_are_not_misread_as_absolute_bbox_coordinates(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 800 800">
          <path id="high-arc" d="M500 500 A10 20 0 0 1 520 500"
                fill="none" stroke="#174f39"/>
        </svg>'''
        result = self._audit(svg)
        path = result["curve_economy_gate"]["highest_anchor_paths"][0]
        self.assertGreaterEqual(path["bbox"][0], 490.0)
        self.assertGreaterEqual(path["bbox"][1], 470.0)
        self.assertLessEqual(path["bbox"][2], 530.0)
        self.assertLessEqual(path["bbox"][3], 530.0)

    def test_closed_anchor_count_handles_explicit_and_implicit_return(self):
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <path id="explicit" d="M10 10 L30 10 L30 30 L10 30 L10 10 Z"
                fill="#174f39"/>
          <path id="implicit" d="M50 10 L80 10 L65 35 Z" fill="#174f39"/>
        </svg>'''
        result = self._audit(svg)
        paths = {item["id"]: item for item in
                 result["curve_economy_gate"]["highest_anchor_paths"]}
        self.assertEqual(paths["explicit"]["node_count"], 4)
        self.assertEqual(paths["implicit"]["node_count"], 3)

    def test_proposal_metadata_reports_selectable_and_unresolved_coverage(self):
        svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">' + ''.join(
            f'<rect id="n{index}" x="{index * 9}" y="10" width="8" height="40" '
            f'fill="#{20 + index:02x}7040"/>' for index in range(10)
        ) + '</svg>'
        proposal = {
            "drawable_count": 10,
            "actual_dom_groups": [{
                "id": "selectable", "mode": "actual-dom", "member_count": 2,
                "paint_count": 2, "node_ids": ["n0", "n1"],
                "reasons": ["cross-paint-overlap"],
            }],
            "manifest_only_groups": [{
                "id": "unresolved", "mode": "manifest-only", "member_count": 8,
                "paint_count": 4, "node_ids": [f"n{index}" for index in range(2, 10)],
                "reasons": ["fragment-proximity", "cross-paint-overlay"],
                "not_applied_reason": "paint-order-inversion",
            }],
        }
        result = self._audit(svg, proposal)

        coverage = result["gradient_object_gate"]["proposal_coverage"]
        self.assertTrue(coverage["available"])
        self.assertEqual(coverage["candidate_member_coverage_rate"], 1.0)
        self.assertEqual(coverage["selectable_member_coverage_rate"], 0.2)
        self.assertEqual(coverage["unresolved_member_coverage_rate"], 0.8)
        self.assertEqual(result["gradient_object_gate"]["status"], "failed")

    def test_ali_beta5_reference_is_not_designer_ready_despite_visual_pass(self):
        path = ROOT / "validation" / "ali_gradient_golden" / "beta5_reference.svg"
        if not path.is_file():
            self.skipTest("Private tea-logo fixture is excluded from the public source package")

        result = audit_designer_quality(path)

        self.assertEqual(result["raster_visual_gate"]["status"], "accepted")
        self.assertEqual(
            result["designer_readiness_status"], "manual_rework_required")
        self.assertTrue(result["raster_visual_designer_status_divergence"])
        self.assertEqual(result["document_inventory"]["drawable_count"], 202)
        gradient = result["gradient_object_gate"]
        self.assertEqual(gradient["gradient_resources"]["resource_count"], 8)
        self.assertEqual(gradient["metrics"]["gradient_fill_drawable_count"], 9)
        self.assertEqual(gradient["metrics"]["solid_fill_drawable_count"], 176)
        self.assertIn(
            "low_gradient_usage_among_many_chromatic_fragments",
            gradient["failure_reasons"],
        )
        curve = result["curve_economy_gate"]
        self.assertEqual(curve["metrics"]["path_count"], 186)
        self.assertEqual(curve["metrics"]["node_count"], 3173)
        self.assertEqual(curve["metrics"]["cubic_segment_count"], 3135)
        self.assertGreater(curve["metrics"]["short_segment_ratio"], 0.45)
        self.assertGreater(curve["metrics"]["near_collinear_cubic_ratio"], 0.80)
        self.assertGreater(curve["metrics"]["anchors_per_100_user_units"], 8.0)

        # Public audit output must be safe to embed in report.json.
        json.dumps(result, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
