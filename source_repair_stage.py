"""Transactional source reconstruction, distinct from simplifying a prior SVG.

An earlier trace is evidence about ownership, not a ground-truth contour. This
stage may repair its errors only when native source-space checks support the
change. Ordinary SVG simplification keeps its existing topology contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


SCHEMA = "ai-vector-cleanroom.source-repair-stage/v1"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _path_sha(text, identifier):
    found = [node for node in ET.fromstring(text).iter()
             if node.get('id') == identifier]
    if len(found) != 1:
        raise ValueError('ambiguous_source_repair_drawable')
    return _sha(found[0].get('d', '').encode('utf-8'))


def _small_source_contact_tail(guard):
    """Permit one extra repair attempt, never exempt its final source check."""
    kinds={'new_local_color_error','new_paint_on_source_empty','new_color_on_source_empty'}
    reasons=set(guard.get('reasons') or [])
    if (guard.get('source_boundary_evidence', {}).get('verified') is not True
            or guard.get('localized_defects_truncated')
            or not reasons or not reasons<=kinds):
        return False
    totals = guard.get('defect_totals') or []
    regions = guard.get('localized_defects') or []
    if not totals or not regions:
        return False
    for item in [*totals, *regions]:
        if (item.get('kind') not in kinds
                or type(item.get('pixels')) is not int or item['pixels'] <= 0):
            return False
    count = sum(item['pixels'] for item in totals)
    return 0 < count <= 4 and sum(item['pixels'] for item in regions) == count


def _paint_is_referenced(root, identifier):
    pattern = re.compile(r'url\(\s*[\"\']?#' + re.escape(identifier) + r'[\"\']?\s*\)')
    return any(pattern.search(value) or (key.rsplit('}', 1)[-1] == 'href' and value == '#' + identifier)
               for node in root.iter() for key, value in node.attrib.items())


def _prune_replaced_paint(svg_text, identifier):
    root = ET.fromstring(svg_text)
    if _paint_is_referenced(root, identifier):
        return svg_text, False
    for parent in root.iter():
        for node in list(parent):
            if (node.get('id') == identifier
                    and node.tag.rsplit('}', 1)[-1] in ('linearGradient', 'radialGradient')):
                parent.remove(node)
    return ET.tostring(root, encoding='unicode'), True


def _is_partial_paint(detail):
    validation = detail.get('validation') if isinstance(detail, dict) else None
    geometry = validation.get('geometry') if isinstance(validation, dict) else None
    proof = geometry.get('source_paint_only') if isinstance(geometry, dict) else None
    return isinstance(proof, dict) and proof.get('partial_selection') is True


def project_gradient_report(stage, details, repair_report):
    """Retain selection history while reporting only paint still in the scene."""
    result = copy.deepcopy(stage)
    removed = repair_report.get('replaced_gradient_details', [])
    if not removed:
        return result
    removed_ids = {d.get('candidate_id') for d in removed}
    complete_details = [d for d in details if not _is_partial_paint(d)]
    partial_details = [d for d in details if _is_partial_paint(d)]
    retained_ids = {d.get('candidate_id') for d in complete_details}
    partial_ids = {d.get('candidate_id') for d in partial_details}
    if (retained_ids & partial_ids or len(retained_ids) != len(complete_details)
            or len(partial_ids) != len(partial_details)
            or None in retained_ids or None in partial_ids
            or '' in retained_ids or '' in partial_ids
            or removed_ids & (retained_ids | partial_ids)):
        raise ValueError('source_flat_selection_projection_mismatch')
    result['source_repair_projection'] = {
        'original_summary': copy.deepcopy(result.get('summary', {})),
        'replaced_candidate_ids': sorted(i for i in removed_ids if i),
        'retained_partial_candidate_ids': sorted(partial_ids),
        'partial_paint_remains_manual_review': bool(partial_details),
        'reason': 'native_source_core_and_edge_favour_solid',
    }
    for row in result.get('decisions', []):
        if (row.get('status') in ('selected', 'partial_paint_selected')
                and row.get('candidate_id') in removed_ids):
            row['previous_status'] = row['status']
            row['status'] = 'replaced_with_source_solid'
    actual = {r.get('candidate_id') for r in result.get('decisions', [])
              if r.get('status') == 'selected'}
    if actual != retained_ids:
        raise ValueError('source_flat_selection_projection_mismatch')
    actual_partial = {r.get('candidate_id') for r in result.get('decisions', [])
                      if r.get('status') == 'partial_paint_selected'}
    if actual_partial != partial_ids:
        raise ValueError('source_flat_partial_selection_projection_mismatch')
    result.setdefault('summary', {})['objects_selected'] = len(complete_details)
    result['summary']['partial_paint_fields'] = len(partial_details)
    return result


def attempt_source_repairs(svg_path, source_png, reference_png, gradient_info,
                           *, error_budget_percent=0.25,
                           maximum_seconds=240.0, maximum_objects=6):
    """Commit only source-verified candidates; return report and new paint proof.

    The original file and all evidence remain unchanged if this stage fails.
    Each fit is finite; no new candidate starts after the search deadline.
    Final validation and the currently running fit can finish after it.
    """
    from source_edge_reconstruction import propose_source_edge_reconstruction
    from source_edge_contacts import preserve_source_edge_contacts, _boundary_outlier_boxes
    from source_light_cleanup import (propose_light_fill_cleanup,
                                      apply_light_fill_candidate)
    from source_flat_paint import propose_source_flat_paint
    from source_scene_guard import validate_source_scene, validate_source_scene_chain
    from svg_postprocess import atomic_replace_bytes

    svg_path = Path(svg_path)
    original_bytes = svg_path.read_bytes()
    original = original_bytes.decode('utf-8')
    details = copy.deepcopy(list(gradient_info or ()))
    partial_paint_ids = {d.get('id') for d in details if _is_partial_paint(d)}
    partial_owner_ids = {identifier for d in details if _is_partial_paint(d)
                         for identifier in (d.get('candidate_id'), d.get('proposal_id'),
                             d['validation']['geometry']['source_paint_only'].get('gradient_object_id'))
                         if identifier}
    report = {'schema': SCHEMA, 'status': 'unchanged', 'committed': [],
              'attempts': [], 'before_svg_sha256': _sha(original_bytes),
              'scope': 'source_reconstruction_not_prior_svg_simplification',
              'human_time_saving_validated': False,
              'search_limit_seconds': maximum_seconds}
    started = time.monotonic()
    source = np.asarray(Image.open(source_png).convert('RGBA'))
    reference = np.asarray(Image.open(reference_png).convert('RGBA'))
    current = original
    transactions = []
    try:
        with tempfile.TemporaryDirectory(prefix='avc-source-repair-') as temp:
            before = Path(temp) / 'before.svg'
            after = Path(temp) / 'candidate.svg'

            def check(candidate, roi=None, geometry=None):
                before.write_bytes(current.encode('utf-8'))
                after.write_bytes(candidate.encode('utf-8'))
                return validate_source_scene(before, after, source_png,
                    processed_reference_png=reference_png, roi_xyxy=roi,
                    source_edge_geometry=geometry)

            # A scaled antialias fringe is not evidence for a designed gradient.
            # Re-propose after each action, preserving the current stack exactly.
            tried_flat = set()
            for _ in range(16):
                if time.monotonic() - started >= maximum_seconds:
                    report['search_budget_exhausted'] = True
                    break
                flat = propose_source_flat_paint(current, source)
                report['flat_paint_decisions'] = flat.get('decisions', [])
                proposals = [p for p in flat.get('proposals', [])
                             if p.get('drawable_id') not in tried_flat]
                # A partial paint certificate binds the exact shared paths,
                # paint and stack. Retiring only one sibling would leave stale
                # evidence for the remaining definition. Keep this validated
                # unit until a dedicated source revalidation can replace it.
                for deferred in proposals:
                    if deferred.get('gradient_id') in partial_paint_ids:
                        tried_flat.add(deferred.get('drawable_id'))
                        report['attempts'].append({
                            'kind': 'source_flat_paint', 'status': 'deferred',
                            'drawable_id': deferred.get('drawable_id'),
                            'gradient_id': deferred.get('gradient_id'),
                            'reason': 'partial_existing_paths_paint_certificate_requires_revalidation'})
                proposals = [p for p in proposals
                             if p.get('drawable_id') not in tried_flat]
                if not proposals:
                    break
                proposal = proposals[0]
                identifier = proposal['drawable_id']
                tried_flat.add(identifier)
                row = {'kind': 'source_flat_paint', 'drawable_id': identifier,
                       'gradient_id': proposal['gradient_id'],
                       'replacement_fill': proposal['replacement_fill'],
                       'evidence': proposal['evidence']}
                try:
                    if proposal['before_svg_sha256'] != _sha(current.encode('utf-8')):
                        raise ValueError('source_flat_candidate_baseline_mismatch')
                    candidate, unused = _prune_replaced_paint(proposal['svg_text'], proposal['gradient_id'])
                    guard = check(candidate, proposal['roi_xyxy'])
                    row.update(status='accepted' if guard.get('accepted') is True
                               else 'rejected', source_guard=guard)
                    if guard.get('accepted') is True:
                        transactions.append({'before_svg_text': current, 'after_svg_text': candidate,
                                             'source_edge_geometry': None, 'roi_xyxy': proposal['roi_xyxy']})
                        current = candidate
                        # Shared gradient definitions can remain, but paint proof
                        # is retired only when no drawable uses that gradient.
                        if unused:
                            removed = [d for d in details if d.get('id') == proposal['gradient_id']]
                            report.setdefault('replaced_gradient_details', []).extend(removed)
                            details = [d for d in details if d not in removed]
                        report['committed'].append({k: row[k] for k in
                            ('kind', 'drawable_id', 'gradient_id', 'replacement_fill')})
                except Exception as exc:
                    row.update(status='error', reason=f'{type(exc).__name__}: {exc}'[:400])
                report['attempts'].append(row)

            # Native primitives and already economical paths require no repair.
            # Choose by structural burden, never by fixture ID or source region.
            paths = []
            for node in ET.fromstring(current).iter():
                if (node.tag.rsplit('}', 1)[-1] != 'path'
                        or not node.get('data-avc-gradient-object')
                        or not node.get('id')):
                    continue
                if (node.get('data-avc-gradient-object') in partial_owner_ids
                        or any(_paint_is_referenced(node, identifier)
                               for identifier in partial_paint_ids if identifier)):
                    report.setdefault('contour_deferred', []).append({
                        'drawable_id': node.get('id'),
                        'reason': 'partial_existing_paths_not_complete_contour_ownership'})
                    continue
                try:
                    anchors = int(node.get('data-avc-designer-anchors', '0'))
                except ValueError:
                    continue
                if anchors > 24:
                    paths.append((anchors, node.get('id')))
            paths.sort(key=lambda row: (-row[0], row[1]))
            for old_anchors, identifier in paths[:maximum_objects]:
                if time.monotonic() - started >= maximum_seconds:
                    report['search_budget_exhausted'] = True
                    break
                # Only hidden contact boundaries can underlap; the candidate
                # builder and full-scene source guard both enforce this scope.
                for underlap in (2.0, 1.0, 0.0):
                    if time.monotonic() - started >= maximum_seconds:
                        report['search_budget_exhausted'] = True
                        break
                    row = {'kind': 'source_edge', 'drawable_id': identifier,
                           'underlap_pixels': underlap}
                    try:
                        proposed = propose_source_edge_reconstruction(current,
                            identifier, source, reference,
                            error_budget_percent=error_budget_percent,
                            underlap_pixels=underlap)
                        if not proposed or not proposed.get('path'):
                            row.update(status='not_proposed',
                                       reason=(proposed or {}).get('reason'))
                            report['attempts'].append(row)
                            break
                        geometry = proposed['geometry']
                        new_anchors = int(geometry['designer_anchor_count'])
                        if new_anchors >= old_anchors:
                            row.update(status='not_more_editable', anchors=new_anchors)
                            report['attempts'].append(row)
                            break
                        candidate = proposed['candidate_svg_text']
                        guard = check(candidate, geometry=geometry)
                        for refinement in range(4):
                            if guard.get('accepted') is True or time.monotonic()-started >= maximum_seconds:
                                break
                            if refinement == 3 and not _small_source_contact_tail(guard):
                                break
                            regions = guard.get('localized_defects', [])
                            has_hole = any(str(r.get('kind', '')).startswith(('created_hole', 'erased_hole')) for r in regions)
                            free_edge_verified = guard.get('source_boundary_evidence', {}).get('verified') is True
                            boundary_outliers = bool(_boundary_outlier_boxes(guard, identifier))
                            if not has_hole and not free_edge_verified and not boundary_outliers:
                                break
                            try:
                                amended = preserve_source_edge_contacts(current, proposed, source, reference, guard,
                                    maximum_seconds=min(30.0, max(0.0, maximum_seconds-(time.monotonic()-started))),
                                    include_unverified_source_defects=free_edge_verified and not has_hole,
                                    include_boundary_outliers=boundary_outliers and not has_hole)
                                amended_guard = check(amended['candidate_svg_text'], geometry=amended['geometry'])
                                row.setdefault('contact_refinements', []).append({
                                    'small_local_tail_attempt': refinement == 3,
                                    'anchors_after': amended['geometry']['designer_anchor_count'],
                                    'source_guard': amended_guard,
                                    'exact_original_contacts': amended['certificate'].get('contact_preservation')})
                                proposed, guard = amended, amended_guard
                                geometry, candidate = proposed['geometry'], proposed['candidate_svg_text']
                                new_anchors = int(geometry['designer_anchor_count'])
                            except Exception as exc:
                                row.setdefault('contact_refinements', []).append({
                                    'status': 'not_applied', 'reason': f'{type(exc).__name__}: {exc}'[:400]})
                                break
                        row.update(status='accepted' if guard.get('accepted') is True
                                   else 'rejected', anchors_before=old_anchors,
                                   anchors_after=new_anchors, source_guard=guard)
                        report['attempts'].append(row)
                        if guard.get('accepted') is not True:
                            continue
                        proof = geometry['source_edge_reconstruction']
                        geometry['source_edge_scene_commit'] = {
                            'status': 'committed', 'accepted': True,
                            'source_guard': guard,
                            'alpha_guard': {'accepted': True,
                                'scope': 'source_supported_alpha_not_prior_svg_topology'},
                            'before_svg_sha256': _sha(current.encode('utf-8')),
                            'after_path_sha256': _path_sha(candidate, identifier),
                            'source_rgba_sha256': proof['source_rgba_sha256'],
                        }
                        root = ET.fromstring(candidate)
                        target = next(n for n in root.iter() if n.get('id') == identifier)
                        owner = target.get('data-avc-gradient-object')
                        matches = [d for d in details
                                   if d.get('candidate_id') == owner
                                   or d.get('proposal_id') == owner
                                   or d.get('id') == proof.get('gradient_id')]
                        if len(matches) != 1:
                            raise ValueError('source_repair_gradient_evidence_join_ambiguous')
                        validation = matches[0]['validation']
                        validation['geometry'] = geometry
                        validation.setdefault('selection', {}).update(
                            colour_used_for_geometry=True,
                            geometry_source='source_coverage_and_guarded_ownership_interface_reconstruction')
                        transactions.append({'before_svg_text': current, 'after_svg_text': candidate,
                                             'source_edge_geometry': copy.deepcopy(geometry), 'roi_xyxy': None})
                        current = candidate
                        report['committed'].append({
                            'kind': 'source_edge', 'drawable_id': identifier,
                            'anchors_before': old_anchors, 'anchors_after': new_anchors,
                            'underlap_pixels': underlap})
                        break
                    except Exception as exc:
                        row.update(status='error', reason=f'{type(exc).__name__}: {exc}'[:400])
                        if not report['attempts'] or report['attempts'][-1] is not row:
                            report['attempts'].append(row)
                        if isinstance(exc, ValueError) and any(
                                token in str(exc) for token in ('topology', 'connectivity')):
                            # A hidden contact underlap can repair the boundary
                            # connectivity of an otherwise valid source field.
                            continue
                        break

            # Each action is re-proposed against the current scene so an earlier
            # accepted action cannot invalidate a later action's parent proof.
            tried = set()
            for _ in range(16):
                if time.monotonic() - started >= maximum_seconds:
                    report['search_budget_exhausted'] = True
                    break
                light = propose_light_fill_cleanup(current, source_png, reference)
                report['light_retained'] = light.get('retained', [])
                proposals = [p for p in light.get('proposals', [])
                             if (p.get('drawable_id'), p.get('operation')) not in tried]
                if not proposals:
                    break
                proposal = proposals[0]
                key = (proposal.get('drawable_id'), proposal.get('operation'))
                tried.add(key)
                row = {'kind': 'light_fill', 'proposal': proposal}
                try:
                    candidate = apply_light_fill_candidate(current, proposal)
                    guard = check(candidate)
                    row.update(status='accepted' if guard.get('accepted') is True
                               else 'rejected', source_guard=guard)
                    if guard.get('accepted') is True:
                        transactions.append({'before_svg_text': current, 'after_svg_text': candidate,
                                             'source_edge_geometry': None, 'roi_xyxy': proposal.get('roi_xyxy')})
                        current = candidate
                        report['committed'].append({'kind': 'light_fill',
                            'drawable_id': key[0], 'operation': key[1]})
                except Exception as exc:
                    row.update(status='error', reason=f'{type(exc).__name__}: {exc}'[:400])
                report['attempts'].append(row)

            if current != original:
                # Recheck aggregate against the one original stage baseline.
                before.write_bytes(original_bytes)
                after.write_bytes(current.encode('utf-8'))
                final_guard = validate_source_scene_chain(before, after, source_png,
                    processed_reference_png=reference_png, transactions=transactions)
                report['final_source_guard'] = final_guard
                if final_guard.get('accepted') is not True:
                    report['status'] = 'rolled_back'
                    report['attempted_commits'] = report.pop('committed')
                    report['committed'] = []
                    return report, copy.deepcopy(list(gradient_info or ()))
                atomic_replace_bytes(svg_path, current.encode('utf-8'))
                report['status'] = 'committed'
            report['after_svg_sha256'] = _sha(svg_path.read_bytes())
            report['elapsed_seconds'] = round(time.monotonic()-started, 3)
            return report, details
    except Exception as exc:
        report.update(status='error', reason=f'{type(exc).__name__}: {exc}'[:400],
                      elapsed_seconds=round(time.monotonic()-started, 3))
        # No writes occur until the final transaction; its replacement is atomic.
        if svg_path.read_bytes() != original_bytes:
            atomic_replace_bytes(svg_path, original_bytes)
        report['attempted_commits'] = report.pop('committed')
        report['committed'] = []
        return report, copy.deepcopy(list(gradient_info or ()))
