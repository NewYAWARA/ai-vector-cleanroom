"""Independent, source-agnostic hints for repeated short cubic reversals."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import designer_quality as dq


def micro_s(count=4, *, step=.5):
    parts=["M0 10"]
    for index in range(count):
        x=index*step
        parts.append(f"C{x+step*.25} 10.5 {x+step*.75} 9.5 {x+step} 10")
    return " ".join(parts)


class CurveCounterturnQualityTests(unittest.TestCase):
    def audit(self, body, metadata=None):
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">{body}</svg>'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.svg'
            path.write_text(svg,encoding='utf8')
            before=path.read_bytes()
            result=dq.audit_designer_quality(path,proposal_metadata=metadata)
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(result['source']['sha256'],hashlib.sha256(before).hexdigest())
            json.dumps(result,allow_nan=False)
            return result

    @staticmethod
    def diagnostic(result):
        return result['curve_economy_gate']['short_counterturn_diagnostic']

    def test_four_micro_s_segments_trigger_warning_and_reproducible_locations(self):
        result=self.audit(f'<path id="wavy" d="{micro_s()}"/>')
        gate=result['curve_economy_gate'];detail=self.diagnostic(result)
        self.assertEqual(detail['counterturn_segment_count'],4)
        self.assertEqual(detail['review_path_count'],1)
        self.assertEqual(gate['status'],'manual_review')
        self.assertEqual(gate['failure_reasons'],[])
        self.assertIn('repeated_short_counterturns_require_review',gate['warning_reasons'])
        record=detail['paths'][0]
        self.assertEqual(record['id'],'wavy')
        self.assertEqual(record['short_counterturn_count'],4)
        self.assertTrue(record['requires_review'])
        self.assertEqual(record['bbox'],[0.0,9.5,2.0,10.5])
        self.assertEqual(record['segments'][0]['parsed_segment_index_1_based'],2)
        self.assertGreaterEqual(record['segments'][0]['sampled_counterturn_degrees'],45)

    def test_counts_remain_public_below_per_path_warning_threshold(self):
        result=self.audit(f'<path id="few" d="{micro_s(3)}"/>')
        detail=self.diagnostic(result)
        self.assertEqual(detail['counterturn_segment_count'],3)
        self.assertEqual(detail['review_path_count'],0)
        self.assertFalse(detail['paths'][0]['requires_review'])
        self.assertNotIn('repeated_short_counterturns_require_review',result['curve_economy_gate']['warning_reasons'])
        self.assertEqual(result['curve_economy_gate']['metrics']['cubic_segment_count'],3)

    def test_counts_do_not_combine_three_on_two_separate_paths_into_one_group(self):
        body=''.join(f'<path id="p{i}" d="{micro_s(3)}"/>' for i in range(2))
        detail=self.diagnostic(self.audit(body))
        self.assertEqual(detail['counterturn_segment_count'],6)
        self.assertEqual(detail['review_path_count'],0)

    def test_tiny_circles_and_high_turn_one_way_semicircles_are_not_counterturns(self):
        circle=('M9.9 10 C9.9 10.1333 10.1 10.1333 10.1 10 '
                'C10.1 9.8667 9.9 9.8667 9.9 10 Z')
        result=self.audit('<circle cx="20" cy="20" r=".1"/>'+''.join(
            f'<path id="circle{i}" d="{circle}"/>' for i in range(4)))
        detail=self.diagnostic(result)
        self.assertEqual(detail['counterturn_segment_count'],0)
        self.assertEqual(detail['review_path_count'],0)
        self.assertNotIn('repeated_short_counterturns_require_review',result['curve_economy_gate']['warning_reasons'])

    def test_long_s_curves_are_not_sampled_for_this_short_curve_metric(self):
        original=dq._curve_point
        # Other pre-existing path length/bounds calculations still use their
        # own samples; the dedicated diagnostic must not add 24-point work.
        segments,_=dq._path_segments(micro_s(4,step=5))
        element=dq.ET.fromstring(f'<path d="{micro_s(4,step=5)}"/>')
        with patch.object(dq,'_curve_point',wraps=original) as observed:
            diagnostic=dq._short_counterturn_diagnostic(element,{},segments,100)
        self.assertEqual(observed.call_count,0)
        self.assertEqual(diagnostic['short_cubic_count'],0)

    def test_affine_transform_chord_and_bboxes_use_canvas_coordinates(self):
        result=self.audit(f'<g transform="translate(20 5)"><path id="small" d="{micro_s()}"/></g>')
        self.assertEqual(self.diagnostic(result)['paths'][0]['bbox'],[20.0,14.5,22.0,15.5])
        result=self.audit(f'<g transform="scale(10)"><path d="{micro_s()}"/></g>')
        self.assertEqual(self.diagnostic(result)['counterturn_segment_count'],0)

    def test_unknown_layout_reports_missing_coverage_without_false_geometry_hint(self):
        result=self.audit(f'<path id="css" style="transform:scale(10)" d="{micro_s()}"/>')
        detail=self.diagnostic(result)
        self.assertEqual(detail['unassessed_or_partial_path_count'],1)
        self.assertEqual(detail['assessed_path_count'],0)
        self.assertEqual(detail['counterturn_segment_count'],0)
        self.assertEqual(detail['coverage_exceptions'][0]['id'],'css')

    def test_detail_output_is_capped_but_counts_are_not(self):
        body=''.join(f'<path id="p{i}" d="{micro_s(15)}"/>' for i in range(53))
        detail=self.diagnostic(self.audit(body))
        self.assertEqual(detail['counterturn_segment_count'],53*15)
        self.assertEqual(detail['review_path_count'],53)
        self.assertEqual(len(detail['paths']),50)
        self.assertTrue(detail['path_records_truncated'])
        self.assertEqual(len(detail['paths'][0]['segments']),12)
        self.assertTrue(detail['paths'][0]['segment_records_truncated'])

    def test_valid_transaction_certificate_does_not_waive_independent_warning(self):
        try:
            from test_designer_quality import _retained_curve_proposal
        except ImportError:
            from tests.test_designer_quality import _retained_curve_proposal
        # Exactly 60 anchors, matching the existing valid retained-geometry
        # certificate fixture: four micro-S cubics plus 55 lines and a close.
        path_data=micro_s(4)+' '+ ' '.join(f'L{2+i*.5} 20' for i in range(1,56))+' Z'
        proposal=_retained_curve_proposal(path_data)
        before=copy.deepcopy(proposal)
        result=self.audit(f'<path id="retained" d="{path_data}" fill="#174f39"/>',proposal)
        self.assertEqual(proposal,before)
        gate=result['curve_economy_gate']
        self.assertTrue(gate['optimizer_economy_evidence']['curve_refit']['authoritative'],gate)
        self.assertEqual(gate['heuristic_scope_metrics']['path_count'],0)
        self.assertEqual(gate['metrics']['node_count'],60)
        self.assertEqual(gate['metrics']['cubic_segment_count'],4)
        self.assertTrue(self.diagnostic(result)['paths'][0]['optimizer_certified'])
        self.assertIn('repeated_short_counterturns_require_review',gate['warning_reasons'])
        self.assertEqual(gate['status'],'manual_review')
        self.assertEqual(gate['failure_reasons'],[])


if __name__=='__main__':
    unittest.main()
