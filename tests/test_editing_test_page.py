from __future__ import annotations

import json
import hashlib
from html import unescape
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from editing_test_page import TASKS, build_editing_test_page


METRICS_FILE = Path(__file__).resolve().parents[1] / "editing_metrics.js"
NODE = shutil.which("node")


def _node(expression):
    """Execute the production calculation code, without a browser or DOM stub."""
    script = f"const metrics = require({json.dumps(str(METRICS_FILE))});\n"
    script += f"console.log(JSON.stringify({expression}));"
    result = subprocess.run([NODE, "-e", script], check=True,
                            capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)


def _completed_task(**overrides):
    task = {
        "id": "complete_deliverable", "applicable": True,
        "vector_seconds": 20, "redraw_seconds": 100,
        "vector_status": "completed", "redraw_status": "completed",
        "vector_evidence": "actual", "redraw_evidence": "actual",
    }
    task.update(overrides)
    return task


def _draft_page_script(*, source_bytes=b"source", svg_markup='<svg xmlns="http://www.w3.org/2000/svg"/>'):
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder)
        result = output / "result_draft"
        result.mkdir()
        (result / "draft_vector.svg").write_text(svg_markup, encoding="utf-8")
        (result / "source_original.png").write_bytes(source_bytes)
        page = build_editing_test_page(output, [{
            "dir": "result_draft", "svg": "result_draft/draft_vector.svg",
        }], tool_version="test")
        return re.search(r'<script>(.*?)</script>', page.read_text(encoding="utf-8"), re.DOTALL).group(1)


def _run_draft_scenario(script, scenario):
    # Execute the actual page persistence/timer handlers with a minimal DOM.
    harness = r'''
const vm=require('vm');
const store=new Map();
function openPage(script,now=0,writable=true){
 let clock=now; const events={},intervals=[];
 function field(value='',type='text'){
  const handlers={};return {value,type,tagName:type==='select-one'?'SELECT':'INPUT',checked:false,
   addEventListener:(event,fn)=>{handlers[event]=fn;},fire:event=>handlers[event]?.()};
 }
 const time=field('','number'),evidence=field('none','select-one'),status=field('not_started','select-one');
 const applicable=field('','checkbox'),note=field(),designer=field(),editor=field(),experience=field('','number');
 const fields=[time,evidence,status,applicable,note,designer,editor,experience];
 const draftStatus={textContent:''},exportButton={};
 const lookup={'.seconds.vector':time,'.vector-kind':evidence,'.vector-status':status,'.applicable':applicable};
 const timer={dataset:{kind:'vector'},textContent:'開始',closest:()=>({querySelector:selector=>lookup[selector]})};
 const ids={'draft-status':draftStatus,'export':exportButton,designer,editor,experience};
 const env={performance:{now:()=>clock},window:{addEventListener:(name,fn)=>{events[name]=fn;}},
  localStorage:{getItem:key=>store.get(key)||null,setItem:(key,value)=>{if(!writable)throw new Error('quota');store.set(key,value);},removeItem:key=>store.delete(key)},
  setInterval:fn=>{intervals.push(fn);return intervals.length;},setTimeout:()=>0,
  Blob:class{},URL:{createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{}},
  document:{querySelectorAll:selector=>selector==='input,select,textarea'?fields:selector==='.timer'?[timer]:[],
   getElementById:id=>ids[id],createElement:()=>({click:()=>{}})}};
 vm.runInNewContext(script,env);
 return {time,evidence,status,note,timer,exportButton,draftStatus,
  setTime:value=>{clock=value;},tick:()=>intervals.forEach(fn=>fn()),pagehide:()=>events.pagehide(),
  before:()=>{let prevented=false;const event={preventDefault:()=>{prevented=true;}};events.beforeunload(event);return prevented;}};
}
'''
    program = harness + "\nconst script=" + json.dumps(script) + ";\n" + scenario
    result = subprocess.run([NODE, "-"], input=program, check=True, capture_output=True,
                            text=True, encoding="utf-8", timeout=15)
    return json.loads(result.stdout)


class EditingTestPageTests(unittest.TestCase):
    @unittest.skipUnless(NODE, "Node is required for draft lifecycle tests")
    def test_draft_warning_begins_only_after_user_input_and_survives_storage_failure(self):
        result = _run_draft_scenario(_draft_page_script(), r'''
const page=openPage(script,0,false);
const fresh=page.before();page.note.value='人工記錄';page.note.fire('input');
console.log(JSON.stringify({fresh,dirty:page.before(),status:page.draftStatus.textContent,stored:store.size}));
''')
        self.assertFalse(result["fresh"])
        self.assertTrue(result["dirty"])
        self.assertEqual(result["stored"], 0)
        self.assertIn("無法儲存草稿", result["status"])

    @unittest.skipUnless(NODE, "Node is required for draft lifecycle tests")
    def test_running_timer_restores_accumulated_seconds_but_never_time_away(self):
        result = _run_draft_scenario(_draft_page_script(), r'''
const first=openPage(script);first.timer.onclick();first.setTime(12500);first.tick();
const periodic=JSON.parse([...store.values()][0]).fields[0].value;
first.setTime(15000);first.pagehide();
const restored=openPage(script,1000000);restored.setTime(2000000);restored.tick();
console.log(JSON.stringify({periodic,value:restored.time.value,status:restored.status.value,
 button:restored.timer.textContent,warning:restored.before(),message:restored.draftStatus.textContent}));
''')
        self.assertEqual(result["periodic"], "12.5")
        self.assertEqual(result["value"], "15")
        self.assertEqual(result["status"], "in_progress")
        self.assertEqual(result["button"], "開始")
        self.assertTrue(result["warning"])
        self.assertIn("停表", result["message"])

    @unittest.skipUnless(NODE, "Node is required for draft lifecycle tests")
    def test_successful_export_clears_dirty_warning_and_saved_draft(self):
        result = _run_draft_scenario(_draft_page_script(), r'''
const page=openPage(script);page.note.value='已完成紀錄';page.note.fire('change');
const before=page.before();page.exportButton.onclick();
console.log(JSON.stringify({before,after:page.before(),stored:store.size}));
''')
        self.assertTrue(result["before"])
        self.assertFalse(result["after"])
        self.assertEqual(result["stored"], 0)

    @unittest.skipUnless(NODE, "Node is required for draft lifecycle tests")
    def test_drafts_are_bound_to_both_svg_and_source_bytes(self):
        original = _draft_page_script()
        changed_source = _draft_page_script(source_bytes=b"different-source")
        changed_svg = _draft_page_script(svg_markup='<svg xmlns="http://www.w3.org/2000/svg"><circle r="2"/></svg>')
        identities = [re.search(r'const DRAFT_IDENTITY=("[^"]+")', script).group(1)
                      for script in (original, changed_source, changed_svg)]
        self.assertEqual(len(set(identities)), 3)
        scenario = r'''
const first=openPage(script);first.note.value='old source notes';first.note.fire('input');
const other=openPage(OTHER_SCRIPT);
console.log(JSON.stringify({note:other.note.value,warning:other.before()}));
'''.replace("OTHER_SCRIPT", json.dumps(changed_source))
        result = _run_draft_scenario(original, scenario)
        self.assertEqual(result["note"], "")
        self.assertFalse(result["warning"])

    def test_original_is_preferred_and_original_only_case_is_included(self):
        for legacy_present in (False, True):
            with self.subTest(legacy_present=legacy_present), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                result = output / "result_derived"
                result.mkdir()
                (result / "derived_vector.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
                original_bytes = b"original-reference-pixels"
                (result / "source_original.png").write_bytes(original_bytes)
                if legacy_present:
                    (result / "source_reference.png").write_bytes(b"cleaned-reference")
                path = build_editing_test_page(output, [{"dir": "result_derived", "svg": "result_derived/derived_vector.svg"}], tool_version="test")
                body = path.read_text(encoding="utf-8")
                meta = json.loads(unescape(re.search(r'<section class="case" data-meta="([^"]+)"', body).group(1)))
                self.assertEqual(meta["reference_kind"], "original")
                self.assertEqual(meta["source_reference"], "result_derived/source_original.png")
                self.assertEqual(meta["source_sha256"], hashlib.sha256(original_bytes).hexdigest())
                self.assertIn('href="result_derived/source_original.png"', body)
                self.assertIn("開啟未處理原圖", body)
                self.assertNotIn('href="result_derived/source_reference.png"', body)
                self.assertNotIn("本案例只有清理後參考圖", body)

    def test_legacy_reference_is_explicitly_labeled_and_hashed(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = output / "result_legacy"
            result.mkdir()
            (result / "legacy_vector.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
            processed_bytes = b"cleaned-reference-pixels"
            (result / "source_reference.png").write_bytes(processed_bytes)
            path = build_editing_test_page(output, [{"dir": "result_legacy", "svg": "result_legacy/legacy_vector.svg"}], tool_version="test")
            body = path.read_text(encoding="utf-8")
            meta = json.loads(unescape(re.search(r'<section class="case" data-meta="([^"]+)"', body).group(1)))
            self.assertEqual(meta["reference_kind"], "processed_reference")
            self.assertEqual(meta["source_reference"], "result_legacy/source_reference.png")
            self.assertEqual(meta["source_sha256"], hashlib.sha256(processed_bytes).hexdigest())
            self.assertIn("清理後參考圖（非原圖）", body)
            self.assertIn("不能當作未處理原圖的省工證據", body)
            self.assertNotIn("開啟原始參考圖", body)

    def test_page_requires_actual_timing_for_saving_metric(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = output / "result_demo"
            result.mkdir()
            svg = result / "demo_vector.svg"
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>',
                           encoding="utf-8")
            (result / "source_reference.png").write_bytes(b"png")
            (result / "report.json").write_text(json.dumps({
                "tool_version": "v3-codex-beta.5",
                "input": "demo.png",
            }), encoding="utf-8")
            path = build_editing_test_page(output, [{
                "dir": "result_demo", "base": "demo", "input": "demo.png",
                "svg": "result_demo/demo_vector.svg",
                "visual_acceptance_status": "accepted",
                "editability_status": "manual_review",
            }], tool_version="v3-codex-beta.5")
            body = path.read_text(encoding="utf-8")

        self.assertIn("Stage 2", body)
        self.assertIn("actual_timed_weighted_saving_percent", body)
        self.assertIn(METRICS_FILE.read_text(encoding="utf-8"), body)
        self.assertIn("EditingMetrics.summarize(rawTasks)", body)
        self.assertIn("product_claim_validated: false", body)
        self.assertIn("multiple designers and representative logos", body)
        self.assertEqual(body.count('<tr data-task="'), len(TASKS))
        self.assertIn("demo_vector.svg", body)
        self.assertIn('data-task="complete_deliverable"', body)
        self.assertIn('value="not_started" selected', body)
        self.assertEqual(body.count('class="applicable" type="checkbox" checked'), 1)
        self.assertNotIn("主要綠色", body)
        self.assertNotIn("主外環", body)
        self.assertNotRegex(body, r'<script\b[^>]*\bsrc=')
        if NODE:
            # Parse the actual embedded page script, not only the module.
            scripts = re.findall(r'<script>(.*?)</script>', body, re.DOTALL)
            self.assertEqual(len(scripts), 1)
            subprocess.run([NODE, "--check"], input=scripts[0], check=True,
                           capture_output=True, text=True, encoding="utf-8")

    def test_missing_files_are_skipped_without_broken_case(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            path = build_editing_test_page(output, [{
                "dir": "result_missing", "svg": "result_missing/no.svg",
            }], tool_version="v3-codex-beta.5")
            body = path.read_text(encoding="utf-8")
        self.assertIn("尚無同時具備 SVG", body)
        self.assertNotIn('class="case"', body)

    def test_result_paths_cannot_escape_output_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "output"
            output.mkdir()
            outside = Path(folder) / "outside.svg"
            outside.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>',
                               encoding="utf-8")
            path = build_editing_test_page(output, [{
                "dir": "..", "svg": "../outside.svg",
            }], tool_version="v3-codex-beta.5")
            body = path.read_text(encoding="utf-8")
        self.assertIn("尚無同時具備 SVG", body)
        self.assertNotIn(str(outside), body)


@unittest.skipUnless(NODE, "Node is required for JavaScript metric regression tests")
class EditingMetricsTests(unittest.TestCase):
    def test_empty_invalid_and_nonfinite_times_stay_missing(self):
        results = _node("['', '  ', null, undefined, false, [], -1, NaN, Infinity, 'Infinity', 'abc'].map(metrics.seconds)")
        self.assertEqual(results, [None] * 11)
        self.assertEqual(_node("[0, '0', '2.5'].map(metrics.seconds)"), [0, 0, 2.5])

    def test_blank_or_zero_vector_never_becomes_hundred_percent_saving(self):
        for value in ("", "  ", 0, None):
            with self.subTest(vector_seconds=value):
                result = _node(f"metrics.summarize([{json.dumps(_completed_task(vector_seconds=value))}])")
                task, summary = result["tasks"][0], result["summary"]
                self.assertFalse(task["comparison_eligible"])
                self.assertIsNone(task["time_saving_percent"])
                self.assertIsNone(summary["primary_time_saving_percent"])
                self.assertFalse(summary["observed_ge_80_percent_for_this_session"])
                self.assertEqual(summary["attempted_tasks"], 1)
                self.assertEqual(summary["uncomparable_attempts"], 1)

    def test_both_conditions_require_explicit_completion_and_actual_positive_time(self):
        invalid = (
            {"vector_status": "partial"}, {"vector_status": "unable"},
            {"vector_status": "not_started"}, {"redraw_status": "partial"},
            {"redraw_status": "unable"}, {"redraw_status": "not_started"},
            {"vector_evidence": "estimated"}, {"vector_evidence": "none"},
            {"redraw_evidence": "estimated"}, {"redraw_evidence": "none"},
            {"redraw_seconds": 0}, {"redraw_seconds": ""},
            {"applicable": False}, {"vector_seconds": -5},
        )
        tasks = [_completed_task(**values) for values in invalid]
        results = _node(f"{json.dumps(tasks)}.map(metrics.evaluateTask)")
        for original, result in zip(tasks, results):
            with self.subTest(original=original):
                self.assertFalse(result["comparison_eligible"])
                self.assertIsNone(result["time_saving_percent"])
                self.assertTrue(result["attempted"])
                self.assertTrue(result["exclusion_reasons"])

    def test_failed_and_deselected_attempts_remain_visible(self):
        tasks = [
            _completed_task(vector_status="unable", vector_seconds=120),
            _completed_task(id="contour_cleanup", applicable=False,
                            vector_status="partial", vector_seconds=30),
        ]
        result = _node(f"metrics.summarize({json.dumps(tasks)})")
        self.assertEqual(result["tasks"][0]["vector_seconds"], 120)
        self.assertEqual(result["tasks"][1]["vector_seconds"], 30)
        self.assertEqual(result["summary"]["attempted_tasks"], 2)
        self.assertEqual(result["summary"]["unable_tasks"], 1)
        self.assertEqual(result["summary"]["incomplete_vector_tasks"], 2)
        self.assertEqual(result["summary"]["uncomparable_attempts"], 2)
        self.assertIsNone(result["summary"]["actual_timed_weighted_saving_percent"])

    def test_easy_diagnostic_cannot_stand_in_for_full_deliverable(self):
        tasks = [_completed_task(id="global_recolour", vector_seconds=1)]
        result = _node(f"metrics.summarize({json.dumps(tasks)})")
        self.assertEqual(result["tasks"][0]["time_saving_percent"], 99)
        self.assertEqual(result["summary"]["diagnostic_comparable_tasks"], 1)
        self.assertFalse(result["summary"]["primary_comparison_eligible"])
        self.assertIsNone(result["summary"]["actual_timed_weighted_saving_percent"])
        self.assertFalse(result["summary"]["observed_ge_80_percent_for_this_session"])

    def test_primary_time_does_not_double_count_diagnostic_measurements(self):
        tasks = [_completed_task(), _completed_task(id="global_recolour", vector_seconds=90)]
        result = _node(f"metrics.summarize({json.dumps(tasks)})")
        summary = result["summary"]
        self.assertEqual(summary["metric_scope"], "complete_deliverable_only")
        self.assertEqual(summary["vector_seconds_sum"], 20)
        self.assertEqual(summary["redraw_seconds_sum"], 100)
        self.assertEqual(summary["primary_time_saving_percent"], 80)
        self.assertTrue(summary["observed_ge_80_percent_for_this_session"])
        self.assertFalse(summary["product_claim_validated"])

    def test_slower_conversion_reports_negative_saving(self):
        result = _node(f"metrics.summarize([{json.dumps(_completed_task(vector_seconds=150))}])")
        self.assertEqual(result["summary"]["primary_time_saving_percent"], -50)
        self.assertFalse(result["summary"]["observed_ge_80_percent_for_this_session"])

    def test_display_rounding_does_not_pass_eighty_percent_threshold(self):
        result = _node(f"metrics.summarize([{json.dumps(_completed_task(vector_seconds=20.0001))}])")
        self.assertLess(result["summary"]["primary_time_saving_percent"], 80)
        self.assertFalse(result["summary"]["observed_ge_80_percent_for_this_session"])

    def test_nonfinite_ratio_is_excluded_even_when_both_times_are_finite(self):
        result = _node(f"metrics.evaluateTask({json.dumps(_completed_task(vector_seconds=1e308, redraw_seconds=1e-308))})")
        self.assertFalse(result["comparison_eligible"])
        self.assertIn("nonfinite_time_ratio", result["exclusion_reasons"])
        self.assertIsNone(result["time_saving_percent"])


if __name__ == "__main__":
    unittest.main()
