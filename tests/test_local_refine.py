from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from designer_handoff import build_handoff_manifest, normalized_svg_text, _geometry
from handoff_service import HandoffConflict
import local_refine as refine


DENSE = "M10 10 L30 10 L50 10 L70 10 L90 10 L90 50 L90 90 L50 90 L10 90 L10 50 Z"
SIMPLE = "M10 10 L90 10 L90 90 L10 90 Z"
PASSED = {"external_render_check": "completed", "accepted": True}


class LocalRefineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "result_demo"
        self.folder.mkdir()
        self.svg = self.folder / "demo_vector.svg"
        self.source = self.folder / "source_reference.png"
        self.source.write_bytes(b"source-bytes-preserved")
        self.report = self.folder / "report.json"
        self.report.write_text(json.dumps({"input": "demo.png", "foreground_match_percent": 99,
                                          "editability_score": 98, "acceptance_status": "accepted"}), encoding="utf-8")
        self._svg()

    def _svg(self, attrs="", group="", data=DENSE):
        self.svg.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                            f'<g fill="#126b45" {group}><path id="target" d="{data}" {attrs}/></g>'
                            '<g id="object-locked"><rect id="locked" x="2" y="3" width="4" height="5" fill="red"/></g>'
                            '</svg>', encoding="utf-8")

    def _payload(self):
        manifest = build_handoff_manifest(self.svg)
        return {"result": self.folder.name, "svg_sha256": manifest["svg_sha256"],
                "decisions": {"target": "redraw", "object-locked": "keep"},
                "object_ids": ["target"], "error_budget_percent": 0.25, "revision": None}

    def _worker(self, request):
        work = Path(request["work"])
        root = ET.parse(work / "before.svg").getroot()
        target = refine._by_id(root)["target"]
        count_before = _geometry(target)[0]
        target.set("d", SIMPLE)
        count_after = _geometry(target)[0]
        refine._write_svg(work / "after.svg", root)
        return {"changed_member_ids": ["target"], "anchors_removed": count_before - count_after,
                "whole_svg_render": {**PASSED, 'composed_alpha': dict(PASSED)}, "proposals": [],
                "target_gates": [{"id": "target", "render": dict(PASSED),
                                  "target_silhouette_topology": "passed"}]}

    def _run(self, payload=None, worker=None):
        with patch.object(refine, "_run_worker", side_effect=worker or self._worker):
            return refine.refine_result(self.root, payload or self._payload(), lock=threading.RLock())

    def _assert_no_derived(self):
        self.assertEqual([item.name for item in self.root.iterdir()], ["result_demo"])

    def test_new_result_preserves_originals_and_locked_decision_and_invalidates_scores(self):
        originals = {path: path.read_bytes() for path in self.folder.iterdir()}
        result = self._run()
        derived = self.root / result["url"].split("result=", 1)[1]
        self.assertTrue(derived.is_dir())
        for path, payload in originals.items():
            self.assertEqual(path.read_bytes(), payload)
        report = json.loads((derived / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["acceptance_status"], "manual_review")
        self.assertIsNone(report["foreground_match_percent"])
        self.assertIsNone(report["editability_score"])
        self.assertEqual(report["local_refine_validation"]["full_image_scores"], "invalidated")
        state = json.loads((derived / "handoff_state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["decisions"], {"target": "review", "object-locked": "keep"})
        self.assertEqual((derived / "source_reference.png").read_bytes(), originals[self.source])
        self.assertEqual(report["local_refine_history"][0]["member_ids"], ["target"])
        self.assertGreater(result["summary"]["anchors_removed"], 0)

    def test_original_only_reference_is_preserved_without_inventing_processed_reference(self):
        original = self.folder / "source_original.png"
        self.source.rename(original)
        result = self._run()
        derived = self.root / result["url"].split("result=", 1)[1]
        self.assertEqual((derived / "source_original.png").read_bytes(), original.read_bytes())
        self.assertFalse((derived / "source_reference.png").exists())

    def test_keep_is_not_modifiable(self):
        payload = self._payload()
        payload["decisions"]["target"] = "keep"
        with self.assertRaisesRegex(ValueError, "鎖定"):
            self._run(payload)
        self._assert_no_derived()

    def test_missing_revision_is_rejected_before_work(self):
        payload = self._payload()
        payload.pop("revision")
        with self.assertRaises(HandoffConflict):
            self._run(payload)
        self._assert_no_derived()

    def test_stale_revision_is_rejected_before_work(self):
        payload = self._payload()
        refine.save_decisions(self.root, payload)
        with self.assertRaises(HandoffConflict):
            self._run(payload)
        self._assert_no_derived()

    def test_repeated_refinement_is_rejected(self):
        self.report.write_text(json.dumps({"local_refine_history": [{"member_ids": ["target"]}]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已精修"):
            self._run()
        self._assert_no_derived()

    def test_unsafe_contexts_are_rejected_before_worker(self):
        cases = [("", 'transform="translate(1,2)"'), ("", 'stroke="black"'),
                 ('style="stroke:black"', ""), ('data-avc-gradient-object="gradient-1"', ""),
                 ('style="filter:none"', ""), ("", 'style="mix-blend-mode:multiply"')]
        for attrs, group in cases:
            with self.subTest(attrs=attrs, group=group):
                self._svg(attrs=attrs, group=group)
                with patch.object(refine, "_run_worker") as worker:
                    with self.assertRaises(ValueError):
                        refine.refine_result(self.root, self._payload(), lock=threading.RLock())
                    worker.assert_not_called()
                self._assert_no_derived()

    def test_open_path_rejected(self):
        self._svg(data=DENSE[:-1])
        with self.assertRaisesRegex(ValueError, "封閉"):
            self._run()
        self._assert_no_derived()

    def test_path_node_limit(self):
        self._svg(data="M0 0 " + " ".join(f"L{i} {i % 2}" for i in range(600)) + " Z")
        with self.assertRaisesRegex(ValueError, "512"):
            self._run()
        self._assert_no_derived()

    def test_stale_initial_svg_digest(self):
        payload = self._payload()
        payload["svg_sha256"] = "0" * 64
        with self.assertRaises(HandoffConflict):
            self._run(payload)
        self._assert_no_derived()

    def test_original_change_during_work_prevents_publish(self):
        def changed(request):
            result = self._worker(request)
            self.source.write_bytes(b"new source")
            return result
        with self.assertRaises(HandoffConflict):
            self._run(worker=changed)
        self._assert_no_derived()

    def test_decision_change_during_work_prevents_publish(self):
        def changed(request):
            result = self._worker(request)
            (self.folder / "handoff_state.json").write_text("{}")
            return result
        with self.assertRaises(HandoffConflict):
            self._run(worker=changed)
        self._assert_no_derived()

    def test_unselected_or_selected_style_change_is_rejected(self):
        for identifier in ("locked", "target"):
            with self.subTest(identifier=identifier):
                def changed(request):
                    result = self._worker(request)
                    path = Path(request["work"]) / "after.svg"
                    tree = ET.parse(path).getroot()
                    refine._by_id(tree)[identifier].set("fill", "blue")
                    refine._write_svg(path, tree)
                    return result
                with self.assertRaisesRegex(ValueError, "樣式"):
                    self._run(worker=changed)
                self._assert_no_derived()

    def test_missing_external_render_never_publishes(self):
        def changed(request):
            result = self._worker(request)
            result["whole_svg_render"]["external_render_check"] = "unavailable"
            return result
        with self.assertRaisesRegex(ValueError, "renderer"):
            self._run(worker=changed)
        self._assert_no_derived()

    def test_missing_target_gate_never_publishes(self):
        def changed(request):
            result = self._worker(request)
            result["target_gates"] = []
            return result
        with self.assertRaisesRegex(ValueError, "局部拓撲"):
            self._run(worker=changed)
        self._assert_no_derived()

    def test_missing_composed_alpha_guard_never_publishes(self):
        def changed(request):
            result = self._worker(request)
            result['whole_svg_render'].pop('composed_alpha')
            return result
        with self.assertRaisesRegex(ValueError, '透明度與連通區'):
            self._run(worker=changed)
        self._assert_no_derived()

    def test_unchanged_has_no_success_result(self):
        def unchanged(request):
            result = self._worker(request)
            result["anchors_removed"] = 0
            return result
        with self.assertRaisesRegex(ValueError, "未減少"):
            self._run(worker=unchanged)
        self._assert_no_derived()

    def test_worker_is_not_run_under_publish_lock(self):
        class PublishLock:
            active = False
            def __enter__(self):
                self.active = True
            def __exit__(self, *args):
                self.active = False
        lock = PublishLock()
        def worker(request):
            self.assertFalse(lock.active)
            return self._worker(request)
        with patch.object(refine, "_run_worker", side_effect=worker):
            refine.refine_result(self.root, self._payload(), lock=lock)

    def test_timeout_has_clear_no_result_error(self):
        with patch.object(refine.subprocess, "run", side_effect=subprocess.TimeoutExpired("refine", 60)):
            with self.assertRaisesRegex(ValueError, "60 秒"):
                refine._run_worker({})

    def test_target_topology_guard_catches_hole_change(self):
        import numpy as np
        from PIL import Image
        import vector_cleanroom as vc
        root = ET.fromstring(normalized_svg_text(self.svg))
        calls = []
        def render(svg, png, **kwargs):
            mask = np.full((50, 50), 255, dtype=np.uint8)
            mask[5:45, 5:45] = 0
            if calls:
                mask[20:30, 20:30] = 255
            Image.fromarray(mask).save(png)
            calls.append(svg)
            return True
        with patch.object(vc, "render_svg_png", side_effect=render), patch.object(vc, "validate_svg_stage_renders", return_value=PASSED):
            with self.assertRaisesRegex(ValueError, "孔洞"):
                refine._target_gate(root, root, "target", self.root)

    def test_real_worker_reduces_closed_shape_and_preserves_locked_object(self):
        from PIL import Image
        Image.new("RGB", (100, 100), "white").save(self.source)
        before_bytes = self.svg.read_bytes()
        before_locked = ET.tostring(refine._by_id(ET.fromstring(normalized_svg_text(self.svg)))["locked"])
        result = refine.refine_result(self.root, self._payload(), lock=threading.RLock())
        derived = self.root / result["url"].split("result=", 1)[1]
        svg = next(derived.glob("*_vector.svg"))
        self.assertEqual(self.svg.read_bytes(), before_bytes)
        after_locked = ET.tostring(refine._by_id(ET.parse(svg).getroot())["locked"])
        self.assertEqual(after_locked, before_locked)
        report = json.loads((derived / "report.json").read_text(encoding="utf-8"))
        self.assertGreater(report["local_refine_validation"]["anchors_removed"], 0)
        self.assertEqual(report["local_refine_validation"]["whole_svg_render"]["external_render_check"], "completed")
        self.assertTrue(report["local_refine_validation"]["whole_svg_render"]["accepted"])
        self.assertTrue(report["local_refine_validation"]["whole_svg_render"]["composed_alpha"]["accepted"])
        self.assertEqual(report["local_refine_validation"]["target_gates"][0]["target_silhouette_topology"], "passed")

    def test_real_worker_rejects_composed_white_object_merge_with_chinese_error(self):
        points = [(20 + 10 * math.cos((i + .5) * math.tau / 32),
                   60 + 30 * math.sin((i + .5) * math.tau / 32)) for i in range(32)]
        data = 'M' + ' L'.join(f'{x:.8f} {y:.8f}' for x, y in points) + ' Z'
        work = self.root / 'white-merge'; work.mkdir()
        before = work / 'before.svg'
        before.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'
            '<rect id="fixed" fill="white" x="10" y="10" width="100" height="20.1"/>'
            f'<path id="oval" fill="white" d="{data}"/></svg>')
        original = before.read_bytes()
        # This actual bounded subprocess used to pass both the full RGBA
        # 2048 renderer and isolated 768 silhouette while merging components.
        with self.assertRaisesRegex(ValueError, '分開的物件黏合') as caught:
            refine._run_worker({'work': str(work), 'member_ids': ['oval'], 'error_budget_percent': .25})
        self.assertNotIn('whole_alpha', str(caught.exception))
        self.assertEqual(before.read_bytes(), original)
        after = ET.parse(work / 'after.svg').getroot()
        self.assertEqual(_geometry(refine._by_id(after)['oval'])[0], 4)
        from alpha_topology import compare_composed_alpha
        with self.assertRaisesRegex(ValueError, 'whole_alpha_components_changed'):
            compare_composed_alpha(work / 'before-alpha.png', work / 'after-alpha.png')

    def test_derived_copies_original_and_processed_reference_separately(self):
        original = self.folder / "source_original.png"
        original.write_bytes(b"unprocessed-original")
        result = self._run()
        derived = self.root / result["url"].split("result=", 1)[1]
        self.assertEqual((derived / "source_original.png").read_bytes(), b"unprocessed-original")
        self.assertEqual((derived / "source_reference.png").read_bytes(), b"source-bytes-preserved")

    def test_embedded_old_acceptance_is_invalidated(self):
        root = ET.parse(self.svg).getroot()
        metadata = ET.SubElement(root, "{http://www.w3.org/2000/svg}metadata", {"id": "ai-vector-cleanroom-metadata"})
        metadata.text = json.dumps({"acceptance_status": "accepted", "editability_score": 99})
        refine._write_svg(self.svg, root)
        result = self._run()
        derived = self.root / result["url"].split("result=", 1)[1]
        new_root = ET.parse(next(derived.glob("*_vector.svg"))).getroot()
        new_meta = refine._by_id(new_root)["ai-vector-cleanroom-metadata"]
        payload = json.loads(new_meta.text)
        self.assertEqual(payload["acceptance_status"], "manual_review")
        self.assertNotIn("editability_score", payload)


if __name__ == "__main__":
    unittest.main()
