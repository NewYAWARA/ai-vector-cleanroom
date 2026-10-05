from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image
import handoff_service as service
from designer_handoff import build_handoff_manifest
import workbench


class HandoffServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "output"
        self.folder = self.output / "result_icon"
        self.folder.mkdir(parents=True)
        self.svg = self.folder / "icon_vector.svg"
        self.original = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                         '<circle id="disc" cx="30" cy="30" r="20" fill="red"/>'
                         '<path id="triangle" d="M60 20 L90 80 L40 80 Z" fill="blue"/></svg>')
        self.svg.write_text(self.original, encoding="utf-8")
        Image.new("RGB", (100, 100), "white").save(self.folder / "source_reference.png")
        (self.folder / "report.json").write_text(json.dumps({"input": "icon.png"}), encoding="utf-8")
        self.manifest = build_handoff_manifest(self.svg)
        self.decisions = {item["id"]: "review" for item in self.manifest["objects"]}
        self.first = next(iter(self.decisions))
        self.payload = {"result": self.folder.name,
                        "svg_sha256": self.manifest["svg_sha256"], "decisions": self.decisions,
                        "revision": None}

    def test_save_reopen_and_lock_are_bound_to_exact_svg(self):
        self.decisions[self.first] = "keep"
        service.save_decisions(self.output, self.payload)
        self.assertEqual(service.load_decisions(self.folder, self.manifest), self.decisions)
        self.assertTrue(service.has_locked_objects(self.output, self.folder.name))
        self.assertEqual(self.svg.read_text(encoding="utf-8"), self.original)

    def test_stale_save_cannot_replace_valid_decisions(self):
        service.save_decisions(self.output, self.payload)
        state = (self.folder / service.STATE_NAME).read_bytes()
        with self.assertRaises(service.HandoffConflict):
            service.save_decisions(self.output, {**self.payload, "svg_sha256": "0" * 64})
        self.assertEqual((self.folder / service.STATE_NAME).read_bytes(), state)

    def test_missing_or_unknown_decisions_cannot_silently_drop_art(self):
        for decisions in ({}, {**self.decisions, "unknown": "keep"}):
            with self.assertRaises(ValueError):
                service.save_decisions(self.output, {**self.payload, "decisions": decisions})
        self.assertFalse((self.folder / service.STATE_NAME).exists())

    def test_existing_stale_record_is_preserved_even_with_new_payload_hash(self):
        service.save_decisions(self.output, self.payload)
        original_state = (self.folder / service.STATE_NAME).read_bytes()
        self.svg.write_text(self.original.replace('fill="red"', 'fill="green"'), encoding="utf-8")
        new_manifest = build_handoff_manifest(self.svg)
        with self.assertRaises(service.HandoffConflict):
            service.save_decisions(self.output, {**self.payload, "svg_sha256": new_manifest["svg_sha256"]})
        self.assertEqual((self.folder / service.STATE_NAME).read_bytes(), original_state)

    def test_original_reference_preferred_and_legacy_reference_labeled(self):
        _folder, _svg, source, report = service.result_context(self.output, self.folder.name)
        self.assertEqual(source.name, "source_reference.png")
        self.assertEqual(report["_handoff_reference_kind"], "processed_reference")
        Image.new("RGBA", (100, 100), (255, 0, 0, 3)).save(self.folder / "source_original.png")
        _folder, _svg, source, report = service.result_context(self.output, self.folder.name)
        self.assertEqual(source.name, "source_original.png")
        self.assertEqual(report["_handoff_reference_kind"], "original")

    def test_traversal_and_ambiguous_svg_are_rejected(self):
        for name in ("../result_icon", "result_../x", "result_x\\y", "C:\\result_icon"):
            with self.assertRaises(ValueError):
                service.result_context(self.output, name)
        (self.folder / "other_vector.svg").write_text(self.original)
        with self.assertRaises(ValueError):
            service.result_context(self.output, self.folder.name)

    def test_export_is_immutable_complete_package_and_honest_partial(self):
        self.decisions[self.first] = "keep"
        before = hashlib.sha256(self.svg.read_bytes()).hexdigest()
        first = service.export_package(self.output, self.payload)
        second = service.export_package(self.output, {**self.payload, "revision": first["revision"]})
        self.assertNotEqual(first["revision"], second["revision"])
        self.assertNotEqual(first["files"][0]["url"], second["files"][0]["url"])
        self.assertTrue(first["summary"]["is_partial"])
        packages = list(self.folder.glob("handoff_*.zip"))
        self.assertEqual(len(packages), 2)
        with zipfile.ZipFile(packages[0]) as archive:
            names = archive.namelist()
            self.assertIn("accepted.svg", names)
            self.assertIn("working.svg", names)
            self.assertIn("draft.svg", names)
            self.assertIn("OPEN_IN_ILLUSTRATOR.txt", names)
            self.assertIn("不完整", archive.read("OPEN_IN_ILLUSTRATOR.txt").decode("utf-8"))
            self.assertIn("清理後參考圖", archive.read("OPEN_IN_ILLUSTRATOR.txt").decode("utf-8"))
            self.assertEqual(json.loads(archive.read("handoff.json"))["reference_kind"], "processed_reference")
            self.assertEqual(json.loads(archive.read("handoff-manifest.json"))["reference_kind"], "processed_reference")
            draft = ET.fromstring(archive.read("draft.svg"))
            reference = next(item for item in draft.iter() if item.get("data-handoff-role") == "raster-reference")
            self.assertIn("清理後參考圖", reference.get("{http://www.inkscape.org/namespaces/inkscape}label"))
        self.assertEqual(hashlib.sha256(self.svg.read_bytes()).hexdigest(), before)

    def test_revision_is_required_and_stale_tab_cannot_unlock_saved_keep(self):
        missing = {key: value for key, value in self.payload.items() if key != "revision"}
        with self.assertRaises(service.HandoffConflict):
            service.save_decisions(self.output, missing)
        kept = {**self.decisions, self.first: "keep"}
        state = service.save_decisions(self.output, {**self.payload, "decisions": kept})
        saved_bytes = (self.folder / service.STATE_NAME).read_bytes()
        self.assertIsInstance(state["revision"], str)
        for stale in (None, "not-current"):
            with self.assertRaises(service.HandoffConflict):
                service.save_decisions(self.output, {**self.payload, "revision": stale})
        self.assertEqual((self.folder / service.STATE_NAME).read_bytes(), saved_bytes)
        self.assertTrue(service.has_locked_objects(self.output, self.folder.name))
        # Explicit edits from the currently saved revision are allowed.
        newer = service.save_decisions(self.output, {**self.payload, "revision": state["revision"]})
        self.assertNotEqual(state["revision"], newer["revision"])

    def test_page_carries_current_revision_or_initial_null(self):
        with mock.patch("handoff_page.build_handoff_page", return_value="page") as page:
            service.build_page(self.output, self.folder.name, "token")
            self.assertIsNone(page.call_args.args[0]["saved_revision"])
            state = service.save_decisions(self.output, self.payload)
            service.build_page(self.output, self.folder.name, "token")
            self.assertEqual(page.call_args.args[0]["saved_revision"], state["revision"])

    def test_failed_export_does_not_advance_revision_or_replace_state(self):
        state = service.save_decisions(self.output, self.payload)
        saved_bytes = (self.folder / service.STATE_NAME).read_bytes()
        payload = {**self.payload, "revision": state["revision"]}
        with mock.patch.object(service, "export_handoff", side_effect=ValueError("invalid source PNG")):
            with self.assertRaises(ValueError):
                service.export_package(self.output, payload)
        self.assertEqual((self.folder / service.STATE_NAME).read_bytes(), saved_bytes)
        self.assertEqual(service.validate_payload_revision(self.folder, payload)["revision"], state["revision"])

    def test_legacy_state_revision_is_read_only_and_still_prevents_lost_updates(self):
        legacy = {"schema": "aivc.handoff-state/v1", "svg_sha256": self.manifest["svg_sha256"], "decisions": self.decisions}
        state_path = self.folder / service.STATE_NAME
        state_path.write_text(json.dumps(legacy), encoding="utf-8")
        original = state_path.read_bytes()
        with mock.patch("handoff_page.build_handoff_page", return_value="page") as page:
            service.build_page(self.output, self.folder.name, "token")
            revision = page.call_args.args[0]["saved_revision"]
        self.assertTrue(revision.startswith("legacy-"))
        self.assertEqual(state_path.read_bytes(), original)
        with self.assertRaises(service.HandoffConflict):
            service.save_decisions(self.output, self.payload)
        state = service.save_decisions(self.output, {**self.payload, "revision": revision})
        self.assertNotEqual(state["revision"], revision)

    def test_concurrent_initial_saves_have_one_winner(self):
        barrier = threading.Barrier(2)
        outcomes = []
        def save():
            barrier.wait(timeout=5)
            try:
                service.save_decisions(self.output, self.payload)
                outcomes.append("saved")
            except service.HandoffConflict:
                outcomes.append("conflict")
        threads = [threading.Thread(target=save) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        self.assertCountEqual(outcomes, ["saved", "conflict"])

    def test_commit_helper_itself_refuses_to_replace_locked_objects(self):
        service.save_decisions(self.output, {**self.payload, "decisions": {**self.decisions, self.first: "keep"}})
        with mock.patch.object(workbench, "OUTPUT_DIR", self.output):
            with self.assertRaisesRegex(RuntimeError, "已採用"):
                workbench._commit_staged_result(Path(self.temp.name) / "staging", "icon")
        self.assertEqual(self.svg.read_text(encoding="utf-8"), self.original)

    def test_original_reference_provenance_survives_export(self):
        original_png = self.folder / "source_original.png"
        Image.new("RGBA", (100, 100), (255, 0, 0, 3)).save(original_png)
        service.export_package(self.output, self.payload)
        archive_path = next(self.folder.glob("handoff_*.zip"))
        with zipfile.ZipFile(archive_path) as archive:
            record = json.loads(archive.read("handoff.json"))
            self.assertEqual(record["reference_kind"], "original")
            self.assertEqual(record["source_png_sha256"], hashlib.sha256(original_png.read_bytes()).hexdigest())
            draft = ET.fromstring(archive.read("draft.svg"))
            reference = next(item for item in draft.iter() if item.get("data-handoff-role") == "raster-reference")
            self.assertIn("原圖參考", reference.get("{http://www.inkscape.org/namespaces/inkscape}label"))

    def _server(self):
        patcher = mock.patch.multiple(workbench, OUTPUT_DIR=self.output, WB_TOKEN="test-token")
        patcher.start()
        self.addCleanup(patcher.stop)
        server = workbench.ThreadingHTTPServer(("127.0.0.1", 0), workbench.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def _request(self, port, method, path, payload=None, token="test-token"):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.addCleanup(connection.close)
        headers = {"X-WB-Token": token, "Content-Type": "application/json"}
        connection.request(method, path, None if payload is None else json.dumps(payload), headers)
        result = connection.getresponse()
        return result.status, result.read()

    def test_http_page_save_and_export(self):
        port = self._server()
        status, page = self._request(port, "GET", "/handoff?result=result_icon")
        self.assertEqual(status, 200)
        self.assertIn("接手".encode(), page)
        status, saved_body = self._request(port, "POST", "/api/handoff/save", self.payload)
        self.assertEqual(status, 200)
        saved = json.loads(saved_body)
        status, body = self._request(port, "POST", "/api/handoff/export", {**self.payload, "revision": saved["revision"]})
        self.assertEqual(status, 200)
        package = json.loads(body)
        self.assertNotEqual(package["revision"], saved["revision"])
        status, body = self._request(port, "GET", package["files"][0]["url"])
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"PK"))

    def test_http_rejects_token_stale_and_bad_body(self):
        port = self._server()
        status, _ = self._request(port, "POST", "/api/handoff/save", self.payload, token="wrong")
        self.assertEqual(status, 403)
        status, _ = self._request(port, "POST", "/api/handoff/save", {**self.payload, "svg_sha256": "old"})
        self.assertEqual(status, 409)
        status, _ = self._request(port, "POST", "/api/handoff/save", [])
        self.assertEqual(status, 400)

    def test_http_prepare_preserves_unchanged_response_and_token_gate(self):
        port = self._server()
        expected = {'url': None, 'summary': {'status': 'unchanged'}, 'units': []}
        with mock.patch('auto_prepare.prepare_result', return_value=expected) as prepare:
            status, body = self._request(port, 'POST', '/api/handoff/prepare', self.payload, token='wrong')
            self.assertEqual(status, 403)
            prepare.assert_not_called()
            status, body = self._request(port, 'POST', '/api/handoff/prepare', self.payload)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), expected)
            self.assertEqual(prepare.call_args.args, (self.output, self.payload))
            self.assertIs(prepare.call_args.kwargs['lock'], workbench._publish_lock)

    def test_page_exposes_preparation_provenance(self):
        record = {'summary': {'anchors_removed': 20}, 'units': []}
        (self.folder / 'report.json').write_text(json.dumps({'auto_prepare': record}))
        with mock.patch('handoff_page.build_handoff_page', return_value='page') as page:
            service.build_page(self.output, self.folder.name, 'token')
            self.assertEqual(page.call_args.args[0]['auto_prepare'], record)


if __name__ == "__main__":
    unittest.main()
