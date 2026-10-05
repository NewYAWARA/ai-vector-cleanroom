"""Derived-result navigation and narrowly scoped client-disconnect handling."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workbench


class WorkbenchHandoffLinksTests(unittest.TestCase):
    def _list(self, *, review=False, reference="source_original.png"):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "result_local_refine_test"
            folder.mkdir()
            (folder / "report.json").write_text(json.dumps({
                "input": "icon.png", "acceptance_status": "manual_review",
                "local_refine_history": [{"parent_result": "result_icon"}],
            }), encoding="utf-8")
            (folder / "local_refine_test_vector.svg").write_text(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>', encoding="utf-8")
            (folder / reference).write_bytes(b"reference")
            if review:
                (folder / "review.html").write_text("<!doctype html><title>Review</title>", encoding="utf-8")
            with mock.patch.multiple(workbench, OUTPUT_DIR=root, HISTORY_DIR=root / "_history"):
                return workbench._list_results()

    def _render_rows(self, records):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node is required to execute the actual workbench row renderer")
        script = workbench.APP_HTML.split("function esc(s)", 1)[1].split("let polling=null;", 1)[0]
        program = ("const records=" + json.dumps(records) + ";\n"
                   "const rows=[]; const table={innerHTML:'',appendChild:row=>rows.push(row.innerHTML)};\n"
                   "const document={getElementById:()=>table,createElement:()=>({innerHTML:''}),querySelectorAll:()=>[]};\n"
                   "const api=async()=>records; const tell=message=>{throw new Error(message)};\n"
                   "function esc(s)" + script + "\nrefresh().then(()=>process.stdout.write(JSON.stringify(rows)));\n")
        result = subprocess.run([node, "-"], input=program, text=True, encoding="utf-8",
                                capture_output=True, timeout=15, check=True)
        return json.loads(result.stdout)

    def test_derived_result_without_review_links_to_handoff_and_never_missing_html(self):
        for reference in ("source_original.png", "source_reference.png"):
            with self.subTest(reference=reference):
                records = self._list(reference=reference)
                self.assertEqual(records[0]["review"], "")
                self.assertEqual(records[0]["handoff"], "/handoff?result=result_local_refine_test")
                row = self._render_rows(records)[0]
                self.assertIn('href="/handoff?result=result_local_refine_test"', row)
                self.assertIn("設計師接手", row)
                self.assertNotIn("review.html", row)
                self.assertNotIn("開啟校稿", row)

    def test_existing_review_still_has_its_working_link(self):
        records = self._list(review=True)
        self.assertEqual(records[0]["review"], "result_local_refine_test/review.html")
        row = self._render_rows(records)[0]
        self.assertIn('href="/output/result_local_refine_test/review.html"', row)
        self.assertIn("開啟校稿", row)

    def test_only_disconnect_errors_are_suppressed(self):
        for error in (ConnectionResetError("client reset"), BrokenPipeError("client left")):
            with self.subTest(error=type(error).__name__):
                handler = workbench.Handler.__new__(workbench.Handler)
                handler.close_connection = False
                with mock.patch.object(workbench.BaseHTTPRequestHandler, "handle_one_request", side_effect=error):
                    handler.handle_one_request()
                self.assertTrue(handler.close_connection)
        for error in (RuntimeError("implementation bug"), OSError("disk failure")):
            with self.subTest(error=type(error).__name__):
                handler = workbench.Handler.__new__(workbench.Handler)
                with mock.patch.object(workbench.BaseHTTPRequestHandler, "handle_one_request", side_effect=error):
                    with self.assertRaises(type(error)):
                        handler.handle_one_request()


if __name__ == "__main__":
    unittest.main()
