"""Targeted tests for the standalone conservative compound-path splitter."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compound_path_splitter import process_compound_paths  # noqa: E402


SVG = "http://www.w3.org/2000/svg"


def _svg(body: str, viewbox: str = "0 0 100 100") -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{viewbox}">{body}</svg>'
    )


def _paths(text: str) -> list[ET.Element]:
    root = ET.fromstring(text)
    return list(root.iter(f"{{{SVG}}}path"))


class CompoundPathSplitterTests(unittest.TestCase):
    def test_evenodd_hole_and_island_stay_with_root(self):
        # The first three contours are one root/hole/island containment
        # family.  Only the distant fourth contour may become a sibling path.
        source = _svg(
            '<path fill="#123456" fill-rule="evenodd" opacity=".75" '
            'd="M0.125 0.25 L40.5 0.25 L40.5 40.75 L0.125 40.75 Z '
            'M5.25 5.5 L5.25 35.5 L35.25 35.5 L35.25 5.5 Z '
            'M10.125 10.375 L15.625 10.375 L15.625 15.875 L10.125 15.875 Z '
            'M70.125 70.25 L80.5 70.25 L80.5 80.75 L70.125 80.75 Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "applied", result.report)
        paths = _paths(result.svg_text)
        self.assertEqual(len(paths), 2)
        self.assertEqual(
            len(re.findall(r"[Mm]", paths[0].get("d", ""))), 3,
        )
        self.assertEqual(
            len(re.findall(r"[Mm]", paths[1].get("d", ""))), 1,
        )
        self.assertIn("M5.25 5.5", paths[0].get("d", ""))
        self.assertIn("M10.125 10.375", paths[0].get("d", ""))
        self.assertNotIn("M70.125 70.25", paths[0].get("d", ""))
        self.assertEqual(paths[1].get("d"),
                         "M70.125 70.25 L80.5 70.25 L80.5 80.75 L70.125 80.75 Z")
        for path in paths:
            self.assertEqual(path.get("fill"), "#123456")
            self.assertEqual(path.get("fill-rule"), "evenodd")
            self.assertEqual(path.get("opacity"), ".75")
        self.assertEqual(result.report["source_paths_split"], 1)
        self.assertEqual(result.report["split_paths"], 2)
        self.assertEqual(result.report["subpaths_redistributed"], 4)
        self.assertEqual(result.report["selectable_path_delta"], 1)

    def test_nonzero_nested_contours_are_not_detached(self):
        source = _svg(
            '<path fill="#111" fill-rule="nonzero" '
            'd="M0 0H30V30H0Z M5 5V25H25V5Z '
            'M60 0H70V10H60Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "applied", result.report)
        paths = _paths(result.svg_text)
        self.assertEqual([len(re.findall(r"[Mm]", p.get("d", "")))
                          for p in paths], [2, 1])

    def test_overlapping_top_level_contours_are_not_split(self):
        source = _svg(
            '<path id="overlap" fill="#f00" '
            'd="M0 0 L30 0 L30 30 L0 30 Z '
            'M20 20 L50 20 L50 50 L20 50 Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "no_change", result.report)
        self.assertIs(result.svg_text, source)
        self.assertEqual(result.svg_text, source)
        self.assertEqual(result.report["overlap_locked_paths"], 1)

    def test_exact_monotone_linear_cubics_simplify_without_splitting(self):
        # All controls lie exactly and monotonically on their endpoint
        # segments.  The nested contour keeps the path overlap-locked, while
        # each cubic can still become the mathematically identical line.
        source = _svg(
            '<path id="straight-cubics" fill="#111" fill-rule="evenodd" '
            'd="M0 0 C3 0 7 0 10 0 C10 3 10 7 10 10 '
            'C7 10 3 10 0 10 C0 7 0 3 0 0 Z '
            'M2 2 C4 2 6 2 8 2 C8 4 8 6 8 8 '
            'C6 8 4 8 2 8 C2 6 2 4 2 2 Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "applied", result.report)
        self.assertEqual(len(_paths(result.svg_text)), 1)
        data = _paths(result.svg_text)[0].get("d", "")
        self.assertNotIn("C", data)
        self.assertEqual(len(re.findall(r"L", data)), 8)
        self.assertEqual(len(re.findall(r"M", data)), 2)
        self.assertEqual(result.report["source_paths_split"], 0)
        self.assertEqual(result.report["selectable_path_delta"], 0)
        self.assertEqual(result.report["source_paths_simplified"], 1)
        self.assertEqual(result.report["linear_cubics_simplified"], 8)
        self.assertGreater(result.report["path_data_bytes_saved"], 0)
        self.assertEqual(result.report["overlap_locked_paths"], 1)

        # Once the exact cubics have become lines, another pass is byte-stable.
        repeated = process_compound_paths(result.svg_text)
        self.assertEqual(repeated.status, "no_change", repeated.report)
        self.assertEqual(repeated.svg_text, result.svg_text)

        rejected = process_compound_paths(
            source, validator=lambda _old, _new: False,
        )
        self.assertEqual(rejected.status, "rolled_back", rejected.report)
        self.assertIs(rejected.svg_text, source)
        self.assertEqual(rejected.report["source_paths_simplified"], 0)
        self.assertEqual(rejected.report["linear_cubics_simplified"], 0)
        self.assertEqual(
            rejected.report["attempted_simplification"][
                "linear_cubics_simplified"
            ],
            8,
        )

    def test_collinear_but_nonmonotone_cubic_is_not_simplified(self):
        # The first cubic doubles back along its line.  Replacing it with a
        # segment could change fill topology, so the exact simplifier refuses.
        source = _svg(
            '<path id="doubling-back" fill="#111" fill-rule="evenodd" '
            'd="M0 0 C15 0 -5 0 10 0 C12 4 8 10 0 0 Z '
            'M2 1 C9 1 -1 1 7 1 C9 3 6 8 2 1 Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "no_change", result.report)
        self.assertIs(result.svg_text, source)
        self.assertEqual(result.report["linear_cubics_simplified"], 0)

    def test_ids_and_output_are_deterministic_and_unique(self):
        source = _svg(
            '<g fill="#abc"><path '
            'd="M1 1L9 1L9 9L1 9Z M21 1L29 1L29 9L21 9Z '
            'M41 1L49 1L49 9L41 9Z"/></g>',
            "0 0 60 20",
        )
        first = process_compound_paths(source)
        second = process_compound_paths(source)
        self.assertEqual(first.status, "applied", first.report)
        self.assertEqual(first.svg_text, second.svg_text)
        ids = [path.get("id") for path in _paths(first.svg_text)]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(node_id and node_id.startswith("compound-path-")
                            for node_id in ids))
        # A second pass is idempotent: every emitted path now has one family.
        third = process_compound_paths(first.svg_text)
        self.assertEqual(third.status, "no_change", third.report)
        self.assertEqual(third.svg_text, first.svg_text)

    def test_existing_source_id_is_retained_on_first_family(self):
        source = _svg(
            '<path id="brand-shape" fill="#222" '
            'd="M0 0H10V10H0Z M30 0H40V10H30Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "applied", result.report)
        ids = [path.get("id") for path in _paths(result.svg_text)]
        self.assertEqual(ids[0], "brand-shape")
        self.assertEqual(len(ids), len(set(ids)))

    def test_validator_rejection_and_exception_roll_back_byte_exactly(self):
        source = _svg(
            '<path fill="#111" '
            'd="M0 0H10V10H0Z M30 0H40V10H30Z"/>'
        )
        calls: list[tuple[str, str]] = []

        def reject(old: str, new: str) -> bool:
            calls.append((old, new))
            return False

        rejected = process_compound_paths(source, validator=reject)
        self.assertEqual(rejected.status, "rolled_back")
        self.assertIs(rejected.svg_text, source)
        self.assertEqual(calls[0][0], source)
        self.assertNotEqual(calls[0][1], source)
        self.assertIn("validator rejected", rejected.report["reason"])

        def broken(_old: str, _new: str) -> bool:
            raise RuntimeError("renderer unavailable")

        raised = process_compound_paths(source, validator=broken)
        self.assertEqual(raised.status, "rolled_back")
        self.assertIs(raised.svg_text, source)
        self.assertIn("RuntimeError", raised.report["reason"])

    def test_transform_reference_and_unsupported_data_fail_closed(self):
        cases = {
            "transform": _svg(
                '<g transform="translate(1 2)"><path fill="#111" '
                'd="M0 0H10V10H0Z M30 0H40V10H30Z"/></g>'
            ),
            "reference": _svg(
                '<defs><path id="source-shape" fill="#111" '
                'd="M0 0H10V10H0Z M30 0H40V10H30Z"/></defs>'
                '<path id="live-shape" fill="#111" '
                'd="M0 0H10V10H0Z M30 0H40V10H30Z"/>'
                '<use href="#live-shape" x="1"/>'
            ),
            "unsupported-command": _svg(
                '<path fill="#111" '
                'd="M0 0 R5 5 10 10 Z M30 0H40V10H30Z"/>'
            ),
            "malformed-geometry": _svg(
                '<path fill="#111" '
                'd="M0 0H10V10H0Z M30 0 L40 Z"/>'
            ),
            "object-bbox-gradient": _svg(
                '<defs><linearGradient id="g"><stop offset="0" '
                'stop-color="#000"/></linearGradient></defs>'
                '<path fill="url(#g)" '
                'd="M0 0H10V10H0Z M30 0H40V10H30Z"/>'
            ),
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                result = process_compound_paths(source)
                self.assertEqual(result.status, "rolled_back", result.report)
                self.assertIs(result.svg_text, source)
                self.assertEqual(result.svg_text, source)
                self.assertTrue(result.report.get("reason"))

    def test_user_space_gradient_is_safe_to_split(self):
        source = _svg(
            '<defs><linearGradient id="g" gradientUnits="userSpaceOnUse" '
            'x1="0" y1="0" x2="100" y2="0"><stop offset="0" '
            'stop-color="#000"/></linearGradient></defs>'
            '<path fill="url(#g)" fill-rule="evenodd" '
            'd="M0 0H10V10H0Z M30 0H40V10H30Z"/>'
        )
        result = process_compound_paths(source)
        self.assertEqual(result.status, "applied", result.report)
        self.assertEqual(len(_paths(result.svg_text)), 2)
        self.assertTrue(all(path.get("fill") == "url(#g)"
                            for path in _paths(result.svg_text)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
